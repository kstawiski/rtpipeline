from __future__ import annotations

"""Pipeline-interpreter operations requested by dependency-light workflow scripts."""

import argparse
import json
import os
import runpy
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DELEGATE_SCHEMA = "rtpipeline-workflow-delegate-v1"


def _write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validated_course_path(
    output_dir: Path, patient: str, course: str, path_text: str
) -> Path:
    root = output_dir.resolve(strict=False)
    course_dir = Path(path_text).resolve(strict=False)
    try:
        relative = course_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"ledger course path is outside output root: {course_dir}") from exc
    if relative.parts != (patient, course):
        raise RuntimeError(
            "ledger path identity does not match patient/course identifiers"
        )
    return course_dir


def _validate_organize(
    output_dir: Path, *, quarantine_invalid: bool
) -> dict[str, Any]:
    from .course_contract import load_course_contract
    from .organize_ledger import (
        STATUS_TECHNICAL_QUARANTINE,
        STATUS_VALIDATED,
        quarantine_course_directory,
        read_organize_ledger,
        write_organize_ledger,
    )

    output_dir = Path(output_dir)
    ledger = read_organize_ledger(output_dir)
    entries = [dict(entry) for entry in ledger["courses"]]
    validated_courses: list[dict[str, str]] = []
    invalid_courses: list[dict[str, Any]] = []
    quarantine_failures: list[str] = []

    for entry in entries:
        if entry["status"] != STATUS_VALIDATED:
            continue
        patient = entry["patient"]
        course = entry["course"]
        course_path = Path(str(entry.get("path") or ""))
        safe_course_path = None
        try:
            safe_course_path = _validated_course_path(
                output_dir, patient, course, str(course_path)
            )
            course_path = safe_course_path
            load_course_contract(course_path)
        except Exception as exc:
            reason = f"manifest validation failed: {type(exc).__name__}: {exc}"
            invalid = {
                "patient": patient,
                "course": course,
                "path": str(course_path),
                "reason": reason,
                "quarantine_path": None,
            }
            if quarantine_invalid:
                quarantine_path = None
                try:
                    quarantine_path = quarantine_course_directory(
                        output_dir,
                        course_path,
                        patient=patient,
                        course=course,
                        reason=reason,
                        phase="manifest_contract_validation",
                    )
                except Exception as quarantine_exc:
                    if safe_course_path is not None:
                        (safe_course_path / ".organized").unlink(missing_ok=True)
                        (
                            safe_course_path / "metadata" / "case_metadata.json"
                        ).unlink(missing_ok=True)
                    failure = (
                        f"{patient}/{course}: {reason}; quarantine failed: "
                        f"{type(quarantine_exc).__name__}: {quarantine_exc}"
                    )
                    quarantine_failures.append(failure)
                    reason = failure
                invalid["reason"] = reason
                invalid["quarantine_path"] = (
                    str(quarantine_path) if quarantine_path is not None else None
                )
                entry.update(
                    {
                        "status": STATUS_TECHNICAL_QUARANTINE,
                        "reason": reason,
                        "quarantine_path": invalid["quarantine_path"],
                    }
                )
            invalid_courses.append(invalid)
            continue

        validated_courses.append(
            {"patient": patient, "course": course, "path": str(course_path)}
        )

    if quarantine_invalid:
        ledger = write_organize_ledger(output_dir, entries)
        if ledger["validated_course_count"] != len(validated_courses):
            raise RuntimeError(
                "validated organize-ledger count does not match "
                "producer-validated manifest courses"
            )
        if quarantine_failures:
            raise RuntimeError(
                "could not revoke one or more manifest-rejected courses after "
                "validating the complete ledger: " + " | ".join(quarantine_failures)
            )

    return {
        "mode": "quarantine" if quarantine_invalid else "check",
        "ledger": ledger,
        "validated_courses": validated_courses,
        "invalid_courses": invalid_courses,
    }


def _assess_segmentation(course_dir: Path) -> dict[str, Any]:
    from .segmentation import assess_course_segmentation

    course_dir = Path(course_dir)
    return {
        "course_dir": str(course_dir.resolve(strict=False)),
        "outcome": assess_course_segmentation(course_dir),
    }


def _publish_radiomics_completion(
    course_dir: Path, sentinel_path: Path
) -> dict[str, Any]:
    from .radiomics_ct_contract import (
        validate_completion_sentinel,
        write_completion_sentinel,
    )

    course_dir = Path(course_dir)
    sentinel_path = write_completion_sentinel(course_dir, Path(sentinel_path))
    sentinel = validate_completion_sentinel(course_dir, sentinel_path)
    return {
        "course_dir": str(course_dir.resolve(strict=False)),
        "sentinel_path": str(sentinel_path.resolve(strict=False)),
        "sentinel": sentinel,
    }


def _to_namespace(value: object) -> object:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{str(key): _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _aggregate(context_path: Path) -> dict[str, Any]:
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"aggregate workflow context is unreadable: {context_path}: {exc}"
        ) from exc
    if not isinstance(context, dict):
        raise RuntimeError("aggregate workflow context must be a JSON object")
    workflow = _to_namespace(context)
    implementation = Path(__file__).with_name("workflow_aggregate.py")
    runpy.run_path(str(implementation), init_globals={"snakemake": workflow})
    return {"implementation": str(implementation), "completed": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    organize = subparsers.add_parser("validate-organize")
    organize.add_argument("--output-dir", required=True)
    organize.add_argument(
        "--mode", choices=("check", "quarantine"), required=True
    )

    segmentation = subparsers.add_parser("assess-segmentation")
    segmentation.add_argument("--course-dir", required=True)

    radiomics = subparsers.add_parser("publish-radiomics-completion")
    radiomics.add_argument("--course-dir", required=True)
    radiomics.add_argument("--sentinel-path", required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--context-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path = Path(args.result_path)
    operation = str(args.operation)
    try:
        if operation == "validate-organize":
            payload = _validate_organize(
                Path(args.output_dir), quarantine_invalid=args.mode == "quarantine"
            )
        elif operation == "assess-segmentation":
            payload = _assess_segmentation(Path(args.course_dir))
        elif operation == "publish-radiomics-completion":
            payload = _publish_radiomics_completion(
                Path(args.course_dir), Path(args.sentinel_path)
            )
        elif operation == "aggregate":
            payload = _aggregate(Path(args.context_path))
        else:
            raise RuntimeError(f"unsupported workflow operation {operation!r}")
    except BaseException as exc:
        _write_json_atomic(
            result_path,
            {
                "schema": DELEGATE_SCHEMA,
                "operation": operation,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        traceback.print_exc()
        return 1

    _write_json_atomic(
        result_path,
        {
            "schema": DELEGATE_SCHEMA,
            "operation": operation,
            "status": "ok",
            "payload": payload,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
