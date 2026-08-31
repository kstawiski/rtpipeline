"""Materialize a producer-validated organized-course manifest."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rtpipeline.snakemake_delegate import invoke, runtime_environment


MANIFEST_SCHEMA = "rtpipeline-organized-course-manifest-v2"
STATUS_VALIDATED = "validated"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _iter_course_dirs(output_dir: Path):
    if not output_dir.is_dir():
        return
    for patient_dir in sorted(output_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        if patient_dir.name.startswith(("_", ".")):
            continue
        if patient_dir.name in {
            "Data",
            "Data_Snakemake_fallback",
            "Logs_Snakemake_fallback",
            "_RESULTS",
        }:
            continue
        for course_dir in sorted(patient_dir.iterdir()):
            if not course_dir.is_dir() or course_dir.name.startswith("_"):
                continue
            yield patient_dir.name, course_dir.name, course_dir


def _estimate_course_complexity(course_path: Path) -> int:
    dicom_root = course_path / "DICOM"
    search_root = dicom_root if dicom_root.exists() else course_path
    count = 0
    for _, _, files in os.walk(search_root):
        count += sum(
            1 for name in files if name.lower().endswith((".dcm", ".ima"))
        )
    if count == 0:
        for _, _, files in os.walk(course_path):
            count += len(files)
    return max(1, count)


def _revoke_all_organized_flags(output_dir: Path) -> None:
    for _, _, course_dir in _iter_course_dirs(output_dir):
        (course_dir / ".organized").unlink(missing_ok=True)


def _manifest_payload(ledger: dict, courses: list[dict]) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "cohort_status": ledger["status"],
        "intended_course_count": ledger["intended_course_count"],
        "attempted_course_count": ledger["attempted_course_count"],
        "validated_course_count": ledger["validated_course_count"],
        "technical_quarantine_count": ledger["technical_quarantine_count"],
        "ledger_path": str(Path("_COURSES") / "organize_ledger.json"),
        "courses": courses,
        "technical_quarantines": ledger["technical_quarantines"],
    }


def _delegate_validation(
    workflow: Any, output_dir: Path, *, mode: str, result_dir: Path
) -> tuple[dict, list[dict], list[dict]]:
    payload = invoke(
        python=str(workflow.params.python),
        operation="validate-organize",
        arguments=("--output-dir", str(output_dir), "--mode", mode),
        result_dir=result_dir,
        env=runtime_environment(workflow.params),
    )
    if payload.get("mode") != mode:
        raise RuntimeError(
            "pipeline manifest validator returned a mismatched validation mode"
        )
    ledger = payload.get("ledger")
    validated = payload.get("validated_courses")
    invalid = payload.get("invalid_courses")
    if (
        not isinstance(ledger, dict)
        or not isinstance(validated, list)
        or not all(isinstance(entry, dict) for entry in validated)
        or not isinstance(invalid, list)
        or not all(isinstance(entry, dict) for entry in invalid)
    ):
        raise RuntimeError("pipeline manifest validator returned a malformed payload")
    ledger_validated = {
        (entry["patient"], entry["course"], str(entry.get("path") or ""))
        for entry in ledger.get("courses", [])
        if isinstance(entry, dict) and entry.get("status") == STATUS_VALIDATED
    }
    delegated_validated = {
        (
            str(entry.get("patient") or ""),
            str(entry.get("course") or ""),
            str(entry.get("path") or ""),
        )
        for entry in validated
    }
    delegated_invalid_ids = {
        (
            str(entry.get("patient") or ""),
            str(entry.get("course") or ""),
        )
        for entry in invalid
    }
    ledger_validated_ids = {
        (patient, course) for patient, course, _ in ledger_validated
    }
    delegated_validated_ids = {
        (patient, course) for patient, course, _ in delegated_validated
    }
    if mode == "check":
        reconciled = (
            ledger.get("validated_course_count")
            == len(validated) + len(invalid)
            and delegated_validated <= ledger_validated
            and delegated_validated_ids.isdisjoint(delegated_invalid_ids)
            and delegated_validated_ids | delegated_invalid_ids
            == ledger_validated_ids
        )
    else:
        reconciled = (
            ledger.get("validated_course_count") == len(validated)
            and delegated_validated == ledger_validated
        )
    if not reconciled:
        raise RuntimeError(
            "pipeline manifest validator result does not reconcile with its ledger"
        )
    return ledger, validated, invalid


def _existing_manifest_is_valid(
    workflow: Any, manifest_path: Path, output_dir: Path
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ledger, validated, invalid = _delegate_validation(
            workflow,
            output_dir,
            mode="check",
            result_dir=manifest_path.parent,
        )
    except Exception:
        return False
    if manifest.get("schema") != MANIFEST_SCHEMA or invalid:
        return False
    if ledger["technical_quarantine_count"]:
        return False
    for field in (
        "intended_course_count",
        "attempted_course_count",
        "validated_course_count",
        "technical_quarantine_count",
    ):
        if manifest.get(field) != ledger[field]:
            return False
    manifest_courses = manifest.get("courses")
    if (
        not isinstance(manifest_courses, list)
        or not all(isinstance(entry, dict) for entry in manifest_courses)
        or manifest.get("technical_quarantines") != ledger["technical_quarantines"]
        or manifest.get("cohort_status") != ledger["status"]
    ):
        return False
    manifest_validated = {
        (
            str(entry.get("patient") or ""),
            str(entry.get("course") or ""),
            str(entry.get("path") or ""),
        )
        for entry in manifest_courses
    }
    delegated_validated = {
        (entry["patient"], entry["course"], str(entry.get("path") or ""))
        for entry in validated
    }
    if (
        len(manifest_courses) != len(delegated_validated)
        or manifest_validated != delegated_validated
    ):
        return False
    for entry in validated:
        course_dir = Path(str(entry["path"]))
        try:
            if (course_dir / ".organized").read_text(
                encoding="utf-8"
            ).strip() != "ok":
                return False
        except Exception:
            (course_dir / ".organized").unlink(missing_ok=True)
            return False
    return True


def _validated_courses_from_delegate(
    workflow: Any,
    output_dir: Path,
    *,
    prioritize_short_courses: bool,
    result_dir: Path,
) -> tuple[dict, list[dict]]:
    ledger, validated, _ = _delegate_validation(
        workflow,
        output_dir,
        mode="quarantine",
        result_dir=result_dir,
    )
    courses = [
        {
            "patient": entry["patient"],
            "course": entry["course"],
            "path": str(entry["path"]),
            "complexity": _estimate_course_complexity(Path(str(entry["path"]))),
        }
        for entry in validated
    ]
    if prioritize_short_courses:
        courses.sort(
            key=lambda entry: (
                entry.get("complexity", 0),
                entry["patient"],
                entry["course"],
            )
        )
    else:
        courses.sort(key=lambda entry: (entry["patient"], entry["course"]))
    return ledger, courses


def main(workflow: Any) -> None:
    manifest_path = Path(workflow.output.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(workflow.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(workflow.params.output_dir)

    if _existing_manifest_is_valid(workflow, manifest_path, output_dir):
        log_path.write_text(
            "Organize stage skipped after manifest and course-contract validation.\n",
            encoding="utf-8",
        )
        return

    # Revoke the prior publication before executing the producer. If execution
    # fails, neither stale manifest bytes nor stale success flags survive.
    manifest_path.unlink(missing_ok=True)
    _revoke_all_organized_flags(output_dir)
    command = [
        str(workflow.params.python),
        "-m",
        "rtpipeline.cli",
        "--dicom-root",
        str(workflow.params.dicom_root),
        "--outdir",
        str(output_dir),
        "--logs",
        str(workflow.params.logs_dir),
        "--stage",
        "organize",
        "--max-workers",
        str(max(1, int(workflow.threads))),
    ]
    custom_structures = str(workflow.params.custom_structures)
    if custom_structures:
        command.extend(["--custom-structures", custom_structures])
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("DEBUG: Starting rtpipeline.cli organize stage...\n")
        log_file.flush()
        subprocess.run(
            command,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=runtime_environment(workflow.params),
        )

    try:
        ledger, courses = _validated_courses_from_delegate(
            workflow,
            output_dir,
            prioritize_short_courses=bool(
                workflow.params.prioritize_short_courses
            ),
            result_dir=manifest_path.parent,
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Manifest contract validation failed: {exc}\n")
        raise

    for course in courses:
        _write_text_atomic(Path(course["path"]) / ".organized", "ok\n")
    payload = _manifest_payload(ledger, courses)
    _write_text_atomic(
        manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if "snakemake" in globals():
    main(snakemake)  # type: ignore[name-defined]
