"""Materialize a producer-validated organized-course manifest."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from rtpipeline.course_contract import load_course_contract
from rtpipeline.organize_ledger import (
    STATUS_TECHNICAL_QUARANTINE,
    STATUS_VALIDATED,
    quarantine_course_directory,
    read_organize_ledger,
    write_organize_ledger,
)


MANIFEST_SCHEMA = "rtpipeline-organized-course-manifest-v2"


def _runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    root_dir = str(snakemake.params.root_dir)  # type: ignore[name-defined]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        if root_dir not in existing_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([root_dir, existing_pythonpath])
    else:
        env["PYTHONPATH"] = root_dir
    env["RTPIPELINE_CONFIGFILE"] = str(snakemake.params.configfile)  # type: ignore[name-defined]
    env["RTPIPELINE_RADIOMICS_ENV"] = str(snakemake.params.radiomics_env)  # type: ignore[name-defined]
    python_bin = str(snakemake.params.python_bin)  # type: ignore[name-defined]
    current_path = env.get("PATH", "")
    if python_bin not in current_path.split(os.pathsep):
        env["PATH"] = os.pathsep.join([python_bin, current_path])
    return env


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
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


def _existing_manifest_is_valid(manifest_path: Path, output_dir: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ledger = read_organize_ledger(output_dir)
    except Exception:
        return False
    if manifest.get("schema") != MANIFEST_SCHEMA:
        return False
    # A prior technical quarantine is retryable. Never let resume turn it into a
    # silently accepted reduced cohort.
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
    ledger_validated = {
        (entry["patient"], entry["course"], str(entry.get("path") or ""))
        for entry in ledger["courses"]
        if entry["status"] == STATUS_VALIDATED
    }
    manifest_validated = {
        (
            str(entry.get("patient") or ""),
            str(entry.get("course") or ""),
            str(entry.get("path") or ""),
        )
        for entry in manifest_courses
        if isinstance(entry, dict)
    }
    if (
        len(manifest_courses) != len(ledger_validated)
        or manifest_validated != ledger_validated
    ):
        return False
    for patient, course, path_text in ledger_validated:
        course_dir = None
        try:
            course_dir = _validated_course_path(
                output_dir, patient, course, path_text
            )
            if (course_dir / ".organized").read_text(encoding="utf-8").strip() != "ok":
                return False
            load_course_contract(course_dir)
        except Exception:
            if course_dir is not None:
                (course_dir / ".organized").unlink(missing_ok=True)
            return False
    return True


def _validated_courses_from_ledger(
    output_dir: Path, *, prioritize_short_courses: bool
) -> tuple[dict, list[dict]]:
    ledger = read_organize_ledger(output_dir)
    entries = [dict(entry) for entry in ledger["courses"]]
    validation_failures: list[str] = []
    courses: list[dict] = []

    for entry in entries:
        if entry["status"] != STATUS_VALIDATED:
            continue
        patient_id = entry["patient"]
        course_id = entry["course"]
        course_path = Path(str(entry.get("path") or ""))
        safe_course_path = None
        try:
            safe_course_path = _validated_course_path(
                output_dir, patient_id, course_id, str(course_path)
            )
            course_path = safe_course_path
            load_course_contract(course_path)
        except Exception as exc:
            reason = f"manifest validation failed: {type(exc).__name__}: {exc}"
            quarantine_path = None
            try:
                quarantine_path = quarantine_course_directory(
                    output_dir,
                    course_path,
                    patient=patient_id,
                    course=course_id,
                    reason=reason,
                    phase="manifest_contract_validation",
                )
            except Exception as quarantine_exc:
                if safe_course_path is not None:
                    (safe_course_path / ".organized").unlink(missing_ok=True)
                    (
                        safe_course_path / "metadata" / "case_metadata.json"
                    ).unlink(missing_ok=True)
                validation_failures.append(
                    f"{patient_id}/{course_id}: {reason}; quarantine failed: "
                    f"{type(quarantine_exc).__name__}: {quarantine_exc}"
                )
                reason = validation_failures[-1]
            entry.update(
                {
                    "status": STATUS_TECHNICAL_QUARANTINE,
                    "reason": reason,
                    "quarantine_path": (
                        str(quarantine_path) if quarantine_path is not None else None
                    ),
                }
            )
            continue

        courses.append(
            {
                "patient": patient_id,
                "course": course_id,
                "path": str(course_path),
                "complexity": _estimate_course_complexity(course_path),
            }
        )

    ledger = write_organize_ledger(output_dir, entries)
    if ledger["validated_course_count"] != len(courses):
        raise RuntimeError(
            "validated organize-ledger count does not match producer-validated manifest courses"
        )
    if validation_failures:
        raise RuntimeError(
            "could not revoke one or more manifest-rejected courses after validating "
            "the complete ledger: " + " | ".join(validation_failures)
        )
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


manifest_path = Path(snakemake.output.manifest)  # type: ignore[name-defined]
manifest_path.parent.mkdir(parents=True, exist_ok=True)
log_path = Path(snakemake.log[0])  # type: ignore[name-defined]
log_path.parent.mkdir(parents=True, exist_ok=True)
output_dir = Path(snakemake.params.output_dir)  # type: ignore[name-defined]

skip_existing = _existing_manifest_is_valid(manifest_path, output_dir)
if skip_existing:
    log_path.write_text(
        "Organize stage skipped after manifest and course-contract validation.\n",
        encoding="utf-8",
    )
else:
    # Revoke the prior publication before executing the producer. If execution
    # fails, neither stale manifest bytes nor stale success flags survive.
    manifest_path.unlink(missing_ok=True)
    _revoke_all_organized_flags(output_dir)
    command = [
        str(snakemake.params.python),  # type: ignore[name-defined]
        "-m",
        "rtpipeline.cli",
        "--dicom-root",
        str(snakemake.params.dicom_root),  # type: ignore[name-defined]
        "--outdir",
        str(output_dir),
        "--logs",
        str(snakemake.params.logs_dir),  # type: ignore[name-defined]
        "--stage",
        "organize",
        "--max-workers",
        str(max(1, int(snakemake.threads))),  # type: ignore[name-defined]
    ]
    custom_structures = str(snakemake.params.custom_structures)  # type: ignore[name-defined]
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
            env=_runtime_environment(),
        )

    ledger, courses = _validated_courses_from_ledger(
        output_dir,
        prioritize_short_courses=bool(  # type: ignore[name-defined]
            snakemake.params.prioritize_short_courses
        ),
    )
    for course in courses:
        _write_text_atomic(Path(course["path"]) / ".organized", "ok\n")
    payload = _manifest_payload(ledger, courses)
    _write_text_atomic(manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
