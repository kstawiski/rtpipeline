"""Materialize the organized-course manifest for the Snakemake checkpoint."""

import json
import os
import subprocess
from pathlib import Path


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


def _iter_course_dirs(output_dir: Path):
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


manifest_path = Path(snakemake.output.manifest)  # type: ignore[name-defined]
manifest_path.parent.mkdir(parents=True, exist_ok=True)
log_path = Path(snakemake.log[0])  # type: ignore[name-defined]
log_path.parent.mkdir(parents=True, exist_ok=True)
output_dir = Path(snakemake.params.output_dir)  # type: ignore[name-defined]

skip_existing = False
if manifest_path.exists():
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        course_entries = manifest_data.get("courses", [])
    except Exception:
        course_entries = []
    if course_entries:
        missing_flags = []
        for entry in course_entries:
            try:
                course_dir = Path(entry.get("path", ""))
            except (TypeError, ValueError):
                missing_flags.append(entry)
                continue
            if not (course_dir / ".organized").exists():
                missing_flags.append(entry)
        skip_existing = not missing_flags

if skip_existing:
    log_path.write_text(
        "Organize stage skipped (manifest already present).\n", encoding="utf-8"
    )
else:
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

    courses = []
    for patient_id, course_id, course_path in _iter_course_dirs(output_dir):
        flag = course_path / ".organized"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("ok\n", encoding="utf-8")
        courses.append(
            {
                "patient": patient_id,
                "course": course_id,
                "path": str(course_path),
                "complexity": _estimate_course_complexity(course_path),
            }
        )
    if bool(snakemake.params.prioritize_short_courses):  # type: ignore[name-defined]
        courses.sort(
            key=lambda entry: (
                entry.get("complexity", 0),
                entry["patient"],
                entry["course"],
            )
        )
    manifest_path.write_text(
        json.dumps({"courses": courses}, indent=2), encoding="utf-8"
    )
