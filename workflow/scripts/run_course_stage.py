"""Run one fail-soft per-course RTpipeline CLI stage from Snakemake."""

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


sentinel_path = Path(snakemake.output.sentinel)  # type: ignore[name-defined]
sentinel_path.parent.mkdir(parents=True, exist_ok=True)
log_path = Path(snakemake.log[0])  # type: ignore[name-defined]
log_path.parent.mkdir(parents=True, exist_ok=True)

segmentation = getattr(snakemake.input, "segmentation", None)  # type: ignore[name-defined]
if segmentation:
    try:
        segmentation_status = Path(str(segmentation)).read_text(encoding="utf-8").strip().lower()
    except Exception:
        segmentation_status = ""
    if segmentation_status.startswith("failed"):
        sentinel_path.write_text(
            "skipped: upstream segmentation failed\n", encoding="utf-8"
        )
        raise SystemExit(0)

command = [
    str(snakemake.params.python),  # type: ignore[name-defined]
    "-m",
    "rtpipeline.cli",
    "--dicom-root",
    str(snakemake.params.dicom_root),  # type: ignore[name-defined]
    "--outdir",
    str(snakemake.params.output_dir),  # type: ignore[name-defined]
    "--logs",
    str(snakemake.params.logs_dir),  # type: ignore[name-defined]
    "--stage",
    str(snakemake.params.stage),  # type: ignore[name-defined]
    "--course-filter",
    f"{snakemake.wildcards.patient}/{snakemake.wildcards.course}",  # type: ignore[name-defined]
    "--manifest",
    str(snakemake.input.manifest),  # type: ignore[name-defined]
    "--max-workers",
    str(max(1, int(snakemake.threads))),  # type: ignore[name-defined]
]
custom_structures = str(snakemake.params.custom_structures)  # type: ignore[name-defined]
if custom_structures:
    command.extend(["--custom-structures", custom_structures])

with log_path.open("w", encoding="utf-8") as log_file:
    result = subprocess.run(
        command,
        check=False,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=_runtime_environment(),
    )

if result.returncode == 0:
    sentinel_path.write_text("ok\n", encoding="utf-8")
else:
    sentinel_path.write_text("failed: see log\n", encoding="utf-8")
