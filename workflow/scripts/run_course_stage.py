"""Run one fail-closed per-course RTpipeline CLI stage from Snakemake."""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_ledger


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


def _require_upstream_status(
    value: object, label: str, allowed_statuses: set[str]
) -> None:
    if not value:
        return
    path = Path(str(value))
    try:
        status = path.read_text(encoding="utf-8").strip().lower()
    except Exception as exc:
        raise RuntimeError(
            f"Required upstream {label} sentinel is unreadable: {path}: {exc}"
        ) from exc
    if status not in allowed_statuses:
        allowed = ", ".join(sorted(allowed_statuses))
        raise RuntimeError(
            f"Required upstream {label} sentinel is not successful: {path} "
            f"(status={status!r}; expected {allowed})"
        )


def _publish_success_sentinel(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text("ok\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


sentinel_path = Path(snakemake.output.sentinel)  # type: ignore[name-defined]
sentinel_path.parent.mkdir(parents=True, exist_ok=True)
sentinel_path.unlink(missing_ok=True)
log_path = Path(snakemake.log[0])  # type: ignore[name-defined]
log_path.parent.mkdir(parents=True, exist_ok=True)

stage_name = str(snakemake.params.stage)  # type: ignore[name-defined]
patient_id = str(snakemake.wildcards.patient)  # type: ignore[name-defined]
course_id = str(snakemake.wildcards.course)  # type: ignore[name-defined]
ledger_root = Path(snakemake.params.output_dir)  # type: ignore[name-defined]
started_at = campaign_ledger._utcnow()
started_monotonic = time.monotonic()


def _record(status: str, returncode: int | None = None, detail: str | None = None) -> None:
    try:
        campaign_ledger.record(
            ledger_root,
            patient_id,
            course_id,
            stage_name,
            status,
            returncode=returncode,
            log_path=str(log_path),
            detail=detail,
            started_at=started_at,
            duration_seconds=round(time.monotonic() - started_monotonic, 3),
        )
    except Exception as exc:  # ledger failure must not mask the stage outcome
        print(f"[campaign-ledger] could not record {stage_name}: {exc}", file=sys.stderr)

for input_name, allowed_statuses in (
    ("segmentation", {"ok"}),
    ("custom", {"disabled", "ok"}),
    ("crop", {"ok"}),
):
    upstream = getattr(snakemake.input, input_name, None)  # type: ignore[name-defined]
    try:
        _require_upstream_status(upstream, input_name, allowed_statuses)
    except RuntimeError as exc:
        _record(campaign_ledger.STATUS_FAILED, detail=str(exc))
        raise

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

if result.returncode != 0:
    sentinel_path.unlink(missing_ok=True)
    _record(
        campaign_ledger.STATUS_FAILED,
        returncode=result.returncode,
        detail=f"stage exited with code {result.returncode}",
    )
    print(
        f"Required stage {stage_name} failed with exit code "
        f"{result.returncode}; see {log_path}",
        file=sys.stderr,
    )
    raise SystemExit(result.returncode or 1)

_publish_success_sentinel(sentinel_path)
_record(campaign_ledger.STATUS_OK, returncode=0)
