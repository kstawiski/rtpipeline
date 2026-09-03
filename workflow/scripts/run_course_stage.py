"""Run one fail-closed per-course RTpipeline CLI stage from Snakemake."""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_ledger

from rtpipeline.snakemake_delegate import invoke, runtime_environment


def _require_upstream_status(
    value: object, label: str, allowed_statuses: set[str]
) -> None:
    if not value:
        return
    path = Path(str(value))
    try:
        text = path.read_text(encoding="utf-8").strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        status = (
            str(decoded.get("status", "")).strip().lower()
            if isinstance(decoded, dict)
            else text.lower()
        )
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


def _require_segmentation_content(workflow: Any, course_dir: Path) -> str:
    """Validate segmentation content inside the dependency-bearing interpreter."""

    payload = invoke(
        python=str(workflow.params.python),
        operation="assess-segmentation",
        arguments=("--course-dir", str(course_dir)),
        result_dir=Path(workflow.log[0]).parent,
        env=runtime_environment(workflow.params),
    )
    expected_course = str(course_dir.resolve(strict=False))
    if payload.get("course_dir") != expected_course:
        raise RuntimeError(
            "Segmentation assessment returned a mismatched course identity"
        )
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        raise RuntimeError("Segmentation assessment returned no structured outcome")
    status = str(outcome.get("status") or "").strip().lower()
    if status not in {"disabled", "ok"}:
        reasons = outcome.get("reasons")
        detail = (
            "; ".join(str(reason) for reason in reasons)
            if isinstance(reasons, list)
            else str(reasons)
        )
        raise RuntimeError(
            f"Required upstream segmentation content is not successful: {course_dir} "
            f"(status={outcome.get('status')!r}; reasons={detail})"
        )
    return status


def _publish_sentinel(path: Path, status: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(f"{status}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_radiomics_completion(
    workflow: Any, course_dir: Path, sentinel_path: Path
) -> None:
    payload = invoke(
        python=str(workflow.params.python),
        operation="publish-radiomics-completion",
        arguments=(
            "--course-dir",
            str(course_dir),
            "--sentinel-path",
            str(sentinel_path),
        ),
        result_dir=Path(workflow.log[0]).parent,
        env=runtime_environment(workflow.params),
    )
    expected_path = str(sentinel_path.resolve(strict=False))
    sentinel = payload.get("sentinel")
    if (
        payload.get("course_dir") != str(course_dir.resolve(strict=False))
        or payload.get("sentinel_path") != expected_path
        or not isinstance(sentinel, dict)
        or sentinel.get("status") != "ok"
    ):
        raise RuntimeError(
            "Radiomics completion validation returned a mismatched structured result"
        )
    try:
        observed = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Radiomics completion sentinel is unreadable after publication: "
            f"{sentinel_path}: {exc}"
        ) from exc
    if observed != sentinel:
        raise RuntimeError(
            "Radiomics completion sentinel differs from the validated delegated result"
        )


def _publish_stage_completion(
    workflow: Any,
    course_dir: Path,
    sentinel_path: Path,
    *,
    status: str,
) -> None:
    configuration = getattr(workflow.input, "configuration", None)
    if not configuration:
        raise RuntimeError(
            f"Stage {workflow.params.stage} has no configuration dependency"
        )
    if status == "disabled":
        from rtpipeline.stage_completion import write_stage_completion_sentinel

        payload = write_stage_completion_sentinel(
            course_dir,
            sentinel_path,
            stage=str(workflow.params.stage),
            status=status,
            configuration_dependency=Path(str(configuration)),
        )
        if payload.get("status") != status:
            raise RuntimeError("Disabled-stage completion validation returned a mismatch")
        return
    payload = invoke(
        python=str(workflow.params.python),
        operation="publish-stage-completion",
        arguments=(
            "--course-dir",
            str(course_dir),
            "--sentinel-path",
            str(sentinel_path),
            "--stage",
            str(workflow.params.stage),
            "--status",
            status,
            "--configuration-dependency",
            str(configuration),
        ),
        result_dir=Path(workflow.log[0]).parent,
        env=runtime_environment(workflow.params),
    )
    if (
        payload.get("course_dir") != str(course_dir.resolve(strict=False))
        or payload.get("sentinel_path") != str(sentinel_path.resolve(strict=False))
        or payload.get("status") != status
        or not isinstance(payload.get("output_count"), int)
        or not str(payload.get("output_set_sha256") or "")
    ):
        raise RuntimeError(
            "Stage completion validation returned a mismatched structured result"
        )


def _producer_terminal_status(path: Path) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()[0].strip().lower() if text else ""
    if isinstance(payload, dict):
        return str(payload.get("status") or "").strip().lower()
    return ""


def main(workflow: Any) -> None:
    sentinel_path = Path(workflow.output.sentinel)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.unlink(missing_ok=True)
    log_path = Path(workflow.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stage_name = str(workflow.params.stage)
    patient_id = str(workflow.wildcards.patient)
    course_id = str(workflow.wildcards.course)
    ledger_root = Path(workflow.params.output_dir)
    started_at = campaign_ledger._utcnow()
    started_monotonic = time.monotonic()

    def record(
        status: str, returncode: int | None = None, detail: str | None = None
    ) -> None:
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
        except Exception as exc:
            print(
                f"[campaign-ledger] could not record {stage_name}: {exc}",
                file=sys.stderr,
            )

    def close_course(
        detail: str,
        returncode: int,
        strict_error: BaseException | None = None,
    ) -> NoReturn:
        """Close this course while allowing an explicit campaign to continue."""

        record(campaign_ledger.STATUS_FAILED, returncode=returncode, detail=detail)
        print(detail, file=sys.stderr)
        if bool(getattr(workflow.params, "campaign_mode", False)):
            _publish_sentinel(sentinel_path, "failed")
            print(
                f"[campaign-mode] course {patient_id}/{course_id} closed at stage "
                f"{stage_name}; campaign continues and the ledger records the failure.",
                file=sys.stderr,
            )
            raise SystemExit(0)
        sentinel_path.unlink(missing_ok=True)
        if strict_error is not None:
            raise strict_error
        raise SystemExit(returncode or 1)

    for input_name, allowed_statuses in (
        ("organized", {"ok"}),
        ("segmentation", {"disabled", "ok"}),
        ("custom", {"disabled", "ok"}),
        ("crop", {"disabled", "ok"}),
    ):
        upstream = getattr(workflow.input, input_name, None)
        try:
            _require_upstream_status(upstream, input_name, allowed_statuses)
        except RuntimeError as exc:
            close_course(str(exc), returncode=1, strict_error=exc)

    if not bool(getattr(workflow.params, "enabled", True)):
        try:
            _publish_stage_completion(
                workflow,
                ledger_root / patient_id / course_id,
                sentinel_path,
                status="disabled",
            )
        except Exception as exc:
            error = RuntimeError(
                f"Disabled-stage completion validation failed for "
                f"{patient_id}/{course_id}: {exc}"
            )
            close_course(str(error), returncode=1, strict_error=error)
        record(campaign_ledger.STATUS_OK, returncode=0, detail="stage disabled")
        return

    segmentation = getattr(workflow.input, "segmentation", None)
    if segmentation:
        try:
            segmentation_status = _require_segmentation_content(
                workflow, ledger_root / patient_id / course_id
            )
        except RuntimeError as exc:
            close_course(str(exc), returncode=1, strict_error=exc)
        if segmentation_status == "disabled":
            try:
                _publish_stage_completion(
                    workflow,
                    ledger_root / patient_id / course_id,
                    sentinel_path,
                    status="disabled",
                )
            except Exception as exc:
                error = RuntimeError(
                    f"Not-applicable-stage completion validation failed for "
                    f"{patient_id}/{course_id}: {exc}"
                )
                close_course(str(error), returncode=1, strict_error=error)
            record(
                campaign_ledger.STATUS_OK,
                returncode=0,
                detail="stage not applicable because no planning CT is declared",
            )
            return

    command = [
        str(workflow.params.python),
        "-m",
        "rtpipeline.cli",
        "--dicom-root",
        str(workflow.params.dicom_root),
        "--outdir",
        str(workflow.params.output_dir),
        "--logs",
        str(workflow.params.logs_dir),
        "--stage",
        str(workflow.params.stage),
        "--course-filter",
        f"{patient_id}/{course_id}",
        "--manifest",
        str(workflow.input.manifest),
        "--max-workers",
        str(max(1, int(workflow.threads))),
    ]
    custom_structures = str(workflow.params.custom_structures)
    if custom_structures:
        command.extend(["--custom-structures", custom_structures])
    extra_args = str(getattr(workflow.params, "extra_args", "") or "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))

    stage_environment = runtime_environment(workflow.params)
    if stage_name == "radiomics":
        worker_limit = str(max(1, int(workflow.threads)))
        stage_environment.update(
            {
                "RTPIPELINE_MAX_WORKERS": worker_limit,
                "RTPIPELINE_RADIOMICS_THREAD_LIMIT": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=stage_environment,
            )
    except OSError as exc:
        close_course(
            f"Stage {stage_name} could not be launched: {exc}",
            returncode=127,
        )

    if result.returncode != 0:
        close_course(
            f"Required stage {stage_name} failed with exit code "
            f"{result.returncode}; see {log_path}",
            returncode=result.returncode,
        )

    if stage_name == "radiomics":
        course_dir = ledger_root / patient_id / course_id
        parquet_path = course_dir / "radiomics_ct.parquet"
        workbook_path = course_dir / "radiomics_ct.xlsx"
        if workbook_path.exists() and not parquet_path.exists():
            error = RuntimeError(
                f"Radiomics stage produced CT Excel without authoritative Parquet: "
                f"{course_dir}"
            )
            close_course(str(error), returncode=1, strict_error=error)
        if not parquet_path.exists():
            error = RuntimeError(
                f"Radiomics stage completed without authoritative CT Parquet: "
                f"{course_dir}"
            )
            close_course(str(error), returncode=1, strict_error=error)
        try:
            _publish_radiomics_completion(workflow, course_dir, sentinel_path)
        except Exception as exc:
            error = RuntimeError(
                f"Radiomics completion validation failed for {course_dir}: {exc}"
            )
            close_course(str(error), returncode=1, strict_error=error)
    else:
        course_dir = ledger_root / patient_id / course_id
        completion_status = (
            "disabled"
            if stage_name == "segmentation"
            and _producer_terminal_status(sentinel_path) == "disabled"
            else "ok"
        )
        try:
            _publish_stage_completion(
                workflow,
                course_dir,
                sentinel_path,
                status=completion_status,
            )
        except Exception as exc:
            error = RuntimeError(
                f"Stage completion validation failed for {course_dir}: {exc}"
            )
            close_course(str(error), returncode=1, strict_error=error)
    record(campaign_ledger.STATUS_OK, returncode=0)


if "snakemake" in globals():
    main(snakemake)  # type: ignore[name-defined]
