"""
Parallel radiomics processing with process isolation.

Goals
-----
- Avoid PyRadiomics/OpenMP instability under threads by using processes.
- Avoid the previous implementation that wrote the full CT volume per ROI task.
- Apply ROI skip lists + voxel limits consistently with the conda fallback.

High-level design
-----------------
1) Parent enumerates tasks as (segmentation_source, RTSTRUCT path, ROI name).
2) Worker processes:
   - Load CT SimpleITK image once.
   - Build/Cache RTStructBuilder per RTSTRUCT path.
   - Reuse a single RadiomicsFeatureExtractor instance per process.
3) Per-ROI timeouts are enforced by restarting the pool (and terminating worker
   processes) whenever any ROI exceeds the timeout.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pydicom

from .layout import build_course_dirs
from .course_contract import ALL_SERIES_RADIOMICS_TEMP_SCOPE, load_course_contract
from .utils import mask_is_cropped, radiomics_mp_context
from .custom_models import (
    list_custom_model_outputs,
    validate_custom_model_output_inventory,
)
from .custom_structures_rtstruct import _create_custom_structures_rtstruct, _is_rs_custom_stale
from .radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    invalidate_radiomics_outputs as _invalidate_radiomics_outputs,
    remove_artifact_strict as _remove_artifact_strict,
    resume_identity_pairs as _resume_identity_pairs,
    write_excel_atomic as _write_excel_atomic,
)

logger = logging.getLogger(__name__)


class RadiomicsRegionExtractionError(RuntimeError):
    """A required ROI could not be extracted, so the course is incomplete."""


_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_THREAD_LIMIT_ENV = "RTPIPELINE_RADIOMICS_THREAD_LIMIT"
_TASK_TIMEOUT_ENV = "RTPIPELINE_RADIOMICS_TASK_TIMEOUT"
_ROI_TIMEOUT_REFERENCE_VOXELS = 100_000
_ROI_TIMEOUT_MAX_MULTIPLIER = 6


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _scaled_roi_timeout(base_timeout: int, estimated_voxels: float) -> int:
    """Scale an ROI budget by estimated post-resampling foreground work.

    ``base_timeout`` remains the floor used by small ROIs. Above 100k estimated
    voxels the budget grows linearly, capped at six times the floor.
    """
    floor = max(1, int(base_timeout))
    work = max(0.0, float(estimated_voxels))
    scaled = math.ceil(floor * max(1.0, work / _ROI_TIMEOUT_REFERENCE_VOXELS))
    return min(floor * _ROI_TIMEOUT_MAX_MULTIPLIER, scaled)


def _resolve_thread_limit(explicit: Optional[int] = None) -> Optional[int]:
    """Resolve thread limit from env or explicit value.

    Environment variable takes precedence (so Snakemake can force a limit).
    """
    env_value = _coerce_positive_int(os.environ.get(_THREAD_LIMIT_ENV))
    if env_value is not None:
        return env_value
    return _coerce_positive_int(explicit)


def _apply_thread_limit(limit: Optional[int]) -> None:
    if limit is None:
        for var in _THREAD_VARS:
            os.environ.pop(var, None)
        return
    limit = max(1, int(limit))
    value = str(limit)
    for var in _THREAD_VARS:
        os.environ[var] = value


def configure_parallel_radiomics(thread_limit: Optional[int] = None) -> None:
    _apply_thread_limit(_resolve_thread_limit(thread_limit))


def enable_parallel_radiomics_processing(thread_limit: Optional[int] = None) -> None:
    """Enable the parallel radiomics backend.

    Note: If RTPIPELINE_RADIOMICS_THREAD_LIMIT is already set (e.g. by Snakemake),
    it is respected and not overridden.
    """
    os.environ["RTPIPELINE_USE_PARALLEL_RADIOMICS"] = "1"

    env_existing = _coerce_positive_int(os.environ.get(_THREAD_LIMIT_ENV))
    explicit = _coerce_positive_int(thread_limit)
    if env_existing is None and explicit is not None:
        os.environ[_THREAD_LIMIT_ENV] = str(explicit)

    configure_parallel_radiomics(thread_limit)
    logger.info("Enabled parallel radiomics processing")


def is_parallel_radiomics_enabled() -> bool:
    return os.environ.get("RTPIPELINE_USE_PARALLEL_RADIOMICS", "").lower() in ("1", "true", "yes")


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch in {"_", "-"}).strip("_-")


def _default_skip_rois() -> Set[str]:
    # Matches the conda implementation defaults.
    return {
        "couchsurface",
        "couchinterior",
        "couchexterior",
        "bones",
        "m1",
        "m2",
        "table",
        "support",
    }


def _derive_voxel_limits(config: Any) -> Tuple[int, int]:
    max_voxels = getattr(config, "radiomics_max_voxels", None)
    max_voxels = 15_000_000 if max_voxels in (None, "") else int(max_voxels)
    if max_voxels < 1:
        max_voxels = 15_000_000

    min_voxels = getattr(config, "radiomics_min_voxels", None)
    min_voxels = 120 if min_voxels in (None, "") else int(min_voxels)
    if min_voxels < 1:
        min_voxels = 1

    return min_voxels, max_voxels


def _calculate_optimal_workers() -> int:
    """Memory-aware worker heuristic.

    Uses RTPIPELINE_MAX_WORKERS as a hard override when set.
    """
    env_override = _coerce_positive_int(os.environ.get("RTPIPELINE_MAX_WORKERS"))
    if env_override is not None:
        return env_override

    cpu_count = os.cpu_count() or 2
    cpu_based = max(1, cpu_count - 1)

    memory_per_worker_gb = float(os.environ.get("RTPIPELINE_MEMORY_PER_WORKER_GB", "2.0"))
    available_gb: Optional[float] = None
    try:
        import psutil  # type: ignore

        available_gb = float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        available_kb = int(line.split()[1])
                        available_gb = available_kb / (1024**2)
                        break
        except Exception:
            available_gb = None

    if available_gb is None:
        return cpu_based

    usable_gb = max(0.0, available_gb - 2.0)  # leave headroom
    memory_based = max(1, int(usable_gb / max(0.25, memory_per_worker_gb)))
    return max(1, min(cpu_based, memory_based))


@dataclass(frozen=True, slots=True)
class _RoiTask:
    source: str
    rs_path: str
    roi_name: str
    course_dir: str


_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(
    ct_dir: str,
    config: Any,
    thread_limit: Optional[int],
    skip_rois: Set[str],
    min_voxels: int,
    max_voxels: int,
    base_timeout: int,
) -> None:
    # Apply OpenMP/BLAS thread limits inside each worker.
    _apply_thread_limit(_resolve_thread_limit(thread_limit) or 1)

    from .radiomics import _extractor, _extractor_large_roi, _load_series_image

    ct_path = Path(ct_dir)
    img = _load_series_image(ct_path)
    ext = _extractor(config, "CT")
    ext_large = _extractor_large_roi(config, "CT")

    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "ct_dir": ct_path,
            "img": img,
            "extractor": ext,
            "extractor_large": ext_large,
            "builders": {},
            "skip_rois": set(skip_rois),
            "min_voxels": int(min_voxels),
            "max_voxels": int(max_voxels),
            "base_timeout": int(base_timeout),
        }
    )


def _get_builder(rs_path: Path):
    builders: Dict[str, Any] = _WORKER_STATE.setdefault("builders", {})
    key = str(rs_path)
    if key in builders:
        return builders[key]

    try:
        from rt_utils import RTStructBuilder  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("rt-utils missing for radiomics worker: %s", exc)
        builders[key] = None
        return None

    ct_dir = _WORKER_STATE.get("ct_dir")
    if ct_dir is None:
        builders[key] = None
        return None

    try:
        builder = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rs_path))
    except Exception as exc:
        logger.debug("RTStructBuilder create_from failed for %s: %s", rs_path, exc)
        builder = None
    builders[key] = builder
    return builder


def _status_record(
    task: _RoiTask,
    status: str,
    detail: str,
    *,
    voxel_count: Optional[int] = None,
) -> Dict[str, Any]:
    course_dir = Path(task.course_dir)
    return {
        "modality": "CT",
        "segmentation_source": task.source,
        "roi_name": task.roi_name,
        "roi_original_name": task.roi_name,
        "course_dir": str(course_dir),
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "structure_cropped": False,
        "extraction_status": status,
        "extraction_status_detail": detail,
        "voxel_count": voxel_count,
    }


def _extract_one(task: _RoiTask) -> Dict[str, Any]:
    skip_rois: Set[str] = _WORKER_STATE.get("skip_rois", set())
    if _norm(task.roi_name) in skip_rois:
        return _status_record(task, "declared_skip", "ROI is listed in radiomics_skip_rois")

    img = _WORKER_STATE.get("img")
    ext = _WORKER_STATE.get("extractor")
    if img is None or ext is None:
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} cannot be extracted because the image or radiomics extractor is unavailable"
        )

    from .radiomics import _mask_from_array_like

    rs_path = Path(task.rs_path)
    builder = _get_builder(rs_path)
    if builder is None:
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} cannot be read because the structure builder is unavailable for {rs_path}"
        )

    try:
        mask = builder.get_roi_mask_by_name(task.roi_name)
    except Exception as exc:
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} mask could not be read from {rs_path}: {exc}"
        ) from exc
    if mask is None:
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} structure builder did not provide a mask from {rs_path}"
        )

    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} from {task.source} produced an empty required mask"
        )

    voxel_count = int(mask_bool.sum())
    min_voxels = int(_WORKER_STATE.get("min_voxels", 120))
    max_voxels = int(_WORKER_STATE.get("max_voxels", 15_000_000))
    if voxel_count < min_voxels:
        return _status_record(
            task,
            "below_minimum_voxels",
            f"ROI contains {voxel_count} voxels; configured minimum is {min_voxels}",
            voxel_count=voxel_count,
        )
    # Decide "large ROI" using an estimate at the extractor resampled spacing.
    #
    # Rationale: voxel_count is measured at native CT spacing (often 1×1×3mm). The
    # default CT radiomics config resamples to 1mm isotropic, which can inflate the
    # effective voxel count ~3× and cause timeouts for big ROIs (especially BODY)
    # even when native voxel_count is below the threshold.
    is_body = _norm(task.roi_name).startswith("body")
    try:
        spacing = tuple(float(x) for x in img.GetSpacing())
    except Exception:
        spacing = (1.0, 1.0, 1.0)
    native_voxel_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    physical_volume_mm3 = float(voxel_count) * max(1e-9, native_voxel_mm3)
    try:
        resampled = ext.settings.get("resampledPixelSpacing") or spacing
        resampled = tuple(float(x) for x in resampled)
    except Exception:
        resampled = spacing
    resampled_voxel_mm3 = float(resampled[0]) * float(resampled[1]) * float(resampled[2])
    estimated_voxels = physical_volume_mm3 / max(1e-9, resampled_voxel_mm3)
    if is_body or estimated_voxels > float(max_voxels):
        ext_large = _WORKER_STATE.get("extractor_large")
        if ext_large is None:
            raise RadiomicsRegionExtractionError(
                f"ROI {task.roi_name} requires the large-ROI extractor, but it is unavailable"
            )
        ext = ext_large

    cropped = mask_is_cropped(mask_bool)
    display_roi = task.roi_name if (not cropped or task.roi_name.endswith("__partial")) else f"{task.roi_name}__partial"

    roi_timeout = _scaled_roi_timeout(
        int(_WORKER_STATE.get("base_timeout", 600)),
        estimated_voxels,
    )

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(
            f"ROI {task.roi_name} exceeded its {roi_timeout}s radiomics budget "
            f"({estimated_voxels:.0f} estimated resampled voxels)"
        )

    use_alarm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler) if use_alarm else None
    if use_alarm:
        signal.alarm(roi_timeout)
    try:
        mask_img = _mask_from_array_like(img, mask_bool)
        res = ext.execute(img, mask_img)
    except TimeoutError as exc:
        raise RadiomicsRegionExtractionError(str(exc)) from exc
    except Exception as exc:
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} radiomics extraction failed: {exc}"
        ) from exc
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    rec: Dict[str, Any] = {}
    for k, v in res.items():
        try:
            rec[k] = float(v)  # numpy scalars
        except Exception:
            rec[k] = str(v)

    course_dir = Path(task.course_dir)
    rec.update(
        {
            "modality": "CT",
            "segmentation_source": task.source,
            "roi_name": display_roi,
            "roi_original_name": task.roi_name,
            "course_dir": str(course_dir),
            "patient_id": course_dir.parent.name,
            "course_id": course_dir.name,
            "structure_cropped": bool(cropped),
        }
    )
    return rec


def _list_roi_names(rs_path: Path) -> List[str]:
    try:
        ds = pydicom.dcmread(str(rs_path), stop_before_pixels=True, force=True)
    except Exception as exc:
        raise RadiomicsCourseExtractionError(f"Failed to read RTSTRUCT {rs_path}: {exc}") from exc
    out: List[str] = []
    for roi in getattr(ds, "StructureSetROISequence", []) or []:
        name = str(getattr(roi, "ROIName", "") or "").strip()
        if name:
            out.append(name)
    if not out:
        raise RadiomicsCourseExtractionError(f"RTSTRUCT contains no named ROIs: {rs_path}")
    return out


def _current_child_pids() -> Set[int]:
    try:
        import psutil  # type: ignore

        return {proc.pid for proc in psutil.Process().children(recursive=False)}
    except Exception:
        return set()


def _terminate_executor_processes(executor: ProcessPoolExecutor, *, baseline_child_pids: Optional[Set[int]] = None) -> None:
    """Best-effort worker termination for timeout recovery.

    Notes:
    - `ProcessPoolExecutor` does not expose worker PIDs via a public API.
    - We use a best-effort approach: try to terminate known worker processes and
      their children (if `psutil` is available), otherwise fall back to `os.kill`.
    """
    pids: List[int] = []

    processes = getattr(executor, "_processes", None)
    if processes:
        for proc in list(processes.values()):
            pid = getattr(proc, "pid", None)
            if isinstance(pid, int) and pid > 0:
                pids.append(pid)
            try:
                proc.terminate()
            except Exception:
                pass

        for proc in list(processes.values()):
            try:
                proc.join(timeout=5)
            except Exception:
                pass

    if not pids and baseline_child_pids:
        # Fallback: identify new children created while this executor was active.
        try:
            current = _current_child_pids()
            pids = sorted(pid for pid in (current - set(baseline_child_pids)) if pid > 0)
        except Exception:
            pids = []
    if not pids:
        return

    # Escalate to SIGKILL if still alive (and clean up child processes).
    try:
        import psutil  # type: ignore

        targets = []
        for pid in pids:
            try:
                targets.append(psutil.Process(pid))
            except Exception:
                continue
        if not targets:
            return

        children = []
        for proc in targets:
            try:
                children.extend(proc.children(recursive=True))
            except Exception:
                pass

        for proc in children + targets:
            try:
                proc.terminate()
            except Exception:
                pass

        _, alive = psutil.wait_procs(children + targets, timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        import signal

        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def _submit_lazy(
    executor: ProcessPoolExecutor,
    pending: Iterator[Any],
    submit_fn: Any,
    futures: Dict[Any, Any],
    task_start: Dict[Any, float],
    count: int,
) -> List[Any]:
    """Submit up to ``count`` more tasks from ``pending``, stamping each future's
    start time at the moment it is actually submitted rather than when a whole
    batch is queued upfront. Submitting everything upfront timestamps tasks that
    sit queued behind the executor's worker slots, inflating timeout measurements
    by their queue-wait time. Mutates ``futures``/``task_start`` in place; returns
    the newly created futures.
    """
    new_futures = []
    for _ in range(count):
        task = next(pending, None)
        if task is None:
            break
        fut = executor.submit(submit_fn, task)
        futures[fut] = task
        task_start[fut] = time.monotonic()
        new_futures.append(fut)
    return new_futures


def parallel_radiomics_for_course(
    config: Any,
    course_dir: Path,
    custom_structures_config: Optional[Path] = None,
    max_workers: Optional[int] = None,
    use_cropped: bool = False,
    *,
    allow_all_series_temp: bool = False,
) -> RadiomicsCourseOutcome:
    """Parallel CT radiomics for one course (process isolation)."""
    course_dir = Path(course_dir)
    contract = load_course_contract(course_dir)
    is_temp = contract.data.get("scope") == ALL_SERIES_RADIOMICS_TEMP_SCOPE
    if is_temp and (
        not allow_all_series_temp or ".all_series_radiomics" not in course_dir.parts
    ):
        raise RadiomicsCourseExtractionError(
            "all-series temporary contract is restricted to the all-series dispatcher"
        )
    if allow_all_series_temp and not is_temp:
        raise RadiomicsCourseExtractionError(
            "all-series dispatcher requires an all-series temporary contract"
        )
    out_path = course_dir / "radiomics_ct.xlsx"
    course_dirs = build_course_dirs(course_dir)
    ct_dir = contract.planning_ct_dir
    ct_files_present = ct_dir is not None
    if not ct_files_present:
        logger.info("No CT image for radiomics in %s", course_dir)
        _invalidate_radiomics_outputs(out_path)
        return RadiomicsCourseOutcome.nothing_to_do("CT series is absent")
    assert ct_dir is not None
    existing_df = None
    if getattr(config, "resume", False) and out_path.exists():
        try:
            import pandas as pd  # type: ignore

            existing_df = pd.read_excel(out_path, engine="openpyxl")
            existing_df = existing_df.drop(
                columns=["extraction_status", "extraction_status_detail", "voxel_count"],
                errors="ignore",
            )
            _resume_identity_pairs(existing_df)
        except Exception as exc:
            logger.warning(
                "Invalidating unusable parallel resume workbook for %s: %s",
                course_dir,
                exc,
            )
            _invalidate_radiomics_outputs(out_path)
            existing_df = None

    try:
        from .radiomics import _extractor, _load_series_image
    except Exception as exc:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Failed importing radiomics extractor for {course_dir}: {exc}"
        ) from exc

    if _load_series_image(ct_dir) is None:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"CT series is present but unreadable for radiomics in {course_dir}"
        )

    # If we can't build a native extractor, delegate to conda backend.
    if _extractor(config, "CT") is None:
        try:
            from .radiomics_conda import radiomics_for_course as conda_radiomics_for_course
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Conda-based radiomics helper unavailable for {course_dir}: {exc}"
            ) from exc
        conda_out = conda_radiomics_for_course(
            course_dir,
            config,
            str(custom_structures_config) if custom_structures_config else None,
            allow_all_series_temp=allow_all_series_temp,
        )
        if conda_out is None:
            _invalidate_radiomics_outputs(out_path)
            return RadiomicsCourseOutcome.nothing_to_do("conda backend found no eligible ROIs")
        return RadiomicsCourseOutcome.extracted(conda_out)

    # Choose RS_auto vs RS_auto_cropped
    rs_auto_name = "RS_auto.dcm"
    if use_cropped:
        rs_auto_cropped = course_dir / "RS_auto_cropped.dcm"
        crop_meta = course_dir / "cropping_metadata.json"
        if rs_auto_cropped.exists() and crop_meta.exists():
            logger.warning(
                "Ignoring RS_auto_cropped.dcm for radiomics in %s due to known geometric misregistration; "
                "using RS_auto.dcm instead.",
                course_dir,
            )

    from .radiomics import _standard_rtstruct_sources

    sources: List[Tuple[str, Path, Optional[List[str]]]] = []
    rs_manual = (
        contract.authoritative_rtstruct_path
        or course_dir / "metadata" / ".contract-rtstruct-absent"
    )
    rs_auto = course_dir / rs_auto_name
    sources.extend(_standard_rtstruct_sources(contract, course_dir))

    # Custom structures (optional): prepare RS_custom but extract only custom ROIs (no duplication)
    rs_custom = course_dir / "RS_custom.dcm"
    desired_custom: set[str] = set()
    configured_custom_value = (
        custom_structures_config
        or getattr(config, "custom_structures_config", None)
    )
    configured_custom_path: Optional[Path] = None
    if configured_custom_value:
        configured_custom_path = Path(configured_custom_value)
        if not configured_custom_path.is_file():
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Configured required custom structure file is missing: {configured_custom_path}"
            )
        try:
            from .radiomics import _custom_roi_names_from_config

            desired_custom = _custom_roi_names_from_config(configured_custom_path)
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to read custom ROI configuration for {course_dir}: {exc}"
            ) from exc
        desired_custom = {
            name for name in desired_custom
            if _norm(name) not in {
                _norm(item)
                for item in getattr(config, "radiomics_skip_rois", []) or []
                if isinstance(item, str) and item.strip()
            }
        }
        try:
            rs_auto_for_custom = course_dir / "RS_auto.dcm"
            if _is_rs_custom_stale(
                rs_custom, configured_custom_path, rs_manual, rs_auto_for_custom
            ):
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir, configured_custom_path, rs_manual, rs_auto_for_custom
                ) or rs_custom
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to prepare RS_custom for radiomics in {course_dir}: {exc}"
            ) from exc

    # Custom model RTSTRUCTs: explicit selections and current per-course output
    # directories are required; dormant definitions under custom_models_root are not.
    try:
        custom_model_expected_rois = validate_custom_model_output_inventory(
            course_dir,
            getattr(config, "custom_model_names", None),
            getattr(config, "custom_models_root", None),
        )
        for model_name, model_course_dir in list_custom_model_outputs(course_dir):
            rs_model = Path(model_course_dir) / "rtstruct.dcm"
            if rs_model.exists():
                sources.append(
                    (
                        f"CustomModel:{model_name}",
                        rs_model,
                        custom_model_expected_rois[model_name],
                    )
                )
    except Exception as exc:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Custom model output scan failed for {course_dir}: {exc}"
        ) from exc

    skip_rois = _default_skip_rois() | {
        _norm(item)
        for item in getattr(config, "radiomics_skip_rois", []) or []
        if isinstance(item, str) and item.strip()
    }

    min_voxels, max_voxels = _derive_voxel_limits(config)

    # Enumerate every current non-skipped identity before accepting a resume
    # workbook. BODY-only top-ups can miss ordinary Manual/AutoRTS/model ROIs.
    tasks: List[_RoiTask] = []
    try:
        for source, rs_path, expected_rois in sources:
            roi_names = expected_rois or _list_roi_names(rs_path)
            for roi_name in roi_names:
                if _norm(roi_name) in skip_rois:
                    continue
                tasks.append(
                    _RoiTask(
                        source=source,
                        rs_path=str(rs_path),
                        roi_name=roi_name,
                        course_dir=str(course_dir),
                    )
                )
    except Exception:
        _invalidate_radiomics_outputs(out_path)
        raise

    # Custom structures: extract every non-skipped configured ROI, or infer
    # custom-only identities from an unconfigured RS_custom.
    if desired_custom and not rs_custom.is_file():
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Required custom RTSTRUCT is missing for configured ROIs in {course_dir}"
        )
    if rs_custom.exists():
        try:
            avail = set(_list_roi_names(rs_custom))
            if configured_custom_path is None and not desired_custom:
                manual_names = set(_list_roi_names(rs_manual)) if rs_manual.exists() else set()
                auto_names = set(_list_roi_names(rs_auto)) if rs_auto.exists() else set()
                inferred = {n for n in (avail - (manual_names | auto_names)) if n}
                desired_custom = {n[:-9] if n.endswith("__partial") else n for n in inferred}

            wanted: list[str] = []
            missing_custom: list[str] = []
            for base in sorted(desired_custom):
                if base in avail:
                    wanted.append(base)
                elif f"{base}__partial" in avail:
                    wanted.append(f"{base}__partial")
                else:
                    missing_custom.append(base)
            if missing_custom:
                raise RadiomicsCourseExtractionError(
                    f"Required configured custom ROI(s) missing from {rs_custom}: "
                    + ", ".join(missing_custom)
                )
            for roi_name in wanted:
                if _norm(roi_name) in skip_rois:
                    continue
                tasks.append(
                    _RoiTask(
                        source="Custom",
                        rs_path=str(rs_custom),
                        roi_name=roi_name,
                        course_dir=str(course_dir),
                    )
                )
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            if isinstance(exc, RadiomicsCourseExtractionError):
                raise
            raise RadiomicsCourseExtractionError(
                f"Failed to enumerate custom radiomics tasks for {course_dir}: {exc}"
            ) from exc

    if existing_df is not None:
        expected_pairs = {(task.source, task.roi_name) for task in tasks}
        try:
            existing_pairs = _resume_identity_pairs(existing_df)
        except ValueError as exc:
            logger.warning(
                "Invalidating unusable parallel resume workbook for %s: %s",
                course_dir,
                exc,
            )
            existing_pairs = set()
        if expected_pairs and existing_pairs == expected_pairs:
            logger.debug(
                "Parallel radiomics resume workbook is complete for %s", course_dir
            )
            return RadiomicsCourseOutcome.extracted(out_path)
        logger.warning(
            "Invalidating incomplete parallel resume workbook for %s: expected %d "
            "source/ROI identities, found %d",
            course_dir,
            len(expected_pairs),
            len(existing_pairs),
        )
        _invalidate_radiomics_outputs(out_path)
        existing_df = None

    if not tasks:
        logger.info("No radiomics tasks for %s", course_dir)
        _invalidate_radiomics_outputs(out_path)
        return RadiomicsCourseOutcome.nothing_to_do("no RTSTRUCT ROIs were enumerated")

    if max_workers is None:
        worker_count = _calculate_optimal_workers()
    else:
        worker_count = max(1, min(int(max_workers), _calculate_optimal_workers()))
    worker_count = max(1, min(worker_count, len(tasks)))

    thread_limit = _resolve_thread_limit(getattr(config, "radiomics_thread_limit", None))
    base_task_timeout = _coerce_positive_int(os.environ.get(_TASK_TIMEOUT_ENV)) or 600
    task_timeout = base_task_timeout * _ROI_TIMEOUT_MAX_MULTIPLIER

    logger.info(
        "Parallel radiomics for %s: %d ROI task(s), %d worker(s), thread_limit=%s, "
        "ROI timeout=%d-%ds (linear above %d estimated resampled voxels)",
        course_dir.name,
        len(tasks),
        worker_count,
        thread_limit if thread_limit is not None else "env/default",
        base_task_timeout,
        task_timeout,
        _ROI_TIMEOUT_REFERENCE_VOXELS,
    )

    total = len(tasks)
    completed: Set[_RoiTask] = set()
    rows: List[Dict[str, Any]] = []

    def _invalidate_incomplete_output() -> None:
        """Remove stale tables that would otherwise look complete after failure."""
        _invalidate_radiomics_outputs(out_path)

    pending_tasks: List[_RoiTask] = list(tasks)
    start_time = time.monotonic()
    last_log = start_time

    while pending_tasks:
        # Fixed snapshot of this round's tasks. Reconstructing the next round's
        # pending_tasks as "round_tasks minus completed" (below) is robust to a
        # submit failing partway through backfill: unlike tracking remaining
        # futures/task_iter state, it doesn't depend on exactly which task was
        # being submitted when the pool broke, so nothing is silently dropped.
        round_tasks = list(pending_tasks)
        baseline_children = _current_child_pids()
        # This pool is created from WITHIN a course-level ThreadPoolExecutor worker thread;
        # a default 'fork' context inherits locked mutexes from the multi-threaded parent and
        # deadlocks. Use a forkserver/spawn context (initializer + initargs are picklable).
        pool_size = min(worker_count, len(pending_tasks))
        executor = ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=radiomics_mp_context(),
            initializer=_worker_init,
            initargs=(
                str(ct_dir), config, thread_limit, skip_rois, min_voxels,
                max_voxels, base_task_timeout,
            ),
        )
        futures: Dict[Any, _RoiTask] = {}
        task_start: Dict[Any, float] = {}
        fatal_error: Optional[RadiomicsCourseExtractionError] = None
        try:
            # Submit lazily (at most `pool_size` in flight) so task_start reflects
            # actual execution start, not queue-wait time behind busy workers.
            task_iter = iter(pending_tasks)
            remaining = set(_submit_lazy(executor, task_iter, _extract_one, futures, task_start, pool_size))
            restart = False

            while remaining:
                done, remaining = wait(remaining, timeout=5.0, return_when=FIRST_COMPLETED)
                now = time.monotonic()

                for fut in done:
                    task = futures[fut]
                    try:
                        rec = fut.result(timeout=0)
                    except BrokenProcessPool as exc:
                        logger.error(
                            "Radiomics worker pool broke while extracting %s/%s (%s); restarting",
                            task.source,
                            task.roi_name,
                            exc,
                        )
                        restart = True
                        break
                    except Exception as exc:
                        fatal_error = RadiomicsCourseExtractionError(
                            f"Radiomics course {course_dir} is incomplete: "
                            f"required ROI {task.source}/{task.roi_name} failed: {exc}"
                        )
                        break
                    if rec is None:
                        fatal_error = RadiomicsCourseExtractionError(
                            f"Radiomics course {course_dir} is incomplete: required ROI "
                            f"{task.source}/{task.roi_name} returned no outcome record"
                        )
                        break
                    completed.add(task)
                    rows.append(rec)

                if fatal_error is not None or restart:
                    break

                # Backfill: submit the next pending task(s) into the slot(s) just freed.
                #
                # A dead worker (e.g. OOM-killed) can make the pool itself broken, so
                # executor.submit() inside _submit_lazy can raise (e.g. BrokenProcessPool).
                # That must not propagate uncaught - it would abandon every unfinished
                # task in this round instead of restarting with a fresh pool. Stop
                # backfilling and restart; round_tasks-minus-completed below recovers
                # both in-flight and never-submitted tasks regardless of exactly which
                # task submission failed.
                try:
                    remaining.update(_submit_lazy(executor, task_iter, _extract_one, futures, task_start, len(done)))
                except Exception as exc:
                    logger.error(
                        "Radiomics: submit failed during backfill for %s (%s); restarting worker pool",
                        course_dir.name,
                        exc,
                    )
                    restart = True
                    break

                # Timeouts: mark tasks as skipped and restart the pool so we don't block on shutdown.
                timed_out_tasks: List[_RoiTask] = []
                for fut in list(remaining):
                    if now - task_start[fut] > task_timeout:
                        task = futures[fut]
                        timed_out_tasks.append(task)
                        completed.add(task)
                        remaining.remove(fut)
                if timed_out_tasks:
                    failed_names = ", ".join(
                        f"{task.source}/{task.roi_name}" for task in timed_out_tasks
                    )
                    fatal_error = RadiomicsCourseExtractionError(
                        f"Radiomics course {course_dir} is incomplete: required ROI task(s) "
                        f"timed out after {task_timeout}s: {failed_names}"
                    )
                    break

                if now - last_log > 30:
                    logger.info("Radiomics progress for %s: %d/%d", course_dir.name, len(completed), total)
                    last_log = now

            if fatal_error is not None:
                pending_tasks = []
            elif restart:
                # Everything in this round not yet finalized (success, failure, or
                # timeout): still-in-flight futures, tasks never submitted because
                # they were still in task_iter, AND a task lost mid-submit if the
                # pool broke during backfill.
                pending_tasks = [task for task in round_tasks if task not in completed]
            else:
                pending_tasks = []

        finally:
            if pending_tasks or fatal_error is not None:
                # Terminate workers BEFORE shutdown(): CPython 3.11 sets executor._processes=None
                # inside shutdown(), so _terminate_executor_processes must read it while still
                # populated. Otherwise it falls back to diffing the MAIN process's direct children,
                # which misses forkserver/spawn workers (forked by the forkserver daemon, not the
                # main process) and leaks the timed-out workers.
                _terminate_executor_processes(executor, baseline_child_pids=baseline_children)
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        if fatal_error is not None:
            _invalidate_incomplete_output()
            logger.error("%s", fatal_error)
            raise fatal_error

    feature_rows = [
        row for row in rows
        if row.get("extraction_status") in (None, "success")
    ]
    if not feature_rows:
        if existing_df is not None and out_path.exists():
            logger.debug("No new radiomics rows for %s (resume top-up)", course_dir)
            return RadiomicsCourseOutcome.extracted(out_path)
        logger.info("No eligible radiomics regions for %s", course_dir)
        _invalidate_radiomics_outputs(out_path)
        return RadiomicsCourseOutcome.nothing_to_do("all enumerated ROIs were explicitly skipped")

    try:
        import pandas as pd  # type: ignore

        df_new = pd.DataFrame(feature_rows).drop(
            columns=["extraction_status", "extraction_status_detail", "voxel_count"],
            errors="ignore",
        )
        if existing_df is not None and out_path.exists():
            output_cols = list(existing_df.columns)
            output_cols.extend(col for col in df_new.columns if col not in existing_df.columns)
            for col in output_cols:
                if col not in existing_df.columns:
                    existing_df[col] = None
                if col not in df_new.columns:
                    df_new[col] = None
            existing_df = existing_df.loc[:, output_cols]
            df_new = df_new.loc[:, output_cols]
            df = pd.concat([existing_df, df_new], ignore_index=True)
            df = df.drop_duplicates(
                subset=["segmentation_source", "roi_original_name", "patient_id", "course_id"],
                keep="first",
            )
        else:
            df = df_new
        _write_excel_atomic(df, out_path)
        # Optional: Parquet sidecar for fast aggregation (best-effort).
        parquet_path = out_path.with_suffix(".parquet")
        tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
        try:
            df.to_parquet(tmp_parquet, index=False, engine="pyarrow")
            tmp_parquet.replace(parquet_path)
        except Exception as exc:
            # Parquet can fail if some diagnostic columns contain non-scalar Python objects.
            # Retry by round-tripping through the just-written XLSX (which forces scalar/string
            # coercion) to keep Parquet sidecars consistent with XLSX exports.
            try:
                import pandas as pd  # type: ignore

                df_roundtrip = pd.read_excel(out_path, engine="openpyxl")
                df_roundtrip.to_parquet(tmp_parquet, index=False, engine="pyarrow")
                tmp_parquet.replace(parquet_path)
            except Exception as exc2:
                _remove_artifact_strict(tmp_parquet, context="cleaning failed Parquet publication")
                _remove_artifact_strict(parquet_path, context="invalidating stale Parquet sidecar")
                logger.debug("Parquet sidecar write failed for %s: %s (retry: %s)", out_path, exc, exc2)
        return RadiomicsCourseOutcome.extracted(out_path)
    except Exception as exc:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Failed to write radiomics output for {course_dir}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Robustness parallel helpers
# ---------------------------------------------------------------------------
# These functions are imported by radiomics_robustness.py to enable parallel
# feature extraction across perturbations.  The main radiomics extraction
# (above) uses _extract_one / _worker_init for a different parallelism
# pattern; these helpers use a simpler file-based approach that is compatible
# with multiprocessing.Pool.imap_unordered().

_ROBUSTNESS_WORKER_STATE: Dict[str, Any] = {}


def _prepare_radiomics_task(
    image,  # SimpleITK Image
    mask,   # SimpleITK Image (binary mask)
    config: Any,
    source: str,
    roi_name: str,
    course_dir: Path,
    temp_dir: Path,
    large_roi: bool,
) -> Tuple[Path, Dict[str, Any]]:
    """Save image/mask to temp files and prepare a task descriptor.

    Returns (mask_path, task_params) tuple suitable for passing to
    ``_isolated_radiomics_extraction_with_retry``.
    """
    import hashlib

    import SimpleITK as sitk
    from .radiomics import _get_params_file

    # Deduplicate only when the exact same image content is reused.
    # Geometry-only keys are unsafe here because noise perturbations share
    # origin/spacing/size but differ in voxel intensities. id(image) is also
    # unsafe: CPython can reuse the id of a garbage-collected image for an
    # unrelated later image, which would silently return a stale on-disk
    # NRRD cache hit for different voxel data. Hash the actual voxel bytes
    # instead so the key tracks content, not object identity.
    img_key = hashlib.sha1(
        sitk.GetArrayViewFromImage(image).tobytes(), usedforsecurity=False
    ).hexdigest()
    img_path = temp_dir / f"img_{img_key}.nrrd"
    if not img_path.exists():
        sitk.WriteImage(image, str(img_path))

    # Each mask is unique
    mask_id = abs(hash((id(mask), roi_name, source)))
    mask_path = temp_dir / f"mask_{mask_id}.nrrd"
    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(mask_path))

    params_file = _get_params_file(config, "CT")
    task_params = {
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "segmentation_source": source,
        "roi_name": roi_name,
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "large_roi": large_roi,
        "params_file": str(params_file) if params_file else None,
    }
    return mask_path, task_params


def _isolated_radiomics_extraction_with_retry(task) -> Optional[Dict[str, Any]]:
    """Extract radiomics features from pre-saved image/mask files.

    Designed for ``multiprocessing.Pool.imap_unordered()``.
    *task* is a ``(mask_path, task_params)`` tuple produced by
    ``_prepare_radiomics_task``.

    The RadiomicsFeatureExtractor is lazily cached per worker process to
    avoid re-creating it for every perturbation.
    """
    _apply_thread_limit(1)

    _mask_path, task_params = task
    params_file = task_params.get("params_file")
    large_roi = task_params.get("large_roi", False)
    extra_metadata = task_params.get("extra_metadata", {})

    # Lazy-init cached extractor (persists within the worker process)
    cache_key = f"{'large' if large_roi else 'normal'}_{params_file or 'default'}"
    if cache_key not in _ROBUSTNESS_WORKER_STATE:
        import warnings
        warnings.filterwarnings("ignore")
        import logging as _logging
        _logging.getLogger("radiomics").setLevel(_logging.ERROR)

        from radiomics import featureextractor  # type: ignore

        if params_file:
            ext = featureextractor.RadiomicsFeatureExtractor(params_file)
        else:
            ext = featureextractor.RadiomicsFeatureExtractor()

        if large_roi:
            try:
                ext.disableAllImageTypes()
                ext.enableImageTypeByName("Original")
                ext.disableAllFeatures()
                ext.enableFeatureClassByName("firstorder")
                ext.enableFeatureClassByName("shape")
                ext.settings["resampledPixelSpacing"] = [2.0, 2.0, 2.0]
            except Exception:
                pass

        _ROBUSTNESS_WORKER_STATE[cache_key] = ext

    ext = _ROBUSTNESS_WORKER_STATE[cache_key]

    try:
        result = ext.execute(task_params["image_path"], task_params["mask_path"])

        output: Dict[str, Any] = {}
        for k, v in result.items():
            if k.startswith("diagnostics_"):
                continue
            try:
                if hasattr(v, "item"):
                    output[k] = v.item()
                elif hasattr(v, "tolist"):
                    output[k] = v.tolist()
                elif isinstance(v, (int, float)):
                    output[k] = v
            except Exception:
                pass

        # Attach metadata expected by the robustness result parser
        output["segmentation_source"] = task_params.get("segmentation_source", "")
        output["roi_name"] = task_params.get("roi_name", "")
        output["patient_id"] = task_params.get("patient_id", "")
        output["course_id"] = task_params.get("course_id", "")
        output.update(extra_metadata)

        return output

    except Exception as e:
        logger.debug(
            "Isolated extraction failed for %s/%s: %s",
            task_params.get("roi_name"),
            extra_metadata.get("perturbation_id", "?"),
            e,
        )
        return None
