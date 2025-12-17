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
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pydicom

from .layout import build_course_dirs
from .utils import mask_is_cropped
from .custom_models import list_custom_model_outputs
from .custom_structures_rtstruct import _create_custom_structures_rtstruct, _is_rs_custom_stale

logger = logging.getLogger(__name__)

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_THREAD_LIMIT_ENV = "RTPIPELINE_RADIOMICS_THREAD_LIMIT"
_TASK_TIMEOUT_ENV = "RTPIPELINE_RADIOMICS_TASK_TIMEOUT"


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


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


def _extract_one(task: _RoiTask) -> Optional[Dict[str, Any]]:
    img = _WORKER_STATE.get("img")
    ext = _WORKER_STATE.get("extractor")
    if img is None or ext is None:
        return None

    from .radiomics import _mask_from_array_like

    rs_path = Path(task.rs_path)
    builder = _get_builder(rs_path)
    if builder is None:
        return None

    skip_rois: Set[str] = _WORKER_STATE.get("skip_rois", set())
    if _norm(task.roi_name) in skip_rois:
        return None

    try:
        mask = builder.get_roi_mask_by_name(task.roi_name)
    except Exception:
        return None
    if mask is None:
        return None

    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None

    voxel_count = int(mask_bool.sum())
    min_voxels = int(_WORKER_STATE.get("min_voxels", 120))
    max_voxels = int(_WORKER_STATE.get("max_voxels", 15_000_000))
    if voxel_count < min_voxels:
        return None
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
            return None
        ext = ext_large

    cropped = mask_is_cropped(mask_bool)
    display_roi = task.roi_name if (not cropped or task.roi_name.endswith("__partial")) else f"{task.roi_name}__partial"

    try:
        mask_img = _mask_from_array_like(img, mask_bool)
        res = ext.execute(img, mask_img)
    except Exception:
        return None

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
    except Exception:
        return []
    out: List[str] = []
    for roi in getattr(ds, "StructureSetROISequence", []) or []:
        name = str(getattr(roi, "ROIName", "") or "").strip()
        if name:
            out.append(name)
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


def parallel_radiomics_for_course(
    config: Any,
    course_dir: Path,
    custom_structures_config: Optional[Path] = None,
    max_workers: Optional[int] = None,
    use_cropped: bool = False,
) -> Optional[Path]:
    """Parallel CT radiomics for one course (process isolation)."""
    course_dir = Path(course_dir)
    out_path = course_dir / "radiomics_ct.xlsx"
    existing_df = None
    if getattr(config, "resume", False) and out_path.exists():
        try:
            import pandas as pd  # type: ignore

            existing_df = pd.read_excel(out_path, engine="openpyxl")
        except Exception:
            existing_df = None

    try:
        from .radiomics import _extractor
    except Exception as exc:
        logger.error("Failed importing radiomics extractor: %s", exc)
        return None

    # If we can't build a native extractor, delegate to conda backend.
    if _extractor(config, "CT") is None:
        try:
            from .radiomics_conda import radiomics_for_course as conda_radiomics_for_course
        except Exception as exc:
            logger.warning("Conda-based radiomics helper unavailable: %s", exc)
            return None
        return conda_radiomics_for_course(course_dir, config, custom_structures_config)

    course_dirs = build_course_dirs(course_dir)
    ct_dir = course_dirs.dicom_ct
    if not ct_dir.exists():
        logger.info("No CT image for radiomics in %s", course_dir)
        return None

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

    sources: List[Tuple[str, Path]] = []
    rs_manual = course_dir / "RS.dcm"
    if rs_manual.exists():
        sources.append(("Manual", rs_manual))
    rs_auto = course_dir / rs_auto_name
    if rs_auto.exists():
        sources.append(("AutoRTS_total", rs_auto))

    # Custom structures (optional): prepare RS_custom but extract only custom ROIs (no duplication)
    rs_custom = course_dir / "RS_custom.dcm"
    desired_custom: set[str] = set()
    if custom_structures_config and Path(custom_structures_config).exists():
        try:
            from .radiomics import _custom_roi_names_from_config

            desired_custom = _custom_roi_names_from_config(Path(custom_structures_config))
        except Exception:
            desired_custom = set()
        try:
            rs_auto_for_custom = course_dir / "RS_auto.dcm"
            if _is_rs_custom_stale(rs_custom, custom_structures_config, rs_manual, rs_auto_for_custom):
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir, custom_structures_config, rs_manual, rs_auto_for_custom
                ) or rs_custom
        except Exception as exc:
            logger.warning("Failed to prepare RS_custom for radiomics in %s: %s", course_dir, exc)

    # Custom model RTSTRUCTs
    try:
        for model_name, model_course_dir in list_custom_model_outputs(course_dir):
            rs_model = Path(model_course_dir) / "rtstruct.dcm"
            if rs_model.exists():
                sources.append((f"CustomModel:{model_name}", rs_model))
    except Exception as exc:
        logger.debug("Custom model outputs scan failed for %s: %s", course_dir, exc)

    skip_rois = _default_skip_rois() | {
        _norm(item)
        for item in getattr(config, "radiomics_skip_rois", []) or []
        if isinstance(item, str) and item.strip()
    }

    min_voxels, max_voxels = _derive_voxel_limits(config)

    # Enumerate tasks
    tasks: List[_RoiTask] = []
    existing_pairs: set[tuple[str, str]] = set()
    if existing_df is not None and "segmentation_source" in existing_df.columns and "roi_original_name" in existing_df.columns:
        try:
            existing_pairs = set(
                zip(
                    existing_df["segmentation_source"].astype(str).tolist(),
                    existing_df["roi_original_name"].astype(str).tolist(),
                )
            )
        except Exception:
            existing_pairs = set()

    full_run = not (getattr(config, "resume", False) and out_path.exists() and existing_df is not None)
    if full_run:
        for source, rs_path in sources:
            for roi_name in _list_roi_names(rs_path):
                if _norm(roi_name) in skip_rois:
                    continue
                tasks.append(_RoiTask(source=source, rs_path=str(rs_path), roi_name=roi_name, course_dir=str(course_dir)))
    else:
        # Resume top-up: only compute ROIs that were previously missing.
        if rs_manual.exists() and ("Manual", "BODY") not in existing_pairs and _norm("BODY") not in skip_rois:
            tasks.append(_RoiTask(source="Manual", rs_path=str(rs_manual), roi_name="BODY", course_dir=str(course_dir)))

    # Custom structures: extract only ROIs declared in config (or inferred from RS_custom diffs)
    if rs_custom.exists():
        try:
            from .radiomics import _list_roi_names_dicom

            avail = set(_list_roi_names_dicom(rs_custom))
            if not desired_custom:
                manual_names = set(_list_roi_names_dicom(course_dir / "RS.dcm"))
                auto_names = set(_list_roi_names_dicom(course_dir / "RS_auto.dcm"))
                inferred = {n for n in (avail - (manual_names | auto_names)) if n}
                desired_custom = {n[:-9] if n.endswith("__partial") else n for n in inferred}

            wanted: list[str] = []
            for base in sorted(desired_custom):
                if base in avail:
                    wanted.append(base)
                elif f"{base}__partial" in avail:
                    wanted.append(f"{base}__partial")
            for roi_name in wanted:
                if _norm(roi_name) in skip_rois:
                    continue
                if ("Custom", roi_name) in existing_pairs:
                    continue
                tasks.append(_RoiTask(source="Custom", rs_path=str(rs_custom), roi_name=roi_name, course_dir=str(course_dir)))
        except Exception as exc:
            logger.warning("Failed to enumerate custom radiomics tasks for %s: %s", course_dir, exc)

    if not tasks:
        if existing_df is not None and out_path.exists():
            logger.debug("Radiomics up-to-date for %s; no missing ROI tasks", course_dir)
            return out_path
        logger.info("No radiomics tasks for %s", course_dir)
        return None

    if max_workers is None:
        worker_count = _calculate_optimal_workers()
    else:
        worker_count = max(1, min(int(max_workers), _calculate_optimal_workers()))
    worker_count = max(1, min(worker_count, len(tasks)))

    thread_limit = _resolve_thread_limit(getattr(config, "radiomics_thread_limit", None))
    task_timeout = _coerce_positive_int(os.environ.get(_TASK_TIMEOUT_ENV)) or 600

    logger.info(
        "Parallel radiomics for %s: %d ROI task(s), %d worker(s), thread_limit=%s, timeout=%ss",
        course_dir.name,
        len(tasks),
        worker_count,
        thread_limit if thread_limit is not None else "env/default",
        task_timeout,
    )

    total = len(tasks)
    completed: Set[_RoiTask] = set()
    rows: List[Dict[str, Any]] = []

    pending_tasks: List[_RoiTask] = list(tasks)
    start_time = time.monotonic()
    last_log = start_time

    while pending_tasks:
        baseline_children = _current_child_pids()
        executor = ProcessPoolExecutor(
            max_workers=min(worker_count, len(pending_tasks)),
            initializer=_worker_init,
            initargs=(str(ct_dir), config, thread_limit, skip_rois, min_voxels, max_voxels),
        )
        futures = {}
        try:
            for task in pending_tasks:
                futures[executor.submit(_extract_one, task)] = task
            task_start = {fut: time.monotonic() for fut in futures}
            remaining = set(futures.keys())
            restart = False

            while remaining:
                done, remaining = wait(remaining, timeout=5.0, return_when=FIRST_COMPLETED)
                now = time.monotonic()

                for fut in done:
                    task = futures[fut]
                    completed.add(task)
                    try:
                        rec = fut.result(timeout=0)
                    except Exception:
                        rec = None
                    if rec:
                        rows.append(rec)

                # Timeouts: mark tasks as skipped and restart the pool so we don't block on shutdown.
                timed_out_tasks: List[_RoiTask] = []
                for fut in list(remaining):
                    if now - task_start[fut] > task_timeout:
                        task = futures[fut]
                        timed_out_tasks.append(task)
                        completed.add(task)
                        remaining.remove(fut)
                if timed_out_tasks:
                    logger.warning(
                        "Radiomics: %d ROI task(s) timed out for %s; restarting worker pool",
                        len(timed_out_tasks),
                        course_dir.name,
                    )
                    restart = True
                    break

                if now - last_log > 30:
                    logger.info("Radiomics progress for %s: %d/%d", course_dir.name, len(completed), total)
                    last_log = now

            if restart:
                pending_tasks = [futures[fut] for fut in remaining if futures[fut] not in completed]
            else:
                pending_tasks = []

        finally:
            if pending_tasks:
                executor.shutdown(wait=False, cancel_futures=True)
                _terminate_executor_processes(executor, baseline_child_pids=baseline_children)
            else:
                executor.shutdown(wait=True)

    if not rows:
        if existing_df is not None and out_path.exists():
            logger.debug("No new radiomics rows for %s (resume top-up)", course_dir)
            return out_path
        logger.warning("No successful radiomics extractions for %s", course_dir)
        return None

    try:
        import pandas as pd  # type: ignore

        df_new = pd.DataFrame(rows)
        if existing_df is not None and out_path.exists():
            template_cols = list(existing_df.columns)
            for col in template_cols:
                if col not in df_new.columns:
                    df_new[col] = None
            df_new = df_new.loc[:, template_cols]
            df = pd.concat([existing_df, df_new], ignore_index=True)
            df = df.drop_duplicates(
                subset=["segmentation_source", "roi_original_name", "patient_id", "course_id"],
                keep="first",
            )
        else:
            df = df_new
        df.to_excel(out_path, index=False, engine="openpyxl")
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
                try:
                    if tmp_parquet.exists():
                        tmp_parquet.unlink()
                except Exception:
                    pass
                # Avoid leaving a stale sidecar if we failed to update it.
                try:
                    if parquet_path.exists():
                        parquet_path.unlink()
                except Exception:
                    pass
                logger.debug("Parquet sidecar write failed for %s: %s (retry: %s)", out_path, exc, exc2)
        return out_path
    except Exception as exc:
        logger.error("Failed to write radiomics output for %s: %s", course_dir, exc)
        return None
