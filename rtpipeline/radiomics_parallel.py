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
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import pydicom

from .acquisition_scale import (
    attach_acquisition_descriptor,
    describe_contract_planning_ct,
    validate_acquisition_descriptor_table,
)
from .layout import build_course_dirs
from .course_contract import ALL_SERIES_RADIOMICS_TEMP_SCOPE, load_course_contract
from .utils import mask_is_cropped, radiomics_mp_context
from .custom_models import (
    list_custom_model_outputs,
    validate_custom_model_output_inventory,
)
from .custom_structures_rtstruct import (
    _create_custom_structures_rtstruct,
    _is_rs_custom_stale,
    record_rs_custom_resume_decision,
)
from .radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    course_diagnostic_columns,
    extraction_status_is_nonfatal_for_required,
    invalidate_radiomics_outputs as _invalidate_radiomics_outputs,
    outcome_from_output,
    remove_artifact_strict as _remove_artifact_strict,
    roi_source_is_required,
    resume_identity_pairs as _resume_identity_pairs,
    write_excel_atomic as _write_excel_atomic,
)
from .radiomics_resource_guard import (
    RESAMPLED_BBOX_LIMIT_CODE,
    estimate_resampled_bounding_box,
    resolve_max_resampled_bbox_voxels,
)
from .roi_requiredness import (
    DenominatorLedger,
    FAILED_RADIOMICS_FEATURE_COMPLETENESS,
    FAILED_RADIOMICS_RESOURCE_LIMIT,
    Requiredness,
    assess_custom_applicability,
    dependency_state_from_observation,
    inspect_rtstruct,
    match_requirements,
    requirements_from_contract,
    write_modality_ledger,
)
from .radiomics_ct_contract import (
    CT_EXTRACTION_ARMS,
    PRIMARY_ARM,
    RADIOMICS_FEATURE_COMPLETENESS_COLUMN,
    SENSITIVITY_ARM,
    RoiClassDecision,
    base_identity_key,
    classify_ct_roi,
    configured_parameter_hash,
    current_code_revision,
    disposition_rows_for_arms,
    effective_parameter_hashes_for_arms,
    expected_publication_keys,
    extract_ct_roi_arms,
    load_custom_structure_provenance,
    new_run_identifier,
    publication_key,
    read_authoritative_ct_publication,
    rtstruct_roi_identities,
    stable_rtstruct_roi_identity,
    validate_ct_publication,
    write_ct_publication_atomic,
)

logger = logging.getLogger(__name__)


class RadiomicsRegionExtractionError(RuntimeError):
    """A required ROI could not be extracted, so the course is incomplete."""

    def __init__(self, detail: str, *, failure_kind: str = "extraction_error") -> None:
        self.failure_kind = failure_kind
        super().__init__(detail)


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
    # No anatomical class is silently dropped. The governed class decision
    # dispositions primary features while retaining the sensitivity arm and shape.
    return set()


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
    series_uid: str
    mask_identity: str
    stable_roi_identifier: str
    decision: RoiClassDecision
    run_identifier: str
    code_revision: str
    configured_parameter_hashes: Dict[str, str] = field(compare=False, hash=False)
    effective_parameter_hashes: Dict[str, str] = field(compare=False, hash=False)
    required: Optional[bool] = None


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
            "config": config,
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


def _task_common_metadata(
    task: _RoiTask,
    *,
    cropped: bool = False,
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
        "series_uid": task.series_uid,
        "mask_identity": task.mask_identity,
        "rtstruct_sop_instance_uid": task.mask_identity,
        "stable_roi_identifier": task.stable_roi_identifier,
        "structure_cropped": bool(cropped),
    }


def _write_parallel_roi_ledger(
    course_dir: Path,
    tasks: Sequence[_RoiTask],
    rows: Sequence[Mapping[str, Any]],
    applicability: Sequence[Any] = (),
    *,
    extracted: bool,
    technical: bool = False,
    indeterminate: bool = False,
) -> None:
    ledger = DenominatorLedger()
    course_id, patient_id = Path(course_dir).name, Path(course_dir).parent.name
    for task in tasks:
        ledger.expect_course_roi(course_id, task.roi_name)
    rows_by_roi: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        name = str(row.get("roi_original_name", row.get("roi_name", "")))
        if name:
            rows_by_roi.setdefault(name, []).append(row)
    technical = technical or any(
        str(row.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or "")
        == "incomplete"
        for row in rows
    )
    for name, roi_rows in rows_by_roi.items():
        row = next(
            (
                candidate
                for candidate in roi_rows
                if str(
                    candidate.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or ""
                )
                == "incomplete"
            ),
            next(
                (
                    candidate
                    for candidate in roi_rows
                    if str(candidate.get("extraction_status") or "success")
                    not in {"success", "declared_skip"}
                ),
                roi_rows[0],
            ),
        )
        status = str(row.get("extraction_status") or "success")
        detail_code = str(row.get("roi_structural_code") or "")
        completeness = str(
            row.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or ""
        )
        reason = (
            FAILED_RADIOMICS_FEATURE_COMPLETENESS
            if completeness == "incomplete"
            else FAILED_RADIOMICS_RESOURCE_LIMIT
            if detail_code == RESAMPLED_BBOX_LIMIT_CODE
            else detail_code
            or (
                "extracted"
                if status in {"success", "declared_skip"}
                else "failed_radiomics_extraction"
            )
        )
        ledger.record_roi(
            course_id,
            patient_id,
            name,
            reason_code=reason,
            disposition="extracted" if reason == "extracted" else "excluded",
            detail_code=detail_code or None,
            detail=str(
                row.get("radiomics_feature_completeness_reason")
                if completeness == "incomplete"
                else row.get("extraction_status_detail")
                or ""
            ),
            estimated_resampled_bbox_voxel_count=row.get(
                "estimated_resampled_bbox_voxel_count"
            ),
            max_resampled_bbox_voxel_count=row.get(
                "max_resampled_bbox_voxel_count"
            ),
        )
    recorded_pairs = {
        (str(row.get("course_id")), str(row.get("roi_name"))): row
        for row in ledger.roi_rows
    }
    for task in tasks:
        if (course_id, task.roi_name) not in recorded_pairs:
            ledger.record_roi(course_id, patient_id, task.roi_name, reason_code="failed_radiomics_extraction", disposition="excluded")
    recorded_pairs = {
        (str(row.get("course_id")), str(row.get("roi_name"))): row
        for row in ledger.roi_rows
    }
    task_names = {task.roi_name for task in tasks}
    for item in applicability:
        configured_name = str(item.roi_name)
        ledger.expect_course_roi(course_id, configured_name)
        if item.reason_code != "extracted":
            ledger.record_roi(
                course_id,
                patient_id,
                configured_name,
                reason_code=item.reason_code,
                disposition="excluded",
                detail=item.detail,
            )
            continue
        if configured_name in task_names:
            continue
        realized_name = next(
            (
                task.roi_name
                for task in tasks
                if _norm(task.roi_name)
                in {_norm(configured_name), _norm(f"{configured_name}__partial")}
            ),
            None,
        )
        realized = recorded_pairs.get((course_id, realized_name or ""))
        if realized is None:
            # Keep the expectation unresolved. ensure_expected_pairs must expose a
            # missing extraction task rather than accepting an invented disposition.
            continue
        reason = str(realized.get("reason_code") or "failed_radiomics_extraction")
        ledger.record_roi(
            course_id,
            patient_id,
            configured_name,
            reason_code=reason,
            disposition="extracted" if reason == "extracted" else "excluded",
            detail=(
                f"Configured ROI was realized as {realized_name!r}. {item.detail}"
            ),
        )
    ledger.record_course(course_id, patient_id, screened=True, in_scope=True, out_of_scope=False, adequate_coverage=bool(rows), insufficient_coverage=not bool(rows), valid_derivation=any(item.reason_code == "extracted" for item in applicability), technical_exclusion=technical, indeterminate=indeterminate or any(item.reason_code == "indeterminate_applicability" for item in applicability), extracted=extracted, reason_code="extracted" if extracted else ("indeterminate_applicability" if indeterminate else "failed_radiomics_extraction"))
    write_modality_ledger(Path(course_dir) / "metadata", ledger, "CT")


def _status_records(
    task: _RoiTask,
    status: str,
    detail: str,
    *,
    voxel_count: Optional[int] = None,
    failure_kind: str = "extraction_error",
    metadata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    common_metadata = _task_common_metadata(task)
    common_metadata.update(dict(metadata or {}))
    return disposition_rows_for_arms(
        common_metadata,
        decision=task.decision,
        disposition=status,
        detail=detail,
        failure_kind=failure_kind,
        run_identifier=task.run_identifier,
        code_revision=task.code_revision,
        native_voxel_count=voxel_count,
        required=_task_is_required(task),
        effective_hashes=task.effective_parameter_hashes,
        configured_parameter_hashes=task.configured_parameter_hashes,
    )


def _task_is_required(task: _RoiTask) -> bool:
    if task.required is not None:
        return bool(task.required)
    return roi_source_is_required(task.source)


def _failure_records(
    task: _RoiTask,
    detail: str,
    *,
    status: str = "failed",
    failure_kind: str = "extraction_error",
) -> List[Dict[str, Any]]:
    return _status_records(
        task,
        status,
        detail,
        failure_kind=failure_kind,
    )


def _record_roi_outcome(
    task: _RoiTask,
    records: Optional[Sequence[Dict[str, Any]]],
    source_counts: Dict[str, Dict[str, int]],
    roi_failures: List[Dict[str, str]],
) -> None:
    """Accumulate one task outcome without treating a best-effort miss as success."""
    record_list = list(records or [])
    statuses = [record.get("extraction_status") for record in record_list]
    status = next(
        (candidate for candidate in statuses if candidate not in (None, "success")),
        "success" if record_list else None,
    )
    try:
        if status != status:
            status = None
    except (TypeError, ValueError):
        pass
    if status == "declared_skip":
        return
    counts = source_counts.setdefault(
        task.source,
        {"attempted": 0, "extracted": 0, "failed": 0},
    )
    counts["attempted"] += 1
    if status in (None, "success"):
        counts["extracted"] += 1
        return
    counts["failed"] += 1
    failure_record = next(
        (record for record in record_list if record.get("extraction_status") == status),
        record_list[0] if record_list else {},
    )
    roi_failures.append(
        {
            "roi_name": task.roi_name,
            "source": task.source,
            "status": str(status),
            "failure_kind": str(failure_record.get("extraction_failure_kind", "extraction_error")),
            "reason": str(failure_record.get("extraction_status_detail", "unknown error")),
        }
    )


def _resume_outcome(
    output_path: Path,
    tasks: Sequence[_RoiTask],
    dataframe: Any,
) -> RadiomicsCourseOutcome:
    """Reconstruct resume status while preserving the required-ROI gate."""
    return outcome_from_output(
        output_path,
        required_by_identity={
            (
                Path(task.course_dir).parent.name,
                Path(task.course_dir).name,
                task.series_uid,
                task.source,
                task.mask_identity,
                task.roi_name,
                task.stable_roi_identifier,
                arm,
            ): _task_is_required(task)
            for task in tasks
            for arm in CT_EXTRACTION_ARMS
        },
    )


def _extract_one(task: _RoiTask) -> List[Dict[str, Any]]:
    skip_rois: Set[str] = _WORKER_STATE.get("skip_rois", set())
    if _norm(task.roi_name) in skip_rois:
        return _status_records(
            task,
            "declared_skip",
            "ROI is listed in radiomics_skip_rois",
            failure_kind="declared_ineligible",
        )

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
            f"ROI {task.roi_name} from {task.source} produced an empty required mask",
            failure_kind="degenerate_mask",
        )

    voxel_count = int(mask_bool.sum())
    min_voxels = int(_WORKER_STATE.get("min_voxels", 120))
    max_voxels = int(_WORKER_STATE.get("max_voxels", 15_000_000))
    if voxel_count < min_voxels:
        return _status_records(
            task,
            "below_minimum_voxels",
            f"ROI contains {voxel_count} voxels; configured minimum is {min_voxels}",
            voxel_count=voxel_count,
            failure_kind="degenerate_mask",
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
    try:
        pad_distance = int(math.ceil(float(ext.settings.get("padDistance", 5))))
    except (TypeError, ValueError, OverflowError):
        pad_distance = 5
    work_estimate = estimate_resampled_bounding_box(
        mask_bool,
        native_spacing_xyz=spacing,
        resampled_spacing_xyz=resampled,
        array_axis_to_xyz=(1, 0, 2),
        pad_distance=pad_distance,
    )
    max_bbox_voxels = resolve_max_resampled_bbox_voxels(
        _WORKER_STATE.get("config")
    )
    if work_estimate.estimated_resampled_bbox_voxels > max_bbox_voxels:
        detail = (
            f"ROI {task.roi_name} requires an estimated padded resampled bounding "
            f"box of {work_estimate.estimated_resampled_bbox_voxels} voxels "
            f"({work_estimate.estimated_resampled_bbox_shape}); configured maximum "
            f"is {max_bbox_voxels}. Full configured radiomics was not started."
        )
        return _status_records(
            task,
            "failed",
            detail,
            voxel_count=voxel_count,
            failure_kind="resource_limit",
            metadata={
                "roi_structural_code": RESAMPLED_BBOX_LIMIT_CODE,
                **work_estimate.metadata(limit=max_bbox_voxels),
            },
        )
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

        def _factory():
            from .radiomics import _extractor, _extractor_large_roi

            candidate = (
                _extractor_large_roi(_WORKER_STATE["config"], "CT")
                if (is_body or estimated_voxels > float(max_voxels))
                else _extractor(_WORKER_STATE["config"], "CT")
            )
            if candidate is None:
                raise RuntimeError("CT radiomics extractor is unavailable in worker")
            return candidate

        display_roi = (
            task.roi_name
            if (not cropped or task.roi_name.endswith("__partial"))
            else f"{task.roi_name}__partial"
        )
        common_metadata = _task_common_metadata(task, cropped=cropped)
        common_metadata["roi_name"] = display_roi
        records = extract_ct_roi_arms(
            img,
            mask_img,
            factory=_factory,
            decision=task.decision,
            common_metadata=common_metadata,
            run_identifier=task.run_identifier,
            code_revision=task.code_revision,
            native_voxel_count=voxel_count,
            required=_task_is_required(task),
            configured_parameter_hashes=task.configured_parameter_hashes,
        )
    except TimeoutError as exc:
        raise RadiomicsRegionExtractionError(str(exc), failure_kind="timeout") from exc
    except Exception as exc:
        raise RadiomicsRegionExtractionError(
            f"ROI {task.roi_name} radiomics extraction failed: {exc}",
            failure_kind="extraction_error",
        ) from exc
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    return records


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
    acquisition_descriptor = describe_contract_planning_ct(contract)
    existing_df = None
    parquet_path = out_path.with_suffix(".parquet")
    if getattr(config, "resume", False) and (parquet_path.exists() or out_path.exists()):
        try:
            if not parquet_path.exists():
                raise ValueError("authoritative CT Parquet is missing")
            existing_df = read_authoritative_ct_publication(parquet_path)
            _resume_identity_pairs(existing_df)
            validate_acquisition_descriptor_table(
                existing_df,
                expected_descriptor=acquisition_descriptor,
                expected_series_instance_uid=contract.planning_ct.get(
                    "series_instance_uid"
                ),
            )
        except Exception as exc:
            logger.warning(
                "Invalidating unusable parallel resume publication for %s: %s",
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
        return outcome_from_output(conda_out)

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
    custom_applicability: list[Any] = []
    pending_custom_assessments: set[str] = set()
    dependency_states: dict[str, Any] = {}
    planning_ct_fov: Any = {}
    observed_required: set[str] = set()
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
        custom_provenance = load_custom_structure_provenance(configured_custom_path)
        # Decide applicability before rebuilding RS_custom. Source-applicable
        # missing ROIs remain repair candidates until the governed builder has run.
        for source_name, source_path, _ in sources:
            try:
                inventory = inspect_rtstruct(source_path)
            except Exception:
                continue
            for observation in inventory.named_rois:
                dependency_states[observation.name] = (
                    dependency_state_from_observation(observation)
                )
        try:
            from .radiomics import _planning_ct_fov, _analysis_contract, _roi_requiredness
            custom_contract = _analysis_contract(config)
            planning_ct_fov = _planning_ct_fov(course_dir)
        except Exception:
            custom_contract = {}
            _planning_ct_fov = lambda _path: {}
            _roi_requiredness = lambda _config, _source, _name: Requiredness.INVENTORY_ONLY
        available_custom = {}
        if rs_custom.is_file():
            try:
                custom_inventory = inspect_rtstruct(rs_custom)
                available_custom = {item.name: item for item in custom_inventory.named_rois}
            except Exception:
                available_custom = {}
        applicable_custom: set[str] = set()
        for base in sorted(desired_custom):
            candidate = next(
                (available_custom[name] for name in (base, f"{base}__partial") if name in available_custom),
                None,
            )
            generated_state = "absent"
            if candidate is not None:
                generated_state = "readable_nonempty" if candidate.has_readable_contour else "unreadable"
            assessment = assess_custom_applicability(
                base,
                dependency_states,
                planning_ct_fov,
                generated_state=generated_state,
                custom_provenance=custom_provenance,
            )
            required_custom = (
                _roi_requiredness(config, "Custom", base)
                == Requiredness.ANALYSIS_REQUIRED
            )
            if assessment.reason_code == "extracted":
                applicable_custom.add(base)
                custom_applicability.append(assessment)
                if required_custom:
                    observed_required.add(base)
            elif assessment.reason_code == "failed_custom_generation":
                applicable_custom.add(base)
                pending_custom_assessments.add(base)
            elif assessment.reason_code == "indeterminate_applicability":
                custom_applicability.append(assessment)
                _write_parallel_roi_ledger(
                    course_dir,
                    (),
                    (),
                    custom_applicability,
                    extracted=False,
                    indeterminate=True,
                )
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Configured custom ROI {base!r} has {assessment.reason_code}: {assessment.detail}"
                )
            else:
                custom_applicability.append(assessment)
                if assessment.reason_code in {"not_applicable_anatomy", "not_applicable_scope"}:
                    if required_custom:
                        observed_required.add(base)
                elif required_custom:
                    _write_parallel_roi_ledger(
                        course_dir,
                        (),
                        (),
                        custom_applicability,
                        extracted=False,
                        technical=True,
                    )
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"Required custom ROI {base!r} has {assessment.reason_code}: {assessment.detail}"
                    )
        desired_custom = applicable_custom
        custom_rebuild_attempted = False
        custom_rebuild_published = False
        try:
            rs_auto_for_custom = course_dir / "RS_auto.dcm"
            custom_is_stale = bool(
                desired_custom
                and _is_rs_custom_stale(
                    rs_custom, configured_custom_path, rs_manual, rs_auto_for_custom
                )
            )
            if not desired_custom:
                record_rs_custom_resume_decision(
                    course_dir,
                    "not_applicable",
                    "no configured custom ROI was applicable to the contracted planning CT",
                )
            elif custom_is_stale:
                custom_rebuild_attempted = True
                from .custom_structures_rtstruct import _quarantine_rejected_rtstruct

                _quarantine_rejected_rtstruct(
                    rs_custom,
                    "RS_custom failed the authoritative currentness check",
                )
                rebuilt = _create_custom_structures_rtstruct(
                    course_dir, configured_custom_path, rs_manual, rs_auto_for_custom
                )
                if rebuilt is None or not Path(rebuilt).is_file():
                    record_rs_custom_resume_decision(
                        course_dir,
                        "failed",
                        "RS_custom replacement could not be published",
                    )
                    raise RadiomicsCourseExtractionError(
                        f"RS_custom rebuild failed for configured ROIs in {course_dir}"
                    )
                rs_custom = Path(rebuilt)
                custom_rebuild_published = True
                record_rs_custom_resume_decision(
                    course_dir,
                    "rebuilt",
                    "rebuilt after the previous RS_custom failed the authoritative currentness check",
                )
            else:
                record_rs_custom_resume_decision(
                    course_dir,
                    "reused",
                    "existing RS_custom passed the authoritative currentness check",
                )
        except Exception as exc:
            if custom_rebuild_attempted and not custom_rebuild_published:
                record_rs_custom_resume_decision(
                    course_dir,
                    "failed",
                    f"RS_custom rebuild raised {type(exc).__name__}: {exc}",
                )
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

    series_uid = str(contract.planning_ct.get("series_instance_uid") or "").strip()
    if not series_uid:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Planning CT contract has no SeriesInstanceUID for radiomics in {course_dir}"
        )
    run_identifier = new_run_identifier()
    code_revision = current_code_revision()
    custom_provenance = load_custom_structure_provenance(configured_custom_path)
    from .radiomics import _extractor, _get_params_file

    parameter_path = _get_params_file(config, "CT")
    identity_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}

    def _build_task(
        source: str,
        rs_path: Path,
        roi_name: str,
        *,
        required: bool,
    ) -> _RoiTask:
        sop_uid, roi_number = stable_rtstruct_roi_identity(rs_path, roi_name)
        decision = classify_ct_roi(
            source,
            roi_name,
            custom_provenance=custom_provenance if source == "Custom" else None,
        )
        configured_hashes = {
            arm: configured_parameter_hash(
                parameter_path,
                arm=arm,
                window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
                large_roi=False,
            )
            for arm in CT_EXTRACTION_ARMS
        }

        def _factory() -> Any:
            candidate = _extractor(config, "CT")
            if candidate is None:
                raise RuntimeError("No CT radiomics extractor is available")
            return candidate

        return _RoiTask(
            source=source,
            rs_path=str(rs_path),
            roi_name=roi_name,
            course_dir=str(course_dir),
            series_uid=series_uid,
            mask_identity=sop_uid,
            stable_roi_identifier=f"rtstruct_roi_number:{roi_number}",
            decision=decision,
            run_identifier=run_identifier,
            code_revision=code_revision,
            configured_parameter_hashes=configured_hashes,
            effective_parameter_hashes=effective_parameter_hashes_for_arms(
                _factory, decision
            ),
            required=required,
        )

    # Enumerate every current non-skipped identity before accepting a resume
    # workbook. BODY-only top-ups can miss ordinary Manual/AutoRTS/model ROIs.
    tasks: List[_RoiTask] = []
    required_contract = requirements_from_contract(
        getattr(config, "radiomics_analysis_contract", {}) or {}, "CT"
    )
    try:
        from .radiomics import _roi_requiredness
        analysis_config = getattr(config, "radiomics_analysis_contract", {}) or {}
        for source, rs_path, expected_rois in sources:
            if expected_rois:
                roi_names = list(expected_rois)
            else:
                try:
                    inventory = inspect_rtstruct(rs_path)
                except Exception:
                    inventory = None
                if inventory is None:
                    roi_names = list(_list_roi_names(rs_path))
                else:
                    for match in match_requirements(inventory, required_contract, source=source):
                        if match.observation is not None:
                            observed_required.add(match.requirement.canonical_name)
                        if match.structural_code == "REQUIRED_ROI_AMBIGUOUS_MATCH":
                            _write_parallel_roi_ledger(
                                course_dir,
                                (),
                                [{
                                    "roi_original_name": match.requirement.canonical_name,
                                    "segmentation_source": source,
                                    "extraction_status": "failed",
                                    "roi_structural_code": "REQUIRED_ROI_AMBIGUOUS_MATCH",
                                }],
                                extracted=False,
                                indeterminate=True,
                            )
                            raise RadiomicsCourseExtractionError(
                                f"Required ROI {match.requirement.canonical_name!r} has ambiguous identity in {rs_path} "
                                "[REQUIRED_ROI_AMBIGUOUS_MATCH]"
                            )
                    for observation in inventory.named_rois:
                        requiredness = _roi_requiredness(config, source, observation.name)
                        if observation.structural_code and requiredness == Requiredness.ANALYSIS_REQUIRED:
                            raise RadiomicsCourseExtractionError(
                                f"Required ROI {observation.name!r} in {rs_path} has "
                                f"{observation.structural_code}"
                            )
                    roi_names = [
                        observation.name for observation in inventory.named_rois
                        if not observation.structural_code
                    ]
                    if not roi_names:
                        try:
                            roi_names = list(_list_roi_names(rs_path))
                        except Exception:
                            roi_names = []
                    if inventory.named_rois:
                        valid_names = {observation.name for observation in inventory.named_rois if not observation.structural_code}
                        roi_names = [name for name in roi_names if name in valid_names]
                    if not roi_names and "RTSTRUCT_NO_NAMED_ROIS" in inventory.structural_codes:
                        logger.info("RTSTRUCT has no named ROIs and contributes no inventory: %s", rs_path)
            for roi_name in roi_names:
                if _norm(roi_name) in skip_rois:
                    continue
                selected_model = source.startswith("CustomModel:")
                requiredness = _roi_requiredness(
                    config, source, roi_name, selected_model=selected_model
                )
                if requiredness == Requiredness.ANALYSIS_REQUIRED:
                    for requirement in required_contract:
                        if (
                            requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
                            and (not requirement.source or _norm(requirement.source) == _norm(source))
                            and any(_norm(accepted) == _norm(roi_name) for accepted in requirement.accepted_names)
                        ):
                            observed_required.add(requirement.canonical_name)
                tasks.append(
                    _build_task(
                        source,
                        rs_path,
                        roi_name,
                        required=requiredness == Requiredness.ANALYSIS_REQUIRED,
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
    if rs_custom.exists() and (desired_custom or configured_custom_path is None):
        try:
            avail = set(_list_roi_names(rs_custom))
            if configured_custom_path is None and not desired_custom:
                manual_names = set(_list_roi_names(rs_manual)) if rs_manual.exists() else set()
                auto_names = set(_list_roi_names(rs_auto)) if rs_auto.exists() else set()
                inferred = {n for n in (avail - (manual_names | auto_names)) if n}
                desired_custom = {n[:-9] if n.endswith("__partial") else n for n in inferred}

            for base in sorted(pending_custom_assessments):
                generated_state = (
                    "readable_nonempty"
                    if base in avail or f"{base}__partial" in avail
                    else "absent"
                )
                final_assessment = assess_custom_applicability(
                    base,
                    dependency_states,
                    planning_ct_fov,
                    generated_state=generated_state,
                    custom_provenance=custom_provenance,
                )
                custom_applicability.append(final_assessment)
                if final_assessment.reason_code == "extracted" and (
                    _roi_requiredness(config, "Custom", base)
                    == Requiredness.ANALYSIS_REQUIRED
                ):
                    observed_required.add(base)

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
                required_custom = _roi_requiredness(config, "Custom", roi_name) == Requiredness.ANALYSIS_REQUIRED
                if required_custom:
                    for requirement in required_contract:
                        if (
                            requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
                            and (not requirement.source or _norm(requirement.source) == "custom")
                            and any(_norm(accepted) == _norm(roi_name) for accepted in requirement.accepted_names)
                        ):
                            observed_required.add(requirement.canonical_name)
                tasks.append(
                    _build_task(
                        "Custom",
                        rs_custom,
                        roi_name,
                        required=required_custom,
                    )
                )
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            if isinstance(exc, RadiomicsCourseExtractionError):
                raise
            raise RadiomicsCourseExtractionError(
                f"Failed to enumerate custom radiomics tasks for {course_dir}: {exc}"
            ) from exc

    missing_required = [
        requirement.canonical_name
        for requirement in required_contract
        if requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
        and requirement.canonical_name not in observed_required
    ]
    if missing_required:
        _write_parallel_roi_ledger(
            course_dir,
            (),
            [{
                "roi_original_name": name,
                "extraction_status": "failed",
                "roi_structural_code": "REQUIRED_ROI_NOT_DECLARED",
            } for name in missing_required],
            extracted=False,
            technical=True,
        )
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            "Required ROI(s) are absent from current RTSTRUCT sources: "
            + ", ".join(missing_required)
            + " [REQUIRED_ROI_NOT_DECLARED]"
        )

    expected_keys = expected_publication_keys(
        {
            "patient_id": Path(task.course_dir).parent.name,
            "course_id": Path(task.course_dir).name,
            "series_uid": task.series_uid,
            "segmentation_source": task.source,
            "mask_identity": task.mask_identity,
            "roi_original_name": task.roi_name,
            "stable_roi_identifier": task.stable_roi_identifier,
        }
        for task in tasks
    )
    if existing_df is not None:
        resume_error: Optional[Exception] = None
        try:
            existing_keys = _resume_identity_pairs(existing_df)
            expected_config_hashes: Dict[Tuple[str, ...], str] = {
                (
                    Path(task.course_dir).parent.name,
                    Path(task.course_dir).name,
                    task.series_uid,
                    task.source,
                    task.mask_identity,
                    task.roi_name,
                    task.stable_roi_identifier,
                    arm,
                ): task.configured_parameter_hashes[arm]
                for task in tasks
                for arm in CT_EXTRACTION_ARMS
            }
            for record in existing_df.to_dict("records"):
                key = publication_key(record)
                if str(record.get("configured_parameter_hash") or "") != str(
                    expected_config_hashes.get(key) or ""
                ):
                    raise ValueError("configured radiomics parameter hash is stale")
        except ValueError as exc:
            resume_error = exc
            existing_keys = set()
        if resume_error is None and expected_keys and existing_keys == expected_keys:
            _write_parallel_roi_ledger(course_dir, tasks, existing_df.to_dict("records"), custom_applicability, extracted=True)
            return _resume_outcome(out_path, tasks, existing_df)
        logger.warning(
            "Invalidating incomplete or stale parallel resume publication for %s: "
            "expected %d full ROI-arm identities, found %d%s",
            course_dir,
            len(expected_keys),
            len(existing_keys),
            f"; {resume_error}" if resume_error is not None else "",
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
    source_counts: Dict[str, Dict[str, int]] = {}
    roi_failures: List[Dict[str, str]] = []

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
                        records = fut.result(timeout=0)
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
                        if _task_is_required(task):
                            fatal_error = RadiomicsCourseExtractionError(
                                f"Radiomics course {course_dir} is incomplete: "
                                f"required ROI {task.source}/{task.roi_name} failed: {exc}"
                            )
                            break
                        records = _failure_records(
                            task,
                            str(exc),
                            failure_kind=getattr(exc, "failure_kind", "extraction_error"),
                        )
                        _record_roi_outcome(task, records, source_counts, roi_failures)
                        completed.add(task)
                        rows.extend(records)
                        continue
                    if not records:
                        if _task_is_required(task):
                            fatal_error = RadiomicsCourseExtractionError(
                                f"Radiomics course {course_dir} is incomplete: required ROI "
                                f"{task.source}/{task.roi_name} returned no outcome record"
                            )
                            break
                        records = _failure_records(
                            task,
                            "worker returned no outcome record",
                        )
                    else:
                        failing_record = next(
                            (
                                record
                                for record in records
                                if record.get("extraction_status")
                                not in (None, "success", "declared_skip")
                            ),
                            None,
                        )
                        if failing_record is not None and (
                            _task_is_required(task)
                            and not extraction_status_is_nonfatal_for_required(
                                failing_record.get("extraction_status")
                            )
                        ):
                            fatal_error = RadiomicsCourseExtractionError(
                                f"Radiomics course {course_dir} is incomplete: "
                                f"required ROI {task.source}/{task.roi_name} failed: "
                                f"{failing_record.get('extraction_status_detail', 'unknown error')}"
                            )
                            break
                    completed.add(task)
                    _record_roi_outcome(task, records, source_counts, roi_failures)
                    rows.extend(records)

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

                # Timeouts are fatal for required ROIs but are recorded and
                # tolerated for best-effort TotalSegmentator organ ROIs.
                timed_out_tasks: List[_RoiTask] = []
                for fut in list(remaining):
                    if now - task_start[fut] > task_timeout:
                        task = futures[fut]
                        timed_out_tasks.append(task)
                        completed.add(task)
                        remaining.remove(fut)
                if timed_out_tasks:
                    required_timeouts = [task for task in timed_out_tasks if _task_is_required(task)]
                    for task in timed_out_tasks:
                        if task in required_timeouts:
                            continue
                        records = _failure_records(
                            task,
                            f"ROI task timed out after {task_timeout}s",
                            failure_kind="timeout",
                        )
                        rows.extend(records)
                        _record_roi_outcome(task, records, source_counts, roi_failures)
                    if required_timeouts:
                        failed_names = ", ".join(
                            f"{task.source}/{task.roi_name}" for task in required_timeouts
                        )
                        fatal_error = RadiomicsCourseExtractionError(
                            f"Radiomics course {course_dir} is incomplete: required ROI task(s) "
                            f"timed out after {task_timeout}s: {failed_names}"
                        )
                        break
                    restart = True
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

    records_to_write = list(rows)
    if not records_to_write:
        if existing_df is not None and out_path.exists():
            _write_parallel_roi_ledger(course_dir, tasks, existing_df.to_dict("records"), custom_applicability, extracted=True)
            return _resume_outcome(out_path, tasks, existing_df)
        logger.info("No eligible radiomics regions for %s", course_dir)
        _invalidate_radiomics_outputs(out_path)
        _write_parallel_roi_ledger(course_dir, tasks, (), custom_applicability, extracted=False, technical=True)
        return RadiomicsCourseOutcome.nothing_to_do(
            "all enumerated ROIs were explicitly skipped",
            roi_counts=source_counts,
        )

    outcome = RadiomicsCourseOutcome.extracted(
        out_path,
        roi_counts=source_counts,
        roi_failures=roi_failures,
        detail=(
            f"extracted {sum(values['extracted'] for values in source_counts.values())} "
            f"of {sum(values['attempted'] for values in source_counts.values())} "
            "attempted ROIs"
        ),
    )
    diagnostics = course_diagnostic_columns(outcome)
    for row in records_to_write:
        row.update(diagnostics)

    try:
        import pandas as pd  # type: ignore

        attach_acquisition_descriptor(
            records_to_write,
            acquisition_descriptor,
        )
        df_new = pd.DataFrame(records_to_write)
        if existing_df is not None:
            output_cols = list(existing_df.columns)
            output_cols.extend(col for col in df_new.columns if col not in existing_df.columns)
            for col in output_cols:
                if col not in existing_df.columns:
                    existing_df[col] = None
                if col not in df_new.columns:
                    df_new[col] = None
            df = pd.concat(
                [existing_df.loc[:, output_cols], df_new.loc[:, output_cols]],
                ignore_index=True,
            )
        else:
            df = df_new
        write_ct_publication_atomic(df, out_path, expected_keys=expected_keys)
        published_df = read_authoritative_ct_publication(out_path)
        _write_parallel_roi_ledger(
            course_dir,
            tasks,
            published_df.to_dict("records"),
            custom_applicability,
            extracted=True,
            technical=bool(roi_failures)
            or bool(
                published_df[RADIOMICS_FEATURE_COMPLETENESS_COLUMN]
                .astype(str)
                .eq("incomplete")
                .any()
            ),
        )
        return outcome
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
    run_identifier: Optional[str] = None,
    source_identity: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Save image/mask to temp files and prepare a task descriptor.

    Returns (mask_path, task_params) tuple suitable for passing to
    ``_isolated_radiomics_extraction_with_retry``.
    """
    import hashlib

    import SimpleITK as sitk
    import numpy as np
    from .radiomics import _get_params_file
    from .radiomics_robustness import (
        ROBUSTNESS_MEASUREMENT_TYPE,
        RobustnessRoiIdentity,
        _perturbed_mask_identity,
    )
    from .radiomics_ct_contract import (
        CT_EXTRACTION_ARMS,
        PRIMARY_ARM,
        classify_ct_roi,
        configured_parameter_hash,
        current_code_revision,
        load_custom_structure_provenance,
        new_run_identifier,
    )

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

    if source_identity is None:
        raise RuntimeError(
            f"CT robustness identity is unavailable for {source}/{roi_name}"
        )
    identity = RobustnessRoiIdentity.from_mapping(source_identity)
    expected_context = {
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "segmentation_source": source,
        "roi_original_name": roi_name,
    }
    mismatches = [
        f"{column}={getattr(identity, column)!r} expected {value!r}"
        for column, value in expected_context.items()
        if getattr(identity, column) != str(value)
    ]
    if mismatches:
        raise RuntimeError(
            "CT robustness source identity does not match task context: "
            + "; ".join(mismatches)
        )

    perturbed_mask_identity = _perturbed_mask_identity(mask)
    mask_path = temp_dir / f"mask_{perturbed_mask_identity.split(':', 1)[1]}.nrrd"
    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(mask_path))

    params_file = _get_params_file(config, "CT")
    custom_path = getattr(config, "custom_structures_config", None)
    decision = classify_ct_roi(
        source,
        roi_name,
        custom_provenance=(
            load_custom_structure_provenance(Path(custom_path))
            if source == "Custom" and custom_path
            else None
        ),
    )
    configured_hashes = {
        arm: configured_parameter_hash(
            params_file,
            arm=arm,
            window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
            large_roi=large_roi,
        )
        for arm in CT_EXTRACTION_ARMS
    }
    task_params = {
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "segmentation_source": source,
        "roi_name": roi_name,
        **identity.as_dict(),
        "large_roi": large_roi,
        "params_file": str(params_file) if params_file else None,
        "dual_arm_ct": True,
        "roi_class_decision": {
            "roi_class": decision.roi_class,
            "map_version": decision.map_version,
            "map_hash": decision.map_hash,
            "map_entry_source": decision.map_entry_source,
            "adjudication_status": decision.adjudication_status,
            "primary_resegment_range_hu": decision.primary_resegment_range_hu,
            "primary_intensity_texture_disposition": decision.primary_intensity_texture_disposition,
            "feature_publication_policy": decision.feature_publication_policy,
        },
        "run_identifier": run_identifier or new_run_identifier(),
        "code_revision": current_code_revision(),
        "native_voxel_count": int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
        "configured_parameter_hashes": configured_hashes,
        "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
        "perturbed_mask_identity": perturbed_mask_identity,
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

    import warnings
    warnings.filterwarnings("ignore")
    import logging as _logging
    _logging.getLogger("radiomics").setLevel(_logging.ERROR)

    from radiomics import featureextractor  # type: ignore
    from .radiomics_ct_contract import RoiClassDecision, extract_ct_roi_arms

    def _factory():
        candidate = (
            featureextractor.RadiomicsFeatureExtractor(params_file)
            if params_file
            else featureextractor.RadiomicsFeatureExtractor()
        )
        if large_roi:
            candidate.disableAllImageTypes()
            candidate.enableImageTypeByName("Original")
            candidate.disableAllFeatures()
            candidate.enableFeatureClassByName("firstorder")
            candidate.enableFeatureClassByName("shape")
            candidate.settings["resampledPixelSpacing"] = [2.0, 2.0, 2.0]
        return candidate

    try:
        decision = RoiClassDecision(**task_params["roi_class_decision"])
        common_metadata = {
            "patient_id": task_params.get("patient_id", ""),
            "course_id": task_params.get("course_id", ""),
            "series_uid": task_params.get("series_uid", ""),
            "segmentation_source": task_params.get("segmentation_source", ""),
            "mask_identity": task_params.get("mask_identity", ""),
            "roi_original_name": task_params.get("roi_original_name", ""),
            "stable_roi_identifier": task_params.get("stable_roi_identifier", ""),
            "measurement_type": task_params.get("measurement_type", ""),
            "perturbed_mask_identity": task_params.get(
                "perturbed_mask_identity", ""
            ),
            "roi_name": task_params.get("roi_name", ""),
            "modality": "CT",
        }
        records = extract_ct_roi_arms(
            task_params["image_path"],
            task_params["mask_path"],
            factory=_factory,
            decision=decision,
            common_metadata=common_metadata,
            run_identifier=task_params["run_identifier"],
            code_revision=task_params["code_revision"],
            native_voxel_count=int(task_params["native_voxel_count"]),
            required=False,
            configured_parameter_hashes=task_params["configured_parameter_hashes"],
        )
        return {
            "__records__": records,
            "segmentation_source": task_params.get("segmentation_source", ""),
            "roi_name": task_params.get("roi_name", ""),
            "patient_id": task_params.get("patient_id", ""),
            "course_id": task_params.get("course_id", ""),
            **extra_metadata,
        }

    except Exception as e:
        logger.debug(
            "Isolated extraction failed for %s/%s: %s",
            task_params.get("roi_name"),
            extra_metadata.get("perturbation_id", "?"),
            e,
        )
        return None
