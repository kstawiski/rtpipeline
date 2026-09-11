#!/usr/bin/env python3
"""
Radiomics module using conda environment isolation.
Runs PyRadiomics in a separate conda environment with NumPy 1.x for compatibility.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Tuple, Sequence, Set, TYPE_CHECKING,
)

if TYPE_CHECKING:
    pass  # For future type hints

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom

from .acquisition_scale import (
    attach_acquisition_descriptor,
    describe_contract_planning_ct,
)
from .custom_models import (
    list_custom_model_outputs,
    validate_custom_model_output_inventory,
)
from .layout import build_course_dirs
from .course_contract import ALL_SERIES_RADIOMICS_TEMP_SCOPE, load_course_contract
from .roi_requiredness import (
    DenominatorLedger,
    FAILED_RADIOMICS_FEATURE_COMPLETENESS,
    FAILED_RADIOMICS_RESOURCE_LIMIT,
    NONVOLUMETRIC_CODES,
    REASON_CODES,
    Requiredness,
    TAXONOMY_CODES,
    assess_custom_applicability,
    requiredness_for,
    requirements_from_contract,
    write_modality_ledger,
)
from .radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    course_diagnostic_columns,
    extraction_status_is_nonfatal_for_required,
    invalidate_radiomics_outputs as _invalidate_radiomics_outputs,
    remove_artifact_strict as _remove_artifact_strict,
)
from .radiomics_memory import permits_second_stage, legacy_rejection
from .radiomics_resource_guard import (
    RESAMPLED_BBOX_LIMIT_CODE,
    configured_grid_settings,
    estimate_resampled_bounding_box,
    resolve_max_resampled_bbox_voxels,
)
from .radiomics_schema import (
    RadiomicsFeatureTypeError,
    assert_radiomics_arrow_schema,
    expected_radiomics_string_columns,
    normalize_radiomics_dataframe,
    normalize_radiomics_result,
    write_radiomics_feature_table_atomic,
)
from .radiomics_ct_contract import (
    CT_EXTRACTION_ARMS,
    PRIMARY_ARM,
    RADIOMICS_FEATURE_COMPLETENESS_COLUMN,
    RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN,
    SENSITIVITY_ARM,
    RoiClassDecision,
    classify_ct_roi,
    configured_parameter_hash,
    current_code_revision,
    disposition_rows_for_arms,
    effective_parameter_hash,
    file_sha256,
    load_custom_structure_provenance,
    new_run_identifier,
    publication_key,
    rtstruct_roi_identities,
    stable_rtstruct_roi_identity,
    validate_ct_publication,
    write_ct_publication_atomic,
)
from .utils import mask_is_cropped, radiomics_mp_context

logger = logging.getLogger(__name__)


_NON_TECHNICAL_LEDGER_REASONS = frozenset({
    "extracted",
    "not_applicable_modality",
    "not_applicable_scope",
    "not_applicable_anatomy",
    "insufficient_fov",
    "not_computed_valid_empty_scope",
    "CONFIGURED_SKIP",
    "CONFIGURED_SOURCE_SCOPE_SKIP",
}) | NONVOLUMETRIC_CODES


def _ledger_row_is_technical(row: Mapping[str, Any]) -> bool:
    """True for a ROI outcome ``DenominatorLedger.summary`` counts as technical.

    Modality absence, anatomy, declared skips and valid-but-empty nonmeasurements
    are not technical exclusions. Keeping those classes apart is the whole point
    of the reason codes, so this reuses the ledger's own classification.
    """
    if str(row.get("disposition", "")) == "extracted":
        return False
    return str(row.get("reason_code", "")) not in _NON_TECHNICAL_LEDGER_REASONS


def _ledger_identity_key(
    entry: Mapping[str, Any], identity_fields: Sequence[str]
) -> Tuple[str, ...]:
    """Source-identity values that make two same-named ROI outcomes distinct.

    A task carries its identity under ``metadata``; a published row carries it at
    the top level. Without this, one ROI name measured from two series collapses
    into a single ledger row and the failed series hides behind the good one.
    """
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    values: List[str] = []
    for field in identity_fields:
        value = entry.get(field)
        if value is None or value == "":
            value = metadata.get(field)
        values.append("" if value is None else str(value))
    return tuple(values)


def _write_conda_roi_ledger(
    course_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    extracted: bool,
    expected_names: Sequence[str] = (),
    missing_reason: str = "failed_radiomics_extraction",
    in_scope: bool = True,
    modality: str = "CT",
    identity_fields: Sequence[str] = (),
    missing_reason_for: Optional[Callable[[str], str]] = None,
    derive_course_states: bool = False,
) -> None:
    ledger = DenominatorLedger()
    course_id, patient_id = Path(course_dir).name, Path(course_dir).parent.name
    for task in tasks:
        ledger.expect_course_roi(course_id, str(task.get("roi_name", "")))
    for name in expected_names:
        ledger.expect_course_roi(course_id, str(name))
    rows_by_identity: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        name = str(row.get("roi_original_name", row.get("roi_name", "")))
        if name:
            key = (name,) + _ledger_identity_key(row, identity_fields)
            rows_by_identity.setdefault(key, []).append(row)
    has_incomplete = any(
        str(row.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or "")
        == "incomplete"
        for row in rows
    )
    for identity, candidates in rows_by_identity.items():
        name = identity[0]
        identity_values = dict(zip(identity_fields, identity[1:]))
        row = next(
            (
                item
                for item in candidates
                if str(item.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or "")
                == "incomplete"
            ),
            candidates[0],
        )
        status = str(row.get("extraction_status") or "success")
        detail_code = str(row.get("roi_structural_code") or "")
        completeness = str(
            row.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or ""
        )
        reason = str(
            row.get("reason_code")
            or detail_code
            or (
                "extracted"
                if status in {"success", "declared_skip"}
                else "failed_radiomics_extraction"
            )
        )
        if completeness == "incomplete":
            reason = FAILED_RADIOMICS_FEATURE_COMPLETENESS
        elif status == "below_minimum_voxels":
            reason = "ROI_MASK_BELOW_MIN_VOXELS"
        elif reason in {RESAMPLED_BBOX_LIMIT_CODE, "ROI_PREDICTED_MEMORY_EXCEEDS_LIMIT"}:
            reason = FAILED_RADIOMICS_RESOURCE_LIMIT
        ledger.record_roi(
            course_id,
            patient_id,
            name,
            reason_code=reason,
            disposition="extracted" if reason == "extracted" else "excluded",
            detail_code=detail_code or None,
            detail=str(
                row.get(RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN)
                if completeness == "incomplete"
                else row.get("extraction_status_detail")
                or ""
            ),
            **identity_values,
            **({"resource_guard_reason_code": row["resource_guard_reason_code"]}
                   if row.get("resource_guard_reason_code") in {"ROI_RESOURCE_BBOX_ADMITTED", "ROI_RESOURCE_MEMORY_ADMITTED"} else {}),
                estimated_resampled_bbox_voxel_count=row.get(
                "estimated_resampled_bbox_voxel_count"
            ),
            max_resampled_bbox_voxel_count=row.get(
                "max_resampled_bbox_voxel_count"
            ),
        )
    recorded_identities = {
        (str(row.get("course_id", "")), str(row.get("roi_name", "")))
        + _ledger_identity_key(row, identity_fields)
        for row in ledger.roi_rows
    }
    for task in tasks:
        name = str(task.get("roi_name", ""))
        identity_values = dict(
            zip(identity_fields, _ledger_identity_key(task, identity_fields))
        )
        identity = (course_id, name) + tuple(identity_values.values())
        if name and identity not in recorded_identities:
            failure = task.get("precomputed_failure") or {}
            reason = str(failure.get("reason_code") or "failed_radiomics_extraction")
            if reason in {RESAMPLED_BBOX_LIMIT_CODE, "ROI_PREDICTED_MEMORY_EXCEEDS_LIMIT"}:
                reason = FAILED_RADIOMICS_RESOURCE_LIMIT
            if reason not in REASON_CODES and reason not in TAXONOMY_CODES:
                reason = "failed_radiomics_extraction"
            failure_metadata = dict(failure.get("metadata") or {})
            ledger.record_roi(
                course_id,
                patient_id,
                name,
                reason_code=reason,
                disposition="excluded",
                detail_code=failure_metadata.get("roi_structural_code"),
                detail=str(failure.get("reason") or ""),
                **identity_values,
                estimated_resampled_bbox_voxel_count=failure_metadata.get(
                    "estimated_resampled_bbox_voxel_count"
                ),
                max_resampled_bbox_voxel_count=failure_metadata.get(
                    "max_resampled_bbox_voxel_count"
                ),
            )
            recorded_identities.add(identity)
    present_names = {str(row.get("roi_name", "")) for row in ledger.roi_rows}
    for name in expected_names:
        if str(name) not in present_names:
            reason = (
                missing_reason_for(str(name))
                if missing_reason_for is not None
                else missing_reason
            )
            ledger.record_roi(course_id, patient_id, str(name), reason_code=reason, disposition="excluded")
    technical_exclusion = not extracted or has_incomplete
    course_reason = "extracted" if extracted else "failed_radiomics_extraction"
    if derive_course_states:
        # A published workbook is not proof that every ROI was measured, and a
        # course that only ever held nonmeasurements was not excluded by a
        # technical failure. Both directions are read from the recorded outcomes.
        technical_exclusion = has_incomplete or any(
            _ledger_row_is_technical(row) for row in ledger.roi_rows
        )
        if extracted:
            course_reason = "extracted"
        elif technical_exclusion:
            course_reason = "failed_radiomics_extraction"
        else:
            remaining = {
                str(row.get("reason_code", ""))
                for row in ledger.roi_rows
                if str(row.get("reason_code", "")) != "extracted"
            }
            course_reason = (
                remaining.pop() if len(remaining) == 1 else "not_applicable_modality"
            )
    ledger.record_course(
        course_id,
        patient_id,
        screened=True,
        in_scope=in_scope,
        out_of_scope=not in_scope,
        adequate_coverage=bool(rows),
        insufficient_coverage=not bool(rows),
        valid_derivation=False,
        technical_exclusion=technical_exclusion,
        indeterminate=False,
        extracted=extracted,
        reason_code=course_reason,
    )
    write_modality_ledger(Path(course_dir) / "metadata", ledger, modality)


RADIOMICS_ENV = os.environ.get("RTPIPELINE_RADIOMICS_ENV", "rtpipeline-radiomics")
_ENV_PROBE_TIMEOUT_ENV = "RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT"
_DEFAULT_ENV_PROBE_TIMEOUT = 180


class RadiomicsEnvironmentProbeTimeout(RuntimeError):
    """The isolated radiomics import probe exhausted its time budget."""

    code = "RADIOMICS_ENV_PROBE_TIMEOUT"

    def __init__(self, command: List[str], timeout: int, attempts: int) -> None:
        self.command = tuple(command)
        self.timeout = int(timeout)
        self.attempts = int(attempts)
        super().__init__(
            "radiomics environment probe timed out: "
            f"command={shlex.join(command)!r}, timeout={timeout}s, attempts={attempts}"
        )

    def __reduce__(self):
        return (type(self), (list(self.command), self.timeout, self.attempts))


def _conda_executable() -> str:
    """Resolve a conda-compatible executable for isolated radiomics calls."""
    configured = os.environ.get("RTPIPELINE_CONDA_EXE")
    if configured:
        return configured
    for candidate in ("conda", "micromamba", "mamba"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "conda"  # Preserve the historical error message when none is installed.


CONDA_EXE = _conda_executable()

# Heartbeat interval for progress logging (seconds)
HEARTBEAT_INTERVAL = 60

# Cache for a confirmed-functional radiomics conda env. The env does not change
# mid-run, but ``process_radiomics_batch`` is called once per course, so a naive
# per-course ``conda run`` probe spawns a fresh subprocess every time. Under the
# nested worker load (course-level threads each driving a ProcessPoolExecutor of
# radiomics workers) that probe's startup contends for the box and times out,
# returning False and silently skipping an otherwise-healthy course. We therefore
# cache the first SUCCESSFUL check process-wide (double-checked under a lock) so a
# transient probe timeout cannot drop courses. A False result is never cached, so
# a genuinely-not-yet-ready env can still recover on a later call.
_ENV_CHECK_LOCK = threading.Lock()
_ENV_CHECK_OK: Optional[bool] = None


def _jsonify_nested_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """JSON-encode any column holding nested (dict/list/set/tuple) values for Parquet.

    PyRadiomics emits diagnostics fields such as
    ``diagnostics_Configuration_EnabledImageTypes`` whose value can be a nested
    dict like ``{'Original': {}}`` — an empty-child struct that PyArrow cannot
    encode ("Cannot write struct type 'Original' with no child field to
    Parquet"), so every checkpoint flush failed. We JSON-encode nested values to
    strings (rather than dropping the columns) so the checkpoint remains a
    FAITHFUL record of every completed ROI: it is the data source used to rebuild
    the per-course workbook on resume, and dropping provenance columns would make
    a resumed course's ``radiomics_ct.xlsx`` lose those columns. Scalar columns
    are left untouched.
    """
    if df.empty:
        return df
    out = df
    copied = False
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (dict, list, set, tuple))).any():
            if not copied:
                out = df.copy()
                copied = True
            out[col] = out[col].map(
                lambda v: json.dumps(v, default=str, sort_keys=True)
                if isinstance(v, (dict, list, set, tuple))
                else v
            )
    return out


def _identity_value(obj: Dict[str, Any], name: str, fallback: str = "") -> str:
    metadata = obj.get("metadata") or obj.get("extra_metadata") or {}
    value = obj.get(name)
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = value is None
    if missing:
        value = metadata.get(name)
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = value is None
    if missing and fallback:
        value = obj.get(fallback)
        try:
            fallback_missing = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            fallback_missing = value is None
        if fallback_missing:
            value = metadata.get(fallback)
    try:
        if value is None or bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        if value is None:
            return ""
    return str(value).strip()


def _ct_publication_key_text(obj: Dict[str, Any], arm: Optional[str] = None) -> str:
    metadata = dict(obj.get("metadata") or obj.get("extra_metadata") or {})
    merged = {**metadata, **obj}
    if arm is not None:
        merged["extraction_arm"] = arm
    key = publication_key(merged)
    if any(not value for value in key):
        raise ValueError("CT checkpoint identity has a blank full publication-key field")
    return "\x1f".join(key)


def _roi_instance_key(obj: Dict[str, Any]) -> str:
    """Stable checkpoint identity, including the arm for governed CT rows."""
    arm = _identity_value(obj, "extraction_arm")
    if arm:
        return _ct_publication_key_text(obj, arm)
    source = _identity_value(obj, "segmentation_source")
    roi = _identity_value(obj, "roi_original_name", fallback="roi_name")
    series = _identity_value(obj, "series_uid")
    return f"{source}\x1f{roi}\x1f{series}"


def _task_expected_keys(task: Dict[str, Any]) -> Set[str]:
    if task.get("dual_arm_ct"):
        return {_ct_publication_key_text(task, arm) for arm in CT_EXTRACTION_ARMS}
    return {_roi_instance_key(task)}


def _validated_identity_keys(records: List[Dict[str, Any]], *, context: str) -> Set[str]:
    keys: List[str] = []
    for record in records:
        if _identity_value(record, "extraction_arm"):
            keys.append(_roi_instance_key(record))
            continue
        source = _identity_value(record, "segmentation_source")
        roi = _identity_value(record, "roi_original_name", fallback="roi_name")
        if not source or not roi:
            raise ValueError(
                f"{context} has blank segmentation_source/roi_original_name identity"
            )
        keys.append(_roi_instance_key(record))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} has duplicate publication identities")
    return set(keys)


def _strip_nii_suffix(path: Path) -> str:
    """Return a NIfTI filename stem, handling .nii.gz as one suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _norm_roi_key(name: str) -> str:
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _ct_skip_rois(config: Any) -> Set[str]:
    cfg_skip = {
        _norm_roi_key(item)
        for item in getattr(config, "radiomics_skip_rois", [])
        if isinstance(item, str) and item.strip()
    }
    return cfg_skip


def _ct_voxel_limits(config: Any) -> Tuple[int, int]:
    max_voxels_limit = getattr(config, "radiomics_max_voxels", None)
    if max_voxels_limit is None:
        max_voxels_limit = 15_000_000
    elif max_voxels_limit < 1:
        max_voxels_limit = 15_000_000

    min_voxels_limit = getattr(config, "radiomics_min_voxels", None)
    if min_voxels_limit is None:
        min_voxels_limit = 120
    elif min_voxels_limit < 1:
        min_voxels_limit = 1
    return int(min_voxels_limit), int(max_voxels_limit)


def _select_usable_rtstruct(*paths: Path) -> Optional[Path]:
    """Return the first RTSTRUCT path that exists and has at least one ROI."""
    for path in paths:
        if not path.exists():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            rois = getattr(ds, "StructureSetROISequence", []) or []
            if rois:
                return path
            logger.warning("Ignoring RTSTRUCT with no ROIs for CT radiomics: %s", path)
        except Exception as exc:
            raise RadiomicsCourseExtractionError(
                f"Required RTSTRUCT is unreadable for CT radiomics: {path}: {exc}"
            ) from exc
    return None


def _ct_nifti_candidates(course_dir: Path) -> Dict[str, Path]:
    nifti_dir = course_dir / "NIFTI"
    if not nifti_dir.exists():
        return {}
    out: Dict[str, Path] = {}
    for path in sorted(nifti_dir.glob("*.nii*")):
        if not path.is_file() or path.name.startswith(".") or "_cropped" in path.name:
            continue
        out[_strip_nii_suffix(path)] = path
    return out


def _series_uid_from_nifti(nifti_path: Path, fallback: str) -> str:
    meta_path = nifti_path.with_name(f"{_strip_nii_suffix(nifti_path)}.metadata.json")
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            uid = data.get("series_instance_uid")
            if uid:
                return str(uid)
        except Exception as exc:
            logger.debug("Failed to read NIfTI metadata %s: %s", meta_path, exc)
    return fallback


def _totalseg_roi_name(mask_path: Path) -> Optional[str]:
    name = _strip_nii_suffix(mask_path)
    if "--" in name:
        _model, name = name.split("--", 1)
    if name.endswith("_cropped"):
        return None
    return name or None


class RadiomicsCheckpoint:
    """Manages checkpoint state for resumable radiomics extraction.

    Saves completed ROI results to a Parquet file so extraction can resume
    if interrupted. This is critical for large datasets where extraction
    can take hours.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        buffer_size: int = 50,
        expected_keys: Optional[Set[str]] = None,
        expected_configured_hashes: Optional[Mapping[str, str]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._completed_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._load_existing(expected_keys, expected_configured_hashes)

    def _load_existing(
        self,
        expected_keys: Optional[Set[str]],
        expected_configured_hashes: Optional[Mapping[str, str]],
    ) -> None:
        """Reuse only a structurally valid current-contract checkpoint subset."""
        if not self.checkpoint_path.exists():
            return
        try:
            df = pd.read_parquet(self.checkpoint_path)
            assert_radiomics_arrow_schema(
                self.checkpoint_path,
                expected_string_columns=expected_radiomics_string_columns(df),
            )
            if df.empty:
                raise ValueError("checkpoint is empty")
            feature_markers = (
                "_firstorder_", "_shape_", "_shape2D_", "_glcm_",
                "_glrlm_", "_glszm_", "_gldm_", "_ngtdm_",
            )
            if not any(
                any(marker in str(column) for marker in feature_markers)
                for column in df.columns
            ):
                raise ValueError("checkpoint has no PyRadiomics feature column")
            records = df.to_dict("records")
            keys = _validated_identity_keys(records, context="checkpoint")
            if expected_keys is not None and keys != expected_keys:
                raise ValueError(
                    "checkpoint identity set is incomplete or outside the current tasks "
                    f"(expected {len(expected_keys)}, found {len(keys)})"
                )
            if any(_identity_value(record, "extraction_arm") for record in records):
                validate_ct_publication(df, fail_on_unclassified_required=False)
                if expected_configured_hashes is None:
                    raise ValueError("CT checkpoint lacks a current configured-parameter contract")
                for record in records:
                    key = _roi_instance_key(record)
                    if str(record.get("configured_parameter_hash") or "") != str(
                        expected_configured_hashes.get(key) or ""
                    ):
                        raise ValueError("CT checkpoint configured-parameter hash is stale")
            self._completed_keys = keys
            logger.info("Checkpoint loaded: %d ROIs already completed", len(keys))
        except Exception as exc:
            logger.warning(
                "Rejecting stale or invalid checkpoint %s: %s",
                self.checkpoint_path,
                exc,
            )
            _remove_artifact_strict(
                self.checkpoint_path,
                context="rejecting invalid radiomics checkpoint",
            )
            self._completed_keys.clear()

    def is_completed(self, key: str) -> bool:
        """Check if a ROI instance (see ``_roi_instance_key``) has been processed."""
        return key in self._completed_keys

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a completed result and flush if buffer is full."""
        with self._lock:
            self._buffer.append(result)
            if result.get('roi_name'):
                self._completed_keys.add(_roi_instance_key(result))

            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered results to checkpoint file."""
        if not self._buffer:
            return

        try:
            new_df = _jsonify_nested_columns(
                normalize_radiomics_dataframe(pd.DataFrame(self._buffer))
            )

            if self.checkpoint_path.exists():
                existing_df = pd.read_parquet(self.checkpoint_path)
                combined_df = normalize_radiomics_dataframe(
                    pd.concat([existing_df, new_df], ignore_index=True)
                )
            else:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df = new_df

            # Write to a temp file and atomically publish via os.replace(): writing
            # the parquet file in place would leave it truncated (and silently
            # discarded by _load_existing on the next run) if the process is
            # killed mid-write.
            tmp_path = self.checkpoint_path.parent / f".{self.checkpoint_path.name}.{os.getpid()}.tmp"
            try:
                combined_df.to_parquet(tmp_path, index=False)
                assert_radiomics_arrow_schema(
                    tmp_path,
                    expected_string_columns=expected_radiomics_string_columns(combined_df),
                )
                os.replace(tmp_path, self.checkpoint_path)
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            self._buffer.clear()
            logger.debug("Checkpoint flushed: %d total ROIs", len(self._completed_keys))
        except Exception as exc:
            logger.error("Failed to flush checkpoint: %s", exc)

    def flush(self) -> None:
        """Force flush any remaining buffered results."""
        with self._lock:
            self._flush_buffer()

    def get_completed_count(self) -> int:
        """Return number of completed ROI instances."""
        return len(self._completed_keys)

    def discard(self) -> None:
        """Delete a checkpoint that cannot represent a complete publication."""
        with self._lock:
            self._buffer.clear()
            self._completed_keys.clear()
            _remove_artifact_strict(
                self.checkpoint_path,
                context="discarding incomplete radiomics checkpoint",
            )

    def load_records(self) -> List[Dict[str, Any]]:
        """Return structurally valid checkpoint records without silent deduplication."""
        with self._lock:
            self._flush_buffer()  # make sure buffered rows are on disk first
            if not self.checkpoint_path.exists():
                return []
            try:
                df = pd.read_parquet(self.checkpoint_path)
            except Exception as exc:
                raise RadiomicsCourseExtractionError(
                    f"Failed to read checkpoint records {self.checkpoint_path}: {exc}"
                ) from exc
        records = df.to_dict("records")
        _validated_identity_keys(records, context="checkpoint")
        return records


class HeartbeatLogger:
    """Logs periodic progress updates during long-running operations.

    Starts a background thread that logs progress every HEARTBEAT_INTERVAL seconds.
    This provides visibility when radiomics extraction is running for hours.
    """

    def __init__(self, total_tasks: int, description: str = "Processing"):
        self.total_tasks = total_tasks
        self.description = description
        self._completed = 0
        self._failed = 0
        self._skipped = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self) -> None:
        """Start the heartbeat logging thread."""
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat logging thread and log final summary."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Log final summary
        self._log_final_summary()

    def _log_final_summary(self) -> None:
        """Log the final completion summary."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            logger.info(
                "[Complete] %s: %d completed, %d failed, %d skipped in %.1f minutes",
                self.description,
                self._completed,
                self._failed,
                self._skipped,
                elapsed / 60,
            )

    def update(self, completed: int = 0, failed: int = 0, skipped: int = 0) -> None:
        """Update progress counters."""
        with self._lock:
            self._completed += completed
            self._failed += failed
            self._skipped += skipped

    def _heartbeat_loop(self) -> None:
        """Background thread that logs progress periodically."""
        last_log = time.monotonic()
        while self._running:
            time.sleep(1)
            now = time.monotonic()
            if now - last_log >= HEARTBEAT_INTERVAL:
                self._log_progress()
                last_log = now

    def _log_progress(self) -> None:
        """Log current progress."""
        with self._lock:
            done = self._completed + self._failed + self._skipped
            elapsed = time.monotonic() - self._start_time

            if done > 0 and elapsed > 0:
                rate = done / elapsed
                remaining = self.total_tasks - done
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}hr"
            else:
                eta_str = "calculating..."

            logger.info(
                "[Heartbeat] %s: %d/%d done (%d completed, %d failed, %d skipped) - ETA: %s",
                self.description,
                done,
                self.total_tasks,
                self._completed,
                self._failed,
                self._skipped,
                eta_str,
            )

    def __enter__(self) -> 'HeartbeatLogger':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()  # stop() now includes final summary logging


# Thread-limiting environment variables for subprocesses
_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)


def _conda_subprocess_env() -> Dict[str, str]:
    """Create environment for conda subprocess with thread limits.

    CRITICAL: Each subprocess must have thread limits set to prevent
    CPU oversubscription. Without this, N workers × M threads each
    can spawn 100+ threads fighting for CPU cores.
    """
    env = os.environ.copy()
    env.setdefault("CONDA_NO_PLUGINS", "1")
    env.setdefault("CONDA_OVERRIDE_CUDA", "0")

    # Get thread limit from environment or default to 1
    # Using 1 thread per subprocess is optimal when running many parallel subprocesses
    thread_limit = os.environ.get("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    try:
        thread_limit = str(max(1, int(thread_limit)))
    except (ValueError, TypeError):
        thread_limit = "1"

    # Set thread limits for all common libraries
    for var in _THREAD_ENV_VARS:
        env.setdefault(var, thread_limit)

    return env


@lru_cache(maxsize=64)
def _materialized_ct_arm_hashes(
    params_file: str,
    large_roi: bool,
    decision_json: str,
) -> Dict[str, str]:
    """Materialize the selected Conda extractors and return both runtime hashes."""
    payload = json.dumps(
        {
            "params_file": params_file or None,
            "large_roi": bool(large_roi),
            "decision": json.loads(decision_json),
        },
        separators=(",", ":"),
    )
    script = r'''
import json
import sys
from radiomics import featureextractor
from rtpipeline.radiomics_ct_contract import (
    RoiClassDecision,
    effective_parameter_hashes_for_arms,
)

payload = json.loads(sys.argv[1])

def factory():
    params_file = payload.get("params_file")
    extractor = (
        featureextractor.RadiomicsFeatureExtractor(params_file)
        if params_file
        else featureextractor.RadiomicsFeatureExtractor()
    )
    # The legacy size flag is not authority to change the configured method.
    return extractor

decision = RoiClassDecision(**payload["decision"])
print(json.dumps(effective_parameter_hashes_for_arms(factory, decision)))
'''
    result = subprocess.run(
        [CONDA_EXE, "run", "-n", RADIOMICS_ENV, "python", "-c", script, payload],
        capture_output=True,
        text=True,
        timeout=120,
        env=_conda_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to materialize Conda radiomics parameter provenance: "
            f"{result.stderr.strip()}"
        )
    try:
        hashes = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Conda radiomics parameter-provenance probe returned invalid JSON"
        ) from exc
    if set(hashes) != set(CT_EXTRACTION_ARMS) or any(
        not str(hashes.get(arm) or "").strip() for arm in CT_EXTRACTION_ARMS
    ):
        raise RuntimeError("Conda radiomics parameter-provenance probe returned incomplete hashes")
    return {arm: str(hashes[arm]) for arm in CT_EXTRACTION_ARMS}


def check_radiomics_env(timeout: Optional[int] = None, retries: int = 1) -> bool:
    """Check if the radiomics conda environment exists and is functional.

    A successful result is cached process-wide: the env cannot disappear
    mid-run, and re-probing per course spawned a ``conda run`` subprocess whose
    cold-start under heavy worker load timed out spuriously, silently skipping
    healthy courses. The first check uses a generous timeout and one retry so a
    transient cold-start stall does not produce a false negative; only a True
    result is cached, so a genuinely-unready env can still recover later. The
    timeout can be set with ``RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT`` (or explicitly)
    and an exhausted timeout raises a typed error instead of masquerading as a
    feature-extraction failure.
    """
    global _ENV_CHECK_OK
    if _ENV_CHECK_OK:
        return True
    with _ENV_CHECK_LOCK:
        if _ENV_CHECK_OK:
            return True
        configured_timeout = timeout
        if configured_timeout is None:
            try:
                configured_timeout = int(os.environ.get(_ENV_PROBE_TIMEOUT_ENV, _DEFAULT_ENV_PROBE_TIMEOUT))
            except (TypeError, ValueError):
                configured_timeout = _DEFAULT_ENV_PROBE_TIMEOUT
        if configured_timeout <= 0:
            configured_timeout = _DEFAULT_ENV_PROBE_TIMEOUT

        command = [
            CONDA_EXE,
            "run",
            "-n",
            RADIOMICS_ENV,
            "python",
            "-c",
            "import radiomics; import numpy; print('OK')",
        ]
        attempts = max(1, int(retries) + 1)
        timeout_failures = 0
        last_err: Optional[str] = None
        for attempt in range(attempts):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=configured_timeout,
                    env=_conda_subprocess_env(),
                )
                if result.returncode == 0 and "OK" in result.stdout:
                    _ENV_CHECK_OK = True
                    return True
                last_err = (result.stderr or "").strip() or f"returncode={result.returncode}"
            except subprocess.TimeoutExpired as e:
                timeout_failures += 1
                last_err = str(e)
                logger.warning(
                    "Radiomics env check attempt %d/%d timed out after %ds",
                    attempt + 1,
                    attempts,
                    configured_timeout,
                )
            except Exception as e:
                last_err = str(e)
                logger.warning(
                    "Radiomics env check attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    e,
                )
        if timeout_failures == attempts:
            raise RadiomicsEnvironmentProbeTimeout(command, configured_timeout, attempts)
        logger.error("Failed to verify radiomics environment after %d attempts: %s", attempts, last_err)
        return False


def extract_radiomics_with_conda(
    image_path: str,
    mask_path: str,
    params_file: Optional[str] = None,
    label: Optional[int] = None,
    large_roi: bool = False,
) -> Dict[str, Any]:
    """
    Extract radiomics features using conda environment.

    Args:
        image_path: Path to image file (NRRD format)
        mask_path: Path to mask file (NRRD format)
        params_file: Optional path to radiomics parameters YAML file
        label: Optional label value for the mask
        large_roi: Compatibility flag. Does not change configured features,
            image types or resampling. Resource failures are not method overrides.

    Returns:
        Dictionary of extracted features
    """
    # Create extraction script
    extraction_script = '''
import sys
import json
import warnings
import SimpleITK as sitk
from radiomics import featureextractor
from rtpipeline.radiomics_schema import normalize_radiomics_result
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

# Read parameters from stdin
params = json.loads(sys.stdin.read())

image_path = params['image_path']
mask_path = params['mask_path']
params_file = params.get('params_file')
label = params.get('label')

# Create extractor
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()

# Execute extraction
if label is not None:
    features = extractor.execute(image_path, mask_path, label=label)
else:
    features = extractor.execute(image_path, mask_path)

# Normalize feature scalars before JSON transport; invalid features fail closed.
output = normalize_radiomics_result(features)

# Output as JSON
print(json.dumps(output))
'''

    # Prepare input parameters
    input_params = {
        'image_path': image_path,
        'mask_path': mask_path,
        'params_file': params_file,
        'label': label,
        'large_roi': bool(large_roi),
    }

    try:
        # Write parameters to temporary file to avoid stdin issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as param_file:
            json.dump(input_params, param_file)
            param_file_path = param_file.name

        # Modified extraction script to read from file
        extraction_script_with_file = f'''
import sys
import json
import warnings
import SimpleITK as sitk
from radiomics import featureextractor
from rtpipeline.radiomics_schema import normalize_radiomics_result
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

# Read parameters from file
with open('{param_file_path}', 'r') as f:
    params = json.load(f)

image_path = params['image_path']
mask_path = params['mask_path']
params_file = params.get('params_file')
label = params.get('label')
large_roi = bool(params.get('large_roi'))


def observed_extractor_state(instance):
    # Only this interpreter can see the settings PyRadiomics really used, so
    # they travel back with the features rather than being reconstructed from
    # the configuration the caller intended.
    state = {{
        'settings': dict(getattr(instance, 'settings', {{}}) or {{}}),
        'image_types': dict(getattr(instance, 'enabledImagetypes', {{}}) or {{}}),
        'features': dict(getattr(instance, 'enabledFeatures', {{}}) or {{}}),
    }}
    return json.loads(json.dumps(state, default=str))

# Create extractor
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()

# Preserve configured features, image types and spacing for every ROI size.
# Resource failures are accounted by callers, never hidden by a reduced method.

# Execute extraction
if label is not None:
    features = extractor.execute(image_path, mask_path, label=label)
else:
    features = extractor.execute(image_path, mask_path)

# Normalize feature scalars before JSON transport; invalid features fail closed.
output = normalize_radiomics_result(features)
output['__effective_extractor_state__'] = observed_extractor_state(extractor)

# Output as JSON
print(json.dumps(output))
'''

        # Run extraction in conda environment
        result = subprocess.run(
            [CONDA_EXE, "run", "-n", RADIOMICS_ENV, "python", "-c", extraction_script_with_file],
            capture_output=True,
            text=True,
            timeout=900,  # allow up to 15 minutes for large ROIs
            env=_conda_subprocess_env(),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Radiomics extraction failed: {result.stderr}")

        # Parse and return features
        features = json.loads(result.stdout)

        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass

        return normalize_radiomics_result(features)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse radiomics output: {e}")
        logger.error(f"stdout: {result.stdout[:500]}")
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass
        raise
    except subprocess.TimeoutExpired:
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass
        raise RuntimeError("Radiomics extraction timed out")
    except Exception as e:
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass
        raise RuntimeError(f"Radiomics extraction failed: {e}")


def extract_radiomics_batch_with_conda(
    tasks: List[Dict[str, Any]],
    params_file: Optional[str] = None,
    timeout_per_roi: int = 900,
) -> List[Dict[str, Any]]:
    """
    Extract radiomics features for multiple ROIs in a SINGLE subprocess.

    This dramatically reduces overhead by loading the radiomics library once
    and processing all ROIs sequentially within that subprocess.

    Args:
        tasks: List of dicts with 'image_path', 'mask_path', optional 'label', 'roi_name'
        params_file: Optional path to radiomics parameters YAML file
        timeout_per_roi: Timeout per ROI in seconds (default 900s = 15 min)

    Returns:
        List of feature dictionaries (one per task), None entries for failed ROIs
    """
    if not tasks:
        return []

    # Calculate total timeout based on number of tasks
    total_timeout = max(300, len(tasks) * timeout_per_roi)  # At least 5 minutes

    # Create batch extraction script that processes all ROIs in one go
    batch_script = r'''
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("radiomics.featureextractor").setLevel(logging.ERROR)

from radiomics import featureextractor
from rtpipeline.radiomics_ct_contract import (
    RoiClassDecision,
    effective_parameter_hash,
    extract_ct_roi_arms,
)
from rtpipeline.radiomics_schema import normalize_radiomics_result

with open(sys.argv[1], "r") as handle:
    batch_params = json.load(handle)

tasks = batch_params["tasks"]
default_params_file = batch_params.get("params_file")


def make_extractor(params_file, large_roi):
    candidate = (
        featureextractor.RadiomicsFeatureExtractor(params_file)
        if params_file
        else featureextractor.RadiomicsFeatureExtractor()
    )
    # large_roi is retained for compatibility, not as a method override.
    return candidate


for task in tasks:
    image_path = task["image_path"]
    mask_path = task["mask_path"]
    label = task.get("label")
    roi_name = task.get("roi_name", "ROI")
    large_roi = bool(task.get("large_roi"))
    task_index = task.get("__task_index__")
    params_file = task.get("params_file") or default_params_file
    try:
        if task.get("dual_arm_ct"):
            decision = RoiClassDecision(**task["roi_class_decision"])

            def factory():
                return make_extractor(params_file, large_roi)

            if task.get("robustness_perturbation_id"):
                from rtpipeline.radiomics_robustness_outcomes import extraction_nonmeasurement
                outcome = extraction_nonmeasurement(image_path, mask_path, factory)
                if outcome is not None:
                    print(json.dumps({
                        "__status__": "success", "__roi_name__": roi_name,
                        "__task_index__": task_index,
                        "__nonmeasurement__": {"reason_code": outcome.reason_code, "evidence": dict(outcome.evidence)},
                    }), flush=True)
                    continue
            records = extract_ct_roi_arms(
                image_path,
                mask_path,
                factory=factory,
                decision=decision,
                common_metadata={**task["metadata"],
                                 "_resource_guard_legacy": task.get("resource_guard_legacy")},
                run_identifier=task["run_identifier"],
                code_revision=task["code_revision"],
                native_voxel_count=int(task["native_voxel_count"]),
                required=bool(task["required"]),
                configured_parameter_hashes=task.get("configured_parameter_hashes"),
            )
            if task.get("robustness_perturbation_id"):
                from rtpipeline.radiomics_robustness_outcomes import returned_geometry_nonmeasurement
                outcome = returned_geometry_nonmeasurement(records, image_path, mask_path, factory)
                if outcome is not None:
                    print(json.dumps({
                        "__status__": "success", "__roi_name__": roi_name,
                        "__task_index__": task_index,
                        "__nonmeasurement__": {"reason_code": outcome.reason_code, "evidence": dict(outcome.evidence)},
                    }), flush=True)
                    continue
            print(
                json.dumps(
                    {
                        "__status__": "success",
                        "__roi_name__": roi_name,
                        "__task_index__": task_index,
                        "__records__": records,
                    },
                    default=str,
                ),
                flush=True,
            )
            continue

        extractor = make_extractor(params_file, large_roi)
        features = (
            extractor.execute(image_path, mask_path, label=label)
            if label is not None
            else extractor.execute(image_path, mask_path)
        )
        output = {
            "__status__": "success",
            "__roi_name__": roi_name,
            "__task_index__": task_index,
        }
        provenance_arm = task.get("parameter_provenance_arm")
        if provenance_arm:
            output["__effective_parameter_hash__"] = effective_parameter_hash(
                extractor,
                arm=str(provenance_arm),
                window=None,
            )
        output.update(normalize_radiomics_result(features))
        print(json.dumps(output, default=str), flush=True)
    except Exception as exc:
        message = str(exc)
        status = "skipped" if "size of the roi is too small" in message.lower() else "error"
        payload = {
            "__status__": status,
            "__roi_name__": roi_name,
            "__task_index__": task_index,
        }
        payload["__reason__" if status == "skipped" else "__error__"] = message
        print(json.dumps(payload), flush=True)
'''

    try:
        # Write batch parameters to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as batch_file:
            json.dump({
                'tasks': [
                    {**t, '__task_index__': i}
                    for i, t in enumerate(tasks)
                ],
                'params_file': params_file,
            }, batch_file)
            batch_file_path = batch_file.name

        # Run batch extraction in conda environment
        result = subprocess.run(
            [CONDA_EXE, "run", "-n", RADIOMICS_ENV, "python", "-c", batch_script, batch_file_path],
            capture_output=True,
            text=True,
            timeout=total_timeout,
            env=_conda_subprocess_env(),
        )

        # Parse results - one JSON per line
        results = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                results.append(parsed)
            except json.JSONDecodeError:
                logger.debug("Failed to parse batch output line: %s", line[:100])

        # Clean up
        try:
            os.unlink(batch_file_path)
        except OSError:
            pass

        if result.returncode != 0 and not results:
            logger.error("Batch radiomics failed: %s", result.stderr[:500] if result.stderr else "unknown error")
            return [None] * len(tasks)

        return results

    except subprocess.TimeoutExpired:
        logger.error("Batch radiomics timed out after %ds for %d tasks", total_timeout, len(tasks))
        try:
            os.unlink(batch_file_path)
        except (OSError, NameError):
            pass
        return [None] * len(tasks)

    except Exception as e:
        logger.error("Batch radiomics failed: %s", e)
        try:
            os.unlink(batch_file_path)
        except (OSError, NameError):
            pass
        return [None] * len(tasks)


def _write_mask_to_file(mask_array: np.ndarray, mask_path: str, ct_info: Dict[str, Any]) -> None:
    """Write a binary mask with CT geometry metadata."""

    arr = np.ascontiguousarray(mask_array.astype(np.uint8).transpose(2, 0, 1))
    img = sitk.GetImageFromArray(arr, isVector=False)

    spacing = tuple(ct_info.get('spacing', (1.0, 1.0, 1.0)))
    if len(spacing) < 3:
        spacing = tuple(list(spacing) + [1.0] * (3 - len(spacing)))
    img.SetSpacing(spacing)

    origin = ct_info.get('origin')
    if origin is not None:
        img.SetOrigin(tuple(origin))

    direction = ct_info.get('direction')
    if direction is not None:
        img.SetDirection(tuple(direction))

    sitk.WriteImage(img, mask_path, useCompression=True)


def _ensure_mask_has_three_dimensions(mask_path: str, ct_info: Dict[str, Any]) -> bool:
    """Ensure mask stored at ``mask_path`` has three dimensions.

    Returns True when the mask was rewritten.
    """

    try:
        mask_img = sitk.ReadImage(mask_path)
    except Exception as exc:
        logger.debug("Failed to read mask %s for dimension fix: %s", mask_path, exc)
        return False

    if mask_img.GetDimension() >= 3:
        return False

    arr2d = sitk.GetArrayFromImage(mask_img)
    arr3d = np.ascontiguousarray(np.expand_dims(arr2d, axis=0).astype(np.uint8))
    img3d = sitk.GetImageFromArray(arr3d, isVector=False)

    spacing = tuple(ct_info.get('spacing', (1.0, 1.0, 1.0)))
    if len(spacing) < 3:
        spacing = tuple(list(spacing) + [1.0] * (3 - len(spacing)))
    img3d.SetSpacing(spacing)

    origin = ct_info.get('origin')
    if origin is not None:
        img3d.SetOrigin(tuple(origin))

    direction = ct_info.get('direction')
    if direction is not None:
        img3d.SetDirection(tuple(direction))

    sitk.WriteImage(img3d, mask_path, useCompression=True)
    return True


class _ObservedExtractorState:
    """The extractor settings an isolated helper reported having actually used."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.settings = dict(state.get("settings") or {})
        self.enabledImagetypes = dict(state.get("image_types") or {})
        self.enabledFeatures = dict(state.get("features") or {})


def _bind_effective_parameter_hash(
    features: Dict[str, Any], task: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind a per-ROI helper result to the arm whose parameters it ran under.

    ``extract_radiomics_batch_with_conda`` knows the arm and hashes the settings
    inside the helper. The per-ROI route takes the same settings the same way --
    observed in the interpreter that ran the extraction -- and hashes them here,
    so a row measured by either route carries the same provenance instead of
    none. A result that already carries the hash keeps it untouched.
    """
    arm = task.get("parameter_provenance_arm")
    if not arm or features.get("__effective_parameter_hash__"):
        return features
    state = features.get("__effective_extractor_state__")
    if isinstance(state, Mapping):
        features["__effective_parameter_hash__"] = effective_parameter_hash(
            _ObservedExtractorState(state), arm=str(arm), window=None
        )
    return features


def _combine_feature_record(features: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    record.update(features)
    record.update(metadata)
    return record


def _process_batch(batch: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Process a batch of radiomics tasks in a subprocess and return (task, features) pairs."""
    params_file = batch[0].get('params_file') if batch else None
    batch_results = extract_radiomics_batch_with_conda(batch, params_file)

    # Match results back to tasks by the unique __task_index__ the subprocess echoes
    # back (injected per-task in extract_radiomics_batch_with_conda), not by roi_name
    # or a positional zip(batch, batch_results). roi_name is not guaranteed unique
    # within a batch (this module documents that elsewhere), so name-based matching
    # can still misattribute features between same-named ROIs when a subprocess
    # output line is dropped/unparseable -- only the injected index is unambiguous.
    results_by_index: Dict[int, Dict[str, Any]] = {}
    for result in batch_results:
        if not result:
            continue
        idx = result.get('__task_index__')
        if isinstance(idx, int) and idx not in results_by_index:
            results_by_index[idx] = result

    pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for i, task in enumerate(batch):
        roi_name = task.get('roi_name', 'ROI')
        result = results_by_index.get(i)
        if result is None:
            logger.warning("Batch radiomics: no matching result for ROI %s (task #%d); dropping", roi_name, i)
        pairs.append((task, result))
    return pairs


def process_radiomics_batch(
    tasks: List[Dict[str, Any]],
    output_path: Path,
    sequential: bool = False,
    max_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    enable_heartbeat: bool = True,
    env_probe_timeout: Optional[int] = None,
    acquisition_descriptor: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Process radiomics extraction tasks and persist them as an Excel sheet.

    Args:
        tasks: List of radiomics task dictionaries
        output_path: Path for output Excel file
        sequential: Force sequential processing
        max_workers: Maximum parallel workers
        checkpoint_path: Optional path for checkpoint file (enables resume)
        enable_heartbeat: Whether to enable progress heartbeat logging
        env_probe_timeout: Seconds allowed for each environment import probe
        acquisition_descriptor: Contract-bound descriptor required for CT rows

    Returns:
        Path to output file if successful, None otherwise
    """

    if not tasks:
        logger.warning("No radiomics tasks to process")
        _invalidate_radiomics_outputs(Path(output_path))
        return None

    try:
        expected_keys = set().union(*(_task_expected_keys(task) for task in tasks))
        expected_count = sum(len(_task_expected_keys(task)) for task in tasks)
        if len(expected_keys) != expected_count:
            raise ValueError("radiomics task inventory has duplicate full publication identities")
        expected_configured_hashes: Dict[str, str] = {}
        for task in tasks:
            if not task.get("dual_arm_ct"):
                continue
            hashes = task.get("configured_parameter_hashes") or {}
            for arm in CT_EXTRACTION_ARMS:
                expected_configured_hashes[_ct_publication_key_text(task, arm)] = str(
                    hashes.get(arm) or ""
                )
        if expected_configured_hashes and any(not value for value in expected_configured_hashes.values()):
            raise ValueError("CT task inventory has a blank configured-parameter hash")
    except ValueError as exc:
        _invalidate_radiomics_outputs(Path(output_path))
        raise RadiomicsCourseExtractionError(str(exc)) from exc

    # A checkpoint is reusable only when it already represents this exact task
    # inventory. Partial, stale, blank, duplicate, missing, or extra identities are
    # deleted and the full current inventory is recomputed.
    checkpoint: Optional[RadiomicsCheckpoint] = None
    if checkpoint_path:
        checkpoint_file_existed = Path(checkpoint_path).exists()
        checkpoint = RadiomicsCheckpoint(
            checkpoint_path,
            expected_keys=expected_keys,
            expected_configured_hashes=(expected_configured_hashes or None),
        )
        already_completed = checkpoint.get_completed_count()
        if checkpoint_file_existed and already_completed == 0:
            _invalidate_radiomics_outputs(Path(output_path))
        if already_completed > 0:
            logger.info("Resume mode: %d ROIs already completed, filtering tasks", already_completed)

    cleanup_paths = {
        task.get('mask_path')
        for task in tasks
        if task.get('mask_path') and task.get('cleanup', True)
    }

    # Filter out already-completed tasks if checkpointing is enabled
    original_task_count = len(tasks)
    if checkpoint:
        tasks = [
            task
            for task in tasks
            if not all(checkpoint.is_completed(key) for key in _task_expected_keys(task))
        ]
        skipped_count = original_task_count - len(tasks)
        if skipped_count > 0:
            logger.info("Skipping %d already-completed ROIs (resume mode)", skipped_count)

    # The conda env is only needed to COMPUTE features. If every ROI was already
    # completed in a prior run (tasks is now empty after checkpoint filtering), skip
    # the probe entirely and fall through to rebuild the workbook from the checkpoint
    # below. Otherwise the env must be functional before we try to extract anything.
    if tasks and not check_radiomics_env(timeout=env_probe_timeout):
        logger.error(
            "Radiomics conda environment '%s' not found or not functional",
            RADIOMICS_ENV,
        )
        logger.error(
            "Please run: conda create -n %s python=3.11 numpy=1.26.* pyradiomics SimpleITK -c conda-forge",
            RADIOMICS_ENV,
        )
        _invalidate_radiomics_outputs(Path(output_path))
        raise RadiomicsCourseExtractionError(
            f"Radiomics conda environment '{RADIOMICS_ENV}' is unavailable"
        )

    for task in tasks:
        if not task.get("dual_arm_ct"):
            continue
        decision_payload = json.dumps(
            task["roi_class_decision"],
            sort_keys=True,
            separators=(",", ":"),
        )
        task["effective_parameter_hashes"] = _materialized_ct_arm_hashes(
            str(task.get("params_file") or ""),
            bool(task.get("large_roi", False)),
            decision_payload,
        )

    def _execute(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        roi_name = task.get('roi_name', 'ROI')
        image_path = task.get('image_path')
        mask_path = task.get('mask_path')
        params_file = task.get('params_file')
        label = task.get('label')
        metadata = dict(task.get('metadata', {}))
        metadata.setdefault('roi_name', roi_name)
        metadata.setdefault('roi_original_name', roi_name)

        if task.get("precomputed_failure"):
            return {
                "__status__": "error",
                "__error__": str(task["precomputed_failure"].get("reason", "unknown error")),
                "__extraction_status__": str(
                    task["precomputed_failure"].get("status", "failed")
                ),
                "__failure_kind__": str(
                    task["precomputed_failure"].get("failure_kind", "extraction_error")
                ),
            }

        if not image_path or not mask_path:
            logger.error("Radiomics task is missing required paths for %s", roi_name)
            return {'__status__': 'error', '__error__': 'missing required image or mask path'}

        try:
            if task.get("dual_arm_ct"):
                batch_results = extract_radiomics_batch_with_conda([task], params_file)
                features = batch_results[0] if batch_results else None
                if features is None:
                    return {"__status__": "error", "__error__": "isolated dual-arm extraction returned no result"}
                return features
            features = extract_radiomics_with_conda(
                image_path,
                mask_path,
                params_file,
                label,
                bool(task.get("large_roi", False)),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if 'size of the roi is too small' in msg:
                logger.info(
                    "Skipping radiomics for ROI %s: %s",
                    roi_name,
                    str(exc).strip(),
                )
                return {'__status__': 'skipped', '__reason__': str(exc).strip()}
            if 'mask has too few dimensions' in msg and task.get('ct_info'):
                repaired = _ensure_mask_has_three_dimensions(mask_path, task['ct_info'])
                if repaired:
                    logger.debug("Rewrote 2D mask for ROI %s", roi_name)
                    try:
                        features = extract_radiomics_with_conda(
                            image_path,
                            mask_path,
                            params_file,
                            label,
                            bool(task.get("large_roi", False)),
                        )
                    except Exception as inner_exc:
                        logger.error("Radiomics retry failed for %s: %s", roi_name, inner_exc)
                        return {'__status__': 'error', '__error__': str(inner_exc)}
                else:
                    logger.error("Unable to repair mask dimensionality for %s", roi_name)
                    return {
                        '__status__': 'error',
                        '__error__': 'mask dimensionality could not be repaired',
                    }
            else:
                logger.error("Failed to extract features for %s: %s", roi_name, exc)
                return {'__status__': 'error', '__error__': str(exc)}

        metadata.setdefault('modality', 'CT')
        return _combine_feature_record(
            _bind_effective_parameter_hash(features, task), metadata
        )

    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    failure_rows: List[Dict[str, Any]] = []

    def _record_failure(
        task: Dict[str, Any],
        detail: str,
        *,
        status: str = "failed",
        failure_kind: str = "extraction_error",
    ) -> None:
        roi_name = str(task.get("roi_name", "ROI"))
        metadata = dict(task.get("metadata") or task.get("extra_metadata") or {})
        metadata.update(
            dict((task.get("precomputed_failure") or {}).get("metadata") or {})
        )
        source = str(metadata.get("segmentation_source", "unknown"))
        if (
            bool(task.get("required", True))
            and not extraction_status_is_nonfatal_for_required(status)
        ):
            failures.append(f"{source}/{roi_name}: {detail}")
            return
        metadata.setdefault("modality", "CT")
        metadata.setdefault("roi_name", roi_name)
        metadata.setdefault("roi_original_name", roi_name)
        if task.get("dual_arm_ct"):
            decision = RoiClassDecision(**task["roi_class_decision"])
            failure_rows.extend(
                disposition_rows_for_arms(
                    metadata,
                    decision=decision,
                    disposition=status,
                    detail=detail,
                    failure_kind=failure_kind,
                    run_identifier=str(task["run_identifier"]),
                    code_revision=str(task["code_revision"]),
                    native_voxel_count=task.get("native_voxel_count"),
                    required=bool(task.get("required", True)),
                    effective_hashes=task.get("effective_parameter_hashes"),
                    configured_parameter_hashes=task.get("configured_parameter_hashes"),
                )
            )
            return
        metadata.update(
            {
                "extraction_status": status,
                "extraction_status_detail": detail,
                "extraction_failure_kind": failure_kind,
            }
        )
        failure_rows.append(metadata)

    def _record_success(task: Dict[str, Any], features: Dict[str, Any]) -> None:
        if task.get("dual_arm_ct"):
            records = features.get("__records__")
            if not isinstance(records, list) or len(records) != len(CT_EXTRACTION_ARMS):
                raise ValueError("isolated CT extraction returned an incomplete arm set")
            arms = {str(record.get("extraction_arm")) for record in records}
            if arms != set(CT_EXTRACTION_ARMS):
                raise ValueError("isolated CT extraction returned invalid extraction arms")
            results.extend(records)
            if checkpoint:
                for record in records:
                    checkpoint.add_result(record)
            return
        features_clean = normalize_radiomics_result(
            {
                key: value
                for key, value in features.items()
                if not key.startswith("__")
            }
        )
        roi_name = task.get("roi_name", "ROI")
        metadata = dict(task.get("metadata") or task.get("extra_metadata") or {})
        effective_hash = features.get("__effective_parameter_hash__")
        if task.get("parameter_provenance_arm") and not effective_hash:
            raise ValueError("isolated extraction omitted effective parameter provenance")
        if effective_hash:
            metadata["effective_parameter_hash"] = str(effective_hash)
        metadata.setdefault("roi_name", roi_name)
        metadata.setdefault("roi_original_name", roi_name)
        metadata.setdefault("modality", "CT")
        record = _combine_feature_record(features_clean, metadata)
        results.append(record)
        if checkpoint:
            checkpoint.add_result(record)

    def _run_sequential(seq: List[Dict[str, Any]]) -> None:
        for idx, task in enumerate(seq, 1):
            roi_name = task.get('roi_name', 'ROI')
            logger.info("Processing %d/%d: %s", idx, len(seq), roi_name)
            rec = _execute(task)
            if rec and rec.get('__status__') == 'skipped':
                _record_failure(
                    task,
                    str(rec.get('__reason__', 'mask was too small for radiomics')),
                    status="below_minimum_voxels",
                    failure_kind="degenerate_mask",
                )
                continue
            if rec and rec.get('__status__') == 'error':
                _record_failure(
                    task,
                    str(rec.get('__error__', 'unknown error')),
                    status=str(rec.get('__extraction_status__', 'failed')),
                    failure_kind=str(rec.get('__failure_kind__', 'extraction_error')),
                )
            elif rec:
                _record_success(task, rec)
            else:
                _record_failure(task, "returned no outcome record")

    tasks_list = list(tasks)
    if max_workers and max_workers > 0:
        worker_limit = max_workers
    else:
        cpu_total = os.cpu_count() or 2
        worker_limit = max(1, cpu_total - 1)
    worker_limit = max(1, min(worker_limit, len(tasks_list)))

    # Use batch processing to reduce subprocess overhead
    # Instead of N subprocesses (each loading radiomics library),
    # we use N/batch_size subprocesses, dramatically reducing startup overhead
    use_batch_processing = os.environ.get('RTPIPELINE_RADIOMICS_BATCH', '1').lower() in ('1', 'true', 'yes')
    if any(task.get("precomputed_failure") for task in tasks_list):
        # Synthetic failure records have no mask file and must be handled by the
        # metadata-aware executor rather than passed to the external batch CLI.
        use_batch_processing = False

    # Initialize heartbeat logger if enabled
    heartbeat: Optional[HeartbeatLogger] = None
    if enable_heartbeat and tasks_list:
        heartbeat = HeartbeatLogger(
            total_tasks=original_task_count,
            description=f"Radiomics ({Path(output_path).stem})"
        )
        heartbeat.start()
        # Account for already-skipped tasks from checkpoint
        if checkpoint:
            heartbeat.update(completed=checkpoint.get_completed_count())

    try:
        if sequential or len(tasks_list) == 1 or worker_limit == 1:
            if use_batch_processing:
                # Even sequential mode benefits from batch processing. Match results
                # back to tasks via _process_batch's __task_index__-based pairing (not a
                # positional zip(tasks_list, batch_results)): the same dropped/
                # unparseable-subprocess-line hazard that motivated the fix in
                # _process_batch applies here too, since this also runs the whole
                # task list through a single extract_radiomics_batch_with_conda() call.
                logger.info("Processing %d radiomics tasks in batch mode (sequential)", len(tasks_list))
                batch_pairs = _process_batch(tasks_list)

                for task, features in batch_pairs:
                    roi_name = task.get('roi_name', 'ROI')
                    if features is None or features.get('__status__') != 'success':
                        status = features.get('__status__', 'failed') if features else 'failed'
                        if status == 'skipped':
                            _record_failure(
                                task,
                                str((features or {}).get('__reason__', 'mask was too small for radiomics')),
                                status="below_minimum_voxels",
                                failure_kind="degenerate_mask",
                            )
                            logger.info("Recorded best-effort/required degenerate ROI %s", roi_name)
                            if heartbeat:
                                heartbeat.update(failed=1)
                        else:
                            detail = (
                                features.get('__error__', 'unknown error')
                                if features else 'returned no outcome record'
                            )
                            logger.error("Radiomics failed for %s: %s", roi_name, detail)
                            _record_failure(
                                task,
                                str(detail),
                                status=(
                                    str(features.get('__extraction_status__', 'failed'))
                                    if features else 'failed'
                                ),
                                failure_kind=(
                                    str(features.get('__failure_kind__', 'extraction_error'))
                                    if features else 'extraction_error'
                                ),
                            )
                            if heartbeat:
                                heartbeat.update(failed=1)
                        continue

                    _record_success(task, features)
                    if heartbeat:
                        heartbeat.update(completed=1)
            else:
                _run_sequential(tasks_list)
        elif use_batch_processing:
            # OPTIMIZED: Split tasks into batches and process each batch in a single subprocess
            # This amortizes the ~2-5 second subprocess startup across multiple ROIs
            batch_size = max(1, len(tasks_list) // worker_limit)  # Allow fine-grained parallelism
            batches = [tasks_list[i:i + batch_size] for i in range(0, len(tasks_list), batch_size)]

            logger.info(
                "Processing %d radiomics tasks in %d batches (%d ROIs/batch) with %d workers",
                len(tasks_list),
                len(batches),
                batch_size,
                min(worker_limit, len(batches)),
            )

            from concurrent.futures import ProcessPoolExecutor, as_completed

            # Use ProcessPoolExecutor for better parallelism (avoids GIL).
            # Each worker processes one batch (multiple ROIs) per subprocess.
            # NOTE: this pool is created from WITHIN a course-level ThreadPoolExecutor
            # worker thread. The default 'fork' start method copies locked mutexes from the
            # multi-threaded parent into workers and deadlocks (workers block forever on an
            # inherited-locked SemLock); use a forkserver/spawn context instead.
            with ProcessPoolExecutor(
                max_workers=min(worker_limit, len(batches)),
                mp_context=radiomics_mp_context(),
            ) as executor:
                future_to_batch = {executor.submit(_process_batch, batch): batch for batch in batches}

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        for task, features in batch_results:
                            roi_name = task.get('roi_name', 'ROI')
                            if features is None or features.get('__status__') != 'success':
                                status = features.get('__status__', 'failed') if features else 'failed'
                                if status == 'skipped':
                                    _record_failure(
                                        task,
                                        str((features or {}).get('__reason__', 'mask was too small for radiomics')),
                                        status="below_minimum_voxels",
                                        failure_kind="degenerate_mask",
                                    )
                                    logger.debug("Recorded best-effort/required degenerate ROI %s", roi_name)
                                    if heartbeat:
                                        heartbeat.update(failed=1)
                                else:
                                    detail = (
                                        features.get('__error__', 'unknown error')
                                        if features else 'returned no outcome record'
                                    )
                                    logger.warning("Radiomics failed for %s: %s", roi_name, detail)
                                    _record_failure(
                                        task,
                                        str(detail),
                                        status=(
                                            str(features.get('__extraction_status__', 'failed'))
                                            if features else 'failed'
                                        ),
                                        failure_kind=(
                                            str(features.get('__failure_kind__', 'extraction_error'))
                                            if features else 'extraction_error'
                                        ),
                                    )
                                    if heartbeat:
                                        heartbeat.update(failed=1)
                                continue

                            _record_success(task, features)
                            if heartbeat:
                                heartbeat.update(completed=1)

                            logger.debug("Completed radiomics for %s", roi_name)
                    except Exception as exc:
                        logger.error("Batch processing failed: %s", exc)
                        for task in batch:
                            _record_failure(task, f"batch crashed: {exc}")
                        if heartbeat:
                            heartbeat.update(failed=len(batch))
        else:
            # Legacy per-ROI processing (can be enabled via RTPIPELINE_RADIOMICS_BATCH=0)
            logger.info(
                "Processing %d radiomics tasks with up to %d worker threads (legacy mode)",
                len(tasks_list),
                worker_limit,
            )
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=worker_limit) as executor:
                future_to_task = {executor.submit(_execute, task): task for task in tasks_list}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    roi_name = task.get('roi_name', 'ROI')
                    try:
                        rec = future.result()
                        if rec and rec.get('__status__') == 'skipped':
                            _record_failure(
                                task,
                                str(rec.get('__reason__', 'mask was too small for radiomics')),
                                status="below_minimum_voxels",
                                failure_kind="degenerate_mask",
                            )
                            if heartbeat:
                                heartbeat.update(failed=1)
                        elif rec and rec.get('__status__') == 'error':
                            _record_failure(
                                task,
                                str(rec.get('__error__', 'unknown error')),
                                status=str(rec.get('__extraction_status__', 'failed')),
                                failure_kind=str(rec.get('__failure_kind__', 'extraction_error')),
                            )
                            if heartbeat:
                                heartbeat.update(failed=1)
                        elif rec:
                            _record_success(task, rec)
                            if heartbeat:
                                heartbeat.update(completed=1)
                            logger.debug("Completed radiomics for %s", roi_name)
                        else:
                            _record_failure(task, "returned no successful feature record")
                            if heartbeat:
                                heartbeat.update(failed=1)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error("Radiomics task crashed for %s: %s", roi_name, exc)
                        _record_failure(task, f"task crashed: {exc}")
                        if heartbeat:
                            heartbeat.update(failed=1)

    except RadiomicsFeatureTypeError:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise
    finally:
        # Always stop heartbeat and flush checkpoint
        if heartbeat:
            heartbeat.stop()
        if checkpoint:
            checkpoint.flush()

    if failures:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(
            "Radiomics course extraction is incomplete: " + "; ".join(failures)
        )

    # Publish only an exact identity set. A valid checkpoint is all-or-nothing;
    # partial or stale rows are never unioned into current results.
    rows: List[Dict[str, Any]] = list(results)
    if checkpoint is not None:
        try:
            done_now = {_roi_instance_key(record) for record in results}
            prior = [
                record for record in checkpoint.load_records()
                if _roi_instance_key(record) not in done_now
            ]
        except Exception:
            _invalidate_radiomics_outputs(Path(output_path))
            checkpoint.discard()
            raise
        if prior:
            logger.info(
                "Resume: reusing %d exact-current checkpoint ROI(s) for %s",
                len(prior), output_path,
            )
            rows.extend(prior)
    rows.extend(failure_rows)

    required_publication_keys = expected_keys
    try:
        publication_keys = _validated_identity_keys(
            rows,
            context="radiomics publication",
        )
    except ValueError as exc:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(str(exc)) from exc
    if publication_keys != required_publication_keys:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(
            "Radiomics publication identity set is incomplete or stale "
            f"(expected {len(required_publication_keys)}, found {len(publication_keys)})"
        )

    if not rows:
        logger.warning("No radiomics features extracted")
        for mask_path in cleanup_paths:
            try:
                if mask_path:
                    os.unlink(mask_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.debug("Cleanup failed for %s: %s", mask_path, exc)
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        return None

    try:
        attach_acquisition_descriptor(rows, acquisition_descriptor)
        source_counts: Dict[str, Dict[str, int]] = {}
        roi_failures: List[Dict[str, str]] = []
        count_rows = (
            [row for row in rows if row.get("extraction_arm") == PRIMARY_ARM]
            if expected_configured_hashes
            else rows
        )
        for row in count_rows:
            source = str(row.get("segmentation_source", "unknown"))
            status = row.get("extraction_status")
            try:
                if status != status:
                    status = None
            except (TypeError, ValueError):
                pass
            counts = source_counts.setdefault(source, {"attempted": 0, "extracted": 0, "failed": 0})
            counts["attempted"] += 1
            if status in (None, "success"):
                counts["extracted"] += 1
            else:
                counts["failed"] += 1
                roi_failures.append(
                    {
                        "roi_name": str(row.get("roi_original_name", row.get("roi_name", "unknown"))),
                        "source": source,
                        "status": str(status),
                        "failure_kind": str(row.get("extraction_failure_kind", "extraction_error")),
                        "reason": str(row.get("extraction_status_detail", "unknown error")),
                    }
                )
        outcome = RadiomicsCourseOutcome.extracted(
            Path(output_path),
            roi_counts=source_counts,
            roi_failures=roi_failures,
            detail=(
                f"extracted {sum(values['extracted'] for values in source_counts.values())} "
                f"of {sum(values['attempted'] for values in source_counts.values())} "
                "attempted ROIs"
            ),
        )
        diagnostics = course_diagnostic_columns(outcome)
        for row in rows:
            row.update(diagnostics)
        df = pd.DataFrame(rows)
        if expected_configured_hashes:
            tuple_expected = {publication_key(row) for row in rows}
            write_ct_publication_atomic(df, Path(output_path), expected_keys=tuple_expected)
        else:
            write_radiomics_feature_table_atomic(df, Path(output_path))
        logger.info("Saved %d radiomics rows to %s", len(df), output_path)
    except Exception as exc:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(
            f"Failed to save radiomics workbook {output_path}: {exc}"
        ) from exc
    finally:
        for mask_path in cleanup_paths:
            try:
                if mask_path:
                    os.unlink(mask_path)
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.debug("Cleanup failed for %s: %s", mask_path, exc)

    return output_path


def radiomics_for_course_ct_nifti_fallback(
    course_dir: Path,
    config: Any,
    *,
    allow_all_series_temp: bool = False,
) -> Optional[Path]:
    """Extract CT radiomics from TotalSegmentator NIfTI masks when no RS is usable."""
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
    course_dirs = build_course_dirs(course_dir)
    seg_dir = course_dirs.segmentation_totalseg
    output_path = course_dir / "radiomics_ct.xlsx"

    if not seg_dir.exists():
        logger.debug("No TotalSegmentator NIfTI mask directory for CT fallback in %s", course_dir)
        _invalidate_radiomics_outputs(output_path)
        return None

    params_file = str(config.radiomics_params_file) if getattr(config, "radiomics_params_file", None) else None
    parameter_path = Path(params_file) if params_file else None
    run_identifier = new_run_identifier()
    code_revision = current_code_revision()

    def _nifti_governed_fields(
        roi_name: str,
        mask_source: Path,
        series_uid: str,
        display_roi: str,
        cropped_flag: bool,
        native_voxel_count: Optional[int],
        large_roi: bool,
        nifti_path_str: str,
    ) -> Dict[str, Any]:
        decision = classify_ct_roi("AutoTS_total_nifti_fallback", roi_name)
        hashes = {
            arm: configured_parameter_hash(
                parameter_path,
                arm=arm,
                window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
                large_roi=large_roi,
            )
            for arm in CT_EXTRACTION_ARMS
        }
        mask_digest = file_sha256(mask_source)
        return {
            "dual_arm_ct": True,
            "roi_class_decision": {
                "roi_class": decision.roi_class,
                "primary_resegment_range_hu": decision.primary_resegment_range_hu,
                "primary_intensity_texture_disposition": decision.primary_intensity_texture_disposition,
                "feature_publication_policy": decision.feature_publication_policy,
                "map_version": decision.map_version,
                "map_hash": decision.map_hash,
                "map_entry_source": decision.map_entry_source,
                "adjudication_status": decision.adjudication_status,
            },
            "run_identifier": run_identifier,
            "code_revision": code_revision,
            "native_voxel_count": native_voxel_count,
            "required": False,
            "configured_parameter_hashes": hashes,
            "metadata": {
                "modality": "CT",
                "series_uid": str(series_uid),
                "segmentation_source": "AutoTS_total_nifti_fallback",
                "mask_identity": mask_digest,
                "stable_roi_identifier": roi_name,
                "course_dir": str(course_dir),
                "patient_id": course_dir.parent.name,
                "course_id": course_dir.name,
                "structure_cropped": bool(cropped_flag),
                "roi_original_name": roi_name,
                "roi_name": display_roi,
                "nifti_path": nifti_path_str,
                "mask_path_source": str(mask_source),
            },
        }
    skip_rois = _ct_skip_rois(config)
    min_voxels_limit, max_voxels_limit = _ct_voxel_limits(config)
    planning_nifti = contract.planning_ct_nifti
    nifti_by_stem = (
        {_strip_nii_suffix(planning_nifti): planning_nifti}
        if planning_nifti is not None
        else {}
    )

    series_dirs = [p for p in sorted(seg_dir.iterdir()) if p.is_dir()]
    if any(seg_dir.glob("*.nii*")):
        series_dirs.insert(0, seg_dir)

    tasks: List[Dict[str, Any]] = []
    preparation_failures: List[Dict[str, Any]] = []
    temp_files: List[Path] = []
    image_cache: Dict[Path, Tuple[str, Tuple[float, float, float]]] = {}
    dicom_image_path: Optional[str] = None
    dicom_spacing: Optional[Tuple[float, float, float]] = None

    def _cleanup_temp_files() -> None:
        for temp_file in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Failed to remove temporary CT fallback file %s: %s", temp_file, exc)

    def _fail_fallback(detail: str, exc: Optional[BaseException] = None) -> None:
        _cleanup_temp_files()
        _invalidate_radiomics_outputs(output_path)
        error = RadiomicsCourseExtractionError(detail)
        if exc is not None:
            raise error from exc
        raise error

    def _record_preparation_failure(
        roi_name: str,
        detail: str,
        *,
        mask_source: Path,
        series_uid: str,
        nifti_path_str: str,
        status: str = "failed",
        failure_kind: str = "extraction_error",
        reason_code: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        preparation_failures.append(
            {
                "roi_name": str(roi_name),
                "mask_source": str(mask_source),
                "series_uid": str(series_uid),
                "nifti_path": str(nifti_path_str),
                "reason": detail,
                "status": status,
                "failure_kind": failure_kind,
                "reason_code": reason_code,
                "metadata": dict(metadata or {}),
            }
        )
        logger.warning(
            "Best-effort radiomics ROI AutoTS_total_nifti_fallback/%s was not prepared: %s",
            roi_name,
            detail,
        )

    def _dicom_image() -> Optional[Tuple[str, Tuple[float, float, float]]]:
        nonlocal dicom_image_path, dicom_spacing
        if dicom_image_path is not None and dicom_spacing is not None:
            return dicom_image_path, dicom_spacing
        ct_dir = contract.planning_ct_dir
        if ct_dir is None:
            return None
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))
            if not dicom_files:
                return None
            reader.SetFileNames(dicom_files)
            img = reader.Execute()
            tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False, prefix="ct_dicom_fallback_")
            tmp.close()
            sitk.WriteImage(img, tmp.name, useCompression=True)
            temp_files.append(Path(tmp.name))
            dicom_image_path = tmp.name
            dicom_spacing = tuple(float(x) for x in img.GetSpacing())
            return dicom_image_path, dicom_spacing
        except Exception as exc:
            _fail_fallback(
                f"Failed to load DICOM CT for NIfTI-mask fallback in {course_dir}: {exc}",
                exc,
            )

    def _image_for_series(series_name: str) -> Optional[Tuple[str, str, Tuple[float, float, float], str]]:
        nifti_path = nifti_by_stem.get(series_name)
        if nifti_path is None and len(nifti_by_stem) == 1:
            nifti_path = next(iter(nifti_by_stem.values()))

        if nifti_path is not None:
            cached = image_cache.get(nifti_path)
            if cached is None:
                try:
                    img = sitk.ReadImage(str(nifti_path))
                    tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False, prefix="ct_nifti_fallback_")
                    tmp.close()
                    sitk.WriteImage(img, tmp.name, useCompression=True)
                    cached = (tmp.name, tuple(float(x) for x in img.GetSpacing()))
                    image_cache[nifti_path] = cached
                    temp_files.append(Path(tmp.name))
                except Exception as exc:
                    _fail_fallback(
                        f"Failed to convert CT NIfTI {nifti_path} for radiomics fallback: {exc}",
                        exc,
                    )
            image_path, spacing = cached
            return image_path, str(nifti_path), spacing, _series_uid_from_nifti(nifti_path, series_name)

        dicom = _dicom_image()
        if dicom is None:
            return None
        image_path, spacing = dicom
        return image_path, "", spacing, series_name

    for series_root in series_dirs:
        series_name = series_root.name if series_root != seg_dir else "CT"
        image_info = _image_for_series(series_name)
        if image_info is None:
            _fail_fallback(
                f"No CT image matched TotalSegmentator mask series {series_name} in {course_dir}"
            )
        image_path, nifti_path_str, image_spacing, series_uid = image_info

        for mask_path in sorted(series_root.glob("*.nii*")):
            roi_name = _totalseg_roi_name(mask_path)
            if not roi_name:
                continue
            norm_key = _norm_roi_key(roi_name)
            if norm_key in skip_rois:
                _record_preparation_failure(
                    roi_name,
                    "ROI is listed in radiomics_skip_rois",
                    mask_source=mask_path,
                    series_uid=series_uid,
                    nifti_path_str=nifti_path_str,
                    status="declared_skip",
                    failure_kind="declared_ineligible",
                )
                continue

            try:
                mask_img = sitk.ReadImage(str(mask_path))
                mask_arr = sitk.GetArrayFromImage(mask_img)
            except Exception as exc:
                _record_preparation_failure(
                    roi_name,
                    f"Failed to read CT fallback mask {mask_path}: {exc}",
                    mask_source=mask_path,
                    series_uid=series_uid,
                    nifti_path_str=nifti_path_str,
                )
                continue

            mask_bool = mask_arr > 0
            if not mask_bool.any():
                _record_preparation_failure(
                    roi_name,
                    f"CT fallback ROI {roi_name} has an empty mask: {mask_path}",
                    mask_source=mask_path,
                    series_uid=series_uid,
                    nifti_path_str=nifti_path_str,
                    failure_kind="degenerate_mask",
                )
                continue

            voxel_count = int(mask_bool.sum())
            if voxel_count < min_voxels_limit:
                _record_preparation_failure(
                    roi_name,
                    f"ROI contains {voxel_count} voxels; configured minimum is "
                    f"{min_voxels_limit}",
                    mask_source=mask_path,
                    series_uid=series_uid,
                    nifti_path_str=nifti_path_str,
                    status="below_minimum_voxels",
                    failure_kind="degenerate_mask",
                )
                continue

            mask_spacing = tuple(float(value) for value in mask_img.GetSpacing())
            resampled_spacing, pad_distance = configured_grid_settings(
                parameter_path,
                native_spacing_xyz=mask_spacing,
            )
            work_estimate = estimate_resampled_bounding_box(
                mask_bool,
                native_spacing_xyz=mask_spacing,
                resampled_spacing_xyz=resampled_spacing,
                array_axis_to_xyz=(2, 1, 0),
                pad_distance=pad_distance,
            )
            max_bbox_voxels = resolve_max_resampled_bbox_voxels(config)
            if (work_estimate.estimated_resampled_bbox_voxels > max_bbox_voxels
                    and not permits_second_stage(
                        work_estimate, mask_bool, native_spacing_xyz=mask_spacing,
                        array_axis_to_xyz=(2, 1, 0), params_file=parameter_path,
                        limit=max_bbox_voxels)):
                detail = (
                    f"ROI {roi_name} requires an estimated padded resampled bounding "
                    f"box of {work_estimate.estimated_resampled_bbox_voxels} voxels "
                    f"({work_estimate.estimated_resampled_bbox_shape}); configured "
                    f"maximum is {max_bbox_voxels}. Full configured radiomics was "
                    "not started."
                )
                _record_preparation_failure(
                    roi_name,
                    detail,
                    mask_source=mask_path,
                    series_uid=series_uid,
                    nifti_path_str=nifti_path_str,
                    failure_kind="resource_limit",
                    reason_code=FAILED_RADIOMICS_RESOURCE_LIMIT,
                    metadata={
                        "reason_code": FAILED_RADIOMICS_RESOURCE_LIMIT,
                        "roi_structural_code": RESAMPLED_BBOX_LIMIT_CODE,
                        **work_estimate.metadata(limit=max_bbox_voxels),
                    },
                )
                continue

            try:
                native_voxel_mm3 = float(image_spacing[0]) * float(image_spacing[1]) * float(image_spacing[2])
            except Exception:
                native_voxel_mm3 = 1.0
            physical_volume_mm3 = float(voxel_count) * max(1e-9, native_voxel_mm3)
            estimated_voxels = physical_volume_mm3
            large_roi = norm_key.startswith("body") or (estimated_voxels > float(max_voxels_limit))
            if large_roi:
                logger.info(
                    "CT fallback ROI %s is large (native=%d voxels, est@1mm=%.0f voxels, cap=%d); preserving full configured radiomics settings",
                    roi_name,
                    voxel_count,
                    estimated_voxels,
                    int(max_voxels_limit),
                )

            try:
                mask_nrrd = tempfile.NamedTemporaryFile(
                    suffix=".nrrd",
                    delete=False,
                    prefix=f"ct_ts_mask_{roi_name}_",
                )
                mask_nrrd.close()
                sitk.WriteImage(mask_img, mask_nrrd.name, useCompression=True)
                temp_files.append(Path(mask_nrrd.name))
            except Exception as exc:
                _record_preparation_failure(
                    roi_name,
                    f"Failed to convert CT fallback mask {mask_path}: {exc}",
                    mask_source=mask_path,
                    series_uid=series_uid,
                    nifti_path_str=nifti_path_str,
                )
                continue

            cropped_flag = mask_is_cropped(mask_bool)
            display_roi = roi_name if (not cropped_flag or roi_name.endswith("__partial")) else f"{roi_name}__partial"

            tasks.append({
                "image_path": image_path,
                "mask_path": mask_nrrd.name,
                "roi_name": display_roi,
                "params_file": params_file,
                "label": None,
                "large_roi": bool(large_roi),
                "cleanup": False,
                **_nifti_governed_fields(
                    roi_name,
                    mask_path,
                    series_uid,
                    display_roi,
                    cropped_flag,
                    voxel_count,
                    bool(large_roi),
                    nifti_path_str,
                ),
            })

            tasks[-1]["resource_guard_legacy"] = legacy_rejection(
                work_estimate, max_bbox_voxels, roi_name)

    for failure in preparation_failures:
        tasks.append(
            {
                "image_path": dicom_image_path or "best-effort-preparation-failure",
                "mask_path": None,
                "roi_name": failure["roi_name"],
                "params_file": params_file,
                "label": None,
                "large_roi": False,
                "cleanup": False,
                **_nifti_governed_fields(
                    failure["roi_name"],
                    Path(failure["mask_source"]),
                    failure["series_uid"],
                    failure["roi_name"],
                    False,
                    None,
                    False,
                    failure["nifti_path"],
                ),
                "precomputed_failure": failure,
            }
        )

    if not tasks:
        logger.warning("No valid TotalSegmentator NIfTI ROIs found for CT fallback in %s", course_dir)
        _cleanup_temp_files()
        _invalidate_radiomics_outputs(output_path)
        return None

    logger.info("Processing %d CT radiomics fallback tasks for %s", len(tasks), course_dir.name)

    max_workers = None
    env_workers = int(os.environ.get("RTPIPELINE_MAX_WORKERS", "0") or 0)
    if env_workers > 0:
        max_workers = env_workers
    elif hasattr(config, "effective_workers") and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
        except Exception:
            pass
    if max_workers is None:
        max_workers = min(4, len(tasks))

    sequential = os.environ.get("RTPIPELINE_RADIOMICS_SEQUENTIAL", "").lower() in ("1", "true", "yes")
    checkpoint_path = course_dir / "metadata" / "radiomics_ct_checkpoint.parquet"

    try:
        result = process_radiomics_batch(
            tasks,
            output_path,
            sequential=sequential,
            max_workers=max_workers,
            checkpoint_path=checkpoint_path,
            enable_heartbeat=True,
            env_probe_timeout=getattr(config, "radiomics_env_probe_timeout", None),
            acquisition_descriptor=describe_contract_planning_ct(contract),
        )
    finally:
        _cleanup_temp_files()

    result_rows: List[Dict[str, Any]] = []
    if result is not None:
        try:
            result_rows = pd.read_parquet(
                Path(result).with_suffix(".parquet")
            ).to_dict("records")
        except Exception:
            result_rows = []
    _write_conda_roi_ledger(
        course_dir,
        tasks,
        result_rows,
        extracted=result is not None,
    )
    if result is None:
        _invalidate_radiomics_outputs(output_path)
    return result


def radiomics_for_course(
    course_dir: Path,
    config: Any,
    custom_structures_config: Optional[str] = None,
    *,
    allow_all_series_temp: bool = False,
) -> Optional[Path]:
    """
    Extract radiomics features for a course using conda environment.

    Args:
        course_dir: Path to the course directory
        config: Pipeline configuration
        custom_structures_config: Optional path to custom structures config

    Returns:
        Path to the radiomics Excel file if successful, None otherwise
    """
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
    course_dirs = build_course_dirs(course_dir)
    output_path = course_dir / "radiomics_ct.xlsx"

    # Check for CT DICOM files
    ct_dir = contract.planning_ct_dir
    has_ct_dicom = ct_dir is not None
    has_ct_nifti = contract.planning_ct_nifti is not None
    if not has_ct_dicom and not has_ct_nifti:
        logger.warning(f"No CT image found in {course_dir}")
        _invalidate_radiomics_outputs(output_path)
        _write_conda_roi_ledger(course_dir, (), (), extracted=False, expected_names=(), missing_reason="not_applicable_scope", in_scope=False)
        return None
    if ct_dir is None:
        raise RadiomicsCourseExtractionError(
            f"Course contract has a CT NIfTI but no planning CT DICOM directory: {course_dir}"
        )

    rs_manual = (
        contract.authoritative_rtstruct_path
        or course_dir / "metadata" / ".contract-rtstruct-absent"
    )
    rs_auto = course_dir / "RS_auto.dcm"
    rs_custom = course_dir / "RS_custom.dcm"

    def _norm(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    skip_rois = _ct_skip_rois(config)
    custom_cfg_value = (
        custom_structures_config
        or getattr(config, "custom_structures_config", None)
    )
    custom_cfg: Optional[Path] = None
    desired_custom: Set[str] = set()
    dependency_states: dict[str, Any] = {}
    pending_custom_assessments: set[str] = set()
    custom_provenance: dict[str, Any] = {}
    planning_ct_fov: Any = {}
    if custom_cfg_value:
        custom_cfg = Path(custom_cfg_value)
        if not custom_cfg.is_file():
            _invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                f"Configured required custom structure file is missing: {custom_cfg}"
            )
        try:
            from .radiomics import _custom_roi_names_from_config

            desired_custom = {
                name for name in _custom_roi_names_from_config(custom_cfg)
            }
        except Exception as exc:
            _invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to read configured custom ROI identities for {course_dir}: {exc}"
            ) from exc
        try:
            from .roi_requiredness import (
                dependency_state_from_observation,
                inspect_rtstruct,
            )
            from .radiomics import _planning_ct_fov
            custom_provenance = load_custom_structure_provenance(custom_cfg)
            planning_ct_fov = _planning_ct_fov(course_dir)
            for source_path in (rs_manual, rs_auto):
                if not Path(source_path).is_file():
                    continue
                try:
                    source_inventory = inspect_rtstruct(source_path)
                except Exception:
                    continue
                for observation in source_inventory.named_rois:
                    dependency_states[observation.name] = (
                        dependency_state_from_observation(observation)
                    )
            custom_inventory = inspect_rtstruct(rs_custom) if rs_custom.is_file() else None
            available_custom = {
                item.name: item for item in getattr(custom_inventory, "named_rois", ())
            }
            applicable_custom: set[str] = set()
            for base in sorted(desired_custom):
                candidate = next(
                    (available_custom[name] for name in (base, f"{base}__partial") if name in available_custom),
                    None,
                )
                generated_state = (
                    "readable_nonempty" if candidate is not None and candidate.has_readable_contour
                    else "unreadable" if candidate is not None else "absent"
                )
                assessment = assess_custom_applicability(
                    base,
                    dependency_states,
                    planning_ct_fov,
                    generated_state=generated_state,
                    custom_provenance=custom_provenance,
                )
                if assessment.reason_code == "extracted":
                    applicable_custom.add(base)
                elif assessment.reason_code == "failed_custom_generation":
                    applicable_custom.add(base)
                    pending_custom_assessments.add(base)
                elif assessment.reason_code == "indeterminate_applicability":
                    raise RadiomicsCourseExtractionError(
                        f"Configured custom ROI {base!r} has {assessment.reason_code}: {assessment.detail}"
                    )
            desired_custom = applicable_custom
        except RadiomicsCourseExtractionError:
            _invalidate_radiomics_outputs(output_path)
            raise
        except Exception as exc:
            logger.warning("Could not complete custom ROI applicability inspection for %s: %s", course_dir, exc)
        custom_rebuild_attempted = False
        custom_rebuild_published = False
        try:
            from .custom_structures_rtstruct import (
                _create_custom_structures_rtstruct,
                _is_rs_custom_stale,
                record_rs_custom_resume_decision,
            )

            custom_is_stale = bool(
                desired_custom
                and _is_rs_custom_stale(rs_custom, custom_cfg, rs_manual, rs_auto)
            )
            if custom_is_stale:
                custom_rebuild_attempted = True
                from .custom_structures_rtstruct import _quarantine_rejected_rtstruct

                _quarantine_rejected_rtstruct(
                    rs_custom,
                    "RS_custom failed the authoritative currentness check",
                )
                rebuilt = _create_custom_structures_rtstruct(
                    course_dir, custom_cfg, rs_manual, rs_auto
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
            elif desired_custom:
                record_rs_custom_resume_decision(
                    course_dir,
                    "reused",
                    "existing RS_custom passed the authoritative currentness check",
                )
        except Exception as exc:
            if custom_rebuild_attempted and not custom_rebuild_published:
                from .custom_structures_rtstruct import record_rs_custom_resume_decision

                record_rs_custom_resume_decision(
                    course_dir,
                    "failed",
                    f"RS_custom rebuild raised {type(exc).__name__}: {exc}",
                )
            _invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to prepare configured custom RTSTRUCT for {course_dir}: {exc}"
            ) from exc

    try:
        from .radiomics import _list_roi_names_dicom

        if custom_cfg is None and not desired_custom and rs_custom.exists():
            manual_names = set(_list_roi_names_dicom(rs_manual))
            auto_names = set(_list_roi_names_dicom(rs_auto))
            custom_names = set(_list_roi_names_dicom(rs_custom))
            inferred = custom_names - (manual_names | auto_names)
            desired_custom = {
                name[:-9] if name.endswith("__partial") else name
                for name in inferred
            }

        custom_wanted: List[str] = []
        if desired_custom:
            if not rs_custom.is_file():
                raise RadiomicsCourseExtractionError(
                    f"Required custom RTSTRUCT is missing for configured ROIs in {course_dir}"
                )
            available_custom = set(_list_roi_names_dicom(rs_custom))
            for base in sorted(pending_custom_assessments):
                final_assessment = assess_custom_applicability(
                    base,
                    dependency_states,
                    planning_ct_fov,
                    generated_state=(
                        "readable_nonempty"
                        if base in available_custom
                        or f"{base}__partial" in available_custom
                        else "absent"
                    ),
                    custom_provenance=custom_provenance,
                )
                if final_assessment.reason_code != "extracted":
                    raise RadiomicsCourseExtractionError(
                        f"Configured custom ROI {base!r} remains unavailable after rebuild: "
                        f"{final_assessment.reason_code}: {final_assessment.detail}"
                    )
            missing_custom: List[str] = []
            for base in sorted(desired_custom):
                if base in available_custom:
                    custom_wanted.append(base)
                elif f"{base}__partial" in available_custom:
                    custom_wanted.append(f"{base}__partial")
                else:
                    missing_custom.append(base)
            if missing_custom:
                raise RadiomicsCourseExtractionError(
                    f"Required configured custom ROI(s) missing from {rs_custom}: "
                    + ", ".join(missing_custom)
                )

        custom_model_expected_rois = validate_custom_model_output_inventory(
            course_dir,
            getattr(config, "custom_model_names", None),
            getattr(config, "custom_models_root", None),
        )
        custom_model_outputs = list_custom_model_outputs(course_dir)
    except Exception as exc:
        _invalidate_radiomics_outputs(output_path)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"Failed to enumerate required segmentation inventory for {course_dir}: {exc}"
        ) from exc

    # Each source is extracted independently. Never collapse colliding ROI names
    # from Manual, AutoRTS, Custom, or a custom model into one merged identity.
    from .radiomics import _deduplicate_rtstruct_sources, _standard_rtstruct_sources

    source_specs = _standard_rtstruct_sources(contract, course_dir)
    if custom_wanted:
        source_specs.append(("Custom", rs_custom, custom_wanted))
    for model_name, model_course_dir in custom_model_outputs:
        source_specs.append(
            (
                f"CustomModel:{model_name}",
                Path(model_course_dir) / "rtstruct.dcm",
                custom_model_expected_rois[model_name],
            )
        )
    source_specs = _deduplicate_rtstruct_sources(source_specs)

    if not source_specs:
        logger.info(
            "No current RTSTRUCT source for %s; trying CT TotalSegmentator NIfTI fallback",
            course_dir,
        )
        return radiomics_for_course_ct_nifti_fallback(course_dir, config)

    if not has_ct_dicom:
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"RTSTRUCT radiomics requires a CT DICOM series in {course_dir}"
        )

    # Load CT image
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))
        reader.SetFileNames(dicom_files)
        ct_image = reader.Execute()
    except Exception as e:
        logger.error(f"Failed to load CT image: {e}")
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"CT series is present but unreadable for radiomics in {course_dir}: {e}"
        ) from e

    ct_info = {
        'spacing': tuple(ct_image.GetSpacing()),
        'origin': tuple(ct_image.GetOrigin()),
        'direction': tuple(ct_image.GetDirection()),
    }

    params_file = (
        str(config.radiomics_params_file)
        if getattr(config, "radiomics_params_file", None)
        else None
    )
    parameter_path = Path(params_file) if params_file else None
    series_uid = str(contract.planning_ct.get("series_instance_uid") or "").strip()
    if not series_uid:
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"Course contract has no planning CT SeriesInstanceUID: {course_dir}"
        )
    run_identifier = new_run_identifier()
    code_revision = current_code_revision()
    custom_provenance = load_custom_structure_provenance(custom_cfg)
    identity_cache: Dict[Path, Dict[str, Tuple[str, str]]] = {}

    def _governed_task_fields(
        source: str,
        rs_file: Path,
        roi_original_name: str,
        display_roi: str,
        cropped_flag: bool,
        native_voxel_count: Optional[int],
        large_roi: bool,
    ) -> Dict[str, Any]:
        rs_file = Path(rs_file).resolve()
        identity = stable_rtstruct_roi_identity(rs_file, roi_original_name)
        decision = classify_ct_roi(
            source,
            roi_original_name,
            custom_provenance=custom_provenance,
        )
        hashes = {
            arm: configured_parameter_hash(
                parameter_path,
                arm=arm,
                window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
                large_roi=large_roi,
            )
            for arm in CT_EXTRACTION_ARMS
        }
        return {
            "dual_arm_ct": True,
            "roi_class_decision": {
                "roi_class": decision.roi_class,
                "primary_resegment_range_hu": decision.primary_resegment_range_hu,
                "primary_intensity_texture_disposition": decision.primary_intensity_texture_disposition,
                "feature_publication_policy": decision.feature_publication_policy,
                "map_version": decision.map_version,
                "map_hash": decision.map_hash,
                "map_entry_source": decision.map_entry_source,
                "adjudication_status": decision.adjudication_status,
            },
            "run_identifier": run_identifier,
            "code_revision": code_revision,
            "native_voxel_count": native_voxel_count,
            "required": requiredness_for(
                source,
                roi_original_name,
                contract=getattr(config, "radiomics_analysis_contract", {}) or {},
                modality="CT",
                explicitly_selected_model=source.startswith("CustomModel:"),
            ) == Requiredness.ANALYSIS_REQUIRED,
            "configured_parameter_hashes": hashes,
            "metadata": {
                "modality": "CT",
                "segmentation_source": source,
                "course_dir": str(course_dir),
                "patient_id": course_dir.parent.name,
                "course_id": course_dir.name,
                "series_uid": series_uid,
                "mask_identity": identity[0],
                "roi_original_name": roi_original_name,
                "stable_roi_identifier": identity[1],
                "roi_name": display_roi,
                "structure_cropped": bool(cropped_flag),
            },
        }

    # Save CT image to a temporary NRRD. Serialization is required acquisition
    # preparation and must not be reclassified as an empty course.
    try:
        with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as ct_file:
            ct_image_path = ct_file.name
        sitk.WriteImage(ct_image, ct_image_path, useCompression=True)
    except Exception as exc:
        try:
            Path(ct_image_path).unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"Failed to serialize CT image for radiomics in {course_dir}: {exc}"
        ) from exc

    tasks: List[Dict[str, Any]] = []
    preparation_failures: List[Dict[str, Any]] = []
    try:
        from rt_utils import RTStructBuilder

        min_voxels_limit, max_voxels_limit = _ct_voxel_limits(config)

        for segmentation_source, rs_file, selected_rois in source_specs:
            best_effort = True

            def _roi_is_required(roi_name: str) -> bool:
                return requiredness_for(
                    segmentation_source,
                    roi_name,
                    contract=getattr(config, "radiomics_analysis_contract", {}) or {},
                    modality="CT",
                    explicitly_selected_model=segmentation_source.startswith("CustomModel:"),
                ) == Requiredness.ANALYSIS_REQUIRED

            def _record_preparation_failure(
                roi_name: str,
                reason: str,
                *,
                status: str = "failed",
                failure_kind: str = "extraction_error",
                reason_code: Optional[str] = None,
                metadata: Optional[Mapping[str, Any]] = None,
            ) -> None:
                if (
                    _roi_is_required(roi_name)
                    and not extraction_status_is_nonfatal_for_required(status)
                ):
                    raise RadiomicsCourseExtractionError(reason)
                preparation_failures.append(
                    {
                        "source": segmentation_source,
                        "rs_file": str(rs_file),
                        "roi_name": str(roi_name),
                        "status": status,
                        "failure_kind": failure_kind,
                        "reason": reason,
                        "reason_code": reason_code,
                        "metadata": dict(metadata or {}),
                    }
                )
                logger.warning(
                    "Best-effort radiomics ROI %s/%s was not prepared: %s",
                    segmentation_source,
                    roi_name,
                    reason,
                )

            try:
                from .roi_requiredness import inspect_rtstruct
                inventory = inspect_rtstruct(rs_file)
            except Exception:
                inventory = None
            preflight_excluded = set()
            if inventory is not None:
                for observation in inventory.named_rois:
                    if observation.structural_code:
                        if _roi_is_required(observation.name):
                            raise RadiomicsCourseExtractionError(
                                f"Required ROI {observation.name!r} in {rs_file} has "
                                f"{observation.structural_code}"
                            )
                        preflight_excluded.add(observation.name)
                        if observation.structural_code not in {
                            "ROI_DECLARED_NO_CONTOUR_ITEM",
                            "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
                        }:
                            _record_preparation_failure(
                            observation.name,
                            f"ROI {observation.name!r} in {rs_file} has structural status {observation.structural_code}",
                            failure_kind="structural_roi_error",
                            reason_code=observation.structural_code,
                        )

            try:
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir),
                    rt_struct_path=str(rs_file),
                )
                roi_names = (
                    list(selected_rois)
                    if selected_rois is not None
                    else list(rtstruct.get_roi_names())
                )
                roi_names = [name for name in roi_names if name not in preflight_excluded]
            except Exception as exc:
                if best_effort:
                    try:
                        from .radiomics import _list_roi_names_dicom

                        advertised_rois = list(_list_roi_names_dicom(rs_file))
                    except Exception:
                        advertised_rois = []
                    if advertised_rois:
                        for advertised_roi in advertised_rois:
                            _record_preparation_failure(
                                advertised_roi,
                                f"Failed to construct RTSTRUCT reader for {segmentation_source} "
                                f"at {rs_file}: {exc}",
                            )
                        continue
                raise RadiomicsCourseExtractionError(
                    f"Failed to construct RTSTRUCT reader for {segmentation_source} at {rs_file}: {exc}"
                ) from exc
            if not roi_names:
                if inventory is not None:
                    continue
                raise RadiomicsCourseExtractionError(
                    f"Required RTSTRUCT contains no named ROIs: {segmentation_source} at {rs_file}"
                )
            if len(roi_names) != len(set(roi_names)):
                raise RadiomicsCourseExtractionError(
                    f"Required RTSTRUCT has duplicate ROI identities: {segmentation_source} at {rs_file}"
                )

            for roi_name in roi_names:
                norm_key = _norm(roi_name)
                if norm_key in skip_rois:
                    _record_preparation_failure(
                        roi_name,
                        "ROI is listed in radiomics_skip_rois",
                        status="declared_skip",
                        failure_kind="declared_ineligible",
                    )
                    continue
                try:
                    mask = rtstruct.get_roi_mask_by_name(roi_name)
                except Exception as exc:
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        f"could not be read: {exc}",
                    )
                    continue
                if mask is None:
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        "did not provide a mask",
                    )
                    continue
                try:
                    mask_bool = np.asarray(mask).astype(bool)
                except Exception as exc:
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        f"could not be converted to a mask: {exc}",
                    )
                    continue
                if not mask_bool.any():
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        "produced an empty mask",
                        failure_kind="degenerate_mask",
                    )
                    continue

                voxel_count = int(mask_bool.sum())
                if voxel_count < min_voxels_limit:
                    _record_preparation_failure(
                        roi_name,
                        f"ROI contains {voxel_count} voxels; configured minimum is "
                        f"{min_voxels_limit}",
                        status="below_minimum_voxels",
                        failure_kind="degenerate_mask",
                    )
                    continue
                spacing = tuple(ct_info.get("spacing", (1.0, 1.0, 1.0)))
                resampled_spacing, pad_distance = configured_grid_settings(
                    parameter_path,
                    native_spacing_xyz=spacing,
                )
                work_estimate = estimate_resampled_bounding_box(
                    mask_bool,
                    native_spacing_xyz=spacing,
                    resampled_spacing_xyz=resampled_spacing,
                    array_axis_to_xyz=(1, 0, 2),
                    pad_distance=pad_distance,
                )
                max_bbox_voxels = resolve_max_resampled_bbox_voxels(config)
                if (work_estimate.estimated_resampled_bbox_voxels > max_bbox_voxels
                        and not permits_second_stage(
                            work_estimate, mask_bool, native_spacing_xyz=spacing,
                            array_axis_to_xyz=(1, 0, 2), params_file=parameter_path,
                            limit=max_bbox_voxels)):
                    detail = (
                        f"ROI {roi_name} requires an estimated padded resampled "
                        f"bounding box of "
                        f"{work_estimate.estimated_resampled_bbox_voxels} voxels "
                        f"({work_estimate.estimated_resampled_bbox_shape}); configured "
                        f"maximum is {max_bbox_voxels}. Full configured radiomics was "
                        "not started."
                    )
                    _record_preparation_failure(
                        roi_name,
                        detail,
                        failure_kind="resource_limit",
                        reason_code=FAILED_RADIOMICS_RESOURCE_LIMIT,
                        metadata={
                            "reason_code": FAILED_RADIOMICS_RESOURCE_LIMIT,
                            "roi_structural_code": RESAMPLED_BBOX_LIMIT_CODE,
                            **work_estimate.metadata(limit=max_bbox_voxels),
                        },
                    )
                    continue
                try:
                    native_voxel_mm3 = (
                        float(spacing[0]) * float(spacing[1]) * float(spacing[2])
                    )
                except Exception:
                    native_voxel_mm3 = 1.0
                physical_volume_mm3 = float(voxel_count) * max(1e-9, native_voxel_mm3)
                estimated_voxels = physical_volume_mm3
                large_roi = norm_key.startswith("body") or (
                    estimated_voxels > float(max_voxels_limit)
                )
                if large_roi:
                    logger.info(
                        "ROI %s/%s is large (native=%d voxels, est@1mm=%.0f voxels, "
                        "cap=%d); preserving full configured radiomics settings",
                        segmentation_source,
                        roi_name,
                        voxel_count,
                        estimated_voxels,
                        int(max_voxels_limit),
                    )

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".nrrd", delete=False
                    ) as mask_file:
                        _write_mask_to_file(mask_bool, mask_file.name, ct_info)
                        mask_path = mask_file.name
                except Exception as exc:
                    _record_preparation_failure(
                        roi_name,
                        f"Failed to serialize mask {segmentation_source}/{roi_name} "
                        f"from {rs_file}: {exc}",
                    )
                    continue

                cropped_flag = mask_is_cropped(mask_bool)
                display_roi = (
                    roi_name
                    if (not cropped_flag or roi_name.endswith("__partial"))
                    else f"{roi_name}__partial"
                )
                tasks.append({
                    "image_path": ct_image_path,
                    "mask_path": mask_path,
                    "roi_name": display_roi,
                    "params_file": params_file,
                    "label": None,
                    "large_roi": bool(large_roi),
                    **_governed_task_fields(
                        segmentation_source,
                        rs_file,
                        roi_name,
                        display_roi,
                        cropped_flag,
                        voxel_count,
                        bool(large_roi),
                    ),
                    "cleanup": True,
                    "ct_info": ct_info,
                })

                tasks[-1]["resource_guard_legacy"] = legacy_rejection(
                    work_estimate, max_bbox_voxels, roi_name)

        for failure in preparation_failures:
            tasks.append(
                {
                    "image_path": ct_image_path,
                    "mask_path": None,
                    "roi_name": failure["roi_name"],
                    "params_file": params_file,
                    "label": None,
                    **_governed_task_fields(
                        failure["source"],
                        Path(failure["rs_file"]),
                        failure["roi_name"],
                        failure["roi_name"],
                        False,
                        None,
                        False,
                    ),
                    "precomputed_failure": failure,
                    "cleanup": False,
                    "ct_info": ct_info,
                }
            )

    except Exception as exc:
        logger.error("Failed to prepare radiomics masks for %s: %s", course_dir, exc)
        Path(ct_image_path).unlink(missing_ok=True)
        for task in tasks:
            mask_path = task.get('mask_path')
            if mask_path:
                Path(mask_path).unlink(missing_ok=True)
        _invalidate_radiomics_outputs(output_path)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"RTSTRUCT radiomics preparation failed for {course_dir}: {exc}"
        ) from exc

    if not tasks:
        logger.warning("No eligible ROIs found in current RTSTRUCT inventory for %s", course_dir)
        Path(ct_image_path).unlink(missing_ok=True)
        _invalidate_radiomics_outputs(output_path)
        return None

    sequential = os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes')

    # Determine worker count - respect Snakemake thread budget via env var or config
    max_workers = None
    worker_source = "unknown"

    # Priority 1: Environment variable (set by CLI from --max-workers or by Snakemake)
    env_workers = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)
    if env_workers > 0:
        max_workers = env_workers
        worker_source = "RTPIPELINE_MAX_WORKERS env"

    # Priority 2: PipelineConfig.effective_workers() (respects --max-workers CLI arg)
    if max_workers is None and hasattr(config, 'effective_workers') and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
            worker_source = "config.effective_workers()"
        except Exception:
            pass

    # Priority 3: Safe default based on task count (not cpu_count to avoid oversubscription)
    if max_workers is None:
        # When no budget is set, use conservative parallelism to avoid oversubscribing
        # With batch processing, each worker runs one subprocess containing multiple ROIs
        max_workers = min(4, len(tasks))  # Max 4 parallel batches by default
        worker_source = "default (no budget set)"
        logger.warning(
            "No worker budget set (RTPIPELINE_MAX_WORKERS env or config.effective_workers). "
            "Using conservative default of %d workers. Set RTPIPELINE_MAX_WORKERS for optimal performance.",
            max_workers
        )

    logger.info("Conda radiomics using %d workers (%s, CPU cores: %d)", max_workers, worker_source, os.cpu_count() or 0)

    # Enable checkpointing for resumable extraction
    checkpoint_path = course_dir / "metadata" / "radiomics_ct_checkpoint.parquet"

    try:
        result = process_radiomics_batch(
            tasks,
            output_path,
            sequential=sequential,
            max_workers=max_workers,
            checkpoint_path=checkpoint_path,
            enable_heartbeat=True,
            env_probe_timeout=getattr(config, "radiomics_env_probe_timeout", None),
            acquisition_descriptor=describe_contract_planning_ct(contract),
        )
        result_rows = []
        if result is not None:
            try:
                import pandas as pd
                result_rows = pd.read_parquet(Path(result).with_suffix(".parquet")).to_dict("records")
            except Exception:
                result_rows = []
        _write_conda_roi_ledger(course_dir, tasks, result_rows, extracted=result is not None)
        return result
    finally:
        Path(ct_image_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MR course radiomics: source identity, mask inventory and ledger (MR-private)
# ---------------------------------------------------------------------------

_MR_MODEL = "total_mr"
_MR_SEGMENTATION_SOURCE = "AutoTS_total_mr"
# ``segmentation._materialize_masks`` copies TotalSegmentator's aggregate label
# volume next to the per-ROI masks under the same ``<model>--`` prefix. It is a
# labelmap, not an ROI; ``auto_rtstruct`` excludes the same two names when it
# enumerates ROI masks. Measuring it publishes features of the union of every
# structure under the ROI name "multilabel".
_MR_NON_ROI_MASK_NAMES = frozenset({"multilabel", "segmentations"})
# Used where an identity field is genuinely unavailable, so a publication key
# never carries a blank field and a missing value is never mistaken for a real
# digest. Mirrors the CT contract's own "unavailable" marker.
_MR_UNAVAILABLE = "unavailable"
# Source-identity fields every MR ledger row keeps, so one ROI name measured
# from two series stays two accounted outcomes. All six are also verified
# against the current tasks by ``_mr_publication_drift`` before publication.
_MR_LEDGER_IDENTITY_FIELDS = (
    "segmentation_source",
    "series_uid",
    "nifti_path",
    "source_content_sha256",
    "mask_path_source",
    "mask_identity",
)
# Fields every published MR row must still agree with after extraction.
_MR_BINDING_FIELDS = (
    "segmentation_source",
    "series_uid",
    "nifti_path",
    "source_content_sha256",
    "mask_path_source",
    "mask_identity",
    "configured_parameter_hash",
    "code_revision",
)


class _MrSourceError(Exception):
    """A discovered MR series cannot be bound to one readable current source."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


def _mr_sidecar_candidates(nifti_path: Path) -> List[Path]:
    """Return the sidecar names the real producers write for one MR NIfTI.

    ``organize._convert_related_series`` writes ``<base>.metadata.json`` while
    ``segmentation`` writes ``<base>.nii.metadata.json`` (``Path.stem`` keeps the
    inner ``.nii`` of ``.nii.gz``). A reader that knows only one spelling loses
    the identity of every series the other producer converted.
    """
    names = dict.fromkeys(
        (
            f"{_strip_nii_suffix(nifti_path)}.metadata.json",
            f"{nifti_path.stem}.metadata.json",
        )
    )
    return [nifti_path.with_name(name) for name in names]


def _mr_is_nifti(path: Path) -> bool:
    """True only for a NIfTI volume. ``*.nii*`` also matches ``*.nii.metadata.json``."""
    return path.name.endswith((".nii", ".nii.gz"))


def _mr_read_sidecar(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _MrSourceError(
            "failed_source_read",
            f"MR metadata sidecar {path.name} is unreadable: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise _MrSourceError(
            "failed_source_read",
            f"MR metadata sidecar {path.name} is not a JSON object",
        )
    return payload


def _mr_dicom_files(dicom_dir: Path, series_root: Path) -> List[Path]:
    """Return the series' DICOM instances without walking sibling derived output."""
    flat = sorted(path for path in dicom_dir.glob("*.dcm") if path.is_file())
    try:
        is_series_root = dicom_dir.resolve() == series_root.resolve()
    except OSError:
        is_series_root = dicom_dir == series_root
    if flat or is_series_root:
        # A series root also holds NIFTI/ and Segmentation_TotalSegmentator/, and
        # the segmentation RTSTRUCT there is a different DICOM series.
        return flat
    return sorted(path for path in dicom_dir.rglob("*.dcm") if path.is_file())


def _mr_dicom_identity(dicom_dir: Path, series_root: Path) -> Dict[str, Any]:
    """Return one unambiguous readable MR series identity, or fail closed."""
    files = _mr_dicom_files(dicom_dir, series_root)
    if not files:
        raise _MrSourceError(
            "failed_source_read", f"MR series has no DICOM instance under {dicom_dir}"
        )
    series_uids: Set[str] = set()
    study_uids: Set[str] = set()
    modalities: Set[str] = set()
    sop_uids: List[str] = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            raise _MrSourceError(
                "failed_source_read",
                f"MR DICOM instance {path.name} is unreadable: {exc}",
            ) from exc
        series_uids.add(str(getattr(ds, "SeriesInstanceUID", "") or "").strip())
        study_uids.add(str(getattr(ds, "StudyInstanceUID", "") or "").strip())
        modalities.add(str(getattr(ds, "Modality", "") or "").strip().upper())
        sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "").strip()
        if not sop_uid:
            raise _MrSourceError(
                "failed_source_read",
                f"MR DICOM instance {path.name} declares no SOPInstanceUID",
            )
        sop_uids.append(sop_uid)
    if len(series_uids) != 1 or not next(iter(series_uids)):
        raise _MrSourceError(
            "failed_source_read",
            f"{dicom_dir} holds {len(series_uids)} SeriesInstanceUID value(s): "
            f"{sorted(series_uids)}",
        )
    if len(study_uids) != 1 or not next(iter(study_uids)):
        # Collapsing a mixed or absent study identity to a blank field would let
        # instances from two studies publish as one measured series.
        raise _MrSourceError(
            "failed_source_read",
            f"{dicom_dir} holds {len(study_uids)} StudyInstanceUID value(s): "
            f"{sorted(study_uids)}",
        )
    if modalities != {"MR"}:
        raise _MrSourceError(
            "failed_source_read",
            f"{dicom_dir} holds non-MR modality {sorted(modalities)}",
        )
    if len(sop_uids) != len(set(sop_uids)):
        raise _MrSourceError(
            "failed_source_read", f"{dicom_dir} holds duplicate SOPInstanceUID values"
        )
    return {
        "series_instance_uid": next(iter(series_uids)),
        "study_instance_uid": next(iter(study_uids)),
        "sop_uids": frozenset(sop_uids),
        "instance_count": len(sop_uids),
    }


def _mr_resolve_series_source(series_root: Path) -> Dict[str, Any]:
    """Bind one MR series to exactly one current, readable, self-consistent source.

    Selection, metadata and DICOM identity are one decision: the chosen image is
    the one its own producer sidecar names, that sidecar declares MR, and the
    DICOM series it points at is readable, unambiguous, and unchanged since the
    sidecar was written.
    """
    nifti_dir = series_root / "NIFTI"
    nifti_files = sorted(
        path
        for path in nifti_dir.glob("*.nii*")
        if path.is_file() and _mr_is_nifti(path) and not path.name.startswith(".")
    )
    if not nifti_files:
        raise _MrSourceError("failed_source_read", "MR series has no NIfTI image")

    candidates: List[Tuple[Path, Path, Dict[str, Any]]] = []
    contradictions: List[str] = []
    for nifti_path in nifti_files:
        present = [path for path in _mr_sidecar_candidates(nifti_path) if path.is_file()]
        if not present:
            contradictions.append(f"{nifti_path.name} has no producer metadata sidecar")
            continue
        if len(present) > 1:
            contradictions.append(
                f"{nifti_path.name} has {len(present)} competing metadata sidecars"
            )
            continue
        sidecar = present[0]
        meta = _mr_read_sidecar(sidecar)
        recorded_name = Path(str(meta.get("nifti_path") or "")).name
        if recorded_name and recorded_name != nifti_path.name:
            contradictions.append(
                f"{sidecar.name} describes {recorded_name}, not {nifti_path.name}"
            )
            continue
        modality = str(meta.get("modality") or "").strip().upper()
        if modality != "MR":
            contradictions.append(
                f"{nifti_path.name} is declared modality {modality or 'unset'}"
            )
            continue
        candidates.append((nifti_path, sidecar, meta))

    if len(candidates) != 1 or contradictions:
        detail = (
            f"MR series does not resolve to exactly one identity-linked MR image "
            f"({len(candidates)} of {len(nifti_files)} NIfTI file(s))"
        )
        if contradictions:
            detail = f"{detail}: " + "; ".join(contradictions)
        raise _MrSourceError("failed_source_read", detail)

    nifti_path, sidecar, meta = candidates[0]

    dicom_dir = series_root / "DICOM" if (series_root / "DICOM").is_dir() else series_root
    recorded_dir = str(meta.get("source_directory") or "").strip()
    if recorded_dir:
        # Only an in-course source directory is honored. A recorded absolute path
        # that now resolves outside this course would silently measure another
        # copy of the series after the course was moved or duplicated.
        candidate_dir = Path(recorded_dir)
        try:
            inside = candidate_dir.resolve().is_relative_to(series_root.resolve())
        except (OSError, ValueError):
            inside = False
        if inside and candidate_dir.is_dir():
            dicom_dir = candidate_dir

    identity = _mr_dicom_identity(dicom_dir, series_root)

    sidecar_series_uid = str(meta.get("series_instance_uid") or "").strip()
    if not sidecar_series_uid:
        raise _MrSourceError(
            "failed_source_read", f"{sidecar.name} records no SeriesInstanceUID"
        )
    if sidecar_series_uid != identity["series_instance_uid"]:
        raise _MrSourceError(
            "failed_source_read",
            f"{sidecar.name} names series {sidecar_series_uid} but {dicom_dir} holds "
            f"{identity['series_instance_uid']}",
        )

    try:
        image_digest = file_sha256(nifti_path)
    except OSError as exc:
        raise _MrSourceError(
            "failed_source_read", f"MR image {nifti_path.name} is unreadable: {exc}"
        ) from exc
    recorded_digest = str(meta.get("nifti_sha256") or "").strip()
    if recorded_digest and recorded_digest != image_digest:
        raise _MrSourceError(
            "failed_source_read",
            f"MR image {nifti_path.name} changed after {sidecar.name} recorded its content",
        )

    recorded_instances = meta.get("instances")
    if isinstance(recorded_instances, (list, tuple)) and recorded_instances:
        if {str(value) for value in recorded_instances} != set(identity["sop_uids"]):
            raise _MrSourceError(
                "failed_source_read",
                f"MR DICOM instances changed after {sidecar.name} recorded them "
                f"({len(recorded_instances)} recorded, {identity['instance_count']} present)",
            )

    study_uid = str(meta.get("study_instance_uid") or "").strip()
    if study_uid and identity["study_instance_uid"] and study_uid != identity["study_instance_uid"]:
        raise _MrSourceError(
            "failed_source_read",
            f"{sidecar.name} names study {study_uid} but {dicom_dir} holds "
            f"{identity['study_instance_uid']}",
        )

    return {
        "nifti_path": nifti_path,
        "sidecar": sidecar,
        "dicom_dir": dicom_dir,
        "series_uid": identity["series_instance_uid"],
        "study_uid": study_uid or identity["study_instance_uid"],
        "image_digest": image_digest,
    }


def _mr_roi_masks(seg_dir: Path) -> List[Tuple[str, Path]]:
    """Return every discovered per-ROI ``total_mr`` mask, aggregates excluded."""
    masks: List[Tuple[str, Path]] = []
    for mask_path in sorted(seg_dir.glob(f"{_MR_MODEL}--*.nii*")):
        if not mask_path.is_file() or not _mr_is_nifti(mask_path):
            continue
        roi_name = _totalseg_roi_name(mask_path)
        if not roi_name or roi_name in _MR_NON_ROI_MASK_NAMES:
            logger.debug("Ignoring non-ROI MR segmentation artifact %s", mask_path)
            continue
        masks.append((roi_name, mask_path))
    return masks


def _mr_source_digest(path: Path) -> str:
    """Digest one source file, recording a read failure instead of raising."""
    try:
        return file_sha256(path)
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}"


def _mr_series_fingerprint(series_root: Path) -> Dict[str, str]:
    """Digest every current source byte one MR series measurement depends on.

    Covers the converted image, its producer metadata sidecar, the ROI mask
    inventory and the raw DICOM instances, keyed by path so an arrival or a
    removal is as visible as a content change. Recorded UIDs cannot show any of
    that: a header can be rewritten and a mask replaced while every identifier
    on the row stays the same.
    """
    entries: Dict[str, str] = {}

    def _record(candidates: Any) -> None:
        for path in sorted(candidates):
            if not path.is_file():
                continue
            try:
                key = str(path.relative_to(series_root))
            except ValueError:
                key = str(path)
            entries[key] = _mr_source_digest(path)

    _record((series_root / "NIFTI").glob("*"))
    _record((series_root / "Segmentation_TotalSegmentator").glob("*.nii*"))
    _record(series_root.rglob("*.dcm"))
    return entries


def _mr_source_inventory(mr_root: Path) -> Dict[str, Dict[str, str]]:
    """Fingerprint every MR series directory currently under the course."""
    try:
        series_dirs = sorted(path for path in mr_root.iterdir() if path.is_dir())
    except OSError:
        return {}
    return {path.name: _mr_series_fingerprint(path) for path in series_dirs}


def _mr_source_drift(
    before: Mapping[str, Mapping[str, str]], after: Mapping[str, Mapping[str, str]]
) -> Optional[str]:
    """Return the first live MR source change observed since ``before``."""
    for series in sorted(set(before) | set(after)):
        original = before.get(series)
        current = after.get(series)
        if original is None:
            return f"MR series directory {series} appeared during extraction"
        if current is None:
            return f"MR series directory {series} disappeared during extraction"
        for name in sorted(set(original) | set(current)):
            if name not in original:
                return f"MR source {series}/{name} appeared during extraction"
            if name not in current:
                return f"MR source {series}/{name} disappeared during extraction"
            if original[name] != current[name]:
                return f"MR source {series}/{name} changed during extraction"
    return None


def _mr_geometry_matches(image: "sitk.Image", mask: "sitk.Image") -> bool:
    if tuple(image.GetSize()) != tuple(mask.GetSize()):
        return False
    for left, right in (
        (image.GetSpacing(), mask.GetSpacing()),
        (image.GetOrigin(), mask.GetOrigin()),
        (image.GetDirection(), mask.GetDirection()),
    ):
        if len(left) != len(right):
            return False
        if any(abs(float(a) - float(b)) > 1e-4 for a, b in zip(left, right)):
            return False
    return True


def _mr_row_identity(
    course_dir: Path,
    *,
    roi_name: str,
    series_uid: str,
    study_uid: str,
    dicom_dir: str,
    nifti_path: str,
    image_digest: str,
    mask_path: str,
    mask_identity: str,
    parameter_arm: str,
    configured_parameter_hash_value: str,
    run_identifier: str,
    code_revision: str,
) -> Dict[str, Any]:
    """Every field a published MR row needs to name its own current source.

    All names are declared publication columns in ``radiomics_schema``; this adds
    no new column to the table.
    """
    return {
        "modality": "MR",
        "image_modality": "MR",
        "segmentation_source": _MR_SEGMENTATION_SOURCE,
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "course_dir": str(course_dir),
        "series_uid": series_uid,
        "study_uid": study_uid,
        "series_dir": dicom_dir,
        "nifti_path": nifti_path,
        "source_content_sha256": image_digest,
        "mask_path_source": mask_path,
        "mask_identity": mask_identity,
        "roi_name": roi_name,
        "roi_original_name": roi_name,
        "stable_roi_identifier": roi_name,
        "extraction_arm": parameter_arm,
        "configured_parameter_hash": configured_parameter_hash_value,
        "run_identifier": run_identifier,
        "code_revision": code_revision,
    }


def _mr_disposition_task(
    disposition: Mapping[str, Any], params_file: Optional[Path]
) -> Dict[str, Any]:
    """Turn a non-measurement outcome into a durable published failure row.

    Mirrors the CT NIfTI fallback: a disposition travels through
    ``process_radiomics_batch`` so it is counted in the course diagnostics, lands
    in the workbook, and fails the course closed when the ROI is required.
    """
    metadata = dict(disposition["metadata"])
    metadata["roi_structural_code"] = disposition["reason_code"]
    return {
        "image_path": "mr-source-disposition",
        "mask_path": None,
        "roi_name": str(metadata["roi_original_name"]),
        "params_file": str(params_file) if params_file else None,
        "label": None,
        "cleanup": False,
        "required": bool(disposition["required"]),
        "metadata": metadata,
        "precomputed_failure": {
            "reason": disposition["detail"],
            "status": disposition["status"],
            "failure_kind": disposition["failure_kind"],
            "reason_code": disposition["reason_code"],
            "metadata": {"roi_structural_code": disposition["reason_code"]},
        },
    }


def _mr_ledger_rows(dispositions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Ledger-only rows for a course that publishes no workbook.

    The full source identity travels with the outcome: a course that published
    nothing must still say which series and which mask bytes it accounted for.
    """
    rows: List[Dict[str, Any]] = []
    for disposition in dispositions:
        row = dict(disposition["metadata"])
        row.update(
            {
                "reason_code": disposition["reason_code"],
                "roi_structural_code": disposition["reason_code"],
                "extraction_status": disposition["status"],
                "extraction_status_detail": disposition["detail"],
            }
        )
        rows.append(row)
    return rows


def _mr_expected_bindings(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    bindings: Dict[str, Dict[str, str]] = {}
    for task in tasks:
        metadata = task.get("metadata") or {}
        bindings[_roi_instance_key(task)] = {
            field: str(metadata.get(field) or "") for field in _MR_BINDING_FIELDS
        }
    return bindings


def _mr_reject_stale_checkpoint(
    checkpoint_path: Path, tasks: Sequence[Mapping[str, Any]], out_path: Path
) -> None:
    """Delete a checkpoint that no longer describes the current MR sources.

    ``RadiomicsCheckpoint`` only compares identities, and its configured-parameter
    check is CT-only, so a resumed MR course could otherwise republish rows
    measured from replaced image or mask bytes.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return
    stale: Optional[str] = None
    try:
        records = pd.read_parquet(checkpoint_path).to_dict("records")
        expected = _mr_expected_bindings(tasks)
    except Exception as exc:
        stale = f"checkpoint is unreadable or the current inventory is invalid: {exc}"
        records, expected = [], {}
    for record in records:
        if stale:
            break
        try:
            key = _roi_instance_key(record)
        except ValueError as exc:
            stale = f"checkpoint row has no usable identity: {exc}"
            break
        binding = expected.get(key)
        if binding is None:
            stale = "checkpoint row is outside the current MR task inventory"
            break
        for field, value in binding.items():
            if _identity_value(record, field) != value:
                stale = f"checkpoint {field} no longer matches the current source"
                break
    if stale:
        logger.warning("Rejecting stale MR radiomics checkpoint %s: %s", checkpoint_path, stale)
        _remove_artifact_strict(
            checkpoint_path, context="rejecting stale MR radiomics checkpoint"
        )
        _invalidate_radiomics_outputs(out_path)


def _mr_publication_drift(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> Optional[str]:
    """Return the first published row that is not bound to a current MR source."""
    expected = _mr_expected_bindings(tasks)
    for row in rows:
        try:
            key = _roi_instance_key(row)
        except ValueError as exc:
            return f"published row has no usable identity: {exc}"
        binding = expected.get(key)
        if binding is None:
            return (
                "published row "
                f"{_identity_value(row, 'roi_original_name', fallback='roi_name')!r} "
                "is not part of the current MR task inventory"
            )
        for field, value in binding.items():
            if _identity_value(row, field) != value:
                return (
                    f"published {field} for "
                    f"{_identity_value(row, 'roi_original_name', fallback='roi_name')!r} "
                    "does not match its current source"
                )
    return None


def _mr_unmet_requirements(
    requirements: Sequence[Any], rows: Sequence[Mapping[str, Any]]
) -> List[str]:
    """Required ROIs that no row actually measured from an accepted source.

    A row only satisfies a requirement when its source matches, its name matches
    an accepted alias, and its outcome is a measurement (or one of the
    dispositions the CT gate has always treated as non-fatal). A failure row
    carrying the required name no longer discharges the requirement.
    """
    measured: Set[Tuple[str, str]] = set()
    for row in rows:
        status = _identity_value(row, "extraction_status") or "success"
        if status != "success" and not extraction_status_is_nonfatal_for_required(status):
            continue
        roi_name = _identity_value(row, "roi_original_name", fallback="roi_name")
        if not roi_name:
            continue
        measured.add(
            (
                _norm_roi_key(_identity_value(row, "segmentation_source")),
                _norm_roi_key(roi_name),
            )
        )
    unmet: List[str] = []
    for requirement in requirements:
        if requirement.requiredness != Requiredness.ANALYSIS_REQUIRED:
            continue
        wanted_source = _norm_roi_key(requirement.source) if requirement.source else None
        accepted = {_norm_roi_key(alias) for alias in requirement.accepted_names}
        if not any(
            roi in accepted and (wanted_source is None or wanted_source == source)
            for source, roi in measured
        ):
            unmet.append(requirement.canonical_name)
    return unmet


def _mr_published_rows(result: Optional[Path]) -> List[Dict[str, Any]]:
    """Read back what was actually published; a read failure is not an empty course."""
    if result is None:
        return []
    parquet_path = Path(result).with_suffix(".parquet")
    try:
        return pd.read_parquet(parquet_path).to_dict("records")
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"Published MR radiomics table {parquet_path} cannot be read back: {exc}"
        ) from exc


def _withdraw_mr_ledger(course_dir: Path, *, context: str) -> None:
    """Retract a superseded MR ledger, including the combined view of it.

    ``invalidate_radiomics_outputs`` removes a workbook and its Parquet sidecar
    and nothing else, so an earlier successful run's ledger outlives the
    measurements it describes. That ledger is what the cohort denominator is
    built from: ``workflow_aggregate._write_radiomics_denominator_aggregate``
    reads the combined ``radiomics_roi_ledger.json`` that ``write_modality_ledger``
    rebuilds out of the per-modality files, so both have to go. A course whose
    live sources changed under it cannot replace the ledger with a current-source
    statement about bytes that are no longer there, so it publishes nothing for MR.
    """
    metadata_dir = Path(course_dir) / "metadata"
    superseded = [
        metadata_dir / f"radiomics_mr_{name}.json"
        for name in ("roi_ledger", "denominators", "patient_ledger")
    ]
    # Withdraw the consumer-facing view first, even if a previous interrupted
    # cleanup already removed all modality files. A failed CT read or rebuild
    # must not leave the combined ledger advertising superseded MR success.
    for name in ("roi_ledger", "denominators", "patient_ledger"):
        _remove_artifact_strict(
            metadata_dir / f"radiomics_{name}.json", context=context
        )
    for path in superseded:
        _remove_artifact_strict(path, context=context)
    surviving = metadata_dir / "radiomics_ct_roi_ledger.json"
    if not surviving.exists():
        return
    try:
        payload = json.loads(surviving.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            not isinstance(payload.get(name), list)
            or any(not isinstance(row, dict) for row in payload[name])
            for name in ("course", "course_roi")
        ):
            raise ValueError("CT ledger must contain course and course_roi row lists")
    except (OSError, ValueError) as exc:
        raise RadiomicsCourseExtractionError(
            f"Surviving CT ledger {surviving} is unreadable while {context}: {exc}"
        ) from exc
    # Re-emitting the surviving CT ledger unchanged is how the combined view is
    # rebuilt from the modality ledgers still on disk. That merge belongs to
    # ``write_modality_ledger`` and is not restated here.
    write_modality_ledger(
        metadata_dir,
        DenominatorLedger(
            course_rows=[dict(row) for row in payload.get("course", ())],
            roi_rows=[dict(row) for row in payload.get("course_roi", ())],
        ),
        "CT",
    )


def radiomics_for_course_mr(
    course_dir: Path,
    config: Any,
    params_file: Optional[Path] = None
) -> Optional[Path]:
    """
    Extract MR radiomics features using conda environment.

    Every MR series directory is accounted for. A series is measured only when
    exactly one NIfTI image, its own producer metadata sidecar, and one readable
    unambiguous MR DICOM series agree, and the recorded content still matches the
    bytes on disk. Every other discovered series or mask keeps an explicit
    disposition instead of disappearing.

    Args:
        course_dir: Path to course directory
        config: Pipeline configuration
        params_file: Optional path to MR radiomics parameters YAML

    Returns:
        Path to output radiomics_mr.xlsx if successful, None otherwise
    """
    course_dir = Path(course_dir)
    mr_root = course_dir / "MR"
    out_path = mr_root / "radiomics_mr.xlsx"
    configured_mr_params = getattr(config, 'radiomics_params_file_mr', None) if config is not None else None
    analysis_contract = getattr(config, "radiomics_analysis_contract", {}) or {}
    mr_requirements = requirements_from_contract(analysis_contract, "MR")
    mr_required_names = {
        requirement.canonical_name
        for requirement in mr_requirements
        if requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
    }
    mr_required = bool(mr_required_names)
    mr_expected_names = [
        requirement.canonical_name
        for requirement in mr_requirements
        if requirement.requiredness != Requiredness.INVENTORY_ONLY
    ]

    def _mr_missing_reason(default: str) -> Optional[Callable[[str], str]]:
        """A required ROI whose whole modality is absent is a technical failure.

        An optional ROI keeps ``not_applicable_modality``; a required one cannot
        be discharged by the same not-applicable skip.
        """
        if default != "not_applicable_modality" or not mr_required_names:
            return None
        return lambda name: (
            "failed_source_read" if name in mr_required_names else default
        )

    def _roi_is_required(roi_name: str) -> bool:
        return requiredness_for(
            _MR_SEGMENTATION_SOURCE,
            roi_name,
            contract=analysis_contract,
            modality="MR",
        ) == Requiredness.ANALYSIS_REQUIRED

    def _account_unbound_configuration(context: str) -> None:
        """Withdraw everything produced under a binding this run cannot resolve.

        The workbook, the checkpoint and the ledger all describe measurements
        made under a configured parameter binding that is no longer resolvable,
        so none of them may keep standing; the course is left saying only that
        its MR radiomics failed technically.
        """
        _invalidate_radiomics_outputs(out_path)
        _remove_artifact_strict(
            mr_root / "radiomics_mr_checkpoint.parquet", context=context
        )
        _write_conda_roi_ledger(
            course_dir,
            [],
            [],
            extracted=False,
            expected_names=mr_expected_names,
            missing_reason="failed_radiomics_extraction",
            identity_fields=_MR_LEDGER_IDENTITY_FIELDS,
            modality="MR",
        )

    if params_file is None and configured_mr_params is not None:
        params_file = Path(configured_mr_params)
    if params_file is not None and not Path(params_file).exists():
        _account_unbound_configuration(
            "discarding the MR checkpoint for a missing configured parameter file"
        )
        raise RadiomicsCourseExtractionError(
            f"Configured required MR radiomics parameter path is missing: {params_file}"
        )
    mr_parameter_arm = "mr_configured"
    try:
        mr_run_identifier = new_run_identifier()
        mr_code_revision = current_code_revision()
        mr_configured_parameter_hash = configured_parameter_hash(
            Path(params_file) if params_file is not None else None,
            arm=mr_parameter_arm,
            window=None,
            large_roi=False,
        )
    except Exception as exc:
        # Nothing may survive under a parameter binding this run could not
        # resolve: the workbook on disk would keep describing itself as current.
        _account_unbound_configuration(
            "discarding the MR checkpoint after a configuration binding failure"
        )
        raise RadiomicsCourseExtractionError(
            f"MR radiomics configuration binding failed for {course_dir}: {exc}"
        ) from exc

    if not mr_root.exists():
        logger.debug("No MR directory in %s", course_dir)
        _invalidate_radiomics_outputs(out_path)
        _write_conda_roi_ledger(
            course_dir,
            [],
            [],
            extracted=False,
            expected_names=mr_expected_names,
            missing_reason="not_applicable_modality",
            missing_reason_for=_mr_missing_reason("not_applicable_modality"),
            identity_fields=_MR_LEDGER_IDENTITY_FIELDS,
            derive_course_states=True,
            modality="MR",
        )
        if mr_required_names:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but {course_dir} has no MR directory; "
                "required ROI(s) were not measured: "
                + ", ".join(sorted(mr_required_names))
            )
        return None

    tasks: List[Dict[str, Any]] = []
    dispositions: List[Dict[str, Any]] = []
    temp_files: List[Path] = []
    min_voxels_limit, _ = _ct_voxel_limits(config)
    series_dirs = sorted(path for path in mr_root.iterdir() if path.is_dir())
    # Bind the live source bytes before anything is read, resolved or prepared,
    # so a source replaced while this course is being measured cannot publish.
    source_inventory = _mr_source_inventory(mr_root)

    def _record_disposition(
        roi_name: str,
        *,
        reason_code: str,
        detail: str,
        status: str = "failed",
        failure_kind: str = "source_read_error",
        series_uid: str,
        study_uid: str = "",
        dicom_dir: str = "",
        nifti_path: str = "",
        image_digest: str = _MR_UNAVAILABLE,
        mask_path: str = "",
        mask_identity: str = _MR_UNAVAILABLE,
    ) -> None:
        dispositions.append(
            {
                "reason_code": reason_code,
                "detail": detail,
                "status": status,
                "failure_kind": failure_kind,
                "required": _roi_is_required(roi_name),
                "metadata": _mr_row_identity(
                    course_dir,
                    roi_name=roi_name,
                    series_uid=series_uid,
                    study_uid=study_uid,
                    dicom_dir=dicom_dir,
                    nifti_path=nifti_path,
                    image_digest=image_digest,
                    mask_path=mask_path,
                    mask_identity=mask_identity,
                    parameter_arm=mr_parameter_arm,
                    configured_parameter_hash_value=mr_configured_parameter_hash,
                    run_identifier=mr_run_identifier,
                    code_revision=mr_code_revision,
                ),
            }
        )
        logger.warning(
            "MR radiomics did not measure %s (%s): %s", roi_name, reason_code, detail
        )

    for series_root in series_dirs:
        series_label = series_root.name
        nifti_dir = series_root / "NIFTI"
        seg_dir = series_root / "Segmentation_TotalSegmentator"

        if not nifti_dir.is_dir() or not seg_dir.is_dir():
            _record_disposition(
                series_label,
                reason_code="failed_source_segmentation",
                detail="MR series has no NIfTI conversion or TotalSegmentator output",
                series_uid=series_label,
            )
            continue

        roi_masks = _mr_roi_masks(seg_dir)

        try:
            source = _mr_resolve_series_source(series_root)
        except _MrSourceError as exc:
            # Every discovered mask keeps its own row, so a required ROI cannot
            # vanish with the series that carried it.
            if roi_masks:
                for roi_name, mask_path in roi_masks:
                    _record_disposition(
                        roi_name,
                        reason_code=exc.reason_code,
                        detail=exc.detail,
                        series_uid=series_label,
                        mask_path=str(mask_path),
                    )
            else:
                _record_disposition(
                    series_label,
                    reason_code=exc.reason_code,
                    detail=exc.detail,
                    series_uid=series_label,
                )
            continue

        series_uid = str(source["series_uid"])
        study_uid = str(source["study_uid"])
        dicom_dir = str(source["dicom_dir"])
        nifti_path = str(source["nifti_path"])
        image_digest = str(source["image_digest"])

        if not roi_masks:
            _record_disposition(
                series_label,
                reason_code="failed_source_segmentation",
                detail=f"MR segmentation produced no {_MR_MODEL} ROI mask in {seg_dir}",
                series_uid=series_uid,
                study_uid=study_uid,
                dicom_dir=dicom_dir,
                nifti_path=nifti_path,
                image_digest=image_digest,
            )
            continue

        def _series_disposition(roi_name: str, mask_path: Path, **kwargs: Any) -> None:
            _record_disposition(
                roi_name,
                series_uid=series_uid,
                study_uid=study_uid,
                dicom_dir=dicom_dir,
                nifti_path=nifti_path,
                image_digest=image_digest,
                mask_path=str(mask_path),
                **kwargs,
            )

        try:
            mr_img = sitk.ReadImage(nifti_path)
            mr_nrrd = tempfile.NamedTemporaryFile(
                suffix=".nrrd", delete=False, prefix="mr_image_"
            )
            mr_nrrd.close()
            sitk.WriteImage(mr_img, mr_nrrd.name)
            temp_files.append(Path(mr_nrrd.name))
            mr_image_path = mr_nrrd.name
        except Exception as exc:
            for roi_name, mask_path in roi_masks:
                _series_disposition(
                    roi_name,
                    mask_path,
                    reason_code="failed_source_read",
                    detail=f"MR image {nifti_path} could not be prepared: {exc}",
                    mask_identity=_MR_UNAVAILABLE,
                )
            continue

        for roi_name, mask_path in roi_masks:
            try:
                mask_identity = file_sha256(mask_path)
                mask_img = sitk.ReadImage(str(mask_path))
                mask_arr = sitk.GetArrayFromImage(mask_img)
            except Exception as exc:
                _series_disposition(
                    roi_name,
                    mask_path,
                    reason_code="failed_source_read",
                    detail=f"MR mask {mask_path.name} is unreadable: {exc}",
                )
                continue

            if not _mr_geometry_matches(mr_img, mask_img):
                _series_disposition(
                    roi_name,
                    mask_path,
                    reason_code="failed_source_read",
                    detail=(
                        f"MR mask {mask_path.name} geometry "
                        f"{tuple(mask_img.GetSize())} does not match its image "
                        f"{tuple(mr_img.GetSize())}"
                    ),
                    failure_kind="source_geometry",
                    mask_identity=mask_identity,
                )
                continue

            mask_bool = mask_arr > 0
            voxel_count = int(mask_bool.sum())
            if voxel_count == 0:
                _series_disposition(
                    roi_name,
                    mask_path,
                    reason_code="not_computed_valid_empty_scope",
                    detail=f"MR mask {mask_path.name} is valid but empty",
                    failure_kind="degenerate_mask",
                    mask_identity=mask_identity,
                )
                continue
            if voxel_count < min_voxels_limit:
                logger.info(
                    "Skipping MR mask %s: %d voxels below minimum %d",
                    mask_path,
                    voxel_count,
                    min_voxels_limit,
                )
                _series_disposition(
                    roi_name,
                    mask_path,
                    reason_code="ROI_MASK_BELOW_MIN_VOXELS",
                    detail=(
                        f"mask contains {voxel_count} voxels; configured minimum is "
                        f"{min_voxels_limit}"
                    ),
                    status="below_minimum_voxels",
                    failure_kind="degenerate_mask",
                    mask_identity=mask_identity,
                )
                continue

            try:
                mask_nrrd = tempfile.NamedTemporaryFile(
                    suffix=".nrrd", delete=False, prefix=f"mr_mask_{roi_name}_"
                )
                mask_nrrd.close()
                sitk.WriteImage(mask_img, mask_nrrd.name)
                temp_files.append(Path(mask_nrrd.name))
            except Exception as exc:
                _series_disposition(
                    roi_name,
                    mask_path,
                    reason_code="failed_radiomics_extraction",
                    detail=f"MR mask {mask_path.name} could not be prepared: {exc}",
                    failure_kind="extraction_error",
                    mask_identity=mask_identity,
                )
                continue

            tasks.append({
                'image_path': mr_image_path,
                'mask_path': mask_nrrd.name,
                'roi_name': roi_name,
                'params_file': str(params_file) if params_file else None,
                'parameter_provenance_arm': mr_parameter_arm,
                'cleanup': False,
                'required': _roi_is_required(roi_name),
                'metadata': _mr_row_identity(
                    course_dir,
                    roi_name=roi_name,
                    series_uid=series_uid,
                    study_uid=study_uid,
                    dicom_dir=dicom_dir,
                    nifti_path=nifti_path,
                    image_digest=image_digest,
                    mask_path=str(mask_path),
                    mask_identity=mask_identity,
                    parameter_arm=mr_parameter_arm,
                    configured_parameter_hash_value=mr_configured_parameter_hash,
                    run_identifier=mr_run_identifier,
                    code_revision=mr_code_revision,
                ),
            })

    def _cleanup_temp_files() -> None:
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except OSError as exc:
                logger.debug("Failed to clean MR temporary file %s: %s", temp_file, exc)

    if not tasks:
        logger.debug("No eligible MR radiomics tasks for %s", course_dir)
        _cleanup_temp_files()
        _invalidate_radiomics_outputs(out_path)
        # A course that measures nothing still publishes a ledger, and that ledger
        # is evidence about specific source bytes. Re-read the live inventory
        # before publishing it: a series or mask that arrived, changed or vanished
        # while this course was screened would otherwise be described by a
        # disposition that no longer matches anything on disk.
        screening_drift = _mr_source_drift(source_inventory, _mr_source_inventory(mr_root))
        if screening_drift is not None:
            _remove_artifact_strict(
                mr_root / "radiomics_mr_checkpoint.parquet",
                context="discarding the MR checkpoint for a source changed during screening",
            )
            _withdraw_mr_ledger(
                course_dir,
                context="withdrawing the MR ledger for a source changed during screening",
            )
            raise RadiomicsCourseExtractionError(
                "MR radiomics sources changed while the course was being screened, "
                f"so its dispositions may not be published: {screening_drift}"
            )
        ledger_rows = _mr_ledger_rows(dispositions)
        missing_reason = (
            "not_applicable_modality" if not series_dirs else "failed_source_segmentation"
        )
        _write_conda_roi_ledger(
            course_dir,
            [],
            ledger_rows,
            extracted=False,
            expected_names=mr_expected_names,
            missing_reason=missing_reason,
            missing_reason_for=_mr_missing_reason(missing_reason),
            identity_fields=_MR_LEDGER_IDENTITY_FIELDS,
            derive_course_states=True,
            modality="MR",
        )
        unmet = _mr_unmet_requirements(mr_requirements, ledger_rows)
        if unmet:
            raise RadiomicsCourseExtractionError(
                "Required MR ROI(s) were not measured: " + ", ".join(unmet)
            )
        return None

    all_tasks = tasks + [
        _mr_disposition_task(disposition, params_file) for disposition in dispositions
    ]
    logger.info(
        "Processing %d MR radiomics tasks (%d measurements, %d dispositions) for %s",
        len(all_tasks),
        len(tasks),
        len(dispositions),
        course_dir.name,
    )

    # Determine worker count - same logic as CT radiomics
    max_workers = None
    env_workers = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)
    if env_workers > 0:
        max_workers = env_workers
    elif hasattr(config, 'effective_workers') and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
        except Exception:
            pass
    if max_workers is None:
        # Conservative default when no budget is set
        max_workers = min(4, len(all_tasks))

    logger.info("MR radiomics using %d workers", max_workers)

    # Enable checkpointing for resumable extraction
    checkpoint_path = mr_root / "radiomics_mr_checkpoint.parquet"
    _mr_reject_stale_checkpoint(checkpoint_path, all_tasks, out_path)

    def _account_rejected_publication(context: str) -> None:
        """Withdraw a rejected MR publication and account every attempted ROI.

        Removing the workbook says nothing about the ROIs this course attempted,
        and it leaves any earlier successful ledger standing for the cohort
        denominator to keep counting. Each independently recorded nonmeasurement
        keeps its own reason; everything that was being measured is accounted as
        the technical failure it turned out to be.
        """
        _invalidate_radiomics_outputs(out_path)
        _remove_artifact_strict(checkpoint_path, context=context)
        _write_conda_roi_ledger(
            course_dir,
            all_tasks,
            [],
            extracted=False,
            expected_names=mr_expected_names,
            missing_reason="failed_radiomics_extraction",
            identity_fields=_MR_LEDGER_IDENTITY_FIELDS,
            derive_course_states=True,
            modality="MR",
        )

    try:
        try:
            result = process_radiomics_batch(
                all_tasks,
                out_path,
                sequential=False,
                max_workers=max_workers,
                checkpoint_path=checkpoint_path,
                enable_heartbeat=True,
                env_probe_timeout=getattr(config, "radiomics_env_probe_timeout", None),
            )
        except Exception as exc:
            # Typed and unexpected extraction failures both owe a course ledger.
            # Lower-level publication withdrawal alone does not account the ROIs
            # that were attempted.
            _account_rejected_publication(
                "discarding the MR checkpoint after an unaccounted extraction failure"
            )
            if isinstance(exc, RadiomicsCourseExtractionError):
                # Keep the original typed error and traceback after accounting it.
                raise
            raise RadiomicsCourseExtractionError(
                f"MR radiomics extraction failed for {course_dir}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        source_drift = _mr_source_drift(source_inventory, _mr_source_inventory(mr_root))
        if source_drift is not None:
            _invalidate_radiomics_outputs(out_path)
            _remove_artifact_strict(
                checkpoint_path,
                context="discarding the MR checkpoint for a source changed during extraction",
            )
            _withdraw_mr_ledger(
                course_dir,
                context="withdrawing the MR ledger for a source changed during extraction",
            )
            raise RadiomicsCourseExtractionError(
                "MR radiomics sources changed while the course was being measured, "
                f"so nothing measured from them may be published: {source_drift}"
            )
        try:
            result_rows = _mr_published_rows(result)
        except RadiomicsCourseExtractionError:
            _account_rejected_publication(
                "discarding MR checkpoint for an unreadable publication"
            )
            raise
        drift = _mr_publication_drift(result_rows, all_tasks)
        if drift is not None:
            _account_rejected_publication(
                "discarding MR checkpoint with drifted sources"
            )
            raise RadiomicsCourseExtractionError(
                f"MR radiomics publication is not bound to its current sources: {drift}"
            )
        _write_conda_roi_ledger(
            course_dir,
            all_tasks,
            result_rows,
            extracted=result is not None,
            expected_names=mr_expected_names,
            missing_reason="failed_source_segmentation",
            identity_fields=_MR_LEDGER_IDENTITY_FIELDS,
            derive_course_states=True,
            modality="MR",
        )
        unmet = _mr_unmet_requirements(mr_requirements, result_rows)
        if unmet:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                "Required MR ROI(s) were not measured: " + ", ".join(unmet)
            )
        if result is None and mr_required:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but produced no workbook for {course_dir}"
            )
        return result
    finally:
        _cleanup_temp_files()
