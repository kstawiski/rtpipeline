from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import os
import tempfile
import re
import shutil
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import Tag
import SimpleITK as sitk
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from .config import DEFAULT_MAX_TOTAL_DOSE_GY, PipelineConfig
from . import nifti_provenance
from .course_contract import (
    COURSE_CONTRACT_VERSION,
    DOSE_GRID_SEMANTICS,
    UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS,
    DOSE_RESPONSE_FIELD,
    CourseContractError,
    _ct_provenance,
    build_dvh_decision,
    load_course_contract,
    relative_contract_path,
)
from .ct import CTInstance, index_ct_series, pick_primary_series, copy_ct_series, _clear_dir
from .dicom_copy import DicomCopyConfig, DicomCopyManager, get_copy_manager, reset_copy_manager
from .inventory import materialize_patient_series_from_inventory
from .layout import CourseDirs, build_course_dirs, course_dir_name
from .metadata import LinkedSet, group_by_course, link_rt_sets, parse_date
from .modality_classifier import classify_series
from .rt_details import extract_rt, StructInfo, has_target_volumes, target_volume_names
from .utils import (
    ensure_dir,
    run_tasks_with_adaptive_workers,
    read_dicom,
    get,
    _scoped_walk,
    parallel_map_files,
    DEFAULT_INDEX_WORKERS,
    FOLLOW_INPUT_SYMLINKS_ENV,
    follow_input_symlinks,
)
from .segmentation import (
    _collect_series_metadata,
    _derive_nifti_name,
    _ensure_ct_nifti,
    _strip_nifti_base,
    run_dcm2niix,
)

logger = logging.getLogger(__name__)


@dataclass
class CourseOutput:
    patient_id: str
    course_key: str
    course_id: str
    course_start: Optional[str]
    dirs: CourseDirs
    rp_path: Path
    rd_path: Path
    rs_path: Optional[Path]
    primary_nifti: Optional[Path]
    related_dicom: List[Path]
    total_prescription_gy: float | None
    plan_sop_uid: str | None = None
    dose_sop_uid: str | None = None
    source_plan_uids: list[str] = field(default_factory=list)
    source_dose_uids: list[str] = field(default_factory=list)
    delivered_dose_gy: float | None = None
    delivery_status: str = "no_records_at_all"
    delivery_method: str | None = None
    delivered_record_count: int = 0
    delivered_fraction_count: int = 0
    planned_fraction_count: int | None = None
    delivery_plan_details: list[dict[str, object]] = field(default_factory=list)
    delivery_warnings: list[str] = field(default_factory=list)
    unresolved_record_plan_uids: list[str] = field(default_factory=list)
    planning_ct_status: str = "unknown"
    planning_ct_referenced_series_uids: list[str] = field(default_factory=list)
    planning_ct_series_uid: str | None = None
    selected_plan_contract: list[dict[str, object]] = field(default_factory=list)
    selected_dose_contract: list[dict[str, object]] = field(default_factory=list)
    per_plan_delivery_contract: list[dict[str, object]] = field(default_factory=list)
    authoritative_rtstruct_uid: str | None = None
    dose_classification: dict[str, object] = field(default_factory=dict)
    dose_qc: dict[str, object] = field(default_factory=dict)
    course_contract: dict[str, object] = field(default_factory=dict)


class OrganizeDiscoveryError(RuntimeError):
    """Raised when organize discovers no supported DICOM objects."""


_VALID_DELIVERY_STATUSES = frozenset(
    {
        "fully_delivered",
        "partially_delivered",
        "delivered_but_records_absent",
        "no_records_at_all",
    }
)


def _has_nonmissing_adjudication(value: object) -> bool:
    """Return whether a serialized adjudication contains an actual decision."""
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return bool(normalized) and normalized.casefold() != "unknown"


def _plan_checkpoint_is_complete(data: dict[str, object], has_discovered_plan: bool) -> bool:
    """Validate plan-bearing checkpoint decisions before applying hydration defaults."""
    if not has_discovered_plan:
        return True
    delivery_status = data.get("delivery_status")
    if not isinstance(delivery_status, str) or delivery_status not in _VALID_DELIVERY_STATUSES:
        return False
    return _has_nonmissing_adjudication(data.get("planning_ct_status"))


def _sop_instance_uid(path: Path) -> str:
    """Return a file's SOPInstanceUID, or "" when it is absent or unreadable."""
    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[Tag(0x0008, 0x0018)],
        )
    except Exception:
        return ""
    return str(getattr(dataset, "SOPInstanceUID", "") or "")


def _safe_copy(
    src: Path,
    dst: Path,
    copy_manager: Optional[DicomCopyManager] = None,
) -> None:
    """Copy DICOM file to destination with optional deduplication."""
    if copy_manager is not None:
        actual, _copied = copy_manager.copy_dicom(src, dst, skip_if_exists=False)
        # SOP dedup may answer from a foreign path without writing dst, and an
        # earlier run may have left a different artifact there. Either way the
        # course contract names dst, so dst must end up being this source.
        source = Path(actual) if Path(actual).is_file() else src
        if _sop_instance_uid(dst) != _sop_instance_uid(source):
            ensure_dir(dst.parent)
            if dst.exists():
                dst.unlink()
            try:
                os.link(source, dst)
            except OSError:
                shutil.copy2(source, dst)
        if not dst.exists():
            raise OSError(f"required DICOM artifact was not materialised at {dst}")
    else:
        ensure_dir(dst.parent)
        if dst.exists() and dst.is_file() and not os.path.samefile(src, dst):
            dst.unlink()
        shutil.copy2(src, dst)


def _copy_into(
    src: Path,
    dst_dir: Path,
    prefix: Optional[str] = None,
    copy_manager: Optional[DicomCopyManager] = None,
) -> Path:
    """Copy src into dst_dir, preserving name and avoiding clashes."""
    if copy_manager is not None:
        dest, _ = copy_manager.copy_dicom_into(src, dst_dir, prefix)
        dest = Path(dest)
        # SOP dedup answers with an existing copy, which for a per-patient object
        # such as an RTRECORD may live under a different course. A course contract
        # may only reference artifacts inside its own course directory, so the
        # object is materialised here instead of being cited across courses.
        if dest.parent.resolve() != dst_dir.resolve():
            ensure_dir(dst_dir)
            source = dest if dest.is_file() else src
            name = f"{prefix}_{src.name}" if prefix else src.name
            target = dst_dir / name
            if target.is_file() and _sop_instance_uid(target) == _sop_instance_uid(source):
                return target
            stem, suffix = target.stem, target.suffix
            counter = 1
            while target.exists():
                target = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            return target
        return dest

    # Fallback to original behavior
    ensure_dir(dst_dir)
    name = src.name
    if prefix:
        name = f"{prefix}_{name}"
    dest = dst_dir / name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            dest = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.copy2(src, dest)
    return dest


def _normalize_dicom_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = " ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "n/a", "na", "null"}:
        return ""
    if not any(ch.isalpha() for ch in text) and text.replace(".", "").isdigit() and float(text or 0) == 0.0:
        return ""
    return text


def infer_plan_rx_gy(ds_plan: Dataset) -> float | None:
    """Infer total prescription dose in Gy from an RTPLAN dataset."""
    dose_seq = getattr(ds_plan, "DoseReferenceSequence", None) or []
    for dr in dose_seq:
        dtype = str(getattr(dr, "DoseReferenceType", ""))
        dtype_norm = dtype.upper()
        if dtype_norm and dtype_norm not in {"TARGET", "TREATED_VOLUME", "PLANNED_TARGET_VOLUME"}:
            continue
        target = getattr(dr, "TargetPrescriptionDose", None)
        if target not in (None, ""):
            try:
                return float(target)
            except Exception:
                pass

    # Fallback to alternative target-dose fields used by some planners.
    # OAR constraints are deliberately excluded: they are dose limits, not Rx.
    for dr in dose_seq:
        dtype = str(getattr(dr, "DoseReferenceType", ""))
        dtype_norm = dtype.upper()
        if dtype_norm and dtype_norm not in {"TARGET", "TREATED_VOLUME", "PLANNED_TARGET_VOLUME"}:
            continue
        alt_fields = [getattr(dr, "DeliveryMaximumDose", None)]
        for val in alt_fields:
            if val not in (None, ""):
                try:
                    return float(val)
                except Exception:
                    continue

    fg_seq = getattr(ds_plan, "FractionGroupSequence", None) or []
    for fg in fg_seq:
        fractions = getattr(fg, "NumberOfFractionsPlanned", None)
        if fractions in (None, "", 0):
            continue
        try:
            fractions = float(fractions)
        except Exception:
            continue
        per_fraction = 0.0
        has_dose = False
        if hasattr(fg, "ReferencedDoseReferenceSequence") and fg.ReferencedDoseReferenceSequence:
            for ref in fg.ReferencedDoseReferenceSequence:
                dose_val = getattr(ref, "TargetPrescriptionDose", None)
                if dose_val not in (None, ""):
                    try:
                        per_fraction = float(dose_val)
                        has_dose = True
                        break
                    except Exception:
                        pass
        if not has_dose and hasattr(fg, "ReferencedBeamSequence") and fg.ReferencedBeamSequence:
            beam_doses = []
            for ref in fg.ReferencedBeamSequence:
                dose_val = getattr(ref, "BeamDose", None)
                if dose_val not in (None, ""):
                    try:
                        beam_doses.append(float(dose_val))
                    except Exception:
                        continue
            if beam_doses:
                per_fraction = sum(beam_doses)
                has_dose = True
        if has_dose and per_fraction:
            return float(per_fraction * fractions)

    return None


def _infer_rx_from_plan_paths(plan_paths: List[Path], *, sum_all: bool = False) -> float | None:
    values: list[float] = []
    for plan_path in plan_paths:
        try:
            ds_plan = pydicom.dcmread(str(plan_path), stop_before_pixels=True)
        except Exception:
            continue
        value = infer_plan_rx_gy(ds_plan)
        if value is not None and value > 0:
            values.append(float(value))
    if not values:
        return None
    if sum_all:
        return float(sum(values))
    return float(values[0])


def _summarize_reconstruction(ds: Dataset) -> str:
    fields = [
        (None, getattr(ds, 'ReconstructionAlgorithm', None)),
        (None, getattr(ds, 'ReconstructionMethod', None)),
        ('Iterative', getattr(ds, 'IterativeReconstructionMethod', None)),
        ('Technique', getattr(ds, 'AlgorithmType', None)),
        ('KernelGroup', getattr(ds, 'ConvolutionKernelGroup', None)),
        ('Kernel', getattr(ds, 'ConvolutionKernel', None)),
        ('Filter', getattr(ds, 'FilterType', None)),
    ]
    parts: list[str] = []
    seen: set[tuple[Optional[str], str]] = set()
    for label, raw in fields:
        text = _normalize_dicom_text(raw)
        if not text:
            continue
        key = (label, text.lower())
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{label}: {text}" if label else text)
    return " | ".join(parts)


def _hydrate_existing_course(
    patient_id: str,
    course_key: str,
    course_dir: Path,
    meta_hint: Optional[dict[str, object]] = None,
) -> Optional[CourseOutput]:
    course_dirs = build_course_dirs(course_dir)
    meta_dir = course_dirs.metadata
    meta_path = meta_dir / "case_metadata.json"
    data: dict[str, object] = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    # Don't hydrate empty directories created by ensure() — require at least
    # metadata JSON or an RS.dcm to consider this course "already processed".
    rs_candidate = course_dirs.dicom_rtstruct if hasattr(course_dirs, 'dicom_rtstruct') else None
    has_rs = (course_dir / "RS.dcm").exists() or (
        rs_candidate is not None and rs_candidate.exists() and any(rs_candidate.iterdir())
    )
    has_ct = course_dirs.dicom_ct.exists() and any(course_dirs.dicom_ct.iterdir())
    if not data and not has_rs and not has_ct:
        return None

    try:
        contract = load_course_contract(course_dir)
    except CourseContractError as exc:
        logger.info(
            "Not hydrating %s/%s: authoritative course contract is missing or stale: %s",
            patient_id,
            course_key,
            exc,
        )
        return None

    course_id = str(contract.data.get("course_id") or course_dir.name)
    course_start = data.get("course_start_date") or data.get("course_start") or (meta_hint.get("start_iso") if meta_hint else None)
    if isinstance(course_start, str) and not course_start:
        course_start = None

    # Contract paths are authoritative. A missing artifact was rejected above.
    absent = course_dir / "metadata" / ".contract-artifact-absent"
    rp_path = contract.plan_artifact_path or absent
    rd_path = contract.dose_grid_path or absent
    rs_path = contract.authoritative_rtstruct_path or absent

    # C4 fix: validate all paths loaded from JSON metadata to prevent path traversal
    primary_nifti: Optional[Path] = contract.planning_ct_nifti

    # A course whose CT was copied but never converted is INCOMPLETE, not done.
    # Hydrating it would mark it processed and leave it without a NIfTI for the
    # rest of the run, so segmentation and radiomics would have nothing to read
    # and the gap would never be repaired by a later resume. Returning None sends
    # it back through conversion instead, which is what resume is for.
    #
    # This is not hypothetical: a dcm2niix race left one 154-patient cohort with
    # 352,707 copied DICOM instances and only 441 NIfTI files. Every affected
    # course would have been skipped on resume.
    if has_ct and primary_nifti is None:
        logger.info(
            "Not hydrating %s/%s: CT is present but no NIfTI was produced; "
            "the course will be reprocessed.",
            patient_id,
            course_key,
        )
        return None

    related_files: List[Path] = []
    related_list = data.get("dicom_related_files") if data else None
    if isinstance(related_list, list):
        from .utils import validate_path as _validate_path
        for entry in related_list:
            if not isinstance(entry, str):
                continue
            cand = Path(entry)
            try:
                cand = _validate_path(cand, course_dir, allow_absolute=True)
                if cand.exists():
                    related_files.append(cand)
            except ValueError:
                logger.warning("Path traversal blocked for related file: %s", entry)
    if not related_files and course_dirs.dicom_related.exists():
        related_files = [p for p in course_dirs.dicom_related.rglob("*.dcm") if p.is_file()]

    total_rx_val = contract.prescribed_dose_gy
    delivered_value = contract.delivered_dose_gy
    delivery_contract = contract.delivery
    delivery_status = str(delivery_contract.get("status") or "")
    delivery_method = delivery_contract.get("method")
    delivery_plan_details = list(delivery_contract.get("per_plan") or [])
    selected_delivery = [
        item for item in delivery_plan_details if item.get("selected_for_dose_grid") is True
    ]
    delivered_record_count = sum(
        int(item.get("delivered_record_count") or 0) for item in selected_delivery
    )
    delivered_fraction_count = sum(
        int(item.get("delivered_fraction_count") or 0) for item in selected_delivery
    )
    planned_total = sum(
        int(item.get("planned_fraction_count") or 0) for item in selected_delivery
    )
    planned_fraction_count = planned_total or None
    unresolved_record_plan_uids = list(
        delivery_contract.get("unresolved_record_plan_uids") or []
    )
    delivery_warnings = list(delivery_contract.get("warnings") or [])
    planning_ct_status = str(contract.planning_ct.get("status") or "")
    planning_ct_referenced_series_uids = list(
        contract.planning_ct.get("referenced_series_uids") or []
    )
    planning_ct_series_uid = str(contract.planning_ct.get("series_instance_uid") or "") or None
    contract_selected_plans: list[dict[str, object]] = []
    for index, item in enumerate(contract.selected_plans):
        hydrated_item = dict(item)
        hydrated_item["path"] = str(
            contract.resolve_path(item.get("path"), f"selected_plans[{index}].path")
        )
        contract_selected_plans.append(hydrated_item)
    contract_selected_doses: list[dict[str, object]] = []
    for index, item in enumerate(contract.selected_doses):
        hydrated_item = dict(item)
        hydrated_item["path"] = str(
            contract.resolve_path(item.get("path"), f"selected_doses[{index}].path")
        )
        contract_selected_doses.append(hydrated_item)
    contract_per_plan: list[dict[str, object]] = []
    for index, item in enumerate(contract.delivery.get("per_plan") or []):
        hydrated_item = dict(item)
        hydrated_item["plan_path"] = str(
            contract.resolve_path(
                item.get("plan_path"), f"delivery.per_plan[{index}].plan_path"
            )
        )
        contract_per_plan.append(hydrated_item)

    plan_artifact = contract.data.get("plan_artifact") or {}
    dose_grid = contract.data.get("dose_grid") or {}
    plan_uid = str(plan_artifact.get("sop_instance_uid") or "") or None
    dose_uid = str(dose_grid.get("sop_instance_uid") or "") or None
    source_plan_uids = {
        str(item.get("sop_instance_uid") or "") for item in contract.selected_plans
    }
    source_dose_uids = {
        str(item.get("sop_instance_uid") or "") for item in contract.selected_doses
    }

    return CourseOutput(
        patient_id=patient_id,
        course_key=course_key,
        course_id=course_id,
        course_start=course_start if isinstance(course_start, str) else None,
        dirs=course_dirs,
        rp_path=rp_path,
        rd_path=rd_path,
        rs_path=rs_path if rs_path.exists() else None,
        primary_nifti=primary_nifti,
        related_dicom=related_files,
        total_prescription_gy=total_rx_val,
        plan_sop_uid=plan_uid,
        dose_sop_uid=dose_uid,
        source_plan_uids=sorted(source_plan_uids) if source_plan_uids else [],
        source_dose_uids=sorted(source_dose_uids) if source_dose_uids else [],
        delivered_dose_gy=delivered_value,
        delivery_status=delivery_status,
        delivery_method=delivery_method if isinstance(delivery_method, str) else None,
        delivered_record_count=delivered_record_count,
        delivered_fraction_count=delivered_fraction_count,
        planned_fraction_count=planned_fraction_count,
        delivery_plan_details=contract_per_plan,
        delivery_warnings=[str(item) for item in delivery_warnings],
        unresolved_record_plan_uids=unresolved_record_plan_uids,
        planning_ct_status=planning_ct_status,
        planning_ct_referenced_series_uids=planning_ct_referenced_series_uids,
        planning_ct_series_uid=planning_ct_series_uid,
        selected_plan_contract=contract_selected_plans,
        selected_dose_contract=contract_selected_doses,
        per_plan_delivery_contract=contract_per_plan,
        authoritative_rtstruct_uid=(
            str((contract.data.get("authoritative_rtstruct") or {}).get("sop_instance_uid") or "")
            or None
        ),
        dose_classification=dict(contract.data.get("dose_classification") or {}),
        dose_qc=dict(contract.dose_qc),
        course_contract=dict(contract.data),
    )


def _sanitize_name(text: str, fallback: str = "item") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned[:80]


def _looks_like_patient_series_layout(dicom_root: Path) -> bool:
    """Detect TCIA-style PatientID/SeriesInstanceUID/*.dcm input trees.

    NOTE: intentionally NOT cohort-scoped. It must inspect the IMMEDIATE children
    of dicom_root so that a nested multi-center layout (CENTER/PATIENT/SERIES) is
    correctly rejected (the CENTER dirs do not look like a patient/series tree). A
    scoped variant that resolved to nested patient dirs would falsely detect a
    flat TCIA tree and skip RT scanning. The full-tree iterdir here is cheap and
    short-circuits on the first non-conforming patient dir.
    """
    try:
        patient_dirs = [path for path in dicom_root.iterdir() if path.is_dir()]
    except OSError:
        return False
    if not patient_dirs:
        return False
    for patient_dir in patient_dirs:
        try:
            entries = list(patient_dir.iterdir())
        except OSError:
            return False
        series_dirs = [path for path in entries if path.is_dir()]
        if not series_dirs or any(path.is_file() for path in entries):
            return False
        for series_dir in series_dirs:
            try:
                names = os.listdir(series_dir)
            except OSError:
                return False
            if not any(name.lower().endswith(".dcm") for name in names):
                return False
    return True


class CourseTargetQCError(RuntimeError):
    """Raised when a plan-and-dose course has no target volume."""


class CTOnlyCohortError(RuntimeError):
    """Raised when CT-only input is supplied without explicit configuration."""


def validate_course_target_qc(
    patient_id: str,
    course_key: str,
    plan_paths: List[Path],
    dose_paths: List[Path],
    struct_path: Optional[Path],
) -> List[str]:
    """Require GTV, CTV, or PTV for every course carrying plan and dose."""
    if not plan_paths or not dose_paths:
        return []
    if struct_path is None:
        raise CourseTargetQCError(
            f"Course target QC failed for patient {patient_id}, course {course_key}: "
            "plan and dose are present but the authoritative structure set is unresolved"
        )
    try:
        ds = pydicom.dcmread(str(struct_path), stop_before_pixels=True, force=True)
    except Exception as exc:
        raise CourseTargetQCError(
            f"Course target QC failed for patient {patient_id}, course {course_key}: "
            f"could not read authoritative structure set {struct_path}: {exc}"
        ) from exc
    roi_names = [
        str(getattr(roi, "ROIName", "") or "")
        for roi in getattr(ds, "StructureSetROISequence", []) or []
    ]
    targets = target_volume_names(roi_names)
    if not targets:
        raise CourseTargetQCError(
            f"Course target QC failed for patient {patient_id}, course {course_key}: "
            f"{len(plan_paths)} plan(s) and {len(dose_paths)} dose(s) but zero target volumes "
            f"in RTSTRUCT {getattr(ds, 'SOPInstanceUID', '<missing>')}"
        )
    return targets


def _authoritative_structure_source(items: Iterable[LinkedSet]) -> Path | None:
    """Resolve duplicate RTSTRUCT paths by SOP UID and reject distinct sets."""
    paths_by_identity: Dict[str, set[Path]] = defaultdict(set)
    for item in items:
        struct = item.struct
        if struct is None:
            continue
        uid = str(struct.sop_instance_uid or "").strip()
        identity = f"SOP:{uid}" if uid else f"PATH:{struct.path.resolve()}"
        paths_by_identity[identity].add(struct.path)
    if not paths_by_identity:
        return None
    if len(paths_by_identity) != 1:
        identities = ", ".join(sorted(paths_by_identity))
        raise CourseTargetQCError(
            "course resolves to multiple distinct RTSTRUCT objects "
            f"({identities}); refusing reference-free CT fallback"
        )
    candidates = next(iter(paths_by_identity.values()))
    return min(candidates, key=lambda path: str(path))


def _classify_organize_ct_series(
    series: List[CTInstance],
    *,
    is_planning_ct: bool,
    allow_ct_only: bool = False,
) -> tuple[bool, str, Optional[str]]:
    """Apply the shared series classifier before organize constructs a course."""
    if not series:
        return False, "exclude", "empty_ct_series"
    datasets: List[Dataset] = []
    for instance in series:
        try:
            datasets.append(
                pydicom.dcmread(str(instance.path), stop_before_pixels=True, force=True)
            )
        except Exception as exc:
            logger.warning("Could not read CT header for organize classification %s: %s", instance.path, exc)
    if not datasets:
        return False, "exclude", "unreadable_ct_series"
    first = datasets[0]
    image_types: List[str] = []
    thicknesses: List[float] = []
    for ds in datasets:
        raw_image_type = getattr(ds, "ImageType", []) or []
        values = raw_image_type if isinstance(raw_image_type, (list, tuple, MultiValue)) else [raw_image_type]
        for value in values:
            text = str(value)
            if text and text not in image_types:
                image_types.append(text)
        try:
            thicknesses.append(float(getattr(ds, "SliceThickness", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    meta = {
        "modality": "CT",
        "series_description": str(getattr(first, "SeriesDescription", "") or ""),
        "manufacturer": str(getattr(first, "Manufacturer", "") or ""),
        "manufacturer_model": str(getattr(first, "ManufacturerModelName", "") or ""),
        "image_type": image_types,
        "n_instances": len(series),
        "rows": int(getattr(first, "Rows", 0) or 0),
        "columns": int(getattr(first, "Columns", 0) or 0),
        "slice_thickness": max(thicknesses) if thicknesses else None,
        "is_planning_ct": is_planning_ct,
        "rt_linked": is_planning_ct,
        "rtstruct_linked": is_planning_ct,
    }
    image_class, exclusion_reason = classify_series(meta)
    if image_class == "exclude":
        return False, image_class, exclusion_reason
    if not is_planning_ct and not allow_ct_only:
        return False, image_class, "non_planning_ct_without_rt_reference"
    return True, image_class, None


# =============================================================================
# DOSE CLASSIFICATION SYSTEM
# =============================================================================
# Prevents incorrect dose summation by:
# 1. Preferring TPS-provided PLAN_SUM objects.
# 2. Separating courses by explicit DICOM reference chains.
# 3. De-duplicating equivalent prescription and fraction signatures.
# 4. Using RTRECORD support to distinguish revisions and delivered phases.
# 5. Applying configurable plausibility safeguards.
# Free-text plan labels are intentionally unavailable to this classifier.
# =============================================================================

@dataclass
class DoseClassification:
    """Result of reference- and delivery-based dose classification."""
    classification: str
    selected_doses: List[Path]
    selected_plans: List[Path]
    excluded_doses: List[Path]  # doses excluded (e.g., replans)
    should_sum: bool
    warnings: List[str]
    reason: str


def _extract_dose_metadata(dose_path: Path) -> dict:
    """Extract relevant metadata from a dose file for classification."""
    try:
        ds = pydicom.dcmread(str(dose_path), stop_before_pixels=True)

        # Get referenced plan UIDs
        ref_plan_uids = []
        if hasattr(ds, "ReferencedRTPlanSequence") and ds.ReferencedRTPlanSequence:
            for ref in ds.ReferencedRTPlanSequence:
                uid = getattr(ref, "ReferencedSOPInstanceUID", None)
                if uid:
                    ref_plan_uids.append(str(uid))

        # Extract geometry for spatial overlap analysis
        origin = list(map(float, getattr(ds, "ImagePositionPatient", [0, 0, 0])))
        pixel_spacing = list(map(float, getattr(ds, "PixelSpacing", [1.0, 1.0])))
        rows = int(getattr(ds, "Rows", 1))
        cols = int(getattr(ds, "Columns", 1))
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        offsets = getattr(ds, "GridFrameOffsetVector", None)

        # Calculate z-extent
        if offsets is not None and len(offsets) >= 2:
            z_offsets = [float(o) for o in offsets]
            z_min = origin[2] + min(z_offsets)
            z_max = origin[2] + max(z_offsets)
        else:
            z_min = origin[2]
            z_max = origin[2] + frames - 1  # Assume 1mm spacing if no offsets

        # Compute bounding box: (x_min, y_min, z_min, x_max, y_max, z_max)
        x_min = origin[0]
        y_min = origin[1]
        x_max = origin[0] + (cols - 1) * pixel_spacing[1]
        y_max = origin[1] + (rows - 1) * pixel_spacing[0]

        bbox = (x_min, y_min, z_min, x_max, y_max, z_max)

        return {
            "path": dose_path,
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "summation_type": str(getattr(ds, "DoseSummationType", "PLAN")).upper(),
            "frame_of_reference": str(getattr(ds, "FrameOfReferenceUID", "")),
            "referenced_plan_uids": ref_plan_uids,
            "creation_date": str(getattr(ds, "InstanceCreationDate", "")),
            "creation_time": str(getattr(ds, "InstanceCreationTime", "")),
            "bbox": bbox,
        }
    except Exception as e:
        logger.warning(f"Failed to extract dose metadata from {dose_path}: {e}")
        return {
            "path": dose_path,
            "sop_uid": "",
            "summation_type": "PLAN",
            "frame_of_reference": "",
            "referenced_plan_uids": [],
            "creation_date": "",
            "creation_time": "",
            "bbox": None,
        }


def _extract_plan_metadata(plan_path: Path) -> dict:
    """Extract relevant metadata from a plan file for classification."""
    try:
        ds = pydicom.dcmread(str(plan_path), stop_before_pixels=True)

        # Extract prescription doses. Free-text plan labels and descriptions are
        # deliberately not loaded into the classifier.
        prescriptions = []
        if hasattr(ds, "DoseReferenceSequence") and ds.DoseReferenceSequence:
            for dr in ds.DoseReferenceSequence:
                rx_dose = getattr(dr, "TargetPrescriptionDose", None)
                if rx_dose is not None:
                    prescriptions.append(
                        {
                            "dose_gy": float(rx_dose),
                            "reference_number": str(getattr(dr, "DoseReferenceNumber", "") or ""),
                            "reference_type": str(getattr(dr, "DoseReferenceType", "") or ""),
                        }
                    )

        plan_date = str(getattr(ds, "RTPlanDate", "") or getattr(ds, "InstanceCreationDate", ""))
        plan_time = str(getattr(ds, "RTPlanTime", "") or getattr(ds, "InstanceCreationTime", ""))

        return {
            "path": plan_path,
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "frame_of_reference": str(getattr(ds, "FrameOfReferenceUID", "")),
            "plan_date": plan_date,
            "plan_time": plan_time,
            "prescriptions": prescriptions,
            "total_rx_gy": sum(p["dose_gy"] for p in prescriptions) if prescriptions else 0.0,
        }
    except Exception as e:
        logger.warning(f"Failed to extract plan metadata from {plan_path}: {e}")
        return {
            "path": plan_path,
            "sop_uid": "",
            "frame_of_reference": "",
            "plan_date": "",
            "plan_time": "",
            "prescriptions": [],
            "total_rx_gy": 0.0,
        }


def _earliest_dated_plan_path(items_sorted: List[LinkedSet], plan_paths: List[Path]) -> Path:
    """Return the earliest-dated plan path for the ITT (first/earliest course plan)
    fallback degrade.

    ``items_sorted`` is sorted ascending by ``plan.plan_date or ""``, so plans MISSING
    a date sort FIRST (an empty string is less than any real date string) -- meaning
    ``plan_paths[0]`` is NOT reliably the chronologically earliest plan whenever any
    plan in the course lacks a date. This scans in that same sorted order and returns
    the first entry that actually HAS a plan_date, which is chronologically earliest
    among the dated plans. If no plan in the course has a date at all, there is no way
    to order them chronologically; this documents that limitation by falling back to
    ``plan_paths[0]`` (the prior behavior) in that edge case.
    """
    for it in items_sorted:
        if it.plan.plan_date:
            return it.plan.path
    return plan_paths[0]


def _plan_paths_for_doses(plan_paths: List[Path], dose_paths: List[Path]) -> List[Path]:
    """Derive the plans actually referenced by ``dose_paths``.

    Mirrors ``dvh.py::_plan_paths_for_doses`` (the safe consumer used for DVH
    computation): used as a fail-closed fallback when a ``DoseClassification``
    legitimately selects no plans (e.g. a replan whose referenced plan UID
    could not be resolved), so callers don't silently substitute every plan
    in the course - which would defeat ITT replan exclusion.
    """
    ref_uids: set[str] = set()
    for dose_path in dose_paths:
        ref_uids.update(_extract_dose_metadata(dose_path).get("referenced_plan_uids", []))
    if not ref_uids:
        return plan_paths[:1] if len(plan_paths) == 1 else []
    selected: List[Path] = []
    for plan_path in plan_paths:
        uid = _extract_plan_metadata(plan_path).get("sop_uid", "")
        if uid and uid in ref_uids:
            selected.append(plan_path)
    return selected


_SESSION_DOSE_SEQUENCE_NAMES = (
    "TreatmentSessionBeamSequence",
    "TreatmentSessionIonBeamSequence",
    "TreatmentSessionApplicationSetupSequence",
)


def _finite_nonnegative(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed >= 0 else None


def _dose_reference_number(item: Dataset) -> str | None:
    value = getattr(item, "ReferencedDoseReferenceNumber", None)
    text = str(value or "").strip()
    return text or None


def _record_session_dose_components(ds: Dataset) -> list[dict[str, object]]:
    """Read additive per-session dose components without walking summary data.

    The value is valid only in the context of the session beam or application
    setup that contains it. A recursive ``iterall`` search would mix beam,
    channel, and cumulative summary values and would also lose the beam identity
    needed to de-duplicate repeated record objects for one session.
    """
    components: list[dict[str, object]] = []
    for sequence_name in _SESSION_DOSE_SEQUENCE_NAMES:
        sequence = getattr(ds, sequence_name, None) or []
        for item_index, session_item in enumerate(sequence):
            if sequence_name == "TreatmentSessionApplicationSetupSequence":
                identity = getattr(session_item, "ReferencedBrachyApplicationSetupNumber", None)
                identity = identity or getattr(session_item, "ApplicationSetupName", None)
                kind = "application_setup"
            else:
                identity = getattr(session_item, "ReferencedBeamNumber", None)
                identity = identity or getattr(session_item, "BeamName", None)
                kind = "beam"
            identity_text = str(identity or f"index:{item_index}").strip()
            dose_sequence = getattr(session_item, "ReferencedCalculatedDoseReferenceSequence", None) or []
            for reference_index, reference_item in enumerate(dose_sequence):
                dose = _finite_nonnegative(
                    getattr(reference_item, "CalculatedDoseReferenceDoseValue", None)
                )
                if dose is None:
                    continue
                components.append(
                    {
                        "component_key": (kind, sequence_name, identity_text),
                        "reference_number": _dose_reference_number(reference_item),
                        "dose_gy": dose,
                        "reference_index": reference_index,
                    }
                )

    # A few exports place the session sequence directly at record level. Keep
    # that representation usable, but do not search nested channel sequences.
    if not components:
        dose_sequence = getattr(ds, "ReferencedCalculatedDoseReferenceSequence", None) or []
        for reference_index, reference_item in enumerate(dose_sequence):
            dose = _finite_nonnegative(
                getattr(reference_item, "CalculatedDoseReferenceDoseValue", None)
            )
            if dose is None:
                continue
            components.append(
                {
                    "component_key": ("record", "ReferencedCalculatedDoseReferenceSequence", "record"),
                    "reference_number": _dose_reference_number(reference_item),
                    "dose_gy": dose,
                    "reference_index": reference_index,
                }
            )
    return components


def _record_cumulative_dose_references(ds: Dataset) -> list[dict[str, object]]:
    """Read cumulative dose-to-reference values from summary records only."""
    values: list[dict[str, object]] = []
    for item in getattr(ds, "TreatmentSummaryCalculatedDoseReferenceSequence", None) or []:
        dose = _finite_nonnegative(getattr(item, "CumulativeDoseToDoseReference", None))
        if dose is not None:
            values.append(
                {
                    "reference_number": _dose_reference_number(item),
                    "dose_gy": dose,
                }
            )
    return values


def _record_dose_reference(ds: Dataset) -> tuple[float | None, str | None]:
    """Read additive per-session calculated dose values from one record.

    This compatibility helper intentionally exposes only the real DICOM
    ``CalculatedDoseReferenceDoseValue`` keyword. Course estimation performs
    plan-reference binding and session de-duplication before using these values.
    """
    components = _record_session_dose_components(ds)
    if not components:
        return None, None
    return float(sum(float(item["dose_gy"]) for item in components)), "calculated_dose_reference"


def _record_delivery_evidence(record_paths: Iterable[Path]) -> Dict[str, dict]:
    """Retain RTRECORD instances and count distinct delivered treatment sessions."""
    evidence: Dict[str, dict] = defaultdict(
        lambda: {"dates": set(), "sessions": set(), "instances": set(), "records": []}
    )
    for path in dict.fromkeys(Path(p) for p in record_paths):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.warning("Could not read RT treatment record %s: %s", path, exc)
            continue
        record_uid = str(getattr(ds, "SOPInstanceUID", "") or path)
        treatment_date = str(getattr(ds, "TreatmentDate", "") or "")
        treatment_time = str(getattr(ds, "TreatmentTime", "") or "")
        fraction_number = (
            getattr(ds, "CurrentFractionNumber", None)
            or getattr(ds, "ReferencedFractionNumber", None)
        )
        fraction_value = int(fraction_number) if str(fraction_number or "").isdigit() else None
        if fraction_value is not None:
            session_key = ("fraction", treatment_date, fraction_value)
        elif treatment_date:
            # RT Beams Treatment Record exports may emit one instance per arc or
            # beam. Records on the same date and plan are one delivered fraction
            # unless a DICOM fraction number distinguishes them.
            session_key = ("date", treatment_date)
        else:
            session_key = ("record", record_uid)
        components = _record_session_dose_components(ds)
        cumulative_references = _record_cumulative_dose_references(ds)
        is_summary_record = bool(
            getattr(ds, "TreatmentSummaryCalculatedDoseReferenceSequence", None)
            or str(getattr(ds, "SOPClassUID", "") or "")
            == "1.2.840.10008.5.1.4.1.1.481.7"
        )
        dose_gy = float(sum(float(item["dose_gy"]) for item in components)) if components else None
        dose_method = "calculated_dose_reference" if dose_gy is not None else None
        for ref in getattr(ds, "ReferencedRTPlanSequence", []) or []:
            plan_uid = str(getattr(ref, "ReferencedSOPInstanceUID", "") or "")
            if not plan_uid:
                continue
            plan_evidence = evidence[plan_uid]
            if record_uid in plan_evidence["instances"]:
                continue
            plan_evidence["instances"].add(record_uid)
            if not is_summary_record:
                plan_evidence["sessions"].add(session_key)
            if treatment_date:
                plan_evidence["dates"].add(treatment_date)
            plan_evidence["records"].append(
                {
                    "path": str(path),
                    "sop_instance_uid": record_uid,
                    "treatment_date": treatment_date,
                    "treatment_time": treatment_time,
                    "fraction_number": fraction_value,
                    "dose_gy": dose_gy,
                    "dose_method": dose_method,
                    "session_key": session_key,
                    "is_summary_record": is_summary_record,
                    "session_components": components,
                    "cumulative_dose_references": cumulative_references,
                }
            )
    return dict(evidence)


def _delivery_reference_audit(
    record_paths: Iterable[Path],
    known_plan_uids: Iterable[str],
    *,
    log_warnings: bool = True,
) -> dict[str, object]:
    """Log and count RTRECORD references that cannot resolve to an exported plan."""
    known = {str(uid) for uid in known_plan_uids if str(uid)}
    unresolved: dict[str, set[str]] = defaultdict(set)
    unresolved_records: set[str] = set()
    for path in dict.fromkeys(Path(p) for p in record_paths):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        record_uid = str(getattr(ds, "SOPInstanceUID", "") or path)
        refs = {
            str(getattr(ref, "ReferencedSOPInstanceUID", "") or "")
            for ref in getattr(ds, "ReferencedRTPlanSequence", []) or []
        }
        for uid in sorted(uid for uid in refs if uid and uid not in known):
            unresolved[uid].add(str(path))
            unresolved_records.add(record_uid)
            if log_warnings:
                logger.warning(
                    "RTRECORD %s references RTPLAN UID %s absent from the export; "
                    "it will not be attributed to another plan",
                    path,
                    uid,
                )
    if unresolved and log_warnings:
        logger.warning(
            "Treatment-record reference audit: %d record(s), %d unresolved RTPLAN UID reference(s); "
            "no plan attribution was made",
            len(unresolved_records),
            len(unresolved),
        )
    elif log_warnings:
        logger.info("Treatment-record reference audit: 0 unresolved RTPLAN UID references")
    return {
        "unresolved_plan_uids": sorted(unresolved),
        "unresolved_record_count": len(unresolved_records),
        "unresolved_reference_count": sum(len(paths) for paths in unresolved.values()),
    }


def _calculate_delivery_summary(
    plan_paths: Iterable[Path],
    record_paths: Iterable[Path],
    *,
    selected_plan_paths: Iterable[Path] | None = None,
    selected_dose_paths: Iterable[Path] | None = None,
    reference_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    """Estimate course dose from DICOM treatment evidence, failing closed.

    A plan's explicit dose values are accepted only after they are bound to the
    plan's target dose reference and de-duplicated by beam or application setup
    within a treatment session. Cumulative summary values are selected as the
    latest cumulative observation and are never summed with one another.
    """
    all_plans = list(dict.fromkeys(Path(p) for p in plan_paths))
    selected = list(
        dict.fromkeys(Path(p) for p in (selected_plan_paths if selected_plan_paths is not None else all_plans))
    )
    records = list(dict.fromkeys(Path(p) for p in record_paths))
    all_meta = {Path(meta["path"]): meta for meta in (_plan_evidence(path) for path in all_plans)}
    plan_meta = {path: all_meta.get(path) or _plan_evidence(path) for path in selected}
    evidence = _record_delivery_evidence(records)
    selected_uids = {str(meta.get("sop_uid") or "") for meta in plan_meta.values()}
    all_plan_uids = {
        str(meta.get("sop_uid") or "")
        for meta in all_meta.values()
        if meta.get("sop_uid")
    }
    unresolved = reference_audit or _delivery_reference_audit(records, all_plan_uids)
    plan_details: list[dict[str, object]] = []
    total_delivered = 0.0
    total_planned_fx = 0
    delivered_records = 0
    delivered_fractions = 0
    estimable = True
    all_fully_delivered = True
    any_matching_records = False
    methods: set[str] = set()
    delivery_warnings: list[str] = []

    def _dose_close(left: float, right: float) -> bool:
        return abs(left - right) <= max(0.1, 0.05 * max(abs(right), 1.0))

    for path in selected:
        meta = plan_meta[path]
        uid = str(meta.get("sop_uid") or "")
        plan_evidence = evidence.get(
            uid,
            {"instances": set(), "sessions": set(), "dates": set(), "records": []},
        )
        records_for_plan = list(plan_evidence.get("records", []))
        record_count = len(plan_evidence.get("instances", set()))
        fraction_count = len(plan_evidence.get("sessions", set()))
        matching = record_count > 0
        any_matching_records = any_matching_records or matching
        delivered_records += record_count
        delivered_fractions += fraction_count
        planned_fx = int(meta.get("fractions_planned") or 0)
        rx = float(meta.get("total_rx_gy") or 0.0)
        if planned_fx > 0:
            total_planned_fx += planned_fx
        target_numbers = {
            str(number).strip()
            for number in (meta.get("target_dose_reference_numbers") or [])
            if str(number).strip()
        }
        expected_per_fraction = rx / planned_fx if rx > 0 and planned_fx > 0 else None
        fallback_dose = (
            rx * min(fraction_count / planned_fx, 1.0)
            if fraction_count and rx > 0 and planned_fx > 0
            else None
        )
        plan_warnings: list[str] = []
        dose: float | None = None
        method = "unknown"

        # A summary record contains a cumulative observation. Select the latest
        # valid target-referenced value instead of adding observations together.
        cumulative_candidates: list[tuple[tuple[object, ...], float]] = []
        if target_numbers:
            for row in records_for_plan:
                for reference in row.get("cumulative_dose_references", []) or []:
                    reference_number = str(reference.get("reference_number") or "").strip()
                    value = _finite_nonnegative(reference.get("dose_gy"))
                    if reference_number in target_numbers and value is not None:
                        cumulative_candidates.append(
                            (
                                (
                                    str(row.get("treatment_date") or ""),
                                    str(row.get("treatment_time") or ""),
                                    int(row.get("fraction_number") or -1),
                                    str(row.get("sop_instance_uid") or ""),
                                ),
                                value,
                            )
                        )
        if cumulative_candidates:
            _order, cumulative_dose = max(cumulative_candidates, key=lambda item: item[0])
            if (
                fallback_dose is not None
                and fraction_count <= planned_fx
                and not _dose_close(cumulative_dose, fallback_dose)
            ):
                reason = (
                    f"cumulative dose {cumulative_dose:.6g} Gy disagrees with fraction-weighted "
                    f"estimate {fallback_dose:.6g} Gy"
                )
                plan_warnings.append(reason)
                logger.warning("RTPLAN %s: %s; using fallback", uid, reason)
            else:
                dose = cumulative_dose
                method = "cumulative_dose_reference"
                methods.add(method)

        # Session values are additive across distinct beam/application components,
        # but repeated records for the same component in one session are duplicates.
        if dose is None and records_for_plan:
            session_rows: dict[object, list[dict[str, object]]] = defaultdict(list)
            for row in records_for_plan:
                if not row.get("is_summary_record"):
                    session_rows[row.get("session_key")].append(row)
            explicit_session_doses: list[float] = []
            explicit_failure: str | None = None
            if target_numbers and session_rows:
                for session_key, rows in session_rows.items():
                    components_by_key: dict[object, float] = {}
                    for row in rows:
                        for component in row.get("session_components", []) or []:
                            reference_number = str(component.get("reference_number") or "").strip()
                            value = _finite_nonnegative(component.get("dose_gy"))
                            if reference_number not in target_numbers or value is None:
                                continue
                            component_key = tuple(component.get("component_key") or ())
                            if component_key in components_by_key:
                                if not _dose_close(components_by_key[component_key], value):
                                    explicit_failure = (
                                        f"conflicting values for component {component_key!r} in session {session_key!r}"
                                    )
                                    break
                                # Same beam/application setup repeated in one
                                # session is a duplicate record, not another dose.
                                continue
                            components_by_key[component_key] = value
                        if explicit_failure:
                            break
                    if explicit_failure:
                        break
                    if not components_by_key:
                        explicit_failure = (
                            f"no target-referenced per-session dose value in session {session_key!r}"
                        )
                        break
                    session_dose = float(sum(components_by_key.values()))
                    if expected_per_fraction is not None and not _dose_close(session_dose, expected_per_fraction):
                        explicit_failure = (
                            f"per-session dose {session_dose:.6g} Gy disagrees with prescribed per-fraction "
                            f"dose {expected_per_fraction:.6g} Gy"
                        )
                        break
                    explicit_session_doses.append(session_dose)
            if explicit_failure:
                plan_warnings.append(explicit_failure)
                logger.warning(
                    "RTPLAN %s: %s; using fraction-weighted prescription when available",
                    uid,
                    explicit_failure,
                )
            elif explicit_session_doses and len(explicit_session_doses) == len(session_rows):
                dose = float(sum(explicit_session_doses))
                method = "calculated_dose_reference"
                methods.add(method)

        if dose is None and fallback_dose is not None:
            dose = float(fallback_dose)
            method = "record_fraction_weighted_prescription"
            methods.add(method)
        elif dose is None:
            estimable = False

        fully = planned_fx > 0 and fraction_count >= planned_fx
        all_fully_delivered = all_fully_delivered and fully
        if matching and dose is not None:
            total_delivered += dose
        elif matching:
            estimable = False
        elif selected:
            # A selected recordless plan makes the course estimate unknown. This
            # is deliberately different from a known zero-dose course.
            estimable = False

        if dose is not None and rx > 0 and dose > rx + max(0.1, 0.05 * rx):
            warning = f"RTPLAN {uid} delivered dose {dose:.6g} Gy exceeds prescribed dose {rx:.6g} Gy"
            plan_warnings.append(warning)
            delivery_warnings.append(warning)
            logger.warning("Dose delivery warning: %s", warning)
        plan_details.append(
            {
                "plan_path": str(path),
                "plan_sop_uid": uid,
                "prescribed_dose_gy": rx if rx > 0 else None,
                "planned_fraction_count": planned_fx or None,
                "target_dose_reference_numbers": sorted(target_numbers),
                "delivered_record_count": record_count,
                "delivered_fraction_count": fraction_count,
                "delivered_dose_gy": dose,
                "fraction_weighted_dose_gy": fallback_dose,
                "method": method,
                "status": "fully_delivered" if fully else ("partially_delivered" if matching else "no_records"),
                "warning_messages": plan_warnings,
                "record_paths": [str(row["path"]) for row in records_for_plan],
            }
        )
    if not selected:
        all_fully_delivered = False
        estimable = False
    if not records:
        status = "no_records_at_all"
    elif not any_matching_records:
        status = "delivered_but_records_absent"
    elif all_fully_delivered and estimable:
        status = "fully_delivered"
    else:
        status = "partially_delivered"
    dose_value = float(total_delivered) if any_matching_records and estimable else None
    if not methods and any_matching_records:
        method = "unknown"
    elif len(methods) == 1:
        method = next(iter(methods))
    elif methods:
        method = "mixed_rtrecord_and_fraction_weighted"
    else:
        method = None
    return {
        "delivered_dose_gy": dose_value,
        "delivery_status": status,
        "delivery_method": method,
        "delivered_record_count": delivered_records,
        "delivered_fraction_count": delivered_fractions,
        "planned_fraction_count": total_planned_fx or None,
        "delivery_plan_details": plan_details,
        "delivery_warnings": delivery_warnings,
        "unresolved_record_plan_uids": unresolved["unresolved_plan_uids"],
        "unresolved_record_count": unresolved["unresolved_record_count"],
        "unresolved_reference_count": unresolved["unresolved_reference_count"],
        "selected_plan_uids": sorted(uid for uid in selected_uids if uid),
        "selected_dose_paths": [str(p) for p in (selected_dose_paths or [])],
    }


def _per_plan_delivery_contract(
    plan_paths: Iterable[Path],
    record_paths: Iterable[Path],
    selected_plan_paths: Iterable[Path],
    copied_plan_paths: dict[Path, Path],
    copied_record_paths: dict[Path, Path],
) -> list[dict[str, object]]:
    """Serialize RTRECORD evidence for every candidate plan in one course.

    Selected membership and zero-record plans remain visible together. This is
    intentionally separate from the course delivered-dose calculation, which
    operates only on selected plans.
    """
    plans = list(dict.fromkeys(Path(path) for path in plan_paths))
    selected = set(Path(path) for path in selected_plan_paths)
    evidence = _record_delivery_evidence(record_paths)
    details: list[dict[str, object]] = []
    for path in plans:
        meta = _plan_evidence(path)
        uid = str(meta.get("sop_uid") or "")
        plan_evidence = evidence.get(
            uid,
            {"instances": set(), "sessions": set(), "dates": set(), "records": []},
        )
        records = list(plan_evidence.get("records", []))
        record_count = len(plan_evidence.get("instances", set()))
        fraction_count = len(plan_evidence.get("sessions", set()))
        copied = copied_plan_paths.get(path)
        details.append(
            {
                "plan_path": str(copied or ""),
                "plan_sop_uid": uid,
                "prescribed_dose_gy": meta.get("total_rx_gy"),
                "planned_fraction_count": int(meta.get("fractions_planned") or 0) or None,
                "delivered_record_count": int(record_count),
                "delivered_fraction_count": int(fraction_count),
                "treatment_dates": sorted(str(value) for value in plan_evidence.get("dates", set())),
                "record_paths": [
                    str(copied_record_paths.get(Path(str(row.get("path") or "")), ""))
                    for row in records
                ],
                "zero_delivery_records": record_count == 0,
                "selected_for_dose_grid": path in selected,
                "status": (
                    "no_records"
                    if record_count == 0
                    else (
                        "fully_delivered"
                        if int(meta.get("fractions_planned") or 0) > 0
                        and fraction_count >= int(meta.get("fractions_planned") or 0)
                        else "partially_delivered"
                    )
                ),
            }
        )
    return details


def _dose_plausibility(
    prescribed_dose_gy: float | None,
    delivered_dose_gy: float | None,
    threshold_gy: float,
) -> dict[str, object]:
    """Return a binding course dose-QC verdict at the configured threshold."""
    prescribed_warning = bool(
        prescribed_dose_gy is not None and prescribed_dose_gy > threshold_gy
    )
    delivered_warning = bool(
        delivered_dose_gy is not None and delivered_dose_gy > threshold_gy
    )
    reasons: list[str] = []
    if prescribed_warning and prescribed_dose_gy is not None:
        reasons.append(
            f"prescribed dose {prescribed_dose_gy:.6g} Gy exceeds configured maximum {float(threshold_gy):.6g} Gy"
        )
    if delivered_warning and delivered_dose_gy is not None:
        reasons.append(
            f"delivered dose {delivered_dose_gy:.6g} Gy exceeds configured maximum {float(threshold_gy):.6g} Gy"
        )
    passed = not (prescribed_warning or delivered_warning)
    return {
        "dose_plausibility_threshold_gy": float(threshold_gy),
        "prescribed_dose_plausibility_warning": prescribed_warning,
        "delivered_dose_plausibility_warning": delivered_warning,
        "dose_plausibility_warning": prescribed_warning or delivered_warning,
        "dose_qc_status": "pass" if passed else "fail",
        "dose_qc_pass": passed,
        "dose_qc_reasons": reasons,
    }


def _dose_close_to_reference(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= max(0.1, 0.05 * max(abs(float(right)), 1.0))


def _plan_evidence(path: Path) -> dict:
    """Read plan identity, prescription, fractions, and deterministic chronology."""
    meta = _extract_plan_metadata(path)
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        meta["fractions_planned"] = 0
        return meta
    fractions = 0
    for group in getattr(ds, "FractionGroupSequence", []) or []:
        value = getattr(group, "NumberOfFractionsPlanned", None)
        try:
            fractions += int(value or 0)
        except (TypeError, ValueError):
            continue
    meta["fractions_planned"] = fractions
    inferred_rx = infer_plan_rx_gy(ds)
    if inferred_rx is not None:
        meta["total_rx_gy"] = float(inferred_rx)
    target_reference_numbers: list[str] = []
    for dose_reference in getattr(ds, "DoseReferenceSequence", []) or []:
        reference_type = str(getattr(dose_reference, "DoseReferenceType", "") or "").upper()
        if reference_type and reference_type not in {
            "TARGET",
            "TREATED_VOLUME",
            "PLANNED_TARGET_VOLUME",
        }:
            continue
        dose = _finite_nonnegative(getattr(dose_reference, "TargetPrescriptionDose", None))
        number = str(getattr(dose_reference, "DoseReferenceNumber", "") or "").strip()
        if dose is not None and number and (inferred_rx is None or _dose_close_to_reference(dose, inferred_rx)):
            target_reference_numbers.append(number)
    meta["target_dose_reference_numbers"] = sorted(set(target_reference_numbers))
    return meta


def _prescriptions_match(left: dict, right: dict) -> bool:
    """Identify revision-equivalent prescriptions without plan-label text."""
    left_rx = float(left.get("total_rx_gy") or 0.0)
    right_rx = float(right.get("total_rx_gy") or 0.0)
    left_fx = int(left.get("fractions_planned") or 0)
    right_fx = int(right.get("fractions_planned") or 0)
    if left_rx <= 0 or right_rx <= 0 or left_fx <= 0 or right_fx <= 0:
        return False
    tolerance = max(0.25, 0.005 * max(left_rx, right_rx))
    return left_fx == right_fx and abs(left_rx - right_rx) <= tolerance


def _same_fraction_dose(left: dict, right: dict) -> bool:
    left_fx = int(left.get("fractions_planned") or 0)
    right_fx = int(right.get("fractions_planned") or 0)
    if left_fx <= 0 or right_fx <= 0:
        return False
    left_dpf = float(left.get("total_rx_gy") or 0.0) / left_fx
    right_dpf = float(right.get("total_rx_gy") or 0.0) / right_fx
    return left_dpf > 0 and abs(left_dpf - right_dpf) <= max(0.05, 0.02 * max(left_dpf, right_dpf))


def _replacement_partition(plans: List[dict]) -> tuple[int, tuple[int, ...]] | None:
    """Find one course-total plan that equals a set of partial replacement plans."""
    if len(plans) < 3:
        return None
    for total_index, total in enumerate(plans):
        total_rx = float(total.get("total_rx_gy") or 0.0)
        total_fx = int(total.get("fractions_planned") or 0)
        if total_rx <= 0:
            continue
        other_indices = [index for index in range(len(plans)) if index != total_index]
        for size in range(2, min(4, len(other_indices)) + 1):
            for parts in combinations(other_indices, size):
                part_rx = sum(float(plans[index].get("total_rx_gy") or 0.0) for index in parts)
                part_fx = sum(int(plans[index].get("fractions_planned") or 0) for index in parts)
                tolerance = max(0.25, 0.005 * total_rx)
                fractions_match = total_fx <= 0 or part_fx <= 0 or part_fx == total_fx
                if abs(part_rx - total_rx) <= tolerance and fractions_match:
                    return total_index, parts
    return None


def _classify_doses(
    plan_paths: List[Path],
    dose_paths: List[Path],
    max_total_dose_gy: float = DEFAULT_MAX_TOTAL_DOSE_GY,
    treatment_record_paths: Optional[Iterable[Path]] = None,
) -> DoseClassification:
    """Select dose evidence without using free-text plan labels.

    Explicit dose-to-plan references establish membership. Equal prescription
    and fraction signatures are revisions and contribute once. A complete plan
    whose prescription equals partial plans is a replacement course total and
    is not added to those parts. Distinct phases are summed only after those two
    de-duplication steps. RT treatment records choose among revisions and can
    distinguish a completed sequential phase from an incompletely delivered
    course-total plan.
    """
    warnings: List[str] = []
    plan_paths = list(dict.fromkeys(Path(path) for path in plan_paths))
    dose_paths = list(dict.fromkeys(Path(path) for path in dose_paths))
    record_paths = list(dict.fromkeys(Path(path) for path in (treatment_record_paths or [])))
    if not dose_paths:
        return DoseClassification(
            classification="no_doses",
            selected_doses=[],
            selected_plans=plan_paths,
            excluded_doses=[],
            should_sum=False,
            warnings=["No dose files found"],
            reason="No dose files available",
        )

    dose_meta = [_extract_dose_metadata(path) for path in dose_paths]
    plan_meta = [_plan_evidence(path) for path in plan_paths]
    plan_by_uid = {meta["sop_uid"]: meta for meta in plan_meta if meta.get("sop_uid")}
    delivery = _record_delivery_evidence(record_paths)

    plan_sum_doses = [meta for meta in dose_meta if meta["summation_type"] == "PLAN_SUM"]
    if plan_sum_doses:
        best_sum = max(
            plan_sum_doses,
            key=lambda meta: (len(meta["referenced_plan_uids"]), str(meta["path"])),
        )
        covered = set(best_sum["referenced_plan_uids"])
        selected_plans = [
            meta["path"] for meta in plan_meta if meta.get("sop_uid") in covered
        ]
        delivered_uids = {
            uid
            for uid in covered
            if delivery.get(uid, {}).get("instances", set())
        }
        if not record_paths or delivered_uids == covered:
            excluded = [
                meta["path"]
                for meta in dose_meta
                if meta is not best_sum
                and set(meta["referenced_plan_uids"])
                and set(meta["referenced_plan_uids"]).issubset(covered)
            ]
            return DoseClassification(
                classification="PLAN_SUM_used",
                selected_doses=[best_sum["path"]],
                selected_plans=selected_plans,
                excluded_doses=excluded,
                should_sum=False,
                warnings=warnings,
                reason=f"TPS-provided PLAN_SUM references {len(covered)} plan(s)",
            )
        warnings.append(
            "Excluded TPS PLAN_SUM because it contains at least one plan with zero delivery records"
        )
        dose_meta = [meta for meta in dose_meta if meta is not best_sum]
        if not delivered_uids or not dose_meta:
            return DoseClassification(
                classification="no_delivered_plan_dose",
                selected_doses=[],
                selected_plans=[],
                excluded_doses=[meta["path"] for meta in plan_sum_doses],
                should_sum=False,
                warnings=warnings,
                reason="No separable RTDOSE remains for a plan supported by RTRECORD",
            )

    plan_level = [meta for meta in dose_meta if meta["summation_type"] == "PLAN"]
    beam_doses = [meta for meta in dose_meta if meta["summation_type"] == "BEAM"]
    unsupported_doses = [
        meta for meta in dose_meta if meta["summation_type"] not in {"PLAN", "BEAM"}
    ]
    if unsupported_doses:
        warnings.append(
            "Excluded unsupported RTDOSE summation types: "
            + ", ".join(sorted({str(meta["summation_type"]) for meta in unsupported_doses}))
        )
    dose_meta = plan_level + beam_doses
    if plan_level and beam_doses:
        covered = {
            uid for meta in plan_level for uid in meta["referenced_plan_uids"] if uid
        }
        covered_beams = [
            meta
            for meta in beam_doses
            if set(meta["referenced_plan_uids"])
            and set(meta["referenced_plan_uids"]).issubset(covered)
        ]
        if covered_beams:
            warnings.append(
                f"Excluded {len(covered_beams)} BEAM dose(s) because matching PLAN dose(s) exist"
            )
        dose_meta = plan_level + [meta for meta in beam_doses if meta not in covered_beams]

    if dose_meta and all(meta["summation_type"] == "BEAM" for meta in dose_meta):
        referenced = {
            uid for meta in dose_meta for uid in meta["referenced_plan_uids"] if uid
        }
        selected_plans = [
            meta["path"] for meta in plan_meta if meta.get("sop_uid") in referenced
        ]
        delivered_references = {
            uid for uid in referenced if delivery.get(uid, {}).get("instances", set())
        }
        if record_paths and delivered_references != referenced:
            return DoseClassification(
                classification="no_delivered_plan_dose",
                selected_doses=[],
                selected_plans=[],
                excluded_doses=[meta["path"] for meta in dose_meta],
                should_sum=False,
                warnings=warnings + ["Excluded BEAM doses for a plan with zero delivery records"],
                reason="RTRECORD does not support delivery of the BEAM-dose plan",
            )
        if len(referenced) == 1 and selected_plans and len(dose_meta) > 1:
            return DoseClassification(
                classification="beam_doses_summed_to_plan",
                selected_doses=[meta["path"] for meta in dose_meta],
                selected_plans=selected_plans,
                excluded_doses=[],
                should_sum=True,
                warnings=warnings,
                reason=f"Summing {len(dose_meta)} BEAM dose(s) for one RTPLAN",
            )
        if len(referenced) == 1 and selected_plans:
            return DoseClassification(
                classification="single_beam_dose_rejected",
                selected_doses=[],
                selected_plans=[],
                excluded_doses=[meta["path"] for meta in dose_meta],
                should_sum=False,
                warnings=warnings + ["A single BEAM RTDOSE is not a plan-level treatment dose grid"],
                reason="Single BEAM RTDOSE cannot represent the complete plan dose",
            )

    candidates_by_plan: Dict[str, List[dict]] = defaultdict(list)
    unmatched_doses: List[dict] = []
    for dose in dose_meta:
        matched = False
        for uid in dose["referenced_plan_uids"]:
            if uid in plan_by_uid:
                candidates_by_plan[uid].append(dose)
                matched = True
        if not matched:
            unmatched_doses.append(dose)

    dose_for_plan: Dict[str, dict] = {}
    for uid, candidates in candidates_by_plan.items():
        ordered = sorted(
            candidates,
            key=lambda meta: (
                0 if meta["summation_type"] == "PLAN" else 1,
                str(meta["path"]),
            ),
        )
        dose_for_plan[uid] = ordered[0]
        if len(ordered) > 1:
            warnings.append(
                f"Plan {uid} had {len(ordered)} candidate non-PLAN_SUM doses; selected {ordered[0]['path'].name} deterministically"
            )

    paired_plans = [meta for meta in plan_meta if meta.get("sop_uid") in dose_for_plan]
    if not paired_plans:
        warnings.append(
            "No RTDOSE reference resolved to a course RTPLAN; excluded every unresolved dose"
        )
        return DoseClassification(
            classification="unresolved_reference_excluded",
            selected_doses=[],
            selected_plans=[],
            excluded_doses=[meta["path"] for meta in dose_meta],
            should_sum=False,
            warnings=warnings,
            reason="Dose-to-plan references did not resolve; no attachment was guessed",
        )

    signature_groups: List[List[dict]] = []
    for plan in paired_plans:
        for group in signature_groups:
            if _prescriptions_match(plan, group[0]):
                group.append(plan)
                break
        else:
            signature_groups.append([plan])

    def representative(group: List[dict]) -> dict:
        def rank(plan: dict) -> tuple:
            evidence = delivery.get(str(plan.get("sop_uid") or ""), {})
            return (
                len(evidence.get("instances", set())),
                str(plan.get("plan_date") or ""),
                str(plan.get("plan_time") or ""),
                str(plan.get("sop_uid") or ""),
            )
        return max(group, key=rank)

    representatives = [representative(group) for group in signature_groups]
    duplicate_count = sum(len(group) - 1 for group in signature_groups)
    if duplicate_count:
        warnings.append(
            f"De-duplicated {duplicate_count} revision plan(s) with equivalent prescription and fraction signatures"
        )

    if record_paths and not any(
        delivery.get(str(plan.get("sop_uid") or ""), {}).get("instances", set())
        for plan in representatives
    ):
        warnings.append(
            "Excluded every plan because RTRECORD objects exist but none references a course plan"
        )
        return DoseClassification(
            classification="no_delivered_plan_dose",
            selected_doses=[],
            selected_plans=[],
            excluded_doses=[dose_for_plan[str(plan["sop_uid"])]["path"] for plan in paired_plans],
            should_sum=False,
            warnings=warnings,
            reason="No course RTPLAN has linked delivery evidence",
        )

    if len(representatives) == 1:
        selected = representatives[0]
        selected_dose = dose_for_plan[str(selected["sop_uid"])]
        classification = "plan_revisions_deduplicated" if duplicate_count else "single_dose"
        return DoseClassification(
            classification=classification,
            selected_doses=[selected_dose["path"]],
            selected_plans=[selected["path"]],
            excluded_doses=[
                dose_for_plan[str(plan["sop_uid"])]["path"]
                for plan in paired_plans
                if plan is not selected
            ],
            should_sum=False,
            warnings=warnings,
            reason=(
                "Equivalent prescription revisions contribute once"
                if duplicate_count
                else "Single referenced plan-level dose"
            ),
        )

    replacement = _replacement_partition(representatives)
    selected_representatives: List[dict]
    classification: str
    should_sum: bool
    if replacement is not None:
        total_index, part_indices = replacement
        total = representatives[total_index]
        parts = [representatives[index] for index in part_indices]
        total_support = len(delivery.get(str(total.get("sop_uid") or ""), {}).get("instances", set()))
        part_support = sum(
            len(delivery.get(str(part.get("sop_uid") or ""), {}).get("instances", set()))
            for part in parts
        )
        if total_support or part_support:
            if total_support >= part_support and total_support > 0:
                selected_representatives = [total]
            else:
                selected_representatives = [
                    plan
                    for plan in parts
                    if delivery.get(str(plan.get("sop_uid") or ""), {}).get("instances", set())
                ]
        else:
            selected_representatives = parts if part_support > total_support else [total]
        classification = "replacement_course_total"
        should_sum = len(selected_representatives) > 1
        warnings.append(
            "A course-total prescription equalled component prescriptions; RTRECORD-selected plans define dose-grid membership"
        )
    else:
        def plan_dates(plan: dict) -> set[str]:
            return set(delivery.get(str(plan.get("sop_uid") or ""), {}).get("dates", set()))

        def plan_records(plan: dict) -> set[str]:
            return set(delivery.get(str(plan.get("sop_uid") or ""), {}).get("instances", set()))

        supported_candidates = [plan for plan in representatives if plan_records(plan)]
        supported: List[dict] = []
        for plan in supported_candidates:
            dates = plan_dates(plan)
            covering = next(
                (
                    other
                    for other in supported_candidates
                    if other is not plan and dates and dates < plan_dates(other)
                ),
                None,
            )
            if covering is not None:
                warnings.append(
                    f"Excluded plan {plan.get('sop_uid')} because its treatment dates are a strict subset of plan {covering.get('sop_uid')}"
                )
            else:
                supported.append(plan)

        largest = max(
            representatives,
            key=lambda plan: (
                int(plan.get("fractions_planned") or 0),
                float(plan.get("total_rx_gy") or 0.0),
                str(plan.get("sop_uid") or ""),
            ),
        )
        if supported:
            selected_representatives = list(supported)
            for plan in representatives:
                if plan in supported or plan in supported_candidates:
                    continue
                warnings.append(
                    f"Excluded plan {plan.get('sop_uid')} because it has zero delivery records"
                )
            classification = (
                "sequential_phases_summed"
                if len(selected_representatives) > 1
                else "delivered_plan_selected"
            )
            should_sum = len(selected_representatives) > 1
        elif record_paths:
            contained = any(
                plan is not largest
                and _same_fraction_dose(plan, largest)
                and int(plan.get("fractions_planned") or 0)
                < int(largest.get("fractions_planned") or 0)
                for plan in representatives
            )
            if contained:
                selected_representatives = [largest]
                classification = "replacement_course_total"
                should_sum = False
                warnings.append(
                    "No RTRECORD referenced this course; retained the encompassing planned prescription with delivery unknown"
                )
            else:
                selected_representatives = representatives
                classification = "sequential_phases_summed"
                should_sum = True
        else:
            selected_representatives = representatives
            classification = "sequential_phases_summed"
            should_sum = True

    selected_plans = [plan["path"] for plan in selected_representatives]
    selected_doses = [dose_for_plan[str(plan["sop_uid"])]["path"] for plan in selected_representatives]
    selected_dose_set = set(selected_doses)
    excluded_doses = [
        dose_for_plan[str(plan["sop_uid"])]["path"]
        for plan in paired_plans
        if dose_for_plan[str(plan["sop_uid"])]["path"] not in selected_dose_set
    ]
    selected_total_rx = sum(float(plan.get("total_rx_gy") or 0.0) for plan in selected_representatives)
    if should_sum and selected_total_rx > max_total_dose_gy:
        warnings.append(
            f"PLAUSIBILITY WARNING: selected prescription total {selected_total_rx:.1f} Gy exceeds {max_total_dose_gy:.1f} Gy"
        )
        logger.error(
            "Dose QC: selected prescription total %.1f Gy exceeds %.1f Gy after evidence-based de-duplication",
            selected_total_rx,
            max_total_dose_gy,
        )
    return DoseClassification(
        classification=classification,
        selected_doses=selected_doses,
        selected_plans=selected_plans,
        excluded_doses=excluded_doses,
        should_sum=should_sum,
        warnings=warnings,
        reason=(
            "Distinct prescription phases remain after revision and replacement-plan de-duplication"
            if should_sum
            else "One delivered course-total prescription selected"
        ),
    )


def _mask_array_to_image(ct_img: sitk.Image, mask: np.ndarray) -> Optional[sitk.Image]:
    arr = np.asarray(mask)
    if arr.size == 0:
        return None
    try:
        ct_z, ct_y, ct_x = ct_img.GetSize()[2], ct_img.GetSize()[1], ct_img.GetSize()[0]
    except Exception:
        size = ct_img.GetSize()
        ct_z, ct_y, ct_x = size[2], size[1], size[0]
    if arr.shape == (ct_z, ct_y, ct_x):
        zyx = arr
    elif arr.shape == (ct_y, ct_x, ct_z):
        zyx = np.transpose(arr, (2, 0, 1))
    elif arr.shape == (ct_x, ct_y, ct_z):
        zyx = np.transpose(arr, (2, 1, 0))
    else:
        logger.debug("Mask shape %s does not match CT shape (%d,%d,%d)", arr.shape, ct_z, ct_y, ct_x)
        return None
    zyx = zyx.astype(np.uint8)
    img = sitk.GetImageFromArray(zyx)
    img.SetSpacing(ct_img.GetSpacing())
    img.SetDirection(ct_img.GetDirection())
    img.SetOrigin(ct_img.GetOrigin())
    return img


def _export_original_segmentation_from_paths(
    *,
    rs_path: Optional[Path],
    primary_nifti: Optional[Path],
    dicom_ct_dir: Path,
    segmentation_original_dir: Path,
    log_root: Path,
    overwrite: bool,
) -> Optional[dict]:
    if not rs_path or not rs_path.exists() or not primary_nifti or not primary_nifti.exists():
        return None
    seg_root = segmentation_original_dir
    base_name = _strip_nifti_base(primary_nifti)
    target_root = seg_root / base_name
    ensure_dir(target_root)
    manifest_path = target_root / "metadata.json"
    if manifest_path.exists() and not overwrite:
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        from rt_utils import RTStructBuilder
    except Exception as exc:
        logger.debug("rt-utils unavailable for segmentation export: %s", exc)
        return None

    try:
        ct_img = sitk.ReadImage(str(primary_nifti))
    except Exception as exc:
        logger.warning("Failed to load primary NIfTI for %s: %s", log_root, exc)
        return None

    try:
        builder = RTStructBuilder.create_from(
            dicom_series_path=str(dicom_ct_dir),
            rt_struct_path=str(rs_path),
        )
    except Exception as exc:
        logger.warning("Failed to load RTSTRUCT for segmentation export (%s): %s", rs_path, exc)
        return None

    used_names: dict[str, int] = {}
    manifest = {
        "model": "manual",
        "source_rtstruct": str(rs_path),
        "source_nifti": str(primary_nifti),
        "structures": [],
    }

    for roi_name in builder.get_roi_names():
        try:
            mask = builder.get_roi_mask_by_name(roi_name)
        except Exception:
            mask = None
        if mask is None:
            continue
        mask_bool = np.asarray(mask).astype(bool)
        if not np.any(mask_bool):
            continue
        sitk_mask = _mask_array_to_image(ct_img, mask_bool)
        if sitk_mask is None:
            continue
        safe = _sanitize_name(roi_name, "ROI")
        idx = used_names.get(safe, 0)
        used_names[safe] = idx + 1
        if idx:
            safe = f"{safe}_{idx+1}"
        out_path = target_root / f"{safe}.nii.gz"
        try:
            sitk.WriteImage(sitk_mask, str(out_path), useCompression=True)
        except Exception as exc:
            logger.debug("Failed to write manual segmentation mask %s: %s", out_path, exc)
            continue
        manifest["structures"].append(
            {
                "roi_name": roi_name,
                "mask": str(out_path.relative_to(seg_root)),
            }
        )

    if manifest["structures"]:
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write segmentation manifest for %s: %s", log_root, exc)
        return manifest
    return None


def _export_original_segmentation(
    course: CourseOutput,
    overwrite: bool,
) -> Optional[dict]:
    return _export_original_segmentation_from_paths(
        rs_path=course.rs_path,
        primary_nifti=course.primary_nifti,
        dicom_ct_dir=course.dirs.dicom_ct,
        segmentation_original_dir=course.dirs.segmentation_original,
        log_root=course.dirs.root,
        overwrite=overwrite,
    )


def _index_series_and_registrations(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
) -> tuple[Dict[tuple[str, str], List[Path]], Dict[str, List[Dict[str, object]]]]:
    """Walk ``dicom_root`` (cohort-scoped when ``patient_ids`` is given) once and
    build the series index, registration index, and per-series modality metadata
    used to attach related (non-course) series/REGs to a course.

    ``max_workers`` controls how many threads read DICOM headers concurrently
    (defaults to ``utils.DEFAULT_INDEX_WORKERS``; ``max_workers=1`` reproduces
    the exact single-threaded behaviour). Only the per-file reads are
    parallelized; the assembly loop below still runs single-threaded, in the
    SAME order the files were discovered in, so the result does not depend on
    thread completion order.
    """
    series_index: Dict[tuple[str, str], List[Path]] = {}
    registrations: Dict[str, List[Dict[str, object]]] = {}
    series_meta: Dict[tuple[str, str], Dict[str, object]] = {}

    paths: List[Path] = []
    for base, _, files in _scoped_walk(dicom_root, patient_ids):
        for name in files:
            if not name.lower().endswith('.dcm'):
                continue
            paths.append(Path(base) / name)

    workers = max_workers if max_workers is not None else DEFAULT_INDEX_WORKERS
    datasets = parallel_map_files(paths, read_dicom, workers)

    for p, ds in zip(paths, datasets):
        if ds is None:
            continue
        modality = str(getattr(ds, 'Modality', '') or '')
        patient_id = str(get(ds, (0x0010, 0x0020), "")) or ""
        series_uid = str(get(ds, (0x0020, 0x000E), "")) or ""
        if patient_id and series_uid:
            series_index.setdefault((patient_id, series_uid), []).append(p)
            meta = series_meta.setdefault((patient_id, series_uid), {})
            if "modality" not in meta and modality:
                meta["modality"] = modality
        modality = str(getattr(ds, 'Modality', '') or '')
        if modality != 'REG' or not patient_id:
            continue
        reg_item: Dict[str, object] = {
            'path': p,
            'for_uids': set(),
            'referenced_series': set(),
            'series_by_for': {},
        }

        def _add_for(uid: str | None) -> None:
            if not uid:
                return
            cast(set, reg_item['for_uids']).add(uid)

        def _add_series(series_uid: str | None, for_uid: str | None = None) -> None:
            if not series_uid:
                return
            cast(set, reg_item['referenced_series']).add(series_uid)
            if for_uid:
                series_by_for = cast(Dict[str, set[str]], reg_item.setdefault('series_by_for', {}))
                series_by_for.setdefault(for_uid, set()).add(series_uid)

        try:
            for ref_for in getattr(ds, 'ReferencedFrameOfReferenceSequence', []) or []:
                for_uid = str(getattr(ref_for, 'FrameOfReferenceUID', '') or '')
                if for_uid:
                    _add_for(for_uid)
                for study in getattr(ref_for, 'RTReferencedStudySequence', []) or []:
                    for series in getattr(study, 'RTReferencedSeriesSequence', []) or []:
                        series_uid_ref = str(getattr(series, 'SeriesInstanceUID', '') or '')
                        if series_uid_ref:
                            _add_series(series_uid_ref, for_uid)

            for reg_seq in getattr(ds, 'RegistrationSequence', []) or []:
                reg_for_uid = str(getattr(reg_seq, 'FrameOfReferenceUID', '') or '')
                if reg_for_uid:
                    _add_for(reg_for_uid)
                for ref_study in getattr(reg_seq, 'ReferencedStudySequence', []) or []:
                    for ref_series in getattr(ref_study, 'ReferencedSeriesSequence', []) or []:
                        series_uid_ref = str(getattr(ref_series, 'SeriesInstanceUID', '') or '')
                        if series_uid_ref:
                            _add_series(series_uid_ref, reg_for_uid)

            for ref_series in getattr(ds, 'ReferencedSeriesSequence', []) or []:
                series_uid_ref = str(getattr(ref_series, 'SeriesInstanceUID', '') or '')
                if series_uid_ref:
                    _add_series(series_uid_ref)

            for other_study in getattr(ds, 'StudiesContainingOtherReferencedInstancesSequence', []) or []:
                for ref_series in getattr(other_study, 'ReferencedSeriesSequence', []) or []:
                    series_uid_ref = str(getattr(ref_series, 'SeriesInstanceUID', '') or '')
                    if series_uid_ref:
                        _add_series(series_uid_ref)

        except Exception as exc:
            logger.debug("Failed indexing registration %s: %s", p, exc)
            continue

        registrations.setdefault(patient_id, []).append(reg_item)
    return series_index, registrations, series_meta

def _create_summed_plan(plan_files: List[Path], total_dose_gy: float | None = None) -> tuple[pydicom.dataset.FileDataset, list[pydicom.dataset.FileDataset], list[str]]:
    """Build an evaluation-only plan sum dataset that references all source RTPLANs."""

    if not plan_files:
        raise ValueError("No plan files to sum")

    plan_datasets: list[pydicom.dataset.FileDataset] = [
        pydicom.dcmread(str(path), stop_before_pixels=False)
        for path in plan_files
    ]

    base_plan = plan_datasets[0]
    plan_sum = copy.deepcopy(base_plan)

    now = datetime.datetime.now()
    plan_sum.SeriesInstanceUID = generate_uid()
    plan_sum.SOPInstanceUID = generate_uid()
    plan_sum.InstanceCreationDate = now.strftime("%Y%m%d")
    plan_sum.InstanceCreationTime = now.strftime("%H%M%S")
    plan_sum.SeriesDescription = f"Plan Sum ({len(plan_files)} plans)"

    def _suffix(value: Optional[str], suffix: str, limit: Optional[int] = None) -> Optional[str]:
        if not value:
            return value
        new_val = f"{value}{suffix}"
        if limit is not None and len(new_val) > limit:
            return new_val[:limit]
        return new_val

    if hasattr(plan_sum, "RTPlanLabel"):
        plan_sum.RTPlanLabel = _suffix(str(getattr(plan_sum, "RTPlanLabel", "")), "_SUM", limit=16) or "PLAN_SUM"
    if hasattr(plan_sum, "RTPlanName"):
        plan_sum.RTPlanName = _suffix(str(getattr(plan_sum, "RTPlanName", "")), "_SUM", limit=64) or "PlanSum"
    if hasattr(plan_sum, "RTPlanDescription"):
        plan_sum.RTPlanDescription = f"Summation of {len(plan_files)} plans generated on {now.isoformat()}"

    if hasattr(plan_sum, "PlanIntent"):
        plan_sum.PlanIntent = "REVIEW"
    if hasattr(plan_sum, "PlanStatus"):
        plan_sum.PlanStatus = "UNPLANNED"
    if hasattr(plan_sum, "ApprovalStatus"):
        plan_sum.ApprovalStatus = "APPROVED"

    beam_mappings: dict[str, dict[int, int]] = {}
    new_beams: list[Dataset] = []
    new_beam_number = 1
    for plan_index, ds_plan in enumerate(plan_datasets):
        plan_uid = str(getattr(ds_plan, "SOPInstanceUID", ""))
        if not plan_uid:
            plan_uid = generate_uid()
            ds_plan.SOPInstanceUID = plan_uid
        beam_mappings.setdefault(plan_uid, {})
        for beam in getattr(ds_plan, "BeamSequence", []) or []:
            beam_copy = copy.deepcopy(beam)
            try:
                original_number = int(getattr(beam_copy, "BeamNumber", new_beam_number))
            except Exception:
                original_number = new_beam_number
            beam_copy.BeamNumber = new_beam_number
            beam_mappings[plan_uid][original_number] = new_beam_number

            beam_name = str(getattr(beam_copy, "BeamName", "") or "")
            suffix = f"_P{plan_index + 1}"
            if beam_name:
                new_name = beam_name + suffix
            else:
                new_name = f"BEAM{new_beam_number:02d}{suffix}"
            if len(new_name) > 16:
                new_name = new_name[:16]
            beam_copy.BeamName = new_name

            new_beams.append(beam_copy)
            new_beam_number += 1
    if new_beams:
        plan_sum.BeamSequence = Sequence(new_beams)
        plan_sum.NumberOfBeams = len(new_beams)

    total_fractions = 0
    for ds_plan in plan_datasets:
        for fg in getattr(ds_plan, "FractionGroupSequence", []) or []:
            val = getattr(fg, "NumberOfFractionsPlanned", None)
            if val not in (None, ""):
                try:
                    total_fractions += int(val)
                except Exception:
                    continue
    if total_fractions <= 0:
        total_fractions = len(plan_datasets)

    fg_dataset = Dataset()
    fg_dataset.FractionGroupNumber = 1
    fg_dataset.NumberOfFractionsPlanned = int(total_fractions)
    fg_dataset.NumberOfBeams = len(new_beams)
    fg_dataset.ReferencedBeamSequence = Sequence()

    ref_beam_items: list[Dataset] = []
    for ds_plan in plan_datasets:
        plan_uid = str(ds_plan.SOPInstanceUID)
        mapping = beam_mappings.get(plan_uid, {})
        for beam in getattr(ds_plan, "BeamSequence", []) or []:
            ref_item = Dataset()
            original_number = int(getattr(beam, "BeamNumber", 0) or 0)
            new_number = mapping.get(original_number)
            if new_number is None:
                continue
            ref_item.ReferencedBeamNumber = new_number
            if hasattr(beam, "BeamMeterset") and beam.BeamMeterset not in (None, ""):
                try:
                    ref_item.BeamMeterset = float(beam.BeamMeterset)
                except Exception:
                    pass
            if hasattr(beam, "BeamDose") and beam.BeamDose not in (None, ""):
                try:
                    ref_item.BeamDose = float(beam.BeamDose)
                except Exception:
                    pass
            ref_beam_items.append(ref_item)
    fg_dataset.ReferencedBeamSequence = Sequence(ref_beam_items)

    plan_sum.FractionGroupSequence = Sequence([fg_dataset])
    plan_sum.NumberOfFractionsPlanned = int(total_fractions)

    if total_dose_gy is not None:
        try:
            if hasattr(plan_sum, "DoseReferenceSequence") and plan_sum.DoseReferenceSequence:
                for dose_ref in plan_sum.DoseReferenceSequence:
                    if hasattr(dose_ref, "TargetPrescriptionDose"):
                        dose_ref.TargetPrescriptionDose = float(total_dose_gy)
        except Exception as exc:
            logger.warning("Failed to update prescription dose in plan sum: %s", exc)

    ref_plan_items: list[Dataset] = []
    for ds_plan in plan_datasets:
        item = Dataset()
        item.ReferencedSOPClassUID = str(getattr(ds_plan, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.481.5"))
        item.ReferencedSOPInstanceUID = str(ds_plan.SOPInstanceUID)
        mapping = beam_mappings.get(str(ds_plan.SOPInstanceUID), {})
        beam_refs: list[Dataset] = []
        for original_number, new_number in sorted(mapping.items()):
            ref_beam = Dataset()
            ref_beam.ReferencedBeamNumber = int(new_number)
            beam_refs.append(ref_beam)
        if beam_refs:
            item.ReferencedBeamSequence = Sequence(beam_refs)
        ref_plan_items.append(item)
    if ref_plan_items:
        plan_sum.ReferencedRTPlanSequence = Sequence(ref_plan_items)

    source_uid_order: list[str] = []
    for ds in plan_datasets:
        uid = str(ds.SOPInstanceUID)
        if uid and uid not in source_uid_order:
            source_uid_order.append(uid)

    return plan_sum, plan_datasets, source_uid_order


def _sum_doses_with_resample(
    dose_files: List[Path],
    plan_sum: pydicom.dataset.FileDataset,
    plan_datasets: list[pydicom.dataset.FileDataset],
) -> tuple[pydicom.dataset.FileDataset, list[pydicom.dataset.FileDataset], list[str]]:
    if not dose_files:
        raise ValueError("No dose files to sum")

    # First pass: Scan headers to find best resolution grid without loading pixels
    ref_idx = 0
    best_resolution = float("inf")
    dose_headers = []
    
    for i, path in enumerate(dose_files):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            dose_headers.append(ds)
            
            if not hasattr(ds, "PixelSpacing") or len(ds.PixelSpacing) < 2:
                continue
                
            pixel_spacing = list(map(float, ds.PixelSpacing))
            slice_thickness = 1.0
            if hasattr(ds, "GridFrameOffsetVector") and len(ds.GridFrameOffsetVector) > 1:
                try:
                    slice_thickness = abs(float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]))
                except Exception:
                    slice_thickness = 1.0
            
            voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
            if voxel_volume < best_resolution:
                best_resolution = voxel_volume
                ref_idx = i
        except Exception as e:
            logger.warning(f"Failed to read dose header {path}: {e}")
            dose_headers.append(None)

    if not dose_headers or dose_headers[ref_idx] is None:
         raise ValueError("Could not read any valid dose headers")

    # Load reference dose fully
    logger.info(f"Using {dose_files[ref_idx].name} as reference dose grid (finest resolution)")
    ds_ref = pydicom.dcmread(str(dose_files[ref_idx]), stop_before_pixels=False)
    
    # Initialize accumulator with reference dose
    arr_ref = ds_ref.pixel_array.astype("float32")
    dose_scaling_ref = float(getattr(ds_ref, "DoseGridScaling", 1.0))
    accumulated = arr_ref * dose_scaling_ref

    # Grid geometry of reference
    rows_ref, cols_ref = ds_ref.Rows, ds_ref.Columns
    frames_ref = int(getattr(ds_ref, "NumberOfFrames", 1) or 1)
    origin_ref = np.array(list(map(float, getattr(ds_ref, "ImagePositionPatient", [0, 0, 0]))))
    pixel_spacing_ref = list(map(float, getattr(ds_ref, "PixelSpacing", [1.0, 1.0])))
    offsets_ref = getattr(ds_ref, "GridFrameOffsetVector", None)

    # C6 fix: extract ImageOrientationPatient direction cosines for reference grid
    iop_ref = list(map(float, getattr(ds_ref, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])))
    row_cosines_ref = np.array(iop_ref[0:3])
    col_cosines_ref = np.array(iop_ref[3:6])
    slice_cosines_ref = np.cross(row_cosines_ref, col_cosines_ref)

    if offsets_ref is not None and len(offsets_ref) == frames_ref:
        z_offsets_ref = np.array([float(offset) for offset in offsets_ref])
    else:
        z_offsets_ref = np.arange(frames_ref, dtype=np.float64)

    # Compute physical positions for reference grid using direction cosines
    # Each voxel (r, c, f) maps to: origin + c * ps[1] * row_cosines + r * ps[0] * col_cosines + offset[f] * slice_cosines
    z_positions_ref = origin_ref[2] + z_offsets_ref * slice_cosines_ref[2]

    # Ensure z_positions_ref is monotonically increasing for proper interpolation
    if len(z_positions_ref) > 1 and z_positions_ref[0] > z_positions_ref[-1]:
        z_positions_ref = z_positions_ref[::-1]
        z_offsets_ref = z_offsets_ref[::-1]
        accumulated = accumulated[::-1, :, :]  # Flip along z-axis to match

    # In-plane (X/Y) reference positions must include BOTH the row-index and the
    # column-index direction-cosine contributions, not just the axis-aligned term.
    # For an oblique-but-consistent grid (in-plane rotation), row_cosines_ref/
    # col_cosines_ref have nonzero cross components (e.g. row direction contributes
    # to Y, column direction contributes to X); dropping them silently resamples to
    # the wrong physical position. This mirrors the full-vector treatment already
    # used for the Z axis above. For axis-aligned grids the added cross terms are
    # exactly zero, so behavior for the common case is unchanged.
    # NOTE (scope): this corrects the REFERENCE grid's forward in-plane mapping.
    # The source-side inverse index mapping and the Z axis' in-plane coupling remain
    # axis-aligned approximations, so a fully in-plane-rotated SOURCE grid is still
    # approximate. Clinical RTDOSE grids are axis-aligned and a source/reference
    # orientation-mismatch warning fires below, so production impact is nil; a full
    # 3D-affine resample would be needed to handle oblique source grids exactly.
    row_idx_ref = np.arange(rows_ref, dtype=np.float64)[:, None]
    col_idx_ref = np.arange(cols_ref, dtype=np.float64)[None, :]
    y_positions_ref = (origin_ref[1]
                       + row_idx_ref * pixel_spacing_ref[0] * col_cosines_ref[1]
                       + col_idx_ref * pixel_spacing_ref[1] * row_cosines_ref[1])
    x_positions_ref = (origin_ref[0]
                       + row_idx_ref * pixel_spacing_ref[0] * col_cosines_ref[0]
                       + col_idx_ref * pixel_spacing_ref[1] * row_cosines_ref[0])

    # Store 1D coordinate arrays for memory-efficient resampling (avoid full meshgrid)
    # These will be broadcast during resampling

    source_dose_uids = []
    uid = str(getattr(ds_ref, "SOPInstanceUID", ""))
    if uid:
        source_dose_uids.append(uid)
        
    # Second pass: Iteratively load and resample other doses
    for i, path in enumerate(dose_files):
        if i == ref_idx:
            continue
            
        logger.debug(f"Resampling and adding dose {path.name}...")
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=False)
            
            uid = str(getattr(ds, "SOPInstanceUID", ""))
            if uid and uid not in source_dose_uids:
                source_dose_uids.append(uid)

            arr = ds.pixel_array.astype("float32")
            arr *= float(getattr(ds, "DoseGridScaling", 1.0))

            rows, cols = ds.Rows, ds.Columns
            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            origin = np.array(list(map(float, getattr(ds, "ImagePositionPatient", [0, 0, 0]))))
            pixel_spacing = list(map(float, getattr(ds, "PixelSpacing", [1.0, 1.0])))
            offsets = getattr(ds, "GridFrameOffsetVector", None)

            # C6 fix: extract source ImageOrientationPatient
            iop_src = list(map(float, getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])))
            row_cos_src = np.array(iop_src[0:3])
            col_cos_src = np.array(iop_src[3:6])
            slice_cos_src = np.cross(row_cos_src, col_cos_src)

            # Warn if source and reference have different orientations
            if (np.dot(slice_cosines_ref, slice_cos_src) < 0.99 or
                    np.dot(row_cosines_ref, row_cos_src) < 0.99):
                logger.warning(
                    "Dose %s has different ImageOrientationPatient than reference. "
                    "Resampling may produce spatially misaligned results.",
                    path.name,
                )

            if offsets is not None and len(offsets) == frames:
                z_offsets_src = np.array([float(offset) for offset in offsets])
            else:
                z_offsets_src = np.arange(frames, dtype=np.float64)

            z_coords = origin[2] + z_offsets_src * slice_cos_src[2]

            if frames == 1:
                arr = arr.reshape((1, rows, cols))

            # Ensure z_coords is monotonically increasing for proper interpolation
            if len(z_coords) > 1 and z_coords[0] > z_coords[-1]:
                z_coords = z_coords[::-1]
                arr = arr[::-1, :, :]  # Flip along z-axis to match

            # Build coordinate grids for reference positions. y_positions_ref/x_positions_ref
            # are now (rows_ref, cols_ref) grids (see above), so broadcast them across frames
            # instead of np.meshgrid (which requires 1-D inputs).
            Z = np.broadcast_to(z_positions_ref[:, None, None], (frames_ref, rows_ref, cols_ref))
            Y = np.broadcast_to(y_positions_ref[None, :, :], (frames_ref, rows_ref, cols_ref))
            X = np.broadcast_to(x_positions_ref[None, :, :], (frames_ref, rows_ref, cols_ref))

            # C6 fix: do NOT clip source coordinates — let map_coordinates use cval=0
            # for out-of-bounds voxels (true zero-fill instead of edge-value leak)

            # Compute fractional indices using proper interpolation
            # np.interp maps physical z-coordinates to array indices
            z_idx_values = np.arange(len(z_coords), dtype=np.float64)
            z_idx = np.interp(Z.ravel(), z_coords, z_idx_values, left=-1, right=-1).reshape(Z.shape)
            y_idx = (Y - origin[1]) / pixel_spacing[0]
            x_idx = (X - origin[0]) / pixel_spacing[1]

            # Mark out-of-bounds z indices for proper zero-fill
            z_oob = (z_idx < 0)
            z_idx = np.clip(z_idx, 0, len(z_coords) - 1)

            coords = np.stack([z_idx, y_idx, x_idx], axis=0)
            resampled = map_coordinates(arr, coords, order=1, mode="constant", cval=0.0)
            resampled = resampled.reshape((frames_ref, rows_ref, cols_ref))

            # Zero out voxels that were outside the source z-range
            resampled[z_oob] = 0.0

            # Free meshgrid memory after use
            del Z, Y, X
            
            accumulated += resampled
            
            # Explicitly clear large arrays to free memory
            del ds
            del arr
            del resampled
            del coords
            
        except Exception as e:
            logger.error(f"Failed to resample dose {path}: {e}")
            # Continue with partial sum or raise? Continuing preserves robustness but might be inaccurate.
            # Given this is a summation, missing a component is critical failure.
            raise RuntimeError(f"Dose summation failed for {path}: {e}")

    # Final scaling and packing
    max_dose = float(np.nanmax(accumulated)) if accumulated.size else 0.0
    if max_dose > 1000:
        scaling_factor = 10.0
    elif max_dose > 100:
        scaling_factor = 100.0
    else:
        scaling_factor = 1000.0

    accumulated = np.nan_to_num(accumulated, nan=0.0)
    # C7 fix: use uint32 to match PixelRepresentation=0 (unsigned) in DICOM header
    accumulated_int = np.rint(accumulated * scaling_factor).astype("uint32")

    new_ds = copy.deepcopy(ds_ref)
    now = datetime.datetime.now()
    new_ds.SOPInstanceUID = generate_uid()
    new_ds.SeriesInstanceUID = generate_uid()
    new_ds.InstanceCreationDate = now.strftime("%Y%m%d")
    new_ds.InstanceCreationTime = now.strftime("%H%M%S")
    new_ds.SeriesDescription = f"Dose Sum ({len(dose_files)} plans)"
    if len(dose_files) > 1:
        new_ds.DoseSummationType = "PLAN_SUM"
    else:
        new_ds.DoseSummationType = str(getattr(new_ds, "DoseSummationType", "PLAN"))

    new_ds.BitsAllocated = 32
    new_ds.BitsStored = 32
    new_ds.HighBit = 31
    new_ds.PixelRepresentation = 0  # unsigned
    new_ds.DoseGridScaling = 1.0 / scaling_factor
    for tag in [(0x0028, 0x0106), (0x0028, 0x0107)]:
        if tag in new_ds:
            del new_ds[tag]
    if hasattr(new_ds, "PerFrameFunctionalGroupsSequence"):
        del new_ds.PerFrameFunctionalGroupsSequence

    fores = {str(getattr(d, "FrameOfReferenceUID", "")) for d in dose_headers if d and getattr(d, "FrameOfReferenceUID", None)}
    fores.discard("")
    if fores:
        if len(fores) > 1:
            logger.warning("Dose sum encountered multiple FrameOfReferenceUIDs: %s", sorted(fores))
        new_ds.FrameOfReferenceUID = sorted(fores)[0]

    new_ds.PixelData = accumulated_int.tobytes()

    # Update references to plans and source doses
    ref_plan_items: list[Dataset] = []
    sum_item = Dataset()
    sum_item.ReferencedSOPClassUID = str(getattr(plan_sum, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.481.5"))
    sum_item.ReferencedSOPInstanceUID = str(plan_sum.SOPInstanceUID)
    ref_plan_items.append(sum_item)
    for ds_plan in plan_datasets:
        item = Dataset()
        item.ReferencedSOPClassUID = str(getattr(ds_plan, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.481.5"))
        item.ReferencedSOPInstanceUID = str(ds_plan.SOPInstanceUID)
        ref_plan_items.append(item)
    new_ds.ReferencedRTPlanSequence = Sequence(ref_plan_items)

    ref_instances: list[Dataset] = []
    # Re-scan files to get SOP UIDs for reference sequence? 
    # We have source_dose_uids collected during the loop.
    # We can't use 'dose_datasets' list anymore as it doesn't exist.
    # We'll construct references from the collected UIDs, assuming standard class.
    
    for uid in source_dose_uids:
        ref_item = Dataset()
        ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2" # RT Dose Storage
        ref_item.ReferencedSOPInstanceUID = uid
        ref_instances.append(ref_item)
        
    if ref_instances:
        new_ds.ReferencedInstanceSequence = Sequence(ref_instances)

    comment = (
        f"Summed from {len(dose_files)} dose distributions on {now.isoformat()}"
    )
    if hasattr(new_ds, "DoseComment"):
        new_ds.DoseComment = comment
    else:
        setattr(new_ds, "DoseComment", comment)
    
    # We no longer return the full list of dose datasets, as we don't keep them in memory.
    # The caller logic needs to be adjusted if it expects this list.
    # Checking usage: caller uses it to extract source_dose_uids? 
    # Caller: "dose_sum_ds, dose_ds_list, source_dose_uids = _sum_doses_with_resample(...)"
    # The caller ignores 'dose_ds_list' in the line: "ref_dose_item.ReferencedSOPInstanceUID = ..."
    # Wait, looking at caller:
    # plan_sum_ds, plan_ds_list, source_plan_uids = _create_summed_plan(...)
    # dose_sum_ds, dose_ds_list, source_dose_uids = _sum_doses_with_resample(...)
    # Then it just saves dose_sum_ds.
    # So returning an empty list for the second element is fine/safer.

    return new_ds, [], source_dose_uids


def _index_rt_files(root: Path, patient_ids: Optional[Iterable[str]] = None) -> Dict[str, List[Path]]:
    """Index RTRECORD objects by PatientID using the DICOM Modality tag.

    RTRECORD files are not required to carry an ``RT`` filename prefix or a
    ``.dcm`` suffix. Discovery therefore reads every file returned by the
    symlink-aware scoped walk and trusts Modality, matching ``extract_rt``.
    """
    index = defaultdict(list)
    logger.info("Indexing RT treatment records in %s...", root)
    candidate_count = 0
    count = 0
    for base, _, files in _scoped_walk(root, patient_ids):
        for fn in files:
            candidate_count += 1
            path = Path(base) / fn
            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    specific_tags=["PatientID", "Modality"],
                    force=True,
                )
                if str(getattr(ds, "Modality", "") or "").strip().upper() != "RTRECORD":
                    continue
                pid = str(getattr(ds, "PatientID", "")).strip()
                if pid:
                    index[pid].append(path)
                    count += 1
            except Exception:
                continue
    logger.info(
        "Indexed %d RTRECORD objects for %d patients from %d input files",
        count,
        len(index),
        candidate_count,
    )
    return index


def _input_tree_diagnostics(root: Path) -> tuple[int, list[str]]:
    """Count visible files and un-followed symlinked directories for an error."""
    file_count = 0
    symlinked_dirs: list[str] = []
    try:
        for base, dirs, files in os.walk(root, followlinks=False):
            file_count += len(files)
            for name in dirs:
                path = Path(base) / name
                try:
                    if path.is_symlink():
                        symlinked_dirs.append(str(path))
                except OSError:
                    continue
    except OSError:
        return file_count, symlinked_dirs
    return file_count, symlinked_dirs


def _raise_if_empty_organize_discovery(
    config: PipelineConfig,
    ct_index: dict,
    plans: list,
    doses: list,
    structs: list,
) -> None:
    """Fail loudly instead of materializing a successful empty manifest."""
    if ct_index or plans or doses or structs:
        return
    file_count, symlinked_dirs = _input_tree_diagnostics(config.dicom_root)
    if file_count == 0 and not symlinked_dirs:
        return
    likely_cause = (
        " The root contains symlinked directories that were not followed. Set "
        f"{FOLLOW_INPUT_SYMLINKS_ENV}=1 for an operator-approved symlinked cohort."
        if symlinked_dirs and not follow_input_symlinks()
        else " Check that the input root is the intended DICOM tree and that its files are readable DICOM objects."
    )
    raise OrganizeDiscoveryError(
        f"Organize discovered zero supported DICOM objects under {config.dicom_root} "
        f"({file_count} visible file(s)).{likely_cause} Refusing to write an empty manifest."
    )


def referenced_ct_series_uids(rtstruct_path: "Path | str") -> set:
    """Return the CT SeriesInstanceUIDs referenced by an RTSTRUCT.

    Walks ReferencedFrameOfReferenceSequence -> RTReferencedStudySequence ->
    RTReferencedSeriesSequence (the structure-set -> planning-image link). Returns an
    empty set when the file is unreadable or carries no such references (e.g. CT-only
    cohorts with no structure set), in which case the caller falls back to a heuristic.
    """
    uids: set = set()
    try:
        ds = pydicom.dcmread(str(rtstruct_path), stop_before_pixels=True, force=True)
    except Exception as exc:
        logger.warning("Could not read RTSTRUCT %s for CT reference: %s", rtstruct_path, exc)
        return uids
    for ref_for in getattr(ds, "ReferencedFrameOfReferenceSequence", []) or []:
        for study in getattr(ref_for, "RTReferencedStudySequence", []) or []:
            for series in getattr(study, "RTReferencedSeriesSequence", []) or []:
                uid = str(getattr(series, "SeriesInstanceUID", "") or "").strip()
                if uid:
                    uids.add(uid)
    return uids


def select_course_ct_series(
    ct_index: dict,
    patient_id: str,
    struct_source_path: "Path | str | None",
    course_study: Optional[str],
    *,
    require_reference: bool = False,
) -> Tuple[Optional[list], str]:
    """Select the planning CT series for a course and report how it was chosen.

    Returns ``(series_instances_or_None, status)``. For auto-OAR DVH the planning CT must
    be the CT the structure set was drawn on, i.e. the series referenced by the course's
    PRIMARY RTSTRUCT -- NOT merely the largest series in the study (verified: referenced
    CTs are frequently smaller than the largest series). Resolution order:

      - references resolve to exactly one indexed CT series -> that series ("referenced")
      - references resolve to several indexed CT series -> deterministic tie-break:
        largest by slice count, then lowest series_uid ("referenced_multi")
      - RTSTRUCT HAS references but none resolve to an indexed CT series -> FAIL CLOSED:
        (None, "unresolved_reference"); the caller skips per-course CT so we never silently
        segment the wrong/largest CT against mismatched structures/dose
      - no RTSTRUCT / no references at all -> legacy largest-in-course-study heuristic via
        pick_primary_series ("fallback_largest"), or (None, "none") if unavailable

    The struct source path must be the ORIGINAL RTSTRUCT, not the per-course RS.dcm copy:
    copy-manager SOP dedup may skip materialising the root RS.dcm.
    """
    patient_studies = ct_index.get(patient_id) or {}

    refs: set = set()
    if struct_source_path:
        refs = referenced_ct_series_uids(struct_source_path)
    elif require_reference:
        return None, "missing_reference"

    if refs:
        resolved: list = []
        for series_map in patient_studies.values():
            for uid in refs:
                series = series_map.get(uid)
                if series:
                    resolved.append((uid, series))
        if len(resolved) == 1:
            return resolved[0][1], "referenced"
        if len(resolved) > 1:
            # Deterministic tie-break: most slices first, then lowest series_uid.
            resolved.sort(key=lambda kv: (-len(kv[1]), kv[0]))
            return resolved[0][1], "referenced_multi"
        # References exist but none are indexed -> fail closed (no silent wrong-CT selection).
        return None, "unresolved_reference"

    if struct_source_path and require_reference:
        return None, "missing_reference"

    # No structure-set references available -> legacy largest-in-study heuristic for direct callers.
    if course_study and course_study in patient_studies:
        series = pick_primary_series(patient_studies[course_study])
        if series:
            return series, "fallback_largest"
    return None, "none"


def _clear_course_ct_outputs(course_dirs) -> None:
    """Remove per-course CT-derived outputs (CT DICOM, CT NIfTI, auto-OAR TotalSegmentator).

    Called when NO planning CT is selected for a course (fail-closed unresolved reference, or
    no CT available). Without this, a resume/re-run over existing outputs could keep a prior
    WRONG CT in ``DICOM/CT`` and segment it into a wrong per-course OAR DVH for a course that
    must be skipped -- and resume hydration / metadata generation would treat the stale CT,
    NIfTI, and masks as valid. ``copy_ct_series`` already purges ``DICOM/CT`` on the success
    path, so this only matters for the no-CT branch.
    """
    # FAIL-SAFE: a course that previously produced valid CT-derived outputs but now
    # resolves to "no CT" is far more often a source/config error (wrong --dicom-root,
    # transient source outage) than a genuine skip. Emptying populated DICOM/CT, NIfTI,
    # and TotalSegmentator folders in that case destroys good data irreversibly (only the
    # raw source can rebuild it). By default we REFUSE to purge a populated dir here and
    # warn loudly; set RTPIPELINE_ALLOW_DESTRUCTIVE_CT_CLEAR=1 to force the aggressive
    # clear when a genuine wrong-CT must be scrubbed.
    force = os.environ.get(
        "RTPIPELINE_ALLOW_DESTRUCTIVE_CT_CLEAR", ""
    ).strip().lower() in ("1", "true", "yes")
    for d in (
        getattr(course_dirs, "dicom_ct", None),
        getattr(course_dirs, "nifti", None),
        getattr(course_dirs, "segmentation_totalseg", None),
    ):
        if d is None:
            continue
        p = Path(d)
        try:
            if p.is_dir() and any(p.iterdir()) and not force:
                logger.warning(
                    "_clear_course_ct_outputs: refusing to purge POPULATED %s for a "
                    "no-CT course (fail-safe against source/config errors, not a real "
                    "skip). Set RTPIPELINE_ALLOW_DESTRUCTIVE_CT_CLEAR=1 to override.", p)
                continue
            _clear_dir(p)
        except Exception as exc:
            logger.warning("Failed to clear stale per-course CT output %s: %s", d, exc)


# ---------------------------------------------------------------------------
# Per-patient organize checkpointing
#
# Organize is one Snakemake checkpoint job over the whole cohort. An
# interruption at 99% therefore re-walked every DICOM header, which on a
# 154-patient / 415,562-instance cohort cost hours before any new work began.
#
# Two records make a patient skippable on resume. The EXPECTED record lists the
# course keys discovered for that patient, written as soon as grouping is known.
# A DONE record is written as each course finishes. A patient is complete only
# when every expected course has a done record, which is what makes it safe to
# drop that patient from the discovery scope entirely, so its headers are never
# re-read. A partially finished patient keeps its finished courses (they hydrate
# cheaply) and re-processes only the rest.
#
# Every record is published with os.replace, so an interruption mid-write leaves
# either the old record or the new one, never a truncated one.
# ---------------------------------------------------------------------------

def _organize_checkpoint_dir(config: PipelineConfig) -> Path:
    return config.output_root / "_COURSES" / "patients"


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _record_expected_courses(config: PipelineConfig, patient_id: str, course_keys: List[str]) -> None:
    try:
        _write_json_atomic(
            _organize_checkpoint_dir(config) / f"{patient_id}.expected.json",
            {"patient": patient_id, "courses": sorted(set(course_keys))},
        )
    except Exception as exc:
        logger.debug("Could not record expected courses for %s: %s", patient_id, exc)


def _record_course_done(config: PipelineConfig, patient_id: str, course_key: str, course_dir: Path) -> None:
    try:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(course_key))
        _write_json_atomic(
            _organize_checkpoint_dir(config) / patient_id / f"{safe}.done.json",
            {"patient": patient_id, "course_key": str(course_key), "course_dir": str(course_dir)},
        )
    except Exception as exc:
        logger.debug("Could not record completed course %s/%s: %s", patient_id, course_key, exc)


def _completed_patients(config: PipelineConfig) -> Dict[str, List[dict]]:
    """Patients whose every discovered course already has a completion record."""
    root = _organize_checkpoint_dir(config)
    if not root.is_dir():
        return {}
    complete: Dict[str, List[dict]] = {}
    for expected_file in sorted(root.glob("*.expected.json")):
        try:
            expected = json.loads(expected_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        patient_id = str(expected.get("patient") or "")
        wanted = [str(c) for c in (expected.get("courses") or [])]
        if not patient_id or not wanted:
            continue
        done_dir = root / patient_id
        if not done_dir.is_dir():
            continue
        done: Dict[str, dict] = {}
        for done_file in done_dir.glob("*.done.json"):
            try:
                entry = json.loads(done_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            key = str(entry.get("course_key") or "")
            if key:
                done[key] = entry
        if all(key in done for key in wanted):
            complete[patient_id] = [done[key] for key in wanted]
    return complete


def organize_and_merge(config: PipelineConfig) -> List[CourseOutput]:
    """End-to-end RT organization and course merging according to config."""
    config.ensure_dirs()

    # Initialize copy manager for optimized DICOM copying
    copy_config = DicomCopyConfig(
        dedup_by_sop_uid=getattr(config, "dicom_copy_dedup_by_sop_uid", True),
        use_hardlinks=getattr(config, "dicom_copy_use_hardlinks", True),
        verify_checksum=getattr(config, "dicom_copy_verify_checksum", False),
        cache_headers=getattr(config, "dicom_copy_cache_headers", True),
        cache_dir=config.output_root / "_CACHE",
    )
    copy_manager = DicomCopyManager(copy_config, config.output_root)
    logger.info(
        "Copy manager initialized: dedup=%s, hardlinks=%s, verify=%s, cache=%s",
        copy_config.dedup_by_sop_uid,
        copy_config.use_hardlinks,
        copy_config.verify_checksum,
        copy_config.cache_headers,
    )

    # Cohort scope for organize-stage discovery: when set (by the CLI from active
    # course/patient filters), only these patient directories are walked instead
    # of the entire dicom_root. None => walk everything (unchanged behaviour).
    scope_ids = list(getattr(config, "discover_patient_ids", []) or []) or None
    if scope_ids:
        logger.info("Organize discovery scoped to %d cohort patient(s)", len(scope_ids))

    # Drop already-complete patients from discovery so their headers are never
    # re-read. Only patients whose every discovered course has a completion
    # record qualify; a partially finished patient stays in scope and keeps its
    # finished courses through hydration.
    resumed_patients: Dict[str, List[dict]] = {}
    if getattr(config, "resume", False):
        checkpoint_candidates = _completed_patients(config)
        for checkpoint_patient, checkpoint_entries in checkpoint_candidates.items():
            valid = True
            for checkpoint_entry in checkpoint_entries:
                checkpoint_dir = Path(str(checkpoint_entry.get("course_dir") or ""))
                checkpoint_key = str(checkpoint_entry.get("course_key") or checkpoint_dir.name)
                if not checkpoint_dir.is_dir() or _hydrate_existing_course(
                    checkpoint_patient,
                    checkpoint_key,
                    checkpoint_dir,
                ) is None:
                    valid = False
                    break
            if valid:
                resumed_patients[checkpoint_patient] = checkpoint_entries
            else:
                logger.info(
                    "Organize resume: checkpoint for patient %s is incomplete under current output invariants; reprocessing.",
                    checkpoint_patient,
                )
        if resumed_patients:
            if scope_ids is None:
                try:
                    all_ids = [d.name for d in config.dicom_root.iterdir() if d.is_dir()]
                except OSError:
                    all_ids = []
                scope_ids = all_ids or None
            if scope_ids is not None:
                remaining = [pid for pid in scope_ids if pid not in resumed_patients]
                logger.info(
                    "Organize resume: %d patient(s) already complete, %d remaining; "
                    "their DICOM headers will not be re-read.",
                    len(resumed_patients),
                    len(remaining),
                )
                scope_ids = remaining

    # Index CTs, extract RT sets, link and group into courses.
    # Header-read concurrency honours the user's --max-workers / config cap (so
    # e.g. --max-workers 1 for debugging really uses a single thread) instead of
    # the static DEFAULT_INDEX_WORKERS the functions fall back to for direct callers.
    index_workers = config.effective_workers()
    ct_index = index_ct_series(config.dicom_root, scope_ids, max_workers=index_workers)
    patient_series_layout = _looks_like_patient_series_layout(config.dicom_root)
    if patient_series_layout:
        logger.info(
            "Detected a patient/series directory shape; scanning RT and registration objects because "
            "directory shape alone does not establish a CT-only cohort"
        )
    rt_file_index = _index_rt_files(config.dicom_root, scope_ids)
    plans, doses, structs = extract_rt(config.dicom_root, scope_ids, max_workers=index_workers)
    record_paths = [path for paths in rt_file_index.values() for path in paths]
    delivery_reference_audit = _delivery_reference_audit(
        record_paths,
        [plan.sop_instance_uid for plan in plans],
    )
    plan_uids_by_patient: Dict[str, list[str]] = defaultdict(list)
    for indexed_plan in plans:
        plan_uids_by_patient[indexed_plan.patient_id].append(indexed_plan.sop_instance_uid)
    delivery_reference_audit_by_patient = {
        patient_id: _delivery_reference_audit(
            paths,
            plan_uids_by_patient.get(patient_id, []),
            log_warnings=False,
        )
        for patient_id, paths in rt_file_index.items()
    }
    logger.info(
        "Organize delivery-reference summary: %d RTRECORD(s), %d unresolved plan reference(s), %d unresolved plan UID(s)",
        len(record_paths),
        delivery_reference_audit["unresolved_reference_count"],
        len(delivery_reference_audit["unresolved_plan_uids"]),
    )
    linked_sets = link_rt_sets(plans, doses, structs)
    courses = group_by_course(linked_sets, config.merge_criteria, config.max_days_between_plans)
    if not courses and not (resumed_patients and scope_ids == []):
        _raise_if_empty_organize_discovery(config, ct_index, plans, doses, structs)
    series_index, registrations_index, series_meta = _index_series_and_registrations(
        config.dicom_root,
        scope_ids,
        max_workers=index_workers,
    )

    planned_struct_uids = {
        plan.referenced_struct_sop for plan in plans if plan.referenced_struct_sop
    }
    linked_struct_uids = {
        item.struct.sop_instance_uid for item in linked_sets if item.struct is not None
    }
    unsupported_target_structs = sorted(
        struct.sop_instance_uid
        for struct in structs
        if struct.sop_instance_uid in planned_struct_uids
        and has_target_volumes(struct.roi_names)
        and struct.sop_instance_uid not in linked_struct_uids
    )
    if unsupported_target_structs:
        logger.error(
            "Course inclusion QC: %d target-bearing plan-referenced RTSTRUCT(s) had no resolvable "
            "RTDOSE→RTPLAN link and cannot be fabricated as courses. Examples: %s",
            len(unsupported_target_structs),
            ", ".join(unsupported_target_structs[:10]),
        )

    target_valid_courses: Dict[Tuple[str, str], List[LinkedSet]] = {}
    for course_identity, items in courses.items():
        patient_id, course_key = course_identity
        plan_paths = list(dict.fromkeys(item.plan.path for item in items))
        dose_paths = list(
            dict.fromkeys(item.dose.path for item in items if item.dose is not None)
        )
        try:
            struct_path = _authoritative_structure_source(items)
        except CourseTargetQCError as exc:
            logger.error("COURSE TARGET QC GATE: %s. Excluding this invalid course.", exc)
            continue
        exact_struct = next(
            (
                item.struct
                for item in items
                if item.struct is not None and item.struct.path == struct_path
            ),
            items[0].struct if items and items[0].struct is not None else None,
        )
        targets = target_volume_names(exact_struct.roi_names if exact_struct is not None else [])
        if not targets:
            if dose_paths:
                try:
                    validate_course_target_qc(
                        str(patient_id),
                        str(course_key),
                        plan_paths,
                        dose_paths,
                        struct_path,
                    )
                except CourseTargetQCError as exc:
                    logger.error("COURSE TARGET QC GATE: %s. Excluding this invalid course.", exc)
            else:
                logger.error(
                    "Course inclusion QC: excluding plan-only course %s/%s because its authoritative structure set has no target volumes",
                    patient_id,
                    course_key,
                )
            continue
        if not dose_paths:
            logger.warning(
                "Course inclusion QC: retaining target-bearing plan-only course %s/%s because no RTDOSE resolves to its plan(s)",
                patient_id,
                course_key,
            )
        logger.info(
            "Course target QC passed for %s/%s with %d target volume(s)",
            patient_id,
            course_key,
            len(targets),
        )
        target_valid_courses[course_identity] = items
    courses = target_valid_courses

    outputs: List[CourseOutput] = []

    existing_names: Dict[str, set[str]] = defaultdict(set)

    def _course_start(items: List[LinkedSet]) -> Optional[datetime.datetime]:
        dates: List[datetime.datetime] = []
        for it in items:
            dt = parse_date(it.plan.plan_date)
            if dt is not None:
                dates.append(dt)
        if not dates:
            return None
        return min(dates)

    raw_entries: List[Tuple[str, str, List[LinkedSet], Optional[datetime.datetime]]] = []
    for (pid, raw_key), items in courses.items():
        raw_entries.append((str(pid), str(raw_key), items, _course_start(items)))

    raw_entries.sort(
        key=lambda entry: (
            entry[0],
            entry[3].strftime("%Y%m%d") if entry[3] else "ZZZZZZZZ",
            entry[1],
        )
    )

    # Persist what was discovered per patient before any of it is processed, so
    # a later resume can tell a finished patient from a partly finished one.
    _expected_by_patient: Dict[str, List[str]] = defaultdict(list)
    for _pid, _raw_key, _items, _start in raw_entries:
        _expected_by_patient[_pid].append(_raw_key)
    for _pid, _keys in _expected_by_patient.items():
        _record_expected_courses(config, _pid, _keys)

    course_tasks: List[Tuple[str, str, List[LinkedSet], Dict[str, Optional[str] | Optional[datetime.datetime]]]] = []
    for pid, raw_key, items, start_dt in raw_entries:
        start_token = start_dt.strftime("%Y-%m") if start_dt else None
        dir_name = course_dir_name(start_token, raw_key, existing_names[pid])
        meta: Dict[str, Optional[str] | Optional[datetime.datetime]] = {
            "dir_name": dir_name,
            "start_token": start_token,
            "start_iso": start_dt.strftime("%Y-%m-%d") if start_dt else None,
            "start_dt": start_dt,
        }
        course_tasks.append((pid, raw_key, items, meta))

    def _process_course(
        patient_id: str,
        course_key_raw: str,
        items: list[LinkedSet],
        meta: Dict[str, Optional[str] | Optional[datetime.datetime]],
    ) -> CourseOutput:
        course_key = "".join(ch if ch.isalnum() else "_" for ch in str(course_key_raw))[:64]
        items_sorted = sorted(items, key=lambda it: it.plan.plan_date or "")
        authoritative_struct_path = _authoritative_structure_source(items_sorted)
        course_for_uids = {it.frame_of_reference_uid for it in items_sorted if it.frame_of_reference_uid}

        patient_root = config.output_root / patient_id
        ensure_dir(patient_root)
        course_id = str(meta.get("dir_name") or course_key)
        course_dir = patient_root / course_id
        course_dirs = build_course_dirs(course_dir)
        course_dirs.ensure()

        if config.resume and course_dir.exists():
            hydrated = _hydrate_existing_course(patient_id, course_key, course_dir, meta)
            if hydrated:
                return hydrated

        primary_nifti: Optional[Path] = None
        related_outputs: List[Path] = []
        seen_related: set[Path] = set()
        course_ct_series_uids: set[str] = set()
        planning_ct_series_uid: str | None = None

        rp_dst = course_dir / "RP.dcm"
        rd_dst = course_dir / "RD.dcm"
        rs_dst = course_dir / "RS.dcm"

        plan_paths: List[Path] = []
        dose_paths: List[Path] = []
        struct_candidates: List[Path] = []

        for it in items_sorted:
            if it.plan.path not in plan_paths:
                plan_paths.append(it.plan.path)
            if it.dose is not None and it.dose.path not in dose_paths:
                dose_paths.append(it.dose.path)
            if it.struct and it.struct.path not in struct_candidates:
                struct_candidates.append(it.struct.path)

        copied_plan_paths: dict[Path, Path] = {}
        copied_dose_paths: dict[Path, Path] = {}
        copied_struct_paths: dict[Path, Path] = {}
        copied_record_paths: dict[Path, Path] = {}
        for src in plan_paths:
            copied_plan_paths[src] = _copy_into(
                src, course_dirs.dicom_rtplan, copy_manager=copy_manager
            )
        for src in dose_paths:
            copied_dose_paths[src] = _copy_into(
                src, course_dirs.dicom_rtdose, copy_manager=copy_manager
            )
        for src in struct_candidates:
            copied_struct_paths[src] = _copy_into(
                src, course_dirs.dicom_rtstruct, copy_manager=copy_manager
            )
        for src in rt_file_index.get(patient_id, []):
            copied_record_paths[src] = _copy_into(
                src, course_dirs.dicom_related / "RTRECORD", copy_manager=copy_manager
            )

        total_rx: float | None = None
        plan_sop_uid: Optional[str] = None
        dose_sop_uid: Optional[str] = None
        source_plan_uids: list[str] = []
        source_dose_uids: list[str] = []
        delivery_plan_paths: list[Path] = list(plan_paths)
        delivery_dose_paths: list[Path] = list(dose_paths)
        dose_classification_info: dict = {}
        selected_plans: list[Path] = []
        selected_doses: list[Path] = []

        # Classify doses from explicit plan references and available delivery records.
        if plan_paths and dose_paths:
            treatment_record_paths = rt_file_index.get(patient_id, [])
            if treatment_record_paths:
                dose_classification = _classify_doses(
                    plan_paths=plan_paths,
                    dose_paths=dose_paths,
                    max_total_dose_gy=float(config.max_total_dose_gy),
                    treatment_record_paths=treatment_record_paths,
                )
            else:
                # Preserve the three-argument call for integrations that replace
                # the classifier and for cohorts with no treatment records.
                dose_classification = _classify_doses(
                    plan_paths=plan_paths,
                    dose_paths=dose_paths,
                    max_total_dose_gy=float(config.max_total_dose_gy),
                )

            logger.info(
                "Dose classification for %s/%s: %s (%s)",
                patient_id, course_id, dose_classification.classification, dose_classification.reason
            )

            # Log any warnings
            for warn in dose_classification.warnings:
                logger.warning("Dose classification warning: %s", warn)

            # Store classification info for metadata
            dose_classification_info = {
                "classification": dose_classification.classification,
                "reason": dose_classification.reason,
                "selected_doses": [str(p) for p in dose_classification.selected_doses],
                "excluded_doses": [str(p) for p in dose_classification.excluded_doses],
                "should_sum": dose_classification.should_sum,
                "warnings": dose_classification.warnings,
            }

            selected_doses = list(dose_classification.selected_doses)
            selected_plans = list(
                dose_classification.selected_plans
                or _plan_paths_for_doses(plan_paths, selected_doses)
            )
            if not selected_plans:
                logger.warning(
                    "Dose classification for %s/%s (%s) selected no delivered plan; "
                    "no treatment dose grid will be emitted from %d candidate plan(s)",
                    patient_id, course_id, dose_classification.classification, len(plan_paths),
                )
                selected_doses = []
            # Recompute total_rx from the FINAL resolved selected_plans (whichever of
            # the three sources above populated it) rather than only
            # dose_classification.selected_plans: a resolved-multi-plan sum via
            # _plan_paths_for_doses(), or the ITT fallback above, previously got no Rx
            # at all because this was gated on the classifier's own (possibly empty)
            # selected_plans instead of the variable actually used downstream.
            if selected_plans:
                total_rx = _infer_rx_from_plan_paths(
                    selected_plans,
                    sum_all=bool(dose_classification.should_sum and len(selected_doses) > 1),
                )
            delivery_plan_paths = list(selected_plans)
            delivery_dose_paths = list(selected_doses)

            if dose_classification.should_sum and len(selected_doses) > 1:
                # Primary + boost case: sum the selected doses
                logger.info("Summing %d doses (primary + boost)", len(selected_doses))
                plan_sum_ds, plan_ds_list, source_plan_uids = _create_summed_plan(
                    selected_plans, total_rx
                )
                dose_sum_ds, dose_ds_list, source_dose_uids = _sum_doses_with_resample(
                    selected_doses, plan_sum_ds, plan_ds_list
                )
                if dose_classification.classification == "beam_doses_summed_to_plan":
                    dose_sum_ds.DoseSummationType = "PLAN"
                    dose_sum_ds.SeriesDescription = f"Plan Dose Sum ({len(selected_doses)} beams)"

                ref_dose_item = Dataset()
                ref_dose_item.ReferencedSOPClassUID = str(getattr(dose_sum_ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.481.2"))
                ref_dose_item.ReferencedSOPInstanceUID = str(dose_sum_ds.SOPInstanceUID)
                plan_sum_ds.ReferencedDoseSequence = Sequence([ref_dose_item])

                plan_sum_ds.save_as(str(rp_dst))
                dose_sum_ds.save_as(str(rd_dst))

                plan_sop_uid = str(plan_sum_ds.SOPInstanceUID)
                dose_sop_uid = str(dose_sum_ds.SOPInstanceUID)

            elif len(selected_doses) == 1:
                # Single dose selected (PLAN_SUM, replan ITT, or ambiguous)
                _safe_copy(selected_doses[0], rd_dst, copy_manager=copy_manager)
                try:
                    ds_dose_single = pydicom.dcmread(str(selected_doses[0]), stop_before_pixels=True)
                    dose_sop_uid = str(getattr(ds_dose_single, "SOPInstanceUID", "") or None)
                except Exception:
                    dose_sop_uid = None
                if dose_sop_uid:
                    source_dose_uids.append(dose_sop_uid)

                # Copy the first plan (or all selected plans for a PLAN_SUM case)
                if len(selected_plans) == 1:
                    _safe_copy(selected_plans[0], rp_dst, copy_manager=copy_manager)
                    try:
                        ds_plan_single = pydicom.dcmread(str(selected_plans[0]), stop_before_pixels=True)
                        plan_sop_uid = str(getattr(ds_plan_single, "SOPInstanceUID", "") or None)
                    except Exception:
                        plan_sop_uid = None
                    if plan_sop_uid:
                        source_plan_uids.append(plan_sop_uid)
                else:
                    # Multiple plans but single dose (e.g., TPS PLAN_SUM)
                    # Create a summed plan to reference them
                    plan_sum_ds, plan_ds_list, source_plan_uids = _create_summed_plan(
                        selected_plans, total_rx
                    )
                    plan_sum_ds.save_as(str(rp_dst))
                    plan_sop_uid = str(plan_sum_ds.SOPInstanceUID)

            else:
                # Keep one plan artifact for treatment intent, but do not call it
                # delivered membership and do not emit an RTDOSE grid.
                logger.warning("No delivered doses selected after classification for %s/%s", patient_id, course_id)
                if rd_dst.exists():
                    rd_dst.unlink()
                if plan_paths:
                    intent_plan = _earliest_dated_plan_path(items_sorted, plan_paths)
                    selected_plans = [intent_plan]
                    delivery_plan_paths = [intent_plan]
                    delivery_dose_paths = []
                    _safe_copy(intent_plan, rp_dst, copy_manager=copy_manager)
                    try:
                        ds_plan_single = pydicom.dcmread(str(intent_plan), stop_before_pixels=True)
                        plan_sop_uid = str(getattr(ds_plan_single, "SOPInstanceUID", "") or None)
                    except Exception:
                        plan_sop_uid = None
                    if plan_sop_uid:
                        source_plan_uids.append(plan_sop_uid)

        else:
            # Fallback: missing plans or doses
            if plan_paths:
                # A plan without a resolvable RTDOSE remains an explicit
                # plan-only course. Preserve its membership for downstream
                # consumers while leaving the dose grid absent.
                selected_plans = [plan_paths[0]]
                delivery_plan_paths = [plan_paths[0]]
                delivery_dose_paths = []
                _safe_copy(plan_paths[0], rp_dst, copy_manager=copy_manager)
                try:
                    ds_plan_single = pydicom.dcmread(str(plan_paths[0]), stop_before_pixels=True)
                    plan_sop_uid = str(getattr(ds_plan_single, "SOPInstanceUID", "") or None)
                except Exception:
                    plan_sop_uid = None
                if plan_sop_uid:
                    source_plan_uids.append(plan_sop_uid)
            if dose_paths:
                first_dose_meta = _extract_dose_metadata(dose_paths[0])
                if (
                    selected_plans
                    and str(first_dose_meta.get("summation_type") or "").upper() in {"PLAN", "PLAN_SUM"}
                ):
                    selected_doses = [dose_paths[0]]
                    _safe_copy(dose_paths[0], rd_dst, copy_manager=copy_manager)
                elif rd_dst.exists():
                    rd_dst.unlink()
                try:
                    ds_dose_single = pydicom.dcmread(str(dose_paths[0]), stop_before_pixels=True)
                    dose_sop_uid = (
                        str(getattr(ds_dose_single, "SOPInstanceUID", "") or None)
                        if selected_doses
                        else None
                    )
                except Exception:
                    dose_sop_uid = None
                if dose_sop_uid:
                    source_dose_uids.append(dose_sop_uid)

        if total_rx is None and rp_dst.exists():
            try:
                total_rx = infer_plan_rx_gy(pydicom.dcmread(str(rp_dst), stop_before_pixels=True))
            except Exception:
                total_rx = None

        # A synthesized artifact must retain explicit source membership even when
        # an integration replacement returns no provenance list.
        if selected_plans and not source_plan_uids:
            source_plan_uids = [
                str(getattr(pydicom.dcmread(str(path), stop_before_pixels=True), "SOPInstanceUID", ""))
                for path in selected_plans
            ]
            source_plan_uids = [uid for uid in source_plan_uids if uid]
        if selected_doses and not source_dose_uids:
            source_dose_uids = [
                str(getattr(pydicom.dcmread(str(path), stop_before_pixels=True), "SOPInstanceUID", ""))
                for path in selected_doses
            ]
            source_dose_uids = [uid for uid in source_dose_uids if uid]

        course_study = items_sorted[0].ct_study_uid if items_sorted else None
        struct_path: Optional[Path] = authoritative_struct_path
        if struct_path is None and struct_candidates:
            logger.error(
                "Authoritative RTSTRUCT could not be resolved for %s/%s; refusing reference-free CT fallback",
                patient_id,
                course_id,
            )
        if struct_path:
            _safe_copy(struct_path, rs_dst, copy_manager=copy_manager)
        authoritative_rtstruct_uid: str | None = None
        if rs_dst.exists():
            try:
                ds_struct = pydicom.dcmread(str(rs_dst), stop_before_pixels=True, force=True)
                authoritative_rtstruct_uid = str(
                    getattr(ds_struct, "SOPInstanceUID", "") or ""
                ) or None
            except Exception:
                authoritative_rtstruct_uid = None
        planning_ct_referenced_series_uids = sorted(
            referenced_ct_series_uids(struct_path) if struct_path else set()
        )

        series, ct_select_status = select_course_ct_series(
            ct_index,
            patient_id,
            struct_path,
            course_study,
            require_reference=True,
        )
        if series:
            eligible, image_class, exclusion_reason = _classify_organize_ct_series(
                list(series),
                is_planning_ct=True,
            )
            if not eligible:
                logger.error(
                    "Planning CT classifier excluded %s/%s series %s: class=%s reason=%s",
                    patient_id,
                    course_id,
                    getattr(series[0], "series_uid", "<missing>"),
                    image_class,
                    exclusion_reason,
                )
                series = None
                ct_select_status = f"classifier_excluded:{exclusion_reason or image_class}"
        if series:
            first_inst = series[0] if series else None
            if first_inst is not None and getattr(first_inst, 'series_uid', None):
                planning_ct_series_uid = str(first_inst.series_uid)
                course_ct_series_uids.add(planning_ct_series_uid)
            logger.info(
                "Per-course planning CT for %s (%s): %s -> series ...%s (%d slices)",
                patient_id, course_id, ct_select_status,
                str(getattr(first_inst, 'series_uid', '') or '')[-12:], len(series),
            )
            copy_ct_series(series, course_dirs.dicom_ct, copy_manager=copy_manager)
            try:
                primary_nifti = _ensure_ct_nifti(
                    config,
                    course_dirs.dicom_ct,
                    course_dirs.nifti,
                    force=bool(config.resume),
                )
            except Exception as exc:
                logger.warning("CT NIfTI conversion failed for %s: %s", course_dir, exc)
        else:
            if ct_select_status == "unresolved_reference":
                logger.warning(
                    "Per-course CT fail-closed for %s (%s): primary RTSTRUCT references a CT series "
                    "absent from the indexed CT; skipping per-course CT/segmentation.",
                    patient_id, course_id,
                )
            # No planning CT -> remove any stale per-course CT/NIfTI/segmentation from a prior run.
            _clear_course_ct_outputs(course_dirs)

        if registrations_index.get(patient_id):
            for reg in registrations_index.get(patient_id, []):
                reg_for = reg.get('for_uids', set())
                if reg_for and course_for_uids and not course_for_uids.intersection(reg_for):
                    continue
                reg_path = Path(reg.get('path'))
                if reg_path.exists() and reg_path not in seen_related:
                    related_outputs.append(_copy_into(reg_path, course_dirs.dicom_related / "REG", copy_manager=copy_manager))
                    seen_related.add(reg_path)
                for series_uid in reg.get('referenced_series', set()):
                    if series_uid in course_ct_series_uids:
                        continue
                    series_paths = series_index.get((patient_id, series_uid), [])
                    if not series_paths:
                        continue
                    modality = str(series_meta.get((patient_id, series_uid), {}).get("modality", "")).upper()
                    if modality == "MR":
                        dest_parent = course_dirs.dicom_mr
                        fallback = "mr"
                    else:
                        dest_parent = course_dirs.dicom_related
                        fallback = "series"
                    dest_dir = dest_parent / _sanitize_name(series_uid, fallback)
                    for src in series_paths:
                        if src not in seen_related and src.exists():
                            related_outputs.append(_copy_into(src, dest_dir, copy_manager=copy_manager))
                            seen_related.add(src)

        for series_subdir in sorted(p for p in course_dirs.dicom_related.iterdir() if p.is_dir() and p.name != "REG"):
            try:
                target_name = _derive_nifti_name(series_subdir)
                meta_path = course_dirs.nifti / f"{target_name}.metadata.json"
                if meta_path.exists() and not config.resume:
                    continue
                tmp_out = course_dirs.nifti / f".tmp_{series_subdir.name}"
                tmp_out.mkdir(parents=True, exist_ok=True)
                generated = run_dcm2niix(config, series_subdir, tmp_out)
                if generated is None:
                    shutil.rmtree(tmp_out, ignore_errors=True)
                    continue
                target_path = course_dirs.nifti / f"{target_name}.nii.gz"
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(generated), str(target_path))
                metadata = _collect_series_metadata(series_subdir)
                metadata.update(
                    {
                        "nifti_path": str(target_path),
                        "source_directory": str(series_subdir),
                        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                    }
                )
                meta_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
                shutil.rmtree(tmp_out, ignore_errors=True)
            except Exception as exc:
                logger.debug("Failed converting related series %s: %s", series_subdir, exc)

        delivery_summary = _calculate_delivery_summary(
            plan_paths,
            rt_file_index.get(patient_id, []),
            selected_plan_paths=delivery_plan_paths,
            selected_dose_paths=delivery_dose_paths,
            reference_audit=delivery_reference_audit_by_patient.get(patient_id),
        )
        per_plan_delivery = _per_plan_delivery_contract(
            plan_paths,
            rt_file_index.get(patient_id, []),
            selected_plans,
            copied_plan_paths,
            copied_record_paths,
        )
        delivery_by_uid = {
            str(item.get("plan_sop_uid") or ""): item
            for item in per_plan_delivery
        }
        selected_plan_contract: list[dict[str, object]] = []
        for selected_path in selected_plans:
            meta_plan = _plan_evidence(selected_path)
            uid = str(meta_plan.get("sop_uid") or "")
            delivery_item = delivery_by_uid.get(uid, {})
            selected_plan_contract.append(
                {
                    "path": str(copied_plan_paths[selected_path]),
                    "sop_instance_uid": uid,
                    "prescribed_dose_gy": meta_plan.get("total_rx_gy"),
                    "planned_fraction_count": int(meta_plan.get("fractions_planned") or 0) or None,
                    "delivered_record_count": int(delivery_item.get("delivered_record_count") or 0),
                    "delivered_fraction_count": int(delivery_item.get("delivered_fraction_count") or 0),
                    "treatment_dates": list(delivery_item.get("treatment_dates") or []),
                }
            )
        selected_dose_contract: list[dict[str, object]] = []
        for selected_path in selected_doses:
            meta_dose = _extract_dose_metadata(selected_path)
            selected_dose_contract.append(
                {
                    "path": str(copied_dose_paths[selected_path]),
                    "sop_instance_uid": str(meta_dose.get("sop_uid") or ""),
                    "dose_summation_type": str(meta_dose.get("summation_type") or "").upper(),
                    "referenced_plan_uids": list(meta_dose.get("referenced_plan_uids") or []),
                }
            )
        dose_threshold_gy = float(config.max_total_dose_gy)
        dose_plausibility = _dose_plausibility(
            float(total_rx) if total_rx is not None else None,
            delivery_summary["delivered_dose_gy"],
            dose_threshold_gy,
        )
        if dose_plausibility["dose_plausibility_warning"]:
            logger.warning(
                "PLAUSIBILITY WARNING: %s/%s prescribed or delivered dose exceeds %.1f Gy "
                "(prescribed=%s, delivered=%s)",
                patient_id,
                course_id,
                dose_threshold_gy,
                total_rx,
                delivery_summary["delivered_dose_gy"],
            )
        return CourseOutput(
            patient_id=patient_id,
            course_key=course_key,
            course_id=course_id,
            course_start=meta.get("start_token") if isinstance(meta.get("start_token"), (str, type(None))) else None,
            dirs=course_dirs,
            rp_path=rp_dst,
            rd_path=rd_dst,
            rs_path=rs_dst if rs_dst.exists() else None,
            primary_nifti=Path(primary_nifti) if primary_nifti else None,
            related_dicom=related_outputs,
            total_prescription_gy=total_rx or None,
            plan_sop_uid=plan_sop_uid,
            dose_sop_uid=dose_sop_uid,
            source_plan_uids=source_plan_uids,
            source_dose_uids=source_dose_uids,
            delivered_dose_gy=delivery_summary["delivered_dose_gy"],
            delivery_status=str(delivery_summary["delivery_status"]),
            delivery_method=delivery_summary["delivery_method"],
            delivered_record_count=int(delivery_summary["delivered_record_count"]),
            delivered_fraction_count=int(delivery_summary["delivered_fraction_count"]),
            planned_fraction_count=delivery_summary["planned_fraction_count"],
            delivery_plan_details=delivery_summary["delivery_plan_details"],
            delivery_warnings=delivery_summary["delivery_warnings"],
            unresolved_record_plan_uids=delivery_summary["unresolved_record_plan_uids"],
            planning_ct_status=ct_select_status,
            planning_ct_referenced_series_uids=planning_ct_referenced_series_uids,
            planning_ct_series_uid=planning_ct_series_uid,
            selected_plan_contract=selected_plan_contract,
            selected_dose_contract=selected_dose_contract,
            per_plan_delivery_contract=per_plan_delivery,
            authoritative_rtstruct_uid=authoritative_rtstruct_uid,
            dose_classification=dose_classification_info,
            dose_qc={
                "status": dose_plausibility["dose_qc_status"],
                "pass": dose_plausibility["dose_qc_pass"],
                "threshold_gy": dose_plausibility["dose_plausibility_threshold_gy"],
                "reasons": dose_plausibility["dose_qc_reasons"],
            },
        )

    if course_tasks:
        results = run_tasks_with_adaptive_workers(
            "Organize",
            course_tasks,
            lambda task: _process_course(task[0], task[1], task[2], task[3]),
            max_workers=config.effective_workers(),
            logger=logger,
            show_progress=True,
            task_timeout=config.task_timeout,
        )
        for co in results:
            if co:
                logger.info(
                    "Organized patient %s course %s at %s",
                    co.patient_id,
                    co.course_id,
                    co.dirs.root,
                )
                outputs.append(co)
    elif structs:
        rs_groups: Dict[tuple[str, str], list[StructInfo]] = {}
        for s in structs:
            key = (str(s.patient_id), s.study_uid or f"FOR:{s.frame_of_reference_uid or 'unknown'}")
            rs_groups.setdefault(key, []).append(s)

        rs_entries: List[Tuple[str, str, List[StructInfo], Dict[str, Optional[str] | Optional[datetime.datetime]]]] = []
        for (pid, raw_key), s_list in rs_groups.items():
            start_dt: Optional[datetime.datetime] = None
            try:
                ds = pydicom.dcmread(str(s_list[0].path), stop_before_pixels=True)
                raw_date = getattr(ds, "StructureSetDate", None) or getattr(ds, "StudyDate", None)
                if raw_date:
                    start_dt = parse_date(str(raw_date))
            except Exception:
                start_dt = None
            start_token = start_dt.strftime("%Y-%m") if start_dt else None
            dir_name = course_dir_name(start_token, raw_key, existing_names[str(pid)])
            meta: Dict[str, Optional[str] | Optional[datetime.datetime]] = {
                "dir_name": dir_name,
                "start_token": start_token,
                "start_iso": start_dt.strftime("%Y-%m-%d") if start_dt else None,
                "start_dt": start_dt,
            }
            rs_entries.append((str(pid), str(raw_key), s_list, meta))

        rs_entries.sort(
            key=lambda entry: (
                entry[0],
                entry[3].get("start_token") or "ZZZZ-99",
                entry[1],
            )
        )

        def _process_rs_group(
            patient_id: str,
            course_key_raw: str,
            s_list: list[StructInfo],
            meta: Dict[str, Optional[str] | Optional[datetime.datetime]],
        ) -> CourseOutput:
            course_key = "".join(ch if ch.isalnum() else "_" for ch in str(course_key_raw))[:64]
            patient_root = config.output_root / patient_id
            ensure_dir(patient_root)
            course_id = str(meta.get("dir_name") or course_key)
            course_dir = patient_root / course_id
            course_dirs = build_course_dirs(course_dir)
            course_dirs.ensure()

            if config.resume and course_dir.exists():
                hydrated = _hydrate_existing_course(patient_id, course_key, course_dir, meta)
                if hydrated:
                    return hydrated

            primary_nifti: Optional[Path] = None
            related_outputs: List[Path] = []
            seen_related: set[Path] = set()
            course_for_uids = {s.frame_of_reference_uid for s in s_list if s.frame_of_reference_uid}
            course_ct_series_uids: set[str] = set()
            planning_ct_series_uid: str | None = None

            rs_dst = course_dir / "RS.dcm"
            primary_struct = s_list[0].path
            _safe_copy(primary_struct, rs_dst, copy_manager=copy_manager)
            authoritative_rtstruct_uid: str | None = None
            try:
                ds_primary_struct = pydicom.dcmread(
                    str(rs_dst), stop_before_pixels=True, force=True
                )
                authoritative_rtstruct_uid = str(
                    getattr(ds_primary_struct, "SOPInstanceUID", "") or ""
                ) or None
            except Exception:
                authoritative_rtstruct_uid = None
            for s in s_list:
                _copy_into(s.path, course_dirs.dicom_rtstruct, copy_manager=copy_manager)

            course_study = s_list[0].study_uid
            series, ct_select_status = select_course_ct_series(
                ct_index,
                patient_id,
                primary_struct,
                course_study,
                require_reference=True,
            )
            if series:
                eligible, image_class, exclusion_reason = _classify_organize_ct_series(
                    list(series),
                    is_planning_ct=True,
                )
                if not eligible:
                    logger.error(
                        "Planning CT classifier excluded RS-only %s/%s series %s: class=%s reason=%s",
                        patient_id,
                        course_id,
                        getattr(series[0], "series_uid", "<missing>"),
                        image_class,
                        exclusion_reason,
                    )
                    series = None
                    ct_select_status = f"classifier_excluded:{exclusion_reason or image_class}"
            if series:
                first_inst = series[0] if series else None
                if first_inst is not None and getattr(first_inst, 'series_uid', None):
                    planning_ct_series_uid = str(first_inst.series_uid)
                    course_ct_series_uids.add(planning_ct_series_uid)
                logger.info(
                    "Per-course planning CT (RS) for %s (%s): %s -> series ...%s (%d slices)",
                    patient_id, course_dir, ct_select_status,
                    str(getattr(first_inst, 'series_uid', '') or '')[-12:], len(series),
                )
                copy_ct_series(series, course_dirs.dicom_ct, copy_manager=copy_manager)
                try:
                    primary_nifti = _ensure_ct_nifti(
                        config,
                        course_dirs.dicom_ct,
                        course_dirs.nifti,
                        force=bool(config.resume),
                    )
                except Exception as exc:
                    logger.warning("CT NIfTI conversion failed (RS-only) for %s: %s", course_dir, exc)
            else:
                if ct_select_status == "unresolved_reference":
                    logger.warning(
                        "Per-course CT fail-closed for %s (%s): primary RTSTRUCT references a CT series "
                        "absent from the indexed CT; skipping per-course CT/segmentation.",
                        patient_id, course_dir,
                    )
                # No planning CT -> remove any stale per-course CT/NIfTI/segmentation from a prior run.
                _clear_course_ct_outputs(course_dirs)

            if registrations_index.get(patient_id):
                for reg in registrations_index.get(patient_id, []):
                    reg_for = reg.get('for_uids', set())
                    if reg_for and course_for_uids and not course_for_uids.intersection(reg_for):
                        continue
                    reg_path = Path(reg.get('path'))
                    if reg_path.exists() and reg_path not in seen_related:
                        related_outputs.append(_copy_into(reg_path, course_dirs.dicom_related / "REG", copy_manager=copy_manager))
                        seen_related.add(reg_path)
                    for series_uid in reg.get('referenced_series', set()):
                        if series_uid in course_ct_series_uids:
                            continue
                        series_paths = series_index.get((patient_id, series_uid), [])
                        if not series_paths:
                            continue
                        modality = str(series_meta.get((patient_id, series_uid), {}).get("modality", "")).upper()
                        if modality == "MR":
                            series_root = course_dirs.dicom_mr / _sanitize_name(series_uid, "mr")
                            dest_dir = series_root / "DICOM"
                        else:
                            dest_dir = course_dirs.dicom_related / _sanitize_name(series_uid, "series")
                        for src in series_paths:
                            if src not in seen_related and src.exists():
                                related_outputs.append(_copy_into(src, dest_dir, copy_manager=copy_manager))
                                seen_related.add(src)

            def _convert_related_series(parent: Path, *, modality_hint: Optional[str] = None) -> None:
                for series_root in sorted(p for p in parent.iterdir() if p.is_dir() and p.name != "REG"):
                    try:
                        if modality_hint == "MR":
                            dicom_dir = series_root / "DICOM"
                            if not dicom_dir.exists():
                                dicom_dir = series_root
                            if not any(dicom_dir.glob("*.dcm")):
                                continue
                            target_root = series_root / "NIFTI"
                            sanitized_sid = _sanitize_name(series_root.name, "mr")
                        else:
                            dicom_dir = series_root
                            if not any(dicom_dir.glob("*.dcm")):
                                continue
                            target_root = course_dirs.nifti
                            sanitized_sid = None
                        target_root.mkdir(parents=True, exist_ok=True)
                        target_name = _derive_nifti_name(dicom_dir)
                        if sanitized_sid and not target_name.endswith(sanitized_sid):
                            target_name = f"{target_name}_{sanitized_sid}"
                        meta_path = target_root / f"{target_name}.metadata.json"
                        if meta_path.exists() and not config.resume:
                            continue
                        tmp_out = target_root / f".tmp_{series_root.name}"
                        tmp_out.mkdir(parents=True, exist_ok=True)
                        generated = run_dcm2niix(config, dicom_dir, tmp_out)
                        if generated is None:
                            shutil.rmtree(tmp_out, ignore_errors=True)
                            continue
                        target_path = target_root / f"{target_name}.nii.gz"
                        if target_path.exists():
                            target_path.unlink()
                        shutil.move(str(generated), str(target_path))
                        metadata = _collect_series_metadata(dicom_dir)
                        if modality_hint and not metadata.get("modality"):
                            metadata["modality"] = modality_hint
                        # Shared with _ensure_ct_nifti so a series converted here
                        # carries the same provenance the course contract requires.
                        nifti_provenance.annotate(
                            metadata,
                            target_path,
                            dicom_dir,
                            regenerated=True,
                            default_modality=modality_hint or "CT",
                        )
                        meta_path.write_text(
                            json.dumps(metadata, indent=2),
                            encoding="utf-8",
                        )
                        shutil.rmtree(tmp_out, ignore_errors=True)
                    except Exception as exc:
                        logger.debug("Failed converting related series %s: %s", series_root, exc)

            def _mr_present() -> bool:
                if not course_dirs.dicom_mr.exists():
                    return False
                for child in course_dirs.dicom_mr.iterdir():
                    if child.is_dir() and child.name != "REG":
                        return True
                return False

            _convert_related_series(course_dirs.dicom_related)
            if course_dirs.dicom_mr.exists():
                _convert_related_series(course_dirs.dicom_mr, modality_hint="MR")

            if not _mr_present():
                fallback_series = []
                for (pid, sid), paths in series_index.items():
                    if pid != patient_id:
                        continue
                    modality = str(series_meta.get((pid, sid), {}).get("modality", "")).upper()
                    if modality != "MR":
                        continue
                    if not paths:
                        continue
                    fallback_series.append((sid, paths))

                if fallback_series:
                    imported_files = 0
                    for idx, (series_uid, paths) in enumerate(fallback_series, start=1):
                        series_name = _sanitize_name(series_uid or f"mr_{idx}", "mr")
                        series_root = course_dirs.dicom_mr / series_name
                        dicom_dir = series_root / "DICOM"
                        for src in paths:
                            try:
                                _copy_into(src, dicom_dir, copy_manager=copy_manager)
                                imported_files += 1
                            except Exception as exc:
                                logger.debug("Failed to import MR file %s: %s", src, exc)
                    if imported_files:
                        logger.info(
                            "Imported %d MR files across %d series (fallback) for %s/%s",
                            imported_files,
                            len(fallback_series),
                            patient_id,
                            course_id,
                        )
                        _convert_related_series(course_dirs.dicom_mr, modality_hint="MR")
                else:
                    logger.debug("No MR series indexed for patient %s", patient_id)

            return CourseOutput(
                patient_id=patient_id,
                course_key=course_key,
                course_id=course_id,
                course_start=meta.get("start_token") if isinstance(meta.get("start_token"), (str, type(None))) else None,
                dirs=course_dirs,
                rp_path=course_dir / "RP.dcm",
                rd_path=course_dir / "RD.dcm",
                rs_path=rs_dst,
                primary_nifti=Path(primary_nifti) if primary_nifti else None,
                related_dicom=related_outputs,
                total_prescription_gy=None,
                planning_ct_status=ct_select_status,
                planning_ct_referenced_series_uids=sorted(
                    referenced_ct_series_uids(primary_struct)
                ),
                planning_ct_series_uid=planning_ct_series_uid,
                authoritative_rtstruct_uid=authoritative_rtstruct_uid,
            )

        results = run_tasks_with_adaptive_workers(
            "Organize (RS-only)",
            rs_entries,
            lambda task: _process_rs_group(task[0], task[1], task[2], task[3]),
            max_workers=config.effective_workers(),
            logger=logger,
            show_progress=True,
            task_timeout=config.task_timeout,
        )
        for res in results:
            if res:
                outputs.append(res)
    elif ct_index:
        if not config.allow_ct_only_courses:
            ct_series_count = sum(
                len(series_map)
                for studies in ct_index.values()
                for series_map in studies.values()
            )
            ct_instance_count = sum(
                len(series)
                for studies in ct_index.values()
                for series_map in studies.values()
                for series in series_map.values()
            )
            raise CTOnlyCohortError(
                "Input contains CT objects but no linked RTPLAN or RTSTRUCT courses "
                f"({ct_series_count} series, {ct_instance_count} instances). "
                "Set organize.allow_ct_only_courses: true only for an intentional diagnostic CT-only cohort."
            )
        ct_entries: List[Tuple[str, str, List[CTInstance], Dict[str, Optional[str] | Optional[datetime.datetime]]]] = []
        for pid, studies in sorted(ct_index.items(), key=lambda item: str(item[0])):
            for study_uid, series_map in sorted(studies.items(), key=lambda item: str(item[0])):
                for series_uid, series in sorted(series_map.items(), key=lambda item: str(item[0])):
                    if not series:
                        continue
                    series_instances = cast(List[CTInstance], list(series))
                    eligible, image_class, exclusion_reason = _classify_organize_ct_series(
                        series_instances,
                        is_planning_ct=False,
                        allow_ct_only=True,
                    )
                    if not eligible:
                        logger.info(
                            "Skipping CT-only series %s/%s as a course: class=%s reason=%s",
                            pid,
                            series_uid,
                            image_class,
                            exclusion_reason,
                        )
                        continue
                    first_inst = series_instances[0]
                    start_dt: Optional[datetime.datetime] = None
                    series_number = getattr(first_inst, "series_number", None)
                    try:
                        ds = pydicom.dcmread(str(first_inst.path), stop_before_pixels=True)
                        raw_date = (
                            getattr(ds, "SeriesDate", None)
                            or getattr(ds, "StudyDate", None)
                            or getattr(ds, "AcquisitionDate", None)
                        )
                        if raw_date:
                            start_dt = parse_date(str(raw_date))
                        series_number = getattr(ds, "SeriesNumber", series_number)
                    except Exception:
                        start_dt = None
                    start_token = start_dt.strftime("%Y-%m") if start_dt else None
                    fallback = f"ct_{series_number or 'series'}_{str(series_uid)[-12:]}"
                    dir_name = course_dir_name(start_token, fallback, existing_names[str(pid)])
                    meta: Dict[str, Optional[str] | Optional[datetime.datetime]] = {
                        "dir_name": dir_name,
                        "start_token": start_token,
                        "start_iso": start_dt.strftime("%Y-%m-%d") if start_dt else None,
                        "start_dt": start_dt,
                    }
                    ct_entries.append((str(pid), str(series_uid), series_instances, meta))

        ct_entries.sort(
            key=lambda entry: (
                entry[0],
                entry[3].get("start_token") or "ZZZZ-99",
                entry[3].get("dir_name") or "",
                entry[1],
            )
        )

        def _process_ct_series(
            patient_id: str,
            series_uid: str,
            series: list,
            meta: Dict[str, Optional[str] | Optional[datetime.datetime]],
        ) -> CourseOutput:
            course_key = "".join(ch if ch.isalnum() else "_" for ch in str(series_uid))[:64]
            patient_root = config.output_root / patient_id
            ensure_dir(patient_root)
            course_id = str(meta.get("dir_name") or course_key)
            course_dir = patient_root / course_id
            course_dirs = build_course_dirs(course_dir)
            course_dirs.ensure()

            if config.resume and course_dir.exists():
                hydrated = _hydrate_existing_course(patient_id, course_key, course_dir, meta)
                if hydrated:
                    return hydrated

            primary_nifti: Optional[Path] = None
            copy_ct_series(series, course_dirs.dicom_ct, copy_manager=copy_manager)
            try:
                primary_nifti = _ensure_ct_nifti(
                    config,
                    course_dirs.dicom_ct,
                    course_dirs.nifti,
                    force=bool(config.resume),
                )
            except Exception as exc:
                logger.warning("CT NIfTI conversion failed (CT-only) for %s: %s", course_dir, exc)

            return CourseOutput(
                patient_id=patient_id,
                course_key=course_key,
                course_id=course_id,
                course_start=meta.get("start_token") if isinstance(meta.get("start_token"), (str, type(None))) else None,
                dirs=course_dirs,
                rp_path=course_dir / "RP.dcm",
                rd_path=course_dir / "RD.dcm",
                rs_path=None,
                primary_nifti=Path(primary_nifti) if primary_nifti else None,
                related_dicom=[],
                total_prescription_gy=None,
                planning_ct_status="ct_only",
                planning_ct_series_uid=str(series_uid),
            )

        results = run_tasks_with_adaptive_workers(
            "Organize (CT-only)",
            ct_entries,
            lambda task: _process_ct_series(task[0], task[1], task[2], task[3]),
            max_workers=config.effective_workers(),
            logger=logger,
            show_progress=True,
            task_timeout=config.task_timeout,
        )
        for res in results:
            if res:
                outputs.append(res)

    for _co in outputs:
        try:
            _record_course_done(config, _co.patient_id, _co.course_key, _co.dirs.root)
        except Exception:
            pass

    # Re-admit the courses of patients that were skipped as already complete, so
    # the manifest still describes the whole cohort rather than only this run.
    if resumed_patients:
        _rehydrated = 0
        for _pid, _entries in resumed_patients.items():
            for _entry in _entries:
                _cdir = Path(str(_entry.get("course_dir") or ""))
                if not _cdir.is_dir():
                    continue
                _co = _hydrate_existing_course(_pid, str(_entry.get("course_key") or _cdir.name), _cdir)
                if _co is not None:
                    outputs.append(_co)
                    _rehydrated += 1
        logger.info("Organize resume: re-admitted %d course(s) from complete patients", _rehydrated)

    if getattr(config, "do_segment_all_series", False):
        if not getattr(config, "inventory_db_path", None):
            raise ValueError("do_segment_all_series=True requires config.inventory_db_path")
        patient_ids = list(getattr(config, "inventory_patient_ids", []) or [])
        if not patient_ids:
            patient_ids = sorted({co.patient_id for co in outputs if co.patient_id})
        if not patient_ids:
            logger.warning("All-series inventory mode requested, but no patient IDs were available")
        failed_patients: list[str] = []
        for patient_id in patient_ids:
            try:
                manifest_path = materialize_patient_series_from_inventory(config, patient_id)
                logger.info("Wrote all-series manifest for %s at %s", patient_id, manifest_path)
            except Exception as exc:
                # Log-and-continue: a single patient's materialization failure (NFS blip, DB
                # lock, unreadable instance) must not abort an entire cohort run. The failed
                # patient simply has no all-series manifest and is surfaced in the summary
                # below; resume reprocesses it on the next launch.
                logger.error(
                    "All-series inventory materialization FAILED for %s: %s (continuing)",
                    patient_id,
                    exc,
                )
                failed_patients.append(str(patient_id))
        if failed_patients:
            logger.error(
                "All-series materialization failed for %d/%d patient(s): %s",
                len(failed_patients),
                len(patient_ids),
                ", ".join(failed_patients[:50]) + (" ..." if len(failed_patients) > 50 else ""),
            )

    # After course-level copying and plan/dose synthesis, write per-case metadata serially to avoid overwhelming IO
    for co in outputs:
        patient_dir = co.dirs.root
        meta_dir = co.dirs.metadata
        meta_dir.mkdir(parents=True, exist_ok=True)
        manual_manifest = _export_original_segmentation(co, overwrite=bool(config.resume))
        nifti_files = sorted(co.dirs.nifti.glob("*.nii*"))
        # ------------------------------------------------------------------
        # Save per-case metadata (Excel + JSON) in the course directory
        # ------------------------------------------------------------------
        try:
            # Aggregate course-level details for research/clinic
            # Rebuild context from files on disk
            plan_uids: set[str] = set()
            if co.plan_sop_uid:
                plan_uids.add(str(co.plan_sop_uid))
            for uid in co.source_plan_uids or []:
                if uid:
                    plan_uids.add(str(uid))
            items_sorted = []  # Not available here; we use on-disk RP only where needed below
            try:
                rp_path = patient_dir / "RP.dcm"
                if rp_path.exists():
                    ds_tmp = pydicom.dcmread(str(rp_path), stop_before_pixels=True)
                    sop_uid = str(getattr(ds_tmp, 'SOPInstanceUID', ''))
                    if sop_uid:
                        plan_uids.add(sop_uid)
            except Exception:
                pass
            plan_total_rx = 0.0
            try:
                ds_plan = pydicom.dcmread(str(co.rp_path), stop_before_pixels=True)
            except Exception:
                ds_plan = None
            if ds_plan is not None:
                inferred = infer_plan_rx_gy(ds_plan)
                if inferred is not None:
                    plan_total_rx += inferred
            logger.info(
                "[organize] %s/%s inferred total prescription=%.3f",
                co.patient_id,
                co.course_id,
                plan_total_rx,
            )
            # Planned fractions, machine, beam energies, beams count
            planned_fractions = None
            machine_name = None
            beam_energies = []
            beams_count = 0
            try:
                rp0 = co.rp_path
                if rp0.exists():
                    ds0 = pydicom.dcmread(str(rp0), stop_before_pixels=True)
                    if hasattr(ds0, 'FractionGroupSequence') and ds0.FractionGroupSequence:
                        fg = ds0.FractionGroupSequence[0]
                        if hasattr(fg, 'NumberOfFractionsPlanned'):
                            planned_fractions = int(fg.NumberOfFractionsPlanned)
                    if hasattr(ds0, 'BeamSequence') and ds0.BeamSequence:
                        beams_count = len(ds0.BeamSequence)
                        for b in ds0.BeamSequence:
                            if hasattr(b, 'NominalBeamEnergy'):
                                try:
                                    beam_energies.append(float(b.NominalBeamEnergy))
                                except Exception:
                                    pass
                    if hasattr(ds0, 'TreatmentMachineName'):
                        machine_name = str(ds0.TreatmentMachineName)
            except Exception:
                pass
            # Course start/stop from RT treatment records (match by patient & referenced plan UID)
            start_date = None
            end_date = None
            fractions_count = 0
            fractions_details: list[dict[str, object]] = []
            try:
                candidate_rt_files = rt_file_index.get(str(co.patient_id), [])
                for p in candidate_rt_files:
                    try:
                        ds_rt = pydicom.dcmread(str(p), stop_before_pixels=True)
                    except Exception:
                        continue
                    if getattr(ds_rt, 'PatientID', None) and str(ds_rt.PatientID).strip() != str(co.patient_id):
                        continue
                    ref_uids = []
                    try:
                        ref_uids = [
                            str(getattr(ref, "ReferencedSOPInstanceUID", "") or "")
                            for ref in getattr(ds_rt, "ReferencedRTPlanSequence", []) or []
                        ]
                        ref_uids = [uid for uid in ref_uids if uid]
                    except Exception:
                        ref_uids = []
                    unknown_ref_uids = sorted(uid for uid in ref_uids if uid not in plan_uids)
                    for unknown_uid in unknown_ref_uids:
                        logger.warning(
                            "RTRECORD %s references RTPLAN UID %s absent from course export; "
                            "the record is excluded from course fractions",
                            p,
                            unknown_uid,
                        )
                    matched_ref_uids = [uid for uid in ref_uids if uid in plan_uids]
                    if not matched_ref_uids:
                        logger.warning(
                            "RTRECORD %s has no RTPLAN reference resolvable to course %s; "
                            "the record is not attributed",
                            p,
                            co.course_id,
                        )
                        continue
                    ref_uid = matched_ref_uids[0]
                    rt_date = getattr(ds_rt, 'TreatmentDate', None) or getattr(ds_rt, 'SeriesDate', None)
                    rt_time = getattr(ds_rt, 'TreatmentTime', None)
                    frac_num = (
                        getattr(ds_rt, "CurrentFractionNumber", None)
                        or getattr(ds_rt, "ReferencedFractionNumber", None)
                    )
                    machine = getattr(ds_rt, 'TreatmentMachineName', None) or getattr(ds_rt, 'ReferencedTreatmentMachineName', None)
                    if rt_date:
                        fractions_count += 1
                        try:
                            d = datetime.datetime.strptime(str(rt_date), '%Y%m%d').date()
                        except Exception:
                            try:
                                d = datetime.datetime.strptime(str(rt_date), '%Y-%m-%d').date()
                            except Exception:
                                d = None
                        if d:
                            start_date = d if start_date is None or d < start_date else start_date
                            end_date = d if end_date is None or d > end_date else end_date
                    machine_name_rt = machine
                    try:
                        seq = getattr(ds_rt, 'TreatmentMachineSequence', None)
                        if seq:
                            tm = getattr(seq[0], 'TreatmentMachineName', None)
                            if tm:
                                machine_name_rt = str(tm)
                    except Exception:
                        pass

                    frac_entry: Dict[str, object] = {
                        "treatment_date": d.isoformat() if d else str(rt_date),
                        "treatment_time": str(rt_time or ""),
                        "plan_sop": ref_uid or "",
                        "fraction_number": int(frac_num) if frac_num is not None else None,
                        "treatment_machine": str(machine_name_rt or ""),
                        "source_path": str(p),
                        "sop_instance_uid": str(getattr(ds_rt, "SOPInstanceUID", "")),
                        "delivered_dose_gy": None,
                        "delivered_dose_method": None,
                        "beam_meterset": None,
                        "delivered_meterset": None,
                        "beam_delivery": [],
                    }
                    try:
                        dose_value, dose_method = _record_dose_reference(ds_rt)
                        if dose_value is not None:
                            frac_entry["delivered_dose_gy"] = dose_value
                            frac_entry["delivered_dose_method"] = dose_method
                    except Exception:
                        pass
                    delivered_total = 0.0
                    try:
                        tsb_seq = getattr(ds_rt, 'TreatmentSessionBeamSequence', None) or []
                        beam_deliveries = []
                        for beam in tsb_seq:
                            beam_num = getattr(beam, 'ReferencedBeamNumber', None)
                            delivered_mu = getattr(beam, 'DeliveredMeterset', None)
                            if delivered_mu not in (None, ""):
                                try:
                                    delivered_val = float(delivered_mu)
                                    delivered_total += delivered_val
                                except Exception:
                                    delivered_val = None
                            else:
                                delivered_val = None
                            cp_seq = getattr(beam, 'ControlPointDeliverySequence', None) or []
                            gantries = []
                            meterset_weights = []
                            for cp in cp_seq:
                                if hasattr(cp, 'GantryAngle'):
                                    try:
                                        gantries.append(float(cp.GantryAngle))
                                    except Exception:
                                        pass
                                if hasattr(cp, 'CumulativeMetersetWeight'):
                                    try:
                                        meterset_weights.append(float(cp.CumulativeMetersetWeight))
                                    except Exception:
                                        pass
                            beam_deliveries.append({
                                'beam_number': int(beam_num) if beam_num is not None else None,
                                'delivered_meterset': delivered_val,
                                'gantry_start': float(gantries[0]) if gantries else None,
                                'gantry_end': float(gantries[-1]) if gantries else None,
                                'control_points': len(cp_seq),
                                'meterset_weights': meterset_weights or None,
                            })
                        if beam_deliveries:
                            frac_entry['beam_delivery'] = beam_deliveries
                        if delivered_total > 0:
                            frac_entry['delivered_meterset'] = delivered_total
                            if not frac_entry.get('beam_meterset'):
                                frac_entry['beam_meterset'] = delivered_total
                    except Exception:
                        pass
                    fractions_details.append(frac_entry)
            except Exception:
                pass

            fractions_path = patient_dir / "fractions.xlsx"
            if fractions_details:
                try:
                    import pandas as _pd

                    df_frac_raw = _pd.DataFrame(fractions_details)
                    if not df_frac_raw.empty:
                        df_frac_raw["treatment_time"] = df_frac_raw["treatment_time"].fillna("")
                        plan_primary = str(co.plan_sop_uid) if co.plan_sop_uid else next(iter(plan_uids), "")
                        df_frac_raw["plan_key"] = df_frac_raw["plan_sop"].fillna("")
                        if plan_primary:
                            df_frac_raw.loc[df_frac_raw["plan_key"] == "", "plan_key"] = plan_primary
                        else:
                            df_frac_raw.loc[df_frac_raw["plan_key"] == "", "plan_key"] = "NO_PLAN"
                        df_frac_raw["treatment_date"] = df_frac_raw["treatment_date"].astype(str)
                        df_frac_raw.sort_values(
                            by=["plan_key", "treatment_date", "treatment_time", "sop_instance_uid"],
                            inplace=True,
                            ignore_index=True,
                        )

                        def _dense_rank_dates(values: _pd.Series) -> _pd.Series:
                            order: dict[str, int] = {}
                            seq: list[int] = []
                            next_idx = 1
                            for date_str in values.astype(str):
                                key = date_str
                                if key not in order:
                                    order[key] = next_idx
                                    next_idx += 1
                                seq.append(order[key])
                            return _pd.Series(seq, index=values.index)

                        inferred = df_frac_raw.groupby("plan_key")['treatment_date'].transform(_dense_rank_dates)
                        df_frac_raw["fraction_number_inferred"] = inferred

                        if df_frac_raw["fraction_number"].notna().any():
                            filled = df_frac_raw.groupby(["plan_key", "treatment_date"])['fraction_number'].transform(
                                lambda s: s.dropna().iloc[0] if not s.dropna().empty else None
                            )
                            df_frac_raw["fraction_number"] = filled.where(
                                filled.notna(),
                                df_frac_raw["fraction_number_inferred"],
                            )
                        else:
                            df_frac_raw["fraction_number"] = df_frac_raw["fraction_number_inferred"]

                        df_frac_raw["fraction_number"] = df_frac_raw["fraction_number"].astype("Int64")

                        aggregated_rows: list[dict[str, object]] = []
                        for (plan_key, frac_num), grp in df_frac_raw.groupby(["plan_key", "fraction_number"], dropna=False):
                            grp_sorted = grp.sort_values(by=["treatment_date", "treatment_time"], kind="stable")
                            first_row = grp_sorted.iloc[0]
                            treatment_date = str(first_row["treatment_date"])
                            start_time = str(grp_sorted["treatment_time"].min()) if not grp_sorted["treatment_time"].isna().all() else ""
                            end_time = str(grp_sorted["treatment_time"].max()) if not grp_sorted["treatment_time"].isna().all() else ""
                            machines = [
                                str(x).strip()
                                for x in grp_sorted["treatment_machine"].tolist()
                                if str(x).strip()
                            ]
                            machine_repr = ";".join(sorted(set(machines))) if machines else str(first_row["treatment_machine"])
                            source_paths_all = ";".join([
                                str(x)
                                for x in _pd.unique(grp_sorted["source_path"])
                                if str(x)
                            ])
                            sops_all = ";".join([
                                str(x)
                                for x in _pd.unique(grp_sorted["sop_instance_uid"])
                                if str(x)
                            ])
                            dose_sum = grp_sorted["delivered_dose_gy"].dropna()
                            meterset_sum = grp_sorted["beam_meterset"].dropna()
                            component_times = ";".join([
                                str(t)
                                for t in grp_sorted["treatment_time"].astype(str)
                                if str(t)
                            ])

                            aggregated_rows.append({
                                "treatment_date": treatment_date,
                                "treatment_time": start_time,
                                "treatment_time_end": end_time,
                                "plan_sop": first_row["plan_sop"] or ("" if plan_key in {"", "NO_PLAN"} else str(plan_key)),
                                "fraction_number": int(frac_num) if _pd.notna(frac_num) else None,
                                "treatment_machine": machine_repr,
                                "source_path": first_row["source_path"],
                                "source_paths_all": source_paths_all,
                                "sop_instance_uid": first_row["sop_instance_uid"],
                                "sop_instance_uids_all": sops_all,
                                "delivered_dose_gy": float(dose_sum.sum()) if not dose_sum.empty else None,
                                "beam_meterset": float(meterset_sum.sum()) if not meterset_sum.empty else None,
                                "records_merged": int(len(grp_sorted)),
                                "component_times": component_times,
                            })

                        df_frac = _pd.DataFrame(aggregated_rows)
                        df_frac.sort_values(by=["treatment_date", "fraction_number"], inplace=True, ignore_index=True)
                        df_frac.to_excel(fractions_path, index=False)

                        raw_export_path = patient_dir / "metadata" / "fractions_raw.xlsx"
                        try:
                            raw_export_path.parent.mkdir(parents=True, exist_ok=True)
                            df_frac_raw.to_excel(raw_export_path, index=False)
                        except Exception:
                            logger.debug("Failed to export raw fractions detail for %s", patient_dir)

                        frac_numbers = df_frac.get("fraction_number")
                        if frac_numbers is not None:
                            fractions_count = int(frac_numbers.dropna().nunique())
                        else:
                            fractions_count = len(df_frac)
                    else:
                        df_frac_raw.to_excel(fractions_path, index=False)
                        fractions_count = 0
                except Exception as exc:
                    logger.warning(
                        "Failed to write fractions summary for %s: %s",
                        patient_dir,
                        exc,
                        exc_info=True,
                    )
            elif fractions_path.exists():
                try:
                    fractions_path.unlink()
                except Exception:
                    pass

            # Dose grid info from organized RD
            dose_grid = {}
            try:
                if co.rd_path.exists():
                    ds_rd = pydicom.dcmread(str(co.rd_path), stop_before_pixels=False)
                    dose_grid = {
                        'DoseSummationType': str(getattr(ds_rd, 'DoseSummationType', '')),
                        'Rows': int(getattr(ds_rd, 'Rows', 0) or 0),
                        'Columns': int(getattr(ds_rd, 'Columns', 0) or 0),
                        'NumberOfFrames': int(getattr(ds_rd, 'NumberOfFrames', 0) or 0),
                    }
                    try:
                        ps = getattr(ds_rd, 'PixelSpacing', [None, None])
                        dose_grid['PixelSpacing'] = [float(ps[0]), float(ps[1])] if ps and len(ps) >= 2 else []
                    except Exception:
                        pass
                    try:
                        ipv = getattr(ds_rd, 'ImagePositionPatient', None)
                        if ipv and len(ipv) == 3:
                            dose_grid['ImagePositionPatient'] = [float(x) for x in ipv]
                    except Exception:
                        pass
                    try:
                        offsets = getattr(ds_rd, 'GridFrameOffsetVector', None)
                        if offsets:
                            dose_grid['GridFrameOffsetVector'] = [float(x) for x in offsets]
                    except Exception:
                        pass
                    try:
                        if hasattr(ds_rd, 'DoseGridScaling') and ds_rd.DoseGridScaling is not None:
                            dose_grid['DoseGridScaling'] = float(ds_rd.DoseGridScaling)
                    except Exception:
                        pass
                    try:
                        import numpy as _np
                        px = getattr(ds_rd, 'PixelData', None)
                        if px is not None:
                            arr = ds_rd.pixel_array.astype(float) * float(getattr(ds_rd, 'DoseGridScaling', 1.0))
                            dose_grid['DoseStats'] = {
                                'minGy': float(_np.min(arr)),
                                'maxGy': float(_np.max(arr)),
                                'meanGy': float(_np.mean(arr)),
                            }
                    except Exception:
                        pass
            except Exception:
                pass

            # CT acquisition summary from CT_DICOM (first file)
            ct_summary = {}
            try:
                ct_dir_path = co.dirs.dicom_ct
                if ct_dir_path.exists():
                    ct_files = sorted([p for p in ct_dir_path.iterdir() if p.is_file()])
                    if ct_files:
                        ds_ct = pydicom.dcmread(str(ct_files[0]), stop_before_pixels=True)
                        ct_summary = {
                            'ct_manufacturer': str(getattr(ds_ct, 'Manufacturer', '')),
                            'ct_model': str(getattr(ds_ct, 'ManufacturerModelName', '')),
                            'ct_institution': str(getattr(ds_ct, 'InstitutionName', '')),
                            'ct_kvp': float(getattr(ds_ct, 'KVP', 0.0) or 0.0) if hasattr(ds_ct, 'KVP') else None,
                            'ct_convolution_kernel': str(getattr(ds_ct, 'ConvolutionKernel', '')),
                            'ct_reconstruction_algorithm': _summarize_reconstruction(ds_ct),
                            'ct_slice_thickness': float(getattr(ds_ct, 'SliceThickness', 0.0) or 0.0) if hasattr(ds_ct, 'SliceThickness') else None,
                            'ct_study_uid': str(getattr(ds_ct, 'StudyInstanceUID', '')),
                            'ct_slice_increment': None,
                            'ct_tube_current_mA': None,
                            'ct_pitch_factor': None,
                            'ct_rotation_time_s': None,
                            'ct_matrix_size': f"{getattr(ds_ct, 'Rows', '')}x{getattr(ds_ct, 'Columns', '')}",
                            'ct_field_of_view_mm': float(getattr(ds_ct, 'ReconstructionDiameter', 0.0) or 0.0) if hasattr(ds_ct, 'ReconstructionDiameter') else None,
                            'ct_pixel_spacing': None,
                            'ct_contrast_agent': str(getattr(ds_ct, 'ContrastBolusAgent', '')),
                            'ct_contrast_flow_rate': None,
                            'ct_contrast_total_volume': None,
                            'ct_contrast_phase': str(getattr(ds_ct, 'ContrastBolusRoute', '')),
                        }
                        try:
                            ps = getattr(ds_ct, 'PixelSpacing', [None, None])
                            ct_summary['ct_pixel_spacing'] = [float(ps[0]), float(ps[1])] if ps and len(ps) >= 2 else []
                        except Exception:
                            pass
                        try:
                            spacing_between = getattr(ds_ct, 'SpacingBetweenSlices', None)
                            if spacing_between is not None:
                                ct_summary['ct_slice_increment'] = float(spacing_between)
                            else:
                                positions = []
                                for ct_file in ct_files[:min(10, len(ct_files))]:
                                    ds_tmp = pydicom.dcmread(str(ct_file), stop_before_pixels=True)
                                    ipp = getattr(ds_tmp, 'ImagePositionPatient', None)
                                    if ipp and len(ipp) == 3:
                                        positions.append(float(ipp[2]))
                                if len(positions) >= 2:
                                    positions = sorted(positions)
                                    deltas = [abs(b - a) for a, b in zip(positions, positions[1:]) if abs(b - a) > 1e-6]
                                    if deltas:
                                        ct_summary['ct_slice_increment'] = float(np.median(deltas))
                        except Exception:
                            pass
                        try:
                            if hasattr(ds_ct, 'XRayTubeCurrent') and ds_ct.XRayTubeCurrent is not None:
                                ct_summary['ct_tube_current_mA'] = float(ds_ct.XRayTubeCurrent)
                            elif hasattr(ds_ct, 'TubeCurrent') and ds_ct.TubeCurrent is not None:
                                ct_summary['ct_tube_current_mA'] = float(ds_ct.TubeCurrent)
                        except Exception:
                            pass
                        try:
                            if hasattr(ds_ct, 'CTPitchFactor') and ds_ct.CTPitchFactor is not None:
                                ct_summary['ct_pitch_factor'] = float(ds_ct.CTPitchFactor)
                        except Exception:
                            pass
                        try:
                            if hasattr(ds_ct, 'GantryRotationTime') and ds_ct.GantryRotationTime is not None:
                                ct_summary['ct_rotation_time_s'] = float(ds_ct.GantryRotationTime)
                            elif hasattr(ds_ct, 'RotationTime') and ds_ct.RotationTime is not None:
                                ct_summary['ct_rotation_time_s'] = float(ds_ct.RotationTime)
                        except Exception:
                            pass
                        try:
                            if hasattr(ds_ct, 'ContrastBolusTotalDose') and ds_ct.ContrastBolusTotalDose is not None:
                                ct_summary['ct_contrast_total_volume'] = float(ds_ct.ContrastBolusTotalDose)
                        except Exception:
                            pass
                        try:
                            if hasattr(ds_ct, 'ContrastFlowRate') and ds_ct.ContrastFlowRate is not None:
                                ct_summary['ct_contrast_flow_rate'] = float(ds_ct.ContrastFlowRate)
                        except Exception:
                            pass
            except Exception:
                pass

            seg_dicom_path = ""
            try:
                seg_dicom_files = sorted(co.dirs.segmentation_totalseg.glob("*/*.dcm"))
                if seg_dicom_files:
                    seg_dicom_path = str(seg_dicom_files[0])
            except Exception:
                seg_dicom_path = ""

            seg_manifest_paths: List[str] = []
            try:
                for manifest_path in co.dirs.segmentation_totalseg.glob("*/manifest.json"):
                    if manifest_path.exists():
                        seg_manifest_paths.append(str(manifest_path))
            except Exception:
                pass

            prescribed_dose_gy = (
                co.total_prescription_gy
                if co.total_prescription_gy is not None
                else (plan_total_rx if plan_total_rx > 0 else None)
            )
            dose_threshold_gy = float(config.max_total_dose_gy)
            dose_plausibility = _dose_plausibility(
                prescribed_dose_gy,
                co.delivered_dose_gy,
                dose_threshold_gy,
            )
            if dose_plausibility["dose_plausibility_warning"]:
                logger.warning(
                    "PLAUSIBILITY WARNING: %s/%s exceeds configured dose threshold %.1f Gy "
                    "(prescribed=%s, delivered=%s)",
                    co.patient_id,
                    co.course_id,
                    dose_threshold_gy,
                    prescribed_dose_gy,
                    co.delivered_dose_gy,
                )
            selected_plan_contract: list[dict[str, object]] = []
            for item in co.selected_plan_contract:
                serialized = dict(item)
                serialized["path"] = relative_contract_path(
                    patient_dir, Path(str(item.get("path") or ""))
                )
                selected_plan_contract.append(serialized)
            selected_dose_contract: list[dict[str, object]] = []
            for item in co.selected_dose_contract:
                serialized = dict(item)
                serialized["path"] = relative_contract_path(
                    patient_dir, Path(str(item.get("path") or ""))
                )
                selected_dose_contract.append(serialized)
            per_plan_delivery_contract: list[dict[str, object]] = []
            for item in co.per_plan_delivery_contract:
                serialized = dict(item)
                plan_path = str(item.get("plan_path") or "")
                serialized["plan_path"] = (
                    relative_contract_path(patient_dir, Path(plan_path))
                    if plan_path
                    else ""
                )
                per_plan_delivery_contract.append(serialized)

            dose_qc_contract = {
                "status": dose_plausibility["dose_qc_status"],
                "pass": dose_plausibility["dose_qc_pass"],
                "threshold_gy": dose_plausibility["dose_plausibility_threshold_gy"],
                "reasons": dose_plausibility["dose_qc_reasons"],
            }
            authoritative_rtstruct = None
            if co.rs_path and co.rs_path.exists() and co.authoritative_rtstruct_uid:
                authoritative_rtstruct = {
                    "sop_instance_uid": co.authoritative_rtstruct_uid,
                    "path": relative_contract_path(patient_dir, co.rs_path),
                }
            plan_artifact = None
            if co.rp_path.exists() and co.plan_sop_uid:
                plan_artifact = {
                    "sop_instance_uid": str(co.plan_sop_uid),
                    "path": relative_contract_path(patient_dir, co.rp_path),
                    "source_plan_uids": list(co.source_plan_uids),
                }
            selected_plan_uids = [
                str(item.get("sop_instance_uid") or "")
                for item in selected_plan_contract
            ]
            selected_dose_uids = [
                str(item.get("sop_instance_uid") or "")
                for item in selected_dose_contract
            ]
            selected_dose_types = [
                str(item.get("dose_summation_type") or "").upper()
                for item in selected_dose_contract
            ]
            dose_grid_contract = None
            if co.rd_path.exists() and co.dose_sop_uid and selected_plan_uids and selected_dose_uids:
                dose_grid_contract = {
                    "sop_instance_uid": str(co.dose_sop_uid),
                    "path": relative_contract_path(patient_dir, co.rd_path),
                    "dose_summation_type": str(dose_grid.get("DoseSummationType") or "").upper(),
                    "semantics": (
                        UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS
                        if co.delivery_status in {"delivered_but_records_absent", "no_records_at_all"}
                        else DOSE_GRID_SEMANTICS
                    ),
                    "source_plan_uids": selected_plan_uids,
                    "source_dose_uids": selected_dose_uids,
                    "source_dose_summation_types": selected_dose_types,
                }
            nifti_provenance = None
            if co.primary_nifti and Path(co.primary_nifti).is_file():
                nifti_path = Path(co.primary_nifti)
                nifti_base = nifti_path.name[:-7] if nifti_path.name.endswith(".nii.gz") else nifti_path.stem
                nifti_sidecar = nifti_path.parent / f"{nifti_base}.metadata.json"
                if not nifti_sidecar.exists():
                    try:
                        ct_provenance = _ct_provenance(co.dirs.dicom_ct)
                        try:
                            nifti_geometry = {
                                "size": [int(value) for value in sitk.ReadImage(str(nifti_path)).GetSize()],
                                "spacing": [float(value) for value in sitk.ReadImage(str(nifti_path)).GetSpacing()],
                                "origin": [float(value) for value in sitk.ReadImage(str(nifti_path)).GetOrigin()],
                                "direction": [float(value) for value in sitk.ReadImage(str(nifti_path)).GetDirection()],
                            }
                        except Exception:
                            nifti_geometry = {}
                        fallback_meta = {
                            **ct_provenance,
                            "nifti_geometry": nifti_geometry,
                            "nifti_sha256": hashlib.sha256(nifti_path.read_bytes()).hexdigest(),
                        }
                        nifti_sidecar.write_text(json.dumps(fallback_meta, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception as exc:
                        raise CourseContractError(
                            f"planning CT NIfTI provenance sidecar is missing or cannot be created: {nifti_sidecar}"
                        ) from exc
                try:
                    nifti_meta = json.loads(nifti_sidecar.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise CourseContractError(
                        f"planning CT NIfTI provenance sidecar is missing or unreadable: {nifti_sidecar}"
                    ) from exc
                if not isinstance(nifti_meta, dict):
                    raise CourseContractError(
                        f"planning CT NIfTI provenance sidecar is not an object: {nifti_sidecar}"
                    )
                required_nifti_keys = (
                    "series_instance_uid",
                    "sop_hash",
                    "geometry",
                    "nifti_geometry",
                    "nifti_sha256",
                )
                if any(key not in nifti_meta for key in required_nifti_keys):
                    raise CourseContractError(
                        f"planning CT NIfTI provenance sidecar is incomplete: {nifti_sidecar}"
                    )
                nifti_provenance = {
                    "sidecar_path": relative_contract_path(patient_dir, nifti_sidecar),
                    "series_instance_uid": nifti_meta["series_instance_uid"],
                    "sop_hash": nifti_meta["sop_hash"],
                    "geometry": nifti_meta["geometry"],
                    "nifti_geometry": nifti_meta["nifti_geometry"],
                    "nifti_sha256": hashlib.sha256(nifti_path.read_bytes()).hexdigest(),
                }
            course_contract = {
                "version": COURSE_CONTRACT_VERSION,
                "authority": "organize",
                "patient_id": str(co.patient_id),
                "course_id": co.course_id,
                "course_key": co.course_key,
                "selected_plans": selected_plan_contract,
                "selected_doses": selected_dose_contract,
                "dose_classification": co.dose_classification,
                "authoritative_rtstruct": authoritative_rtstruct,
                "planning_ct": {
                    "status": co.planning_ct_status,
                    "series_instance_uid": str(co.planning_ct_series_uid or ""),
                    "referenced_series_uids": co.planning_ct_referenced_series_uids,
                    "nifti_provenance": nifti_provenance,
                    "dicom_dir": (
                        relative_contract_path(patient_dir, co.dirs.dicom_ct)
                        if co.dirs.dicom_ct.is_dir() and any(co.dirs.dicom_ct.iterdir())
                        else ""
                    ),
                    "nifti_path": (
                        relative_contract_path(patient_dir, co.primary_nifti)
                        if co.primary_nifti and Path(co.primary_nifti).is_file()
                        else ""
                    ),
                },
                "plan_artifact": plan_artifact,
                "dose_grid": dose_grid_contract,
                "dvh": build_dvh_decision(
                    len(selected_plan_contract),
                    len(selected_dose_contract),
                    str(co.delivery_status),
                ),
                "delivery": {
                    "prescribed_dose_gy": prescribed_dose_gy,
                    "delivered_dose_gy": co.delivered_dose_gy,
                    "status": co.delivery_status,
                    "method": co.delivery_method,
                    "dose_response_field": DOSE_RESPONSE_FIELD,
                    "per_plan": per_plan_delivery_contract,
                    "warnings": co.delivery_warnings,
                    "unresolved_record_plan_uids": co.unresolved_record_plan_uids,
                },
                "dose_qc": dose_qc_contract,
            }
            co.course_contract = course_contract
            case_meta = {
                "patient_id": str(co.patient_id),
                "course_key": co.course_key,
                "course_id": co.course_id,
                "rp_path": str(co.rp_path) if co.rp_path.exists() else "",
                "rd_path": str(co.rd_path) if co.rd_path.exists() else "",
                "rs_path": str(co.rs_path) if co.rs_path and co.rs_path.exists() else "",
                "plan_sop_uid": str(co.plan_sop_uid or ""),
                "dose_sop_uid": str(co.dose_sop_uid or ""),
                "source_plan_uids": co.source_plan_uids or [],
                "source_dose_uids": co.source_dose_uids or [],
                "rs_auto_path": str((patient_dir / "RS_auto.dcm")) if (patient_dir / "RS_auto.dcm").exists() else "",
                "seg_dicom_path": seg_dicom_path,
                "seg_dir": str(co.dirs.segmentation_totalseg) if co.dirs.segmentation_totalseg.exists() else "",
                "segmentation_original_dir": str(co.dirs.segmentation_original) if co.dirs.segmentation_original.exists() else "",
                "segmentation_totalseg_manifests": seg_manifest_paths,
                "ct_dir": str(co.dirs.dicom_ct) if co.dirs.dicom_ct.exists() else "",
                "primary_nifti": str(co.primary_nifti) if co.primary_nifti and Path(co.primary_nifti).exists() else "",
                "ct_study_uid": ct_summary.get('ct_study_uid', ''),
                "dicom_related_files": [str(p) for p in co.related_dicom],
                "dicom_related_count": len(co.related_dicom),
                "nifti_files": [str(p) for p in nifti_files],
                "planned_fractions": planned_fractions,
                "beams_count": beams_count,
                "beam_energies": beam_energies,
                "treatment_machine": machine_name,
                "total_prescription_gy": prescribed_dose_gy,
                "delivered_dose_gy": co.delivered_dose_gy,
                "delivery_status": co.delivery_status,
                "delivery_method": co.delivery_method,
                "delivered_record_count": co.delivered_record_count,
                "delivered_fraction_count": co.delivered_fraction_count,
                "planned_fraction_count": co.planned_fraction_count,
                "delivery_plan_details": co.delivery_plan_details,
                "delivery_warnings": co.delivery_warnings,
                "unresolved_record_plan_uids": co.unresolved_record_plan_uids,
                "planning_ct_status": co.planning_ct_status,
                "planning_ct_referenced_series_uids": co.planning_ct_referenced_series_uids,
                **dose_plausibility,
                "course_start_date": (start_date.isoformat() if start_date else (co.course_start or "")),
                "course_end_date": end_date.isoformat() if end_date else "",
                "fractions_count": fractions_count,
                "fractions_file": str(fractions_path) if fractions_details else "",
                "dose_grid": dose_grid,
                "dose_grid_semantics": (
                    dose_grid_contract.get("semantics") if dose_grid_contract is not None else None
                ),
                "dose_response_dose_field": DOSE_RESPONSE_FIELD,
                "course_contract": course_contract,
            }
            case_meta.update(ct_summary)
            if manual_manifest:
                case_meta["segmentation_original_manifest"] = manual_manifest
            # Enrich with RP/RS tags if available
            try:
                if co.rp_path.exists():
                    ds_rp = pydicom.dcmread(str(co.rp_path), stop_before_pixels=True)
                    # Prescriptions per target (if available)
                    prescriptions = []
                    try:
                        for dr in getattr(ds_rp, 'DoseReferenceSequence', []) or []:
                            entry = {
                                'DoseReferenceNumber': getattr(dr, 'DoseReferenceNumber', None),
                                'DoseReferenceDescription': str(getattr(dr, 'DoseReferenceDescription', '')),
                                'DoseReferenceStructureType': str(getattr(dr, 'DoseReferenceStructureType', '')),
                                'TargetPrescriptionDose': float(getattr(dr, 'TargetPrescriptionDose', 0.0) or 0.0) if hasattr(dr, 'TargetPrescriptionDose') else None,
                            }
                            prescriptions.append(entry)
                    except Exception:
                        pass
                    # Plan demographic/time
                    plan_date = str(getattr(ds_rp, 'RTPlanDate', ''))
                    plan_time = str(getattr(ds_rp, 'RTPlanTime', ''))
                    plan_intent = str(getattr(ds_rp, 'RTPlanIntent', '')) if hasattr(ds_rp, 'RTPlanIntent') else ''
                    # Approval timestamps (if present)
                    approval_date = str(getattr(ds_rp, 'ApprovalStatusDate', '')) if hasattr(ds_rp, 'ApprovalStatusDate') else ''
                    approval_time = str(getattr(ds_rp, 'ApprovalStatusTime', '')) if hasattr(ds_rp, 'ApprovalStatusTime') else ''
                    review_date = str(getattr(ds_rp, 'ReviewDate', '')) if hasattr(ds_rp, 'ReviewDate') else ''
                    review_time = str(getattr(ds_rp, 'ReviewTime', '')) if hasattr(ds_rp, 'ReviewTime') else ''
                    reviewer_name = str(getattr(ds_rp, 'ReviewerName', '')) if hasattr(ds_rp, 'ReviewerName') else ''
                    # Patient demographics
                    patient_name = str(getattr(ds_rp, 'PatientName', ''))
                    patient_sex = str(getattr(ds_rp, 'PatientSex', '')) if hasattr(ds_rp, 'PatientSex') else ''
                    patient_birth_date = str(getattr(ds_rp, 'PatientBirthDate', '')) if hasattr(ds_rp, 'PatientBirthDate') else ''
                    patient_weight_kg = None
                    patient_height_m = None
                    try:
                        if hasattr(ds_rp, 'PatientWeight') and ds_rp.PatientWeight is not None:
                            patient_weight_kg = float(ds_rp.PatientWeight)
                    except Exception:
                        pass
                    try:
                        if hasattr(ds_rp, 'PatientSize') and ds_rp.PatientSize is not None:
                            patient_height_m = float(ds_rp.PatientSize)
                    except Exception:
                        pass
                    # Patient DOB to compute age at plan
                    age_years = None
                    try:
                        dob = str(getattr(ds_rp, 'PatientBirthDate', '') or getattr(ds_rp, 'PatientBirthDate', ''))
                        if dob and plan_date and len(dob) == 8 and len(plan_date) == 8:
                            d_dob = datetime.datetime.strptime(dob, '%Y%m%d').date()
                            d_plan = datetime.datetime.strptime(plan_date, '%Y%m%d').date()
                            age_years = int((d_plan - d_dob).days // 365.25)
                    except Exception:
                        pass
                    patient_bmi = None
                    try:
                        if patient_weight_kg and patient_height_m and patient_height_m > 0:
                            patient_bmi = float(patient_weight_kg / (patient_height_m ** 2))
                    except Exception:
                        pass
                    # Beam geometry summaries
                    gantry_angles = []
                    coll_angles = []
                    couch_angles = []
                    try:
                        for b in getattr(ds_rp, 'BeamSequence', []) or []:
                            cps = getattr(b, 'ControlPointSequence', []) or []
                            if cps:
                                for cp in cps:
                                    if hasattr(cp, 'GantryAngle'):
                                        try: gantry_angles.append(float(cp.GantryAngle))
                                        except (ValueError, TypeError) as e:
                                            logger.debug("Failed to parse GantryAngle: %s", e)
                                    if hasattr(cp, 'BeamLimitingDeviceAngle'):
                                        try: coll_angles.append(float(cp.BeamLimitingDeviceAngle))
                                        except (ValueError, TypeError) as e:
                                            logger.debug("Failed to parse BeamLimitingDeviceAngle: %s", e)
                                    if hasattr(cp, 'PatientSupportAngle'):
                                        try: couch_angles.append(float(cp.PatientSupportAngle))
                                        except (ValueError, TypeError) as e:
                                            logger.debug("Failed to parse PatientSupportAngle: %s", e)
                            else:
                                if hasattr(b, 'GantryAngle'):
                                    try: gantry_angles.append(float(b.GantryAngle))
                                    except (ValueError, TypeError) as e:
                                        logger.debug("Failed to parse GantryAngle: %s", e)
                                if hasattr(b, 'BeamLimitingDeviceAngle'):
                                    try: coll_angles.append(float(b.BeamLimitingDeviceAngle))
                                    except (ValueError, TypeError) as e:
                                        logger.debug("Failed to parse BeamLimitingDeviceAngle: %s", e)
                                if hasattr(b, 'PatientSupportAngle'):
                                    try: couch_angles.append(float(b.PatientSupportAngle))
                                    except (ValueError, TypeError) as e:
                                        logger.debug("Failed to parse PatientSupportAngle: %s", e)
                    except Exception:
                        pass
                    def _stats(arr):
                        import numpy as _np
                        if not arr:
                            return {'mean': None, 'std': None, 'min': None, 'max': None, 'unique': 0}
                        a = _np.array(arr, dtype=float)
                        return {
                            'mean': float(_np.nanmean(a)),
                            'std': float(_np.nanstd(a)),
                            'min': float(_np.nanmin(a)),
                            'max': float(_np.nanmax(a)),
                            'unique': int(len(_np.unique(_np.round(a, 1))))
                        }
                    geom = {
                        'gantry': _stats(gantry_angles),
                        'collimator': _stats(coll_angles),
                        'couch': _stats(couch_angles),
                    }
                    # Heuristic technique inference
                    def _infer(ga, nbeams):
                        import numpy as _np
                        if not ga:
                            return ''
                        ua = len(set(int(round(x)) for x in ga))
                        if ua > max(1, nbeams*2) and (_np.ptp(ga) > 20 or _np.std(ga) > 15):
                            return 'VMAT/ARC'
                        return 'STATIC/IMRT'
                    geom['technique_inferred'] = _infer(gantry_angles, beams_count)

                    # Per-ROI prescription mapping (if Ref ROI present)
                    prescriptions_by_roi = []
                    roiname_by_num = {}
                    try:
                        if (patient_dir / "RS.dcm").exists():
                            ds_rs_map = pydicom.dcmread(str(patient_dir / "RS.dcm"), stop_before_pixels=True)
                            for roi in getattr(ds_rs_map, 'StructureSetROISequence', []) or []:
                                roiname_by_num[int(getattr(roi, 'ROINumber', -1))] = str(getattr(roi, 'ROIName', ''))
                    except Exception:
                        pass
                    for dr in getattr(ds_rp, 'DoseReferenceSequence', []) or []:
                        try:
                            roi_num = getattr(dr, 'ReferencedROINumber', None)
                            if roi_num is None:
                                continue
                            prescriptions_by_roi.append({
                                'ROI_Number': int(roi_num),
                                'ROI_Name': roiname_by_num.get(int(roi_num), ''),
                                'TargetPrescriptionDose': float(getattr(dr, 'TargetPrescriptionDose', 0.0) or 0.0) if hasattr(dr, 'TargetPrescriptionDose') else None,
                            })
                        except Exception:
                            continue
                    # Clinicians
                    physicians_of_record = []
                    try:
                        por = getattr(ds_rp, 'PhysiciansOfRecord', None)
                        if por is not None:
                            if isinstance(por, (list, tuple)):
                                physicians_of_record = [str(x) for x in por]
                            else:
                                physicians_of_record = [str(por)]
                    except Exception:
                        pass
                    referring_physician = str(getattr(ds_rp, 'ReferringPhysicianName', '')) if hasattr(ds_rp, 'ReferringPhysicianName') else ''
                    performing_physician = str(getattr(ds_rp, 'PerformingPhysicianName', '')) if hasattr(ds_rp, 'PerformingPhysicianName') else ''
                    operators_name = []
                    try:
                        opn = getattr(ds_rp, 'OperatorsName', None)
                        if opn is not None:
                            if isinstance(opn, (list, tuple)):
                                operators_name = [str(x) for x in opn]
                            else:
                                operators_name = [str(opn)]
                    except Exception:
                        pass

                    # Per-beam information and meterset from FractionGroup
                    meterset_by_beam = {}
                    total_meterset = None
                    try:
                        total = 0.0
                        for fg in getattr(ds_rp, 'FractionGroupSequence', []) or []:
                            for rb in getattr(fg, 'ReferencedBeamSequence', []) or []:
                                bnum = getattr(rb, 'ReferencedBeamNumber', None)
                                bm = getattr(rb, 'BeamMeterset', None)
                                if bm is not None:
                                    try:
                                        bm_val = float(bm)
                                        total += bm_val
                                        if bnum is not None:
                                            meterset_by_beam[int(bnum)] = bm_val
                                    except Exception:
                                        pass
                        total_meterset = total if total > 0 else None
                    except Exception:
                        pass

                    # Beam modality/type/energy per beam and arc detection
                    beam_info = []
                    try:
                        for b in getattr(ds_rp, 'BeamSequence', []) or []:
                            cps = getattr(b, 'ControlPointSequence', []) or []
                            num_cps = len(cps) if cps else 0
                            gantry_angles = []
                            if cps:
                                for cp in cps:
                                    if hasattr(cp, 'GantryAngle'):
                                        try: gantry_angles.append(float(cp.GantryAngle))
                                        except (ValueError, TypeError) as e:
                                            logger.debug("Failed to parse GantryAngle: %s", e)
                            else:
                                if hasattr(b, 'GantryAngle'):
                                    try: gantry_angles.append(float(b.GantryAngle))
                                    except (ValueError, TypeError) as e:
                                        logger.debug("Failed to parse GantryAngle: %s", e)
                            gantry_span = float(np.ptp(gantry_angles)) if gantry_angles else None
                            is_arc = bool(num_cps >= 30 and (gantry_span or 0) > 80)
                            bnum = int(getattr(b, 'BeamNumber', 0) or 0)
                            bname = str(getattr(b, 'BeamName', ''))
                            beam_info.append({
                                'BeamNumber': bnum,
                                'BeamName': bname,
                                'BeamType': str(getattr(b, 'BeamType', '')),
                                'RadiationType': str(getattr(b, 'RadiationType', '')),
                                'NominalBeamEnergy': float(getattr(b, 'NominalBeamEnergy', 0.0) or 0.0) if hasattr(b, 'NominalBeamEnergy') else None,
                                'NumberOfControlPoints': num_cps,
                                'GantrySpan': gantry_span,
                                'IsArc': is_arc,
                                'BeamMeterset': meterset_by_beam.get(bnum),
                            })
                    except Exception:
                        pass

                    # Heuristic technique inference (enhanced)
                    try:
                        arcs = [bi for bi in beam_info if bi.get('IsArc')]
                        static = [bi for bi in beam_info if not bi.get('IsArc')]
                        technique_inferred = 'VMAT/ARC' if arcs else 'STATIC/IMRT'
                    except Exception:
                        technique_inferred = case_meta.get('beam_geometry', {}).get('technique_inferred', '')

                    case_meta.update({
                        "plan_name": getattr(ds_rp, "RTPlanLabel", "") or getattr(ds_rp, "RTPlanName", ""),
                        "plan_date": plan_date,
                        "plan_time": plan_time,
                        "plan_intent": plan_intent,
                        "approval": getattr(ds_rp, "ApprovalStatus", ""),
                        "approval_status_date": approval_date,
                        "approval_status_time": approval_time,
                        "review_date": review_date,
                        "review_time": review_time,
                        "reviewer_name": reviewer_name,
                        "patient_name": patient_name,
                        "patient_sex": patient_sex,
                        "patient_birth_date": patient_birth_date,
                        "patient_weight_kg": patient_weight_kg,
                        "patient_height_m": patient_height_m,
                        "patient_bmi": patient_bmi,
                        "prescriptions": prescriptions,
                        "prescriptions_by_roi": prescriptions_by_roi,
                        "patient_age_at_plan": age_years,
                        "beam_geometry": geom,
                        "beam_info": beam_info,
                        "total_meterset": total_meterset,
                        "meterset_by_beam": meterset_by_beam or None,
                        "physicians_of_record": physicians_of_record,
                        "referring_physician": referring_physician,
                        "performing_physician": performing_physician,
                        "operators_name": operators_name,
                        "technique_inferred": technique_inferred,
                    })
            except Exception:
                pass
            try:
                if (patient_dir / "RS.dcm").exists():
                    ds_rs = pydicom.dcmread(str(patient_dir / "RS.dcm"), stop_before_pixels=True)
                    rois = []
                    for roi in getattr(ds_rs, 'StructureSetROISequence', []) or []:
                        nm = getattr(roi, 'ROIName', None)
                        if nm:
                            rois.append(str(nm))
                    case_meta["structures"] = ", ".join(rois)
                    # ROI counts by type (heuristic by name)
                    try:
                        name_l = [x.lower() for x in rois]
                        case_meta['roi_count'] = len(rois)
                        case_meta['ptv_count'] = sum(1 for x in name_l if 'ptv' in x)
                        case_meta['ctv_count'] = sum(1 for x in name_l if 'ctv' in x)
                        case_meta['oar_count'] = case_meta['roi_count'] - case_meta['ptv_count'] - case_meta['ctv_count']
                    except Exception:
                        pass
            except Exception:
                pass
            # RD dose metadata already collected above
            if dose_grid:
                case_meta['dose_units'] = None
                case_meta['dose_type'] = None
                try:
                    ds_rd2 = pydicom.dcmread(str(co.rd_path), stop_before_pixels=True)
                    case_meta['dose_units'] = str(getattr(ds_rd2, 'DoseUnits', ''))
                    case_meta['dose_type'] = str(getattr(ds_rd2, 'DoseType', ''))
                except Exception:
                    pass

            # Write JSON + XLSX (one-row sheet)
            metadata_path = meta_dir / "case_metadata.json"
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(case_meta, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                raise CourseContractError(
                    f"failed to write authoritative course contract for {patient_dir}: {exc}"
                ) from exc
            # Producer-side validation closes the same path, UID, delivery,
            # summation-type, and dose-QC checks enforced downstream.
            load_course_contract(patient_dir)
            try:
                import pandas as _pd

                _pd.DataFrame([case_meta]).to_excel(meta_dir / "case_metadata.xlsx", index=False)
            except Exception as exc:
                logger.debug("Failed to write case metadata XLSX for %s: %s", patient_dir, exc)
        except CourseContractError:
            raise
        except Exception as e:
            logger.debug("Failed to write per-case metadata for %s: %s", patient_dir, e)

    # Save copy manager caches and log statistics
    copy_manager.save_caches()
    logger.info("DICOM copy statistics: %s", copy_manager.stats)

    return outputs
