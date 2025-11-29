from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import SimpleITK as sitk
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from .config import PipelineConfig
from .ct import index_ct_series, pick_primary_series, copy_ct_series
from .dicom_copy import DicomCopyConfig, DicomCopyManager, get_copy_manager, reset_copy_manager
from .layout import CourseDirs, build_course_dirs, course_dir_name
from .metadata import LinkedSet, group_by_course, link_rt_sets, parse_date
from .rt_details import extract_rt, StructInfo
from .utils import ensure_dir, run_tasks_with_adaptive_workers, read_dicom, get
from .segmentation import _ensure_ct_nifti, _strip_nifti_base, run_dcm2niix, _derive_nifti_name

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


def _safe_copy(
    src: Path,
    dst: Path,
    copy_manager: Optional[DicomCopyManager] = None,
) -> None:
    """Copy DICOM file to destination with optional deduplication."""
    if copy_manager is not None:
        copy_manager.copy_dicom(src, dst, skip_if_exists=False)
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

    course_id = str(data.get("course_id") or (meta_hint.get("dir_name") if meta_hint else course_dir.name))
    course_start = data.get("course_start_date") or data.get("course_start") or (meta_hint.get("start_iso") if meta_hint else None)
    if isinstance(course_start, str) and not course_start:
        course_start = None

    rp_path = course_dir / "RP.dcm"
    rd_path = course_dir / "RD.dcm"
    rs_path = course_dir / "RS.dcm"
    if not rp_path.exists() or not rd_path.exists():
        return None

    primary_nifti: Optional[Path] = None
    primary_str = data.get("primary_nifti") if data else None
    if isinstance(primary_str, str) and primary_str:
        cand = Path(primary_str)
        if cand.exists():
            primary_nifti = cand
    if primary_nifti is None and course_dirs.nifti.exists():
        candidates = sorted(course_dirs.nifti.glob("*.nii*"))
        for cand in candidates:
            if cand.is_file():
                primary_nifti = cand
                break

    related_files: List[Path] = []
    related_list = data.get("dicom_related_files") if data else None
    if isinstance(related_list, list):
        for entry in related_list:
            if not isinstance(entry, str):
                continue
            cand = Path(entry)
            if cand.exists():
                related_files.append(cand)
    if not related_files and course_dirs.dicom_related.exists():
        related_files = [p for p in course_dirs.dicom_related.rglob("*.dcm") if p.is_file()]

    total_rx = data.get("total_prescription_gy") if data else None
    try:
        total_rx_val = float(total_rx) if total_rx not in (None, "") else None
    except (TypeError, ValueError):
        total_rx_val = None

    plan_uid: Optional[str] = None
    dose_uid: Optional[str] = None
    source_plan_uids: set[str] = set()
    source_dose_uids: set[str] = set()

    try:
        ds_plan = pydicom.dcmread(str(rp_path), stop_before_pixels=True)
        plan_uid = str(getattr(ds_plan, "SOPInstanceUID", "") or None)
        for ref in getattr(ds_plan, "ReferencedRTPlanSequence", []) or []:
            uid = getattr(ref, "ReferencedSOPInstanceUID", None)
            if uid:
                source_plan_uids.add(str(uid))
    except Exception:
        plan_uid = None

    try:
        ds_dose = pydicom.dcmread(str(rd_path), stop_before_pixels=True)
        dose_uid = str(getattr(ds_dose, "SOPInstanceUID", "") or None)
        for ref in getattr(ds_dose, "ReferencedRTPlanSequence", []) or []:
            uid = getattr(ref, "ReferencedSOPInstanceUID", None)
            if uid:
                source_plan_uids.add(str(uid))
        for ref in getattr(ds_dose, "ReferencedInstanceSequence", []) or []:
            uid = getattr(ref, "ReferencedSOPInstanceUID", None)
            if uid:
                source_dose_uids.add(str(uid))
    except Exception:
        dose_uid = None

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
    )


def _sanitize_name(text: str, fallback: str = "item") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned[:80]


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


def _export_original_segmentation(
    course: CourseOutput,
    overwrite: bool,
) -> Optional[dict]:
    if not course.rs_path or not course.rs_path.exists() or not course.primary_nifti or not course.primary_nifti.exists():
        return None
    seg_root = course.dirs.segmentation_original
    base_name = _strip_nifti_base(course.primary_nifti)
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
        ct_img = sitk.ReadImage(str(course.primary_nifti))
    except Exception as exc:
        logger.warning("Failed to load primary NIfTI for %s: %s", course.dirs.root, exc)
        return None

    try:
        builder = RTStructBuilder.create_from(
            dicom_series_path=str(course.dirs.dicom_ct),
            rt_struct_path=str(course.rs_path),
        )
    except Exception as exc:
        logger.warning("Failed to load RTSTRUCT for segmentation export (%s): %s", course.rs_path, exc)
        return None

    used_names: dict[str, int] = {}
    manifest = {
        "model": "manual",
        "source_rtstruct": str(course.rs_path),
        "source_nifti": str(course.primary_nifti),
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
            logger.warning("Failed to write segmentation manifest for %s: %s", course.dirs.root, exc)
        return manifest
    return None


def _index_series_and_registrations(
    dicom_root: Path,
) -> tuple[Dict[tuple[str, str], List[Path]], Dict[str, List[Dict[str, object]]]]:
    series_index: Dict[tuple[str, str], List[Path]] = {}
    registrations: Dict[str, List[Dict[str, object]]] = {}
    series_meta: Dict[tuple[str, str], Dict[str, object]] = {}
    for base, _, files in os.walk(dicom_root):
        for name in files:
            if not name.lower().endswith('.dcm'):
                continue
            p = Path(base) / name
            ds = read_dicom(p)
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
    origin_ref = list(map(float, getattr(ds_ref, "ImagePositionPatient", [0, 0, 0])))
    pixel_spacing_ref = list(map(float, getattr(ds_ref, "PixelSpacing", [1.0, 1.0])))
    offsets_ref = getattr(ds_ref, "GridFrameOffsetVector", None)
    
    if offsets_ref is not None and len(offsets_ref) == frames_ref:
        z_positions_ref = np.array([origin_ref[2] + float(offset) for offset in offsets_ref])
    else:
        z_positions_ref = np.array([origin_ref[2] + i for i in range(frames_ref)])
    
    y_positions_ref = np.array([origin_ref[1] + r * pixel_spacing_ref[0] for r in range(rows_ref)])
    x_positions_ref = np.array([origin_ref[0] + c * pixel_spacing_ref[1] for c in range(cols_ref)])

    # Pre-calculate meshgrid for reference (optimization)
    Z, Y, X = np.meshgrid(z_positions_ref, y_positions_ref, x_positions_ref, indexing="ij")

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
            origin = list(map(float, getattr(ds, "ImagePositionPatient", [0, 0, 0])))
            pixel_spacing = list(map(float, getattr(ds, "PixelSpacing", [1.0, 1.0])))
            offsets = getattr(ds, "GridFrameOffsetVector", None)
            
            if offsets is not None and len(offsets) == frames:
                z_coords = np.array([origin[2] + float(offset) for offset in offsets])
            else:
                z_coords = np.array([origin[2] + j for j in range(frames)])

            if frames == 1:
                arr = arr.reshape((1, rows, cols))

            # Resampling logic
            z_min, z_max = z_coords.min(), z_coords.max()
            y_min, y_max = origin[1], origin[1] + (rows - 1) * pixel_spacing[0]
            x_min, x_max = origin[0], origin[0] + (cols - 1) * pixel_spacing[1]

            Zc = np.clip(Z, z_min, z_max)
            Yc = np.clip(Y, y_min, y_max)
            Xc = np.clip(X, x_min, x_max)

            z_idx = np.searchsorted(z_coords, Zc) - 0.5
            y_idx = (Yc - origin[1]) / pixel_spacing[0]
            x_idx = (Xc - origin[0]) / pixel_spacing[1]

            coords = np.stack([z_idx, y_idx, x_idx], axis=0)
            resampled = map_coordinates(arr, coords, order=1, mode="constant", cval=0.0)
            resampled = resampled.reshape((frames_ref, rows_ref, cols_ref))
            
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
    accumulated_int = np.rint(accumulated * scaling_factor).astype("int32")

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


def _index_rt_files(root: Path) -> Dict[str, List[Path]]:
    """Index all RT DICOM files by PatientID to avoid repeated scanning."""
    index = defaultdict(list)
    logger.info("Indexing RT files in %s...", root)
    count = 0
    for base, _, files in os.walk(root):
        for fn in files:
            if not (fn.startswith('RT') and fn.lower().endswith('.dcm')):
                continue
            path = Path(base) / fn
            try:
                # We read specific tags to be fast. 
                # We need PatientID for grouping.
                # We might read more details later, but for indexing this is enough.
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["PatientID"])
                pid = str(getattr(ds, "PatientID", "")).strip()
                if pid:
                    index[pid].append(path)
                    count += 1
            except Exception:
                continue
    logger.info("Indexed %d RT files for %d patients", count, len(index))
    return index


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

    # Index CTs, extract RT sets, link and group into courses
    ct_index = index_ct_series(config.dicom_root)
    rt_file_index = _index_rt_files(config.dicom_root)
    plans, doses, structs = extract_rt(config.dicom_root)
    linked_sets = link_rt_sets(plans, doses, structs)
    courses = group_by_course(linked_sets, config.merge_criteria, config.max_days_between_plans)
    series_index, registrations_index, series_meta = _index_series_and_registrations(config.dicom_root)

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

        rp_dst = course_dir / "RP.dcm"
        rd_dst = course_dir / "RD.dcm"
        rs_dst = course_dir / "RS.dcm"

        plan_paths: List[Path] = []
        dose_paths: List[Path] = []
        struct_candidates: List[Path] = []

        for it in items_sorted:
            if it.plan.path not in plan_paths:
                plan_paths.append(it.plan.path)
            if it.dose.path not in dose_paths:
                dose_paths.append(it.dose.path)
            if it.struct and it.struct.path not in struct_candidates:
                struct_candidates.append(it.struct.path)

        for src in plan_paths:
            _copy_into(src, course_dirs.dicom_rtplan, copy_manager=copy_manager)
        for src in dose_paths:
            _copy_into(src, course_dirs.dicom_rtdose, copy_manager=copy_manager)
        for src in struct_candidates:
            _copy_into(src, course_dirs.dicom_rtstruct, copy_manager=copy_manager)

        total_rx = 0.0
        for p in plan_paths:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if hasattr(ds, "DoseReferenceSequence") and ds.DoseReferenceSequence:
                    for dr in ds.DoseReferenceSequence:
                        if hasattr(dr, "TargetPrescriptionDose") and dr.TargetPrescriptionDose is not None:
                            total_rx += float(dr.TargetPrescriptionDose)
                            break
            except Exception:
                continue

        plan_sop_uid: Optional[str] = None
        dose_sop_uid: Optional[str] = None
        source_plan_uids: list[str] = []
        source_dose_uids: list[str] = []

        if plan_paths and dose_paths:
            if len(plan_paths) == 1 and len(dose_paths) == 1:
                _safe_copy(plan_paths[0], rp_dst, copy_manager=copy_manager)
                _safe_copy(dose_paths[0], rd_dst, copy_manager=copy_manager)
                try:
                    ds_plan_single = pydicom.dcmread(str(plan_paths[0]), stop_before_pixels=True)
                    plan_sop_uid = str(getattr(ds_plan_single, "SOPInstanceUID", "") or None)
                except Exception:
                    plan_sop_uid = None
                try:
                    ds_dose_single = pydicom.dcmread(str(dose_paths[0]), stop_before_pixels=False)
                    dose_sop_uid = str(getattr(ds_dose_single, "SOPInstanceUID", "") or None)
                except Exception:
                    dose_sop_uid = None
                if plan_sop_uid:
                    source_plan_uids.append(plan_sop_uid)
                if dose_sop_uid:
                    source_dose_uids.append(dose_sop_uid)
            else:
                plan_sum_ds, plan_ds_list, source_plan_uids = _create_summed_plan(plan_paths, total_rx if total_rx else None)
                dose_sum_ds, dose_ds_list, source_dose_uids = _sum_doses_with_resample(dose_paths, plan_sum_ds, plan_ds_list)

                ref_dose_item = Dataset()
                ref_dose_item.ReferencedSOPClassUID = str(getattr(dose_sum_ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.481.2"))
                ref_dose_item.ReferencedSOPInstanceUID = str(dose_sum_ds.SOPInstanceUID)
                plan_sum_ds.ReferencedDoseSequence = Sequence([ref_dose_item])

                plan_sum_ds.save_as(str(rp_dst))
                dose_sum_ds.save_as(str(rd_dst))

                plan_sop_uid = str(plan_sum_ds.SOPInstanceUID)
                dose_sop_uid = str(dose_sum_ds.SOPInstanceUID)
        else:
            if plan_paths:
                _safe_copy(plan_paths[0], rp_dst, copy_manager=copy_manager)
                try:
                    ds_plan_single = pydicom.dcmread(str(plan_paths[0]), stop_before_pixels=True)
                    plan_sop_uid = str(getattr(ds_plan_single, "SOPInstanceUID", "") or None)
                except Exception:
                    plan_sop_uid = None
                if plan_sop_uid:
                    source_plan_uids.append(plan_sop_uid)
            if dose_paths:
                _safe_copy(dose_paths[0], rd_dst, copy_manager=copy_manager)
                try:
                    ds_dose_single = pydicom.dcmread(str(dose_paths[0]), stop_before_pixels=False)
                    dose_sop_uid = str(getattr(ds_dose_single, "SOPInstanceUID", "") or None)
                except Exception:
                    dose_sop_uid = None
                if dose_sop_uid:
                    source_dose_uids.append(dose_sop_uid)

        course_study = items_sorted[0].ct_study_uid if items_sorted else None
        struct_path = struct_candidates[0] if struct_candidates else None
        if struct_path is None and course_study:
            for s in structs:
                if str(s.patient_id) == patient_id and (s.study_uid == course_study or not course_study):
                    struct_path = s.path
                    _copy_into(s.path, course_dirs.dicom_rtstruct, copy_manager=copy_manager)
                    break
        if struct_path:
            _safe_copy(struct_path, rs_dst, copy_manager=copy_manager)

        if course_study and patient_id in ct_index and course_study in ct_index[patient_id]:
            series = pick_primary_series(ct_index[patient_id][course_study])
            if series:
                first_inst = series[0] if series else None
                if first_inst is not None and getattr(first_inst, 'series_uid', None):
                    course_ct_series_uids.add(str(first_inst.series_uid))
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

            rs_dst = course_dir / "RS.dcm"
            primary_struct = s_list[0].path
            _safe_copy(primary_struct, rs_dst, copy_manager=copy_manager)
            for s in s_list:
                _copy_into(s.path, course_dirs.dicom_rtstruct, copy_manager=copy_manager)

            course_study = s_list[0].study_uid
            if course_study and patient_id in ct_index and course_study in ct_index[patient_id]:
                series = pick_primary_series(ct_index[patient_id][course_study])
                if series:
                    first_inst = series[0] if series else None
                    if first_inst is not None and getattr(first_inst, 'series_uid', None):
                        course_ct_series_uids.add(str(first_inst.series_uid))
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
                        metadata.update(
                            {
                                "nifti_path": str(target_path),
                                "source_directory": str(dicom_dir),
                                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                            }
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
            # Total prescription dose across plans
            def _infer_plan_rx(ds_plan: pydicom.dataset.FileDataset) -> float | None:
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
                # fallback to alternative dose fields (common in some planners)
                for dr in dose_seq:
                    alt_fields = [
                        getattr(dr, "DeliveredDoseReferenceDoseValue", None),
                        getattr(dr, "DeliveryMaximumDose", None),
                        getattr(dr, "OrganAtRiskMaximumDose", None),
                    ]
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

            plan_total_rx = 0.0
            try:
                ds_plan = pydicom.dcmread(str(co.rp_path), stop_before_pixels=True)
            except Exception:
                ds_plan = None
            if ds_plan is not None:
                inferred = _infer_plan_rx(ds_plan)
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
                        ref_uid = None
                        try:
                            for ref in getattr(ds_rt, 'ReferencedRTPlanSequence', []) or []:
                                ref_uid = getattr(ref, 'ReferencedSOPInstanceUID', None)
                                if ref_uid:
                                    break
                        except Exception:
                            pass
                        if plan_uids and ref_uid and ref_uid not in plan_uids:
                            continue
                        rt_date = getattr(ds_rt, 'TreatmentDate', None) or getattr(ds_rt, 'SeriesDate', None)
                        rt_time = getattr(ds_rt, 'TreatmentTime', None)
                        frac_num = getattr(ds_rt, 'ReferencedFractionNumber', None)
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
                            "beam_meterset": None,
                            "delivered_meterset": None,
                            "beam_delivery": [],
                        }
                        try:
                            delivered = getattr(ds_rt, "ReferencedDoseReferenceSequence", None)
                            if delivered:
                                first = delivered[0]
                                dose_value = getattr(first, "DeliveredDoseReferenceDoseValue", None)
                                if dose_value is not None:
                                    frac_entry["delivered_dose_gy"] = float(dose_value)
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
                "total_prescription_gy": co.total_prescription_gy or plan_total_rx,
                "course_start_date": (start_date.isoformat() if start_date else (co.course_start or "")),
                "course_end_date": end_date.isoformat() if end_date else "",
                "fractions_count": fractions_count,
                "fractions_file": str(fractions_path) if fractions_details else "",
                "dose_grid": dose_grid,
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
            try:
                with open(meta_dir / "case_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(case_meta, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.warning("Failed to write case metadata JSON for %s: %s", patient_dir, exc)
            try:
                import pandas as _pd

                _pd.DataFrame([case_meta]).to_excel(meta_dir / "case_metadata.xlsx", index=False)
            except Exception as exc:
                logger.debug("Failed to write case metadata XLSX for %s: %s", patient_dir, exc)
        except Exception as e:
            logger.debug("Failed to write per-case metadata for %s: %s", patient_dir, e)

    # Save copy manager caches and log statistics
    copy_manager.save_caches()
    logger.info("DICOM copy statistics: %s", copy_manager.stats)

    return outputs
