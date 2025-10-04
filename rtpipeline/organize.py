from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from .config import PipelineConfig
from .ct import index_ct_series, pick_primary_series, copy_ct_series
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


def _safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and dst.is_file() and not os.path.samefile(src, dst):
        dst.unlink()
    shutil.copy2(src, dst)


def _copy_into(src: Path, dst_dir: Path, prefix: Optional[str] = None) -> Path:
    """Copy src into dst_dir, preserving name and avoiding clashes."""
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
    for base, _, files in os.walk(dicom_root):
        for name in files:
            if not name.lower().endswith('.dcm'):
                continue
            p = Path(base) / name
            ds = read_dicom(p)
            if ds is None:
                continue
            patient_id = str(get(ds, (0x0010, 0x0020), "")) or ""
            series_uid = str(get(ds, (0x0020, 0x000E), "")) or ""
            if patient_id and series_uid:
                series_index.setdefault((patient_id, series_uid), []).append(p)
            modality = str(getattr(ds, 'Modality', '') or '')
            if modality != 'REG' or not patient_id:
                continue
            reg_item = {
                'path': p,
                'for_uids': set(),
                'referenced_series': set(),
            }
            try:
                for ref_for in getattr(ds, 'ReferencedFrameOfReferenceSequence', []) or []:
                    for_uid = str(getattr(ref_for, 'FrameOfReferenceUID', '') or '')
                    if for_uid:
                        reg_item['for_uids'].add(for_uid)
                    for study in getattr(ref_for, 'RTReferencedStudySequence', []) or []:
                        for series in getattr(study, 'RTReferencedSeriesSequence', []) or []:
                            series_uid_ref = str(getattr(series, 'SeriesInstanceUID', '') or '')
                            if series_uid_ref:
                                reg_item['referenced_series'].add(series_uid_ref)
            except Exception as exc:
                logger.debug("Failed indexing registration %s: %s", p, exc)
                continue
            registrations.setdefault(patient_id, []).append(reg_item)
    return series_index, registrations

def _create_summed_plan(plan_files: List[Path], total_dose_gy: float, out_plan_path: Path) -> None:
    if not plan_files:
        raise ValueError("No plan files to sum")
    ds_plan = pydicom.dcmread(str(plan_files[0]), stop_before_pixels=True)
    new_plan = copy.deepcopy(ds_plan)

    new_plan.SeriesInstanceUID = generate_uid()
    new_plan.SOPInstanceUID = generate_uid()
    new_plan.SeriesDescription = f"Summed Plan ({len(plan_files)} fractions)"
    new_plan.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
    new_plan.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S")

    # Update prescription
    try:
        if hasattr(new_plan, "DoseReferenceSequence") and new_plan.DoseReferenceSequence:
            for dose_ref in new_plan.DoseReferenceSequence:
                if hasattr(dose_ref, "TargetPrescriptionDose"):
                    dose_ref.TargetPrescriptionDose = float(total_dose_gy)
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning("Failed to update prescription dose in summed plan: %s", e)

    # Update fraction group meterset proportionally if possible
    try:
        if hasattr(ds_plan, "DoseReferenceSequence") and ds_plan.DoseReferenceSequence:
            base_rx = float(ds_plan.DoseReferenceSequence[0].TargetPrescriptionDose)
        else:
            base_rx = float(total_dose_gy)
        if hasattr(new_plan, "FractionGroupSequence") and new_plan.FractionGroupSequence and base_rx > 0:
            ratio = float(total_dose_gy) / base_rx
            for fg in new_plan.FractionGroupSequence:
                if hasattr(fg, "ReferencedBeamSequence") and fg.ReferencedBeamSequence:
                    for beam_ref in fg.ReferencedBeamSequence:
                        if hasattr(beam_ref, "BeamMeterset"):
                            beam_ref.BeamMeterset = float(beam_ref.BeamMeterset) * ratio
    except (AttributeError, ValueError, TypeError, ZeroDivisionError) as e:
        logger.warning("Failed to update meterset in summed plan: %s", e)

    new_plan.save_as(str(out_plan_path))


def _sum_doses_with_resample(dose_files: List[Path], out_dose_path: Path) -> None:
    if not dose_files:
        raise ValueError("No dose files to sum")

    ref_idx = 0
    best_resolution = float("inf")
    for i, p in enumerate(dose_files):
        ds = pydicom.dcmread(str(p))
        if not hasattr(ds, "PixelSpacing") or len(ds.PixelSpacing) < 2:
            continue
        pixel_spacing = list(map(float, ds.PixelSpacing))
        slice_thickness = 1.0
        if hasattr(ds, "GridFrameOffsetVector") and len(ds.GridFrameOffsetVector) > 1:
            slice_thickness = abs(float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]))
        voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
        if voxel_volume < best_resolution:
            best_resolution = voxel_volume
            ref_idx = i

    ds_ref = pydicom.dcmread(str(dose_files[ref_idx]))
    arr_ref = ds_ref.pixel_array.astype('float32')
    dose_scaling_ref = float(getattr(ds_ref, "DoseGridScaling", 1.0))
    arr_ref *= dose_scaling_ref

    rows_ref, cols_ref = ds_ref.Rows, ds_ref.Columns
    frames_ref = getattr(ds_ref, "NumberOfFrames", 1)
    origin_ref = list(map(float, getattr(ds_ref, "ImagePositionPatient", [0, 0, 0])))
    pixel_spacing_ref = list(map(float, getattr(ds_ref, "PixelSpacing", [1.0, 1.0])))
    offsets_ref = getattr(ds_ref, "GridFrameOffsetVector", None)
    if offsets_ref is not None and len(offsets_ref) == frames_ref:
        z_positions_ref = np.array([origin_ref[2] + float(offset) for offset in offsets_ref])
    else:
        z_positions_ref = np.array([origin_ref[2] + i for i in range(frames_ref)])
    y_positions_ref = np.array([origin_ref[1] + r * pixel_spacing_ref[0] for r in range(rows_ref)])
    x_positions_ref = np.array([origin_ref[0] + c * pixel_spacing_ref[1] for c in range(cols_ref)])

    accumulated = arr_ref.copy()

    for i, p in enumerate(dose_files):
        if i == ref_idx:
            continue
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array.astype('float32')
        arr *= float(getattr(ds, "DoseGridScaling", 1.0))

        rows, cols = ds.Rows, ds.Columns
        frames = getattr(ds, "NumberOfFrames", 1)
        origin = list(map(float, getattr(ds, "ImagePositionPatient", [0, 0, 0])))
        pixel_spacing = list(map(float, getattr(ds, "PixelSpacing", [1.0, 1.0])))
        offsets = getattr(ds, "GridFrameOffsetVector", None)
        if offsets is not None and len(offsets) == frames:
            z_coords = np.array([origin[2] + float(offset) for offset in offsets])
        else:
            z_coords = np.array([origin[2] + j for j in range(frames)])

        if frames == 1:
            arr = arr.reshape((1, rows, cols))

        Z, Y, X = np.meshgrid(z_positions_ref, y_positions_ref, x_positions_ref, indexing="ij")
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

    # Convert to int representation with scaling
    max_dose = float(np.nanmax(accumulated)) if accumulated.size else 0.0
    if max_dose > 1000:
        scaling_factor = 10.0
    elif max_dose > 100:
        scaling_factor = 100.0
    else:
        scaling_factor = 1000.0

    accumulated = np.nan_to_num(accumulated, nan=0.0)
    accumulated_int = np.rint(accumulated * scaling_factor).astype('int32')

    new_ds = copy.deepcopy(ds_ref)
    new_ds.SOPInstanceUID = generate_uid()
    new_ds.SeriesInstanceUID = generate_uid()
    new_ds.SeriesDescription = f"Summed Dose ({len(dose_files)} fractions)"
    new_ds.DoseSummationType = "PLAN"
    new_ds.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
    new_ds.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S")

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

    new_ds.PixelData = accumulated_int.tobytes()
    new_ds.save_as(str(out_dose_path))


def organize_and_merge(config: PipelineConfig) -> List[CourseOutput]:
    """End-to-end RT organization and course merging according to config."""
    config.ensure_dirs()

    # Index CTs, extract RT sets, link and group into courses
    ct_index = index_ct_series(config.dicom_root)
    plans, doses, structs = extract_rt(config.dicom_root)
    linked_sets = link_rt_sets(plans, doses, structs)
    courses = group_by_course(linked_sets, config.merge_criteria, config.max_days_between_plans)
    series_index, registrations_index = _index_series_and_registrations(config.dicom_root)

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
            _copy_into(src, course_dirs.dicom_rtplan)
        for src in dose_paths:
            _copy_into(src, course_dirs.dicom_rtdose)
        for src in struct_candidates:
            _copy_into(src, course_dirs.dicom_rtstruct)

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

        if plan_paths and dose_paths:
            if len(plan_paths) == 1 and len(dose_paths) == 1:
                _safe_copy(plan_paths[0], rp_dst)
                _safe_copy(dose_paths[0], rd_dst)
            else:
                _sum_doses_with_resample(dose_paths, rd_dst)
                _create_summed_plan(plan_paths, total_rx, rp_dst)

        course_study = items_sorted[0].ct_study_uid if items_sorted else None
        struct_path = struct_candidates[0] if struct_candidates else None
        if struct_path is None and course_study:
            for s in structs:
                if str(s.patient_id) == patient_id and (s.study_uid == course_study or not course_study):
                    struct_path = s.path
                    _copy_into(s.path, course_dirs.dicom_rtstruct)
                    break
        if struct_path:
            _safe_copy(struct_path, rs_dst)

        if course_study and patient_id in ct_index and course_study in ct_index[patient_id]:
            series = pick_primary_series(ct_index[patient_id][course_study])
            if series:
                copy_ct_series(series, course_dirs.dicom_ct)
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
                    related_outputs.append(_copy_into(reg_path, course_dirs.dicom_related / "REG"))
                    seen_related.add(reg_path)
                for series_uid in reg.get('referenced_series', set()):
                    for src in series_index.get((patient_id, series_uid), []):
                        if src not in seen_related and src.exists():
                            dest_dir = course_dirs.dicom_related / _sanitize_name(series_uid, "series")
                            related_outputs.append(_copy_into(src, dest_dir))
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
        )

    if course_tasks:
        results = run_tasks_with_adaptive_workers(
            "Organize",
            course_tasks,
            lambda task: _process_course(task[0], task[1], task[2], task[3]),
            max_workers=config.effective_workers(),
            logger=logger,
            show_progress=True,
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

            rs_dst = course_dir / "RS.dcm"
            primary_struct = s_list[0].path
            _safe_copy(primary_struct, rs_dst)
            for s in s_list:
                _copy_into(s.path, course_dirs.dicom_rtstruct)

            course_study = s_list[0].study_uid
            if course_study and patient_id in ct_index and course_study in ct_index[patient_id]:
                series = pick_primary_series(ct_index[patient_id][course_study])
                if series:
                    copy_ct_series(series, course_dirs.dicom_ct)
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
                        related_outputs.append(_copy_into(reg_path, course_dirs.dicom_related / "REG"))
                        seen_related.add(reg_path)
                    for series_uid in reg.get('referenced_series', set()):
                        for src in series_index.get((patient_id, series_uid), []):
                            if src not in seen_related and src.exists():
                                dest_dir = course_dirs.dicom_related / _sanitize_name(series_uid, "series")
                                related_outputs.append(_copy_into(src, dest_dir))
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
            plan_uids = set()
            items_sorted = []  # Not available here; we use on-disk RP only where needed below
            try:
                rp_path = patient_dir / "RP.dcm"
                if rp_path.exists():
                    ds_tmp = pydicom.dcmread(str(rp_path), stop_before_pixels=True)
                    plan_uids = {str(getattr(ds_tmp, 'SOPInstanceUID', ''))}
            except Exception:
                pass
            # Total prescription dose across plans
            plan_total_rx = 0.0
            for pth in [co.rp_path]:
                try:
                    ds_p = pydicom.dcmread(str(pth), stop_before_pixels=True)
                    if hasattr(ds_p, "DoseReferenceSequence") and ds_p.DoseReferenceSequence:
                        for dr in ds_p.DoseReferenceSequence:
                            if hasattr(dr, "TargetPrescriptionDose") and dr.TargetPrescriptionDose is not None:
                                plan_total_rx += float(dr.TargetPrescriptionDose)
                                break
                except Exception:
                    continue
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
                for base, _, files in os.walk(config.dicom_root):
                    for fn in files:
                        if not (fn.startswith('RT') and fn.lower().endswith('.dcm')):
                            continue
                        p = Path(base) / fn
                        try:
                            ds_rt = pydicom.dcmread(str(p), stop_before_pixels=True)
                        except Exception:
                            continue
                        if getattr(ds_rt, 'PatientID', None) and str(ds_rt.PatientID) != str(co.patient_id):
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
                        frac_entry: Dict[str, object] = {
                            "treatment_date": d.isoformat() if d else str(rt_date),
                            "treatment_time": str(rt_time or ""),
                            "plan_sop": ref_uid or "",
                            "fraction_number": int(frac_num) if frac_num is not None else None,
                            "treatment_machine": str(machine or ""),
                            "source_path": str(p),
                            "sop_instance_uid": str(getattr(ds_rt, "SOPInstanceUID", "")),
                            "delivered_dose_gy": None,
                            "beam_meterset": None,
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
                        try:
                            if hasattr(ds_rt, "BeamSequence") and ds_rt.BeamSequence:
                                meterset = 0.0
                                for beam in ds_rt.BeamSequence:
                                    delivered = getattr(beam, "AccumulatedDose", None)
                                    if delivered is not None:
                                        meterset += float(delivered)
                                if meterset:
                                    frac_entry["beam_meterset"] = meterset
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
                        plan_primary = plan_uids[0] if plan_uids else ""
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

                        fractions_count = len(df_frac)
                    else:
                        df_frac_raw.to_excel(fractions_path, index=False)
                except Exception as exc:
                    logger.warning("Failed to write fractions summary for %s: %s", patient_dir, exc)
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
                            'ct_reconstruction_algorithm': str(getattr(ds_ct, 'ReconstructionAlgorithm', '')) if hasattr(ds_ct, 'ReconstructionAlgorithm') else str(getattr(ds_ct, 'ReconstructionMethod', '')),
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
                                        except Exception: pass
                                    if hasattr(cp, 'BeamLimitingDeviceAngle'):
                                        try: coll_angles.append(float(cp.BeamLimitingDeviceAngle))
                                        except Exception: pass
                                    if hasattr(cp, 'PatientSupportAngle'):
                                        try: couch_angles.append(float(cp.PatientSupportAngle))
                                        except Exception: pass
                            else:
                                if hasattr(b, 'GantryAngle'):
                                    try: gantry_angles.append(float(b.GantryAngle))
                                    except Exception: pass
                                if hasattr(b, 'BeamLimitingDeviceAngle'):
                                    try: coll_angles.append(float(b.BeamLimitingDeviceAngle))
                                    except Exception: pass
                                if hasattr(b, 'PatientSupportAngle'):
                                    try: couch_angles.append(float(b.PatientSupportAngle))
                                    except Exception: pass
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
                                        except Exception: pass
                            else:
                                if hasattr(b, 'GantryAngle'):
                                    try: gantry_angles.append(float(b.GantryAngle))
                                    except Exception: pass
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

    return outputs
