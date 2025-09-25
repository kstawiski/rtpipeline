from __future__ import annotations

import copy
import json
import datetime
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from .config import PipelineConfig
from .ct import index_ct_series, pick_primary_series, copy_ct_series
from .metadata import LinkedSet, group_by_course, link_rt_sets
from .rt_details import extract_rt
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class CourseOutput:
    patient_id: str
    course_key: str
    dir: Path
    rp_path: Path
    rd_path: Path
    rs_path: Optional[Path]
    ct_dir: Optional[Path]
    total_prescription_gy: float | None


def _safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and dst.is_file() and not os.path.samefile(src, dst):
        dst.unlink()
    shutil.copy2(src, dst)


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
    except Exception:
        pass

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
    except Exception:
        pass

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
    arr_ref = ds_ref.pixel_array.astype(np.float32)
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
        arr = ds.pixel_array.astype(np.float32)
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
    accumulated_int = np.rint(accumulated * scaling_factor).astype(np.int32)

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

    outputs: List[CourseOutput] = []

    if not courses and structs:
        # Fallback: RS+CT only grouping by RS study UID (still a useful course container)
        rs_groups: Dict[tuple[str, str], list] = {}
        for s in structs:
            key = (s.patient_id, s.study_uid or f"FOR:{s.frame_of_reference_uid or 'unknown'}")
            rs_groups.setdefault(key, []).append(s)
        for (patient_id, course_key_raw), s_list in rs_groups.items():
            course_key = "".join(ch if ch.isalnum() else "_" for ch in str(course_key_raw))[:64]
            patient_dir = config.output_root / str(patient_id) / f"course_{course_key}"
            ensure_dir(patient_dir)
            rs_dst = patient_dir / "RS.dcm"
            # Copy the first RS as representative
            _safe_copy(s_list[0].path, rs_dst)
            # Copy CT series of this study if available
            ct_dir_out: Optional[Path] = None
            if patient_id in ct_index and (s_list[0].study_uid or "") in ct_index[patient_id]:
                series = pick_primary_series(ct_index[patient_id][s_list[0].study_uid])
                if series:
                    ct_dir_out = patient_dir / "CT_DICOM"
                    copy_ct_series(series, ct_dir_out)
            outputs.append(
                CourseOutput(
                    patient_id=patient_id,
                    course_key=course_key,
                    dir=patient_dir,
                    rp_path=patient_dir / "RP.dcm",
                    rd_path=patient_dir / "RD.dcm",
                    rs_path=rs_dst,
                    ct_dir=ct_dir_out,
                    total_prescription_gy=None,
                )
            )

    for (patient_id, course_key_raw), items in courses.items():
        # sanitize course key for filesystem
        course_key = "".join(ch if ch.isalnum() else "_" for ch in str(course_key_raw))[:64]
        # Sort plans by date for deterministic behavior
        items_sorted = sorted(items, key=lambda it: it.plan.plan_date or "")

        patient_dir = config.output_root / str(patient_id) / f"course_{course_key}"
        ensure_dir(patient_dir)
        rp_dst = patient_dir / "RP.dcm"
        rd_dst = patient_dir / "RD.dcm"
        rs_dst = patient_dir / "RS.dcm"

        plan_paths = [it.plan.path for it in items_sorted]
        dose_paths = [it.dose.path for it in items_sorted]
        struct_path = next((it.struct.path for it in items_sorted if it.struct), None)

        # Sum or copy as needed
        if len(plan_paths) == 1 and len(dose_paths) == 1:
            _safe_copy(plan_paths[0], rp_dst)
            _safe_copy(dose_paths[0], rd_dst)
            if struct_path is None:
                # fallback: find RS by study UID
                course_study = items_sorted[0].ct_study_uid
                for s in structs:
                    if s.patient_id == patient_id and (s.study_uid == course_study or not course_study):
                        struct_path = s.path
                        break
            if struct_path:
                _safe_copy(struct_path, rs_dst)
        else:
            # Compute total rx from plan dose references where available
            total_rx = 0.0
            for p in plan_paths:
                try:
                    ds = pydicom.dcmread(str(p))
                    if hasattr(ds, "DoseReferenceSequence") and ds.DoseReferenceSequence:
                        for dr in ds.DoseReferenceSequence:
                            if hasattr(dr, "TargetPrescriptionDose") and dr.TargetPrescriptionDose is not None:
                                total_rx += float(dr.TargetPrescriptionDose)
                                break
                except Exception:
                    pass
            _sum_doses_with_resample(dose_paths, rd_dst)
            _create_summed_plan(plan_paths, total_rx, rp_dst)
            if struct_path is None:
                course_study = items_sorted[0].ct_study_uid
                for s in structs:
                    if s.patient_id == patient_id and (s.study_uid == course_study or not course_study):
                        struct_path = s.path
                        break
            if struct_path:
                _safe_copy(struct_path, rs_dst)

        # Copy CT series that matches the course (same study UID) if available
        ct_dir_out: Optional[Path] = None
        course_study = items_sorted[0].ct_study_uid
        if course_study and patient_id in ct_index and course_study in ct_index[patient_id]:
            # Pick the largest series in this study
            series = pick_primary_series(ct_index[patient_id][course_study])
            if series:
                ct_dir_out = patient_dir / "CT_DICOM"
                copy_ct_series(series, ct_dir_out)

        outputs.append(
            CourseOutput(
                patient_id=patient_id,
                course_key=course_key,
                dir=patient_dir,
                rp_path=rp_dst,
                rd_path=rd_dst,
                rs_path=rs_dst if rs_dst.exists() else None,
                ct_dir=ct_dir_out,
                total_prescription_gy=None,  # optional to compute separately
            )
        )

        logger.info("Organized patient %s course %s at %s", patient_id, course_key, patient_dir)

        # ------------------------------------------------------------------
        # Save per-case metadata (Excel + JSON) in the course directory
        # ------------------------------------------------------------------
        try:
            # Aggregate course-level details for research/clinic
            plan_uids = {it.plan.sop_instance_uid for it in items_sorted if it.plan and it.plan.sop_instance_uid}
            # Total prescription dose across plans
            plan_total_rx = 0.0
            for pth in plan_paths:
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
                if plan_paths:
                    ds0 = pydicom.dcmread(str(plan_paths[0]), stop_before_pixels=True)
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
                        if getattr(ds_rt, 'PatientID', None) and str(ds_rt.PatientID) != str(patient_id):
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
            except Exception:
                pass

            # Dose grid info from organized RD
            dose_grid = {}
            try:
                if rd_dst.exists():
                    ds_rd = pydicom.dcmread(str(rd_dst), stop_before_pixels=False)
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
                if ct_dir_out and Path(ct_dir_out).exists():
                    ct_files = sorted([p for p in Path(ct_dir_out).iterdir() if p.suffix.lower() == '.dcm'])
                    if ct_files:
                        ds_ct = pydicom.dcmread(str(ct_files[0]), stop_before_pixels=True)
                        ct_summary = {
                            'ct_manufacturer': str(getattr(ds_ct, 'Manufacturer', '')),
                            'ct_model': str(getattr(ds_ct, 'ManufacturerModelName', '')),
                            'ct_institution': str(getattr(ds_ct, 'InstitutionName', '')),
                            'ct_kvp': float(getattr(ds_ct, 'KVP', 0.0) or 0.0) if hasattr(ds_ct, 'KVP') else None,
                            'ct_convolution_kernel': str(getattr(ds_ct, 'ConvolutionKernel', '')),
                            'ct_slice_thickness': float(getattr(ds_ct, 'SliceThickness', 0.0) or 0.0) if hasattr(ds_ct, 'SliceThickness') else None,
                        }
                        try:
                            ps = getattr(ds_ct, 'PixelSpacing', [None, None])
                            ct_summary['ct_pixel_spacing'] = [float(ps[0]), float(ps[1])] if ps and len(ps) >= 2 else []
                        except Exception:
                            pass
            except Exception:
                pass

            case_meta = {
                "patient_id": str(patient_id),
                "course_key": course_key,
                "rp_path": str(rp_dst) if rp_dst.exists() else "",
                "rd_path": str(rd_dst) if rd_dst.exists() else "",
                "rs_path": str(rs_dst) if rs_dst.exists() else "",
                "rs_auto_path": str((patient_dir / "RS_auto.dcm")) if (patient_dir / "RS_auto.dcm").exists() else "",
                "seg_dicom_path": str((patient_dir / "TotalSegmentator_DICOM" / "segmentations.dcm")) if (patient_dir / "TotalSegmentator_DICOM" / "segmentations.dcm").exists() else "",
                "seg_nifti_dir": str((patient_dir / "TotalSegmentator_NIFTI")) if (patient_dir / "TotalSegmentator_NIFTI").exists() else "",
                "ct_dir": str(ct_dir_out) if ct_dir_out and Path(ct_dir_out).exists() else "",
                "ct_study_uid": items_sorted[0].ct_study_uid if items_sorted else "",
                "planned_fractions": planned_fractions,
                "beams_count": beams_count,
                "beam_energies": beam_energies,
                "treatment_machine": machine_name,
                "total_prescription_gy": plan_total_rx,
                "course_start_date": start_date.isoformat() if start_date else "",
                "course_end_date": end_date.isoformat() if end_date else "",
                "fractions_count": fractions_count,
                "dose_grid": dose_grid,
            }
            case_meta.update(ct_summary)
            # Enrich with RP/RS tags if available
            try:
                if rp_dst.exists():
                    ds_rp = pydicom.dcmread(str(rp_dst), stop_before_pixels=True)
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
                    ds_rd2 = pydicom.dcmread(str(rd_dst), stop_before_pixels=True)
                    case_meta['dose_units'] = str(getattr(ds_rd2, 'DoseUnits', ''))
                    case_meta['dose_type'] = str(getattr(ds_rd2, 'DoseType', ''))
                except Exception:
                    pass

            # Write JSON + XLSX (one-row sheet)
            with open(patient_dir / "case_metadata.json", "w", encoding="utf-8") as f:
                json.dump(case_meta, f, ensure_ascii=False, indent=2)
            try:
                import pandas as _pd
                _pd.DataFrame([case_meta]).to_excel(patient_dir / "case_metadata.xlsx", index=False)
            except Exception:
                pass
        except Exception as e:
            logger.debug("Failed to write per-case metadata for %s: %s", patient_dir, e)

    return outputs
