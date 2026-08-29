from __future__ import annotations

"""Production-shaped organizer-contract fixtures for downstream stage tests."""

import json
import hashlib
from pathlib import Path
from typing import Iterable

import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTDoseStorage,
    RTPlanStorage,
    RTStructureSetStorage,
    UID,
    generate_uid,
)

from rtpipeline.course_contract import (
    build_dvh_decision,
    DOSE_GRID_SEMANTICS,
    DOSE_RESPONSE_FIELD,
    UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS,
)


def _header(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def _relative(course_dir: Path, path: Path) -> str:
    return str(path.resolve(strict=False).relative_to(course_dir.resolve(strict=False)))


def _first_dicom(paths: Iterable[Path], modality: str) -> Path | None:
    for path in sorted(set(Path(item) for item in paths)):
        if not path.is_file():
            continue
        try:
            if str(getattr(_header(path), "Modality", "") or "").upper() == modality:
                return path
        except Exception:
            continue
    return None


def write_synthetic_planning_ct(course_dir: Path) -> Path:
    """Create one readable planning-CT header and return its series directory."""
    course_dir = Path(course_dir)
    ct_dir = course_dir / "DICOM" / "CT"
    ct_dir.mkdir(parents=True, exist_ok=True)
    path = ct_dir / "ct_000.dcm"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.PatientID = course_dir.parent.name
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.save_as(path, enforce_file_format=True)
    return ct_dir


def write_synthetic_rtstruct(path: Path) -> Path:
    """Create a readable RTSTRUCT header at a test-owned path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "RTSTRUCT"
    dataset.StructureSetLabel = "TEST"
    dataset.save_as(path, enforce_file_format=True)
    return path


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(sop_class_uid)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    return dataset


def write_synthetic_plan_and_dose(
    course_dir: Path,
    *,
    prescribed_dose_gy: float = 50.0,
    planned_fraction_count: int = 20,
) -> tuple[Path, Path]:
    """Write a referenced plan-level RTPLAN and RTDOSE pair."""
    plan_path = course_dir / "DICOM" / "RTPLAN" / "selected_plan.dcm"
    dose_path = course_dir / "DICOM" / "RTDOSE" / "selected_dose.dcm"
    plan_uid = generate_uid()
    plan = _file_dataset(plan_path, RTPlanStorage, plan_uid)
    plan.Modality = "RTPLAN"
    plan.PatientID = course_dir.parent.name
    dose_reference = Dataset()
    dose_reference.DoseReferenceType = "TARGET"
    dose_reference.TargetPrescriptionDose = float(prescribed_dose_gy)
    plan.DoseReferenceSequence = Sequence([dose_reference])
    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = int(planned_fraction_count)
    plan.FractionGroupSequence = Sequence([fraction_group])
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.save_as(str(plan_path), enforce_file_format=True)

    dose = _file_dataset(dose_path, RTDoseStorage, generate_uid())
    dose.Modality = "RTDOSE"
    dose.PatientID = course_dir.parent.name
    dose.DoseSummationType = "PLAN"
    plan_reference = Dataset()
    plan_reference.ReferencedSOPClassUID = RTPlanStorage
    plan_reference.ReferencedSOPInstanceUID = plan_uid
    dose.ReferencedRTPlanSequence = Sequence([plan_reference])
    dose_path.parent.mkdir(parents=True, exist_ok=True)
    dose.save_as(str(dose_path), enforce_file_format=True)
    return plan_path, dose_path


def write_minimal_course_contract(
    course_dir: Path,
    *,
    selected_plans: Iterable[Path] | None = None,
    selected_doses: Iterable[Path] | None = None,
    authoritative_rtstruct: Path | None = None,
    planning_ct_dir: Path | None = None,
    planning_ct_nifti: Path | None = None,
    delivery_status: str = "no_records_at_all",
    dose_qc_pass: bool = True,
    dose_qc_threshold_gy: float = 100.0,
) -> Path:
    """Write a valid version-1 contract around artifacts created by a test."""
    course_dir = Path(course_dir)
    dicom_dir = course_dir / "DICOM"
    all_dicom = list(dicom_dir.rglob("*.dcm")) if dicom_dir.exists() else []
    all_dicom += [path for path in course_dir.glob("*.dcm") if path.is_file()]

    plans = list(selected_plans or [])
    doses = list(selected_doses or [])
    if selected_plans is None:
        first = _first_dicom(all_dicom, "RTPLAN")
        plans = [first] if first is not None else []
    if selected_doses is None:
        first = _first_dicom(all_dicom, "RTDOSE")
        doses = [first] if first is not None else []
    if planning_ct_dir is None:
        candidate = dicom_dir / "CT"
        if candidate.is_dir() and any(candidate.iterdir()):
            planning_ct_dir = candidate
    planning_series_uid = ""
    nifti_sidecar_path: Path | None = None
    if planning_ct_dir is not None:
        readable = []
        for path in sorted(planning_ct_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                uid = str(getattr(_header(path), "SeriesInstanceUID", "") or "")
            except Exception:
                continue
            if uid:
                readable.append(uid)
        unique = sorted(set(readable))
        if len(unique) != 1:
            raise AssertionError(f"test planning CT directory must contain one series, found {unique}")
        planning_series_uid = unique[0]
        if planning_ct_nifti is None:
            planning_ct_nifti = course_dir / "NIFTI" / "ct.nii.gz"
            planning_ct_nifti.parent.mkdir(parents=True, exist_ok=True)
            if not planning_ct_nifti.is_file():
                sitk.WriteImage(sitk.Image([2, 2, 2], sitk.sitkInt16), str(planning_ct_nifti))

        first_ct = _header(sorted(planning_ct_dir.iterdir())[0])
        ct_geometry = {
            "rows": int(getattr(first_ct, "Rows", 0) or 0),
            "columns": int(getattr(first_ct, "Columns", 0) or 0),
            "pixel_spacing": (
                [float(value) for value in first_ct.PixelSpacing]
                if hasattr(first_ct, "PixelSpacing")
                else None
            ),
            "image_orientation_patient": (
                [float(value) for value in first_ct.ImageOrientationPatient]
                if hasattr(first_ct, "ImageOrientationPatient")
                else None
            ),
            "slice_thickness": (
                float(first_ct.SliceThickness)
                if hasattr(first_ct, "SliceThickness")
                else None
            ),
        }
        try:
            nifti_image = sitk.ReadImage(str(planning_ct_nifti))
            nifti_geometry = {
                "size": [int(value) for value in nifti_image.GetSize()],
                "spacing": [float(value) for value in nifti_image.GetSpacing()],
                "origin": [float(value) for value in nifti_image.GetOrigin()],
                "direction": [float(value) for value in nifti_image.GetDirection()],
            }
        except Exception:
            nifti_geometry = {}
        nifti_base = (
            planning_ct_nifti.name[:-7]
            if planning_ct_nifti.name.endswith(".nii.gz")
            else planning_ct_nifti.stem
        )
        nifti_sidecar_path = planning_ct_nifti.parent / f"{nifti_base}.metadata.json"
        nifti_sidecar_path.write_text(
            json.dumps(
                {
                    "series_instance_uid": planning_series_uid,
                    "sop_hash": hashlib.sha256(
                        "".join(
                            str(getattr(_header(path), "SOPInstanceUID", ""))
                            for path in sorted(planning_ct_dir.iterdir())
                            if path.is_file()
                        ).encode("utf-8")
                    ).hexdigest(),
                    "geometry": ct_geometry,
                    "nifti_geometry": nifti_geometry,
                    "nifti_sha256": hashlib.sha256(planning_ct_nifti.read_bytes()).hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    plan_entries: list[dict[str, object]] = []
    plan_uids: list[str] = []
    for path in plans:
        dataset = _header(path)
        uid = str(dataset.SOPInstanceUID)
        plan_uids.append(uid)
        plan_entries.append(
            {
                "sop_instance_uid": uid,
                "path": _relative(course_dir, path),
                "delivered_record_count": 0,
                "delivered_fraction_count": 0,
                "treatment_dates": [],
            }
        )

    dose_entries: list[dict[str, object]] = []
    dose_uids: list[str] = []
    dose_types: list[str] = []
    for path in doses:
        dataset = _header(path)
        uid = str(dataset.SOPInstanceUID)
        summation_type = str(getattr(dataset, "DoseSummationType", "PLAN") or "PLAN").upper()
        refs = [
            str(item.ReferencedSOPInstanceUID)
            for item in getattr(dataset, "ReferencedRTPlanSequence", []) or []
        ]
        dose_uids.append(uid)
        dose_types.append(summation_type)
        dose_entries.append(
            {
                "sop_instance_uid": uid,
                "path": _relative(course_dir, path),
                "dose_summation_type": summation_type,
                "referenced_plan_uids": refs,
            }
        )

    per_plan = [
        {
            "plan_path": item["path"],
            "plan_sop_uid": item["sop_instance_uid"],
            "prescribed_dose_gy": None,
            "planned_fraction_count": None,
            "delivered_record_count": 0,
            "delivered_fraction_count": 0,
            "treatment_dates": [],
            "record_paths": [],
            "zero_delivery_records": True,
            "selected_for_dose_grid": True,
            "status": "no_records",
        }
        for item in plan_entries
    ]
    semantics = (
        UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS
        if delivery_status in {"no_records_at_all", "delivered_but_records_absent"}
        else DOSE_GRID_SEMANTICS
    )
    rtstruct_entry = None
    if authoritative_rtstruct is not None:
        dataset = _header(authoritative_rtstruct)
        rtstruct_entry = {
            "sop_instance_uid": str(dataset.SOPInstanceUID),
            "path": _relative(course_dir, authoritative_rtstruct),
        }
    plan_artifact = (
        {
            **dict(plan_entries[0]),
            "source_plan_uids": plan_uids,
        }
        if plan_entries
        else None
    )
    dose_grid = None
    if plan_entries and dose_entries:
        dose_grid = {
            **dose_entries[0],
            "semantics": semantics,
            "source_plan_uids": plan_uids,
            "source_dose_uids": dose_uids,
            "source_dose_summation_types": dose_types,
        }
    payload = {
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "course_contract": {
            "version": 1,
            "authority": "organize",
            "patient_id": course_dir.parent.name,
            "course_id": course_dir.name,
            "course_key": course_dir.name,
            "selected_plans": plan_entries,
            "selected_doses": dose_entries,
            "dose_classification": {
                "classification": "single_dose" if dose_entries else "no_doses"
            },
            "dvh": build_dvh_decision(
                len(plan_entries),
                len(dose_entries),
                delivery_status,
            ),
            "authoritative_rtstruct": rtstruct_entry,
            "planning_ct": {
                "status": "referenced" if planning_ct_dir is not None else "missing_reference",
                "series_instance_uid": planning_series_uid,
                "referenced_series_uids": [planning_series_uid] if planning_series_uid else [],
                "dicom_dir": _relative(course_dir, planning_ct_dir) if planning_ct_dir is not None else "",
                "nifti_path": _relative(course_dir, planning_ct_nifti) if planning_ct_nifti is not None else "",
                "nifti_provenance": (
                    {
                        "sidecar_path": _relative(
                            course_dir,
                            nifti_sidecar_path,
                        ),
                        **json.loads(nifti_sidecar_path.read_text(encoding="utf-8")),
                    }
                    if planning_ct_nifti is not None and nifti_sidecar_path is not None
                    else None
                ),
            },
            "plan_artifact": plan_artifact,
            "dose_grid": dose_grid,
            "delivery": {
                "prescribed_dose_gy": None,
                "delivered_dose_gy": None,
                "status": delivery_status,
                "method": None,
                "dose_response_field": DOSE_RESPONSE_FIELD,
                "per_plan": per_plan,
                "warnings": [],
                "unresolved_record_plan_uids": [],
            },
            "dose_qc": {
                "status": "pass" if dose_qc_pass else "fail",
                "pass": dose_qc_pass,
                "threshold_gy": float(dose_qc_threshold_gy),
                "reasons": [] if dose_qc_pass else ["test dose-QC failure"],
            },
        },
    }
    metadata_path = course_dir / "metadata" / "case_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path
