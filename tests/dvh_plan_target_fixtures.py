"""Synthetic single-course DVH fixture for plan-target binding tests.

Every UID, name and dose value is invented. The course has one RTPLAN, one
PLAN-summed RTDOSE and one contracted RTSTRUCT on a shared frame of reference.
The left half of the dose grid receives a uniform dose and the right half a
near-zero dose, so a box ROI on either side has a predictable D95.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    RTBeamsTreatmentRecordStorage,
    RTDoseStorage,
    RTPlanStorage,
    RTStructureSetStorage,
    UID,
)

from rtpipeline.course_contract import (
    DOSE_GRID_SEMANTICS,
    DOSE_RESPONSE_ELIGIBILITY_BASIS,
    build_dvh_decision,
    classify_course_dose_completeness,
)
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_planning_ct,
)

UID_ROOT = "2.25.4242"
FRAME_OF_REFERENCE_UID = f"{UID_ROOT}.1"
PLAN_UID = f"{UID_ROOT}.2"
DOSE_UID = f"{UID_ROOT}.3"
RTSTRUCT_UID = f"{UID_ROOT}.4"
OTHER_RTSTRUCT_UID = f"{UID_ROOT}.5"
STUDY_UID = f"{UID_ROOT}.6"

# Dose grid: 20 x 20 voxels in-plane at 2 mm, five planes at 2 mm.
GRID_SPACING_MM = 2.0
GRID_SIZE = 20
GRID_PLANES = 5
HIGH_DOSE_GY = 20.0
LOW_DOSE_GY = 0.005

# Box ROIs: (name, interpreted type, x range mm). y spans 6-30 mm, z spans 2-6 mm.
DOSED_LEFT_X = (4.0, 16.0)
UNDOSED_RIGHT_X = (24.0, 36.0)
WHOLE_X = (4.0, 36.0)


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(sop_class_uid)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = STUDY_UID
    dataset.SeriesInstanceUID = f"{sop_instance_uid}.1"
    dataset.PatientID = "SYNTH_P"
    return dataset


def dose_reference(
    *,
    number: int = 1,
    structure_type: str = "SITE",
    description: str | None = None,
    roi_number: int | None = None,
    reference_type: str = "TARGET",
    prescription_gy: float | None = HIGH_DOSE_GY,
) -> dict[str, object]:
    return {
        "number": number,
        "structure_type": structure_type,
        "description": description,
        "roi_number": roi_number,
        "reference_type": reference_type,
        "prescription_gy": prescription_gy,
    }


def write_plan(
    path: Path,
    references: Iterable[Mapping[str, object]],
    *,
    referenced_rtstruct_uid: str | None = RTSTRUCT_UID,
    fractions: int = 5,
) -> Path:
    plan = _file_dataset(path, RTPlanStorage, PLAN_UID)
    plan.Modality = "RTPLAN"
    plan.ApprovalStatus = "APPROVED"
    plan.FrameOfReferenceUID = FRAME_OF_REFERENCE_UID
    items = []
    for spec in references:
        item = Dataset()
        item.DoseReferenceNumber = int(spec["number"])
        item.DoseReferenceUID = f"{UID_ROOT}.10.{int(spec['number'])}"
        item.DoseReferenceStructureType = str(spec["structure_type"])
        item.DoseReferenceType = str(spec["reference_type"])
        if spec.get("description") is not None:
            item.DoseReferenceDescription = str(spec["description"])
        if spec.get("roi_number") is not None:
            item.ReferencedROINumber = int(spec["roi_number"])
        if spec.get("prescription_gy") is not None:
            item.TargetPrescriptionDose = float(spec["prescription_gy"])
        items.append(item)
    plan.DoseReferenceSequence = Sequence(items)
    if referenced_rtstruct_uid is not None:
        reference = Dataset()
        reference.ReferencedSOPClassUID = RTStructureSetStorage
        reference.ReferencedSOPInstanceUID = referenced_rtstruct_uid
        plan.ReferencedStructureSetSequence = Sequence([reference])
    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = int(fractions)
    fraction_group.NumberOfBeams = 1
    beam_reference = Dataset()
    beam_reference.ReferencedBeamNumber = 1
    beam_reference.BeamDose = HIGH_DOSE_GY / int(fractions)
    beam_reference.BeamDoseType = "PHYSICAL"
    if items:
        binding = Dataset()
        binding.ReferencedDoseReferenceUID = items[0].DoseReferenceUID
        beam_reference.ReferencedDoseReferenceSequence = Sequence([binding])
    fraction_group.ReferencedBeamSequence = Sequence([beam_reference])
    plan.FractionGroupSequence = Sequence([fraction_group])
    beam = Dataset()
    beam.BeamNumber = 1
    beam.TreatmentDeliveryType = "TREATMENT"
    beam.RadiationType = "PHOTON"
    plan.BeamSequence = Sequence([beam])
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.save_as(str(path), enforce_file_format=True)
    return path


def write_dose(path: Path) -> Path:
    dose = _file_dataset(path, RTDoseStorage, DOSE_UID)
    dose.Modality = "RTDOSE"
    dose.DoseSummationType = "PLAN"
    dose.DoseUnits = "GY"
    dose.DoseType = "PHYSICAL"
    dose.FrameOfReferenceUID = FRAME_OF_REFERENCE_UID
    reference = Dataset()
    reference.ReferencedSOPClassUID = RTPlanStorage
    reference.ReferencedSOPInstanceUID = PLAN_UID
    dose.ReferencedRTPlanSequence = Sequence([reference])
    dose.Rows = GRID_SIZE
    dose.Columns = GRID_SIZE
    dose.NumberOfFrames = GRID_PLANES
    dose.PixelSpacing = [GRID_SPACING_MM, GRID_SPACING_MM]
    dose.ImagePositionPatient = [0.0, 0.0, 0.0]
    dose.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dose.GridFrameOffsetVector = [GRID_SPACING_MM * k for k in range(GRID_PLANES)]
    dose.SamplesPerPixel = 1
    dose.PhotometricInterpretation = "MONOCHROME2"
    dose.BitsAllocated = 32
    dose.BitsStored = 32
    dose.HighBit = 31
    dose.PixelRepresentation = 0
    scaling = 1e-4
    dose.DoseGridScaling = scaling
    values = np.full((GRID_PLANES, GRID_SIZE, GRID_SIZE), LOW_DOSE_GY, dtype=float)
    values[:, :, : GRID_SIZE // 2] = HIGH_DOSE_GY
    dose.PixelData = np.round(values / scaling).astype("<u4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    dose.save_as(str(path), enforce_file_format=True)
    return path


def write_rtstruct(
    path: Path,
    rois: Iterable[tuple[str, str, tuple[float, float]]],
    *,
    sop_instance_uid: str = RTSTRUCT_UID,
) -> Path:
    dataset = _file_dataset(path, RTStructureSetStorage, sop_instance_uid)
    dataset.Modality = "RTSTRUCT"
    dataset.StructureSetLabel = "SYNTH"
    dataset.FrameOfReferenceUID = FRAME_OF_REFERENCE_UID
    referenced_frame = Dataset()
    referenced_frame.FrameOfReferenceUID = FRAME_OF_REFERENCE_UID
    dataset.ReferencedFrameOfReferenceSequence = Sequence([referenced_frame])
    structures, contours, observations = [], [], []
    for number, (name, interpreted_type, (x0, x1)) in enumerate(rois, start=1):
        structure = Dataset()
        structure.ROINumber = number
        structure.ROIName = name
        structure.ReferencedFrameOfReferenceUID = FRAME_OF_REFERENCE_UID
        structures.append(structure)
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        items = []
        for z_value in (2.0, 4.0, 6.0):
            contour = Dataset()
            contour.ContourGeometricType = "CLOSED_PLANAR"
            contour.NumberOfContourPoints = 4
            contour.ContourData = [
                x0, 6.0, z_value, x1, 6.0, z_value,
                x1, 30.0, z_value, x0, 30.0, z_value,
            ]
            items.append(contour)
        roi_contour.ContourSequence = Sequence(items)
        contours.append(roi_contour)
        observation = Dataset()
        observation.ObservationNumber = number
        observation.ReferencedROINumber = number
        observation.RTROIInterpretedType = interpreted_type
        observations.append(observation)
    dataset.StructureSetROISequence = Sequence(structures)
    dataset.ROIContourSequence = Sequence(contours)
    dataset.RTROIObservationsSequence = Sequence(observations)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), enforce_file_format=True)
    return path


def write_treatment_records(course: Path, fractions: int = 5) -> list[Path]:
    """One synthetic RTRECORD session per planned fraction."""
    paths = []
    for fraction in range(1, fractions + 1):
        uid = f"{UID_ROOT}.20.{fraction}"
        path = course / "DICOM_related" / "RTRECORD" / f"record_{fraction}.dcm"
        record = _file_dataset(path, RTBeamsTreatmentRecordStorage, uid)
        record.Modality = "RTRECORD"
        record.TreatmentDate = f"202001{fraction:02d}"
        reference = Dataset()
        reference.ReferencedSOPInstanceUID = PLAN_UID
        record.ReferencedRTPlanSequence = Sequence([reference])
        record.CurrentFractionNumber = fraction
        session = Dataset()
        session.TreatmentDeliveryType = "TREATMENT"
        session.TreatmentTerminationStatus = "NORMAL"
        record.TreatmentSessionBeamSequence = Sequence([session])
        path.parent.mkdir(parents=True, exist_ok=True)
        record.save_as(str(path), enforce_file_format=True)
        paths.append(path)
    return paths


def mark_fully_delivered(course: Path, fractions: int = 5) -> None:
    """Rewrite the synthetic contract as a fully delivered, dose-response-eligible course."""
    records = write_treatment_records(course, fractions)
    metadata_path = course / "metadata" / "case_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = payload["course_contract"]
    dates = sorted({f"202001{k:02d}" for k in range(1, fractions + 1)})
    evidence = {
        "delivered_record_count": len(records),
        "delivered_fraction_count": fractions,
        "treatment_dates": dates,
    }
    contract["selected_plans"][0].update({**evidence, "planned_fraction_count": fractions})
    contract["delivery"]["per_plan"][0].update(
        {
            **evidence,
            "planned_fraction_count": fractions,
            "record_paths": [path.relative_to(course).as_posix() for path in records],
            "zero_delivery_records": False,
            "status": "fully_delivered",
        }
    )
    contract["delivery"].update(
        {
            "status": "fully_delivered",
            "delivered_dose_gy": HIGH_DOSE_GY,
            "dose_response_eligible": True,
            "dose_response_eligibility_basis": DOSE_RESPONSE_ELIGIBILITY_BASIS,
        }
    )
    contract["dose_grid"]["semantics"] = DOSE_GRID_SEMANTICS
    contract["dose_completeness"] = classify_course_dose_completeness(
        selected_plans=contract["selected_plans"],
        selected_doses=contract["selected_doses"],
        dose_classification=contract["dose_classification"],
        dose_grid=contract["dose_grid"],
        per_plan_delivery=contract["delivery"]["per_plan"],
        delivery_status="fully_delivered",
        spatial_mapping_validated=True,
    )
    contract["dvh"] = build_dvh_decision(
        1,
        1,
        "fully_delivered",
        dose_response_eligible=True,
        dose_completeness=contract["dose_completeness"],
    )
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


DEFAULT_ROIS = (
    ("PTV1", "PTV", DOSED_LEFT_X),
    ("CTV 1", "CTV", (6.0, 14.0)),
    ("PTV2", "PTV", UNDOSED_RIGHT_X),
    ("CTV2", "CTV", (26.0, 34.0)),
    ("Bladder", "ORGAN", WHOLE_X),
)
NO_NEAR_ZERO_ROIS = (
    ("PTV1", "PTV", DOSED_LEFT_X),
    ("CTV 1", "CTV", (6.0, 14.0)),
    ("Bladder", "ORGAN", WHOLE_X),
    ("Rectum", "ORGAN", UNDOSED_RIGHT_X),
)


def build_course(
    root: Path,
    *,
    references: Iterable[Mapping[str, object]] = (
        dose_reference(description="PTV1"),
    ),
    rois=DEFAULT_ROIS,
    plan_referenced_rtstruct_uid: str | None = RTSTRUCT_UID,
    delivered: bool = True,
) -> Path:
    course = Path(root) / "SYNTH_P" / "SYNTH_C"
    write_synthetic_planning_ct(course)
    plan = write_plan(
        course / "DICOM" / "RTPLAN" / "plan.dcm",
        references,
        referenced_rtstruct_uid=plan_referenced_rtstruct_uid,
    )
    dose = write_dose(course / "DICOM" / "RTDOSE" / "dose.dcm")
    rtstruct = write_rtstruct(course / "DICOM" / "RTSTRUCT" / "rs.dcm", rois)
    write_minimal_course_contract(
        course,
        selected_plans=[plan],
        selected_doses=[dose],
        authoritative_rtstruct=rtstruct,
    )
    if delivered:
        mark_fully_delivered(course)
    return course


def canonical_dvh_table(course: Path) -> str:
    """Serialize the published DVH table with exact float reprs and no tmp paths."""
    frame = pd.read_parquet(course / "dvh_metrics.parquet")
    prefix = str(course)

    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (np.floating, float)):
            return repr(float(value))
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        text = str(value)
        return text.replace(prefix, "<COURSE>")

    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": [[cell(value) for value in row] for row in frame.itertuples(index=False)],
    }
    return json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=False)
