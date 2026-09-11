"""Synthetic RT object fixtures shared by the organizer regression tests.

Every dataset here is written from the production reading contracts, not from
any exported study: identifiers are generated, geometry is a small axis-aligned
lattice, and dose values are round numbers chosen to make one arithmetic
interpretation resolvable. Nothing in this module is derived from a patient.

The builders satisfy these readers:

* ``rtpipeline.rt_details.extract_rt_with_records`` classifies by ``Modality``
  and indexes ``PatientID``, ``SOPInstanceUID``, ``StudyInstanceUID``,
  ``FrameOfReferenceUID`` and the referenced-SOP sequences.
* ``rtpipeline.metadata.link_rt_sets`` resolves RTDOSE to RTPLAN to RTSTRUCT
  through those references alone.
* ``rtpipeline.prescription.resolve_plan_prescriptions`` needs a target dose
  reference plus complete therapeutic ``BeamDose`` membership for one fraction
  group, so ``rx_gy`` and ``fractions`` resolve as a confirmed plan total.
* ``rtpipeline.course_contract._record_delivery_session_evidence`` accepts a
  session item only when ``TreatmentDeliveryType`` is TREATMENT or CONTINUATION
  and ``TreatmentTerminationStatus`` is NORMAL.

Tests are free to reopen a written file and mutate tags to build the negative
case they need; the defaults are the resolvable case.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PATIENT_ID = "P1"

RTPLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"
RTDOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
RTRECORD_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.8"

__all__ = [
    "PATIENT_ID",
    "make_plan",
    "make_dose",
    "make_struct",
    "make_record",
    "extract_linked",
    "organize_workflow",
]


def _new_dataset(sop_class_uid: str, sop_instance_uid: str, *, patient_id: str) -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop_class_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.PatientID = patient_id
    ds.PatientName = "SYNTH^FIXTURE"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.SpecificCharacterSet = "ISO_IR 100"
    return ds


def _referenced_sop(sop_class_uid: str, sop_instance_uid: str) -> Dataset:
    item = Dataset()
    item.ReferencedSOPClassUID = sop_class_uid
    item.ReferencedSOPInstanceUID = sop_instance_uid
    return item


def _write(ds: Dataset, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return path


def make_struct(
    path,
    sop_uid: str,
    *,
    study_uid: str | None = None,
    frame_uid: str | None = None,
    roi_names=("BODY", "PTV1"),
    patient_id: str = PATIENT_ID,
) -> Path:
    """Write an RTSTRUCT whose ROI names drive organizer target screening."""
    ds = _new_dataset(RTSTRUCT_SOP_CLASS, sop_uid, patient_id=patient_id)
    ds.Modality = "RTSTRUCT"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = "20240101"
    ds.SeriesDescription = "synthetic structures"
    ds.StructureSetLabel = "synthetic"
    ds.StructureSetDate = "20240101"

    frame_of_reference_uid = frame_uid or generate_uid()
    ds.FrameOfReferenceUID = frame_of_reference_uid
    referenced_frame = Dataset()
    referenced_frame.FrameOfReferenceUID = frame_of_reference_uid
    referenced_frame.RTReferencedStudySequence = Sequence(
        [_referenced_sop("1.2.840.10008.3.1.2.3.1", ds.StudyInstanceUID)]
    )
    ds.ReferencedFrameOfReferenceSequence = Sequence([referenced_frame])

    rois, contours, observations = [], [], []
    for number, name in enumerate(roi_names, start=1):
        roi = Dataset()
        roi.ROINumber = number
        roi.ReferencedFrameOfReferenceUID = frame_of_reference_uid
        roi.ROIName = name
        roi.ROIGenerationAlgorithm = "MANUAL"
        rois.append(roi)

        contour = Dataset()
        contour.ReferencedROINumber = number
        contour.ROIDisplayColor = [255, 0, 0]
        slice_contour = Dataset()
        slice_contour.ContourGeometricType = "CLOSED_PLANAR"
        slice_contour.NumberOfContourPoints = 4
        slice_contour.ContourData = [
            -10.0, -10.0, 0.0,
            10.0, -10.0, 0.0,
            10.0, 10.0, 0.0,
            -10.0, 10.0, 0.0,
        ]
        contour.ContourSequence = Sequence([slice_contour])
        contours.append(contour)

        observation = Dataset()
        observation.ObservationNumber = number
        observation.ReferencedROINumber = number
        observation.RTROIInterpretedType = "PTV" if name.upper().startswith("PTV") else "ORGAN"
        observation.ROIInterpreter = ""
        observations.append(observation)

    ds.StructureSetROISequence = Sequence(rois)
    ds.ROIContourSequence = Sequence(contours)
    ds.RTROIObservationsSequence = Sequence(observations)
    return _write(ds, path)


def make_plan(
    path,
    sop_uid: str,
    *,
    struct_uid: str | None = None,
    study_uid: str | None = None,
    frame_uid: str | None = None,
    date: str = "20240101",
    rx_gy: float = 60.0,
    fractions: int = 30,
    label: str = "synthetic",
    approval_status: str = "APPROVED",
    beam_count: int = 1,
    patient_id: str = PATIENT_ID,
) -> Path:
    """Write an RTPLAN whose prescription resolves to ``rx_gy`` as a total.

    ``BeamDose`` is the per-fraction share of ``rx_gy``, so BeamDose sum times
    ``fractions`` matches the target prescription while the per-fraction
    interpretation does not. That is the one case the resolver accepts as
    TOTAL_CONFIRMED, which keeps the fixture out of the ambiguous branches.
    """
    ds = _new_dataset(RTPLAN_SOP_CLASS, sop_uid, patient_id=patient_id)
    ds.Modality = "RTPLAN"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = date
    ds.FrameOfReferenceUID = frame_uid or generate_uid()
    ds.RTPlanLabel = label
    ds.RTPlanName = label
    ds.RTPlanDate = date
    ds.RTPlanTime = "080000"
    ds.RTPlanGeometry = "PATIENT"
    ds.ApprovalStatus = approval_status
    ds.PlanIntent = "CURATIVE"
    if approval_status == "APPROVED":
        ds.ReviewDate = date
        ds.ReviewTime = "090000"
        ds.ReviewerName = "SYNTH^REVIEWER"

    if struct_uid:
        ds.ReferencedStructureSetSequence = Sequence(
            [_referenced_sop(RTSTRUCT_SOP_CLASS, struct_uid)]
        )

    dose_reference_uid = generate_uid()
    dose_reference = Dataset()
    dose_reference.DoseReferenceNumber = 1
    dose_reference.DoseReferenceUID = dose_reference_uid
    dose_reference.DoseReferenceStructureType = "SITE"
    dose_reference.DoseReferenceDescription = "synthetic target"
    dose_reference.DoseReferenceType = "TARGET"
    dose_reference.TargetPrescriptionDose = float(rx_gy)
    ds.DoseReferenceSequence = Sequence([dose_reference])

    beam_count = max(1, int(beam_count))
    per_beam_dose = float(rx_gy) / (int(fractions) * beam_count)
    beams, beam_references = [], []
    for number in range(1, beam_count + 1):
        beam = Dataset()
        beam.BeamNumber = number
        beam.BeamName = f"beam{number}"
        beam.BeamDescription = "synthetic beam"
        beam.BeamType = "STATIC"
        beam.RadiationType = "PHOTON"
        beam.TreatmentDeliveryType = "TREATMENT"
        beam.TreatmentMachineName = "SYNTH_LINAC"
        beam.SourceAxisDistance = 1000.0
        beam.NumberOfWedges = 0
        beam.NumberOfCompensators = 0
        beam.NumberOfBoli = 0
        beam.NumberOfBlocks = 0
        beam.FinalCumulativeMetersetWeight = 1.0
        beam.NumberOfControlPoints = 2
        control_points = []
        for index in range(2):
            control_point = Dataset()
            control_point.ControlPointIndex = index
            control_point.CumulativeMetersetWeight = float(index)
            control_point.GantryAngle = 0.0
            control_point.BeamLimitingDeviceAngle = 0.0
            control_point.PatientSupportAngle = 0.0
            control_point.NominalBeamEnergy = 6.0
            if index == 0:
                control_point.IsocenterPosition = [0.0, 0.0, 0.0]
            jaw = Dataset()
            jaw.RTBeamLimitingDeviceType = "X"
            jaw.LeafJawPositions = [-50.0, 50.0]
            control_point.BeamLimitingDevicePositionSequence = Sequence([jaw])
            control_points.append(control_point)
        beam.ControlPointSequence = Sequence(control_points)
        beams.append(beam)

        reference = Dataset()
        reference.ReferencedBeamNumber = number
        reference.BeamDose = per_beam_dose
        reference.BeamDoseType = "PHYSICAL"
        reference.BeamMeterset = 100.0
        reference.ReferencedDoseReferenceUID = dose_reference_uid
        beam_references.append(reference)

    ds.BeamSequence = Sequence(beams)

    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = int(fractions)
    fraction_group.NumberOfBeams = beam_count
    fraction_group.NumberOfBrachyApplicationSetups = 0
    fraction_group.ReferencedBeamSequence = Sequence(beam_references)
    group_reference = Dataset()
    group_reference.ReferencedDoseReferenceNumber = 1
    fraction_group.ReferencedDoseReferenceSequence = Sequence([group_reference])
    ds.FractionGroupSequence = Sequence([fraction_group])
    return _write(ds, path)


def make_dose(
    path,
    sop_uid: str,
    *,
    plan_uid: str | None = None,
    study_uid: str | None = None,
    frame_uid: str | None = None,
    with_pixels: bool = False,
    summation_type: str = "PLAN",
    dose_gy: float = 60.0,
    shape: tuple[int, int, int] = (3, 4, 4),
    patient_id: str = PATIENT_ID,
) -> Path:
    """Write an RTDOSE on a small uniform lattice referencing ``plan_uid``."""
    ds = _new_dataset(RTDOSE_SOP_CLASS, sop_uid, patient_id=patient_id)
    ds.Modality = "RTDOSE"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = "20240101"
    ds.FrameOfReferenceUID = frame_uid or generate_uid()
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = summation_type
    if plan_uid:
        ds.ReferencedRTPlanSequence = Sequence(
            [_referenced_sop(RTPLAN_SOP_CLASS, plan_uid)]
        )

    frames, rows, columns = shape
    ds.ImagePositionPatient = [-10.0, -10.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [5.0, 5.0]
    ds.SliceThickness = 5.0
    ds.GridFrameOffsetVector = [float(5 * index) for index in range(frames)]
    ds.NumberOfFrames = frames
    ds.Rows = rows
    ds.Columns = columns
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    if with_pixels:
        scaling = float(dose_gy) / 1000.0
        grid = np.full((frames, rows, columns), 1000, dtype=np.uint16)
        ds.DoseGridScaling = scaling
        ds.PixelData = grid.tobytes()
    else:
        ds.DoseGridScaling = float(dose_gy) / 1000.0
    return _write(ds, path)


def make_record(
    path,
    plan_uid: str,
    *,
    date: str = "20240102",
    fraction_number: int = 1,
    beam_number: int = 1,
    termination: str = "NORMAL",
    delivery_type: str = "TREATMENT",
    sop_uid: str | None = None,
    patient_id: str = PATIENT_ID,
) -> Path:
    """Write an RTRECORD that counts as one validated delivered session."""
    ds = _new_dataset(
        RTRECORD_SOP_CLASS, sop_uid or generate_uid(), patient_id=patient_id
    )
    ds.Modality = "RTRECORD"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = date
    ds.TreatmentDate = date
    ds.TreatmentTime = "101500"
    ds.ReferencedRTPlanSequence = Sequence(
        [_referenced_sop(RTPLAN_SOP_CLASS, plan_uid)]
    )

    machine = Dataset()
    machine.TreatmentMachineName = "SYNTH_LINAC"
    ds.TreatmentMachineSequence = Sequence([machine])

    session_beam = Dataset()
    session_beam.ReferencedBeamNumber = beam_number
    session_beam.BeamName = f"beam{beam_number}"
    session_beam.CurrentFractionNumber = int(fraction_number)
    session_beam.TreatmentDeliveryType = delivery_type
    session_beam.TreatmentTerminationStatus = termination
    session_beam.TreatmentVerificationStatus = "VERIFIED"
    session_beam.SpecifiedPrimaryMeterset = 100.0
    session_beam.DeliveredPrimaryMeterset = 100.0
    session_beam.TreatmentMachineName = "SYNTH_LINAC"
    ds.TreatmentSessionBeamSequence = Sequence([session_beam])
    return _write(ds, path)


def extract_linked(root):
    """Return ``LinkedSet`` items for a fixture tree, as the organizer sees it."""
    from rtpipeline.metadata import link_rt_sets
    from rtpipeline.rt_details import extract_rt_with_records

    plans, doses, structs, _records = extract_rt_with_records(Path(root))
    return link_rt_sets(plans, doses, structs)


def organize_workflow(tmp_path, output_dir, *, dicom_root=None, threads: int = 1):
    """Build the Snakemake namespace the organize checkpoint script receives.

    The attribute set mirrors ``checkpoint organize_courses`` in the Snakefile:
    the script reads ``output.manifest``, ``params.dicom_root``,
    ``params.output_dir`` and the interpreter/environment params.
    """
    import sys

    tmp_path = Path(tmp_path)
    output_dir = Path(output_dir)
    root_dir = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        input=SimpleNamespace(
            configuration=str(tmp_path / "config.yaml"), clinical_records=[]
        ),
        output=SimpleNamespace(manifest=str(output_dir / "manifests" / "courses.json")),
        log=[str(tmp_path / "logs" / "stage_organize.log")],
        threads=threads,
        params=SimpleNamespace(
            python=sys.executable,
            python_bin=str(Path(sys.executable).parent),
            root_dir=str(root_dir),
            configfile=str(tmp_path / "config.yaml"),
            radiomics_env="rtpipeline-radiomics",
            dicom_root=str(Path(dicom_root) if dicom_root else tmp_path / "input"),
            output_dir=str(output_dir),
            logs_dir=str(tmp_path / "logs"),
            custom_structures="",
            clinical_prescription_records="",
            prioritize_short_courses=False,
        ),
    )
