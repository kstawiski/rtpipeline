"""Synthetic RTPLAN and RTRECORD builders for the dose-semantics suite.

Every UID, date, dose and meterset below was invented for this suite. None is
derived from a patient, a clinical export or an earlier fixture. Dates sit in
2091 so they cannot be mistaken for a treatment calendar.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

RTPLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"
RTRECORD_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.4"
RTSUMMARY_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.7"

PLAN_A = "2.25.418006270091000000000000000000000011"
PLAN_B = "2.25.418006270091000000000000000000000012"
PLAN_C = "2.25.418006270091000000000000000000000013"
REF_A = "2.25.418006270091000000000000000000000021"
REF_B = "2.25.418006270091000000000000000000000022"
FRAME = "2.25.418006270091000000000000000000000031"

TARGET_NUMBER = "4"
BEAM_DOSES = (1.35, 0.85)
PER_FRACTION = 2.2
PLANNED_FRACTIONS = 7
RESOLVED_TOTAL = 15.4
UNMATCHED_TARGET_RX = 19.0

_record_counter = [0]


def _without_value_validation(builder):
    """Let a builder write deliberately invalid values such as a NaN dose."""

    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        with pydicom.config.disable_value_validation():
            return builder(*args, **kwargs)

    return wrapper


def date_for(day: int) -> str:
    return f"2091{3 + day // 28:02d}{1 + day % 28:02d}"


def next_record_uid() -> str:
    _record_counter[0] += 1
    return f"2.25.4180062700910000000000000009{_record_counter[0]:08d}"


def _save(ds: Dataset, path: Path, sop_class: str, sop_uid: str) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.PatientID = "SYNTHETIC"
    ds.save_as(str(path), enforce_file_format=True)
    return path


@_without_value_validation
def make_plan(
    path: Path,
    *,
    plan_uid: str = PLAN_A,
    reference_number: str = TARGET_NUMBER,
    reference_uid: str = REF_A,
    reference_type: str = "TARGET",
    target_rx: float | None = RESOLVED_TOTAL,
    fractions: int = PLANNED_FRACTIONS,
    beam_doses: tuple[float, ...] = BEAM_DOSES,
    beam_reference_uid: str | None = REF_A,
    extra_fraction_group: bool = False,
    approval: str = "APPROVED",
) -> Path:
    ds = Dataset()
    ds.Modality = "RTPLAN"
    ds.ApprovalStatus = approval
    ds.RTPlanLabel = "synthetic"
    ds.RTPlanDate = date_for(0)
    ds.RTPlanGeometry = "PATIENT"
    ds.FrameOfReferenceUID = FRAME
    reference = Dataset()
    reference.DoseReferenceNumber = reference_number
    reference.DoseReferenceUID = reference_uid
    reference.DoseReferenceStructureType = "COORDINATES"
    reference.DoseReferenceDescription = "synthetic point"
    reference.DoseReferenceType = reference_type
    if target_rx is not None:
        reference.TargetPrescriptionDose = target_rx
    ds.DoseReferenceSequence = Sequence([reference])
    beams = []
    references = []
    for index, dose in enumerate(beam_doses, start=1):
        beam = Dataset()
        beam.BeamNumber = index
        beam.BeamName = f"Arc {index}"
        beam.BeamType = "DYNAMIC"
        beam.TreatmentDeliveryType = "TREATMENT"
        beam.RadiationType = "PHOTON"
        beams.append(beam)
        item = Dataset()
        item.ReferencedBeamNumber = index
        item.BeamDose = dose
        item.BeamMeterset = round(100.0 * dose, 3)
        if beam_reference_uid:
            item.ReferencedDoseReferenceUID = beam_reference_uid
        references.append(item)
    setup = Dataset()
    setup.BeamNumber = 8
    setup.BeamName = "kV setup"
    setup.BeamType = "STATIC"
    setup.TreatmentDeliveryType = "SETUP"
    setup.RadiationType = "PHOTON"
    beams.append(setup)
    setup_reference = Dataset()
    setup_reference.ReferencedBeamNumber = 8
    references.append(setup_reference)
    ds.BeamSequence = Sequence(beams)
    group = Dataset()
    group.FractionGroupNumber = 1
    group.NumberOfFractionsPlanned = fractions
    group.NumberOfBeams = len(references)
    group.ReferencedBeamSequence = Sequence(references)
    groups = [group]
    if extra_fraction_group:
        second = Dataset()
        second.FractionGroupNumber = 2
        second.NumberOfFractionsPlanned = 1
        second.NumberOfBeams = 1
        only = Dataset()
        only.ReferencedBeamNumber = 1
        only.BeamDose = beam_doses[0]
        second.ReferencedBeamSequence = Sequence([only])
        groups.append(second)
    ds.FractionGroupSequence = Sequence(groups)
    return _save(ds, path, RTPLAN_SOP_CLASS, plan_uid)


@dataclass(frozen=True)
class Event:
    beam: int
    delivery_type: str = "TREATMENT"
    termination: str = "NORMAL"
    dose: float | str | None = None
    reference_number: str = TARGET_NUMBER
    start: str = "071500"
    meterset: float | None = None


@_without_value_validation
def make_record(
    path: Path,
    *,
    plan_uid: str = PLAN_A,
    day: int = 0,
    time: str = "071500",
    fraction: int | None = 1,
    events: list[Event],
    record_uid: str | None = None,
) -> Path:
    ds = Dataset()
    ds.Modality = "RTRECORD"
    ds.TreatmentDate = date_for(day)
    ds.TreatmentTime = time
    link = Dataset()
    link.ReferencedSOPClassUID = RTPLAN_SOP_CLASS
    link.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([link])
    items = []
    for event in events:
        item = Dataset()
        item.ReferencedBeamNumber = event.beam
        item.BeamName = f"Arc {event.beam}"
        item.BeamType = "DYNAMIC"
        item.RadiationType = "PHOTON"
        if fraction is not None:
            item.CurrentFractionNumber = fraction
        item.TreatmentDeliveryType = event.delivery_type
        item.TreatmentTerminationStatus = event.termination
        item.TreatmentVerificationStatus = "VERIFIED"
        if event.dose is not None:
            calculated = Dataset()
            calculated.CalculatedDoseReferenceDoseValue = event.dose
            calculated.ReferencedDoseReferenceNumber = event.reference_number
            item.ReferencedCalculatedDoseReferenceSequence = Sequence([calculated])
        meterset = event.meterset
        if meterset is None and isinstance(event.dose, float):
            meterset = round(100.0 * event.dose, 3)
        first = Dataset()
        first.ReferencedControlPointIndex = 0
        first.TreatmentControlPointDate = date_for(day)
        first.TreatmentControlPointTime = event.start
        first.DeliveredMeterset = 0.0
        last = Dataset()
        last.ReferencedControlPointIndex = 1
        last.TreatmentControlPointDate = date_for(day)
        last.TreatmentControlPointTime = event.start
        if meterset is not None:
            last.DeliveredMeterset = meterset
        item.ControlPointDeliverySequence = Sequence([first, last])
        items.append(item)
    ds.TreatmentSessionBeamSequence = Sequence(items)
    return _save(ds, path, RTRECORD_SOP_CLASS, record_uid or next_record_uid())


@_without_value_validation
def make_summary_record(path: Path, *, plan_uid: str = PLAN_A, day: int, cumulative: float) -> Path:
    ds = Dataset()
    ds.Modality = "RTRECORD"
    ds.TreatmentDate = date_for(day)
    ds.TreatmentTime = "180000"
    link = Dataset()
    link.ReferencedSOPClassUID = RTPLAN_SOP_CLASS
    link.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([link])
    item = Dataset()
    item.CumulativeDoseToDoseReference = cumulative
    item.ReferencedDoseReferenceNumber = TARGET_NUMBER
    ds.TreatmentSummaryCalculatedDoseReferenceSequence = Sequence([item])
    return _save(ds, path, RTSUMMARY_SOP_CLASS, next_record_uid())


def complete_session(
    directory: Path,
    fraction: int,
    *,
    plan_uid: str = PLAN_A,
    day: int | None = None,
    doses: tuple[float | str | None, ...] = BEAM_DOSES,
    reference_number: str = TARGET_NUMBER,
    one_record_per_beam: bool = True,
) -> list[Path]:
    """Both therapeutic arcs of one fraction, each ending NORMAL."""
    day = fraction if day is None else day
    events = [
        Event(beam=beam, dose=dose, reference_number=reference_number, start=f"07{15 + beam:02d}00",
              meterset=None if isinstance(dose, float) else 120.0)
        for beam, dose in zip((1, 2), doses)
    ]
    tag = plan_uid[-2:]
    if one_record_per_beam:
        return [
            make_record(directory / f"p{tag}_f{fraction}_b{event.beam}.dcm", plan_uid=plan_uid, day=day,
                        time=event.start, fraction=fraction, events=[event])
            for event in events
        ]
    return [make_record(directory / f"p{tag}_f{fraction}.dcm", plan_uid=plan_uid, day=day, fraction=fraction, events=events)]
