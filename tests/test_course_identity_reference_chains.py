"""Regression tests for reference-driven radiotherapy course construction.

Each test uses synthetic DICOM. No production patient data is embedded here.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTBeamsTreatmentRecordStorage,
    RTDoseStorage,
    RTPlanStorage,
    RTStructureSetStorage,
    UID,
    generate_uid,
)

from rtpipeline import cli as cli_module
from rtpipeline import organize as org
from rtpipeline.clinical_prescription import (
    confirm_two_phase_fractionation,
    parse_kopernik_treatment_description,
)
from rtpipeline.config import PipelineConfig
from rtpipeline.ct import CTInstance
from rtpipeline.course_contract import load_course_contract
from rtpipeline.organize_ledger import read_organize_ledger
from rtpipeline.metadata import LinkedSet, group_by_course, link_rt_sets
from rtpipeline.rt_details import extract_rt
from rtpipeline.dvh import dvh_for_course


def _write_placeholder_ct_nifti(_config, _ct_dir, nifti_dir, **_kwargs):
    nifti_dir = Path(nifti_dir)
    nifti_dir.mkdir(parents=True, exist_ok=True)
    path = nifti_dir / "ct.nii.gz"
    path.write_bytes(b"synthetic planning CT NIfTI placeholder")
    return path


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(sop_class_uid)
    meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.PatientID = "P1"
    ds.SeriesInstanceUID = generate_uid()
    return ds


def _write(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def _mk_struct(
    path: Path,
    sop_uid: str,
    *,
    study_uid: str,
    frame_uid: str,
    roi_names: list[str],
    ct_series_uid: str | None = None,
    with_contours: bool = False,
) -> Path:
    ds = _file_dataset(path, RTStructureSetStorage, sop_uid)
    ds.Modality = "RTSTRUCT"
    ds.StudyInstanceUID = study_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.add_new((0x3006, 0x0024), "UI", frame_uid)
    rois = []
    for number, name in enumerate(roi_names, start=1):
        roi = Dataset()
        roi.ROINumber = number
        roi.ReferencedFrameOfReferenceUID = frame_uid
        roi.ROIName = name
        roi.ROIGenerationAlgorithm = "MANUAL"
        rois.append(roi)
    ds.StructureSetROISequence = Sequence(rois)
    if with_contours:
        contour_rois = []
        for number in range(1, len(roi_names) + 1):
            roi_contour = Dataset()
            roi_contour.ReferencedROINumber = number
            roi_contour.ROIDisplayColor = [255, 0, 0]
            contours = []
            for z in (0.0, 1.0):
                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = 4
                contour.ContourData = [
                    0.1, 0.1, z,
                    1.9, 0.1, z,
                    1.9, 1.9, z,
                    0.1, 1.9, z,
                ]
                contours.append(contour)
            roi_contour.ContourSequence = Sequence(contours)
            contour_rois.append(roi_contour)
        ds.ROIContourSequence = Sequence(contour_rois)
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = frame_uid
    if ct_series_uid:
        ref_series = Dataset()
        ref_series.SeriesInstanceUID = ct_series_uid
        ref_study = Dataset()
        ref_study.RTReferencedSeriesSequence = Sequence([ref_series])
        ref_for.RTReferencedStudySequence = Sequence([ref_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])
    return _write(ds, path)


def _mk_plan(
    path: Path,
    sop_uid: str,
    *,
    struct_uid: str,
    study_uid: str,
    frame_uid: str,
    date: str,
    rx_gy: float,
    fractions: int,
    label: str,
) -> Path:
    ds = _file_dataset(path, RTPlanStorage, sop_uid)
    ds.Modality = "RTPLAN"
    ds.StudyInstanceUID = study_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.add_new((0x3006, 0x0024), "UI", frame_uid)
    ds.RTPlanDate = date
    ds.RTPlanTime = "120000"
    ds.RTPlanLabel = label
    ds.RTPlanName = label
    ref_struct = Dataset()
    ref_struct.ReferencedSOPClassUID = RTStructureSetStorage
    ref_struct.ReferencedSOPInstanceUID = struct_uid
    ds.ReferencedStructureSetSequence = Sequence([ref_struct])
    dose_ref = Dataset()
    dose_ref.DoseReferenceNumber = 1
    dose_ref.DoseReferenceUID = generate_uid()
    dose_ref.DoseReferenceType = "TARGET"
    dose_ref.TargetPrescriptionDose = float(rx_gy)
    ds.DoseReferenceSequence = Sequence([dose_ref])
    beam = Dataset()
    beam.BeamNumber = 1
    beam.TreatmentDeliveryType = "TREATMENT"
    ds.BeamSequence = Sequence([beam])
    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = fractions
    fraction_group.NumberOfBeams = 1
    referenced_beam = Dataset()
    referenced_beam.ReferencedBeamNumber = 1
    referenced_beam.BeamDose = float(rx_gy) / fractions
    referenced_beam.BeamDoseType = "PHYSICAL"
    referenced_beam.ReferencedDoseReferenceUID = dose_ref.DoseReferenceUID
    fraction_group.ReferencedBeamSequence = Sequence([referenced_beam])
    ds.FractionGroupSequence = Sequence([fraction_group])
    return _write(ds, path)


def _mk_dose(
    path: Path,
    sop_uid: str,
    *,
    plan_uid: str,
    study_uid: str,
    frame_uid: str,
    with_pixels: bool = False,
) -> Path:
    ds = _file_dataset(path, RTDoseStorage, sop_uid)
    ds.Modality = "RTDOSE"
    ds.StudyInstanceUID = study_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.add_new((0x3006, 0x0024), "UI", frame_uid)
    ds.DoseSummationType = "PLAN"
    ds.Rows = 2
    ds.Columns = 2
    ds.NumberOfFrames = 2
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.GridFrameOffsetVector = [0.0, 1.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if with_pixels:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.DoseUnits = "GY"
        ds.DoseGridScaling = 0.1
        ds.PixelData = np.full((2, 2, 2), 10, dtype=np.uint16).tobytes()
    ref_plan = Dataset()
    ref_plan.ReferencedSOPClassUID = RTPlanStorage
    ref_plan.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref_plan])
    return _write(ds, path)


def _mk_multi_plan_dose(
    path: Path,
    sop_uid: str,
    *,
    plan_uids: list[str],
    study_uid: str,
    frame_uid: str,
) -> Path:
    ds = _file_dataset(path, RTDoseStorage, sop_uid)
    ds.Modality = "RTDOSE"
    ds.StudyInstanceUID = study_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.add_new((0x3006, 0x0024), "UI", frame_uid)
    ds.DoseSummationType = "PLAN"
    refs = []
    for plan_uid in plan_uids:
        ref_plan = Dataset()
        ref_plan.ReferencedSOPClassUID = RTPlanStorage
        ref_plan.ReferencedSOPInstanceUID = plan_uid
        refs.append(ref_plan)
    ds.ReferencedRTPlanSequence = Sequence(refs)
    return _write(ds, path)


def _mk_record(path: Path, plan_uid: str, *, date: str = "20240102") -> Path:
    ds = _file_dataset(path, RTBeamsTreatmentRecordStorage, generate_uid())
    ds.Modality = "RTRECORD"
    ds.StudyInstanceUID = generate_uid()
    ds.TreatmentDate = date
    ref_plan = Dataset()
    ref_plan.ReferencedSOPClassUID = RTPlanStorage
    ref_plan.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref_plan])
    return _write(ds, path)


def _mk_ct(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    frame_uid: str,
    description: str,
    manufacturer: str = "Siemens",
    model: str = "SOMATOM",
) -> CTInstance:
    ds = _file_dataset(path, CTImageStorage, generate_uid())
    ds.Modality = "CT"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.SeriesDescription = description
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = model
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.SliceThickness = 2.0
    ds.Rows = 512
    ds.Columns = 512
    ds.InstanceNumber = 1
    _write(ds, path)
    return CTInstance(
        path=path,
        patient_id="P1",
        study_uid=study_uid,
        series_uid=series_uid,
        series_number=1,
        instance_number=1,
    )


def _extract_linked(root: Path):
    plans, doses, structs = extract_rt(root, max_workers=1)
    return link_rt_sets(plans, doses, structs)


def test_plan_referenced_structure_set_is_selected_among_setup_sets(tmp_path):
    """Patient 292929 must receive the 36-ROI target set, not InitLaserIso setup contours."""
    study_uid, frame_uid = generate_uid(), generate_uid()
    setup_uid, rich_uid, plan_uid = generate_uid(), generate_uid(), generate_uid()
    _mk_struct(
        tmp_path / "00_setup.dcm",
        setup_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["InitLaserIso", "AcqIsocenter"],
    )
    _mk_struct(
        tmp_path / "99_rich.dcm",
        rich_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "CTV1", "PTV1", "PTV2"],
    )
    _mk_plan(
        tmp_path / "plan.dcm",
        plan_uid,
        struct_uid=rich_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240416",
        rx_gy=64.0,
        fractions=32,
        label="plan",
    )
    _mk_dose(tmp_path / "dose.dcm", generate_uid(), plan_uid=plan_uid, study_uid=study_uid, frame_uid=frame_uid)

    linked = _extract_linked(tmp_path)

    assert len(linked) == 1
    assert linked[0].struct is not None
    assert linked[0].struct.sop_instance_uid == rich_uid
    assert "PTV1" in linked[0].struct.roi_names


def test_plan_reference_beats_a_larger_zero_target_structure_set(tmp_path):
    """Patient 481077 must not receive the 115-ROI auto-OAR set with zero targets."""
    study_uid, frame_uid = generate_uid(), generate_uid()
    auto_uid, target_uid, plan_uid = generate_uid(), generate_uid(), generate_uid()
    _mk_struct(
        tmp_path / "00_auto_oar.dcm",
        auto_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=[f"OAR_{index}" for index in range(115)],
    )
    _mk_struct(
        tmp_path / "99_target.dcm",
        target_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "CTV", "PTV"],
    )
    _mk_plan(
        tmp_path / "plan.dcm",
        plan_uid,
        struct_uid=target_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20250314",
        rx_gy=45.0,
        fractions=5,
        label="plan",
    )
    _mk_dose(tmp_path / "dose.dcm", generate_uid(), plan_uid=plan_uid, study_uid=study_uid, frame_uid=frame_uid)

    linked = _extract_linked(tmp_path)

    assert linked[0].struct is not None
    assert linked[0].struct.sop_instance_uid == target_uid
    assert len(linked[0].struct.roi_names) == 3


@pytest.mark.parametrize("shared_study", [False, True], ids=["dfci_cross_study", "kopernik_shared_study"])
def test_referenced_planning_ct_links_with_or_without_a_shared_study(tmp_path, shared_study):
    """DFCI cross-study references and the existing Kopernik shared-study shape must both resolve."""
    rt_study = generate_uid()
    ct_study = rt_study if shared_study else generate_uid()
    frame_uid, series_uid, struct_uid = generate_uid(), generate_uid(), generate_uid()
    ct = _mk_ct(
        tmp_path / "ct.dcm",
        study_uid=ct_study,
        series_uid=series_uid,
        frame_uid=frame_uid,
        description="Planning CT",
    )
    struct_path = _mk_struct(
        tmp_path / "struct.dcm",
        struct_uid,
        study_uid=rt_study,
        frame_uid=frame_uid,
        roi_names=["CTV", "PTV"],
        ct_series_uid=series_uid,
    )
    ct_index = {"P1": {ct_study: {series_uid: [ct]}}}

    selected, status = org.select_course_ct_series(ct_index, "P1", struct_path, rt_study)

    assert status == "referenced"
    assert selected == [ct]


def test_multi_plan_dose_links_every_referenced_plan_in_one_structure_course(tmp_path):
    """A PLAN_SUM dose tied to two phases on one planning structure set must retain both phases."""
    study_uid = generate_uid()
    frame_uid = generate_uid()
    struct_uid = generate_uid()
    struct = _mk_struct(
        tmp_path / "struct.dcm", struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid, roi_names=["CTV", "PTV"],
    )
    plan_uids = [str(generate_uid()), str(generate_uid())]
    plans = [
        _mk_plan(
            tmp_path / f"plan_{index}.dcm", plan_uid,
            struct_uid=struct_uid, study_uid=study_uid, frame_uid=frame_uid,
            date="20240101", rx_gy=50.0 if index == 0 else 10.0,
            fractions=25 if index == 0 else 5, label=f"phase-{index}",
        )
        for index, plan_uid in enumerate(plan_uids)
    ]
    dose = _mk_multi_plan_dose(
        tmp_path / "plan_sum.dcm", generate_uid(),
        plan_uids=plan_uids, study_uid=study_uid, frame_uid=frame_uid,
    )

    plan_info, dose_info, struct_info = extract_rt(tmp_path)
    linked = link_rt_sets(plan_info, dose_info, struct_info)
    grouped = group_by_course(linked)

    assert len(plans) == 2 and struct.exists() and dose.exists()
    assert len(linked) == 2
    assert len(grouped) == 1
    assert {item.plan.sop_instance_uid for item in linked} == set(plan_uids)
    assert {item.struct.sop_instance_uid for item in linked if item.struct} == {struct_uid}


def test_unresolved_dose_reference_is_not_attached_to_the_only_plan(tmp_path):
    """An explicit dose reference to a missing plan must not be guessed onto another plan."""
    study_uid = generate_uid()
    frame_uid = generate_uid()
    struct_uid = generate_uid()
    plan_uid = generate_uid()
    plan = _mk_plan(
        tmp_path / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        label="clinical",
        date="20241001",
        rx_gy=55.0,
        fractions=20,
    )
    dose = _mk_dose(
        tmp_path / "dose.dcm",
        generate_uid(),
        plan_uid=generate_uid(),
        study_uid=study_uid,
        frame_uid=frame_uid,
    )

    classified = getattr(org, "_classify_doses")([plan], [dose])

    assert classified.selected_doses == []
    assert classified.selected_plans == []
    assert classified.classification == "unresolved_reference_excluded"


def test_target_bearing_referenced_structure_without_dose_remains_a_plan_only_course(tmp_path):
    """Patient 351107 had a target-bearing planning structure set with plans but no RTDOSE."""
    study_uid = generate_uid()
    frame_uid = generate_uid()
    struct_uid = generate_uid()
    _mk_struct(
        tmp_path / "struct.dcm", struct_uid,
        study_uid=study_uid, frame_uid=frame_uid,
        roi_names=["CTV", "PTV"],
    )
    plan_uid = generate_uid()
    _mk_plan(
        tmp_path / "plan.dcm", plan_uid,
        struct_uid=struct_uid, study_uid=study_uid, frame_uid=frame_uid,
        date="20211122", rx_gy=45.0, fractions=25, label="course",
    )

    plans, doses, structs = extract_rt(tmp_path)
    linked = link_rt_sets(plans, doses, structs)
    grouped = group_by_course(linked)

    assert len(linked) == 1
    assert linked[0].dose is None
    assert linked[0].struct is not None
    assert linked[0].struct.sop_instance_uid == struct_uid
    assert list(grouped) == [("P1", struct_uid)]


def test_four_target_bearing_references_produce_four_courses(tmp_path):
    """Patient 481077 must retain all 4 target-bearing planning structure sets as courses."""
    study_uid, frame_uid = generate_uid(), generate_uid()
    for index in range(4):
        struct_uid, plan_uid = generate_uid(), generate_uid()
        _mk_struct(
            tmp_path / f"struct_{index}.dcm",
            struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            roi_names=[f"CTV{index + 1}", f"PTV{index + 1}"],
        )
        _mk_plan(
            tmp_path / f"plan_{index}.dcm",
            plan_uid,
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date=f"20240{index + 1}01",
            rx_gy=8.0 + index,
            fractions=1,
            label=f"course {index + 1}",
        )
        _mk_dose(
            tmp_path / f"dose_{index}.dcm",
            generate_uid(),
            plan_uid=plan_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )

    grouped = group_by_course(_extract_linked(tmp_path))

    assert len(grouped) == 4
    assert all(len(items) == 1 for items in grouped.values())


def test_identical_prescription_revisions_contribute_once(tmp_path):
    """Patient 353398 must remain 70 Gy in 33 fractions rather than summing 6 revisions."""
    study_uid, frame_uid, struct_uid = generate_uid(), generate_uid(), generate_uid()
    plans: list[Path] = []
    doses: list[Path] = []
    plan_uids: list[str] = []
    for index in range(6):
        plan_uid = generate_uid()
        plan_uids.append(plan_uid)
        plans.append(
            _mk_plan(
                tmp_path / f"plan_{index}.dcm",
                plan_uid,
                struct_uid=struct_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
                date="20180518",
                rx_gy=70.0,
                fractions=33,
                label=f"sib{index}",
            )
        )
        doses.append(
            _mk_dose(
                tmp_path / f"dose_{index}.dcm",
                generate_uid(),
                plan_uid=plan_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
            )
        )
    delivered = _mk_record(tmp_path / "record.dcm", plan_uids[-1])

    classified = getattr(org, "_classify_doses")(plans, doses, treatment_record_paths=[delivered])
    selected_rx = sum(org.infer_plan_rx_gy(pydicom.dcmread(path, stop_before_pixels=True)) or 0 for path in classified.selected_plans)

    assert classified.classification == "plan_revisions_deduplicated"
    assert classified.selected_plans == [plans[-1]]
    assert len(classified.selected_doses) == 1
    assert selected_rx == pytest.approx(70.0)
    assert not classified.should_sum


def test_sequential_phase_without_delivery_records_is_excluded_after_revision_deduplication(tmp_path):
    """Only the delivered 50 Gy revision contributes when the 10 Gy phase has no RTRECORD."""
    study_uid, frame_uid, struct_uid = generate_uid(), generate_uid(), generate_uid()
    specs = [(50.0, 25), (10.0, 5), (50.0, 25)]
    plans: list[Path] = []
    doses: list[Path] = []
    plan_uids: list[str] = []
    for index, (rx_gy, fractions) in enumerate(specs):
        plan_uid = generate_uid()
        plan_uids.append(plan_uid)
        plans.append(
            _mk_plan(
                tmp_path / f"plan_{index}.dcm",
                plan_uid,
                struct_uid=struct_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
                date="20180518",
                rx_gy=rx_gy,
                fractions=fractions,
                label=f"phase {index + 1}",
            )
        )
        doses.append(
            _mk_dose(
                tmp_path / f"dose_{index}.dcm",
                generate_uid(),
                plan_uid=plan_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
            )
        )
    delivered = [
        _mk_record(tmp_path / f"record_{index}.dcm", plan_uids[-1], date=f"201801{index + 1:02d}")
        for index in range(25)
    ]

    classified = getattr(org, "_classify_doses")(plans, doses, treatment_record_paths=delivered)
    selected_rx = sum(org.infer_plan_rx_gy(pydicom.dcmread(path, stop_before_pixels=True)) or 0 for path in classified.selected_plans)

    assert classified.classification == "delivered_plan_selected"
    assert classified.selected_plans == [plans[-1]]
    assert selected_rx == pytest.approx(50.0)
    assert not classified.should_sum


def test_replacement_plans_within_one_structure_course_do_not_double_count(tmp_path):
    """A 20-fraction plan and its 17 plus 3 fraction replacements must remain 55 Gy within one course."""
    study_uid, frame_uid, struct_uid = generate_uid(), generate_uid(), generate_uid()
    specs = [(55.0, 20), (46.75, 17), (8.25, 3)]
    plans: list[Path] = []
    doses: list[Path] = []
    for index, (rx_gy, fractions) in enumerate(specs):
        plan_uid = generate_uid()
        plans.append(
            _mk_plan(
                tmp_path / f"plan_{index}.dcm",
                plan_uid,
                struct_uid=struct_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
                date=f"2024120{index + 1}",
                rx_gy=rx_gy,
                fractions=fractions,
                label=f"plan {index + 1}",
            )
        )
        doses.append(
            _mk_dose(
                tmp_path / f"dose_{index}.dcm",
                generate_uid(),
                plan_uid=plan_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
            )
        )

    classified = org._classify_doses(plans, doses)
    selected_rx = sum(org.infer_plan_rx_gy(pydicom.dcmread(path, stop_before_pixels=True)) or 0 for path in classified.selected_plans)

    assert classified.classification == "replacement_course_total"
    assert selected_rx == pytest.approx(55.0)
    assert len(classified.selected_plans) in {1, 2}


def test_replans_on_three_structure_sets_keep_undelivered_courses_out_of_grid(tmp_path):
    """Only the structure-defined course linked to RTRECORD contributes a treatment grid."""
    frame_uid = generate_uid()
    course_specs = [
        [(55.0, 20), (55.0, 20)],
        [(55.0, 20), (8.25, 3)],
        [(55.0, 20), (46.75, 17)],
    ]
    record_plan_uid = ""
    for course_index, plan_specs in enumerate(course_specs):
        study_uid = generate_uid()
        struct_uid = generate_uid()
        _mk_struct(
            tmp_path / f"struct_{course_index}.dcm",
            struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            roi_names=[f"CTV{course_index + 1}", f"PTV{course_index + 1}"],
        )
        for plan_index, (rx_gy, fractions) in enumerate(plan_specs):
            plan_uid = generate_uid()
            _mk_plan(
                tmp_path / f"plan_{course_index}_{plan_index}.dcm",
                plan_uid,
                struct_uid=struct_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
                date=f"2024120{course_index + 1}",
                rx_gy=rx_gy,
                fractions=fractions,
                label=f"course {course_index + 1} plan {plan_index + 1}",
            )
            _mk_dose(
                tmp_path / f"dose_{course_index}_{plan_index}.dcm",
                generate_uid(),
                plan_uid=plan_uid,
                study_uid=study_uid,
                frame_uid=frame_uid,
            )
            if course_index == 1 and plan_index == 0:
                record_plan_uid = plan_uid

    records = [
        _mk_record(
            tmp_path / f"record_{index}.dcm",
            record_plan_uid,
            date=f"202412{index + 1:02d}",
        )
        for index in range(6)
    ]
    grouped = group_by_course(_extract_linked(tmp_path))
    totals: list[float] = []
    classifications: list[str] = []
    for items in grouped.values():
        plans = list(dict.fromkeys(item.plan.path for item in items))
        doses = list(dict.fromkeys(item.dose.path for item in items if item.dose is not None))
        classified = org._classify_doses(plans, doses, treatment_record_paths=records)
        totals.append(
            sum(
                org.infer_plan_rx_gy(pydicom.dcmread(path, stop_before_pixels=True)) or 0
                for path in classified.selected_plans
            )
        )
        classifications.append(classified.classification)

    assert len(grouped) == 3
    assert sorted(totals) == pytest.approx([0.0, 0.0, 55.0])
    assert sorted(classifications) == [
        "delivered_plan_selected",
        "no_delivered_plan_dose",
        "no_delivered_plan_dose",
    ]


def test_target_bearing_courses_a_year_apart_are_not_merged(tmp_path):
    """Patient 440657 bladder treatment and lung SBRT 1 year later must remain 2 courses."""
    study_uid, frame_uid = generate_uid(), generate_uid()
    for index, date in enumerate(("20241001", "20251001")):
        struct_uid, plan_uid = generate_uid(), generate_uid()
        _mk_struct(
            tmp_path / f"struct_{index}.dcm",
            struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            roi_names=["CTV", "PTV"],
        )
        _mk_plan(
            tmp_path / f"plan_{index}.dcm",
            plan_uid,
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date=date,
            rx_gy=55.0 if index == 0 else 50.0,
            fractions=20 if index == 0 else 5,
            label=f"course {index + 1}",
        )
        _mk_dose(
            tmp_path / f"dose_{index}.dcm",
            generate_uid(),
            plan_uid=plan_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )

    grouped = group_by_course(_extract_linked(tmp_path))

    assert len(grouped) == 2


def _treatment_records(
    tmp_path: Path,
    prefix: str,
    plan_uid: str,
    count: int,
    *,
    start: date = date(2024, 1, 1),
) -> list[Path]:
    return [
        _mk_record(
            tmp_path / f"{prefix}_{index:02d}.dcm",
            plan_uid,
            date=(start + timedelta(days=index)).strftime("%Y%m%d"),
        )
        for index in range(count)
    ]


def test_genuine_sequential_boost_with_records_on_both_phases_sums(tmp_path: Path) -> None:
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    phase_uids = [generate_uid(), generate_uid()]
    plans = [
        _mk_plan(
            tmp_path / "phase1.dcm",
            phase_uids[0],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20240101",
            rx_gy=50.0,
            fractions=25,
            label="phase 1",
        ),
        _mk_plan(
            tmp_path / "phase2.dcm",
            phase_uids[1],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20240205",
            rx_gy=10.0,
            fractions=5,
            label="phase 2",
        ),
    ]
    doses = [
        _mk_dose(
            tmp_path / f"phase{index + 1}_dose.dcm",
            generate_uid(),
            plan_uid=uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        for index, uid in enumerate(phase_uids)
    ]
    records = _treatment_records(tmp_path, "phase1_record", phase_uids[0], 25)
    records += _treatment_records(
        tmp_path,
        "phase2_record",
        phase_uids[1],
        5,
        start=date(2024, 2, 5),
    )

    classified = org._classify_doses(plans, doses, treatment_record_paths=records)

    assert classified.classification == "sequential_phases_summed"
    assert classified.selected_plans == plans
    assert classified.selected_doses == doses
    assert classified.should_sum


def test_delivered_remainder_replan_does_not_sum_full_plan_doses(tmp_path: Path) -> None:
    """10102925269 used 2 fractions of a 20-fraction plan, then a 17-fraction replan."""
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    plan_uids = [generate_uid(), generate_uid()]
    plans = [
        _mk_plan(
            tmp_path / "course_total.dcm",
            plan_uids[0],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20250925",
            rx_gy=55.0,
            fractions=20,
            label="course total",
        ),
        _mk_plan(
            tmp_path / "remainder.dcm",
            plan_uids[1],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20251008",
            rx_gy=46.75,
            fractions=17,
            label="remainder",
        ),
    ]
    doses = [
        _mk_dose(
            tmp_path / f"dose_{index}.dcm",
            generate_uid(),
            plan_uid=uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        for index, uid in enumerate(plan_uids)
    ]
    records = _treatment_records(tmp_path, "initial", plan_uids[0], 2)
    records += _treatment_records(
        tmp_path,
        "remainder",
        plan_uids[1],
        17,
        start=date(2024, 1, 1),
    )

    classified = org._classify_doses(
        plans,
        doses,
        treatment_record_paths=records,
    )

    assert classified.classification == "replacement_plan_chain"
    assert classified.selected_plans == plans
    assert classified.selected_doses == []
    assert set(classified.excluded_doses) == set(doses)
    assert classified.should_sum is False
    assert classified.prescription_plans == [plans[0]]


def test_remainder_chain_keeps_independent_boost_in_prescription_scope(
    tmp_path: Path,
) -> None:
    """10149603697 has a 16-fraction remainder and an independent 3-fraction boost."""
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    plan_uids = [generate_uid() for _ in range(3)]
    plans = [
        _mk_plan(
            tmp_path / "initial.dcm",
            plan_uids[0],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20220531",
            rx_gy=50.0,
            fractions=25,
            label="A1 pelvis",
        ),
        _mk_plan(
            tmp_path / "remainder.dcm",
            plan_uids[1],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20220601",
            rx_gy=32.0,
            fractions=16,
            label="A1 pelvis:1",
        ),
        _mk_plan(
            tmp_path / "boost.dcm",
            plan_uids[2],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20220608",
            rx_gy=12.0,
            fractions=3,
            label="A2 pelvis",
        ),
    ]
    doses = [
        _mk_dose(
            tmp_path / f"dose_{index}.dcm",
            generate_uid(),
            plan_uid=uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        for index, uid in enumerate(plan_uids)
    ]
    records = _treatment_records(tmp_path, "initial", plan_uids[0], 1)
    records += _treatment_records(
        tmp_path,
        "remainder",
        plan_uids[1],
        16,
        start=date(2022, 6, 1),
    )
    records += _treatment_records(
        tmp_path,
        "boost",
        plan_uids[2],
        3,
        start=date(2022, 6, 8),
    )

    classified = org._classify_doses(
        plans,
        doses,
        treatment_record_paths=records,
    )

    assert classified.classification == "replacement_plan_chain"
    assert classified.selected_plans == plans
    assert classified.selected_doses == []
    assert classified.prescription_plans == [plans[0], plans[2]]
    assert classified.should_sum is False


def test_kopernik_419783_replacement_chain_withholds_course_dose_totals(
    tmp_path: Path,
) -> None:
    """Reproduce the three-plan, 35-planned, 16-delivered course configuration."""

    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    plan_uids = [generate_uid() for _ in range(3)]
    plans = [
        _mk_plan(
            tmp_path / "RAP_50Gy.dcm",
            plan_uids[0],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20200224",
            rx_gy=50.0,
            fractions=25,
            label="RAP",
        ),
        _mk_plan(
            tmp_path / "RAP_25Gy_nowy.dcm",
            plan_uids[1],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20200312",
            rx_gy=25.0,
            fractions=10,
            label="RAP 25Gy Nowy",
        ),
        _mk_plan(
            tmp_path / "RAP_12Gy_stary.dcm",
            plan_uids[2],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20200224",
            rx_gy=12.0,
            fractions=6,
            label="RAP12Gy stary",
        ),
    ]
    doses = [
        _mk_dose(
            tmp_path / f"dose_{index}.dcm",
            generate_uid(),
            plan_uid=uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        for index, uid in enumerate(plan_uids)
    ]
    records = _treatment_records(tmp_path, "RAP", plan_uids[0], 6)
    records += _treatment_records(
        tmp_path,
        "RAP_25Gy_nowy",
        plan_uids[1],
        10,
        start=date(2020, 2, 25),
    )

    classified = org._classify_doses(
        plans,
        doses,
        treatment_record_paths=records,
    )
    delivery = org._calculate_delivery_summary(
        plans,
        records,
        selected_plan_paths=classified.selected_plans,
    )

    assert classified.classification == "replacement_plan_chain"
    assert classified.prescription_plans == []
    assert classified.selected_plans == plans[:2]
    assert classified.selected_doses == []
    assert delivery["planned_fraction_count"] == 35
    assert delivery["delivered_fraction_count"] == 16
    delivered_dose_gy = delivery["delivered_dose_gy"]
    delivery_method = delivery["delivery_method"]
    assert isinstance(delivered_dose_gy, (int, float))
    assert delivery_method is None or isinstance(delivery_method, str)
    assert delivered_dose_gy == pytest.approx(37.0)
    assert 100 * 16 / 35 == pytest.approx(45.7142857143)

    publication = org._scope_aware_course_dose_publication(
        prescribed_dose_scope="UNRESOLVED_REPLACEMENT_CHAIN",
        course_prescribed_dose_gy=None,
        course_resolved_prescribed_dose_total_gy=None,
        plan_prescribed_dose_gy=50.0,
        plan_resolved_prescribed_dose_total_gy=50.0,
        delivered_dose_gy=float(delivered_dose_gy),
        delivery_status=str(delivery["delivery_status"]),
        delivery_method=delivery_method,
    )
    assert publication["resolved_prescribed_dose_total_gy"] is None
    assert publication["delivered_dose_gy"] is None
    assert publication["delivery_status"] == "delivery_unresolved"

    parsed = parse_kopernik_treatment_description(
        "ICD9: 92.27: Teleradioterapia radykalna technika IMRT na obszar "
        "pecherza moczowego do dawki 37,0Gy/p.ref po 2,0Gy (6 frakcji) "
        "i 2,5Gy (10 frakcji)"
    )
    per_plan_delivery = cast(
        list[dict[str, object]], delivery["delivery_plan_details"]
    )
    phase_confirmation = confirm_two_phase_fractionation(
        parsed["sites"], per_plan_delivery
    )
    clinical_evidence = {
        "outcome": "RESOLVED_FROM_CLINICAL_RECORD",
        "clinical_resolved_total_gy": 37.0,
        "fractionation_classification": phase_confirmation,
    }
    recovered = org._clinical_delivery_publication(
        clinical_prescription_evidence=clinical_evidence,
        delivered_dose_gy=publication["delivered_dose_gy"],
        delivery_status=str(publication["delivery_status"]),
        delivery_method=publication["delivery_method"],
    )
    resolved_publication = org._scope_aware_course_dose_publication(
        prescribed_dose_scope="COURSE_TOTAL_CLINICAL_RECORD",
        course_prescribed_dose_gy=37.0,
        course_resolved_prescribed_dose_total_gy=37.0,
        plan_prescribed_dose_gy=50.0,
        plan_resolved_prescribed_dose_total_gy=50.0,
        delivered_dose_gy=recovered["delivered_dose_gy"],
        delivery_status=recovered["delivery_status"],
        delivery_method=recovered["delivery_method"],
    )

    assert phase_confirmation is not None
    assert phase_confirmation["dicom_delivered_total_gy"] == pytest.approx(37.0)
    assert recovered["independently_established"] is True
    assert resolved_publication == {
        "prescribed_dose_gy": 37.0,
        "resolved_prescribed_dose_total_gy": 37.0,
        "delivered_dose_gy": 37.0,
        "delivery_status": "fully_delivered",
        "delivery_method": "clinical_two_phase_fractionation_with_dicom_records",
    }


def test_eighty_percent_overlapping_phases_remain_additive(
    tmp_path: Path,
) -> None:
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    plan_uids = [generate_uid(), generate_uid()]
    plans = [
        _mk_plan(
            tmp_path / "initial_12.dcm",
            plan_uids[0],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20200330",
            rx_gy=36.0,
            fractions=12,
            label="initial",
        ),
        _mk_plan(
            tmp_path / "remaining_3.dcm",
            plan_uids[1],
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date="20200428",
            rx_gy=9.0,
            fractions=3,
            label="remaining",
        ),
    ]
    doses = [
        _mk_dose(
            tmp_path / f"overlap_dose_{index}.dcm",
            generate_uid(),
            plan_uid=uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        for index, uid in enumerate(plan_uids)
    ]
    records = _treatment_records(tmp_path, "initial_overlap", plan_uids[0], 9)
    records += _treatment_records(
        tmp_path,
        "remainder_overlap",
        plan_uids[1],
        3,
        start=date(2024, 1, 9),
    )

    classified = org._classify_doses(
        plans,
        doses,
        treatment_record_paths=records,
    )

    assert classified.classification == "sequential_phases_summed"
    assert classified.selected_plans == plans
    assert classified.selected_doses == doses
    assert classified.should_sum is True


def test_plan_sum_containing_an_undelivered_phase_is_rejected(tmp_path: Path) -> None:
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    plan_uids = [str(generate_uid()), str(generate_uid())]
    plans = [
        _mk_plan(
            tmp_path / f"phase_{index}.dcm",
            uid,
            struct_uid=struct_uid,
            study_uid=study_uid,
            frame_uid=frame_uid,
            date=f"2024010{index + 1}",
            rx_gy=50.0 if index == 0 else 10.0,
            fractions=25 if index == 0 else 5,
            label=f"phase {index + 1}",
        )
        for index, uid in enumerate(plan_uids)
    ]
    plan_sum = _mk_multi_plan_dose(
        tmp_path / "plan_sum.dcm",
        generate_uid(),
        plan_uids=plan_uids,
        study_uid=study_uid,
        frame_uid=frame_uid,
    )
    plan_sum_dataset = pydicom.dcmread(str(plan_sum))
    plan_sum_dataset.DoseSummationType = "PLAN_SUM"
    plan_sum_dataset.save_as(str(plan_sum), write_like_original=False)
    records = _treatment_records(tmp_path, "phase1_record", plan_uids[0], 25)

    classified = org._classify_doses(
        plans,
        [plan_sum],
        treatment_record_paths=records,
    )

    assert classified.classification == "no_delivered_plan_dose"
    assert classified.selected_plans == []
    assert classified.selected_doses == []
    assert classified.excluded_doses == [plan_sum]


def test_single_plan_with_zero_linked_records_is_not_a_treatment_grid(tmp_path: Path) -> None:
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    plan_uid = generate_uid()
    plan = _mk_plan(
        tmp_path / "planned_only.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=50.0,
        fractions=20,
        label="planned only",
    )
    dose = _mk_dose(
        tmp_path / "planned_only_dose.dcm",
        generate_uid(),
        plan_uid=plan_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
    )
    unrelated_record = _mk_record(
        tmp_path / "unrelated_record.dcm",
        generate_uid(),
    )

    classified = org._classify_doses(
        [plan],
        [dose],
        treatment_record_paths=[unrelated_record],
    )

    assert classified.classification == "no_delivered_plan_dose"
    assert classified.selected_plans == []
    assert classified.selected_doses == []
    assert classified.excluded_doses == [dose]
    assert classified.should_sum is False


def test_records_on_subset_of_treatment_dates_do_not_add_verification_dose(tmp_path: Path) -> None:
    """The 353398 failure added a low-dose verification plan to a delivered 70 Gy course."""
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    full_uid = generate_uid()
    verification_uid = generate_uid()
    full = _mk_plan(
        tmp_path / "full.dcm", full_uid, struct_uid=struct_uid,
        study_uid=study_uid, frame_uid=frame_uid,
        date="20180518", rx_gy=70.0, fractions=33, label="course",
    )
    verification = _mk_plan(
        tmp_path / "verification.dcm", verification_uid, struct_uid=struct_uid,
        study_uid=study_uid, frame_uid=frame_uid,
        date="20180518", rx_gy=1.0, fractions=10, label="verification",
    )
    doses = [
        _mk_dose(
            tmp_path / "full_dose.dcm", generate_uid(),
            plan_uid=full_uid, study_uid=study_uid, frame_uid=frame_uid,
        ),
        _mk_dose(
            tmp_path / "verification_dose.dcm", generate_uid(),
            plan_uid=verification_uid, study_uid=study_uid, frame_uid=frame_uid,
        ),
    ]
    full_records = _treatment_records(tmp_path, "full_record", full_uid, 33)
    verification_records = _treatment_records(tmp_path, "verification_record", verification_uid, 6)

    classified = getattr(org, "_classify_doses")(
        [full, verification], doses,
        treatment_record_paths=full_records + verification_records,
    )

    assert classified.selected_plans == [full]
    assert sum(
        org.infer_plan_rx_gy(pydicom.dcmread(path, stop_before_pixels=True)) or 0
        for path in classified.selected_plans
    ) == pytest.approx(70.0)
    assert classified.should_sum is False


def test_zero_record_course_total_does_not_replace_delivered_partial_replan(tmp_path: Path) -> None:
    """A recordless course-total plan cannot replace a delivered partial replan grid."""
    struct_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    total_uid = generate_uid()
    partial_uid = generate_uid()
    total = _mk_plan(
        tmp_path / "total.dcm", total_uid, struct_uid=struct_uid,
        study_uid=study_uid, frame_uid=frame_uid,
        date="20241001", rx_gy=55.0, fractions=20, label="course",
    )
    partial = _mk_plan(
        tmp_path / "partial.dcm", partial_uid, struct_uid=struct_uid,
        study_uid=study_uid, frame_uid=frame_uid,
        date="20241024", rx_gy=36.0, fractions=12, label="replan",
    )
    doses = [
        _mk_dose(
            tmp_path / "total_dose.dcm", generate_uid(),
            plan_uid=total_uid, study_uid=study_uid, frame_uid=frame_uid,
        ),
        _mk_dose(
            tmp_path / "partial_dose.dcm", generate_uid(),
            plan_uid=partial_uid, study_uid=study_uid, frame_uid=frame_uid,
        ),
    ]
    partial_records = _treatment_records(tmp_path, "partial_record", partial_uid, 12)

    classified = getattr(org, "_classify_doses")(
        [total, partial], doses,
        treatment_record_paths=partial_records,
    )

    assert classified.selected_plans == [partial]
    assert sum(
        org.infer_plan_rx_gy(pydicom.dcmread(path, stop_before_pixels=True)) or 0
        for path in classified.selected_plans
    ) == pytest.approx(36.0)
    assert classified.should_sum is False


def test_qa_phantom_and_topogram_series_are_ineligible_for_courses(tmp_path):
    """DFCI ScandiDos VirtualCT and scanner topograms must not become patient courses."""
    frame_uid = generate_uid()
    phantom_series = generate_uid()
    phantom = [
        _mk_ct(
            tmp_path / "phantom.dcm",
            study_uid=generate_uid(),
            series_uid=phantom_series,
            frame_uid=frame_uid,
            description="ARIA RadOnc Images - Verification Plan Phantom",
            manufacturer="ScandiDos AB",
            model="VirtualCT",
        )
    ] * 20
    topogram_series = generate_uid()
    topogram = [
        _mk_ct(
            tmp_path / "topogram.dcm",
            study_uid=generate_uid(),
            series_uid=topogram_series,
            frame_uid=frame_uid,
            description="Topogram",
        )
    ] * 20

    classify = getattr(org, "_classify_organize_ct_series")
    phantom_ok, _, phantom_reason = classify(phantom, is_planning_ct=True)
    topogram_ok, _, topogram_reason = classify(topogram, is_planning_ct=True)

    assert not phantom_ok
    assert phantom_reason and "qa_phantom" in phantom_reason
    assert not topogram_ok
    assert topogram_reason and ("topogram" in topogram_reason or "localizer" in topogram_reason)


def test_plan_and_dose_without_targets_fail_the_course_qc_gate(tmp_path):
    """Kopernik zero-target courses must fail loudly instead of passing every existing gate."""
    study_uid, frame_uid, struct_uid, plan_uid = generate_uid(), generate_uid(), generate_uid(), generate_uid()
    struct = _mk_struct(
        tmp_path / "struct.dcm",
        struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "InitLaserIso", "AcqIsocenter"],
    )
    plan = _mk_plan(
        tmp_path / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=55.0,
        fractions=20,
        label="plan",
    )
    dose = _mk_dose(tmp_path / "dose.dcm", generate_uid(), plan_uid=plan_uid, study_uid=study_uid, frame_uid=frame_uid)

    qc_error = getattr(org, "CourseTargetQCError")
    validate = getattr(org, "validate_course_target_qc")
    with pytest.raises(qc_error, match="zero target volumes"):
        validate("P1", "2024-01", [plan], [dose], struct)


def test_compact_target_suffix_remains_a_target_name():
    """DFCI compact target names such as PTVbt must retain target status."""
    assert org.target_volume_names(["PTVbt", "CTVn", "notPTV"]) == ["PTVbt", "CTVn"]


def test_boolean_crop_name_does_not_satisfy_course_target_qc(tmp_path):
    """Kopernik Pecherz - PTV alone is a boolean crop, not a target volume."""
    study_uid, frame_uid, struct_uid, plan_uid = generate_uid(), generate_uid(), generate_uid(), generate_uid()
    struct = _mk_struct(
        tmp_path / "crop_only.dcm",
        struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "Pecherz - PTV"],
    )
    plan = _mk_plan(
        tmp_path / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=55.0,
        fractions=20,
        label="plan",
    )
    dose = _mk_dose(tmp_path / "dose.dcm", generate_uid(), plan_uid=plan_uid, study_uid=study_uid, frame_uid=frame_uid)

    with pytest.raises(org.CourseTargetQCError, match="zero target volumes"):
        org.validate_course_target_qc("P1", "2024-01", [plan], [dose], struct)


def test_optimization_helper_name_does_not_satisfy_course_target_qc(tmp_path):
    """DFCI zPtvOpt alone is an optimization helper, not a target volume."""
    study_uid, frame_uid, struct_uid, plan_uid = generate_uid(), generate_uid(), generate_uid(), generate_uid()
    struct = _mk_struct(
        tmp_path / "optimization_only.dcm",
        struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "zPtvOpt"],
    )
    plan = _mk_plan(
        tmp_path / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=55.0,
        fractions=20,
        label="plan",
    )
    dose = _mk_dose(tmp_path / "dose.dcm", generate_uid(), plan_uid=plan_uid, study_uid=study_uid, frame_uid=frame_uid)

    with pytest.raises(org.CourseTargetQCError, match="zero target volumes"):
        org.validate_course_target_qc("P1", "2024-01", [plan], [dose], struct)


def test_duplicate_paths_for_one_structure_uid_keep_rs_and_referenced_ct(tmp_path, monkeypatch, caplog):
    """Duplicate DFCI copies of one RTSTRUCT must not bypass reference-based CT selection."""
    dicom_root = tmp_path / "dicom"
    rt_study, frame_uid = generate_uid(), generate_uid()
    small_series, large_series = generate_uid(), generate_uid()
    struct_uid, plan_uid = generate_uid(), generate_uid()
    struct_a = _mk_struct(
        dicom_root / "a" / "struct.dcm",
        struct_uid,
        study_uid=rt_study,
        frame_uid=frame_uid,
        roi_names=["BODY", "PTV1"],
        ct_series_uid=small_series,
    )
    struct_b = _mk_struct(
        dicom_root / "b" / "duplicate.dcm",
        struct_uid,
        study_uid=rt_study,
        frame_uid=frame_uid,
        roi_names=["BODY", "PTV1"],
        ct_series_uid=small_series,
    )
    plan = _mk_plan(
        dicom_root / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=rt_study,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=55.0,
        fractions=20,
        label="plan",
    )
    dose = _mk_dose(
        dicom_root / "dose.dcm",
        generate_uid(),
        plan_uid=plan_uid,
        study_uid=rt_study,
        frame_uid=frame_uid,
    )
    small_ct = [
        _mk_ct(
            dicom_root / f"small_{index}.dcm",
            study_uid=rt_study,
            series_uid=small_series,
            frame_uid=frame_uid,
            description="Referenced planning CT",
        )
        for index in range(10)
    ]
    large_ct = [
        _mk_ct(
            dicom_root / f"large_{index}.dcm",
            study_uid=rt_study,
            series_uid=large_series,
            frame_uid=frame_uid,
            description="Larger unreferenced CT",
        )
        for index in range(11)
    ]

    plan_infos, dose_infos, struct_infos = extract_rt(dicom_root, max_workers=1)
    struct_by_path = {info.path: info for info in struct_infos}
    linked = [
        LinkedSet(
            patient_id="P1",
            plan=plan_infos[0],
            dose=dose_infos[0],
            struct=struct_by_path[path],
            ct_study_uid=rt_study,
            frame_of_reference_uid=frame_uid,
        )
        for path in (struct_a, struct_b)
    ]
    ct_index = {
        "P1": {
            rt_study: {
                small_series: small_ct,
                large_series: large_ct,
            }
        }
    }

    monkeypatch.setattr(org, "index_ct_series", lambda *args, **kwargs: ct_index)
    monkeypatch.setattr(
        org,
        "extract_rt_with_records",
        lambda *args, **kwargs: (plan_infos, dose_infos, struct_infos, {}),
    )
    monkeypatch.setattr(org, "link_rt_sets", lambda *args, **kwargs: linked)
    monkeypatch.setattr(org, "group_by_course", lambda *args, **kwargs: {("P1", struct_uid): linked})
    monkeypatch.setattr(org, "_index_series_and_registrations", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(org, "_index_rt_files", lambda *args, **kwargs: {})
    monkeypatch.setattr(org, "_looks_like_patient_series_layout", lambda *args, **kwargs: False)
    monkeypatch.setattr(org, "_ensure_ct_nifti", _write_placeholder_ct_nifti)

    cfg = PipelineConfig(
        dicom_root=dicom_root,
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        dicom_copy_dedup_by_sop_uid=False,
    )
    caplog.set_level("INFO")

    outputs = org.organize_and_merge(cfg)

    assert len(outputs) == 1
    assert outputs[0].rs_path is not None and outputs[0].rs_path.exists()
    copied_ct = sorted(outputs[0].dirs.dicom_ct.glob("*.dcm"))
    assert copied_ct
    copied_series = {
        str(pydicom.dcmread(path, stop_before_pixels=True).SeriesInstanceUID)
        for path in copied_ct
    }
    assert copied_series == {small_series}
    assert any("referenced" in record.getMessage() for record in caplog.records)
    assert plan.exists() and dose.exists()


def test_organize_contract_round_trips_to_dvh_for_plan_only_course(tmp_path, monkeypatch):
    """The real organizer serializer and DVH must preserve plan-only membership."""
    dicom_root = tmp_path / "dicom"
    study_uid, frame_uid = generate_uid(), generate_uid()
    struct_uid, plan_uid = generate_uid(), generate_uid()
    struct = _mk_struct(
        dicom_root / "struct.dcm",
        struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "PTV1"],
    )
    plan = _mk_plan(
        dicom_root / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=55.0,
        fractions=20,
        label="plan-only",
    )
    ct_series_uid = generate_uid()
    ct = [
        _mk_ct(
            dicom_root / f"ct_{index}.dcm",
            study_uid=study_uid,
            series_uid=ct_series_uid,
            frame_uid=frame_uid,
            description="Referenced planning CT",
        )
        for index in range(3)
    ]

    ct_index = {"P1": {study_uid: {ct_series_uid: ct}}}
    monkeypatch.setattr(org, "index_ct_series", lambda *args, **kwargs: ct_index)
    monkeypatch.setattr(org, "_index_series_and_registrations", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(org, "_index_rt_files", lambda *args, **kwargs: {})
    monkeypatch.setattr(org, "_looks_like_patient_series_layout", lambda *args, **kwargs: False)
    monkeypatch.setattr(org, "_ensure_ct_nifti", _write_placeholder_ct_nifti)

    cfg = PipelineConfig(
        dicom_root=dicom_root,
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        dicom_copy_dedup_by_sop_uid=False,
    )
    outputs = org.organize_and_merge(cfg)

    assert len(outputs) == 1
    course_dir = outputs[0].dirs.root
    contract = load_course_contract(course_dir)
    contract_path = course_dir / "metadata" / "case_metadata.json"
    contract_bytes_before_dvh = contract_path.read_bytes()
    selected_plan_uids = [item["sop_instance_uid"] for item in contract.selected_plans]
    assert selected_plan_uids == [plan_uid]
    assert contract.selected_doses == []
    assert contract.data["dvh"]["reason_code"] == "plan_only_no_authoritative_dose_grid"

    monkeypatch.setattr(cli_module, "organize_and_merge", lambda _cfg: outputs)
    monkeypatch.setattr(
        cli_module,
        "run_tasks_with_adaptive_workers",
        lambda _label, tasks, function, **_kwargs: [function(task) for task in tasks],
    )
    exit_status = cli_module.main(
        [
            "--dicom-root",
            str(dicom_root),
            "--outdir",
            str(cfg.output_root),
            "--logs",
            str(cfg.logs_root),
            "--stage",
            "dvh",
            "--no-metadata",
        ]
    )
    assert exit_status == 0
    assert not (course_dir / "dvh_metrics.xlsx").exists()
    assert contract_path.read_bytes() == contract_bytes_before_dvh
    qc = json.loads((course_dir / "metadata" / "dvh_qc.json").read_text(encoding="utf-8"))
    resolution = qc["dose_resolution"]
    assert resolution["source_plan_sop_instance_uids"] == selected_plan_uids
    assert resolution["selected_plan_paths"] == [
        str(contract.resolve_path(contract.selected_plans[0]["path"], "selected_plan.path"))
    ]
    assert resolution["selected_dose_paths"] == []
    assert resolution["dvh"] == contract.data["dvh"]


def test_organize_contract_round_trips_dose_membership_to_dvh(
    tmp_path, monkeypatch
):
    """The organizer's dose-bearing serializer is consumed unchanged by DVH."""
    dicom_root = tmp_path / "dicom"
    study_uid, frame_uid = generate_uid(), generate_uid()
    ct_series_uid = generate_uid()
    struct_uid, plan_uid, dose_uid = generate_uid(), generate_uid(), generate_uid()
    _mk_struct(
        dicom_root / "struct.dcm",
        struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        roi_names=["BODY", "PTV1"],
        ct_series_uid=ct_series_uid,
        with_contours=True,
    )
    _mk_plan(
        dicom_root / "plan.dcm",
        plan_uid,
        struct_uid=struct_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        date="20240101",
        rx_gy=55.0,
        fractions=20,
        label="dose-bearing",
    )
    _mk_dose(
        dicom_root / "dose.dcm",
        dose_uid,
        plan_uid=plan_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        with_pixels=True,
    )
    _mk_record(dicom_root / "record.dcm", plan_uid)
    ct = [
        _mk_ct(
            dicom_root / f"ct_{index}.dcm",
            study_uid=study_uid,
            series_uid=ct_series_uid,
            frame_uid=frame_uid,
            description="Referenced planning CT",
        )
        for index in range(3)
    ]

    ct_index = {"P1": {study_uid: {ct_series_uid: ct}}}
    monkeypatch.setattr(org, "index_ct_series", lambda *args, **kwargs: ct_index)
    monkeypatch.setattr(org, "_index_series_and_registrations", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(org, "_looks_like_patient_series_layout", lambda *args, **kwargs: False)
    monkeypatch.setattr(org, "_ensure_ct_nifti", _write_placeholder_ct_nifti)

    cfg = PipelineConfig(
        dicom_root=dicom_root,
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        dicom_copy_dedup_by_sop_uid=False,
    )
    outputs = org.organize_and_merge(cfg)

    assert len(outputs) == 1
    course_dir = outputs[0].dirs.root
    contract = load_course_contract(course_dir)
    contract_path = course_dir / "metadata" / "case_metadata.json"
    contract_bytes_before_dvh = contract_path.read_bytes()
    selected_plan_uids = [item["sop_instance_uid"] for item in contract.selected_plans]
    selected_dose_uids = [item["sop_instance_uid"] for item in contract.selected_doses]
    assert selected_plan_uids
    assert selected_dose_uids
    assert selected_plan_uids == outputs[0].source_plan_uids
    assert selected_plan_uids == [plan_uid]
    assert selected_dose_uids == [dose_uid]
    assert contract.data["dose_grid"] is not None
    delivery = contract.delivery
    assert delivery["prescribed_dose_gy"] == pytest.approx(55.0)
    assert delivery["resolved_prescribed_dose_total_gy"] == pytest.approx(55.0)
    assert delivery["status"] == "partially_delivered"
    assert delivery["delivered_dose_gy"] == pytest.approx(2.75)
    assert delivery["method"] == "record_fraction_weighted_prescription"
    assert len(delivery["per_plan"]) == 1
    plan_delivery = delivery["per_plan"][0]
    assert plan_delivery["plan_sop_uid"] == plan_uid
    assert plan_delivery["source_prescribed_dose_gy"] == pytest.approx(55.0)
    assert plan_delivery["beam_dose_sum_per_fraction_gy"] == pytest.approx(2.75)
    assert plan_delivery["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert (
        plan_delivery["prescription_resolution_method"]
        == "BEAMDOSE_TOTAL_5PCT_V1"
    )
    assert plan_delivery["resolved_prescribed_dose_total_gy"] == pytest.approx(55.0)
    assert plan_delivery["delivered_record_count"] == 1
    assert plan_delivery["delivered_fraction_count"] == 1
    assert plan_delivery["treatment_dates"] == ["20240102"]
    assert plan_delivery["zero_delivery_records"] is False
    assert plan_delivery["record_paths"]
    record_path = contract.resolve_path(
        plan_delivery["record_paths"][0],
        "delivery.per_plan[0].record_paths[0]",
    )
    assert record_path is not None
    assert record_path.is_file()
    dose_grid_before = dict(contract.data["dose_grid"])
    dose_qc_before = dict(contract.dose_qc)

    output = dvh_for_course(course_dir, parallel_workers=1)

    assert output == course_dir / "dvh_metrics.xlsx"
    assert output is not None
    assert output.is_file()
    assert contract_path.read_bytes() == contract_bytes_before_dvh
    workbook = pd.read_excel(output)
    assert not workbook.empty
    assert set(workbook["source_plan_sop_instance_uids"]) == set(selected_plan_uids)
    assert set(workbook["source_dose_sop_instance_uids"]) == set(selected_dose_uids)
    assert set(workbook["Dose_Grid_Semantics"]) == {dose_grid_before["semantics"]}
    assert set(workbook["Delivery_Status"]) == {delivery["status"]}
    assert bool(workbook["Delivered_Dose_Gy"].notna().all())
    assert workbook["Delivered_Dose_Gy"].tolist() == pytest.approx(
        [delivery["delivered_dose_gy"]] * len(workbook)
    )
    assert "DmeanGy" in workbook
    assert workbook["DmeanGy"].count() == len(workbook)
    qc = json.loads((course_dir / "metadata" / "dvh_qc.json").read_text(encoding="utf-8"))
    emitted_dvh = qc["dose_resolution"]["dvh"]
    assert emitted_dvh["metrics_status"] == "computed"
    assert emitted_dvh["output"] == "dvh_metrics.xlsx"
    assert (course_dir / emitted_dvh["output"]).is_file()
    assert emitted_dvh == contract.data["dvh"]
    assert qc["dose_resolution"]["source_plan_sop_instance_uids"] == selected_plan_uids
    assert qc["dose_resolution"]["delivery_status"] == delivery["status"]
    assert qc["dose_resolution"]["delivered_dose_gy"] == pytest.approx(
        delivery["delivered_dose_gy"]
    )
    contract_after = load_course_contract(course_dir)
    assert contract_after.data["dose_grid"] == dose_grid_before
    assert contract_after.dose_qc == dose_qc_before


def test_plan_label_text_classifier_cannot_be_called():
    """Course dose selection must expose no path that keys decisions to plan labels."""
    assert not hasattr(org, "_classify_doses_legacy")
    assert not hasattr(org, "_is_replan_text")
    assert not hasattr(org, "_is_boost_text")


def _ct_only_config(tmp_path: Path, *, allow: bool) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        allow_ct_only_courses=allow,
    )


def _write_ct_only_series(root: Path) -> None:
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for index in range(10):
        _mk_ct(
            root / "P1" / f"CT_{index}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            frame_uid=frame_uid,
            description="Diagnostic CT",
        )


def test_ct_only_input_fails_loudly_when_support_is_not_enabled(tmp_path):
    """A diagnostic CT-only cohort must not silently produce zero courses by default."""
    cfg = _ct_only_config(tmp_path, allow=False)
    _write_ct_only_series(cfg.dicom_root)

    with pytest.raises(org.CTOnlyCohortError, match="allow_ct_only_courses"):
        org.organize_and_merge(cfg)


def test_ct_only_input_produces_a_course_when_explicitly_enabled(tmp_path, monkeypatch):
    """The explicit CT-only option must restore diagnostic radiomics course output."""
    cfg = _ct_only_config(tmp_path, allow=True)
    _write_ct_only_series(cfg.dicom_root)
    monkeypatch.setattr(org, "_ensure_ct_nifti", _write_placeholder_ct_nifti)

    outputs = org.organize_and_merge(cfg)

    assert len(outputs) == 1
    assert len(list(outputs[0].dirs.dicom_ct.glob("*.dcm"))) == 10


def test_organize_quarantines_one_contract_failure_and_validates_later_course(
    tmp_path, monkeypatch
):
    cfg = _ct_only_config(tmp_path, allow=True)
    for series_index in range(2):
        study_uid = generate_uid()
        series_uid = generate_uid()
        frame_uid = generate_uid()
        for slice_index in range(10):
            _mk_ct(
                cfg.dicom_root
                / "P1"
                / f"series_{series_index}_ct_{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                frame_uid=frame_uid,
                description=f"Diagnostic CT {series_index}",
            )
    monkeypatch.setattr(org, "_ensure_ct_nifti", _write_placeholder_ct_nifti)
    original_publish = org._validate_and_publish_case_metadata
    publication_calls = 0

    def _fail_first_publication(course_dir, case_metadata):
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 1:
            raise org.CourseContractError("intentional stale selected plan")
        return original_publish(course_dir, case_metadata)

    monkeypatch.setattr(
        org, "_validate_and_publish_case_metadata", _fail_first_publication
    )

    outputs = org.organize_and_merge(cfg)

    assert publication_calls == 2
    assert len(outputs) == 1
    ledger = read_organize_ledger(cfg.output_root)
    assert ledger["attempted_course_count"] == 2
    assert ledger["validated_course_count"] == 1
    assert ledger["technical_quarantine_count"] == 1
    quarantine = ledger["technical_quarantines"][0]
    assert quarantine["clinical_exclusion"] is False
    assert "intentional stale selected plan" in quarantine["reason"]
    assert Path(quarantine["quarantine_path"]).is_dir()
    assert not Path(quarantine["path"]).exists()
