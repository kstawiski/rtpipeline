from pathlib import Path
import json

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, RTPlanStorage, RTStructureSetStorage, generate_uid

from rtpipeline.dvh import _resolve_dvh_dose, _resolve_dvh_structures
from rtpipeline.layout import build_course_dirs
from rtpipeline.organize import _classify_doses


def _write(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.PatientID = "P1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    return ds


def _mk_plan(
    path: Path,
    plan_uid: str,
    *,
    label: str = "plan",
    rx: float = 70.0,
    rtstruct_uid: str | None = None,
) -> Path:
    ds = _file_dataset(path, RTPlanStorage, plan_uid)
    ds.Modality = "RTPLAN"
    ds.RTPlanLabel = label
    ds.RTPlanName = label
    ds.RTPlanDate = "20240101"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.498.1"
    dose_ref = Dataset()
    dose_ref.DoseReferenceType = "TARGET"
    dose_ref.TargetPrescriptionDose = float(rx)
    ds.DoseReferenceSequence = Sequence([dose_ref])
    if rtstruct_uid:
        ref = Dataset()
        ref.ReferencedSOPClassUID = RTStructureSetStorage
        ref.ReferencedSOPInstanceUID = rtstruct_uid
        ds.ReferencedStructureSetSequence = Sequence([ref])
    return _write(ds, path)


def _mk_dose(
    path: Path,
    dose_uid: str,
    plan_uid: str,
    summation_type: str,
    *,
    frame_of_reference_uid: str = "1.2.826.0.1.3680043.8.498.1",
) -> Path:
    ds = _file_dataset(path, RTDoseStorage, dose_uid)
    ds.Modality = "RTDOSE"
    ds.FrameOfReferenceUID = frame_of_reference_uid
    ds.DoseSummationType = summation_type
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.Rows = 2
    ds.Columns = 2
    ds.NumberOfFrames = 2
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.GridFrameOffsetVector = [0.0, 1.0]
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTPlanStorage
    ref.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref])
    return _write(ds, path)


def _mk_rtstruct(path: Path, sop_uid: str, frame_of_reference_uid: str) -> Path:
    ds = _file_dataset(path, RTStructureSetStorage, sop_uid)
    ds.Modality = "RTSTRUCT"
    ds.FrameOfReferenceUID = frame_of_reference_uid
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = frame_of_reference_uid
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])
    ds.StructureSetROISequence = Sequence([])
    ds.ROIContourSequence = Sequence([])
    ds.RTROIObservationsSequence = Sequence([])
    return _write(ds, path)


def test_dvh_resolver_uses_plan_dose_not_first_beam(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    plan_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "RP.dcm", plan_uid)
    beam = _mk_dose(dirs.dicom_rtdose / "000_BEAM.dcm", generate_uid(), plan_uid, "BEAM")
    plan_dose = _mk_dose(dirs.dicom_rtdose / "999_PLAN.dcm", generate_uid(), plan_uid, "PLAN")

    resolved = _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")

    assert resolved.ok
    assert resolved.rp_path == plan
    assert resolved.rd_path == plan_dose
    assert resolved.output_dose_summation_type == "PLAN"
    assert resolved.source_dose_summation_types == ["PLAN"]
    assert beam.name not in str(resolved.rd_path)


def test_dvh_resolver_rejects_single_beam_dose(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    plan_uid = generate_uid()
    _mk_plan(dirs.dicom_rtplan / "RP.dcm", plan_uid)
    _mk_dose(dirs.dicom_rtdose / "000_BEAM.dcm", generate_uid(), plan_uid, "BEAM")

    resolved = _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")

    assert not resolved.ok
    assert resolved.skip_reason
    assert "BEAM" in resolved.skip_reason or "summation" in resolved.skip_reason


def test_classifier_excludes_beams_when_matching_plan_dose_exists(tmp_path):
    plan_uid = generate_uid()
    plan = _mk_plan(tmp_path / "RP.dcm", plan_uid)
    beam_1 = _mk_dose(tmp_path / "001_BEAM.dcm", generate_uid(), plan_uid, "BEAM")
    plan_dose = _mk_dose(tmp_path / "002_PLAN.dcm", generate_uid(), plan_uid, "PLAN")
    beam_2 = _mk_dose(tmp_path / "003_BEAM.dcm", generate_uid(), plan_uid, "BEAM")

    classified = _classify_doses([plan], [beam_1, plan_dose, beam_2])

    assert classified.classification == "single_dose"
    assert classified.selected_doses == [plan_dose]
    assert set(classified.excluded_doses) == set()
    assert any("BEAM" in warning for warning in classified.warnings)


def test_classifier_sums_same_plan_beam_doses_when_no_plan_dose_exists(tmp_path):
    plan_uid = generate_uid()
    plan = _mk_plan(tmp_path / "RP.dcm", plan_uid)
    beam_1 = _mk_dose(tmp_path / "001_BEAM.dcm", generate_uid(), plan_uid, "BEAM")
    beam_2 = _mk_dose(tmp_path / "002_BEAM.dcm", generate_uid(), plan_uid, "BEAM")

    classified = _classify_doses([plan], [beam_1, beam_2])

    assert classified.classification == "beam_doses_summed_to_plan"
    assert classified.selected_doses == [beam_1, beam_2]
    assert classified.selected_plans == [plan]
    assert classified.should_sum


def test_structure_resolver_uses_plan_referenced_rtstruct_from_source_study(tmp_path):
    project = tmp_path / "project"
    course = project / "work" / "rtpipe" / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()

    plan_uid = generate_uid()
    plan_for_uid = generate_uid()
    wrong_for_uid = generate_uid()
    rtstruct_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "RP.dcm", plan_uid, rtstruct_uid=rtstruct_uid)
    dose = _mk_dose(
        dirs.dicom_rtdose / "RD.dcm",
        generate_uid(),
        plan_uid,
        "PLAN",
        frame_of_reference_uid=plan_for_uid,
    )
    course_rs = _mk_rtstruct(dirs.dicom_rtstruct / "RS_wrong.dcm", generate_uid(), wrong_for_uid)

    (course / "metadata").mkdir(parents=True, exist_ok=True)
    (course / "metadata" / "case_metadata.json").write_text(
        json.dumps({"patient_id": "P1", "ct_study_uid": "STUDY1"}),
        encoding="utf-8",
    )
    source_dir = project / "data_bucket" / "dicom" / "P1" / "STUDY1"
    source_rs = _mk_rtstruct(source_dir / f"RTSTRUCT_{rtstruct_uid}.dcm", rtstruct_uid, plan_for_uid)

    resolved = _resolve_dvh_structures(
        course,
        dirs,
        plan,
        pydicom.dcmread(str(dose)),
        course_rs,
        course / "missing_auto.dcm",
        None,
    )

    assert resolved.ok
    assert resolved.classification == "plan_referenced_rtstruct"
    assert resolved.sources[0].source_label == "PlanRTSTRUCT"
    assert resolved.sources[0].sop_instance_uid == rtstruct_uid
    assert resolved.sources[0].path == course / "RS_plan.dcm"
    assert (course / "RS_plan.dcm").exists()
    assert source_rs.exists()
