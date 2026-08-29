from pathlib import Path
import json

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, RTPlanStorage, RTStructureSetStorage, generate_uid

from rtpipeline.dvh import _resolve_dvh_dose, _resolve_dvh_structures
from rtpipeline.layout import build_course_dirs
from rtpipeline.organize import _classify_doses
from rtpipeline.course_contract import (
    CourseContractError,
    build_dvh_decision,
    load_course_contract,
)


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


def _write_contract(
    course: Path,
    *,
    selected_plans: list[Path],
    selected_doses: list[Path],
    candidate_plans: list[Path] | None = None,
    rtstruct: Path | None = None,
    dose_classification: str = "single_dose",
) -> None:
    root = course

    def rel(path: Path | None) -> str:
        return str(path.resolve().relative_to(root.resolve())) if path is not None else ""

    plan_entries = []
    selected_plan_uids = []
    for path in selected_plans:
        uid = str(pydicom.dcmread(str(path), stop_before_pixels=True).SOPInstanceUID)
        selected_plan_uids.append(uid)
        plan_entries.append(
            {
                "sop_instance_uid": uid,
                "path": rel(path),
                "source_plan_uids": [uid],
                "delivered_record_count": 0,
                "delivered_fraction_count": 0,
                "treatment_dates": [],
            }
        )
    dose_entries = []
    for path in selected_doses:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        refs = [
            str(item.ReferencedSOPInstanceUID)
            for item in getattr(ds, "ReferencedRTPlanSequence", []) or []
        ]
        dose_entries.append(
            {
                "sop_instance_uid": str(ds.SOPInstanceUID),
                "path": rel(path),
                "dose_summation_type": str(ds.DoseSummationType),
                "referenced_plan_uids": refs,
            }
        )
    per_plan = []
    for path in candidate_plans or selected_plans:
        uid = str(pydicom.dcmread(str(path), stop_before_pixels=True).SOPInstanceUID)
        per_plan.append(
            {
                "plan_path": rel(path),
                "plan_sop_uid": uid,
                "prescribed_dose_gy": 70.0,
                "planned_fraction_count": None,
                "delivered_record_count": 0,
                "delivered_fraction_count": 0,
                "treatment_dates": [],
                "record_paths": [],
                "zero_delivery_records": True,
                "selected_for_dose_grid": uid in selected_plan_uids,
                "status": "no_records",
            }
        )
    rt_uid = ""
    if rtstruct is not None:
        rt_uid = str(pydicom.dcmread(str(rtstruct), stop_before_pixels=True).SOPInstanceUID)
    payload = {
        "patient_id": course.parent.name,
        "course_id": course.name,
        "course_contract": {
            "version": 1,
            "authority": "organize",
            "patient_id": course.parent.name,
            "course_id": course.name,
            "selected_plans": plan_entries,
            "selected_doses": dose_entries,
            "authoritative_rtstruct": (
                {"sop_instance_uid": rt_uid, "path": rel(rtstruct)}
                if rtstruct is not None
                else None
            ),
            "planning_ct": {
                "status": "missing_reference",
                "series_instance_uid": "",
                "referenced_series_uids": [],
                "dicom_dir": "",
                "nifti_path": "",
            },
            "delivery": {
                "prescribed_dose_gy": 70.0 if selected_plans else None,
                "status": "no_records_at_all",
                "method": None,
                "delivered_dose_gy": None,
                "dose_response_field": "delivered_dose_gy",
                "delivered_record_count": 0,
                "delivered_fraction_count": 0,
                "planned_fraction_count": None,
                "unresolved_record_plan_uids": [],
                "unresolved_record_count": 0,
                "unresolved_reference_count": 0,
                "selected_plan_uids": selected_plan_uids,
                "per_plan": per_plan,
            },
            "dose_classification": {"classification": dose_classification},
            "dvh": build_dvh_decision(
                len(plan_entries),
                len(dose_entries),
                "no_records_at_all",
            ),
            "plan_artifact": (
                plan_entries[0] if selected_plans else None
            ),
            "dose_grid": (
                {
                    **dose_entries[0],
                    "semantics": "planned_dose_for_selected_plan_set_delivery_unknown",
                    "source_plan_uids": selected_plan_uids,
                    "source_dose_uids": [item["sop_instance_uid"] for item in dose_entries],
                    "source_dose_summation_types": [item["dose_summation_type"] for item in dose_entries],
                }
                if selected_doses
                else None
            ),
            "dose_qc": {"status": "pass", "pass": True, "threshold_gy": 100.0, "reasons": []},
        },
    }
    metadata = course / "metadata" / "case_metadata.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_dvh_resolver_uses_plan_dose_not_first_beam(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    plan_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "RP.dcm", plan_uid)
    beam = _mk_dose(dirs.dicom_rtdose / "000_BEAM.dcm", generate_uid(), plan_uid, "BEAM")
    plan_dose = _mk_dose(dirs.dicom_rtdose / "999_PLAN.dcm", generate_uid(), plan_uid, "PLAN")
    _write_contract(course, selected_plans=[plan], selected_doses=[plan_dose])

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
    plan = _mk_plan(dirs.dicom_rtplan / "RP.dcm", plan_uid)
    _mk_dose(dirs.dicom_rtdose / "000_BEAM.dcm", generate_uid(), plan_uid, "BEAM")
    _write_contract(
        course,
        selected_plans=[],
        selected_doses=[],
        candidate_plans=[plan],
        dose_classification="single_beam_dose_rejected",
    )

    resolved = _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")

    assert not resolved.ok
    assert resolved.classification == "single_beam_dose_rejected"
    assert "no authoritative treatment dose grid" in resolved.reason


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


def test_structure_resolver_uses_contracted_rtstruct_not_directory_superset(tmp_path):
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
    _mk_rtstruct(dirs.dicom_rtstruct / "RS_wrong.dcm", generate_uid(), wrong_for_uid)
    course_rs = _mk_rtstruct(dirs.dicom_rtstruct / "RS_selected.dcm", rtstruct_uid, plan_for_uid)
    _write_contract(
        course,
        selected_plans=[plan],
        selected_doses=[dose],
        rtstruct=course_rs,
    )

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
    assert resolved.classification == "contract_rtstruct"
    assert resolved.sources[0].source_label == "Manual"
    assert resolved.sources[0].sop_instance_uid == rtstruct_uid
    assert resolved.sources[0].path == course_rs


def test_dvh_contract_ignores_extra_plan_and_dose_files(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    selected_uid = generate_uid()
    extra_uid = generate_uid()
    selected_plan = _mk_plan(dirs.dicom_rtplan / "selected.dcm", selected_uid, rx=50.0)
    extra_plan = _mk_plan(dirs.dicom_rtplan / "extra.dcm", extra_uid, rx=55.0)
    selected_dose = _mk_dose(dirs.dicom_rtdose / "selected.dcm", generate_uid(), selected_uid, "PLAN")
    _mk_dose(dirs.dicom_rtdose / "extra.dcm", generate_uid(), extra_uid, "PLAN")
    _write_contract(
        course,
        selected_plans=[selected_plan],
        selected_doses=[selected_dose],
        candidate_plans=[selected_plan, extra_plan],
    )

    resolved = _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")

    assert resolved.ok
    assert resolved.rp_path == selected_plan
    assert resolved.rd_path == selected_dose
    assert resolved.source_plan_sop_instance_uids == [selected_uid]


def test_missing_or_mismatched_course_contract_fails_closed(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    with pytest.raises(CourseContractError, match="missing"):
        _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")

    plan_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "plan.dcm", plan_uid)
    dose = _mk_dose(dirs.dicom_rtdose / "dose.dcm", generate_uid(), plan_uid, "PLAN")
    _write_contract(course, selected_plans=[plan], selected_doses=[dose])
    metadata_path = course / "metadata" / "case_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["course_contract"]["selected_plans"][0]["sop_instance_uid"] = generate_uid()
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CourseContractError, match="does not match"):
        load_course_contract(course)


def test_stale_dose_to_plan_reference_fails_closed(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    plan_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "plan.dcm", plan_uid)
    dose = _mk_dose(
        dirs.dicom_rtdose / "dose.dcm",
        generate_uid(),
        plan_uid,
        "PLAN",
    )
    _write_contract(course, selected_plans=[plan], selected_doses=[dose])
    dataset = pydicom.dcmread(str(dose))
    dataset.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = generate_uid()
    dataset.save_as(str(dose), write_like_original=False)

    with pytest.raises(CourseContractError, match="referenced RTPLAN UIDs"):
        _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")


def test_dose_qc_failure_and_semantics_are_contract_fields(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    plan_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "plan.dcm", plan_uid, rx=105.0)
    dose = _mk_dose(dirs.dicom_rtdose / "dose.dcm", generate_uid(), plan_uid, "PLAN")
    _write_contract(course, selected_plans=[plan], selected_doses=[dose])
    metadata_path = course / "metadata" / "case_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = payload["course_contract"]
    contract["delivery"]["prescribed_dose_gy"] = 105.0
    contract["dose_qc"] = {
        "status": "fail",
        "pass": False,
        "threshold_gy": 100.0,
        "reasons": ["prescribed dose 105 Gy exceeds configured maximum 100 Gy"],
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")

    assert resolved.dose_qc_pass is False
    assert resolved.dose_qc_status == "fail"
    assert resolved.dose_grid_semantics == "planned_dose_for_selected_plan_set_delivery_unknown"
    assert resolved.dose_qc_reasons is not None
    assert "105 Gy" in resolved.dose_qc_reasons[0]
    with pytest.raises(CourseContractError, match="threshold"):
        _resolve_dvh_dose(
            course,
            dirs,
            course / "missing_RS.dcm",
            max_total_dose_gy=90.0,
        )


def test_implausible_total_with_passing_qc_contract_fails_closed(tmp_path):
    course = tmp_path / "P1" / "2024-01"
    dirs = build_course_dirs(course)
    dirs.ensure()
    plan_uid = generate_uid()
    plan = _mk_plan(dirs.dicom_rtplan / "plan.dcm", plan_uid, rx=105.0)
    dose = _mk_dose(
        dirs.dicom_rtdose / "dose.dcm",
        generate_uid(),
        plan_uid,
        "PLAN",
    )
    _write_contract(course, selected_plans=[plan], selected_doses=[dose])
    metadata_path = course / "metadata" / "case_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["course_contract"]["delivery"]["prescribed_dose_gy"] = 105.0
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CourseContractError, match="dose_qc verdict disagrees"):
        _resolve_dvh_dose(course, dirs, course / "missing_RS.dcm")
