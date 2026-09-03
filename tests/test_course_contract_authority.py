from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import RTDoseStorage, UID

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
    write_synthetic_planning_ct,
)
from rtpipeline import cli as cli_module
from rtpipeline.course_contract import (
    CourseContractError,
    build_dvh_decision,
    load_course_contract,
)
from rtpipeline.dvh import _resolve_dvh_dose, dvh_for_course
from rtpipeline.layout import build_course_dirs


def _metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_metadata(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_contract_rejects_role_swap_with_preserved_sop_uid(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    metadata = write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )

    dataset = pydicom.dcmread(str(plan))
    dataset.Modality = "RTDOSE"
    dataset.SOPClassUID = RTDoseStorage
    dataset.save_as(str(plan), enforce_file_format=True)

    with pytest.raises(CourseContractError, match="expected RTPLAN"):
        load_course_contract(course)

    assert metadata.is_file()


def test_contract_rejects_stale_beamdose_resolution(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    write_minimal_course_contract(
        course,
        selected_plans=[plan],
        selected_doses=[dose],
    )
    assert load_course_contract(course).resolved_prescribed_dose_total_gy == 50.0

    dataset = pydicom.dcmread(str(plan))
    dataset.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose = 5.0
    dataset.save_as(str(plan), enforce_file_format=True)

    with pytest.raises(CourseContractError, match="prescription_groups"):
        load_course_contract(course)


def test_contract_round_trips_group_scoped_prescription(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(
        course,
        prescribed_dose_gy=30.0,
        planned_fraction_count=2,
    )
    dataset = pydicom.dcmread(str(plan))
    dataset.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose = 3.0
    group_reference = Dataset()
    group_reference.ReferencedDoseReferenceNumber = 1
    group_reference.TargetPrescriptionDose = 6.0
    dataset.FractionGroupSequence[0].ReferencedDoseReferenceSequence = Sequence(
        [group_reference]
    )
    dataset.save_as(str(plan), enforce_file_format=True)
    write_minimal_course_contract(
        course,
        selected_plans=[plan],
        selected_doses=[dose],
    )

    contract = load_course_contract(course)

    assert contract.prescribed_dose_gy == 6.0
    assert contract.resolved_prescribed_dose_total_gy == 6.0
    assert contract.selected_plans[0]["source_prescribed_dose_tag_path"].startswith(
        "FractionGroupSequence[0]"
    )


def test_contract_requires_auditable_rtrecord_for_nonzero_delivery(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    metadata_path = write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )
    payload = _metadata(metadata_path)
    contract = payload["course_contract"]
    contract["delivery"].update(
        {
            "status": "fully_delivered",
            "prescribed_dose_gy": 50.0,
            "delivered_dose_gy": 2.5,
        }
    )
    entry = contract["delivery"]["per_plan"][0]
    entry.update(
        {
            "delivered_record_count": 1,
            "delivered_fraction_count": 1,
            "treatment_dates": ["20240101"],
            "record_paths": [],
            "zero_delivery_records": False,
        }
    )
    contract["dvh"] = build_dvh_decision(1, 1, "fully_delivered")
    _save_metadata(metadata_path, payload)

    with pytest.raises(CourseContractError, match="delivery.*RTRECORD evidence"):
        load_course_contract(course)


def test_derived_artifacts_are_bound_to_selected_membership(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    other_plan = course / "DICOM" / "RTPLAN" / "other_plan.dcm"
    other_plan.write_bytes(plan.read_bytes())
    other_dataset = pydicom.dcmread(str(other_plan))
    other_dataset.SOPInstanceUID = "2.25.999999999999999999999999999999999999"
    other_dataset.file_meta.MediaStorageSOPInstanceUID = other_dataset.SOPInstanceUID
    other_dataset.save_as(str(other_plan), enforce_file_format=True)
    metadata_path = write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )
    payload = _metadata(metadata_path)
    artifact = payload["course_contract"]["plan_artifact"]
    artifact["path"] = str(other_plan.relative_to(course))
    artifact["sop_instance_uid"] = str(other_dataset.SOPInstanceUID)
    _save_metadata(metadata_path, payload)

    with pytest.raises(CourseContractError, match="source_plan_uids"):
        load_course_contract(course)


def test_copied_plan_artifact_can_represent_one_member_of_nonadditive_plan_set(
    tmp_path: Path,
) -> None:
    course = tmp_path / "P1" / "replacement"
    plan, _ = write_synthetic_plan_and_dose(course)
    second_plan = course / "DICOM" / "RTPLAN" / "replacement_plan.dcm"
    second_plan.write_bytes(plan.read_bytes())
    second_dataset = pydicom.dcmread(str(second_plan))
    second_dataset.SOPInstanceUID = UID("2.25.888888888888888888888888888888888888")
    second_dataset.file_meta.MediaStorageSOPInstanceUID = second_dataset.SOPInstanceUID
    second_dataset.save_as(str(second_plan), enforce_file_format=True)

    metadata_path = write_minimal_course_contract(
        course,
        selected_plans=[plan, second_plan],
        selected_doses=[],
    )
    contract = load_course_contract(course)
    first_uid = str(pydicom.dcmread(plan, stop_before_pixels=True).SOPInstanceUID)
    second_uid = str(second_dataset.SOPInstanceUID)

    plan_artifact = contract.data["plan_artifact"]
    assert len(contract.selected_plans) == 2
    assert plan_artifact["sop_instance_uid"] == first_uid
    assert plan_artifact["source_plan_uids"] == [first_uid]

    payload = _metadata(metadata_path)
    payload["course_contract"]["plan_artifact"]["source_plan_uids"] = [
        first_uid,
        second_uid,
    ]
    _save_metadata(metadata_path, payload)
    with pytest.raises(CourseContractError, match="copied source plan_artifact"):
        load_course_contract(course)


def test_contract_rejects_stale_planning_nifti_sidecar(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    ct_dir = write_synthetic_planning_ct(course)
    metadata_path = write_minimal_course_contract(course, planning_ct_dir=ct_dir)
    payload = _metadata(metadata_path)
    sidecar_value = payload["course_contract"]["planning_ct"]["nifti_provenance"][
        "sidecar_path"
    ]
    sidecar = course / sidecar_value
    sidecar_data = _metadata(sidecar)
    sidecar_data["series_instance_uid"] = "2.25.123456789"
    _save_metadata(sidecar, sidecar_data)

    with pytest.raises(CourseContractError, match="sidecar.*does not match"):
        load_course_contract(course)


def test_dvh_skips_a_valid_contract_without_a_dose_grid(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    write_minimal_course_contract(course, selected_plans=[], selected_doses=[])
    stale = course / "dvh_metrics.xlsx"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale output")

    assert dvh_for_course(course) is None
    assert not stale.exists()
    qc_path = course / "metadata" / "dvh_qc.json"
    qc = _metadata(qc_path)
    assert qc["dose_resolution"]["classification"] == "no_doses"
    assert qc["course_contract_sha256"]


def test_plan_only_course_has_plan_membership_and_explicit_no_metrics_outcome(
    tmp_path: Path,
) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    dose.unlink()
    write_minimal_course_contract(course, selected_plans=[plan], selected_doses=[])

    contract = load_course_contract(course)
    plan_artifact, dose_grid = contract.require_dvh_artifacts()
    assert plan_artifact == plan.resolve()
    assert dose_grid is None

    stale = course / "dvh_metrics.xlsx"
    stale.write_bytes(b"stale output")
    assert dvh_for_course(course) is None
    assert not stale.exists()

    qc = _metadata(course / "metadata" / "dvh_qc.json")
    resolution = qc["dose_resolution"]
    assert resolution["source_plan_sop_instance_uids"] == [
        str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    ]
    assert resolution["selected_plan_paths"] == [str(plan.resolve())]
    assert resolution["selected_dose_paths"] == []
    assert resolution["dvh"]["reason_code"] == "plan_only_no_authoritative_dose_grid"
    assert resolution["dvh"]["metrics_status"] == "not_computed"
    assert resolution["dvh"]["dose_record_status"] == "delivery_unknown_no_rtrecord"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "computed", "wrong_reason", "missing_field"],
)
def test_contract_rejects_noncanonical_plan_only_dvh_decision(
    tmp_path: Path, mutation: str
) -> None:
    course = tmp_path / mutation / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    dose.unlink()
    metadata_path = write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[]
    )
    payload = _metadata(metadata_path)
    contract = payload["course_contract"]
    if mutation == "missing":
        contract.pop("dvh")
    elif mutation == "computed":
        contract["dvh"]["status"] = "ready"
        contract["dvh"]["metrics_status"] = "computed"
        contract["dvh"]["output"] = "dvh_metrics.xlsx"
    elif mutation == "wrong_reason":
        contract["dvh"]["reason_code"] = "authoritative_dose_grid"
    else:
        contract["dvh"].pop("dose_record_status")
    _save_metadata(metadata_path, payload)

    with pytest.raises(CourseContractError, match="field dvh"):
        load_course_contract(course)


def test_dvh_invalidates_stale_output_when_contract_dose_qc_fails(
    tmp_path: Path, monkeypatch
) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(
        course, prescribed_dose_gy=105.0
    )
    metadata_path = write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )
    payload = _metadata(metadata_path)
    contract = payload["course_contract"]
    contract["delivery"].update(
        {
            "status": "delivered_but_records_absent",
            "delivered_dose_gy": None,
        }
    )
    contract["dvh"] = build_dvh_decision(
        1,
        1,
        "delivered_but_records_absent",
        dose_response_eligible=False,
        dose_completeness=contract["dose_completeness"],
    )
    contract["dose_qc"] = {
        "status": "fail",
        "pass": False,
        "threshold_gy": 100.0,
        "reasons": ["resolved prescribed dose 105.0 Gy exceeds 100.0 Gy"],
    }
    _save_metadata(metadata_path, payload)
    stale = course / "dvh_metrics.xlsx"
    stale.write_bytes(b"stale output")

    assert dvh_for_course(course) is None
    assert not stale.exists()
    qc = _metadata(course / "metadata" / "dvh_qc.json")
    assert qc["dose_resolution"]["dose_qc_pass"] is False

    course_output = SimpleNamespace(
        dirs=build_course_dirs(course), total_prescription_gy=105.0
    )
    monkeypatch.setattr(
        cli_module, "organize_and_merge", lambda _cfg: [course_output]
    )
    monkeypatch.setattr(
        cli_module,
        "run_tasks_with_adaptive_workers",
        lambda _label, tasks, function, **_kwargs: [
            function(task) for task in tasks
        ],
    )
    exit_status = cli_module.main(
        [
            "--dicom-root",
            str(tmp_path / "dicom"),
            "--outdir",
            str(tmp_path / "output"),
            "--logs",
            str(tmp_path / "logs"),
            "--stage",
            "dvh",
            "--no-metadata",
        ]
    )
    assert exit_status == 1


def test_dvh_uses_only_contract_artifacts_when_directory_has_extra_objects(
    tmp_path: Path,
) -> None:
    course = tmp_path / "P1" / "C1"
    selected_plan, selected_dose = write_synthetic_plan_and_dose(course)
    extra_plan = course / "DICOM" / "RTPLAN" / "extra_plan.dcm"
    extra_plan.write_bytes(selected_plan.read_bytes())
    extra_dataset = pydicom.dcmread(str(extra_plan))
    extra_dataset.SOPInstanceUID = "2.25.888888888888888888888888888888888888"
    extra_dataset.file_meta.MediaStorageSOPInstanceUID = extra_dataset.SOPInstanceUID
    extra_dataset.save_as(str(extra_plan), enforce_file_format=True)
    write_minimal_course_contract(
        course, selected_plans=[selected_plan], selected_doses=[selected_dose]
    )

    resolution = _resolve_dvh_dose(
        course, build_course_dirs(course), course / "missing-rtstruct.dcm"
    )
    assert resolution.ok
    assert resolution.selected_plan_paths == [selected_plan.resolve()]
    assert resolution.source_plan_sop_instance_uids == [
        str(pydicom.dcmread(str(selected_plan), stop_before_pixels=True).SOPInstanceUID)
    ]
    assert str(extra_dataset.SOPInstanceUID) not in resolution.source_plan_sop_instance_uids
