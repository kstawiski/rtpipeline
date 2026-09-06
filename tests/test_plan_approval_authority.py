"""Synthetic clinical-authority regressions. No patient evidence is embedded."""
from dataclasses import asdict
import json
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from rtpipeline import organize as org
from rtpipeline.config import PipelineConfig
from rtpipeline.course_contract import CourseContractError, load_course_contract
from rtpipeline.plan_approval import approval_audit, approved_course_start
from test_course_identity_reference_chains import _mk_plan, _mk_dose, _mk_struct, _extract_linked


def sources(tmp_path, statuses):
    root = tmp_path / "input"
    study, frame, struct_uid = [generate_uid() for _ in range(3)]
    _mk_struct(root / "struct.dcm", struct_uid, study_uid=study, frame_uid=frame,
               roi_names=["PTV1"])
    plans, doses = [], []
    for i, status in enumerate(statuses):
        uid = generate_uid()
        p = _mk_plan(root / f"plan{i}.dcm", uid, struct_uid=struct_uid,
                     study_uid=study, frame_uid=frame, date=f"20240{i+1}01",
                     rx_gy=60.0, fractions=30, label="synthetic")
        ds = pydicom.dcmread(p)
        if status is not None:
            ds.ApprovalStatus = status
        elif "ApprovalStatus" in ds:
            del ds.ApprovalStatus
        ds.save_as(p, enforce_file_format=True)
        plans.append(p)
        doses.append(_mk_dose(root / f"dose{i}.dcm", generate_uid(), plan_uid=uid,
                             study_uid=study, frame_uid=frame, with_pixels=True))
    return root, plans, doses


@pytest.mark.parametrize("status,disposition", [
    ("REJECTED", "no_approved_plan_rejected_only"),
    ("UNAPPROVED", "no_approved_plan_unapproved_only"),
    (None, "no_approved_plan_approval_status_absent"),
    ("", "no_approved_plan_approval_status_absent"),
    ("UNKNOWN", "no_approved_plan_approval_status_unknown"),
])
def test_nonapproved_only_is_not_authoritative(tmp_path, status, disposition):
    root, plans, doses = sources(tmp_path, [status])
    before = {p: p.read_bytes() for p in plans + doses}
    result = org._classify_doses(plans, doses)
    assert result.classification == disposition
    assert result.selected_plans == result.selected_doses == []
    assert org._infer_rx_from_plan_paths(plans) is None
    assert org._infer_source_rx_from_plan_paths(plans) is None
    assert approved_course_start(_extract_linked(root)) is None
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(ValueError, match="APPROVED"):
        org._create_summed_plan(plans)
    with pytest.raises(ValueError, match="APPROVED"):
        org._sum_doses_with_resample(doses, Dataset(), [pydicom.dcmread(plans[0])])


@pytest.mark.parametrize("status", ["REJECTED", "UNAPPROVED", None])
def test_approved_sibling_controls_plan_dose_prescription_and_identity(tmp_path, status):
    root, plans, doses = sources(tmp_path, [status, "APPROVED", status])
    result = org._classify_doses(plans, doses)
    assert result.selected_plans == [plans[1]]
    assert result.selected_doses == [doses[1]]
    assert org._infer_rx_from_plan_paths(result.selected_plans) == 60
    assert approved_course_start(_extract_linked(root)).strftime("%Y%m%d") == "20240201"


@pytest.mark.parametrize("kind", ["PLAN_SUM", "MULTI_PLAN", "BEAM", "PLAN"])
def test_composite_grid_with_rejected_component_is_not_salvaged(tmp_path, kind):
    _, plans, doses = sources(tmp_path, ["APPROVED", "REJECTED"])
    ds = pydicom.dcmread(doses[0]); ds.DoseSummationType = kind
    rejected_ref = Dataset()
    rejected_ref.ReferencedSOPInstanceUID = pydicom.dcmread(plans[1]).SOPInstanceUID
    ds.ReferencedRTPlanSequence.append(rejected_ref)
    ds.save_as(doses[0], enforce_file_format=True)
    result = org._classify_doses(plans, doses)
    assert result.selected_doses == []
    assert result.selected_plans == [plans[0]]


def organize(root, tmp_path, monkeypatch):
    monkeypatch.setattr(org, "_index_series_and_registrations", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(org, "_looks_like_patient_series_layout", lambda *a, **k: False)
    config = PipelineConfig(root, tmp_path / "output", tmp_path / "logs", max_workers_override=1,
                            dicom_copy_dedup_by_sop_uid=False)
    output = org.organize_and_merge(config)
    assert len(output) == 1
    return output[0], load_course_contract(output[0].dirs.root)


@pytest.mark.parametrize("status,disposition", [
    ("REJECTED", "no_approved_plan_rejected_only"),
    ("UNAPPROVED", "no_approved_plan_unapproved_only"),
    (None, "no_approved_plan_approval_status_absent"),
])
def test_real_organizer_preserves_context_and_publishes_explicit_hold(tmp_path, monkeypatch, status, disposition):
    root, plans, doses = sources(tmp_path, [status])
    before = plans[0].read_bytes()
    output, contract = organize(root, tmp_path, monkeypatch)
    assert contract.selected_plans == contract.selected_doses == []
    assert contract.data["dose_classification"]["plan_approval"]["disposition"] == disposition
    assert contract.data["delivery"]["prescribed_dose_gy"] is None
    assert contract.data["delivery"]["resolved_prescribed_dose_total_gy"] is None
    assert contract.data["delivery"]["delivered_dose_gy"] is None
    assert contract.data["delivery"]["dose_response_eligible"] is False
    assert not output.rp_path.exists() and not output.rd_path.exists()
    context = contract.data["delivery"]["per_plan"]
    assert len(context) == 1 and context[0]["selected_for_dose_grid"] is False
    path = contract.resolve_path(context[0]["plan_path"], "plan_path")
    assert path.read_bytes() == before
    assert contract.data["dose_classification"]["plan_approval"]["plans"][0]["approval_status"] == (status or "ABSENT")


def test_real_organizer_uses_approved_sibling_and_rejects_stale_status(tmp_path, monkeypatch):
    root, plans, doses = sources(tmp_path, ["REJECTED", "APPROVED"])
    output, contract = organize(root, tmp_path, monkeypatch)
    assert output.course_id == "2024-02"
    assert output.rp_path.read_bytes() == plans[1].read_bytes()
    assert output.rd_path.read_bytes() == doses[1].read_bytes()
    assert len(contract.data["delivery"]["per_plan"]) == 2
    assert contract.data["delivery"]["resolved_prescribed_dose_total_gy"] == 60
    selected = contract.resolve_path(contract.selected_plans[0]["path"], "path")
    ds = pydicom.dcmread(selected); ds.ApprovalStatus = "REJECTED"; ds.save_as(selected, enforce_file_format=True)
    with pytest.raises(CourseContractError, match="APPROVED"):
        load_course_contract(output.dirs.root)


def test_resume_compares_original_source_approval_not_only_archived_copy(tmp_path, monkeypatch):
    root, plans, doses = sources(tmp_path, ["APPROVED", "APPROVED"])
    output, contract = organize(root, tmp_path, monkeypatch)
    assert contract.selected_plans[0]["sop_instance_uid"] == str(pydicom.dcmread(plans[1]).SOPInstanceUID)
    ds = pydicom.dcmread(plans[1])
    ds.ApprovalStatus = "REJECTED"
    ds.save_as(plans[1], enforce_file_format=True)
    monkeypatch.setattr(org, "_hydrate_existing_course", lambda *a, **k: output)
    config = PipelineConfig(root, tmp_path / "output", tmp_path / "logs", max_workers_override=1,
                            dicom_copy_dedup_by_sop_uid=False, resume=True)
    outputs = org.organize_and_merge(config)
    assert len(outputs) == 1
    current = load_course_contract(outputs[0].dirs.root)
    assert current.selected_plans[0]["sop_instance_uid"] == str(pydicom.dcmread(plans[0]).SOPInstanceUID)
    assert current.data["dose_classification"]["plan_approval"]["plans"][1]["approval_status"] == "REJECTED"


def test_all_approved_classification_is_byte_identical(tmp_path):
    root, plans, doses = sources(tmp_path, ["APPROVED", "APPROVED"])
    old = org._classify_approved_doses(plans, doses)
    new = org._classify_doses(plans, doses)
    serialize = lambda x: json.dumps(asdict(x), default=str, sort_keys=True).encode()
    assert serialize(old) == serialize(new)
    assert approved_course_start(_extract_linked(root)).strftime("%Y%m%d") == "20240101"
