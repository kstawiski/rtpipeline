"""Synthetic source-disposition regressions. No real patient data."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from rtpipeline import organize as org
from rtpipeline.config import PipelineConfig
from rtpipeline.metadata import link_rt_sets, group_by_course
from rtpipeline.rt_details import extract_rt_with_records, target_volume_names
from rtpipeline.plan_disposition import (
    RELATIVE_PATH, build_source_plan_dispositions, compare_plan_variant,
    validate_source_plan_dispositions, write_source_plan_dispositions,
)
from rtpipeline.organize_ledger import read_organize_ledger, OrganizeLedgerError, validate_organize_ledger
from test_course_identity_reference_chains import _mk_plan, _mk_record, _mk_struct


def plan(root, *, structured=False, status="APPROVED", intent="", label="synthetic"):
    study, frame, struct_uid, uid = [generate_uid() for _ in range(4)]
    path = root / (uid + ".dcm")
    _mk_plan(path, uid, struct_uid=struct_uid, study_uid=study, frame_uid=frame,
             date="20240101", rx_gy=60, fractions=30, label=label)
    ds = pydicom.dcmread(path)
    ds.ApprovalStatus = status
    ds.PlanIntent = intent
    ds.RTPlanGeometry = "PATIENT" if structured else "TREATMENT_DEVICE"
    if not structured:
        del ds.ReferencedStructureSetSequence
    else:
        _mk_struct(root / (struct_uid + ".dcm"), struct_uid, study_uid=study,
                   frame_uid=frame, roi_names=["PTV1"])
    b = ds.BeamSequence[0]
    b.BeamName = "field"
    b.TreatmentMachineName = "SYNTHETIC_MACHINE"
    b.BeamType = "DYNAMIC"
    b.RadiationType = "PHOTON"
    b.FinalCumulativeMetersetWeight = 1
    cps = []
    for i in range(2):
        cp = Dataset()
        cp.ControlPointIndex = i
        cp.CumulativeMetersetWeight = i
        cp.GantryAngle = 0
        cp.BeamLimitingDeviceAngle = 0
        cp.PatientSupportAngle = 0
        cp.NominalBeamEnergy = 6
        jaw = Dataset()
        jaw.RTBeamLimitingDeviceType = "X"
        jaw.LeafJawPositions = [-10, 10]
        cp.BeamLimitingDevicePositionSequence = Sequence([jaw])
        cps.append(cp)
    b.ControlPointSequence = Sequence(cps)
    b.NumberOfControlPoints = 2
    ds.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset = 100
    ds.save_as(path, enforce_file_format=True)
    return path


def inventory(root):
    plans, doses, structs, records = extract_rt_with_records(root)
    groups = group_by_course(link_rt_sets(plans, doses, structs))
    accepted = {k: items for k, items in groups.items()
                if any(x.struct and target_volume_names(x.struct.roi_names) for x in items)}
    return plans, records, accepted


def run(root, out, *, resume=False):
    config = PipelineConfig(root, out, out.parent / "logs", max_workers_override=1,
                            dicom_copy_use_hardlinks=False, dicom_copy_dedup_by_sop_uid=False,
                            resume=resume)
    courses = org.organize_and_merge(config)
    payload = json.loads((out / RELATIVE_PATH).read_text())
    assert read_organize_ledger(out)["source_plan_dispositions"] == payload
    return courses, payload


def add_record(root, path, *, date="20240102", extra_plan=None):
    uid = str(pydicom.dcmread(path).SOPInstanceUID)
    record = _mk_record(root / (generate_uid() + ".dcm"), uid, date=date)
    ds = pydicom.dcmread(record)
    ds.TreatmentSessionBeamSequence[0].DeliveredPrimaryMeterset = 100
    if extra_plan:
        ref = copy.deepcopy(ds.ReferencedRTPlanSequence[0])
        ref.ReferencedSOPInstanceUID = pydicom.dcmread(extra_plan).SOPInstanceUID
        ds.ReferencedRTPlanSequence.append(ref)
    ds.save_as(record, enforce_file_format=True)
    return record


@pytest.mark.parametrize("status", ["APPROVED", "UNAPPROVED", "REJECTED", ""])
def test_real_organizer_retains_delivered_structureless_plan(tmp_path, status):
    root, out = tmp_path / "input", tmp_path / "output"
    p = plan(root, status=status)
    add_record(root, p)
    before = p.read_bytes()
    courses, result = run(root, out)
    assert courses == []
    row, = result["plans"]
    assert row["disposition_type"] == "non_measurable_delivery"
    assert row["reason_code"] == "DELIVERED_PLAN_WITHOUT_IMAGING_AUTHORITY"
    assert row["validated_record_count"] == row["validated_session_count"] == 1
    assert row["target_prescriptions"][0]["prescription_gy"] == 60
    assert row["planned_fractions"] == [30]
    assert row["clinical_exclusion"] is False
    assert row["additional_clinical_course_count"] is None
    assert row["structure_set_path"] is row["dose_grid_path"] is row["delivered_dose_gy"] is None
    assert len(row["non_measurements"]) == 5
    assert not list(out.rglob("RS.dcm")) and not list(out.rglob("RD.dcm"))
    assert p.read_bytes() == before


def variant_pair(root):
    authoritative = plan(root, structured=True)
    ds = pydicom.dcmread(authoritative)
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    del ds.ReferencedStructureSetSequence
    ds.RTPlanGeometry = "TREATMENT_DEVICE"
    ds.BeamSequence[0].BeamNumber = 17
    ds.BeamSequence[0].BeamName = "renumbered"
    ds.FractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber = 17
    variant = root / (str(ds.SOPInstanceUID) + ".dcm")
    ds.save_as(variant, enforce_file_format=True)
    return authoritative, variant


def test_genuine_variant_recorded_with_authority_and_not_double_counted(tmp_path):
    root = tmp_path / "input"
    authoritative, variant = variant_pair(root)
    add_record(root, variant, extra_plan=authoritative)
    courses, result = run(root, tmp_path / "out")
    assert len(courses) == 1
    row = next(r for r in result["plans"] if r["plan_uid"] == str(pydicom.dcmread(variant).SOPInstanceUID))
    assert row["disposition_type"] == "recorded_beam_variant"
    assert row["authoritative_plan_uid"] == str(pydicom.dcmread(authoritative).SOPInstanceUID)
    assert row["additional_clinical_course_count"] == 0
    assert row["dose_grid_path"] is row["structure_set_path"] is None
    assert len(result["plans"]) == 2
    assert result["disposition_counts"] == {"course_member": 1, "recorded_beam_variant": 1}


@pytest.mark.parametrize("change", ["machine", "jaw", "energy", "meterset", "dose_reference", "new_record"])
def test_similar_plan_cannot_authorize_unproven_delivery_deduplication(tmp_path, change):
    root = tmp_path / "input"
    authoritative, variant = variant_pair(root)
    ds = pydicom.dcmread(variant)
    if change == "machine":
        ds.BeamSequence[0].TreatmentMachineName = "OTHER"
    elif change == "jaw":
        ds.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions = [-9.5, 10]
    elif change == "energy":
        ds.BeamSequence[0].ControlPointSequence[0].NominalBeamEnergy = 10
    elif change == "meterset":
        ds.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset = 101
    elif change == "dose_reference":
        ds.DoseReferenceSequence[0].DoseReferenceUID = generate_uid()
    ds.save_as(variant, enforce_file_format=True)
    add_record(root, variant, extra_plan=None if change == "new_record" else authoritative)
    payload = build_source_plan_dispositions(*inventory(root))
    row = next(r for r in payload["plans"] if r["plan_uid"] == str(ds.SOPInstanceUID))
    assert row["disposition_type"] == "non_measurable_delivery"
    assert row["authoritative_plan_uid"] is None
    assert row["validated_record_count"] == 1


@pytest.mark.parametrize("status,intent,reason", [
    ("APPROVED", "VERIFICATION", "VERIFICATION_OR_QA_PLAN"),
    ("UNAPPROVED", "VERIFICATION", "VERIFICATION_OR_QA_PLAN"),
    ("UNAPPROVED", "CURATIVE", "NONAPPROVED_PLAN_WITHOUT_IMAGING_AUTHORITY_OR_VALIDATED_DELIVERY"),
    ("REJECTED", "", "REJECTED_PLAN_WITHOUT_VALIDATED_DELIVERY"),
    ("APPROVED", "", "NO_TARGET_BEARING_AUTHORITATIVE_RTSTRUCT"),
])
def test_every_declined_source_plan_has_typed_reason(tmp_path, status, intent, reason):
    root = tmp_path / "input"
    plan(root, status=status, intent=intent)
    courses, result = run(root, tmp_path / "out")
    assert courses == []
    row, = result["plans"]
    assert row["disposition_type"] == "excluded"
    assert row["reason_code"] == reason
    assert row["mechanical_reason"] == "NO_RTSTRUCT_REFERENCE"


def test_beam_records_are_sessions_not_fractions_and_duplicates_deduplicate(tmp_path):
    root = tmp_path / "input"
    p = plan(root)
    r = add_record(root, p)
    add_record(root, p)
    add_record(root, p, date="20240103")
    plans, records, accepted = inventory(root)
    records["P1"].append(r)
    result = build_source_plan_dispositions(plans, records, accepted)
    row, = result["plans"]
    assert row["validated_record_count"] == 3
    assert row["validated_session_count"] == 2


def test_resume_discovers_new_detached_delivery_even_if_courses_completed(tmp_path, monkeypatch):
    root, out = tmp_path / "input", tmp_path / "out"
    plan(root, structured=True)
    courses, first = run(root, out)
    assert len(courses) == 1
    p = plan(root)
    add_record(root, p)
    # This previously caused the entire patient to disappear from discovery.
    monkeypatch.setattr(org, "_completed_patients", lambda _: {"P1": [{"course_dir": str(courses[0].dirs.root), "course_key": courses[0].course_key}]})
    courses, result = run(root, out, resume=True)
    assert len(courses) == 1
    assert result["source_plan_count"] == 2
    assert result["non_course_delivery_plan_count"] == 1


def test_unpublished_course_is_not_reported_as_a_member(tmp_path, monkeypatch):
    root = tmp_path / "input"
    plan(root, structured=True)
    monkeypatch.setattr(org, "run_tasks_with_adaptive_workers", lambda *a, **k: [])
    courses, result = run(root, tmp_path / "out")
    assert courses == []
    row, = result["plans"]
    assert row["disposition_type"] == "technical_hold"
    assert row["reason_code"] == "COURSE_NOT_PUBLISHED"


@pytest.mark.parametrize("tamper", ["count", "reason", "dose", "stage", "session"])
def test_validator_rejects_forged_disposition(tmp_path, tamper):
    root = tmp_path / "input"
    p = plan(root)
    add_record(root, p)
    payload = build_source_plan_dispositions(*inventory(root))
    row = payload["plans"][0]
    if tamper == "count": payload["source_plan_count"] += 1
    if tamper == "reason": row["reason_code"] = ""
    if tamper == "dose": row["dose_grid_path"] = "borrowed.dcm"
    if tamper == "stage": del row["non_measurements"]["dvh"]
    if tamper == "session": row["validated_session_count"] += 1
    with pytest.raises(ValueError): validate_source_plan_dispositions(payload)


def test_record_parse_failure_remains_a_technical_hold(tmp_path):
    root = tmp_path / "input"
    plan(root)
    plans, records, accepted = inventory(root)
    bad = root / "bad_record.dcm"
    bad.write_bytes(b"not a DICOM")
    payload = build_source_plan_dispositions(plans, {"P1": [bad]}, accepted)
    assert payload["status"] == "complete_with_holds"
    assert len(payload["record_read_errors"]) == 1


def test_unwritable_disposition_register_fails_loudly(tmp_path, monkeypatch):
    import rtpipeline.plan_disposition as pd
    payload = build_source_plan_dispositions([], {}, {})
    def fail(*a, **k): raise OSError("synthetic write failure")
    monkeypatch.setattr(pd, "_write_json_atomic", fail)
    with pytest.raises(OSError, match="synthetic write failure"):
        write_source_plan_dispositions(tmp_path, payload)


def test_archive_is_byte_identical_and_does_not_mutate_source(tmp_path):
    root, out = tmp_path / "input", tmp_path / "out"
    p = plan(root)
    r = add_record(root, p)
    _, payload = run(root, out)
    row = payload["plans"][0]
    assert (out / row["archived_path"]).read_bytes() == p.read_bytes()
    assert (out / row["records"][0]["archived_path"]).read_bytes() == r.read_bytes()
    assert not (out / row["archived_path"]).samefile(p)
    _, second = run(root, out, resume=True)
    assert second == payload


def test_missing_source_is_not_a_reusable_empty_history(tmp_path):
    from rtpipeline.plan_disposition import source_scope_fingerprint, source_dispositions_match_source
    root = tmp_path / "absent"
    assert source_scope_fingerprint(root) is None
    payload = build_source_plan_dispositions([], {}, {})
    payload.update(source_root=str(root.resolve()), source_scope_fingerprint=None,
                   source_archive_status="complete", course_publication_status="complete")
    assert not source_dispositions_match_source(payload, root)


@pytest.mark.parametrize("change", ["add", "delete", "bytes_same_stat"])
def test_source_history_freshness_requires_complete_current_identity(tmp_path, change):
    from rtpipeline.plan_disposition import source_dispositions_match_source
    root, out = tmp_path / "input", tmp_path / "out"
    p = plan(root)
    add_record(root, p)
    _, payload = run(root, out)
    assert source_dispositions_match_source(payload, root)
    if change == "add": plan(root)
    elif change == "delete": p.unlink()
    else:
        import os
        st = p.stat()
        data = p.read_bytes()
        assert b"APPROVED" in data
        p.write_bytes(data.replace(b"APPROVED", b"REJECTED"))
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert not source_dispositions_match_source(payload, root)


def test_archive_failure_keeps_pending_typed_delivery_receipt(tmp_path, monkeypatch):
    import rtpipeline.plan_disposition as pd
    root, out = tmp_path / "input", tmp_path / "out"
    p = plan(root)
    add_record(root, p)
    payload = build_source_plan_dispositions(*inventory(root))
    def fail(*a, **k): raise OSError("copy denied")
    monkeypatch.setattr(pd.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="copy denied"):
        write_source_plan_dispositions(out, payload)
    saved = json.loads((out / RELATIVE_PATH).read_text())
    assert saved["source_archive_status"] == "pending"
    assert saved["plans"][0]["validated_record_count"] == 1
    assert saved["plans"][0]["disposition_type"] == "non_measurable_delivery"


def test_workflow_manifest_and_campaign_rollup_preserve_nonmeasurement_denominator(tmp_path):
    import runpy
    from rtpipeline.workflow_delegate import _validate_organize
    root, out = tmp_path / "input", tmp_path / "out"
    p = plan(root)
    add_record(root, p)
    _, before = run(root, out)
    result = _validate_organize(out, quarantine_invalid=True)
    assert result["ledger"]["source_plan_dispositions"] == before
    assert result["ledger"]["source_history_current"] is True
    scripts = Path(__file__).resolve().parents[1] / "workflow/scripts"
    manifest = runpy.run_path(str(scripts / "organize_courses.py"))["_manifest_payload"](result["ledger"], [])
    assert manifest["courses"] == []
    assert manifest["source_plan_dispositions"]["non_course_delivery_plan_count"] == 1
    campaign = runpy.run_path(str(scripts / "campaign_ledger.py"))
    summary = campaign["rollup"](out)
    assert summary["source_plan_dispositions"]["non_course_validated_record_count"] == 1
    assert summary["course_count"] == 0


def test_legacy_course_only_manifest_does_not_skip_source_census(tmp_path, monkeypatch):
    import runpy
    from test_organize_course_quarantine import _organize_workflow
    script = Path(__file__).resolve().parents[1] / "workflow/scripts/organize_courses.py"
    module = runpy.run_path(str(script))
    out = tmp_path / "out"
    workflow = _organize_workflow(tmp_path, out)
    manifest = Path(workflow.output.manifest)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema": module["MANIFEST_SCHEMA"]}))
    function = module["_existing_manifest_is_valid"]
    monkeypatch.setitem(function.__globals__, "_delegate_validation", lambda *a, **k: ({"courses": []}, [], []))
    assert function(workflow, manifest, out) is False
