"""Normal regressions adapted from the six WS270 review reproducers.

Only deterministic synthetic DICOM is used. F3 lives in the resume regression.
"""
import importlib.util
import json
from pathlib import Path

import pydicom

from rtpipeline import organize as org
from rtpipeline.config import PipelineConfig
from organize_io_fixture import synthetic_ct_conversion
from review_classify_fixture import c_concurrent_same_uid_diff_iso_same_date, c_split_iso
from review_organize_fixture import build


def declaration(path, uids):
    path.write_text(json.dumps({"declarations": [{
        "plan_sop_instance_uids": list(uids),
        "basis": "synthetic concurrent components (WS270 reviewer)",
        "declared_on": "2026-11-03", "issued_by": "WS270 reviewer (synthetic)",
    }]}))
    return path


def classify(folder):
    return org._classify_doses(
        sorted(folder.glob("p*.dcm")), sorted(folder.glob("d*.dcm")),
        treatment_record_paths=sorted(folder.glob("rec*.dcm")),
        structure_set_paths=sorted(folder.glob("rs*.dcm")),
        course_dose_paths=sorted(folder.glob("d*.dcm")),
    )


def test_same_target_uid_distinct_isocentre_same_day_is_not_collapsed(tmp_path, monkeypatch):
    """F1: shared TARGET identity cannot alone discard a distinct isocentre."""
    monkeypatch.delenv("RTPIPELINE_CONCURRENT_COMPONENTS", raising=False)
    case = c_concurrent_same_uid_diff_iso_same_date(tmp_path)
    plans, doses = case["plans"], case["doses"]
    head = org._classify_doses(plans, doses)
    assert head.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert set(head.selected_plans) == set(plans) and head.selected_doses == []

    original = [pydicom.dcmread(p) for p in plans]
    # Plan dates take precedence over approval dates; times never authorize a revision.
    for earlier, later, inferred in [
        ({}, {"RTPlanTime": "090000"}, False),
        ({}, {"ReviewDate": "20240103"}, False),
        ({}, {"RTPlanDate": "20240102"}, True),
        ({"RTPlanDate": "", "ReviewDate": "20240101"}, {"RTPlanDate": "20240102"}, True),
        ({"RTPlanDate": "20240102"}, {"RTPlanDate": "", "ReviewDate": "20240101"}, True),
        ({"RTPlanDate": "", "ReviewDate": "20240101"}, {}, False),
        ({}, {"RTPlanDate": "", "ReviewDate": "20240101"}, False),
        ({"RTPlanDate": "", "ReviewDate": ""}, {"RTPlanDate": "20240102"}, False),
        ({"RTPlanDate": "20240102"}, {"RTPlanDate": "", "ReviewDate": ""}, False),
    ]:
        import copy
        for path, template, updates in zip(plans, original, (earlier, later)):
            ds = copy.deepcopy(template)
            for key, value in updates.items():
                setattr(ds, key, value)
            ds.save_as(path, enforce_file_format=True)
        for order in (plans, plans[::-1]):
            result = org._classify_doses(order, doses[::-1])
            assert (result.classification == "plan_revisions_deduplicated") is inferred
            if inferred:
                assert any("revision inferred across distinct isocentres" in w.lower()
                           for w in result.warnings)
            else:
                assert result.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
                assert set(result.selected_plans) == set(plans) and not result.selected_doses
    # A later third plan cannot erase an earlier same-day distinct pair.
    import copy
    for path, ds in zip(plans, original):
        ds.save_as(path, enforce_file_format=True)
    third = copy.deepcopy(original[1])
    third.SOPInstanceUID = str(third.SOPInstanceUID) + ".1"
    third.RTPlanDate = "20240102"
    third_plan = tmp_path / "pC.dcm"
    third.save_as(third_plan, enforce_file_format=True)
    third_dose = copy.deepcopy(pydicom.dcmread(doses[1]))
    third_dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = third.SOPInstanceUID
    third_dose.SOPInstanceUID = str(third_dose.SOPInstanceUID) + ".1"
    third_grid = tmp_path / "dC.dcm"
    third_dose.save_as(third_grid, enforce_file_format=True)
    held = org._classify_doses(plans + [third_plan], doses + [third_grid])
    assert held.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert len(held.selected_plans) == 3 and not held.selected_doses
    # Agreeing first-control-point isocentres permit same-day revisions.
    original[0].BeamSequence[0].ControlPointSequence[0].IsocenterPosition = [0, 0, 150]
    for path, ds in zip(plans, original):
        ds.save_as(path, enforce_file_format=True)
    assert org._classify_doses(plans, doses).classification == "plan_revisions_deduplicated"


def test_partial_records_contradiction_is_visible(tmp_path, monkeypatch):
    """F2: the reviewer's 10/10 + 5/10 records must refuse the full sum."""
    uids = build(tmp_path / "in", "partial_records")
    path = declaration(tmp_path / "decl.json", uids["D"].values())
    folder = tmp_path / "in" / "SYNTH_D" / "c0"
    monkeypatch.delenv("RTPIPELINE_CONCURRENT_COMPONENTS", raising=False)
    legacy = classify(folder)
    monkeypatch.setenv("RTPIPELINE_CONCURRENT_COMPONENTS", str(path))
    result = classify(folder)
    assert any("RTRECORD shows incomplete delivery of a declared plan" in w
               for w in result.warnings)
    assert result.classification == legacy.classification
    assert result.selected_plans == legacy.selected_plans
    assert result.selected_doses == legacy.selected_doses
    assert result.should_sum == legacy.should_sum
    # Existing records with no validated delivered fractions are also below
    # the planned count; an empty delivered-session set must not evade refusal.
    for record in folder.glob("rec*.dcm"):
        ds = pydicom.dcmread(record)
        if str(ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID) != uids["D"]["pB"]:
            continue
        ds.TreatmentSessionBeamSequence[0].TreatmentTerminationStatus = "ABORTED"
        ds.save_as(record, enforce_file_format=True)
    zero_sessions = org._record_delivery_evidence(sorted(folder.glob("rec*.dcm")))
    assert not zero_sessions[uids["D"]["pB"]]["sessions"]
    assert any("RTRECORD shows incomplete delivery of a declared plan" in w
               for w in classify(folder).warnings)
    # No records and complete records retain the declaration behaviour.
    for variant in ("none", "full_records"):
        other = tmp_path / variant
        uids = build(other, variant)
        declaration(path, uids["D"].values())
        assert classify(other / "SYNTH_D" / "c0").classification == org.DECLARED_CONCURRENT_COMPONENTS


def test_record_free_hold_does_not_claim_delivery(tmp_path, monkeypatch):
    """F4: keep the classification key, state the record-free basis."""
    monkeypatch.delenv("RTPIPELINE_CONCURRENT_COMPONENTS", raising=False)
    case = c_split_iso(tmp_path)
    head = org._classify_doses(case["plans"], case["doses"])
    assert head.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    text = " ".join(head.warnings + [head.reason]).lower()
    assert "delivered" not in text
    phrase = "equal-signature plans retained as membership without revision evidence"
    assert phrase in head.reason.lower() and phrase in " ".join(head.warnings).lower()
    summary = org._calculate_delivery_summary(
        case["plans"], [], selected_plan_paths=head.selected_plans,
        selected_dose_paths=[], classification=head.classification,
    )
    assert phrase in " ".join(summary["delivery_warnings"]).lower()


def test_classification_without_structure_sets_matches_organize(tmp_path, monkeypatch):
    """F5: exercise the audit caller, rather than bypassing RTSTRUCT validation."""
    uids = build(tmp_path / "in")
    path = declaration(tmp_path / "decl.json", uids["D"].values())
    monkeypatch.setenv("RTPIPELINE_CONCURRENT_COMPONENTS", str(path))
    monkeypatch.setattr(org, "run_dcm2niix", synthetic_ct_conversion)
    config = PipelineConfig(tmp_path / "in", tmp_path / "out", tmp_path / "logs",
                            max_workers_override=1, dicom_copy_use_hardlinks=False,
                            dicom_copy_dedup_by_sop_uid=False)
    org.organize_and_merge(config)
    course = tmp_path / "out" / "SYNTH_D" / "2024-01"
    meta = json.loads((course / "metadata" / "case_metadata.json").read_text())
    contract = meta["course_contract"]
    assert contract["dose_classification"]["classification"] == org.DECLARED_CONCURRENT_COMPONENTS
    spec = importlib.util.spec_from_file_location(
        "audit_course_dose_completeness", Path(org.__file__).parents[1] / "scripts" / "audit_course_dose_completeness.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    for configured in (True, False):
        if not configured:
            monkeypatch.delenv("RTPIPELINE_CONCURRENT_COMPONENTS")
        plans, doses, classification, delivery = audit._prospective_contract_inputs(course, contract)
        assert classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS
        assert {p["sop_instance_uid"] for p in plans} == set(uids["D"].values())
        assert len(doses) == 2
        assert delivery["delivered_dose_gy"] is None


def test_declaration_naming_only_unapproved_plans_is_visible(tmp_path, monkeypatch):
    """F6: relevance includes the course's unapproved plans."""
    uids = build(tmp_path / "in", "unapproved_pair")
    path = declaration(tmp_path / "decl.json", [uids["D"]["pU"], uids["D"]["pV"]])
    folder = tmp_path / "in" / "SYNTH_D" / "c0"
    monkeypatch.delenv("RTPIPELINE_CONCURRENT_COMPONENTS", raising=False)
    legacy = classify(folder)
    monkeypatch.setenv("RTPIPELINE_CONCURRENT_COMPONENTS", str(path))
    result = classify(folder)
    assert any("not applied: a declared plan is not APPROVED" in w for w in result.warnings)
    assert result.classification == legacy.classification
    assert result.selected_plans == legacy.selected_plans
    assert result.selected_doses == legacy.selected_doses

    # Organize filters approval before classifying; the complete course UID
    # context must survive that filter on publication and resume as well.
    monkeypatch.setattr(org, "run_dcm2niix", synthetic_ct_conversion)
    config = PipelineConfig(tmp_path / "in", tmp_path / "out", tmp_path / "logs",
                            max_workers_override=1, dicom_copy_use_hardlinks=False,
                            dicom_copy_dedup_by_sop_uid=False)
    for resume in (False, True):
        config.resume = resume
        org.organize_and_merge(config)
        meta = json.loads((tmp_path / "out/SYNTH_D/2024-01/metadata/case_metadata.json").read_text())
        assert any("not applied: a declared plan is not APPROVED" in w
                   for w in meta["course_contract"]["dose_classification"]["warnings"])
