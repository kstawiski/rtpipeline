from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rtpipeline import radiomics_ct_contract as contract
from rtpipeline import roi_name_rules as rules
from rtpipeline.rt_details import is_target_volume_name


@pytest.mark.parametrize("name", [
    "CTV 1", " cTv\t1 ", "CTV 2", "ctv   3", "CTV4", "PTV 1",
    "pTv\n2", "PTV 3", "PTV4", "gtv", "GTVn", "gTvP", "GTVm",
    "GTV1", "GTV2", "GTV3", "GTV 2", "GTVn1", "GTVn2", "GTVn7",
    "GTVm1", "GTVm2", "GTVp12", "GTVn 1", "PTV 01",
])
def test_unqualified_targets_reach_real_class_decision(name):
    decision = contract.classify_ct_roi("Manual", name)
    assert decision.roi_class == "target"
    assert decision.primary_resegment_range_hu == (-1000, 400)
    assert decision.feature_publication_policy == contract.FEATURE_POLICY_EXTRACT
    assert decision.map_version == "ct-roi-class-map-2026-09-06-v6"


@pytest.mark.parametrize("name", [
    "Jelita - PTV", "jelita-ptv", "Odbytnica - PTV", "odbytnica-ptv",
    "ODBYTNICA\t−\tPTV 2", "Esica-PTV", "Pecherz - PTV", "Watroba - GTVm2",
    "Nerka L-PTV", "PTV2 - PTV1", "ptv2-ptv1", "PTV 2–PTV 1",
    "PTV1 - GTVm1", "CTV1-GTVm", "PTV3 - PTV21", "PTV1-CTV1",
    "2cm od PTV1", "2cm od PTV2", "2cm od PTV3", "1cm od PTV1",
    "1 mm od PTV 4", "zz_2cm od PTV12", "zPTV1", "z_PTV 2", "marg PTV2",
    "zmargPTV1", "z_marg_PTV1", "z_min_PTV1", "Jelita - PTV__partial",
])
def test_definite_crops_margins_and_controls_are_inventory_helpers(name):
    decision = contract.classify_ct_roi("Manual", name)
    assert decision.roi_class == "planning_helper"
    assert decision.feature_publication_policy == contract.FEATURE_POLICY_INVENTORY_ONLY
    assert decision.primary_resegment_range_hu is None


@pytest.mark.parametrize("name", [
    "Jelita + PTV1", "Jelita w PTV", "odb and PTV", "PTV1 + PTV2",
    "PTV1orPTV2", "PTV1-0.5jel", "PTV1-jelita", "GTV1-prostata",
    "GTV2-pęcherz", "GTVm - lopatka", "GTVn4p", "GTVn_SBRT?",
    "CTV1_v1", "CTV1-61,6", "CTVN_Prostate", "PTV_20/4_01/2024",
    "PTV2-PTV1 60/2.4", "PTV3 - PTV2_fiz", "PTV1_2_SUMA", "nPTV1",
    "NS_GTV1_2", "sl-ptv", "sp-ptv", "zBowel + PTV1", "m3/ptv1",
    "unknown mixed tissue", "bladder-prostate", "PTVunknown", "xPTV1",
    "PTV 1 2", "G TV1", "P T V1", "PTV-1", "PTV+1", "PTV/1",
    "PTV_1", "PTV1.0", "PTV1,0", "GTVn+1", "PTV1__unknown",
])
def test_ambiguous_and_mixed_names_fail_closed(name):
    decision = contract.classify_ct_roi("Manual", name)
    assert decision.roi_class == "unresolved_mixed"
    assert decision.adjudication_status == "operator_adjudication_required"
    assert decision.primary_resegment_range_hu is None


@pytest.mark.parametrize("left,right", [
    ("PTV1", "PTV-1"), ("PTV1", "PTV+1"), ("PTV1", "PTV_1"),
    ("PTV1", "PTV/1"), ("PTV1", "PTV01"), ("PTV12", "PTV1 2"),
    ("PTV1", "xPTV1"), ("PTV1", "zPTV1"), ("GTVn1", "GTVp1"),
    ("PTV1-PTV2", "PTV1+PTV2"), ("PTV1-PTV2", "PTV2-PTV1"),
    ("PTV1-PTV2", "PTV1PTV2"), ("CTV1_v1", "CTV1_v2"),
    ("PTV1.2", "PTV12"), ("PTV1,2", "PTV1.2"),
    ("maße", "masse"), ("pęcherz", "pecherz"), ("K", "K"),
])
def test_normalization_retains_semantic_differences(left, right):
    assert rules.normalize_roi_class_name(left) != rules.normalize_roi_class_name(right)


@pytest.mark.parametrize("name", ["PTV 1", " pTv\t1 ", "PTV1", "Odbytnica − PTV", "a__b"])
def test_normalization_is_idempotent(name):
    normalized = rules.normalize_roi_class_name(name)
    assert rules.normalize_roi_class_name(normalized) == normalized


def test_conflicting_normalized_map_entries_veto_exact_and_rule():
    data = copy.deepcopy(contract.load_roi_class_map()[0])
    data["manual_custom_crosswalk"]["ptv 1"] = {
        "roi_class": "planning_helper", "adjudication_status": "approved",
        "evidence_basis": "Synthetic explicit conflicting fixture",
    }
    for name in ("PTV1", "ptv 1", "PtV 1"):
        entry, source = rules.lookup_manual_roi_class(name, data)
        assert entry["roi_class"] == "unresolved_mixed"
        assert source.endswith("normalization_conflict")


def test_pending_status_is_a_veto_not_a_target_approval():
    data = copy.deepcopy(contract.load_roi_class_map()[0])
    data["manual_custom_crosswalk"]["GTVp99"] = {
        "roi_class": "target", "adjudication_status": "pending_human_review",
        "evidence_basis": "Synthetic pending decision fixture",
    }
    entry, _ = rules.lookup_manual_roi_class("gtvp 99", data)
    assert entry["roi_class"] == "unresolved_mixed"
    assert entry["adjudication_status"] == "operator_adjudication_required"


def test_shared_target_detector_is_reused_and_not_sufficient(monkeypatch):
    assert is_target_volume_name("PTV2 - PTV1")
    assert is_target_volume_name("2cm od PTV2")
    assert not is_target_volume_name("Jelita - PTV")
    assert contract.classify_ct_roi("Manual", "PTV2 - PTV1").roi_class != "target"
    calls = []
    monkeypatch.setattr(rules, "is_target_volume_name", lambda text: calls.append(text) or False)
    assert contract.classify_ct_roi("Manual", "GTVp99").roi_class == "unresolved_mixed"
    assert "gtvp99" in calls


def test_legacy_map_contract_is_not_silently_reinterpreted():
    data = copy.deepcopy(contract.load_roi_class_map()[0])
    data.pop("name_rules")
    assert rules.lookup_manual_roi_class("CTV 1", data)[0] is None
    assert rules.lookup_manual_roi_class("PTV1", data)[0]["roi_class"] == "target"


def test_unsupported_rule_contract_is_rejected():
    data = copy.deepcopy(contract.load_roi_class_map()[0])
    data["name_rules"]["version"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported CT ROI name rule version"):
        rules.lookup_manual_roi_class("PTV1", data)


def test_canonical_source_remains_exact_and_does_not_admit_manual_rules():
    assert contract.classify_ct_roi("AutoRTS_total", "PTV 1").roi_class == "unresolved_mixed"
    assert contract.classify_ct_roi("AutoRTS_total", "LIVER").roi_class == "unresolved_mixed"
    assert contract.classify_ct_roi("AutoRTS_total", "liver").roi_class == "solid_soft_tissue_neural"


@pytest.mark.parametrize("margin,declared,expected", [
    (0, "target", "target"), ({"uniform_mm": 0}, "target", "target"),
    (2, "planning_helper", "planning_helper"), (-2, "planning_helper", "planning_helper"),
    ({"left_mm": 1}, "planning_helper", "planning_helper"),
    (2, "target", "unresolved_mixed"), (0, "planning_helper", "unresolved_mixed"),
    ("unreadable", "target", "unresolved_mixed"), (True, "target", "unresolved_mixed"),
    ({"unsupported_mm": 0}, "target", "unresolved_mixed"),
    (float("nan"), "target", "unresolved_mixed"),
])
def test_derived_margin_contract_remains_fail_closed(tmp_path, margin, declared, expected):
    data = copy.deepcopy(contract.load_roi_class_map()[0])
    data["derived_crosswalk"]["fixture_derived"] = {
        "roi_class": declared, "require_provenance": True,
        "adjudication_status": "fixture", "evidence_basis": "Synthetic derived fixture",
    }
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    decision = contract.classify_ct_roi("Custom", "fixture_derived", map_path=path,
        custom_provenance={"fixture_derived": {
            "operation": "union", "source_structures": ["PTV1", "PTV2"], "margin": margin,
        }})
    assert decision.roi_class == expected
    assert decision.map_entry_source.startswith("derived_crosswalk:")


def test_derived_missing_and_nested_unreadable_provenance_remain_closed():
    assert contract.classify_ct_roi("Custom", "bowel_bag").roi_class == "unresolved_mixed"
    provenance = {
        "bowel_bag": {"operation": "union", "source_structures": ["pelvic_bones"], "margin": 0},
        "pelvic_bones": {"operation": "union", "source_structures": ["hip_left"], "margin": "bad"},
    }
    decision = contract.classify_ct_roi("Custom", "bowel_bag", custom_provenance=provenance)
    assert decision.roi_class == "unresolved_mixed"


@pytest.mark.parametrize("field,value", [
    ("roi_map_version", "ct-roi-class-map-2026-09-04-v5"),
    ("roi_map_hash", "stale-hash"),
])
def test_publication_rejects_stale_map_identity(monkeypatch, field, value):
    from test_radiomics_ct_dual_arm import _dual_arm_rows
    rows = _dual_arm_rows(monkeypatch)
    assert len(contract.validate_ct_publication(pd.DataFrame(rows))) == 2
    for row in rows:
        row[field] = value
    with pytest.raises(ValueError, match="stale ROI class map identity"):
        contract.validate_ct_publication(pd.DataFrame(rows))


def test_pending_map_entry_still_blocks_required_publication(monkeypatch):
    from test_radiomics_ct_dual_arm import _dual_arm_rows
    decision = contract.classify_ct_roi("Manual", "PTV1_v1")
    rows = _dual_arm_rows(monkeypatch, decision=decision)
    with pytest.raises(ValueError, match="operator adjudication is required"):
        contract.validate_ct_publication(pd.DataFrame(rows))


def test_new_rule_metadata_is_complete():
    data = contract.load_roi_class_map()[0]
    for entry in data["name_rules"]["rules"].values():
        assert all(entry.get(key) for key in ("roi_class", "adjudication_status", "evidence_basis"))
