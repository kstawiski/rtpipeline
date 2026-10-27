"""RF18: brachytherapy CTV selection and DN02 empty worker results. No clinical data.

Selector. Robustness selection patterns are matched against the whole ROI name,
so ``CTV*`` cannot select the GEC-ESTRO brachytherapy targets ``HR-CTV`` and
``IR-CTV``. A course whose only targets carry those names closed as an
unmatched selection. The fix is an explicit, opt-in pattern set; the default
selection semantics are unchanged.

DN02. In production, a worker returned only the extractor's resource-guard
disposition records (no feature keys) for every perturbation of a large target.
The parent logged "returned no scalar features" and dropped the result, so the
course failed as an incomplete extraction with no recorded cause. The extractor
is replaced here by synthetic records with the exact extract_ct_roi_arms shape;
the robustness course, admission and validation code is real.
"""
from __future__ import annotations

import json
from fnmatch import fnmatch

import pandas as pd
import pytest

from rtpipeline import radiomics_parallel as rp
from rtpipeline import radiomics_robustness as rr
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from rtpipeline.robustness_watchdog import technical_rows, validate_technical_frame
from test_robustness_feature_gaps import _course, fake_parallel_worker

CAMPAIGN_SELECTION = ["GTV*", "CTV*", "PTV*"]
# Name shapes taken from the campaign ROI inventory (aggregate, no identifiers).
INVENTORY_NAMES = [
    "HR-CTV", "IR-CTV", "hr-ctv", "CTV1", "CTV 2", "CTV_3", "PTV1", "PTV 1", "GTV1",
    "PECHERZ - PTV", "ODBYTNICA-PTV", "2CM OD PTV1", "ZPTV1", "ZMARGPTV1", "NS_GTV",
    "NS_CTV", "MARG/PTV1", "urinary_bladder", "Bladder", "BODY", "HR", "IR", "IRIS",
    "HRT", "CTV-HR", "ZHR-CTV",
]


def _legacy_match(name, selection):
    """The pre-RF18 inline matcher, copied verbatim from four call sites."""
    return any(fnmatch(str(name).upper(), pattern.upper()) for pattern in selection)


def test_brachytherapy_pattern_set_is_explicit():
    assert rr.BRACHYTHERAPY_CTV_PATTERNS == ("HR-CTV*", "IR-CTV*")


@pytest.mark.parametrize("selection", [
    CAMPAIGN_SELECTION,
    CAMPAIGN_SELECTION + ["Bladder", "Whole_Bladder", "Bladder KM", "bladder 1", "urinary_bladder"],
    ["ROI"], [],
])
def test_shared_matcher_keeps_default_selection_semantics(selection):
    for name in INVENTORY_NAMES:
        assert rr.robustness_selection_matches(name, selection) == _legacy_match(name, selection), name


def test_campaign_selection_misses_brachytherapy_targets():
    assert not rr.robustness_selection_matches("HR-CTV", CAMPAIGN_SELECTION)
    assert not rr.robustness_selection_matches("IR-CTV", CAMPAIGN_SELECTION)


def test_opt_in_patterns_select_brachytherapy_ctvs_only():
    selection = rr.with_brachytherapy_ctv_patterns(CAMPAIGN_SELECTION)
    assert selection == CAMPAIGN_SELECTION + ["HR-CTV*", "IR-CTV*"]
    added = {
        name for name in INVENTORY_NAMES
        if rr.robustness_selection_matches(name, selection)
        and not rr.robustness_selection_matches(name, CAMPAIGN_SELECTION)
    }
    assert added == {"HR-CTV", "IR-CTV", "hr-ctv"}
    # Idempotent, order-preserving, and case-insensitive about existing entries.
    assert rr.with_brachytherapy_ctv_patterns(selection) == selection
    assert rr.with_brachytherapy_ctv_patterns(["hr-ctv*"]) == ["hr-ctv*", "IR-CTV*"]
    assert rr.with_brachytherapy_ctv_patterns(None) == ["HR-CTV*", "IR-CTV*"]


def test_requiredness_expansion_follows_the_selection():
    from types import SimpleNamespace

    from rtpipeline.roi_requiredness import Requiredness

    inventory = SimpleNamespace(named_rois=[
        SimpleNamespace(name=n) for n in ("HR-CTV", "IR-CTV", "Bladder", "Rectum")])
    inspect = lambda _path: inventory  # noqa: E731
    assert rr._robustness_selection_requiredness(
        "RS.dcm", CAMPAIGN_SELECTION, inspect=inspect)["HR-CTV"] == Requiredness.INVENTORY_ONLY
    expanded = rr._robustness_selection_requiredness(
        "RS.dcm", rr.with_brachytherapy_ctv_patterns(CAMPAIGN_SELECTION), inspect=inspect)
    assert {n for n, r in expanded.items() if r == Requiredness.ANALYSIS_REQUIRED} == {"HR-CTV", "IR-CTV"}


def _resource_records(identity, extra=None):
    """Disposition-only records as extract_ct_roi_arms returns on a resource refusal."""
    return [{
        **identity, "extraction_arm": arm, "extraction_status": "failed",
        "extraction_failure_kind": "resource_limit",
        "roi_structural_code": "ROI_PREDICTED_MEMORY_EXCEEDS_LIMIT",
        "intensity_texture_disposition": "failed",
        "resource_guard_predicted_peak_bytes": 9_000_000_000,
        "resource_guard_memory_budget_bytes": 8 * 1024 ** 3,
        **(extra or {}),
    } for arm in CT_EXTRACTION_ARMS]


def test_empty_result_cause_is_named():
    resource = rr._empty_worker_result_failure({"__records__": _resource_records({})})
    assert resource["reason_code"] == "resource_limit"
    assert resource["evidence"]["record_count"] == 2
    assert {r["roi_structural_code"] for r in resource["evidence"]["records"]} == {
        "ROI_PREDICTED_MEMORY_EXCEEDS_LIMIT"}
    mixed = _resource_records({})
    mixed[0]["extraction_failure_kind"] = "timeout"
    assert rr._empty_worker_result_failure({"__records__": mixed})["reason_code"] == "no_scalar_features"
    empty = rr._empty_worker_result_failure({"__records__": []})
    assert empty["reason_code"] == "no_scalar_features" and empty["evidence"] == {
        "record_count": 0, "records": []}


@pytest.mark.parametrize("reason", ["resource_limit", "no_scalar_features"])
def test_empty_result_rows_pass_the_technical_validator(reason):
    failure = {"reason_code": reason, "evidence": {"record_count": 0, "records": []}}
    rows = technical_rows({"__technical_failure__": failure, "roi_name": "GTVm",
                           "perturbation_id": "ntcv_v0"},
                          {"patient_id": "P"}, "run", "sha256:" + "0" * 64)
    frame = pd.DataFrame(rows)
    assert validate_technical_frame(frame, {"ntcv_v0"}, "Manual/GTVm") == {"ntcv_v0"}
    frame["reason_code"] = "unexplained"
    with pytest.raises(RuntimeError, match="unexplained technical non-measurement"):
        validate_technical_frame(frame, {"ntcv_v0"}, "Manual/GTVm")


def _resource_limited_worker(limited):
    def worker(task):
        params = task[1]
        pid = params["extra_metadata"]["perturbation_id"]
        if pid not in limited:
            return fake_parallel_worker(task)
        identity = {c: params[c] for c in rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
        return {"__records__": _resource_records(identity),
                "segmentation_source": params["segmentation_source"],
                "roi_name": params["roi_name"], "perturbation_id": pid}
    return worker


def test_resource_limited_condition_is_recorded_not_dropped(tmp_path, monkeypatch, caplog):
    """DN02 reproducer. Before RF18 this raised 'incomplete robustness extraction'
    with the condition only listed as missing; now the condition is a technical
    failure row carrying the worker's resource-guard evidence."""
    course, cfg, rob = _course(tmp_path, monkeypatch, parallel=True)
    monkeypatch.setattr(rp, "_isolated_radiomics_extraction_with_retry",
                        _resource_limited_worker({"ntcv_v0"}))
    with pytest.raises(RuntimeError) as raised:
        rr.robustness_for_course(cfg, rob, course)
    assert "missing=" not in str(raised.value)
    assert "published partial results" in str(raised.value)
    assert "technical_failure_arm_rows=2" in str(raised.value)
    evidence = pd.read_parquet(course / "radiomics_robustness_ct.failed_evidence.parquet")
    failed = evidence[evidence.robustness_status.eq("technical_failure")]
    assert set(failed.perturbation_id) == {"ntcv_v0"}
    assert set(failed.extraction_arm) == set(CT_EXTRACTION_ARMS)
    assert set(failed.reason_code) == {"resource_limit"}
    assert failed.value.isna().all() and failed.feature_name.isna().all()
    detail = json.loads(failed.technical_evidence.iloc[0])
    assert {r["roi_structural_code"] for r in detail["records"]} == {
        "ROI_PREDICTED_MEMORY_EXCEEDS_LIMIT"}
    # The other conditions keep their measurements as failed-stage evidence.
    assert set(evidence.loc[evidence.robustness_status.eq("measured"), "perturbation_id"]) == {
        "ntcv_v-15", "ntcv_n10_v-15", "ntcv_n10_v0"}
    assert not (course / "radiomics_robustness_ct.parquet").exists()


def test_unaffected_course_is_unchanged(tmp_path, monkeypatch):
    course, cfg, rob = _course(tmp_path, monkeypatch, parallel=True)
    monkeypatch.setattr(rp, "_isolated_radiomics_extraction_with_retry",
                        _resource_limited_worker(set()))
    table = pd.read_parquet(rr.robustness_for_course(cfg, rob, course))
    assert set(table.robustness_status) == {"measured"}
