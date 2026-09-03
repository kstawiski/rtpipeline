from __future__ import annotations

from rtpipeline.course_contract import (
    build_dvh_decision,
    classify_course_dose_completeness,
)
from rtpipeline.dvh import annotate_dvh_metrics


def _plan(uid: str, planned: int = 10, delivered: int = 10) -> dict:
    return {
        "sop_instance_uid": uid,
        "planned_fraction_count": planned,
        "delivered_fraction_count": delivered,
    }


def _dose(uid: str, refs: list[str], summation: str = "PLAN") -> dict:
    return {
        "sop_instance_uid": uid,
        "referenced_plan_uids": refs,
        "dose_summation_type": summation,
    }


def _class(kind: str) -> dict:
    return {"classification": kind}


def test_exact_multi_plan_rtdose_is_course_level_eligible() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1"), _plan("p2")],
        selected_doses=[_dose("d", ["p1", "p2"], "MULTI_PLAN")],
        dose_classification=_class("MULTI_PLAN_exact_coverage"),
        dose_grid=None,
        per_plan_delivery=[
            {"plan_sop_uid": "p1", "delivered_record_count": 1, "delivered_fraction_count": 10},
            {"plan_sop_uid": "p2", "delivered_record_count": 1, "delivered_fraction_count": 10},
        ],
        delivery_status="fully_delivered",
        spatial_mapping_validated=True,
    )
    assert result["status"] == "eligible"
    assert result["category"] == "multi_plan_rtdose_exact_uid_coverage"


def test_single_plan_multi_plan_rtdose_exact_uid_coverage_is_eligible() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1")],
        selected_doses=[_dose("d", ["p1"], "MULTI_PLAN")],
        dose_classification=_class("MULTI_PLAN_exact_coverage"),
        dose_grid=None,
        per_plan_delivery=[
            {
                "plan_sop_uid": "p1",
                "delivered_record_count": 1,
                "delivered_fraction_count": 10,
            }
        ],
        delivery_status="fully_delivered",
        spatial_mapping_validated=True,
    )
    assert result["status"] == "eligible"
    assert result["category"] == "multi_plan_rtdose_exact_uid_coverage"


def test_single_plan_dose_requires_physical_grid_validation() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1")],
        selected_doses=[_dose("d", ["p1"])],
        dose_classification=_class("delivered_plan_selected"),
        dose_grid=None,
        per_plan_delivery=[
            {
                "plan_sop_uid": "p1",
                "delivered_record_count": 1,
                "delivered_fraction_count": 10,
            }
        ],
        delivery_status="fully_delivered",
        spatial_mapping_validated=False,
    )
    assert result["status"] == "not_defensible"
    assert result["reason_code"] == "dose_grid_not_validated"


def test_record_without_verified_fraction_is_not_delivery_evidence() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1", delivered=0)],
        selected_doses=[_dose("d", ["p1"])],
        dose_classification=_class("delivered_plan_selected"),
        dose_grid=None,
        per_plan_delivery=[
            {
                "plan_sop_uid": "p1",
                "delivered_record_count": 1,
                "delivered_fraction_count": 0,
            }
        ],
        delivery_status="delivered_but_records_absent",
        spatial_mapping_validated=True,
    )
    assert result["status"] == "not_defensible"
    assert result["reason_code"] == "delivery_evidence_unresolved"


def test_multi_plan_uid_subset_is_not_course_level_eligible() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1"), _plan("p2")],
        selected_doses=[_dose("d", ["p1"], "MULTI_PLAN")],
        dose_classification=_class("MULTI_PLAN_exact_coverage"),
        dose_grid=None,
        per_plan_delivery=[
            {"plan_sop_uid": "p1", "delivered_record_count": 1, "delivered_fraction_count": 10},
            {"plan_sop_uid": "p2", "delivered_record_count": 1, "delivered_fraction_count": 10},
        ],
        delivery_status="fully_delivered",
    )
    assert result["status"] == "not_defensible"
    assert result["reason_code"] == "dose_plan_uid_coverage_mismatch"


def test_distinct_unselected_delivered_plan_is_not_silently_dropped() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1")],
        selected_doses=[_dose("d", ["p1"])],
        dose_classification=_class("delivered_plan_selected"),
        dose_grid=None,
        per_plan_delivery=[
            {"plan_sop_uid": "p1", "delivered_record_count": 1, "delivered_fraction_count": 10},
            {"plan_sop_uid": "p2", "delivered_record_count": 1, "delivered_fraction_count": 3},
        ],
        delivery_status="fully_delivered",
    )
    assert result["status"] == "not_defensible"
    assert result["reason_code"] == "unselected_delivered_plan_requires_reconciliation"


def test_per_plan_accumulation_records_delivered_fraction_weights() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1", 10, 5), _plan("p2", 5, 5)],
        selected_doses=[_dose("d1", ["p1"]), _dose("d2", ["p2"])],
        dose_classification=_class("sequential_phases_summed"),
        dose_grid=None,
        per_plan_delivery=[
            {"plan_sop_uid": "p1", "delivered_record_count": 1, "delivered_fraction_count": 5},
            {"plan_sop_uid": "p2", "delivered_record_count": 1, "delivered_fraction_count": 5},
        ],
        delivery_status="partially_delivered",
        spatial_mapping_validated=True,
    )
    assert result["status"] == "eligible"
    assert result["delivered_fraction_weights"] == {"p1": 0.5, "p2": 1.0}


def test_per_plan_accumulation_fails_closed_without_spatial_validation() -> None:
    result = classify_course_dose_completeness(
        selected_plans=[_plan("p1"), _plan("p2")],
        selected_doses=[_dose("d1", ["p1"]), _dose("d2", ["p2"])],
        dose_classification=_class("sequential_phases_summed"),
        dose_grid=None,
        per_plan_delivery=[
            {"plan_sop_uid": "p1", "delivered_record_count": 1, "delivered_fraction_count": 10},
            {"plan_sop_uid": "p2", "delivered_record_count": 1, "delivered_fraction_count": 10},
        ],
        delivery_status="fully_delivered",
    )
    assert result["status"] == "not_defensible"
    assert result["reason_code"] == "spatial_mapping_not_validated"


def test_near_zero_is_quarantined_without_zero_assignment() -> None:
    result = annotate_dvh_metrics(
        {
            "D95Gy": 0.01,
            "D98Gy": 0.0,
            "D2Gy": 0.03,
            "DmeanGy": 0.01,
            "V20Gy": 0.0,
            "V50Gy": 0.0,
            "V100Gy": 0.0,
        },
        technique="EBRT",
        structure_name="PTV",
        structure_interpreted_type="PTV",
        target_like=True,
        prescription_resolved=True,
        dose_response_eligible=True,
        quarantine_near_zero=True,
        rtstruct_sop_instance_uid=None,
        rtstruct_path=None,
        zero_dose_status="zero_dose_in_grid",
        zero_dose_reason="target-like ROI has near-zero dose in grid",
        zero_dose_trigger_metric="D95Gy",
        zero_dose_trigger_value_gy=0.01,
    )
    assert result["dose_response_eligible"] is False
    assert result["dose_metric_status"] == "quarantined_near_zero_requires_reconciliation"
    assert result["dose_response_quarantine_status"] == "pending_plan_target_reconciliation"
    assert result["relative_metric_status"] == "quarantined_near_zero_requires_reconciliation"
    assert result["D95Gy"] == 0.01


def test_dvh_decision_retains_qc_grid_but_records_course_dose_exclusion() -> None:
    completeness = {
        "status": "not_defensible",
        "reason_code": "unselected_delivered_plan_requires_reconciliation",
        "reason": "A clinically delivered plan is absent from the selected dose grid.",
    }
    decision = build_dvh_decision(
        2,
        1,
        "partially_delivered",
        dose_response_eligible=False,
        dose_completeness=completeness,
    )
    assert decision["status"] == "ready"
    assert decision["metrics_status"] == "computed"
    assert decision["dose_response_eligible"] is False
    assert decision["reason_code"] == "dose_response_course_dose_incomplete"
    assert decision["dose_completeness_reason_code"] == (
        "unselected_delivered_plan_requires_reconciliation"
    )
