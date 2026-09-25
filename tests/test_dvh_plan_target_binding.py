"""Plan-target binding for the near-zero target course quarantine.

All inputs are synthetic DICOM objects written under pytest's tmp_path. The two
stored tables were produced by the unmodified ae60f01 DVH stage from the same
fixture (see tests/dvh_plan_target_fixtures.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import dvh_plan_target_fixtures as fx
from rtpipeline.dvh import (
    EXCLUDED_TARGET_NOT_BOUND_STATUS,
    RELATIVE_DVH_METRIC_COLUMNS,
    _is_dose_derived_metric_column,
    apply_unbound_target_exclusion,
    dvh_for_course,
    reconcile_near_zero_plan_targets,
)

BASELINE_DIR = Path(__file__).parent / "synthetic_dvh_plan_target"
NO_NEAR_ZERO_BASELINE = BASELINE_DIR / "ae60f01_no_near_zero_targets.json"
QUARANTINE_BASELINE = BASELINE_DIR / "ae60f01_near_zero_course_quarantine.json"
UNBOUND_TARGETS = {"PTV2", "CTV2"}


def _run(tmp_path: Path, **kwargs) -> tuple[Path, pd.DataFrame, dict]:
    course = fx.build_course(tmp_path, **kwargs)
    assert dvh_for_course(course, parallel_workers=1, max_total_dose_gy=100.0)
    frame = pd.read_parquet(course / "dvh_metrics.parquet")
    qc = json.loads((course / "metadata" / "dvh_qc.json").read_text(encoding="utf-8"))
    return course, frame, qc


def _baseline_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(zip(payload["columns"], values)) for values in payload["rows"]]
    return {row["ROI_Name"]: row for row in rows}


def _assert_course_quarantined_as_at_ae60f01(course: Path, qc: dict, rule: str) -> None:
    assert fx.canonical_dvh_table(course) + "\n" == QUARANTINE_BASELINE.read_text(
        encoding="utf-8"
    )
    reconciliation = qc["plan_target_reconciliation"]
    assert reconciliation["decision"] == (
        "course_quarantined_pending_plan_target_reconciliation"
    )
    assert rule in reconciliation["quarantine_rules"]
    assert qc["target_near_zero_dose_grid_coverage"]["near_zero_target_row_count"] == 2


def test_site_description_binding_excludes_only_unbound_near_zero_targets(tmp_path):
    """Scenario 1: SITE dose reference names PTV1; PTV2/CTV2 are near zero and unbound."""
    course, frame, qc = _run(
        tmp_path, references=[fx.dose_reference(structure_type="SITE", description="PTV1")]
    )
    rows = {row["ROI_Name"]: row for row in frame.to_dict("records")}

    for name in UNBOUND_TARGETS:
        row = rows[name]
        assert row["dose_response_quarantine_status"] == EXCLUDED_TARGET_NOT_BOUND_STATUS
        assert not row["dose_response_eligible"]
        assert not row["dose_metric_usable_for_dose_response"]
        assert row["dose_metric_status"] == EXCLUDED_TARGET_NOT_BOUND_STATUS
        assert row["relative_metric_status"] == EXCLUDED_TARGET_NOT_BOUND_STATUS
        assert "PTV1" in row["dose_response_quarantine_reason"]
        assert "D95Gy <= 0.1 Gy" in row["dose_response_quarantine_reason"]
        assert all(pd.isna(row[column]) for column in RELATIVE_DVH_METRIC_COLUMNS)

    for name in {"PTV1", "CTV 1", "Bladder"}:
        row = rows[name]
        assert row["dose_response_eligible"]
        assert row["dose_metric_usable_for_dose_response"]
        assert pd.isna(row["dose_response_quarantine_status"])
        assert row["relative_metric_status"] == "computed"
        assert row["Dose_Response_Eligible"]
    assert rows["PTV1"]["D95%"] == pytest.approx(100.05)

    # Physical measurements are those ae60f01 published for the quarantined course.
    baseline = _baseline_rows(QUARANTINE_BASELINE)
    for name, row in rows.items():
        for column in frame.columns:
            if _is_dose_derived_metric_column(column) and column not in RELATIVE_DVH_METRIC_COLUMNS:
                expected = baseline[name][column]
                actual = row[column]
                assert (expected is None and pd.isna(actual)) or repr(float(actual)) == expected

    reconciliation = qc["plan_target_reconciliation"]
    assert reconciliation["decision"] == "unbound_near_zero_targets_excluded"
    assert reconciliation["quarantine_rules"] == []
    assert [(t["roi_name"], t["binding_methods"]) for t in reconciliation["plan_bound_targets"]] == [
        ("PTV1", ["dose_reference_description"])
    ]
    assert {t["roi_name"] for t in reconciliation["unbound_near_zero_targets"]} == UNBOUND_TARGETS
    assert qc["target_near_zero_dose_grid_coverage"]["near_zero_target_row_count"] == 2
    curves = json.loads((course / "dvh_curves.json").read_text(encoding="utf-8"))["points"]
    assert {c["roi_name"]: c["dose_metric_status"] for c in curves}["PTV2"] == (
        EXCLUDED_TARGET_NOT_BOUND_STATUS
    )


def test_bound_near_zero_target_quarantines_the_course_as_before(tmp_path):
    """Scenario 2: the plan binds PTV2, which is near zero."""
    course, _frame, qc = _run(
        tmp_path,
        references=[
            fx.dose_reference(number=1, description="PTV1"),
            fx.dose_reference(number=2, description="ptv2 "),
        ],
    )
    _assert_course_quarantined_as_at_ae60f01(course, qc, "plan_bound_target_near_zero")


@pytest.mark.parametrize("description", ["Pelvis 20Gy", "PTV", "PTV1 boost", None])
def test_no_identifiable_plan_bound_target_quarantines_the_course(tmp_path, description):
    """Scenario 3: no exact description or ROI-number binding, including prefixes."""
    course, _frame, qc = _run(
        tmp_path, references=[fx.dose_reference(description=description)]
    )
    _assert_course_quarantined_as_at_ae60f01(course, qc, "no_plan_bound_target_identified")
    assert qc["plan_target_reconciliation"]["plan_bound_targets"] == []


def test_volume_reference_binds_by_referenced_roi_number(tmp_path):
    """Scenario 4: VOLUME dose reference to ROI 1 (PTV1) of the contracted RTSTRUCT."""
    _course, frame, qc = _run(
        tmp_path,
        references=[fx.dose_reference(structure_type="VOLUME", roi_number=1)],
    )
    rows = {row["ROI_Name"]: row for row in frame.to_dict("records")}
    assert {
        name for name, row in rows.items()
        if row["dose_response_quarantine_status"] == EXCLUDED_TARGET_NOT_BOUND_STATUS
    } == UNBOUND_TARGETS
    assert rows["PTV1"]["dose_response_eligible"]
    reconciliation = qc["plan_target_reconciliation"]
    assert reconciliation["decision"] == "unbound_near_zero_targets_excluded"
    assert reconciliation["roi_number_binding"][0]["status"] == "used"
    assert [(t["roi_name"], t["binding_methods"]) for t in reconciliation["plan_bound_targets"]] == [
        ("PTV1", ["dose_reference_roi_number"])
    ]


def test_course_without_near_zero_targets_matches_ae60f01_output(tmp_path):
    """Scenario 5: byte-identical canonical DVH table to the stored ae60f01 output."""
    course, _frame, qc = _run(tmp_path, rois=fx.NO_NEAR_ZERO_ROIS)
    assert fx.canonical_dvh_table(course) + "\n" == NO_NEAR_ZERO_BASELINE.read_text(
        encoding="utf-8"
    )
    assert qc["plan_target_reconciliation"]["decision"] == "not_required_no_near_zero_targets"
    assert qc["plan_target_reconciliation"]["course_quarantine"] is False


@pytest.mark.parametrize("referenced_uid", [fx.OTHER_RTSTRUCT_UID, None])
def test_roi_number_binding_requires_plan_reference_to_contracted_rtstruct(
    tmp_path, referenced_uid
):
    """Scenario 6: ROI number 1 is ignored when the plan names another RTSTRUCT."""
    course, _frame, qc = _run(
        tmp_path,
        references=[fx.dose_reference(structure_type="VOLUME", roi_number=1)],
        plan_referenced_rtstruct_uid=referenced_uid,
    )
    _assert_course_quarantined_as_at_ae60f01(course, qc, "no_plan_bound_target_identified")
    binding = qc["plan_target_reconciliation"]["roi_number_binding"][0]
    assert binding["status"] == (
        "not_used_plan_references_other_rtstruct"
        if referenced_uid
        else "not_used_plan_has_no_structure_set_reference"
    )
    assert binding["roi_number_reference_count"] == 1


# In-memory checks of the course rule on synthetic rows.

RS = "2.25.99.1"


def _row(name, number, d95, *, target=True, zero="not_zero", coverage="fully_covered",
         source="Manual", rs=RS):
    return {
        "ROI_Name": name, "ROI_OriginalName": name, "ROI_Number": number,
        "target_like": target, "D95Gy": d95, "zero_dose_status": zero,
        "dose_grid_coverage_status": coverage, "Segmentation_Source": source,
        "rtstruct_sop_instance_uid": rs,
        "dose_response_eligible": True, "dose_metric_usable_for_dose_response": True,
        "dose_metric_status": ("quarantined_near_zero_requires_reconciliation"
                               if zero != "not_zero" else "computed"),
        "relative_metric_status": ("quarantined_near_zero_requires_reconciliation"
                                   if zero != "not_zero" else "computed"),
        "dose_response_quarantine_status": (
            "pending_plan_target_reconciliation" if zero != "not_zero" else None),
    }


def _plans(*references, rs_refs=(RS,)):
    return {"plans": [{"plan_sop_instance_uid": "2.25.99.2", "plan_path": "plan.dcm",
                       "referenced_rtstruct_sop_instance_uids": list(rs_refs),
                       "dose_references": list(references)}],
            "unreadable_plans": []}


def _ref(description=None, roi_number=None, reference_type="TARGET"):
    return {"index": 0, "dose_reference_number": "1", "dose_reference_type": reference_type,
            "structure_type": "SITE", "description": description,
            "referenced_roi_number": roi_number}


def test_bound_target_not_fully_on_grid_quarantines_the_course():
    rows = [_row("PTV1", 1, None, coverage="partial_grid"),
            _row("PTV2", 2, 0.0, zero="zero_dose_in_grid")]
    result = reconcile_near_zero_plan_targets(rows, _plans(_ref("PTV1")), RS)
    assert result["course_quarantine"] is True
    assert result["quarantine_rules"] == ["plan_bound_target_not_measured_on_selected_dose_grid"]


def test_description_match_is_exact_after_whitespace_and_case_normalization():
    rows = [_row("PTV 1", 1, 20.0), _row("PTV2", 2, 0.0, zero="zero_dose_in_grid")]
    result = reconcile_near_zero_plan_targets(rows, _plans(_ref("  ptv \t 1 ")), RS)
    assert result["decision"] == "unbound_near_zero_targets_excluded"
    prefix = reconcile_near_zero_plan_targets(rows, _plans(_ref("PTV")), RS)
    assert prefix["course_quarantine"] is True


def test_unreadable_selected_plan_quarantines_the_course():
    rows = [_row("PTV1", 1, 20.0), _row("PTV2", 2, 0.0, zero="zero_dose_in_grid")]
    plans = _plans(_ref("PTV1"))
    plans["unreadable_plans"] = ["second.dcm: synthetic read failure"]
    result = reconcile_near_zero_plan_targets(rows, plans, RS)
    assert result["course_quarantine"] is True
    assert "selected_plan_unreadable" in result["quarantine_rules"]


def test_derived_copy_of_a_roi_number_bound_target_is_never_excluded():
    rows = [_row("PTV1", 1, 20.0),
            _row("PTV1", 7, 0.0, zero="zero_dose_in_grid", source="Merged", rs="2.25.99.9"),
            _row("PTV2", 2, 0.0, zero="zero_dose_in_grid")]
    result = reconcile_near_zero_plan_targets(rows, _plans(_ref(roi_number=1)), RS)
    assert result["course_quarantine"] is True
    assert "plan_bound_target_near_zero" in result["quarantine_rules"]


def test_organ_at_risk_reference_alone_does_not_anchor_the_plan_target_set():
    rows = [_row("PTV1", 1, 20.0), _row("PTV2", 2, 0.0, zero="zero_dose_in_grid")]
    result = reconcile_near_zero_plan_targets(
        rows, _plans(_ref("PTV1", reference_type="ORGAN_AT_RISK")), RS
    )
    assert result["quarantine_rules"] == ["no_plan_bound_target_identified"]


def test_exclusion_keeps_absolute_values_and_touches_only_unbound_rows():
    rows = [_row("PTV1", 1, 20.0), _row("PTV2", 2, 0.0, zero="zero_dose_in_grid")]
    rows[1]["DmeanGy"] = 9.0
    before = dict(rows[0])
    result = reconcile_near_zero_plan_targets(rows, _plans(_ref("PTV1")), RS)
    apply_unbound_target_exclusion(rows, result)
    assert rows[0] == before
    assert rows[1]["DmeanGy"] == 9.0
    assert rows[1]["D95Gy"] == 0.0
    assert rows[1]["dose_response_quarantine_status"] == EXCLUDED_TARGET_NOT_BOUND_STATUS
    assert rows[1]["dose_response_eligible"] is False


def _audit_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_course_dose_completeness.py"
    spec = importlib.util.spec_from_file_location("audit_course_dose_completeness_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("description", "expected_status", "pending_rows", "excluded_rows"),
    [("PTV1", "clear", 0, 2), ("PTV2", "pending_plan_target_reconciliation", 2, 0)],
)
def test_dose_completeness_audit_accepts_governed_unbound_exclusion(
    tmp_path, description, expected_status, pending_rows, excluded_rows
):
    course, _frame, _qc = _run(tmp_path, references=[fx.dose_reference(description=description)])
    evidence = _audit_module()._near_zero_target_evidence(course)
    assert evidence["status"] == expected_status
    assert evidence["row_count"] == pending_rows
    assert len(evidence["excluded_target_not_bound_to_course_plan_rows"]) == excluded_rows
