from pathlib import Path

import pandas as pd
import pytest

from rtpipeline.dvh_aggregate import (
    DVH_AGGREGATE_SCHEMA_VERSION,
    build_dvh_aggregate,
)


def test_dvh_aggregate_retains_computed_rows_and_failure_rows() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "course_id": ["C1"],
            "ROI_Number": [1],
            "ROI_Name": ["PTV"],
            "DmaxGy": [50.0],
            "structure_cropped": [False],
            "treatment_technique": ["EBRT"],
            "rtstruct_sop_instance_uid": ["1.2.3"],
            "rtstruct_path": ["RS.dcm"],
        }
    )
    courses = [("P1", "C1", Path("P1/C1")), ("P2", "C2", Path("P2/C2"))]

    aggregate = build_dvh_aggregate(
        [frame],
        courses,
        incomplete={("P2", "C2"): ["dvh_metrics.xlsx is missing"]},
    )

    assert len(aggregate) == 2
    failure = aggregate.loc[aggregate["course_id"] == "C2"].iloc[0]
    assert failure["row_status"] == "failed"
    assert "dvh_metrics.xlsx" in failure["failure_reason"]
    assert failure["structure_provenance_status"] == "not_available"
    assert failure["Prescribed_Dose_Status"] == "not_available"
    assert failure["Prescribed_Dose_Reason"]
    assert failure["Delivered_Dose_Status"] == "not_available"
    assert failure["Delivered_Dose_Reason"]
    assert aggregate["DmaxGy"].dtype.name == "Float64"
    assert aggregate["ROI_Number"].dtype.name == "Float64"
    assert aggregate["structure_cropped"].dtype.name == "boolean"
    assert set(aggregate["aggregate_schema_version"].dropna()) == {
        "rtpipeline-dvh-aggregate-v3"
    }
    assert DVH_AGGREGATE_SCHEMA_VERSION == "rtpipeline-dvh-aggregate-v3"


def test_dvh_aggregate_records_expected_not_computed_course() -> None:
    aggregate = build_dvh_aggregate(
        [],
        [("P1", "C1", Path("P1/C1"))],
        expected_noncomputed={("P1", "C1"): "no authoritative dose grid"},
    )

    row = aggregate.iloc[0]
    assert row["row_status"] == "not_computed"
    assert row["failure_reason"] == "no authoritative dose grid"
    assert row["dose_metric_status"] == "not_available"
    assert row["Dose_Plan_Scope_Status"] == "not_available"


def test_a_not_computed_course_states_its_ineligibility_rather_than_leaving_it_unknown() -> None:
    """NA claims eligibility is unknown. For a course with no dose grid it is not.

    The two booleans were simply unset, so the schema filled them with NA. A
    consumer then had to read "unknown" on the one row whose eligibility is
    least in doubt; in one campaign that was 20 of 122 courses, and it read as
    40 unresolved eligibility cells rather than 40 settled ones.
    """

    aggregate = build_dvh_aggregate(
        [],
        [("P1", "C1", Path("P1/C1"))],
        expected_noncomputed={("P1", "C1"): "no authoritative dose grid"},
    )

    row = aggregate.iloc[0]
    assert row["dose_response_eligible"] is False or row["dose_response_eligible"] == False  # noqa: E712
    assert row["dose_metric_usable_for_dose_response"] is False or row["dose_metric_usable_for_dose_response"] == False  # noqa: E712
    assert not pd.isna(row["dose_response_eligible"])
    assert not pd.isna(row["dose_metric_usable_for_dose_response"])


def test_dvh_aggregate_preserves_target_geometry_and_null_nonmeasurement() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "course_id": ["C1"],
            "ROI_Number": [11],
            "ROI_Name": ["PTV 1"],
            "ROI_OriginalName": ["PTV 1"],
            "ROI_Interpreted_Type": ["PTV"],
            "target_like": [True],
            "D95Gy": [0.0],
            "DmeanGy": [0.0],
            "V1Gy (cm³)": [0.0],
            "zero_dose_status": ["zero_dose_outside_dose_grid"],
            "zero_dose_reason": ["Outside selected grid."],
            "zero_dose_trigger_metric": ["D95Gy"],
            "zero_dose_trigger_value_gy": [0.01],
            "dose_metric_status": ["not_measurable_outside_dose_grid"],
            "dose_metric_reason": ["Outside selected grid."],
            "dose_metric_usable_for_dose_response": [False],
            "Dose_Plan_Scope_Status": ["partial_course_plan_set"],
            "Dose_Plan_Scope_Reason": ["One plan is unrepresented."],
            "Course_Treatment_Plan_Count": [2],
            "Dose_Grid_Plan_Count": [1],
            "Unrepresented_Treatment_Plan_Count": [1],
            "Course_Treatment_Isocenter_Status": ["multiple_plan_isocenters"],
            "Course_Treatment_Isocenter_Reason": [
                "Two RTPLAN isocenters are present."
            ],
            "Course_Treatment_Isocenter_Count": [2],
            "Course_Treatment_Isocenter_Max_Separation_mm": [233.137],
            "Course_Treatment_Isocenter_Readable_Plan_Count": [2],
            "Course_Target_Dose_Coverage_Status": [
                "near_zero_target_outside_selected_dose_grid"
            ],
        }
    )

    aggregate = build_dvh_aggregate(
        [frame], [("P1", "C1", Path("P1/C1"))]
    )
    row = aggregate.iloc[0]

    assert pd.isna(row["D95Gy"])
    assert pd.isna(row["DmeanGy"])
    assert pd.isna(row["V1Gy (cm³)"])
    assert row["zero_dose_status"] == "zero_dose_outside_dose_grid"
    assert row["zero_dose_trigger_value_gy"] == 0.01
    assert row["dose_metric_status"] == "not_measurable_outside_dose_grid"
    assert bool(row["target_like"]) is True
    assert bool(row["dose_metric_usable_for_dose_response"]) is False
    assert row["Dose_Plan_Scope_Status"] == "partial_course_plan_set"
    assert row["Course_Treatment_Plan_Count"] == 2.0
    assert row["Course_Treatment_Isocenter_Status"] == "multiple_plan_isocenters"
    assert row["Course_Treatment_Isocenter_Count"] == 2.0
    assert row["Course_Treatment_Isocenter_Max_Separation_mm"] == 233.137


def test_dvh_aggregate_keeps_producer_outside_grid_disposition() -> None:
    from rtpipeline.dvh import annotate_dvh_metrics

    published = annotate_dvh_metrics(
        {"Volume (cm³)": 12.0, "DmeanGy": 0.0, "D2ccGy": 0.0, "D0.03ccGy": 0.0},
        technique="EBRT",
        structure_name="PTV 1",
        structure_interpreted_type="PTV",
        prescription_resolved=True,
        dose_response_eligible=True,
        rtstruct_sop_instance_uid="1.2.3",
        rtstruct_path="RS.dcm",
        zero_dose_status="zero_dose_outside_dose_grid",
        zero_dose_reason="Outside grid.",
        grid_coverage={"status": "outside_grid", "fraction": 0.0},
    )
    reason = "Whole-ROI metrics require complete independent dose-grid support."
    assert published["dose_metric_status"] == "outside_grid"
    assert published["dose_metric_reason"] == reason
    # Aggregation must also discard stale numeric values on a declared nonmeasurement.
    published.update(patient_id="P1", course_id="C1", DmeanGy=3.0,
                     D90Gy=4.0, D2ccGy=50.0,
                     **{"D0.03ccGy": 7.5, "V1Gy (cm³)": 4.0,
                        "dose_metric_usable_for_dose_response": True})
    source = pd.DataFrame([published])
    aggregate = build_dvh_aggregate([source], [("P1", "C1", Path("P1/C1"))])
    row = aggregate.iloc[0]
    assert row["dose_grid_coverage_status"] == "outside_grid"
    assert row["dose_metric_status"] == "outside_grid"
    assert row["dose_metric_reason"] == reason
    assert row["zero_dose_status"] == "zero_dose_outside_dose_grid"
    assert row["Volume (cm³)"] == 12.0
    for column in ("DmeanGy", "D90Gy", "D2ccGy", "D0.03ccGy", "V1Gy (cm³)"):
        assert pd.isna(row[column])
    assert not bool(row["dose_metric_usable_for_dose_response"])
    assert aggregate.loc[aggregate["dose_metric_usable_for_dose_response"].fillna(False)].empty
    assert source.iloc[0]["DmeanGy"] == 3.0


@pytest.mark.parametrize("reason", ["Producer-specific outside-grid evidence.", None, pd.NA])
def test_dvh_aggregate_preserves_published_outside_grid_reason_verbatim(reason) -> None:
    frame = pd.DataFrame([{
        "patient_id": "P1", "course_id": "C1",
        "zero_dose_status": "zero_dose_outside_dose_grid",
        "dose_grid_coverage_status": "outside_grid",
        "dose_metric_status": "outside_grid", "dose_metric_reason": reason,
    }])
    row = build_dvh_aggregate([frame], []).iloc[0]
    assert row["dose_metric_status"] == "outside_grid"
    if pd.isna(reason):
        assert pd.isna(row["dose_metric_reason"])
    else:
        assert row["dose_metric_reason"] == reason


@pytest.mark.parametrize("published_status", [None, pd.NA, "computed", "not_measurable_outside_dose_grid"])
@pytest.mark.parametrize("coverage", [None, "outside_grid"])
def test_dvh_aggregate_keeps_legacy_outside_grid_behavior(published_status, coverage) -> None:
    record = {
        "patient_id": "P1", "course_id": "C1", "DmeanGy": 3.0,
        "D90Gy": 4.0, "D2ccGy": 50.0, "D0.03ccGy": 7.5, "Volume (cm³)": 12.0,
        "zero_dose_status": "zero_dose_outside_dose_grid",
        "dose_metric_reason": "Legacy reason to replace.",
        "dose_metric_usable_for_dose_response": True,
    }
    # None means the column was absent in an older export; pd.NA is a null cell.
    if published_status is not None:
        record["dose_metric_status"] = published_status
    if coverage is not None:
        record["dose_grid_coverage_status"] = coverage
    row = build_dvh_aggregate([pd.DataFrame([record])], []).iloc[0]
    assert row["dose_metric_status"] == "not_measurable_outside_dose_grid"
    assert row["dose_metric_reason"] == (
        "Dose-derived metrics are null because the ROI does not intersect "
        "the selected RTDOSE grid."
    )
    for column in ("DmeanGy", "D90Gy", "D2ccGy", "D0.03ccGy"):
        assert pd.isna(row[column])
    assert row["Volume (cm³)"] == 12.0
    assert not bool(row["dose_metric_usable_for_dose_response"])


@pytest.mark.parametrize("coverage", [None, pd.NA, "fully_covered", "partial_grid"])
def test_dvh_aggregate_requires_matching_canonical_outside_grid_coverage(coverage) -> None:
    record = {
        "zero_dose_status": "zero_dose_outside_dose_grid",
        "dose_metric_status": "outside_grid",
        "dose_metric_reason": "Unconfirmed outside-grid label.",
    }
    if coverage is not None:
        record["dose_grid_coverage_status"] = coverage
    row = build_dvh_aggregate([pd.DataFrame([record])], []).iloc[0]
    assert row["dose_metric_status"] == "not_measurable_outside_dose_grid"
    assert row["dose_metric_reason"].startswith("Dose-derived metrics are null")


def test_dvh_aggregate_handles_mixed_producer_and_legacy_rows() -> None:
    frame = pd.DataFrame({
        "patient_id": ["P1"] * 3, "course_id": ["C1"] * 3,
        "zero_dose_status": ["zero_dose_outside_dose_grid"] * 2 + ["not_zero_dose"],
        "dose_grid_coverage_status": ["outside_grid", "outside_grid", "fully_covered"],
        "dose_metric_status": ["outside_grid", pd.NA, "computed"],
        "dose_metric_reason": ["Producer evidence.", "Legacy evidence.", "Measured."],
        "DmeanGy": [0.0, 0.0, 20.0],
        "dose_metric_usable_for_dose_response": [True, True, True],
    }, index=[12, 18, 25])
    aggregate = build_dvh_aggregate([frame], [])
    assert aggregate["dose_metric_status"].tolist() == [
        "outside_grid", "not_measurable_outside_dose_grid", "computed",
    ]
    assert aggregate["dose_metric_reason"].iloc[0] == "Producer evidence."
    assert aggregate["dose_metric_reason"].iloc[2] == "Measured."
    assert aggregate["DmeanGy"].isna().tolist() == [True, True, False]
    assert aggregate["dose_metric_usable_for_dose_response"].tolist() == [False, False, True]


@pytest.mark.parametrize("published_status", ["outside_grid", "not_measurable_outside_dose_grid"])
def test_dvh_aggregate_nulls_exactly_the_producer_dose_metrics(published_status) -> None:
    from rtpipeline.dvh import RELATIVE_DVH_METRIC_COLUMNS, _is_dose_derived_metric_column

    metrics = {
        "DmeanGy", "DmaxGy", "DminGy", "D95Gy", "D90Gy", "D98Gy", "D2Gy", "D50Gy",
        "D2ccGy", "D1ccGy", "D0.1ccGy", "D0.03ccGy", "HI", "SpreadGy", "IntegralDose_Gycm3",
        *RELATIVE_DVH_METRIC_COLUMNS,
        "V0Gy (cm³)", "V1Gy (%)", "V2.5Gy (cm³)", "V12.75Gy (%)",
    }
    retained = {
        "Volume (cm³)", "Prescribed_Dose_Gy", "zero_dose_trigger_value_gy",
        "VnoteGy (cm³)", "V99%Rx (%)", "V1Gy (%)_note", "D2cc_status", "D0.03cc_status",
    }
    record = dict.fromkeys(metrics | retained, 12.0)
    record.update(
        patient_id="P1", course_id="C1", zero_dose_status="zero_dose_outside_dose_grid",
        dose_grid_coverage_status="outside_grid", dose_metric_status=published_status,
    )
    computed = dict(record, zero_dose_status="not_zero_dose", dose_metric_status="computed")
    source = pd.DataFrame([record, computed])
    aggregate = build_dvh_aggregate([source], [])
    for column in metrics | retained:
        assert _is_dose_derived_metric_column(column) == (column in metrics)
        assert pd.isna(aggregate.iloc[0][column]) == (column in metrics), column
        assert aggregate.iloc[1][column] == 12.0, column
        assert source.iloc[0][column] == 12.0, column


def _dvh_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "patient_id": "P1",
        "course_id": "C1",
        "ROI_Number": 79.0,
        "ROI_Name": "vertebrae_T2",
        "ROI_OriginalName": "vertebrae_T2",
        "row_status": "computed",
        "DmeanGy": 10.0,
        "DmaxGy": 12.0,
        "treatment_technique": "EBRT",
    }
    base.update(overrides)
    return base


def test_dvh_aggregate_collapses_identical_same_mask_double_measurement() -> None:
    """A support ROI copied into the published custom set is measured once
    per source file with identical values (observed: 20 such pairs with
    458/458 numeric fields identical). The census keeps one row."""
    auto = _dvh_row(
        Segmentation_Source="AutoRTS",
        rtstruct_path="Output/P1/C1/RS_auto.dcm",
    )
    merged = _dvh_row(
        Segmentation_Source="Merged",
        rtstruct_path="Output/P1/C1/RS_custom.dcm",
    )
    aggregate = build_dvh_aggregate(
        [pd.DataFrame([auto, merged])], [("P1", "C1", Path("P1/C1"))]
    )
    kept = aggregate.loc[
        (aggregate["patient_id"] == "P1") & (aggregate["ROI_Name"] == "vertebrae_T2")
    ]
    assert len(kept) == 1
    assert kept.iloc[0]["Segmentation_Source"] == "Merged"
    assert kept.iloc[0]["DmeanGy"] == 10.0


def test_dvh_aggregate_keeps_same_identity_rows_with_differing_values() -> None:
    """Pairs whose measured values differ are genuinely different masks and
    stay for the readiness gate to flag under row identity (fail-closed)."""
    auto = _dvh_row(Segmentation_Source="AutoRTS", DmeanGy=10.0)
    merged = _dvh_row(Segmentation_Source="Merged", DmeanGy=11.0)
    aggregate = build_dvh_aggregate(
        [pd.DataFrame([auto, merged])], [("P1", "C1", Path("P1/C1"))]
    )
    kept = aggregate.loc[
        (aggregate["patient_id"] == "P1") & (aggregate["ROI_Name"] == "vertebrae_T2")
    ]
    assert len(kept) == 2


def test_dvh_aggregate_dedup_tolerates_frames_without_identity_columns() -> None:
    """Aggregate inputs without ROI identity columns (e.g. synthetic or
    partial course frames) must pass through instead of raising inside
    the same-mask deduplication (observed AttributeError on scalar NaN)."""
    frame = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "course_id": "C1",
                "row_status": "computed",
                "DmeanGy": 1.0,
            }
        ]
    )
    aggregate = build_dvh_aggregate([frame], [("P1", "C1", Path("P1/C1"))])
    assert len(aggregate) == 1


def test_aggregate_keeps_producer_nonmeasurement_and_reason_fields():
    frame = pd.DataFrame([
        _dvh_row(row_status='not_computed', DmeanGy=None, DmaxGy=None,
                 dvh_nonmeasurement_class='declared_empty',
                 dose_grid_coverage_reason='ROI has no contour items.',
                 dose_grid_coverage_reason_code='ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'),
        _dvh_row(ROI_Number=80, row_status=None),
    ])
    aggregate = build_dvh_aggregate([frame], [('P1', 'C1', Path('P1/C1'))])
    empty = aggregate.loc[aggregate.ROI_Number == 79].iloc[0]
    assert empty.row_status == 'not_computed'
    assert empty.dose_grid_coverage_reason == 'ROI has no contour items.'
    assert empty.dose_grid_coverage_reason_code == 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'
    assert pd.isna(empty.DmeanGy)
    assert aggregate.loc[aggregate.ROI_Number == 80, 'row_status'].tolist() == ['computed']
