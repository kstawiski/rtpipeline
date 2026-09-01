from pathlib import Path

import pandas as pd

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
        DVH_AGGREGATE_SCHEMA_VERSION
    }


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
