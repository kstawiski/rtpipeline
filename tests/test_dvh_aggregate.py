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
