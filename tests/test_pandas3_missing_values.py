"""Pandas-3 missing-value regression: parquet round trips must not write 'nan'.

Uses invented ROI names and synthetic identifiers only. No patient data.
"""
import math

import pandas as pd
import pytest

from rtpipeline.missing_values import is_missing_value, text_or, value_or


def test_missing_value_helper_treats_pandas_scalars_as_missing():
    assert is_missing_value(None)
    assert is_missing_value(float("nan"))
    assert is_missing_value(pd.NA)
    assert is_missing_value(pd.NaT)
    assert not is_missing_value("")
    assert not is_missing_value("success")
    assert not is_missing_value(0)
    assert not is_missing_value(0.0)
    assert not is_missing_value(False)
    assert not is_missing_value({})
    assert value_or(float("nan"), "fallback") == "fallback"
    assert value_or("kept", "fallback") == "kept"
    assert value_or(0, "fallback") == 0
    assert text_or({"reason_code": float("nan")}, "reason_code", "fallback") == "fallback"
    assert text_or({"reason_code": pd.NA}, "reason_code", "fallback") == "fallback"
    assert text_or({}, "reason_code", "fallback") == "fallback"
    assert text_or({"reason_code": "ROI_MASK_BELOW_MIN_VOXELS"}, "reason_code", "fallback") == "ROI_MASK_BELOW_MIN_VOXELS"


def test_conda_ledger_survives_parquet_round_trip_with_absent_reason(tmp_path):
    from rtpipeline.radiomics_conda import _write_conda_roi_ledger

    course_dir = tmp_path / "SYNTHPAT" / "SYNTHCOURSE"
    (course_dir / "metadata").mkdir(parents=True)
    measured = {
        "roi_original_name": "SYNTH_ROI_ALPHA",
        "roi_name": "SYNTH_ROI_ALPHA",
        "extraction_status": "success",
        "extraction_status_detail": "",
        "segmentation_source": "SYNTH_SOURCE",
        "series_uid": "1.2.3.4.5.100",
        "mask_identity": "synth-mask-alpha",
        "stable_roi_identifier": "synth-stable-alpha",
        "roi_structural_code": "extracted",
        "reason_code": "extracted",
    }
    disposition = {
        "roi_original_name": "SYNTH_ROI_BETA",
        "roi_name": "SYNTH_ROI_BETA",
        "extraction_status": "failed",
        "extraction_status_detail": "synthetic failure detail",
        "segmentation_source": "SYNTH_SOURCE",
        "series_uid": "1.2.3.4.5.100",
        "mask_identity": "synth-mask-beta",
        "stable_roi_identifier": "synth-stable-beta",
        "roi_structural_code": "ROI_CONTOUR_UNPARSEABLE",
        # No reason_code key: pd.DataFrame fills the cell, and pandas 3
        # returns float nan for it after a parquet round trip.
    }
    frame = pd.DataFrame([measured, disposition])
    assert frame["reason_code"].isna().iloc[1]
    parquet = tmp_path / "mr.parquet"
    frame.to_parquet(parquet, index=False)
    rows = pd.read_parquet(parquet).to_dict("records")
    missing = rows[1].get("reason_code")
    assert missing is None or (isinstance(missing, float) and math.isnan(missing)) or missing is pd.NA or pd.isna(missing)

    tasks = [
        {"roi_name": "SYNTH_ROI_ALPHA", "metadata": dict(measured)},
        {"roi_name": "SYNTH_ROI_BETA", "metadata": dict(disposition)},
    ]
    _write_conda_roi_ledger(
        course_dir,
        tasks,
        rows,
        extracted=True,
        expected_names=["SYNTH_ROI_ALPHA", "SYNTH_ROI_BETA"],
        modality="MR",
    )
    import json

    ledger = json.loads((course_dir / "metadata" / "radiomics_mr_roi_ledger.json").read_text())
    by_roi = {row["roi_name"]: row for row in ledger["course_roi"]}
    assert by_roi["SYNTH_ROI_ALPHA"]["reason_code"] == "extracted"
    assert by_roi["SYNTH_ROI_BETA"]["reason_code"] == "ROI_CONTOUR_UNPARSEABLE"
    assert all(row["reason_code"] != "nan" for row in ledger["course_roi"])


def _parallel_task(roi_name):
    from rtpipeline.radiomics_parallel import _RoiTask
    return _RoiTask(
        source="SYNTH_SOURCE",
        rs_path="synth-rs",
        roi_name=roi_name,
        course_dir="synth-course",
        series_uid="1.2.3.4.5.100",
        mask_identity="synth-mask",
        stable_roi_identifier="synth-stable",
        decision=None,
        run_identifier="synth-run",
        code_revision="synth-rev",
        configured_parameter_hashes={},
        effective_parameter_hashes={},
    )


def test_parallel_ledger_survives_parquet_round_trip_with_absent_fields(tmp_path):
    from rtpipeline.radiomics_parallel import _write_parallel_roi_ledger

    course_dir = tmp_path / "SYNTHPAT" / "SYNTHCOURSE"
    (course_dir / "metadata").mkdir(parents=True)
    measured = {
        "roi_original_name": "SYNTH_ROI_GAMMA",
        "roi_name": "SYNTH_ROI_GAMMA",
        "extraction_status": "success",
        "segmentation_source": "SYNTH_SOURCE",
    }
    failed = {
        "roi_original_name": "SYNTH_ROI_DELTA",
        "roi_name": "SYNTH_ROI_DELTA",
        "extraction_status": "failed",
        "extraction_status_detail": "synthetic failure detail",
        "segmentation_source": "SYNTH_SOURCE",
        # roi_structural_code absent: pandas 3 round trip yields nan.
    }
    frame = pd.DataFrame([measured, failed])
    parquet = tmp_path / "ct.parquet"
    frame.to_parquet(parquet, index=False)
    rows = pd.read_parquet(parquet).to_dict("records")
    tasks = [_parallel_task("SYNTH_ROI_GAMMA"), _parallel_task("SYNTH_ROI_DELTA")]
    _write_parallel_roi_ledger(course_dir, tasks, rows, (), extracted=True)
    import json

    ledger = json.loads((course_dir / "metadata" / "radiomics_ct_roi_ledger.json").read_text())
    by_roi = {row["roi_name"]: row for row in ledger["course_roi"]}
    assert by_roi["SYNTH_ROI_GAMMA"]["reason_code"] == "extracted"
    assert by_roi["SYNTH_ROI_DELTA"]["reason_code"] == "failed_radiomics_extraction"
    assert all(row["reason_code"] != "nan" for row in ledger["course_roi"])
