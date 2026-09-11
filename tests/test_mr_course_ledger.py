"""Denominator-ledger accounting for the executed conda MR course helper.

Covers what ``rtpipeline.radiomics_conda.radiomics_for_course_mr`` records in
``metadata/radiomics_mr_roi_ledger.json``: one ROI measured from two series must
keep two source identities, a course that publishes a workbook must still carry
its technical failures, and modality absence must stay distinguishable from a
technical exclusion. Every input is generated in ``tmp_path`` by the source
contract module's synthetic producers; no conda subprocess, PyRadiomics run or
clinical input is involved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rtpipeline import radiomics_conda as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from test_mr_course_source_contract import (  # noqa: F401  (``batch`` is a fixture)
    OTHER_SERIES_UID,
    SERIES_UID,
    _Config,
    _contract,
    _make_course,
    _make_series,
    batch,
)


def _payload(course_dir: Path) -> Dict[str, Any]:
    return json.loads(
        (course_dir / "metadata" / "radiomics_mr_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )


def _roi_rows(course_dir: Path, roi_name: str) -> List[Dict[str, Any]]:
    return [
        row for row in _payload(course_dir)["course_roi"] if row["roi_name"] == roi_name
    ]


def _course_row(course_dir: Path) -> Dict[str, Any]:
    rows = _payload(course_dir)["course"]
    assert len(rows) == 1
    return rows[0]


def _empty_course(tmp_path: Path) -> Path:
    course_dir = tmp_path / "PAT" / "COURSE"
    course_dir.mkdir(parents=True)
    return course_dir


# --------------------------------------------------------------------------
# one ROI name is not one measurement
# --------------------------------------------------------------------------
def test_one_roi_in_two_series_keeps_both_source_identities(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    good = _make_series(course_dir, dir_name="mr_ok", rois=(("liver", 40),))
    broken = _make_series(
        course_dir,
        dir_name="mr_broken",
        series_uid=OTHER_SERIES_UID,
        rois=(("liver", 24),),
        image_name="mr_series_b",
        with_dicom=False,
    )

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is not None

    rows = _roi_rows(course_dir, "liver")
    assert len(rows) == 2
    by_reason = {row["reason_code"]: row for row in rows}
    assert set(by_reason) == {"extracted", "failed_source_read"}
    measured = by_reason["extracted"]
    assert measured["series_uid"] == SERIES_UID
    assert measured["mask_path_source"] == str(good["masks"]["liver"])
    assert measured["mask_identity"] == rc.file_sha256(good["masks"]["liver"])
    assert measured["source_content_sha256"] == rc.file_sha256(good["nifti_path"])
    failed = by_reason["failed_source_read"]
    assert failed["disposition"] == "excluded"
    assert failed["mask_path_source"] == str(broken["masks"]["liver"])
    assert failed["mask_identity"] != measured["mask_identity"]


def test_published_workbook_does_not_clear_a_technical_failure(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, dir_name="mr_ok", rois=(("liver", 40),))
    _make_series(
        course_dir,
        dir_name="mr_broken",
        series_uid=OTHER_SERIES_UID,
        rois=(("spleen", 40),),
        with_dicom=False,
    )

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is not None

    course_row = _course_row(course_dir)
    assert course_row["extracted"] is True
    assert course_row["technical_exclusion"] is True


# --------------------------------------------------------------------------
# a nonmeasurement is not a technical exclusion
# --------------------------------------------------------------------------
def test_absent_optional_mr_is_not_a_technical_exclusion(tmp_path, batch):
    course_dir = _empty_course(tmp_path)

    config = _Config(contract=_contract(optional=["liver"]))
    assert rc.radiomics_for_course_mr(course_dir, config) is None

    course_row = _course_row(course_dir)
    assert course_row["extracted"] is False
    assert course_row["technical_exclusion"] is False
    assert course_row["reason_code"] == "not_applicable_modality"
    assert _roi_rows(course_dir, "liver")[0]["reason_code"] == "not_applicable_modality"


def test_all_valid_empty_masks_are_not_a_technical_exclusion(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, rois=(("liver", 0), ("spleen", 0)))

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None

    course_row = _course_row(course_dir)
    assert course_row["extracted"] is False
    assert course_row["technical_exclusion"] is False
    assert course_row["reason_code"] == "not_computed_valid_empty_scope"


def test_unreadable_mask_stays_a_technical_exclusion(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir, rois=(("liver", 0), ("spleen", 40)))
    series["masks"]["spleen"].write_bytes(b"not a nifti")

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None

    course_row = _course_row(course_dir)
    assert course_row["extracted"] is False
    assert course_row["technical_exclusion"] is True
    assert _roi_rows(course_dir, "spleen")[0]["reason_code"] == "failed_source_read"
    assert (
        _roi_rows(course_dir, "liver")[0]["reason_code"]
        == "not_computed_valid_empty_scope"
    )


# --------------------------------------------------------------------------
# an unpublished course still names its sources
# --------------------------------------------------------------------------
def test_unpublished_dispositions_still_name_their_series_and_mask(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir, rois=(("liver", 0),))

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None

    row = _roi_rows(course_dir, "liver")[0]
    assert row["segmentation_source"] == "AutoTS_total_mr"
    assert row["series_uid"] == SERIES_UID
    assert row["nifti_path"] == str(series["nifti_path"])
    assert row["source_content_sha256"] == rc.file_sha256(series["nifti_path"])
    assert row["mask_path_source"] == str(series["masks"]["liver"])
    assert row["mask_identity"] == rc.file_sha256(series["masks"]["liver"])


# --------------------------------------------------------------------------
# required missing MR is accounted, not skipped
# --------------------------------------------------------------------------
def test_required_missing_mr_is_accounted_as_a_technical_exclusion(tmp_path, batch):
    course_dir = _empty_course(tmp_path)

    with pytest.raises(RadiomicsCourseExtractionError, match="liver"):
        rc.radiomics_for_course_mr(
            course_dir, _Config(contract=_contract(required=["liver"]))
        )

    assert batch.calls == 0
    row = _roi_rows(course_dir, "liver")[0]
    assert row["reason_code"] == "failed_source_read"
    assert row["disposition"] == "excluded"
    course_row = _course_row(course_dir)
    assert course_row["extracted"] is False
    assert course_row["technical_exclusion"] is True


def test_absent_mr_separates_required_failure_from_optional_absence(tmp_path, batch):
    course_dir = _empty_course(tmp_path)

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(
            course_dir,
            _Config(contract=_contract(required=["liver"], optional=["spleen"])),
        )

    assert _roi_rows(course_dir, "liver")[0]["reason_code"] == "failed_source_read"
    assert (
        _roi_rows(course_dir, "spleen")[0]["reason_code"] == "not_applicable_modality"
    )
