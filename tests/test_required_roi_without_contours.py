"""An unused ROI name must not void a course that has measurable ROIs.

A planner may declare a structure and never contour it. That name carries no
contour data at all, so nothing can be extracted from it under any tolerance --
but it is an ROI that was never drawn, not a measurement that failed. Treating
a required one as fatal discarded every other contoured ROI in the course: a
campaign lost 92 of 122 courses, and 1197 measurable ROIs with them, to empty
placeholders named 'Pecherz' while a contoured urinary_bladder sat beside them.
"""

import pydicom
import pytest

from rtpipeline import radiomics
from rtpipeline.roi_requiredness import Requiredness

from test_roi_requiredness import write_synthetic_rtstruct


def _uncontoured(path):
    dataset = pydicom.dcmread(path)
    del dataset.ROIContourSequence
    dataset.save_as(path, write_like_original=False)
    return path


def test_a_required_uncontoured_roi_no_longer_raises(tmp_path):
    path = _uncontoured(
        write_synthetic_rtstruct(tmp_path / "RS.dcm", roi_names=("Pecherz",))
    )
    failures = []

    masks = radiomics._rtstruct_masks(
        tmp_path / "ct",
        path,
        best_effort=True,
        failure_outcomes=failures,
        requiredness_by_roi={"Pecherz": Requiredness.ANALYSIS_REQUIRED},
        contourless_required_is_absence=True,
    )

    assert masks == {}
    assert [f["roi_name"] for f in failures] == ["Pecherz"]
    assert failures[0]["status"] == "structural_nonmeasurement"
    assert failures[0]["failure_kind"] == "declared_without_contour_data"
    assert failures[0]["structural_code"] in {
        "ROI_DECLARED_NO_CONTOUR_ITEM",
        "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
    }


def test_a_generated_source_keeps_an_empty_required_roi_fatal(tmp_path):
    """An empty ROI from a generator means it produced nothing. Stay strict."""

    path = _uncontoured(
        write_synthetic_rtstruct(tmp_path / "RS.dcm", roi_names=("Pecherz",))
    )
    failures = []

    with pytest.raises(radiomics.RadiomicsCourseExtractionError, match="ROI_DECLARED_"):
        radiomics._rtstruct_masks(
            tmp_path / "ct",
            path,
            best_effort=True,
            failure_outcomes=failures,
            requiredness_by_roi={"Pecherz": Requiredness.ANALYSIS_REQUIRED},
            contourless_required_is_absence=False,
        )


def test_a_non_required_uncontoured_roi_stays_a_silent_skip(tmp_path):
    path = _uncontoured(
        write_synthetic_rtstruct(tmp_path / "RS.dcm", roi_names=("TemplateEmpty",))
    )
    failures = []

    masks = radiomics._rtstruct_masks(
        tmp_path / "ct",
        path,
        best_effort=True,
        failure_outcomes=failures,
        requiredness_by_roi={"TemplateEmpty": Requiredness.INVENTORY_ONLY},
    )

    assert masks == {}
    assert failures == []


def test_the_contourless_codes_are_exactly_the_no_data_cases():
    assert radiomics._CONTOURLESS_STRUCTURAL_CODES == frozenset(
        {"ROI_DECLARED_NO_CONTOUR_ITEM", "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE"}
    )


@pytest.mark.parametrize(
    "code", ["ROI_CONTOUR_PARTIALLY_UNPARSEABLE", "ROI_EXTRACTION_FAILED"]
)
def test_a_genuine_structural_failure_is_not_treated_as_absence(code):
    """Only "no contour data at all" is an absence; a broken contour is not."""

    assert code not in radiomics._CONTOURLESS_STRUCTURAL_CODES
