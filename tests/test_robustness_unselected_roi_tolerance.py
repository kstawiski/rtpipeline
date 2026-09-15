"""Robustness must not die on ROIs outside its selection (D22a).

Production case: course 477918/2025-11 closed its robustness stage (and
stopped the workflow) because the auto ROI adrenal_gland_right carried
ROI_CONTOUR_PARTIALLY_UNPARSEABLE, although robustness never selected
that ROI. Per-organ, not per-course: selected targets stay fail-closed
via ANALYSIS_REQUIRED; anything else records into the source sink and
extraction continues.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pydicom
import pytest

from rtpipeline.radiomics import _rtstruct_masks
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rtpipeline.roi_requiredness import inspect_rtstruct, Requiredness
from test_science_batch_c import _build_real_rtstruct


def _two_roi_rtstruct(tmp_path: Path) -> tuple[Path, Path]:
    ct = tmp_path / "ct"
    ct.mkdir()
    rtstruct = _build_real_rtstruct(ct, n_slices=4, side=12)
    mask = np.zeros((12, 12, 4), dtype=bool)
    mask[2:5, 2:5, 1:3] = True
    rtstruct.add_roi(mask=mask, name="adrenal_gland_right")
    source = tmp_path / "RS.dcm"
    rtstruct.save(str(source))
    # Corrupt one adrenal contour (2 points: invalid closed planar) while
    # leaving a valid contour, so inventory reports PARTIALLY_UNPARSEABLE.
    ds = pydicom.dcmread(str(source))
    names = [r.ROIName for r in ds.StructureSetROISequence]
    numbers = [r.ROINumber for r in ds.StructureSetROISequence]
    item = next(
        i for i in ds.ROIContourSequence
        if names[numbers.index(i.ReferencedROINumber)] == "adrenal_gland_right"
    )
    bad = copy.deepcopy(item.ContourSequence[0])
    bad.ContourData = [1.0, 1.0, 0.0, 2.0, 2.0, 0.0]
    bad.NumberOfContourPoints = 2
    item.ContourSequence.append(bad)
    ds.save_as(str(source))
    return ct, source


def test_unselected_structural_failure_is_recorded_not_raised(tmp_path):
    ct, source = _two_roi_rtstruct(tmp_path)
    codes = {o.name: o.structural_code for o in inspect_rtstruct(source).named_rois}
    assert codes["PTV"] is None
    assert codes["adrenal_gland_right"] == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"

    sink: list = []
    masks = _rtstruct_masks(
        ct,
        source,
        failure_outcomes=sink,
        best_effort=True,
        requiredness_by_roi={"PTV": Requiredness.ANALYSIS_REQUIRED},
    )
    assert "PTV" in masks
    assert "adrenal_gland_right" not in masks
    assert [
        (entry["roi_name"], entry.get("structural_code")) for entry in sink
    ] == [("adrenal_gland_right", "ROI_CONTOUR_PARTIALLY_UNPARSEABLE")]


def test_unscoped_call_still_fails_closed(tmp_path):
    """The legacy call shape (no selection scoping) keeps failing closed."""
    ct, source = _two_roi_rtstruct(tmp_path)
    with pytest.raises(RadiomicsCourseExtractionError, match="PARTIALLY_UNPARSEABLE"):
        _rtstruct_masks(ct, source, failure_outcomes=[])


def test_selected_structural_failure_stays_fatal(tmp_path):
    """A broken SELECTED ROI still fails the course even when scoped."""
    ct, source = _two_roi_rtstruct(tmp_path)
    with pytest.raises(RadiomicsCourseExtractionError, match="PARTIALLY_UNPARSEABLE"):
        _rtstruct_masks(
            ct,
            source,
            failure_outcomes=[],
            best_effort=True,
            requiredness_by_roi={
                "PTV": Requiredness.ANALYSIS_REQUIRED,
                "adrenal_gland_right": Requiredness.ANALYSIS_REQUIRED,
            },
        )
