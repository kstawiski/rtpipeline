"""Robustness must not die on ROIs outside its selection (D22a, corrected).

Production case: course 477918/2025-11 closed its robustness stage (and
stopped the workflow) because the auto ROI adrenal_gland_right carried
ROI_CONTOUR_PARTIALLY_UNPARSEABLE, although the configured
apply_to_structures selection (GTV*/CTV*/PTV*) never included it.
Per-organ, not per-course: selected targets stay fail-closed via
ANALYSIS_REQUIRED; anything else records into the source sink and
extraction continues, with best_effort left False so whole-source
failures stay fatal.

Unlike the first version of this file, these tests exercise the
production wiring (selection expansion + tolerate_unselected call
shape), not just the pre-existing mask loader in isolation: the
reviewer showed the first version passed identically without the fix.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pydicom
import pytest

from rtpipeline.radiomics import _rtstruct_masks
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rtpipeline.radiomics_robustness import _robustness_selection_requiredness
from rtpipeline.roi_requiredness import Requiredness, inspect_rtstruct
from test_science_batch_c import _build_real_rtstruct

SELECTION = ["GTV*", "CTV*", "PTV*"]


def _two_roi_rtstruct(tmp_path: Path) -> tuple[Path, Path]:
    ct = tmp_path / "ct"
    ct.mkdir()
    rtstruct = _build_real_rtstruct(ct, n_slices=4, side=12)
    mask = np.zeros((12, 12, 4), dtype=bool)
    mask[2:5, 2:5, 1:3] = True
    rtstruct.add_roi(mask=mask, name="adrenal_gland_right")
    source = tmp_path / "RS.dcm"
    rtstruct.save(str(source))
    # Corrupt one adrenal contour while leaving a valid one, so the inventory
    # reports PARTIALLY_UNPARSEABLE. The corruption must bound area -- a
    # non-planar quad -- because an invalid item enclosing nothing is a
    # mask-to-contour artifact that the shared validator drops.
    ds = pydicom.dcmread(str(source))
    names = [r.ROIName for r in ds.StructureSetROISequence]
    numbers = [r.ROINumber for r in ds.StructureSetROISequence]
    item = next(
        i for i in ds.ROIContourSequence
        if names[numbers.index(i.ReferencedROINumber)] == "adrenal_gland_right"
    )
    bad = copy.deepcopy(item.ContourSequence[0])
    bad.ContourData = [1.0, 1.0, 0.0, 5.0, 1.0, 3.0, 5.0, 5.0, 0.0, 1.0, 5.0, 4.0]
    bad.NumberOfContourPoints = 4
    item.ContourSequence.append(bad)
    ds.save_as(str(source))
    return ct, source


def _production_call(ct, source, *, requiredness, sink):
    """The exact call shape robustness_for_course uses (wiring under test)."""
    return _rtstruct_masks(
        ct,
        source,
        failure_outcomes=sink,
        tolerate_unselected=requiredness is not None,
        requiredness_by_roi=requiredness,
        unmeasurable_required_is_disposition=True,
    )


def test_selection_expansion_marks_only_selected_required(tmp_path):
    _, source = _two_roi_rtstruct(tmp_path)
    requiredness = _robustness_selection_requiredness(source, SELECTION)
    assert requiredness is not None
    # PTV is the synthetic target; the adrenal is outside the selection.
    assert requiredness["PTV"] == Requiredness.ANALYSIS_REQUIRED
    assert requiredness["adrenal_gland_right"] == Requiredness.INVENTORY_ONLY


def test_selection_expansion_is_case_insensitive(tmp_path):
    _, source = _two_roi_rtstruct(tmp_path)
    requiredness = _robustness_selection_requiredness(source, ["ptv*"])
    assert requiredness is not None
    assert requiredness["PTV"] == Requiredness.ANALYSIS_REQUIRED


def test_empty_selection_and_uninspectable_source_fail_closed(tmp_path):
    _, source = _two_roi_rtstruct(tmp_path)
    assert _robustness_selection_requiredness(source, []) is None
    assert _robustness_selection_requiredness(source, None) is None
    assert (
        _robustness_selection_requiredness(tmp_path / "absent.dcm", SELECTION)
        is None
    )


def test_production_wiring_tolerates_unselected_failure(tmp_path):
    """The reviewer's empirical protocol as a regression test."""
    ct, source = _two_roi_rtstruct(tmp_path)
    codes = {o.name: o.structural_code for o in inspect_rtstruct(source).named_rois}
    assert codes["PTV"] is None
    assert codes["adrenal_gland_right"] == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"

    requiredness = _robustness_selection_requiredness(source, SELECTION)
    assert requiredness is not None
    sink: list = []
    masks = _production_call(ct, source, requiredness=requiredness, sink=sink)
    assert "PTV" in masks
    assert "adrenal_gland_right" not in masks
    assert [
        (entry["roi_name"], entry.get("structural_code")) for entry in sink
    ] == [("adrenal_gland_right", "ROI_CONTOUR_PARTIALLY_UNPARSEABLE")]


def test_production_wiring_selected_structural_status_is_disposition(tmp_path):
    """Changed 2026-09-24: a selected ROI with an unreadable contour is a
    governed structural non-measurement in robustness, not a course failure.
    The main-radiomics call shape (no opt-in flag) still raises."""
    ct, source = _two_roi_rtstruct(tmp_path)
    requiredness = _robustness_selection_requiredness(
        source, ["PTV", "adrenal_gland_right"]
    )
    sink: list = []
    masks = _production_call(ct, source, requiredness=requiredness, sink=sink)
    assert "PTV" in masks and "adrenal_gland_right" not in masks
    assert [(e["roi_name"], e["status"], e["failure_kind"], e["structural_code"])
            for e in sink] == [(
        "adrenal_gland_right", "structural_nonmeasurement",
        "unmeasurable_source_contour", "ROI_CONTOUR_PARTIALLY_UNPARSEABLE",
    )]
    with pytest.raises(RadiomicsCourseExtractionError, match="PARTIALLY_UNPARSEABLE"):
        _rtstruct_masks(ct, source, failure_outcomes=[], tolerate_unselected=True,
                        requiredness_by_roi=requiredness)


def test_production_wiring_without_expansion_stays_fatal(tmp_path):
    """No map, no tolerance: the legacy fail-closed shape is preserved."""
    ct, source = _two_roi_rtstruct(tmp_path)
    with pytest.raises(RadiomicsCourseExtractionError, match="PARTIALLY_UNPARSEABLE"):
        _production_call(ct, source, requiredness=None, sink=[])
