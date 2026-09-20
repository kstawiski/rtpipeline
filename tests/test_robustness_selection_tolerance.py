"""A source holding nothing the selection asks for cannot void the course.

Robustness measures only the selected structures. When a structure set holds
none of them it contributes nothing -- but it used to be loaded with zero
tolerance, which made every structurally flagged ROI fatal. An organ well
outside the scan, whose auto-derived contour is degenerate, then voided a course
it had no bearing on: 12 of 230 courses in one campaign aborted on
adrenal_gland_left, costal_cartilages or humerus_right.
"""

from pathlib import Path

import pytest

from rtpipeline.radiomics_robustness import _robustness_selection_requiredness
from rtpipeline.roi_requiredness import Requiredness

from test_roi_requiredness import write_synthetic_rtstruct


def _rtstruct(tmp_path: Path, *names: str) -> Path:
    return write_synthetic_rtstruct(tmp_path / "RS.dcm", roi_names=names)


def test_a_source_without_any_selected_roi_is_tolerated_not_fatal(tmp_path):
    path = _rtstruct(tmp_path, "adrenal_gland_left", "costal_cartilages")

    result = _robustness_selection_requiredness(path, ["PTV*", "urinary_bladder"])

    assert result is not None, "returning None loads with zero tolerance"
    assert set(result) == {"adrenal_gland_left", "costal_cartilages"}
    assert all(v == Requiredness.INVENTORY_ONLY for v in result.values())


def test_selected_names_are_still_required(tmp_path):
    path = _rtstruct(tmp_path, "PTV_4500", "adrenal_gland_left")

    result = _robustness_selection_requiredness(path, ["PTV*"])

    assert result["PTV_4500"] == Requiredness.ANALYSIS_REQUIRED
    assert result["adrenal_gland_left"] == Requiredness.INVENTORY_ONLY


def test_an_uninspectable_source_stays_fatal(tmp_path):
    """A source that cannot be read is a whole-source failure, not tolerance."""

    missing = tmp_path / "does_not_exist.dcm"

    assert _robustness_selection_requiredness(missing, ["PTV*"]) is None


def test_an_empty_selection_stays_fatal(tmp_path):
    """No selection at all is a configuration error, not a licence to tolerate."""

    path = _rtstruct(tmp_path, "PTV_4500")

    assert _robustness_selection_requiredness(path, []) is None
    assert _robustness_selection_requiredness(path, None) is None


@pytest.mark.parametrize("pattern", ["ptv*", "PTV*", "PtV*"])
def test_pattern_matching_is_case_insensitive(tmp_path, pattern):
    path = _rtstruct(tmp_path, "PTV_4500")

    result = _robustness_selection_requiredness(path, [pattern])

    assert result["PTV_4500"] == Requiredness.ANALYSIS_REQUIRED
