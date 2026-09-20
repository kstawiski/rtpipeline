"""A contour that bounds no area cannot make an ROI unparseable.

Converting a segmentation mask to contours emits a one- or two-point item for a
stray boundary voxel. Such an item bounds nothing and the rasteriser draws
nothing from it, yet it used to mark the whole ROI
ROI_CONTOUR_PARTIALLY_UNPARSEABLE and void the course. In one campaign that cost
19 courses on urinary_bladder -- the study target -- where 533 contours were
valid and all 28 invalid ones had one or two points.

An invalid item that *does* bound area is different: real geometry could not be
read, and the ROI stays partially unparseable.
"""

import numpy as np
import pytest
from pydicom.dataset import Dataset

from rtpipeline.rtstruct_geometry import (
    contour_encloses_area,
    roi_geometry_code,
)


def _contour(points, kind="CLOSED_PLANAR"):
    item = Dataset()
    item.ContourGeometricType = kind
    flat = [float(v) for p in points for v in p]
    item.ContourData = flat
    item.NumberOfContourPoints = len(points)
    return item


def _square(z=0.0):
    return _contour([(0, 0, z), (10, 0, z), (10, 10, z), (0, 10, z)])


def test_a_one_or_two_point_item_bounds_no_area():
    assert not contour_encloses_area(_contour([(0, 0, 0)]))
    assert not contour_encloses_area(_contour([(0, 0, 0), (1, 1, 0)]))
    assert not contour_encloses_area(_contour([(2, 2, 0), (2, 2, 0), (2, 2, 0)]))
    assert contour_encloses_area(_square())


def test_degenerate_slices_do_not_make_the_roi_unparseable():
    contours = [_square(0.0), _square(2.0), _contour([(5, 5, 4.0), (5, 5, 4.0)])]

    assert roi_geometry_code(contours) is None


def test_an_invalid_item_that_bounds_area_still_fails():
    """Non-planar but real geometry means something could not be read."""

    skewed = _contour([(0, 0, 0), (10, 0, 5), (10, 10, 0), (0, 10, 9)])
    contours = [_square(0.0), skewed]

    assert roi_geometry_code(contours) == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"


def test_all_items_degenerate_is_still_unparseable():
    contours = [_contour([(0, 0, 0)]), _contour([(1, 1, 0), (2, 2, 0)])]

    assert roi_geometry_code(contours) == "ROI_CONTOUR_UNPARSEABLE"


def test_an_empty_sequence_is_unchanged():
    assert roi_geometry_code([]) == "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE"


def test_a_clean_roi_is_unchanged():
    assert roi_geometry_code([_square(0.0), _square(2.0)]) is None


@pytest.mark.parametrize("n_degenerate", [1, 2, 5])
def test_the_surviving_geometry_decides_the_kind(n_degenerate):
    contours = [_square(float(i)) for i in range(4)]
    contours += [_contour([(9, 9, 99.0)]) for _ in range(n_degenerate)]

    assert roi_geometry_code(contours) is None
