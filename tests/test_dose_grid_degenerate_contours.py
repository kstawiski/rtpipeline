"""An area-less contour must not make dose-grid coverage undeterminable.

_dose_grid_contour_coordinates counted a planar item with fewer than three
points as "unresolved", and one such item made classify_zero_dose_roi_geometry
return zero_dose_geometry_unresolved for the whole ROI. That became
dose_grid_coverage_status=coverage_unresolved, which suppressed the ROI's
prescription-relative DVH metrics.

In one campaign that affected 2,926 of 12,353 DVH rows -- the single largest
contributor to the treatment-technique criterion failing. The items are the same
mask-to-contour conversion artifacts already handled in radiomics: one or two
points, bounding nothing.
"""

import numpy as np
import pytest
from pydicom.dataset import Dataset

from rtpipeline.dvh import _dose_grid_contour_coordinates


def _contour(points, kind="CLOSED_PLANAR"):
    item = Dataset()
    item.ContourGeometricType = kind
    item.ContourData = [float(v) for p in points for v in p]
    item.NumberOfContourPoints = len(points)
    return item


def _rtstruct(*contours, roi_number=1):
    roi = Dataset()
    roi.ReferencedROINumber = roi_number
    roi.ContourSequence = list(contours)
    ds = Dataset()
    ds.ROIContourSequence = [roi]
    return ds


def _square(z=0.0):
    return _contour([(0, 0, z), (10, 0, z), (10, 10, z), (0, 10, z)])


def test_a_degenerate_planar_item_is_skipped_not_unresolved():
    ds = _rtstruct(_square(0.0), _square(2.0), _contour([(5, 5, 4.0), (5, 5, 4.0)]))

    contours, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 0, "one area-less item must not block coverage"
    assert len(contours) == 2


def test_a_single_point_planar_item_is_skipped():
    ds = _rtstruct(_square(0.0), _contour([(1, 1, 1.0)]))

    contours, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 0
    assert len(contours) == 1


def test_a_genuine_point_roi_is_still_supported():
    ds = _rtstruct(_contour([(1, 1, 1.0)], kind="POINT"))

    contours, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 0
    assert len(contours) == 1


def test_non_finite_coordinates_remain_unresolved():
    bad = _contour([(0, 0, 0), (1, 1, 0), (2, 2, 0)])
    bad.ContourData = [0.0, 0.0, 0.0, float("nan"), 1.0, 0.0, 2.0, 2.0, 0.0]
    ds = _rtstruct(_square(0.0), bad)

    contours, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 1, "a malformed item that is not merely area-less still counts"
    assert len(contours) == 1


def test_a_ragged_coordinate_triple_remains_unresolved():
    bad = _contour([(0, 0, 0)])
    bad.ContourData = [0.0, 0.0]
    ds = _rtstruct(_square(0.0), bad)

    _, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 1


def test_an_unsupported_geometric_type_with_area_remains_unresolved():
    ds = _rtstruct(_square(0.0), _contour([(0, 0, 0), (1, 0, 0), (1, 1, 0)], kind="OPEN_NONPLANAR"))

    _, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 1


@pytest.mark.parametrize("n_degenerate", [1, 3, 7])
def test_many_degenerate_items_still_leave_coverage_determinable(n_degenerate):
    items = [_square(float(i)) for i in range(3)]
    items += [_contour([(9, 9, 50.0)]) for _ in range(n_degenerate)]
    ds = _rtstruct(*items)

    contours, unresolved = _dose_grid_contour_coordinates(ds, 1)

    assert unresolved == 0
    assert len(contours) == 3
