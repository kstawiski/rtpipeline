"""An area-less contour cannot make an ROI's source series unresolvable.

roi_geometry_code drops an invalid item that bounds no area -- a mask-to-contour
artifact of one or two points. resolve_roi_scopes then walked the same contours
and counted each such item as a scope failure, so the ROI it had just judged
readable became ROI_UNRESOLVED_SOURCE_SCOPE. The defect moved one step
downstream rather than being fixed.

On one campaign course that was 26 of 73 auto-segmented ROIs, including
hip_left, hip_right and sacrum. Losing those lost the derived pelvic_bones
custom structure, and every course whose configuration required it aborted:
39 of 122 produced no radiomics at all.
"""

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from rtpipeline.rtstruct_geometry import resolve_roi_scopes

SERIES_UID = generate_uid()
FRAME_UID = generate_uid()


def _ct_slice(z: float) -> Dataset:
    image = Dataset()
    image.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    image.SOPInstanceUID = generate_uid()
    image.SeriesInstanceUID = SERIES_UID
    image.FrameOfReferenceUID = FRAME_UID
    image.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    image.ImagePositionPatient = [0.0, 0.0, float(z)]
    image.PixelSpacing = [1.0, 1.0]
    image.Rows = 64
    image.Columns = 64
    return image


def _contour(points, kind="CLOSED_PLANAR"):
    item = Dataset()
    item.ContourGeometricType = kind
    item.ContourData = [float(v) for p in points for v in p]
    item.NumberOfContourPoints = len(points)
    return item


def _square(z):
    return _contour([(5, 5, z), (20, 5, z), (20, 20, z), (5, 20, z)])


def _speck(z):
    """A stray boundary voxel: one point, bounding nothing."""
    return _contour([(10, 10, z)])


def _rtstruct(contours):
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "hip_left"
    roi.ReferencedFrameOfReferenceUID = FRAME_UID
    contour_roi = Dataset()
    contour_roi.ReferencedROINumber = 1
    contour_roi.ContourSequence = Sequence(list(contours))
    ds = Dataset()
    ds.StructureSetROISequence = Sequence([roi])
    ds.ROIContourSequence = Sequence([contour_roi])
    return ds


def _resolve(contours, slices):
    return resolve_roi_scopes(_rtstruct(contours), slices)[1]


def test_a_clean_roi_binds_one_source_series():
    slices = [_ct_slice(0.0), _ct_slice(1.0)]

    result = _resolve([_square(0.0), _square(1.0)], slices)

    assert result.code is None
    assert result.source_series_uids == (SERIES_UID,)


def test_specks_do_not_make_the_source_scope_unresolvable():
    """This is the defect: the ROI was readable, then lost its series."""

    slices = [_ct_slice(0.0), _ct_slice(1.0), _ct_slice(2.0)]

    result = _resolve([_square(0.0), _square(1.0), _speck(2.0)], slices)

    assert result.code is None, result.detail
    assert result.source_series_uids == (SERIES_UID,)


@pytest.mark.parametrize("n_specks", [1, 4, 9])
def test_many_specks_still_leave_the_scope_resolved(n_specks):
    slices = [_ct_slice(float(i)) for i in range(2 + n_specks)]
    contours = [_square(0.0), _square(1.0)]
    contours += [_speck(float(2 + i)) for i in range(n_specks)]

    result = _resolve(contours, slices)

    assert result.code is None
    assert result.source_series_uids == (SERIES_UID,)


def test_an_invalid_contour_that_bounds_area_still_holds_the_roi():
    """Real geometry that could not be read still stops the ROI -- earlier, and
    with the more precise code, before scope resolution is even attempted."""

    slices = [_ct_slice(0.0), _ct_slice(1.0)]
    skewed = _contour([(5, 5, 1.0), (20, 5, 1.6), (20, 20, 1.0), (5, 20, 1.9)])

    result = _resolve([_square(0.0), skewed], slices)

    assert result.code == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"


def test_a_readable_contour_with_no_matching_image_still_fails_the_scope():
    """The scope check itself must keep working: skipping artifacts must not
    turn into skipping real geometry that no CT slice supports."""

    slices = [_ct_slice(0.0)]

    result = _resolve([_square(0.0), _square(77.0)], slices)

    assert result.code == "ROI_UNRESOLVED_SOURCE_SCOPE"
    assert "authoritative geometric matches" in result.detail


def test_a_speck_off_the_scanned_volume_is_still_ignored():
    """The artifact carries no geometry, so its position cannot matter."""

    slices = [_ct_slice(0.0), _ct_slice(1.0)]

    result = _resolve([_square(0.0), _square(1.0), _speck(900.0)], slices)

    assert result.code is None
    assert result.source_series_uids == (SERIES_UID,)


def test_an_roi_of_only_specks_never_reaches_scope_resolution():
    """roi_geometry_code stops it first, with the honest unparseable code."""

    slices = [_ct_slice(0.0)]

    result = _resolve([_speck(0.0), _speck(0.0)], slices)

    assert result.code == "ROI_CONTOUR_UNPARSEABLE"


def test_specks_are_withheld_from_the_rasterizer_copy():
    """One step further downstream than scope: the accepted ROI's speck items
    aborted some rasterizer builds (cv2 fillPoly assertion) while others drew
    nothing from them. The rasterizer copy now withholds invalid area-less
    items; the verdict, the full source and the bound series are unchanged."""
    from rtpipeline.rtstruct_geometry import _contour_rasterizable

    assert _contour_rasterizable(_square(0.0)) is True
    assert _contour_rasterizable(_speck(2.0)) is False
    skewed = _contour([(5, 5, 1.0), (20, 5, 1.6), (20, 20, 1.0), (5, 20, 1.9)])
    assert _contour_rasterizable(skewed) is True


def test_rasterizer_copy_keeps_valid_contours_and_drops_only_specks():
    """The prepared copy carries exactly the scope-validated contours minus
    the validator-skipped specks; the source dataset keeps everything."""
    from rtpipeline.rtstruct_geometry import _contour_rasterizable

    contours = [_square(0.0), _square(1.0), _speck(2.0)]
    kept = [c for c in contours if _contour_rasterizable(c)]
    assert kept == contours[:2]
