"""One validator decides whether an ROI's geometry is readable.

``roi_geometry_code`` holds the rule that an invalid contour item bounding no
area carries no geometry to lose -- a mask-to-contour artifact of one or two
points that the rasteriser ignores. Two other places classified the same ROI
themselves and called it partially unparseable the moment any item failed
validation, bypassing that rule entirely.

In one 122-course campaign that discarded the auto-segmented urinary_bladder --
the study target -- in 15 courses whose remaining slices described it
completely: 18 valid contours and 2 area-less ones was enough to void it.
"""

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline.radiomics_source_inventory import source_observations
from rtpipeline.roi_requiredness import inspect_rtstruct
from rtpipeline.rtstruct_geometry import roi_geometry_code


def _contour(points, kind="CLOSED_PLANAR"):
    item = Dataset()
    item.ContourGeometricType = kind
    item.ContourData = [float(v) for p in points for v in p]
    item.NumberOfContourPoints = len(points)
    return item


def _square(z=0.0):
    return _contour([(0, 0, z), (10, 0, z), (10, 10, z), (0, 10, z)])


def _speck(z=0.0):
    """A stray boundary voxel: one point, bounding nothing."""
    return _contour([(5, 5, z)])


def _dataset(contours, name="urinary_bladder"):
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = name
    contour_roi = Dataset()
    contour_roi.ReferencedROINumber = 1
    contour_roi.ContourSequence = Sequence(list(contours))
    ds = Dataset()
    ds.StructureSetROISequence = Sequence([roi])
    ds.ROIContourSequence = Sequence([contour_roi])
    return ds


def _codes(contours, name="urinary_bladder"):
    """The structural code each of the three paths assigns to one ROI."""
    ds = _dataset(contours, name)
    inventory = [o for o in source_observations(None, dataset=ds) if o.name == name][0]
    requiredness = [o for o in inspect_rtstruct(None, dataset=ds).rois if o.name == name][0]
    return roi_geometry_code(list(contours)), inventory.structural_code, requiredness.structural_code


def test_the_campaign_case_is_no_longer_a_defect():
    """18 readable slices and 2 area-less specks describe the bladder fine."""

    contours = [_square(float(i)) for i in range(18)] + [_speck(90.0), _speck(91.0)]

    assert _codes(contours) == (None, None, None)


def test_an_invalid_item_that_bounds_area_still_condemns_the_roi():
    """Real geometry that could not be read is still a defect, in all three."""

    skewed = _contour([(0, 0, 0), (10, 0, 5), (10, 10, 0), (0, 10, 9)])
    contours = [_square(0.0), skewed]

    assert _codes(contours) == ("ROI_CONTOUR_PARTIALLY_UNPARSEABLE",) * 3


def test_an_roi_of_only_specks_is_unparseable_everywhere():
    contours = [_speck(0.0), _speck(1.0)]

    assert _codes(contours) == ("ROI_CONTOUR_UNPARSEABLE",) * 3


def test_an_empty_contour_sequence_is_unchanged():
    assert _codes([]) == ("ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",) * 3


def test_a_clean_roi_is_unchanged():
    assert _codes([_square(0.0), _square(2.0)]) == (None, None, None)


@pytest.mark.parametrize("n_specks", [1, 2, 7])
def test_the_three_paths_never_disagree(n_specks):
    """They disagreed before: only one applied the area rule."""

    contours = [_square(float(i)) for i in range(4)]
    contours += [_speck(50.0 + i) for i in range(n_specks)]

    canonical, inventory, requiredness = _codes(contours)

    assert canonical == inventory == requiredness is None


def test_a_declared_roi_without_any_contour_item_keeps_its_own_code():
    """Only the two inventories can tell "no item" from "empty sequence"."""

    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "urinary_bladder"
    ds = Dataset()
    ds.StructureSetROISequence = Sequence([roi])
    ds.ROIContourSequence = Sequence([])

    inventory = [o for o in source_observations(None, dataset=ds)][0]
    requiredness = [o for o in inspect_rtstruct(None, dataset=ds).rois][0]

    assert inventory.structural_code == "ROI_DECLARED_NO_CONTOUR_ITEM"
    assert requiredness.structural_code == "ROI_DECLARED_NO_CONTOUR_ITEM"


def test_the_raw_item_counts_are_still_reported():
    """The verdict changes; the observation of what was there does not."""

    ds = _dataset([_square(0.0), _square(1.0), _speck(9.0)])

    observation = [o for o in source_observations(None, dataset=ds)][0]

    assert observation.valid_contours == 2
    assert observation.invalid_contours == 1
