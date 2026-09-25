"""The ROI fixer's grouped rt-utils rasterization equals rt-utils exactly.

``fix_rtstruct_rois`` rasterizes every ROI with rt-utils to find the ones it
must rebuild. ``_rt_utils_roi_mask`` groups contours by referenced image once
instead of matching every contour against every slice. These tests require the
same mask, or an exception wherever rt-utils raises, and the same fixer result.
All data here are synthetic.
"""
from __future__ import annotations

import copy

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from test_auto_rtstruct_exact_planes import _mixed_z, _uniform_z, _write_ct

from rtpipeline import auto_rtstruct as ar
from rtpipeline import roi_fixer
from rtpipeline.roi_fixer import _rt_utils_roi_mask, fix_rtstruct_rois


def _builder(ct_dir, zs):
    from rt_utils import RTStructBuilder

    _write_ct(ct_dir, zs, rows=24, columns=20)
    builder = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    shape = (24, 20, len(zs))
    ring = np.zeros(shape, dtype=bool)
    ring[4:18, 3:15, 2:9] = True
    ring[8:13, 7:11, 2:9] = False  # holes
    two = np.zeros(shape, dtype=bool)
    two[2:6, 2:6, 5:12] = True
    two[15:21, 12:18, 7:14] = True  # two contours per slice
    border = np.zeros(shape, dtype=bool)
    border[0:5, 14:20, :] = True
    builder.add_roi(mask=ring, name="ring")
    builder.add_roi(mask=two, name="two")
    builder.add_roi(mask=border, name="border")
    return builder


def _outcome(function, rtstruct, name):
    try:
        return "array", function(name).tobytes() if function is not None else _rt_utils_roi_mask(rtstruct, name).tobytes()
    except Exception:
        return "raised", None


def _assert_same(rtstruct, names) -> None:
    for name in names:
        expected = _outcome(rtstruct.get_roi_mask_by_name, rtstruct, name)
        actual = _outcome(None, rtstruct, name)
        assert actual == expected, name
        if expected[0] == "array":
            reference = rtstruct.get_roi_mask_by_name(name)
            mask = _rt_utils_roi_mask(rtstruct, name)
            assert mask.dtype == reference.dtype and mask.shape == reference.shape
            assert mask.strides == reference.strides


def _contours(rtstruct, name):
    number = next(int(r.ROINumber) for r in rtstruct.ds.StructureSetROISequence if r.ROIName == name)
    item = next(i for i in rtstruct.ds.ROIContourSequence if int(i.ReferencedROINumber) == number)
    return item.ContourSequence


@pytest.mark.parametrize("zs", [_uniform_z(16), _mixed_z(16)], ids=["uniform", "mixed"])
def test_grouped_masks_equal_rt_utils(tmp_path, zs) -> None:
    rtstruct = _builder(tmp_path / "ct", zs)
    _assert_same(rtstruct, ["ring", "two", "border", "absent"])
    assert _rt_utils_roi_mask(rtstruct, "two").any()


@pytest.mark.parametrize(
    "defect",
    [
        "no_image_sequence",
        "no_referenced_uid",
        "unknown_image",
        "two_images",
        "duplicate_reference",
        "single_point",
        "no_contour_data_on_unknown_image",
        "no_contour_data",
        "empty_contour_sequence",
        "reordered",
        "slice_without_uid",
    ],
)
def test_grouped_masks_equal_rt_utils_on_defective_contours(tmp_path, defect) -> None:
    rtstruct = _builder(tmp_path / "ct", _uniform_z(16))
    contours = _contours(rtstruct, "two")
    first = contours[0]
    uids = [str(s.SOPInstanceUID) for s in rtstruct.series_data]
    if defect == "no_image_sequence":
        del contours[3].ContourImageSequence
    elif defect == "no_referenced_uid":
        del contours[3].ContourImageSequence[0].ReferencedSOPInstanceUID
    elif defect == "unknown_image":
        contours[2].ContourImageSequence[0].ReferencedSOPInstanceUID = "1.2.3.4.5"
    elif defect == "two_images":
        extra = copy.deepcopy(first.ContourImageSequence[0])
        extra.ReferencedSOPInstanceUID = uids[1]
        first.ContourImageSequence.append(extra)
    elif defect == "duplicate_reference":
        first.ContourImageSequence.append(copy.deepcopy(first.ContourImageSequence[0]))
    elif defect == "single_point":
        first.ContourData = list(first.ContourData[:3])
        first.NumberOfContourPoints = 1
    elif defect == "no_contour_data_on_unknown_image":
        contours[2].ContourImageSequence[0].ReferencedSOPInstanceUID = "1.2.3.4.5"
        del contours[2].ContourData
    elif defect == "no_contour_data":
        del contours[2].ContourData
    elif defect == "empty_contour_sequence":
        contours.clear()
    elif defect == "reordered":
        items = list(contours)
        contours.clear()
        for contour in reversed(items):
            contours.append(contour)
    elif defect == "slice_without_uid":
        del rtstruct.series_data[4].SOPInstanceUID
    _assert_same(rtstruct, ["ring", "two", "border"])


def _signature(path):
    return ar._rtstruct_contour_signature(path)


@pytest.mark.parametrize("broken", [False, True], ids=["nothing_to_fix", "one_roi_unrasterizable"])
def test_fixer_result_is_unchanged(tmp_path, monkeypatch, broken) -> None:
    ct_dir = tmp_path / "ct"
    builder = _builder(ct_dir, _uniform_z(16))
    if broken:
        # Contours referencing an image outside the series rasterize to
        # nothing in rt-utils and in the fixer's rebuild: the ROI fails.
        for contour in _contours(builder, "ring"):
            contour.ContourImageSequence[0].ReferencedSOPInstanceUID = "1.2.3.4.5"
    source = tmp_path / "rs.dcm"
    builder.save(str(source))
    results = {}
    for label in ("rt_utils", "grouped"):
        target = tmp_path / f"{label}.dcm"
        target.write_bytes(source.read_bytes())
        with monkeypatch.context() as patch:
            if label == "rt_utils":
                patch.setattr(roi_fixer, "_rt_utils_roi_mask", lambda rtstruct, name: rtstruct.get_roi_mask_by_name(name))
            summary = fix_rtstruct_rois(ct_dir, target)
        results[label] = (summary.changed, summary.fixed, summary.failed, _signature(target), target.read_bytes() == source.read_bytes())
    assert results["grouped"] == results["rt_utils"]
    assert results["grouped"][4] is (not results["grouped"][0])


def test_fixer_rebuild_path_is_unchanged(tmp_path, monkeypatch) -> None:
    """A ROI rt-utils cannot rasterize is rebuilt; every other ROI is re-added."""
    ct_dir = tmp_path / "ct"
    builder = _builder(ct_dir, _uniform_z(16))
    for contour in _contours(builder, "two")[:2]:
        contour.ContourData = list(contour.ContourData[:3])  # one point: cv2 rejects it
        contour.NumberOfContourPoints = 1
    source = tmp_path / "rs.dcm"
    builder.save(str(source))
    results = {}
    for label in ("rt_utils", "grouped"):
        target = tmp_path / f"{label}.dcm"
        target.write_bytes(source.read_bytes())
        with monkeypatch.context() as patch:
            if label == "rt_utils":
                patch.setattr(roi_fixer, "_rt_utils_roi_mask", lambda rtstruct, name: rtstruct.get_roi_mask_by_name(name))
            summary = fix_rtstruct_rois(ct_dir, target)
        results[label] = (summary.changed, summary.fixed, summary.failed, _signature(target))
    assert results["grouped"] == results["rt_utils"]
    assert results["grouped"][0] is True and results["grouped"][1] == ["two"]
    reread = pydicom.dcmread(str(tmp_path / "grouped.dcm"))
    assert {str(r.ROIName) for r in reread.StructureSetROISequence} == {"ring", "two", "border"}


def test_a_replaced_rasterizer_is_still_called(tmp_path, monkeypatch) -> None:
    """Only rt-utils' own method is reproduced; any replacement is called."""
    from rt_utils import RTStruct

    rtstruct = _builder(tmp_path / "ct", _uniform_z(16))
    calls = []
    original = RTStruct.get_roi_mask_by_name

    def replaced(self, name):
        calls.append(name)
        return original(self, name)

    monkeypatch.setattr(RTStruct, "get_roi_mask_by_name", replaced)
    _rt_utils_roi_mask(rtstruct, "ring")
    assert calls == ["ring"]
    monkeypatch.undo()
    _rt_utils_roi_mask(rtstruct, "ring")
    assert calls == ["ring"]
