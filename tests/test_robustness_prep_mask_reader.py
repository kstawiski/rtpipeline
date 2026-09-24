"""The indexed RTSTRUCT reader equals rt_utils, and robustness keeps only selectable masks.

``_rt_utils_indexed_roi_mask`` must return the rt_utils array bit for bit (or
exactly its emptiness) and raise the same exception type and text for every
ROI, including defective ones. ``_rtstruct_masks(retain_mask=...)`` must record
the same outcomes as without it. All data is synthetic.
"""
from __future__ import annotations

import copy

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline import radiomics as rm
from robustness_prep_fixture import (
    append_roi, apply_defects, contour, ellipsoid, slice_z_and_uid, square_contour,
    write_ct, write_rtstruct,
)

ROWS = COLUMNS = 40
SLICES = 12


def _cylinder(center, radii, slices):
    """Elliptic cylinder: every contour bounds area, so rt_utils can read it back."""
    mask = np.zeros((ROWS, COLUMNS, SLICES), dtype=bool)
    disc = ellipsoid((ROWS, COLUMNS, 1), (center[0], center[1], 0), (radii[0], radii[1], 1))[..., 0]
    for index in slices:
        mask[..., index] = disc
    return mask


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("reader")
    ct = root / "CT"
    write_ct(ct, rows=ROWS, columns=COLUMNS, slices=SLICES, spacing=(2.0, 2.0, 3.0))
    shape = (ROWS, COLUMNS, SLICES)
    masks = [
        ("GTV1", _cylinder((20, 20), (6, 8), range(3, 10))),
        ("edge", _cylinder((0, 39), (5, 5), range(8, 12))),
        ("ring", _cylinder((20, 20), (12, 12), range(1, 10)) & ~_cylinder((20, 20), (6, 6), range(1, 10))),
        ("thin", np.pad(np.ones((30, 1, 1), bool), ((5, 5), (20, 19), (0, 11)))),
        ("two_parts", _cylinder((8, 8), (3, 3), range(2, 5)) | _cylinder((30, 30), (4, 4), range(8, 11))),
    ]
    rs = root / "RS.dcm"
    write_rtstruct(ct, rs, masks)
    apply_defects(ct, rs, [
        {"name": "far", "kind": "outside_fov"},
        {"name": "orphan_ref", "kind": "unreferenced_slice"},
        {"name": "no_ref", "kind": "no_image_reference"},
        {"name": "point", "kind": "point"},
    ])
    ds = pydicom.dcmread(rs)
    slices = slice_z_and_uid(ct)
    z0, uid0 = slices[0]
    z5, uid5 = slices[5]
    # Referenced twice by one item, and a slice-0 item before a malformed one.
    twice = square_contour(-10.0, -10.0, 8.0, z5, uid5)
    extra = Dataset()
    extra.ReferencedSOPClassUID = twice.ContourImageSequence[0].ReferencedSOPClassUID
    extra.ReferencedSOPInstanceUID = uid5
    twice.ContourImageSequence.append(extra)
    append_roi(ds, "double_ref", [twice, square_contour(0.0, 0.0, 6.0, z0, uid0)])
    bad_late = square_contour(0.0, 0.0, 6.0, z5, uid5)
    bad_late.ContourData = bad_late.ContourData[:-1]  # not a multiple of three
    append_roi(ds, "bad_late", [square_contour(-4.0, -4.0, 6.0, z0, uid0), bad_late])
    missing_data = square_contour(0.0, 0.0, 6.0, z5, uid5)
    del missing_data.ContourData
    append_roi(ds, "missing_data", [square_contour(-4.0, -4.0, 6.0, z0, uid0), missing_data])
    append_roi(ds, "no_sequence", None)
    no_uid = square_contour(0.0, 0.0, 6.0, z5, uid5)
    del no_uid.ContourImageSequence[0].ReferencedSOPInstanceUID
    append_roi(ds, "no_uid", [no_uid])
    append_roi(ds, "empty_sequence", [])
    ds.save_as(rs)
    return ct, rs


def _outcome(call):
    try:
        value = call()
    except Exception as exc:  # compared by type and text
        return ("raised", type(exc), str(exc))
    return ("value", value)


def _names(rs):
    return [str(r.ROIName) for r in pydicom.dcmread(rs).StructureSetROISequence] + ["absent"]


def test_indexed_reader_is_used_for_the_pinned_rt_utils(source):
    from rt_utils import RTStructBuilder

    ct, rs = source
    rt = RTStructBuilder.create_from(dicom_series_path=str(ct), rt_struct_path=str(rs))
    assert rm._rt_utils_replicated_digest() == rm._RT_UTILS_REPLICATED_SHA256
    assert rm._rt_utils_reader_is_replicated(rt)
    assert not rm._rt_utils_reader_is_replicated(copy.copy(object()))


@pytest.mark.parametrize("retain", [True, False])
def test_indexed_reader_matches_rt_utils_for_every_roi(source, retain):
    from rt_utils import RTStructBuilder

    ct, rs = source
    names = _names(rs)
    # Separate readers: pydicom caches converted values in the dataset.
    library = RTStructBuilder.create_from(dicom_series_path=str(ct), rt_struct_path=str(rs))
    indexed = RTStructBuilder.create_from(dicom_series_path=str(ct), rt_struct_path=str(rs))
    raised = set()
    for name in names:
        expected = _outcome(lambda: library.get_roi_mask_by_name(name))
        observed = _outcome(lambda: rm._rt_utils_indexed_roi_mask(indexed, name, retain=retain))
        if expected[0] == "raised":
            raised.add(name)
            assert observed == expected, name
        elif retain:
            assert observed[0] == "value" and observed[1].dtype == expected[1].dtype
            assert np.array_equal(observed[1], expected[1]), name
        else:
            assert observed == ("value", bool(expected[1].any())), name
    # rt_utils cannot fill a one-point polygon; the robustness reader never
    # sees "point" because its inventory marks it non-volumetric first.
    assert raised == {"bad_late", "missing_data", "no_sequence", "no_uid", "no_ref", "absent", "point"}


def test_indexed_reader_falls_back_when_uids_are_not_plain_strings(source, monkeypatch):
    from rt_utils import RTStructBuilder

    ct, rs = source
    monkeypatch.setattr(rm, "_dict_lookup_matches_equality", lambda: ())
    library = RTStructBuilder.create_from(dicom_series_path=str(ct), rt_struct_path=str(rs))
    indexed = RTStructBuilder.create_from(dicom_series_path=str(ct), rt_struct_path=str(rs))
    for name in ("GTV1", "double_ref", "ring"):
        assert np.array_equal(rm._rt_utils_indexed_roi_mask(indexed, name, retain=True),
                              library.get_roi_mask_by_name(name))


def _read(ct, rs, **kwargs):
    sink = []
    try:
        out = rm._rtstruct_masks(ct, rs, failure_outcomes=sink, **kwargs)
    except Exception as exc:
        return ("raised", type(exc), str(exc), sink)
    return ("value", out, sink)


@pytest.mark.parametrize("use_index", [True, False])
def test_retain_mask_changes_only_which_arrays_are_kept(source, monkeypatch, use_index):
    from rtpipeline.roi_requiredness import Requiredness

    ct, rs = source
    if not use_index:
        monkeypatch.setattr(rm, "_rt_utils_reader_is_replicated", lambda rt: False)
    names = _names(rs)[:-1]
    policy = dict(
        tolerate_unselected=True,
        requiredness_by_roi={n: (Requiredness.ANALYSIS_REQUIRED if n == "GTV1" else
                                 Requiredness.INVENTORY_ONLY) for n in names},
        unmeasurable_required_is_disposition=True,
    )
    expected = _read(ct, rs, **policy)
    observed = _read(ct, rs, **policy, retain_mask=lambda n: n in {"GTV1", "edge"})
    assert expected[0] == observed[0] == "value"
    assert observed[2] == expected[2]
    assert list(observed[1]) == list(expected[1])
    for name, array in expected[1].items():
        if name in {"GTV1", "edge"}:
            assert np.array_equal(observed[1][name], array)
        else:
            assert observed[1][name] is None
    failed = {row["roi_name"]: row["failure_kind"] for row in expected[2] if row["status"] == "failed"}
    assert failed["far"] == failed["orphan_ref"] == "degenerate_mask"
    assert failed["no_ref"] == failed["bad_late"] == failed["missing_data"] == "extraction_error"


def test_retain_mask_keeps_fatal_selected_failures(source):
    from rtpipeline.roi_requiredness import Requiredness

    ct, rs = source
    names = _names(rs)[:-1]
    policy = dict(
        tolerate_unselected=True,
        requiredness_by_roi={n: (Requiredness.ANALYSIS_REQUIRED if n == "far" else
                                 Requiredness.INVENTORY_ONLY) for n in names},
    )
    expected = _read(ct, rs, **policy)
    observed = _read(ct, rs, **policy, retain_mask=lambda n: False)
    assert expected[0] == "raised" and observed[:3] == expected[:3]
    assert observed[3] == expected[3]


def test_a_replaced_rt_utils_reader_is_used_instead_of_the_replica(source, monkeypatch):
    from rt_utils import RTStruct, RTStructBuilder

    ct, rs = source
    rt = RTStructBuilder.create_from(dicom_series_path=str(ct), rt_struct_path=str(rs))
    assert rm._rt_utils_reader_is_replicated(rt)
    real = RTStruct.get_roi_mask_by_name
    monkeypatch.setattr(RTStruct, "get_roi_mask_by_name", lambda self, name: real(self, name))
    assert not rm._rt_utils_reader_is_replicated(rt)
    monkeypatch.undo()
    assert rm._rt_utils_reader_is_replicated(rt)
