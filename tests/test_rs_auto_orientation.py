"""RS_auto orientation and the reuse of RS_auto written with x and y exchanged.

From 49eb9c9 until 2026-09-25, RS_auto rebuilt from masks sampled them in
``(column, row)`` layout, which rt-utils contours with x and y exchanged. These
tests check the physical x/y extent of published contours against the source
mask on square and non-square series, and that reuse rejects such a file while
it keeps reusing the TotalSegmentator RTSTRUCT copy. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from course_contract_test_utils import write_minimal_course_contract
from test_auto_rtstruct_exact_planes import (
    STRUCTURES,
    _align_contract_ct_geometry,
    _course,
    _mixed_z,
    _uniform_z,
    _write_ct,
)
from test_rtstruct_writers_exact_planes import (
    MODEL,
    MODEL_ROI,
    _assert_box_placed,
    _build_rs_custom,
    _custom_course,
    _force_rebuild,
)

from rtpipeline import auto_rtstruct as ar
from rtpipeline import custom_structures_rtstruct as crs
from rtpipeline.rtstruct_geometry import create_scoped_rtstruct, image_array_for_rtstruct

OLD = 1_000_000_000
# (rows, columns, PixelSpacing [row spacing (y), column spacing (x)], origin x/y)
SQUARE = (16, 16, (2.0, 2.0), (-15.0, -15.0))
NON_SQUARE = (40, 60, (1.5, 1.5), (-45.0, -40.0))
# Non-square pixels: PixelSpacing[0] (between rows, y) differs from [1] (x).
ANISOTROPIC = (40, 60, (2.0, 1.5), (-45.0, -40.0))
# Asymmetric, off-centre boxes (x, y, z ranges in mm) inside both series.
BOXES = {
    "wide_box": ((-12.0, 8.0), (2.0, 10.0), (-22.0, 12.0)),
    "tall_box": ((-3.0, 1.0), (-13.0, 7.0), (-26.0, 4.0)),
}


def _transposed_sampler(image, series_data):
    """The RS_auto sampler of 49eb9c9..2026-09-25: (column, row, slice)."""
    return np.transpose(image_array_for_rtstruct(image, series_data), (1, 0, 2))


def _grid(zs, geometry) -> sitk.Image:
    rows, columns, (row_mm, column_mm), (ox, oy) = geometry
    image = sitk.Image(columns, rows, len(zs), sitk.sitkUInt8)
    image.SetOrigin((ox, oy, float(zs[0])))
    image.SetSpacing((column_mm, row_mm, float(zs[-1] - zs[0]) / (len(zs) - 1)))
    return image


def _centres(geometry):
    rows, columns, (row_mm, column_mm), (ox, oy) = geometry
    return ox + np.arange(columns) * column_mm, oy + np.arange(rows) * row_mm


def _mask_image(zs, geometry, box) -> sitk.Image:
    grid = _grid(zs, geometry)
    x, y = _centres(geometry)
    z = float(zs[0]) + np.arange(len(zs)) * grid.GetSpacing()[2]
    (x0, x1), (y0, y1), (z0, z1) = box
    inside = lambda v, lo, hi: (v >= lo) & (v <= hi)  # noqa: E731
    array = (
        inside(z, z0, z1)[:, None, None]
        & inside(y, y0, y1)[None, :, None]
        & inside(x, x0, x1)[None, None, :]
    ).astype(np.uint8)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(grid)
    return image


def _geometry_course(tmp_path: Path, zs, geometry):
    rows, columns, spacing, origin = geometry
    course = tmp_path / "course"
    dirs = ar.build_course_dirs(course)
    _write_ct(dirs.dicom_ct, zs, rows=rows, columns=columns, pixel_spacing=spacing, origin_xy=origin)
    _align_contract_ct_geometry(write_minimal_course_contract(course, planning_ct_dir=dirs.dicom_ct), dirs.dicom_ct)
    seg_dir = dirs.segmentation_totalseg / "CT_series"
    seg_dir.mkdir(parents=True)
    for name, box in BOXES.items():
        sitk.WriteImage(_mask_image(zs, geometry, box), str(seg_dir / f"total--{name}.nii.gz"))
    return course, dirs.dicom_ct, seg_dir


def _age(*paths: Path) -> None:
    for path in paths:
        for item in [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]:
            os.utime(item, ns=(OLD, OLD))


def _roi_points(path: Path) -> dict[str, list[np.ndarray]]:
    ds = pydicom.dcmread(str(path))
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    return {
        names[int(item.ReferencedROINumber)]: [
            np.asarray(c.ContourData, dtype=float).reshape(-1, 3) for c in item.ContourSequence
        ]
        for item in ds.ROIContourSequence
    }


def _assert_physical_extent(path: Path, zs, geometry, boxes=BOXES) -> None:
    """Every contour spans exactly the box's pixel centres in x and in y."""
    x, y = _centres(geometry)
    step = float(zs[-1] - zs[0]) / (len(zs) - 1)
    published = {name.lower().replace(" ", "_"): value for name, value in _roi_points(path).items()}
    assert set(published) == set(boxes)
    for name, ((x0, x1), (y0, y1), (z0, z1)) in boxes.items():
        xs, ys = x[(x >= x0) & (x <= x1)], y[(y >= y0) & (y <= y1)]
        assert published[name], name
        for points in published[name]:
            assert np.allclose(points[:, 0].min(), xs.min(), atol=1e-3), (name, "x min")
            assert np.allclose(points[:, 0].max(), xs.max(), atol=1e-3), (name, "x max")
            assert np.allclose(points[:, 1].min(), ys.min(), atol=1e-3), (name, "y min")
            assert np.allclose(points[:, 1].max(), ys.max(), atol=1e-3), (name, "y max")
            assert z0 - step <= points[0, 2] <= z1 + step, (name, points[0, 2])


def _origin(course: Path) -> dict:
    return json.loads(ar._rs_auto_origin_path(course).read_text(encoding="utf-8"))


def _decision(course: Path) -> dict:
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    return audit["decisions"]["RS_auto"]


# Orientation -----------------------------------------------------------------


@pytest.mark.parametrize("geometry", [SQUARE, NON_SQUARE], ids=["square", "rows40_columns60"])
def test_model_rtstruct_and_copied_rs_auto_keep_mask_x_y_extent(tmp_path, geometry) -> None:
    """Copy path: the model RTSTRUCT producer, then RS_auto copied from it."""
    from rtpipeline import segmentation

    zs = _uniform_z()
    course, ct_dir, seg_dir = _geometry_course(tmp_path, zs, geometry)
    total = segmentation._ensure_model_rtstruct_from_masks(ct_dir, seg_dir, "CT_series", "total")
    assert total == seg_dir / "CT_series--total.dcm"
    _assert_physical_extent(total, zs, geometry)
    _age(ct_dir, course / "NIFTI", seg_dir)

    out = ar.build_auto_rtstruct(course)
    assert out.read_bytes() == total.read_bytes()
    assert _origin(course)["origin"] == ar.RS_AUTO_ORIGIN_COPY
    _assert_physical_extent(out, zs, geometry)


@pytest.mark.parametrize("geometry", [SQUARE, NON_SQUARE], ids=["square", "rows40_columns60"])
def test_rs_auto_from_masks_keeps_mask_x_y_extent(tmp_path, geometry) -> None:
    """Mask path on a mixed-spacing CT (no TotalSegmentator RTSTRUCT)."""
    zs = _mixed_z()
    course, ct_dir, seg_dir = _geometry_course(tmp_path, zs, geometry)
    _age(ct_dir, course / "NIFTI", seg_dir)
    out = ar.build_auto_rtstruct(course)
    assert out is not None
    _assert_physical_extent(out, zs, geometry)
    record = _origin(course)
    assert record["origin"] == ar.RS_AUTO_ORIGIN_MASKS
    assert record["mask_layout"] == ar.RS_AUTO_MASK_LAYOUT


def test_the_old_sampler_exchanges_x_and_y_on_a_non_square_series(tmp_path, monkeypatch) -> None:
    """The extent check detects the defect it guards against."""
    zs = _mixed_z()
    course, ct_dir, seg_dir = _geometry_course(tmp_path, zs, NON_SQUARE)
    _age(ct_dir, course / "NIFTI", seg_dir)
    monkeypatch.setattr(ar, "_image_array_for_rtstruct", _transposed_sampler)
    out = ar.build_auto_rtstruct(course)
    with pytest.raises(AssertionError):
        _assert_physical_extent(out, zs, NON_SQUARE)


# Reuse of RS_auto written before the fix --------------------------------------


def _legacy_mask_build(course: Path, monkeypatch) -> Path:
    """RS_auto as 49eb9c9..2026-09-25 wrote it from masks: transposed, no record."""
    with monkeypatch.context() as patch:
        patch.setattr(ar, "_image_array_for_rtstruct", _transposed_sampler)
        legacy = ar.build_auto_rtstruct(course)
    ar._rs_auto_origin_path(course).unlink()
    return legacy


def test_legacy_mask_built_rs_auto_is_rejected_and_rebuilt(tmp_path, monkeypatch) -> None:
    course, ct_dir, _seg, _total = _course(tmp_path, _mixed_z())
    legacy = _legacy_mask_build(course, monkeypatch)
    scoped = create_scoped_rtstruct(ct_dir, legacy)
    for name, box in STRUCTURES.items():
        _assert_box_placed(np.transpose(scoped.get_roi_mask_by_name(name), (1, 0, 2)), _mixed_z(), box)
    legacy_bytes = legacy.read_bytes()

    out = ar.build_auto_rtstruct(course)
    assert out.read_bytes() != legacy_bytes
    decision = _decision(course)
    assert decision["action"] == "rebuilt"
    reason = decision["rejected_artifact"]["reason"]
    assert "no origin record" in reason
    assert "off the planning CT planes" in reason
    assert "x and y exchanged" in reason
    scoped = create_scoped_rtstruct(ct_dir, out)
    for name, box in STRUCTURES.items():
        _assert_box_placed(scoped.get_roi_mask_by_name(name), _mixed_z(), box)
    assert _origin(course)["mask_layout"] == ar.RS_AUTO_MASK_LAYOUT

    rebuilt = out.read_bytes()
    assert ar.build_auto_rtstruct(course) == out
    assert out.read_bytes() == rebuilt
    assert _decision(course)["action"] == "reused"


def test_legacy_mask_built_rs_auto_beside_an_on_plane_totalseg_rtstruct_is_rejected(
    tmp_path, monkeypatch
) -> None:
    """A mask-built file is not accepted just because an on-plane copy source exists."""
    course, _ct_dir, seg_dir, total = _course(tmp_path, _uniform_z())
    hidden = seg_dir.parent / "hidden--total.dcm"
    total.rename(hidden)
    _legacy_mask_build(course, monkeypatch)
    hidden.rename(total)
    os.utime(total, ns=(OLD, OLD))

    out = ar.build_auto_rtstruct(course)
    reason = _decision(course)["rejected_artifact"]["reason"]
    assert "differ from the copy of the selected TotalSegmentator RTSTRUCT" in reason
    assert out.read_bytes() == total.read_bytes()
    assert _origin(course)["origin"] == ar.RS_AUTO_ORIGIN_COPY


def test_legacy_mask_built_rs_auto_without_a_totalseg_rtstruct_is_rejected(tmp_path, monkeypatch) -> None:
    course, _ct_dir, _seg, _total = _course(tmp_path, _uniform_z(), rtstruct=False)
    _legacy_mask_build(course, monkeypatch)
    ar.build_auto_rtstruct(course)
    reason = _decision(course)["rejected_artifact"]["reason"]
    assert "no selected TotalSegmentator RTSTRUCT exists" in reason


@pytest.mark.parametrize("fixer_rebuilds", [False, True], ids=["copied", "fixer_rebuilt"])
def test_legacy_totalseg_copy_is_reused_unchanged(tmp_path, monkeypatch, fixer_rebuilds) -> None:
    """A copy-path RS_auto written before the origin record existed is reused as is."""
    course, _ct_dir, _seg, total = _course(tmp_path, _uniform_z())
    if fixer_rebuilds:
        _force_rebuild(monkeypatch, "vertebrae_S1")
    out = ar.build_auto_rtstruct(course)
    assert (out.read_bytes() == total.read_bytes()) is not fixer_rebuilds
    ar._rs_auto_origin_path(course).unlink()
    before = out.read_bytes()

    assert ar.build_auto_rtstruct(course) == out
    assert out.read_bytes() == before
    assert _decision(course)["action"] == "reused"
    record = _origin(course)
    assert record["origin"] == ar.RS_AUTO_ORIGIN_COPY
    assert record["recognized_legacy_copy"] is True


@pytest.mark.parametrize("tamper", ["old_layout", "other_bytes"])
def test_origin_record_must_name_current_layout_and_these_bytes(tmp_path, tamper) -> None:
    course, _ct_dir, _seg, _total = _course(tmp_path, _mixed_z())
    out = ar.build_auto_rtstruct(course)
    record = _origin(course)
    if tamper == "old_layout":
        record["mask_layout"] = "columns-rows-v1"
    else:
        record["rs_auto_sha256"] = "0" * 64
    ar._rs_auto_origin_path(course).write_text(json.dumps(record), encoding="utf-8")

    ar.build_auto_rtstruct(course)
    reason = _decision(course)["rejected_artifact"]["reason"]
    if tamper == "old_layout":
        assert "does not name the current mask layout" in reason
    else:
        assert "no origin record" in reason
    assert _origin(course)["mask_layout"] == ar.RS_AUTO_MASK_LAYOUT
    assert out.exists()


# RS_custom follows a rebuilt RS_auto ------------------------------------------


def test_rs_custom_regenerates_after_legacy_rs_auto_is_rebuilt(tmp_path, monkeypatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(ar, "_image_array_for_rtstruct", _transposed_sampler)
        case = _custom_course(tmp_path, _mixed_z())
    course, ct_dir = case["course"], case["ct_dir"]
    ar._rs_auto_origin_path(course).unlink()
    rs_custom = _build_rs_custom(case)
    os.utime(rs_custom, ns=(OLD + 2, OLD + 2))
    assert not crs._is_rs_custom_stale(rs_custom, case["config"], case["manual"], case["rs_auto"])
    stale_custom = create_scoped_rtstruct(ct_dir, rs_custom)
    for name, box in STRUCTURES.items():
        _assert_box_placed(np.transpose(stale_custom.get_roi_mask_by_name(name), (1, 0, 2)), _mixed_z(), box)

    assert ar.build_auto_rtstruct(course) == case["rs_auto"]
    assert _decision(course)["action"] == "rebuilt"
    assert crs._is_rs_custom_stale(rs_custom, case["config"], case["manual"], case["rs_auto"])

    out = _build_rs_custom(case)
    custom = create_scoped_rtstruct(ct_dir, out)
    auto = create_scoped_rtstruct(ct_dir, case["rs_auto"])
    for name, box in STRUCTURES.items():
        _assert_box_placed(custom.get_roi_mask_by_name(name), _mixed_z(), box)
    expected = np.logical_or(*(auto.get_roi_mask_by_name(n) for n in STRUCTURES))
    assert np.array_equal(custom.get_roi_mask_by_name("auto_union"), expected)
    assert np.any(custom.get_roi_mask_by_name(f"{MODEL}_{MODEL_ROI}"))
    assert not crs._is_rs_custom_stale(out, case["config"], case["manual"], case["rs_auto"])


# Known rt-utils limitation, not changed here -------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "rt-utils get_pixel_to_patient_transformation_matrix scales the row "
        "direction (x) by PixelSpacing[0], the spacing between rows (y); every "
        "rt-utils writer misplaces contours on non-square pixels (2026-09-25, see REPORT.md)"
    ),
)
@pytest.mark.parametrize("path", ["model_rtstruct", "rs_auto_masks"])
def test_contours_keep_mask_x_y_extent_on_non_square_pixels(tmp_path, path) -> None:
    from rtpipeline import segmentation

    zs = _uniform_z() if path == "model_rtstruct" else _mixed_z()
    course, ct_dir, seg_dir = _geometry_course(tmp_path, zs, ANISOTROPIC)
    _age(ct_dir, course / "NIFTI", seg_dir)
    if path == "model_rtstruct":
        out = segmentation._ensure_model_rtstruct_from_masks(ct_dir, seg_dir, "CT_series", "total")
    else:
        out = ar.build_auto_rtstruct(course)
    assert out is not None
    _assert_physical_extent(out, zs, ANISOTROPIC)
