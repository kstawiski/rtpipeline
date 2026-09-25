"""Every rt-utils RTSTRUCT writer must place its contours on the CT planes.

rt-utils ``RTStruct.add_roi`` converts mask slice ``i`` with one affine built
from the first slice and a uniform step, so on a planning CT with mixed slice
spacing every contour it writes misses the plane of the image it references,
and the scoped reader (``rtstruct_geometry``) holds the ROI. fe50cc0 fixed
RS_auto. These tests cover the other writers: RS_custom, the model RTSTRUCT
derived from masks in ``segmentation``, the custom-model RTSTRUCT, the ROI
fixer and RS_auto_cropped. All data here are synthetic.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from course_contract_test_utils import write_minimal_course_contract
from test_auto_rtstruct_exact_planes import (
    CT_SIZE,
    ORIGIN_XY,
    PIXEL_MM,
    STRUCTURES,
    _align_contract_ct_geometry,
    _grid_mask,
    _mixed_z,
    _regular_grid,
    _series,
    _uniform_z,
    _write_ct,
    _write_totalseg_outputs,
)

from rtpipeline import auto_rtstruct as ar
from rtpipeline import custom_structures_rtstruct as crs
from rtpipeline.rtstruct_geometry import (
    contours_on_referenced_planes,
    create_scoped_rtstruct,
    image_array_for_rtstruct,
    place_added_rois_on_planes,
)

REPO = Path(__file__).resolve().parents[1]
OLD_REVISION = "a177c59"
OLD = 1_000_000_000  # inputs predate every publication
MODEL = "testmodel"
MODEL_ROI = "box"
MODEL_BOX = ((3.0, 11.0), (-11.0, -3.0), (-20.0, 6.0))
GTV_BOX = ((-5.0, 5.0), (-3.0, 9.0), (-18.0, -6.0))
CONFIG = """\
custom_structures:
  - name: "auto_union"
    operation: "union"
    source_structures: ["iliac_artery_left", "vertebrae_S1"]
  - name: "model_gtv_union"
    operation: "union"
    source_structures: ["testmodel_box", "GTV"]
"""


def _box_image(zs, box) -> sitk.Image:
    """A NIfTI-style mask on the regularized grid a segmentation model uses."""
    grid = _regular_grid(zs)
    image = sitk.GetImageFromArray(_grid_mask(grid, box).astype(np.uint8))
    image.CopyInformation(grid)
    return image


def _expected_in_plane(box) -> np.ndarray:
    """The box on one CT slice, in rt-utils mask layout (row index, column index)."""
    (x0, x1), (y0, y1), _z = box
    x = ORIGIN_XY[0] + np.arange(CT_SIZE) * PIXEL_MM
    y = ORIGIN_XY[1] + np.arange(CT_SIZE) * PIXEL_MM
    return ((y >= y0) & (y <= y1))[:, None] & ((x >= x0) & (x <= x1))[None, :]


def _assert_box_placed(mask: np.ndarray, zs, box) -> None:
    """Occupied slices lie within the box's z range and carry its exact x/y extent."""
    step = float(zs[-1] - zs[0]) / (len(zs) - 1)
    occupied = np.any(mask, axis=(0, 1))
    assert occupied.any()
    expected = _expected_in_plane(box)
    for index in np.flatnonzero(occupied):
        assert box[2][0] - step <= zs[index] <= box[2][1] + step, (index, zs[index])
        assert np.array_equal(mask[:, :, index], expected), index


def _write_manual_rtstruct(ct_dir: Path, out: Path, zs, *, gtv_box=GTV_BOX, offplane_mm: float = 0.5) -> Path:
    """A clinical-style base: GTV on its planes, plus one ROI held off-plane."""
    from rt_utils import RTStructBuilder

    builder = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    builder.add_roi(mask=image_array_for_rtstruct(_box_image(zs, gtv_box), builder.series_data) > 0, name="GTV")
    place_added_rois_on_planes(builder.ds, builder.series_data)
    mask = np.zeros((CT_SIZE, CT_SIZE, len(zs)), dtype=bool)
    mask[4:9, 4:9, 3] = True
    builder.add_roi(mask=mask, name="manual_offplane")
    contour = builder.ds.ROIContourSequence[-1].ContourSequence[0]
    points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
    points[:, 2] += offplane_mm
    contour.ContourData = points.ravel().tolist()
    builder.save(str(out))
    return out


def _custom_course(tmp_path: Path, zs, *, name: str = "course", ct_change=None, with_auto=True) -> dict:
    course = tmp_path / name
    dirs = ar.build_course_dirs(course)
    if ct_change:
        _write_ct_with_changing_slice(dirs.dicom_ct, zs, ct_change)
        # Only the lower slices keep the first slice's geometry.
        manual = _write_manual_rtstruct(dirs.dicom_ct, course / "RS.dcm", zs, gtv_box=((-5.0, 5.0), (-3.0, 9.0), (-30.0, -12.0)))
    else:
        _write_ct(dirs.dicom_ct, zs)
        manual = _write_manual_rtstruct(dirs.dicom_ct, course / "RS.dcm", zs)
    metadata = write_minimal_course_contract(course, planning_ct_dir=dirs.dicom_ct, authoritative_rtstruct=manual)
    _align_contract_ct_geometry(metadata, dirs.dicom_ct)
    for path in [*dirs.dicom_ct.iterdir(), *(course / "NIFTI").iterdir(), manual]:
        os.utime(path, ns=(OLD, OLD))
    rs_auto = None
    if with_auto:
        _write_totalseg_outputs(dirs.segmentation_totalseg / "CT_series", dirs.dicom_ct, zs)
        rs_auto = ar.build_auto_rtstruct(course)
        assert rs_auto is not None
        os.utime(rs_auto, ns=(OLD + 1, OLD + 1))
    model_dir = course / "Segmentation_CustomModels" / MODEL
    model_dir.mkdir(parents=True)
    sitk.WriteImage(_box_image(zs, MODEL_BOX), str(model_dir / f"{MODEL_ROI}.nii.gz"))
    config = tmp_path / f"{name}_custom.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    for path in [model_dir / f"{MODEL_ROI}.nii.gz", config]:
        os.utime(path, ns=(OLD, OLD))
    return {"course": course, "ct_dir": dirs.dicom_ct, "manual": manual, "rs_auto": rs_auto, "config": config}


def _build_rs_custom(case: dict) -> Path:
    return crs._create_custom_structures_rtstruct(
        case["course"], case["config"], rs_manual=case["manual"], rs_auto=case["rs_auto"]
    )


def _roi_contours(path: Path) -> dict[str, list[list[float]]]:
    ds = pydicom.dcmread(str(path))
    names = {int(roi.ROINumber): str(roi.ROIName) for roi in ds.StructureSetROISequence}
    return {
        names[int(item.ReferencedROINumber)]: [list(c.ContourData) for c in getattr(item, "ContourSequence", []) or []]
        for item in ds.ROIContourSequence
    }


def _assert_resolves(path: Path, ct_dir: Path, names) -> None:
    scoped = create_scoped_rtstruct(ct_dir, path)
    for name in names:
        result = scoped.by_name[name]
        assert result.code is None, (name, result.code, result.detail)
        assert np.any(scoped.get_roi_mask_by_name(name)), name


# RS_custom ----------------------------------------------------------------


def test_rs_custom_on_mixed_spacing_ct_resolves_every_added_roi(tmp_path) -> None:
    case = _custom_course(tmp_path, _mixed_z())
    out = _build_rs_custom(case)
    assert out == case["course"] / "RS_custom.dcm"
    manual_names = {"GTV", "manual_offplane"}
    added = set(_roi_contours(out)) - manual_names
    assert added == {*STRUCTURES, f"{MODEL}_{MODEL_ROI}", "auto_union", "model_gtv_union"}
    _assert_resolves(out, case["ct_dir"], added)

    custom = create_scoped_rtstruct(case["ct_dir"], out)
    auto = create_scoped_rtstruct(case["ct_dir"], case["rs_auto"])
    manual = create_scoped_rtstruct(case["ct_dir"], case["manual"])
    # Derived unions equal the unions of the scoped reader's own source masks.
    expected_auto = np.logical_or(*(auto.get_roi_mask_by_name(n) for n in STRUCTURES))
    assert np.array_equal(custom.get_roi_mask_by_name("auto_union"), expected_auto)
    expected_model = custom.get_roi_mask_by_name(f"{MODEL}_{MODEL_ROI}") | manual.get_roi_mask_by_name("GTV")
    assert np.array_equal(custom.get_roi_mask_by_name("model_gtv_union"), expected_model)
    # The NIfTI model mask sits on the right slices with its own x/y extent.
    _assert_box_placed(custom.get_roi_mask_by_name(f"{MODEL}_{MODEL_ROI}"), _mixed_z(), MODEL_BOX)

    # Base contours are copied unchanged, including the one the reader holds.
    before, after = _roi_contours(case["manual"]), _roi_contours(out)
    assert after["GTV"] == before["GTV"]
    assert after["manual_offplane"] == before["manual_offplane"]
    assert custom.by_name["manual_offplane"].code == "ROI_UNRESOLVED_SOURCE_SCOPE"
    # The unresolved base ROI does not make the publication stale.
    assert not crs._is_rs_custom_stale(out, case["config"], case["manual"], case["rs_auto"])


def test_existing_off_plane_rs_custom_is_rejected_and_rebuilt(tmp_path, monkeypatch, caplog) -> None:
    """An RS_custom written without anchoring (what a177c59 published) is rebuilt."""
    import rtpipeline.rtstruct_geometry as geometry

    case = _custom_course(tmp_path, _mixed_z())
    with monkeypatch.context() as patch:
        patch.setattr(geometry, "place_added_rois_on_planes", lambda *args, **kwargs: 0)
        stale = _build_rs_custom(case)
    ok, _detail = contours_on_referenced_planes(pydicom.dcmread(str(stale)), _series(case["ct_dir"]))
    assert not ok
    stale_bytes = stale.read_bytes()
    with caplog.at_level("WARNING", logger="rtpipeline.custom_structures_rtstruct"):
        assert crs._is_rs_custom_stale(stale, case["config"], case["manual"], case["rs_auto"])
    assert any("added ROIs off the planning CT planes" in r.getMessage() for r in caplog.records)

    out = _build_rs_custom(case)
    assert out.read_bytes() != stale_bytes
    added = set(_roi_contours(out)) - {"GTV", "manual_offplane"}
    _assert_resolves(out, case["ct_dir"], added)
    assert not crs._is_rs_custom_stale(out, case["config"], case["manual"], case["rs_auto"])


def _write_ct_with_changing_slice(ct_dir: Path, zs, change: str) -> None:
    """Mixed-spacing CT whose upper slices change pixel spacing or orientation."""
    _write_ct(ct_dir, zs)
    for index, path in enumerate(sorted(ct_dir.iterdir())):
        if index < len(zs) // 2:
            continue
        ds = pydicom.dcmread(str(path))
        if change == "spacing":
            ds.PixelSpacing = [PIXEL_MM * 1.05, PIXEL_MM * 1.05]
        else:
            ds.ImageOrientationPatient = [1, 0, 0, 0, 0.9998, 0.019998999]
        ds.save_as(str(path), enforce_file_format=True)


@pytest.mark.parametrize("change", ["spacing", "orientation"])
def test_rs_custom_fails_closed_when_anchoring_is_impossible(tmp_path, change) -> None:
    """Pixel spacing or orientation changes between slices: nothing is published."""
    case = _custom_course(tmp_path, _mixed_z(), ct_change=change, with_auto=False)
    with pytest.raises(crs.CustomStructureRTStructError, match="orientation or pixel spacing"):
        _build_rs_custom(case)
    assert not (case["course"] / "RS_custom.dcm").exists()


# Model RTSTRUCT derived from masks (segmentation) ----------------------------


def _model_masks(seg_dir: Path, zs) -> None:
    seg_dir.mkdir(parents=True, exist_ok=True)
    for name, box in STRUCTURES.items():
        sitk.WriteImage(_box_image(zs, box), str(seg_dir / f"total--{name}.nii.gz"))


def test_model_rtstruct_from_masks_on_mixed_ct(tmp_path) -> None:
    from rtpipeline import segmentation

    ct_dir, seg_dir = tmp_path / "CT", tmp_path / "seg"
    _write_ct(ct_dir, _mixed_z())
    _model_masks(seg_dir, _mixed_z())
    out = segmentation._ensure_model_rtstruct_from_masks(ct_dir, seg_dir, "CT_series", "total")
    assert out == seg_dir / "CT_series--total.dcm"
    names = set(_roi_contours(out))
    assert len(names) == len(STRUCTURES)
    _assert_resolves(out, ct_dir, names)
    scoped = create_scoped_rtstruct(ct_dir, out)
    for name, box in STRUCTURES.items():
        roi = next(n for n in names if n.lower().replace(" ", "_") == name.lower())
        _assert_box_placed(scoped.get_roi_mask_by_name(roi), _mixed_z(), box)


@pytest.mark.parametrize("change", ["spacing", "orientation"])
def test_model_rtstruct_from_masks_fails_closed(tmp_path, change) -> None:
    from rtpipeline import segmentation

    ct_dir, seg_dir = tmp_path / "CT", tmp_path / "seg"
    _write_ct_with_changing_slice(ct_dir, _mixed_z(), change)
    _model_masks(seg_dir, _mixed_z())
    assert segmentation._ensure_model_rtstruct_from_masks(ct_dir, seg_dir, "CT_series", "total") is None
    assert not (seg_dir / "CT_series--total.dcm").exists()


# Custom-model RTSTRUCT --------------------------------------------------------


def test_custom_model_rtstruct_on_mixed_ct(tmp_path) -> None:
    from rtpipeline import custom_models

    ct_dir, out_dir = tmp_path / "CT", tmp_path / "model"
    _write_ct(ct_dir, _mixed_z())
    out_dir.mkdir()
    masks = {name: _box_image(_mixed_z(), box) for name, box in STRUCTURES.items()}
    paths = {name: out_dir / f"{name}.nii.gz" for name in STRUCTURES}
    out = custom_models._build_rtstruct(ct_dir, masks, out_dir, paths)
    _assert_resolves(out, ct_dir, STRUCTURES)
    scoped = create_scoped_rtstruct(ct_dir, out)
    for name, box in STRUCTURES.items():
        _assert_box_placed(scoped.get_roi_mask_by_name(name), _mixed_z(), box)


def test_custom_model_rtstruct_fails_closed(tmp_path) -> None:
    from rtpipeline import custom_models

    ct_dir, out_dir = tmp_path / "CT", tmp_path / "model"
    _write_ct_with_changing_slice(ct_dir, _mixed_z(), "spacing")
    out_dir.mkdir()
    masks = {name: _box_image(_mixed_z(), box) for name, box in STRUCTURES.items()}
    with pytest.raises(RuntimeError, match="off the planning CT planes|orientation or pixel spacing"):
        custom_models._build_rtstruct(ct_dir, masks, out_dir, {n: out_dir / n for n in STRUCTURES})
    assert not (out_dir / "rtstruct.dcm").exists()


# ROI fixer ---------------------------------------------------------------------


def _force_rebuild(monkeypatch, name: str) -> None:
    """rt-utils cannot rasterise ``name``; the fixer rebuilds it from contours."""
    from rt_utils import RTStruct

    original = RTStruct.get_roi_mask_by_name

    def failing(self, roi_name):
        if roi_name == name:
            raise ValueError("synthetic rasterisation failure")
        return original(self, roi_name)

    monkeypatch.setattr(RTStruct, "get_roi_mask_by_name", failing)


def _on_plane_rtstruct(ct_dir: Path, out: Path, zs) -> Path:
    from rt_utils import RTStructBuilder

    builder = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    for name, box in STRUCTURES.items():
        builder.add_roi(mask=image_array_for_rtstruct(_box_image(zs, box), builder.series_data) > 0, name=name)
    place_added_rois_on_planes(builder.ds, builder.series_data)
    builder.save(str(out))
    return out


def test_roi_fixer_output_on_mixed_ct_stays_on_plane(tmp_path, monkeypatch) -> None:
    from rtpipeline.roi_fixer import fix_rtstruct_rois

    ct_dir = tmp_path / "CT"
    _write_ct(ct_dir, _mixed_z())
    path = _on_plane_rtstruct(ct_dir, tmp_path / "rs.dcm", _mixed_z())
    before = create_scoped_rtstruct(ct_dir, path)
    before_masks = {n: before.get_roi_mask_by_name(n) for n in STRUCTURES}
    with monkeypatch.context() as patch:
        _force_rebuild(patch, "iliac_artery_left")
        summary = fix_rtstruct_rois(ct_dir, path)
    assert summary is not None and summary.changed and summary.fixed == ["iliac_artery_left"]
    _assert_resolves(path, ct_dir, STRUCTURES)
    after = create_scoped_rtstruct(ct_dir, path)
    for name in STRUCTURES:
        assert np.array_equal(after.get_roi_mask_by_name(name), before_masks[name]), name


# RS_auto_cropped -----------------------------------------------------------------


def _cropping_course(tmp_path: Path, zs, name: str = "crop") -> Path:
    course = tmp_path / name
    dirs = ar.build_course_dirs(course)
    _write_ct(dirs.dicom_ct, zs)
    _align_contract_ct_geometry(write_minimal_course_contract(course, planning_ct_dir=dirs.dicom_ct), dirs.dicom_ct)
    seg_dir = dirs.segmentation_totalseg / "CT_series"
    seg_dir.mkdir(parents=True)
    for roi, box in STRUCTURES.items():
        sitk.WriteImage(_box_image(zs, box), str(seg_dir / f"total--{roi}_cropped.nii.gz"))
    return course


def test_cropped_rtstruct_on_mixed_ct(tmp_path) -> None:
    from rtpipeline.anatomical_cropping import _create_rtstruct_from_cropped_masks

    course = _cropping_course(tmp_path, _mixed_z())
    out = _create_rtstruct_from_cropped_masks(course, ar.build_course_dirs(course), "_cropped")
    assert out == course / "RS_auto_cropped.dcm"
    ct_dir = ar.build_course_dirs(course).dicom_ct
    _assert_resolves(out, ct_dir, STRUCTURES)


# Uniform spacing: byte-identical to a177c59 ------------------------------------

_RUNNER = r'''
import datetime, hashlib, itertools, json, shutil, sys, types
from pathlib import Path

tree, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(tree))
import numpy as np
import pydicom.uid
import SimpleITK as sitk
import rt_utils.ds_helper as ds_helper
from rt_utils import RTStruct

counter = itertools.count()
ds_helper.generate_uid = lambda *a, **k: pydicom.uid.generate_uid(entropy_srcs=[f"uid-{next(counter)}"])

class _Fixed(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 12, 0, 0)

ds_helper.datetime = types.SimpleNamespace(datetime=_Fixed)

import rtpipeline
assert Path(rtpipeline.__file__).resolve().is_relative_to(tree), rtpipeline.__file__
from rtpipeline import anatomical_cropping, custom_models, segmentation
from rtpipeline import custom_structures_rtstruct as crs
from rtpipeline.layout import build_course_dirs
from rtpipeline.roi_fixer import fix_rtstruct_rois

spec = json.loads((work / "spec.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {}
course = work / "course"
rs = crs._create_custom_structures_rtstruct(course, work / "custom.yaml", course / "RS.dcm", course / "RS_auto.dcm")
out["RS_custom"] = sha(rs)
ct_dir = build_course_dirs(course).dicom_ct
out["model_rtstruct"] = sha(segmentation._ensure_model_rtstruct_from_masks(ct_dir, work / "seg", "CT_series", "total"))
masks = {p.name.split(".")[0]: sitk.ReadImage(str(p)) for p in sorted((work / "seg").glob("total--*.nii.gz"))}
(work / "model").mkdir()
out["custom_model_rtstruct"] = sha(custom_models._build_rtstruct(ct_dir, masks, work / "model", {n: work / "model" / n for n in masks}))
original = RTStruct.get_roi_mask_by_name
def failing(self, name):
    if name == spec["rebuild"]:
        raise ValueError("synthetic rasterisation failure")
    return original(self, name)
RTStruct.get_roi_mask_by_name = failing
summary = fix_rtstruct_rois(ct_dir, work / "fix.dcm")
RTStruct.get_roi_mask_by_name = original
assert summary is not None and summary.changed
out["roi_fixer"] = sha(work / "fix.dcm")
crop = work / "crop"
out["RS_auto_cropped"] = sha(anatomical_cropping._create_rtstruct_from_cropped_masks(crop, build_course_dirs(crop), "_cropped"))
print(json.dumps(out))
'''


def _old_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "old_tree"
    tree.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", OLD_REVISION, "rtpipeline"],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"revision {OLD_REVISION} is unavailable: {exc}")
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def _run_writers(runner: Path, tree: Path, work: Path) -> dict:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(tree),
        "OMP_NUM_THREADS": "1",
        "SITK_NUMBER_OF_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, str(runner), str(tree), str(work)],
        capture_output=True, text=True, env=env, cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_uniform_spacing_writer_outputs_are_byte_identical_to_a177c59(tmp_path) -> None:
    old_tree = _old_tree(tmp_path)
    zs = _uniform_z()
    template = tmp_path / "template"
    template.mkdir()
    case = _custom_course(template, zs)
    shutil.move(str(case["config"]), template / "custom.yaml")
    _model_masks(template / "seg", zs)
    fix_ct = ar.build_course_dirs(template / "course").dicom_ct
    _on_plane_rtstruct(fix_ct, template / "fix.dcm", zs)
    _cropping_course(template, zs, name="crop")
    (template / "spec.json").write_text(json.dumps({"rebuild": "iliac_artery_left"}), encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER, encoding="utf-8")

    results = {}
    for label, tree in (("old", old_tree), ("new", REPO)):
        # The course contract binds the course to its parent directory name.
        work = tmp_path / label / template.name
        shutil.copytree(template, work, copy_function=shutil.copy2)
        results[label] = _run_writers(runner, tree, work)
    assert set(results["new"]) == {
        "RS_custom", "model_rtstruct", "custom_model_rtstruct", "roi_fixer", "RS_auto_cropped"
    }
    assert results["new"] == results["old"]


# RS_auto orientation (x/y exchange fixed 2026-09-25) ------------------------------


def test_rs_auto_rebuilt_from_masks_keeps_structure_x_y_extent(tmp_path) -> None:
    from test_auto_rtstruct_exact_planes import _course

    course, ct_dir, _seg, _total = _course(tmp_path, _mixed_z())
    scoped = create_scoped_rtstruct(ct_dir, ar.build_auto_rtstruct(course))
    for name, box in STRUCTURES.items():
        _assert_box_placed(scoped.get_roi_mask_by_name(name), _mixed_z(), box)
