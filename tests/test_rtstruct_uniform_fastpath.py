"""The one-resample fast path of ``image_array_for_rtstruct`` is byte-identical.

``image_array_for_rtstruct`` samples a mask once per DICOM slice. On a regular
series it may instead resample once onto a 3-D reference, but only when that
provably returns the same bytes. These tests compare both paths on synthetic
series and require the per-slice path wherever identity is not proven. All
data here are synthetic.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset

from rtpipeline import rtstruct_geometry as rg

AXIAL = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _ds(value: float) -> str:
    """A DICOM DS string as scanners write it (at most 16 characters)."""
    text = f"{value:.10g}"
    return text if len(text) <= 16 else f"{value:.8f}"[:16]


def _rotation(a: float, b: float, c: float) -> np.ndarray:
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c))
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return rz @ ry @ rx


def _series(*, rows=36, columns=44, pixel=(0.9765625, 0.9765625), orientation=AXIAL,
            origin=(-21.5, -17.25, -40.0), steps=None, count=24, step=2.5):
    """Datasets carrying only the geometry the sampler reads, in series order."""
    row, column = (np.asarray(v, dtype=float) for v in orientation)
    normal = np.cross(row, column)
    normal /= np.linalg.norm(normal)
    steps = [step] * (count - 1) if steps is None else list(steps)
    offsets = np.concatenate([[0.0], np.cumsum(steps)])
    slices = []
    for offset in offsets:
        ds = Dataset()
        ds.Rows = rows
        ds.Columns = columns
        ds.PixelSpacing = [_ds(pixel[0]), _ds(pixel[1])]
        ds.ImageOrientationPatient = [_ds(v) for v in (*row, *column)]
        ds.ImagePositionPatient = [_ds(v) for v in np.asarray(origin, dtype=float) + offset * normal]
        slices.append(ds)
    return slices


def _series_grid(slices) -> sitk.Image:
    """The regular grid a converter derives from the series, as a mask would carry it."""
    first = slices[0]
    row = np.asarray([float(v) for v in first.ImageOrientationPatient[:3]])
    column = np.asarray([float(v) for v in first.ImageOrientationPatient[3:]])
    normal = np.cross(row, column)
    normal /= np.linalg.norm(normal)
    origins = np.asarray([[float(v) for v in s.ImagePositionPatient] for s in slices])
    step = float((origins[-1] - origins[0]) @ normal) / (len(slices) - 1)
    grid = sitk.Image(int(first.Columns), int(first.Rows), len(slices), sitk.sitkUInt8)
    grid.SetOrigin(tuple(origins[0] if step > 0 else origins[-1]))
    grid.SetSpacing((float(first.PixelSpacing[1]), float(first.PixelSpacing[0]), abs(step)))
    grid.SetDirection(tuple(np.column_stack((row, column, normal)).ravel()))
    return grid


def _mask(grid: sitk.Image, *, seed=0, border=False, pixel=sitk.sitkUInt8) -> sitk.Image:
    """A mask of blobs on ``grid``; ``border`` also fills every face of the grid."""
    rng = np.random.default_rng(seed)
    size = grid.GetSize()[::-1]
    array = np.zeros(size, dtype=np.int16)
    zz, yy, xx = np.indices(size)
    for label in range(1, 6):
        centre = rng.uniform(0, 1, 3) * np.asarray(size)
        radius = rng.uniform(0.15, 0.35, 3) * np.asarray(size)
        inside = ((zz - centre[0]) / radius[0]) ** 2 + ((yy - centre[1]) / radius[1]) ** 2 + ((xx - centre[2]) / radius[2]) ** 2 <= 1
        array[inside] = label
    array[rng.uniform(size=size) < 0.02] = 7
    if border:
        array[0], array[-1] = 3, 4
        array[:, 0], array[:, -1] = 5, 6
        array[:, :, 0], array[:, :, -1] = 7, 8
    image = sitk.Cast(sitk.GetImageFromArray(array), pixel)
    image.CopyInformation(grid)
    return image


def _nifti_roundtrip(image: sitk.Image, tmp_path: Path) -> sitk.Image:
    """Write and read the mask as a model output would (float32 NIfTI geometry)."""
    path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(path))
    return sitk.ReadImage(str(path))


def _coarse_grid(slices, spacing, shift) -> sitk.Image:
    """A grid covering the series at a coarser spacing, offset from its voxels."""
    fine = _series_grid(slices)
    extent = np.asarray(fine.GetSize()) * np.asarray(fine.GetSpacing())
    size = [int(math.ceil(e / s)) + 2 for e, s in zip(extent, spacing)]
    grid = sitk.Image(size, sitk.sitkUInt8)
    direction = np.asarray(fine.GetDirection()).reshape(3, 3)
    grid.SetOrigin(tuple(np.asarray(fine.GetOrigin()) + direction @ np.asarray(shift)))
    grid.SetSpacing(tuple(spacing))
    grid.SetDirection(fine.GetDirection())
    return grid


def _assert_identical(image: sitk.Image, slices, *, fast: bool) -> None:
    geometries = rg._slice_geometries(slices)
    assert (rg._uniform_reference(image, geometries) is not None) is fast
    expected = rg._image_array_per_slice(image, geometries)
    actual = rg.image_array_for_rtstruct(image, slices)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape == (int(slices[0].Rows), int(slices[0].Columns), len(slices))
    assert actual.strides == expected.strides
    assert actual.tobytes() == expected.tobytes()
    assert actual.any()


OBLIQUE = _rotation(0.35, -0.2, 0.15)
OBLIQUE_ORIENTATION = (tuple(OBLIQUE[:, 0]), tuple(OBLIQUE[:, 1]))

SAME_GRID_CASES = {
    "axial_2.5": {},
    "axial_2.4": {"step": 2.4},
    "axial_3.0": {"step": 3.0, "count": 17},
    "axial_1.25": {"step": 1.25, "count": 40},
    "axial_non_square_pixels": {"pixel": (1.2, 0.8), "step": 2.0},
    "non_integer_origin": {"origin": (-249.51171875, -180.3, -97.6), "step": 2.4},
    "negative_step_order": {"step": -3.0, "origin": (10.3, -4.7, 55.5)},
    "oblique": {"orientation": OBLIQUE_ORIENTATION, "step": 2.4, "origin": (-12.34, 8.9, -60.1)},
}


@pytest.mark.parametrize("case", sorted(SAME_GRID_CASES))
@pytest.mark.parametrize("roundtrip", [False, True], ids=["in_memory", "nifti"])
def test_fast_path_matches_per_slice_on_the_series_grid(tmp_path, case, roundtrip) -> None:
    slices = _series(**SAME_GRID_CASES[case])
    image = _mask(_series_grid(slices), seed=len(case))
    if roundtrip:
        image = _nifti_roundtrip(image, tmp_path)
    _assert_identical(image, slices, fast=True)


@pytest.mark.parametrize("case", ["axial_2.4", "negative_step_order", "oblique"])
def test_fast_path_matches_per_slice_on_a_coarser_mask_grid(tmp_path, case) -> None:
    slices = _series(**SAME_GRID_CASES[case])
    grid = _coarse_grid(slices, spacing=(1.5, 1.7, 3.3), shift=(-1.1, -0.9, -2.2))
    image = _nifti_roundtrip(_mask(grid, seed=3), tmp_path)
    _assert_identical(image, slices, fast=True)


@pytest.mark.parametrize("case", ["axial_2.5", "oblique"])
def test_fast_path_matches_per_slice_when_the_mask_touches_the_border(tmp_path, case) -> None:
    slices = _series(**SAME_GRID_CASES[case])
    image = _mask(_series_grid(slices), seed=5, border=True)
    _assert_identical(image, slices, fast=True)
    # A smaller mask grid: samples beyond its border take the default value.
    grid = _series_grid(slices)
    cropped = sitk.RegionOfInterest(_mask(grid, seed=6, border=True), (30, 20, 16), (5, 7, 3))
    _assert_identical(cropped, slices, fast=True)


@pytest.mark.parametrize("pixel", [sitk.sitkInt16, sitk.sitkFloat32])
def test_fast_path_keeps_the_mask_pixel_type(pixel) -> None:
    slices = _series(step=2.4)
    _assert_identical(_mask(_series_grid(slices), seed=9, pixel=pixel), slices, fast=True)


def test_mixed_spacing_series_takes_the_per_slice_path() -> None:
    slices = _series(steps=[2.5] * 8 + [5.0] * 4 + [2.5] * 8)
    grid = _coarse_grid(slices, spacing=(0.9765625, 0.9765625, 1.25), shift=(0.0, 0.0, 0.0))
    _assert_identical(_mask(grid, seed=1), slices, fast=False)


@pytest.mark.parametrize(
    "change",
    ["gap", "one_position_off", "pixel_spacing", "orientation", "single_slice", "duplicate_positions"],
)
def test_irregular_series_take_the_per_slice_path(change) -> None:
    slices = _series(step=2.0)
    grid = _coarse_grid(slices, spacing=(1.1, 1.1, 1.3), shift=(-0.3, -0.3, -0.4))
    if change == "gap":
        slices = slices[:10] + slices[11:]
    elif change == "one_position_off":
        slices[7].ImagePositionPatient[2] = _ds(float(slices[7].ImagePositionPatient[2]) + 2e-4)
    elif change == "pixel_spacing":
        slices[4].PixelSpacing = ["0.9765624", "0.9765625"]
    elif change == "orientation":
        slices[4].ImageOrientationPatient = ["1", "0", "0", "0", "0.99999999", "0.0001414"]
    elif change == "single_slice":
        slices = slices[:1]
    elif change == "duplicate_positions":
        for ds in slices:
            ds.ImagePositionPatient = list(slices[0].ImagePositionPatient)
    _assert_identical(_mask(grid, seed=2), slices, fast=False)


def test_samples_on_a_rounding_tie_take_the_per_slice_path() -> None:
    # Series voxel centres every 1 mm and mask voxel centres every 2 mm from the
    # same origin: every other sample lies exactly between two mask voxels.
    slices = _series(pixel=(1.0, 1.0), step=1.0, origin=(0.0, 0.0, 0.0))
    coarse = _coarse_grid(slices, spacing=(2.0, 2.0, 2.0), shift=(0.0, 0.0, 0.0))
    _assert_identical(_mask(coarse, seed=4), slices, fast=False)


def test_in_plane_rotated_mask_takes_the_per_slice_path() -> None:
    slices = _series(step=2.0)
    grid = _coarse_grid(slices, spacing=(1.1, 1.1, 1.3), shift=(-0.3, -0.3, -0.4))
    turn = _rotation(0.3, 0.0, 0.0)
    grid.SetDirection(tuple(turn.ravel()))
    _assert_identical(_mask(grid, seed=8), slices, fast=False)


def test_regular_series_is_sampled_with_one_resample(monkeypatch) -> None:
    slices = _series(step=2.4)
    image = _mask(_series_grid(slices), seed=11)
    calls = []
    original = sitk.Resample

    def counting(*args, **kwargs):
        calls.append(args[1].GetSize())
        return original(*args, **kwargs)

    monkeypatch.setattr(sitk, "Resample", counting)
    rg.image_array_for_rtstruct(image, slices)
    assert calls == [(44, 36, 24)]


def test_invalid_slice_geometry_still_raises() -> None:
    slices = _series()
    slices[3].Rows = 35
    with pytest.raises(ValueError, match="inconsistent slice dimensions"):
        rg.image_array_for_rtstruct(_mask(_series_grid(_series()), seed=1), slices)
    slices = _series()
    del slices[2].PixelSpacing
    with pytest.raises(ValueError, match="lacks rows, columns"):
        rg.image_array_for_rtstruct(_mask(_series_grid(_series()), seed=1), slices)
    with pytest.raises(ValueError, match="no DICOM slices"):
        rg.image_array_for_rtstruct(_mask(_series_grid(_series()), seed=1), [])
