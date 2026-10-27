"""Physical-space agreement using synthetic scalar SimpleITK masks only."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline.eval.geometry import (
    GRID_TOLERANCE,
    compare_masks,
    dice,
    mask_geometry,
    summarize_masks,
)


def _image(array, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None):
    image = sitk.GetImageFromArray(np.asarray(array), isVector=False)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    return image


def _point(index_zyx=(4, 4, 4), *, size=12, **geometry):
    array = np.zeros((size, size, size), dtype=np.uint8)
    array[index_zyx] = 1
    return _image(array, **geometry)


def _assert_no_scores(result):
    assert result.dice is None
    assert result.hd95_mm is None
    assert result.mean_surface_distance_mm is None
    assert result.surface_dice is None


def test_identical_masks_have_physical_volumes_and_perfect_scores():
    array = np.zeros((8, 9, 10), dtype=np.uint16)
    array[1:4, 2:6, 3:8] = 7  # All positive values are foreground.
    image = _image(array, spacing=(1.5, 2.0, 3.0))

    result = compare_masks(image, image, surface_tolerance_mm=0)

    assert result.status == "OK"
    assert result.dice == result.surface_dice == 1.0
    assert result.hd95_mm == result.mean_surface_distance_mm == 0.0
    assert result.reference_volume_cm3 == pytest.approx(60 * 9 / 1000)
    assert result.test_volume_cm3 == result.reference_volume_cm3
    assert result.comparison_test_volume_cm3 == result.reference_volume_cm3
    assert result.reference_empty is result.test_empty is result.both_empty is False
    assert result.resampled is False
    assert result.reason is None
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_same_array_with_shifted_origin_uses_physical_axis_spacing(axis):
    spacing = (2.0, 3.0, 4.0)
    reference = _point(spacing=spacing)
    origin = [0.0] * 3
    origin[axis] = spacing[axis]
    test = _point(spacing=spacing, origin=origin)

    result = compare_masks(reference, test, surface_tolerance_mm=spacing[axis] - 0.01)

    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.resampled is True
    assert result.dice == 0.0
    assert result.hd95_mm == pytest.approx(spacing[axis])
    # Adjacent voxel surfaces share four of their eight corner samples.
    assert result.mean_surface_distance_mm == pytest.approx(spacing[axis] / 2)
    assert result.surface_dice == 0.5
    inclusive = compare_masks(reference, test, surface_tolerance_mm=spacing[axis])
    assert inclusive.surface_dice == 1.0


def test_shifted_array_and_origin_can_describe_identical_physical_masks():
    reference = _point((4, 4, 5))
    test = _point((4, 4, 4), origin=(1.0, 0.0, 0.0))
    result = compare_masks(reference, test)
    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.dice == result.surface_dice == 1.0
    assert result.hd95_mm == result.mean_surface_distance_mm == 0.0


def test_different_spacing_is_resampled_with_nearest_neighbour():
    array = np.zeros((12, 12, 12), dtype=np.uint8)
    array[4, 4, 3:5] = 1
    reference = _image(array)
    # Coarse voxel centred at x=4 mm covers fine-grid centres x=3 and x=4.
    test = _point((4, 4, 2), spacing=(2.0, 1.0, 1.0))

    result = compare_masks(reference, test)

    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.resampled is True
    assert result.dice == result.surface_dice == 1.0
    assert result.hd95_mm == result.mean_surface_distance_mm == 0.0
    assert result.reference_volume_cm3 == pytest.approx(0.002)
    assert result.test_volume_cm3 == pytest.approx(0.002)
    assert result.comparison_test_volume_cm3 == pytest.approx(0.002)


def test_native_test_volume_is_retained_when_resampling_changes_voxel_count():
    # A 1.5-mm-wide voxel centred at x=4 maps to a single 1-mm reference voxel.
    reference = _point((4, 4, 4))
    test = _point((4, 4, 3), spacing=(1.5, 1.0, 1.0), origin=(-0.5, 0, 0))
    result = compare_masks(reference, test)
    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.dice == 1.0
    assert result.test_volume_cm3 == pytest.approx(0.0015)
    assert result.comparison_test_volume_cm3 == pytest.approx(0.001)


def test_different_grid_sizes_can_cover_the_same_contour():
    result = compare_masks(_point(size=12), _point(size=8))
    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.resampled is True
    assert result.dice == result.surface_dice == 1.0


def test_flipped_direction_and_shifted_origin_preserve_physical_location():
    reference = _point((4, 4, 2), size=10)
    test = _point(
        (4, 4, 7), size=10, origin=(9.0, 0.0, 0.0),
        direction=(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    result = compare_masks(reference, test)
    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.dice == result.surface_dice == 1.0
    assert result.hd95_mm == result.mean_surface_distance_mm == 0.0


def test_rotated_direction_is_resampled_in_physical_coordinates():
    reference = _point((4, 3, 8))
    test = _point(
        (4, 3, 3), origin=(11.0, 0.0, 0.0),
        direction=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    result = compare_masks(reference, test)
    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.dice == 1.0


def test_common_oblique_grid_has_correct_euclidean_distances():
    angle = math.pi / 4
    c, s = math.cos(angle), math.sin(angle)
    geometry = dict(
        spacing=(2.0, 3.0, 4.0), origin=(12.0, -20.0, 30.0),
        direction=(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
    )
    result = compare_masks(
        _point((4, 4, 4), **geometry), _point((5, 5, 5), **geometry),
        surface_tolerance_mm=math.sqrt(29) + 1e-10,
    )
    assert result.status == "OK"
    assert result.hd95_mm == pytest.approx(math.sqrt(29))
    # Eight equally weighted corners have the distances of all subsets of the
    # three axis steps, including the shared corner at distance zero.
    expected_mean = (2 + 3 + 4 + math.sqrt(13) + math.sqrt(20) + 5 + math.sqrt(29)) / 8
    assert result.mean_surface_distance_mm == pytest.approx(expected_mean)
    assert result.surface_dice == 1.0


@pytest.mark.parametrize("metadata", ["origin", "spacing", "direction"])
@pytest.mark.parametrize("factor, resampled", [(0.5, False), (2.0, True)])
def test_grid_metadata_tolerance_is_explicit(metadata, factor, resampled):
    reference = _point()
    test = _point()
    delta = GRID_TOLERANCE * factor
    if metadata == "origin":
        test.SetOrigin((delta, 0.0, 0.0))
    elif metadata == "spacing":
        test.SetSpacing((1.0 + delta, 1.0, 1.0))
    else:
        c, s = math.cos(delta), math.sin(delta)
        test.SetDirection((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
    result = compare_masks(reference, test)
    assert result.resampled is resampled
    assert result.status == ("GRID_MISMATCH_RESAMPLED" if resampled else "OK")
    assert result.dice == 1.0


def test_grid_tolerance_is_absolute_at_large_physical_coordinates():
    result = compare_masks(
        _point(origin=(1e6, 0, 0)), _point(origin=(1e6 + 1e-3, 0, 0)),
    )
    assert result.resampled is True
    assert result.status == "GRID_MISMATCH_RESAMPLED"


@pytest.mark.parametrize("shift, expected", [(0.01, False), (0.2, True)])
def test_grid_tolerance_can_be_selected(shift, expected):
    result = compare_masks(_point(), _point(origin=(shift, 0, 0)), grid_tolerance=0.1)
    assert result.resampled is expected


@pytest.mark.parametrize("reference_empty, test_empty, status", [
    (True, True, "BOTH_EMPTY"),
    (True, False, "REFERENCE_EMPTY"),
    (False, True, "TEST_EMPTY"),
])
@pytest.mark.parametrize("mismatch", [False, True])
def test_empty_masks_have_explicit_status_and_undefined_distances(
    reference_empty, test_empty, status, mismatch,
):
    empty = np.zeros((12, 12, 12), dtype=np.uint8)
    reference = _image(empty) if reference_empty else _point()
    test = _image(empty) if test_empty else _point()
    if mismatch:
        test.SetOrigin((1.0, 0.0, 0.0))

    result = compare_masks(reference, test)

    assert result.status == status
    assert result.reference_empty is reference_empty
    assert result.test_empty is test_empty
    assert result.both_empty is (reference_empty and test_empty)
    assert result.resampled is mismatch
    assert result.reference_volume_cm3 == pytest.approx(0 if reference_empty else 0.001)
    assert result.test_volume_cm3 == pytest.approx(0 if test_empty else 0.001)
    assert result.hd95_mm is result.mean_surface_distance_mm is None
    if result.both_empty:
        _assert_no_scores(result)
    else:
        assert result.dice == result.surface_dice == 0.0


@pytest.mark.parametrize("partial", [False, True])
def test_foreground_outside_reference_fov_is_incompatible_not_absent(partial):
    array = np.zeros((12, 12, 12), dtype=np.uint8)
    array[4, 4, 10] = 1
    if partial:
        array[4, 4, 2] = 1  # Part of the test contour still overlaps reference.
    result = compare_masks(_point(), _image(array, origin=(4.0, 0, 0)))
    assert result.status == "INCOMPATIBLE"
    assert result.resampled is True
    assert result.reference_empty is result.test_empty is result.both_empty is False
    assert result.test_volume_cm3 == pytest.approx(0.002 if partial else 0.001)
    assert "field of view" in result.reason
    _assert_no_scores(result)


def test_foreground_lost_on_coarse_grid_is_incompatible_not_absent():
    reference = _point(spacing=(4.0, 4.0, 4.0))
    test = _point((5, 5, 5), size=20)
    result = compare_masks(reference, test)
    assert result.status == "INCOMPATIBLE"
    assert result.resampled is True
    assert result.test_empty is False
    assert result.test_volume_cm3 == pytest.approx(0.001)
    assert "removed all test foreground" in result.reason
    _assert_no_scores(result)


@pytest.mark.parametrize("array", [np.zeros((8, 8), np.uint8), np.zeros((2, 3, 4, 5), np.uint8)])
def test_non_3d_masks_are_incompatible(array):
    result = compare_masks(sitk.GetImageFromArray(array, isVector=False), _point())
    assert result.status == "INCOMPATIBLE"
    assert "three-dimensional" in result.reason
    _assert_no_scores(result)


def test_vector_masks_are_incompatible():
    vector = sitk.GetImageFromArray(np.zeros((8, 8, 8, 3), np.uint8), isVector=True)
    result = compare_masks(_point(), vector)
    assert result.status == "INCOMPATIBLE"
    _assert_no_scores(result)


def test_nonorthogonal_grid_is_incompatible():
    test = _point(direction=(1.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    result = compare_masks(_point(), test)
    assert result.status == "INCOMPATIBLE"
    assert "orthonormal" in result.reason
    _assert_no_scores(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_mask_values_are_incompatible(value):
    array = np.zeros((8, 8, 8), dtype=np.float32)
    array[4, 4, 4] = value
    result = compare_masks(_point(), _image(array))
    assert result.status == "INCOMPATIBLE"
    assert "finite" in result.reason
    _assert_no_scores(result)


@pytest.mark.parametrize("parameter", ["surface_tolerance_mm", "grid_tolerance"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_comparison_parameters_raise(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        compare_masks(_point(), _point(), **{parameter: value})


def test_unreadable_mask_has_incompatible_status(tmp_path):
    result = compare_masks(_point(), tmp_path / "missing.nii.gz")
    assert result.status == "INCOMPATIBLE"
    assert result.reason
    assert result.test_empty is None
    _assert_no_scores(result)


def test_entirely_filled_masks_include_image_edge_surfaces():
    image = _image(np.ones((4, 5, 6), dtype=np.uint8), spacing=(1, 2, 3))
    result = compare_masks(image, image, surface_tolerance_mm=0)
    assert result.status == "OK"
    assert result.dice == result.surface_dice == 1.0
    assert result.hd95_mm == result.mean_surface_distance_mm == 0.0


def test_disjoint_contained_masks_have_zero_overlap_and_finite_distances():
    result = compare_masks(_point((2, 2, 2)), _point((8, 2, 2)))
    assert result.status == "OK"
    assert result.dice == result.surface_dice == 0
    assert result.hd95_mm == 6.0
    assert result.mean_surface_distance_mm == 5.5


def test_hd95_uses_directed_percentiles_and_mean_pools_both_surfaces():
    reference = np.zeros((4, 4, 30), dtype=np.uint8)
    reference[2, 2, 2:22:2] = 1  # Ten separate foreground voxels.
    test = reference.copy()
    test[2, 2, 26] = 1  # One additional point, 6 mm from the nearest reference.
    result = compare_masks(_image(reference), _image(test), surface_tolerance_mm=0)
    # Eight equal-area corners per voxel: the extra voxel contributes four
    # distances of 5 mm and four of 6 mm. Its directed inverse-CDF HD95 is 5 mm;
    # pooling both directions would instead put the 95th percentile at zero.
    assert result.hd95_mm == pytest.approx(5.0)
    assert result.mean_surface_distance_mm == pytest.approx(5.5 / 21)
    assert result.surface_dice == pytest.approx(20 / 21)


def _sphere(radius, center=(0.0, 0.0, 0.0), spacing=0.5):
    # 73 cubed samples, covering [-18, 18] mm in each axis at default spacing.
    axis = np.arange(-18, 18 + spacing / 2, spacing)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    cx, cy, cz = center
    array = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2)
    return _image(array.astype(np.uint8), spacing=(spacing,) * 3, origin=(-18.0,) * 3)


def _sphere_volume(radius):
    return 4 * math.pi * radius ** 3 / 3


def _equal_sphere_intersection(radius, separation):
    return math.pi * (4 * radius + separation) * (2 * radius - separation) ** 2 / 12


def test_translated_spheres_against_analytic_overlap_and_surface_distances():
    radius, separation, spacing = 10.0, 6.0, 0.5
    reference = _sphere(radius, center=(-separation / 2, 0, 0), spacing=spacing)
    test = _sphere(radius, center=(separation / 2, 0, 0), spacing=spacing)
    result = compare_masks(reference, test, surface_tolerance_mm=separation / 2)

    expected_dice = _equal_sphere_intersection(radius, separation) / _sphere_volume(radius)
    # A centre-sampled sphere's voxel union is bounded by spheres of radii
    # r +/- half the voxel diagonal. Bound numerator and denominator separately.
    half_diagonal = math.sqrt(3) * spacing / 2
    lower = _equal_sphere_intersection(radius - half_diagonal, separation) / _sphere_volume(radius + half_diagonal)
    upper = _equal_sphere_intersection(radius + half_diagonal, separation) / _sphere_volume(radius - half_diagonal)
    assert result.status == "OK"
    assert lower <= result.dice <= upper
    assert abs(result.dice - expected_dice) <= 3 * half_diagonal / radius
    assert _sphere_volume(radius - half_diagonal) / 1000 <= result.reference_volume_cm3 <= _sphere_volume(radius + half_diagonal) / 1000

    # For equal spheres with 0 < separation < radius, unsigned nearest surface
    # distances have uniform [0, separation] distribution under continuous area
    # sampling: HD95=.95*d, mean=d/2, and surface Dice at d/2 is .5.
    # Boundary voxel centres discretize each surface within one voxel diagonal;
    # two diagonals allow for both sampled surfaces and nonuniform sample density.
    distance_tolerance = 2 * math.sqrt(3) * spacing
    assert result.hd95_mm == pytest.approx(0.95 * separation, abs=distance_tolerance)
    assert result.mean_surface_distance_mm == pytest.approx(separation / 2, abs=distance_tolerance)
    assert result.surface_dice == pytest.approx(0.5, abs=distance_tolerance / separation)


def test_concentric_spheres_have_known_surface_separation():
    inner_radius, outer_radius, spacing = 8.0, 12.0, 0.5
    reference, test = _sphere(inner_radius), _sphere(outer_radius)
    gap = outer_radius - inner_radius
    # Each digital boundary is within one voxel diagonal of its analytic sphere.
    distance_tolerance = 2 * math.sqrt(3) * spacing
    below = compare_masks(reference, test, surface_tolerance_mm=gap - distance_tolerance)
    above = compare_masks(reference, test, surface_tolerance_mm=gap + distance_tolerance)
    expected_dice = 2 * inner_radius ** 3 / (inner_radius ** 3 + outer_radius ** 3)
    half_diagonal = math.sqrt(3) * spacing / 2
    lower = 2 * (inner_radius - half_diagonal) ** 3 / (
        (inner_radius - half_diagonal) ** 3 + (outer_radius + half_diagonal) ** 3
    )
    upper = 2 * (inner_radius + half_diagonal) ** 3 / (
        (inner_radius + half_diagonal) ** 3 + (outer_radius - half_diagonal) ** 3
    )
    assert lower <= below.dice <= upper
    assert below.dice == pytest.approx(expected_dice, abs=3 * half_diagonal / inner_radius)
    assert below.hd95_mm == pytest.approx(gap, abs=distance_tolerance)
    assert below.mean_surface_distance_mm == pytest.approx(gap, abs=distance_tolerance)
    assert below.surface_dice == 0.0
    assert above.surface_dice == 1.0


def test_file_inputs_deprecated_wrapper_and_existing_geometry_helpers(tmp_path):
    reference = _point(spacing=(1.0, 2.0, 3.0))
    test = _point(spacing=(1.0, 2.0, 3.0), origin=(1.0, 0, 0))
    reference_path, test_path = tmp_path / "reference.mha", tmp_path / "test.mha"
    sitk.WriteImage(reference, str(reference_path))
    sitk.WriteImage(test, str(test_path))
    result = compare_masks(reference_path, str(test_path))
    assert result.status == "GRID_MISMATCH_RESAMPLED"
    assert result.dice == 0.0
    with pytest.warns(DeprecationWarning, match="compare_masks"):
        assert dice(reference_path, test_path) == 0.0
    geometry = mask_geometry(reference_path)
    assert geometry.voxels == 1
    assert geometry.volume_cm3 == pytest.approx(0.006)
    assert geometry.bbox_zyx == [[4, 4, 4], [4, 4, 4]]
    assert summarize_masks([reference_path]) == [geometry.to_dict()]


def test_deprecated_wrapper_never_calls_empty_pairs_perfect():
    empty = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    with pytest.warns(DeprecationWarning):
        assert dice(empty, empty) is None
    with pytest.warns(DeprecationWarning):
        assert dice(_point(), sitk.Image([8, 8], sitk.sitkUInt8)) is None
