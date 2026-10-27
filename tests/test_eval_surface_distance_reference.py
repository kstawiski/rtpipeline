"""Reference agreement and convergence on continuous spheres.

Offset case: radius R=10 mm, centre separation d=6 mm (d<R). On either
continuous surface, q, the distance to the other centre, has density q/(2Rd)
on [R-d, R+d]. Folding about R gives |q-R| a uniform [0,d] distribution.
Thus surface Dice(t)=t/d for 0<=t<=d, mean=d/2 and directed HD95=0.95*d.

Concentric case: radii 8 and 12 mm. All continuous surface distances are 4 mm;
surface Dice is zero below 4 mm and one at or above 4 mm. The 1/2/3-mm
tolerances avoid that discontinuity. Exact zero errors may remain zero as
resolution improves; every initially nonzero error must decrease overall.

Run this file as a script to print computed values, analytic values and absolute
errors as JSON. These checks quantify discretization error; passing them does
not establish clinical or manuscript readiness.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline.eval._vendor import surface_distance
from rtpipeline.eval.geometry import compare_masks


SPACINGS_MM = (1.0, 0.5, 0.25)
TOLERANCES_MM = (1.0, 2.0, 3.0)
METRICS = (
    "surface_dice_at_1_mm",
    "surface_dice_at_2_mm",
    "surface_dice_at_3_mm",
    "mean_surface_distance_mm",
    "hd95_mm",
)


def _image(mask, spacing_xyz, origin=(0.0, 0.0, 0.0)):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin)
    return image


def _sphere(radius_mm, centre_x_mm, spacing_mm):
    axis = np.arange(-18.0, 18.0 + spacing_mm / 2, spacing_mm)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    mask = (x - centre_x_mm) ** 2 + y ** 2 + z ** 2 <= radius_mm ** 2
    return _image(mask, (spacing_mm,) * 3, origin=(-18.0,) * 3)


def sphere_measurements():
    """Six rows, each with computed metrics and independent analytic targets."""
    rows = []
    for case in ("offset", "concentric"):
        analytic = dict(zip(METRICS, (1 / 6, 2 / 6, 3 / 6, 3.0, 5.7)))
        if case == "concentric":
            analytic = dict(zip(METRICS, (0.0, 0.0, 0.0, 4.0, 4.0)))
        for spacing in SPACINGS_MM:
            if case == "offset":
                reference = _sphere(10.0, -3.0, spacing)
                test = _sphere(10.0, 3.0, spacing)
            else:
                reference = _sphere(8.0, 0.0, spacing)
                test = _sphere(12.0, 0.0, spacing)
            results = [compare_masks(reference, test, surface_tolerance_mm=t)
                       for t in TOLERANCES_MM]
            assert all(result.status == "OK" for result in results)
            computed = dict(zip(METRICS, (
                *(result.surface_dice for result in results),
                results[0].mean_surface_distance_mm,
                results[0].hd95_mm,
            )))
            rows.append({
                "case": case,
                "spacing_mm": spacing,
                "computed": computed,
                "analytic": analytic.copy(),
                "absolute_error": {key: abs(computed[key] - analytic[key])
                                   for key in METRICS},
            })
    return rows


@pytest.fixture(scope="module")
def measurements():
    return sphere_measurements()


@pytest.mark.parametrize("case", ["offset", "concentric"])
@pytest.mark.parametrize("metric", METRICS)
def test_sphere_errors_decrease_with_voxel_size(measurements, case, metric):
    rows = [row for row in measurements if row["case"] == case]
    assert [row["spacing_mm"] for row in rows] == list(SPACINGS_MM)
    errors = [row["absolute_error"][metric] for row in rows]
    assert all(fine <= coarse + 1e-12 for coarse, fine in zip(errors, errors[1:])), errors
    if errors[0] > 1e-12:
        assert errors[-1] < errors[0], errors
    if case == "offset":
        assert errors[2] < errors[1] < errors[0], errors


@pytest.mark.parametrize("case", ["offset", "concentric"])
@pytest.mark.parametrize("spacing", SPACINGS_MM)
def test_sphere_errors_are_bounded_at_each_resolution(measurements, case, spacing):
    row = next(row for row in measurements
               if row["case"] == case and row["spacing_mm"] == spacing)
    voxel_diagonal_mm = math.sqrt(3) * spacing
    for key in ("mean_surface_distance_mm", "hd95_mm"):
        assert row["absolute_error"][key] < voxel_diagonal_mm
    for tolerance, key in zip(TOLERANCES_MM, METRICS[:3]):
        assert 0 <= row["computed"][key] <= 1
        if case == "offset":
            # The continuous distance CDF has slope 1/d for d=6 mm.
            assert row["absolute_error"][key] < voxel_diagonal_mm / 6
        elif tolerance < 4 - voxel_diagonal_mm:
            assert row["computed"][key] == 0


@pytest.mark.parametrize("spacing_xyz", [(1.0, 2.0, 3.0), (2.7, 0.6, 1.1)])
@pytest.mark.parametrize("at_image_edge", [False, True])
@pytest.mark.parametrize("resample", [False, True])
def test_reference_metrics_on_anisotropic_reference_grid(
    spacing_xyz, at_image_edge, resample,
):
    # Unequal surfaces distinguish area pooling from averaging directed means;
    # unequal spacings distinguish NumPy z/y/x order from SimpleITK x/y/z order.
    reference_mask = np.zeros((9, 11, 14), dtype=bool)
    test_mask = np.zeros_like(reference_mask)
    start = 0 if at_image_edge else 1
    reference_mask[start:4, start:7, start:6] = True
    test_mask[2:8, 3:10, 3:12] = True
    reference = _image(reference_mask, spacing_xyz)
    origin = (spacing_xyz[0], 0.0, 0.0) if resample else (0.0,) * 3
    test = _image(test_mask, spacing_xyz, origin=origin)
    if resample:
        # An integer x shift has an exact expected nearest-neighbour result.
        aligned_test = np.zeros_like(test_mask)
        aligned_test[:, :, 1:] = test_mask[:, :, :-1]
    else:
        aligned_test = test_mask
    expected_distances = surface_distance.compute_surface_distances(
        reference_mask, aligned_test, spacing_mm=spacing_xyz[::-1],
    )
    directed_means = surface_distance.compute_average_surface_distance(expected_distances)
    areas = [expected_distances[key].sum()
             for key in ("surfel_areas_gt", "surfel_areas_pred")]
    expected_mean = np.average(directed_means, weights=areas)
    assert not np.isclose(expected_mean, np.mean(directed_means))
    tolerance_mm = 1.25

    result = compare_masks(reference, test, surface_tolerance_mm=tolerance_mm)

    assert result.status == ("GRID_MISMATCH_RESAMPLED" if resample else "OK")
    assert result.resampled is resample
    assert result.mean_surface_distance_mm == pytest.approx(expected_mean, abs=1e-12)
    assert result.hd95_mm == pytest.approx(
        surface_distance.compute_robust_hausdorff(expected_distances, 95), abs=1e-12,
    )
    assert result.surface_dice == pytest.approx(
        surface_distance.compute_surface_dice_at_tolerance(expected_distances, tolerance_mm),
        abs=1e-12,
    )
    assert result.dice == pytest.approx(
        2 * np.count_nonzero(reference_mask & aligned_test)
        / (reference_mask.sum() + aligned_test.sum()),
    )


if __name__ == "__main__":
    print(json.dumps(sphere_measurements(), indent=2, allow_nan=False))
