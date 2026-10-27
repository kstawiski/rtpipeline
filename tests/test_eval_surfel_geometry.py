"""Area-weighted surfaces on synthetic physical-space masks."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

from rtpipeline.eval.geometry import compare_masks, _area_percentile, _surfel_area_table


def image(array, spacing):
    result = sitk.GetImageFromArray(array.astype(np.uint8))
    result.SetSpacing(tuple(spacing))
    return result


def sphere(radius, centre, spacing):
    axis = np.arange(-18, 18 + spacing / 2, spacing)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    return image((x-centre)**2 + y*y + z*z <= radius*radius, (spacing,) * 3)


def measurements():
    rows = []
    for h in (1., .5, .25):
        reference, test = sphere(10., -3., h), sphere(10., 3., h)
        result = compare_masks(reference, test, surface_tolerance_mm=3.)
        arrays = [sitk.GetArrayFromImage(i) > 0 for i in (reference, test)]
        boundary = [m & ~ndimage.binary_erosion(m) for m in arrays]
        distances = [ndimage.distance_transform_edt(~boundary[1-i], sampling=(h,)*3)[boundary[i]]
                     for i in (0, 1)]
        old_dice = sum(np.count_nonzero(d <= 3) for d in distances) / sum(d.size for d in distances)
        rows.append(dict(case="offset", spacing_mm=h, analytic_hd95_mm=5.7,
                         analytic_mean_mm=3., analytic_surface_dice=.5,
                         hd95_mm=result.hd95_mm, mean_mm=result.mean_surface_distance_mm,
                         surface_dice=result.surface_dice, equal_voxel_surface_dice=old_dice))
        # The continuous surfaces have constant separation 4 mm. Tolerances
        # avoid the discontinuity of surface Dice at exactly that separation.
        for tolerance, expected in ((3., 0.), (5., 1.)):
            result = compare_masks(sphere(8., 0., h), sphere(12., 0., h),
                                   surface_tolerance_mm=tolerance)
            rows.append(dict(case=f"concentric_tolerance_{tolerance:g}", spacing_mm=h,
                             analytic_hd95_mm=4., analytic_mean_mm=4.,
                             analytic_surface_dice=expected, hd95_mm=result.hd95_mm,
                             mean_mm=result.mean_surface_distance_mm,
                             surface_dice=result.surface_dice))
    return rows


def test_spheres_approach_continuous_area_weighted_results():
    rows = measurements()
    offset = [r for r in rows if r["case"] == "offset"]
    for key, target in (("hd95_mm", 5.7), ("mean_mm", 3.), ("surface_dice", .5)):
        errors = [abs(r[key] - target) for r in offset]
        assert errors[2] < errors[1] < errors[0]
    for row in rows:
        h = row["spacing_mm"]
        assert abs(row["hd95_mm"] - row["analytic_hd95_mm"]) < math.sqrt(3) * h
        assert abs(row["mean_mm"] - row["analytic_mean_mm"]) < math.sqrt(3) * h
        assert abs(row["surface_dice"] - row["analytic_surface_dice"]) < h / 4
    # The brief's earlier 0.537 versus 0.500 is reproducible with this geometry.
    assert offset[1]["equal_voxel_surface_dice"] == pytest.approx(.537, abs=.001)


def test_weighted_percentile_is_inverse_cdf_and_accounts_for_area():
    assert _area_percentile(np.array([0., 100.]), np.array([96., 4.]), .95) == 0.
    assert _area_percentile(np.array([0., 100.]), np.array([94., 6.]), .95) == 100.


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_adjacent_single_voxels_use_corner_surfaces_in_physical_space(axis):
    spacing = (2., 3., 4.)
    a = np.zeros((5, 5, 5), bool); a[2, 2, 2] = True
    b = np.roll(a, 1, axis=2-axis)
    result = compare_masks(image(a, spacing), image(b, spacing),
                           surface_tolerance_mm=spacing[axis] - .01)
    assert result.hd95_mm == pytest.approx(spacing[axis])
    assert result.mean_surface_distance_mm == pytest.approx(spacing[axis] / 2)
    assert result.surface_dice == pytest.approx(.5)
    assert compare_masks(image(a, spacing), image(b, spacing),
                         surface_tolerance_mm=spacing[axis]).surface_dice == 1.


def test_anisotropic_surfel_areas_and_image_edge_closure():
    table = _surfel_area_table((2., 3., 4.))
    assert table[1] == pytest.approx(np.linalg.norm(np.array([12., 8., 6.]) / 8))
    assert table[15] == pytest.approx(12.)
    assert table[0] == table[255] == 0.
    a = np.ones((3, 4, 5), bool)
    result = compare_masks(image(a, (2., 3., 4.)), image(a, (2., 3., 4.)), surface_tolerance_mm=0)
    assert result.surface_dice == 1.
    assert result.mean_surface_distance_mm == result.hd95_mm == 0.


if __name__ == "__main__":
    import sys
    Path(sys.argv[1]).write_text(json.dumps(measurements(), indent=2) + "\n")
