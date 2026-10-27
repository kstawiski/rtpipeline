from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import SimpleITK as sitk

from ._vendor import surface_distance
from ._vendor.surface_distance import lookup_tables


# Absolute tolerances: millimetres for origin/spacing, unitless for direction.
GRID_TOLERANCE = 1e-6
MaskInput = Path | str | sitk.Image
AgreementStatus = Literal[
    "OK", "BOTH_EMPTY", "REFERENCE_EMPTY", "TEST_EMPTY",
    "GRID_MISMATCH_RESAMPLED", "INCOMPATIBLE",
]


@dataclass(slots=True)
class MaskGeometry:
    path: str
    voxels: int
    volume_cm3: float
    bbox_zyx: list[list[int]] | None

    def to_dict(self) -> dict:
        return asdict(self)


def mask_geometry(path: Path) -> MaskGeometry:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    voxels = int(arr.sum())
    voxel_volume_cm3 = float(np.prod(img.GetSpacing()) / 1000.0)
    bbox = None
    if voxels:
        coords = np.argwhere(arr)
        bbox = [coords.min(axis=0).astype(int).tolist(), coords.max(axis=0).astype(int).tolist()]
    return MaskGeometry(str(path), voxels, float(voxels * voxel_volume_cm3), bbox)


@dataclass(slots=True)
class MaskAgreement:
    """Contour agreement; volumes are native mask volumes unless named otherwise.

    Empty flags describe the input masks, before resampling. Unknown volumes and
    flags are None when an input cannot be evaluated. Distances are undefined
    (None) if either mask is empty; Dice and surface Dice are zero for exactly
    one empty mask and None for two empty masks. INCOMPATIBLE has no scores.
    """

    dice: float | None = None
    reference_volume_cm3: float | None = None
    test_volume_cm3: float | None = None
    comparison_test_volume_cm3: float | None = None
    both_empty: bool | None = None
    reference_empty: bool | None = None
    test_empty: bool | None = None
    resampled: bool = False
    status: AgreementStatus = "INCOMPATIBLE"
    hd95_mm: float | None = None
    mean_surface_distance_mm: float | None = None
    surface_dice: float | None = None
    surface_tolerance_mm: float = 1.0
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _read_mask(source: MaskInput) -> tuple[sitk.Image, np.ndarray]:
    image = source if isinstance(source, sitk.Image) else sitk.ReadImage(str(source))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Masks must be scalar, three-dimensional images.")
    if any(size == 0 for size in image.GetSize()):
        raise ValueError("Masks must have a nonzero grid size.")
    spacing = np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if (
        not np.all(np.isfinite(image.GetOrigin()))
        or not np.all(np.isfinite(spacing))
        or not np.all(spacing > 0)
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("Mask geometry must be finite, with positive spacing.")
    # EDT sampling represents orthogonal axes. Reflections are valid; shear is not.
    if not np.allclose(direction.T @ direction, np.eye(3), atol=GRID_TOLERANCE, rtol=0):
        raise ValueError("Mask direction must be orthonormal.")
    array = sitk.GetArrayFromImage(image)
    if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise ValueError("Mask values must be finite, real numbers.")
    return image, array > 0


def _same_grid(reference: sitk.Image, test: sitk.Image, tolerance: float) -> bool:
    return reference.GetSize() == test.GetSize() and all(
        np.allclose(a, b, atol=tolerance, rtol=0)
        for a, b in (
            (reference.GetOrigin(), test.GetOrigin()),
            (reference.GetSpacing(), test.GetSpacing()),
            (reference.GetDirection(), test.GetDirection()),
        )
    )


def _resample_mask(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        image, reference, sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )


def compare_masks(
    reference: MaskInput,
    test: MaskInput,
    *,
    surface_tolerance_mm: float = 1.0,
    grid_tolerance: float = GRID_TOLERANCE,
) -> MaskAgreement:
    """Compare positive foreground in scalar 3-D masks sharing physical coordinates.

    Image coordinates and spacing must be in mm. This function does not perform
    registration or establish that two images belong to the same patient/frame.
    Size must match exactly. Origin/spacing (mm) and direction entries (unitless)
    must match within ``grid_tolerance`` (default 1e-6, absolute, no relative
    tolerance). Otherwise the test mask is resampled onto the reference grid with
    an identity physical transform and nearest-neighbour interpolation.

    Native volumes and input empty flags are retained. Empty status takes
    precedence over GRID_MISMATCH_RESAMPLED; ``resampled`` records the operation
    independently. Nonempty test foreground whose voxel centres fall outside
    the reference voxel support, or which disappears on resampling, makes a
    nonempty-mask comparison INCOMPATIBLE rather than silently scoring truncation.

    Scores use the reference grid and the Nikolov/DeepMind surface-distance
    method: 2x2x2 neighbourhoods locate surface elements on the corner lattice,
    with marching-cubes areas in mm², including anisotropic spacing. Exterior
    background closes surfaces at the image edge. HD95 is the maximum of the
    two directed area-weighted 95th percentiles (inverse empirical CDF). Mean
    surface distance pools both directions by area. Surface Dice is the fraction
    of total area within ``surface_tolerance_mm`` (inclusive). Distances between
    corner-lattice samples approximate continuous surfaces as resolution improves.

    Invalid comparison parameters raise ValueError. Unreadable or unsupported
    masks return INCOMPATIBLE with a reason and no scores.
    """
    for name, value in (
        ("surface_tolerance_mm", surface_tolerance_mm),
        ("grid_tolerance", grid_tolerance),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    result = MaskAgreement(surface_tolerance_mm=float(surface_tolerance_mm))
    try:
        reference_image, reference_array = _read_mask(reference)
        test_image, test_array = _read_mask(test)
    except (RuntimeError, OSError, ValueError) as error:
        result.reason = str(error)
        return result

    reference_count = int(np.count_nonzero(reference_array))
    test_count = int(np.count_nonzero(test_array))
    reference_voxel_cm3 = float(np.prod(reference_image.GetSpacing()) / 1000.0)
    result.reference_volume_cm3 = reference_count * reference_voxel_cm3
    result.test_volume_cm3 = test_count * float(np.prod(test_image.GetSpacing()) / 1000.0)
    result.reference_empty = reference_count == 0
    result.test_empty = test_count == 0
    result.both_empty = result.reference_empty and result.test_empty

    if not _same_grid(reference_image, test_image, grid_tolerance):
        try:
            binary_test = sitk.GetImageFromArray(test_array.astype(np.uint8))
            binary_test.CopyInformation(test_image)
            aligned_test = _resample_mask(binary_test, reference_image)
            result.resampled = True
            aligned_array = sitk.GetArrayFromImage(aligned_test) > 0
            if reference_count and test_count:
                # Check coverage at native test voxel centres before accepting a
                # score. Zero padding must not hide foreground outside the FOV.
                coverage = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8) + 1
                coverage.CopyInformation(reference_image)
                covered = sitk.GetArrayFromImage(_resample_mask(coverage, test_image)) > 0
                if np.any(test_array & ~covered):
                    result.reason = "Test foreground extends outside the reference field of view."
                    return result
                if not np.any(aligned_array):
                    result.reason = "Resampling removed all test foreground from the comparison grid."
                    return result
            test_array = aligned_array
        except RuntimeError as error:
            result.reason = f"Mask resampling failed: {error}"
            return result

    aligned_test_count = int(np.count_nonzero(test_array))
    result.comparison_test_volume_cm3 = aligned_test_count * reference_voxel_cm3
    if result.both_empty:
        result.status = "BOTH_EMPTY"
        return result
    if result.reference_empty or result.test_empty:
        result.status = "REFERENCE_EMPTY" if result.reference_empty else "TEST_EMPTY"
        result.dice = 0.0
        result.surface_dice = 0.0
        return result

    result.status = "GRID_MISMATCH_RESAMPLED" if result.resampled else "OK"
    result.dice = float(
        2 * np.count_nonzero(reference_array & test_array)
        / (reference_count + aligned_test_count)
    )
    spacing_zyx = reference_image.GetSpacing()[::-1]
    distances = surface_distance.compute_surface_distances(
        reference_array, test_array, spacing_mm=spacing_zyx,
    )
    result.hd95_mm = float(surface_distance.compute_robust_hausdorff(distances, 95))
    reference_mean, test_mean = surface_distance.compute_average_surface_distance(distances)
    reference_area = distances["surfel_areas_gt"].sum()
    test_area = distances["surfel_areas_pred"].sum()
    # The reference returns two directed means. Retain the API's symmetric,
    # area-pooled mean, rather than weighting unequal surfaces equally.
    result.mean_surface_distance_mm = float(
        (reference_mean * reference_area + test_mean * test_area)
        / (reference_area + test_area)
    )
    result.surface_dice = float(surface_distance.compute_surface_dice_at_tolerance(
        distances, surface_tolerance_mm,
    ))
    return result


def _area_percentile(distances: np.ndarray, areas: np.ndarray, fraction: float) -> float:
    order = np.argsort(distances)
    cumulative = np.cumsum(areas[order])
    index = np.searchsorted(cumulative / areas.sum(), fraction, side="left")
    return float(distances[order[min(index, len(order) - 1)]])


@lru_cache(maxsize=32)
def _surfel_area_table(spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Reference area table retained for existing callers of this private helper."""
    return lookup_tables.create_table_neighbour_code_to_surface_area(spacing_zyx)


def dice(mask_a: MaskInput, mask_b: MaskInput) -> float | None:
    """Deprecated scalar wrapper for compare_masks; empty pairs return None."""
    warnings.warn(
        "dice() is deprecated; use compare_masks() for metrics and status.",
        DeprecationWarning, stacklevel=2,
    )
    return compare_masks(mask_a, mask_b).dice


def summarize_masks(paths: Iterable[Path]) -> list[dict]:
    return [mask_geometry(path).to_dict() for path in paths]
