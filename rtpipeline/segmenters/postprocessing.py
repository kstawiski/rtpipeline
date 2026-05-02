from __future__ import annotations

from glob import glob
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def _as_reference_geometry(img: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if _same_geometry(img, reference):
        return img
    return sitk.Resample(
        img,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        img.GetPixelID(),
    )


def total_lung_mask(course_root: Path, reference: sitk.Image) -> sitk.Image:
    lung_paths = sorted(
        (course_root / "Segmentation_TotalSegmentator").glob("CT_*/total--lung*.nii.gz")
    )
    if not lung_paths:
        raise FileNotFoundError(
            f"No TotalSegmentator lung lobe masks found under {course_root / 'Segmentation_TotalSegmentator'}"
        )

    lung = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=bool)
    for path in lung_paths:
        img = _as_reference_geometry(sitk.ReadImage(str(path)), reference)
        lung |= sitk.GetArrayFromImage(img) > 0

    out = sitk.GetImageFromArray(lung.astype(np.uint8))
    out.CopyInformation(reference)
    return out


def _empty_like(mask_img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask_img)
    empty = sitk.GetImageFromArray(np.zeros_like(arr, dtype=np.uint8))
    empty.CopyInformation(mask_img)
    return empty


def _zyx_affine_from_sitk(img: sitk.Image) -> np.ndarray:
    """Return an affine that maps SimpleITK array coordinates (z, y, x) to mm."""
    spacing_xyz = np.asarray(img.GetSpacing(), dtype=float)
    origin_xyz = np.asarray(img.GetOrigin(), dtype=float)
    direction = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
    zyx_to_xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction @ np.diag(spacing_xyz) @ zyx_to_xyz
    affine[:3, 3] = origin_xyz
    return affine


def centroid_distance_cm(
    centroid_a_voxel: tuple[float, float, float] | list[float] | np.ndarray,
    centroid_b_voxel: tuple[float, float, float] | list[float] | np.ndarray,
    affine: np.ndarray,
) -> float:
    """Compute physical centroid distance in cm for z, y, x voxel coordinates."""
    a = np.asarray(centroid_a_voxel, dtype=float)
    b = np.asarray(centroid_b_voxel, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Centroids must be length-3 z, y, x coordinates")
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError("Affine must be a 4x4 matrix")
    a_mm = affine @ np.array([a[0], a[1], a[2], 1.0], dtype=float)
    b_mm = affine @ np.array([b[0], b[1], b[2], 1.0], dtype=float)
    return float(np.linalg.norm(a_mm[:3] - b_mm[:3]) / 10.0)


def _resolve_reference_paths(reference_mask_path: str | Path) -> list[Path]:
    pattern = str(reference_mask_path)
    if any(ch in pattern for ch in "*?[]"):
        return [Path(p) for p in sorted(glob(pattern))]
    path = Path(reference_mask_path)
    return [path] if path.exists() else []


def _jsonable_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    out = dict(anchor)
    affine = out.get("affine")
    if isinstance(affine, np.ndarray):
        out["affine"] = affine.tolist()
    for key in ("centroid_zyx",):
        if key in out and out[key] is not None:
            out[key] = [float(v) for v in out[key]]
    if "bbox_zyx" in out and out["bbox_zyx"] is not None:
        mins, maxs = out["bbox_zyx"]
        out["bbox_zyx"] = [[int(v) for v in mins], [int(v) for v in maxs]]
    return out


def derive_anchor_from_reference(reference_mask_path: str | Path) -> dict[str, Any] | None:
    """Derive a lesion anchor from one reference mask or a glob of masks.

    If ``reference_mask_path`` contains glob wildcards, all matching masks are
    resampled to the first mask geometry and unioned before the centroid and
    bounding box are computed.  This is intentional for interobserver GTVs:
    the union makes the anchor robust to individual specialist contour style.
    Coordinates are returned in SimpleITK array order: z, y, x.
    """
    paths = _resolve_reference_paths(reference_mask_path)
    if not paths:
        return None

    reference = sitk.ReadImage(str(paths[0]))
    union = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=bool)
    used_paths: list[str] = []
    for path in paths:
        img = _as_reference_geometry(sitk.ReadImage(str(path)), reference)
        arr = sitk.GetArrayFromImage(img) > 0
        if np.any(arr):
            union |= arr
            used_paths.append(str(path))

    coords = np.argwhere(union)
    if coords.size == 0:
        return None

    mins = tuple(int(v) for v in coords.min(axis=0))
    maxs = tuple(int(v) for v in coords.max(axis=0))
    centroid = tuple(float(v) for v in coords.mean(axis=0))
    voxels = int(coords.shape[0])
    volume_cm3 = float(voxels * np.prod(reference.GetSpacing()) / 1000.0)
    return {
        "centroid_zyx": centroid,
        "bbox_zyx": (mins, maxs),
        "voxels": voxels,
        "volume_cm3": volume_cm3,
        "affine": _zyx_affine_from_sitk(reference),
        "reference_paths": used_paths,
    }


def select_component_nearest_anchor(
    mask_img: sitk.Image,
    anchor_centroid_zyx: tuple[float, float, float] | list[float] | np.ndarray,
    max_distance_cm: float = 5.0,
    min_component_volume_cm3: float = 0.5,
) -> tuple[sitk.Image, dict[str, Any]]:
    arr = sitk.GetArrayFromImage(mask_img) > 0
    empty = _empty_like(mask_img)
    affine = _zyx_affine_from_sitk(mask_img)
    voxel_volume_cm3 = float(np.prod(mask_img.GetSpacing()) / 1000.0)
    metadata: dict[str, Any] = {
        "status": "no_anchor_match",
        "anchor_centroid_zyx": [float(v) for v in anchor_centroid_zyx],
        "max_distance_cm": float(max_distance_cm),
        "min_component_volume_cm3": float(min_component_volume_cm3),
        "n_components_evaluated": 0,
        "n_components_passing": 0,
    }
    if not np.any(arr):
        metadata["reason"] = "empty_mask"
        return empty, metadata

    structure = ndimage.generate_binary_structure(arr.ndim, 1)
    labels, n_components = ndimage.label(arr, structure=structure)
    metadata["n_components_evaluated"] = int(n_components)
    components: list[dict[str, Any]] = []
    for component_id in range(1, int(n_components) + 1):
        coords = np.argwhere(labels == component_id)
        if coords.size == 0:
            continue
        voxels = int(coords.shape[0])
        volume_cm3 = float(voxels * voxel_volume_cm3)
        centroid = tuple(float(v) for v in coords.mean(axis=0))
        distance_cm = centroid_distance_cm(centroid, anchor_centroid_zyx, affine)
        components.append(
            {
                "component_id": int(component_id),
                "voxels": voxels,
                "volume_cm3": volume_cm3,
                "centroid_zyx": centroid,
                "distance_cm": distance_cm,
                "passes_volume": volume_cm3 >= float(min_component_volume_cm3),
                "passes_distance": distance_cm <= float(max_distance_cm),
            }
        )

    passing = [
        comp
        for comp in components
        if comp["passes_volume"] and comp["passes_distance"]
    ]
    metadata["n_components_passing"] = len(passing)
    if components:
        closest = min(components, key=lambda comp: comp["distance_cm"])
        metadata["closest_component_distance_cm"] = float(closest["distance_cm"])
        metadata["closest_component_volume_cm3"] = float(closest["volume_cm3"])
        metadata["closest_component_voxels"] = int(closest["voxels"])
    if not passing:
        metadata["reason"] = "no_component_within_anchor_gate"
        return empty, metadata

    selected = min(passing, key=lambda comp: comp["distance_cm"])
    kept = (labels == int(selected["component_id"])).astype(np.uint8)
    out = sitk.GetImageFromArray(kept)
    out.CopyInformation(mask_img)
    metadata.update(
        {
            "status": "selected",
            "selected_component_id": int(selected["component_id"]),
            "selected_component_distance_cm": float(selected["distance_cm"]),
            "selected_component_volume_cm3": float(selected["volume_cm3"]),
            "selected_component_voxels": int(selected["voxels"]),
            "selected_component_centroid_zyx": [
                float(v) for v in selected["centroid_zyx"]
            ],
        }
    )
    return out, metadata


def largest_component(mask_img: sitk.Image, min_component_volume_cm3: float = 0.5) -> sitk.Image:
    arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    empty = sitk.GetImageFromArray(np.zeros_like(arr, dtype=np.uint8))
    empty.CopyInformation(mask_img)
    if not np.any(arr):
        return empty

    voxel_volume_cm3 = float(np.prod(mask_img.GetSpacing()) / 1000.0)
    min_voxels = max(1, int(ceil(float(min_component_volume_cm3) / voxel_volume_cm3)))

    binary = sitk.GetImageFromArray(arr)
    binary.CopyInformation(mask_img)
    components = sitk.ConnectedComponent(binary)
    relabeled = sitk.RelabelComponent(
        components,
        sortByObjectSize=True,
        minimumObjectSize=min_voxels,
    )
    kept = (sitk.GetArrayFromImage(relabeled) == 1).astype(np.uint8)
    out = sitk.GetImageFromArray(kept)
    out.CopyInformation(mask_img)
    return out


def confine_to_total_lung_largest_component(
    mask_img: sitk.Image,
    course_root: Path,
    min_component_volume_cm3: float = 0.5,
) -> sitk.Image:
    lung_img = total_lung_mask(course_root, mask_img)
    confined = (
        (sitk.GetArrayFromImage(mask_img) > 0)
        & (sitk.GetArrayFromImage(lung_img) > 0)
    ).astype(np.uint8)
    confined_img = sitk.GetImageFromArray(confined)
    confined_img.CopyInformation(mask_img)
    return largest_component(confined_img, min_component_volume_cm3=min_component_volume_cm3)


def confine_to_total_lung(mask_img: sitk.Image, course_root: Path) -> sitk.Image:
    lung_img = total_lung_mask(course_root, mask_img)
    confined = (
        (sitk.GetArrayFromImage(mask_img) > 0)
        & (sitk.GetArrayFromImage(lung_img) > 0)
    ).astype(np.uint8)
    confined_img = sitk.GetImageFromArray(confined)
    confined_img.CopyInformation(mask_img)
    return confined_img


def anchor_to_jsonable(anchor: dict[str, Any] | None) -> dict[str, Any] | None:
    if anchor is None:
        return None
    return _jsonable_anchor(anchor)
