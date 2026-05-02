from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


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
