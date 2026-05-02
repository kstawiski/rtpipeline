from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


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


def dice(mask_a: Path, mask_b: Path) -> float | None:
    img_a = sitk.ReadImage(str(mask_a))
    img_b = sitk.ReadImage(str(mask_b))
    arr_a = sitk.GetArrayFromImage(img_a) > 0
    arr_b = sitk.GetArrayFromImage(img_b) > 0
    if arr_a.shape != arr_b.shape:
        return None
    denom = int(arr_a.sum() + arr_b.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(arr_a, arr_b).sum() / denom)


def summarize_masks(paths: Iterable[Path]) -> list[dict]:
    return [mask_geometry(path).to_dict() for path in paths]
