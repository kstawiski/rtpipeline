from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import yaml


DEFAULT_MAX_RESAMPLED_BBOX_VOXELS = 15_000_000
DEFAULT_PAD_DISTANCE = 5
RESAMPLED_BBOX_LIMIT_CODE = "ROI_RESAMPLED_BBOX_EXCEEDS_LIMIT"


def _positive_triplet(values: Iterable[Any], *, label: str) -> tuple[float, float, float]:
    converted = tuple(float(value) for value in values)
    if len(converted) != 3 or any(not math.isfinite(value) or value <= 0 for value in converted):
        raise ValueError(f"{label} must contain three finite positive values")
    return converted


def resolve_max_resampled_bbox_voxels(config: Any) -> int:
    value = getattr(config, "radiomics_max_resampled_bbox_voxels", None)
    if value in (None, ""):
        return DEFAULT_MAX_RESAMPLED_BBOX_VOXELS
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESAMPLED_BBOX_VOXELS
    return converted if converted > 0 else DEFAULT_MAX_RESAMPLED_BBOX_VOXELS


def configured_grid_settings(
    params_file: Optional[Path],
    *,
    native_spacing_xyz: Sequence[float],
) -> tuple[tuple[float, float, float], int]:
    """Read the resampled grid and padding without importing PyRadiomics."""

    native = _positive_triplet(native_spacing_xyz, label="native spacing")
    settings: dict[str, Any] = {}
    if params_file is not None:
        data = yaml.safe_load(Path(params_file).read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and isinstance(data.get("setting"), dict):
            settings = data["setting"]
    raw_resampled = settings.get("resampledPixelSpacing")
    resampled = (
        native
        if raw_resampled in (None, "", [])
        else _positive_triplet(raw_resampled, label="resampledPixelSpacing")
    )
    raw_padding = settings.get("padDistance", DEFAULT_PAD_DISTANCE)
    try:
        padding = max(0, int(math.ceil(float(raw_padding))))
    except (TypeError, ValueError, OverflowError):
        padding = DEFAULT_PAD_DISTANCE
    return resampled, padding


@dataclass(frozen=True, slots=True)
class ResampledBoundingBoxEstimate:
    native_foreground_voxels: int
    estimated_resampled_foreground_voxels: int
    native_bbox_shape: tuple[int, int, int]
    estimated_resampled_bbox_shape: tuple[int, int, int]
    estimated_resampled_bbox_voxels: int
    pad_distance_voxels: int

    def metadata(self, *, limit: int) -> dict[str, Any]:
        return {
            "native_mask_voxel_count": self.native_foreground_voxels,
            "estimated_resampled_foreground_voxel_count": (
                self.estimated_resampled_foreground_voxels
            ),
            "native_mask_bbox_shape": list(self.native_bbox_shape),
            "estimated_resampled_bbox_shape": list(
                self.estimated_resampled_bbox_shape
            ),
            "estimated_resampled_bbox_voxel_count": (
                self.estimated_resampled_bbox_voxels
            ),
            "max_resampled_bbox_voxel_count": int(limit),
            "radiomics_pad_distance_voxels": self.pad_distance_voxels,
        }


def estimate_resampled_bounding_box(
    mask: Any,
    *,
    native_spacing_xyz: Sequence[float],
    resampled_spacing_xyz: Sequence[float],
    array_axis_to_xyz: Sequence[int],
    pad_distance: int,
) -> ResampledBoundingBoxEstimate:
    """Conservatively estimate the padded PyRadiomics resampling work grid.

    PyRadiomics resamples a crop around the mask, not just its foreground voxels.
    Sparse or disjoint masks can therefore require a large dense image. The estimate
    uses the occupied native bounding box, maps each array axis to physical spacing,
    rounds each target dimension upward, and adds configured padding on both sides.
    """

    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 3:
        raise ValueError(f"radiomics mask must be three-dimensional, got {foreground.ndim}D")
    axis_map = tuple(int(axis) for axis in array_axis_to_xyz)
    if sorted(axis_map) != [0, 1, 2]:
        raise ValueError("array_axis_to_xyz must be a permutation of (0, 1, 2)")
    native_xyz = _positive_triplet(native_spacing_xyz, label="native spacing")
    resampled_xyz = _positive_triplet(
        resampled_spacing_xyz, label="resampled spacing"
    )
    padding = max(0, int(pad_distance))
    foreground_count = int(foreground.sum())
    if foreground_count == 0:
        return ResampledBoundingBoxEstimate(
            native_foreground_voxels=0,
            estimated_resampled_foreground_voxels=0,
            native_bbox_shape=(0, 0, 0),
            estimated_resampled_bbox_shape=(0, 0, 0),
            estimated_resampled_bbox_voxels=0,
            pad_distance_voxels=padding,
        )

    native_shape_parts: list[int] = []
    for array_axis in range(3):
        reduce_axes = tuple(axis for axis in range(3) if axis != array_axis)
        occupied = np.flatnonzero(np.any(foreground, axis=reduce_axes))
        native_shape_parts.append(int(occupied[-1] - occupied[0] + 1))
    native_shape = tuple(native_shape_parts)
    resampled_shape = tuple(
        int(
            math.ceil(
                native_shape[array_axis]
                * native_xyz[physical_axis]
                / resampled_xyz[physical_axis]
            )
        )
        + 2 * padding
        for array_axis, physical_axis in enumerate(axis_map)
    )
    physical_foreground_volume = foreground_count * math.prod(native_xyz)
    resampled_foreground = int(
        math.ceil(physical_foreground_volume / math.prod(resampled_xyz))
    )
    return ResampledBoundingBoxEstimate(
        native_foreground_voxels=foreground_count,
        estimated_resampled_foreground_voxels=resampled_foreground,
        native_bbox_shape=native_shape,
        estimated_resampled_bbox_shape=resampled_shape,
        estimated_resampled_bbox_voxels=int(math.prod(resampled_shape)),
        pad_distance_voxels=padding,
    )
