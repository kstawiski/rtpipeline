from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import sanitize_rtstruct

try:
    from rt_utils import RTStructBuilder
except ImportError:  # pragma: no cover - optional dependency
    RTStructBuilder = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from skimage.draw import polygon as _polygon
except Exception:  # pragma: no cover - scikit-image not available
    _polygon = None

try:  # pragma: no cover - matplotlib always available in project deps
    from matplotlib.path import Path as _MplPath
except Exception:  # pragma: no cover - extremely unlikely
    _MplPath = None

logger = logging.getLogger(__name__)


@dataclass
class FixSummary:
    changed: bool
    fixed: List[str]
    failed: List[str]
    output_path: Path


class _ROIRebuilder:
    """Rebuild ROI voxel masks directly from contour data when rt-utils fails."""

    def __init__(self, rtstruct, *, minimum_points: int = 3) -> None:
        self.rtstruct = rtstruct
        self.ds = rtstruct.ds
        self.series_data = rtstruct.series_data
        self.minimum_points = minimum_points
        self._geometry_cache: Optional[dict] = None
        self._rebuilt: Dict[str, np.ndarray] = {}
        self._roi_number_cache: Dict[str, int] = {}
        self._roi_colors: Dict[str, Sequence[int]] = {}
        self._init_roi_maps()

    def _init_roi_maps(self) -> None:
        for roi in getattr(self.ds, "StructureSetROISequence", []) or []:
            name = str(getattr(roi, "ROIName", "") or "").strip()
            number = int(getattr(roi, "ROINumber", 0) or 0)
            if name:
                self._roi_number_cache[name] = number
        name_map = {v: k for k, v in self._roi_number_cache.items()}
        for roi_contour in getattr(self.ds, "ROIContourSequence", []) or []:
            number = int(getattr(roi_contour, "ReferencedROINumber", 0) or 0)
            name = name_map.get(number)
            if not name:
                continue
            color = getattr(roi_contour, "ROIDisplayColor", None)
            if color and len(color) == 3:
                try:
                    self._roi_colors[name] = [int(c) for c in color]
                except Exception:
                    continue

    def was_rebuilt(self, roi_name: str) -> bool:
        return roi_name in self._rebuilt

    def get_roi_color(self, roi_name: str) -> Optional[Sequence[int]]:
        return self._roi_colors.get(roi_name)

    def get_mask(self, roi_name: str) -> Optional[np.ndarray]:
        try:
            mask = self.rtstruct.get_roi_mask_by_name(roi_name)
        except Exception:
            mask = None
        if mask is not None and np.any(mask):
            return mask.astype(bool)
        rebuilt = self._rebuild_mask(roi_name)
        if rebuilt is not None and np.any(rebuilt):
            self._rebuilt[roi_name] = rebuilt.astype(bool)
            return self._rebuilt[roi_name]
        return None

    # --- Internal helpers -------------------------------------------------

    def _get_geometry(self) -> dict:
        if self._geometry_cache is not None:
            return self._geometry_cache
        if not self.series_data:
            raise RuntimeError("No CT series data available for ROI rebuild")
        first = self.series_data[0]
        pixel_spacing = [float(x) for x in getattr(first, "PixelSpacing", [1.0, 1.0])]
        positions = np.array(
            [[float(c) for c in getattr(slice_ds, "ImagePositionPatient", (0.0, 0.0, 0.0))]
             for slice_ds in self.series_data],
            dtype=float,
        )
        orientation_vals = getattr(first, "ImageOrientationPatient", None)
        if orientation_vals is None:
            orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        else:
            orientation = np.array([float(x) for x in orientation_vals], dtype=float).reshape(2, 3)
        geom = {
            "pixel_spacing": np.asarray(pixel_spacing, dtype=float),
            "positions": positions,
            "orientation": orientation,
        }
        self._geometry_cache = geom
        return geom

    def _roi_number(self, roi_name: str) -> Optional[int]:
        return self._roi_number_cache.get(roi_name)

    def _contour_sequence(self, roi_number: int):
        for roi_contour in getattr(self.ds, "ROIContourSequence", []) or []:
            if int(getattr(roi_contour, "ReferencedROINumber", 0) or 0) == roi_number:
                return getattr(roi_contour, "ContourSequence", None)
        return None

    def _fix_contour_points(self, contour_data: Sequence[float]) -> Optional[np.ndarray]:
        if contour_data is None:
            return None
        try:
            pts = np.asarray(contour_data, dtype=float).reshape(-1, 3)
        except Exception:
            return None
        if pts.shape[0] < self.minimum_points:
            return None
        if np.isnan(pts).any() or np.isinf(pts).any():
            return None
        if not np.allclose(pts[0], pts[-1], atol=1e-5):
            pts = np.vstack([pts, pts[0]])
        return pts

    def _slice_contours(self, roi_name: str, slice_uid: str) -> List[np.ndarray]:
        roi_number = self._roi_number(roi_name)
        if roi_number is None:
            return []
        contour_seq = self._contour_sequence(roi_number)
        if not contour_seq:
            return []
        slice_contours: List[np.ndarray] = []
        for contour in contour_seq:
            try:
                contour_images = getattr(contour, "ContourImageSequence", None)
                if not contour_images:
                    continue
                ref_uid = getattr(contour_images[0], "ReferencedSOPInstanceUID", None)
                if ref_uid != slice_uid:
                    continue
                fixed = self._fix_contour_points(getattr(contour, "ContourData", None))
                if fixed is not None:
                    slice_contours.append(fixed)
            except Exception:
                continue
        return slice_contours

    def _world_to_pixel(self, world_coords: np.ndarray, slice_index: int) -> np.ndarray:
        geom = self._get_geometry()
        origin = geom["positions"][slice_index]
        translated = world_coords - origin
        row_cos = geom["orientation"][0]
        col_cos = geom["orientation"][1]
        pixel_spacing = geom["pixel_spacing"]
        x = np.dot(translated, col_cos) / (pixel_spacing[1] if pixel_spacing[1] else 1.0)
        y = np.dot(translated, row_cos) / (pixel_spacing[0] if pixel_spacing[0] else 1.0)
        coords = np.column_stack([x, y])
        return np.rint(coords).astype(np.int32)

    def _fill_polygon(self, slice_mask: np.ndarray, points: np.ndarray) -> None:
        if points.shape[0] < 3:
            return
        h, w = slice_mask.shape
        pts = points.copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        if _polygon is not None:
            rr, cc = _polygon(pts[:, 1], pts[:, 0], shape=slice_mask.shape)
            slice_mask[rr, cc] = True
            return
        if _MplPath is None:
            logger.debug("matplotlib.path unavailable; polygon filling skipped")
            return
        path = _MplPath(pts[:, :2])
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        mask = path.contains_points(grid)
        slice_mask |= mask.reshape(slice_mask.shape)

    def _rebuild_mask(self, roi_name: str) -> Optional[np.ndarray]:
        rows = getattr(self.series_data[0], "Rows", None)
        cols = getattr(self.series_data[0], "Columns", None)
        if rows is None or cols is None:
            return None
        num_slices = len(self.series_data)
        mask = np.zeros((rows, cols, num_slices), dtype=bool)
        try:
            for slice_index, slice_ds in enumerate(self.series_data):
                slice_uid = getattr(slice_ds, "SOPInstanceUID", None)
                if not slice_uid:
                    continue
                contours = self._slice_contours(roi_name, slice_uid)
                if not contours:
                    continue
                slice_mask = np.zeros((rows, cols), dtype=bool)
                for contour in contours:
                    pixels = self._world_to_pixel(contour, slice_index)
                    self._fill_polygon(slice_mask, pixels)
                mask[:, :, slice_index] = slice_mask
        except Exception as exc:
            logger.debug("Rebuild mask failed for ROI %s: %s", roi_name, exc)
            return None
        return mask if np.any(mask) else None


def fix_rtstruct_rois(
    dicom_series_path: Path,
    rtstruct_path: Path,
    *,
    output_path: Optional[Path] = None,
    minimum_points: int = 3,
) -> Optional[FixSummary]:
    """Attempt to rebuild failed ROIs and save a cleaned RTSTRUCT.

    Parameters
    ----------
    dicom_series_path: Path
        Directory containing CT DICOM slices for the RTSTRUCT.
    rtstruct_path: Path
        Path to the RTSTRUCT file to fix.
    output_path: Optional[Path]
        Optional destination path; defaults to overwriting ``rtstruct_path``.
    minimum_points: int
        Minimum contour vertices required to keep a contour.
    """

    if RTStructBuilder is None:
        logger.debug("rt-utils not available; skipping ROI fix for %s", rtstruct_path)
        return None

    dicom_series_path = Path(dicom_series_path)
    rtstruct_path = Path(rtstruct_path)
    if not rtstruct_path.exists() or not dicom_series_path.exists():
        logger.debug("ROI fix skipped: missing paths (%s, %s)", rtstruct_path, dicom_series_path)
        return None

    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(dicom_series_path),
            rt_struct_path=str(rtstruct_path),
        )
    except Exception as exc:
        logger.warning("Failed to load RTSTRUCT %s for ROI fix: %s", rtstruct_path, exc)
        return None

    rebuilder = _ROIRebuilder(rtstruct, minimum_points=minimum_points)
    roi_names = rtstruct.get_roi_names()
    if not roi_names:
        logger.debug("No ROIs found in %s", rtstruct_path)
        return None

    mask_map: Dict[str, np.ndarray] = {}
    fixed: List[str] = []
    failed: List[str] = []

    for roi_name in roi_names:
        mask = rebuilder.get_mask(roi_name)
        if mask is None or not np.any(mask):
            failed.append(roi_name)
            continue
        mask_map[roi_name] = mask.astype(bool)
        if rebuilder.was_rebuilt(roi_name):
            fixed.append(roi_name)

    if not fixed:
        logger.debug("No ROI fixes required for %s", rtstruct_path)
        return FixSummary(changed=False, fixed=[], failed=failed, output_path=rtstruct_path)

    output_path = Path(output_path) if output_path else rtstruct_path

    try:
        new_rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dicom_series_path))
    except Exception as exc:
        logger.warning("Failed to create new RTSTRUCT for %s: %s", rtstruct_path, exc)
        return None

    # Preserve descriptive metadata when possible
    try:
        new_rtstruct.ds.StructureSetLabel = getattr(rtstruct.ds, "StructureSetLabel", "Fixed")
        new_rtstruct.ds.StructureSetName = getattr(rtstruct.ds, "StructureSetName", "")
        new_rtstruct.ds.StructureSetDescription = getattr(rtstruct.ds, "StructureSetDescription", "")
    except Exception:
        pass

    for roi_name in roi_names:
        mask = mask_map.get(roi_name)
        if mask is None:
            continue
        color = rebuilder.get_roi_color(roi_name)
        try:
            if color:
                new_rtstruct.add_roi(mask=mask, name=roi_name, color=list(color))
            else:
                new_rtstruct.add_roi(mask=mask, name=roi_name)
        except Exception as exc:
            logger.debug("Failed adding ROI %s during rebuild: %s", roi_name, exc)
            failed.append(roi_name)

    try:
        new_rtstruct.save(str(output_path))
    except Exception as exc:
        logger.error("Failed to save fixed RTSTRUCT to %s: %s", output_path, exc)
        return None

    try:
        sanitize_rtstruct(output_path, minimum_points=minimum_points)
    except Exception:
        pass

    logger.info(
        "Rebuilt %d ROI(s) in %s (output: %s)",
        len(fixed),
        rtstruct_path,
        output_path,
    )
    if failed:
        logger.debug("ROIs still problematic after rebuild in %s: %s", output_path, ", ".join(failed))

    return FixSummary(changed=True, fixed=fixed, failed=failed, output_path=output_path)
