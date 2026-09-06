"""Resolve complete ROI geometry without clipping or conflating CT series."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

NONVOLUMETRIC_CODES = frozenset({'ROI_NONVOLUMETRIC_POINT', 'ROI_NONVOLUMETRIC_OPEN_NONPLANAR', 'ROI_NONVOLUMETRIC_OPEN_PLANAR', 'ROI_NONVOLUMETRIC_MIXED'})
PLANE_TOLERANCE_MM = 1e-3  # Numerical coordinate precision, not slice snapping.


def contour_geometry(contour) -> tuple[str, bool]:
    kind = str(getattr(contour, 'ContourGeometricType', '') or '')
    try:
        points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        count = len(points)
        if not np.isfinite(points).all():
            return kind, False
        declared = getattr(contour, 'NumberOfContourPoints', count)
        if int(declared) != count:
            return kind, False
        if kind == 'POINT':
            return kind, count == 1
        if kind in {'OPEN_NONPLANAR', 'OPEN_PLANAR'}:
            valid = count >= 2 and len(np.unique(points, axis=0)) >= 2
        elif kind in {'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'}:
            valid = count >= 3 and len(np.unique(points, axis=0)) >= 3
            if valid and np.linalg.matrix_rank(points - points[0], tol=PLANE_TOLERANCE_MM) < 2:
                valid = False
        elif not kind:
            # Legacy inventory compatibility only. Geometry resolution below
            # never admits untyped contours to a volumetric builder.
            return kind, count >= 2
        else:
            return kind, False
        if valid and count >= 3 and kind in {'OPEN_PLANAR', 'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'}:
            _, _, vt = np.linalg.svd(points - points[0], full_matrices=False)
            valid = bool(np.max(np.abs((points - points[0]) @ vt[-1])) <= PLANE_TOLERANCE_MM)
        return kind, bool(valid)
    except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
        return kind, False


def roi_geometry_code(contours) -> str | None:
    items = [contour_geometry(c) for c in contours]
    if not items:
        return 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'
    if not all(valid for _, valid in items):
        return 'ROI_CONTOUR_PARTIALLY_UNPARSEABLE' if any(v for _, v in items) else 'ROI_CONTOUR_UNPARSEABLE'
    kinds = {kind for kind, _ in items}
    nonvolume = {'POINT', 'OPEN_NONPLANAR', 'OPEN_PLANAR'}
    if kinds <= nonvolume:
        return 'ROI_NONVOLUMETRIC_' + (next(iter(kinds)) if len(kinds) == 1 else 'MIXED')
    if kinds & nonvolume:
        return 'ROI_CONTOUR_MIXED_GEOMETRY'
    return None


@dataclass
class ScopeResult:
    roi_number: int
    roi_name: str
    code: str | None
    source_series_uids: tuple[str, ...]
    contours: list
    detail: str = ''


def _on_image(points, image, frame_uid):
    if frame_uid and str(getattr(image, 'FrameOfReferenceUID', '')) != frame_uid:
        return False
    try:
        orientation = np.asarray(image.ImageOrientationPatient, dtype=float)
        row, column = orientation[:3], orientation[3:]
        normal = np.cross(row, column)
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
            return False
        delta = points - np.asarray(image.ImagePositionPatient, dtype=float)
        if np.max(np.abs(delta @ normal)) > PLANE_TOLERANCE_MM:
            return False
        spacing = np.asarray(image.PixelSpacing, dtype=float)
        x, y = delta @ row / spacing[1], delta @ column / spacing[0]
        return bool(np.all(x >= -0.5) and np.all(x <= int(image.Columns)-0.5) and np.all(y >= -0.5) and np.all(y <= int(image.Rows)-0.5))
    except (ValueError, TypeError, AttributeError):
        return False


def resolve_roi_scopes(dataset, ct_images) -> dict[int, ScopeResult]:
    """Bind every volumetric contour. One unresolved contour holds the entire ROI.

    No reference is pruned here, no contour is removed, and coordinates are never
    changed. CT instances from different series remain separate candidates.
    """
    images = list(ct_images)
    by_uid = {}
    for image in images:
        uid = str(image.SOPInstanceUID)
        if uid in by_uid:
            raise ValueError('CT_DUPLICATE_SOP_INSTANCE_UID')
        by_uid[uid] = image
    # Use the first vertex only to shortlist planes, then validate ALL vertices
    # and in-plane bounds. This is exact filtering, never a geometric fallback.
    plane_candidates = []
    for image in images:
        try:
            orientation = np.asarray(image.ImageOrientationPatient, dtype=float)
            normal = np.cross(orientation[:3], orientation[3:])
            plane_candidates.append((image, np.asarray(image.ImagePositionPatient, dtype=float), normal))
        except (AttributeError, ValueError, TypeError):
            continue
    by_number = {}
    for item in getattr(dataset, 'ROIContourSequence', []) or []:
        by_number.setdefault(int(item.ReferencedROINumber), []).extend(list(getattr(item, 'ContourSequence', []) or []))
    results = {}
    for roi in getattr(dataset, 'StructureSetROISequence', []) or []:
        number = int(roi.ROINumber)
        contours = by_number.get(number, [])
        result = ScopeResult(number, str(roi.ROIName), roi_geometry_code(contours), (), copy.deepcopy(contours))
        results[number] = result
        if result.code:
            continue
        scopes = set()
        failures = []
        frame_uid = str(getattr(roi, 'ReferencedFrameOfReferenceUID', '') or '')
        for index, contour in enumerate(result.contours):
            kind, valid = contour_geometry(contour)
            if not kind or not valid:
                failures.append(f'contour {index} has invalid or unspecified geometric type')
                continue
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            refs = list(getattr(contour, 'ContourImageSequence', []) or [])
            if refs:
                candidates = [by_uid.get(str(getattr(ref, 'ReferencedSOPInstanceUID', ''))) for ref in refs]
                if any(image is None or not _on_image(points, image, frame_uid) for image in candidates):
                    failures.append(f'contour {index} image reference is unavailable or geometry disagrees')
                    continue
                # Multiple distinct image references cannot silently choose a series.
                candidates = list({str(image.SOPInstanceUID): image for image in candidates}.values())
            else:
                candidates = [image for image, origin, normal in plane_candidates
                              if abs(float((points[0] - origin) @ normal)) <= PLANE_TOLERANCE_MM
                              and _on_image(points, image, frame_uid)]
            if len(candidates) != 1:
                failures.append(f'contour {index} has {len(candidates)} authoritative geometric matches')
                continue
            image = candidates[0]
            scopes.add(str(image.SeriesInstanceUID))
            if not refs:
                ref = Dataset()
                ref.ReferencedSOPClassUID = image.SOPClassUID
                ref.ReferencedSOPInstanceUID = image.SOPInstanceUID
                contour.ContourImageSequence = Sequence([ref])
        result.source_series_uids = tuple(sorted(scopes))
        if failures:
            result.code = 'ROI_UNRESOLVED_SOURCE_SCOPE'
            result.detail = '; '.join(failures)
        elif len(scopes) != 1:
            result.code = 'ROI_MULTISERIES_SOURCE_SCOPE'
            result.detail = 'Complete contours span distinct source series; no combined mask is defined'
    return results


class ROIContourDisposition(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f'{result.code}: {result.detail}')


class ScopedRTStruct:
    def __init__(self, dataset, series_data):
        from rt_utils import RTStruct
        self.ds = dataset  # Preserve the entire source, including unresolved ROIs.
        self.series_data = series_data
        series_uids = {str(image.SeriesInstanceUID) for image in series_data}
        if len(series_uids) != 1:
            raise ValueError('RTSTRUCT_SCOPE_REQUIRES_ONE_CT_SERIES')
        self.scopes = resolve_roi_scopes(dataset, series_data)
        self.by_name = {}
        for result in self.scopes.values():
            if result.roi_name in self.by_name:
                raise ValueError('RTSTRUCT_DUPLICATE_ROI_NAME')
            self.by_name[result.roi_name] = result
        # Only fully bound volumetric ROIs enter the rasterizer. The full source
        # and every rejected identity remain in ds/scopes, never in a clipped mask.
        prepared = copy.deepcopy(dataset)
        accepted = {n for n, r in self.scopes.items() if r.code is None}
        prepared.StructureSetROISequence = Sequence([r for r in prepared.StructureSetROISequence if int(r.ROINumber) in accepted])
        prepared.ROIContourSequence = Sequence([])
        for number in sorted(accepted):
            item = Dataset()
            item.ReferencedROINumber = number
            item.ContourSequence = Sequence(self.scopes[number].contours)
            prepared.ROIContourSequence.append(item)
        prepared.RTROIObservationsSequence = Sequence([r for r in getattr(prepared, 'RTROIObservationsSequence', []) if int(r.ReferencedROINumber) in accepted])
        # Rebuild global references only after all retained contours are bound.
        series = Dataset()
        series.SeriesInstanceUID = next(iter(series_uids))
        series.ContourImageSequence = Sequence([])
        for image in series_data:
            ref = Dataset()
            ref.ReferencedSOPClassUID = image.SOPClassUID
            ref.ReferencedSOPInstanceUID = image.SOPInstanceUID
            series.ContourImageSequence.append(ref)
        frame = copy.deepcopy(dataset.ReferencedFrameOfReferenceSequence[0])
        study = copy.deepcopy(frame.RTReferencedStudySequence[0])
        study.RTReferencedSeriesSequence = Sequence([series])
        frame.RTReferencedStudySequence = Sequence([study])
        prepared.ReferencedFrameOfReferenceSequence = Sequence([frame])
        self.builder = RTStruct(series_data, prepared)

    def add_roi(self, **kwargs):
        from rt_utils import RTStruct
        return RTStruct(self.series_data, self.ds).add_roi(**kwargs)

    def save(self, file_path):
        self.ds.save_as(file_path)

    def get_roi_names(self):
        return list(self.by_name)

    def get_roi_mask_by_name(self, name):
        result = self.by_name[name]
        if result.code:
            raise ROIContourDisposition(result)
        return self.builder.get_roi_mask_by_name(name)


def create_scoped_rtstruct(ct_dir: Path, rs_path: Path):
    from rt_utils import image_helper
    from .rtstruct_identity import validate_rtstruct_identity
    dataset = pydicom.dcmread(rs_path)
    validate_rtstruct_identity(dataset)
    series_data = image_helper.load_sorted_image_series(str(ct_dir))
    return ScopedRTStruct(dataset, series_data)
