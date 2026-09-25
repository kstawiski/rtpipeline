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


def contour_encloses_area(contour) -> bool:
    """Whether this item can bound any area at all.

    Fewer than three distinct points bounds nothing, whatever the declared
    geometric type. Converting a segmentation mask to contours emits such items
    for a stray boundary voxel, and a rasteriser draws nothing from them.
    """
    try:
        data = getattr(contour, 'ContourData', None)
        if data is None:
            return False
        points = np.asarray(data, dtype=float).reshape(-1, 3)
        if not np.isfinite(points).all():
            return False
        return len(np.unique(points, axis=0)) >= 3
    except (ValueError, TypeError, AttributeError):
        return False


def roi_geometry_code(contours) -> str | None:
    contours = list(contours)
    items = [contour_geometry(c) for c in contours]
    if not items:
        return 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'
    if not all(valid for _, valid in items):
        if not any(v for _, v in items):
            return 'ROI_CONTOUR_UNPARSEABLE'
        # An invalid item that bounds no area carries no geometry to lose: it is
        # a mask-to-contour conversion artifact, and the rasteriser ignores it.
        # Judging the ROI unparseable because of one would discard a structure
        # the rest of whose slices describe it completely. Only an invalid item
        # that *does* bound area means real geometry could not be read.
        if all(v or not contour_encloses_area(c) for (_, v), c in zip(items, contours)):
            items = [item for item in items if item[1]]
        else:
            return 'ROI_CONTOUR_PARTIALLY_UNPARSEABLE'
    kinds = {kind for kind, _ in items}
    nonvolume = {'POINT', 'OPEN_NONPLANAR', 'OPEN_PLANAR'}
    if kinds <= nonvolume:
        return 'ROI_NONVOLUMETRIC_' + (next(iter(kinds)) if len(kinds) == 1 else 'MIXED')
    if kinds & nonvolume:
        return 'ROI_CONTOUR_MIXED_GEOMETRY'
    return None


def _contour_rasterizable(contour) -> bool:
    """Whether a contour item may enter the rasterizer copy.

    Mirrors the scope validator's skip rule exactly: a valid item is kept,
    and an invalid item is kept only when it bounds area (in which case the
    containing ROI cannot be scope-accepted anyway, so keeping it can only
    fail loudly, never silently). An invalid area-less item is withheld: it
    contributes zero voxels in rasterizers that tolerate it and aborts
    rasterizers that do not.
    """
    kind, valid = contour_geometry(contour)
    if kind and valid:
        return True
    try:
        return bool(contour_encloses_area(contour))
    except Exception:
        return False


@dataclass
class ScopeResult:
    roi_number: int
    roi_name: str
    code: str | None
    source_series_uids: tuple[str, ...]
    contours: list
    detail: str = ''


def plane_offset_mm(points, image) -> float:
    """Largest distance (mm) of ``points`` from the plane of ``image``.

    Infinite when the image orientation cosines do not define a unit normal.
    """
    orientation = np.asarray(image.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
        return float('inf')
    delta = points - np.asarray(image.ImagePositionPatient, dtype=float)
    return float(np.max(np.abs(delta @ normal)))


def _on_image(points, image, frame_uid):
    if frame_uid and str(getattr(image, 'FrameOfReferenceUID', '')) != frame_uid:
        return False
    try:
        if plane_offset_mm(points, image) > PLANE_TOLERANCE_MM:
            return False
        orientation = np.asarray(image.ImageOrientationPatient, dtype=float)
        row, column = orientation[:3], orientation[3:]
        delta = points - np.asarray(image.ImagePositionPatient, dtype=float)
        spacing = np.asarray(image.PixelSpacing, dtype=float)
        x, y = delta @ row / spacing[1], delta @ column / spacing[0]
        return bool(np.all(x >= -0.5) and np.all(x <= int(image.Columns)-0.5) and np.all(y >= -0.5) and np.all(y <= int(image.Rows)-0.5))
    except (ValueError, TypeError, AttributeError):
        return False


def contours_on_referenced_planes(dataset, ct_images, roi_numbers=None) -> tuple[bool, str]:
    """Whether every contour lies on the plane of the CT image it references.

    Uses the scoped reader's plane arithmetic and ``PLANE_TOLERANCE_MM``. Items
    the reader skips (invalid and bounding no area) are skipped here too. A
    contour without an image reference must lie on some CT plane. When
    ``roi_numbers`` is given, only those ROIs are checked. Returns
    ``(ok, detail)``; ``detail`` is empty when ``ok``.
    """
    images = list(ct_images)
    by_uid = {str(image.SOPInstanceUID): image for image in images}
    names = {int(roi.ROINumber): str(getattr(roi, 'ROIName', ''))
             for roi in getattr(dataset, 'StructureSetROISequence', []) or []}
    checked = 0
    failed = 0
    worst = 0.0
    first = ''
    for item in getattr(dataset, 'ROIContourSequence', []) or []:
        number = int(item.ReferencedROINumber)
        if roi_numbers is not None and number not in roi_numbers:
            continue
        for index, contour in enumerate(getattr(item, 'ContourSequence', []) or []):
            kind, valid = contour_geometry(contour)
            if (not kind or not valid) and not contour_encloses_area(contour):
                continue
            checked += 1
            reason = ''
            try:
                points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
                refs = list(getattr(contour, 'ContourImageSequence', []) or [])
                if refs:
                    targets = [by_uid.get(str(getattr(ref, 'ReferencedSOPInstanceUID', ''))) for ref in refs]
                    if any(image is None for image in targets):
                        reason = 'references an image outside the CT series'
                    else:
                        offset = max(plane_offset_mm(points, image) for image in targets)
                else:
                    offset = min((plane_offset_mm(points, image) for image in images), default=float('inf'))
            except (ValueError, TypeError, AttributeError) as exc:
                reason = f'has unreadable geometry ({exc})'
            if not reason and offset > PLANE_TOLERANCE_MM:
                reason = f'lies {offset:.3f} mm off its CT plane'
                worst = max(worst, offset)
            if reason:
                failed += 1
                if not first:
                    first = f"ROI {names.get(number, number)!r} contour {index} {reason}"
    if not failed:
        return True, ''
    detail = f'{failed} of {checked} contours are not on the CT plane they reference'
    if worst:
        detail += f' (max offset {worst:.3f} mm, tolerance {PLANE_TOLERANCE_MM} mm)'
    return False, f'{detail}; first: {first}'


# A series is sampled with one 3-D resample only when that resample provably
# returns the bytes of the per-slice resample. Each slice position may differ
# from the regular grid by at most REGULAR_GRID_TOLERANCE_MM, and every sample
# must then lie further than that deviation plus _ROUNDING_MARGIN_VOXELS from
# the half-integer input index at which nearest-neighbour rounding or the
# buffer bound changes its result. Floating-point differences between the two
# evaluation orders are orders of magnitude below that margin.
REGULAR_GRID_TOLERANCE_MM = 1e-4
_ROUNDING_MARGIN_VOXELS = 1e-6
# A term of the input index that changes by less than this over the whole
# series enters the margin instead of the enumerated sample positions.
_MINOR_TERM_LIMIT_VOXELS = 1e-2


def _slice_geometries(slices) -> list[tuple]:
    """Per-slice reference geometry exactly as the per-slice sampler builds it."""
    geometries = []
    expected_shape: tuple[int, int] | None = None
    for dataset in slices:
        try:
            rows = int(dataset.Rows)
            columns = int(dataset.Columns)
            spacing = [float(value) for value in dataset.PixelSpacing]
            orientation = np.asarray(
                [float(value) for value in dataset.ImageOrientationPatient],
                dtype=float,
            ).reshape(2, 3)
            origin = tuple(float(value) for value in dataset.ImagePositionPatient)
        except Exception as exc:
            raise ValueError(
                "RTSTRUCT source slice lacks rows, columns, pixel spacing, "
                "orientation, or image position"
            ) from exc
        shape = (rows, columns)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("RTSTRUCT source series has inconsistent slice dimensions")

        normal = np.cross(orientation[0], orientation[1])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("RTSTRUCT source slice has invalid orientation cosines")
        normal /= norm
        direction = np.column_stack((orientation[0], orientation[1], normal))
        geometries.append((rows, columns, spacing, orientation, origin, normal, direction))
    return geometries


def _image_array_per_slice(image, geometries) -> np.ndarray:
    """One nearest-neighbour resample per DICOM slice (the reference path)."""
    import SimpleITK as sitk

    sampled: list[np.ndarray] = []
    identity = sitk.Transform(3, sitk.sitkIdentity)
    for rows, columns, spacing, _orientation, origin, _normal, direction in geometries:
        reference = sitk.Image(columns, rows, 1, image.GetPixelID())
        reference.SetOrigin(origin)
        reference.SetSpacing((spacing[1], spacing[0], 1.0))
        reference.SetDirection(tuple(float(value) for value in direction.ravel()))
        plane = sitk.Resample(
            image,
            reference,
            identity,
            sitk.sitkNearestNeighbor,
            0,
            image.GetPixelID(),
        )
        # SimpleITK returns (z, rows, columns).
        sampled.append(sitk.GetArrayFromImage(plane)[0])
    return np.stack(sampled, axis=2)


def _half_integer_distance(values: np.ndarray) -> float:
    """Smallest distance of ``values`` from a half-integer."""
    return float(np.min(np.abs(values - np.floor(values) - 0.5)))


def _uniform_reference(image, geometries):
    """A 3-D reference whose one resample equals the per-slice resamples, or None.

    Requires identical rows, columns, pixel spacing and orientation values on
    every slice, and each slice position within REGULAR_GRID_TOLERANCE_MM of
    ``first + k * step * normal``. Then the samples of both paths differ only
    by that deviation and floating-point rounding. Every sample of the image's
    continuous index must lie far enough from a half-integer (where
    nearest-neighbour rounding and the inside-buffer test change) that both
    paths select the same voxel or the same default value. Anything not
    proven returns None and the per-slice path is used.
    """
    import SimpleITK as sitk

    count = len(geometries)
    if count < 2:
        return None
    rows, columns, spacing, orientation, _origin, normal, _direction = geometries[0]
    for other in geometries[1:]:
        if other[2] != spacing or not np.array_equal(other[3], orientation):
            return None
    origins = np.asarray([geometry[4] for geometry in geometries], dtype=float)
    step = float((origins[-1] - origins[0]) @ normal) / (count - 1)
    if not np.isfinite(step) or abs(step) <= 100 * REGULAR_GRID_TOLERANCE_MM:
        return None
    regular = origins[0] + np.arange(count)[:, None] * (step * normal)[None, :]
    if float(np.max(np.abs(regular - origins))) > REGULAR_GRID_TOLERANCE_MM:
        return None

    # Continuous index of the image at the per-slice samples:
    # base[k] + along_row * column_index + along_column * row_index.
    try:
        image_matrix = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3) * np.asarray(
            image.GetSpacing(), dtype=float
        )[None, :]
        to_index = np.linalg.inv(image_matrix)
    except (ValueError, np.linalg.LinAlgError):
        return None
    image_origin = np.asarray(image.GetOrigin(), dtype=float)
    base = (origins - image_origin) @ to_index.T
    deviation = np.max(np.abs((regular - origins) @ to_index.T), axis=0)
    along_row = to_index @ (orientation[0] * spacing[1])
    along_column = to_index @ (orientation[1] * spacing[0])
    if not (np.isfinite(base).all() and np.isfinite(along_row).all() and np.isfinite(along_column).all()):
        return None
    for axis in range(3):
        terms = [(along_row[axis], columns), (along_column[axis], rows)]
        margin = deviation[axis] + _ROUNDING_MARGIN_VOXELS
        major = []
        for coefficient, length in terms:
            span = abs(coefficient) * (length - 1)
            if span <= _MINOR_TERM_LIMIT_VOXELS:
                margin += span
            else:
                major.append((coefficient, length))
        if len(major) > 1:
            # In-plane rotation between the image and the series; not enumerated.
            return None
        values = base[:, axis][:, None]
        if major:
            coefficient, length = major[0]
            values = values + coefficient * np.arange(length)[None, :]
        if _half_integer_distance(values) <= margin:
            return None

    direction = np.column_stack((orientation[0], orientation[1], normal * np.sign(step)))
    reference = sitk.Image(columns, rows, count, image.GetPixelID())
    reference.SetOrigin(geometries[0][4])
    reference.SetSpacing((spacing[1], spacing[0], abs(step)))
    reference.SetDirection(tuple(float(value) for value in direction.ravel()))
    return reference


def image_array_for_rtstruct(image, series_data) -> np.ndarray:
    """Sample an image on the exact, potentially non-uniform DICOM planes.

    Returns one plane per source DICOM object in the in-plane layout rt-utils
    ``add_roi`` contours: axis 0 is the row index (along the column direction)
    and axis 1 the column index (along the row direction), as
    ``np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)`` gives on a regular
    grid. SimpleITK regularizes mixed slice spacing onto a uniform z grid, so
    moving axes from that image puts slice ``i`` at the wrong position.

    A regular series is sampled with one 3-D resample when ``_uniform_reference``
    proves it returns the same bytes as one resample per slice; any other
    series is sampled slice by slice.
    """
    import SimpleITK as sitk

    slices = list(series_data)
    if not slices:
        raise ValueError("RTSTRUCT source series contains no DICOM slices")
    geometries = _slice_geometries(slices)
    reference = _uniform_reference(image, geometries)
    if reference is None:
        return _image_array_per_slice(image, geometries)
    volume = sitk.Resample(
        image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        image.GetPixelID(),
    )
    # SimpleITK returns (slices, rows, columns).
    return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(volume), 0, -1))


def anchor_contours_to_referenced_planes(ds, series_data, roi_numbers=None) -> int:
    """Move rt-utils contours from its uniform slice grid onto their referenced planes.

    rt-utils converts mask slice ``i`` with one affine built from the first
    slice and a uniform step ``(z_last - z_first) / (N - 1)``. On a series with
    mixed slice spacing, slice ``i`` then lands off the plane of the image it
    references. The mask was sampled on each slice's own pixel grid, so the
    correct position of a contour is that slice's origin plus the same in-plane
    offset. Only contours off their referenced plane are changed; on a uniform
    series nothing moves. When ``roi_numbers`` is given, only those ROIs are
    touched. Raises ``ValueError`` when orientation or pixel spacing changes
    between slices. Returns the number of contours moved.
    """
    from rt_utils import image_helper

    slices = list(series_data)
    if not slices:
        return 0
    index_by_uid = {str(s.SOPInstanceUID): i for i, s in enumerate(slices)}
    matrix = np.asarray(
        image_helper.get_pixel_to_patient_transformation_matrix(slices), dtype=float
    )
    first_orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
    first_spacing = np.asarray(slices[0].PixelSpacing, dtype=float)
    moved = 0
    for item in getattr(ds, "ROIContourSequence", []) or []:
        if roi_numbers is not None and int(item.ReferencedROINumber) not in roi_numbers:
            continue
        for contour in getattr(item, "ContourSequence", []) or []:
            refs = list(getattr(contour, "ContourImageSequence", []) or [])
            if len(refs) != 1:
                continue
            index = index_by_uid.get(str(getattr(refs[0], "ReferencedSOPInstanceUID", "")))
            if index is None:
                continue
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            image = slices[index]
            if plane_offset_mm(points, image) <= PLANE_TOLERANCE_MM:
                continue
            if not (
                np.allclose(np.asarray(image.ImageOrientationPatient, dtype=float), first_orientation, atol=1e-6)
                and np.allclose(np.asarray(image.PixelSpacing, dtype=float), first_spacing, atol=1e-6)
            ):
                raise ValueError(
                    "RTSTRUCT source series changes orientation or pixel spacing between slices"
                )
            placed_origin = matrix[:3, 3] + matrix[:3, 2] * index
            shift = np.asarray(image.ImagePositionPatient, dtype=float) - placed_origin
            contour.ContourData = (points + shift).ravel().tolist()
            moved += 1
    return moved


def place_added_rois_on_planes(ds, series_data, roi_numbers=None) -> int:
    """Anchor ROIs written by rt-utils ``add_roi`` and require them on-plane.

    ``roi_numbers`` names the ROIs this writer added; contours copied unchanged
    from a source RTSTRUCT are left alone. ``None`` means every ROI. Raises
    ``ValueError`` when anchoring is impossible or a contour stays off its
    referenced plane, so the caller never publishes an off-plane RTSTRUCT.
    Returns the number of contours moved.
    """
    numbers = None if roi_numbers is None else {int(number) for number in roi_numbers}
    slices = list(series_data)
    moved = anchor_contours_to_referenced_planes(ds, slices, numbers)
    on_planes, detail = contours_on_referenced_planes(ds, slices, numbers)
    if not on_planes:
        raise ValueError(f"contours are off the planning CT planes: {detail}")
    return moved


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
                if not contour_encloses_area(contour):
                    # The same rule roi_geometry_code applies. An invalid item
                    # bounding no area is a mask-to-contour artifact: it carries
                    # no geometry, so it can neither define nor contradict a
                    # source scope. Counting it a failure made the whole ROI
                    # ROI_UNRESOLVED_SOURCE_SCOPE -- moving the defect one step
                    # downstream rather than dropping the artifact.
                    continue
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
        # Area-less contour items (one or two points bounding no area) are also
        # withheld from the rasterizer copy: the scope validator skips them as
        # mask-to-contour artifacts carrying no geometry, but some rasterizer
        # builds reject them outright (cv2 fillPoly assertion) while others
        # draw nothing from them. Withholding is geometrically neutral -- such
        # an item contributes zero voxels either way -- and makes the accepted
        # ROI readable in every environment.
        prepared = copy.deepcopy(dataset)
        accepted = {n for n, r in self.scopes.items() if r.code is None}
        prepared.StructureSetROISequence = Sequence([r for r in prepared.StructureSetROISequence if int(r.ROINumber) in accepted])
        prepared.ROIContourSequence = Sequence([])
        for number in sorted(accepted):
            item = Dataset()
            item.ReferencedROINumber = number
            item.ContourSequence = Sequence(
                contour for contour in self.scopes[number].contours
                if _contour_rasterizable(contour)
            )
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
