"""Dose support and byte-bound mask identities, independent of dose intensity."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _trapezoid(y, x):
    """Integrate with the NumPy API available in the active environment.

    ``numpy.trapezoid`` was added in NumPy 2.0, while the isolated radiomics
    environment intentionally remains on NumPy 1.x. Both APIs implement the
    same composite trapezoidal rule, so selecting the available name keeps the
    DVH grid-coverage calculation numerically identical across both stacks.
    """
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def grid_identity(image):
    return {"size": list(image.GetSize()), "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def nifti_identity(path):
    """Validate format AND decoded bytes. A suffix or a YAML path is not identity."""
    path = Path(path)
    if not str(path).endswith(('.nii', '.nii.gz')):
        raise ValueError('Source is not a NIfTI mask')
    digest = sha256_file(path)
    try:
        _validate_nifti_bytes(str(path.resolve()), digest)
    except Exception as exc:
        raise ValueError(f'Invalid NIfTI mask bytes: {exc}') from exc
    return {"path": str(path.resolve()), "sha256": digest}


@lru_cache(maxsize=2048)
def _validate_nifti_bytes(path, digest):
    # Every caller first hashes current file bytes. Cache never relies on mtime.
    import nibabel as nib
    img = nib.load(str(path))
    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)) or len(img.shape) != 3:
        raise ValueError('Source is not a three-dimensional NIfTI image')
    data = np.asanyarray(img.dataobj)
    if not np.isfinite(data).all() or not ((data == 0) | (data == 1)).all():
        raise ValueError('Source mask contains nonfinite or nonbinary voxels')
    if sha256_file(path) != digest:
        raise ValueError("Mask changed while its bytes were being validated")


def resample_grid_support(dose_image, reference_image):
    """Use an independent constant-one image and the dose sampler's support.

    SimpleITK linear interpolation accepts its half-voxel edge interval. The
    same resampler applied to ones records that interval, not dose > 0.
    """
    support = sitk.Image(dose_image.GetSize(), sitk.sitkFloat32) + 1.0
    support.CopyInformation(dose_image)
    sampled = sitk.Resample(support, reference_image, sitk.Transform(),
                            sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(sampled) >= 1.0 - 1e-6


def mask_grid_coverage(mask, support, voxel_volume_cm3):
    mask = np.asarray(mask, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if mask.shape != support.shape:
        raise ValueError('ROI and independent grid support must have identical shapes')
    total = int(np.count_nonzero(mask))
    inside = int(np.count_nonzero(mask & support))
    return {"status": ('empty_roi' if not total else 'outside_grid' if not inside
                        else 'fully_covered' if inside == total else 'partial_grid'),
            "fraction": inside / total if total else None,
            "method": 'independent_support_resampled_to_ct_voxel_centres',
            "roi_voxels": total, "covered_voxels": inside,
            "roi_volume_cm3": total * voxel_volume_cm3,
            "covered_volume_cm3": inside * voxel_volume_cm3}


def roi_nonvolume_coverage(rtstruct, roi_number):
    """Dispose of explicit nonvolume geometry before either grid comparison.

    An absent ROI is not a declared empty ROI. Malformed coordinates and mixed
    open/closed geometry remain unresolved, including when another item is valid.
    """
    items = []
    for item in getattr(rtstruct, 'ROIContourSequence', None) or []:
        try:
            if int(getattr(item, 'ReferencedROINumber', -1)) == int(roi_number):
                items.append(item)
        except (TypeError, ValueError):
            continue
    if not items:
        return None

    def disposition(status, code, reason):
        return {'status': status, 'fraction': None,
                'method': 'contour_geometry_disposition',
                'reason': reason, 'reason_code': code}

    contours = [c for item in items for c in getattr(item, 'ContourSequence', None) or []]
    if not contours:
        return disposition('declared_empty', 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE',
                           'ROI is declared with no contour items; no volume was measured.')
    kinds, points = [], []
    for contour in contours:
        kind = str(getattr(contour, 'ContourGeometricType', '') or '').strip().upper()
        try:
            p = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            if not len(p) or not np.isfinite(p).all():
                raise ValueError('Empty or nonfinite coordinates')
            if int(getattr(contour, 'NumberOfContourPoints', len(p))) != len(p):
                raise ValueError('Point count mismatch')
        except (ValueError, TypeError, AttributeError, OverflowError):
            return disposition('coverage_unresolved', 'ROI_MALFORMED_CONTOUR_DATA',
                               'ROI contains missing, malformed or inconsistent contour coordinates.')
        kinds.append(kind)
        points.append(p)
    nonvolume = [(k == 'POINT' and len(p) == 1) or
                 (k in {'OPEN_PLANAR', 'OPEN_NONPLANAR'} and len(p) >= 2)
                 for k, p in zip(kinds, points)]
    if all(nonvolume):
        return disposition('non_volumetric', 'ROI_NONVOLUMETRIC_CONTOURS',
                           'ROI contains only points or open lines; no volumetric DVH is defined.')
    if any(nonvolume) or any(k not in {'', 'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'} for k in kinds):
        return disposition('coverage_unresolved', 'ROI_MIXED_OR_UNSUPPORTED_CONTOURS',
                           'ROI contains mixed volumetric/non-volumetric or unsupported contour types.')
    if all(k in {'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'} and len(np.unique(p, axis=0)) < 3
           for k, p in zip(kinds, points)):
        return disposition('degenerate_contours', 'ROI_NO_AREA_BEARING_CONTOURS',
                           'All closed contour items have fewer than three distinct points.')
    return None


def _parity_plane_areas(polygons, xmax, ymax):
    """Integrate even/odd fill and its intersection with [0,xmax] x [0,ymax].

    A vertical scanline pairs sorted edge crossings across ALL rings. Split x
    at vertices, edge intersections and crossings of the clipping bounds. Within
    each open slab the pairing is fixed and both filled and clipped lengths are
    linear, so midpoint integration is exact up to floating-point roundoff.
    Retraced edges cancel by parity; self-touching/crossing rings need no winding
    or polygon repair library. Vertical edges have zero measure in this integral.
    Edge intersections use O(E^2) comparisons with O(E) temporary storage.
    """
    starts, ends = [], []
    for polygon in polygons:
        p = np.asarray(polygon, dtype=float)[:, :2]
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite planar coordinates')
        if len(p) < 3:
            continue
        q = np.roll(p, -1, axis=0)
        keep = p[:, 0] != q[:, 0]
        starts.append(p[keep])
        ends.append(q[keep])
    if not starts:
        return 0.0, 0.0
    a, b = np.concatenate(starts), np.concatenate(ends)
    if not len(a):
        return 0.0, 0.0
    # Evaluate an edge from the same endpoint in both traversal directions.
    # Exact retraces then cancel exactly instead of leaving a tiny positive area
    # from different floating-point evaluation orders.
    reverse = (a[:, 0] > b[:, 0])[:, None]
    a, b = np.where(reverse, b, a), np.where(reverse, a, b)
    slope = (b[:, 1] - a[:, 1]) / (b[:, 0] - a[:, 0])
    left, right = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
    knots = [left, right, np.array([0., xmax])]
    for i in range(len(a) - 1):
        lo = np.maximum(left[i], left[i+1:])
        hi = np.minimum(right[i], right[i+1:])
        delta = slope[i] - slope[i+1:]
        usable = (hi > lo) & (delta != 0)
        j = np.flatnonzero(usable) + i + 1
        x = a[i, 0] + (a[j, 1] + slope[j] * (a[i, 0] - a[j, 0]) - a[i, 1]) / delta[usable]
        knots.append(x[(x > lo[usable]) & (x < hi[usable])])
    nonhorizontal = slope != 0
    for bound in (0., ymax):
        x = a[nonhorizontal, 0] + (bound - a[nonhorizontal, 1]) / slope[nonhorizontal]
        knots.append(x[(x > left[nonhorizontal]) & (x < right[nonhorizontal])])
    xs = np.unique(np.concatenate(knots))
    total = inside = 0.0
    for low, high in zip(xs[:-1], xs[1:]):
        x = low + (high - low) / 2
        active = (left < x) & (x < right)
        y = np.sort(a[active, 1] + slope[active] * (x - a[active, 0]))
        if len(y) % 2:
            raise ValueError('Unpaired planar edge crossings')
        lower, upper = y[::2], y[1::2]
        total += float(np.sum(upper - lower)) * (high - low)
        if 0 < x < xmax:
            lengths = np.maximum(0., np.minimum(upper, ymax) - np.maximum(lower, 0.))
            inside += float(np.sum(lengths)) * (high - low)
    return total, inside


def rtstruct_grid_coverage(rtstruct, roi_number, dose):
    """Measure unsnapped contour coverage independently of the DVH values.

    Fraction is cross-sectional contour area integrated with trapezoidal
    interpolation between contour planes. No superior/inferior extrapolation
    is made. A single plane uses an area fraction (explicit method code).
    Nested contours use even/odd parity, as in dicompyler's rasterizer.
    """
    from .dvh import classify_zero_dose_roi_geometry, _dose_grid_contour_coordinates
    disposition = roi_nonvolume_coverage(rtstruct, roi_number)
    if disposition is not None:
        return disposition
    classified = classify_zero_dose_roi_geometry(rtstruct, roi_number, dose)
    status = {'zero_dose_in_grid': 'fully_covered',
              'zero_dose_partly_inside_dose_grid': 'partial_grid',
              'zero_dose_outside_dose_grid': 'outside_grid'}.get(classified['status'], 'coverage_unresolved')
    result = {'status': status, 'fraction': None, 'method': 'contour_volume_trapezoid_even_odd_linework',
              'reason': classified['reason']}
    if status in {'fully_covered', 'outside_grid', 'coverage_unresolved'}:
        result['fraction'] = 1.0 if status == 'fully_covered' else 0.0 if status == 'outside_grid' else None
        result['method'] = 'complete_contour_polygon_containment'
        return result
    contours, _ = _dose_grid_contour_coordinates(rtstruct, roi_number)
    origin = np.asarray(dose.ImagePositionPatient, float)
    orient = np.asarray(dose.ImageOrientationPatient, float)
    col = orient[:3] / np.linalg.norm(orient[:3])
    row = orient[3:] / np.linalg.norm(orient[3:])
    normal = np.cross(col, row)
    offsets = np.asarray(dose.GridFrameOffsetVector, float)
    if np.allclose(orient, [1, 0, 0, 0, 1, 0]) and np.isclose(offsets[0], origin[2], atol=1e-3):
        offsets = offsets - origin[2]
    xmax = (int(dose.Columns) - 1) * float(dose.PixelSpacing[1])
    ymax = (int(dose.Rows) - 1) * float(dose.PixelSpacing[0])
    planes = {}
    for contour in contours:
        q = (contour - origin) @ np.column_stack((col, row, normal))
        if len(q) < 3 or np.ptp(q[:, 2]) > 1e-3:
            return {**result, 'status': 'coverage_unresolved',
                    'reason': 'Volumetric coverage needs closed contours parallel to dose planes.'}
        planes.setdefault(round(float(np.mean(q[:, 2])), 4), []).append(q)
    zs, totals, covered = [], [], []
    for z, polygons in sorted(planes.items()):
        try:
            total, inside = _parity_plane_areas(polygons, xmax, ymax)
        except ValueError as exc:
            return {**result, 'status': 'coverage_unresolved', 'reason': str(exc),
                    'reason_code': 'ROI_PLANAR_AREA_UNRESOLVED'}
        zs.append(z); totals.append(total); covered.append(inside)
    if not zs or not np.isfinite(totals).all() or min(totals) < 0:
        return {**result, 'status': 'coverage_unresolved', 'reason': 'Invalid contour area.',
                'reason_code': 'ROI_PLANAR_AREA_UNRESOLVED'}
    zs = np.asarray(zs); totals = np.asarray(totals); covered = np.asarray(covered)
    if len(zs) == 1:
        total = totals[0]
        inside = covered[0] if offsets.min()-1e-3 <= zs[0] <= offsets.max()+1e-3 else 0.0
        result['method'] = 'single_plane_contour_area_fraction'
    else:
        total = float(_trapezoid(totals, zs))
        low, high = max(zs[0], offsets.min()), min(zs[-1], offsets.max())
        knots = np.unique(np.r_[low, zs[(zs > low) & (zs < high)], high])
        inside = float(_trapezoid(np.interp(knots, zs, covered), knots)) if high > low else 0.0
        result['roi_volume_cm3'] = total / 1000
        result['covered_volume_cm3'] = inside / 1000
    if not np.isfinite(total) or total <= 0 or not np.isfinite(inside):
        return {**result, 'status': 'coverage_unresolved',
                'reason': 'Nonpositive integrated contour volume or single-plane area.',
                'reason_code': 'ROI_NONPOSITIVE_CONTOUR_VOLUME'}
    fraction = float(np.clip(inside / total, 0, 1))
    result['fraction'] = 1.0 if status == 'fully_covered' else 0.0 if status == 'outside_grid' else fraction
    return result


def publish_derived_mask(directory, name, mask, ct_image, component_paths, config_path, definitions, outcomes, ct_files):
    """Publish exact consumed mask plus a conservative complete dependency closure.

    All loaded component masks are bound, including inputs to nested derived
    structures. This can invalidate extra rows but never overlooks a dependency.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    closure = {'schema': 'dvh-derived-mask-v1', 'name': name,
               'ct_grid': grid_identity(ct_image),
               'ct_sources': [{'path': str(Path(p).resolve()), 'sha256': sha256_file(p)} for p in ct_files],
               'components': [nifti_identity(p) for p in sorted(set(component_paths))],
               'configuration': {'path': str(Path(config_path).resolve()), 'sha256': sha256_file(config_path)},
               'definitions': definitions, 'outcomes': outcomes,
               'implementation_sha256': sha256_file(Path(__file__).with_name('custom_structures.py'))}
    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8)); image.CopyInformation(ct_image)
    key = hashlib.sha256(name.encode()).hexdigest()[:20]
    mask_path = directory / (key + '.nii.gz')
    temp = directory / (key + '.tmp.nii.gz')
    sitk.WriteImage(image, str(temp))
    identity = nifti_identity(temp)
    if not np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(temp))), mask):
        raise ValueError('Published derived mask bytes differ from consumed mask')
    temp.replace(mask_path)
    closure['mask'] = {'path': str(mask_path.resolve()), 'sha256': identity['sha256']}
    path = directory / (key + '.json')
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(closure, sort_keys=True), encoding='utf-8'); tmp.replace(path)
    validate_derived_mask(path)
    return path


def validate_derived_mask(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError('Derived mask closure bytes changed')
    closure = json.loads(path.read_text(encoding='utf-8'))
    if closure.get('schema') != 'dvh-derived-mask-v1':
        raise ValueError('Unknown derived mask provenance schema')
    if not closure.get('components') or not closure.get('definitions') or not closure.get('ct_sources'):
        raise ValueError('Incomplete derived-mask dependency closure')
    if not {'size', 'spacing', 'origin', 'direction'} <= closure.get('ct_grid', {}).keys():
        raise ValueError('Missing CT-grid identity')
    for item in closure['components'] + [closure['mask']]:
        if nifti_identity(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Source or derived mask bytes changed')
    for item in closure['ct_sources'] + [closure['configuration']]:
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('CT or configuration bytes changed')
    if sha256_file(Path(__file__).with_name('custom_structures.py')) != closure.get('implementation_sha256'):
        raise ValueError('Derived-mask implementation changed')
    image = sitk.ReadImage(closure['mask']['path'])
    actual = grid_identity(image)
    if any(not np.allclose(actual[k], closure['ct_grid'][k], rtol=1e-6, atol=1e-5) for k in actual):
        raise ValueError('Derived mask does not use the recorded CT grid')
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}
