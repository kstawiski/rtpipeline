"""Dose support and byte-bound mask identities, independent of dose intensity."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk


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


def rtstruct_grid_coverage(rtstruct, roi_number, dose):
    """Measure unsnapped contour coverage independently of the DVH values.

    Fraction is cross-sectional contour area integrated with trapezoidal
    interpolation between contour planes. No superior/inferior extrapolation
    is made. A single plane uses an area fraction (explicit method code).
    Nested contours use even/odd parity, as in dicompyler's rasterizer.
    """
    from .dvh import (classify_zero_dose_roi_geometry,
                      _dose_grid_contour_coordinates, _clip_polygon_to_axis_bound)
    from matplotlib.path import Path as PolygonPath
    classified = classify_zero_dose_roi_geometry(rtstruct, roi_number, dose)
    status = {'zero_dose_in_grid': 'fully_covered',
              'zero_dose_partly_inside_dose_grid': 'partial_grid',
              'zero_dose_outside_dose_grid': 'outside_grid'}.get(classified['status'], 'coverage_unresolved')
    result = {'status': status, 'fraction': None, 'method': 'contour_volume_trapezoid',
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
    def area(p):
        return abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))) / 2 if len(p) >= 3 else 0.0
    zs, totals, covered = [], [], []
    for z, polygons in sorted(planes.items()):
        total = inside = 0.0
        for p in polygons:
            # Hole sign is determined by containment, not contour winding.
            depth = sum(bool(PolygonPath(other[:, :2]).contains_points(p[:, :2]).all())
                        for other in polygons if other is not p)
            sign = -1 if depth % 2 else 1
            clipped = p
            for axis, bound in [(0, xmax), (1, ymax)]:
                clipped = _clip_polygon_to_axis_bound(clipped, axis=axis, boundary=0.0, keep_above=True, tolerance=0.0)
                clipped = _clip_polygon_to_axis_bound(clipped, axis=axis, boundary=bound, keep_above=False, tolerance=0.0)
            total += sign * area(p)
            inside += sign * area(clipped)
        zs.append(z); totals.append(total); covered.append(inside)
    if not zs or min(totals) <= 0:
        return {**result, 'status': 'coverage_unresolved', 'reason': 'Nonpositive contour area.'}
    zs = np.asarray(zs); totals = np.asarray(totals); covered = np.asarray(covered)
    if len(zs) == 1:
        total = totals[0]
        inside = covered[0] if offsets.min()-1e-3 <= zs[0] <= offsets.max()+1e-3 else 0.0
        result['method'] = 'single_plane_contour_area_fraction'
    else:
        total = float(np.trapezoid(totals, zs))
        low, high = max(zs[0], offsets.min()), min(zs[-1], offsets.max())
        knots = np.unique(np.r_[low, zs[(zs > low) & (zs < high)], high])
        inside = float(np.trapezoid(np.interp(knots, zs, covered), knots)) if high > low else 0.0
        result['roi_volume_cm3'] = total / 1000
        result['covered_volume_cm3'] = inside / 1000
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
