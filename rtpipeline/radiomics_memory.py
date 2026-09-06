"""Budgeted second-stage qualification for large, sparse CT crops.

The geometry envelope is not permission for unbounded texture matrices. Every
filtered image is checked again before PyRadiomics constructs a feature class.
No installed PyRadiomics code is modified.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
from typing import Any
import numpy as np
import yaml
from .radiomics_resource_guard import DEFAULT_MAX_RESAMPLED_BBOX_VOXELS

MEMORY_BUDGET_BYTES = 8 * 1024**3
FIXED_BYTES = 512 * 1024**2
MATRIX_RESERVE_BYTES = 1024**3
CROP_BYTES_PER_VOXEL = 128
NATIVE_BYTES_PER_VOXEL = 32
MASK_BYTES_PER_VOXEL = 128
STAGE1_CODE = "ROI_RESOURCE_BBOX_ADMITTED"
STAGE2_CODE = "ROI_RESOURCE_MEMORY_ADMITTED"

def legacy_rejection(estimate, limit, roi_name):
    from .radiomics_resource_guard import RESAMPLED_BBOX_LIMIT_CODE
    return {"metadata": {"roi_structural_code": RESAMPLED_BBOX_LIMIT_CODE,
                         **estimate.metadata(limit=limit)},
            "detail": (f"ROI {roi_name} requires an estimated padded resampled bounding "
                       f"box of {estimate.estimated_resampled_bbox_voxels} voxels "
                       f"({estimate.estimated_resampled_bbox_shape}); configured maximum "
                       f"is {limit}. Full configured radiomics was not started.")}


class RadiomicsMemoryLimit(RuntimeError):
    def __init__(self, message, predicted_peak_bytes=None):
        self.predicted_peak_bytes = predicted_peak_bytes
        super().__init__(message)


def predicted_peak_bytes(*, crop_voxels: int, native_image_voxels: int,
                         mask_voxels: int, gray_levels: int = 0,
                         max_gray_frequency: int | None = None,
                         max_extent: int = 1, reserve: bool = False,
                         glcm_directions: int = 62, gldm_neighbors: int = 124) -> int:
    """An allocation envelope, not an empirical RSS regression.

    GLSZM uses maxRegion, not the image bbox, as its last dimension. The
    frequency of the most common gray level bounds maxRegion without a costly
    connectivity pass. Two simultaneous dense matrices cover numpy.delete.
    Distances [1,2] yield at most 62 GLCM directions and 124 GLDM neighbors.
    """
    values = (crop_voxels, native_image_voxels, mask_voxels, gray_levels, max_extent, glcm_directions, gldm_neighbors)
    if any(not math.isfinite(v) or int(v) != v or v < 0 for v in values):
        raise ValueError("memory dimensions must be finite nonnegative integers")
    n, g = int(mask_voxels), int(gray_levels)
    z = n if max_gray_frequency is None else int(max_gray_frequency)
    if z < 0 or z > n:
        raise ValueError("gray frequency must lie within the foreground count")
    matrices = max(16*g*z, 32*g*g*glcm_directions, 32*g*int(max_extent)*13,
                   32*g*(gldm_neighbors+1), 32*g*3, MATRIX_RESERVE_BYTES if reserve else 0)
    return int(FIXED_BYTES + NATIVE_BYTES_PER_VOXEL*int(native_image_voxels)
               + CROP_BYTES_PER_VOXEL*int(crop_voxels)
               + MASK_BYTES_PER_VOXEL*n + matrices)


def _profile(settings, image_types):
    """Only the memory-qualified production filter profile may override bbox."""
    if settings.get('normalize', False) or settings.get('voxelBased', False):
        return False
    if set(image_types) - {'Original', 'LoG', 'Wavelet'}:
        return False
    wavelet = image_types.get('Wavelet', {}) or {}
    if wavelet.get('level', 1) != 1 or wavelet.get('start_level', 0) != 0:
        return False
    # Image-specific overrides are merged into kwargs by PyRadiomics. They
    # must not silently increase distance counts or alter the qualified filter
    # storage profile after this check of the global settings.
    permitted = {'Original': set(), 'LoG': {'sigma'},
                 'Wavelet': {'wavelet', 'level', 'start_level'}}
    for name, options in image_types.items():
        if set(options or {}) - permitted[name]:
            return False
    if wavelet.get('wavelet', 'coif1') != 'coif1':
        return False
    distances = settings.get('distances', [1])
    return bool(distances) and all(int(x) == x and 1 <= x <= 2 for x in distances)


def preflight_metadata(estimate, mask, *, native_spacing_xyz, array_axis_to_xyz,
                       settings=None, image_types=None, limit=DEFAULT_MAX_RESAMPLED_BBOX_VOXELS):
    """Retain the original fast path and bound preprocessing before its start.

    A per-native-voxel lattice bound is used, not the volume-ratio estimate.
    The latter is not an upper bound on nearest-neighbor occupied voxels.
    Filter-specific matrix costs are checked against actual post-resampling
    counts by install_texture_budget before any C feature allocation.
    """
    if estimate.estimated_resampled_bbox_voxels <= limit:
        return {'resource_guard_reason_code': STAGE1_CODE,
                'resource_guard_predicted_peak_bytes': 0,
                'resource_guard_memory_budget_bytes': MEMORY_BUDGET_BYTES,
                'resource_guard_crop_voxels_bound': math.prod(int(x)+2 for x in estimate.estimated_resampled_bbox_shape),
                'resource_guard_native_image_voxels': int(np.asarray(mask).size)}
    if limit < DEFAULT_MAX_RESAMPLED_BBOX_VOXELS:
        return None  # Respect an explicitly stricter administrator ceiling.
    settings = settings or {}
    image_types = image_types or {'Original': {}, 'LoG': {}, 'Wavelet': {}}
    if not _profile(settings, image_types):
        return None
    target = settings.get('resampledPixelSpacing') or native_spacing_xyz
    if len(target) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in target):
        return None
    # The legacy formula can undercount a floor/ceil boundary by two cells.
    b = math.prod(int(x)+2 for x in estimate.estimated_resampled_bbox_shape)
    replication = math.prod(math.ceil(float(a)/float(t)) for a,t in zip(native_spacing_xyz,target))
    n = min(b, int(estimate.native_foreground_voxels)*replication)
    native = int(np.asarray(mask).size)
    peak = predicted_peak_bytes(crop_voxels=b,native_image_voxels=native,
                                mask_voxels=n,reserve=True)
    if peak > MEMORY_BUDGET_BYTES:
        return None
    return {'resource_guard_reason_code': STAGE2_CODE,
            'resource_guard_predicted_peak_bytes': peak,
            'resource_guard_memory_budget_bytes': MEMORY_BUDGET_BYTES,
            'resource_guard_crop_voxels_bound': b,
            'resource_guard_mask_voxels_bound': n,
            'resource_guard_native_image_voxels': native}


def permits_second_stage(estimate, mask, *, native_spacing_xyz,
                         array_axis_to_xyz, params_file=None, settings=None,
                         image_types=None, limit=DEFAULT_MAX_RESAMPLED_BBOX_VOXELS):
    if params_file:
        data = yaml.safe_load(Path(params_file).read_text()) or {}
        settings = data.get('setting', {})
        image_types = data.get('imageType', {'Original': {}})
    return preflight_metadata(estimate, mask, native_spacing_xyz=native_spacing_xyz,
                              array_axis_to_xyz=array_axis_to_xyz, settings=settings,
                              image_types=image_types,limit=limit) is not None


def install_texture_budget(extractor: Any, metadata: dict) -> Any:
    """Check the real binned filtered-image dimensions before C allocation."""
    if metadata.get('resource_guard_reason_code', STAGE2_CODE) == STAGE2_CODE:
        extractor.settings['preCrop'] = True
    original = extractor.computeFeatures
    native = int(metadata['resource_guard_native_image_voxels'])
    crop = int(metadata['resource_guard_crop_voxels_bound'])
    def guarded(self, image, mask, imageTypeName, **kwargs):
        import SimpleITK as sitk
        a = sitk.GetArrayViewFromImage(image)
        foreground = sitk.GetArrayViewFromImage(mask) == kwargs.get('label', 1)
        values = a[foreground]
        if not values.size or not np.isfinite(values).all():
            raise RadiomicsMemoryLimit('nonfinite or empty filtered ROI')
        if kwargs.get('binCount'):
            _, edges = np.histogram(values, int(kwargs['binCount']))
            edges[-1] += 1
            bins = np.digitize(values, edges)
        else:
            width = float(kwargs.get('binWidth', 25))
            if not math.isfinite(width) or width <= 0:
                raise RadiomicsMemoryLimit('invalid bin width')
            # Match binImage/getBinEdges without allocating a full int64 crop.
            g_bound = math.floor(float(values.max())/width) - math.floor(float(values.min())/width) + 1
            if predicted_peak_bytes(crop_voxels=max(crop, a.size), native_image_voxels=native,
                                    mask_voxels=int(values.size), gray_levels=g_bound,
                                    max_gray_frequency=1, max_extent=max(a.shape)) > MEMORY_BUDGET_BYTES:
                raise RadiomicsMemoryLimit('gray-level range exceeds byte budget')
            low = values.min() - (values.min() % width)
            edges = np.arange(low, values.max() + 2 * width, width)
            if len(edges) == 1:
                edges = [edges[0] - .5, edges[0] + .5]
            bins = np.digitize(values, edges)
        g = int(bins.max())
        _, frequencies = np.unique(bins, return_counts=True)
        z = int(frequencies.max())
        n = int(values.size)
        distances = kwargs.get('distances', [1])
        # Chebyshev shells bound directions before force2D/short-axis pruning.
        neighbors = sum((2*int(d)+1)**3 - (2*int(d)-1)**3 for d in set(distances))
        peak = predicted_peak_bytes(crop_voxels=max(crop,a.size),
                                    native_image_voxels=native,mask_voxels=n,
                                    gray_levels=g,max_gray_frequency=z,
                                    max_extent=max(a.shape), glcm_directions=neighbors//2,
                                    gldm_neighbors=neighbors)
        metadata['resource_guard_predicted_peak_bytes'] = max(
            metadata['resource_guard_predicted_peak_bytes'],peak)
        if peak > MEMORY_BUDGET_BYTES:
            raise RadiomicsMemoryLimit('filtered-image allocation envelope exceeds byte budget', peak)
        del values, bins, foreground, frequencies
        enabled = getattr(self, 'enabledFeatures', None)
        if enabled is None:
            return original(image, mask, imageTypeName, **kwargs)
        from collections import OrderedDict
        result = OrderedDict()
        try:
            for family, names in enabled.items():
                self.enabledFeatures = {family: names}
                result.update(original(image, mask, imageTypeName, **kwargs))
        finally:
            self.enabledFeatures = enabled
        return result
    extractor.computeFeatures = MethodType(guarded, extractor)
    return extractor
