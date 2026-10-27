"""Explicit IBSI CT preprocessing, adapted from validation/ibsi/scripts/run_validation.py.

The benchmark implementation supplies the centred grid, edge padding, linear
mask interpolation, HU rounding and independent range/sigma criteria. Feature
formulas and discretisation remain those of PyRadiomics.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import numpy as np
import SimpleITK as sitk


PREPROCESSING_KEY = "rtpipelinePreprocessing"
SIGMA_KEY = "rtpipelineResegmentSigma"


def preprocessing_options(params: Mapping) -> dict:
    options = params.get("radiomics") or {}
    if not isinstance(options, Mapping):
        raise ValueError("radiomics must be a mapping")
    unknown = options.keys() - {"preprocessing", "resegment_sigma"}
    if unknown:
        raise ValueError(f"Unknown radiomics preprocessing options: {sorted(unknown)}")
    mode = options.get("preprocessing", "native")
    if mode not in ("native", "ibsi"):
        raise ValueError("radiomics.preprocessing must be native or ibsi")
    if mode == "native":
        if "resegment_sigma" in options:
            raise ValueError("resegment_sigma requires radiomics.preprocessing: ibsi")
        return {}
    sigma = options.get("resegment_sigma")
    result = {PREPROCESSING_KEY: "ibsi"}
    if sigma is not None:
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("radiomics.resegment_sigma must be finite and positive")
        result[SIGMA_KEY] = sigma
    return result


def _read(source):
    return sitk.ReadImage(str(source)) if isinstance(source, (str, Path)) else source


def _rounded(image):
    rounded = sitk.GetImageFromArray(
        np.rint(sitk.GetArrayFromImage(image)).astype(np.float32)
    )
    rounded.CopyInformation(image)
    return rounded


def spatial_preprocess(image, mask, settings: Mapping):
    """Return the rounded CT and unresegmented binary morphology mask.

    A zero target spacing preserves that axis. Dimensions use ceil(N*s/s_new),
    and the physical centres of the old and new grids coincide. No crop precedes
    interpolation. Three replicated edge voxels reproduce the public benchmark.
    """
    image, mask = _read(image), _read(mask)
    if mask.GetNumberOfComponentsPerPixel() > 1:
        mask = sitk.VectorIndexSelectionCast(mask, int(settings.get("label_channel", 0)))
    if (image.GetDimension() != 3 or mask.GetDimension() != 3
            or image.GetNumberOfComponentsPerPixel() != 1):
        raise ValueError("IBSI preprocessing requires scalar 3-D CT and mask images")
    if image.GetSize() != mask.GetSize() or any(
        not np.allclose(a, b, atol=1e-6, rtol=0) for a, b in (
            (image.GetOrigin(), mask.GetOrigin()),
            (image.GetSpacing(), mask.GetSpacing()),
            (image.GetDirection(), mask.GetDirection()),
        )
    ):
        raise ValueError("IBSI CT and morphology mask must share their input grid")
    if settings.get("normalize", False) or settings.get("removeOutliers") is not None:
        raise ValueError("IBSI CT preprocessing requires normalize: false and no removeOutliers")
    if settings.get("resegmentShape", False):
        raise ValueError("IBSI preprocessing keeps morphology unresegmented; resegmentShape must be false")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("IBSI CT intensities must be finite")
    if not np.isfinite(sitk.GetArrayViewFromImage(mask)).all():
        raise ValueError("IBSI mask values must be finite")
    image = _rounded(image)
    morph = sitk.Cast(mask == int(settings.get("label", 1)), sitk.sitkUInt8)
    target = settings.get("resampledPixelSpacing")
    if target is None:
        return image, morph
    spacing = np.asarray(target, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing < 0):
        raise ValueError("IBSI target spacing must contain three finite nonnegative values")
    old_spacing = np.asarray(image.GetSpacing())
    spacing = np.where(spacing == 0, old_spacing, spacing)
    old_size = np.asarray(image.GetSize())
    size = np.ceil(old_size * old_spacing / spacing).astype(int)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    origin = np.asarray(image.GetOrigin()) + direction @ (
        (old_size - 1) * old_spacing / 2 - (size - 1) * spacing / 2
    )
    method = settings.get("interpolator", "sitkBSpline")
    if isinstance(method, str):
        method = getattr(sitk, method, None)
    if method is None:
        raise ValueError("IBSI resampling requires a configured SimpleITK interpolator")
    grid = sitk.Image([int(n) for n in size], sitk.sitkFloat32)
    grid.SetSpacing(spacing.tolist())
    grid.SetOrigin(origin.tolist())
    grid.SetDirection(image.GetDirection())
    padded = sitk.ZeroFluxNeumannPad(image, [3] * 3, [3] * 3)
    processed = sitk.Resample(padded, grid, sitk.Transform(), method, -1000., sitk.sitkFloat32)
    fractional = sitk.Resample(
        sitk.Cast(morph, sitk.sitkFloat32), grid, sitk.Transform(),
        sitk.sitkLinear, 0., sitk.sitkFloat32,
    )
    return _rounded(processed), sitk.Cast(fractional >= 0.5, sitk.sitkUInt8)


def resegmentation_bounds(values: np.ndarray, settings: Mapping):
    """Inclusive criteria, each evaluated on the unresegmented morphology ROI."""
    lower, upper = -np.inf, np.inf
    limits = settings.get("resegmentRange")
    if limits is not None:
        limits = np.asarray(limits, dtype=float)
        if (limits.ndim != 1 or len(limits) not in (1, 2)
                or not np.isfinite(limits).all()
                or (len(limits) == 2 and limits[0] > limits[1])):
            raise ValueError("resegmentRange must contain one or two ordered finite limits")
        mode = settings.get("resegmentMode", "absolute")
        if mode == "sigma":
            limits = values.mean(dtype=np.float64) + limits * values.std(dtype=np.float64, ddof=0)
        elif mode == "relative":
            limits = limits * values.max()
        elif mode != "absolute":
            raise ValueError("Unknown resegmentMode")
        lower = limits[0]
        if len(limits) == 2:
            upper = limits[1]
    sigma = settings.get(SIGMA_KEY)
    if sigma is not None:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("IBSI sigma must be finite and positive")
        mean, sd = values.mean(dtype=np.float64), values.std(dtype=np.float64, ddof=0)
        lower, upper = max(lower, mean - sigma * sd), min(upper, mean + sigma * sd)
    return lower, upper


def intensity_mask(image, morph, settings: Mapping):
    array = sitk.GetArrayViewFromImage(image)
    foreground = sitk.GetArrayFromImage(morph) > 0
    values = array[foreground]
    if values.size == 0:
        raise ValueError("IBSI morphology mask is empty after interpolation")
    lower, upper = resegmentation_bounds(values.astype(np.float64), settings)
    keep = foreground & (array >= lower) & (array <= upper)
    result = sitk.GetImageFromArray(keep.astype(np.uint8))
    result.CopyInformation(morph)
    return result


def ibsi_preprocess(image, mask, settings: Mapping):
    """Return CT, morphology mask and resegmented intensity mask."""
    processed, morph = spatial_preprocess(image, mask, settings)
    return processed, morph, intensity_mask(processed, morph, settings)


def extractor_class():
    """Import PyRadiomics only inside its supported NumPy environment."""
    from radiomics.featureextractor import RadiomicsFeatureExtractor

    class IBSIFeatureExtractor(RadiomicsFeatureExtractor):
        @staticmethod
        def loadImage(imageFilepath, maskFilepath, generalInfo=None, **settings):
            image, morph = spatial_preprocess(imageFilepath, maskFilepath, settings)
            native = dict(settings, resampledPixelSpacing=None, interpolator=None,
                          preCrop=False, normalize=False, label=1)
            return RadiomicsFeatureExtractor.loadImage(image, morph, generalInfo, **native)

        def execute(self, imageFilepath, maskFilepath, label=None, label_channel=None, voxelBased=False):
            settings = dict(self.settings)
            if label is not None:
                settings["label"] = label
            if label_channel is not None:
                settings["label_channel"] = label_channel
            image, morph = spatial_preprocess(imageFilepath, maskFilepath, settings)
            shape_names = {"shape", "shape2D"} & self.enabledFeatures.keys()
            intensity_names = self.enabledFeatures.keys() - shape_names

            def prepared(features):
                # Independent instance state; no mutation of caller settings,
                # no second resampling, and no second resegmentation.
                ext = RadiomicsFeatureExtractor()
                ext.settings = dict(settings, resampledPixelSpacing=None, interpolator=None,
                                    preCrop=False, normalize=False, label=1,
                                    resegmentRange=None, resegmentShape=False)
                ext.enabledFeatures = {k: deepcopy(v) for k, v in self.enabledFeatures.items()
                                       if k in features}
                ext.enabledImagetypes = deepcopy(self.enabledImagetypes)
                # Retain instance-level feature guards installed by the caller.
                for name in ("computeFeatures", "computeShape"):
                    if name == "computeFeatures" and not (features - shape_names):
                        continue
                    if name in self.__dict__:
                        original = self.__dict__[name]

                        def guarded(*args, _original=original, _ext=ext, **kwargs):
                            # Guards can close over bound methods of the source
                            # extractor. Keep their feature selection consistent
                            # for this call and restore it even after a failure.
                            previous = self.enabledFeatures
                            self.enabledFeatures = _ext.enabledFeatures
                            try:
                                return _original(*args, **kwargs)
                            finally:
                                self.enabledFeatures = previous

                        setattr(ext, name, guarded)
                return ext

            result = {}
            if intensity_names:
                intensity = intensity_mask(image, morph, settings)
                ext = prepared(intensity_names)
                result.update(ext.execute(image, intensity, voxelBased=voxelBased))
            if shape_names and not voxelBased:
                ext = prepared(shape_names)
                ext.enabledImagetypes = {"Original": {}}
                result.update(ext.execute(image, morph))
            if settings.get("additionalInfo", False):
                result["diagnostics_RTpipeline_Preprocessing"] = "ibsi"
                result["diagnostics_RTpipeline_PreprocessingSettings"] = settings
            return result

    return IBSIFeatureExtractor
