from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import weakref
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import pydicom


from .config import DEFAULT_MAX_TOTAL_DOSE_GY, PipelineConfig
from .course_contract import (
    ALL_SERIES_RADIOMICS_TEMP_AUTHORITY,
    ALL_SERIES_RADIOMICS_TEMP_SCOPE,
    AUTO_RTSTRUCT_SOURCE,
    COURSE_CONTRACT_VERSION,
    build_dvh_decision,
    load_course_contract,
)
from .layout import build_course_dirs
from .modality_classifier import is_quantitative_image_class
from importlib import resources as importlib_resources
import yaml
from .utils import run_tasks_with_adaptive_workers, mask_is_cropped, _scoped_walk
from .custom_models import (
    list_custom_model_outputs,
    validate_custom_model_output_inventory,
)
from .custom_structures_rtstruct import (
    _create_custom_structures_rtstruct,
    _is_rs_custom_stale,
    record_rs_custom_resume_decision,
)
from .radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    course_diagnostic_columns,
    extraction_status_is_nonfatal_for_required,
    invalidate_radiomics_outputs as _invalidate_radiomics_outputs,
    outcome_from_output,
    remove_artifact_strict as _remove_artifact_strict,
    roi_source_is_required,
    resume_identity_pairs as _resume_identity_pairs,
    write_excel_atomic as _write_excel_atomic,
)

if TYPE_CHECKING:
    from radiomics import featureextractor

# Image cache for avoiding repeated DICOM loading (significant I/O savings)
# Configurable via environment variables: RTPIPELINE_IMAGE_CACHE_SIZE, RTPIPELINE_IMAGE_CACHE_AGE_SEC
_IMAGE_CACHE: Dict[str, Tuple[sitk.Image, float]] = {}
_IMAGE_CACHE_LOCK = threading.Lock()
_IMAGE_CACHE_MAX_SIZE = int(os.environ.get('RTPIPELINE_IMAGE_CACHE_SIZE', '8'))
_IMAGE_CACHE_MAX_AGE_SEC = int(os.environ.get('RTPIPELINE_IMAGE_CACHE_AGE_SEC', '300'))

_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)
_THREAD_LIMIT_ENV = 'RTPIPELINE_RADIOMICS_THREAD_LIMIT'


def _resolve_thread_limit(value: Optional[int]) -> Optional[int]:
    if value is not None:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            coerced = None
    else:
        env_raw = os.environ.get(_THREAD_LIMIT_ENV)
        try:
            coerced = int(env_raw) if env_raw is not None else None
        except (TypeError, ValueError):
            coerced = None
    if coerced is None or coerced <= 0:
        return None
    return coerced


def _apply_radiomics_thread_limit(limit: Optional[int]) -> None:
    if limit is None:
        for var in _THREAD_ENV_VARS:
            os.environ.pop(var, None)
        return
    limit = max(1, int(limit))
    value = str(limit)
    for var in _THREAD_ENV_VARS:
        os.environ[var] = value

logger = logging.getLogger(__name__)


def _deduplicate_rtstruct_sources(
    sources: Iterable[Tuple[str, Path, Optional[List[str]]]],
) -> List[Tuple[str, Path, Optional[List[str]]]]:
    """Keep one source identity per resolved RTSTRUCT path, preserving order."""
    unique: List[Tuple[str, Path, Optional[List[str]]]] = []
    seen: set[Path] = set()
    for source, path, roi_names in sources:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((source, Path(path), roi_names))
    return unique


def _standard_rtstruct_sources(
    contract, course_dir: Path
) -> List[Tuple[str, Path, Optional[List[str]]]]:
    """Resolve standard manual/auto sources with contracted provenance."""
    sources: List[Tuple[str, Path, Optional[List[str]]]] = []
    auto = Path(course_dir) / "RS_auto.dcm"
    auto_is_current = False
    if auto.exists():
        planning_ct = contract.planning_ct
        planning_series_uid = str(planning_ct.get("series_instance_uid") or "").strip()
        try:
            from .auto_rtstruct import (
                _derived_rtstruct_dependencies_are_current,
                _rtstruct_matches_planning_ct,
            )

            auto_is_current = _rtstruct_matches_planning_ct(auto, planning_series_uid)
            if auto_is_current:
                auto_is_current = _derived_rtstruct_dependencies_are_current(
                    auto,
                    ct_dir=contract.planning_ct_dir,
                    nifti_path=contract.planning_ct_nifti,
                    segmentation_root=build_course_dirs(course_dir).segmentation_totalseg,
                )
        except Exception as exc:
            logger.warning("Could not validate RS_auto provenance for %s: %s", course_dir, exc)
            auto_is_current = False
        if not auto_is_current:
            logger.warning(
                "Excluding rejected RS_auto.dcm from radiomics sources for %s",
                course_dir,
            )

    authoritative = contract.authoritative_rtstruct_path
    if authoritative is not None and authoritative.exists():
        authoritative_is_auto = authoritative.resolve() == auto.resolve()
        if not authoritative_is_auto or auto_is_current:
            sources.append((contract.authoritative_rtstruct_source, authoritative, None))
    if auto.exists() and auto_is_current:
        sources.append((AUTO_RTSTRUCT_SOURCE, auto, None))
    return _deduplicate_rtstruct_sources(sources)


def _mr_radiomics_required(config: PipelineConfig) -> bool:
    """MR is optional unless the caller configured an MR parameter path."""
    return getattr(config, "radiomics_params_file_mr", None) is not None


_DEFAULT_MIN_VOXELS = 120
_DEFAULT_MAX_VOXELS_FULL = 15_000_000
_LARGE_ROI_RESAMPLED_SPACING_MM = (2.0, 2.0, 2.0)


_PARAM_CACHE_LOCK = threading.Lock()
_PARAM_CACHE: Dict[Path, Tuple[Tuple[float, int], Dict[str, Any]]] = {}


def _load_radiomics_params_dict(pfile: Path) -> Optional[Dict[str, Any]]:
    """Load radiomics YAML parameters with caching to avoid repeated parsing.

    Returns a deep copy so callers are free to mutate the structure without
    impacting the cached reference.
    """

    try:
        stat = pfile.stat()
    except FileNotFoundError:
        logger.warning("Radiomics params file missing: %s", pfile)
        return None
    except Exception as exc:
        logger.warning("Unable to stat radiomics params file %s: %s", pfile, exc)
        return None

    key = pfile.resolve()
    meta = (stat.st_mtime, stat.st_size)

    with _PARAM_CACHE_LOCK:
        cached = _PARAM_CACHE.get(key)
        if cached and cached[0] == meta:
            return deepcopy(cached[1])

    try:
        with pfile.open('r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to parse radiomics params %s: %s", pfile, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Radiomics params in %s must be a mapping", pfile)
        return None

    with _PARAM_CACHE_LOCK:
        _PARAM_CACHE[key] = (meta, data)

    return deepcopy(data)


def _apply_params_to_extractor(ext: "featureextractor.RadiomicsFeatureExtractor", params: Dict[str, Any]) -> None:
    """Apply cached YAML parameters to a freshly created extractor instance."""

    settings = params.get('setting') or {}
    voxel_settings = params.get('voxelSetting') or {}
    image_types = params.get('imageType') or {}
    feature_classes = params.get('featureClass') or {}

    # Ensure numeric settings retain numeric types even if parsed as strings
    if isinstance(settings, dict) and isinstance(settings.get('geometryTolerance'), str):
        try:
            settings['geometryTolerance'] = float(settings['geometryTolerance'])
        except ValueError:
            pass

    # Merge settings with voxel specific overrides
    if isinstance(settings, dict):
        ext.settings.update(settings)
    if isinstance(voxel_settings, dict):
        ext.settings.update(voxel_settings)

    # Configure enabled image types
    if isinstance(image_types, dict) and image_types:
        normalized_images: Dict[str, Dict[str, Any]] = {}
        for name, cfg in image_types.items():
            try:
                normalized_images[str(name)] = dict(cfg) if isinstance(cfg, dict) else {}
            except Exception:
                normalized_images[str(name)] = {}
        ext.enabledImagetypes = normalized_images
    else:
        ext.enabledImagetypes = {"Original": {}}

    # Configure exactly the feature classes named by the parameter file.
    normalized_features: Dict[str, List[str]] = {}
    if isinstance(feature_classes, dict) and feature_classes:
        for name, values in feature_classes.items():
            if isinstance(values, list):
                normalized_features[str(name)] = list(values)
            elif isinstance(values, dict):
                # PyRadiomics treats dict specification similar to enabling all entries
                normalized_features[str(name)] = list(values.keys())
            elif values is None:
                normalized_features[str(name)] = []
            else:
                normalized_features[str(name)] = [str(values)]
    if not normalized_features:
        normalized_features = {fc: [] for fc in ext.featureClassNames if fc != 'shape2D'}
    else:
        unknown_classes = set(normalized_features) - set(ext.featureClassNames)
        if unknown_classes:
            raise ValueError(
                f"Unknown PyRadiomics feature classes: {sorted(unknown_classes)}"
            )
    ext.enabledFeatures = normalized_features

    # Update SimpleITK tolerance in case geometryTolerance was provided.
    ext._setTolerance()


def _have_pyradiomics() -> bool:
    """Return True when radiomics features can be extracted (directly or via conda)."""

    import numpy as np

    major_version = int(np.__version__.split('.')[0])

    if major_version >= 2:
        try:
            from .radiomics_conda import check_radiomics_env, RADIOMICS_ENV
        except ImportError:
            logger.warning(
                "NumPy %s detected but conda fallback helpers are unavailable",
                np.__version__,
            )
            return False
        if check_radiomics_env():
            logger.info(
                "NumPy %s detected; will route PyRadiomics calls through conda environment '%s'",
                np.__version__,
                RADIOMICS_ENV,
            )
            return True
        logger.warning(
            "NumPy %s detected and radiomics conda environment '%s' is not available",
            np.__version__,
            RADIOMICS_ENV,
        )
        return False

    try:
        from radiomics import featureextractor  # type: ignore
        logger.debug("Native PyRadiomics available under NumPy %s", np.__version__)
        return True
    except ImportError:
        logger.debug("Native PyRadiomics unavailable; probing conda fallback")
        try:
            from .radiomics_conda import check_radiomics_env, RADIOMICS_ENV
        except ImportError:
            logger.warning("PyRadiomics missing and conda fallback helpers not importable")
            return False
        if check_radiomics_env():
            logger.info(
                "Using conda-based PyRadiomics fallback (env '%s') under NumPy %s",
                RADIOMICS_ENV,
                np.__version__,
            )
            return True
        logger.warning("PyRadiomics missing and conda fallback environment unavailable")
        return False


def _load_series_image(dicom_dir: Path, series_uid: Optional[str] = None, use_cache: bool = True) -> Optional[sitk.Image]:
    """Load a DICOM series as a SimpleITK image with optional caching.

    The cache stores recently loaded images to avoid repeated disk I/O when processing
    multiple structures from the same scan. This can provide 10-50x speedup for
    subsequent accesses to the same series.

    Args:
        dicom_dir: Path to the DICOM directory
        series_uid: Optional specific series UID to load
        use_cache: Whether to use the image cache (default: True)

    Returns:
        SimpleITK Image or None if loading fails
    """
    import time

    cache_key = f"{dicom_dir}:{series_uid or 'default'}"

    # Check cache first
    if use_cache:
        with _IMAGE_CACHE_LOCK:
            if cache_key in _IMAGE_CACHE:
                cached_img, cached_time = _IMAGE_CACHE[cache_key]
                # Check if cache entry is still valid
                if time.time() - cached_time < _IMAGE_CACHE_MAX_AGE_SEC:
                    # Update timestamp on hit (proper LRU behavior)
                    _IMAGE_CACHE[cache_key] = (cached_img, time.time())
                    logger.debug("Image cache hit for %s", cache_key)
                    return cached_img
                else:
                    # Expired entry
                    del _IMAGE_CACHE[cache_key]

    # Load from disk
    try:
        reader = sitk.ImageSeriesReader()
        sids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not sids:
            return None
        sid = series_uid if (series_uid and series_uid in sids) else sids[0]
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)
        reader.SetFileNames(files)
        img = reader.Execute()

        # Store in cache
        if use_cache and img is not None:
            with _IMAGE_CACHE_LOCK:
                # Evict old entries if cache is full
                if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX_SIZE:
                    # Remove oldest entry
                    oldest_key = min(_IMAGE_CACHE.keys(), key=lambda k: _IMAGE_CACHE[k][1])
                    del _IMAGE_CACHE[oldest_key]
                    logger.debug("Evicted oldest cache entry: %s", oldest_key)

                _IMAGE_CACHE[cache_key] = (img, time.time())
                logger.debug("Cached image for %s", cache_key)

        return img
    except Exception as e:
        logger.debug("Failed loading series %s in %s: %s", series_uid, dicom_dir, e)
        return None


def clear_image_cache() -> int:
    """Clear the image cache and return number of entries cleared."""
    with _IMAGE_CACHE_LOCK:
        count = len(_IMAGE_CACHE)
        _IMAGE_CACHE.clear()
        logger.debug("Cleared %d entries from image cache", count)
        return count


def _resample_to_reference(img: sitk.Image, ref: sitk.Image, nn: bool = True) -> sitk.Image:
    if (img.GetSize() == ref.GetSize() and img.GetSpacing() == ref.GetSpacing() and img.GetDirection() == ref.GetDirection() and img.GetOrigin() == ref.GetOrigin()):
        return img
    return sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor if nn else sitk.sitkLinear, 0, img.GetPixelID())


def _mask_from_array_like(ct_img: sitk.Image, mask3d: np.ndarray) -> sitk.Image:
    # Radiomics expects a label image with same geometry as ct_img.
    # SimpleITK GetImageFromArray expects [z,y,x] order. rt-utils may return [y,x,z].
    sx, sy, sz = ct_img.GetSize()
    zyx = mask3d
    if mask3d.shape == (sz, sy, sx):
        zyx = mask3d
    elif mask3d.shape == (sy, sx, sz):
        zyx = np.transpose(mask3d, (2, 0, 1))
    elif mask3d.shape == (sx, sy, sz):
        zyx = np.transpose(mask3d, (2, 1, 0))
    m = sitk.GetImageFromArray(zyx.astype(np.uint8))
    m.SetSpacing(ct_img.GetSpacing())
    m.SetDirection(ct_img.GetDirection())
    m.SetOrigin(ct_img.GetOrigin())
    return m


def _get_params_file(config: PipelineConfig | None, modality: str = 'CT') -> Optional[Path]:
    """Return a filesystem path to a radiomics params YAML.
    Prefer a user file, then a packaged filesystem resource. Copy only when a
    configured logs directory or a non-filesystem package resource requires it.
    """
    try:
        modality_upper = (modality or 'CT').upper()
        if modality_upper == 'MR':
            candidate = getattr(config, 'radiomics_params_file_mr', None) if config else None
            packaged_name = 'radiomics_params_mr.yaml'
        else:
            candidate = getattr(config, 'radiomics_params_file', None) if config else None
            packaged_name = 'radiomics_params.yaml'

        if candidate and Path(candidate).exists():
            return Path(candidate)
        # Copy packaged file to logs_root for stable path
        packaged = importlib_resources.files('rtpipeline').joinpath(packaged_name)
        if packaged.is_file():
            if not (config and config.logs_root):
                try:
                    packaged_path = Path(packaged)
                except TypeError:
                    packaged_path = None
                if packaged_path is not None and packaged_path.is_file():
                    return packaged_path
            target_root = (
                Path(config.logs_root)
                if (config and config.logs_root)
                else Path(tempfile.gettempdir()) / "rtpipeline-resources"
            )
            target_root.mkdir(parents=True, exist_ok=True)
            out = Path(target_root) / packaged_name
            try:
                out.write_bytes(packaged.read_bytes())
            except Exception:
                # Fallback: return a temp-like path via as_file context
                try:
                    from importlib.resources import as_file
                except Exception:
                    as_file = None  # type: ignore
                if as_file is not None:
                    with as_file(packaged) as p:
                        return Path(p)
            return out
    except Exception as e:
        logger.warning("Failed to prepare radiomics params file: %s", e)
    return None


def _extractor(config: PipelineConfig, modality: str = 'CT', normalize_override: Optional[bool] = None) -> Optional["featureextractor.RadiomicsFeatureExtractor"]:
    # Check NumPy version to decide which approach to use
    import numpy as np
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

    if numpy_version[0] >= 2:
        # For NumPy 2.x we always delegate to the conda executor.
        logger.debug(
            "NumPy %s detected – returning None to trigger conda-based radiomics",
            np.__version__,
        )
        return None  # Signals to use conda-based execution

    # Direct usage with NumPy 1.x
    logger.debug("NumPy 1.x detected (%s), using PyRadiomics directly", np.__version__)
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
        logging.getLogger('radiomics.featureextractor').setLevel(logging.WARNING)

        pfile = _get_params_file(config, modality)
        params_dict: Optional[Dict[str, Any]] = None
        if pfile is not None:
            params_dict = _load_radiomics_params_dict(pfile)

        ext = RadiomicsFeatureExtractor()

        if params_dict is not None:
            _apply_params_to_extractor(ext, params_dict)
        elif pfile is not None:
            # Fallback to the library parser if caching fails for any reason
            ext.loadParams(str(pfile))

        # Adjust per-modality recommendations
        if modality.upper() == 'MR':
            # Prefer binCount=64 for MRI; toggle normalization based on detected weighting
            try:
                ext.settings['binCount'] = 64
                if 'binWidth' in ext.settings:
                    del ext.settings['binWidth']
            except Exception:
                pass
            if normalize_override is not None:
                try:
                    ext.settings['normalize'] = bool(normalize_override)
                except Exception:
                    pass

        return ext
    except Exception as e:
        logger.warning("Failed to create RadiomicsFeatureExtractor: %s", e)
        return None


def _derive_voxel_limits(config: PipelineConfig) -> tuple[int, int]:
    min_voxels = getattr(config, "radiomics_min_voxels", None)
    max_voxels = getattr(config, "radiomics_max_voxels", None)
    try:
        min_v = int(min_voxels) if min_voxels not in (None, "") else _DEFAULT_MIN_VOXELS
    except Exception:
        min_v = _DEFAULT_MIN_VOXELS
    try:
        max_v = int(max_voxels) if max_voxels not in (None, "") else _DEFAULT_MAX_VOXELS_FULL
    except Exception:
        max_v = _DEFAULT_MAX_VOXELS_FULL
    if min_v < 1:
        min_v = 1
    if max_v < 1:
        max_v = _DEFAULT_MAX_VOXELS_FULL
    return min_v, max_v


def _extractor_large_roi(config: PipelineConfig, modality: str = "CT") -> Optional["featureextractor.RadiomicsFeatureExtractor"]:
    """Reduced extractor for very large ROIs (e.g., BODY).

    Uses:
    - Original image only (no LoG/Wavelet)
    - shape + firstorder only
    - coarser isotropic resampling (default 2mm) for feasibility
    """
    ext = _extractor(config, modality)
    if ext is None:
        return None
    try:
        ext.disableAllImageTypes()
        ext.enableImageTypeByName("Original")
    except Exception:
        pass
    try:
        ext.disableAllFeatures()
        ext.enableFeatureClassByName("firstorder")
        ext.enableFeatureClassByName("shape")
    except Exception:
        pass
    try:
        # Coarser resampling dramatically reduces runtime/memory for large ROIs.
        ext.settings["resampledPixelSpacing"] = list(_LARGE_ROI_RESAMPLED_SPACING_MM)
    except Exception:
        pass
    return ext


def _custom_roi_names_from_config(path: Path) -> set[str]:
    """Return the exact declared custom-ROI inventory or fail closed."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"Configured custom structure file could not be read: {path}: {exc}"
        ) from exc

    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RadiomicsCourseExtractionError(
            f"Configured custom structure file could not be parsed as YAML/JSON: "
            f"{path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RadiomicsCourseExtractionError(
            f"Configured custom structure file top level must be a mapping: {path}"
        )
    if "custom_structures" not in data:
        raise RadiomicsCourseExtractionError(
            f"Configured custom structure file is missing required "
            f"'custom_structures' section: {path}"
        )

    items = data["custom_structures"]
    if not isinstance(items, list) or not items:
        raise RadiomicsCourseExtractionError(
            f"Configured 'custom_structures' section must be a non-empty list: {path}"
        )

    out: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure entry {index} must be a mapping: {path}"
            )
        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure entry {index} has an invalid 'name': {path}"
            )
        name = raw_name.strip()
        if name in out:
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure file has duplicate ROI name "
                f"{name!r}: {path}"
            )

        operation = item.get("operation", "union")
        if not isinstance(operation, str) or operation not in {
            "union",
            "intersection",
            "subtract",
            "xor",
        }:
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure entry {index} has an invalid "
                f"'operation': {path}"
            )
        sources = item.get("source_structures")
        if (
            not isinstance(sources, list)
            or not sources
            or any(
                not isinstance(source, str) or not source.strip()
                for source in sources
            )
        ):
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure entry {index} has invalid "
                f"'source_structures'; expected a non-empty list of names: {path}"
            )
        if operation == "subtract" and len(sources) < 2:
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure entry {index} uses 'subtract' but "
                f"declares fewer than two source structures: {path}"
            )

        margin = item.get("margin")
        if margin is not None:
            if isinstance(margin, bool):
                valid_margin = False
            elif isinstance(margin, (int, float)):
                valid_margin = bool(np.isfinite(float(margin)))
            elif isinstance(margin, dict):
                allowed_margin_fields = {
                    "anterior_mm",
                    "posterior_mm",
                    "left_mm",
                    "right_mm",
                    "superior_mm",
                    "inferior_mm",
                    "uniform_mm",
                }
                valid_margin = bool(margin) and set(margin) <= allowed_margin_fields
                valid_margin = valid_margin and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and np.isfinite(float(value))
                    for value in margin.values()
                )
            else:
                valid_margin = False
            if not valid_margin:
                raise RadiomicsCourseExtractionError(
                    f"Configured custom structure entry {index} has an invalid "
                    f"'margin': {path}"
                )

        description = item.get("description")
        if description is not None and not isinstance(description, str):
            raise RadiomicsCourseExtractionError(
                f"Configured custom structure entry {index} has an invalid "
                f"'description': {path}"
            )
        out.add(name)
    return out


def _list_roi_names_dicom(rs_path: Path) -> list[str]:
    rs_path = Path(rs_path)
    if not rs_path.exists():
        return []
    try:
        ds = pydicom.dcmread(str(rs_path), stop_before_pixels=True, force=True)
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"Failed to read RTSTRUCT identities from {rs_path}: {exc}"
        ) from exc
    out: list[str] = []
    for roi in getattr(ds, "StructureSetROISequence", []) or []:
        name = str(getattr(roi, "ROIName", "") or "").strip()
        if name:
            out.append(name)
    if not out:
        raise RadiomicsCourseExtractionError(
            f"RTSTRUCT contains no named ROI identities: {rs_path}"
        )
    return out


def _rtstruct_masks(
    dicom_series_path: Path,
    rs_path: Path,
    *,
    skip_rois: Optional[set[str]] = None,
    expected_rois: Optional[List[str]] = None,
    best_effort: bool = False,
    failure_outcomes: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, np.ndarray]:
    """Convert RTSTRUCT ROIs to boolean masks under an explicit source policy.

    Required sources fail closed when an advertised ROI cannot be read or has a
    degenerate mask. For best-effort TotalSegmentator sources, the same failure
    is returned in ``failure_outcomes`` and extraction continues for other ROIs.
    """
    normalized_skips = {
        ''.join(ch for ch in str(name).lower() if ch.isalnum())
        for name in (skip_rois or set())
    }

    def _record_or_raise(name: str, detail: str, *, failure_kind: str = "extraction_error") -> None:
        if not best_effort:
            raise RadiomicsCourseExtractionError(detail)
        if failure_outcomes is not None:
            failure_outcomes.append(
                {
                    "roi_name": str(name),
                    "status": "failed",
                    "failure_kind": failure_kind,
                    "reason": detail,
                }
            )
        logger.warning("Best-effort radiomics ROI %s was not extracted: %s", name, detail)
    if best_effort and failure_outcomes is None:
        raise ValueError("best-effort RTSTRUCT extraction requires a failure_outcomes sink")
    try:
        from rt_utils import RTStructBuilder
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"RTSTRUCT mask conversion is unavailable for {rs_path}: {exc}"
        ) from exc

    try:
        rt = RTStructBuilder.create_from(
            dicom_series_path=str(dicom_series_path), rt_struct_path=str(rs_path)
        )
    except Exception as exc:
        if best_effort:
            try:
                advertised_rois = list(_list_roi_names_dicom(rs_path))
            except Exception:
                advertised_rois = []
            if advertised_rois:
                for advertised_roi in advertised_rois:
                    _record_or_raise(
                        advertised_roi,
                        f"Failed to construct RTSTRUCT reader for {rs_path}: {exc}",
                    )
                return {}
        raise RadiomicsCourseExtractionError(
            f"Failed to construct RTSTRUCT reader for {rs_path}: {exc}"
        ) from exc

    try:
        available_roi_names = list(rt.get_roi_names())
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"Failed to read expected ROI names from {rs_path}: {exc}"
        ) from exc

    roi_names = available_roi_names
    if expected_rois is not None:
        if len(available_roi_names) != len(set(available_roi_names)):
            raise RadiomicsCourseExtractionError(
                f"Required RTSTRUCT has duplicate ROI identities: {rs_path}"
            )
        missing = sorted(set(expected_rois) - set(available_roi_names))
        unexpected = sorted(set(available_roi_names) - set(expected_rois))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise RadiomicsCourseExtractionError(
                f"Required RTSTRUCT inventory does not exactly match the "
                f"prespecified expectation ({'; '.join(details)}): {rs_path}"
            )
        roi_names = list(expected_rois)

    out: Dict[str, np.ndarray] = {}
    for name in roi_names:
        norm_name = ''.join(ch for ch in str(name).lower() if ch.isalnum())
        if norm_name in normalized_skips:
            logger.debug("Skipping configured radiomics ROI %s in %s", name, rs_path)
            continue
        try:
            if hasattr(rt, 'get_mask_for_roi'):
                mask = rt.get_mask_for_roi(name)
            elif hasattr(rt, 'get_roi_mask'):
                mask = rt.get_roi_mask(name)
            elif hasattr(rt, 'get_roi_mask_by_name'):
                mask = rt.get_roi_mask_by_name(name)
            else:
                raise AttributeError("RTSTRUCT reader exposes no ROI mask method")
        except Exception as exc:
            _record_or_raise(
                name,
                f"Expected ROI {name!r} in {rs_path} could not be read: {exc}",
            )
            continue
        if mask is None:
            _record_or_raise(
                name,
                f"Expected ROI {name!r} in {rs_path} did not provide a mask",
            )
            continue
        try:
            mask_bool = np.asarray(mask).astype(bool)
        except Exception as exc:
            _record_or_raise(
                name,
                f"Expected ROI {name!r} in {rs_path} could not be converted to a mask: {exc}",
            )
            continue
        if not mask_bool.any():
            _record_or_raise(
                name,
                f"Expected ROI {name!r} in {rs_path} produced an "
                f"{'empty mask' if best_effort else 'empty required mask'}",
                failure_kind="degenerate_mask",
            )
            continue
        out[str(name)] = mask_bool
    return out


def _check_radiomics_contract_scope(
    contract,
    course_dir: Path,
    *,
    allow_all_series_temp: bool,
) -> None:
    """Keep the all-series contract exception out of course-level radiomics."""
    is_temp = contract.data.get("scope") == ALL_SERIES_RADIOMICS_TEMP_SCOPE
    if is_temp:
        if not allow_all_series_temp or ".all_series_radiomics" not in course_dir.parts:
            raise RadiomicsCourseExtractionError(
                "all-series temporary contract is restricted to the all-series dispatcher"
            )
    elif allow_all_series_temp:
        raise RadiomicsCourseExtractionError(
            "all-series dispatcher requires an all-series temporary contract"
        )


def radiomics_for_course(
    config: PipelineConfig,
    course_dir: Path,
    custom_structures_config: Optional[Path] = None,
    use_cropped: bool = False,
    *,
    allow_all_series_temp: bool = False,
) -> RadiomicsCourseOutcome:
    """Run pyradiomics on CT course with manual RS, RS_auto, and custom structures if present."""

    contract = load_course_contract(course_dir)
    _check_radiomics_contract_scope(
        contract,
        Path(course_dir),
        allow_all_series_temp=allow_all_series_temp,
    )
    course_dirs = build_course_dirs(course_dir)
    contracted_ct_dir = contract.planning_ct_dir
    contracted_rs_manual = (
        contract.authoritative_rtstruct_path
        or course_dir / "metadata" / ".contract-rtstruct-absent"
    )
    out_path = course_dir / 'radiomics_ct.xlsx'

    # Resume-friendly: if output exists, only recompute when required ROIs are missing
    existing_df = None
    if getattr(config, "resume", False) and out_path.exists():
        try:
            import pandas as pd

            existing_df = pd.read_excel(out_path, engine="openpyxl")
            _resume_identity_pairs(existing_df)
        except Exception as exc:
            logger.warning(
                "Invalidating unusable resume workbook for %s: %s", course_dir, exc
            )
            _invalidate_radiomics_outputs(out_path)
            existing_df = None

    # CT radiomics uses masks derived from RTSTRUCT contours paired with the original
    # DICOM CT series. Cropped RTSTRUCTs (RS_auto_cropped.dcm) have been observed to
    # be misregistered in physical space when used with the original CT series
    # (e.g., shifted into air), producing invalid radiomics.
    rs_auto_cropped = course_dir / "RS_auto_cropped.dcm"
    if use_cropped and rs_auto_cropped.exists():
        logger.warning(
            "Ignoring RS_auto_cropped.dcm for radiomics in %s due to known geometric misregistration; "
            "using RS_auto.dcm instead.",
            course_dir,
        )

    ct_files_present = contracted_ct_dir is not None
    img = _load_series_image(contracted_ct_dir) if contracted_ct_dir is not None else None
    if img is None:
        if not ct_files_present:
            logger.info("No CT image for radiomics in %s", course_dir)
            _invalidate_radiomics_outputs(out_path)
            return RadiomicsCourseOutcome.nothing_to_do("CT image is absent")
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"CT series is present but unreadable for radiomics in {course_dir}"
        )
    assert contracted_ct_dir is not None

    extractor = _extractor(config, 'CT')
    if extractor is None:
        try:
            from .radiomics_conda import radiomics_for_course as conda_radiomics_for_course
        except ImportError as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Conda-based radiomics helper unavailable for {course_dir}: {exc}"
            ) from exc
        logger.info("Delegating CT radiomics for %s to conda environment", course_dir)
        conda_out = conda_radiomics_for_course(
            course_dir,
            config,
            custom_structures_config,
            allow_all_series_temp=allow_all_series_temp,
        )
        if conda_out is None:
            _invalidate_radiomics_outputs(out_path)
            return RadiomicsCourseOutcome.nothing_to_do("conda backend found no eligible ROIs")
        return outcome_from_output(conda_out)
    rows: List[Dict] = []
    tasks: List[tuple[str, str, np.ndarray, bool]] = []
    roi_failures: List[Dict[str, str]] = []
    source_counts: Dict[str, Dict[str, int]] = {}
    prepared_failure_pairs: set[tuple[str, str]] = set()

    # Determine which custom ROIs belong to the current source/ROI identity set.
    # A configured skip is an explicit ineligibility decision, not a failed mask.
    configured_skip_rois = {
        str(name) for name in getattr(config, "radiomics_skip_rois", [])
        if isinstance(name, str) and name.strip()
    }
    normalized_skip_rois = {
        ''.join(ch for ch in name.lower() if ch.isalnum()) for name in configured_skip_rois
    }
    desired_custom_bases: set[str] = set()
    configured_custom_path_value = (
        custom_structures_config
        or getattr(config, "custom_structures_config", None)
    )
    configured_custom_path: Optional[Path] = None
    if configured_custom_path_value:
        configured_custom_path = Path(configured_custom_path_value)
        if not configured_custom_path.is_file():
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Configured required custom structure file is missing: {configured_custom_path}"
            )
        try:
            desired_custom_bases = {
                name for name in _custom_roi_names_from_config(configured_custom_path)
                if ''.join(ch for ch in name.lower() if ch.isalnum())
                not in normalized_skip_rois
            }
        except RadiomicsCourseExtractionError:
            _invalidate_radiomics_outputs(out_path)
            raise

    # Process every current non-skipped identity before deciding whether a resume
    # workbook is complete. Partial top-ups can silently miss newly added sources.

    # Process standard RTSTRUCTs
    rs_manual_path = contracted_rs_manual
    rs_auto_path_name = "RS_auto.dcm"

    sources = _standard_rtstruct_sources(contract, course_dir)
    for source, rs_path, _roi_names in sources:
        if not rs_path.exists():
            continue
        try:
            source_failures: List[Dict[str, str]] = []
            is_best_effort = not roi_source_is_required(source)
            masks = _rtstruct_masks(
                contracted_ct_dir,
                rs_path,
                skip_rois=configured_skip_rois,
                best_effort=is_best_effort,
                failure_outcomes=source_failures if is_best_effort else None,
            )
            for failure in source_failures:
                failure["source"] = source
                failure_record = {
                    "modality": "CT",
                    "segmentation_source": source,
                    "roi_name": failure["roi_name"],
                    "roi_original_name": failure["roi_name"],
                    "course_dir": str(course_dir),
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "structure_cropped": False,
                    "extraction_status": failure["status"],
                    "extraction_status_detail": failure["reason"],
                    "extraction_failure_kind": failure["failure_kind"],
                }
                rows.append(failure_record)
                prepared_failure_pairs.add((source, failure["roi_name"]))
            roi_failures.extend(source_failures)
            source_counts.setdefault(
                source,
                {"attempted": 0, "extracted": 0, "failed": 0},
            )["failed"] += len(source_failures)
            source_counts[source]["attempted"] += len(source_failures)
        except RadiomicsCourseExtractionError:
            _invalidate_radiomics_outputs(out_path)
            raise
        for roi, mask in masks.items():
            tasks.append((source, roi, mask, mask_is_cropped(mask)))

    # Process custom structures (extract only custom ROIs; avoid duplicating base ROIs in RS_custom)
    rs_custom = course_dir / "RS_custom.dcm"
    want_custom = bool(desired_custom_bases)
    if configured_custom_path is None and not want_custom and rs_custom.exists():
        # Unconfigured legacy fallback only. A configured file defines the exact inventory.
        try:
            base_names = set(_list_roi_names_dicom(rs_custom))
            manual_names = set(_list_roi_names_dicom(rs_manual_path))
            auto_names = set(_list_roi_names_dicom(course_dir / "RS_auto.dcm"))
            inferred = {n for n in (base_names - (manual_names | auto_names)) if n}
            # Strip __partial suffix for base matching.
            desired_custom_bases = {n[:-9] if n.endswith("__partial") else n for n in inferred}
            want_custom = bool(desired_custom_bases)
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to enumerate custom RTSTRUCT identities for {course_dir}: {exc}"
            ) from exc

    if want_custom:
        custom_rebuild_attempted = False
        custom_rebuild_published = False
        try:
            from rt_utils import RTStructBuilder

            rs_manual_path = contracted_rs_manual
            rs_auto_path = course_dir / "RS_auto.dcm"

            custom_is_stale = bool(
                configured_custom_path
                and _is_rs_custom_stale(
                    rs_custom, configured_custom_path, rs_manual_path, rs_auto_path
                )
            )
            if custom_is_stale:
                custom_rebuild_attempted = True
                logger.info("Regenerating RS_custom.dcm for radiomics in %s", course_dir.name)
                from .custom_structures_rtstruct import _quarantine_rejected_rtstruct

                _quarantine_rejected_rtstruct(
                    rs_custom,
                    "RS_custom failed the authoritative currentness check",
                )
                rebuilt = _create_custom_structures_rtstruct(
                    course_dir,
                    configured_custom_path,
                    rs_manual_path,
                    rs_auto_path,
                )
                if rebuilt is None or not Path(rebuilt).is_file():
                    record_rs_custom_resume_decision(
                        course_dir,
                        "failed",
                        "RS_custom replacement could not be published",
                    )
                    raise RadiomicsCourseExtractionError(
                        f"RS_custom rebuild failed for configured ROIs in {course_dir}"
                    )
                rs_custom = Path(rebuilt)
                custom_rebuild_published = True
                record_rs_custom_resume_decision(
                    course_dir,
                    "rebuilt",
                    "rebuilt after the previous RS_custom failed the authoritative currentness check",
                )
            elif configured_custom_path:
                record_rs_custom_resume_decision(
                    course_dir,
                    "reused",
                    "existing RS_custom passed the authoritative currentness check",
                )

            if not rs_custom or not rs_custom.is_file():
                raise RadiomicsCourseExtractionError(
                    f"Required custom RTSTRUCT is missing for configured ROIs in {course_dir}"
                )

            # Resolve every declared ROI (base vs base__partial); omission is fatal.
            available = set(_list_roi_names_dicom(rs_custom))
            wanted_names: list[str] = []
            missing_custom: list[str] = []
            for base in sorted(desired_custom_bases):
                if base in available:
                    wanted_names.append(base)
                elif f"{base}__partial" in available:
                    wanted_names.append(f"{base}__partial")
                else:
                    missing_custom.append(base)
            if missing_custom:
                raise RadiomicsCourseExtractionError(
                    f"Required configured custom ROI(s) missing from {rs_custom}: "
                    + ", ".join(missing_custom)
                )

            rt = RTStructBuilder.create_from(
                dicom_series_path=str(contracted_ct_dir),
                rt_struct_path=str(rs_custom),
            )
            for roi_name in wanted_names:
                try:
                    mask = rt.get_roi_mask_by_name(roi_name)
                except Exception as exc:
                    raise RadiomicsCourseExtractionError(
                        f"Expected custom ROI {roi_name!r} in {rs_custom} could not be read: {exc}"
                    ) from exc
                if mask is None:
                    raise RadiomicsCourseExtractionError(
                        f"Expected custom ROI {roi_name!r} in {rs_custom} did not provide a mask"
                    )
                mask_bool = np.asarray(mask).astype(bool)
                if not mask_bool.any():
                    raise RadiomicsCourseExtractionError(
                        f"Expected custom ROI {roi_name!r} in {rs_custom} produced an empty required mask"
                    )
                tasks.append(("Custom", roi_name, mask_bool, mask_is_cropped(mask_bool)))
        except Exception as exc:
            if custom_rebuild_attempted and not custom_rebuild_published:
                record_rs_custom_resume_decision(
                    course_dir,
                    "failed",
                    f"RS_custom rebuild raised {type(exc).__name__}: {exc}",
                )
            _invalidate_radiomics_outputs(out_path)
            if isinstance(exc, RadiomicsCourseExtractionError):
                raise
            raise RadiomicsCourseExtractionError(
                f"Failed to prepare custom RTSTRUCT masks for {course_dir}: {exc}"
            ) from exc

    # Include current course outputs and explicitly selected custom models. Model
    # definitions present only under custom_models_root are dormant, not required.
    try:
        custom_model_expected_rois = validate_custom_model_output_inventory(
            course_dir,
            getattr(config, "custom_model_names", None),
            getattr(config, "custom_models_root", None),
        )
        custom_model_outputs = list_custom_model_outputs(course_dir)
    except Exception as exc:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Custom model output scan failed for {course_dir}: {exc}"
        ) from exc
    for model_name, model_course_dir in custom_model_outputs:
        rs_path = model_course_dir / "rtstruct.dcm"
        if not rs_path.exists():
            continue
        try:
            masks = _rtstruct_masks(
                contracted_ct_dir,
                rs_path,
                skip_rois=configured_skip_rois,
                expected_rois=custom_model_expected_rois[model_name],
            )
        except RadiomicsCourseExtractionError:
            _invalidate_radiomics_outputs(out_path)
            raise
        for roi, mask in masks.items():
            tasks.append((f"CustomModel:{model_name}", roi, mask, mask_is_cropped(mask)))

    if existing_df is not None:
        expected_pairs = {(source, roi) for source, roi, _mask, _cropped in tasks}
        expected_pairs |= prepared_failure_pairs
        try:
            existing_pairs = _resume_identity_pairs(existing_df)
        except ValueError as exc:
            logger.warning("Invalidating unusable resume workbook for %s: %s", course_dir, exc)
            existing_pairs = set()
        missing_required = sorted(
            (source, roi)
            for source, roi in (expected_pairs - existing_pairs)
            if roi_source_is_required(source)
        )
        if missing_required:
            _invalidate_radiomics_outputs(out_path)
            missing_text = ", ".join(
                f"{source}/{roi}" for source, roi in missing_required
            )
            raise RadiomicsCourseExtractionError(
                f"Persisted radiomics output is missing required ROI outcome(s): {missing_text}"
            )
        if expected_pairs and existing_pairs == expected_pairs:
            return outcome_from_output(
                out_path,
                required_by_identity={
                    (source, roi): roi_source_is_required(source)
                    for source, roi in expected_pairs
                },
            )
        logger.warning(
            "Invalidating incomplete resume workbook for %s: expected %d source/ROI "
            "identities, found %d",
            course_dir,
            len(expected_pairs),
            len(existing_pairs),
        )
        _invalidate_radiomics_outputs(out_path)
        existing_df = None
    def _do_ct_task(t):
        source, roi, mask, cropped = t
        try:
            # Create fresh extractor instance for each task to avoid threading issues
            min_voxels, max_voxels_full = _derive_voxel_limits(config)
            voxel_count = int(np.asarray(mask).astype(bool).sum())
            if voxel_count < min_voxels:
                return {
                    "__status__": "below_min_voxels",
                    "extraction_status": "below_minimum_voxels",
                    "extraction_status_detail": (
                        f"ROI contains {voxel_count} voxels; configured minimum is {min_voxels}"
                    ),
                    "extraction_failure_kind": "degenerate_mask",
                    "segmentation_source": source,
                    "roi_name": roi,
                    "roi_original_name": roi,
                    "modality": "CT",
                    "course_dir": str(course_dir),
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "structure_cropped": bool(cropped),
                    "voxel_count": voxel_count,
                }

            # Large-ROI detection should reflect the *effective* workload after resampling.
            # CT radiomics typically resamples to ~1mm isotropic, so thick-slice CT can
            # inflate voxel counts substantially. Treat BODY as large unconditionally.
            try:
                spacing = tuple(float(x) for x in img.GetSpacing())
            except Exception:
                spacing = (1.0, 1.0, 1.0)
            native_voxel_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
            physical_volume_mm3 = float(voxel_count) * max(1e-9, native_voxel_mm3)
            estimated_voxels = physical_volume_mm3  # ~ voxels at 1mm isotropic
            is_body = str(roi).strip().lower().startswith("body")
            use_large = is_body or (estimated_voxels > float(max_voxels_full))
            ext = _extractor_large_roi(config, "CT") if use_large else _extractor(config, 'CT')
            if ext is None:
                raise RuntimeError(f"No radiomics extractor available for {source}/{roi}")
            m_img = _mask_from_array_like(img, mask)
            res = ext.execute(img, m_img)
            rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
            display_roi = roi if (not cropped or roi.endswith("__partial")) else f"{roi}__partial"
            rec.update({
                'modality': 'CT',
                'segmentation_source': source,
                'roi_name': display_roi,
                'roi_original_name': roi,
                'course_dir': str(course_dir),
                'patient_id': course_dir.parent.name,
                'course_id': course_dir.name,
                'structure_cropped': bool(cropped),
            })
            return rec
        except Exception as e:
            detail = f"Radiomics failed for {source}/{roi}: {e}"
            if not roi_source_is_required(source):
                return {
                    "__status__": "failed",
                    "extraction_status": "failed",
                    "extraction_status_detail": detail,
                    "extraction_failure_kind": "extraction_error",
                    "segmentation_source": source,
                    "roi_name": roi,
                    "roi_original_name": roi,
                    "course_dir": str(course_dir),
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "structure_cropped": bool(cropped),
                }
            raise RuntimeError(detail) from e
    if tasks:
        sequential_env = os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes')
        if sequential_env:
            max_workers = 1
            logger.info("Radiomics sequential mode requested via RTPIPELINE_RADIOMICS_SEQUENTIAL")
        else:
            max_workers = config.effective_workers()
            env_limit = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)
            if env_limit > 0:
                max_workers = min(max_workers, env_limit)
            max_workers = max(1, max_workers)
        logger.info("Running radiomics for %s with up to %d worker(s)", course_dir.name, max_workers)
        results = run_tasks_with_adaptive_workers(
            f"Radiomics CT ({course_dir.name})",
            tasks,
            _do_ct_task,
            max_workers=max_workers,
            logger=logger,
        )
        if len(results) != len(tasks):
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Radiomics course {course_dir} returned {len(results)} ROI outcomes for "
                f"{len(tasks)} attempted tasks"
            )
        for task, rec in zip(tasks, results):
            source, roi, _mask, cropped = task
            status = (rec or {}).get("extraction_status")
            internal_status = (rec or {}).get("__status__")
            if internal_status == "skipped" or status == "declared_skip":
                continue
            counts = source_counts.setdefault(
                source,
                {"attempted": 0, "extracted": 0, "failed": 0},
            )
            counts["attempted"] += 1
            if not rec:
                if roi_source_is_required(source):
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"Radiomics course {course_dir} is incomplete: required ROI "
                        f"{source}/{roi} returned no outcome record"
                    )
                rec = {
                    "modality": "CT",
                    "segmentation_source": source,
                    "roi_name": roi,
                    "roi_original_name": roi,
                    "course_dir": str(course_dir),
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "structure_cropped": bool(cropped),
                    "extraction_status": "failed",
                    "extraction_status_detail": "worker returned no outcome record",
                    "extraction_failure_kind": "extraction_error",
                }
                status = "failed"
            if status not in (None, "success"):
                counts["failed"] += 1
                detail = str(rec.get("extraction_status_detail", "unknown error"))
                if (
                    roi_source_is_required(source)
                    and not extraction_status_is_nonfatal_for_required(status)
                ):
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"Radiomics course {course_dir} is incomplete: required ROI "
                        f"{source}/{roi} failed: {detail}"
                    )
                roi_failures.append(
                    {
                        "roi_name": roi,
                        "source": source,
                        "status": str(status),
                        "failure_kind": str(rec.get("extraction_failure_kind", "extraction_error")),
                        "reason": detail,
                    }
                )
                rows.append({k: v for k, v in rec.items() if not k.startswith("__")})
                continue
            counts["extracted"] += 1
            rows.append(rec)
    if not rows:
        try:
            from .radiomics_conda import (
                _select_usable_rtstruct,
                radiomics_for_course_ct_nifti_fallback,
            )

            if _select_usable_rtstruct(course_dir / "RS_custom.dcm", course_dir / "RS_auto.dcm") is None:
                logger.info(
                    "No usable RS_custom.dcm/RS_auto.dcm rows for %s; trying CT TotalSegmentator NIfTI fallback",
                    course_dir,
                )
                fallback_out = radiomics_for_course_ct_nifti_fallback(
                    course_dir,
                    config,
                    allow_all_series_temp=allow_all_series_temp,
                )
                if fallback_out is not None:
                    return outcome_from_output(fallback_out)
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            if isinstance(exc, RadiomicsCourseExtractionError):
                raise
            raise RadiomicsCourseExtractionError(
                f"CT TotalSegmentator NIfTI fallback failed for {course_dir}: {exc}"
            ) from exc
        _invalidate_radiomics_outputs(out_path)
        return RadiomicsCourseOutcome.nothing_to_do(
            "no eligible radiomics regions",
            roi_counts=source_counts,
        )
    outcome = RadiomicsCourseOutcome.extracted(
        out_path,
        roi_counts=source_counts,
        roi_failures=roi_failures,
        detail=(
            f"extracted {sum(values['extracted'] for values in source_counts.values())} "
            f"of {sum(values['attempted'] for values in source_counts.values())} "
            "attempted ROIs"
        ),
    )
    diagnostics = course_diagnostic_columns(outcome)
    for row in rows:
        row.update(diagnostics)
    try:
        import pandas as pd
        df_new = pd.DataFrame(rows)
        if existing_df is not None and out_path.exists():
            # Append-only update (resume top-up): keep original column order and avoid duplicates.
            try:
                template_cols = list(existing_df.columns)
                for col in template_cols:
                    if col not in df_new.columns:
                        df_new[col] = np.nan
                df_new = df_new.loc[:, template_cols]
                df = pd.concat([existing_df, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=["segmentation_source", "roi_original_name", "patient_id", "course_id"], keep="last")
            except Exception:
                df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df = df_new
        out = out_path
        _write_excel_atomic(df, out)
        # Also save Parquet for faster aggregation (10-50x I/O speedup)
        parquet_path = out_path.with_suffix('.parquet')
        tmp_parquet = parquet_path.with_suffix('.parquet.tmp')
        try:
            df.to_parquet(tmp_parquet, index=False, engine='pyarrow')
            tmp_parquet.replace(parquet_path)
            logger.debug("Saved Parquet: %s", parquet_path)
        except Exception as parquet_err:
            # Retry via an Excel round-trip to coerce non-scalar objects to strings.
            try:
                df_roundtrip = pd.read_excel(out_path, engine="openpyxl")
                df_roundtrip.to_parquet(tmp_parquet, index=False, engine="pyarrow")
                tmp_parquet.replace(parquet_path)
                logger.debug("Saved Parquet (round-trip): %s", parquet_path)
            except Exception as parquet_err2:
                _remove_artifact_strict(
                    tmp_parquet, context="cleaning a failed temporary Parquet write"
                )
                # Parquet is optional only when no stale or partial sidecar survives.
                _remove_artifact_strict(
                    parquet_path, context="invalidating a failed Parquet refresh"
                )
                logger.debug("Parquet save failed (non-critical): %s (retry: %s)", parquet_err, parquet_err2)
        return outcome
    except Exception as e:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Failed to write CT radiomics for {course_dir}: {e}"
        ) from e


@dataclass
class MRSeries:
    patient_id: str
    series_uid: str
    dir: Path


def _strip_nii_name(nifti_path: Path) -> str:
    name = nifti_path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return nifti_path.stem


# Must match the segmentation_source written onto MR rows below, so that
# roi_source_is_required() adjudicates the same source name the table records.
MR_AUTO_SOURCE = "AutoTS_total_mr"


def _collect_total_mr_masks(
    series_dir: Path,
    seg_dir: Path,
    failure_outcomes: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    if not seg_dir.exists():
        return masks
    candidates = sorted(seg_dir.glob("*--total_mr.dcm"))
    for rtstruct_path in candidates:
        masks.update(_rtstruct_masks(series_dir, rtstruct_path))
        if masks:
            return masks
    # These masks come from the TotalSegmentator MR model, which emits its whole
    # structure list regardless of field of view. A pelvic MR legitimately yields
    # empty brain/lung/liver masks. roi_source_is_required() already classifies
    # this source (AutoTS_total_mr) as not required, so an empty or unreadable
    # mask is recorded and skipped rather than aborting the course.
    for mask_path in sorted(seg_dir.glob("total_mr--*.nii*")):
        try:
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)
        except Exception as exc:
            if roi_source_is_required(MR_AUTO_SOURCE):
                raise RadiomicsCourseExtractionError(
                    f"Required MR mask is unreadable: {mask_path}: {exc}"
                ) from exc
            if failure_outcomes is not None:
                failure_outcomes.append(
                    {"roi_name": mask_path.name, "source": MR_AUTO_SOURCE,
                     "status": "failed", "detail": f"unreadable: {exc}"}
                )
            continue
        arr = np.moveaxis(arr, 0, -1)
        mask = arr > 0
        if not mask.any():
            if roi_source_is_required(MR_AUTO_SOURCE):
                raise RadiomicsCourseExtractionError(
                    f"Required MR mask is empty: {mask_path}"
                )
            if failure_outcomes is not None:
                failure_outcomes.append(
                    {"roi_name": mask_path.name, "source": MR_AUTO_SOURCE,
                     "status": "below_minimum_voxels", "detail": "mask is empty"}
                )
            continue
        name = mask_path.name
        if name.endswith('.nii.gz'):
            name = name[:-7]
        elif name.endswith('.nii'):
            name = name[:-4]
        if name.startswith('total_mr--'):
            name = name[len('total_mr--'):]
        masks[name] = mask
    return masks


def radiomics_for_course_mr(config: PipelineConfig, course) -> Optional[Path]:
    course_dirs = course.dirs if hasattr(course, 'dirs') else build_course_dirs(Path(course))
    mr_root = course_dirs.dicom_mr
    out_path = mr_root / 'radiomics_mr.xlsx'
    mr_required = _mr_radiomics_required(config)
    # Non-fatal outcomes for the non-required auto MR source, recorded rather
    # than raised so one empty out-of-FOV mask cannot void the whole course.
    mr_failures: List[Dict[str, str]] = []
    configured_params = getattr(config, 'radiomics_params_file_mr', None)
    if configured_params is not None and not Path(configured_params).exists():
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Configured required MR radiomics parameter path is missing: {configured_params}"
        )
    if not mr_root.exists():
        _invalidate_radiomics_outputs(out_path)
        if mr_required:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but no MR directory exists for {course_dirs.root}"
            )
        return None

    # Check if we need conda fallback (NumPy 2.x)
    import numpy as np
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version[0] >= 2:
        # Delegate to conda-based MR radiomics
        try:
            from .radiomics_conda import radiomics_for_course_mr as conda_radiomics_for_course_mr
        except ImportError as exc:
            _invalidate_radiomics_outputs(out_path)
            if mr_required:
                raise RadiomicsCourseExtractionError(
                    f"Required conda-based MR radiomics helper is unavailable: {exc}"
                ) from exc
            logger.warning("Optional conda-based MR radiomics helper unavailable: %s", exc)
            return None
        logger.info("Delegating MR radiomics for %s to conda environment", course_dirs.root)
        result = conda_radiomics_for_course_mr(course_dirs.root, config)
        if result is None and mr_required:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but produced no eligible output for {course_dirs.root}"
            )
        return result

    rows: List[Dict[str, object]] = []
    for series_root in sorted(p for p in mr_root.iterdir() if p.is_dir()):
        nifti_dir = series_root / 'NIFTI'
        dicom_dir = series_root / 'DICOM'
        if not dicom_dir.exists():
            dicom_dir = series_root
        seg_dir = series_root / 'Segmentation_TotalSegmentator'
        if not nifti_dir.exists() or not seg_dir.exists():
            continue
        if not any(dicom_dir.rglob('*.dcm')):
            logger.debug("Skipping MR series %s (no DICOM slices)", series_root)
            continue
        meta_files = sorted(nifti_dir.glob('*.metadata.json'))
        if not meta_files:
            continue
        try:
            data = json.loads(meta_files[0].read_text(encoding='utf-8'))
        except Exception:
            continue
        if str(data.get('modality', '')).upper() != 'MR':
            continue
        nifti_path = Path(data.get('nifti_path') or '')
        source_dir = Path(data.get('source_directory') or dicom_dir)
        if not source_dir.exists():
            source_dir = dicom_dir
        if not nifti_path.exists() or not source_dir.exists():
            continue
        series_uid = str(data.get('series_instance_uid') or source_dir.name)
        weighting = _infer_mr_weighting(source_dir, series_uid)
        normalize_override = True if weighting == 'T1' else False if weighting == 'T2' else False
        extractor = _extractor(config, 'MR', normalize_override=normalize_override)
        if extractor is None:
            continue
        img = _load_series_image(source_dir, series_uid)
        if img is None:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"MR series is present but unreadable for radiomics: {source_dir}"
            )
        base_name = _strip_nii_name(nifti_path)
        masks = _collect_total_mr_masks(source_dir, seg_dir, mr_failures)
        if not masks:
            continue
        for roi_name, mask in masks.items():
            try:
                m_img = _mask_from_array_like(img, mask)
                res = extractor.execute(img, m_img)
                rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                rec.update({
                    'patient_id': getattr(course, 'patient_id', course_dirs.root.parent.name),
                    'course_id': getattr(course, 'course_id', course_dirs.root.name),
                    'modality': 'MR',
                    'segmentation_source': 'AutoTS_total_mr',
                    'roi_name': roi_name,
                    'series_dir': str(source_dir),
                    'series_uid': series_uid,
                    'nifti_path': str(nifti_path),
                })
                rows.append(rec)
            except Exception as exc:
                if roi_source_is_required(MR_AUTO_SOURCE):
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"MR radiomics failed for required ROI {series_uid}/{roi_name}: {exc}"
                    ) from exc
                mr_failures.append(
                    {"roi_name": roi_name, "source": MR_AUTO_SOURCE,
                     "status": "failed", "detail": str(exc)[:200]}
                )
    if not rows:
        _invalidate_radiomics_outputs(out_path)
        if mr_required:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but produced no eligible rows for {course_dirs.root}"
            )
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        _write_excel_atomic(df, out_path)
        # Also save Parquet for faster aggregation, without exposing partial bytes.
        parquet_path = out_path.with_suffix('.parquet')
        tmp_parquet = parquet_path.with_suffix('.parquet.tmp')
        try:
            df.to_parquet(tmp_parquet, index=False, engine='pyarrow')
            tmp_parquet.replace(parquet_path)
            logger.debug("Saved MR Parquet: %s", parquet_path)
        except Exception as parquet_err:
            _remove_artifact_strict(
                tmp_parquet, context="cleaning a failed temporary MR Parquet write"
            )
            _remove_artifact_strict(
                parquet_path, context="invalidating a failed MR Parquet refresh"
            )
            logger.debug("MR Parquet save failed (non-critical): %s", parquet_err)
        return out_path
    except Exception as exc:
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Failed to write MR radiomics for {course_dirs.root}: {exc}"
        ) from exc


def _find_mr_manual_rs(dicom_root: Path, patient_id: str, mr_for_uid: str) -> List[Path]:
    rs_list: List[Path] = []
    # Already filters by PatientID below; scope the walk to that patient's dir so
    # this does not stat the entire DICOM root for every MR series.
    for base, _, files in _scoped_walk(dicom_root, [patient_id]):
        for fn in files:
            if not fn.startswith('RS') or not fn.lower().endswith('.dcm'):
                continue
            p = Path(base) / fn
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            except Exception:
                continue
            if str(getattr(ds, 'PatientID', '')) != str(patient_id):
                continue
            try:
                rs_for = str(getattr(ds, 'FrameOfReferenceUID', '') or getattr(ds, (0x3006, 0x0024), ''))
            except Exception:
                rs_for = ''
            if rs_for and rs_for == mr_for_uid:
                rs_list.append(p)
    return rs_list


def _mr_series_for_uid(series_dir: Path, series_uid: str) -> Optional[str]:
    try:
        # Grab FrameOfReferenceUID from one MR slice in that series
        for base, _, files in os.walk(series_dir):
            for fn in files:
                p = Path(base) / fn
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                except Exception:
                    continue
                if str(getattr(ds, 'Modality', '')) != 'MR':
                    continue
                if str(getattr(ds, 'SeriesInstanceUID', '')) != series_uid:
                    continue
                return str(getattr(ds, 'FrameOfReferenceUID', ''))
    except Exception:
        pass
    return None


def _infer_mr_weighting(series_dir: Path, series_uid: str) -> Optional[str]:
    """Heuristic detection of MR weighting (T1 vs T2) from DICOM headers.
    Returns 'T1', 'T2', or None.
    """
    keys = ('SeriesDescription', 'ProtocolName', 'SequenceName')
    try:
        for base, _, files in os.walk(series_dir):
            for fn in files:
                p = Path(base) / fn
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                except Exception:
                    continue
                if str(getattr(ds, 'Modality', '')) != 'MR':
                    continue
                if str(getattr(ds, 'SeriesInstanceUID', '')) != series_uid:
                    continue
                hay = ' '.join(str(getattr(ds, k, '') or '') for k in keys).lower()
                if 't2' in hay or 'flair' in hay:
                    return 'T2'
                if 't1' in hay:
                    return 'T1'
                # Also check ImageType for 'T1' or 'T2'
                try:
                    it = ds.ImageType
                    it_s = ' '.join([str(x) for x in it]).lower() if it else ''
                    if 't2' in it_s:
                        return 'T2'
                    if 't1' in it_s:
                        return 'T1'
                except Exception:
                    pass
                # First matching slice is enough
                return None
    except Exception:
        pass
    return None


def radiomics_for_mr_series(config: PipelineConfig, series: MRSeries) -> Optional[Path]:
    # Determine weighting to toggle normalization: T2 -> False, T1 -> True, else default False
    wt = _infer_mr_weighting(series.dir, series.series_uid)
    normalize_override = True if wt == 'T1' else False if wt == 'T2' else False
    extractor = _extractor(config, 'MR', normalize_override=normalize_override)
    if extractor is None:
        return None
    img = _load_series_image(series.dir, series.series_uid)
    if img is None:
        logger.info("No MR image for radiomics in %s", series.dir)
        return None
    out_root = config.output_root / series.patient_id / f"MR_{series.series_uid}"
    out_feat = out_root / 'radiomics_features_MR.xlsx'
    if getattr(config, 'resume', False) and out_feat.exists():
        return out_feat
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    # Manual MR RS (if any)
    for_uid = _mr_series_for_uid(series.dir, series.series_uid) or ''
    for rs in _find_mr_manual_rs(config.dicom_root, series.patient_id, for_uid):
        masks = _rtstruct_masks(series.dir, rs)
        for roi, mask in masks.items():
            try:
                m_img = _mask_from_array_like(img, mask)
                res = extractor.execute(img, m_img)
                rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                rec.update({
                    'modality': 'MR',
                    'segmentation_source': 'Manual',
                    'roi_name': roi,
                    'series_dir': str(series.dir),
                    'series_uid': series.series_uid,
                })
                rows.append(rec)
            except Exception as e:
                logger.debug("Radiomics MR manual failed for %s: %s", roi, e)
                continue
    # Auto total_mr segmentation (DICOM-SEG/NIfTI) if present
    try:
        from .auto_rtstruct import _load_seg_dicom, _load_seg_nifti  # type: ignore
    except Exception:
        _load_seg_dicom = _load_seg_nifti = None  # type: ignore
    # DICOM-SEG
    if _load_seg_dicom is not None:
        seg_dicom = out_root / 'TotalSegmentator_total_mr_DICOM' / 'segmentations.dcm'
        if seg_dicom.exists():
            seg_img, label_map = _load_seg_dicom(seg_dicom)
            if seg_img is not None:
                seg_img = _resample_to_reference(seg_img, img, nn=True)
                arr = sitk.GetArrayFromImage(seg_img)
                labels = [int(v) for v in np.unique(arr) if int(v) != 0]

                def _do_lab(lab: int) -> Optional[Dict[str, Any]]:
                    try:
                        mask = (arr == lab)
                        if not mask.any():
                            return None
                        ext = _extractor(config, 'MR', normalize_override=normalize_override)
                        m_img = _mask_from_array_like(img, mask)
                        res = ext.execute(img, m_img)
                        rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                        rec.update({
                            'modality': 'MR',
                            'segmentation_source': 'AutoTS_total_mr',
                            'roi_name': label_map.get(lab, f'Segment_{lab}'),
                            'series_dir': str(series.dir),
                            'series_uid': series.series_uid,
                        })
                        return rec
                    except Exception as e:
                        logger.debug("Radiomics MR total_mr failed for label %s: %s", lab, e)
                        return None

                dicom_results = run_tasks_with_adaptive_workers(
                    "Radiomics MR (TotalSegmentator DICOM)",
                    labels,
                    _do_lab,
                    max_workers=config.effective_workers(),
                    logger=logger,
                )
                for rec in dicom_results:
                    if rec:
                        rows.append(rec)
    # NIfTI fallback
    if _load_seg_nifti is not None:
        seg_nifti_dir = out_root / 'TotalSegmentator_total_mr_NIFTI'
        seg_img, label_map = _load_seg_nifti(seg_nifti_dir)
        if seg_img is not None:
            seg_img = _resample_to_reference(seg_img, img, nn=True)
            arr = sitk.GetArrayFromImage(seg_img)
            labels = [int(v) for v in np.unique(arr) if int(v) != 0]

            def _do_lab_nifti(lab: int) -> Optional[Dict[str, Any]]:
                try:
                    mask = (arr == lab)
                    if not mask.any():
                        return None
                    ext = _extractor(config, 'MR', normalize_override=normalize_override)
                    m_img = _mask_from_array_like(img, mask)
                    res = ext.execute(img, m_img)
                    rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                    rec.update({
                        'modality': 'MR',
                        'segmentation_source': 'AutoTS_total_mr',
                        'roi_name': label_map.get(lab, f'Segment_{lab}'),
                        'series_dir': str(series.dir),
                        'series_uid': series.series_uid,
                    })
                    return rec
                except Exception as e:
                    logger.debug("Radiomics MR total_mr (NIfTI) failed for label %s: %s", lab, e)
                    return None

            nifti_results = run_tasks_with_adaptive_workers(
                "Radiomics MR (TotalSegmentator NIfTI)",
                labels,
                _do_lab_nifti,
                max_workers=config.effective_workers(),
                logger=logger,
            )
            for rec in nifti_results:
                if rec:
                    rows.append(rec)
    if not rows:
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        out = out_feat
        df.to_excel(out, index=False)
        return out
    except Exception as e:
        logger.warning("Failed to write MR radiomics: %s", e)
        return None


def run_radiomics(config: PipelineConfig, courses: List["object"], custom_structures_config: Optional[Path] = None) -> None:
    """Top-level orchestrator: per-course CT radiomics and per-series MR radiomics.
    'courses' elements are CourseOutput-like with patient_id, course_key, dirs.root.
    """
    _apply_radiomics_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))

    aggregate_path = config.output_root / 'Data' / 'radiomics_all.xlsx'

    # A requested radiomics stage must not preserve outputs from an older run when
    # neither the native nor isolated backend can execute.
    can_use_radiomics = _have_pyradiomics()
    if not can_use_radiomics:
        for course in courses:
            course_root = Path(course.dirs.root)
            _invalidate_radiomics_outputs(course_root / "radiomics_ct.xlsx")
            for mr_output in course_root.rglob("radiomics_features_MR.xlsx"):
                _invalidate_radiomics_outputs(mr_output)
        _invalidate_radiomics_outputs(aggregate_path)
        raise RuntimeError("PyRadiomics is unavailable; requested radiomics extraction failed")

    # Check if enhanced parallel radiomics processing is enabled
    try:
        from .radiomics_parallel import is_parallel_radiomics_enabled, parallel_radiomics_for_course
        use_parallel_impl = is_parallel_radiomics_enabled()
    except ImportError:
        use_parallel_impl = False

    # Determine if we should use cropped volumes
    use_cropped = getattr(config, 'ct_cropping_use_for_radiomics', True)
    env_course_limit = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)

    if use_parallel_impl:
        logger.info("Using enhanced parallel radiomics implementation")

        per_course_worker_cap = config.effective_workers()
        if env_course_limit > 0:
            per_course_worker_cap = min(per_course_worker_cap, env_course_limit)

        # Use the new parallel implementation for each course
        def _parallel_radiomics_wrapper(course):
            return parallel_radiomics_for_course(
                config,
                course.dirs.root,
                custom_structures_config,
                max_workers=per_course_worker_cap,
                use_cropped=use_cropped,
            )

        radiomics_func = _parallel_radiomics_wrapper
        max_course_workers = per_course_worker_cap
    else:
        # Use traditional implementation
        radiomics_func = lambda course: radiomics_for_course(config, course.dirs.root, custom_structures_config, use_cropped=use_cropped)

        # CT per course (parallel, but limited for memory safety)
        if os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes'):
            max_course_workers = 1
            logger.info("Using sequential course processing for radiomics (RTPIPELINE_RADIOMICS_SEQUENTIAL set)")
        else:
            max_course_workers = config.effective_workers()
            max_course_workers = max(1, max_course_workers)

    if env_course_limit > 0:
        max_course_workers = min(max_course_workers, env_course_limit)
    max_course_workers = max(1, max_course_workers)
    logger.info("Processing radiomics with up to %d course workers", max_course_workers)

    ct_results = run_tasks_with_adaptive_workers(
        "Radiomics (CT courses)",
        courses,
        radiomics_func,
        max_workers=max_course_workers,
        logger=logger,
        show_progress=True,
    )
    if len(ct_results) != len(courses):
        _invalidate_radiomics_outputs(aggregate_path)
        raise RuntimeError(
            "CT radiomics worker returned an incomplete result vector; "
            f"expected {len(courses)} course outcomes and received {len(ct_results)}"
        )
    failed_courses = [
        str(getattr(getattr(course, "dirs", None), "root", course))
        for course, result in zip(courses, ct_results)
        if result is None
    ]
    if failed_courses:
        _invalidate_radiomics_outputs(aggregate_path)
        raise RuntimeError(
            "CT radiomics failed for course(s); cohort aggregation was not written: "
            + ", ".join(failed_courses)
        )

    mr_required = _mr_radiomics_required(config)
    mr_outputs: List[Path] = []
    for course in courses:
        try:
            mr_output = radiomics_for_course_mr(config, course)
        except Exception as exc:
            if mr_required:
                _invalidate_radiomics_outputs(aggregate_path)
                raise RuntimeError(
                    f"Required MR radiomics failed for course {getattr(course, 'dirs', course)}; "
                    f"cohort aggregation was not written: {exc}"
                ) from exc
            logger.warning(
                "Optional MR radiomics failed for course %s: %s",
                getattr(course, 'dirs', course),
                exc,
            )
        else:
            if mr_output is not None:
                mr_outputs.append(Path(mr_output))
            elif mr_required:
                _invalidate_radiomics_outputs(aggregate_path)
                raise RuntimeError(
                    f"Required MR radiomics produced no workbook for {getattr(course, 'dirs', course)}; "
                    "cohort aggregation was not written"
                )

    # Cohort merge. Every workbook declared EXTRACTED by a course is expected and
    # must be readable; aggregate publication is atomic and invalidated on failure.
    try:
        import pandas as _pd

        out_rows = []
        for course, outcome in zip(courses, ct_results):
            p = getattr(outcome, "output_path", None)
            if p is None:
                continue
            p = Path(p)
            try:
                df = _pd.read_excel(p, engine="openpyxl")
            except Exception as exc:
                raise RadiomicsCourseExtractionError(
                    f"Expected course radiomics workbook is unreadable: {p}: {exc}"
                ) from exc
            if isinstance(outcome, RadiomicsCourseOutcome):
                for column, value in course_diagnostic_columns(outcome).items():
                    df[column] = value
            for position, (column, value) in enumerate((
                ('patient_id', getattr(course, 'patient_id', Path(course.dirs.root).parts[-2])),
                ('course_key', getattr(course, 'course_key', Path(course.dirs.root).name)),
                ('course_dir', str(course.dirs.root)),
            )):
                if column in df.columns:
                    df.pop(column)
                df.insert(position, column, value)
            out_rows.append(df)

        legacy_mr_paths = list(config.output_root.rglob('MR_*/radiomics_features_MR.xlsx'))
        expected_mr_paths = list(dict.fromkeys(mr_outputs + legacy_mr_paths))
        legacy_mr_set = set(legacy_mr_paths)
        for p in expected_mr_paths:
            try:
                df = _pd.read_excel(p, engine="openpyxl")
            except Exception as exc:
                raise RadiomicsCourseExtractionError(
                    f"Expected MR radiomics workbook is unreadable: {p}: {exc}"
                ) from exc
            if p in legacy_mr_set:
                parts = p.parts
                patient_id = parts[-4] if len(parts) > 4 else (parts[0] if parts else 'unknown')
                series_uid = (
                    parts[-2].replace('MR_', '') if len(parts) > 2 and parts[-2].startswith('MR_')
                    else (parts[-2] if len(parts) > 2 else 'unknown')
                )
                for position, (column, value) in enumerate((
                    ('patient_id', patient_id),
                    ('series_uid', series_uid),
                    ('series_dir', str(p.parent)),
                )):
                    if column in df.columns:
                        df.pop(column)
                    df.insert(position, column, value)
            out_rows.append(df)

        if out_rows:
            all_df = _pd.concat(out_rows, ignore_index=True)
            _write_excel_atomic(all_df, aggregate_path)
        else:
            _invalidate_radiomics_outputs(aggregate_path)
    except Exception as exc:
        _invalidate_radiomics_outputs(aggregate_path)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"Failed to publish cohort radiomics workbook {aggregate_path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# B4 + C4 — all-series (non-course) CT radiomics
#
# Course radiomics (run_radiomics / radiomics_for_course / parallel_radiomics_for_course)
# is left byte-identical. This block adds a SEPARATE entry point that radiomics the
# materialized all-series CT series (planning/diagnostic/PET-CT/4DCT-averaged), one
# representative volume per 4DCT study (C4), and writes its own Data/radiomics_all_series.csv.
# It NEVER calls run_radiomics (which runs the per-course MR loop and rewrites the cohort
# merge Data/radiomics_all.xlsx via rglob).
# ---------------------------------------------------------------------------

# Base CT classes radiomic'd as-is. 4DCT (fourdct_ave / fourdct_phase) is handled separately
# by the C4 per-study dedup below.
_ALL_SERIES_RADIOMICS_BASE_CLASSES = frozenset({"planning_ct", "diagnostic_ct", "petct_ct"})
_FOURDCT_RADIOMICS_CLASSES = frozenset({"fourdct_ave", "fourdct_phase"})


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink ``src`` -> ``dst`` (absolute target, for inode/disk safety at scale); copy on failure."""
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
    except OSError:
        pass
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_all_series_auto_rtstruct(input_dir: Path, model: str) -> Optional[Path]:
    """Locate the per-series TotalSegmentator auto-RTSTRUCT written by the all-series stage.

    The all-series segmenter writes it at
    ``<input_dir.parent>/Segmentation_TotalSegmentator/<input_dir.name>/<base>/<base>--<model>.dcm``
    (segmentation._series_artifact_dirs + the ``rt_out`` naming, segmentation.py:919/965). CT radiomics
    binds masks via this RTSTRUCT (radiomics.py:640), so a series without one is skipped, not failed.
    """
    from .segmentation import _series_artifact_dirs  # lazy: avoid any import cycle
    _, seg_root = _series_artifact_dirs(Path(input_dir))
    if not seg_root.exists():
        return None
    for rt in sorted(seg_root.glob(f"*/*--{model}.dcm")):
        if rt.is_file():
            return rt
    return None


def _materialize_temp_course_tree(course_dir: Path, ct_slices_dir: Path, rtstruct_path: Path) -> bool:
    """Build the minimal real on-disk course tree B4 radiomics needs: ``DICOM/CT/`` slices + root
    ``RS_auto.dcm``. Symlinks where possible. Returns False (caller skips the series) if no CT slices."""
    course_dirs = build_course_dirs(course_dir)
    course_dirs.dicom_ct.mkdir(parents=True, exist_ok=True)
    n_slices = 0
    for slice_path in sorted(Path(ct_slices_dir).glob("*.dcm")):
        if slice_path.is_file():
            # The contract validator rejects paths whose symlink targets leave
            # the temporary course tree. Copy the selected source objects so
            # the temporary contract is self-contained and freshness-checkable.
            shutil.copy2(slice_path, course_dirs.dicom_ct / slice_path.name)
            n_slices += 1
    if n_slices == 0:
        return False
    shutil.copy2(Path(rtstruct_path), course_dir / "RS_auto.dcm")
    ct_datasets = []
    for path in sorted(course_dirs.dicom_ct.iterdir()):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "") or "").strip().upper() == "CT":
            ct_datasets.append(dataset)
    series_uids = {
        str(getattr(dataset, "SeriesInstanceUID", "") or "").strip()
        for dataset in ct_datasets
        if str(getattr(dataset, "SeriesInstanceUID", "") or "").strip()
    }
    if not ct_datasets:
        # A consumer-facing temporary tree must never exist without a contract
        # that identifies its selected CT series.
        return False
    if len(series_uids) != 1:
        return False
    rtstruct = pydicom.dcmread(
        str(course_dir / "RS_auto.dcm"), stop_before_pixels=True, force=True
    )
    rtstruct_uid = str(getattr(rtstruct, "SOPInstanceUID", "") or "").strip()
    if not rtstruct_uid:
        return False
    metadata_path = course_dir / "metadata" / "case_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    patient_id = course_dir.parent.name
    course_id = course_dir.name
    metadata_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "course_id": course_id,
                "course_contract": {
                    "version": COURSE_CONTRACT_VERSION,
                    "authority": ALL_SERIES_RADIOMICS_TEMP_AUTHORITY,
                    "scope": ALL_SERIES_RADIOMICS_TEMP_SCOPE,
                    "scope_reason": (
                        "This temporary tree is an explicit all-series exception. It contains one "
                        "materialized CT series and generated RTSTRUCT for series-level radiomics, "
                        "not a treatment course. It cannot be used for course-level output."
                    ),
                    "patient_id": patient_id,
                    "course_id": course_id,
                    "course_key": course_id,
                    "selected_plans": [],
                    "selected_doses": [],
                    "dose_classification": {"classification": "no_doses"},
                    "dvh": build_dvh_decision(0, 0, "no_records_at_all"),
                    "authoritative_rtstruct": {
                        "sop_instance_uid": rtstruct_uid,
                        "path": "RS_auto.dcm",
                        "segmentation_source": AUTO_RTSTRUCT_SOURCE,
                    },
                    "planning_ct": {
                        "status": "referenced",
                        "series_instance_uid": next(iter(series_uids)),
                        "referenced_series_uids": sorted(series_uids),
                        "dicom_dir": "DICOM/CT",
                        "nifti_path": "",
                        "nifti_provenance": None,
                        "dicom_only": True,
                    },
                    "plan_artifact": None,
                    "dose_grid": None,
                    "delivery": {
                        "prescribed_dose_gy": None,
                        "delivered_dose_gy": None,
                        "status": "no_records_at_all",
                        "method": None,
                        "dose_response_field": "delivered_dose_gy",
                        "per_plan": [],
                        "warnings": [],
                        "unresolved_record_plan_uids": [],
                    },
                    "dose_qc": {
                        "status": "pass",
                        "pass": True,
                        "threshold_gy": DEFAULT_MAX_TOTAL_DOSE_GY,
                        "reasons": [],
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        load_course_contract(course_dir)
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"all-series temporary course contract failed validation for {course_dir}: {exc}"
        ) from exc
    return True
def _pick_representative_4dct_phase(phase_rows: List[dict]) -> dict:
    """C4 fallback when a 4DCT study has no averaged reconstruction: prefer the 50% phase (end-exhale,
    conventionally the most stable) identifiable from the series description, else the first phase in
    manifest order (mirrors segmentation._limit_fourdct_to_representative)."""
    for row in phase_rows:
        desc = str(row.get("series_description") or "").lower().replace(" ", "")
        if "50%" in desc:
            return row
    return phase_rows[0]


def _select_all_series_radiomics_rows(
    rows: List[dict],
    has_rtstruct: Optional[Callable[[dict], bool]] = None,
) -> List[Tuple[dict, bool]]:
    """B4 selection + C4 dedup. Returns ``(row, is_4d_phase)`` pairs to radiomic.

    Base CT classes (planning_ct/diagnostic_ct/petct_ct) are taken as-is (``is_4d_phase=False``).
    For 4DCT, only one volume per ``study_uid`` is radiomic'd: the averaged reconstruction if present
    (``is_4d_phase=False``), else one representative phase (``is_4d_phase=True``, excluded from pooling).

    The 4DCT representative is chosen ONLY among volumes that were actually segmented: ``has_rtstruct``
    (default: every row) gates 4DCT eligibility. This matters when segmentation keeps a single 4DCT
    representative (``all_series_fourdct_single_representative``, patient-level, first-ave-else-first-phase):
    without the gate B4's per-study 50%-preference could pick a phase that was never segmented, find no
    RTSTRUCT, and silently drop the whole study. With the gate, B4 picks the segmented volume.

    ``is_quantitative_image_class`` (C3) is enforced as a denylist guard so CBCT can never be radiomic'd
    even if a future edit adds it to the class sets. A missing/empty ``study_uid`` is keyed per-series
    (not collapsed into one bucket) so distinct acquisitions with absent UIDs are not silently lost."""
    if has_rtstruct is None:
        has_rtstruct = lambda _row: True
    selected: List[Tuple[dict, bool]] = []
    fourdct_by_study: Dict[str, Dict[str, List[dict]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cls = str(row.get("image_class") or "")
        if not is_quantitative_image_class(cls):  # C3 denylist guard
            continue
        if cls in _ALL_SERIES_RADIOMICS_BASE_CLASSES:
            selected.append((row, False))
        elif cls in _FOURDCT_RADIOMICS_CLASSES:
            study = str(row.get("study_uid") or "").strip() or f"__nostudy__:{row.get('series_uid') or ''}"
            bucket = fourdct_by_study.setdefault(study, {"ave": [], "phase": []})
            bucket["ave" if cls == "fourdct_ave" else "phase"].append(row)
    for bucket in fourdct_by_study.values():
        aves = [r for r in bucket["ave"] if has_rtstruct(r)]
        if aves:
            selected.append((aves[0], False))   # representative = first segmented ave; all phases dropped
            continue
        phases = [r for r in bucket["phase"] if has_rtstruct(r)]
        if phases:
            selected.append((_pick_representative_4dct_phase(phases), True))  # excluded from pooling
    return selected


def _dispatch_radiomics_for_course(config: PipelineConfig, course_dir: Path) -> Optional[Path]:
    """Mirror the per-course CT dispatch (radiomics.py:1244-1274) on one temp course dir.
    NEVER calls run_radiomics (which runs the MR loop + rewrites the cohort merge radiomics_all.xlsx)."""
    try:
        from .radiomics_parallel import is_parallel_radiomics_enabled, parallel_radiomics_for_course
        use_parallel = is_parallel_radiomics_enabled()
    except ImportError:
        use_parallel = False
    if use_parallel:
        max_workers = max(1, config.effective_workers())
        result = parallel_radiomics_for_course(
            config,
            course_dir,
            None,
            max_workers=max_workers,
            use_cropped=False,
            allow_all_series_temp=True,
        )
    else:
        result = radiomics_for_course(
            config,
            course_dir,
            None,
            use_cropped=False,
            allow_all_series_temp=True,
        )
    return getattr(result, "output_path", result)


def run_radiomics_all_series(config: PipelineConfig, patient_ids: List[str]) -> Optional[Path]:
    """B4+C4: CT radiomics for the all-series (non-course) imaging.

    For every eligible CT-class series in each patient's all_series manifest, a real temporary
    course-shaped tree is materialized (``DICOM/CT/`` slices + ``RS_auto.dcm`` <- the per-series
    TotalSegmentator ``{base}--{task}.dcm``), the per-course CT radiomics worker is run on it, and
    ``radiomics_ct.xlsx`` is read back and tagged with provenance. The temp tree is deleted on series
    completion. Output is aggregated to ``Data/radiomics_all_series.csv`` with columns
    patient_id / series_uid / study_uid / image_class / is_4d_phase / series_dir.

    Course-path artifacts (per-course ``radiomics_ct.xlsx`` and ``Data/radiomics_all.xlsx``) are NOT
    touched — this function never calls run_radiomics and only writes the new CSV.
    """
    import pandas as pd

    # Parity with run_radiomics: honor a configured radiomics thread limit (BLAS oversubscription
    # control) and skip cleanly if no extraction backend (native OR conda) is available, so we don't
    # materialize temp trees only to no-op per series.
    _apply_radiomics_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))
    if not _have_pyradiomics():
        logger.warning("PyRadiomics unavailable (native + conda) - skipping all-series radiomics")
        return None

    output_root = Path(config.output_root)
    requested_patient_ids = sorted({str(patient_id) for patient_id in patient_ids})
    if not requested_patient_ids:
        return None
    data_dir = output_root / "Data"
    out_csv = data_dir / "radiomics_all_series.csv"
    preserved_df = None
    if out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            if "patient_id" not in existing_df.columns:
                raise ValueError("existing CSV lacks required patient_id column")
            preserved_df = existing_df[
                ~existing_df["patient_id"].astype(str).isin(requested_patient_ids)
            ].copy()
        except Exception as exc:
            raise RuntimeError(f"Cannot safely merge existing all-series radiomics CSV: {exc}") from exc
    temp_base = output_root / ".all_series_radiomics"
    all_dfs = []

    def _row_has_rtstruct(row: dict) -> bool:
        """A 4DCT volume is eligible as the per-study representative only if it was actually segmented."""
        od = str(row.get("output_dir") or "")
        if not od:
            return False
        return _find_all_series_auto_rtstruct(Path(od), str(row.get("ts_task") or "total")) is not None

    for patient_id in requested_patient_ids:
        course_dirs = build_course_dirs(output_root / str(patient_id) / "all_series")
        manifest_path = course_dirs.metadata / "series_manifest.json"
        if not manifest_path.exists():
            logger.info("All-series manifest not found for patient %s; skipping radiomics", patient_id)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Unable to read all-series manifest for patient %s: %s", patient_id, exc)
            continue
        rows = manifest.get("series", [])
        if not isinstance(rows, list):
            continue

        for row, is_4d_phase in _select_all_series_radiomics_rows(rows, has_rtstruct=_row_has_rtstruct):
            series_uid = str(row.get("series_uid") or "")
            image_class = str(row.get("image_class") or "")
            study_uid = str(row.get("study_uid") or "")
            model = str(row.get("ts_task") or "total")
            output_dir_text = str(row.get("output_dir") or "")
            if not output_dir_text:
                continue
            input_dir = Path(output_dir_text)
            if not input_dir.exists():
                logger.info("All-series CT dir missing for patient %s series %s; skipping", patient_id, series_uid)
                continue
            rtstruct_path = _find_all_series_auto_rtstruct(input_dir, model)
            if rtstruct_path is None:
                logger.info(
                    "No auto-RTSTRUCT for patient %s series %s (%s); skipping radiomics",
                    patient_id, series_uid, image_class,
                )
                continue

            course_dir = temp_base / str(patient_id) / (series_uid or "series")
            if course_dir.exists():
                shutil.rmtree(course_dir, ignore_errors=True)
            course_dir.mkdir(parents=True, exist_ok=True)
            try:
                if not _materialize_temp_course_tree(course_dir, input_dir, rtstruct_path):
                    logger.info("No CT slices to radiomic for patient %s series %s; skipping", patient_id, series_uid)
                    continue
                out_path = _dispatch_radiomics_for_course(config, course_dir)
                if not out_path or not Path(out_path).exists():
                    continue
                try:
                    df = pd.read_excel(Path(out_path), engine="openpyxl")
                except Exception as exc:
                    logger.warning(
                        "Failed reading all-series radiomics for patient %s series %s: %s",
                        patient_id, series_uid, exc,
                    )
                    continue
                if df.empty:
                    continue
                df["patient_id"] = str(patient_id)
                df["series_uid"] = series_uid
                df["study_uid"] = study_uid
                df["image_class"] = image_class
                df["is_4d_phase"] = bool(is_4d_phase)
                df["series_dir"] = str(input_dir)
                # the temp course dir is deleted; repoint course_dir at the persistent series dir and
                # drop the course-shaped course_id (meaningless in an all-series artifact — series_uid
                # is the identifier; both CT workers emit it as the temp course basename).
                if "course_dir" in df.columns:
                    df["course_dir"] = str(input_dir)
                df = df.drop(columns=[c for c in ("course_id",) if c in df.columns])
                all_dfs.append(df)
            finally:
                shutil.rmtree(course_dir, ignore_errors=True)
        try:
            (temp_base / str(patient_id)).rmdir()  # tidy per-patient parent if now empty
        except OSError:
            pass

    try:
        temp_base.rmdir()
    except OSError:
        pass

    frames = []
    if preserved_df is not None and not preserved_df.empty:
        frames.append(preserved_df)
    frames.extend(all_dfs)
    if not frames:
        out_csv.unlink(missing_ok=True)
        logger.info(
            "No all-series CT radiomics remain after refreshing %d patient(s)",
            len(requested_patient_ids),
        )
        return None
    out_df = pd.concat(frames, ignore_index=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{out_csv.name}.", suffix=".tmp", dir=data_dir, delete=False
    ) as handle:
        tmp_csv = Path(handle.name)
    try:
        out_df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(out_csv)
    finally:
        tmp_csv.unlink(missing_ok=True)
    logger.info("Wrote all-series CT radiomics: %s (%d rows)", out_csv, len(out_df))
    return out_csv
