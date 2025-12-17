from __future__ import annotations

import json
import logging
import os
import threading
import weakref
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import pydicom


from .config import PipelineConfig
from .layout import build_course_dirs
from importlib import resources as importlib_resources
import yaml
from .utils import run_tasks_with_adaptive_workers, mask_is_cropped
from .custom_models import list_custom_model_outputs
from .custom_structures_rtstruct import _create_custom_structures_rtstruct, _is_rs_custom_stale

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


def _apply_params_to_extractor(ext: "radiomics.featureextractor.RadiomicsFeatureExtractor", params: Dict[str, Any]) -> None:
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

    # Configure feature classes; ensure every known class has an entry to
    # avoid KeyErrors downstream.
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
        for fc in ext.featureClassNames:
            if fc == 'shape2D':
                continue
            normalized_features.setdefault(fc, [])
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
    Prefers user-provided file; else copies packaged defaults into logs_root for reuse.
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
            target_root = Path(config.logs_root) if (config and config.logs_root) else Path('.')
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


def _extractor(config: PipelineConfig, modality: str = 'CT', normalize_override: Optional[bool] = None) -> Optional["radiomics.featureextractor.RadiomicsFeatureExtractor"]:
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


def _extractor_large_roi(config: PipelineConfig, modality: str = "CT") -> Optional["radiomics.featureextractor.RadiomicsFeatureExtractor"]:
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
    """Parse a custom-structures YAML/JSON file and return ROI base names."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return set()
    data: Any
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(raw)
        except Exception:
            return set()
    if not isinstance(data, dict):
        return set()
    items = data.get("custom_structures") or data.get("custom_structures_config") or []
    if not isinstance(items, list):
        return set()
    out: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                out.add(name)
    return out


def _list_roi_names_dicom(rs_path: Path) -> list[str]:
    try:
        ds = pydicom.dcmread(str(rs_path), stop_before_pixels=True, force=True)
    except Exception:
        return []
    out: list[str] = []
    for roi in getattr(ds, "StructureSetROISequence", []) or []:
        name = str(getattr(roi, "ROIName", "") or "").strip()
        if name:
            out.append(name)
    return out


def _rtstruct_masks(dicom_series_path: Path, rs_path: Path) -> Dict[str, np.ndarray]:
    try:
        from rt_utils import RTStructBuilder
    except Exception as e:
        logger.warning("rt-utils missing for RTSTRUCT to mask: %s", e)
        return {}
    try:
        rt = RTStructBuilder.create_from(dicom_series_path=str(dicom_series_path), rt_struct_path=str(rs_path))
        out: Dict[str, np.ndarray] = {}
        for name in rt.get_roi_names():
            mask = None
            try:
                if hasattr(rt, 'get_mask_for_roi'):
                    mask = rt.get_mask_for_roi(name)
                elif hasattr(rt, 'get_roi_mask'):
                    mask = rt.get_roi_mask(name)  # alternative API name
                elif hasattr(rt, 'get_roi_mask_by_name'):
                    mask = rt.get_roi_mask_by_name(name)
            except Exception:
                mask = None
            if mask is None:
                continue
            if mask.dtype != np.bool_:
                mask = mask.astype(bool)
            if mask.any():
                out[name] = mask
        return out
    except Exception as e:
        logger.debug("RTSTRUCT to mask failed: %s", e)
        return {}


def radiomics_for_course(config: PipelineConfig, course_dir: Path, custom_structures_config: Optional[Path] = None, use_cropped: bool = False) -> Optional[Path]:
    """Run pyradiomics on CT course with manual RS, RS_auto, and custom structures if present."""

    course_dirs = build_course_dirs(course_dir)
    out_path = course_dir / 'radiomics_ct.xlsx'

    # Resume-friendly: if output exists, only recompute when required ROIs are missing
    existing_df = None
    if getattr(config, "resume", False) and out_path.exists():
        try:
            import pandas as pd

            existing_df = pd.read_excel(out_path, engine="openpyxl")
        except Exception as exc:
            logger.debug("Failed reading existing radiomics_ct.xlsx for %s: %s", course_dir, exc)
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

    extractor = _extractor(config, 'CT')
    if extractor is None:
        try:
            from .radiomics_conda import radiomics_for_course as conda_radiomics_for_course
        except ImportError as exc:
            logger.warning("Conda-based radiomics helper unavailable: %s", exc)
            return None
        logger.info("Delegating CT radiomics for %s to conda environment", course_dir)
        return conda_radiomics_for_course(course_dir, config, custom_structures_config)
    img = _load_series_image(course_dirs.dicom_ct)
    if img is None:
        logger.info("No CT image for radiomics in %s", course_dir)
        return None
    rows: List[Dict] = []
    tasks: List[tuple[str, str, np.ndarray, bool]] = []

    # Determine which custom ROIs should exist (used for resume top-ups)
    desired_custom_bases: set[str] = set()
    if custom_structures_config and custom_structures_config.exists():
        desired_custom_bases = _custom_roi_names_from_config(custom_structures_config)

    missing_custom_bases: set[str] = set(desired_custom_bases)
    body_missing = False
    if existing_df is not None:
        try:
            # Custom bases are considered present when either base or base__partial exists.
            if "segmentation_source" in existing_df.columns and "roi_original_name" in existing_df.columns:
                existing_custom = set(
                    existing_df.loc[existing_df["segmentation_source"] == "Custom", "roi_original_name"]
                    .astype(str)
                    .tolist()
                )
            else:
                existing_custom = set()
            for base in list(missing_custom_bases):
                if base in existing_custom or f"{base}__partial" in existing_custom:
                    missing_custom_bases.discard(base)
        except Exception:
            pass
        try:
            if "segmentation_source" in existing_df.columns and "roi_original_name" in existing_df.columns:
                body_missing = not bool(
                    (
                        (existing_df["segmentation_source"] == "Manual")
                        & (existing_df["roi_original_name"].astype(str).str.upper() == "BODY")
                    ).any()
                )
            else:
                body_missing = False
        except Exception:
            body_missing = False

    # If we are resuming and nothing is missing, skip work.
    if (
        getattr(config, "resume", False)
        and out_path.exists()
        and existing_df is not None
        and not missing_custom_bases
        and not body_missing
    ):
        return out_path

    # Process standard RTSTRUCTs
    rs_manual = "RS.dcm"
    rs_auto = "RS_auto.dcm"

    full_run = not (getattr(config, "resume", False) and out_path.exists() and existing_df is not None)
    if full_run:
        for source, rs_name in (("Manual", rs_manual), ("AutoRTS_total", rs_auto)):
            rs_path = course_dir / rs_name
            if not rs_path.exists():
                continue
            masks = _rtstruct_masks(course_dirs.dicom_ct, rs_path)
            for roi, mask in masks.items():
                tasks.append((source, roi, mask, mask_is_cropped(mask)))
    else:
        # Resume top-up: add only missing BODY from manual RS if needed.
        if body_missing:
            try:
                from rt_utils import RTStructBuilder

                rs_path = course_dir / rs_manual
                if rs_path.exists():
                    rt = RTStructBuilder.create_from(dicom_series_path=str(course_dirs.dicom_ct), rt_struct_path=str(rs_path))
                    if "BODY" in rt.get_roi_names():
                        mask = rt.get_roi_mask_by_name("BODY")
                        if mask is not None and np.asarray(mask).astype(bool).any():
                            tasks.append(("Manual", "BODY", mask.astype(bool), mask_is_cropped(mask)))
            except Exception as exc:
                logger.debug("Resume BODY top-up failed for %s: %s", course_dir, exc)

    # Process custom structures (extract only custom ROIs; avoid duplicating base ROIs in RS_custom)
    rs_custom = course_dir / "RS_custom.dcm"
    want_custom = bool(desired_custom_bases)
    if not want_custom and rs_custom.exists():
        # Fallback: infer custom-only ROIs as those present in RS_custom but absent in RS and RS_auto.
        try:
            base_names = set(_list_roi_names_dicom(rs_custom))
            manual_names = set(_list_roi_names_dicom(course_dir / "RS.dcm"))
            auto_names = set(_list_roi_names_dicom(course_dir / "RS_auto.dcm"))
            inferred = {n for n in (base_names - (manual_names | auto_names)) if n}
            # Strip __partial suffix for base matching.
            desired_custom_bases = {n[:-9] if n.endswith("__partial") else n for n in inferred}
            missing_custom_bases = set(desired_custom_bases)
            want_custom = bool(desired_custom_bases)
        except Exception:
            want_custom = False

    if want_custom and (full_run or missing_custom_bases):
        try:
            from rt_utils import RTStructBuilder

            rs_manual_path = course_dir / "RS.dcm"
            rs_auto_path = course_dir / "RS_auto.dcm"

            # Ensure RS_custom exists and is current when a config is available.
            if custom_structures_config and custom_structures_config.exists() and _is_rs_custom_stale(
                rs_custom, custom_structures_config, rs_manual_path, rs_auto_path
            ):
                logger.info("Regenerating RS_custom.dcm for radiomics in %s", course_dir.name)
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir,
                    custom_structures_config,
                    rs_manual_path,
                    rs_auto_path,
                )

            if rs_custom and rs_custom.exists():
                # Resolve which actual ROI names to extract (base vs base__partial).
                available = set(_list_roi_names_dicom(rs_custom))
                bases_to_do = desired_custom_bases if full_run else missing_custom_bases
                wanted_names: list[str] = []
                for base in sorted(bases_to_do):
                    if base in available:
                        wanted_names.append(base)
                    elif f"{base}__partial" in available:
                        wanted_names.append(f"{base}__partial")
                if wanted_names:
                    rt = RTStructBuilder.create_from(dicom_series_path=str(course_dirs.dicom_ct), rt_struct_path=str(rs_custom))
                    for roi_name in wanted_names:
                        try:
                            mask = rt.get_roi_mask_by_name(roi_name)
                        except Exception:
                            continue
                        if mask is None:
                            continue
                        mask_bool = np.asarray(mask).astype(bool)
                        if not mask_bool.any():
                            continue
                        tasks.append(("Custom", roi_name, mask_bool, mask_is_cropped(mask_bool)))
        except Exception as e:
            logger.warning("Failed to process custom structures for radiomics: %s", e)

    # Include segmentation outputs from custom nnUNet models by default
    if full_run:
        for model_name, model_course_dir in list_custom_model_outputs(course_dir):
            rs_path = model_course_dir / "rtstruct.dcm"
            if not rs_path.exists():
                continue
            masks = _rtstruct_masks(course_dirs.dicom_ct, rs_path)
            for roi, mask in masks.items():
                tasks.append((f"CustomModel:{model_name}", roi, mask, mask_is_cropped(mask)))
    def _do_ct_task(t):
        source, roi, mask, cropped = t
        try:
            # Create fresh extractor instance for each task to avoid threading issues
            min_voxels, max_voxels_full = _derive_voxel_limits(config)
            voxel_count = int(np.asarray(mask).astype(bool).sum())
            if voxel_count < min_voxels:
                return None

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
                logger.debug("No radiomics extractor available for %s/%s", source, roi)
                return None
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
            logger.debug("Radiomics failed for %s/%s: %s", source, roi, e)
            return None
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
        for rec in results:
            if rec:
                rows.append(rec)
    if not rows:
        return None
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
                df = df.drop_duplicates(subset=["segmentation_source", "roi_original_name", "patient_id", "course_id"], keep="first")
            except Exception:
                df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df = df_new
        out = out_path
        df.to_excel(out, index=False)
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
                try:
                    if tmp_parquet.exists():
                        tmp_parquet.unlink()
                except Exception:
                    pass
                # Avoid leaving a stale sidecar if we failed to update it.
                try:
                    if parquet_path.exists():
                        parquet_path.unlink()
                except Exception:
                    pass
                logger.debug("Parquet save failed (non-critical): %s (retry: %s)", parquet_err, parquet_err2)
        return out
    except Exception as e:
        logger.warning("Failed to write CT radiomics: %s", e)
        return None


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


def _collect_total_mr_masks(series_dir: Path, seg_dir: Path) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    if not seg_dir.exists():
        return masks
    candidates = sorted(seg_dir.glob("*--total_mr.dcm"))
    for rtstruct_path in candidates:
        try:
            masks.update(_rtstruct_masks(series_dir, rtstruct_path))
            if masks:
                return masks
        except Exception as exc:
            logger.debug("Failed reading MR RTSTRUCT %s: %s", rtstruct_path, exc)
    for mask_path in sorted(seg_dir.glob("total_mr--*.nii*")):
        try:
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)
            arr = np.moveaxis(arr, 0, -1)
            mask = arr > 0
            if not mask.any():
                continue
            name = mask_path.name
            if name.endswith('.nii.gz'):
                name = name[:-7]
            elif name.endswith('.nii'):
                name = name[:-4]
            if name.startswith('total_mr--'):
                name = name[len('total_mr--'):]
            masks[name] = mask
        except Exception as exc:
            logger.debug("Failed reading MR mask %s: %s", mask_path, exc)
    return masks


def radiomics_for_course_mr(config: PipelineConfig, course) -> Optional[Path]:
    course_dirs = course.dirs if hasattr(course, 'dirs') else build_course_dirs(Path(course))
    mr_root = course_dirs.dicom_mr
    out_path = mr_root / 'radiomics_mr.xlsx'
    if not mr_root.exists():
        if out_path.exists() and not getattr(config, 'resume', False):
            try:
                out_path.unlink()
            except Exception:
                pass
        return None

    # Check if we need conda fallback (NumPy 2.x)
    import numpy as np
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version[0] >= 2:
        # Delegate to conda-based MR radiomics
        try:
            from .radiomics_conda import radiomics_for_course_mr as conda_radiomics_for_course_mr
        except ImportError as exc:
            logger.warning("Conda-based MR radiomics helper unavailable: %s", exc)
            return None
        logger.info("Delegating MR radiomics for %s to conda environment", course_dirs.root)
        return conda_radiomics_for_course_mr(course_dirs.root, config)

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
            logger.debug("No MR image for radiomics in %s", source_dir)
            continue
        base_name = _strip_nii_name(nifti_path)
        masks = _collect_total_mr_masks(source_dir, seg_dir)
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
                logger.debug("Radiomics MR failed for %s/%s: %s", series_uid, roi_name, exc)
                continue
    if not rows:
        if out_path.exists() and not getattr(config, 'resume', False):
            try:
                out_path.unlink()
            except Exception:
                pass
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(out_path, index=False)
        # Also save Parquet for faster aggregation
        try:
            parquet_path = out_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False, engine='pyarrow')
            logger.debug("Saved MR Parquet: %s", parquet_path)
        except Exception as parquet_err:
            logger.debug("MR Parquet save failed (non-critical): %s", parquet_err)
        return out_path
    except Exception as exc:
        logger.warning("Failed to write MR radiomics for %s: %s", course_dirs.root, exc)
        return None


def _find_mr_manual_rs(dicom_root: Path, patient_id: str, mr_for_uid: str) -> List[Path]:
    rs_list: List[Path] = []
    for base, _, files in os.walk(dicom_root):
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

    # Check if we can use PyRadiomics
    can_use_radiomics = _have_pyradiomics()

    if not can_use_radiomics:
        logger.warning("PyRadiomics not available - skipping radiomics extraction")
        return

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

    run_tasks_with_adaptive_workers(
        "Radiomics (CT courses)",
        courses,
        radiomics_func,
        max_workers=max_course_workers,
        logger=logger,
        show_progress=True,
    )
    for course in courses:
        try:
            radiomics_for_course_mr(config, course)
        except Exception as exc:
            logger.warning("MR radiomics failed for course %s: %s", getattr(course, 'dirs', course), exc)
    # Cohort merge
    try:
        import pandas as _pd
        out_rows = []
        # CT cohort
        for c in courses:
            p = Path(c.dirs.root) / 'radiomics_ct.xlsx'
            if p.exists():
                try:
                    df = _pd.read_excel(p)
                    df.insert(0, 'patient_id', getattr(c, 'patient_id', Path(c.dirs.root).parts[-2]))
                    df.insert(1, 'course_key', getattr(c, 'course_key', Path(c.dirs.root).name))
                    df.insert(2, 'course_dir', str(c.dirs.root))
                    out_rows.append(df)
                except Exception:
                    pass
        # MR cohort
        for p in config.output_root.rglob('MR_*/radiomics_features_MR.xlsx'):
            try:
                df = _pd.read_excel(p)
                # Attempt to infer patient_id and series_uid from path
                parts = p.parts

                # Ensure we have enough path components before accessing
                if len(parts) >= 4:
                    try:
                        idx = parts.index(str(config.output_root))
                    except ValueError:
                        idx = len(parts)-4

                    # Safely access path components with bounds checking
                    if len(parts) > 4:
                        patient_id = parts[-4]
                    else:
                        patient_id = parts[0] if parts else 'unknown'

                    if len(parts) > 2:
                        series_uid = parts[-2].replace('MR_','') if parts[-2].startswith('MR_') else parts[-2]
                    else:
                        series_uid = 'unknown'

                    df.insert(0, 'patient_id', patient_id)
                    df.insert(1, 'series_uid', series_uid)
                    df.insert(2, 'series_dir', str(p.parent))
                    out_rows.append(df)
            except Exception:
                pass
        if out_rows:
            all_df = _pd.concat(out_rows, ignore_index=True)
            data_dir = config.output_root / 'Data'
            data_dir.mkdir(parents=True, exist_ok=True)
            all_df.to_excel(data_dir / 'radiomics_all.xlsx', index=False)
    except Exception as e:
        logger.warning("Failed to write cohort radiomics: %s", e)
