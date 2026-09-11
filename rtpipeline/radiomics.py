from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import weakref
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

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
    build_treatment_technique_contract,
    classify_course_dose_completeness,
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
    load_rs_custom_outcomes,
    record_rs_custom_resume_decision,
)
from .acquisition_scale import (
    attach_acquisition_descriptor,
    describe_contract_planning_ct,
    validate_acquisition_descriptor_table,
)
from .radiomics_cohort import (
    attach_radiomics_cohort_provenance,
    build_radiomics_cohort_provenance,
    is_valid_radiomics_cohort_table,
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
)
from .radiomics_memory import permits_second_stage, legacy_rejection
from .radiomics_resource_guard import (
    RESAMPLED_BBOX_LIMIT_CODE,
    estimate_resampled_bounding_box,
    resolve_max_resampled_bbox_voxels,
)
from .radiomics_schema import (
    RadiomicsFeatureTypeError,
    assert_radiomics_arrow_schema,
    expected_radiomics_string_columns,
    normalize_radiomics_dataframe,
    normalize_radiomics_result,
    write_radiomics_feature_table_atomic,
)
from .roi_requiredness import (
    _norm as _normalize_required_roi_name,
    DenominatorLedger,
    FAILED_RADIOMICS_FEATURE_COMPLETENESS,
    FAILED_RADIOMICS_RESOURCE_LIMIT,
    REASON_CODES,
    Requiredness,
    TAXONOMY_CODES,
    assess_custom_applicability,
    inspect_rtstruct,
    match_requirements,
    requirements_from_contract,
    requiredness_for,
    write_modality_ledger,
)
from .radiomics_ct_contract import (
    CT_EXTRACTION_ARMS,
    PRIMARY_ARM,
    RADIOMICS_FEATURE_COMPLETENESS_COLUMN,
    RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN,
    SENSITIVITY_ARM,
    classify_ct_roi,
    configured_parameter_hash,
    current_code_revision,
    disposition_rows_for_arms,
    effective_parameter_hash,
    effective_parameter_hashes_for_arms,
    expected_publication_keys,
    extract_ct_roi_arms,
    load_custom_structure_provenance,
    new_run_identifier,
    publication_key,
    read_authoritative_ct_publication,
    rtstruct_roi_identities,
    stable_rtstruct_roi_identity,
    validate_ct_publication,
    write_ct_publication_atomic,
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


def _write_course_exclusion_ledger(course_dir: Path, config: Any, *, reason_code: str, in_scope: bool = True) -> None:
    ledger = DenominatorLedger()
    course_id = course_dir.name
    patient_id = course_dir.parent.name
    contract = getattr(config, "radiomics_analysis_contract", {}) or {}
    for requirement in requirements_from_contract(contract, "CT"):
        if requirement.requiredness == Requiredness.INVENTORY_ONLY:
            continue
        ledger.expect_course_roi(course_id, requirement.canonical_name)
        ledger.record_roi(course_id, patient_id, requirement.canonical_name, reason_code=reason_code, disposition="excluded")
    ledger.record_course(course_id, patient_id, screened=True, in_scope=in_scope, out_of_scope=not in_scope, adequate_coverage=False, insufficient_coverage=in_scope, valid_derivation=False, technical_exclusion=reason_code not in {"not_applicable_scope", "not_applicable_anatomy"}, indeterminate=reason_code == "indeterminate_applicability", extracted=False, reason_code=reason_code)
    _write_course_roi_ledger(course_dir, ledger)


def _analysis_contract(config: Any) -> Any:
    return getattr(config, "radiomics_analysis_contract", None) or {}


def _roi_requiredness(config: Any, source: str, roi_name: str, *, modality: str = "CT", selected_model: bool = False) -> Requiredness:
    return requiredness_for(
        source,
        roi_name,
        contract=_analysis_contract(config),
        modality=modality,
        explicitly_selected_model=selected_model,
    )


def _mr_radiomics_required(config: PipelineConfig) -> bool:
    """MR requiredness comes from the MR analysis contract, not params presence."""
    return any(
        requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
        for requirement in requirements_from_contract(_analysis_contract(config), "MR")
    )


def _planning_ct_fov(course_dir: Path) -> dict[str, Any]:
    """Resolve campaign body-region evidence without inspecting cropped CT bytes."""
    candidates = (
        Path(course_dir) / "qc_reports" / "body_regions.json",
        Path(course_dir) / "metadata" / "body_regions.json",
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        regions = data.get("regions", data.get("body_regions", data)) if isinstance(data, dict) else data
        if isinstance(regions, dict):
            regions = [
                str(key).replace("CONTAINS_", "").replace("contains_", "")
                for key, value in regions.items()
                if value
            ]
        if isinstance(regions, (list, tuple, set)):
            normalized = {str(value).casefold().replace("-", "_") for value in regions}
            if normalized:
                return {
                    "contains_regions": tuple(normalized),
                    "excluded_regions": ("pelvis",) if "pelvis" not in normalized else (),
                }
    return {}


def _write_course_roi_ledger(course_dir: Path, ledger: DenominatorLedger) -> None:
    try:
        write_modality_ledger(Path(course_dir) / "metadata", ledger, "CT")
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"Could not write radiomics denominator ledger for {course_dir}: {exc}"
        ) from exc


_DEFAULT_MIN_VOXELS = 120
_DEFAULT_MAX_VOXELS_FULL = 15_000_000


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
    """Compatibility wrapper: a large or BODY-named ROI keeps the configured method.

    ROI size and name never select a different extraction method. Feasibility is
    governed by the resource bounding-box, memory, and timeout guards, which
    reject an infeasible ROI as a non-measurement instead of silently pruning
    image types, feature classes, or resampled spacing.
    """
    return _extractor(config, modality)


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


def _list_roi_names_dicom(rs_path: Path, *, allow_empty: bool = False) -> list[str]:
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
    if not out and not allow_empty:
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
    requiredness_by_roi: Optional[Dict[str, Any]] = None,
    structural_inventory: Any = None,
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

    def _is_required(name: str) -> bool:
        value = (requiredness_by_roi or {}).get(str(name), Requiredness.INVENTORY_ONLY)
        return getattr(value, "value", value) == Requiredness.ANALYSIS_REQUIRED.value

    def _record_or_raise(
        name: str,
        detail: str,
        *,
        failure_kind: str = "extraction_error",
        structural_code: Optional[str] = None,
    ) -> None:
        if structural_code is None and (requiredness_by_roi is not None or structural_inventory is not None):
            structural_code = "ROI_EXTRACTION_FAILED"
        outcome = {
            "roi_name": str(name),
            "status": "failed",
            "failure_kind": failure_kind,
            "reason": detail,
        }
        if structural_code:
            outcome["structural_code"] = structural_code
        fatal = not best_effort or _is_required(name)
        if fatal:
            if failure_outcomes is not None:
                failure_outcomes.append(outcome)
            suffix = f" [{structural_code}]" if structural_code else ""
            raise RadiomicsCourseExtractionError(detail + suffix)
        if failure_outcomes is not None:
            failure_outcomes.append(outcome)
        logger.warning("Best-effort radiomics ROI %s was not extracted: %s", name, detail)
    if best_effort and failure_outcomes is None:
        raise ValueError("best-effort RTSTRUCT extraction requires a failure_outcomes sink")

    # Inspect the source before invoking rt-utils. In particular, a declared ROI
    # without ContourSequence is an inventory observation, not an AttributeError
    # from which source corruption may be inferred.
    if structural_inventory is None and Path(rs_path).exists():
        try:
            structural_inventory = inspect_rtstruct(Path(rs_path))
        except Exception as exc:
            if not best_effort:
                raise RadiomicsCourseExtractionError(
                    f"Failed to inspect RTSTRUCT {rs_path}: {exc}"
                ) from exc
    def _validate_expected_inventory(names: List[str]) -> None:
        if expected_rois is None:
            return
        if len(names) != len(set(names)):
            raise RadiomicsCourseExtractionError(
                f"Required RTSTRUCT has duplicate ROI identities: {rs_path}"
            )
        missing = sorted(set(expected_rois) - set(names))
        unexpected = sorted(set(names) - set(expected_rois))
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

    # Validate before any non-volumetric early return or name-keyed collapse.
    if structural_inventory is not None:
        _validate_expected_inventory([
            observation.name for observation in structural_inventory.named_rois
        ])
    inventory_observations = {
        observation.name: observation
        for observation in getattr(structural_inventory, "named_rois", ())
    }
    structurally_extractable = {
        observation.name for observation in inventory_observations.values()
        if not observation.structural_code
    }
    for observation in inventory_observations.values():
        from .rtstruct_geometry import NONVOLUMETRIC_CODES
        if observation.structural_code in NONVOLUMETRIC_CODES:
            if ''.join(ch for ch in observation.name.lower() if ch.isalnum()) in normalized_skips:
                # The caller publishes the declared-skip arm rows. Do not also
                # publish a second disposition for the same ROI-arm identity.
                continue
            if failure_outcomes is None:
                raise ValueError("non-volumetric ROI dispositions require an outcome sink")
            from .rtstruct_identity import require_rtstruct_identity
            failure_outcomes.append({
                "roi_name": observation.name, "status": "nonvolumetric_nonmeasurement",
                "failure_kind": "nonvolumetric_geometry", "reason": observation.structural_code,
                "structural_code": observation.structural_code,
                "rtstruct_sop_instance_uid": require_rtstruct_identity(rs_path),
                "roi_number": str(observation.roi_number), "source_path": str(rs_path),
            })
            continue
        if observation.structural_code:
            if (
                observation.structural_code in {
                    "ROI_DECLARED_NO_CONTOUR_ITEM",
                    "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
                }
                and not _is_required(observation.name)
            ):
                continue
            _record_or_raise(
                observation.name,
                f"ROI {observation.name!r} in {rs_path} has structural status {observation.structural_code}",
                failure_kind="structural_roi_error",
                structural_code=observation.structural_code,
            )
    if structural_inventory is not None and not inventory_observations:
        if "RTSTRUCT_NO_NAMED_ROIS" in getattr(structural_inventory, "structural_codes", ()):
            if not best_effort:
                raise RadiomicsCourseExtractionError(
                    f"RTSTRUCT contains no named ROIs: {rs_path} [RTSTRUCT_NO_NAMED_ROIS]"
                )
            if failure_outcomes is not None:
                failure_outcomes.append({
                    "roi_name": "",
                    "status": "failed",
                    "failure_kind": "structural_roi_error",
                    "reason": f"RTSTRUCT contains no named ROIs: {rs_path}",
                    "structural_code": "RTSTRUCT_NO_NAMED_ROIS",
                })
        return {}
    if structural_inventory is not None and not structurally_extractable:
        return {}
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

    _validate_expected_inventory(available_roi_names)
    roi_names = list(expected_rois) if expected_rois is not None else available_roi_names
    if structural_inventory is not None:
        # Expected declarations include points and open contours, but only
        # volumetric identities may reach the mask reader.
        roi_names = [name for name in roi_names if name in structurally_extractable]

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
                structural_code=(
                    "ROI_MASK_EMPTY_AFTER_RASTERIZATION"
                    if requiredness_by_roi is not None or structural_inventory is not None
                    else None
                ),
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


@dataclass(frozen=True)
class _DirectCtTask:
    source: str
    roi_name: str
    mask: np.ndarray
    cropped: bool
    mask_identity: str
    stable_roi_identifier: str
    source_content_sha256: str
    decision: Any
    required: bool
    configured_parameter_hashes: Dict[str, str]
    effective_parameter_hashes: Dict[str, str]



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
    acquisition_descriptor = describe_contract_planning_ct(contract)

    # Resume-friendly: if output exists, only recompute when required ROIs are missing
    existing_df = None
    parquet_path = out_path.with_suffix(".parquet")
    if getattr(config, "resume", False) and (parquet_path.exists() or out_path.exists()):
        try:
            if not parquet_path.exists():
                raise ValueError("authoritative CT Parquet is missing")
            existing_df = read_authoritative_ct_publication(parquet_path)
            _resume_identity_pairs(existing_df)
            validate_acquisition_descriptor_table(
                existing_df,
                expected_descriptor=acquisition_descriptor,
                expected_series_instance_uid=contract.planning_ct.get(
                    "series_instance_uid"
                ),
            )
        except Exception as exc:
            logger.warning(
                "Invalidating unusable resume publication for %s: %s", course_dir, exc
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
            _write_course_exclusion_ledger(course_dir, config, reason_code="not_applicable_scope", in_scope=False)
            return RadiomicsCourseOutcome.nothing_to_do("CT image is absent")
        _invalidate_radiomics_outputs(out_path)
        _write_course_exclusion_ledger(course_dir, config, reason_code="failed_source_read")
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
    series_uid = str(contract.planning_ct.get("series_instance_uid") or "").strip()
    if not series_uid:
        _invalidate_radiomics_outputs(out_path)
        _write_course_exclusion_ledger(course_dir, config, reason_code="failed_source_read")
        raise RadiomicsCourseExtractionError(
            f"Planning CT contract has no SeriesInstanceUID for radiomics in {course_dir}"
        )
    run_identifier = new_run_identifier()
    code_revision = current_code_revision()
    rows: List[Dict] = []
    tasks: List[_DirectCtTask] = []
    roi_failures: List[Dict[str, str]] = []
    custom_applicability: list[Any] = []
    source_counts: Dict[str, Dict[str, int]] = {}
    roi_ledger = DenominatorLedger()
    ledger_course_id = course_dir.name
    ledger_patient_id = course_dir.parent.name
    ledger_expected_names = {
        requirement.canonical_name
        for requirement in requirements_from_contract(_analysis_contract(config), "CT")
        if requirement.requiredness != Requiredness.INVENTORY_ONLY
    }

    def _finalize_ct_ledger(*, extracted: bool, technical: bool = False, indeterminate: bool = False) -> None:
        technical = technical or any(
            str(row.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or "")
            == "incomplete"
            for row in rows
        )
        for name in sorted(ledger_expected_names):
            roi_ledger.expect_course_roi(ledger_course_id, name)
        rows_by_roi: Dict[tuple[str, str], List[Mapping[str, Any]]] = {}
        for row in rows:
            key = (
                str(row.get("segmentation_source", "")),
                str(row.get("roi_original_name", row.get("roi_name", ""))),
            )
            if key[1]:
                rows_by_roi.setdefault(key, []).append(row)
        for key, pair_rows in rows_by_roi.items():
            row = pair_rows[0]
            name = key[1]
            statuses = {
                str(item.get("extraction_status") or "success")
                for item in pair_rows
            }
            completeness = {
                str(item.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or "")
                for item in pair_rows
            }
            detail_code = str(row.get("roi_structural_code") or "")
            if "incomplete" in completeness:
                reason = FAILED_RADIOMICS_FEATURE_COMPLETENESS
            elif statuses == {"success"}:
                reason = "extracted"
            elif statuses == {"nonvolumetric_nonmeasurement"}:
                from .rtstruct_geometry import NONVOLUMETRIC_CODES
                if detail_code not in NONVOLUMETRIC_CODES:
                    raise RadiomicsCourseExtractionError(
                        "Non-volumetric disposition lacks a valid structural reason"
                    )
                reason = detail_code
            elif statuses == {"declared_skip"}:
                reason = "CONFIGURED_SKIP"
            elif "below_minimum_voxels" in statuses:
                reason = "ROI_MASK_BELOW_MIN_VOXELS"
            elif detail_code == RESAMPLED_BBOX_LIMIT_CODE:
                reason = FAILED_RADIOMICS_RESOURCE_LIMIT
            else:
                reason = "failed_radiomics_extraction"
            completeness_detail = next(
                (
                    str(
                        item.get(RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN)
                        or ""
                    )
                    for item in pair_rows
                    if str(
                        item.get(RADIOMICS_FEATURE_COMPLETENESS_COLUMN) or ""
                    )
                    == "incomplete"
                ),
                "",
            )
            roi_ledger.expect_course_roi(ledger_course_id, name)
            roi_ledger.record_roi(
                ledger_course_id,
                ledger_patient_id,
                name,
                reason_code=reason,
                disposition=(
                    "nonmeasurement" if statuses == {"nonvolumetric_nonmeasurement"}
                    else "extracted" if reason == "extracted" else "excluded"
                ),
                segmentation_source=key[0],
                detail_code=detail_code or None,
                detail=(
                    completeness_detail
                    if reason == FAILED_RADIOMICS_FEATURE_COMPLETENESS
                    else str(row.get("extraction_status_detail") or "")
                ),
                **({"resource_guard_reason_code": row["resource_guard_reason_code"]}
                   if row.get("resource_guard_reason_code") in {"ROI_RESOURCE_BBOX_ADMITTED", "ROI_RESOURCE_MEMORY_ADMITTED"} else {}),
                estimated_resampled_bbox_voxel_count=row.get(
                    "estimated_resampled_bbox_voxel_count"
                ),
                max_resampled_bbox_voxel_count=row.get(
                    "max_resampled_bbox_voxel_count"
                ),
            )
        for failure in roi_failures:
            name = str(failure.get("roi_name", ""))
            source = str(failure.get("source", ""))
            if name and (source, name) not in rows_by_roi:
                reason = str(failure.get("structural_code") or failure.get("reason_code") or "failed_radiomics_extraction")
                if reason not in REASON_CODES and reason not in TAXONOMY_CODES:
                    reason = "failed_radiomics_extraction"
                roi_ledger.record_roi(ledger_course_id, ledger_patient_id, name, reason_code=reason, disposition="excluded", segmentation_source=source)
        present_names = {str(row.get("roi_name", "")) for row in roi_ledger.roi_rows}
        for assessment in custom_applicability:
            roi_ledger.expect_course_roi(ledger_course_id, assessment.roi_name)
            if assessment.reason_code == "extracted" and assessment.roi_name in present_names:
                continue
            roi_ledger.record_roi(ledger_course_id, ledger_patient_id, assessment.roi_name, reason_code=assessment.reason_code, disposition="extracted" if assessment.reason_code == "extracted" else "excluded", detail=assessment.detail)
        observed_rows = list(roi_ledger.roi_rows)
        for requirement in requirements_from_contract(_analysis_contract(config), "CT"):
            if requirement.requiredness == Requiredness.INVENTORY_ONLY:
                continue
            aliases = {_normalize_required_roi_name(name) for name in requirement.accepted_names}
            matches = [
                row for row in observed_rows
                if _normalize_required_roi_name(row.get("roi_name")) in aliases
                and (
                    not requirement.source
                    or _normalize_required_roi_name(row.get("segmentation_source", row.get("source")))
                    == _normalize_required_roi_name(requirement.source)
                )
            ]
            if any(row.get("roi_name") == requirement.canonical_name for row in matches):
                continue
            if len(matches) == 1:
                matched = matches[0]
                # An accepted name is an identity bridge, not proof of extraction.
                roi_ledger.record_roi(
                    ledger_course_id, ledger_patient_id, requirement.canonical_name,
                    reason_code=matched["reason_code"], disposition=matched["disposition"],
                    segmentation_source=matched.get("segmentation_source", matched.get("source", "")),
                    alias_used=True, alias_name=matched["roi_name"],
                    detail_code=matched.get("detail_code"), detail=matched.get("detail", ""),
                )
            else:
                roi_ledger.record_roi(
                    ledger_course_id, ledger_patient_id, requirement.canonical_name,
                    reason_code=("REQUIRED_ROI_AMBIGUOUS_MATCH" if matches else "REQUIRED_ROI_NOT_DECLARED"),
                    disposition="excluded", segmentation_source=requirement.source or "",
                )
        all_nonvolumetric = bool(rows) and all(
            row.get("extraction_status") == "nonvolumetric_nonmeasurement"
            for row in rows
        )
        roi_ledger.record_course(
            ledger_course_id,
            ledger_patient_id,
            screened=True,
            in_scope=True,
            out_of_scope=False,
            adequate_coverage=bool(rows) and not all_nonvolumetric,
            insufficient_coverage=not bool(rows),
            valid_derivation=any(item.reason_code == "extracted" for item in custom_applicability),
            technical_exclusion=technical,
            indeterminate=indeterminate,
            extracted=extracted,
            reason_code=(
                "extracted" if extracted else
                "nonvolumetric_nonmeasurement" if all_nonvolumetric else
                "indeterminate_applicability" if indeterminate else
                "failed_radiomics_extraction"
            ),
        )
        _write_course_roi_ledger(course_dir, roi_ledger)
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
            }
        except RadiomicsCourseExtractionError:
            _invalidate_radiomics_outputs(out_path)
            raise

    if configured_custom_path is not None:
        ledger_expected_names.update(_custom_roi_names_from_config(configured_custom_path))
    custom_provenance = load_custom_structure_provenance(configured_custom_path)
    parameter_path = _get_params_file(config, "CT")
    source_content_hashes: Dict[Path, str] = {}

    def _source_fingerprint(rs_path: Path) -> str:
        if rs_path not in source_content_hashes:
            try:
                source_content_hashes[rs_path] = hashlib.sha256(rs_path.read_bytes()).hexdigest()
            except Exception as exc:
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Failed to fingerprint RTSTRUCT source {rs_path}: {exc}"
                ) from exc
        return source_content_hashes[rs_path]

    def _verify_source_fingerprints() -> None:
        try:
            for rs_path, expected in source_content_hashes.items():
                if hashlib.sha256(rs_path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"RTSTRUCT source changed during radiomics: {rs_path}")
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Cannot publish or reuse CT radiomics with changed or unreadable source fingerprints: {exc}"
            ) from exc

    def _uses_large_extractor(roi_name: str, mask: np.ndarray) -> bool:
        _, max_voxels_full = _derive_voxel_limits(config)
        try:
            spacing = tuple(float(value) for value in img.GetSpacing())
        except Exception:
            spacing = (1.0, 1.0, 1.0)
        native_voxel_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
        physical_volume_mm3 = float(np.asarray(mask).astype(bool).sum()) * max(1e-9, native_voxel_mm3)
        return roi_name.strip().lower().startswith("body") or physical_volume_mm3 > float(max_voxels_full)

    def _effective_hashes(decision: Any, *, large_roi: bool = False) -> Dict[str, str]:
        def _factory() -> Any:
            candidate = (
                _extractor_large_roi(config, "CT")
                if large_roi
                else _extractor(config, "CT")
            )
            if candidate is None:
                raise RuntimeError("No CT radiomics extractor is available")
            return candidate

        return effective_parameter_hashes_for_arms(_factory, decision)

    def _identity_for(rs_path: Path, roi_name: str) -> tuple[str, str]:
        return stable_rtstruct_roi_identity(rs_path, roi_name)

    def _make_task(
        source: str,
        rs_path: Path,
        roi_name: str,
        mask: np.ndarray,
        *,
        required: bool,
    ) -> _DirectCtTask:
        sop_uid, roi_number = _identity_for(rs_path, roi_name)
        decision = classify_ct_roi(
            source,
            roi_name,
            custom_provenance=custom_provenance if source == "Custom" else None,
        )
        configured_hashes = {
            arm: configured_parameter_hash(
                parameter_path,
                arm=arm,
                window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
                large_roi=False,
            )
            for arm in CT_EXTRACTION_ARMS
        }
        return _DirectCtTask(
            source=source,
            roi_name=roi_name,
            mask=mask,
            cropped=mask_is_cropped(mask),
            mask_identity=sop_uid,
            stable_roi_identifier=f"rtstruct_roi_number:{roi_number}",
            source_content_sha256=_source_fingerprint(rs_path),
            decision=decision,
            required=required,
            configured_parameter_hashes=configured_hashes,
            effective_parameter_hashes=_effective_hashes(
                decision, large_roi=_uses_large_extractor(roi_name, mask)
            ),
        )

    def _common_metadata(task: _DirectCtTask) -> Dict[str, Any]:
        display_roi = (
            task.roi_name
            if (not task.cropped or task.roi_name.endswith("__partial"))
            else f"{task.roi_name}__partial"
        )
        return {
            "modality": "CT",
            "segmentation_source": task.source,
            "roi_name": display_roi,
            "roi_original_name": task.roi_name,
            "course_dir": str(course_dir),
            "patient_id": course_dir.parent.name,
            "course_id": course_dir.name,
            "series_uid": series_uid,
            "mask_identity": task.mask_identity,
            "rtstruct_sop_instance_uid": task.mask_identity,
            "stable_roi_identifier": task.stable_roi_identifier,
            "source_content_sha256": task.source_content_sha256,
            "configured_minimum_native_voxel_count": _derive_voxel_limits(config)[0],
            "structure_cropped": bool(task.cropped),
        }

    def _append_declared_skip(
        source: str, rs_path: Path, roi_name: str, *, required: bool
    ) -> None:
        task = _make_task(
            source,
            rs_path,
            roi_name,
            np.zeros((0, 0, 0), dtype=bool),
            required=required,
        )
        rows.extend(
            disposition_rows_for_arms(
                _common_metadata(task),
                decision=task.decision,
                disposition="declared_skip",
                detail="ROI is listed in radiomics_skip_rois",
                failure_kind="declared_ineligible",
                run_identifier=run_identifier,
                code_revision=code_revision,
                native_voxel_count=0,
                required=required,
                effective_hashes=task.effective_parameter_hashes,
                configured_parameter_hashes=task.configured_parameter_hashes,
            )
        )

    # Process every current non-skipped identity before deciding whether a resume
    # workbook is complete. Partial top-ups can silently miss newly added sources.

    required_contract = requirements_from_contract(_analysis_contract(config), "CT")
    observed_required: set[str] = set()

    # Process standard RTSTRUCTs. Every declared ROI is inventoried, but only
    # contract-matched identities are analysis-required.
    rs_manual_path = contracted_rs_manual
    rs_auto_path_name = "RS_auto.dcm"

    sources = _standard_rtstruct_sources(contract, course_dir)
    for source, rs_path, source_roi_names in sources:
        if not rs_path.exists():
            continue
        _source_fingerprint(rs_path)
        try:
            source_inventory = inspect_rtstruct(rs_path)
            source_roi_names = list(source_inventory.named_rois[i].name for i in range(len(source_inventory.named_rois)))
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to inspect RTSTRUCT identities for {rs_path}: {exc}"
            ) from exc
        requiredness_by_roi = {
            roi_name: _roi_requiredness(config, source, roi_name)
            for roi_name in source_roi_names
        }
        for match in match_requirements(source_inventory, required_contract, source=source):
            if match.observation is not None:
                observed_required.add(match.requirement.canonical_name)
            if match.structural_code == "REQUIRED_ROI_AMBIGUOUS_MATCH":
                roi_failures.append({
                    "roi_name": match.requirement.canonical_name,
                    "source": source,
                    "status": "failed",
                    "failure_kind": "structural_roi_error",
                    "reason_code": "REQUIRED_ROI_AMBIGUOUS_MATCH",
                    "structural_code": "REQUIRED_ROI_AMBIGUOUS_MATCH",
                    "reason": f"required ROI has ambiguous identity in {rs_path}",
                })
                _finalize_ct_ledger(extracted=False, technical=False, indeterminate=True)
                raise RadiomicsCourseExtractionError(
                    f"Required ROI {match.requirement.canonical_name!r} has ambiguous identity in {rs_path} "
                    "[REQUIRED_ROI_AMBIGUOUS_MATCH]"
                )
            if match.structural_code and match.observation is not None:
                requiredness_by_roi[match.observation.name] = match.requirement.requiredness

        if normalized_skip_rois:
            for roi_name in source_roi_names:
                if ''.join(ch for ch in roi_name.lower() if ch.isalnum()) in normalized_skip_rois:
                    _append_declared_skip(
                        source,
                        rs_path,
                        roi_name,
                        required=requiredness_by_roi[roi_name] == Requiredness.ANALYSIS_REQUIRED,
                    )
        try:
            source_failures: List[Dict[str, str]] = []
            masks = _rtstruct_masks(
                contracted_ct_dir,
                rs_path,
                skip_rois=configured_skip_rois,
                best_effort=True,
                failure_outcomes=source_failures,
                requiredness_by_roi=requiredness_by_roi,
                structural_inventory=source_inventory,
            )
            for failure in source_failures:
                failure["source"] = source
                failure_name = failure.get("roi_name", "")
                if not failure_name:
                    continue
                failed_task = _make_task(
                    source,
                    rs_path,
                    failure_name,
                    np.zeros((0, 0, 0), dtype=bool),
                    required=requiredness_by_roi.get(failure_name, Requiredness.INVENTORY_ONLY) == Requiredness.ANALYSIS_REQUIRED,
                )
                rows.extend(
                    disposition_rows_for_arms(
                        {**_common_metadata(failed_task),
                         "roi_structural_code": failure.get("structural_code")},
                        decision=failed_task.decision,
                        disposition=failure["status"],
                        detail=failure["reason"],
                        failure_kind=failure["failure_kind"],
                        run_identifier=run_identifier,
                        code_revision=code_revision,
                        native_voxel_count=(
                            None if failure["status"] == "nonvolumetric_nonmeasurement" else 0
                        ),
                        required=failed_task.required,
                        effective_hashes=failed_task.effective_parameter_hashes,
                        configured_parameter_hashes=failed_task.configured_parameter_hashes,
                    )
                )
            extraction_failures = [
                item for item in source_failures
                if item["status"] != "nonvolumetric_nonmeasurement"
            ]
            roi_failures.extend(extraction_failures)
            source_counts.setdefault(source, {"attempted": 0, "extracted": 0, "failed": 0})["failed"] += len(extraction_failures)
            source_counts[source]["attempted"] += len(extraction_failures)
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            _finalize_ct_ledger(extracted=False, technical=True)
            if isinstance(exc, RadiomicsCourseExtractionError):
                raise
            raise RadiomicsCourseExtractionError(
                f"Failed to prepare RTSTRUCT source {source} for {course_dir}: {exc}"
            ) from exc
        for roi, mask in masks.items():
            tasks.append(
                _make_task(
                    source,
                    rs_path,
                    roi,
                    mask,
                    required=requiredness_by_roi.get(roi, Requiredness.INVENTORY_ONLY) == Requiredness.ANALYSIS_REQUIRED,
                )
            )

    # Process custom structures (extract only custom ROIs; avoid duplicating base ROIs in RS_custom)
    rs_custom = course_dir / "RS_custom.dcm"
    want_custom = bool(desired_custom_bases)
    dependency_states: dict[str, Any] = {}
    pending_custom_assessments: set[str] = set()
    custom_stage_outcomes: dict[str, dict[str, object]] = {}
    if (
        configured_custom_path is not None
        and rs_custom.is_file()
        and not _is_rs_custom_stale(
            rs_custom,
            configured_custom_path,
            contracted_rs_manual,
            course_dir / "RS_auto.dcm",
        )
    ):
        custom_stage_outcomes = load_rs_custom_outcomes(course_dir)
    if want_custom and configured_custom_path is not None:
        # First classify source applicability. A missing but source-applicable ROI is
        # a rebuild candidate, not a terminal finding. Final applicability is recorded
        # only after the governed RS_custom builder has had one repair attempt.
        custom_inventory = None
        if rs_custom.is_file():
            try:
                custom_inventory = inspect_rtstruct(rs_custom)
            except Exception as exc:
                custom_inventory = None
                logger.warning("Could not inspect existing RS_custom.dcm for %s: %s", course_dir, exc)
        available_custom = {
            observation.name: observation
            for observation in getattr(custom_inventory, "named_rois", ())
        }
        for task in tasks:
            dependency_states[task.roi_name] = {
                "readable": True,
                "non_empty": bool(np.asarray(task.mask).astype(bool).any()),
            }
        eligible_custom: set[str] = set()
        for base in sorted(desired_custom_bases):
            candidate = next(
                (available_custom[name] for name in (base, f"{base}__partial") if name in available_custom),
                None,
            )
            generated_state = "absent"
            if candidate is not None:
                generated_state = "readable_nonempty" if candidate.has_readable_contour else "unreadable"
            assessment = assess_custom_applicability(
                base,
                dependency_states,
                _planning_ct_fov(course_dir),
                generated_state=generated_state,
                custom_provenance=custom_provenance,
            )
            custom_required = _roi_requiredness(config, "Custom", base) == Requiredness.ANALYSIS_REQUIRED
            if assessment.reason_code == "extracted":
                eligible_custom.add(base)
                custom_applicability.append(assessment)
                if custom_required:
                    observed_required.add(base)
            elif assessment.reason_code == "failed_custom_generation":
                eligible_custom.add(base)
                pending_custom_assessments.add(base)
            elif assessment.reason_code == "indeterminate_applicability":
                custom_applicability.append(assessment)
                _finalize_ct_ledger(extracted=False, indeterminate=True)
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Configured custom ROI {base!r} has {assessment.reason_code}: {assessment.detail}"
                )
            else:
                custom_applicability.append(assessment)
                if assessment.reason_code in {"not_applicable_anatomy", "not_applicable_scope"}:
                    if custom_required:
                        observed_required.add(base)
                elif custom_required:
                    _finalize_ct_ledger(extracted=False, technical=True)
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"Required custom ROI {base!r} has {assessment.reason_code}: {assessment.detail}"
                    )
        desired_custom_bases = eligible_custom
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

            _source_fingerprint(rs_custom)
            # Resolve every declared ROI (base vs base__partial). A configured
            # source-applicable ROI gets one governed rebuild attempt before a
            # missing outcome is treated as a technical generation failure.
            available = set(_list_roi_names_dicom(rs_custom))
            for base in sorted(pending_custom_assessments):
                generated_state = (
                    "readable_nonempty"
                    if base in available or f"{base}__partial" in available
                    else "absent"
                )
                final_assessment = assess_custom_applicability(
                    base,
                    dependency_states,
                    _planning_ct_fov(course_dir),
                    generated_state=generated_state,
                    custom_provenance=custom_provenance,
                )
                custom_applicability.append(final_assessment)
                if final_assessment.reason_code == "extracted" and (
                    _roi_requiredness(config, "Custom", base)
                    == Requiredness.ANALYSIS_REQUIRED
                ):
                    observed_required.add(base)

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

            try:
                rt = RTStructBuilder.create_from(
                    dicom_series_path=str(contracted_ct_dir),
                    rt_struct_path=str(rs_custom),
                )
            except Exception as exc:
                if any(_roi_requiredness(config, "Custom", name) == Requiredness.ANALYSIS_REQUIRED for name in wanted_names):
                    raise RadiomicsCourseExtractionError(
                        f"Required custom RTSTRUCT could not be read: {exc}"
                    ) from exc
                for name in wanted_names:
                    roi_failures.append({"roi_name": name, "source": "Custom", "status": "failed", "failure_kind": "custom_read", "reason_code": "failed_custom_read", "reason": str(exc)})
                rt = None
            if rt is None:
                wanted_names = []
            for roi_name in wanted_names:
                required_custom = _roi_requiredness(config, "Custom", roi_name) == Requiredness.ANALYSIS_REQUIRED
                if ''.join(ch for ch in roi_name.lower() if ch.isalnum()) in normalized_skip_rois:
                    _append_declared_skip(
                        "Custom", rs_custom, roi_name,
                        required=_roi_requiredness(config, "Custom", roi_name) == Requiredness.ANALYSIS_REQUIRED,
                    )
                    continue
                try:
                    mask = rt.get_roi_mask_by_name(roi_name)
                except Exception as exc:
                    if required_custom:
                        raise RadiomicsCourseExtractionError(
                            f"Expected custom ROI {roi_name!r} in {rs_custom} could not be read: {exc}"
                        ) from exc
                    roi_failures.append({"roi_name": roi_name, "source": "Custom", "status": "failed", "failure_kind": "custom_read", "reason_code": "failed_custom_read", "reason": str(exc)})
                    continue
                if mask is None:
                    if required_custom:
                        raise RadiomicsCourseExtractionError(
                            f"Expected custom ROI {roi_name!r} in {rs_custom} did not provide a mask"
                        )
                    roi_failures.append({"roi_name": roi_name, "source": "Custom", "status": "failed", "failure_kind": "custom_read", "reason_code": "failed_custom_read", "reason": "mask is absent"})
                    continue
                mask_bool = np.asarray(mask).astype(bool)
                if not mask_bool.any():
                    if required_custom:
                        raise RadiomicsCourseExtractionError(
                            f"Expected custom ROI {roi_name!r} in {rs_custom} produced an empty required mask [ROI_MASK_EMPTY_AFTER_RASTERIZATION]"
                        )
                    roi_failures.append({"roi_name": roi_name, "source": "Custom", "status": "failed", "failure_kind": "degenerate_mask", "reason_code": "ROI_MASK_EMPTY_AFTER_RASTERIZATION", "reason": "mask is empty"})
                    continue
                tasks.append(
                    _make_task(
                        "Custom",
                        rs_custom,
                        roi_name,
                        mask_bool,
                        required=_roi_requiredness(config, "Custom", roi_name) == Requiredness.ANALYSIS_REQUIRED,
                    )
                )
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
        model_dispositions: List[Dict[str, str]] = []
        source = f"CustomModel:{model_name}"
        try:
            _source_fingerprint(rs_path)
            masks = _rtstruct_masks(
                contracted_ct_dir,
                rs_path,
                skip_rois=configured_skip_rois,
                expected_rois=custom_model_expected_rois[model_name],
                failure_outcomes=model_dispositions,
                requiredness_by_roi={
                    name: Requiredness.ANALYSIS_REQUIRED
                    for name in custom_model_expected_rois[model_name]
                },
            )
            for disposition in model_dispositions:
                disposition["source"] = source
                task = _make_task(
                    source, rs_path, disposition["roi_name"],
                    np.zeros((0, 0, 0), dtype=bool), required=True,
                )
                rows.extend(disposition_rows_for_arms(
                    {**_common_metadata(task),
                     "roi_structural_code": disposition.get("structural_code")},
                    decision=task.decision,
                    disposition=disposition["status"],
                    detail=disposition["reason"],
                    failure_kind=disposition["failure_kind"],
                    run_identifier=run_identifier,
                    code_revision=code_revision,
                    native_voxel_count=None,
                    required=task.required,
                    effective_hashes=task.effective_parameter_hashes,
                    configured_parameter_hashes=task.configured_parameter_hashes,
                ))
            extraction_failures = [
                item for item in model_dispositions
                if item["status"] != "nonvolumetric_nonmeasurement"
            ]
            roi_failures.extend(extraction_failures)
            counts = source_counts.setdefault(
                source, {"attempted": 0, "extracted": 0, "failed": 0}
            )
            counts["attempted"] += len(extraction_failures)
            counts["failed"] += len(extraction_failures)
        except Exception as exc:
            _invalidate_radiomics_outputs(out_path)
            if isinstance(exc, RadiomicsCourseExtractionError):
                raise
            raise RadiomicsCourseExtractionError(
                f"Failed to prepare custom-model RTSTRUCT masks for {course_dir}: {exc}"
            ) from exc
        for roi_name in custom_model_expected_rois[model_name]:
            if ''.join(ch for ch in roi_name.lower() if ch.isalnum()) in normalized_skip_rois:
                _append_declared_skip(
                    f"CustomModel:{model_name}", rs_path, roi_name, required=True
                )
        for roi, mask in masks.items():
            tasks.append(
                _make_task(
                    f"CustomModel:{model_name}",
                    rs_path,
                    roi,
                    mask,
                    required=True,
                )
            )

    missing_required = [
        requirement.canonical_name
        for requirement in required_contract
        if requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
        and requirement.canonical_name not in observed_required
    ]
    if missing_required:
        _finalize_ct_ledger(extracted=False, technical=True)
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            "Required ROI(s) are not declared in the current analysis sources "
            "[REQUIRED_ROI_NOT_DECLARED]: " + ", ".join(sorted(missing_required))
        )

    _verify_source_fingerprints()
    expected_keys = expected_publication_keys(
        [*(_common_metadata(task) for task in tasks), *rows]
    )
    task_by_base = {
        (
            course_dir.parent.name,
            course_dir.name,
            series_uid,
            task.source,
            task.mask_identity,
            task.roi_name,
            task.stable_roi_identifier,
        ): task
        for task in tasks
    }
    if existing_df is not None:
        resume_error: Optional[Exception] = None
        try:
            existing_keys = _resume_identity_pairs(existing_df)
            expected_config_hashes: Dict[tuple[str, ...], str] = {
                (*base, arm): task.configured_parameter_hashes[arm]
                for base, task in task_by_base.items()
                for arm in CT_EXTRACTION_ARMS
            }
            expected_config_hashes.update(
                {
                    publication_key(record): str(record["configured_parameter_hash"])
                    for record in rows
                }
            )
            current_dispositions = {
                publication_key(record): record for record in rows
            }
            for record in existing_df.to_dict("records"):
                key = publication_key(record)
                if str(record.get("configured_parameter_hash") or "") != str(
                    expected_config_hashes.get(key) or ""
                ):
                    raise ValueError("configured radiomics parameter hash is stale")
                task = task_by_base.get(key[:-1])
                if task is not None:
                    if record.get("configured_minimum_native_voxel_count") != _derive_voxel_limits(config)[0]:
                        raise ValueError("configured native-mask minimum is stale")
                    expected_execution = {
                        "source_content_sha256": task.source_content_sha256,
                        "code_revision": code_revision,
                        "effective_parameter_hash": task.effective_parameter_hashes[key[-1]],
                    }
                    for field, expected in expected_execution.items():
                        if str(record.get(field) or "") != str(expected):
                            raise ValueError(f"current measured source {field} is stale")
                current = current_dispositions.get(key)
                if current is not None:
                    for field in (
                        "source_content_sha256", "extraction_status",
                        "roi_structural_code", "code_revision",
                        "effective_parameter_hash",
                    ):
                        if current.get(field) is None:
                            continue
                        if str(record.get(field) or "") != str(current[field]):
                            raise ValueError(f"current source disposition {field} is stale")
        except ValueError as exc:
            resume_error = exc
            existing_keys = set()
        if resume_error is None and expected_keys and existing_keys == expected_keys:
            outcome = outcome_from_output(
                out_path,
                required_by_identity={
                    (*base, arm): task.required
                    for base, task in task_by_base.items()
                    for arm in CT_EXTRACTION_ARMS
                },
            )
            # Reconstruct denominators from the admitted table, not the presence
            # of an earlier ledger or newly prepared task placeholders.
            rows[:] = existing_df.to_dict("records")
            roi_failures[:] = list(outcome.roi_failures)
            try:
                _finalize_ct_ledger(
                    extracted=outcome.extracted_count > 0,
                    technical=bool(roi_failures),
                )
            except Exception:
                _invalidate_radiomics_outputs(out_path)
                raise
            return outcome
        logger.warning(
            "Invalidating incomplete or stale resume publication for %s: expected %d "
            "full ROI-arm identities, found %d%s",
            course_dir,
            len(expected_keys),
            len(existing_keys),
            f"; {resume_error}" if resume_error is not None else "",
        )
        _invalidate_radiomics_outputs(out_path)
        existing_df = None
    def _do_ct_task(task: _DirectCtTask) -> List[Dict[str, Any]]:
        effective_hashes = task.effective_parameter_hashes
        try:
            min_voxels, _ = _derive_voxel_limits(config)
            voxel_count = int(np.asarray(task.mask).astype(bool).sum())
            if voxel_count < min_voxels:
                return disposition_rows_for_arms(
                    {**_common_metadata(task), "roi_structural_code": "ROI_MASK_BELOW_MIN_VOXELS"},
                    decision=task.decision,
                    disposition="below_minimum_voxels",
                    detail=(
                        f"ROI contains {voxel_count} voxels; configured minimum is {min_voxels}"
                    ),
                    failure_kind="degenerate_mask",
                    run_identifier=run_identifier,
                    code_revision=code_revision,
                    native_voxel_count=voxel_count,
                    required=task.required,
                    effective_hashes=effective_hashes,
                    configured_parameter_hashes=task.configured_parameter_hashes,
                )

            try:
                spacing = tuple(float(x) for x in img.GetSpacing())
            except Exception:
                spacing = (1.0, 1.0, 1.0)
            try:
                resampled_spacing = tuple(
                    float(value)
                    for value in (
                        extractor.settings.get("resampledPixelSpacing") or spacing
                    )
                )
            except Exception:
                resampled_spacing = spacing
            try:
                pad_distance = int(
                    math.ceil(float(extractor.settings.get("padDistance", 5)))
                )
            except (TypeError, ValueError, OverflowError):
                pad_distance = 5
            work_estimate = estimate_resampled_bounding_box(
                task.mask,
                native_spacing_xyz=spacing,
                resampled_spacing_xyz=resampled_spacing,
                array_axis_to_xyz=(1, 0, 2),
                pad_distance=pad_distance,
            )
            max_bbox_voxels = resolve_max_resampled_bbox_voxels(config)
            if (work_estimate.estimated_resampled_bbox_voxels > max_bbox_voxels
                    and not permits_second_stage(
                        work_estimate, task.mask, native_spacing_xyz=spacing,
                        array_axis_to_xyz=(1, 0, 2), settings=extractor.settings,
                        image_types=getattr(extractor, "enabledImagetypes", {}),
                        limit=max_bbox_voxels)):
                detail = (
                    f"ROI {task.roi_name} requires an estimated padded resampled "
                    f"bounding box of "
                    f"{work_estimate.estimated_resampled_bbox_voxels} voxels "
                    f"({work_estimate.estimated_resampled_bbox_shape}); configured "
                    f"maximum is {max_bbox_voxels}. Full configured radiomics was "
                    "not started."
                )
                return disposition_rows_for_arms(
                    {
                        **_common_metadata(task),
                        "roi_structural_code": RESAMPLED_BBOX_LIMIT_CODE,
                        **work_estimate.metadata(limit=max_bbox_voxels),
                    },
                    decision=task.decision,
                    disposition="failed",
                    detail=detail,
                    failure_kind="resource_limit",
                    run_identifier=run_identifier,
                    code_revision=code_revision,
                    native_voxel_count=voxel_count,
                    required=task.required,
                    effective_hashes=effective_hashes,
                    configured_parameter_hashes=task.configured_parameter_hashes,
                )
            use_large = _uses_large_extractor(task.roi_name, task.mask)

            def _factory():
                candidate = (
                    _extractor_large_roi(config, "CT")
                    if use_large
                    else _extractor(config, "CT")
                )
                if candidate is None:
                    raise RuntimeError(
                        f"No radiomics extractor available for {task.source}/{task.roi_name}"
                    )
                return candidate

            mask_image = _mask_from_array_like(img, task.mask)
            return extract_ct_roi_arms(
                img,
                mask_image,
                factory=_factory,
                decision=task.decision,
                common_metadata={**_common_metadata(task),
                                 "_resource_guard_legacy": legacy_rejection(
                                     work_estimate, max_bbox_voxels, task.roi_name)},
                run_identifier=run_identifier,
                code_revision=code_revision,
                native_voxel_count=voxel_count,
                required=task.required,
                configured_parameter_hashes=task.configured_parameter_hashes,
            )
        except Exception as exc:
            detail = f"Radiomics failed for {task.source}/{task.roi_name}: {exc}"
            if not task.required:
                return disposition_rows_for_arms(
                    _common_metadata(task),
                    decision=task.decision,
                    disposition="failed",
                    detail=detail,
                    failure_kind="extraction_error",
                    run_identifier=run_identifier,
                    code_revision=code_revision,
                    native_voxel_count=int(np.asarray(task.mask).astype(bool).sum()),
                    required=False,
                    effective_hashes=effective_hashes,
                    configured_parameter_hashes=task.configured_parameter_hashes,
                )
            raise RuntimeError(detail) from exc
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
        for task, records in zip(tasks, results):
            counts = source_counts.setdefault(
                task.source,
                {"attempted": 0, "extracted": 0, "failed": 0},
            )
            counts["attempted"] += 1
            if not records:
                if task.required:
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"Radiomics course {course_dir} is incomplete: required ROI "
                        f"{task.source}/{task.roi_name} returned no outcome records"
                    )
                records = disposition_rows_for_arms(
                    _common_metadata(task),
                    decision=task.decision,
                    disposition="failed",
                    detail="worker returned no outcome records",
                    failure_kind="extraction_error",
                    run_identifier=run_identifier,
                    code_revision=code_revision,
                    native_voxel_count=int(np.asarray(task.mask).astype(bool).sum()),
                    required=False,
                    effective_hashes=task.effective_parameter_hashes,
                    configured_parameter_hashes=task.configured_parameter_hashes,
                )
            if len(records) != len(CT_EXTRACTION_ARMS) or {
                str(record.get("extraction_arm")) for record in records
            } != set(CT_EXTRACTION_ARMS):
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Radiomics course {course_dir} returned an incomplete arm set for "
                    f"{task.source}/{task.roi_name}"
                )
            failing_record = next(
                (
                    record
                    for record in records
                    if record.get("extraction_status") not in (None, "success")
                ),
                None,
            )
            if failing_record is not None:
                counts["failed"] += 1
                status = failing_record.get("extraction_status")
                detail = str(failing_record.get("extraction_status_detail", "unknown error"))
                if task.required and not extraction_status_is_nonfatal_for_required(status):
                    _invalidate_radiomics_outputs(out_path)
                    raise RadiomicsCourseExtractionError(
                        f"Radiomics course {course_dir} is incomplete: required ROI "
                        f"{task.source}/{task.roi_name} failed: {detail}"
                    )
                roi_failures.append(
                    {
                        "roi_name": task.roi_name,
                        "source": task.source,
                        "status": str(status),
                        "failure_kind": str(
                            failing_record.get("extraction_failure_kind", "extraction_error")
                        ),
                        "reason": detail,
                    }
                )
            else:
                counts["extracted"] += 1
            rows.extend(records)
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
        _finalize_ct_ledger(extracted=False, technical=bool(roi_failures), indeterminate=False)
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
    _verify_source_fingerprints()
    try:
        attach_acquisition_descriptor(
            rows,
            acquisition_descriptor,
        )
        import pandas as pd
        df = pd.DataFrame(rows)
        write_ct_publication_atomic(df, out_path, expected_keys=expected_keys)
        rows[:] = read_authoritative_ct_publication(
            out_path.with_suffix(".parquet")
        ).to_dict("records")
        _finalize_ct_ledger(
            extracted=outcome.extracted_count > 0,
            technical=bool(roi_failures),
            indeterminate=False,
        )
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

MR_AUTO_SOURCE = "AutoTS_total_mr"


def _write_mr_ledger(course_dirs: Any, *, course_state: Mapping[str, Any], roi_rows: Iterable[Mapping[str, Any]] = ()) -> None:
    ledger = DenominatorLedger()
    course_id = str(getattr(course_dirs, "root", "").name if hasattr(getattr(course_dirs, "root", None), "name") else getattr(course_dirs, "root", ""))
    patient_id = str(Path(getattr(course_dirs, "root", ".")).parent.name)
    ledger.record_course(course_id, patient_id, screened=True, **dict(course_state))
    for row in roi_rows:
        ledger.record_roi(
            course_id,
            patient_id,
            str(row.get("roi_name", "")),
            reason_code=str(row.get("reason_code", "failed_radiomics_extraction")),
            disposition=str(row.get("disposition", "excluded")),
            **{key: value for key, value in row.items() if key not in {"roi_name", "reason_code", "disposition"}},
        )
    try:
        write_modality_ledger(Path(course_dirs.root) / "metadata", ledger, "MR")
    except Exception as exc:
        logger.warning("Could not write MR denominator ledger for %s: %s", course_dirs.root, exc)


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
        source_failures: list[dict[str, str]] = []
        try:
            candidate_masks = _rtstruct_masks(
                series_dir,
                rtstruct_path,
                best_effort=True,
                failure_outcomes=source_failures,
            )
        except RadiomicsCourseExtractionError as exc:
            source_failures.append({
                "roi_name": rtstruct_path.name,
                "source": MR_AUTO_SOURCE,
                "status": "failed",
                "detail": str(exc),
                "reason_code": "failed_source_read",
            })
            candidate_masks = {}
        if failure_outcomes is not None:
            failure_outcomes.extend(source_failures)
        masks.update(candidate_masks)
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
                     "status": "failed", "detail": f"unreadable: {exc}",
                     "reason_code": "failed_source_read"}
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
                     "status": "below_minimum_voxels", "detail": "mask is empty",
                     "reason_code": "not_computed_valid_empty_scope"}
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
    mr_failures: List[Dict[str, str]] = []
    mr_ledger_rows: List[Dict[str, Any]] = []
    configured_params = getattr(config, 'radiomics_params_file_mr', None)
    if configured_params is not None and not Path(configured_params).exists():
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Configured required MR radiomics parameter path is missing: {configured_params}"
        )
    if not mr_root.exists():
        _invalidate_radiomics_outputs(out_path)
        _write_mr_ledger(
            course_dirs,
            course_state={
                "in_scope": False,
                "out_of_scope": True,
                "adequate_coverage": False,
                "insufficient_coverage": False,
                "valid_derivation": False,
                "technical_exclusion": False,
                "indeterminate": False,
                "extracted": False,
                "reason_code": "not_applicable_modality",
            },
        )
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
    run_identifier = new_run_identifier()
    code_revision = current_code_revision()
    for series_root in sorted(p for p in mr_root.iterdir() if p.is_dir()):
        nifti_dir = series_root / 'NIFTI'
        dicom_dir = series_root / 'DICOM'
        if not dicom_dir.exists():
            dicom_dir = series_root
        seg_dir = series_root / 'Segmentation_TotalSegmentator'
        if not nifti_dir.exists() or not seg_dir.exists():
            mr_failures.append({
                "roi_name": series_root.name,
                "source": MR_AUTO_SOURCE,
                "status": "failed",
                "detail": "MR segmentation is missing or stale",
                "reason_code": "failed_source_segmentation",
            })
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
            failure = {
                "roi_name": series_uid,
                "source": MR_AUTO_SOURCE,
                "status": "failed",
                "detail": f"MR series is present but unreadable: {source_dir}",
                "reason_code": "failed_source_read",
            }
            mr_failures.append(failure)
            _write_mr_ledger(
                course_dirs,
                course_state={
                    "in_scope": True, "out_of_scope": False,
                    "adequate_coverage": False, "insufficient_coverage": True,
                    "valid_derivation": False, "technical_exclusion": True,
                    "indeterminate": False, "extracted": False,
                    "reason_code": "failed_source_read",
                },
                roi_rows=mr_failures,
            )
            _invalidate_radiomics_outputs(out_path)
            if mr_required:
                raise RadiomicsCourseExtractionError(
                    f"MR series is present but unreadable for radiomics: {source_dir}"
                )
            continue
        base_name = _strip_nii_name(nifti_path)
        masks = _collect_total_mr_masks(source_dir, seg_dir, mr_failures)
        if not masks:
            continue
        for roi_name, mask in masks.items():
            try:
                m_img = _mask_from_array_like(img, mask)
                res = extractor.execute(img, m_img)
                rec = normalize_radiomics_result(res)
                rec.update({
                    'patient_id': getattr(course, 'patient_id', course_dirs.root.parent.name),
                    'course_id': getattr(course, 'course_id', course_dirs.root.name),
                    'modality': 'MR',
                    'segmentation_source': 'AutoTS_total_mr',
                    'roi_name': roi_name,
                    'series_dir': str(source_dir),
                    'series_uid': series_uid,
                    'nifti_path': str(nifti_path),
                    **_mr_parameter_provenance(
                        config,
                        extractor,
                        normalize_override=normalize_override,
                        run_identifier=run_identifier,
                        code_revision=code_revision,
                    ),
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
                     "status": "failed", "detail": str(exc)[:200],
                     "reason_code": "failed_radiomics_extraction"}
                )
    if not rows:
        _invalidate_radiomics_outputs(out_path)
        reason_code = (
            str(mr_failures[0].get("reason_code"))
            if mr_failures else "not_computed_valid_empty_scope"
        )
        _write_mr_ledger(
            course_dirs,
            course_state={
                "in_scope": True, "out_of_scope": False,
                "adequate_coverage": False,
                "insufficient_coverage": any(
                    str(row.get("reason_code")) in {"failed_source_read", "failed_source_segmentation"}
                    for row in mr_failures
                ),
                "valid_derivation": False,
                "technical_exclusion": any(
                    str(row.get("reason_code")) not in {
                        "not_computed_valid_empty_scope", "not_applicable_scope", "not_applicable_anatomy"
                    }
                    for row in mr_failures
                ),
                "indeterminate": reason_code == "indeterminate_applicability",
                "extracted": False,
                "reason_code": reason_code,
            },
            roi_rows=mr_failures,
        )
        if mr_required:
            required_names = [
                requirement.canonical_name
                for requirement in requirements_from_contract(_analysis_contract(config), "MR")
                if requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
            ]
            detail = (
                "[REQUIRED_ROI_NOT_DECLARED] " + ", ".join(required_names)
                if required_names else "no eligible MR rows"
            )
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but produced no eligible rows for {course_dirs.root}: {detail}"
            )
        return None

    required_mr = [
        requirement for requirement in requirements_from_contract(_analysis_contract(config), "MR")
        if requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
    ]
    row_names = {str(row.get("roi_name", "")) for row in rows}
    missing_required_mr = [
        requirement.canonical_name
        for requirement in required_mr
        if not any(
            "".join(ch for ch in accepted.casefold() if ch.isalnum())
            == "".join(ch for ch in name.casefold() if ch.isalnum())
            for accepted in requirement.accepted_names for name in row_names
        )
    ]
    if missing_required_mr:
        _invalidate_radiomics_outputs(out_path)
        _write_mr_ledger(
            course_dirs,
            course_state={
                "in_scope": True, "out_of_scope": False,
                "adequate_coverage": False, "insufficient_coverage": True,
                "valid_derivation": False, "technical_exclusion": True,
                "indeterminate": False, "extracted": False,
                "reason_code": "REQUIRED_ROI_NOT_DECLARED",
            },
            roi_rows=[{
                "roi_name": name,
                "reason_code": "REQUIRED_ROI_NOT_DECLARED",
                "disposition": "excluded",
                "detail": "required MR ROI is absent",
            } for name in missing_required_mr],
        )
        raise RadiomicsCourseExtractionError(
            "Required MR ROI(s) are absent [REQUIRED_ROI_NOT_DECLARED]: "
            + ", ".join(missing_required_mr)
        )
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        parquet_path = write_radiomics_feature_table_atomic(df, out_path)
        logger.debug("Saved and schema-validated MR Parquet: %s", parquet_path)
        _write_mr_ledger(
            course_dirs,
            course_state={
                "in_scope": True, "out_of_scope": False,
                "adequate_coverage": True, "insufficient_coverage": bool(mr_failures),
                "valid_derivation": True, "technical_exclusion": bool(mr_failures),
                "indeterminate": any(row.get("reason_code") == "indeterminate_applicability" for row in mr_failures),
                "extracted": True, "reason_code": "extracted",
            },
            roi_rows=(
                mr_failures
                + [
                    {"roi_name": row.get("roi_name", ""), "reason_code": "extracted", "disposition": "extracted"}
                    for row in rows
                ]
            ),
        )
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


def _mr_parameter_arm(normalize_override: Optional[bool]) -> str:
    if normalize_override is None:
        return "mr_configured"
    return "mr_normalized" if normalize_override else "mr_native_intensity"


def _mr_parameter_provenance(
    config: PipelineConfig,
    extractor: Any,
    *,
    normalize_override: Optional[bool],
    run_identifier: str,
    code_revision: str,
) -> dict[str, str]:
    arm = _mr_parameter_arm(normalize_override)
    params_path = _get_params_file(config, "MR")
    return {
        "extraction_arm": arm,
        "configured_parameter_hash": configured_parameter_hash(
            params_path,
            arm=arm,
            window=None,
            large_roi=False,
        ),
        "effective_parameter_hash": effective_parameter_hash(
            extractor,
            arm=arm,
            window=None,
        ),
        "run_identifier": run_identifier,
        "code_revision": code_revision,
    }


def _mr_resume_parameter_provenance_is_current(
    dataframe: Any,
    params_path: Optional[Path],
    *,
    expected_arm: Optional[str] = None,
    expected_effective_parameter_hash: Optional[str] = None,
    expected_code_revision: Optional[str] = None,
    expected_sources: Optional[set[tuple[str, str]]] = None,
) -> bool:
    """Return whether a stored MR table still carries the current identity.

    Without keywords this only asserts the parameter-provenance contract the DAG
    parameter manifest checks. The keyword expectations bind the decision to what
    is observed *now*: the extraction arm, the effective extractor configuration,
    the code revision, and the exact source inventory. A table written by another
    revision, by a differently configured extractor, or from different source
    bytes is therefore never reusable, even when every UID is unchanged.
    """
    required = {
        "extraction_arm",
        "configured_parameter_hash",
        "effective_parameter_hash",
        "run_identifier",
        "code_revision",
    }
    if expected_sources is not None:
        required |= {"mask_path_source", "source_content_sha256"}
    columns = {str(column) for column in dataframe.columns}
    if dataframe.empty or not required.issubset(columns):
        return False
    observed_sources: set[tuple[str, str]] = set()
    for row in dataframe.to_dict("records"):
        arm = str(row.get("extraction_arm") or "")
        if arm not in {"mr_configured", "mr_normalized", "mr_native_intensity"}:
            return False
        if expected_arm is not None and arm != expected_arm:
            return False
        expected = configured_parameter_hash(
            params_path,
            arm=arm,
            window=None,
            large_roi=False,
        )
        if str(row.get("configured_parameter_hash") or "") != expected:
            return False
        for field in ("effective_parameter_hash", "run_identifier", "code_revision"):
            if not str(row.get(field) or "").strip():
                return False
        if (
            expected_effective_parameter_hash is not None
            and str(row.get("effective_parameter_hash")) != expected_effective_parameter_hash
        ):
            return False
        if (
            expected_code_revision is not None
            and str(row.get("code_revision")) != expected_code_revision
        ):
            return False
        if expected_sources is not None:
            source_path = str(row.get("mask_path_source") or "").strip()
            source_digest = str(row.get("source_content_sha256") or "").strip()
            if not source_path or not source_digest:
                return False
            observed_sources.add((source_path, source_digest))
    # Set equality, not containment: a changed or removed source drops out of the
    # current inventory, and a newly appeared source is absent from the stored
    # table. Both must force re-extraction.
    if expected_sources is not None and observed_sources != expected_sources:
        return False
    return True


@dataclass(frozen=True)
class _MRMaskSource:
    """One MR mask source, bound to the exact bytes that define it."""

    kind: str  # "manual" | "auto_dicom_seg" | "auto_nifti"
    path: Path
    files: tuple[Path, ...]
    content_sha256: str

    @property
    def identity(self) -> tuple[str, str]:
        return (str(self.path), self.content_sha256)


def _mr_content_digest(files: Iterable[Path]) -> str:
    """Digest the exact current bytes of the files defining one mask source.

    A single-file source keeps the plain content SHA-256 of that file, which is
    the fingerprint every other radiomics publication records. A multi-file
    source (a NIfTI segmentation directory) folds each member's name and content
    digest in deterministic order, so adding, removing, or editing any member
    changes the source identity.
    """
    members = sorted((Path(path) for path in files), key=lambda item: str(item))
    if len(members) == 1:
        return hashlib.sha256(members[0].read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for path in members:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _mr_nifti_seg_files(seg_dir: Path) -> tuple[Path, ...]:
    """Return every file the NIfTI segmentation loader may consume."""
    candidates: set[Path] = set()
    for pattern in ("*.nii", "*.nii.gz", "*.json"):
        candidates.update(path for path in seg_dir.glob(pattern) if path.is_file())
    return tuple(sorted(candidates, key=lambda item: str(item)))


def _mr_mask_sources(
    config: PipelineConfig, series: MRSeries, out_root: Path, *, for_uid: str
) -> tuple[_MRMaskSource, ...]:
    """Fingerprint every mask source this series will actually be read from.

    The auto source is chosen by presence, not by yield: when the DICOM-SEG file
    exists it is authoritative and the NIfTI directory is not a second source.
    That keeps the inventory decidable from the filesystem alone, so a resume
    decision can be compared against it before anything is decoded, and it stops
    the same TotalSegmentator label being published twice from two encodings.
    """
    sources: list[_MRMaskSource] = []
    seen: set[Path] = set()
    for rs in _find_mr_manual_rs(config.dicom_root, series.patient_id, for_uid):
        resolved = Path(rs)
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            digest = _mr_content_digest([resolved])
        except OSError as exc:
            raise RadiomicsCourseExtractionError(
                f"Cannot fingerprint MR manual RTSTRUCT {resolved}: {exc}"
            ) from exc
        sources.append(
            _MRMaskSource("manual", resolved, (resolved,), digest)
        )

    seg_dicom = out_root / 'TotalSegmentator_total_mr_DICOM' / 'segmentations.dcm'
    seg_nifti_dir = out_root / 'TotalSegmentator_total_mr_NIFTI'
    if seg_dicom.exists():
        try:
            digest = _mr_content_digest([seg_dicom])
        except OSError as exc:
            raise RadiomicsCourseExtractionError(
                f"Cannot fingerprint MR DICOM-SEG source {seg_dicom}: {exc}"
            ) from exc
        sources.append(
            _MRMaskSource("auto_dicom_seg", seg_dicom, (seg_dicom,), digest)
        )
    elif seg_nifti_dir.is_dir():
        files = _mr_nifti_seg_files(seg_nifti_dir)
        if files:
            try:
                digest = _mr_content_digest(files)
            except OSError as exc:
                raise RadiomicsCourseExtractionError(
                    f"Cannot fingerprint MR NIfTI segmentation source {seg_nifti_dir}: {exc}"
                ) from exc
            sources.append(
                _MRMaskSource("auto_nifti", seg_nifti_dir, files, digest)
            )
    return tuple(sources)


def _mr_verify_mask_sources(sources: Iterable[_MRMaskSource]) -> None:
    """Fail when any mask source changed after it was read for this run."""
    for source in sources:
        if source.kind == "auto_nifti":
            current_files = _mr_nifti_seg_files(source.path)
        else:
            current_files = source.files
        if set(current_files) != set(source.files):
            raise RadiomicsCourseExtractionError(
                f"MR mask source inventory changed during radiomics: {source.path}"
            )
        try:
            current_digest = _mr_content_digest(current_files)
        except OSError as exc:
            raise RadiomicsCourseExtractionError(
                f"MR mask source became unreadable during radiomics: {source.path}: {exc}"
            ) from exc
        if current_digest != source.content_sha256:
            raise RadiomicsCourseExtractionError(
                f"MR mask source changed during radiomics: {source.path}"
            )


def _mr_required_coverage_failures(
    requirements: Iterable[Any], rows: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Report every required MR ROI that the given rows do not actually cover.

    A required ROI is covered only by its own row. Another ROI extracting
    successfully never substitutes for it, and a row that records a technical
    failure for it does not count as coverage either.
    """
    materialized = list(rows)
    failures: list[str] = []
    for requirement in requirements:
        accepted = {
            _normalize_required_roi_name(name) for name in requirement.accepted_names
        }
        matches = [
            row
            for row in materialized
            if _normalize_required_roi_name(str(row.get("roi_name", ""))) in accepted
            and (
                not requirement.source
                or _normalize_required_roi_name(str(row.get("segmentation_source", "")))
                == _normalize_required_roi_name(requirement.source)
            )
        ]
        if not matches:
            failures.append(
                f"{requirement.canonical_name} [REQUIRED_ROI_NOT_DECLARED]"
            )
            continue
        if len(matches) > 1:
            failures.append(
                f"{requirement.canonical_name} [REQUIRED_ROI_AMBIGUOUS_MATCH]"
            )
            continue
        status = str(matches[0].get("extraction_status") or "").strip()
        if status != "success" and not extraction_status_is_nonfatal_for_required(status):
            detail = str(matches[0].get("extraction_status_detail") or "").strip()
            failures.append(
                f"{requirement.canonical_name} [{status or 'unknown_status'}]"
                + (f": {detail}" if detail else "")
            )
    return failures


def _mr_publication_identity_conflicts(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Report rows that share one full publication identity."""
    seen: Dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(
            str(row.get(column) or "")
            for column in (
                "segmentation_source",
                "roi_name",
                "stable_roi_identifier",
                "source_content_sha256",
            )
        )
        seen[key] = seen.get(key, 0) + 1
    return [
        f"{key[0]}/{key[1]} appears {count} times for one source identity"
        for key, count in sorted(seen.items())
        if count > 1
    ]


def radiomics_for_mr_series(config: PipelineConfig, series: MRSeries) -> Optional[Path]:
    """Extract MR radiomics for one directly addressed series.

    Every published row names the source bytes it came from, the effective
    extractor configuration that produced it, and the code revision of the run.
    Resume reuses a stored table only while all three still match the current
    observation and the current required-ROI contract is genuinely covered.
    Publication is refused when a source changes while the series is processed.

    Provenance limit: ``current_code_revision`` resolves the committed Git
    revision (or ``RTPIPELINE_CODE_REVISION``). It does not fingerprint
    uncommitted working-tree edits, so an unchanged revision does not by itself
    prove the executed code content was unchanged.
    """
    # Determine weighting to toggle normalization: T2 -> False, T1 -> True, else default False
    wt = _infer_mr_weighting(series.dir, series.series_uid)
    normalize_override = True if wt == 'T1' else False if wt == 'T2' else False
    out_root = config.output_root / series.patient_id / f"MR_{series.series_uid}"
    out_feat = out_root / 'radiomics_features_MR.xlsx'
    extractor = _extractor(config, 'MR', normalize_override=normalize_override)
    if extractor is None:
        _invalidate_radiomics_outputs(out_feat)
        if _mr_radiomics_required(config):
            raise RadiomicsCourseExtractionError(
                f"MR radiomics extraction is unavailable for {series.patient_id}/{series.series_uid}: "
                "extractor could not be constructed"
            )
        logger.warning("MR radiomics extraction unavailable for %s; stale outputs invalidated", series.dir)
        return None
    img = _load_series_image(series.dir, series.series_uid)
    if img is None:
        _invalidate_radiomics_outputs(out_feat)
        if _mr_radiomics_required(config):
            raise RadiomicsCourseExtractionError(
                f"MR radiomics required but no MR image could be loaded for "
                f"{series.patient_id}/{series.series_uid}: {series.dir}"
            )
        logger.warning("No MR image for radiomics in %s; stale outputs invalidated", series.dir)
        return None

    arm = _mr_parameter_arm(normalize_override)
    params_path = _get_params_file(config, "MR")
    code_revision = current_code_revision()
    required_contract = tuple(
        requirement
        for requirement in requirements_from_contract(_analysis_contract(config), "MR")
        if requirement.requiredness == Requiredness.ANALYSIS_REQUIRED
    )
    for_uid = _mr_series_for_uid(series.dir, series.series_uid) or ''
    try:
        current_effective_hash = effective_parameter_hash(extractor, arm=arm, window=None)
        sources = _mr_mask_sources(config, series, out_root, for_uid=for_uid)
    except Exception as exc:
        _invalidate_radiomics_outputs(out_feat)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"Cannot establish MR radiomics execution or source identity for "
            f"{series.patient_id}/{series.series_uid}: {exc}"
        ) from exc

    if getattr(config, 'resume', False) and out_feat.exists():
        try:
            import pandas as pd

            parquet_path = out_feat.with_suffix('.parquet')
            assert_radiomics_arrow_schema(parquet_path)
            existing = pd.read_parquet(parquet_path, engine="pyarrow")
            if not _mr_resume_parameter_provenance_is_current(
                existing,
                params_path,
                expected_arm=arm,
                expected_effective_parameter_hash=current_effective_hash,
                expected_code_revision=code_revision,
                expected_sources={source.identity for source in sources},
            ):
                raise ValueError(
                    "MR execution identity, parameter provenance or source inventory is stale"
                )
            unmet = _mr_required_coverage_failures(
                required_contract, existing.to_dict("records")
            )
            if unmet:
                raise ValueError(
                    "stored MR table does not cover the current required ROI contract: "
                    + "; ".join(unmet)
                )
        except Exception as exc:
            logger.info("Rejecting non-conforming MR radiomics resume output %s: %s", out_feat, exc)
            _invalidate_radiomics_outputs(out_feat)
        else:
            return out_feat

    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    run_identifier = new_run_identifier()
    parameter_provenance = _mr_parameter_provenance(
        config,
        extractor,
        normalize_override=normalize_override,
        run_identifier=run_identifier,
        code_revision=code_revision,
    )

    try:
        for source in sources:
            if source.kind == "manual":
                rows.extend(
                    _mr_manual_rows(
                        config,
                        series,
                        img,
                        extractor,
                        source,
                        parameter_provenance=parameter_provenance,
                    )
                )
            else:
                rows.extend(
                    _mr_auto_rows(
                        config,
                        series,
                        img,
                        source,
                        normalize_override=normalize_override,
                        parameter_provenance=parameter_provenance,
                        expected_effective_parameter_hash=current_effective_hash,
                    )
                )

        if not rows:
            _invalidate_radiomics_outputs(out_feat)
            if _mr_radiomics_required(config):
                raise RadiomicsCourseExtractionError(
                    f"MR radiomics required but produced no rows for "
                    f"{series.patient_id}/{series.series_uid}"
                )
            logger.warning("MR radiomics produced no rows for %s/%s; stale outputs invalidated",
                           series.patient_id, series.series_uid)
            return None

        conflicts = _mr_publication_identity_conflicts(rows)
        if conflicts:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics rows for {series.patient_id}/{series.series_uid} are not "
                "uniquely identified: " + "; ".join(conflicts)
            )
        divergent = {
            tuple(str(row.get(field) or "") for field in parameter_provenance)
            for row in rows
        }
        if divergent != {tuple(str(parameter_provenance[field]) for field in parameter_provenance)}:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics rows for {series.patient_id}/{series.series_uid} do not share "
                "one execution identity"
            )
        unmet = _mr_required_coverage_failures(required_contract, rows)
        if unmet:
            raise RadiomicsCourseExtractionError(
                "Required MR ROI(s) are not covered for "
                f"{series.patient_id}/{series.series_uid}: " + "; ".join(unmet)
            )
        _mr_verify_mask_sources(sources)

        import pandas as pd
        df = pd.DataFrame(rows)
        write_radiomics_feature_table_atomic(df, out_feat)
        return out_feat
    except RadiomicsFeatureTypeError:
        _invalidate_radiomics_outputs(out_feat)
        raise
    except RadiomicsCourseExtractionError:
        _invalidate_radiomics_outputs(out_feat)
        raise
    except Exception as exc:
        _invalidate_radiomics_outputs(out_feat)
        raise RadiomicsCourseExtractionError(
            f"Failed to write MR radiomics for {series.patient_id}/{series.series_uid}: {exc}"
        ) from exc


def _mr_manual_rows(
    config: PipelineConfig,
    series: MRSeries,
    img: Any,
    extractor: Any,
    source: _MRMaskSource,
    *,
    parameter_provenance: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Rows for one manual MR RTSTRUCT, each bound to its source and ROI identity.

    A required ROI that cannot be extracted raises here. The caller invalidates
    the stored workbook and Parquet for every raised failure, so no partially
    accounted table survives.
    """
    from .rtstruct_identity import require_rtstruct_identity

    rs = source.path
    try:
        sop_uid = require_rtstruct_identity(rs)
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"MR manual RTSTRUCT {rs} has no usable identity: {exc}"
        ) from exc
    try:
        roi_numbers = {
            observation.name: str(observation.roi_number)
            for observation in inspect_rtstruct(rs).named_rois
        }
    except Exception as exc:
        raise RadiomicsCourseExtractionError(
            f"Cannot read the ROI inventory of MR manual RTSTRUCT {rs}: {exc}"
        ) from exc

    def _manual_row(roi: str, *, status: str, failure_kind: Optional[str] = None,
                    detail: str = '', structural_code: Optional[str] = None,
                    row_sop_uid: Optional[str] = None,
                    roi_number: Optional[str] = None,
                    measurements: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        number = roi_number if roi_number is not None else roi_numbers.get(roi)
        rec: Dict[str, Any] = dict(measurements or {})
        rec.update({
            'modality': 'MR',
            'segmentation_source': 'Manual',
            'roi_name': roi,
            'series_dir': str(series.dir),
            'series_uid': series.series_uid,
            'mask_path_source': str(rs),
            'source_content_sha256': source.content_sha256,
            'rtstruct_sop_instance_uid': row_sop_uid or sop_uid,
            'stable_roi_identifier': (
                f"rtstruct_roi_number:{number}" if number is not None else None
            ),
            'extraction_status': status,
            **parameter_provenance,
        })
        if structural_code:
            rec['roi_structural_code'] = structural_code
        if failure_kind:
            rec['extraction_failure_kind'] = failure_kind
            rec['extraction_status_detail'] = detail
        return rec

    def _fail_closed_required(roi: str, detail: str) -> None:
        if _roi_requiredness(config, 'Manual', roi, modality='MR') == Requiredness.ANALYSIS_REQUIRED:
            raise RadiomicsCourseExtractionError(
                f"Required MR ROI {roi!r} in {rs} could not be extracted: {detail}"
            )

    rows: List[Dict[str, Any]] = []
    sink: List[Dict] = []
    masks = _rtstruct_masks(series.dir, rs, best_effort=True, failure_outcomes=sink)
    for entry in sink:
        reason = str(entry.get("reason", ""))
        if entry.get("status") == "nonvolumetric_nonmeasurement":
            # POINT and other open geometry is a nonmeasurement with null
            # measurements, not a technical extraction failure.
            rows.append(_manual_row(
                entry["roi_name"], status='nonvolumetric_nonmeasurement',
                failure_kind='nonvolumetric_geometry',
                detail=reason,
                structural_code=str(entry.get("structural_code", "")) or None,
                row_sop_uid=entry.get("rtstruct_sop_instance_uid"),
                roi_number=entry.get("roi_number"),
            ))
            continue
        rows.append(_manual_row(
            entry["roi_name"], status='failed',
            failure_kind=str(entry.get("failure_kind", "extraction_error")),
            detail=reason,
            structural_code=entry.get("structural_code"),
            row_sop_uid=entry.get("rtstruct_sop_instance_uid"),
            roi_number=entry.get("roi_number"),
        ))
        _fail_closed_required(entry["roi_name"], reason)
    for roi, mask in masks.items():
        try:
            m_img = _mask_from_array_like(img, mask)
            res = extractor.execute(img, m_img)
            rows.append(_manual_row(
                roi, status='success', measurements=normalize_radiomics_result(res),
            ))
        except RadiomicsFeatureTypeError:
            raise
        except Exception as e:
            logger.warning("Radiomics MR manual failed for %s: %s", roi, e)
            rows.append(_manual_row(
                roi, status='failed', failure_kind='extraction_error', detail=str(e),
            ))
            _fail_closed_required(roi, str(e))
    return rows


def _mr_auto_rows(
    config: PipelineConfig,
    series: MRSeries,
    img: Any,
    source: _MRMaskSource,
    *,
    normalize_override: Optional[bool],
    parameter_provenance: Mapping[str, str],
    expected_effective_parameter_hash: str,
) -> List[Dict[str, Any]]:
    """Rows for one TotalSegmentator MR source, accounting for every label.

    Every submitted label produces exactly one row. A label whose result the
    worker pool cannot return, and a schema violation raised inside a worker,
    both fail the series instead of silently shrinking the publication.
    """
    def _source_row(roi: str, *, status: str, failure_kind: Optional[str] = None,
                    detail: str = '', stable_identifier: str,
                    measurements: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        rec: Dict[str, Any] = dict(measurements or {})
        rec.update({
            'modality': 'MR',
            'segmentation_source': MR_AUTO_SOURCE,
            'roi_name': roi,
            'series_dir': str(series.dir),
            'series_uid': series.series_uid,
            'mask_path_source': str(source.path),
            'source_content_sha256': source.content_sha256,
            'stable_roi_identifier': stable_identifier,
            'extraction_status': status,
            **parameter_provenance,
        })
        if failure_kind:
            rec['extraction_failure_kind'] = failure_kind
            rec['extraction_status_detail'] = detail
        return rec

    def _unusable_source_row(detail: str) -> Dict[str, Any]:
        return _source_row(
            source.path.name, status='failed', failure_kind='source_read_error',
            detail=detail, stable_identifier=f"total_mr_source:{source.content_sha256}",
        )

    try:
        from .auto_rtstruct import _load_seg_dicom, _load_seg_nifti  # type: ignore
    except Exception as exc:  # noqa: BLE001 - an unusable decoder is a durable disposition
        return [_unusable_source_row(f"segmentation decoders are unavailable: {exc}")]
    if source.kind == "auto_dicom_seg":
        if _load_seg_dicom is None:
            return [_unusable_source_row("DICOM-SEG decoder is unavailable")]
        seg_img, label_map = _load_seg_dicom(source.path)
    else:
        if _load_seg_nifti is None:
            return [_unusable_source_row("NIfTI segmentation reader is unavailable")]
        seg_img, label_map = _load_seg_nifti(source.path, None)
    if seg_img is None:
        return [_unusable_source_row("segmentation source could not be decoded")]

    seg_img = _resample_to_reference(seg_img, img, nn=True)
    arr = sitk.GetArrayFromImage(seg_img)
    labels = [int(value) for value in np.unique(arr) if int(value) != 0]
    if not labels:
        return [_source_row(
            source.path.name, status='failed', failure_kind='degenerate_mask',
            detail='segmentation source declares no foreground label',
            stable_identifier=f"total_mr_source:{source.content_sha256}",
        )]

    label_map = dict(label_map or {})
    schema_failures: List[BaseException] = []

    def _do_lab(lab: int) -> Optional[Dict[str, Any]]:
        roi = label_map.get(lab, f'Segment_{lab}')
        identifier = f"total_mr_label:{lab}"
        try:
            mask = (arr == lab)
            if not mask.any():
                logger.warning("Radiomics MR total_mr label %s has an empty mask", lab)
                return _source_row(roi, status='failed', failure_kind='degenerate_mask',
                                   detail='empty mask', stable_identifier=identifier)
            ext = _extractor(config, 'MR', normalize_override=normalize_override)
            if ext is None:
                return _source_row(roi, status='failed', failure_kind='extraction_error',
                                   detail='no MR extractor could be constructed',
                                   stable_identifier=identifier)
            observed = effective_parameter_hash(
                ext, arm=str(parameter_provenance['extraction_arm']), window=None
            )
            if observed != expected_effective_parameter_hash:
                return _source_row(
                    roi, status='failed', failure_kind='extraction_error',
                    detail=(
                        "task extractor configuration does not match the run's "
                        f"effective parameter identity ({observed})"
                    ),
                    stable_identifier=identifier,
                )
            m_img = _mask_from_array_like(img, mask)
            res = ext.execute(img, m_img)
            return _source_row(roi, status='success', stable_identifier=identifier,
                               measurements=normalize_radiomics_result(res))
        except RadiomicsFeatureTypeError as exc:
            # The pool converts a raised exception into a dropped result. Carry
            # the schema violation out explicitly so it fails the series.
            schema_failures.append(exc)
            return None
        except Exception as e:
            logger.warning("Radiomics MR total_mr failed for label %s: %s", lab, e)
            return _source_row(roi, status='failed', failure_kind='extraction_error',
                               detail=str(e), stable_identifier=identifier)

    results = run_tasks_with_adaptive_workers(
        f"Radiomics MR (TotalSegmentator {'DICOM' if source.kind == 'auto_dicom_seg' else 'NIfTI'})",
        labels,
        _do_lab,
        max_workers=config.effective_workers(),
        logger=logger,
    )
    if schema_failures:
        raise schema_failures[0]
    unaccounted = [
        lab for lab, rec in zip(labels, list(results) + [None] * (len(labels) - len(results)))
        if not isinstance(rec, dict)
    ]
    if unaccounted:
        raise RadiomicsCourseExtractionError(
            f"MR radiomics did not account for TotalSegmentator label(s) "
            f"{', '.join(str(lab) for lab in unaccounted)} from {source.path}"
        )
    return [rec for rec in results if isinstance(rec, dict)]


def _radiomics_course_identity(course: Any) -> tuple[str, str]:
    root = Path(course.dirs.root)
    patient = str(getattr(course, "patient_id", "") or root.parent.name).strip()
    course_key = str(getattr(course, "course_key", "") or root.name).strip()
    if not patient or not course_key:
        raise RuntimeError("radiomics course is missing patient or course identity")
    return patient, course_key


def _full_organize_cohort(
    config: PipelineConfig, courses: list[Any]
) -> Optional[dict[str, Any]]:
    """Return the organize ledger only for a complete validated-course invocation."""

    from .organize_ledger import OrganizeLedgerError, ledger_path, validate_organize_ledger

    path = ledger_path(config.output_root)
    supplied = {_radiomics_course_identity(course) for course in courses}
    if len(supplied) != len(courses):
        raise RuntimeError("Radiomics invocation contains duplicate patient/course identities")
    if not path.exists():
        # A caller's course list cannot establish the intended source census or
        # prove that quarantined courses were retained in the denominator.
        return None
    try:
        source_bytes = path.read_bytes()
        ledger = validate_organize_ledger(json.loads(source_bytes))
    except (OrganizeLedgerError, OSError, ValueError) as exc:
        raise RuntimeError(f"Radiomics organize ledger is invalid: {exc}") from exc
    expected = {
        (entry["patient"], entry["course"])
        for entry in ledger["courses"]
        if entry["status"] == "validated"
    }
    if supplied != expected:
        return None
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    ledger["technical_quarantines"] = [
        {**entry, "source_record_sha256": source_sha256}
        for entry in ledger["technical_quarantines"]
    ]
    ledger["denominator_source_sha256"] = source_sha256
    return ledger


def _invalidate_unprovenanced_radiomics_cohort(path: Path) -> None:
    """Remove legacy/invalid aggregates while preserving a valid prior publication."""

    if is_valid_radiomics_cohort_table(path):
        logger.warning(
            "Preserving prior radiomics cohort aggregate with valid denominator "
            "provenance after the current invocation failed or was incomplete: %s",
            path,
        )
        return
    _invalidate_radiomics_outputs(path)


def run_radiomics(config: PipelineConfig, courses: List["object"], custom_structures_config: Optional[Path] = None) -> None:
    """Top-level orchestrator: per-course CT radiomics and per-series MR radiomics.
    'courses' elements are CourseOutput-like with patient_id, course_key, dirs.root.
    """
    _apply_radiomics_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))

    aggregate_path = config.output_root / 'Data' / 'radiomics_all.xlsx'
    organize_cohort = _full_organize_cohort(config, courses)

    # A requested radiomics stage must not preserve outputs from an older run when
    # neither the native nor isolated backend can execute.
    can_use_radiomics = _have_pyradiomics()
    if not can_use_radiomics:
        for course in courses:
            course_root = Path(course.dirs.root)
            _invalidate_radiomics_outputs(course_root / "radiomics_ct.xlsx")
            for mr_output in course_root.rglob("radiomics_features_MR.xlsx"):
                _invalidate_radiomics_outputs(mr_output)
        _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
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
        _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
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
        _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
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
                _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
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
                _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
                raise RuntimeError(
                    f"Required MR radiomics produced no workbook for {getattr(course, 'dirs', course)}; "
                    "cohort aggregation was not written"
                )

    # A filtered workflow invocation extracts one course but cannot prove the cohort
    # denominator. The final aggregate rule owns cohort publication in that mode.
    if organize_cohort is None:
        _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
        logger.info(
            "Skipping cohort radiomics publication because an authoritative organize "
            "ledger is absent or the invocation does not match its validated courses"
        )
        return

    missing_outputs = [
        _radiomics_course_identity(course)
        for course, outcome in zip(courses, ct_results)
        if getattr(outcome, "output_path", None) is None
    ]
    if missing_outputs:
        _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
        if len(missing_outputs) == len(courses):
            return
        formatted = ", ".join(
            f"{patient}/{course}" for patient, course in missing_outputs
        )
        raise RuntimeError(
            "CT radiomics produced no publication for organize-validated course(s); "
            f"cohort aggregation was not written: {formatted}"
        )

    provenance = build_radiomics_cohort_provenance(
        intended_count=int(organize_cohort["intended_course_count"]),
        validated_count=int(organize_cohort["validated_course_count"]),
        extracted_courses=[_radiomics_course_identity(course) for course in courses],
        technical_quarantines=organize_cohort["technical_quarantines"],
        denominator_source_sha256=organize_cohort[
            "denominator_source_sha256"
        ],
    )

    # Cohort merge. Every workbook declared EXTRACTED by a course is expected and
    # must be readable. Temporary files are validated before replacing a prior table.
    try:
        import pandas as _pd

        out_rows = []
        for course, outcome in zip(courses, ct_results):
            p = getattr(outcome, "output_path", None)
            if p is None:
                continue
            p = Path(p)
            try:
                df = read_authoritative_ct_publication(p.with_suffix(".parquet"))
            except Exception as exc:
                raise RadiomicsCourseExtractionError(
                    f"Expected course radiomics Parquet is unreadable: {p.with_suffix('.parquet')}: {exc}"
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
            all_df = attach_radiomics_cohort_provenance(all_df, provenance)
            write_radiomics_feature_table_atomic(all_df, aggregate_path)
        else:
            _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
    except Exception as exc:
        _invalidate_unprovenanced_radiomics_cohort(aggregate_path)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"Failed to publish cohort radiomics workbook {aggregate_path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# B4 + C4 — all-series (non-course) CT radiomics
#
# This block is a separate entry point that extracts radiomics from the
# materialized all-series CT series (planning/diagnostic/PET-CT/4DCT-averaged), one
# representative volume per 4DCT study (C4), and writes its own Data/radiomics_all_series.csv.
# It never calls run_radiomics, which also runs the per-course MR loop and may publish
# Data/radiomics_all.xlsx only for a complete, reconciled organize cohort.
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
    temp_dose_classification = {"classification": "no_doses"}
    temp_dose_completeness = classify_course_dose_completeness(
        selected_plans=[],
        selected_doses=[],
        dose_classification=temp_dose_classification,
        dose_grid=None,
        per_plan_delivery=[],
        delivery_status="no_records_at_all",
    )
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
                    "treatment_technique": build_treatment_technique_contract([]),
                    "dose_classification": temp_dose_classification,
                    "dose_completeness": temp_dose_completeness,
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
                        "resolved_prescribed_dose_total_gy": None,
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
    Never call run_radiomics because it also owns complete-cohort publication."""
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
    out_parquet = data_dir / "radiomics_all_series.parquet"
    preserved_df = None
    if out_csv.exists() or out_parquet.exists():
        try:
            if not out_parquet.exists():
                raise ValueError("authoritative all-series CT Parquet is missing")
            existing_df = read_authoritative_ct_publication(out_parquet)
            if "patient_id" not in existing_df.columns:
                raise ValueError("existing CSV lacks required patient_id column")
            validate_acquisition_descriptor_table(existing_df)
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
                    df = read_authoritative_ct_publication(Path(out_path).with_suffix(".parquet"))
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
                validate_acquisition_descriptor_table(df)
                # the temp course dir is deleted; repoint course_dir at the persistent series dir and
                # Keep course_id because it is part of the full publication identity.
                if "course_dir" in df.columns:
                    df["course_dir"] = str(input_dir)
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
        out_parquet.unlink(missing_ok=True)
        logger.info(
            "No all-series CT radiomics remain after refreshing %d patient(s)",
            len(requested_patient_ids),
        )
        return None
    out_df = normalize_radiomics_dataframe(pd.concat(frames, ignore_index=True))
    expected_strings = expected_radiomics_string_columns(out_df)
    validate_acquisition_descriptor_table(out_df)
    expected_keys = {publication_key(record) for record in out_df.to_dict("records")}
    validate_ct_publication(out_df, expected_keys=expected_keys)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{out_csv.name}.", suffix=".tmp", dir=data_dir, delete=False
    ) as handle:
        tmp_csv = Path(handle.name)
    parquet_tmp = tmp_csv.with_suffix(".tmp.parquet")
    try:
        out_df.to_csv(tmp_csv, index=False)
        out_df.to_parquet(parquet_tmp, index=False, engine="pyarrow")
        assert_radiomics_arrow_schema(
            parquet_tmp, expected_string_columns=expected_strings
        )
        parquet_check = pd.read_parquet(parquet_tmp, engine="pyarrow")
        validate_ct_publication(parquet_check, expected_keys=expected_keys)
        parquet_tmp.replace(out_parquet)
        tmp_csv.replace(out_csv)
    except Exception:
        out_parquet.unlink(missing_ok=True)
        out_csv.unlink(missing_ok=True)
        raise
    finally:
        tmp_csv.unlink(missing_ok=True)
        parquet_tmp.unlink(missing_ok=True)
    logger.info("Wrote all-series CT radiomics: %s (%d rows)", out_csv, len(out_df))
    return out_csv
