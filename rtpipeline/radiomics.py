from __future__ import annotations

import logging
import os
import threading
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


def _load_series_image(dicom_dir: Path, series_uid: Optional[str] = None) -> Optional[sitk.Image]:
    try:
        reader = sitk.ImageSeriesReader()
        sids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not sids:
            return None
        sid = series_uid if (series_uid and series_uid in sids) else sids[0]
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)
        reader.SetFileNames(files)
        return reader.Execute()
    except Exception as e:
        logger.debug("Failed loading series %s in %s: %s", series_uid, dicom_dir, e)
        return None


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


def _get_params_file(config: PipelineConfig | None) -> Optional[Path]:
    """Return a filesystem path to a radiomics params YAML.
    Prefers user-provided file; else copies packaged defaults into logs_root for reuse.
    """
    try:
        if config and config.radiomics_params_file and Path(config.radiomics_params_file).exists():
            return Path(config.radiomics_params_file)
        # Copy packaged file to logs_root for stable path
        packaged = importlib_resources.files('rtpipeline').joinpath('radiomics_params.yaml')
        if packaged.is_file():
            target_root = Path(config.logs_root) if (config and config.logs_root) else Path('.')
            target_root.mkdir(parents=True, exist_ok=True)
            out = Path(target_root) / 'radiomics_params.yaml'
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

        pfile = _get_params_file(config)
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


def radiomics_for_course(config: PipelineConfig, course_dir: Path, custom_structures_config: Optional[Path] = None) -> Optional[Path]:
    """Run pyradiomics on CT course with manual RS, RS_auto, and custom structures if present."""

    _apply_radiomics_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))

    course_dirs = build_course_dirs(course_dir)
    # Resume-friendly: skip if output exists
    out_path = course_dir / 'radiomics_ct.xlsx'
    if getattr(config, 'resume', False) and out_path.exists():
        return out_path
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

    # Process standard RTSTRUCTs
    for source, rs_name in (("Manual", "RS.dcm"), ("AutoRTS_total", "RS_auto.dcm")):
        rs_path = course_dir / rs_name
        if not rs_path.exists():
            continue
        masks = _rtstruct_masks(course_dirs.dicom_ct, rs_path)
        for roi, mask in masks.items():
            tasks.append((source, roi, mask, mask_is_cropped(mask)))

    # Process custom structures if configuration provided
    if custom_structures_config and custom_structures_config.exists():
        try:
            # Check if custom RTSTRUCT exists or create it
            rs_custom = course_dir / "RS_custom.dcm"
            if not rs_custom.exists():
                from .dvh import _create_custom_structures_rtstruct
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir,
                    custom_structures_config,
                    course_dir / "RS.dcm",
                    course_dir / "RS_auto.dcm"
                )
            if rs_custom and rs_custom.exists():
                masks = _rtstruct_masks(course_dirs.dicom_ct, rs_custom)
                for roi, mask in masks.items():
                    tasks.append(("Custom", roi, mask, mask_is_cropped(mask)))
        except Exception as e:
            logger.warning("Failed to process custom structures for radiomics: %s", e)
    def _do_ct_task(t):
        source, roi, mask, cropped = t
        try:
            # Create fresh extractor instance for each task to avoid threading issues
            ext = _extractor(config, 'CT')
            if ext is None:
                logger.debug("No radiomics extractor available for %s/%s", source, roi)
                return None
            m_img = _mask_from_array_like(img, mask)
            res = ext.execute(img, m_img)
            rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
            rec.update({
                'modality': 'CT',
                'segmentation_source': source,
                'roi_name': roi,
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
            env_limit = int(os.environ.get('RTPIPELINE_RADIOMICS_MAX_WORKERS', '0') or 0)
            if env_limit > 0:
                max_workers = min(max_workers, env_limit)
            max_workers = max(1, min(max_workers, 4))
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
        df = pd.DataFrame(rows)
        out = out_path
        df.to_excel(out, index=False)
        return out
    except Exception as e:
        logger.warning("Failed to write CT radiomics: %s", e)
        return None


@dataclass
class MRSeries:
    patient_id: str
    series_uid: str
    dir: Path


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
    _apply_radiomics_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))
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

    if use_parallel_impl:
        logger.info("Using enhanced parallel radiomics implementation")
        # Use the new parallel implementation for each course
        def _parallel_radiomics_wrapper(course):
            return parallel_radiomics_for_course(config, course.dirs.root, custom_structures_config)

        radiomics_func = _parallel_radiomics_wrapper
        # Reduce course workers when using internal parallelization
        max_course_workers = max(1, min(2, len(courses)))
    else:
        # Use traditional implementation
        radiomics_func = lambda course: radiomics_for_course(config, course.dirs.root, custom_structures_config)

        # CT per course (parallel, but limited for memory safety)
        if os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes'):
            max_course_workers = 1
            logger.info("Using sequential course processing for radiomics (RTPIPELINE_RADIOMICS_SEQUENTIAL set)")
        else:
            max_course_workers = config.effective_workers()
            env_limit = int(os.environ.get('RTPIPELINE_RADIOMICS_MAX_WORKERS', '0') or 0)
            if env_limit > 0:
                max_course_workers = min(max_course_workers, env_limit)
            max_course_workers = max(1, min(max_course_workers, 4))
            logger.info("Processing radiomics with up to %d course workers", max_course_workers)

    run_tasks_with_adaptive_workers(
        "Radiomics (CT courses)",
        courses,
        radiomics_func,
        max_workers=max_course_workers,
        logger=logger,
        show_progress=True,
    )
    # MR per series (sequential; number of series is usually small)
    try:
        from .segmentation import _scan_mr_series
    except Exception:
        _scan_mr_series = None  # type: ignore
    if _scan_mr_series is None:
        return
    series = _scan_mr_series(config.dicom_root)
    for pid, suid, sdir in series:
        radiomics_for_mr_series(config, MRSeries(pid, suid, sdir))
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
