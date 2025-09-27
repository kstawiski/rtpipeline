from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import pydicom

from .config import PipelineConfig
from importlib import resources as importlib_resources
import yaml

logger = logging.getLogger(__name__)


def _have_pyradiomics() -> bool:
    """Check if pyradiomics is available and properly installed."""
    try:
        # First check if we can import without C extensions
        import os
        original_env = os.environ.get('PYRADIOMICS_USE_CEXTENSIONS')
        
        try:
            # Try importing pyradiomics normally first
            import radiomics  # noqa: F401
            from radiomics import featureextractor  # noqa: F401
            logger.info("pyradiomics loaded successfully with full C extensions support")
            return True
        except ImportError as e:
            error_msg = str(e)
            
            # Check if this is a NumPy 2.x compatibility issue with C extensions
            if "NumPy 1.x cannot be run in" in error_msg or "_multiarray_umath" in error_msg:
                logger.warning("pyradiomics C extensions incompatible with NumPy 2.x - attempting fallback")
                
                # Try disabling C extensions
                os.environ['PYRADIOMICS_USE_CEXTENSIONS'] = '0'
                
                try:
                    # Clear any cached imports
                    import sys
                    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('radiomics')]
                    for mod in modules_to_clear:
                        del sys.modules[mod]
                    
                    # Try importing again with C extensions disabled
                    import radiomics  # noqa: F401
                    from radiomics import featureextractor  # noqa: F401
                    
                    logger.warning("pyradiomics loaded with C extensions disabled due to NumPy 2.x compatibility")
                    logger.info("Radiomics will work but may be slower without C extensions")
                    return True
                    
                except Exception as fallback_error:
                    logger.error(f"pyradiomics failed even with C extensions disabled: {fallback_error}")
                    return False
                finally:
                    # Restore original environment
                    if original_env is None:
                        os.environ.pop('PYRADIOMICS_USE_CEXTENSIONS', None)
                    else:
                        os.environ['PYRADIOMICS_USE_CEXTENSIONS'] = original_env
            else:
                logger.info("pyradiomics not available. Install with: pip install pyradiomics")
                logger.info("Or install rtpipeline with radiomics support: pip install -e '.[radiomics]'")
                return False
                
    except Exception as e:
        logger.warning("pyradiomics import failed (possibly Python version compatibility): %s", e)
        logger.info("Try using Python 3.11: conda create -n rtpipeline python=3.11")
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
    if not _have_pyradiomics():
        return None
    from radiomics.featureextractor import RadiomicsFeatureExtractor
    try:
            pfile = _get_params_file(config)
            if pfile is not None:
                ext = RadiomicsFeatureExtractor(str(pfile))
            else:
                ext = RadiomicsFeatureExtractor()
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
    # Resume-friendly: skip if output exists
    out_path = course_dir / 'radiomics_features_CT.xlsx'
    if getattr(config, 'resume', False) and out_path.exists():
        return out_path
    extractor = _extractor(config, 'CT')
    if extractor is None:
        return None
    img = _load_series_image(course_dir / 'CT_DICOM')
    if img is None:
        logger.info("No CT image for radiomics in %s", course_dir)
        return None
    rows: List[Dict] = []
    tasks = []

    # Process standard RTSTRUCTs
    for source, rs_name in (("Manual", "RS.dcm"), ("AutoRTS_total", "RS_auto.dcm")):
        rs_path = course_dir / rs_name
        if not rs_path.exists():
            continue
        masks = _rtstruct_masks(course_dir / 'CT_DICOM', rs_path)
        for roi, mask in masks.items():
            tasks.append((source, roi, mask))

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
                masks = _rtstruct_masks(course_dir / 'CT_DICOM', rs_custom)
                for roi, mask in masks.items():
                    tasks.append(("Custom", roi, mask))
        except Exception as e:
            logger.warning("Failed to process custom structures for radiomics: %s", e)
    def _do_ct_task(t):
        source, roi, mask = t
        try:
            # Create fresh extractor instance for each task to avoid threading issues
            ext = _extractor(config, 'CT')
            if ext is None:
                logger.debug("No radiomics extractor available for %s/%s", source, roi)
                return None
            m_img = _mask_from_array_like(img, mask)
            res = ext.execute(img, m_img)
            rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
            rec.update({'modality': 'CT','segmentation_source': source,'roi_name': roi,'course_dir': str(course_dir)})
            return rec
        except Exception as e:
            logger.debug("Radiomics failed for %s/%s: %s", source, roi, e)
            return None
            return None
    if tasks:
        # Force sequential radiomics processing to prevent segmentation faults
        # Parallel processing causes memory issues with pyradiomics/OpenMP interactions
        max_radiomics_workers = 1
        logger.info("Using sequential radiomics processing (enforced to prevent segmentation faults)")
        
        # Process tasks sequentially to avoid threading issues
        for t in tasks:
            r = _do_ct_task(t)
            if r:
                rows.append(r)
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
                from concurrent.futures import ThreadPoolExecutor, as_completed
                def _do_lab(lab:int):
                    try:
                        mask = (arr == lab)
                        if not mask.any():
                            return None
                        ext = _extractor(config, 'MR', normalize_override=normalize_override)
                        m_img = _mask_from_array_like(img, mask)
                        res = ext.execute(img, m_img)
                        rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                        rec.update({'modality':'MR','segmentation_source':'AutoTS_total_mr','roi_name':label_map.get(lab,f'Segment_{lab}'),'series_dir':str(series.dir),'series_uid':series.series_uid})
                        return rec
                    except Exception as e:
                        logger.debug("Radiomics MR total_mr failed for label %s: %s", lab, e)
                        return None
                with ThreadPoolExecutor(max_workers=max(1, min(config.effective_workers(), 2))) as ex:
                    for f in as_completed([ex.submit(_do_lab, lab) for lab in labels]):
                        rr=f.result()
                        if rr:
                            rows.append(rr)
    # NIfTI fallback
    if _load_seg_nifti is not None:
        seg_nifti_dir = out_root / 'TotalSegmentator_total_mr_NIFTI'
        seg_img, label_map = _load_seg_nifti(seg_nifti_dir)
        if seg_img is not None:
            seg_img = _resample_to_reference(seg_img, img, nn=True)
            arr = sitk.GetArrayFromImage(seg_img)
            labels = [int(v) for v in np.unique(arr) if int(v) != 0]
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _do_lab(lab:int):
                try:
                    mask = (arr == lab)
                    if not mask.any():
                        return None
                    ext = _extractor(config, 'MR', normalize_override=normalize_override)
                    m_img = _mask_from_array_like(img, mask)
                    res = ext.execute(img, m_img)
                    rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                    rec.update({'modality':'MR','segmentation_source':'AutoTS_total_mr','roi_name':label_map.get(lab,f'Segment_{lab}'),'series_dir':str(series.dir),'series_uid':series.series_uid})
                    return rec
                except Exception as e:
                    logger.debug("Radiomics MR total_mr (NIfTI) failed for label %s: %s", lab, e)
                    return None
            with ThreadPoolExecutor(max_workers=max(1, min(config.effective_workers(), 2))) as ex:
                for f in as_completed([ex.submit(_do_lab, lab) for lab in labels]):
                    rr=f.result()
                    if rr:
                        rows.append(rr)
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
    'courses' elements are CourseOutput-like with attributes patient_id, course_key, dir.
    """
    if not _have_pyradiomics():
        return
    # CT per course (parallel, but limited for memory safety)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Limit course-level parallelization for radiomics to prevent memory issues
    if os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes'):
        max_course_workers = 1
        logger.info("Using sequential course processing for radiomics (RTPIPELINE_RADIOMICS_SEQUENTIAL set)")
    else:
        max_course_workers = min(2, config.effective_workers())
        logger.info("Processing radiomics with %d course workers (limited for memory safety)", max_course_workers)

    with ThreadPoolExecutor(max_workers=max_course_workers) as ex:
        futs = {ex.submit(radiomics_for_course, config, c.dir, custom_structures_config): c for c in courses}
        for _ in as_completed(futs):
            pass
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
            p = Path(c.dir) / 'radiomics_features_CT.xlsx'
            if p.exists():
                try:
                    df = _pd.read_excel(p)
                    df.insert(0, 'patient_id', getattr(c, 'patient_id', Path(c.dir).parts[-2]))
                    df.insert(1, 'course_key', getattr(c, 'course_key', Path(c.dir).name))
                    df.insert(2, 'course_dir', str(c.dir))
                    out_rows.append(df)
                except Exception:
                    pass
        # MR cohort
        for p in config.output_root.rglob('MR_*/radiomics_features_MR.xlsx'):
            try:
                df = _pd.read_excel(p)
                # Attempt to infer patient_id and series_uid from path
                parts = p.parts
                try:
                    idx = parts.index(str(config.output_root))
                except ValueError:
                    idx = len(parts)-4
                patient_id = parts[-4]
                series_uid = parts[-2].replace('MR_','')
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
