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
    try:
        import radiomics  # noqa: F401
        return True
    except Exception as e:
        logger.warning("pyradiomics not available: %s", e)
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
    m = sitk.GetImageFromArray(mask3d.astype(np.uint8))
    m.SetSpacing(ct_img.GetSpacing())
    m.SetDirection(ct_img.GetDirection())
    m.SetOrigin(ct_img.GetOrigin())
    return m


def _load_params_yaml(config: PipelineConfig | None) -> Optional[dict]:
    try:
        # Prefer user-provided file if present
        if config and config.radiomics_params_file:
            pth = Path(config.radiomics_params_file)
            if pth.exists():
                return yaml.safe_load(pth.read_text(encoding='utf-8'))
        # Fallback to packaged file
        p = importlib_resources.files('rtpipeline').joinpath('radiomics_params.yaml')
        if p.is_file():
            return yaml.safe_load(p.read_bytes())
    except Exception as e:
        logger.warning("Failed to load radiomics params YAML: %s", e)
    return None


def _extractor(config: PipelineConfig, modality: str = 'CT', normalize_override: Optional[bool] = None) -> Optional["radiomics.featureextractor.RadiomicsFeatureExtractor"]:
    if not _have_pyradiomics():
        return None
    from radiomics.featureextractor import RadiomicsFeatureExtractor
    try:
            params = _load_params_yaml(config)
            if params is not None:
                ext = RadiomicsFeatureExtractor(**params)
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
            mask = rt.get_mask_for_roi(name)
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


def radiomics_for_course(config: PipelineConfig, course_dir: Path) -> Optional[Path]:
    """Run pyradiomics on CT course with manual RS and RS_auto if present."""
    extractor = _extractor(config, 'CT')
    if extractor is None:
        return None
    img = _load_series_image(course_dir / 'CT_DICOM')
    if img is None:
        logger.info("No CT image for radiomics in %s", course_dir)
        return None
    rows: List[Dict] = []
    for source, rs_name in (("Manual", "RS.dcm"), ("AutoRTS_total", "RS_auto.dcm")):
        rs_path = course_dir / rs_name
        if not rs_path.exists():
            continue
        masks = _rtstruct_masks(course_dir / 'CT_DICOM', rs_path)
        for roi, mask in masks.items():
            try:
                m_img = _mask_from_array_like(img, mask)
                res = extractor.execute(img, m_img)
                rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                rec.update({
                    'modality': 'CT',
                    'segmentation_source': source,
                    'roi_name': roi,
                    'course_dir': str(course_dir),
                })
                rows.append(rec)
            except Exception as e:
                logger.debug("Radiomics failed for %s/%s: %s", source, roi, e)
                continue
    if not rows:
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        out = course_dir / 'radiomics_features_CT.xlsx'
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
                for lab in labels:
                    try:
                        mask = (arr == lab)
                        if not mask.any():
                            continue
                        m_img = _mask_from_array_like(img, mask)
                        res = extractor.execute(img, m_img)
                        rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                        rec.update({
                            'modality': 'MR',
                            'segmentation_source': 'AutoTS_total_mr',
                            'roi_name': label_map.get(lab, f'Segment_{lab}'),
                            'series_dir': str(series.dir),
                            'series_uid': series.series_uid,
                        })
                        rows.append(rec)
                    except Exception as e:
                        logger.debug("Radiomics MR total_mr failed for label %s: %s", lab, e)
                        continue
    # NIfTI fallback
    if _load_seg_nifti is not None:
        seg_nifti_dir = out_root / 'TotalSegmentator_total_mr_NIFTI'
        seg_img, label_map = _load_seg_nifti(seg_nifti_dir)
        if seg_img is not None:
            seg_img = _resample_to_reference(seg_img, img, nn=True)
            arr = sitk.GetArrayFromImage(seg_img)
            labels = [int(v) for v in np.unique(arr) if int(v) != 0]
            for lab in labels:
                try:
                    mask = (arr == lab)
                    if not mask.any():
                        continue
                    m_img = _mask_from_array_like(img, mask)
                    res = extractor.execute(img, m_img)
                    rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
                    rec.update({
                        'modality': 'MR',
                        'segmentation_source': 'AutoTS_total_mr',
                        'roi_name': label_map.get(lab, f'Segment_{lab}'),
                        'series_dir': str(series.dir),
                        'series_uid': series.series_uid,
                    })
                    rows.append(rec)
                except Exception as e:
                    logger.debug("Radiomics MR total_mr (NIfTI) failed for label %s: %s", lab, e)
                    continue
    if not rows:
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        out = out_root / 'radiomics_features_MR.xlsx'
        df.to_excel(out, index=False)
        return out
    except Exception as e:
        logger.warning("Failed to write MR radiomics: %s", e)
        return None


def run_radiomics(config: PipelineConfig, courses: List["object"]) -> None:
    """Top-level orchestrator: per-course CT radiomics and per-series MR radiomics.
    'courses' elements are CourseOutput-like with attributes patient_id, course_key, dir.
    """
    if not _have_pyradiomics():
        return
    # CT per course (parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    workers = config.effective_workers()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(radiomics_for_course, config, c.dir): c for c in courses}
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
