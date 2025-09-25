from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pydicom
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)


def _load_ct_image(ct_dir: Path) -> Optional[sitk.Image]:
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
        if not series_ids:
            return None
        files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
        reader.SetFileNames(files)
        return reader.Execute()
    except Exception as e:
        logger.error("CT load failed: %s", e)
        return None


def _load_seg_dicom(seg_path: Path) -> tuple[Optional[sitk.Image], Dict[int, str]]:
    try:
        import pydicom_seg
    except Exception as e:
        logger.debug("pydicom-seg unavailable: %s", e)
        return None, {}
    try:
        ds = pydicom.dcmread(str(seg_path))
        # Validate SOP Class UID for DICOM-SEG
        if str(getattr(ds, 'SOPClassUID', '')) != "1.2.840.10008.5.1.4.1.1.66.4":
            logger.debug("Not a DICOM-SEG: %s", getattr(ds, 'SOPClassUID', None))
            return None, {}
        reader = pydicom_seg.SegmentReader()
        seg_img = reader.read(ds)
        # Build label map from SegmentSequence
        label_map: Dict[int, str] = {}
        for seg in getattr(ds, 'SegmentSequence', []):
            num = int(getattr(seg, 'SegmentNumber', 0) or 0)
            name = str(getattr(seg, 'SegmentLabel', f'Segment_{num}') or f'Segment_{num}')
            if num:
                label_map[num] = name
        return seg_img, label_map
    except Exception as e:
        logger.error("DICOM-SEG load failed: %s", e)
        return None, {}


def _load_seg_nifti(nifti_dir: Path) -> tuple[Optional[sitk.Image], Dict[int, str]]:
    if not nifti_dir.exists():
        return None, {}
    label_map: Dict[int, str] = {}
    # Try TotalSegmentator's segmentations.nii.gz and segmentations.json
    json_path = nifti_dir / 'segmentations.json'
    nii_candidates = [nifti_dir / 'segmentations.nii.gz', nifti_dir / 'segmentation.nii.gz']
    seg_img = None
    try:
        for p in nii_candidates:
            if p.exists():
                seg_img = sitk.ReadImage(str(p))
                break
        if seg_img is None:
            for p in nifti_dir.iterdir():
                if p.name.endswith('.nii') or p.name.endswith('.nii.gz'):
                    try:
                        seg_img = sitk.ReadImage(str(p))
                        break
                    except Exception:
                        continue
    except Exception as e:
        logger.error("NIfTI seg load failed: %s", e)
        return None, {}

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            # Expect mapping label name -> index or list of objects with 'id'/'name'
            if isinstance(data, dict):
                # e.g., {"organ_name": index}
                for k, v in data.items():
                    try:
                        idx = int(v)
                        label_map[idx] = str(k)
                    except Exception:
                        continue
            elif isinstance(data, list):
                for item in data:
                    try:
                        idx = int(item.get('id'))
                        name = str(item.get('name', f'Segment_{idx}'))
                        label_map[idx] = name
                    except Exception:
                        continue
        except Exception:
            pass
    # Ensure correct tuple shape regardless of seg_img presence
    if seg_img is None:
        return None, {}
    return seg_img, label_map


def _resample_to_reference(seg_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    if (seg_img.GetSize() == ref_img.GetSize() and
        seg_img.GetSpacing() == ref_img.GetSpacing() and
        seg_img.GetDirection() == ref_img.GetDirection() and
        seg_img.GetOrigin() == ref_img.GetOrigin()):
        return seg_img
    res = sitk.Resample(seg_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg_img.GetPixelID())
    return res


def build_auto_rtstruct(course_dir: Path) -> Optional[Path]:
    """Create an RTSTRUCT (RS_auto.dcm) from TotalSegmentator output if present.
    Returns path to RTSTRUCT or None.
    """
    try:
        from rt_utils import RTStructBuilder
    except Exception as e:
        logger.error("rt-utils not available: %s", e)
        return None

    ct_dir = course_dir / 'CT_DICOM'
    if not ct_dir.exists():
        logger.info("No CT_DICOM for %s", course_dir)
        return None

    # Resume-friendly: if already built, skip
    out_path = course_dir / 'RS_auto.dcm'
    if out_path.exists():
        logger.info("Auto RTSTRUCT already exists: %s", out_path)
        return out_path

    ct_img = _load_ct_image(ct_dir)
    if ct_img is None:
        return None

    # Prefer DICOM-SEG, detect if RTSTRUCT already produced, fallback to NIfTI
    seg_img: Optional[sitk.Image] = None
    label_map: Dict[int, str] = {}
    dicom_dir = course_dir / 'TotalSegmentator_DICOM'
    dicom_seg_path = dicom_dir / 'segmentations.dcm'
    if dicom_seg_path.exists():
        # Check whether this is SEG or RTSTRUCT; handle both
        try:
            ds = pydicom.dcmread(str(dicom_seg_path), stop_before_pixels=True)
            sop = str(getattr(ds, 'SOPClassUID', ''))
            if sop == '1.2.840.10008.5.1.4.1.1.66.4':
                seg_img, label_map = _load_seg_dicom(dicom_seg_path)
            elif sop == '1.2.840.10008.5.1.4.1.1.481.3':
                # Already an RTSTRUCT from TotalSegmentator: copy/normalize as RS_auto
                out_path = course_dir / 'RS_auto.dcm'
                try:
                    if out_path.exists():
                        out_path.unlink()
                    dicom_seg_path.rename(out_path)
                except Exception:
                    try:
                        import shutil as _sh
                        _sh.copy2(str(dicom_seg_path), str(out_path))
                    except Exception as e:
                        logger.error('Failed to copy RTSTRUCT to RS_auto: %s', e)
                        return None
                logger.info("Wrote auto RTSTRUCT (from RTSTRUCT): %s", out_path)
                return out_path
        except Exception as e:
            logger.debug('Inspecting DICOM output failed: %s', e)
    if seg_img is None:
        seg_img, label_map = _load_seg_nifti(course_dir / 'TotalSegmentator_NIFTI')
    if seg_img is None:
        logger.info("No segmentation outputs found to build RTSTRUCT in %s", course_dir)
        return None

    # Resample segmentation to CT geometry
    seg_res = _resample_to_reference(seg_img, ct_img)
    seg_arr = sitk.GetArrayFromImage(seg_res)  # [z,y,x] integer labels
    labels = [int(v) for v in np.unique(seg_arr) if int(v) != 0]
    if not labels:
        logger.info("Segmentation contains no labels in %s", course_dir)
        return None

    # Create RTSTRUCT
    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    except Exception as e:
        logger.error("Failed to create RTSTRUCT: %s", e)
        return None

    # Add each label as ROI
    for idx in labels:
        name = label_map.get(idx, f'Segment_{idx}')
        mask = (seg_arr == idx)
        try:
            rtstruct.add_roi(mask=mask, name=name)
        except Exception as e:
            logger.debug("Failed to add ROI %s: %s", name, e)
            continue

    try:
        rtstruct.save(str(out_path))
        logger.info("Wrote auto RTSTRUCT: %s", out_path)
        return out_path
    except Exception as e:
        logger.error("Saving auto RTSTRUCT failed: %s", e)
        return None
