from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pydicom
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)

from .layout import build_course_dirs
from .utils import sanitize_rtstruct
from .roi_fixer import fix_rtstruct_rois
from .layout import build_course_dirs


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


def _strip_nifti_base(nifti_path: Path) -> str:
    name = nifti_path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return nifti_path.stem


def _load_seg_nifti(seg_dir: Path, base_name: Optional[str]) -> tuple[Optional[sitk.Image], Dict[int, str]]:
    if not seg_dir.exists():
        return None, {}

    label_map: Dict[int, str] = {}
    seg_img: Optional[sitk.Image] = None

    candidates: list[Path] = []
    if base_name:
        specific = seg_dir / f"{base_name}_total_multilabel.nii.gz"
        if specific.exists():
            candidates.append(specific)
    if not candidates:
        candidates = sorted(seg_dir.glob("*_total_multilabel.nii.gz"))

    try:
        for p in candidates:
            seg_img = sitk.ReadImage(str(p))
            break
    except Exception as e:
        logger.error("NIfTI seg load failed: %s", e)
        seg_img = None

    json_candidates: list[Path] = []
    if base_name:
        json_specific = seg_dir / f"{base_name}_total_segmentations.json"
        if json_specific.exists():
            json_candidates.append(json_specific)
    if not json_candidates:
        json_candidates = sorted(seg_dir.glob("*_total_segmentations.json"))

    for json_path in json_candidates:
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(data, dict):
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
        break

    if seg_img is None:
        return None, {}
    return seg_img, label_map


def _iter_binary_masks(nifti_dir: Path, prefix: Optional[str] = None) -> Iterable[Tuple[str, sitk.Image]]:
    """Yield (name, image) pairs for TotalSegmentator-style binary mask outputs."""
    if not nifti_dir.exists():
        return []

    masks: list[Tuple[str, sitk.Image]] = []
    for mask_path in sorted(nifti_dir.glob("*.nii*")):
        name_lower = mask_path.name.lower()
        if name_lower in {"segmentations.nii", "segmentations.nii.gz", "segmentation.nii", "segmentation.nii.gz"}:
            # Skip potential multi-label files handled separately
            continue
        try:
            img = sitk.ReadImage(str(mask_path))
        except Exception as e:
            logger.debug("Skipping mask %s: %s", mask_path.name, e)
            continue
        # Clean name: remove .nii and .nii.gz suffixes
        name = mask_path.name
        if name.endswith('.nii.gz'):
            name = name[:-7]  # Remove .nii.gz
        elif name.endswith('.nii'):
            name = name[:-4]  # Remove .nii
        if prefix and name.startswith(prefix):
            stripped = name[len(prefix):]
            name = stripped or name
        masks.append((name, img))
    return masks


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

    course_dirs = build_course_dirs(course_dir)
    ct_dir = course_dirs.dicom_ct
    if not ct_dir.exists():
        logger.info("No CT DICOM for %s", course_dir)
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
    seg_root = course_dirs.segmentation_totalseg
    dicom_seg_path: Optional[Path] = None
    base_name: Optional[str] = None
    selected_dir: Optional[Path] = None

    if seg_root.exists():
        candidate_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())
        for base_dir in candidate_dirs:
            cand = base_dir / f"{base_dir.name}--total.dcm"
            if cand.exists():
                selected_dir = base_dir
                dicom_seg_path = cand
                base_name = base_dir.name
                break
        if selected_dir is None and candidate_dirs:
            selected_dir = candidate_dirs[0]
            base_name = selected_dir.name
            cand = selected_dir / f"{base_name}--total.dcm"
            if cand.exists():
                dicom_seg_path = cand

    if dicom_seg_path and dicom_seg_path.exists():
        try:
            ds = pydicom.dcmread(str(dicom_seg_path), stop_before_pixels=True)
            sop = str(getattr(ds, 'SOPClassUID', ''))
            if sop == '1.2.840.10008.5.1.4.1.1.66.4':
                seg_img, label_map = _load_seg_dicom(dicom_seg_path)
            elif sop == '1.2.840.10008.5.1.4.1.1.481.3':
                try:
                    if out_path.exists():
                        out_path.unlink()
                    shutil.copy2(str(dicom_seg_path), str(out_path))
                except Exception as e:
                    logger.error('Failed to copy RTSTRUCT to RS_auto: %s', e)
                    return None
                logger.info("Wrote auto RTSTRUCT (from RTSTRUCT): %s", out_path)
                return out_path
        except Exception as e:
            logger.debug('Inspecting DICOM output failed: %s', e)

    if seg_img is None:
        seg_img, label_map = _load_seg_nifti(selected_dir or seg_root, base_name)

    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    except Exception as e:
        logger.error("Failed to create RTSTRUCT: %s", e)
        return None

    added_any = False

    if seg_img is not None:
        # Resample segmentation to CT geometry and add each label present
        seg_res = _resample_to_reference(seg_img, ct_img)
        seg_arr = sitk.GetArrayFromImage(seg_res)  # [z,y,x] integer labels
        seg_arr = np.moveaxis(seg_arr, 0, -1)  # -> [y,x,z] for rt-utils
        labels = [int(v) for v in np.unique(seg_arr) if int(v) != 0]
        if not labels:
            logger.info("Segmentation contains no labels in %s", course_dir)
        else:
            for idx in labels:
                name = label_map.get(idx, f'Segment_{idx}')
                mask = seg_arr == idx
                if not np.any(mask):
                    continue
                try:
                    rtstruct.add_roi(mask=mask, name=name)
                    added_any = True
                except Exception as e:
                    logger.debug("Failed to add ROI %s: %s", name, e)

    fallback_dir = selected_dir or seg_root
    if not added_any and fallback_dir.exists():
        # Fall back to per-ROI binary masks produced by TotalSegmentator
        mask_prefix = f"{base_name}--total--" if base_name else None
        for name, mask_img in _iter_binary_masks(fallback_dir, prefix=mask_prefix):
            try:
                resampled = _resample_to_reference(mask_img, ct_img)
                mask_arr = sitk.GetArrayFromImage(resampled)
                mask_arr = np.moveaxis(mask_arr, 0, -1)  # -> [y,x,z]
            except Exception as e:
                logger.debug("Failed to resample mask %s: %s", name, e)
                continue

            mask_bin = mask_arr > 0
            if not np.any(mask_bin):
                continue

            roi_name = name
            try:
                rtstruct.add_roi(mask=mask_bin, name=roi_name)
                added_any = True
            except Exception as e:
                logger.debug("Failed to add ROI %s: %s", roi_name, e)

    if not added_any:
        logger.info("No RTSTRUCT ROIs added for %s", course_dir)
        return None

    try:
        rtstruct.save(str(out_path))
        sanitize_rtstruct(out_path)
        summary = fix_rtstruct_rois(ct_dir, out_path)
        if summary and summary.changed:
            logger.info(
                "Auto RTSTRUCT ROI fix: %d repaired, %d still problematic",
                len(summary.fixed),
                len(summary.failed),
            )
        logger.info("Wrote auto RTSTRUCT: %s", out_path)
        return out_path
    except Exception as e:
        logger.error("Saving auto RTSTRUCT failed: %s", e)
        return None
