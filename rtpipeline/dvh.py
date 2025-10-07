from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)

from .layout import build_course_dirs
from .utils import sanitize_rtstruct, mask_is_cropped, run_tasks_with_adaptive_workers
from .layout import build_course_dirs

# Compatibility shim for dicompyler-core with pydicom>=3
try:
    import sys as _sys, types as _types, pydicom as _pyd
    _m = _sys.modules.get('dicom') or _types.ModuleType('dicom')
    _m.read_file = getattr(_pyd, 'dcmread', None)
    _sys.modules['dicom'] = _m
    _mds = _types.ModuleType('dicom.dataset')
    _mds.Dataset = _pyd.dataset.Dataset
    _mds.FileDataset = _pyd.dataset.FileDataset
    _sys.modules['dicom.dataset'] = _mds
    _mtag = _types.ModuleType('dicom.tag')
    _mtag.Tag = _pyd.tag.Tag
    _sys.modules['dicom.tag'] = _mtag
    _muid = _types.ModuleType('dicom.uid')
    _muid.UID = _pyd.uid.UID
    _sys.modules['dicom.uid'] = _muid
    # Patch pydicom.filewriter API expected by older libs
    try:
        import pydicom.filewriter as _fw
        if not hasattr(_fw, 'validate_file_meta'):
            def validate_file_meta(*args, **kwargs):  # type: ignore
                return True
            _fw.validate_file_meta = validate_file_meta  # type: ignore[attr-defined]
    except Exception:
        pass
except (ImportError, AttributeError) as e:
    logger.warning("Failed to set up dicompyler-core compatibility: %s", e)
except Exception as e:
    logger.error("Unexpected error in dicompyler-core compatibility setup: %s", e)

try:
    from dicompylercore import dvhcalc
    # Patch missing validate_file_meta into dicompyler-core modules for pydicom>=3
    try:
        import sys as _sys2, types as _types2, pydicom as _pyd2
        import pydicom.filewriter as _fw2
        if not hasattr(_fw2, 'validate_file_meta'):
            def _validate_file_meta(*args, **kwargs):  # type: ignore
                return True
            _fw2.validate_file_meta = _validate_file_meta  # type: ignore[attr-defined]
        # Also expose via builtins for modules that reference it unqualified
        try:
            import builtins as _bi
            if not hasattr(_bi, 'validate_file_meta'):
                _bi.validate_file_meta = _fw2.validate_file_meta  # type: ignore[attr-defined]
        except Exception:
            pass
        for _name, _mod in list(_sys2.modules.items()):
            if _name and _name.startswith('dicompylercore') and hasattr(_mod, '__dict__'):
                if 'validate_file_meta' not in _mod.__dict__:
                    try:
                        _mod.validate_file_meta = _fw2.validate_file_meta  # type: ignore[attr-defined]
                    except Exception:
                        pass
    except Exception:
        pass
except ImportError as e:
    logger.error("Failed to import dicompylercore: %s. Install with: pip install dicompyler-core", e)
    raise


def _dose_at_fraction(bins, cumulative, fraction: float) -> float:
    V_total = cumulative[0]
    target = fraction * V_total
    idx = np.where(cumulative <= target)[0]
    if idx.size == 0:
        return float(bins[-1])
    i = int(idx[0])
    if i == 0:
        return float(bins[0])
    x0, x1 = float(bins[i - 1]), float(bins[i])
    y0, y1 = float(cumulative[i - 1]), float(cumulative[i])
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def _get_volume_at_threshold(bins, cumulative, threshold: float) -> float:
    V_total = cumulative[0]
    if threshold <= bins[0]:
        return float(V_total)
    if threshold >= bins[-1]:
        return 0.0
    idx = np.where(bins >= threshold)[0]
    if idx.size == 0:
        return 0.0
    i = int(idx[0])
    if i == 0:
        return float(V_total)
    x0, x1 = float(bins[i - 1]), float(bins[i])
    y0, y1 = float(cumulative[i - 1]), float(cumulative[i])
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (threshold - x0) / (x1 - x0)


def _compute_metrics(abs_dvh, rx_dose: float) -> Optional[Dict[str, float]]:
    bins_abs = abs_dvh.bincenters
    cum_abs = abs_dvh.counts
    if cum_abs.size == 0 or cum_abs[0] == 0:
        return None
    V_total = float(cum_abs[0])
    DmeanGy = float(abs_dvh.mean)
    DmaxGy = float(abs_dvh.max)
    DminGy = float(abs_dvh.min)
    D95Gy = _dose_at_fraction(bins_abs, cum_abs, 0.95)
    D98Gy = _dose_at_fraction(bins_abs, cum_abs, 0.98)
    D2Gy = _dose_at_fraction(bins_abs, cum_abs, 0.02)
    D50Gy = _dose_at_fraction(bins_abs, cum_abs, 0.50)
    HI_abs = (D2Gy - D98Gy) / D50Gy if D50Gy != 0 else float("nan")
    SpreadGy = DmaxGy - DminGy

    V_abs: Dict[str, float] = {}
    V_rel: Dict[str, float] = {}
    for x in range(1, 61):
        vol = _get_volume_at_threshold(bins_abs, cum_abs, float(x))
        V_abs[f"V{x}Gy (cm³)"] = float(vol)
        V_rel[f"V{x}Gy (%)"] = float((vol / V_total) * 100.0)

    IntegralDose = DmeanGy * V_total

    # Additional clinically useful metrics
    # Hottest small volumes (absolute), commonly used for OARs
    D1ccGy = None
    D0_1ccGy = None
    try:
        if V_total > 0:
            if V_total >= 1.0:
                D1ccGy = _dose_at_fraction(bins_abs, cum_abs, 1.0 / V_total)
            if V_total >= 0.1:
                D0_1ccGy = _dose_at_fraction(bins_abs, cum_abs, 0.1 / V_total)
    except Exception:
        pass

    # Coverage at 95% and 100% of Rx dose
    V95Rx_cc = None
    V95Rx_pct = None
    V100Rx_cc = None
    V100Rx_pct = None
    try:
        if rx_dose and rx_dose > 0:
            V95Rx_cc = _get_volume_at_threshold(bins_abs, cum_abs, 0.95 * rx_dose)
            V100Rx_cc = _get_volume_at_threshold(bins_abs, cum_abs, 1.00 * rx_dose)
            V95Rx_pct = (V95Rx_cc / V_total) * 100.0 if V_total > 0 else None
            V100Rx_pct = (V100Rx_cc / V_total) * 100.0 if V_total > 0 else None
    except Exception:
        pass

    rel_dvh = abs_dvh.relative_dose(float(rx_dose))
    bins_rel = rel_dvh.bincenters
    cum_rel = rel_dvh.counts
    Dmean_pct = float(rel_dvh.mean)
    Dmax_pct = float(rel_dvh.max)
    Dmin_pct = float(rel_dvh.min)
    D95_pct = _dose_at_fraction(bins_rel, cum_rel, 0.95)
    D98_pct = _dose_at_fraction(bins_rel, cum_rel, 0.98)
    D2_pct = _dose_at_fraction(bins_rel, cum_rel, 0.02)
    D50_pct = _dose_at_fraction(bins_rel, cum_rel, 0.50)
    HI_pct = (D2_pct - D98_pct) / D50_pct if D50_pct != 0 else float("nan")
    Spread_pct = Dmax_pct - Dmin_pct

    metrics: Dict[str, float] = {
        "DmeanGy": DmeanGy,
        "DmaxGy": DmaxGy,
        "DminGy": DminGy,
        "D95Gy": D95Gy,
        "D98Gy": D98Gy,
        "D2Gy": D2Gy,
        "D50Gy": D50Gy,
        "HI": HI_abs,
        "SpreadGy": SpreadGy,
        "D1ccGy": D1ccGy,
        "D0.1ccGy": D0_1ccGy,
        "Dmean%": Dmean_pct,
        "Dmax%": Dmax_pct,
        "Dmin%": Dmin_pct,
        "D95%": D95_pct,
        "D98%": D98_pct,
        "D2%": D2_pct,
        "D50%": D50_pct,
        "HI%": HI_pct,
        "Spread%": Spread_pct,
        "Volume (cm³)": V_total,
        "IntegralDose_Gycm3": float(IntegralDose),
        "V95%Rx (cm³)": V95Rx_cc,
        "V95%Rx (%)": V95Rx_pct,
        "V100%Rx (cm³)": V100Rx_cc,
        "V100%Rx (%)": V100Rx_pct,
    }
    metrics.update(V_abs)
    metrics.update(V_rel)
    return metrics


def _create_custom_structures_rtstruct(
    course_dir: Path,
    config_path: Optional[Union[str, Path]] = None,
    rs_manual: Optional[Path] = None,
    rs_auto: Optional[Path] = None
) -> Optional[Path]:
    """Create a new RTSTRUCT with custom structures from boolean operations."""
    try:
        from .custom_structures import CustomStructureProcessor
        from rt_utils import RTStructBuilder
    except ImportError as e:
        logger.warning("rt-utils not available for custom structures: %s", e)
        return None

    # Choose base RTSTRUCT (prefer manual over auto)
    base_rs = None
    base_source = ""
    if rs_manual and Path(rs_manual).exists():
        base_rs = Path(rs_manual)
        base_source = "manual"
    elif rs_auto and Path(rs_auto).exists():
        base_rs = Path(rs_auto)
        base_source = "auto"
    else:
        logger.warning("No base RTSTRUCT available for custom structures")
        return None

    course_dirs = build_course_dirs(course_dir)
    ct_dir = course_dirs.dicom_ct
    if not ct_dir.exists():
        logger.warning("CT_DICOM not found for custom structures")
        return None

    try:
        # Load the base RTSTRUCT
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(base_rs)
        )

        existing_names = set()
        available_masks: Dict[str, np.ndarray] = {}

        totalseg_mask_cache: Dict[str, Optional[np.ndarray]] = {}
        ct_image: Optional[sitk.Image] = None

        def _ensure_ct_image() -> Optional[sitk.Image]:
            nonlocal ct_image
            if ct_image is not None:
                return ct_image
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
            if not series_ids:
                logger.warning("No CT series found for spacing calculation")
                return None
            dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
            reader.SetFileNames(dicom_files)
            try:
                ct_image = reader.Execute()
            except Exception as exc:
                logger.warning("Failed to load CT series for %s: %s", course_dir, exc)
                ct_image = None
            return ct_image

        def _totalseg_mask(roi_name: str) -> Optional[np.ndarray]:
            key = roi_name.strip().lower()
            if key in totalseg_mask_cache:
                return totalseg_mask_cache[key]
            seg_root = course_dir / "Segmentation_TotalSegmentator"
            if not seg_root.exists():
                totalseg_mask_cache[key] = None
                return None
            mask_path: Optional[Path] = None
            for subdir in seg_root.iterdir():
                if not subdir.is_dir():
                    continue
                manifest_path = subdir / "manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for model in data.get("models", []):
                    for mask_file in model.get("masks", []):
                        if "--" not in mask_file:
                            continue
                        _, roi_part = mask_file.split("--", 1)
                        roi_base = roi_part.replace('.nii.gz', '').strip().lower()
                        if roi_base == key:
                            candidate = subdir / mask_file
                            if candidate.exists():
                                mask_path = candidate
                                break
                    if mask_path:
                        break
                if mask_path:
                    break
            if not mask_path:
                totalseg_mask_cache[key] = None
                return None
            try:
                img = sitk.ReadImage(str(mask_path))
                reference_ct = _ensure_ct_image()
                if reference_ct is not None:
                    img = sitk.Resample(
                        img,
                        reference_ct,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        0,
                        img.GetPixelID(),
                    )
                arr = sitk.GetArrayFromImage(img)
                mask = np.moveaxis(arr.astype(bool), 0, -1)
            except Exception as exc:
                logger.debug("TotalSegmentator fallback failed for %s: %s", roi_name, exc)
                totalseg_mask_cache[key] = None
                return None
            totalseg_mask_cache[key] = mask
            return mask

        def _harvest_masks(builder: "RTStructBuilder", label: str, add_missing: bool = False) -> None:
            nonlocal existing_names, available_masks
            for roi_name in builder.get_roi_names():
                try:
                    mask = builder.get_roi_mask_by_name(roi_name)
                except Exception as exc:  # pragma: no cover - safety
                    logger.debug("Failed to fetch mask for %s from %s: %s", roi_name, label, exc)
                    mask = None
                if mask is None or not np.any(mask):
                    fallback = _totalseg_mask(roi_name)
                    if fallback is None or not np.any(fallback):
                        continue
                    mask_bool = fallback.astype(bool)
                else:
                    mask_bool = mask.astype(bool)
                available_masks.setdefault(roi_name, mask_bool)
                already_present = roi_name in existing_names
                if add_missing and not already_present:
                    try:
                        rtstruct.add_roi(mask=mask_bool, name=roi_name)
                        existing_names.add(roi_name)
                    except Exception as exc:
                        logger.debug("Unable to add ROI %s from %s: %s", roi_name, label, exc)
                elif not already_present:
                    existing_names.add(roi_name)

        # Harvest base masks first (manual preferred)
        _harvest_masks(rtstruct, f"base:{base_source}")

        # Integrate additional sources so that downstream custom operations can reference them
        if rs_manual and Path(rs_manual).exists() and base_source != "manual":
            try:
                manual_builder = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir),
                    rt_struct_path=str(rs_manual)
                )
                _harvest_masks(manual_builder, "manual", add_missing=True)
            except Exception as exc:
                logger.warning("Failed to integrate manual structures: %s", exc)

        if rs_auto and Path(rs_auto).exists() and base_source != "auto":
            try:
                auto_builder = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir),
                    rt_struct_path=str(rs_auto)
                )
                _harvest_masks(auto_builder, "auto", add_missing=True)
            except Exception as exc:
                logger.warning("Failed to integrate auto structures: %s", exc)

        ct_image = _ensure_ct_image()
        if ct_image is None:
            return None
        spacing = ct_image.GetSpacing()

        # Process custom structures
        processor = CustomStructureProcessor(spacing=spacing)
        if config_path:
            processor.load_config(config_path)

        custom_masks = processor.process_all_custom_structures(available_masks)
        partial_map = getattr(processor, "partial_structures", {})
        warning_entries = []

        # Add custom structures to RTSTRUCT
        for name, mask in custom_masks.items():
            missing_sources = partial_map.get(name, [])
            cropped_flag = mask_is_cropped(mask)
            final_name = name
            if missing_sources:
                logger.warning(
                    "Custom structure %s built with missing sources %s; marking as partial",
                    name,
                    ", ".join(missing_sources),
                )
                final_name = f"{final_name}__partial"
            if cropped_flag and not final_name.endswith("__partial"):
                logger.warning(
                    "Custom structure %s is cropped at image boundary; marking as partial",
                    name,
                )
                final_name = f"{final_name}__partial"
            if missing_sources or cropped_flag:
                warning_entries.append(
                    {
                        "structure": final_name,
                        "original_structure": name,
                        "missing_sources": missing_sources,
                        "cropped": bool(cropped_flag),
                    }
                )
            try:
                rtstruct.add_roi(
                    mask=mask.astype(bool),
                    name=final_name,
                    color=[255, 0, 0]  # Red color for custom structures
                )
                logger.info("Added custom structure: %s", final_name)
            except Exception as e:
                logger.warning("Failed to add custom structure %s: %s", final_name, e)

        if warning_entries:
            try:
                meta_dir = course_dir / "metadata"
                meta_dir.mkdir(parents=True, exist_ok=True)
                flag_path = meta_dir / "custom_structure_warnings.json"
                payload = {
                    "note": "Custom structures generated with incomplete inputs or cropped masks; treat listed structures as partial",
                    "entries": warning_entries,
                }
                flag_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to record custom structure warnings: %s", exc)
        else:
            try:
                flag_path = course_dir / "metadata" / "custom_structure_warnings.json"
                flag_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Save the new RTSTRUCT
        out_path = course_dir / "RS_custom.dcm"
        rtstruct.save(str(out_path))
        try:
            sanitize_rtstruct(out_path)
        except Exception as exc:
            logger.debug("Sanitising RS_custom failed for %s: %s", out_path, exc)
        return out_path

    except Exception as e:
        logger.error("Failed to create custom structures RTSTRUCT: %s", e)
        return None


def _estimate_rx_from_ctv1(rtstruct: pydicom.dataset.FileDataset, rtdose: pydicom.dataset.FileDataset) -> float:
    default_rx = 50.0
    try:
        for roi in rtstruct.StructureSetROISequence:
            name = str(getattr(roi, "ROIName", "") or "")
            if "ctv1" in name.replace(" ", "").lower():
                abs_dvh_ctv = dvhcalc.get_dvh(rtstruct, rtdose, roi.ROINumber)
                if abs_dvh_ctv.volume > 0:
                    bins_ctv = abs_dvh_ctv.bincenters
                    cum_ctv = abs_dvh_ctv.counts
                    return float(_dose_at_fraction(bins_ctv, cum_ctv, 0.95))
    except Exception as e:
        logger.debug("RX estimate failed: %s", e)
    return default_rx


def dvh_for_course(
    course_dir: Path,
    custom_structures_config: Optional[Union[str, Path]] = None,
    parallel_workers: Optional[int] = None,
) -> Optional[Path]:
    course_dirs = build_course_dirs(course_dir)
    rp = course_dir / "RP.dcm"
    rd = course_dir / "RD.dcm"
    rs_manual = course_dir / "RS.dcm"
    rs_auto = course_dir / "RS_auto.dcm"
    if not rd.exists() or not rp.exists():
        logger.warning("Missing RP/RD in %s; skipping DVH", course_dir)
        return None
    try:
        rtdose = pydicom.dcmread(str(rd))
        rtplan = pydicom.dcmread(str(rp))
    except Exception as e:
        logger.error("Failed to read RP/RD: %s", e)
        return None

    results: List[Dict] = []

    try:
        from rt_utils import RTStructBuilder
    except Exception:
        RTStructBuilder = None  # type: ignore

    builder_cache: Dict[Path, Optional["RTStructBuilder"]] = {}  # type: ignore[name-defined]

    def _get_builder(rs_path: Path) -> Optional["RTStructBuilder"]:  # type: ignore[name-defined]
        if RTStructBuilder is None:
            return None
        if rs_path in builder_cache:
            return builder_cache[rs_path]
        if not course_dirs.dicom_ct.exists():
            builder_cache[rs_path] = None
            return None
        try:
            builder = RTStructBuilder.create_from(
                dicom_series_path=str(course_dirs.dicom_ct),
                rt_struct_path=str(rs_path),
            )
        except Exception as exc:
            logger.debug("RTStructBuilder create_from failed for %s: %s", rs_path, exc)
            builder = None
        builder_cache[rs_path] = builder
        return builder

    def _calc_roi(task: tuple) -> Optional[Dict]:
        (
            roi_number,
            roi_name,
            source_label,
            rx_value,
            rtstruct_ds,
            builder_obj,
        ) = task
        try:
            abs_dvh = dvhcalc.get_dvh(rtstruct_ds, rtdose, roi_number)
        except Exception as exc:
            logger.debug("DVH failed for ROI %s: %s", roi_name, exc)
            return None
        if abs_dvh.volume == 0:
            return None
        metrics = _compute_metrics(abs_dvh, rx_value)
        if metrics is None:
            return None
        cropped = False
        if builder_obj is not None:
            try:
                mask = builder_obj.get_roi_mask_by_name(roi_name)
                cropped = mask_is_cropped(mask)
            except Exception:
                cropped = False
        display_name = roi_name
        if cropped and not display_name.endswith("__partial"):
            display_name = f"{display_name}__partial"
        metrics.update(
            {
                "ROI_Number": int(roi_number),
                "ROI_Name": display_name,
                "ROI_OriginalName": roi_name,
                "Segmentation_Source": source_label,
                "structure_cropped": cropped,
            }
        )
        return metrics

    def process_struct(rs_path: Path, source: str, rx_dose: float) -> None:
        try:
            rtstruct = pydicom.dcmread(str(rs_path))
        except Exception as e:
            logger.warning("Cannot read RTSTRUCT %s: %s", rs_path, e)
            return
        builder = _get_builder(rs_path)
        rois = list(getattr(rtstruct, "StructureSetROISequence", []) or [])
        if not rois:
            return
        tasks = []
        for roi in rois:
            roi_name = str(getattr(roi, "ROIName", ""))
            try:
                roi_number = int(roi.ROINumber)
            except Exception:
                roi_number = int(getattr(roi, "ROINumber", 0) or 0)
            if roi_number <= 0:
                continue
            tasks.append((roi_number, roi_name, source, rx_dose, rtstruct, builder))

        if not tasks:
            return

        worker_cap = max(1, parallel_workers or max(1, (os.cpu_count() or 2) // 2))
        task_results = run_tasks_with_adaptive_workers(
            f"DVH ({source})",
            tasks,
            _calc_roi,
            max_workers=worker_cap,
            logger=logger,
        )
        for item in task_results:
            if item:
                results.append(item)

    # Estimate rx from manual if present; else default
    rx_est = 50.0
    if rs_manual.exists():
        try:
            rx_est = _estimate_rx_from_ctv1(pydicom.dcmread(str(rs_manual)), rtdose)
        except Exception:
            pass

    # Use RS_custom.dcm if it exists (created by structure merger)
    rs_custom = course_dir / "RS_custom.dcm"
    if rs_custom.exists():
        process_struct(rs_custom, "Merged", rx_est)
    else:
        # Fallback to individual files if RS_custom doesn't exist
        if rs_manual.exists():
            process_struct(rs_manual, "Manual", rx_est)
        if rs_auto.exists():
            process_struct(rs_auto, "AutoRTS", rx_est)

        # Process custom structures if configuration provided (legacy approach)
        if custom_structures_config:
            try:
                from .custom_structures import CustomStructureProcessor
                rs_custom_legacy = _create_custom_structures_rtstruct(
                    course_dir, custom_structures_config, rs_manual, rs_auto
                )
                if rs_custom_legacy and rs_custom_legacy.exists():
                    process_struct(rs_custom_legacy, "Custom", rx_est)
            except Exception as e:
                logger.warning("Failed to process custom structures: %s", e)

    if not results:
        logger.info("No DVH results for %s", course_dir)
        return None

    df = pd.DataFrame(results)
    out_xlsx = course_dir / "dvh_metrics.xlsx"
    df.to_excel(out_xlsx, index=False)
    return out_xlsx
