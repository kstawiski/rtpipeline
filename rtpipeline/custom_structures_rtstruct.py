"""Helpers for generating RS_custom.dcm without importing DVH dependencies.

Why this module exists:
`rtpipeline.dvh` imports dicompyler-core at import time. The radiomics stage often
runs in a separate environment where DVH dependencies may be absent. Custom
structures (e.g., `pelvic_bones`, `iliac_area`) must be available for radiomics
without requiring dicompyler-core. This module provides the RS_custom build and
staleness logic with minimal imports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import SimpleITK as sitk

from .layout import build_course_dirs
from .utils import mask_is_cropped, sanitize_rtstruct

logger = logging.getLogger(__name__)

_RS_CUSTOM_META_VERSION = 2


def _is_rs_custom_stale(
    rs_custom_path: Path,
    config_path: Optional[Union[str, Path]],
    rs_manual: Optional[Path],
    rs_auto: Optional[Path],
) -> bool:
    """Return True when RS_custom.dcm should be regenerated."""
    if not rs_custom_path.exists():
        return True

    try:
        course_dir = rs_custom_path.parent
        # RS_auto_cropped.dcm has been observed to be geometrically misregistered
        # when paired with the original CT series. Many existing RS_custom.dcm
        # files were generated in workflows that preferred RS_auto_cropped, so we
        # force a one-time regeneration (tracked via a small metadata file) when
        # a cropped auto RTSTRUCT is present.
        rs_custom_meta = course_dir / "metadata" / "rs_custom_meta.json"
        if (course_dir / "RS_auto_cropped.dcm").exists():
            try:
                meta = json.loads(rs_custom_meta.read_text(encoding="utf-8")) if rs_custom_meta.exists() else {}
                if int(meta.get("version", 0) or 0) < _RS_CUSTOM_META_VERSION:
                    logger.info("RS_custom meta missing/outdated in %s; regenerating to avoid cropped-geometry issues", course_dir)
                    return True
            except Exception:
                logger.info("Failed to read rs_custom_meta.json in %s; regenerating", course_dir)
                return True

        rs_custom_mtime = rs_custom_path.stat().st_mtime

        if config_path:
            config_path = Path(config_path)
            if config_path.exists() and config_path.stat().st_mtime > rs_custom_mtime:
                logger.info("Custom structures config is newer than RS_custom.dcm, regenerating")
                return True

        for source_rs in [rs_manual, rs_auto]:
            if source_rs and Path(source_rs).exists() and Path(source_rs).stat().st_mtime > rs_custom_mtime:
                logger.info("Source RTSTRUCT %s is newer than RS_custom.dcm, regenerating", Path(source_rs).name)
                return True

        seg_root = course_dir / "Segmentation_TotalSegmentator"
        if seg_root.exists():
            for item in seg_root.rglob("*.nii.gz"):
                if item.stat().st_mtime > rs_custom_mtime:
                    logger.info("TotalSegmentator output is newer than RS_custom.dcm, regenerating")
                    return True

        # Also check CustomModels outputs for staleness (cardiac_STOPSTORM, etc.)
        custom_seg_root = course_dir / "Segmentation_CustomModels"
        if custom_seg_root.exists():
            for item in custom_seg_root.rglob("*.nii.gz"):
                if item.stat().st_mtime > rs_custom_mtime:
                    logger.info("CustomModel output is newer than RS_custom.dcm, regenerating")
                    return True

        logger.debug("RS_custom.dcm is up-to-date, reusing existing file")
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to check RS_custom.dcm staleness (%s); regenerating", exc)
        return True


def _create_custom_structures_rtstruct(
    course_dir: Path,
    config_path: Optional[Union[str, Path]] = None,
    rs_manual: Optional[Path] = None,
    rs_auto: Optional[Path] = None,
) -> Optional[Path]:
    """Create a new RTSTRUCT with custom structures from boolean operations."""
    try:
        from .custom_structures import CustomStructureProcessor
        from rt_utils import RTStructBuilder
    except ImportError as exc:
        logger.warning("rt-utils not available for custom structures: %s", exc)
        return None

    # Choose base RTSTRUCT (prefer manual over auto)
    base_rs: Optional[Path] = None
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
        # Load base RTSTRUCT
        rtstruct = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(base_rs))

        existing_names: set[str] = set()
        available_masks: Dict[str, np.ndarray] = {}
        totalseg_mask_cache: Dict[str, Optional[np.ndarray]] = {}
        custom_model_mask_cache: Dict[str, Optional[np.ndarray]] = {}
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
                        roi_base = roi_part.replace(".nii.gz", "").strip().lower()
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

        def _custom_model_mask(roi_name: str, model_name: str) -> Optional[np.ndarray]:
            """Load a mask from Segmentation_CustomModels/<model_name>/<mask_name>.nii.gz"""
            cache_key = f"{model_name}:{roi_name}".lower()
            if cache_key in custom_model_mask_cache:
                return custom_model_mask_cache[cache_key]

            custom_seg_root = course_dir / "Segmentation_CustomModels" / model_name
            if not custom_seg_root.exists():
                custom_model_mask_cache[cache_key] = None
                return None

            # Look for mask file matching roi_name
            mask_path: Optional[Path] = None
            roi_lower = roi_name.strip().lower()

            for nii_file in custom_seg_root.glob("*.nii.gz"):
                # Custom model masks are typically named directly as structure name
                file_stem = nii_file.stem.replace(".nii", "").strip().lower()
                if file_stem == roi_lower:
                    mask_path = nii_file
                    break

            if not mask_path:
                custom_model_mask_cache[cache_key] = None
                return None

            try:
                img = sitk.ReadImage(str(mask_path))
                reference_ct = _ensure_ct_image()
                if reference_ct is not None:
                    # Resample to CT geometry
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
                logger.warning("CustomModel mask loading failed for %s/%s: %s", model_name, roi_name, exc)
                custom_model_mask_cache[cache_key] = None
                return None

            custom_model_mask_cache[cache_key] = mask
            return mask

        def _harvest_custom_model_masks() -> None:
            """Harvest all masks from Segmentation_CustomModels and add to RS_custom.dcm"""
            nonlocal existing_names

            custom_seg_root = course_dir / "Segmentation_CustomModels"
            if not custom_seg_root.exists():
                return

            for model_dir in sorted(custom_seg_root.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name

                for nii_file in sorted(model_dir.glob("*.nii.gz")):
                    roi_name = nii_file.stem.replace(".nii", "")
                    # Prefix with model name to avoid collisions (e.g., STOPSTORM_Heart)
                    prefixed_name = f"{model_name}_{roi_name}"

                    if prefixed_name in existing_names:
                        logger.debug("CustomModel ROI %s already exists, skipping", prefixed_name)
                        continue

                    mask = _custom_model_mask(roi_name, model_name)
                    if mask is None or not np.any(mask):
                        logger.debug("CustomModel mask %s/%s is empty, skipping", model_name, roi_name)
                        continue

                    try:
                        cropped_flag = mask_is_cropped(mask)
                        final_name = prefixed_name
                        if cropped_flag:
                            logger.warning("CustomModel structure %s is cropped at image boundary; marking as partial", prefixed_name)
                            final_name = f"{final_name}__partial"

                        rtstruct.add_roi(mask=mask.astype(bool), name=final_name, color=[0, 128, 255])
                        existing_names.add(final_name)
                        available_masks[final_name] = mask
                        logger.info("Added CustomModel structure: %s from %s", final_name, model_name)
                    except Exception as exc:
                        logger.warning("Failed to add CustomModel ROI %s: %s", prefixed_name, exc)

        def _harvest_masks(builder: "RTStructBuilder", label: str, add_missing: bool = False) -> None:
            nonlocal existing_names, available_masks
            for roi_name in builder.get_roi_names():
                try:
                    mask = builder.get_roi_mask_by_name(roi_name)
                except Exception as exc:  # pragma: no cover - defensive
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

        # Integrate additional sources to enable custom ops that reference them
        if rs_manual and Path(rs_manual).exists() and base_source != "manual":
            try:
                manual_builder = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rs_manual))
                _harvest_masks(manual_builder, "manual", add_missing=True)
            except Exception as exc:
                logger.warning("Failed to integrate manual structures: %s", exc)

        if rs_auto and Path(rs_auto).exists() and base_source != "auto":
            try:
                auto_builder = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rs_auto))
                _harvest_masks(auto_builder, "auto", add_missing=True)
            except Exception as exc:
                logger.warning("Failed to integrate auto structures: %s", exc)

        # Harvest masks from custom models (cardiac_STOPSTORM, etc.) and add to RS_custom
        _harvest_custom_model_masks()

        ct_image = _ensure_ct_image()
        if ct_image is None:
            return None
        spacing = ct_image.GetSpacing()

        processor = CustomStructureProcessor(spacing=spacing)
        if config_path:
            processor.load_config(config_path)

        custom_masks = processor.process_all_custom_structures(available_masks)
        partial_map = getattr(processor, "partial_structures", {})
        warning_entries = []

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
                logger.warning("Custom structure %s is cropped at image boundary; marking as partial", name)
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
                rtstruct.add_roi(mask=mask.astype(bool), name=final_name, color=[255, 0, 0])
                logger.info("Added custom structure: %s", final_name)
            except Exception as exc:
                logger.warning("Failed to add custom structure %s: %s", final_name, exc)

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

        out_path = course_dir / "RS_custom.dcm"
        rtstruct.save(str(out_path))
        try:
            sanitize_rtstruct(out_path)
        except Exception as exc:
            logger.debug("Sanitising RS_custom failed for %s: %s", out_path, exc)

        # Record generator metadata to enable safe/stable staleness checks.
        try:
            meta_dir = course_dir / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_payload = {
                "version": _RS_CUSTOM_META_VERSION,
                "base_source": base_source,
                "base_rtstruct": str(base_rs.name if base_rs else ""),
                "rs_manual_present": bool(rs_manual and Path(rs_manual).exists()),
                "rs_auto_present": bool(rs_auto and Path(rs_auto).exists()),
                "note": "Generated in CT DICOM coordinates; do not rely on RS_auto_cropped.dcm for radiomics",
            }
            (meta_dir / "rs_custom_meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed writing rs_custom_meta.json for %s: %s", course_dir, exc)
        return out_path

    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to create custom structures RTSTRUCT: %s", exc)
        return None
