"""Helpers for generating RS_custom.dcm without importing DVH dependencies.

Why this module exists:
`rtpipeline.dvh` imports dicompyler-core at import time. The radiomics stage often
runs in a separate environment where DVH dependencies may be absent. Custom
structures (e.g., `pelvic_bones`, `iliac_area`) must be available for radiomics
without requiring dicompyler-core. This module provides the RS_custom build and
staleness logic with minimal imports.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pydicom
import SimpleITK as sitk

from .layout import build_course_dirs
from .course_contract import CourseContractError, load_course_contract
from .utils import mask_is_cropped, sanitize_rtstruct

logger = logging.getLogger(__name__)

_RS_CUSTOM_META_VERSION = 2
_RTSTRUCT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.3"


class CustomStructureRTStructError(RuntimeError):
    """RS_custom could not be built for the named configured ROIs."""

    def __init__(
        self,
        course_dir: Path,
        roi_names: list[str],
        cause: BaseException,
    ) -> None:
        self.course_dir = Path(course_dir)
        self.roi_names = tuple(str(name) for name in roi_names)
        self.cause = cause
        names = ", ".join(self.roi_names) if self.roi_names else "<unresolved>"
        super().__init__(
            f"configured ROI(s) [{names}] could not be built in {self.course_dir}: "
            f"{type(cause).__name__}: {cause}"
        )


def _rtstruct_builder_source(
    source: Path,
    ct_dir: Path,
    temporary_dir: Optional[Path],
) -> tuple[Path, Optional[Path]]:
    """Prune unused cross-series image references from a temporary RTSTRUCT copy.

    Some clinical RTSTRUCTs retain references to an older image series even though
    every surviving ROI contour references the contracted planning CT. rt-utils
    rejects the whole object in that case. Pruning is safe only when every contour
    image reference is present in the contracted CT. A contour outside that series
    remains a hard failure.
    """

    ct_uids: set[str] = set()
    ct_series_uids: set[str] = set()
    for path in Path(ct_dir).rglob("*"):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        sop_uid = str(getattr(dataset, "SOPInstanceUID", "")).strip()
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if sop_uid:
            ct_uids.add(sop_uid)
        if series_uid:
            ct_series_uids.add(series_uid)
    if not ct_uids or len(ct_series_uids) != 1:
        raise ValueError(
            f"contracted planning CT inventory is not one readable series: "
            f"instances={len(ct_uids)}, series={sorted(ct_series_uids)}"
        )

    dataset = pydicom.dcmread(source, stop_before_pixels=True)
    contour_refs = {
        str(image.ReferencedSOPInstanceUID)
        for roi_contour in getattr(dataset, "ROIContourSequence", []) or []
        for contour in getattr(roi_contour, "ContourSequence", []) or []
        for image in getattr(contour, "ContourImageSequence", []) or []
        if getattr(image, "ReferencedSOPInstanceUID", None)
    }
    missing_contour_refs = sorted(contour_refs - ct_uids)
    if missing_contour_refs:
        raise ValueError(
            "RTSTRUCT ROI contours reference image(s) outside the contracted planning "
            f"CT; first missing SOP Instance UID: {missing_contour_refs[0]}"
        )

    unbound_contours = [
        contour
        for roi_contour in getattr(dataset, "ROIContourSequence", []) or []
        for contour in getattr(roi_contour, "ContourSequence", []) or []
        if not (getattr(contour, "ContourImageSequence", None) or [])
    ]
    referenced_images = {
        str(image.ReferencedSOPInstanceUID)
        for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []) or []
        for study in getattr(frame, "RTReferencedStudySequence", []) or []
        for series in getattr(study, "RTReferencedSeriesSequence", []) or []
        for image in getattr(series, "ContourImageSequence", []) or []
        if getattr(image, "ReferencedSOPInstanceUID", None)
    }
    if referenced_images <= ct_uids:
        return Path(source), None
    if unbound_contours:
        raise ValueError(
            "RTSTRUCT contains contour geometry without image-level references; "
            "unused cross-series references cannot be pruned safely"
        )

    planning_series_uid = next(iter(ct_series_uids))
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []) or []:
        studies = []
        for study in getattr(frame, "RTReferencedStudySequence", []) or []:
            series_items = []
            for series in getattr(study, "RTReferencedSeriesSequence", []) or []:
                images = [
                    image
                    for image in getattr(series, "ContourImageSequence", []) or []
                    if str(getattr(image, "ReferencedSOPInstanceUID", "")) in ct_uids
                ]
                if not images:
                    continue
                series.SeriesInstanceUID = planning_series_uid
                series.ContourImageSequence = images
                series_items.append(series)
            if series_items:
                study.RTReferencedSeriesSequence = series_items
                studies.append(study)
        frame.RTReferencedStudySequence = studies

    remaining = {
        str(image.ReferencedSOPInstanceUID)
        for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []) or []
        for study in getattr(frame, "RTReferencedStudySequence", []) or []
        for series in getattr(study, "RTReferencedSeriesSequence", []) or []
        for image in getattr(series, "ContourImageSequence", []) or []
        if getattr(image, "ReferencedSOPInstanceUID", None)
    }
    if not remaining or not remaining <= ct_uids:
        unresolved = sorted(remaining - ct_uids)
        raise ValueError(
            "RTSTRUCT cross-series references could not be safely restricted to the "
            f"contracted planning CT: {unresolved[:3]}"
        )

    temp_kwargs = {}
    if temporary_dir is not None:
        temporary_dir = Path(temporary_dir)
        temporary_dir.mkdir(parents=True, exist_ok=True)
        temp_kwargs["dir"] = str(temporary_dir)
    handle, name = tempfile.mkstemp(
        prefix=f".{Path(source).name}.",
        suffix=".planning-ct.dcm",
        **temp_kwargs,
    )
    os.close(handle)
    temporary = Path(name)
    try:
        pydicom.dcmwrite(temporary, dataset, write_like_original=False)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    logger.info(
        "Prepared a temporary planning-CT-only RTSTRUCT reference copy for %s; "
        "the authoritative source was not modified",
        source,
    )
    return temporary, temporary


def _is_valid_rtstruct(path: Path) -> bool:
    """Return True only if `path` parses as a DICOM RTSTRUCT with >=1 ROI.

    A process killed mid-write leaves a truncated file: pydicom.dcmread either raises
    on it, or parses a dataset missing SOPClassUID/StructureSetROISequence. Both must
    be treated as incomplete so resume regenerates instead of accepting a truncated
    file forever (the exact class of bug 3bd8c5d fixed for segmentation masks/manifests).
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception:
        return False
    if str(getattr(ds, "SOPClassUID", "")) != _RTSTRUCT_SOP_CLASS_UID:
        return False
    return bool(getattr(ds, "StructureSetROISequence", None))


def _write_rtstruct_atomic(out_path: Path, write_fn, validate_fn=None) -> None:
    """Write an RTSTRUCT by calling `write_fn(tmp_path_str)`, then atomically publish it
    at `out_path` via os.replace().

    rt_utils' RTStruct.save() writes straight to the destination path; a process killed
    mid-write would leave a truncated RTSTRUCT that a presence-only resume check accepts
    forever.

    The temp path must end in `.dcm`: rt_utils' RTStruct.save() auto-appends `.dcm`
    to any path that doesn't already end with it, which would silently redirect the
    write to `tmp_path + ".dcm"` and make the subsequent os.replace(tmp_path, ...)
    raise FileNotFoundError.
    """
    tmp_path = out_path.parent / f".{out_path.name}.{os.getpid()}.tmp.dcm"
    try:
        write_fn(str(tmp_path))
        if validate_fn is not None:
            validate_fn(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _quarantine_rejected_rtstruct(path: Path, reason: str) -> Optional[Path]:
    """Revoke a rejected RS_custom before attempting its replacement."""
    if not path.exists():
        return None
    counter = 0
    while True:
        suffix = f".{os.getpid()}" if counter == 0 else f".{os.getpid()}.{counter}"
        quarantine = path.parent / f".{path.name}.rejected{suffix}"
        if not quarantine.exists():
            break
        counter += 1
    try:
        os.replace(path, quarantine)
    except OSError:
        try:
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise OSError(f"could not revoke rejected RTSTRUCT {path}") from exc
        quarantine = None
    logger.info("Revoked rejected RTSTRUCT %s (%s)", path, reason)
    return quarantine


def record_rs_custom_resume_decision(course_dir: Path, action: str, reason: str) -> None:
    """Persist the RS_custom decision without making auditing a rebuild dependency."""
    try:
        from .course_contract import load_course_contract
        from .segmentation import record_segmentation_resume_decision

        contract = load_course_contract(course_dir)
        record_segmentation_resume_decision(
            course_dir,
            {
                "RS_custom": {
                    "action": action,
                    "model_run": False,
                    "artefact": "RS_custom.dcm",
                    "reason": reason,
                }
            },
            source={
                "planning_ct_series_instance_uid": str(
                    contract.planning_ct.get("series_instance_uid") or ""
                )
            },
        )
    except Exception as exc:
        logger.warning("Could not record RS_custom resume decision for %s: %s", course_dir, exc)

def _roi_numbers(rtstruct_ds) -> list[int]:
    numbers: list[int] = []
    for roi in getattr(rtstruct_ds, "StructureSetROISequence", []) or []:
        try:
            numbers.append(int(roi.ROINumber))
        except Exception:
            continue
    return numbers


def _assert_unique_roi_numbers(rtstruct_ds, context: str) -> None:
    numbers = _roi_numbers(rtstruct_ds)
    duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
    if duplicates:
        raise ValueError(f"Duplicate ROI numbers in {context}: {duplicates}")


def _add_roi_with_unique_number(rtstruct, *, mask: np.ndarray, name: str, **kwargs) -> None:
    """Add an ROI while avoiding rt-utils len(sequence)+1 number collisions."""
    next_number = max(_roi_numbers(rtstruct.ds), default=0) + 1
    rtstruct.add_roi(mask=mask, name=name, **kwargs)

    rtstruct.ds.StructureSetROISequence[-1].ROINumber = next_number
    rtstruct.ds.ROIContourSequence[-1].ReferencedROINumber = next_number
    rtstruct.ds.RTROIObservationsSequence[-1].ObservationNumber = next_number
    rtstruct.ds.RTROIObservationsSequence[-1].ReferencedROINumber = next_number
    _assert_unique_roi_numbers(rtstruct.ds, f"after adding {name}")


def _is_rs_custom_stale(
    rs_custom_path: Path,
    config_path: Optional[Union[str, Path]],
    rs_manual: Optional[Path],
    rs_auto: Optional[Path],
    *,
    allow_contractless: bool = False,
) -> bool:
    """Return True when RS_custom.dcm should be regenerated.

    Production callers require the authoritative course contract.  The explicit
    ``allow_contractless`` mode is retained only for small utility callers that
    intentionally use the historical mtime-only behavior.
    """
    if not rs_custom_path.exists():
        return True

    if not _is_valid_rtstruct(rs_custom_path):
        # A process killed mid-write leaves a truncated file; mtime/metadata-based
        # staleness checks below never parse the file, so they would otherwise accept
        # it forever. Regenerate instead of trusting a file that doesn't even parse.
        logger.warning("RS_custom.dcm at %s failed to parse/validate; regenerating", rs_custom_path)
        return True

    try:
        course_dir = rs_custom_path.parent
        contract = None
        if not allow_contractless:
            try:
                contract = load_course_contract(course_dir)
            except Exception as exc:
                logger.warning(
                    "Cannot establish the authoritative course contract for %s; "
                    "rejecting RS_custom.dcm: %s",
                    course_dir,
                    exc,
                )
                return True
        if contract is not None:
            planning_ct = contract.planning_ct
            planning_series_uid = str(planning_ct.get("series_instance_uid") or "").strip()
            if not planning_series_uid:
                logger.warning("Planning CT series identity is absent for %s; regenerating RS_custom.dcm", course_dir)
                return True
            try:
                from .auto_rtstruct import _seg_source_series_uids

                source_series_uids = _seg_source_series_uids(rs_custom_path)
            except Exception:
                source_series_uids = set()
            if source_series_uids != {planning_series_uid}:
                logger.warning(
                    "RS_custom.dcm at %s does not reference planning CT series %s; regenerating",
                    rs_custom_path,
                    planning_series_uid,
                )
                return True
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

        if contract is not None:
            dependency_paths: list[Path] = []
            planning_ct_dir = contract.planning_ct_dir
            if planning_ct_dir is not None:
                dependency_paths.extend(item for item in planning_ct_dir.rglob("*") if item.is_file())
            planning_ct_nifti = contract.planning_ct_nifti
            if planning_ct_nifti is not None and planning_ct_nifti.is_file():
                dependency_paths.append(planning_ct_nifti)
            if any(item.stat().st_mtime > rs_custom_mtime for item in dependency_paths):
                logger.info("Contracted planning CT input is newer than RS_custom.dcm, regenerating")
                return True

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
        if contract is not None:
            try:
                from .segmentation import record_segmentation_resume_decision

                record_segmentation_resume_decision(
                    course_dir,
                    {
                        "RS_custom": {
                            "action": "reused",
                            "model_run": False,
                            "artefact": "RS_custom.dcm",
                            "reason": "valid RTSTRUCT references the contracted planning CT series and inputs are current",
                        }
                    },
                    source={
                        "planning_ct_series_instance_uid": str(
                            contract.planning_ct.get("series_instance_uid") or ""
                        ),
                        "artifact": "RS_custom.dcm",
                    },
                )
            except Exception as exc:
                logger.warning("Could not record RS_custom reuse for %s: %s", course_dir, exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to check RS_custom.dcm staleness (%s); regenerating", exc)
        return True


def _create_custom_structures_rtstruct_unlocked(
    course_dir: Path,
    config_path: Optional[Union[str, Path]] = None,
    rs_manual: Optional[Path] = None,
    rs_auto: Optional[Path] = None,
) -> Optional[Path]:
    """Create a new RTSTRUCT with custom structures from boolean operations."""
    contract = load_course_contract(course_dir)
    contracted_manual = contract.authoritative_rtstruct_path
    if rs_manual is not None and contracted_manual is not None:
        if Path(rs_manual).resolve(strict=False) != contracted_manual.resolve(strict=False):
            raise CourseContractError(
                "custom-structure caller RTSTRUCT disagrees with the authoritative course contract"
            )
    rs_manual = contracted_manual
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
    ct_dir = contract.planning_ct_dir
    if ct_dir is None:
        logger.warning("Course contract has no planning CT for custom structures")
        return None

    processor = CustomStructureProcessor()
    if config_path:
        processor.load_config(config_path)
    temporary_sources: list[Path] = []
    try:
        # rt-utils rejects any cross-series image reference, even an unused
        # historical series. Restrict only a temporary copy and only when every
        # surviving contour is already bound to the contracted planning CT.
        builder_source, temporary_source = _rtstruct_builder_source(
            base_rs,
            ct_dir,
            None,
        )
        if temporary_source is not None:
            temporary_sources.append(temporary_source)
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(builder_source),
        )

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

                        _add_roi_with_unique_number(
                            rtstruct,
                            mask=mask.astype(bool),
                            name=final_name,
                            color=[0, 128, 255],
                        )
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
                        _add_roi_with_unique_number(rtstruct, mask=mask_bool, name=roi_name)
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
                manual_source, manual_temporary = _rtstruct_builder_source(
                    Path(rs_manual), ct_dir, None
                )
                if manual_temporary is not None:
                    temporary_sources.append(manual_temporary)
                manual_builder = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir),
                    rt_struct_path=str(manual_source),
                )
                _harvest_masks(manual_builder, "manual", add_missing=True)
            except Exception as exc:
                logger.warning("Failed to integrate manual structures: %s", exc)

        if rs_auto and Path(rs_auto).exists() and base_source != "auto":
            try:
                auto_source, auto_temporary = _rtstruct_builder_source(
                    Path(rs_auto), ct_dir, None
                )
                if auto_temporary is not None:
                    temporary_sources.append(auto_temporary)
                auto_builder = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir),
                    rt_struct_path=str(auto_source),
                )
                _harvest_masks(auto_builder, "auto", add_missing=True)
            except Exception as exc:
                logger.warning("Failed to integrate auto structures: %s", exc)

        # Harvest masks from custom models (cardiac_STOPSTORM, etc.) and add to RS_custom
        _harvest_custom_model_masks()

        ct_image = _ensure_ct_image()
        if ct_image is None:
            return None
        spacing = ct_image.GetSpacing()

        processor.spacing = spacing

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
                _add_roi_with_unique_number(
                    rtstruct,
                    mask=mask.astype(bool),
                    name=final_name,
                    color=[255, 0, 0],
                )
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
        _assert_unique_roi_numbers(rtstruct.ds, f"RS_custom before save for {course_dir}")

        def _validate_temporary_publication(path: Path) -> None:
            try:
                sanitize_rtstruct(path)
            except Exception as exc:
                logger.debug("Sanitising temporary RS_custom failed for %s: %s", path, exc)
            dataset = pydicom.dcmread(str(path))
            if str(getattr(dataset, "SOPClassUID", "")) != _RTSTRUCT_SOP_CLASS_UID:
                raise ValueError(f"temporary RS_custom is not an RTSTRUCT: {path}")
            if not getattr(dataset, "StructureSetROISequence", None):
                raise ValueError(f"temporary RS_custom has no named ROIs: {path}")
            _assert_unique_roi_numbers(dataset, f"temporary RS_custom for {course_dir}")

        _write_rtstruct_atomic(
            out_path,
            rtstruct.save,
            validate_fn=_validate_temporary_publication,
        )

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
        try:
            from .segmentation import record_segmentation_resume_decision

            record_segmentation_resume_decision(
                course_dir,
                {
                    "RS_custom": {
                        "action": "rebuilt",
                        "model_run": False,
                        "artefact": "RS_custom.dcm",
                        "reason": "rebuilt from current contracted RTSTRUCT and source masks",
                    }
                },
                source={
                    "planning_ct_series_instance_uid": str(
                        contract.planning_ct.get("series_instance_uid") or ""
                    ),
                    "artifact": "RS_custom.dcm",
                },
            )
        except Exception as exc:
            logger.warning("Could not record RS_custom rebuild for %s: %s", course_dir, exc)
        return out_path

    except Exception as exc:  # pragma: no cover - defensive
        processor_value = locals().get("processor")
        roi_names = [
            str(config.name)
            for config in getattr(processor_value, "custom_configs", [])
        ]
        if isinstance(exc, CustomStructureRTStructError):
            logger.error("Failed to create custom structures RTSTRUCT: %s", exc)
            raise
        error = CustomStructureRTStructError(course_dir, roi_names, exc)
        logger.error("Failed to create custom structures RTSTRUCT: %s", error)
        raise error from exc
    finally:
        for temporary_source in temporary_sources:
            temporary_source.unlink(missing_ok=True)


def _create_custom_structures_rtstruct(
    course_dir: Path,
    config_path: Optional[Union[str, Path]] = None,
    rs_manual: Optional[Path] = None,
    rs_auto: Optional[Path] = None,
) -> Optional[Path]:
    """Serialize RS_custom production and reuse a current competing publication."""

    course_dir = Path(course_dir)
    lock_dir = course_dir / "metadata"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".rs_custom.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        out_path = course_dir / "RS_custom.dcm"
        if out_path.is_file() and not _is_rs_custom_stale(
            out_path,
            config_path,
            rs_manual,
            rs_auto,
        ):
            logger.info("Reusing RS_custom published by a competing course stage: %s", out_path)
            return out_path
        return _create_custom_structures_rtstruct_unlocked(
            course_dir,
            config_path,
            rs_manual,
            rs_auto,
        )
