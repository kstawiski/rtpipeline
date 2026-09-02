from __future__ import annotations

import json
import logging
import os
import shutil
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pydicom
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)

from .layout import build_course_dirs
from .course_contract import load_course_contract
from .utils import sanitize_rtstruct
from .roi_fixer import fix_rtstruct_rois

_RTSTRUCT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.3"


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


def _write_rtstruct_atomic(out_path: Path, write_fn) -> None:
    """Write an RTSTRUCT by calling `write_fn(tmp_path_str)`, then atomically publish it
    at `out_path` via os.replace().

    rt_utils' RTStruct.save() (and a raw file copy) write straight to the destination
    path; a process killed mid-write would leave a truncated RTSTRUCT that a
    presence-only resume check accepts forever.

    The temp path must end in `.dcm`: rt_utils' RTStruct.save() auto-appends `.dcm`
    to any path that doesn't already end with it, which would silently redirect the
    write to `tmp_path + ".dcm"` and make the subsequent os.replace(tmp_path, ...)
    raise FileNotFoundError.
    """
    tmp_path = out_path.parent / f".{out_path.name}.{os.getpid()}.tmp.dcm"
    try:
        write_fn(str(tmp_path))
        os.replace(tmp_path, out_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


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
        from .dicom_seg import load_dicom_seg_multiclass
    except Exception as e:
        logger.debug("DICOM-SEG decoder unavailable: %s", e)
        return None, {}
    try:
        return load_dicom_seg_multiclass(seg_path)
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
        for specific_name in (
            f"{base_name}_total_multilabel.nii.gz",
            f"{base_name}--total--multilabel.nii.gz",
            "total--multilabel.nii.gz",
        ):
            specific = seg_dir / specific_name
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
        json_candidates.extend(sorted(seg_dir.glob("total--segmentations.json")))

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
        # Clean name: remove .nii and .nii.gz suffixes
        name = mask_path.name
        if name.endswith('.nii.gz'):
            name = name[:-7]  # Remove .nii.gz
        elif name.endswith('.nii'):
            name = name[:-4]  # Remove .nii
        if prefix and not name.startswith(prefix):
            continue
        if name.lower().endswith(("--segmentations", "--multilabel")):
            continue
        try:
            img = sitk.ReadImage(str(mask_path))
        except Exception as e:
            logger.debug("Skipping mask %s: %s", mask_path.name, e)
            continue
        if prefix:
            name = name[len(prefix):] or name
        masks.append((name, img))
    return masks


def _pretty_roi_name(name: str) -> str:
    """Strip TotalSegmentator prefixes (e.g., 'total--') from ROI names."""
    if not name:
        return name

    working = name.strip()
    suffix = ""
    if working.endswith("__partial"):
        working = working[:-9]
        suffix = "__partial"

    dash_parts = [part for part in working.split("--") if part]
    if dash_parts:
        candidate = dash_parts[-1]
    else:
        candidate = working

    # Remove known TotalSegmentator prefixes
    lowered = candidate.lower()
    for prefix in ("total__", "total_", "total-"):
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix):]
            lowered = candidate.lower()
            break
    if lowered.startswith("total"):
        trimmed = candidate[5:].lstrip("_-")
        if trimmed:
            candidate = trimmed

    cleaned = candidate or working
    return cleaned + suffix


def _unique_roi_name(base_name: str, used_names: set[str]) -> str:
    """Return a deterministic ROI name that cannot loop on an existing suffix."""
    if base_name not in used_names:
        return base_name
    candidate = f"{base_name}_dup"
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_dup{suffix}"
        suffix += 1
    return candidate


def _resample_to_reference(seg_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    if (seg_img.GetSize() == ref_img.GetSize() and
        seg_img.GetSpacing() == ref_img.GetSpacing() and
        seg_img.GetDirection() == ref_img.GetDirection() and
        seg_img.GetOrigin() == ref_img.GetOrigin()):
        return seg_img
    res = sitk.Resample(seg_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg_img.GetPixelID())
    return res


def _read_for_uid(dcm_path: Path) -> str:
    """FrameOfReferenceUID from a DICOM file; '' if unreadable/absent.

    Reads the top-level tag (present on CT slices and DICOM-SEG) and, when absent,
    the nested ``ReferencedFrameOfReferenceSequence`` carried by an RTSTRUCT. The
    rtpipeline ``--total.dcm`` default output type is ``dicom_rtstruct`` (SOP
    481.3), which has no top-level FrameOfReferenceUID, so the nested read is the
    common path here — without it every RTSTRUCT ``--total.dcm`` would report ''.
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    except Exception as e:
        logger.debug("Could not read FrameOfReferenceUID from %s: %s", dcm_path, e)
        return ""
    uid = str(getattr(ds, "FrameOfReferenceUID", "") or "")
    if uid:
        return uid
    for ref_for in getattr(ds, "ReferencedFrameOfReferenceSequence", []) or []:
        nested = str(getattr(ref_for, "FrameOfReferenceUID", "") or "")
        if nested:
            return nested
    return ""


def _seg_source_series_uids(dcm_path: Path) -> set[str]:
    """Source CT SeriesInstanceUIDs that a ``--total.dcm`` references.

    The exact provenance link used to bind masks to the planning CT:
      * RTSTRUCT (rtpipeline default): ReferencedFrameOfReferenceSequence ->
        RTReferencedStudySequence -> RTReferencedSeriesSequence, reusing the
        already-reviewed :func:`rtpipeline.organize.referenced_ct_series_uids`.
      * DICOM-SEG: the top-level ``ReferencedSeriesSequence``.
    Empty set when the file is unreadable or carries no such link.
    """
    uids: set[str] = set()
    try:
        from .organize import referenced_ct_series_uids
        uids |= set(referenced_ct_series_uids(dcm_path))
    except Exception as e:
        logger.debug("RTSTRUCT source-series read failed for %s: %s", dcm_path, e)
    if not uids:
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
            for series in getattr(ds, "ReferencedSeriesSequence", []) or []:
                uid = str(getattr(series, "SeriesInstanceUID", "") or "").strip()
                if uid:
                    uids.add(uid)
        except Exception as e:
            logger.debug("SEG source-series read failed for %s: %s", dcm_path, e)
    return uids


def _rtstruct_matches_planning_ct(path: Path, planning_ct_series_uid: str) -> bool:
    """Require an existing derived RTSTRUCT to reference exactly the current CT series."""
    expected = str(planning_ct_series_uid or "").strip()
    if not expected or not _is_valid_rtstruct(path):
        return False
    return _seg_source_series_uids(path) == {expected}


def _derived_rtstruct_dependencies_are_current(
    output_path: Path,
    *,
    ct_dir: Path,
    nifti_path: Optional[Path],
    segmentation_root: Path,
) -> bool:
    """Reject an otherwise valid RTSTRUCT when a source artifact is newer."""
    try:
        output_mtime = output_path.stat().st_mtime
        dependencies: list[Path] = [
            item for item in ct_dir.rglob("*") if item.is_file()
        ]
        if nifti_path is not None and nifti_path.is_file():
            dependencies.append(nifti_path)
        if segmentation_root.exists():
            dependencies.extend(
                item
                for item in segmentation_root.rglob("*")
                if item.is_file() and item.suffix in {".gz", ".nii", ".dcm"}
            )
        return not any(item.stat().st_mtime > output_mtime for item in dependencies)
    except Exception:
        return False


def _record_auto_resume_decision(
    course_dir: Path,
    action: str,
    reason: str,
    *,
    rejected_artifact: Optional[dict[str, str]] = None,
) -> None:
    """Append the RS_auto reuse/rebuild decision without coupling imports at module load."""
    try:
        from .segmentation import record_segmentation_resume_decision

        contract = load_course_contract(course_dir)
        planning_ct = contract.planning_ct
        source = {
            "planning_ct_series_instance_uid": str(planning_ct.get("series_instance_uid") or ""),
            "artifact": "RS_auto.dcm",
        }
        decision: dict[str, object] = {
            "action": action,
            "model_run": False,
            "artefact": "RS_auto.dcm",
            "reason": reason,
        }
        if rejected_artifact is not None:
            decision["rejected_artifact"] = dict(rejected_artifact)
        record_segmentation_resume_decision(
            course_dir,
            {"RS_auto": decision},
            source=source,
        )
    except Exception as exc:
        logger.warning("Could not record RS_auto resume decision for %s: %s", course_dir, exc)


def _remove_rejected_rtstruct(path: Path, reason: str) -> None:
    """Remove a rejected RTSTRUCT before any replacement work begins."""
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        raise OSError(f"could not remove rejected RTSTRUCT {path}") from exc
    logger.warning("Removed rejected RTSTRUCT %s before rebuild because %s", path, reason)


def _read_ct_for_uid(ct_dir: Path) -> str:
    """FrameOfReferenceUID of the planning-CT series (first readable slice); '' if unknown."""
    try:
        for slice_path in sorted(ct_dir.glob("*.dcm")):
            uid = _read_for_uid(slice_path)
            if uid:
                return uid
    except Exception as e:
        logger.debug("Could not determine CT FrameOfReferenceUID for %s: %s", ct_dir, e)
    return ""


def _read_ct_series_uid(ct_dir: Path) -> str:
    """Read identity within an already selected CT directory.

    This helper does not choose a planning series. Course callers must obtain
    ``ct_dir`` from the authoritative course contract before calling it.
    """
    try:
        for slice_path in sorted(ct_dir.glob("*.dcm")):
            try:
                dataset = pydicom.dcmread(
                    str(slice_path),
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=["SeriesInstanceUID"],
                )
            except Exception:
                continue
            uid = str(getattr(dataset, "SeriesInstanceUID", "") or "").strip()
            if uid:
                return uid
    except Exception:
        pass
    return ""


def _select_seg_dir_for_ct(
    candidate_dirs: Iterable[Path],
    ct_series_uid: Optional[str],
    ct_for_uid: Optional[str],
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Pick the TotalSegmentator seg dir built from the planning CT.

    Returns ``(selected_dir, dicom_seg_path, base_name)``. Selection order:

      1. **Source-series identity** (exact): the seg dir whose ``--total.dcm``
         references the planning CT's SeriesInstanceUID. This is the only signal
         that disambiguates 4DCT phases / multiple reconstructions that *share* a
         FrameOfReferenceUID but are distinct series.
      2. **FrameOfReferenceUID** (coarser): when no source-series link is readable,
         the seg dir whose ``--total.dcm`` shares the planning CT's FoR.
      3. **Lone-candidate back-compat**: a single candidate with no *contradicting*
         readable identity (legacy single-series / NIfTI-only dirs).

    Fail-closed at every rung: several candidates with no unambiguous match (or a
    lone candidate whose readable identity contradicts the planning CT) returns
    ``(None, None, None)`` rather than guessing ``candidate_dirs[0]`` — binding masks
    from the wrong series/frame onto the planning CT is a correctness defect, not a
    degraded-but-acceptable output. ``dicom_seg_path`` is ``None`` for a NIfTI-only
    seg dir (the caller then geometry-checks the loaded image via
    :func:`_geometry_compatible`).
    """
    dirs = [d for d in candidate_dirs if d.is_dir()]
    if not dirs:
        return (None, None, None)

    def _total_dcm(d: Path) -> Optional[Path]:
        cand = d / f"{d.name}--total.dcm"
        return cand if cand.exists() else None

    segs = [(d, _total_dcm(d)) for d in dirs]

    # 1. Source-series identity — exact provenance, disambiguates same-FoR series.
    if ct_series_uid:
        matches = [
            (d, seg, d.name)
            for d, seg in segs
            if seg is not None and ct_series_uid in _seg_source_series_uids(seg)
        ]
        if len(matches) == 1:
            d, seg, name = matches[0]
            # Defense-in-depth: a well-formed TS RTSTRUCT references the planning
            # series AND carries its FrameOfReferenceUID. A readable disagreement
            # signals a malformed/stale structure set -> fail-closed (the RTSTRUCT
            # is later copied verbatim, bypassing the geometry net).
            seg_for = _read_for_uid(seg)
            if ct_for_uid and seg_for and seg_for != ct_for_uid:
                logger.error(
                    "Auto RTSTRUCT: %s references the planning CT series %s but carries "
                    "FrameOfReferenceUID %s (!= planning %s); malformed provenance, "
                    "refusing to bind.",
                    seg, ct_series_uid, seg_for, ct_for_uid,
                )
                return (None, None, None)
            return (d, seg, name)
        if len(matches) > 1:
            logger.error(
                "Auto RTSTRUCT: %d TotalSegmentator outputs reference the planning CT "
                "series %s; ambiguous provenance, refusing to guess.",
                len(matches), ct_series_uid,
            )
            return (None, None, None)

    # 2. FrameOfReferenceUID — coarser fallback only when a candidate has no readable
    # source-series link. A readable link to another series is contradictory provenance,
    # even when that series shares the planning CT's frame (for example 4DCT phases).
    if ct_for_uid:
        for_matches = [
            (d, seg, d.name)
            for d, seg in segs
            if seg is not None
            and not _seg_source_series_uids(seg)
            and _read_for_uid(seg) == ct_for_uid
        ]
        if len(for_matches) == 1:
            return for_matches[0]
        if len(for_matches) > 1:
            logger.error(
                "Auto RTSTRUCT: %d TotalSegmentator outputs share the planning CT "
                "FrameOfReferenceUID %s but none resolves by source series; refusing "
                "to guess among same-frame series (e.g. 4DCT phases).",
                len(for_matches), ct_for_uid,
            )
            return (None, None, None)

    # 3. Lone-candidate back-compat (legacy single-series / NIfTI-only). Use it unless
    #    its readable identity actively contradicts the planning CT.
    if len(dirs) == 1:
        only = dirs[0]
        seg = _total_dcm(only)
        if seg is not None:
            src = _seg_source_series_uids(seg)
            if ct_series_uid and src and ct_series_uid not in src:
                logger.error(
                    "Auto RTSTRUCT: the only TotalSegmentator output references series "
                    "%s, not the planning CT series %s; refusing to bind.",
                    sorted(src), ct_series_uid,
                )
                return (None, None, None)
            for_uid = _read_for_uid(seg)
            if ct_for_uid and for_uid and for_uid != ct_for_uid:
                logger.error(
                    "Auto RTSTRUCT: the only TotalSegmentator output FrameOfReferenceUID "
                    "%s != planning CT %s; refusing to bind.",
                    for_uid, ct_for_uid,
                )
                return (None, None, None)
        return (only, seg, only.name)

    # Several candidates, none identifiable -> fail-closed.
    return (None, None, None)


def _geometry_compatible(
    seg_img: sitk.Image, ct_img: sitk.Image, tol_mm: float = 2.0
) -> bool:
    """Return whether two 3-D images cover the same physical volume.

    TotalSegmentator may resample a source CT before producing masks, so compatible
    images need not have equal spacing or voxel counts. Course and segmentation
    provenance establish source-series identity before this safety check. Here we
    require matching physical axes under a signed permutation and matching physical
    extents, with one half-voxel allowance for resampling-rounding differences.
    """
    try:
        if seg_img.GetDimension() != 3 or ct_img.GetDimension() != 3:
            return False

        seg_size = np.asarray(seg_img.GetSize(), dtype=int)
        ct_size = np.asarray(ct_img.GetSize(), dtype=int)
        if seg_size.shape != (3,) or ct_size.shape != (3,):
            return False
        if np.any(seg_size <= 0) or np.any(ct_size <= 0):
            return False

        seg_spacing = np.asarray(seg_img.GetSpacing(), dtype=float)
        ct_spacing = np.asarray(ct_img.GetSpacing(), dtype=float)
        if seg_spacing.shape != (3,) or ct_spacing.shape != (3,):
            return False
        if not np.all(np.isfinite(seg_spacing)) or not np.all(np.isfinite(ct_spacing)):
            return False
        if np.any(seg_spacing <= 0) or np.any(ct_spacing <= 0):
            return False

        seg_direction = np.asarray(seg_img.GetDirection(), dtype=float).reshape(3, 3)
        ct_direction = np.asarray(ct_img.GetDirection(), dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(seg_direction)) or not np.all(np.isfinite(ct_direction)):
            return False

        # Direction columns are physical image axes. Spacing is deliberately not
        # included because masks may be produced on a resampled grid.
        orientation_match = False
        for permutation in permutations(range(3)):
            if all(
                any(
                    np.allclose(
                        seg_direction[:, seg_axis],
                        sign * ct_direction[:, ref_axis],
                        atol=1e-3,
                        rtol=0,
                    )
                    for sign in (1.0, -1.0)
                )
                for seg_axis, ref_axis in enumerate(permutation)
            ):
                orientation_match = True
                break
        if not orientation_match:
            return False

        def _extent(img: sitk.Image):
            # Physical bounding box over the 8 grid corners. Unlike origin plus
            # size*spacing, this remains valid for flipped, reordered, and oblique
            # direction cosines.
            sx, sy, sz = img.GetSize()
            corners = np.array([
                img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
                for x in (0, sx - 1)
                for y in (0, sy - 1)
                for z in (0, sz - 1)
            ], dtype=float)
            return corners.min(axis=0), corners.max(axis=0)

        seg_lo, seg_hi = _extent(seg_img)
        ct_lo, ct_hi = _extent(ct_img)
        extent_tolerance = tol_mm + 0.5 * max(
            float(seg_spacing.max()), float(ct_spacing.max())
        )
        return bool(
            np.allclose(seg_lo, ct_lo, atol=extent_tolerance, rtol=0)
            and np.allclose(seg_hi, ct_hi, atol=extent_tolerance, rtol=0)
        )
    except Exception as e:
        logger.debug("Geometry compatibility check failed: %s", e)
        return False


def build_auto_rtstruct(course_dir: Path) -> Optional[Path]:
    """Create an RTSTRUCT (RS_auto.dcm) from TotalSegmentator output if present.
    Returns path to RTSTRUCT or None.
    """
    contract = load_course_contract(course_dir)
    course_dirs = build_course_dirs(course_dir)
    out_path = course_dir / "RS_auto.dcm"
    ct_dir = contract.planning_ct_dir
    ct_series_uid = str(contract.planning_ct.get("series_instance_uid") or "").strip()
    rejected_artifact: Optional[dict[str, str]] = None

    def _failed(reason: str) -> Optional[Path]:
        logger.error("Auto RTSTRUCT rebuild failed for %s: %s", course_dir, reason)
        if out_path.exists():
            cleanup_reason = "the attempted replacement was not successfully completed"
            try:
                _remove_rejected_rtstruct(out_path, cleanup_reason)
            except OSError as exc:
                fatal_reason = f"{reason}; failed output could not be removed: {exc}"
                _record_auto_resume_decision(
                    course_dir,
                    "failed",
                    fatal_reason,
                    rejected_artifact=rejected_artifact,
                )
                raise RuntimeError(fatal_reason) from exc
        _record_auto_resume_decision(
            course_dir,
            "failed",
            reason,
            rejected_artifact=rejected_artifact,
        )
        return None

    def _reject_existing(reason: str) -> None:
        nonlocal rejected_artifact
        logger.warning("Existing Auto RTSTRUCT %s was rejected because %s", out_path, reason)
        try:
            _remove_rejected_rtstruct(out_path, reason)
        except OSError as exc:
            fatal_reason = f"could not remove rejected RS_auto.dcm: {exc}"
            _record_auto_resume_decision(course_dir, "failed", fatal_reason)
            raise RuntimeError(fatal_reason) from exc
        rejected_artifact = {
            "action": "removed",
            "path": out_path.name,
            "reason": reason,
        }
        # Persist the revocation before any rebuild work can raise. The terminal
        # rebuilt/failed decision below updates this same per-artifact record.
        _record_auto_resume_decision(
            course_dir,
            "rejected",
            f"removed before rebuild: {reason}",
            rejected_artifact=rejected_artifact,
        )

    if out_path.exists():
        current = False
        if ct_dir is None:
            rejection_reason = "the course contract has no planning CT DICOM directory"
        elif not ct_series_uid:
            rejection_reason = "the course contract has no planning CT series identity"
        elif not _is_valid_rtstruct(out_path):
            rejection_reason = "it is not a readable, non-empty DICOM RTSTRUCT"
        else:
            source_uids = _seg_source_series_uids(out_path)
            if source_uids != {ct_series_uid}:
                if not source_uids:
                    rejection_reason = "it does not record verifiable planning CT series provenance"
                elif len(source_uids) == 1:
                    rejection_reason = (
                        f"referenced planning CT series {next(iter(source_uids))}, not the "
                        f"current planning CT series {ct_series_uid}"
                    )
                else:
                    rejection_reason = (
                        f"referenced planning CT series set {sorted(source_uids)}, not only the "
                        f"current planning CT series {ct_series_uid}"
                    )
            elif not _derived_rtstruct_dependencies_are_current(
                out_path,
                ct_dir=ct_dir,
                nifti_path=contract.planning_ct_nifti,
                segmentation_root=course_dirs.segmentation_totalseg,
            ):
                rejection_reason = "a contracted CT or segmentation dependency is newer or unreadable"
            else:
                current = True
                rejection_reason = ""

        if current:
            logger.info(
                "Segmentation resume course=%s artefact=RS_auto.dcm action=reused "
                "reason=valid RTSTRUCT references planning CT series %s",
                course_dir,
                ct_series_uid,
            )
            _record_auto_resume_decision(
                course_dir,
                "reused",
                f"valid RTSTRUCT references planning CT series {ct_series_uid}",
            )
            return out_path
        _reject_existing(rejection_reason)

    if ct_dir is None:
        return _failed("course contract has no planning CT DICOM directory")
    if not ct_series_uid:
        return _failed("course contract has no planning CT series identity")

    try:
        from rt_utils import RTStructBuilder
    except Exception as e:
        logger.error("rt-utils not available: %s", e)
        return _failed(f"rt-utils is unavailable: {e}")

    ct_img = _load_ct_image(ct_dir)
    if ct_img is None:
        return _failed("planning CT could not be loaded")

    # Prefer DICOM-SEG, detect if RTSTRUCT already produced, fallback to NIfTI
    seg_img: Optional[sitk.Image] = None
    label_map: Dict[int, str] = {}
    seg_root = course_dirs.segmentation_totalseg
    dicom_seg_path: Optional[Path] = None
    base_name: Optional[str] = None
    selected_dir: Optional[Path] = None

    if seg_root.exists():
        candidate_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())
        # Bind masks to the planning CT by FrameOfReferenceUID, never candidate_dirs[0]:
        # in all-series mode several series (CBCT, 4DCT phases, diagnostic CT) are
        # segmented and an alpha-first pick can map the wrong series onto this CT.
        ct_for_uid = _read_ct_for_uid(ct_dir)
        selected_dir, dicom_seg_path, base_name = _select_seg_dir_for_ct(
            candidate_dirs, ct_series_uid, ct_for_uid
        )
        if selected_dir is None and candidate_dirs:
            logger.error(
                "Auto RTSTRUCT: no TotalSegmentator output matches the planning CT "
                "(series %s, FrameOfReference %s) among %d candidate series in %s; "
                "refusing to bind masks from a different series. Skipping RS_auto for %s.",
                ct_series_uid or "unknown",
                ct_for_uid or "unknown",
                len(candidate_dirs),
                seg_root,
                course_dir,
            )
            return _failed(
                "no TotalSegmentator output matches the contracted planning CT series"
            )

    if dicom_seg_path and dicom_seg_path.exists():
        try:
            ds = pydicom.dcmread(str(dicom_seg_path), stop_before_pixels=True)
            sop = str(getattr(ds, 'SOPClassUID', ''))
            if sop == '1.2.840.10008.5.1.4.1.1.66.4':
                seg_img, label_map = _load_seg_dicom(dicom_seg_path)
            elif sop == '1.2.840.10008.5.1.4.1.1.481.3':
                try:
                    # Copy bytes without preserving the producer timestamp. The new
                    # publication must be newer than every input dependency or its
                    # own freshness gate rejects it as stale.
                    _write_rtstruct_atomic(
                        out_path,
                        lambda tmp: shutil.copyfile(str(dicom_seg_path), tmp),
                    )
                except Exception as e:
                    logger.error('Failed to copy RTSTRUCT to RS_auto: %s', e)
                    return _failed(f"copying the matched RTSTRUCT failed: {e}")
                # Keep behavior consistent with NIfTI-derived RTSTRUCTs.
                try:
                    sanitize_rtstruct(out_path)
                    summary = fix_rtstruct_rois(ct_dir, out_path)
                    if summary and summary.changed:
                        logger.info(
                            "Auto RTSTRUCT ROI fix: %d repaired, %d still problematic",
                            len(summary.fixed),
                            len(summary.failed),
                        )
                except Exception as e:
                    logger.debug("Post-processing copied RTSTRUCT failed: %s", e)
                logger.info("Wrote auto RTSTRUCT (from RTSTRUCT): %s", out_path)
                _record_auto_resume_decision(
                    course_dir,
                    "rebuilt",
                    "rebuilt from a TotalSegmentator RTSTRUCT referencing the planning CT series",
                    rejected_artifact=rejected_artifact,
                )
                return out_path
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug('Inspecting DICOM output failed: %s', e)

    if seg_img is None:
        seg_img, label_map = _load_seg_nifti(selected_dir or seg_root, base_name)

    if seg_img is not None and not _geometry_compatible(seg_img, ct_img):
        logger.error(
            "Auto RTSTRUCT: selected segmentation (%s) does not share the planning CT's "
            "physical space; refusing to resample cross-frame masks. Skipping RS_auto for %s.",
            selected_dir or seg_root,
            course_dir,
        )
        return _failed(
            "selected TotalSegmentator output does not share the planning CT physical space"
        )

    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    except Exception as e:
        logger.error("Failed to create RTSTRUCT: %s", e)
        return _failed(f"creating the RTSTRUCT builder failed: {e}")

    added_any = False
    added_names: set[str] = set()

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
                name = _pretty_roi_name(name)
                mask = seg_arr == idx
                if not np.any(mask):
                    continue
                try:
                    roi_name = _unique_roi_name(_pretty_roi_name(name), added_names)
                    rtstruct.add_roi(mask=mask, name=roi_name)
                    added_names.add(roi_name)
                    added_any = True
                except Exception as e:
                    logger.debug("Failed to add ROI %s: %s", name, e)

    fallback_dir = selected_dir or seg_root
    if not added_any and fallback_dir.exists():
        # Fall back to per-ROI binary masks produced by TotalSegmentator.  The
        # two historical naming conventions are normalized in one pass.  Prefer
        # current ``total--ROI`` files over legacy ``<base>--total--ROI`` files
        # when both represent the same ROI, so a stale duplicate cannot win by
        # directory sort order.
        binary_by_roi: dict[str, tuple[int, str, sitk.Image]] = {}
        for raw_name, mask_img in _iter_binary_masks(fallback_dir):
            if base_name and raw_name.startswith(f"{base_name}--total--"):
                rank = 1
                roi_key = raw_name[len(base_name) + len("--total--"):]
            elif raw_name.startswith("total--"):
                rank = 0
                roi_key = raw_name[len("total--"):]
            else:
                continue
            if not roi_key:
                continue
            previous = binary_by_roi.get(roi_key)
            if previous is None or rank < previous[0]:
                binary_by_roi[roi_key] = (rank, raw_name, mask_img)
        binary_masks = []
        for _rank, raw_name, mask_img in sorted(
            binary_by_roi.values(), key=lambda item: item[1]
        ):
            if raw_name.startswith("total--"):
                roi_name = raw_name[len("total--"):]
            elif base_name and raw_name.startswith(f"{base_name}--total--"):
                roi_name = raw_name[len(base_name) + len("--total--"):]
            else:
                continue
            binary_masks.append((roi_name, mask_img))
        # Universal geometry net: EVERY per-ROI mask must share the planning CT's
        # physical space before any is resampled. Checking only one would let a
        # mixed/stale dir (a compatible mask plus an incompatible later one) bind
        # a cross-frame ROI, so a single incompatible mask fail-closes the whole
        # fallback (matching the seg_img path's whole-build fail-closed).
        incompatible = [n for n, m in binary_masks if not _geometry_compatible(m, ct_img)]
        if incompatible:
            logger.error(
                "Auto RTSTRUCT: %d of %d per-ROI binary masks in %s do not share the "
                "planning CT's physical space (e.g. %s); refusing the fallback to avoid "
                "binding cross-frame ROIs. Skipping RS_auto for %s.",
                len(incompatible), len(binary_masks), fallback_dir, incompatible[0], course_dir,
            )
            binary_masks = []
        for name, mask_img in binary_masks:
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

            roi_name = _unique_roi_name(_pretty_roi_name(name), added_names)
            try:
                rtstruct.add_roi(mask=mask_bin, name=roi_name)
                added_names.add(roi_name)
                added_any = True
            except Exception as e:
                logger.debug("Failed to add ROI %s: %s", roi_name, e)

    if not added_any:
        return _failed("no readable non-empty segmentation ROI could be added")

    try:
        _write_rtstruct_atomic(out_path, rtstruct.save)
        sanitize_rtstruct(out_path)
        summary = fix_rtstruct_rois(ct_dir, out_path)
        if summary and summary.changed:
            logger.info(
                "Auto RTSTRUCT ROI fix: %d repaired, %d still problematic",
                len(summary.fixed),
                len(summary.failed),
            )
        logger.info("Wrote auto RTSTRUCT: %s", out_path)
        _record_auto_resume_decision(
            course_dir,
            "rebuilt",
            "rebuilt from TotalSegmentator masks matched to the planning CT",
            rejected_artifact=rejected_artifact,
        )
        return out_path
    except Exception as e:
        logger.error("Saving auto RTSTRUCT failed: %s", e)
        return _failed(f"publishing the rebuilt RTSTRUCT failed: {e}")
