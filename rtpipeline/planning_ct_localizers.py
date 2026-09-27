"""Conservative publication of planning CT without ImageType[2] localizers.

This module acts on organize's course copies only. It never edits source DICOM.
A uniform volume and exact rescaled voxel equality (after axis permutation/flip,
without interpolation) bind the retained instances to the planning NIfTI.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from .ct_series_hygiene import CT_STORAGE, rtstruct_image_references

METHOD = "planning_ct_image_type_localizer_exclusion"
POSITION_TOLERANCE_MM = 0.01


class PlanningCTLocalizerError(RuntimeError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def is_localizer(ds) -> bool:
    values = getattr(ds, "ImageType", []) or []
    if isinstance(values, str):
        values = values.split("\\")
    return len(values) >= 3 and str(values[2]).strip().upper() == "LOCALIZER"


def _headers(ct_dir: Path):
    return [(p, pydicom.dcmread(p, stop_before_pixels=True))
            for p in sorted(ct_dir.iterdir()) if p.is_file()]


def _volume(entries):
    """Require one regular acquisition; never select another geometry subset."""
    if len(entries) < 3:
        raise PlanningCTLocalizerError("ct_localizer_insufficient_volume")
    keys, positions = set(), []
    for _, ds in entries:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        spacing = np.asarray(ds.PixelSpacing, dtype=float)
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        if (orientation.shape != (6,) or spacing.shape != (2,) or position.shape != (3,)
                or not all(np.isfinite(v).all() for v in (orientation, spacing, position))
                or min(spacing) <= 0 or min(int(ds.Rows), int(ds.Columns)) <= 0):
            raise ValueError("invalid geometry")
        keys.add((int(ds.Rows), int(ds.Columns), *spacing, *orientation,
                  str(getattr(ds, "AcquisitionNumber", "")),
                  str(getattr(ds, "TemporalPositionIdentifier", "")),
                  str(getattr(ds, "EchoNumbers", ""))))
        positions.append(position)
    if len(keys) != 1:
        raise PlanningCTLocalizerError("ct_localizer_inconsistent_geometry")
    row, col = orientation[:3], orientation[3:]
    normal = np.cross(row, col)
    if (abs(np.linalg.norm(row) - 1) > 1e-4 or abs(np.linalg.norm(col) - 1) > 1e-4
            or abs(np.dot(row, col)) > 1e-4):
        raise PlanningCTLocalizerError("ct_localizer_inconsistent_geometry")
    positions = np.asarray(positions)
    projections = positions @ normal
    steps = np.diff(np.sort(projections))
    residual = positions - positions[0] - np.outer(projections - projections[0], normal)
    if (min(steps) <= POSITION_TOLERANCE_MM
            or max(steps) - min(steps) > POSITION_TOLERANCE_MM
            or np.max(np.linalg.norm(residual, axis=1)) > POSITION_TOLERANCE_MM):
        raise PlanningCTLocalizerError("ct_localizer_inconsistent_volume_positions")
    return {
        "rows": int(entries[0][1].Rows), "columns": int(entries[0][1].Columns),
        "pixel_spacing": spacing.tolist(), "image_orientation_patient": orientation.tolist(),
        "slice_step_mm": float(np.mean(steps)),
    }


def select_localizers(ct_dir: Path, rtstruct: Path | None):
    """Return excluded paths and evidence, or ([], None) for unchanged series."""
    try:
        # Read failures do not change the pre-existing no-localizer path. Once a
        # localizer is found, every file must be readable and validated below.
        entries, unreadable = [], False
        for path in sorted(ct_dir.iterdir()):
            try:
                entries.append((path, pydicom.dcmread(path, stop_before_pixels=True)))
            except Exception:
                unreadable = True
        excluded = [(p, ds) for p, ds in entries if is_localizer(ds)]
        if not excluded:
            return [], None
        if unreadable or any(p.is_symlink() or not p.is_file() for p, _ in entries):
            raise PlanningCTLocalizerError("ct_localizer_invalid_series_directory")
        if rtstruct is None:
            raise PlanningCTLocalizerError("ct_localizer_rtstruct_references_missing")
        refs = rtstruct_image_references(rtstruct)
        if not refs:
            raise PlanningCTLocalizerError("ct_localizer_rtstruct_references_missing")
        identities = [str(ds.SOPInstanceUID) for _, ds in entries]
        if len(set(identities)) != len(identities) or "" in identities:
            raise PlanningCTLocalizerError("ct_localizer_ambiguous_instance_identity")
        for _, ds in entries:
            if (str(ds.SOPClassUID) != CT_STORAGE or str(ds.Modality) != "CT"
                    or int(getattr(ds, "NumberOfFrames", 1)) != 1):
                raise PlanningCTLocalizerError("ct_localizer_unsupported_ct_object")
        for field in ("SeriesInstanceUID", "FrameOfReferenceUID", "StudyInstanceUID"):
            values = {str(getattr(ds, field, "")) for _, ds in entries}
            if len(values) != 1 or "" in values:
                raise PlanningCTLocalizerError("ct_localizer_series_or_frame_mismatch")
        excluded_uids = {str(ds.SOPInstanceUID) for _, ds in excluded}
        if refs & excluded_uids:
            raise PlanningCTLocalizerError("ct_localizer_referenced_by_rtstruct")
        if refs - set(identities):
            raise PlanningCTLocalizerError("ct_localizer_rtstruct_references_unresolvable")
        kept = [(p, ds) for p, ds in entries if not is_localizer(ds)]
        geometry = _volume(kept)
        evidence = {
            "method": METHOD, "reason": "localizer_image_type", "status": "excluded",
            "source_instance_count": len(entries), "kept_instance_count": len(kept),
            "excluded_instance_count": len(excluded),
            "kept_sop_instance_uids": sorted(str(ds.SOPInstanceUID) for _, ds in kept),
            "excluded_instances": [{"sop_instance_uid": uid, "reason_code": "localizer_image_type"}
                                   for uid in sorted(excluded_uids)],
            "rtstruct_referenced_instance_count": len(refs),
            "rtstruct_references_all_kept": True,
            "authoritative_rtstruct_sha256": hashlib.sha256(rtstruct.read_bytes()).hexdigest(),
            "selected_geometry": geometry,
        }
        return [p for p, _ in excluded], evidence
    except PlanningCTLocalizerError:
        raise
    except Exception as exc:
        raise PlanningCTLocalizerError("ct_localizer_invalid_dicom_or_references") from exc


def verify_volume(ct_dir: Path, nifti: Path) -> None:
    """Prove that the existing NIfTI contains exactly the retained CT volume."""
    import SimpleITK as sitk
    try:
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(str(ct_dir))
        if len(names) != len(list(ct_dir.iterdir())):
            raise ValueError("reader omitted instances")
        reader.SetFileNames(names)
        ct = reader.Execute()
        image = sitk.ReadImage(str(nifti))
        orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(ct.GetDirection())
        image = sitk.DICOMOrient(image, orientation)
        if (ct.GetSize() != image.GetSize()
                or not np.allclose(ct.GetSpacing(), image.GetSpacing(), atol=1e-4, rtol=0)
                or not np.allclose(ct.GetOrigin(), image.GetOrigin(), atol=1e-3, rtol=0)
                or not np.allclose(ct.GetDirection(), image.GetDirection(), atol=1e-5, rtol=0)
                or not np.array_equal(sitk.GetArrayViewFromImage(ct), sitk.GetArrayViewFromImage(image))):
            raise ValueError("volume differs")
    except Exception as exc:
        raise PlanningCTLocalizerError("ct_localizer_nifti_volume_mismatch") from exc


def validate_selection(ct_dir: Path, rtstruct: Path | None, evidence: dict[str, Any]) -> None:
    """Validate recorded exclusions against the published retained instances."""
    try:
        entries = _headers(ct_dir)
        kept = sorted(str(ds.SOPInstanceUID) for _, ds in entries)
        excluded = evidence["excluded_instances"]
        dropped = [item["sop_instance_uid"] for item in excluded]
        refs = rtstruct_image_references(rtstruct)
        if (evidence.get("method") != METHOD or evidence.get("reason") != "localizer_image_type"
                or evidence.get("status") != "excluded" or not dropped
                or len(set(dropped)) != len(dropped) or set(dropped) & set(kept)
                or any(item["reason_code"] != "localizer_image_type" for item in excluded)
                or evidence.get("kept_sop_instance_uids") != kept
                or evidence.get("kept_instance_count") != len(kept)
                or evidence.get("excluded_instance_count") != len(dropped)
                or evidence.get("source_instance_count") != len(kept) + len(dropped)
                or any(is_localizer(ds) for _, ds in entries)
                or not refs or not refs <= set(kept)
                or evidence.get("rtstruct_references_all_kept") is not True
                or evidence.get("rtstruct_referenced_instance_count") != len(refs)
                or rtstruct is None
                or evidence.get("authoritative_rtstruct_sha256") != hashlib.sha256(rtstruct.read_bytes()).hexdigest()
                or evidence.get("selected_geometry") != _volume(entries)):
            raise ValueError("inconsistent selection")
    except Exception as exc:
        raise PlanningCTLocalizerError("ct_localizer_stale_selection_evidence") from exc
