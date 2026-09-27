"""Conservative instance selection, used only after existing CT conversions fail.

Source DICOM files are never changed. Mixed geometry requires image-level
references from the authoritative RTSTRUCT; series-level references alone do
not authorize excluding images. Duplicate equality uses canonical float64
rescaled pixels, with no numeric tolerance or interpolation.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from .nonuniform_ct_conversion import Converter, convert_nonuniform_unsigned_ct

METHOD = "dcm2niix_ct_series_hygiene"
POSITION_TOLERANCE_MM = 0.01
ORIENTATION_TOLERANCE = 1e-4
CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


class CTSeriesHygieneError(RuntimeError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def rtstruct_image_references(path: Path | None) -> set[str]:
    """Read both series-level and per-contour ContourImageSequence items."""
    if path is None:
        return set()
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if str(ds.Modality) != "RTSTRUCT":
            raise ValueError("not RTSTRUCT")
        refs: set[str] = set()
        for element in ds.iterall():
            if element.keyword != "ContourImageSequence":
                continue
            for item in element.value:
                uid = str(getattr(item, "ReferencedSOPInstanceUID", ""))
                sop_class = str(getattr(item, "ReferencedSOPClassUID", ""))
                if not uid or sop_class not in ("", CT_STORAGE):
                    raise ValueError("unresolved CT image reference")
                refs.add(uid)
        return refs
    except Exception as exc:
        raise CTSeriesHygieneError("ct_hygiene_rtstruct_references_unresolvable") from exc


def _vector(ds, name: str, length: int) -> np.ndarray:
    values = np.asarray(getattr(ds, name), dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError("invalid geometry")
    return values


def _pixels(path: Path) -> bytes:
    ds = pydicom.dcmread(path)
    # A modality LUT cannot be represented by the linear rescale checked here.
    if hasattr(ds, "ModalityLUTSequence"):
        raise CTSeriesHygieneError("ct_hygiene_duplicate_rescale_unsupported")
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    if not np.isfinite([slope, intercept]).all() or slope == 0:
        raise CTSeriesHygieneError("ct_hygiene_duplicate_rescale_unsupported")
    values = np.asarray(ds.pixel_array, dtype="<f8") * slope + intercept
    if values.shape != (int(ds.Rows), int(ds.Columns)) or not np.isfinite(values).all():
        raise CTSeriesHygieneError("ct_hygiene_duplicate_pixels_invalid")
    return values.astype("<f8").tobytes()


def select_ct_instances(
    ct_dir: Path, authoritative_rtstruct: Path | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Select one consistent volume or raise a stable, identifier-free reason."""
    try:
        return _select(ct_dir, authoritative_rtstruct)
    except CTSeriesHygieneError:
        raise
    except Exception as exc:
        raise CTSeriesHygieneError("ct_hygiene_invalid_dicom_or_geometry") from exc


def _select(ct_dir, authoritative_rtstruct):
    entries = sorted(ct_dir.iterdir())
    if not entries or any(not p.is_file() for p in entries):
        raise CTSeriesHygieneError("ct_hygiene_invalid_series_directory")
    partitions: dict[tuple, list[dict]] = {}
    series, frames, identities = set(), set(), set()
    for path in entries:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if (str(getattr(ds, "SOPClassUID", "")) != CT_STORAGE
                or str(getattr(ds, "Modality", "")) != "CT"
                or int(getattr(ds, "NumberOfFrames", 1)) != 1):
            raise CTSeriesHygieneError("ct_hygiene_unsupported_ct_object")
        uid = str(ds.SOPInstanceUID)
        if not uid or uid in identities:
            raise CTSeriesHygieneError("ct_hygiene_ambiguous_instance_identity")
        identities.add(uid)
        series.add(str(ds.SeriesInstanceUID))
        frames.add(str(ds.FrameOfReferenceUID))
        orientation = _vector(ds, "ImageOrientationPatient", 6)
        position = _vector(ds, "ImagePositionPatient", 3)
        spacing = _vector(ds, "PixelSpacing", 2)
        rows, columns = int(ds.Rows), int(ds.Columns)
        if min(rows, columns) <= 0 or min(spacing) <= 0:
            raise ValueError("invalid dimensions")
        # Exact numeric partitioning is deliberately conservative. Tolerance is
        # used for volume consistency, never to merge different geometries.
        key = (*orientation, rows, columns, *spacing)
        partitions.setdefault(key, []).append({
            "path": path, "uid": uid, "position": position,
            "instance": int(getattr(ds, "InstanceNumber", 0)),
        })
    if len(series) != 1 or "" in series or len(frames) != 1 or "" in frames:
        raise CTSeriesHygieneError("ct_hygiene_series_or_frame_mismatch")
    refs = rtstruct_image_references(authoritative_rtstruct)
    if refs - identities:
        raise CTSeriesHygieneError("ct_hygiene_rtstruct_references_unresolvable")
    mixed = len(partitions) > 1
    if mixed:
        if not refs:
            raise CTSeriesHygieneError("ct_hygiene_mixed_geometry_without_image_references")
        candidates = [(key, group) for key, group in partitions.items()
                      if refs <= {item["uid"] for item in group}]
        if len(candidates) != 1:
            raise CTSeriesHygieneError("ct_hygiene_references_span_geometry_partitions")
        key, group = candidates[0]
    else:
        key, group = next(iter(partitions.items()))
    excluded = [{"sop_instance_uid": item["uid"], "reason_code": "unreferenced_geometry_partition"}
                for other_key, other_group in partitions.items() if other_key != key
                for item in other_group]
    orientation = np.array(key[:6])
    row, col = orientation[:3], orientation[3:]
    normal = np.cross(row, col)
    if (abs(np.linalg.norm(row) - 1) > ORIENTATION_TOLERANCE
            or abs(np.linalg.norm(col) - 1) > ORIENTATION_TOLERANCE
            or abs(np.dot(row, col)) > ORIENTATION_TOLERANCE
            or (mixed and abs(abs(normal[2]) - 1) > ORIENTATION_TOLERANCE)):
        raise CTSeriesHygieneError("ct_hygiene_not_consistent_axial_volume")
    # Prefer a referenced equivalent instance, then lowest InstanceNumber/UID.
    ordered = sorted(group, key=lambda item: (item["uid"] not in refs, item["instance"], item["uid"]))
    kept: list[dict] = []
    for item in ordered:
        matches = [other for other in kept if np.linalg.norm(item["position"] - other["position"]) <= POSITION_TOLERANCE_MM]
        if len(matches) > 1:
            raise CTSeriesHygieneError("ct_hygiene_ambiguous_duplicate_position")
        if matches:
            other = matches[0]
            if _pixels(item["path"]) != _pixels(other["path"]):
                raise CTSeriesHygieneError("ct_hygiene_duplicate_position_pixels_differ")
            if mixed and item["uid"] in refs:
                raise CTSeriesHygieneError("ct_hygiene_referenced_duplicate_would_be_excluded")
            excluded.append({"sop_instance_uid": item["uid"],
                             "reason_code": "identical_rescaled_duplicate_position",
                             "retained_sop_instance_uid": other["uid"]})
        else:
            kept.append(item)
    positions = np.array([item["position"] for item in kept])
    projections = positions @ normal
    steps = np.diff(np.sort(projections))
    residual = positions - positions[0] - np.outer(projections - projections[0], normal)
    if (len(kept) < 3 or np.min(steps) <= POSITION_TOLERANCE_MM
            or np.max(np.linalg.norm(residual, axis=1)) > POSITION_TOLERANCE_MM):
        raise CTSeriesHygieneError("ct_hygiene_inconsistent_volume_positions")
    if not excluded:
        raise CTSeriesHygieneError("ct_hygiene_no_safe_exclusions")
    kept_uids = {item["uid"] for item in kept}
    evidence = {
        "method": METHOD, "status": "selected",
        "reason": "Failed CT conversion retried after verified instance exclusions",
        "source_instance_count": len(entries), "kept_instance_count": len(kept),
        "excluded_instance_count": len(excluded), "geometry_partition_count": len(partitions),
        "position_tolerance_mm": POSITION_TOLERANCE_MM,
        "kept_sop_instance_uids": sorted(kept_uids),
        "excluded_instances": sorted(excluded, key=lambda item: item["sop_instance_uid"]),
        "rtstruct_referenced_instance_count": len(refs),
        "rtstruct_references_all_kept": refs <= kept_uids,
        "authoritative_rtstruct_sha256": (hashlib.sha256(authoritative_rtstruct.read_bytes()).hexdigest()
                                           if authoritative_rtstruct else None),
        "selected_geometry": {"image_orientation_patient": list(key[:6]),
                              "rows": key[6], "columns": key[7], "pixel_spacing": list(key[8:])},
        "slice_step_min_mm": float(min(steps)), "slice_step_max_mm": float(max(steps)),
        "verification": "Duplicate rescaled float64 pixel bytes are identical; excluded geometry partitions have no RTSTRUCT image references",
    }
    return sorted(item["path"] for item in kept), evidence


def convert_ct_series_hygiene(
    ct_dir: Path, work_dir: Path, converter: Converter,
    authoritative_rtstruct: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    files, evidence = select_ct_instances(ct_dir, authoritative_rtstruct)
    staging = work_dir / "hygiene_input"
    output = work_dir / "hygiene_output"
    staging.mkdir()
    output.mkdir()
    for path in files:
        shutil.copyfile(path, staging / path.name)
    generated = converter(staging, output)
    if generated is None:
        retry = convert_nonuniform_unsigned_ct(staging, output, converter)
        if retry is None:
            raise CTSeriesHygieneError("ct_hygiene_selected_volume_conversion_failed")
        generated, evidence["downstream_conversion"] = retry
    # A consistent geometry can still hide different acquisitions. Never use
    # the general converter's largest-file choice to discard a second volume.
    volumes = [path for path in generated.parent.iterdir()
               if path.name.endswith((".nii", ".nii.gz"))]
    unequalized = [path for path in volumes if "_Eq_" not in path.name]
    equalized = [path for path in volumes if "_Eq_" in path.name]
    if len(unequalized) != 1 or len(equalized) > 1 or (equalized and generated != equalized[0]):
        raise CTSeriesHygieneError("ct_hygiene_ambiguous_converter_outputs")
    import nibabel as nib

    try:
        acquired_shape = nib.load(str(unequalized[0])).shape
    except Exception as exc:
        raise CTSeriesHygieneError("ct_hygiene_unreadable_converter_output") from exc
    geometry = evidence["selected_geometry"]
    if (
        len(acquired_shape) != 3
        or acquired_shape[2] != len(files)
        or sorted(acquired_shape[:2]) != sorted([geometry["rows"], geometry["columns"]])
    ):
        raise CTSeriesHygieneError("ct_hygiene_converter_did_not_preserve_selected_extent")
    evidence["status"] = "converted"
    return generated, evidence
