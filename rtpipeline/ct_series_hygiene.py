"""Conservative instance selection, used only after existing CT conversions fail.

Source DICOM files are never changed. Mixed geometry requires image-level
references from the authoritative RTSTRUCT; series-level references alone do
not authorize excluding images. Duplicate equality uses canonical float64
rescaled pixels, with no numeric tolerance or interpolation.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pydicom

from .nonuniform_ct_conversion import Converter, convert_nonuniform_unsigned_ct

METHOD = "dcm2niix_ct_series_hygiene"
POSITION_TOLERANCE_MM = 0.01
ORIENTATION_TOLERANCE = 1e-4
CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
STACK_DISTINGUISHING_ATTRIBUTES = (
    "ScanOptions",
    "ConvolutionKernel",
    "ReconstructionDiameter",
    "WindowCenter",
    "Exposure",
)
STACK_SIGNATURE_KEYS = (
    *STACK_DISTINGUISHING_ATTRIBUTES,
    "instance_count",
    "sop_uids_sha256",
)
_EXPLICIT_VR_LONG_LENGTH_VRS = frozenset({
    b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV",
})


class CTSeriesHygieneError(RuntimeError):
    def __init__(self, reason_code: str, *, evidence: dict[str, Any] | None = None):
        self.reason_code = reason_code
        self.evidence = dict(evidence) if evidence else {}
        super().__init__(reason_code)


def _extract_contour_image_refs(sequence: Any) -> set[str]:
    refs: set[str] = set()
    for item in sequence or []:
        uid = str(getattr(item, "ReferencedSOPInstanceUID", ""))
        sop_class = str(getattr(item, "ReferencedSOPClassUID", ""))
        if not uid or sop_class not in ("", CT_STORAGE):
            raise ValueError("unresolved CT image reference")
        refs.add(uid)
    return refs


def rtstruct_reference_levels(path: Path | None) -> tuple[set[str], set[str]]:
    """Return ``(frame_refs, contour_refs)`` from an authoritative RTSTRUCT."""
    if path is None:
        return set(), set()
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if str(ds.Modality) != "RTSTRUCT":
            raise ValueError("not RTSTRUCT")
        frame_refs: set[str] = set()
        contour_refs: set[str] = set()
        for ref_for in getattr(ds, "ReferencedFrameOfReferenceSequence", None) or []:
            for study in getattr(ref_for, "RTReferencedStudySequence", None) or []:
                for series in getattr(study, "RTReferencedSeriesSequence", None) or []:
                    frame_refs |= _extract_contour_image_refs(
                        getattr(series, "ContourImageSequence", None)
                    )
        for roi_contour in getattr(ds, "ROIContourSequence", None) or []:
            for contour in getattr(roi_contour, "ContourSequence", None) or []:
                contour_refs |= _extract_contour_image_refs(
                    getattr(contour, "ContourImageSequence", None)
                )
        all_refs: set[str] = set()
        for element in ds.iterall():
            if element.keyword != "ContourImageSequence":
                continue
            all_refs |= _extract_contour_image_refs(element.value)
        extra = all_refs - (frame_refs | contour_refs)
        if extra:
            frame_refs |= extra
        return frame_refs, contour_refs
    except Exception as exc:
        raise CTSeriesHygieneError("ct_hygiene_rtstruct_references_unresolvable") from exc


def rtstruct_image_references(path: Path | None) -> set[str]:
    """Read both series-level and per-contour ContourImageSequence items."""
    frame_refs, contour_refs = rtstruct_reference_levels(path)
    return frame_refs | contour_refs


def _vector(ds, name: str, length: int) -> np.ndarray:
    values = np.asarray(getattr(ds, name), dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError("invalid geometry")
    return values


def _image_type_tokens(ds: Any) -> tuple[str, ...]:
    raw = getattr(ds, "ImageType", None)
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        parts = raw.split("\\")
    else:
        try:
            parts = list(raw)
        except TypeError:
            parts = [raw]
    return tuple(str(part).strip().upper() for part in parts if str(part).strip())


def _is_localizer_tokens(tokens: tuple[str, ...]) -> bool:
    return len(tokens) >= 3 and tokens[2] == "LOCALIZER"


def _stack_attr_value(ds: Any, keyword: str) -> Any:
    raw = getattr(ds, keyword, None)
    if raw in (None, ""):
        return None
    if keyword in ("ScanOptions", "ConvolutionKernel"):
        if isinstance(raw, (list, tuple)) or (
            hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes))
        ):
            tokens = [str(v).strip() for v in raw if str(v).strip()]
            return "\\".join(tokens) if tokens else None
        text = str(raw).strip()
        return text if text else None
    if keyword in ("ReconstructionDiameter", "WindowCenter", "Exposure"):
        def exact_value(value: Any) -> str:
            # pydicom's DSfloat retains the on-disk decimal in original_string.
            # Converting it through float can merge distinct 16-character DS values.
            text = str(getattr(value, "original_string", value)).strip()
            if _canonical_decimal(text) is None:
                raise ValueError(f"non-finite {keyword}")
            return text

        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
            values = [exact_value(value) for value in raw]
            if not values:
                raise ValueError(f"empty {keyword}")
            return values[0] if len(values) == 1 else values
        return exact_value(raw)
    return str(raw).strip() or None


def _canonical_decimal(value: Any) -> Decimal | None:
    """Parse a finite numeric attribute into an exact Decimal; NaN/Inf/bool return None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        try:
            dec = Decimal(str(value).strip())
        except InvalidOperation:
            return None
        return dec if dec.is_finite() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dec = Decimal(text)
        except InvalidOperation:
            return None
        return dec if dec.is_finite() else None
    return None


def _stack_attr_key(attrs: dict[str, Any]) -> tuple[Any, ...]:
    key_parts: list[Any] = []
    for name in STACK_DISTINGUISHING_ATTRIBUTES:
        val = attrs.get(name)
        if isinstance(val, list):
            decs = [_canonical_decimal(x) for x in val]
            if any(d is None for d in decs):
                raise ValueError(f"non-finite {name}")
            key_parts.append(tuple(decs))
        elif name in ("ReconstructionDiameter", "WindowCenter", "Exposure") and val is not None:
            dec = _canonical_decimal(val)
            if dec is None:
                raise ValueError(f"non-finite {name}")
            key_parts.append(dec)
        else:
            key_parts.append(val)
    return tuple(key_parts)


def compute_stack_sop_uids_sha256(uids: Iterable[str]) -> str:
    sorted_uids = sorted(str(u).strip() for u in uids if str(u).strip())
    return hashlib.sha256("\n".join(sorted_uids).encode("utf-8")).hexdigest()


def compute_stack_signature(
    attributes: dict[str, Any], instance_count: int, sop_uids_sha256: str
) -> dict[str, Any]:
    return {
        **{name: attributes.get(name) for name in STACK_DISTINGUISHING_ATTRIBUTES},
        "instance_count": int(instance_count),
        "sop_uids_sha256": str(sop_uids_sha256),
    }


def _signatures_match(candidate_sig: dict[str, Any], requested_sig: dict[str, Any]) -> bool:
    for key in STACK_SIGNATURE_KEYS:
        if key not in requested_sig:
            return False
        left = candidate_sig.get(key)
        right = requested_sig.get(key)
        if key in ("ReconstructionDiameter", "Exposure"):
            if left is None or right is None:
                if left is not right:
                    return False
            else:
                d_left = _canonical_decimal(left)
                d_right = _canonical_decimal(right)
                if d_left is None or d_right is None or d_left != d_right:
                    return False
        elif key == "WindowCenter":
            if left is None or right is None:
                if left is not right:
                    return False
            elif isinstance(left, list) and isinstance(right, list):
                if not left or len(left) != len(right):
                    return False
                for a, b in zip(left, right):
                    d_a = _canonical_decimal(a)
                    d_b = _canonical_decimal(b)
                    if d_a is None or d_b is None or d_a != d_b:
                        return False
            elif not isinstance(left, list) and not isinstance(right, list):
                d_left = _canonical_decimal(left)
                d_right = _canonical_decimal(right)
                if d_left is None or d_right is None or d_left != d_right:
                    return False
            else:
                return False
        elif key in ("ScanOptions", "ConvolutionKernel"):
            if left is None or right is None:
                if left is not right:
                    return False
            elif not isinstance(left, str) or not isinstance(right, str) or left != right:
                return False
        elif key == "instance_count":
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, int)
                or not isinstance(right, int)
                or left != right
            ):
                return False
        else:
            if not isinstance(left, str) or not isinstance(right, str) or left != right:
                return False
    return True


def _validate_stack_override_governance(
    stack_override: Any, candidate_stacks: list[dict[str, Any]] | None = None
) -> tuple[str, str, dict[str, Any]]:
    err_evidence = {"candidate_stacks": candidate_stacks} if candidate_stacks is not None else None
    if not isinstance(stack_override, dict):
        raise CTSeriesHygieneError(
            "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
        )
    adjudication_record_id = stack_override.get("adjudication_record_id")
    reason = stack_override.get("reason")
    if (
        not isinstance(adjudication_record_id, str)
        or not adjudication_record_id.strip()
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise CTSeriesHygieneError(
            "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
        )
    sig_raw = (
        stack_override.get("stack_signature")
        if isinstance(stack_override.get("stack_signature"), dict)
        else stack_override
    )
    if not isinstance(sig_raw, dict) or any(key not in sig_raw for key in STACK_SIGNATURE_KEYS):
        raise CTSeriesHygieneError(
            "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
        )
    instance_count = sig_raw.get("instance_count")
    sop_hash = sig_raw.get("sop_uids_sha256")
    if (
        isinstance(instance_count, bool)
        or not isinstance(instance_count, int)
        or instance_count <= 0
        or not isinstance(sop_hash, str)
        or len(sop_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in sop_hash)
    ):
        raise CTSeriesHygieneError(
            "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
        )
    for str_key in ("ScanOptions", "ConvolutionKernel"):
        val = sig_raw.get(str_key)
        if val is not None and not isinstance(val, str):
            raise CTSeriesHygieneError(
                "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
            )
    for num_key in ("ReconstructionDiameter", "Exposure"):
        val = sig_raw.get(num_key)
        if val is not None and _canonical_decimal(val) is None:
            raise CTSeriesHygieneError(
                "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
            )
    wc_val = sig_raw.get("WindowCenter")
    if wc_val is not None:
        if isinstance(wc_val, list):
            if not wc_val or any(_canonical_decimal(item) is None for item in wc_val):
                raise CTSeriesHygieneError(
                    "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
                )
        elif _canonical_decimal(wc_val) is None:
            raise CTSeriesHygieneError(
                "ct_hygiene_stack_override_invalid_governance", evidence=err_evidence
            )
    return adjudication_record_id.strip(), reason.strip(), sig_raw


def conversion_evidence_matches_expected(
    recorded: Mapping[str, Any], expected_selection: Mapping[str, Any]
) -> bool:
    """Compare stored CT hygiene conversion evidence against current selection.

    Accepts both the current WS201 evidence schema (with `reason_code` and
    `selection_basis`) and the legacy parent (`1aeb0c5`) hygiene evidence schema
    when the selected instance set and all legacy fields match exactly.
    """
    if not isinstance(recorded, Mapping):
        return False
    expected = dict(expected_selection)
    expected["status"] = "converted"
    is_legacy_schema = (
        "reason_code" not in recorded
        and "selection_basis" not in recorded
        and "stack_override" not in recorded
        and "stack_override" not in expected
        and expected.get("reason_code") == "verified_instance_exclusions"
        and expected.get("selection_basis")
        == "rtstruct_referenced_geometry_or_identical_duplicates"
        and recorded.get("kept_sop_instance_uids") == expected.get("kept_sop_instance_uids")
    )
    if is_legacy_schema:
        expected_compare = {
            k: v
            for k, v in expected.items()
            if k not in ("reason_code", "selection_basis")
        }
    else:
        expected_compare = expected
    recorded_keys = set(recorded.keys()) - {"downstream_conversion"}
    if not (recorded_keys <= set(expected_compare.keys())):
        return False
    return all(recorded.get(key) == value for key, value in expected_compare.items())


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


def select_duplicate_positions(group, refs, *, position_tolerance_mm=POSITION_TOLERANCE_MM,
                               refuse_referenced=True, include_pixel_evidence=False,
                               allow_referenced_differences=False):
    """Share RF10's pixel proof and deterministic tie break with publication.

    A zero tolerance requires exact numeric positions. Publication always
    protects referenced instances; RF10 retains its historical mixed-series gate.
    Differing pixels require explicit opt-in and complete image-level authority.
    """
    excluded = []
    differing = False
    # Prefer a referenced equivalent instance, then lowest InstanceNumber/UID.
    ordered = sorted(group, key=lambda item: (item["uid"] not in refs, item["instance"], item["uid"]))
    kept: list[dict] = []
    for item in ordered:
        matches = [other for other in kept if (
            np.array_equal(item["position"], other["position"])
            if position_tolerance_mm == 0 else
            np.linalg.norm(item["position"] - other["position"]) <= position_tolerance_mm)]
        if len(matches) > 1:
            raise CTSeriesHygieneError("ct_hygiene_ambiguous_duplicate_position")
        if matches:
            other = matches[0]
            retained_pixels = _pixels(other["path"])
            excluded_pixels = _pixels(item["path"])
            different = excluded_pixels != retained_pixels
            if different and not allow_referenced_differences:
                raise CTSeriesHygieneError("ct_hygiene_duplicate_position_pixels_differ")
            differing |= different
            if (refuse_referenced or different) and item["uid"] in refs:
                raise CTSeriesHygieneError("ct_hygiene_referenced_duplicate_would_be_excluded")
            excluded.append({"sop_instance_uid": item["uid"],
                             "reason_code": ("unreferenced_duplicate_position" if different else
                                             "identical_rescaled_duplicate_position"),
                             "retained_sop_instance_uid": other["uid"]})
            if include_pixel_evidence:
                excluded[-1].update(
                    image_position_patient=other["position"].tolist(),
                    rescaled_pixels_sha256=hashlib.sha256(retained_pixels).hexdigest())
            if different:
                excluded[-1].update(
                    image_position_patient=other["position"].tolist(),
                    retained_rescaled_pixels_sha256=hashlib.sha256(retained_pixels).hexdigest(),
                    excluded_rescaled_pixels_sha256=hashlib.sha256(excluded_pixels).hexdigest())
        else:
            kept.append(item)
    if differing and refs != {item["uid"] for item in kept}:
        # Exactly one reference at EVERY retained position, including positions
        # whose copies happen to match. Outside references are also refused.
        raise CTSeriesHygieneError("ct_hygiene_duplicate_position_reference_coverage")
    return kept, excluded


def select_geometry_partition(partitions, refs):
    """RF10 image-level authority: all references must select exactly one group."""
    if len(partitions) == 1:
        return next(iter(partitions.items()))
    if not refs:
        raise CTSeriesHygieneError("ct_hygiene_mixed_geometry_without_image_references")
    candidates = [(key, group) for key, group in partitions.items()
                  if refs <= {item["uid"] for item in group}]
    if len(candidates) != 1:
        raise CTSeriesHygieneError("ct_hygiene_references_span_geometry_partitions")
    return candidates[0]


def _check_axial_orientation(orientation: np.ndarray, *, require_axial_z: bool = True) -> np.ndarray:
    row, col = orientation[:3], orientation[3:]
    normal = np.cross(row, col)
    if (
        abs(np.linalg.norm(row) - 1) > ORIENTATION_TOLERANCE
        or abs(np.linalg.norm(col) - 1) > ORIENTATION_TOLERANCE
        or abs(np.dot(row, col)) > ORIENTATION_TOLERANCE
        or (require_axial_z and abs(abs(normal[2]) - 1) > ORIENTATION_TOLERANCE)
    ):
        raise CTSeriesHygieneError("ct_hygiene_not_consistent_axial_volume")
    return normal


def _volume_projections_and_steps(
    items: list[dict[str, Any]], normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    if len(items) < 3:
        return np.array([]), np.array([]), False, False
    positions = np.array([item["position"] for item in items], dtype=float)
    projections = positions @ normal
    steps = np.diff(np.sort(projections))
    residual = positions - positions[0] - np.outer(projections - projections[0], normal)
    consistent = bool(
        np.min(steps) > POSITION_TOLERANCE_MM
        and np.max(np.linalg.norm(residual, axis=1)) <= POSITION_TOLERANCE_MM
    )
    uniform = bool(consistent and (np.max(steps) - np.min(steps) <= POSITION_TOLERANCE_MM))
    return projections, steps, consistent, uniform


def _detect_spanning_reconstruction_stacks(
    partitions: dict[tuple, list[dict[str, Any]]],
    refs: set[str],
    frame_refs: set[str],
    contour_refs: set[str],
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[dict[str, Any]]] | None:
    """Detect Pattern D: RTSTRUCT references span >=2 complete same-geometry stacks."""
    if not refs:
        return None
    referenced_partitions = [
        (pkey, group)
        for pkey, group in partitions.items()
        if any(item["uid"] in refs for item in group)
    ]
    if not referenced_partitions:
        return None
    active_items = [item for _, group in referenced_partitions for item in group]
    if any(item["is_localizer"] for item in active_items):
        return None
    active_uids = {item["uid"] for item in active_items}
    if not (refs <= active_uids):
        return None
    base_orient_rows_cols = {
        (*item["partition_key"][:6], item["partition_key"][6], item["partition_key"][7])
        for item in active_items
    }
    if len(base_orient_rows_cols) != 1:
        return None
    orientation = np.array(active_items[0]["partition_key"][:6], dtype=float)
    try:
        normal = _check_axial_orientation(orientation, require_axial_z=True)
    except CTSeriesHygieneError:
        return None

    stacks_by_attr: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in active_items:
        stacks_by_attr.setdefault(_stack_attr_key(item["stack_attrs"]), []).append(item)
    if len(stacks_by_attr) < 2:
        return None

    reference_projections: np.ndarray | None = None
    stack_records: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    referenced_stack_count = 0
    for stack_items in stacks_by_attr.values():
        if len({item["partition_key"] for item in stack_items}) != 1:
            return None
        projections, _steps, _consistent, uniform = _volume_projections_and_steps(
            stack_items, normal
        )
        if not uniform:
            return None
        sorted_proj = np.sort(projections)
        if reference_projections is None:
            reference_projections = sorted_proj
        elif (
            len(sorted_proj) != len(reference_projections)
            or float(np.max(np.abs(sorted_proj - reference_projections))) > POSITION_TOLERANCE_MM
        ):
            return None
        stack_uids = {item["uid"] for item in stack_items}
        if stack_uids & refs:
            referenced_stack_count += 1
        attrs = dict(stack_items[0]["stack_attrs"])
        sop_hash = compute_stack_sop_uids_sha256(stack_uids)
        sig = compute_stack_signature(attrs, len(stack_items), sop_hash)
        candidate = {
            **{name: attrs.get(name) for name in STACK_DISTINGUISHING_ATTRIBUTES},
            "instance_count": len(stack_items),
            "contour_reference_count": len(stack_uids & contour_refs),
            "frame_reference_count": len(stack_uids & frame_refs),
            "sop_uids_sha256": sop_hash,
            "stack_signature": sig,
        }
        stack_records.append((candidate, stack_items))

    if referenced_stack_count < 2:
        return None

    stack_records.sort(
        key=lambda pair: (
            str(pair[0].get("ConvolutionKernel") or ""),
            str(pair[0].get("ReconstructionDiameter") or ""),
            str(pair[0].get("ScanOptions") or ""),
            str(pair[0].get("WindowCenter") or ""),
            str(pair[0].get("Exposure") or ""),
            str(pair[0].get("sop_uids_sha256") or ""),
        )
    )
    candidate_stacks = [pair[0] for pair in stack_records]
    return stack_records, candidate_stacks


def _validate_pattern_e_mixed_image_type(
    all_items: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    partitions: dict[tuple, list[dict[str, Any]]],
    refs: set[str],
    frame_refs: set[str],
    contour_refs: set[str],
    steps: np.ndarray,
) -> tuple[dict[str, int], list[str]]:
    """Validate Pattern E prerequisites or fail closed with ct_hygiene_mixed_image_type_refused."""
    kept_uids = {item["uid"] for item in kept}
    if (
        excluded
        or len(partitions) != 1
        or len(kept) != len(all_items)
        or len(kept) < 3
        or not refs
        or refs != kept_uids
        or (frame_refs and frame_refs != kept_uids)
        or (contour_refs and not (contour_refs <= kept_uids))
        or float(np.max(steps) - np.min(steps)) > POSITION_TOLERANCE_MM
    ):
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    if len({item["study_uid"] for item in kept}) != 1 or "" in {item["study_uid"] for item in kept}:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    if len({item["acquisition_key"] for item in kept}) != 1:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    if len({item["pixel_meta"] for item in kept}) != 1:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    pixel_meta = kept[0]["pixel_meta"]
    samples_per_pixel, bits_allocated, bits_stored, high_bit, _pixel_rep, has_lut, is_raw_le, slope, intercept = pixel_meta
    if (
        samples_per_pixel != 1
        or bits_allocated != 16
        or bits_stored <= 0
        or high_bit != bits_stored - 1
        or has_lut
        or not is_raw_le
        or not np.isfinite([slope, intercept]).all()
        or slope == 0
    ):
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    image_types = [item["image_type"] for item in kept]
    if any(len(it) < 3 or it[2] != "AXIAL" for it in image_types):
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    if len({it[2:] for it in image_types}) != 1:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    counts: dict[str, int] = {}
    tuple_counts: dict[tuple[str, ...], int] = {}
    for it in image_types:
        key_text = "\\".join(it)
        counts[key_text] = counts.get(key_text, 0) + 1
        tuple_counts[it] = tuple_counts.get(it, 0) + 1
    preferred = ("ORIGINAL", "PRIMARY", *image_types[0][2:])
    if preferred in tuple_counts:
        harmonized = list(preferred)
    else:
        harmonized = list(
            max(tuple_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        )
    return dict(sorted(counts.items())), harmonized


def select_ct_instances(
    ct_dir: Path,
    authoritative_rtstruct: Path | None = None,
    *,
    stack_override: dict[str, Any] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Select one consistent volume or raise a stable, identifier-free reason."""
    try:
        return _select(ct_dir, authoritative_rtstruct, stack_override=stack_override)
    except CTSeriesHygieneError:
        raise
    except Exception as exc:
        raise CTSeriesHygieneError("ct_hygiene_invalid_dicom_or_geometry") from exc


def _select(
    ct_dir: Path,
    authoritative_rtstruct: Path | None,
    *,
    stack_override: dict[str, Any] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    entries = sorted(ct_dir.iterdir())
    if not entries or any(not p.is_file() for p in entries):
        raise CTSeriesHygieneError("ct_hygiene_invalid_series_directory")
    partitions: dict[tuple, list[dict[str, Any]]] = {}
    all_items: list[dict[str, Any]] = []
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
        study_uid = str(getattr(ds, "StudyInstanceUID", ""))
        orientation = _vector(ds, "ImageOrientationPatient", 6)
        position = _vector(ds, "ImagePositionPatient", 3)
        spacing = _vector(ds, "PixelSpacing", 2)
        rows, columns = int(ds.Rows), int(ds.Columns)
        if min(rows, columns) <= 0 or min(spacing) <= 0:
            raise ValueError("invalid dimensions")
        image_type = _image_type_tokens(ds)
        is_loc = _is_localizer_tokens(image_type)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        ts_uid = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "") or "")
        is_raw_le = ts_uid in (
            "",
            "1.2.840.10008.1.2",
            "1.2.840.10008.1.2.1",
        )
        pixel_meta = (
            int(getattr(ds, "SamplesPerPixel", 1) or 1),
            int(getattr(ds, "BitsAllocated", 0) or 0),
            int(getattr(ds, "BitsStored", 0) or 0),
            int(getattr(ds, "HighBit", -1) if getattr(ds, "HighBit", None) is not None else -1),
            int(getattr(ds, "PixelRepresentation", 0) or 0),
            bool(hasattr(ds, "ModalityLUTSequence")),
            is_raw_le,
            slope,
            intercept,
        )
        acquisition_key = (
            str(getattr(ds, "AcquisitionNumber", "")),
            str(getattr(ds, "TemporalPositionIdentifier", "")),
            str(getattr(ds, "EchoNumbers", "")),
        )
        stack_attrs = {
            name: _stack_attr_value(ds, name)
            for name in STACK_DISTINGUISHING_ATTRIBUTES
        }
        # Exact numeric partitioning is deliberately conservative. Tolerance is
        # used for volume consistency, never to merge different geometries.
        key = (*orientation, rows, columns, *spacing)
        item = {
            "path": path,
            "uid": uid,
            "position": position,
            "instance": int(getattr(ds, "InstanceNumber", 0)),
            "partition_key": key,
            "image_type": image_type,
            "is_localizer": is_loc,
            "study_uid": study_uid,
            "acquisition_key": acquisition_key,
            "pixel_meta": pixel_meta,
            "stack_attrs": stack_attrs,
        }
        partitions.setdefault(key, []).append(item)
        all_items.append(item)

    if len(series) != 1 or "" in series or len(frames) != 1 or "" in frames:
        raise CTSeriesHygieneError("ct_hygiene_series_or_frame_mismatch")
    frame_refs, contour_refs = rtstruct_reference_levels(authoritative_rtstruct)
    refs = frame_refs | contour_refs
    if refs - identities:
        raise CTSeriesHygieneError("ct_hygiene_rtstruct_references_unresolvable")

    non_localizers = [item for item in all_items if not item["is_localizer"]]
    has_derived_image_type = (
        len({item["image_type"][:2] for item in non_localizers if item["image_type"]}) > 1
        and any("DERIVED" in item["image_type"][:2] for item in non_localizers)
    )

    # Pattern D: check if RTSTRUCT references span >=2 complete same-geometry reconstruction stacks.
    spanning_stacks = _detect_spanning_reconstruction_stacks(
        partitions, refs, frame_refs, contour_refs
    )
    if spanning_stacks is not None:
        stack_records, candidate_stacks = spanning_stacks
        if stack_override is None:
            raise CTSeriesHygieneError(
                "ct_hygiene_references_span_reconstruction_stacks",
                evidence={"candidate_stacks": candidate_stacks},
            )
        adjudication_id, override_reason, requested_sig = _validate_stack_override_governance(
            stack_override, candidate_stacks
        )
        matched = [
            (cand, items)
            for cand, items in stack_records
            if _signatures_match(cand["stack_signature"], requested_sig)
        ]
        if len(matched) != 1:
            raise CTSeriesHygieneError(
                "ct_hygiene_stack_override_signature_mismatch",
                evidence={"candidate_stacks": candidate_stacks},
            )
        selected_cand, kept = matched[0]
        key = kept[0]["partition_key"]
        kept_uids = {item["uid"] for item in kept}
        other_stack_uids = {
            item["uid"]
            for cand, items in stack_records
            if cand is not selected_cand
            for item in items
        }
        excluded = [
            {
                "sop_instance_uid": item["uid"],
                "reason_code": (
                    "governed_stack_override_excluded"
                    if item["uid"] in other_stack_uids
                    else "unreferenced_geometry_partition"
                ),
            }
            for item in all_items
            if item["uid"] not in kept_uids
        ]
        orientation = np.array(key[:6], dtype=float)
        normal = _check_axial_orientation(orientation, require_axial_z=True)
        _proj, steps, _cons, _unif = _volume_projections_and_steps(kept, normal)
        normalized_override = {
            "adjudication_record_id": adjudication_id,
            "reason": override_reason,
            "stack_signature": dict(selected_cand["stack_signature"]),
        }
        evidence = {
            "method": METHOD,
            "status": "selected",
            "reason": "Governed stack override selected one verified reconstruction stack",
            "reason_code": "governed_stack_override_selected",
            "selection_basis": "governed_stack_override",
            "source_instance_count": len(entries),
            "kept_instance_count": len(kept),
            "excluded_instance_count": len(excluded),
            "geometry_partition_count": len(partitions),
            "position_tolerance_mm": POSITION_TOLERANCE_MM,
            "kept_sop_instance_uids": sorted(kept_uids),
            "excluded_instances": sorted(excluded, key=lambda item: item["sop_instance_uid"]),
            "rtstruct_referenced_instance_count": len(refs),
            "rtstruct_references_all_kept": refs <= kept_uids,
            "kept_all_rtstruct_referenced": kept_uids <= refs,
            "authoritative_rtstruct_sha256": (
                hashlib.sha256(authoritative_rtstruct.read_bytes()).hexdigest()
                if authoritative_rtstruct
                else None
            ),
            "selected_geometry": {
                "image_orientation_patient": list(key[:6]),
                "rows": key[6],
                "columns": key[7],
                "pixel_spacing": list(key[8:]),
            },
            "slice_step_min_mm": float(min(steps)),
            "slice_step_max_mm": float(max(steps)),
            "candidate_stacks": candidate_stacks,
            "stack_override": normalized_override,
            "verification": (
                "Governed stack override signature matched exactly; selected reconstruction "
                "stack verified as one uniform axial volume"
            ),
        }
        return sorted(item["path"] for item in kept), evidence

    if stack_override is not None:
        _validate_stack_override_governance(stack_override)
        raise CTSeriesHygieneError("ct_hygiene_stack_override_signature_mismatch")

    mixed = len(partitions) > 1
    try:
        key, group = select_geometry_partition(partitions, refs)
    except CTSeriesHygieneError:
        if has_derived_image_type and (not refs or refs == {item["uid"] for item in non_localizers}):
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        raise

    excluded = [
        {"sop_instance_uid": item["uid"], "reason_code": "unreferenced_geometry_partition"}
        for other_key, other_group in partitions.items()
        if other_key != key
        for item in other_group
    ]
    orientation = np.array(key[:6], dtype=float)
    row, col = orientation[:3], orientation[3:]
    normal = np.cross(row, col)
    if (
        abs(np.linalg.norm(row) - 1) > ORIENTATION_TOLERANCE
        or abs(np.linalg.norm(col) - 1) > ORIENTATION_TOLERANCE
        or abs(np.dot(row, col)) > ORIENTATION_TOLERANCE
        or (mixed and abs(abs(normal[2]) - 1) > ORIENTATION_TOLERANCE)
    ):
        if has_derived_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        raise CTSeriesHygieneError("ct_hygiene_not_consistent_axial_volume")

    kept, duplicates = select_duplicate_positions(
        group,
        refs,
        refuse_referenced=mixed,
        allow_referenced_differences=bool(refs),
    )
    differing_duplicates = [
        item for item in duplicates if item["reason_code"] == "unreferenced_duplicate_position"
    ]
    if differing_duplicates:
        kept_uids_check = {item["uid"] for item in kept}
        if (
            not frame_refs
            or not contour_refs
            or frame_refs != kept_uids_check
            or not (contour_refs <= kept_uids_check)
            or any(
                item["retained_sop_instance_uid"] not in contour_refs
                or item["retained_sop_instance_uid"] not in frame_refs
                for item in differing_duplicates
            )
        ):
            raise CTSeriesHygieneError("ct_hygiene_duplicate_position_reference_coverage")
    excluded.extend(duplicates)

    projections, steps, consistent, uniform = _volume_projections_and_steps(kept, normal)

    # Pattern A: when RTSTRUCT image-level references select a subset of `kept`,
    # exclude unreferenced stray slices before falling back to interpolation.
    if refs:
        ref_kept = [item for item in kept if item["uid"] in refs]
        unref_kept = [item for item in kept if item["uid"] not in refs]
        if 0 < len(ref_kept) < len(kept):
            ref_acqs = {item["acquisition_key"] for item in ref_kept}
            unref_acqs = {item["acquisition_key"] for item in unref_kept}
            distinct_unref_acq = bool(unref_acqs and not (unref_acqs & ref_acqs))
            if not excluded or not uniform or distinct_unref_acq:
                ref_uids = {item["uid"] for item in ref_kept}
                if (
                    refs != ref_uids
                    or (frame_refs and frame_refs != ref_uids)
                    or (contour_refs and not (contour_refs <= ref_uids))
                ):
                    raise CTSeriesHygieneError("ct_hygiene_incomplete_rtstruct_references")
                ref_proj, ref_steps, ref_consistent, ref_uniform = _volume_projections_and_steps(
                    ref_kept, normal
                )
                if not ref_consistent:
                    raise CTSeriesHygieneError("ct_hygiene_incomplete_rtstruct_references")
                all_primary_acq_referenced = distinct_unref_acq and (
                    {item["uid"] for item in kept if item["acquisition_key"] in ref_acqs} == ref_uids
                )
                if not all_primary_acq_referenced:
                    if not ref_uniform or uniform:
                        raise CTSeriesHygieneError("ct_hygiene_incomplete_rtstruct_references")
                    mean_step = float(np.mean(ref_steps))
                    sorted_ref_proj = np.sort(ref_proj)
                    for item in unref_kept:
                        if item["acquisition_key"] in ref_acqs:
                            proj = float(np.dot(item["position"], normal))
                            min_dist = float(np.min(np.abs(sorted_ref_proj - proj)))
                            if min_dist >= mean_step - POSITION_TOLERANCE_MM:
                                raise CTSeriesHygieneError("ct_hygiene_incomplete_rtstruct_references")
                for item in unref_kept:
                    excluded.append(
                        {
                            "sop_instance_uid": item["uid"],
                            "reason_code": "unreferenced_slice_position",
                        }
                    )
                kept = ref_kept
                projections, steps, consistent, uniform = (
                    ref_proj,
                    ref_steps,
                    ref_consistent,
                    ref_uniform,
                )

    if not consistent:
        if has_derived_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        raise CTSeriesHygieneError("ct_hygiene_inconsistent_volume_positions")

    kept_image_type_prefixes = {item["image_type"][:2] for item in kept if item["image_type"]}
    is_mixed_image_type = len(kept_image_type_prefixes) > 1 or has_derived_image_type
    original_image_type_counts: dict[str, int] | None = None
    harmonized_image_type: list[str] | None = None

    if not excluded:
        if is_mixed_image_type:
            original_image_type_counts, harmonized_image_type = (
                _validate_pattern_e_mixed_image_type(
                    all_items, kept, excluded, partitions, refs, frame_refs, contour_refs, steps
                )
            )
        else:
            raise CTSeriesHygieneError("ct_hygiene_no_safe_exclusions")
    elif len(kept_image_type_prefixes) > 1:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")

    kept_uids = {item["uid"] for item in kept}
    exclusion_reasons = {item["reason_code"] for item in excluded}
    if original_image_type_counts is not None:
        reason_text = "Failed CT conversion retried after verified ImageType harmonization"
        reason_code = "mixed_image_type_harmonized"
        selection_basis = "rtstruct_complete_references_image_type_harmonized"
        verification_text = (
            "Staged ImageType tag harmonized; single NIfTI volume verified slice-by-slice "
            "against source rescaled float64 pixels and DICOM slice positions without interpolation"
        )
    elif "unreferenced_slice_position" in exclusion_reasons:
        reason_text = "Failed CT conversion retried after excluding unreferenced slice positions"
        reason_code = "unreferenced_slice_position"
        selection_basis = "rtstruct_complete_referenced_volume"
        verification_text = (
            "Authoritative RTSTRUCT references select a consistent volume; excluded "
            "unreferenced slice positions have no RTSTRUCT image references"
        )
    elif "unreferenced_duplicate_position" in exclusion_reasons:
        reason_text = "Failed CT conversion retried after excluding unreferenced duplicate positions"
        reason_code = "unreferenced_duplicate_position"
        selection_basis = "rtstruct_frame_and_contour_complete_references"
        verification_text = (
            "Authoritative RTSTRUCT frame and contour references select exactly one instance "
            "per position; excluded duplicate rescaled float64 pixel SHA-256 hashes recorded"
        )
    else:
        reason_text = "Failed CT conversion retried after verified instance exclusions"
        reason_code = "verified_instance_exclusions"
        selection_basis = "rtstruct_referenced_geometry_or_identical_duplicates"
        verification_text = (
            "Duplicate rescaled float64 pixel bytes are identical; excluded geometry "
            "partitions have no RTSTRUCT image references"
        )

    evidence: dict[str, Any] = {
        "method": METHOD,
        "status": "selected",
        "reason": reason_text,
        "reason_code": reason_code,
        "selection_basis": selection_basis,
        "source_instance_count": len(entries),
        "kept_instance_count": len(kept),
        "excluded_instance_count": len(excluded),
        "geometry_partition_count": len(partitions),
        "position_tolerance_mm": POSITION_TOLERANCE_MM,
        "kept_sop_instance_uids": sorted(kept_uids),
        "excluded_instances": sorted(excluded, key=lambda item: item["sop_instance_uid"]),
        "rtstruct_referenced_instance_count": len(refs),
        "rtstruct_references_all_kept": refs <= kept_uids,
        "authoritative_rtstruct_sha256": (
            hashlib.sha256(authoritative_rtstruct.read_bytes()).hexdigest()
            if authoritative_rtstruct
            else None
        ),
        "selected_geometry": {
            "image_orientation_patient": list(key[:6]),
            "rows": key[6],
            "columns": key[7],
            "pixel_spacing": list(key[8:]),
        },
        "slice_step_min_mm": float(min(steps)),
        "slice_step_max_mm": float(max(steps)),
        "verification": verification_text,
    }
    if original_image_type_counts is not None and harmonized_image_type is not None:
        evidence["original_image_type_counts"] = original_image_type_counts
        evidence["harmonized_image_type"] = harmonized_image_type
    return sorted(item["path"] for item in kept), evidence


def _rewrite_image_type_tag_bytes(raw: bytes, target_image_type: Sequence[str]) -> bytes:
    """Rewrite only top-level (0008, 0008) ImageType value bytes in a Part 10 file."""
    if len(raw) < 132 or raw[128:132] != b"DICM":
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    offset = 132
    n = len(raw)
    # Skip Group 0002 (Explicit VR Little Endian)
    while offset + 8 <= n:
        group, elem = struct.unpack_from("<HH", raw, offset)
        if group != 0x0002:
            break
        vr = raw[offset + 4 : offset + 6]
        if vr in _EXPLICIT_VR_LONG_LENGTH_VRS:
            if offset + 12 > n:
                raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
            length = struct.unpack_from("<I", raw, offset + 8)[0]
            header_len = 12
        else:
            length = struct.unpack_from("<H", raw, offset + 6)[0]
            header_len = 8
        offset += header_len + length

    new_val = "\\".join(str(part).strip() for part in target_image_type).encode("ascii")
    if len(new_val) % 2 == 1:
        new_val += b" "

    while offset + 8 <= n:
        group, elem = struct.unpack_from("<HH", raw, offset)
        if group > 0x0008 or (group == 0x0008 and elem > 0x0008):
            break
        maybe_vr = raw[offset + 4 : offset + 6]
        is_explicit = all(65 <= b <= 90 for b in maybe_vr)
        if is_explicit:
            if maybe_vr in _EXPLICIT_VR_LONG_LENGTH_VRS:
                if offset + 12 > n:
                    raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
                length = struct.unpack_from("<I", raw, offset + 8)[0]
                header_len = 12
            else:
                length = struct.unpack_from("<H", raw, offset + 6)[0]
                header_len = 8
        else:
            length = struct.unpack_from("<I", raw, offset + 4)[0]
            header_len = 8
        if length == 0xFFFFFFFF:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        val_end = offset + header_len + length
        if val_end > n:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        if group == 0x0008 and elem == 0x0008:
            if is_explicit and maybe_vr != b"CS":
                raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
            if is_explicit:
                new_header = struct.pack("<HH2sH", group, elem, b"CS", len(new_val))
            else:
                new_header = struct.pack("<HHI", group, elem, len(new_val))
            return raw[:offset] + new_header + new_val + raw[val_end:]
        offset = val_end
    raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")


def _stage_harmonized_image_type(
    files: list[Path], staging: Path, target_image_type: list[str]
) -> None:
    target_tuple = tuple(str(x).strip().upper() for x in target_image_type)
    for path in files:
        ds = pydicom.dcmread(path)
        current = _image_type_tokens(ds)
        dest = staging / path.name
        if current == target_tuple:
            shutil.copyfile(path, dest)
            continue
        rewritten = _rewrite_image_type_tag_bytes(path.read_bytes(), target_tuple)
        dest.write_bytes(rewritten)
        verify_ds = pydicom.dcmread(dest)
        if (
            _image_type_tokens(verify_ds) != target_tuple
            or verify_ds.PixelData != ds.PixelData
            or str(verify_ds.SOPInstanceUID) != str(ds.SOPInstanceUID)
        ):
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")


def _verify_nifti_voxel_payload(
    nifti_path: Path,
    expected_rows: int,
    expected_cols: int,
    expected_slices: int,
    *,
    allow_equalized_slices: bool = False,
    is_mixed_image_type: bool = False,
) -> None:
    """Verify 3D extent and complete, finite, decompressable voxel payload."""
    import nibabel as nib

    try:
        img = nib.load(str(nifti_path))
        shape = tuple(int(dim) for dim in img.shape)
    except Exception as exc:
        if is_mixed_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused") from exc
        raise CTSeriesHygieneError("ct_hygiene_unreadable_converter_output") from exc
    slice_ok = (
        len(shape) == 3 and shape[2] >= int(expected_slices)
        if allow_equalized_slices
        else len(shape) == 3 and shape[2] == int(expected_slices)
    )
    if (
        len(shape) != 3
        or not slice_ok
        or sorted(shape[:2]) != sorted([int(expected_rows), int(expected_cols)])
    ):
        if is_mixed_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        raise CTSeriesHygieneError("ct_hygiene_converter_did_not_preserve_selected_extent")
    try:
        voxels = img.get_fdata(dtype=np.float64, caching="unchanged")
    except Exception as exc:
        if is_mixed_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused") from exc
        raise CTSeriesHygieneError("ct_hygiene_unreadable_converter_output") from exc
    if (
        voxels.shape != shape
        or voxels.size != int(np.prod(shape))
        or voxels.size <= 0
        or not np.isfinite(voxels).all()
    ):
        if is_mixed_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        raise CTSeriesHygieneError("ct_hygiene_unreadable_converter_output")


def _verify_slice_by_slice_volume(files: list[Path], nifti_path: Path) -> None:
    """Verify exact slice-by-slice rescaled float64 pixels and float32-bounded NIfTI-1 geometry."""
    import nibabel as nib

    try:
        slices = []
        for path in files:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "ModalityLUTSequence"):
                raise ValueError("ModalityLUTSequence unsupported")
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            if not np.isfinite([slope, intercept]).all() or slope == 0:
                raise ValueError("invalid rescale slope/intercept")
            rescaled = np.asarray(ds.pixel_array, dtype=np.float64) * slope + intercept
            orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
            position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
            normal = np.cross(orientation[:3], orientation[3:])
            slices.append((float(np.dot(position, normal)), position, orientation, spacing, rescaled))
        slices.sort(key=lambda item: item[0])
        if len(slices) < 2:
            raise ValueError("insufficient slices")
        expected_stack = np.stack([item[4] for item in slices], axis=0)
        first_pos = slices[0][1]
        last_pos = slices[-1][1]
        first_orient = slices[0][2]
        first_spacing = slices[0][3]
        for _proj, _pos, orient, sp, _pix in slices[1:]:
            if not np.array_equal(orient, first_orient) or not np.array_equal(sp, first_spacing):
                raise ValueError("non-uniform DICOM slice orientation or spacing")
        rows = int(expected_stack.shape[1])
        n_slices = len(slices)
        row_dir = first_orient[:3]
        col_dir = first_orient[3:]
        row_spacing = float(first_spacing[0])
        col_spacing = float(first_spacing[1])
        slice_vec_lps = (last_pos - first_pos) / float(n_slices - 1)
        lps_to_ras = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
        col_i_ras = row_dir * col_spacing * lps_to_ras
        col_j_ras = (-col_dir) * row_spacing * lps_to_ras
        col_k_ras = slice_vec_lps * lps_to_ras
        origin_ras = (first_pos + (rows - 1) * row_spacing * col_dir) * lps_to_ras
        expected_affine_f64 = np.eye(4, dtype=np.float64)
        expected_affine_f64[:3, 0] = col_i_ras
        expected_affine_f64[:3, 1] = col_j_ras
        expected_affine_f64[:3, 2] = col_k_ras
        expected_affine_f64[:3, 3] = origin_ras
        # NIfTI-1 stores sform entries as IEEE binary32. Some converter
        # calculations also round the source attributes to binary32 before
        # forming the affine, notably the translation after the row flip.
        # Accept only these two exact representations, component by component;
        # no additional millimetre or angular tolerance is permitted.
        expected_sform_f32 = expected_affine_f64.astype(np.float32)
        converter_sform_f32 = np.eye(4, dtype=np.float32)
        ras_f32 = lps_to_ras.astype(np.float32)
        row_f32, col_f32 = row_dir.astype(np.float32), col_dir.astype(np.float32)
        pos_f32 = first_pos.astype(np.float32)
        converter_sform_f32[:3, 0] = row_f32 * np.float32(col_spacing) * ras_f32
        converter_sform_f32[:3, 1] = -col_f32 * np.float32(row_spacing) * ras_f32
        converter_sform_f32[:3, 2] = (
            (last_pos.astype(np.float32) - pos_f32) / np.float32(n_slices - 1) * ras_f32
        )
        converter_sform_f32[:3, 3] = (
            pos_f32 + np.float32(rows - 1) * np.float32(row_spacing) * col_f32
        ) * ras_f32
        for idx, (_proj, pos, _orient, _sp, _pix) in enumerate(slices):
            expected_pos = first_pos + idx * slice_vec_lps
            if not np.array_equal(pos.astype(np.float32), expected_pos.astype(np.float32)):
                raise ValueError("source positions cannot be represented by one uniform affine")

        nib_img = nib.load(str(nifti_path))
        hdr = nib_img.header
        if type(hdr) is not nib.Nifti1Header or hdr.get_xyzt_units()[0] != "mm":
            raise ValueError("expected millimetre NIfTI-1 geometry")
        sform, sform_code = hdr.get_sform(coded=True)
        if int(sform_code) <= 0 or sform is None:
            raise ValueError("missing NIfTI sform")
        if not np.all((sform == expected_sform_f32) | (sform == converter_sform_f32)):
            raise ValueError("NIfTI sform mismatch")
        # Verify the independently encoded qform too: readers may prefer it.
        # Decode a canonical binary32 quaternion rather than adding a rotation
        # tolerance. This also handles equivalent quaternion signs at 180 degrees.
        qform, qform_code = hdr.get_qform(coded=True)
        if int(qform_code) <= 0:
            raise ValueError("missing NIfTI qform")
        for affine in (expected_affine_f64, sform):
            expected_header = nib.Nifti1Header()
            expected_header.set_qform(affine, code=1)
            if np.array_equal(qform, expected_header.get_qform()):
                break
        else:
            raise ValueError("NIfTI qform mismatch")

        # dcm2niix's native CT layout flips DICOM rows; use the same index
        # mapping as the affine above, without another reader's reorientation
        # or quaternion normalization changing the geometry being checked.
        actual_array = nib_img.get_fdata(dtype=np.float64, caching="unchanged")
        expected_array = expected_stack[:, ::-1, :].transpose(2, 1, 0)
        if actual_array.shape != expected_array.shape or not np.array_equal(actual_array, expected_array):
            raise ValueError("slice rescaled pixel mismatch")
    except CTSeriesHygieneError:
        raise
    except Exception as exc:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused") from exc


def convert_ct_series_hygiene(
    ct_dir: Path,
    work_dir: Path,
    converter: Converter,
    authoritative_rtstruct: Path | None = None,
    *,
    stack_override: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    files, evidence = select_ct_instances(
        ct_dir, authoritative_rtstruct, stack_override=stack_override
    )
    is_mixed_image_type = evidence.get("reason_code") == "mixed_image_type_harmonized"
    staging = work_dir / "hygiene_input"
    output = work_dir / "hygiene_output"
    staging.mkdir()
    output.mkdir()
    if is_mixed_image_type:
        _stage_harmonized_image_type(files, staging, evidence["harmonized_image_type"])
    else:
        for path in files:
            shutil.copyfile(path, staging / path.name)
    generated = converter(staging, output)
    if generated is None:
        if is_mixed_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
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
    if is_mixed_image_type and equalized:
        raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
    if len(unequalized) != 1 or len(equalized) > 1 or (equalized and generated != equalized[0]):
        if is_mixed_image_type:
            raise CTSeriesHygieneError("ct_hygiene_mixed_image_type_refused")
        raise CTSeriesHygieneError("ct_hygiene_ambiguous_converter_outputs")
    geometry = evidence["selected_geometry"]
    _verify_nifti_voxel_payload(
        unequalized[0],
        geometry["rows"],
        geometry["columns"],
        len(files),
        allow_equalized_slices=False,
        is_mixed_image_type=is_mixed_image_type,
    )
    if generated != unequalized[0]:
        _verify_nifti_voxel_payload(
            generated,
            geometry["rows"],
            geometry["columns"],
            len(files),
            allow_equalized_slices=True,
            is_mixed_image_type=is_mixed_image_type,
        )
    if is_mixed_image_type:
        _verify_slice_by_slice_volume(files, generated)
    evidence["status"] = "converted"
    return generated, evidence
