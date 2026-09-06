"""Identity-complete CT source inventory, independent of feature eligibility.

Keep the historical name-level analysis denominator and publish a separate
source-identity ledger. A source file and declaration ordinal disambiguate even
malformed duplicate names/numbers and reused SOPInstanceUIDs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import math
import pydicom
import numpy as np
from .roi_requiredness import ROIObservation, _atomic_json
from .rtstruct_geometry import contour_geometry, roi_geometry_code


def source_observations(path, dataset=None):
    """Classify each declaration once, without materializing a voxel mask."""
    ds = dataset if dataset is not None else pydicom.dcmread(str(path), stop_before_pixels=True)
    declared = list(getattr(ds, "StructureSetROISequence", ()) or ())
    def number(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    identities = [(number(getattr(x, "ROINumber", None)), str(getattr(x, "ROIName", "") or "").strip()) for x in declared]
    numbers, names = Counter(x[0] for x in identities), Counter(x[1] for x in identities)
    items = {}
    for item in getattr(ds, "ROIContourSequence", ()) or ():
        items.setdefault(number(getattr(item, "ReferencedROINumber", None)), []).append(item)
    result = []
    for roi_number, name in identities:
        contour_items = items.get(roi_number, [])
        sequence = [contour for item in contour_items for contour in (getattr(item, "ContourSequence", ()) or ())]
        types, valid_count, invalid_count = [], 0, 0
        for contour in sequence:
            kind = str(getattr(contour, "ContourGeometricType", ""))
            types.append(kind)
            # The same strict validator governs mask preparation and inventory.
            # Counts alone cannot prove planarity or non-collinearity.
            _, valid = contour_geometry(contour)
            valid_count += int(valid)
            invalid_count += int(not valid)
        if not name or roi_number is None or numbers[roi_number] > 1 or names[name] > 1:
            code = "ROI_MALFORMED_IDENTITY"
        elif not contour_items:
            code = "ROI_DECLARED_NO_CONTOUR_ITEM"
        elif not sequence:
            code = "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE"
        elif invalid_count:
            code = "ROI_CONTOUR_PARTIALLY_UNPARSEABLE" if valid_count else "ROI_CONTOUR_UNPARSEABLE"
        else:
            code = roi_geometry_code(sequence)
        result.append(ROIObservation(roi_number, name, structural_code=code,
            valid_contours=valid_count, invalid_contours=invalid_count,
            contour_item_present=bool(contour_items),
            contour_sequence_present=any(hasattr(item, "ContourSequence") for item in contour_items)))
    return result


def structural_disposition(code):
    if code.startswith("ROI_DECLARED_"):
        return "declared_empty"
    if code.startswith("ROI_NONVOLUMETRIC_"):
        return "non_volumetric"
    return "malformed_source_roi"


def source_ledger_rows(sources, tasks, rows, *, datasets=None):
    """One terminal row per declaration, preserving original source identity.

    Unselected inherited declarations in an RS_custom object are explicitly
    outside its configured custom-only extraction scope, not missing anatomy.
    Missing outcomes of scheduled tasks remain technical failures.
    """
    output = []
    expected = set()
    for path in sorted({Path(p) for p in sources}):
        ds = datasets[path] if datasets is not None else pydicom.dcmread(str(path), stop_before_pixels=True)
        observations = source_observations(path, dataset=ds)
        expected.update((str(path), i) for i in range(len(getattr(ds, "StructureSetROISequence", ()))))
        file_tasks = [t for t in tasks if Path(t.rs_path) == path]
        for ordinal, observation in enumerate(observations):
            task = next((t for t in file_tasks
                         if (t.stable_roi_identifier == f"rtstruct_declaration:{ordinal}" or
                             (t.roi_name == observation.name and
                              t.stable_roi_identifier == f"rtstruct_roi_number:{observation.roi_number}"))), None)
            matched = [r for r in rows if task is not None and
                       r.get("segmentation_source") == task.source and
                       r.get("mask_identity") == task.mask_identity and
                       r.get("stable_roi_identifier") == task.stable_roi_identifier and
                       r.get("roi_original_name", r.get("roi_name")) == task.roi_name]
            if observation.structural_code:
                reason = observation.structural_code
                disposition = structural_disposition(reason)
            elif matched:
                failures = [r for r in matched if r.get("extraction_status") != "success"]
                selected = failures[0] if failures else matched[0]
                disposition = str(selected.get("extraction_status", "failed"))
                reason = str(selected.get("roi_structural_code") or disposition)
                if disposition == "success":
                    reason = disposition = "extracted"
                elif disposition == "declared_skip":
                    reason = "CONFIGURED_SKIP"
            elif task is not None:
                reason, disposition = "ROI_EXTRACTION_FAILED", "failed"
            elif path.name == "RS_custom.dcm":
                reason, disposition = "CONFIGURED_SOURCE_SCOPE_SKIP", "declared_skip"
            else:
                # Never misrepresent absent task construction as configured skip.
                reason, disposition = "ROI_EXTRACTION_FAILED", "failed"
            output.append({
                "source_file": path.name, "source_path": str(path),
                "source_sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
                "declaration_ordinal": ordinal, "roi_number": observation.roi_number,
                "roi_name": observation.name, "reason_code": reason,
                "disposition": disposition,
                "arm_dispositions": {r["extraction_arm"]: r.get("extraction_status") for r in matched},
            })
    identities = {(r["source_path"], r["declaration_ordinal"]) for r in output}
    assert len(identities) == len(output), "source declarations were collapsed"
    if expected != identities:
        raise ValueError(f"Source ledger mismatch: missing={sorted(expected-identities)}, extra={sorted(identities-expected)}")
    return output


def write_source_ledger(course_dir, tasks, rows):
    sources = {Path(t.rs_path) for t in tasks
               if getattr(t, "rs_path", None) and Path(t.rs_path).is_file()}
    custom = Path(course_dir) / "RS_custom.dcm"
    if custom.is_file():
        sources.add(custom)
    result = source_ledger_rows(sources, tasks, rows)
    _atomic_json(Path(course_dir) / "metadata" / "radiomics_ct_source_roi_ledger.json", result)
    _atomic_json(Path(course_dir) / "metadata" / "radiomics_ct_source_denominators.json", {
        "source_roi_count": len(result),
        "terminal_disposition_counts": dict(Counter(r["disposition"] for r in result)),
        "reason_code_counts": dict(Counter(r["reason_code"] for r in result)),
        "missing_source_identities": [], "extra_source_identities": [],
    })
    return result
