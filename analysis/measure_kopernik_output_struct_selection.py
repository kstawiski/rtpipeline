#!/usr/bin/env python3
"""Compare copied Kopernik RTSTRUCTs with output RTPLAN references read-only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset


def all_refs(ds: Dataset, sequence_name: str) -> set[str]:
    return {
        str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
        for item in getattr(ds, sequence_name, []) or []
        if getattr(item, "ReferencedSOPInstanceUID", None)
    }


def roi_summary(path: Path) -> tuple[str, int, int]:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    names = [
        str(getattr(item, "ROIName", "") or "")
        for item in getattr(ds, "StructureSetROISequence", []) or []
    ]
    targets = sum(any(token in name.upper() for token in ("GTV", "CTV", "PTV")) for name in names)
    return str(getattr(ds, "SOPInstanceUID", "") or ""), len(names), targets


def source_structs(root: Path) -> dict[tuple[str, str], tuple[int, int]]:
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for patient in sorted(path for path in root.iterdir() if path.is_dir()):
        for base, dirs, files in os.walk(patient, followlinks=True):
            dirs.sort()
            for name in sorted(files):
                upper = name.upper()
                if not (upper.startswith("RS.") or upper.startswith("RTSTRUCT")):
                    continue
                path = Path(base) / name
                try:
                    uid, rois, targets = roi_summary(path)
                    ds = pydicom.dcmread(
                        str(path),
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=["PatientID"],
                    )
                except Exception:
                    continue
                patient_id = str(getattr(ds, "PatientID", "") or patient.name)
                if uid:
                    result[(patient_id, uid)] = (rois, targets)
    return result


def course_files(course: Path, modality: str) -> list[Path]:
    paths: list[Path] = []
    modality_dir = course / "DICOM" / modality
    if modality_dir.is_dir():
        paths.extend(sorted(path for path in modality_dir.iterdir() if path.is_file()))
    if modality == "RTSTRUCT":
        root_struct = course / "RS.dcm"
        if root_struct.is_file():
            paths.append(root_struct)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    structs = source_structs(args.source)
    rows: list[dict] = []
    for patient_dir in sorted(path for path in args.output.iterdir() if path.is_dir() and not path.name.startswith("_")):
        patient_id = patient_dir.name
        for course in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
            plan_refs: set[str] = set()
            for path in course_files(course, "RTPLAN"):
                try:
                    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                except Exception:
                    continue
                plan_refs.update(all_refs(ds, "ReferencedStructureSetSequence"))
            selected: list[tuple[str, int, int]] = []
            for path in course_files(course, "RTSTRUCT"):
                try:
                    selected.append(roi_summary(path))
                except Exception:
                    continue
            selected = list(dict.fromkeys(selected))
            selected_uid = selected[0][0] if len(selected) == 1 else ""
            selected_rois = selected[0][1] if len(selected) == 1 else 0
            selected_targets = selected[0][2] if len(selected) == 1 else 0
            authoritative = [structs[(patient_id, uid)] for uid in sorted(plan_refs) if (patient_id, uid) in structs]
            max_authoritative_rois = max((item[0] for item in authoritative), default=0)
            max_authoritative_targets = max((item[1] for item in authoritative), default=0)
            rows.append(
                {
                    "patient_id": patient_id,
                    "course": course.name,
                    "plan_referenced_struct_uids": sorted(plan_refs),
                    "selected_struct_uid": selected_uid,
                    "selected_roi_count": selected_rois,
                    "selected_target_count": selected_targets,
                    "max_authoritative_roi_count": max_authoritative_rois,
                    "max_authoritative_target_count": max_authoritative_targets,
                    "selection_matches_plan_reference": bool(selected_uid and selected_uid in plan_refs),
                }
            )

    comparable = [row for row in rows if row["plan_referenced_struct_uids"]]
    mismatches = [row for row in comparable if not row["selection_matches_plan_reference"]]
    result = {
        "source": str(args.source),
        "output": str(args.output),
        "access": "read-only",
        "output_courses": len(rows),
        "courses_with_plan_structure_reference": len(comparable),
        "correct_structure_selection": sum(row["selection_matches_plan_reference"] for row in comparable),
        "wrong_or_missing_structure_selection": len(mismatches),
        "mismatches_where_authoritative_has_more_targets": sum(
            row["max_authoritative_target_count"] > row["selected_target_count"] for row in mismatches
        ),
        "mismatch_examples": mismatches[:10],
        "worked_examples": [
            row for row in rows if row["patient_id"] in {"292929", "333944", "481077"}
        ],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
