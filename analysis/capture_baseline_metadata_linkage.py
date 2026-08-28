#!/usr/bin/env python3
"""Capture the pre-fix metadata linkage and grouping decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pydicom

from rtpipeline.metadata import group_by_course, link_rt_sets
from rtpipeline.rt_details import extract_rt


def _plan_struct_uid(path: Path) -> str:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    for item in getattr(ds, "ReferencedStructureSetSequence", []) or []:
        uid = str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
        if uid:
            return uid
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("patients", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plans, doses, structs = extract_rt(args.root, args.patients, max_workers=4)
    linked = link_rt_sets(plans, doses, structs)
    grouped = group_by_course(linked)
    result: dict[str, Any] = {"patients": {}}
    for patient in args.patients:
        rows = []
        for item in linked:
            if str(item.patient_id) != patient:
                continue
            assigned_names = item.struct.roi_names if item.struct else []
            rows.append(
                {
                    "plan_uid": item.plan.sop_instance_uid,
                    "dose_uid": item.dose.sop_instance_uid if item.dose is not None else "",
                    "authoritative_struct_uid": _plan_struct_uid(item.plan.path),
                    "assigned_struct_uid": item.struct.sop_instance_uid if item.struct else "",
                    "assigned_struct_roi_count": len(assigned_names),
                    "assigned_struct_target_count": sum(
                        any(token in name.upper() for token in ("GTV", "CTV", "PTV"))
                        for name in assigned_names
                    ),
                    "course_study_uid": item.ct_study_uid or "",
                    "frame_uid": item.frame_of_reference_uid or "",
                }
            )
        course_rows = [
            {"course_key": key, "linked_sets": len(items)}
            for (pid, key), items in grouped.items()
            if str(pid) == patient
        ]
        result["patients"][patient] = {
            "linked_sets": rows,
            "grouped_courses": course_rows,
        }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
