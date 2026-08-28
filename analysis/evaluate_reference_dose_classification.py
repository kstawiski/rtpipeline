#!/usr/bin/env python3
"""Evaluate reference-driven dose classification on selected read-only patients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pydicom

from rtpipeline.metadata import group_by_course, link_rt_sets
from rtpipeline.organize import _classify_doses, _index_rt_files, infer_plan_rx_gy
from rtpipeline.rt_details import extract_rt, has_target_volumes


def _uid(path: Path) -> str:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return str(getattr(ds, "SOPInstanceUID", "") or "")


def _rx(path: Path) -> float:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return float(infer_plan_rx_gy(ds) or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("patients", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plans, doses, structs = extract_rt(args.root, args.patients, max_workers=4)
    courses = group_by_course(link_rt_sets(plans, doses, structs))
    rt_files = _index_rt_files(args.root, args.patients)
    result = {"patients": {}}
    for patient in args.patients:
        rows = []
        for (pid, struct_uid), items in courses.items():
            if str(pid) != patient or not items or items[0].struct is None:
                continue
            if not has_target_volumes(items[0].struct.roi_names):
                continue
            plan_paths = list(dict.fromkeys(item.plan.path for item in items))
            dose_paths = list(
                dict.fromkeys(item.dose.path for item in items if item.dose is not None)
            )
            classified = _classify_doses(
                plan_paths,
                dose_paths,
                treatment_record_paths=rt_files.get(patient, []),
            )
            rows.append(
                {
                    "struct_uid": struct_uid,
                    "source_plans": len(plan_paths),
                    "source_doses": len(dose_paths),
                    "classification": classified.classification,
                    "selected_plan_uids": [_uid(path) for path in classified.selected_plans],
                    "selected_dose_uids": [_uid(path) for path in classified.selected_doses],
                    "selected_total_rx_gy": sum(_rx(path) for path in classified.selected_plans),
                    "should_sum": classified.should_sum,
                    "warnings": classified.warnings,
                }
            )
        result["patients"][patient] = sorted(rows, key=lambda row: row["struct_uid"])
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
