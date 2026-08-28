#!/usr/bin/env python3
"""Inventory whether existing course checkpoints need dose reprocessing.

The script reads aggregate checkpoint state only. It does not modify the output
root and does not emit patient or course identifiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _course_directories(output_root: Path) -> list[Path]:
    courses: list[Path] = []
    for patient_dir in sorted(output_root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("_"):
            continue
        for course_dir in sorted(patient_dir.iterdir()):
            if not course_dir.is_dir():
                continue
            if (course_dir / "DICOM").is_dir() or (
                course_dir / "metadata" / "case_metadata.json"
            ).is_file():
                courses.append(course_dir)
    return courses


def _read_metadata(course_dir: Path) -> tuple[dict[str, Any], bool]:
    path = course_dir / "metadata" / "case_metadata.json"
    if not path.is_file():
        return {}, False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, True
    return (value if isinstance(value, dict) else {}), not isinstance(value, dict)


def _nested_plan_files(course_dir: Path) -> list[Path]:
    plan_dir = course_dir / "DICOM" / "RTPLAN"
    if not plan_dir.is_dir():
        return []
    return sorted(path for path in plan_dir.rglob("*") if path.is_file())


def census(output_root: Path, cohort: str) -> dict[str, Any]:
    courses = _course_directories(output_root)
    totals: dict[str, Any] = {
        "total_courses": len(courses),
        "courses_with_truthy_metadata_rp_path": 0,
        "courses_with_empty_or_missing_metadata_rp_path": 0,
        "courses_with_nested_rtplan_files": 0,
        "courses_with_legacy_root_rp_dcm": 0,
        "courses_with_any_discovered_plan": 0,
        "courses_with_delivery_status": 0,
        "courses_with_planning_ct_status": 0,
        "plan_bearing_courses_missing_either_adjudication": 0,
        "previous_logic_would_reprocess": 0,
        "previous_logic_would_silently_skip": 0,
        "metadata_missing_or_malformed": 0,
        "nested_rtplan_file_count": 0,
    }

    for course_dir in courses:
        metadata, malformed = _read_metadata(course_dir)
        if malformed or not metadata:
            totals["metadata_missing_or_malformed"] += 1
        truthy_rp_path = bool(metadata.get("rp_path"))
        nested_plans = _nested_plan_files(course_dir)
        legacy_root_plan = (course_dir / "RP.dcm").is_file()
        has_discovered_plan = bool(nested_plans or legacy_root_plan or truthy_rp_path)
        has_delivery_status = "delivery_status" in metadata
        has_planning_ct_status = "planning_ct_status" in metadata
        missing_either = not has_delivery_status or not has_planning_ct_status
        old_reprocess = bool((truthy_rp_path or legacy_root_plan) and missing_either)
        silently_skipped = bool(has_discovered_plan and missing_either and not old_reprocess)

        totals["courses_with_truthy_metadata_rp_path"] += int(truthy_rp_path)
        totals["courses_with_empty_or_missing_metadata_rp_path"] += int(not truthy_rp_path)
        totals["courses_with_nested_rtplan_files"] += int(bool(nested_plans))
        totals["courses_with_legacy_root_rp_dcm"] += int(legacy_root_plan)
        totals["courses_with_any_discovered_plan"] += int(has_discovered_plan)
        totals["courses_with_delivery_status"] += int(has_delivery_status)
        totals["courses_with_planning_ct_status"] += int(has_planning_ct_status)
        totals["plan_bearing_courses_missing_either_adjudication"] += int(
            has_discovered_plan and missing_either
        )
        totals["previous_logic_would_reprocess"] += int(old_reprocess)
        totals["previous_logic_would_silently_skip"] += int(silently_skipped)
        totals["nested_rtplan_file_count"] += len(nested_plans)

    bypass = totals["previous_logic_would_silently_skip"]
    denominator = totals["total_courses"]
    totals["previous_logic_silent_skip_percentage"] = (
        round(100.0 * bypass / denominator, 1) if denominator else None
    )
    return {
        "schema_version": 1,
        "cohort": cohort,
        "source": {
            "output_root": str(output_root),
            "access_mode": "read-only",
            "course_definition": (
                "second-level non-internal directory containing DICOM or "
                "metadata/case_metadata.json"
            ),
        },
        "claim_kinds": {
            "inventory": "calculation from current output files",
            "silent_skip": (
                "calculation applying the previous resume predicate to plan-bearing "
                "checkpoints missing delivery_status or planning_ct_status"
            ),
        },
        "counts": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    print(json.dumps(census(root, args.cohort), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
