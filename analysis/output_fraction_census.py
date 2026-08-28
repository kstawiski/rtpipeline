#!/usr/bin/env python3
"""Read-only delivered-dose census using the production estimator."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from rtpipeline.organize import _calculate_delivery_summary, _plan_evidence


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def clean(value):
    if value is None:
        return None
    value = round(float(value), 6)
    return 0.0 if abs(value) < 1e-9 else value


def source_record_paths(course_dir: Path) -> list[Path]:
    try:
        table = pd.read_excel(course_dir / "fractions.xlsx")
    except Exception:
        return []
    paths: list[Path] = []
    for text in table.get("source_paths_all", pd.Series(dtype=str)).dropna():
        for value in str(text).split(";"):
            path = Path(value)
            if "RTRECORD" in value.upper() and path.is_file():
                paths.append(path)
    return list(dict.fromkeys(paths))


def plan_paths(course_dir: Path) -> list[Path]:
    paths = [path for path in (course_dir / "DICOM" / "RTPLAN").rglob("*") if path.is_file()]
    root_plan = course_dir / "RP.dcm"
    if root_plan.is_file():
        paths.append(root_plan)
    return list(dict.fromkeys(paths))


def main(output_root: Path) -> dict:
    metadata_paths = sorted(output_root.glob("*/*/metadata/case_metadata.json"))
    plans_by_patient: dict[str, list[Path]] = defaultdict(list)
    records_by_patient: dict[str, list[Path]] = defaultdict(list)
    metadata_rows: list[tuple[Path, dict]] = []

    for metadata_path in metadata_paths:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        patient_id = str(data.get("patient_id") or metadata_path.parts[-4])
        course_dir = metadata_path.parents[1]
        plans_by_patient[patient_id].extend(plan_paths(course_dir))
        records_by_patient[patient_id].extend(source_record_paths(course_dir))
        metadata_rows.append((metadata_path, data))

    plan_uid_paths: dict[str, dict[str, Path]] = {}
    for patient_id, paths in plans_by_patient.items():
        mapping: dict[str, Path] = {}
        for path in dict.fromkeys(paths):
            uid = str(_plan_evidence(path).get("sop_uid") or "")
            if uid:
                mapping.setdefault(uid, path)
        plan_uid_paths[patient_id] = mapping
        plans_by_patient[patient_id] = list(mapping.values())
        records_by_patient[patient_id] = list(dict.fromkeys(records_by_patient[patient_id]))

    courses = []
    for metadata_path, data in metadata_rows:
        patient_id = str(data.get("patient_id") or metadata_path.parts[-4])
        course_id = str(data.get("course_id") or metadata_path.parts[-3])
        selected_uids = [str(uid) for uid in data.get("source_plan_uids", []) if str(uid)]
        if not selected_uids and data.get("plan_sop_uid"):
            selected_uids = [str(data["plan_sop_uid"])]
        selected_paths = [
            plan_uid_paths[patient_id][uid]
            for uid in selected_uids
            if uid in plan_uid_paths[patient_id]
        ]
        summary = _calculate_delivery_summary(
            plans_by_patient[patient_id],
            records_by_patient[patient_id],
            selected_plan_paths=selected_paths,
        )
        reported = number(data.get("total_prescription_gy"))
        delivered = number(summary["delivered_dose_gy"])
        difference = reported - delivered if reported is not None and delivered is not None else None
        details = summary["delivery_plan_details"]
        partial = any(
            int(detail.get("delivered_fraction_count") or 0) > 0
            and int(detail.get("planned_fraction_count") or 0)
            > int(detail.get("delivered_fraction_count") or 0)
            for detail in details
        )
        courses.append(
            {
                "patient_id": patient_id,
                "course_id": course_id,
                "metadata_path": str(metadata_path),
                "reported_prescription_gy": reported,
                "delivered_dose_gy": delivered,
                "reported_minus_delivered_gy": difference,
                "delivery_status": summary["delivery_status"],
                "delivery_method": summary["delivery_method"],
                "has_partially_delivered_plan": partial,
                "delivery_warnings": summary["delivery_warnings"],
                "plan_details": details,
            }
        )

    differences = [
        row["reported_minus_delivered_gy"]
        for row in courses
        if row["reported_minus_delivered_gy"] is not None
    ]
    partial_courses = [row for row in courses if row["has_partially_delivered_plan"]]
    worst = sorted(
        [row for row in courses if row["reported_minus_delivered_gy"] is not None],
        key=lambda row: row["reported_minus_delivered_gy"],
        reverse=True,
    )[:10]
    result = {
        "output_root": str(output_root),
        "course_count": len(courses),
        "patient_count": len({row["patient_id"] for row in courses}),
        "partial_plan_course_count": len(partial_courses),
        "delivered_dose_known_course_count": sum(row["delivered_dose_gy"] is not None for row in courses),
        "delivered_dose_unknown_course_count": sum(row["delivered_dose_gy"] is None for row in courses),
        "delivery_status_counts": dict(Counter(row["delivery_status"] for row in courses)),
        "reported_minus_delivered_gy": {
            "n": len(differences),
            "min": clean(min(differences)) if differences else None,
            "p25": clean(percentile(differences, 0.25)),
            "median": clean(statistics.median(differences)) if differences else None,
            "p75": clean(percentile(differences, 0.75)),
            "max": clean(max(differences)) if differences else None,
            "mean": clean(statistics.mean(differences)) if differences else None,
        },
        "worst_cases": worst,
        "partial_case_examples": partial_courses[:30],
        "courses": courses,
        "calculation": (
            "Every course was evaluated by rtpipeline.organize._calculate_delivery_summary. "
            "The same production function binds records to selected plans through DICOM UIDs, "
            "de-duplicates treatment sessions and beam/application components, validates target "
            "dose references, and applies the fail-closed rule when any selected plan is not estimable."
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.output_root), indent=2, sort_keys=True))
