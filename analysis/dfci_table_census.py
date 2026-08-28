#!/usr/bin/env python3
"""Read-only DFCI delivered-dose census using the production estimator."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from rtpipeline.organize import (
    _calculate_delivery_summary,
    _delivery_reference_audit,
    _plan_evidence,
)


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


def main(output_root: Path, data_dir: Path) -> dict:
    fractions = pd.read_excel(data_dir / "fractions.xlsx")
    plans_table = pd.read_excel(data_dir / "plans.xlsx")
    plans_by_patient: dict[str, list[Path]] = defaultdict(list)
    records_by_patient: dict[str, list[Path]] = defaultdict(list)
    plan_read_failures = []

    for _, row in plans_table.iterrows():
        path = Path(str(row.get("file_path") or ""))
        patient_id = str(row.get("patient_id") or "")
        if path.is_file() and patient_id:
            plans_by_patient[patient_id].append(path)

    plan_uid_paths: dict[str, dict[str, Path]] = {}
    for patient_id, paths in plans_by_patient.items():
        mapping: dict[str, Path] = {}
        for path in dict.fromkeys(paths):
            try:
                uid = str(_plan_evidence(path).get("sop_uid") or "")
            except Exception as exc:
                plan_read_failures.append({"path": str(path), "error": str(exc)})
                continue
            if uid:
                mapping.setdefault(uid, path)
        plan_uid_paths[patient_id] = mapping
        plans_by_patient[patient_id] = list(mapping.values())

    for _, row in fractions.iterrows():
        path = Path(str(row.get("file_path") or ""))
        patient_id = str(row.get("patient_id") or "")
        if path.is_file() and patient_id:
            records_by_patient[patient_id].append(path)
    for patient_id in records_by_patient:
        records_by_patient[patient_id] = list(dict.fromkeys(records_by_patient[patient_id]))

    patient_audits = {}
    for patient_id, record_paths in records_by_patient.items():
        patient_audits[patient_id] = _delivery_reference_audit(
            record_paths,
            plan_uid_paths.get(patient_id, {}),
            log_warnings=False,
        )

    courses = []
    for metadata_path in sorted(output_root.glob("*/*/metadata/case_metadata.json")):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        patient_id = str(data.get("patient_id") or metadata_path.parts[-4])
        course_id = str(data.get("course_id") or metadata_path.parts[-3])
        selected_uids = [str(uid) for uid in data.get("source_plan_uids", []) if str(uid)]
        if not selected_uids and data.get("plan_sop_uid"):
            selected_uids = [str(data["plan_sop_uid"])]
        selected_paths = [
            plan_uid_paths.get(patient_id, {})[uid]
            for uid in selected_uids
            if uid in plan_uid_paths.get(patient_id, {})
        ]
        summary = _calculate_delivery_summary(
            plans_by_patient.get(patient_id, []),
            records_by_patient.get(patient_id, []),
            selected_plan_paths=selected_paths,
            reference_audit=patient_audits.get(patient_id),
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
    absent_uids = sorted(
        {
            uid
            for audit in patient_audits.values()
            for uid in audit["unresolved_plan_uids"]
        }
    )
    result = {
        "source_tables": {
            "fractions": str(data_dir / "fractions.xlsx"),
            "plans": str(data_dir / "plans.xlsx"),
            "course_metadata_glob": str(output_root / "*/*/metadata/case_metadata.json"),
        },
        "course_count": len(courses),
        "patient_count": len({row["patient_id"] for row in courses}),
        "rtrecord_row_count": len(fractions),
        "rtrecord_path_count": sum(len(paths) for paths in records_by_patient.values()),
        "exported_plan_count": sum(len(paths) for paths in plans_by_patient.values()),
        "plan_read_failure_count": len(plan_read_failures),
        "plan_read_failures": plan_read_failures[:20],
        "record_referenced_absent_plan_uid_count": len(absent_uids),
        "record_referenced_absent_plan_record_count": sum(
            int(audit["unresolved_record_count"]) for audit in patient_audits.values()
        ),
        "record_referenced_absent_plan_uids": absent_uids,
        "partial_plan_course_count": len(partial_courses),
        "delivery_status_counts": dict(Counter(row["delivery_status"] for row in courses)),
        "delivered_dose_known_course_count": sum(row["delivered_dose_gy"] is not None for row in courses),
        "delivered_dose_unknown_course_count": sum(row["delivered_dose_gy"] is None for row in courses),
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.output_root, args.data_dir), indent=2, sort_keys=True))
