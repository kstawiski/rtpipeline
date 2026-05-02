#!/usr/bin/env python3
"""Audit scanner/protocol metadata availability for residual attribution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path("/umed-projekty/rtpipeline")
DATASET_ROOT = Path("/umed-projekty/DICOMRT-datasets")
SCANNER_TABLE = PROJECT_ROOT / "manuscript/analysis/tables/scanner_demographics.csv"

FIELD_LABELS = {
    "ct_manufacturer": "manufacturer",
    "ct_institution": "site_proxy_institution",
    "ct_slice_thickness": "slice_thickness",
    "ct_convolution_kernel": "convolution_kernel",
    "contrast_labeled": "contrast_labeled",
}


def clean_string(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null", "n/a", "na"}:
        return ""
    return text


def clean_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or number <= 0:
        return None
    return number


def has_contrast_label(meta: dict[str, object]) -> bool:
    string_fields = [
        "ct_contrast_agent",
        "ct_contrast_phase",
    ]
    numeric_fields = [
        "ct_contrast_total_volume",
    ]
    if any(clean_string(meta.get(field)) for field in string_fields):
        return True
    return any(clean_float(meta.get(field)) is not None for field in numeric_fields)


def load_cohorts(scanner_table: Path) -> list[str]:
    with scanner_table.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["cohort"].strip() for row in reader if row.get("cohort", "").strip()]


def iter_course_dirs(dataset_root: Path, cohort: str) -> Iterable[Path]:
    output_dir = dataset_root / cohort / "output"
    if not output_dir.exists():
        return
    for patient_dir in sorted(output_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("_"):
            continue
        for course_dir in sorted(patient_dir.iterdir()):
            if course_dir.is_dir() and not course_dir.name.startswith("_"):
                yield course_dir


def summarize_counter(counter: Counter[str], limit: int = 5) -> str:
    if not counter:
        return ""
    parts: list[str] = []
    for value, count in counter.most_common(limit):
        parts.append(f"{value} ({count})")
    if len(counter) > limit:
        parts.append(f"... +{len(counter) - limit} more")
    return "; ".join(parts)


def render_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    body = [
        " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def build_course_record(cohort: str, course_dir: Path) -> dict[str, object]:
    metadata_path = course_dir / "metadata" / "case_metadata.json"
    robustness_path = course_dir / "radiomics_robustness_ct.parquet"
    nifti_metadata_paths = sorted((course_dir / "NIFTI").glob("*.metadata.json"))

    record: dict[str, object] = {
        "cohort": cohort,
        "patient_dir": course_dir.parent.name,
        "course_dir": course_dir.name,
        "course_path": str(course_dir),
        "metadata_path": str(metadata_path),
        "robustness_path": str(robustness_path),
        "has_case_metadata": metadata_path.exists(),
        "has_robustness_parquet": robustness_path.exists(),
        "has_nifti_metadata": bool(nifti_metadata_paths),
        "patient_id_meta": "",
        "course_id_meta": "",
        "ct_manufacturer": "",
        "ct_model": "",
        "ct_institution": "",
        "ct_slice_thickness": None,
        "ct_slice_increment": None,
        "ct_convolution_kernel": "",
        "ct_contrast_agent": "",
        "ct_contrast_phase": "",
        "ct_kvp": None,
        "ct_study_uid": "",
        "contrast_labeled": False,
    }
    if not metadata_path.exists():
        return record

    meta = json.loads(metadata_path.read_text())
    record.update(
        {
            "patient_id_meta": clean_string(meta.get("patient_id")),
            "course_id_meta": clean_string(meta.get("course_id")),
            "ct_manufacturer": clean_string(meta.get("ct_manufacturer")),
            "ct_model": clean_string(meta.get("ct_model")),
            "ct_institution": clean_string(meta.get("ct_institution")),
            "ct_slice_thickness": clean_float(meta.get("ct_slice_thickness")),
            "ct_slice_increment": clean_float(meta.get("ct_slice_increment")),
            "ct_convolution_kernel": clean_string(meta.get("ct_convolution_kernel")),
            "ct_contrast_agent": clean_string(meta.get("ct_contrast_agent")),
            "ct_contrast_phase": clean_string(meta.get("ct_contrast_phase")),
            "ct_kvp": clean_float(meta.get("ct_kvp")),
            "ct_study_uid": clean_string(meta.get("ct_study_uid")),
            "contrast_labeled": has_contrast_label(meta),
        }
    )
    return record


def field_present(record: dict[str, object], field: str) -> bool:
    value = record.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value is not None
    return bool(clean_string(value))


def field_value(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "yes" if value else ""
    return clean_string(value)


def summarize_cohort(records: list[dict[str, object]]) -> dict[str, object]:
    metadata_records = [r for r in records if r["has_case_metadata"]]
    joinable_records = [
        r for r in records if r["has_case_metadata"] and r["has_robustness_parquet"]
    ]
    example_records = joinable_records or metadata_records

    summary: dict[str, object] = {
        "cohort": records[0]["cohort"] if records else "",
        "course_dirs": len(records),
        "metadata_json": len(metadata_records),
        "robustness_parquet": sum(bool(r["has_robustness_parquet"]) for r in records),
        "joinable_courses": len(joinable_records),
    }

    for field in FIELD_LABELS:
        summary[f"raw_{FIELD_LABELS[field]}_present"] = sum(
            field_present(record, field) for record in metadata_records
        )
        summary[f"raw_{FIELD_LABELS[field]}_pct"] = (
            f"{100.0 * summary[f'raw_{FIELD_LABELS[field]}_present'] / len(metadata_records):.1f}%"
            if metadata_records
            else "0.0%"
        )
        summary[f"{FIELD_LABELS[field]}_present"] = sum(
            field_present(record, field) for record in joinable_records
        )
        summary[f"{FIELD_LABELS[field]}_pct"] = (
            f"{100.0 * summary[f'{FIELD_LABELS[field]}_present'] / len(joinable_records):.1f}%"
            if joinable_records
            else "0.0%"
        )

    manufacturer_counter = Counter(
        field_value(record, "ct_manufacturer")
        for record in example_records
        if field_present(record, "ct_manufacturer")
    )
    institution_counter = Counter(
        field_value(record, "ct_institution")
        for record in example_records
        if field_present(record, "ct_institution")
    )
    kernel_counter = Counter(
        field_value(record, "ct_convolution_kernel")
        for record in example_records
        if field_present(record, "ct_convolution_kernel")
    )
    slice_values = sorted(
        {
            field_value(record, "ct_slice_thickness")
            for record in example_records
            if field_present(record, "ct_slice_thickness")
        }
    )

    summary["unique_manufacturers"] = len(manufacturer_counter)
    summary["unique_site_proxy"] = len(institution_counter)
    summary["unique_kernels"] = len(kernel_counter)
    summary["unique_slice_thickness"] = len(slice_values)
    summary["manufacturer_examples"] = summarize_counter(manufacturer_counter, limit=4)
    summary["site_proxy_examples"] = summarize_counter(institution_counter, limit=4)
    summary["kernel_examples"] = summarize_counter(kernel_counter, limit=4)
    summary["slice_examples"] = ", ".join(slice_values[:6])
    return summary


def summarize_field_diversity(
    cohort_rows: list[dict[str, object]], records_by_cohort: dict[str, list[dict[str, object]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field, label in FIELD_LABELS.items():
        cohorts_with_any = 0
        cohorts_with_within_cohort_variation = 0
        total_nonempty = 0
        overall_counter: Counter[str] = Counter()
        for cohort_row in cohort_rows:
            cohort = str(cohort_row["cohort"])
            joinable_records = [
                record
                for record in records_by_cohort[cohort]
                if record["has_case_metadata"] and record["has_robustness_parquet"]
            ]
            sample_value = joinable_records[0].get(field) if joinable_records else None
            if isinstance(sample_value, bool):
                values = ["yes" if bool(record.get(field)) else "no" for record in joinable_records]
            else:
                values = [
                    field_value(record, field)
                    for record in joinable_records
                    if field_present(record, field)
                ]
            if values:
                cohorts_with_any += 1
                total_nonempty += len(values)
                overall_counter.update(values)
                if len(set(values)) > 1:
                    cohorts_with_within_cohort_variation += 1
        rows.append(
            {
                "field": label,
                "cohorts_with_any": cohorts_with_any,
                "cohorts_with_within_cohort_variation": cohorts_with_within_cohort_variation,
                "overall_nonempty_joinable_rows": total_nonempty,
                "top_values": summarize_counter(overall_counter, limit=6),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scanner-table",
        default=str(SCANNER_TABLE),
        help="Path to scanner_demographics.csv used to define the manuscript cohorts.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DATASET_ROOT),
        help="Root containing per-cohort output directories.",
    )
    args = parser.parse_args()

    scanner_table = Path(args.scanner_table)
    dataset_root = Path(args.dataset_root)
    cohorts = load_cohorts(scanner_table)

    records_by_cohort: dict[str, list[dict[str, object]]] = {}
    for cohort in cohorts:
        records_by_cohort[cohort] = [
            build_course_record(cohort, course_dir)
            for course_dir in iter_course_dirs(dataset_root, cohort)
        ]

    cohort_rows = [summarize_cohort(records_by_cohort[cohort]) for cohort in cohorts]
    field_rows = summarize_field_diversity(cohort_rows, records_by_cohort)

    total_course_dirs = sum(int(row["course_dirs"]) for row in cohort_rows)
    total_metadata = sum(int(row["metadata_json"]) for row in cohort_rows)
    total_robustness = sum(int(row["robustness_parquet"]) for row in cohort_rows)
    total_joinable = sum(int(row["joinable_courses"]) for row in cohort_rows)

    print("Scanner/protocol residual-attribution audit")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"scanner_table: {scanner_table}")
    print(f"dataset_root: {dataset_root}")
    print(
        "candidate_course_metadata_glob: "
        f"{dataset_root}/<cohort>/output/*/*/metadata/case_metadata.json"
    )
    print(
        "candidate_join_glob: "
        f"{dataset_root}/<cohort>/output/*/*/radiomics_robustness_ct.parquet"
    )
    print()
    print(
        f"totals: course_dirs={total_course_dirs}, metadata_json={total_metadata}, "
        f"robustness_parquet={total_robustness}, joinable_courses={total_joinable}"
    )
    print()
    print("Per-cohort raw metadata coverage")
    print(
        render_table(
            cohort_rows,
            [
                "cohort",
                "course_dirs",
                "metadata_json",
                "raw_manufacturer_pct",
                "raw_site_proxy_institution_pct",
                "raw_slice_thickness_pct",
                "raw_convolution_kernel_pct",
                "raw_contrast_labeled_pct",
            ],
        )
    )
    print()
    print("Per-cohort joinable coverage")
    print(
        render_table(
            cohort_rows,
            [
                "cohort",
                "course_dirs",
                "metadata_json",
                "robustness_parquet",
                "joinable_courses",
                "manufacturer_pct",
                "site_proxy_institution_pct",
                "slice_thickness_pct",
                "convolution_kernel_pct",
                "contrast_labeled_pct",
                "unique_manufacturers",
                "unique_site_proxy",
                "unique_slice_thickness",
                "unique_kernels",
            ],
        )
    )
    print()
    print("Per-cohort value examples")
    print(
        render_table(
            cohort_rows,
            [
                "cohort",
                "manufacturer_examples",
                "site_proxy_examples",
                "slice_examples",
                "kernel_examples",
            ],
        )
    )
    print()
    print("Field diversity across joinable courses")
    print(
        render_table(
            field_rows,
            [
                "field",
                "cohorts_with_any",
                "cohorts_with_within_cohort_variation",
                "overall_nonempty_joinable_rows",
                "top_values",
            ],
        )
    )


if __name__ == "__main__":
    main()
