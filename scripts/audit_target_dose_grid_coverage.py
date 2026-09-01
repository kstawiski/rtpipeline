#!/usr/bin/env python3
"""Read-only audit of near-zero target geometry against selected RTDOSE grids."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

from rtpipeline.dvh import _is_target_structure, classify_zero_dose_roi_geometry

NEAR_ZERO_TARGET_D95_GY = 0.1

STATUS_TO_CLASS = {
    "zero_dose_outside_dose_grid": "entirely_outside_grid",
    "zero_dose_partly_inside_dose_grid": "partly_inside_grid",
    "zero_dose_in_grid": "inside_grid_near_zero_selected_dose",
    "zero_dose_geometry_unresolved": "not_decidable",
}


def _parse_cohort(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Cohorts must use NAME=/absolute/output/path")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not name.strip() or not path.is_absolute():
        raise argparse.ArgumentTypeError("Cohorts require a name and absolute path")
    return name.strip(), path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _course_workbooks(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Cohort output root does not exist: {root}")
    workbooks = sorted(root.glob("*/*/dvh_metrics.xlsx"))
    if not workbooks:
        raise RuntimeError(f"No per-course dvh_metrics.xlsx files found below {root}")
    return workbooks


def _resolve_rtstruct_path(course_dir: Path, raw_value: object, source: str) -> Path | None:
    raw = str(raw_value or "").strip()
    candidates: list[Path] = []
    if raw and raw.lower() not in {"nan", "none", "<na>"}:
        supplied = Path(raw)
        candidates.extend([supplied, course_dir / supplied, course_dir / supplied.name])
    source_upper = source.upper()
    if "ORIGINAL" in source_upper or "MANUAL" in source_upper:
        candidates.append(course_dir / "RS.dcm")
    elif "CUSTOM" in source_upper:
        candidates.append(course_dir / "RS_custom.dcm")
    elif "TOTAL" in source_upper or "AUTO" in source_upper:
        candidates.extend(
            [course_dir / "RS_auto_cropped.dcm", course_dir / "RS_auto.dcm"]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _dose_source_plan_uids(contract: object, dose: object) -> tuple[str, ...]:
    if isinstance(contract, dict):
        artifact = contract.get("dose_artifact") or contract.get("dose_grid") or {}
    else:
        artifact = getattr(contract, "dose_artifact", {}) or {}
    values: list[object] = []
    if isinstance(artifact, dict):
        values.extend(artifact.get("source_plan_uids") or [])
    if not values:
        for item in getattr(dose, "ReferencedRTPlanSequence", []) or []:
            values.append(getattr(item, "ReferencedSOPInstanceUID", ""))
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True)
class _PlanScope:
    status: str


def _classify_plan_scope(contract: object, source_plan_uids: tuple[str, ...]) -> _PlanScope:
    if isinstance(contract, dict):
        delivery = contract.get("delivery") or {}
    else:
        delivery = getattr(contract, "delivery", {}) or {}
    entries = list(delivery.get("per_plan") or []) if isinstance(delivery, dict) else []
    expected = {
        str(item.get("plan_sop_uid") or "").strip()
        for item in entries
        if isinstance(item, dict) and str(item.get("plan_sop_uid") or "").strip()
    }
    represented = {value for value in source_plan_uids if value}
    if not expected:
        return _PlanScope("course_treatment_plan_set_unresolved")
    if represented == expected:
        return _PlanScope("complete_course_plan_set")
    if represented & expected:
        return _PlanScope("partial_course_plan_set")
    return _PlanScope("dose_plan_scope_unresolved")


def _roi_interpreted_types(rtstruct: object) -> dict[int, str]:
    interpreted: dict[int, str] = {}
    for observation in getattr(rtstruct, "RTROIObservationsSequence", []) or []:
        try:
            number = int(observation.ReferencedROINumber)
        except (AttributeError, TypeError, ValueError):
            continue
        interpreted[number] = str(
            getattr(observation, "RTROIInterpretedType", "") or ""
        ).strip()
    return interpreted


def audit_cohort(name: str, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workbooks = _course_workbooks(root)
    workbook_manifest: list[str] = []
    records: list[dict[str, Any]] = []
    total_rows = 0
    target_rows = 0
    scope_counts: Counter[str] = Counter()
    scope_by_course: dict[tuple[str, str], str] = {}
    input_issue_counts: Counter[str] = Counter()

    for workbook in workbooks:
        course_dir = workbook.parent
        patient_id = course_dir.parent.name
        course_id = course_dir.name
        frame = pd.read_excel(workbook)
        total_rows += len(frame)
        workbook_manifest.append(
            f"{workbook.relative_to(root)}\t{workbook.stat().st_size}\t{_sha256(workbook)}"
        )

        dose_path = course_dir / "RD.dcm"
        dose = None
        if not dose_path.is_file():
            input_issue_counts["selected_rtdose_missing"] += 1
        else:
            try:
                dose = pydicom.dcmread(str(dose_path), stop_before_pixels=True)
            except Exception:
                input_issue_counts["selected_rtdose_unreadable"] += 1
        metadata_path = course_dir / "metadata" / "case_metadata.json"
        contract: dict[str, Any] = {}
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_contract = metadata.get("course_contract") or metadata
            if isinstance(raw_contract, dict):
                contract = raw_contract
            else:
                input_issue_counts["course_contract_not_object"] += 1
        except (OSError, ValueError, AttributeError):
            input_issue_counts["course_contract_unreadable"] += 1
        plan_scope = _classify_plan_scope(
            contract, _dose_source_plan_uids(contract, dose)
        )
        course_key = (patient_id, course_id)
        scope_counts[plan_scope.status] += 1
        scope_by_course[course_key] = plan_scope.status

        rtstruct_cache: dict[Path, tuple[object, dict[int, str]]] = {}
        unreadable_rtstruct_paths: set[Path] = set()
        for _, row in frame.iterrows():
            roi_name = str(row.get("ROI_OriginalName") or row.get("ROI_Name") or "")
            source = str(row.get("Segmentation_Source") or "")
            rtstruct_path = _resolve_rtstruct_path(
                course_dir, row.get("rtstruct_path"), source
            )
            roi_number: int | None
            raw_roi_number = row.get("ROI_Number")
            try:
                roi_number = (
                    int(float(str(raw_roi_number)))
                    if raw_roi_number is not None
                    else None
                )
                if roi_number is not None and roi_number <= 0:
                    roi_number = None
            except (TypeError, ValueError):
                roi_number = None

            rtstruct = None
            interpreted_type = ""
            if rtstruct_path is not None and roi_number is not None:
                if (
                    rtstruct_path not in rtstruct_cache
                    and rtstruct_path not in unreadable_rtstruct_paths
                ):
                    try:
                        dataset = pydicom.dcmread(
                            str(rtstruct_path), stop_before_pixels=True
                        )
                    except Exception:
                        input_issue_counts["referenced_rtstruct_unreadable"] += 1
                        unreadable_rtstruct_paths.add(rtstruct_path)
                    else:
                        rtstruct_cache[rtstruct_path] = (
                            dataset,
                            _roi_interpreted_types(dataset),
                        )
                cached = rtstruct_cache.get(rtstruct_path)
                if cached is not None:
                    rtstruct, interpreted_types = cached
                    interpreted_type = interpreted_types.get(roi_number, "")
            target_like = _is_target_structure(roi_name, interpreted_type)
            if not target_like:
                continue
            target_rows += 1

            raw_d95_gy = row.get("D95Gy")
            try:
                d95_gy = (
                    float(str(raw_d95_gy)) if raw_d95_gy is not None else float("nan")
                )
            except (TypeError, ValueError):
                continue
            if pd.isna(d95_gy) or d95_gy < 0 or d95_gy > NEAR_ZERO_TARGET_D95_GY:
                continue

            if rtstruct is None or roi_number is None or dose is None:
                missing_surfaces = []
                if rtstruct is None or roi_number is None:
                    missing_surfaces.append("traceable RTSTRUCT contour geometry")
                if dose is None:
                    missing_surfaces.append("selected RTDOSE grid geometry")
                qc = {
                    "status": "zero_dose_geometry_unresolved",
                    "reason": (
                        "No " + " or ".join(missing_surfaces) + " is available for this row."
                    ),
                }
            else:
                qc = classify_zero_dose_roi_geometry(rtstruct, roi_number, dose)
            status = str(qc["status"])
            if status not in STATUS_TO_CLASS:
                raise RuntimeError(f"Unexpected geometry status {status!r} for {workbook}")
            records.append(
                {
                    "cohort": name,
                    "patient_id": patient_id,
                    "course_id": course_id,
                    "ROI_Number": roi_number,
                    "ROI_Name": roi_name,
                    "ROI_Interpreted_Type": interpreted_type or None,
                    "D95Gy": d95_gy,
                    "geometry_class": STATUS_TO_CLASS[status],
                    "zero_dose_status": status,
                    "zero_dose_reason": str(qc["reason"]),
                    "rtstruct_geometry_available": rtstruct is not None,
                    "dose_plan_scope_status": plan_scope.status,
                }
            )

    row_counts = Counter(record["geometry_class"] for record in records)
    courses_by_class: dict[str, set[tuple[str, str]]] = defaultdict(set)
    class_sets_by_course: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        key = (record["patient_id"], record["course_id"])
        geometry_class = record["geometry_class"]
        courses_by_class[geometry_class].add(key)
        class_sets_by_course[key].add(geometry_class)
    class_order = tuple(STATUS_TO_CLASS.values())
    row_count_complete = sum(row_counts.values()) == len(records)
    if not row_count_complete:
        raise RuntimeError("Near-zero target rows did not reconcile to geometry classes")
    combinations = Counter(
        "+".join(sorted(classes)) for classes in class_sets_by_course.values()
    )
    near_zero_scope_counts = Counter(
        scope_by_course[key] for key in class_sets_by_course
    )
    manifest_sha256 = hashlib.sha256(
        "\n".join(workbook_manifest).encode("utf-8")
    ).hexdigest()
    summary = {
        "cohort": name,
        "output_root": str(root),
        "input_workbook_count": len(workbooks),
        "input_row_count": total_rows,
        "target_like_row_count": target_rows,
        "near_zero_definition": f"target_like D95Gy <= {NEAR_ZERO_TARGET_D95_GY:g} Gy",
        "near_zero_target_row_count": len(records),
        "course_count_with_near_zero_target": len(class_sets_by_course),
        "target_row_count_by_class": {
            geometry_class: row_counts[geometry_class] for geometry_class in class_order
        },
        "course_count_by_class_nonexclusive": {
            geometry_class: len(courses_by_class[geometry_class])
            for geometry_class in class_order
        },
        "course_count_by_exclusive_class_combination": dict(sorted(combinations.items())),
        "all_computed_course_plan_scope_counts": dict(sorted(scope_counts.items())),
        "near_zero_target_course_plan_scope_counts": dict(
            sorted(near_zero_scope_counts.items())
        ),
        "physical_input_issue_counts": dict(sorted(input_issue_counts.items())),
        "reconciliation": {
            "classified_row_count": sum(row_counts.values()),
            "near_zero_target_row_count": len(records),
            "class_counts_are_mutually_exclusive": row_count_complete,
        },
        "workbook_manifest_sha256": manifest_sha256,
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        type=_parse_cohort,
        metavar="NAME=/absolute/output/path",
    )
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for name, root in args.cohort:
        summary, cohort_records = audit_cohort(name, root)
        summaries.append(summary)
        records.extend(cohort_records)

    combined_row_counts = Counter(record["geometry_class"] for record in records)
    class_order = tuple(STATUS_TO_CLASS.values())
    output = {
        "schema_version": "task18-target-dose-grid-audit-v1",
        "read_only_inputs": True,
        "near_zero_target_d95_threshold_gy": NEAR_ZERO_TARGET_D95_GY,
        "cohorts": summaries,
        "combined": {
            "input_workbook_count": sum(item["input_workbook_count"] for item in summaries),
            "input_row_count": sum(item["input_row_count"] for item in summaries),
            "target_like_row_count": sum(item["target_like_row_count"] for item in summaries),
            "near_zero_target_row_count": len(records),
            "target_row_count_by_class": {
                geometry_class: combined_row_counts[geometry_class]
                for geometry_class in class_order
            },
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.rows_out is not None:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.rows_out, index=False)


if __name__ == "__main__":
    main()
