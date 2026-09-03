#!/usr/bin/env python3
"""Audit course-level dose completeness from read-only campaign outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, cast

import pydicom
import pandas as pd

from rtpipeline.course_contract import classify_course_dose_completeness
from rtpipeline.dvh import _is_target_structure, summarize_plan_isocenter_positions
from rtpipeline.organize import (
    _calculate_delivery_summary,
    _classify_doses,
    _extract_dose_metadata,
    _plan_evidence,
    _validated_dose_grid_geometry,
)


def _parse_cohort(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("cohort must be NAME=/absolute/output/root")
    name, path = raw.split("=", 1)
    root = Path(path)
    if not name or not root.is_absolute():
        raise argparse.ArgumentTypeError("cohort must have a name and absolute root")
    return name, root


def _parse_course(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("course selector must be COHORT=PATIENT/COURSE")
    cohort, key = raw.split("=", 1)
    if key.count("/") != 1:
        raise argparse.ArgumentTypeError("course key must be PATIENT/COURSE")
    return cohort, key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(course_dir: Path, value: object) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [course_dir / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _course_contract(course_dir: Path) -> tuple[dict[str, Any], Path]:
    metadata_path = course_dir / "metadata" / "case_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = payload.get("course_contract") or payload
    if not isinstance(contract, dict):
        raise ValueError(f"course contract is not an object: {metadata_path}")
    return contract, metadata_path


def _spatial_mapping_evidence(
    course_dir: Path, selected_doses: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    paths: list[Path] = []
    frames: set[str] = set()
    geometry: list[dict[str, object]] = []
    errors: list[str] = []
    for item in selected_doses:
        path = _resolve(course_dir, item.get("path"))
        if path is None or not path.is_file():
            errors.append(f"missing:{path}")
            continue
        paths.append(path)
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            current = _validated_dose_grid_geometry(dataset, path)
        except Exception as exc:
            errors.append(f"unreadable_or_invalid:{path}:{type(exc).__name__}:{exc}")
            continue
        frame = str(getattr(dataset, "FrameOfReferenceUID", "") or "").strip()
        if not frame:
            errors.append(f"missing_frame_of_reference_uid:{path}")
        else:
            frames.add(frame)
        geometry.append(
            {
                "path": str(path),
                "frame_of_reference_uid": frame or None,
                "shape": [
                    int(cast(int, current["frames"])),
                    int(cast(int, current["rows"])),
                    int(cast(int, current["cols"])),
                ],
                "spacing_mm": [
                    float(value) for value in cast(Iterable[float], current["spacing"])
                ],
            }
        )
    valid = bool(paths and len(geometry) == len(paths) and not errors and len(frames) == 1)
    return {
        "status": "validated" if valid else "not_validated",
        "method": "finite_invertible_patient_coordinate_grids_with_common_frame_of_reference",
        "source_count": len(paths),
        "common_frame_of_reference": len(frames) == 1,
        "geometry": geometry,
        "errors": errors,
    }


def _delivery_record_evidence(
    course_dir: Path,
    selected_plan_uids: set[str],
    per_plan: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    record_paths: set[Path] = set()
    declared_paths = 0
    for item in per_plan:
        if str(item.get("plan_sop_uid") or "").strip() not in selected_plan_uids:
            continue
        for raw in item.get("treatment_record_paths") or []:
            declared_paths += 1
            path = _resolve(course_dir, raw)
            if path is not None:
                record_paths.add(path)
    if not record_paths:
        record_paths.update((course_dir / "DICOM_related" / "RTRECORD").glob("*.dcm"))
    readable = 0
    plan_references: set[str] = set()
    errors: list[str] = []
    for path in sorted(record_paths):
        if not path.is_file():
            errors.append(f"missing:{path}")
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception as exc:
            errors.append(f"unreadable:{path}:{type(exc).__name__}:{exc}")
            continue
        readable += 1
        plan_references.update(
            str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
            for item in getattr(dataset, "ReferencedRTPlanSequence", []) or []
            if str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
        )
    coverage = selected_plan_uids.issubset(plan_references)
    return {
        "status": "validated" if readable and coverage and not errors else "not_validated",
        "declared_path_count": declared_paths,
        "unique_path_count": len(record_paths),
        "readable_record_count": readable,
        "referenced_plan_uids": sorted(plan_references),
        "selected_plan_uid_coverage": coverage,
        "errors": errors,
    }


def _prospective_contract_inputs(
    course_dir: Path,
    contract: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    plan_paths = sorted((course_dir / "DICOM" / "RTPLAN").glob("*.dcm"))
    dose_paths = sorted((course_dir / "DICOM" / "RTDOSE").glob("*.dcm"))
    record_paths = sorted((course_dir / "DICOM_related" / "RTRECORD").glob("*.dcm"))
    serialized_per_plan = list(
        (contract.get("delivery") or {}).get("per_plan") or []
    )
    for item in serialized_per_plan:
        plan_path = _resolve(course_dir, item.get("plan_path"))
        if plan_path is not None and plan_path.is_file():
            plan_paths.append(plan_path)
        for raw in (
            list(item.get("record_paths") or [])
            + list(item.get("treatment_record_paths") or [])
        ):
            record_path = _resolve(course_dir, raw)
            if record_path is not None and record_path.is_file():
                record_paths.append(record_path)
    plan_paths = sorted(set(plan_paths))
    record_paths = sorted(set(record_paths))
    if not plan_paths or not dose_paths:
        return (
            list(contract.get("selected_plans") or []),
            list(contract.get("selected_doses") or []),
            dict(contract.get("dose_classification") or {}),
            dict(contract.get("delivery") or {}),
        )
    classification = _classify_doses(
        plan_paths,
        dose_paths,
        treatment_record_paths=record_paths,
    )
    prospective_delivery = _calculate_delivery_summary(
        plan_paths,
        record_paths,
        selected_plan_paths=classification.selected_plans,
        selected_dose_paths=classification.selected_doses,
    )
    all_plan_delivery = _calculate_delivery_summary(
        plan_paths,
        record_paths,
        selected_plan_paths=plan_paths,
        selected_dose_paths=classification.selected_doses,
    )
    prospective_delivery["all_plan_delivery_details"] = list(
        all_plan_delivery.get("delivery_plan_details") or []
    )
    delivery = dict(contract.get("delivery") or {})
    per_plan_by_uid = {
        str(item.get("plan_sop_uid") or "").strip(): item
        for item in delivery.get("per_plan") or []
        if str(item.get("plan_sop_uid") or "").strip()
    }
    prospective_by_uid = {
        str(item.get("plan_sop_uid") or "").strip(): item
        for item in cast(
            list[dict[str, Any]],
            prospective_delivery.get("delivery_plan_details") or [],
        )
        if str(item.get("plan_sop_uid") or "").strip()
    }
    selected_plans: list[dict[str, Any]] = []
    for path in classification.selected_plans:
        evidence = _plan_evidence(path)
        plan_uid = str(evidence.get("sop_uid") or "").strip()
        delivery_item = prospective_by_uid.get(plan_uid) or per_plan_by_uid.get(
            plan_uid,
            {},
        )
        selected_plans.append(
            {
                "path": str(path),
                "sop_instance_uid": plan_uid,
                "planned_fraction_count": int(
                    evidence.get("fractions_planned") or 0
                )
                or delivery_item.get("planned_fraction_count"),
                "resolved_prescribed_dose_total_gy": evidence.get(
                    "resolved_total_rx_gy"
                ),
                "delivered_fraction_count": delivery_item.get("delivered_fraction_count"),
                "delivered_record_count": delivery_item.get("delivered_record_count"),
            }
        )
    selected_doses: list[dict[str, Any]] = []
    for path in classification.selected_doses:
        metadata = _extract_dose_metadata(path)
        selected_doses.append(
            {
                "path": str(path),
                "sop_instance_uid": str(metadata.get("sop_uid") or "").strip(),
                "dose_summation_type": str(metadata.get("summation_type") or "").strip(),
                "referenced_plan_uids": list(metadata.get("referenced_plan_uids") or []),
            }
        )
    classification_payload = {
        "classification": classification.classification,
        "reason": classification.reason,
        "selected_doses": [str(path) for path in classification.selected_doses],
        "excluded_doses": [str(path) for path in classification.excluded_doses],
        "should_sum": classification.should_sum,
        "prescription_plan_uids": [
            str(_plan_evidence(path).get("sop_uid") or "")
            for path in classification.prescription_plans
        ],
        "warnings": list(classification.warnings),
    }
    return (
        selected_plans,
        selected_doses,
        classification_payload,
        prospective_delivery,
    )


def _isocenter_evidence(
    course_dir: Path,
    selected_plans: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    positions: dict[str, list[tuple[float, float, float]]] = {}
    entries = list(selected_plans)
    for index, item in enumerate(entries):
        uid = str(item.get("sop_instance_uid") or index)
        path = _resolve(course_dir, item.get("path"))
        current: list[tuple[float, float, float]] = []
        if path is not None and path.is_file():
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                for beam in getattr(dataset, "BeamSequence", []) or []:
                    for point in getattr(beam, "ControlPointSequence", []) or []:
                        raw = getattr(point, "IsocenterPosition", None)
                        if raw is not None and len(raw) == 3:
                            current.append(
                                (float(raw[0]), float(raw[1]), float(raw[2]))
                            )
            except Exception:
                current = []
        positions[uid] = current
    summary = summarize_plan_isocenter_positions(
        positions,
        expected_plan_count=len(entries),
    )
    return {
        "status": summary.status,
        "isocenter_count": summary.isocenter_count,
        "max_separation_mm": summary.max_separation_mm,
        "plan_count": summary.plan_count,
        "readable_plan_count": summary.readable_plan_count,
        "treated_region_count": None,
    }


def _resolve_rtstruct_path(
    course_dir: Path,
    raw_value: object,
    source: str,
) -> Path | None:
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


def _near_zero_target_evidence(course_dir: Path) -> dict[str, Any]:
    workbook = course_dir / "dvh_metrics.xlsx"
    if not workbook.is_file():
        return {"status": "workbook_absent", "row_count": 0, "rows": []}
    try:
        frame = pd.read_excel(workbook)
    except Exception as exc:
        return {
            "status": "workbook_unreadable",
            "row_count": 0,
            "rows": [],
            "error": str(exc),
        }

    records: list[dict[str, object]] = []
    rtstruct_cache: dict[Path, dict[int, str]] = {}
    unreadable_rtstruct_paths: set[Path] = set()
    for _, row in frame.iterrows():
        raw_d95 = row.get("D95Gy")
        try:
            d95_gy = float(str(raw_d95)) if raw_d95 is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if pd.isna(d95_gy) or d95_gy < 0 or d95_gy > 0.1:
            continue

        roi_name = str(row.get("ROI_OriginalName") or row.get("ROI_Name") or "")
        source = str(row.get("Segmentation_Source") or "")
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

        interpreted_type = ""
        rtstruct_path = _resolve_rtstruct_path(
            course_dir,
            row.get("rtstruct_path"),
            source,
        )
        if rtstruct_path is not None and roi_number is not None:
            if (
                rtstruct_path not in rtstruct_cache
                and rtstruct_path not in unreadable_rtstruct_paths
            ):
                try:
                    dataset = pydicom.dcmread(
                        str(rtstruct_path),
                        stop_before_pixels=True,
                    )
                except Exception:
                    unreadable_rtstruct_paths.add(rtstruct_path)
                else:
                    rtstruct_cache[rtstruct_path] = _roi_interpreted_types(dataset)
            interpreted_type = rtstruct_cache.get(rtstruct_path, {}).get(
                roi_number,
                "",
            )
        if not _is_target_structure(roi_name, interpreted_type):
            continue
        records.append(
            {
                "roi_number": roi_number,
                "roi_name": roi_name,
                "roi_interpreted_type": interpreted_type or None,
                "d95_gy": d95_gy,
            }
        )

    return {
        "status": (
            "pending_plan_target_reconciliation" if records else "clear"
        ),
        "row_count": len(records),
        "rows": records,
        "minimum_d95_gy": min(
            (float(str(item["d95_gy"])) for item in records),
            default=None,
        ),
        "maximum_d95_gy": max(
            (float(str(item["d95_gy"])) for item in records),
            default=None,
        ),
    }


def _json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> int:
    logging.getLogger("rtpipeline.organize").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", type=_parse_cohort, required=True)
    parser.add_argument("--include-course", action="append", type=_parse_course, default=[])
    parser.add_argument("--exclude-course", action="append", type=_parse_course, default=[])
    parser.add_argument("--ledger-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    includes: dict[str, set[str]] = {}
    excludes: dict[str, set[str]] = {}
    for cohort, key in args.include_course:
        includes.setdefault(cohort, set()).add(key)
    for cohort, key in args.exclude_course:
        excludes.setdefault(cohort, set()).add(key)

    rows: list[dict[str, object]] = []
    for cohort, root in args.cohort:
        course_keys = {
            f"{path.parent.parent.name}/{path.parent.name}"
            for path in root.glob("*/*/dvh_metrics.xlsx")
        }
        course_keys.update(includes.get(cohort, set()))
        course_keys.difference_update(excludes.get(cohort, set()))
        for key in sorted(course_keys):
            patient_id, course_id = key.split("/", 1)
            course_dir = root / patient_id / course_id
            contract, metadata_path = _course_contract(course_dir)
            source_classification = dict(contract.get("dose_classification") or {})
            (
                selected_plans,
                selected_doses,
                prospective_classification,
                prospective_delivery,
            ) = _prospective_contract_inputs(course_dir, contract)
            delivery = dict(contract.get("delivery") or {})
            serialized_per_plan = list(delivery.get("per_plan") or [])
            prospective_per_plan = list(
                prospective_delivery.get("all_plan_delivery_details") or []
            )
            per_plan_by_uid = {
                str(item.get("plan_sop_uid") or "").strip(): dict(item)
                for item in serialized_per_plan
                if str(item.get("plan_sop_uid") or "").strip()
            }
            for item in prospective_per_plan:
                plan_uid = str(item.get("plan_sop_uid") or "").strip()
                if not plan_uid:
                    continue
                merged = dict(per_plan_by_uid.get(plan_uid) or {})
                merged.update(item)
                per_plan_by_uid[plan_uid] = merged
            per_plan = list(per_plan_by_uid.values())
            spatial = _spatial_mapping_evidence(course_dir, selected_doses)
            result = classify_course_dose_completeness(
                selected_plans=selected_plans,
                selected_doses=selected_doses,
                dose_classification=prospective_classification,
                dose_grid=contract.get("dose_grid"),
                per_plan_delivery=per_plan,
                delivery_status=str(
                    prospective_delivery.get("delivery_status") or ""
                ),
                spatial_mapping_validated=spatial["status"] == "validated",
            )
            selected_uids = {
                str(item.get("sop_instance_uid") or "").strip()
                for item in selected_plans
                if str(item.get("sop_instance_uid") or "").strip()
            }
            records = _delivery_record_evidence(course_dir, selected_uids, per_plan)
            delivered_isocenter_plans = [
                {
                    "sop_instance_uid": item.get("plan_sop_uid"),
                    "path": item.get("plan_path"),
                }
                for item in per_plan
                if int(item.get("delivered_fraction_count") or 0) > 0
            ]
            isocenters = _isocenter_evidence(
                course_dir,
                delivered_isocenter_plans or selected_plans,
            )
            near_zero = _near_zero_target_evidence(course_dir)
            if result["status"] == "eligible" and records["status"] != "validated":
                result = dict(result)
                result.update(
                    {
                        "status": "not_defensible",
                        "category": "not_defensible",
                        "reason_code": "delivery_record_artifacts_not_validated",
                        "reason": "Selected-plan RTRECORD artifacts are missing, unreadable, or do not cover every selected plan UID.",
                    }
                )
            prospective_delivery_status = str(
                prospective_delivery.get("delivery_status") or ""
            )
            other_eligibility_resolved = bool(
                prospective_delivery.get("delivered_dose_gy") is not None
                and prospective_delivery_status
                in {"fully_delivered", "partially_delivered"}
            )
            completeness_eligible = result["status"] == "eligible"
            dose_response_eligible = bool(
                completeness_eligible
                and other_eligibility_resolved
                and near_zero["status"] != "pending_plan_target_reconciliation"
            )
            plan_delivery_evidence = [
                {
                    field: item.get(field)
                    for field in (
                        "plan_sop_uid",
                        "plan_path",
                        "planned_fraction_count",
                        "delivered_record_count",
                        "delivered_fraction_count",
                        "delivery_status",
                        "status",
                        "method",
                    )
                }
                for item in per_plan
            ]
            rows.append(
                {
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "course_id": course_id,
                    "course_key": key,
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": _sha256(metadata_path),
                    "source_dose_classification": str(
                        source_classification.get("classification") or ""
                    ),
                    "prospective_dose_classification": str(
                        prospective_classification.get("classification") or ""
                    ),
                    "delivery_status": str(delivery.get("status") or ""),
                    "delivery_method": str(delivery.get("method") or ""),
                    "delivered_dose_gy": delivery.get("delivered_dose_gy"),
                    "prospective_delivery_status": prospective_delivery_status,
                    "prospective_delivery_method": str(
                        prospective_delivery.get("delivery_method") or ""
                    ),
                    "prospective_delivered_dose_gy": prospective_delivery.get(
                        "delivered_dose_gy"
                    ),
                    "selected_plan_count": len(selected_plans),
                    "selected_dose_count": len(selected_doses),
                    "selected_plan_uids": _json_cell(sorted(selected_uids)),
                    "selected_dose_summation_types": _json_cell(
                        [str(item.get("dose_summation_type") or "") for item in selected_doses]
                    ),
                    "selected_dose_referenced_plan_uids": _json_cell(
                        [item.get("referenced_plan_uids") or [] for item in selected_doses]
                    ),
                    "delivered_plan_uids": _json_cell(result.get("delivered_plan_uids", [])),
                    "unselected_delivered_plan_uids": _json_cell(
                        result.get("unselected_delivered_plan_uids", [])
                    ),
                    "dose_completeness_expected_plan_uids": _json_cell(
                        result.get("expected_plan_uids", [])
                    ),
                    "dose_completeness_represented_plan_uids": _json_cell(
                        result.get("represented_plan_uids", [])
                    ),
                    "delivered_fraction_weights": _json_cell(
                        result.get("delivered_fraction_weights", {})
                    ),
                    "per_plan_delivery_evidence": _json_cell(plan_delivery_evidence),
                    "spatial_mapping_status": spatial["status"],
                    "spatial_mapping_evidence": _json_cell(spatial),
                    "delivery_record_status": records["status"],
                    "delivery_record_evidence": _json_cell(records),
                    "isocenter_evidence": _json_cell(isocenters),
                    "treated_region_count": None,
                    "near_zero_target_status": near_zero["status"],
                    "near_zero_target_evidence": _json_cell(near_zero),
                    "dose_completeness_status": result["status"],
                    "dose_completeness_category": result["category"],
                    "dose_completeness_reason_code": result["reason_code"],
                    "dose_completeness_reason": result["reason"],
                    "dose_completeness_eligible": completeness_eligible,
                    "other_dose_response_requirements_resolved": other_eligibility_resolved,
                    "dose_response_eligible_after_contract": dose_response_eligible,
                }
            )

    fieldnames = list(rows[0]) if rows else []
    args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cohort_summaries: dict[str, object] = {}
    for cohort, _ in args.cohort:
        subset = [row for row in rows if row["cohort"] == cohort]
        cohort_summaries[cohort] = {
            "course_count": len(subset),
            "category_counts": dict(
                sorted(Counter(str(row["dose_completeness_category"]) for row in subset).items())
            ),
            "reason_code_counts": dict(
                sorted(Counter(str(row["dose_completeness_reason_code"]) for row in subset).items())
            ),
            "dose_completeness_eligible": sum(
                bool(row["dose_completeness_eligible"]) for row in subset
            ),
            "dose_response_eligible_after_contract": sum(
                bool(row["dose_response_eligible_after_contract"]) for row in subset
            ),
            "near_zero_target_quarantine_courses": sum(
                row["near_zero_target_status"]
                == "pending_plan_target_reconciliation"
                for row in subset
            ),
            "spatial_mapping_validated": sum(
                row["spatial_mapping_status"] == "validated" for row in subset
            ),
            "delivery_records_validated": sum(
                row["delivery_record_status"] == "validated" for row in subset
            ),
        }
    summary = {
        "schema_version": "task29-course-dose-completeness-v2",
        "read_only_inputs": True,
        "classification_contract": {
            "multi_plan": "one MULTI_PLAN RTDOSE with exact selected-plan SOP Instance UID coverage and full delivery",
            "single_plan": "one clinically contributing plan supported by a treatment-bearing RTRECORD session with NORMAL termination and one linked PLAN RTDOSE with a positive bounded delivered/planned fraction weight",
            "per_plan": "one PLAN RTDOSE per selected plan, positive delivered/planned fraction weight from treatment-bearing RTRECORD sessions with NORMAL termination, validated RTRECORD plan references, and validated patient-coordinate mapping",
            "not_defensible": "any unresolved course plan set, UID linkage, delivery weight, record evidence, or spatial mapping",
        },
        "evidence_surfaces": [
            "metadata/case_metadata.json course contract",
            "selected RTPLAN and RTDOSE SOP Instance UID references",
            "selected per-plan planned fraction counts",
            "treatment-bearing RTRECORD sessions with NORMAL termination",
            "copied RTRECORD DICOM ReferencedRTPlanSequence",
            "RTDOSE grid geometry and FrameOfReferenceUID",
        ],
        "cohorts": cohort_summaries,
        "combined": {
            "course_count": len(rows),
            "category_counts": dict(
                sorted(Counter(str(row["dose_completeness_category"]) for row in rows).items())
            ),
            "dose_completeness_eligible": sum(
                bool(row["dose_completeness_eligible"]) for row in rows
            ),
            "dose_response_eligible_after_contract": sum(
                bool(row["dose_response_eligible_after_contract"]) for row in rows
            ),
        },
        "ledger": str(args.ledger_out),
        "ledger_sha256": _sha256(args.ledger_out),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
