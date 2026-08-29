from __future__ import annotations

"""Reconcile organizer and DVH provenance for the read-only Kopernik cohort.

The script reads the campaign output and source RTRECORD headers. It never
modifies either source tree. It exercises the fixed DVH resolver in temporary
workspace-owned course directories populated with header-only copies.
"""

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pydicom

from rtpipeline.course_contract import (
    DOSE_GRID_SEMANTICS,
    DOSE_RESPONSE_FIELD,
    UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS,
)
from rtpipeline.dvh import _resolve_dvh_dose
from rtpipeline.layout import build_course_dirs
from rtpipeline.organize import _record_delivery_evidence


TARGET_RE = re.compile(r"^PTV(?:\s*\d+|\d*)$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _split_uids(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    for value in values:
        for uid in str(value or "").split(";"):
            uid = uid.strip()
            if uid and uid.lower() != "nan" and uid not in output:
                output.append(uid)
    return output


def _dicom_inventory(directory: Path, modality: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.is_dir():
        return rows
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "") or "").upper() != modality:
            continue
        row: dict[str, Any] = {
            "path": str(path),
            "sop_instance_uid": str(getattr(dataset, "SOPInstanceUID", "") or ""),
        }
        if modality == "RTPLAN":
            row.update(
                {
                    "plan_label": str(getattr(dataset, "RTPlanLabel", "") or ""),
                    "plan_name": str(getattr(dataset, "RTPlanName", "") or ""),
                }
            )
        elif modality == "RTDOSE":
            row.update(
                {
                    "dose_summation_type": str(
                        getattr(dataset, "DoseSummationType", "") or ""
                    ).upper(),
                    "referenced_plan_uids": [
                        str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
                        for item in getattr(dataset, "ReferencedRTPlanSequence", []) or []
                        if str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
                    ],
                }
            )
        rows.append(row)
    return rows


def _record_evidence(patient_root: Path) -> dict[str, dict[str, Any]]:
    record_paths: list[Path] = []
    if not patient_root.exists():
        return {}
    for path in sorted(item for item in patient_root.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "Modality",
                    "SOPClassUID",
                    "SOPInstanceUID",
                    "TreatmentDate",
                    "ReferencedRTPlanSequence",
                ],
            )
        except Exception:
            continue
        modality = str(getattr(dataset, "Modality", "") or "").upper()
        if "RECORD" not in modality:
            continue
        record_paths.append(path)
    evidence = _record_delivery_evidence(record_paths)
    return {
        uid: {
            "delivered_record_count": len(row.get("instances", set())),
            "delivered_fraction_count": len(row.get("sessions", set())),
            "treatment_dates": sorted(row.get("dates", set())),
            "record_paths": sorted(
                {
                    str(record.get("path") or "")
                    for record in row.get("records", [])
                    if str(record.get("path") or "")
                }
            ),
        }
        for uid, row in sorted(evidence.items())
    }


def _header_copy(source: Path, destination: Path) -> Path:
    dataset = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(destination), enforce_file_format=True)
    return destination


def _exercise_fixed_resolver(
    *,
    workspace: Path,
    patient_id: str,
    course_id: str,
    plans: list[dict[str, Any]],
    doses: list[dict[str, Any]],
    selected_plan_uids: list[str],
    selected_dose_uids: list[str],
    record_evidence: dict[str, dict[str, Any]],
    delivery_status: str,
    prescribed_dose_gy: float | None,
    delivered_dose_gy: float | None,
    threshold_gy: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="course-contract-evidence-", dir=workspace) as tmp:
        course = Path(tmp) / patient_id / course_id
        plan_paths: dict[str, Path] = {}
        dose_paths: dict[str, Path] = {}
        for index, row in enumerate(plans):
            uid = str(row["sop_instance_uid"])
            plan_paths[uid] = _header_copy(
                Path(row["path"]),
                course / "DICOM" / "RTPLAN" / f"plan_{index:03d}.dcm",
            )
        for index, row in enumerate(doses):
            uid = str(row["sop_instance_uid"])
            dose_paths[uid] = _header_copy(
                Path(row["path"]),
                course / "DICOM" / "RTDOSE" / f"dose_{index:03d}.dcm",
            )

        def relative(path: Path) -> str:
            return str(path.relative_to(course))

        record_paths: dict[str, Path] = {}
        record_index = 0
        for evidence in record_evidence.values():
            for source_value in evidence.get("record_paths") or []:
                source = Path(str(source_value))
                copied = _header_copy(
                    source,
                    course / "DICOM" / "RTRECORD" / f"record_{record_index:04d}.dcm",
                )
                record_paths[str(source)] = copied
                record_index += 1

        selected_plans = []
        per_plan = []
        for uid, path in plan_paths.items():
            evidence = record_evidence.get(uid, {})
            count = int(evidence.get("delivered_record_count") or 0)
            fraction_count = int(evidence.get("delivered_fraction_count") or 0)
            dates = list(evidence.get("treatment_dates") or [])
            if uid in selected_plan_uids:
                selected_plans.append(
                    {
                        "sop_instance_uid": uid,
                        "path": relative(path),
                        "delivered_record_count": count,
                        "delivered_fraction_count": fraction_count,
                        "treatment_dates": dates,
                    }
                )
            per_plan.append(
                {
                    "plan_path": relative(path),
                    "plan_sop_uid": uid,
                    "prescribed_dose_gy": None,
                    "planned_fraction_count": None,
                    "delivered_record_count": count,
                    "delivered_fraction_count": fraction_count,
                    "treatment_dates": dates,
                    "record_paths": [
                        relative(record_paths[source])
                        for source in evidence.get("record_paths") or []
                        if source in record_paths
                    ],
                    "zero_delivery_records": count == 0,
                    "selected_for_dose_grid": uid in selected_plan_uids,
                    "status": "no_records" if count == 0 else "partially_delivered",
                }
            )

        selected_doses = []
        dose_rows = {str(row["sop_instance_uid"]): row for row in doses}
        for uid in selected_dose_uids:
            row = dose_rows[uid]
            selected_doses.append(
                {
                    "sop_instance_uid": uid,
                    "path": relative(dose_paths[uid]),
                    "dose_summation_type": row["dose_summation_type"],
                    "referenced_plan_uids": row["referenced_plan_uids"],
                }
            )

        qc_failure = any(
            value is not None and value > threshold_gy
            for value in (prescribed_dose_gy, delivered_dose_gy)
        )
        semantics = (
            DOSE_GRID_SEMANTICS
            if delivery_status in {"fully_delivered", "partially_delivered"}
            else UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS
        )
        dose_grid = None
        if selected_doses:
            dose_grid = {
                **selected_doses[0],
                "semantics": semantics,
                "source_plan_uids": selected_plan_uids,
                "source_dose_uids": selected_dose_uids,
                "source_dose_summation_types": [
                    item["dose_summation_type"] for item in selected_doses
                ],
            }
        contract = {
            "version": 1,
            "authority": "organize",
            "patient_id": patient_id,
            "course_id": course_id,
            "selected_plans": selected_plans,
            "selected_doses": selected_doses,
            "authoritative_rtstruct": None,
            "planning_ct": {
                "status": "missing_reference",
                "series_instance_uid": "",
                "referenced_series_uids": [],
                "dicom_dir": "",
                "nifti_path": "",
            },
            "delivery": {
                "prescribed_dose_gy": prescribed_dose_gy,
                "status": delivery_status,
                "method": None,
                "delivered_dose_gy": delivered_dose_gy,
                "dose_response_field": DOSE_RESPONSE_FIELD,
                "delivered_record_count": sum(
                    int(item["delivered_record_count"])
                    for item in per_plan
                    if item["selected_for_dose_grid"]
                ),
                "delivered_fraction_count": sum(
                    int(item["delivered_fraction_count"])
                    for item in per_plan
                    if item["selected_for_dose_grid"]
                ),
                "planned_fraction_count": None,
                "unresolved_record_plan_uids": [],
                "unresolved_record_count": 0,
                "unresolved_reference_count": 0,
                "selected_plan_uids": selected_plan_uids,
                "per_plan": per_plan,
            },
            "dose_classification": {"classification": "evidence_replay"},
            "plan_artifact": (
                {
                    **selected_plans[0],
                    "source_plan_uids": selected_plan_uids,
                    "semantics": "selected_plan_artifact",
                }
                if selected_plans
                else None
            ),
            "dose_grid": dose_grid,
            "dose_qc": {
                "status": "fail" if qc_failure else "pass",
                "pass": not qc_failure,
                "threshold_gy": threshold_gy,
                "reasons": ["dose exceeds configured threshold"] if qc_failure else [],
            },
        }
        metadata = course / "metadata" / "case_metadata.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps({"patient_id": patient_id, "course_id": course_id, "course_contract": contract}),
            encoding="utf-8",
        )
        resolved = _resolve_dvh_dose(
            course,
            build_course_dirs(course),
            course / "metadata" / ".contract-rtstruct-absent",
            max_total_dose_gy=threshold_gy,
        )
        return {
            "source_plan_uids": resolved.source_plan_sop_instance_uids,
            "source_dose_uids": resolved.source_dose_sop_instance_uids,
            "dose_grid_semantics": resolved.dose_grid_semantics,
            "dose_qc_status": resolved.dose_qc_status,
        }


def build_evidence(output_root: Path, input_root: Path, workspace: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    record_cache: dict[str, dict[str, dict[str, Any]]] = {}
    metadata_paths = [
        course / "metadata" / "case_metadata.json"
        for patient in sorted(item for item in output_root.iterdir() if item.is_dir())
        for course in sorted(item for item in patient.iterdir() if item.is_dir())
        if (course / "metadata" / "case_metadata.json").is_file()
    ]
    for metadata_path in metadata_paths:
        course = metadata_path.parent.parent
        workbook = course / "dvh_metrics.xlsx"
        if not workbook.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prescribed = _as_float(metadata.get("total_prescription_gy"))
        if prescribed is None:
            continue
        patient_id = course.parent.name
        course_id = course.name
        plans = _dicom_inventory(course / "DICOM" / "RTPLAN", "RTPLAN")
        doses = _dicom_inventory(course / "DICOM" / "RTDOSE", "RTDOSE")
        selected_plan_uids = [str(value) for value in metadata.get("source_plan_uids") or []]
        selected_dose_uids = [str(value) for value in metadata.get("source_dose_uids") or []]
        frame = pd.read_excel(workbook)
        workbook_plan_uids = _split_uids(
            frame.get("source_plan_sop_instance_uids", [])
        )
        workbook_dose_uids = _split_uids(
            frame.get("source_dose_sop_instance_uids", [])
        )
        target_rows = frame[
            frame.get("ROI_Name", pd.Series(dtype=str)).astype(str).map(
                lambda value: bool(TARGET_RE.fullmatch(value.strip()))
            )
        ]
        if "Segmentation_Source" in target_rows.columns:
            plan_rows = target_rows[
                target_rows["Segmentation_Source"].astype(str) == "PlanRTSTRUCT"
            ]
            if not plan_rows.empty:
                target_rows = plan_rows
        target = None
        if not target_rows.empty:
            target_rows = target_rows.copy()
            target_rows["D95_numeric"] = pd.to_numeric(
                target_rows.get("D95Gy"), errors="coerce"
            )
            target_rows = target_rows.dropna(subset=["D95_numeric"])
            if not target_rows.empty:
                picked = target_rows.loc[target_rows["D95_numeric"].idxmax()]
                d95 = float(picked["D95_numeric"])
                target = {
                    "roi_name": str(picked["ROI_Name"]),
                    "d95_gy": d95,
                    "d95_to_prescription_percent": d95 / prescribed * 100.0,
                    "absolute_percent_difference_from_prescription": abs(
                        d95 / prescribed * 100.0 - 100.0
                    ),
                }

        if patient_id not in record_cache:
            record_cache[patient_id] = _record_evidence(input_root / patient_id)
        record_evidence = record_cache[patient_id]
        threshold = _as_float(metadata.get("dose_plausibility_threshold_gy")) or 100.0
        replay = _exercise_fixed_resolver(
            workspace=workspace,
            patient_id=patient_id,
            course_id=course_id,
            plans=plans,
            doses=doses,
            selected_plan_uids=selected_plan_uids,
            selected_dose_uids=selected_dose_uids,
            record_evidence=record_evidence,
            delivery_status=str(metadata.get("delivery_status") or "no_records_at_all"),
            prescribed_dose_gy=prescribed,
            delivered_dose_gy=_as_float(metadata.get("delivered_dose_gy")),
            threshold_gy=threshold,
        )
        plan_details = []
        for plan in plans:
            uid = str(plan["sop_instance_uid"])
            plan_details.append(
                {
                    **plan,
                    "selected_by_organize": uid in selected_plan_uids,
                    "delivery_evidence": record_evidence.get(
                        uid,
                        {
                            "delivered_record_count": 0,
                            "delivered_fraction_count": 0,
                            "treatment_dates": [],
                            "record_paths": [],
                        },
                    ),
                }
            )
        rows.append(
            {
                "course": f"{patient_id}/{course_id}",
                "claim_kinds": {
                    "source_fields_and_workbook_values": "fact",
                    "ratios_and_membership_comparisons": "calculation",
                    "clinical_interpretation": "inference",
                },
                "source_files": {
                    "case_metadata": str(metadata_path),
                    "case_metadata_sha256": _sha256(metadata_path),
                    "dvh_workbook": str(workbook),
                    "dvh_workbook_sha256": _sha256(workbook),
                },
                "observed": {
                    "rtplan_file_count": len(plans),
                    "rtdose_file_count": len(doses),
                    "organize_source_plan_uids": selected_plan_uids,
                    "organize_source_dose_uids": selected_dose_uids,
                    "legacy_dvh_source_plan_uids": workbook_plan_uids,
                    "legacy_dvh_source_dose_uids": workbook_dose_uids,
                    "prescribed_dose_gy": prescribed,
                    "delivered_dose_gy": _as_float(metadata.get("delivered_dose_gy")),
                    "delivery_status": metadata.get("delivery_status"),
                    "target_d95": target,
                    "plans": plan_details,
                    "doses": doses,
                },
                "calculated": {
                    "legacy_plan_membership_agrees_with_organize": set(workbook_plan_uids)
                    == set(selected_plan_uids),
                    "fixed_resolver": replay,
                    "fixed_plan_membership_agrees_with_organize": set(
                        replay["source_plan_uids"]
                    )
                    == set(selected_plan_uids),
                    "target_d95_differs_from_prescription_by_more_than_10_percent": bool(
                        target
                        and target["absolute_percent_difference_from_prescription"] > 10.0
                    ),
                },
            }
        )

    if len(rows) != 4:
        raise RuntimeError(
            f"expected four courses with both DVH and prescription, found {len(rows)}"
        )
    return {
        "ledger_version": 1,
        "purpose": "Read-only real-course membership reconciliation for the single-authority course contract",
        "provenance": {
            "campaign_output_root": str(output_root),
            "source_dicom_root": str(input_root),
            "source_policy": "Facts are copied from hashed metadata, workbooks, and DICOM headers. Calculations and inferences are labeled separately.",
            "scan": "All case_metadata.json files were enumerated. Courses were included when total_prescription_gy was numeric and DVH/dvh_metrics.xlsx existed.",
            "fixed_resolver_replay": "Header-only copies were written under a temporary workspace-owned directory, resolved through rtpipeline.dvh._resolve_dvh_dose, and deleted before exit.",
        },
        "courses": rows,
        "aggregate": {
            "course_count": len(rows),
            "legacy_plan_membership_mismatch_count": sum(
                not row["calculated"]["legacy_plan_membership_agrees_with_organize"]
                for row in rows
            ),
            "fixed_plan_membership_mismatch_count": sum(
                not row["calculated"]["fixed_plan_membership_agrees_with_organize"]
                for row in rows
            ),
            "target_d95_more_than_10_percent_from_prescription_count": sum(
                row["calculated"][
                    "target_d95_differs_from_prescription_by_more_than_10_percent"
                ]
                for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    ledger = args.ledger.resolve()
    ledger.relative_to(workspace)
    evidence = build_evidence(
        args.output_root.resolve(),
        args.input_root.resolve(),
        workspace,
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
