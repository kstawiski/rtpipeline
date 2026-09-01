#!/usr/bin/env python3
"""Measure clinical prescription evidence against existing course contracts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pydicom

from rtpipeline.clinical_prescription import (
    CLINICAL_EVIDENCE_SCHEMA,
    adjudicate_clinical_prescription,
    load_kopernik_treatment_records,
)


def _plan_dates(course_dir: Path, contract: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    paths = [
        str(item.get("plan_path") or "")
        for item in (contract.get("delivery") or {}).get("per_plan", []) or []
    ]
    paths.extend(
        str(item.get("path") or "")
        for item in contract.get("selected_plans", []) or []
    )
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = course_dir / path
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        date_value = str(getattr(dataset, "RTPlanDate", "") or "").strip()
        if date_value:
            values.add(date_value)
    return sorted(values)


def _course_contracts(
    roots: list[Path],
) -> dict[tuple[str, str], tuple[Path, dict[str, Any], Path, str]]:
    contracts: dict[
        tuple[str, str], tuple[Path, dict[str, Any], Path, str]
    ] = {}
    for root in roots:
        for path in sorted(root.glob("*/*/metadata/case_metadata.json")):
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
            patient_id = str(data.get("patient_id") or path.parents[2].name)
            course_id = str(data.get("course_id") or path.parents[1].name)
            key = (patient_id, course_id)
            if key in contracts:
                raise RuntimeError(
                    f"duplicate course contract across analysis roots: {patient_id}/{course_id}"
                )
            contracts[key] = (
                path.parents[1],
                data,
                path,
                hashlib.sha256(raw).hexdigest(),
            )
    return contracts


def _verify_unchanged_snapshot(
    contracts: dict[
        tuple[str, str], tuple[Path, dict[str, Any], Path, str]
    ],
) -> str:
    inventory_rows: list[str] = []
    for key, (_, _, path, expected_hash) in sorted(contracts.items()):
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_hash != expected_hash:
            raise RuntimeError(f"course contract changed during audit: {path}")
        inventory_rows.append(f"{key[0]}/{key[1]}\t{expected_hash}\n")
    return hashlib.sha256("".join(inventory_rows).encode("utf-8")).hexdigest()


def _evidence_for_course(
    *,
    index,
    patient_id: str,
    course_id: str,
    course_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    course_contract = contract.get("course_contract") or contract
    existing = course_contract.get("clinical_prescription_evidence")
    if (
        isinstance(existing, dict)
        and existing.get("schema") == CLINICAL_EVIDENCE_SCHEMA
        and (existing.get("source") or {}).get("workbook_sha256")
        == index.workbook_sha256
    ):
        return existing
    delivery = course_contract.get("delivery") or {}
    per_plan = list(delivery.get("per_plan") or [])
    treatment_dates = sorted(
        {
            str(value)
            for item in per_plan
            for value in item.get("treatment_dates", []) or []
            if str(value).strip()
        }
    )
    return adjudicate_clinical_prescription(
        index,
        patient_id=patient_id,
        course_id=course_id,
        course_start_date=contract.get("course_start_date"),
        course_end_date=contract.get("course_end_date"),
        plan_dates=_plan_dates(course_dir, course_contract),
        treatment_dates=treatment_dates,
        dicom_resolved_total_gy=delivery.get(
            "resolved_prescribed_dose_total_gy"
        ),
        dicom_prescribed_dose_scope=str(
            delivery.get("prescribed_dose_scope") or ""
        ),
        dicom_classification=str(
            (course_contract.get("dose_classification") or {}).get("classification") or ""
        )
        or None,
        per_plan_delivery=per_plan,
    )


def _compact_course(
    patient_id: str,
    course_id: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    record = evidence.get("record") or {}
    parsed = evidence.get("parse") or {}
    return {
        "patient_id": patient_id,
        "course_id": course_id,
        "outcome": evidence.get("outcome"),
        "reason": evidence.get("reason"),
        "dicom_resolved_total_gy": (evidence.get("dicom") or {}).get(
            "resolved_prescribed_dose_total_gy"
        ),
        "dicom_prescribed_dose_scope": (evidence.get("dicom") or {}).get(
            "prescribed_dose_scope"
        ),
        "clinical_resolved_total_gy": evidence.get(
            "clinical_resolved_total_gy"
        ),
        "clinical_sites": [
            {
                "site": site.get("site"),
                "total_dose_gy": site.get("total_dose_gy"),
                "fraction_count": site.get("fraction_count"),
                "phases": site.get("phases"),
            }
            for site in parsed.get("sites", []) or []
        ],
        "record_id": record.get("record_id"),
        "excel_row": record.get("excel_row"),
        "treatment_start_date": record.get("treatment_start_date"),
        "treatment_end_date": record.get("treatment_end_date"),
        "parsed_field": record.get("parsed_field"),
        "source_text": record.get("source_text"),
        "match": evidence.get("match"),
        "disagreement": evidence.get("disagreement"),
        "fractionation_classification": evidence.get(
            "fractionation_classification"
        ),
    }


def build_report(
    clinical_records: Path,
    output_roots: list[Path],
) -> dict[str, Any]:
    index = load_kopernik_treatment_records(clinical_records)
    contracts = _course_contracts(output_roots)
    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    courses: list[dict[str, Any]] = []
    for (patient_id, course_id), (
        course_dir,
        contract,
        _contract_path,
        contract_hash,
    ) in sorted(contracts.items()):
        evidence = _evidence_for_course(
            index=index,
            patient_id=patient_id,
            course_id=course_id,
            course_dir=course_dir,
            contract=contract,
        )
        outcome = str(evidence.get("outcome") or "UNRESOLVED")
        reason = str(evidence.get("reason") or "NONE")
        outcomes[outcome] += 1
        reasons[reason] += 1
        compact = _compact_course(patient_id, course_id, evidence)
        compact["case_metadata_sha256"] = contract_hash
        courses.append(compact)

    declared_total = sum(outcomes.values())
    if declared_total != len(contracts):
        raise RuntimeError(
            f"outcome reconciliation failed: {declared_total} != {len(contracts)}"
        )
    disagreements = [
        item for item in courses if item["outcome"] == "DISAGREES_WITH_DICOM"
    ]
    resolved = [
        item
        for item in courses
        if item["outcome"] == "RESOLVED_FROM_CLINICAL_RECORD"
    ]
    corroborated = [
        item for item in courses if item["outcome"] == "CORROBORATED_DICOM"
    ]
    inventory_hash = _verify_unchanged_snapshot(contracts)
    return {
        "schema": "rtpipeline-clinical-prescription-cohort-audit-v1",
        "source": index.source_dict(),
        "course_contract_count": len(contracts),
        "contract_inventory_sha256": inventory_hash,
        "snapshot_verification": (
            "all case_metadata.json hashes were unchanged during the audit"
        ),
        "patient_count": len({patient for patient, _ in contracts}),
        "counts": {
            "unresolved_to_resolved": len(resolved),
            "corroborated": len(corroborated),
            "disagrees_with_dicom": len(disagreements),
            "clinical_evidence_unresolved": outcomes["UNRESOLVED"],
        },
        "outcome_reconciliation": dict(sorted(outcomes.items())),
        "unresolved_reason_counts": dict(sorted(reasons.items())),
        "resolved_courses": resolved,
        "corroborated_courses": corroborated,
        "disagreements": disagreements,
        "all_courses": courses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-records", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, required=True, action="append"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.clinical_records, args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("course_contract_count", "patient_count", "counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
