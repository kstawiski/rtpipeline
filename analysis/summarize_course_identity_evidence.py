#!/usr/bin/env python3
"""Reduce read-only probe output to the diagnosis evidence ledger."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import pydicom


def _output_courses(output_root: Path, patient: str) -> list[dict[str, Any]]:
    patient_dir = output_root / patient
    if not patient_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for course_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
        rs_path = course_dir / "RS.dcm"
        rtstruct_dir = course_dir / "DICOM" / "RTSTRUCT"
        rtstruct_files = sorted(path for path in rtstruct_dir.glob("*") if path.is_file())
        selected_rs_path = rs_path if rs_path.is_file() else (rtstruct_files[0] if rtstruct_files else rs_path)
        selected_rs_uid = ""
        target_names: list[str] = []
        roi_names: list[str] = []
        if selected_rs_path.is_file():
            try:
                ds = pydicom.dcmread(str(selected_rs_path), stop_before_pixels=True, force=True)
                selected_rs_uid = str(getattr(ds, "SOPInstanceUID", "") or "")
                roi_names = [
                    str(getattr(item, "ROIName", "") or "")
                    for item in (getattr(ds, "StructureSetROISequence", None) or [])
                    if getattr(item, "ROIName", None)
                ]
                target_names = [
                    name for name in roi_names
                    if any(token in name.upper() for token in ("GTV", "CTV", "PTV"))
                ]
            except Exception:
                pass
        rows.append(
            {
                "course": course_dir.name,
                "ct_files": sum(1 for path in (course_dir / "DICOM" / "CT").glob("*") if path.is_file()),
                "rtstruct_files": len(rtstruct_files),
                "rtplan_files": sum(1 for path in (course_dir / "DICOM" / "RTPLAN").glob("*") if path.is_file()),
                "rtdose_files": sum(1 for path in (course_dir / "DICOM" / "RTDOSE").glob("*") if path.is_file()),
                "primary_rs_present": rs_path.is_file(),
                "selected_rtstruct_path": str(selected_rs_path) if selected_rs_path.is_file() else "",
                "selected_rtstruct_uid": selected_rs_uid,
                "primary_rs_roi_count": len(roi_names),
                "primary_rs_target_names": target_names,
            }
        )
    return rows


def _patient_summary(patient: dict[str, Any]) -> dict[str, Any]:
    structs = {row["sop_uid"]: row for row in patient["structs"]}
    ref_groups: dict[str, dict[str, Any]] = {}
    for plan in patient["plans"]:
        ref = plan["referenced_struct_uid"]
        struct = structs.get(ref)
        group = ref_groups.setdefault(
            ref,
            {
                "struct_uid": ref,
                "roi_count": struct["roi_count"] if struct else None,
                "target_names": struct["target_names"] if struct else [],
                "plan_uids": [],
                "plans": [],
            },
        )
        group["plan_uids"].append(plan["sop_uid"])
        group["plans"].append(
            {
                "uid": plan["sop_uid"],
                "date": plan["date"],
                "time": plan["time"],
                "label": plan["label"],
                "rx_values_gy": plan["rx_values_gy"],
                "planned_fractions": plan["planned_fractions"],
            }
        )
    record_counts = collections.Counter(
        record["referenced_plan_uid"]
        for record in patient["records"]
        if record["referenced_plan_uid"]
    )
    dose_counts = collections.Counter(
        uid
        for dose in patient["doses"]
        for uid in dose["referenced_plan_uids"]
        if uid
    )
    return {
        "modality_counts": patient["modality_counts"],
        "plan_count": len(patient["plans"]),
        "dose_count": len(patient["doses"]),
        "struct_count": len(patient["structs"]),
        "record_count": len(patient["records"]),
        "plan_referenced_struct_sets": sorted(ref_groups.values(), key=lambda row: row["struct_uid"]),
        "dose_count_by_plan_uid": dict(sorted(dose_counts.items())),
        "record_count_by_plan_uid": dict(sorted(record_counts.items())),
    }


def _dfci_sample_summary(data: dict[str, Any]) -> dict[str, Any]:
    totals = collections.Counter()
    per_patient: dict[str, Any] = {}
    for patient_id, patient in data["patients"].items():
        links = patient["plan_struct_links"]
        ct_by_uid = {row["series_uid"]: row for row in patient["ct_series"]}
        cross_study = sum(
            1 for link in links
            if link["resolved"]
            and link["plan_study_uid"]
            and link["struct_study_uid"]
            and link["plan_study_uid"] != link["struct_study_uid"]
        )
        cross_study_rt_to_ct = sum(
            1
            for link in links
            for uid in link["resolved_ct_series_uids"]
            if ct_by_uid.get(uid, {}).get("study_uid")
            and link["struct_study_uid"]
            and ct_by_uid[uid]["study_uid"] != link["struct_study_uid"]
        )
        referenced_ct = sum(len(link["referenced_ct_series_uids"]) for link in links)
        resolved_ct = sum(len(link["resolved_ct_series_uids"]) for link in links)
        totals.update(
            {
                "plans": len(patient["plans"]),
                "doses": len(patient["doses"]),
                "structs": len(patient["structs"]),
                "plan_struct_links": len(links),
                "resolved_plan_struct_links": sum(1 for link in links if link["resolved"]),
                "cross_study_plan_struct_links": cross_study,
                "cross_study_rt_to_ct_links": cross_study_rt_to_ct,
                "referenced_ct_series_links": referenced_ct,
                "resolved_ct_series_links": resolved_ct,
            }
        )
        per_patient[patient_id] = {
            "plans": len(patient["plans"]),
            "doses": len(patient["doses"]),
            "structs": len(patient["structs"]),
            "cross_study_plan_struct_links": cross_study,
            "cross_study_rt_to_ct_links": cross_study_rt_to_ct,
            "referenced_ct_series_links": referenced_ct,
            "resolved_ct_series_links": resolved_ct,
        }
    return {"totals": dict(totals), "patients": per_patient}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kopernik-probe", type=Path, required=True)
    parser.add_argument("--dfci-probe", type=Path, required=True)
    parser.add_argument("--kopernik-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kopernik = json.loads(args.kopernik_probe.read_text(encoding="utf-8"))
    dfci = json.loads(args.dfci_probe.read_text(encoding="utf-8"))
    result = {
        "claim_kinds": {
            "patient_probe_summaries": "calculation from DICOM headers",
            "output_course_inventory": "calculation from current output files",
            "interpretation": "inference reserved for docs/diagnosis-course-identity.md",
        },
        "sources": {
            "kopernik_probe": str(args.kopernik_probe),
            "dfci_probe": str(args.dfci_probe),
            "kopernik_output": str(args.kopernik_output),
        },
        "kopernik": {
            patient_id: {
                **_patient_summary(patient),
                "current_output_courses": _output_courses(args.kopernik_output, patient_id),
            }
            for patient_id, patient in kopernik["patients"].items()
        },
        "dfci_sample": _dfci_sample_summary(dfci),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
