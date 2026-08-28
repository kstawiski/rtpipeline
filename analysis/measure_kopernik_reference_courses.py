#!/usr/bin/env python3
"""Measure reference-supported Kopernik courses without writing cohort data."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pydicom
from pydicom.dataset import Dataset

TARGET_TOKENS = ("GTV", "CTV", "PTV")
RT_PREFIXES = ("RP.", "RS.", "RD.", "RTPLAN", "RTSTRUCT", "RTDOSE")


def _rt_files(patient_dir: Path) -> Iterator[Path]:
    for base, dirs, files in os.walk(patient_dir, followlinks=True):
        dirs.sort()
        for name in sorted(files):
            upper = name.upper()
            if name.lower().endswith(".dcm") and upper.startswith(RT_PREFIXES):
                yield Path(base) / name


def _first_ref(ds: Dataset, sequence_name: str) -> str:
    sequence = getattr(ds, sequence_name, None) or []
    if not sequence:
        return ""
    return str(getattr(sequence[0], "ReferencedSOPInstanceUID", "") or "")


def _all_refs(ds: Dataset, sequence_name: str) -> tuple[str, ...]:
    sequence = getattr(ds, sequence_name, None) or []
    return tuple(
        str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
        for item in sequence
        if getattr(item, "ReferencedSOPInstanceUID", None)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    patient_dirs = sorted(entry.name for entry in os.scandir(args.input_root) if entry.is_dir())
    structs: dict[tuple[str, str], bool] = {}
    plans: list[tuple[str, str, str]] = []
    doses: list[tuple[str, tuple[str, ...]]] = []
    modality_counts: dict[str, int] = defaultdict(int)
    for patient_dir in patient_dirs:
        for path in _rt_files(args.input_root / patient_dir):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            except Exception:
                continue
            modality = str(getattr(ds, "Modality", "") or "").upper()
            patient_id = str(getattr(ds, "PatientID", "") or patient_dir)
            sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "")
            modality_counts[modality] += 1
            if modality == "RTSTRUCT":
                names = [
                    str(getattr(item, "ROIName", "") or "")
                    for item in (getattr(ds, "StructureSetROISequence", None) or [])
                ]
                structs[(patient_id, sop_uid)] = any(
                    token in name.upper() for name in names for token in TARGET_TOKENS
                )
            elif modality == "RTPLAN":
                plans.append((patient_id, sop_uid, _first_ref(ds, "ReferencedStructureSetSequence")))
            elif modality == "RTDOSE":
                doses.append((patient_id, _all_refs(ds, "ReferencedRTPlanSequence")))

    plan_by_key = {(patient_id, plan_uid): ref_uid for patient_id, plan_uid, ref_uid in plans}
    referenced_refsets: set[tuple[str, str]] = set()
    target_refsets_by_patient: dict[str, set[str]] = defaultdict(set)
    for patient_id, _, ref_uid in plans:
        if not ref_uid:
            continue
        key = (patient_id, ref_uid)
        referenced_refsets.add(key)
        if structs.get(key, False):
            target_refsets_by_patient[patient_id].add(ref_uid)

    linked_target_by_patient: dict[str, set[str]] = defaultdict(set)
    unresolved_doses = 0
    for patient_id, plan_uids in doses:
        resolved_any = False
        for plan_uid in plan_uids:
            ref_uid = plan_by_key.get((patient_id, plan_uid), "")
            if not ref_uid:
                continue
            resolved_any = True
            if structs.get((patient_id, ref_uid), False):
                linked_target_by_patient[patient_id].add(ref_uid)
        if not resolved_any:
            unresolved_doses += 1

    output_courses_by_patient: dict[str, int] = {}
    for patient_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        output_courses_by_patient[patient_dir.name] = sum(
            1 for course in patient_dir.iterdir() if course.is_dir() and not course.name.startswith("_")
        )

    undergenerated = {}
    for patient_id, refsets in target_refsets_by_patient.items():
        output_count = output_courses_by_patient.get(patient_id, 0)
        if output_count < len(refsets):
            undergenerated[patient_id] = {
                "target_bearing_referenced_structure_sets": len(refsets),
                "current_output_courses": output_count,
            }

    result = {
        "patient_directories": len(patient_dirs),
        "plans": modality_counts.get("RTPLAN", 0),
        "doses": modality_counts.get("RTDOSE", 0),
        "structs": modality_counts.get("RTSTRUCT", 0),
        "distinct_referenced_structure_sets": len(referenced_refsets),
        "target_bearing_referenced_structure_sets": sum(len(value) for value in target_refsets_by_patient.values()),
        "patients_with_target_bearing_referenced_structure_sets": len(target_refsets_by_patient),
        "dose_linked_target_courses": sum(len(value) for value in linked_target_by_patient.values()),
        "target_refsets_without_dose_linked_course": sum(
            len(target_refsets_by_patient[patient_id] - linked_target_by_patient.get(patient_id, set()))
            for patient_id in target_refsets_by_patient
        ),
        "target_refsets_without_dose_linked_course_by_patient": {
            patient_id: sorted(refsets - linked_target_by_patient.get(patient_id, set()))
            for patient_id, refsets in target_refsets_by_patient.items()
            if refsets - linked_target_by_patient.get(patient_id, set())
        },
        "unresolved_doses": unresolved_doses,
        "current_output_courses": sum(output_courses_by_patient.values()),
        "undergenerated_patients": undergenerated,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
