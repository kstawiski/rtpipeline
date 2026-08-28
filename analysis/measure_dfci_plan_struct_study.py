#!/usr/bin/env python3
"""Measure RTPLAN to RTSTRUCT StudyInstanceUID relationships read-only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pydicom


def main(root: Path) -> None:
    structs: dict[tuple[str, str], str] = {}
    structs_by_uid: dict[str, str] = {}
    plans: list[tuple[str, str, str]] = []
    for patient_entry in sorted(os.scandir(root), key=lambda item: item.name):
        if not patient_entry.is_dir():
            continue
        for base, dirs, files in os.walk(patient_entry.path, followlinks=True):
            dirs.sort()
            dicom_files = sorted(
                name
                for name in files
                if name.lower().endswith(".dcm")
                and name.upper().startswith(("RP.", "RS."))
            )
            for name in dicom_files:
                path = Path(base) / name
                try:
                    ds = pydicom.dcmread(
                        str(path),
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=[
                            "Modality", "PatientID", "SOPInstanceUID",
                            "StudyInstanceUID", "ReferencedStructureSetSequence",
                        ],
                    )
                except Exception:
                    continue
                modality = str(getattr(ds, "Modality", "") or "").upper()
                patient_id = str(getattr(ds, "PatientID", "") or patient_entry.name)
                study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                if modality == "RTSTRUCT":
                    struct_uid = str(getattr(ds, "SOPInstanceUID", "") or "")
                    structs[(patient_id, struct_uid)] = study_uid
                    structs_by_uid[struct_uid] = study_uid
                elif modality == "RTPLAN":
                    refs = getattr(ds, "ReferencedStructureSetSequence", None) or []
                    ref_uid = str(getattr(refs[0], "ReferencedSOPInstanceUID", "") or "") if refs else ""
                    plans.append((patient_id, study_uid, ref_uid))

    resolved = 0
    cross_study = 0
    missing = 0
    no_reference = 0
    globally_resolvable_only = 0
    distinct_refsets: set[tuple[str, str]] = set()
    cross_refsets: set[tuple[str, str]] = set()
    examples: list[dict[str, str]] = []
    for patient_id, plan_study, ref_uid in plans:
        if not ref_uid:
            no_reference += 1
            continue
        struct_study = structs.get((patient_id, ref_uid))
        if not struct_study:
            if ref_uid in structs_by_uid:
                globally_resolvable_only += 1
            else:
                missing += 1
            continue
        resolved += 1
        distinct_refsets.add((patient_id, ref_uid))
        if plan_study != struct_study:
            cross_study += 1
            cross_refsets.add((patient_id, ref_uid))
            if len(examples) < 5:
                examples.append(
                    {
                        "patient_directory": patient_id,
                        "plan_study_uid": plan_study,
                        "struct_study_uid": struct_study,
                        "referenced_struct_uid": ref_uid,
                    }
                )
    print(
        json.dumps(
            {
                "patient_directories": sum(1 for entry in os.scandir(root) if entry.is_dir()),
                "plans": len(plans),
                "plans_without_structure_reference": no_reference,
                "resolved_plan_struct_links": resolved,
                "globally_resolvable_only": globally_resolvable_only,
                "missing_plan_struct_links": missing,
                "cross_study_plan_struct_links": cross_study,
                "distinct_referenced_structs": len(distinct_refsets),
                "cross_study_distinct_referenced_structs": len(cross_refsets),
                "examples": examples,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
