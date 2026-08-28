#!/usr/bin/env python3
"""Read-only DICOM reference-chain probe for course-identity diagnosis.

The script scans selected patient directories and emits JSON to stdout. It never
writes to the source tree and reads DICOM headers without pixel data.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pydicom
from pydicom.dataset import Dataset

from rtpipeline.rt_details import target_volume_names


def _target_names(names: Iterable[str]) -> list[str]:
    return sorted(set(target_volume_names([str(name) for name in names])))


def _first_uid(sequence: Iterable[Any] | None) -> str:
    for item in sequence or []:
        value = str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
        if value:
            return value
    return ""


def _referenced_ct_series(ds: Dataset) -> list[str]:
    values: set[str] = set()
    for ref_for in getattr(ds, "ReferencedFrameOfReferenceSequence", []) or []:
        for study in getattr(ref_for, "RTReferencedStudySequence", []) or []:
            for series in getattr(study, "RTReferencedSeriesSequence", []) or []:
                uid = str(getattr(series, "SeriesInstanceUID", "") or "").strip()
                if uid:
                    values.add(uid)
    return sorted(values)


def _iter_files(root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        for name in files:
            yield Path(base) / name


def scan_patient(path: Path) -> dict[str, Any]:
    plans: list[dict[str, Any]] = []
    doses: list[dict[str, Any]] = []
    structs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    ct_series: dict[str, dict[str, Any]] = {}
    modality_counts: collections.Counter[str] = collections.Counter()
    unreadable = 0

    for dicom_path in _iter_files(path):
        try:
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
        except Exception:
            unreadable += 1
            continue
        modality = str(getattr(ds, "Modality", "") or "").upper()
        if not modality:
            continue
        modality_counts[modality] += 1
        common = {
            "path": str(dicom_path),
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
            "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
            "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "") or ""),
        }
        if modality == "CT":
            uid = common["series_uid"]
            entry = ct_series.setdefault(
                uid,
                {
                    "series_uid": uid,
                    "study_uid": common["study_uid"],
                    "frame_uid": common["frame_uid"],
                    "instances": 0,
                    "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
                    "manufacturer_model": str(getattr(ds, "ManufacturerModelName", "") or ""),
                    "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
                    "image_types": [],
                    "slice_thickness": getattr(ds, "SliceThickness", None),
                },
            )
            entry["instances"] += 1
            entry["image_types"] = sorted(
                set(entry["image_types"])
                | {str(value) for value in (getattr(ds, "ImageType", None) or [])}
            )
        elif modality == "RTPLAN":
            plans.append(
                {
                    **common,
                    "label": str(getattr(ds, "RTPlanLabel", "") or ""),
                    "name": str(getattr(ds, "RTPlanName", "") or ""),
                    "date": str(getattr(ds, "RTPlanDate", "") or getattr(ds, "InstanceCreationDate", "") or ""),
                    "time": str(getattr(ds, "RTPlanTime", "") or getattr(ds, "InstanceCreationTime", "") or ""),
                    "referenced_struct_uid": _first_uid(getattr(ds, "ReferencedStructureSetSequence", None)),
                    "rx_values_gy": [
                        float(value)
                        for item in (getattr(ds, "DoseReferenceSequence", None) or [])
                        for value in [getattr(item, "TargetPrescriptionDose", None)]
                        if value not in (None, "")
                    ],
                    "planned_fractions": [
                        int(value)
                        for item in (getattr(ds, "FractionGroupSequence", None) or [])
                        for value in [getattr(item, "NumberOfFractionsPlanned", None)]
                        if value not in (None, "")
                    ],
                }
            )
        elif modality == "RTDOSE":
            doses.append(
                {
                    **common,
                    "summation_type": str(getattr(ds, "DoseSummationType", "") or ""),
                    "referenced_plan_uids": sorted(
                        {
                            str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
                            for item in (getattr(ds, "ReferencedRTPlanSequence", None) or [])
                            if getattr(item, "ReferencedSOPInstanceUID", None)
                        }
                    ),
                }
            )
        elif modality == "RTSTRUCT":
            names = [
                str(getattr(item, "ROIName", "") or "")
                for item in (getattr(ds, "StructureSetROISequence", None) or [])
                if getattr(item, "ROIName", None)
            ]
            structs.append(
                {
                    **common,
                    "roi_count": len(names),
                    "target_names": _target_names(names),
                    "roi_names": names,
                    "referenced_ct_series_uids": _referenced_ct_series(ds),
                }
            )
        elif "RECORD" in modality or hasattr(ds, "TreatmentSessionBeamSequence"):
            records.append(
                {
                    **common,
                    "modality": modality,
                    "referenced_plan_uid": _first_uid(getattr(ds, "ReferencedRTPlanSequence", None)),
                    "treatment_date": str(getattr(ds, "TreatmentDate", "") or getattr(ds, "SeriesDate", "") or ""),
                    "fraction_number": getattr(ds, "ReferencedFractionNumber", None),
                }
            )

    struct_by_uid = {item["sop_uid"]: item for item in structs if item["sop_uid"]}
    ct_by_uid = {uid: item for uid, item in ct_series.items() if uid}
    plan_struct_links = []
    for plan in plans:
        ref = plan["referenced_struct_uid"]
        struct = struct_by_uid.get(ref)
        plan_struct_links.append(
            {
                "plan_uid": plan["sop_uid"],
                "plan_date": plan["date"],
                "plan_label": plan["label"],
                "plan_study_uid": plan["study_uid"],
                "referenced_struct_uid": ref,
                "resolved": struct is not None,
                "struct_study_uid": struct["study_uid"] if struct else "",
                "struct_roi_count": struct["roi_count"] if struct else None,
                "target_names": struct["target_names"] if struct else [],
                "referenced_ct_series_uids": struct["referenced_ct_series_uids"] if struct else [],
                "resolved_ct_series_uids": [
                    uid for uid in (struct["referenced_ct_series_uids"] if struct else []) if uid in ct_by_uid
                ],
            }
        )

    return {
        "patient_directory": str(path),
        "modality_counts": dict(sorted(modality_counts.items())),
        "unreadable_files": unreadable,
        "plans": plans,
        "doses": doses,
        "structs": structs,
        "records": records,
        "ct_series": sorted(ct_series.values(), key=lambda item: item["series_uid"]),
        "plan_struct_links": plan_struct_links,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("patients", nargs="+")
    args = parser.parse_args()
    result = {
        "root": str(args.root),
        "patients": {
            patient: scan_patient(args.root / patient)
            for patient in args.patients
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
