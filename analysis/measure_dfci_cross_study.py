#!/usr/bin/env python3
"""Measure RTSTRUCT to CT linkage across DICOM StudyInstanceUIDs.

Run with the cohort root as the only argument. The script is read-only and emits
one JSON object to stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pydicom
from pydicom.dataset import Dataset


def _selected_files(patient_dir: Path) -> Iterable[Path]:
    """Read one CT header per series directory and every RTSTRUCT header."""
    for base, dirs, files in os.walk(patient_dir):
        dirs.sort()
        files.sort()
        ct_names = [name for name in files if name.upper().startswith("CT") and name.lower().endswith(".dcm")]
        if ct_names:
            yield Path(base) / ct_names[0]
        for name in files:
            if name.upper().startswith("RS") and name.lower().endswith(".dcm"):
                yield Path(base) / name


def _ct_refs(ds: Dataset) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for ref_for in getattr(ds, "ReferencedFrameOfReferenceSequence", []) or []:
        frame_uid = str(getattr(ref_for, "FrameOfReferenceUID", "") or "")
        for study in getattr(ref_for, "RTReferencedStudySequence", []) or []:
            for series in getattr(study, "RTReferencedSeriesSequence", []) or []:
                uid = str(getattr(series, "SeriesInstanceUID", "") or "")
                if uid:
                    refs.append((uid, frame_uid))
    return refs


def main() -> None:
    root = Path(sys.argv[1])
    ct_by_series: dict[tuple[str, str], dict[str, str]] = {}
    structs: list[dict[str, object]] = []
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        patient = patient_dir.name
        for path in _selected_files(patient_dir):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            except Exception:
                continue
            modality = str(getattr(ds, "Modality", "") or "").upper()
            if modality == "CT":
                series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
                key = (patient, series_uid)
                if series_uid and key not in ct_by_series:
                    ct_by_series[key] = {
                        "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
                        "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "") or ""),
                    }
            elif modality == "RTSTRUCT":
                structs.append(
                    {
                        "patient": patient,
                        "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
                        "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
                        "refs": _ct_refs(ds),
                    }
                )

    referenced_links = 0
    resolved_links = 0
    frame_matches = 0
    cross_study_links = 0
    examples: list[dict[str, str]] = []
    for struct in structs:
        for series_uid, referenced_frame in struct["refs"]:  # type: ignore[index]
            referenced_links += 1
            ct = ct_by_series.get((str(struct["patient"]), series_uid))
            if not ct:
                continue
            resolved_links += 1
            if referenced_frame and ct["frame_uid"] == referenced_frame:
                frame_matches += 1
            if ct["study_uid"] != struct["study_uid"]:
                cross_study_links += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "patient_directory": str(struct["patient"]),
                            "rtstruct_sop_uid": str(struct["sop_uid"]),
                            "rt_study_uid": str(struct["study_uid"]),
                            "ct_series_uid": series_uid,
                            "ct_study_uid": ct["study_uid"],
                            "frame_uid": ct["frame_uid"],
                        }
                    )

    result = {
        "patient_directories": sum(1 for path in root.iterdir() if path.is_dir()),
        "ct_series": len(ct_by_series),
        "rtstructs": len(structs),
        "referenced_ct_series_links": referenced_links,
        "resolved_ct_series_links": resolved_links,
        "frame_uid_matches": frame_matches,
        "cross_study_rtstruct_ct_links": cross_study_links,
        "examples": examples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
