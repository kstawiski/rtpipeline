#!/usr/bin/env python3
"""Measure modality indexing and target-definition effects on completed cohorts.

The script is read-only with respect to cohort inputs. It emits aggregate JSON
only and does not include patient identifiers or full DICOM paths.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import openpyxl
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag

from rtpipeline.meta import _core_key_from_filename, _index_dicom_files_by_modality
from rtpipeline.rt_details import target_volume_names
from rtpipeline.utils import _scoped_walk

_MODALITY = Tag(0x0008, 0x0060)
_SOP_INSTANCE_UID = Tag(0x0008, 0x0018)
_REFERENCED_STRUCT = Tag(0x300C, 0x0060)
_ROI_SEQUENCE = Tag(0x3006, 0x0020)
_REFERENCED_PLAN = Tag(0x300C, 0x0002)
_EXPORT_TABLES = (
    "CT_images.xlsx",
    "plans.xlsx",
    "structure_sets.xlsx",
    "dosimetrics.xlsx",
    "fractions.xlsx",
    "metadata.xlsx",
)


def _all_dcm_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for base, _, files in _scoped_walk(root, None):
        paths.extend(Path(base) / name for name in files if name.lower().endswith(".dcm"))
    paths.sort(key=str)
    return paths


def _legacy_prefix_counts(paths: Iterable[Path]) -> dict[str, int]:
    return {
        prefix: sum(path.name.startswith(prefix) for path in paths)
        for prefix in ("RP", "RS", "RD", "RT", "CT")
    }


def _first_reference_uid(ds: Dataset, sequence_name: str) -> str:
    for item in getattr(ds, sequence_name, []) or []:
        value = str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
        if value:
            return value
    return ""


def _read_plan(path: Path) -> dict:
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[_SOP_INSTANCE_UID, _REFERENCED_STRUCT],
        )
    except Exception:
        return {"readable": False, "sop_uid": "", "struct_uid": "", "core_key": None}
    return {
        "readable": True,
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or "").strip(),
        "struct_uid": _first_reference_uid(ds, "ReferencedStructureSetSequence"),
        "core_key": _core_key_from_filename(path.name),
    }


def _read_struct(path: Path) -> dict:
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[_SOP_INSTANCE_UID, _ROI_SEQUENCE],
        )
    except Exception:
        return {"readable": False, "sop_uid": "", "permissive": False, "strict": False}
    names = [
        str(getattr(item, "ROIName", "") or "")
        for item in getattr(ds, "StructureSetROISequence", []) or []
        if getattr(item, "ROIName", None)
    ]
    return {
        "readable": True,
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or "").strip(),
        "permissive": any(
            token in name.upper()
            for name in names
            for token in ("GTV", "CTV", "PTV")
        ),
        "strict": bool(target_volume_names(names)),
    }


def _read_dose(path: Path) -> dict:
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[_REFERENCED_PLAN],
        )
    except Exception:
        return {"readable": False, "plan_uids": [], "core_key": None}
    return {
        "readable": True,
        "plan_uids": sorted(
            {
                str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
                for item in getattr(ds, "ReferencedRTPlanSequence", []) or []
                if getattr(item, "ReferencedSOPInstanceUID", None)
            }
        ),
        "core_key": _core_key_from_filename(path.name),
    }


def _parallel_read(paths: list[Path], function, workers: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(function, paths))


def _target_census(modalities: dict[str, list[Path]], workers: int) -> dict:
    plans = _parallel_read(modalities.get("RTPLAN", []), _read_plan, workers)
    structs = _parallel_read(modalities.get("RTSTRUCT", []), _read_struct, workers)
    doses = _parallel_read(modalities.get("RTDOSE", []), _read_dose, workers)

    structs_by_uid: dict[str, list[dict]] = defaultdict(list)
    for row in structs:
        if row["readable"] and row["sop_uid"]:
            structs_by_uid[row["sop_uid"]].append(row)
    struct_status = {
        uid: {
            "permissive": any(row["permissive"] for row in rows),
            "strict": any(row["strict"] for row in rows),
        }
        for uid, rows in structs_by_uid.items()
    }
    referenced_struct_uids = {
        row["struct_uid"]
        for row in plans
        if row["readable"] and row["struct_uid"]
    }
    permissive = {
        uid
        for uid in referenced_struct_uids
        if struct_status.get(uid, {}).get("permissive", False)
    }
    strict = {
        uid
        for uid in referenced_struct_uids
        if struct_status.get(uid, {}).get("strict", False)
    }

    plan_uids = {
        row["sop_uid"] for row in plans if row["readable"] and row["sop_uid"]
    }
    referenced_plan_uids = {
        uid
        for row in doses
        if row["readable"]
        for uid in row["plan_uids"]
    }
    plan_core_keys = {
        row["core_key"] for row in plans if row["readable"] and row["core_key"]
    }
    dose_core_keys = {
        row["core_key"] for row in doses if row["readable"] and row["core_key"]
    }

    duplicate_struct_groups = sum(len(rows) > 1 for rows in structs_by_uid.values())
    duplicate_struct_files = sum(max(0, len(rows) - 1) for rows in structs_by_uid.values())
    return {
        "plan_files": len(plans),
        "readable_plan_files": sum(row["readable"] for row in plans),
        "dose_files": len(doses),
        "readable_dose_files": sum(row["readable"] for row in doses),
        "structure_files": len(structs),
        "readable_structure_files": sum(row["readable"] for row in structs),
        "distinct_structure_sop_uids": len(structs_by_uid),
        "duplicate_structure_sop_uid_groups": duplicate_struct_groups,
        "duplicate_structure_files_beyond_first": duplicate_struct_files,
        "distinct_plan_referenced_structure_uids": len(referenced_struct_uids),
        "resolved_plan_referenced_structure_uids": sum(
            uid in struct_status for uid in referenced_struct_uids
        ),
        "permissive_target_bearing_plan_referenced_sets": len(permissive),
        "strict_target_bearing_plan_referenced_sets": len(strict),
        "sets_losing_target_status": len(permissive - strict),
        "strict_only_sets": len(strict - permissive),
        "plan_files_with_legacy_core_key": sum(
            row["readable"] and bool(row["core_key"]) for row in plans
        ),
        "dose_files_with_legacy_core_key": sum(
            row["readable"] and bool(row["core_key"]) for row in doses
        ),
        "shared_legacy_core_keys": len(plan_core_keys.intersection(dose_core_keys)),
        "distinct_plan_sop_uids": len(plan_uids),
        "dose_referenced_plan_uids": len(referenced_plan_uids),
        "dose_references_resolving_to_indexed_plans": len(plan_uids.intersection(referenced_plan_uids)),
    }


def _xlsx_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        if worksheet is None:
            return 0
        return max(0, int(worksheet.max_row or 0) - 1)
    finally:
        workbook.close()


def _export_inventory(data_root: Path | None) -> dict[str, dict] | None:
    if data_root is None:
        return None
    result: dict[str, dict] = {}
    for name in _EXPORT_TABLES:
        path = data_root / name
        result[name] = {
            "present": path.exists(),
            "rows": _xlsx_rows(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    enumeration_started = time.perf_counter()
    all_files = _all_dcm_files(args.dicom_root)
    enumeration_seconds = time.perf_counter() - enumeration_started
    legacy_started = time.perf_counter()
    legacy_counts = _legacy_prefix_counts(all_files)
    legacy_predicate_seconds = time.perf_counter() - legacy_started

    tag_started = time.perf_counter()
    modalities = _index_dicom_files_by_modality(
        args.dicom_root,
        max_workers=max(1, args.max_workers),
    )
    tag_index_seconds = time.perf_counter() - tag_started
    census_started = time.perf_counter()
    census = _target_census(modalities, max(1, args.max_workers))
    census_seconds = time.perf_counter() - census_started

    total = len(all_files)
    result = {
        "schema_version": 1,
        "cohort": args.cohort,
        "source": {
            "dicom_root": str(args.dicom_root),
            "data_root": str(args.data_root) if args.data_root else None,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pydicom": pydicom.__version__,
            "follow_input_symlinks_env": os.environ.get("RTPIPELINE_FOLLOW_INPUT_SYMLINKS", ""),
        },
        "method": {
            "filename_baseline": "one scoped walk plus startswith checks for RP, RS, RD, RT, and CT",
            "tag_index": "one scoped walk plus pydicom.dcmread(stop_before_pixels=True, specific_tags=[Modality])",
            "max_workers": max(1, args.max_workers),
            "clock": "time.perf_counter wall seconds",
        },
        "file_count": total,
        "legacy_prefix_counts": legacy_counts,
        "dicom_modality_counts": {
            modality: len(paths) for modality, paths in sorted(modalities.items())
        },
        "timing_seconds": {
            "filename_enumeration": round(enumeration_seconds, 6),
            "filename_predicates_after_enumeration": round(legacy_predicate_seconds, 6),
            "filename_total": round(enumeration_seconds + legacy_predicate_seconds, 6),
            "tag_index_total": round(tag_index_seconds, 6),
            "tag_index_per_1000_files": round(tag_index_seconds * 1000 / total, 6) if total else None,
            "tag_to_filename_total_ratio": round(
                tag_index_seconds / (enumeration_seconds + legacy_predicate_seconds), 6
            ) if enumeration_seconds + legacy_predicate_seconds else None,
            "target_and_reference_census": round(census_seconds, 6),
        },
        "target_and_reference_census": census,
        "completed_export_tables": _export_inventory(args.data_root),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
