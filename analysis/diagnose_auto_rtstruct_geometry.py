#!/usr/bin/env python3
"""Audit auto-RTSTRUCT geometry acceptance without writing cohort data.

The audit discovers course directories from the declared cohort root, selects the
TotalSegmentator directory with the production provenance logic, reads every
binary-mask header, and evaluates one representative image for each distinct
geometry through the production gate. Grouping by complete SimpleITK geometry
avoids decompressing duplicate masks while preserving whole-fallback semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk

from rtpipeline.auto_rtstruct import (
    _geometry_compatible,
    _load_ct_image,
    _read_ct_for_uid,
    _read_ct_series_uid,
    _select_seg_dir_for_ct,
)

WORKSPACE = Path("/umed-projekty/rtpipeline")
DEFAULT_ROOT = Path("/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output")
GENERIC_MULTILABEL_NAMES = {
    "segmentation.nii",
    "segmentation.nii.gz",
    "segmentations.nii",
    "segmentations.nii.gz",
}


class _GeometryProxy:
    """Expose header geometry through the image methods used by the gate."""

    def __init__(self, geometry: dict[str, Any]) -> None:
        self.geometry = geometry

    def GetDimension(self) -> int:
        return int(self.geometry["dimension"])

    def GetSize(self) -> tuple[int, ...]:
        return self.geometry["size"]

    def GetSpacing(self) -> tuple[float, ...]:
        return self.geometry["spacing"]

    def GetOrigin(self) -> tuple[float, ...]:
        return self.geometry["origin"]

    def GetDirection(self) -> tuple[float, ...]:
        return self.geometry["direction"]

    def TransformIndexToPhysicalPoint(self, index: tuple[int, ...]) -> tuple[float, ...]:
        spacing = np.asarray(self.GetSpacing(), dtype=float)
        origin = np.asarray(self.GetOrigin(), dtype=float)
        direction = np.asarray(self.GetDirection(), dtype=float).reshape(3, 3)
        point = origin + direction @ (np.asarray(index, dtype=float) * spacing)
        return tuple(float(value) for value in point)


def _image_geometry(path: Path) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return {
        "dimension": int(reader.GetDimension()),
        "size": tuple(int(v) for v in reader.GetSize()),
        "spacing": tuple(float(v) for v in reader.GetSpacing()),
        "origin": tuple(float(v) for v in reader.GetOrigin()),
        "direction": tuple(float(v) for v in reader.GetDirection()),
    }


def _geometry_key(geometry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        geometry["dimension"],
        geometry["size"],
        geometry["spacing"],
        geometry["origin"],
        geometry["direction"],
    )


def _extent(geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(geometry["size"], dtype=int)
    spacing = np.asarray(geometry["spacing"], dtype=float)
    origin = np.asarray(geometry["origin"], dtype=float)
    direction = np.asarray(geometry["direction"], dtype=float).reshape(3, 3)
    corners = []
    for index in itertools.product(*[(0, int(n) - 1) for n in size]):
        corners.append(origin + direction @ (np.asarray(index, dtype=float) * spacing))
    points = np.asarray(corners, dtype=float)
    return points.min(axis=0), points.max(axis=0)


def _legacy_geometry_compatible(
    seg: dict[str, Any], ct: dict[str, Any], tol_mm: float = 2.0
) -> bool:
    """Metadata-equivalent copy of the gate at commit 7d45561."""
    try:
        if seg["dimension"] != 3 or ct["dimension"] != 3:
            return False
        if not np.allclose(seg["direction"], ct["direction"], atol=1e-3):
            return False
        if seg["size"] != ct["size"]:
            return False
        if not np.allclose(seg["spacing"], ct["spacing"], atol=1e-3):
            return False
        if not np.allclose(seg["origin"], ct["origin"], atol=tol_mm):
            return False
        seg_lo, seg_hi = _extent(seg)
        ct_lo, ct_hi = _extent(ct)
        return bool(
            np.allclose(seg_lo, ct_lo, atol=tol_mm)
            and np.allclose(seg_hi, ct_hi, atol=tol_mm)
        )
    except Exception:
        return False


def _binary_mask_paths(seg_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(seg_dir.glob("*.nii*"))
        if path.name.lower() not in GENERIC_MULTILABEL_NAMES
        and "_total_multilabel.nii" not in path.name.lower()
    ]


def _planning_nifti_path(
    course: Path, ct_dir: Path, ct_series_uid: str
) -> tuple[Path, str]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for metadata_path in sorted((course / "NIFTI").glob("*.metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        nifti_path = Path(str(metadata.get("nifti_path", "")))
        if nifti_path.is_file():
            records.append((nifti_path, metadata))

    uid_matches = [
        path
        for path, metadata in records
        if str(metadata.get("series_instance_uid", "")) == ct_series_uid
    ]
    if len(uid_matches) == 1:
        return uid_matches[0], "series_instance_uid"
    if len(uid_matches) > 1:
        raise RuntimeError(
            f"SeriesInstanceUID {ct_series_uid} matched {len(uid_matches)} NIfTI records"
        )

    ct_files = sorted(path for path in ct_dir.iterdir() if path.is_file())
    if not ct_files:
        raise RuntimeError("DICOM/CT contains no files")
    dataset = pydicom.dcmread(str(ct_files[0]), stop_before_pixels=True, force=True)
    description_tokens = re.findall(
        r"[a-z0-9]+", str(getattr(dataset, "SeriesDescription", "")).lower()
    )
    series_date = str(
        getattr(dataset, "SeriesDate", "")
        or getattr(dataset, "AcquisitionDate", "")
        or getattr(dataset, "StudyDate", "")
    )
    context_matches = []
    for path, metadata in records:
        stem_tokens = re.findall(r"[a-z0-9]+", path.name.lower())
        description_matches = all(token in stem_tokens for token in description_tokens)
        date_matches = not series_date or series_date in stem_tokens
        count_matches = int(metadata.get("instance_count", -1)) == len(ct_files)
        if description_tokens and description_matches and date_matches and count_matches:
            context_matches.append(path)
    if len(context_matches) == 1:
        return context_matches[0], "description_date_instance_count"
    raise RuntimeError(
        f"Expected one planning NIfTI for CT series {ct_series_uid}, found "
        f"{len(uid_matches)} UID matches and {len(context_matches)} contextual matches"
    )


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(WORKSPACE), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha256(revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(WORKSPACE), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def audit(root: Path, expected_courses: int) -> dict[str, Any]:
    candidate_dirs = sorted(path for path in root.glob("*/*") if path.is_dir())
    courses = [
        path
        for path in candidate_dirs
        if (path / "DICOM" / "CT").is_dir()
        and (path / "Segmentation_TotalSegmentator").is_dir()
    ]
    if len(courses) != expected_courses:
        raise RuntimeError(
            f"Expected {expected_courses} courses with DICOM/CT and "
            f"Segmentation_TotalSegmentator, found {len(courses)}"
        )

    per_course: list[dict[str, Any]] = []
    aggregate = {
        "courses_discovered": len(courses),
        "courses_with_selected_binary_masks": 0,
        "courses_without_binary_masks": 0,
        "mask_headers_inspected": 0,
        "distinct_mask_geometry_groups": 0,
        "legacy_gate_accepted_courses": 0,
        "legacy_gate_rejected_courses": 0,
        "production_gate_accepted_courses": 0,
        "production_gate_rejected_courses": 0,
        "selection_failures": 0,
        "header_or_load_failures": 0,
        "planning_nifti_matched_courses": 0,
        "planning_nifti_series_uid_matches": 0,
        "planning_nifti_context_matches": 0,
        "planning_nifti_match_failures": 0,
        "planning_nifti_legacy_gate_accepted_courses": 0,
        "planning_nifti_legacy_gate_rejected_courses": 0,
        "planning_nifti_production_gate_accepted_courses": 0,
        "planning_nifti_production_gate_rejected_courses": 0,
    }

    for course in courses:
        ct_dir = course / "DICOM" / "CT"
        seg_root = course / "Segmentation_TotalSegmentator"
        row: dict[str, Any] = {
            "course": str(course.relative_to(root)),
            "status": "pending",
        }
        reader = sitk.ImageSeriesReader()
        series_ids = list(reader.GetGDCMSeriesIDs(str(ct_dir)) or [])
        row["ct_series_count"] = len(series_ids)
        row["ct_file_count"] = len([path for path in ct_dir.iterdir() if path.is_file()])

        ct_img = _load_ct_image(ct_dir)
        if ct_img is None:
            row["status"] = "header_or_load_failure"
            row["error"] = "production CT loader returned None"
            aggregate["header_or_load_failures"] += 1
            per_course.append(row)
            continue
        ct_geometry = {
            "dimension": int(ct_img.GetDimension()),
            "size": tuple(int(v) for v in ct_img.GetSize()),
            "spacing": tuple(float(v) for v in ct_img.GetSpacing()),
            "origin": tuple(float(v) for v in ct_img.GetOrigin()),
            "direction": tuple(float(v) for v in ct_img.GetDirection()),
        }
        ct_series_uid = _read_ct_series_uid(ct_dir)
        try:
            planning_nifti, planning_match_method = _planning_nifti_path(
                course, ct_dir, ct_series_uid
            )
            planning_geometry = _image_geometry(planning_nifti)
            planning_legacy_accepts = _legacy_geometry_compatible(
                planning_geometry, ct_geometry
            )
            planning_production_accepts = bool(
                _geometry_compatible(_GeometryProxy(planning_geometry), ct_img)
            )
            row.update(
                {
                    "planning_nifti": str(planning_nifti.relative_to(course)),
                    "planning_nifti_match_method": planning_match_method,
                    "planning_nifti_legacy_gate_accepts": planning_legacy_accepts,
                    "planning_nifti_production_gate_accepts": planning_production_accepts,
                }
            )
            aggregate["planning_nifti_matched_courses"] += 1
            aggregate[
                "planning_nifti_series_uid_matches"
                if planning_match_method == "series_instance_uid"
                else "planning_nifti_context_matches"
            ] += 1
            aggregate[
                "planning_nifti_legacy_gate_accepted_courses"
                if planning_legacy_accepts
                else "planning_nifti_legacy_gate_rejected_courses"
            ] += 1
            aggregate[
                "planning_nifti_production_gate_accepted_courses"
                if planning_production_accepts
                else "planning_nifti_production_gate_rejected_courses"
            ] += 1
        except Exception as exc:
            row["planning_nifti_error"] = f"{type(exc).__name__}: {exc}"
            aggregate["planning_nifti_match_failures"] += 1

        candidate_seg_dirs = sorted(path for path in seg_root.iterdir() if path.is_dir())
        selected_dir = None
        if candidate_seg_dirs:
            selected_dir, _, _ = _select_seg_dir_for_ct(
                candidate_seg_dirs,
                ct_series_uid,
                _read_ct_for_uid(ct_dir),
            )
        if candidate_seg_dirs and selected_dir is None:
            row["status"] = "selection_failure"
            aggregate["selection_failures"] += 1
            per_course.append(row)
            continue

        fallback_dir = selected_dir or seg_root
        mask_paths = _binary_mask_paths(fallback_dir)
        row["selected_seg_dir"] = str(fallback_dir.relative_to(course))
        row["binary_mask_count"] = len(mask_paths)
        if not mask_paths:
            row["status"] = "no_binary_masks"
            aggregate["courses_without_binary_masks"] += 1
            per_course.append(row)
            continue

        aggregate["courses_with_selected_binary_masks"] += 1
        aggregate["mask_headers_inspected"] += len(mask_paths)
        try:
            groups: dict[tuple[Any, ...], dict[str, Any]] = {}
            for mask_path in mask_paths:
                geometry = _image_geometry(mask_path)
                groups.setdefault(
                    _geometry_key(geometry),
                    {"geometry": geometry, "representative": mask_path, "count": 0},
                )["count"] += 1
            aggregate["distinct_mask_geometry_groups"] += len(groups)

            legacy_group_acceptance = [
                _legacy_geometry_compatible(group["geometry"], ct_geometry)
                for group in groups.values()
            ]
            production_group_acceptance = []
            for group in groups.values():
                production_group_acceptance.append(
                    bool(
                        _geometry_compatible(
                            _GeometryProxy(group["geometry"]), ct_img
                        )
                    )
                )

            legacy_accepts = bool(legacy_group_acceptance) and all(
                legacy_group_acceptance
            )
            production_accepts = bool(production_group_acceptance) and all(
                production_group_acceptance
            )
            row.update(
                {
                    "status": "evaluated",
                    "distinct_mask_geometry_groups": len(groups),
                    "legacy_gate_accepts": legacy_accepts,
                    "production_gate_accepts": production_accepts,
                    "legacy_incompatible_mask_count": sum(
                        group["count"]
                        for group, accepted in zip(
                            groups.values(), legacy_group_acceptance, strict=True
                        )
                        if not accepted
                    ),
                    "production_incompatible_mask_count": sum(
                        group["count"]
                        for group, accepted in zip(
                            groups.values(), production_group_acceptance, strict=True
                        )
                        if not accepted
                    ),
                }
            )
            aggregate[
                "legacy_gate_accepted_courses"
                if legacy_accepts
                else "legacy_gate_rejected_courses"
            ] += 1
            aggregate[
                "production_gate_accepted_courses"
                if production_accepts
                else "production_gate_rejected_courses"
            ] += 1
        except Exception as exc:
            row["status"] = "header_or_load_failure"
            row["error"] = f"{type(exc).__name__}: {exc}"
            aggregate["header_or_load_failures"] += 1
        per_course.append(row)

    return {
        "record_type": "calculation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "root": str(root),
            "access_mode": "read-only",
            "course_discovery_rule": (
                "two-level directories containing both DICOM/CT and "
                "Segmentation_TotalSegmentator"
            ),
            "candidate_two_level_directories": len(candidate_dirs),
            "expected_courses": expected_courses,
        },
        "implementation": {
            "workspace": str(WORKSPACE),
            "baseline_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "baseline_auto_rtstruct_sha256": _git_blob_sha256(
                "HEAD", "rtpipeline/auto_rtstruct.py"
            ),
            "current_worktree_auto_rtstruct_sha256": _sha256(
                WORKSPACE / "rtpipeline" / "auto_rtstruct.py"
            ),
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "simpleitk_version": sitk.Version_VersionString(),
            "method": (
                "Every available binary-mask header was inspected. Masks were grouped "
                "by complete SimpleITK dimension, size, spacing, origin, and direction, "
                "and each distinct group was evaluated by the production gate. Course "
                "acceptance required every group to pass. The planning NIfTI was matched "
                "to DICOM/CT by SeriesInstanceUID and evaluated separately for every "
                "course as an inference about masks not yet generated."
            ),
        },
        "calculations": aggregate,
        "per_course": per_course,
    }


def _write_record(output: Path, label: str, record: dict[str, Any]) -> None:
    resolved = output.resolve()
    if WORKSPACE.resolve() not in resolved.parents:
        raise ValueError(f"Output must be inside {WORKSPACE}: {resolved}")
    payload: dict[str, Any]
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": 1,
            "purpose": (
                "Byte-preserved cohort calculations for the auto-RTSTRUCT geometry "
                "diagnosis. Each run is labeled as fact or calculation; scientific "
                "interpretation remains in the diagnosis report."
            ),
            "runs": [],
        }
    payload["runs"].append({"label": label, **record})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-courses", type=int, default=122)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = audit(args.root.resolve(), args.expected_courses)
    if args.output is not None:
        _write_record(args.output, args.label, record)
    print(json.dumps({"label": args.label, **record["calculations"]}, sort_keys=True))

    calculations = record["calculations"]
    if calculations["planning_nifti_matched_courses"] != args.expected_courses:
        return 2
    if (
        calculations["selection_failures"]
        or calculations["header_or_load_failures"]
        or calculations["planning_nifti_match_failures"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
