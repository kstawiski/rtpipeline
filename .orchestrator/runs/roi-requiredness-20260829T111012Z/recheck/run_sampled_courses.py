from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_parallel import parallel_radiomics_for_course

WORKSPACE = Path("/umed-projekty/rtpipeline")
CAMPAIGN = Path("/home/konrad/rtpipeline_campaign/kopernik_bladder_v3")
RECHECK = WORKSPACE / ".orchestrator/runs/roi-requiredness-20260829T111012Z/recheck"
STAGE_ROOT = RECHECK / "sampled-courses"
COURSES = (
    ("428073", "2020-05"),
    ("431057", "2020-07"),
    ("475201", "2025-03"),
    ("482967", "2024-11"),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage_course(patient_id: str, course_id: str) -> Path:
    source = CAMPAIGN / "Output" / patient_id / course_id
    staged = STAGE_ROOT / patient_id / course_id
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "DICOM").mkdir(exist_ok=True)
    for name in ("CT", "RTSTRUCT"):
        source_dir = source / "DICOM" / name
        if source_dir.exists():
            (staged / "DICOM" / name).symlink_to(source_dir, target_is_directory=True)
    for name in ("RS_auto.dcm",):
        source_file = source / name
        if source_file.exists():
            (staged / name).symlink_to(source_file)
    source_custom = source / "RS_custom.dcm"
    if source_custom.exists():
        shutil.copy2(source_custom, staged / "RS_custom.dcm")
    source_meta = source / "metadata" / "rs_custom_meta.json"
    if source_meta.exists():
        (staged / "metadata").mkdir(exist_ok=True)
        shutil.copy2(source_meta, staged / "metadata" / source_meta.name)
    return staged


def _config() -> PipelineConfig:
    return PipelineConfig(
        dicom_root=CAMPAIGN / "Input",
        output_root=STAGE_ROOT,
        logs_root=RECHECK / "logs",
        max_workers_override=4,
        radiomics_params_file=WORKSPACE / "rtpipeline/radiomics_params.yaml",
        radiomics_skip_rois=[
            "body",
            "couchsurface",
            "couchinterior",
            "couchexterior",
            "bones",
            "m1",
            "m2",
        ],
        radiomics_max_voxels=1_500_000_000,
        radiomics_min_voxels=64,
        radiomics_thread_limit=1,
        custom_structures_config=WORKSPACE / "rtpipeline/custom_structures_pelvic.yaml",
        resume=False,
    )


def _run_one(patient_id: str, course_id: str) -> dict[str, Any]:
    staged = STAGE_ROOT / patient_id / course_id
    receipt_path = RECHECK / "receipts" / f"{patient_id}_{course_id}.json"
    try:
        outcome = parallel_radiomics_for_course(
            _config(),
            staged,
            max_workers=4,
        )
        output_path = outcome.output_path
        if output_path is None or not output_path.is_file():
            raise RuntimeError(f"course returned {outcome.status.value} without a workbook")
        frame = pd.read_excel(output_path, engine="openpyxl")
        status_values = sorted(
            str(value) for value in frame["radiomics_course_status"].dropna().unique()
        )
        payload: dict[str, Any] = {
            "patient_id": patient_id,
            "course_id": course_id,
            "status": outcome.status.value,
            "detail": outcome.detail,
            "roi_counts": outcome.roi_counts or {},
            "roi_failures": list(outcome.roi_failures),
            "workbook": str(output_path),
            "workbook_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "workbook_rows": int(len(frame)),
            "workbook_status_values": status_values,
        }
    except Exception as exc:
        payload = {
            "patient_id": patient_id,
            "course_id": course_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "detail": str(exc),
        }
    _write_json(receipt_path, payload)
    return payload


def main() -> int:
    os.environ["RTPIPELINE_RADIOMICS_THREAD_LIMIT"] = "1"
    os.environ["RTPIPELINE_RADIOMICS_TASK_TIMEOUT"] = "600"
    os.environ["RTPIPELINE_MAX_WORKERS"] = "4"
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    receipts = RECHECK / "receipts"
    if receipts.exists():
        shutil.rmtree(receipts)
    for patient_id, course_id in COURSES:
        _stage_course(patient_id, course_id)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(COURSES)) as executor:
        futures = {
            executor.submit(_run_one, patient_id, course_id): (patient_id, course_id)
            for patient_id, course_id in COURSES
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    results.sort(key=lambda item: (item["patient_id"], item["course_id"]))
    summary = {
        "configuration": {
            "revision": "efce0145e7d771f4440f62fc2734c2865ab0b61f",
            "implementation": "working tree repair",
            "radiomics_min_voxels": 64,
            "radiomics_max_voxels": 1_500_000_000,
            "radiomics_thread_limit": 1,
            "workers_per_course": 4,
            "custom_structures_config": str(
                WORKSPACE / "rtpipeline/custom_structures_pelvic.yaml"
            ),
        },
        "source_campaign": str(CAMPAIGN),
        "source_access": "read-only through staged symlinks",
        "staged_output_root": str(STAGE_ROOT),
        "courses": results,
    }
    _write_json(RECHECK / "sampled-course-summary.json", summary)
    return 0 if all(item["status"] != "failed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
