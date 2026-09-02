from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from rtpipeline import cli
from rtpipeline import radiomics_parallel as rp
from rtpipeline.radiomics_ct_contract import classify_ct_roi
from rtpipeline.radiomics_resource_guard import (
    DEFAULT_MAX_RESAMPLED_BBOX_VOXELS,
    RESAMPLED_BBOX_LIMIT_CODE,
    estimate_resampled_bounding_box,
    resolve_max_resampled_bbox_voxels,
)
from rtpipeline.roi_requiredness import FAILED_RADIOMICS_RESOURCE_LIMIT


def test_sparse_roi_bound_uses_roi_extent_not_full_image() -> None:
    mask = np.zeros((100, 100, 100), dtype=bool)
    mask[10:12, 20:22, 30:32] = True

    estimate = estimate_resampled_bounding_box(
        mask,
        native_spacing_xyz=(2.0, 2.0, 2.0),
        resampled_spacing_xyz=(1.0, 1.0, 1.0),
        array_axis_to_xyz=(2, 1, 0),
        pad_distance=5,
    )

    assert estimate.native_bbox_shape == (2, 2, 2)
    assert estimate.estimated_resampled_bbox_shape == (14, 14, 14)
    assert estimate.estimated_resampled_bbox_voxels == 2744


def test_resource_limit_resolution_is_positive_and_configurable() -> None:
    assert resolve_max_resampled_bbox_voxels(None) == DEFAULT_MAX_RESAMPLED_BBOX_VOXELS
    assert (
        resolve_max_resampled_bbox_voxels(
            SimpleNamespace(radiomics_max_resampled_bbox_voxels=321)
        )
        == 321
    )
    assert (
        resolve_max_resampled_bbox_voxels(
            SimpleNamespace(radiomics_max_resampled_bbox_voxels=0)
        )
        == DEFAULT_MAX_RESAMPLED_BBOX_VOXELS
    )


def test_measured_m3_extent_exceeds_default_working_crop_limit() -> None:
    mask = np.zeros((276, 456, 155), dtype=bool)
    mask[0, 0, 0] = True
    mask[-1, -1, -1] = True

    estimate = estimate_resampled_bounding_box(
        mask,
        native_spacing_xyz=(0.9765625, 0.9765625, 3.0),
        resampled_spacing_xyz=(1.0, 1.0, 1.0),
        array_axis_to_xyz=(1, 0, 2),
        pad_distance=5,
    )

    assert estimate.estimated_resampled_bbox_shape == (280, 456, 475)
    assert estimate.estimated_resampled_bbox_voxels == 60_648_000
    assert estimate.estimated_resampled_bbox_voxels > DEFAULT_MAX_RESAMPLED_BBOX_VOXELS


def test_cli_accepts_resampled_bbox_limit() -> None:
    args = cli.build_parser().parse_args(
        [
            "--dicom-root",
            "/input",
            "--radiomics-max-resampled-bbox-voxels",
            "123456",
        ]
    )
    assert args.radiomics_max_resampled_bbox_voxels == 123456


def test_parallel_worker_records_oversized_crop_without_extraction(
    monkeypatch,
    tmp_path,
) -> None:
    mask = np.ones((20, 30, 40), dtype=bool)

    class Builder:
        def get_roi_mask_by_name(self, _roi_name):
            return mask

    class Image:
        def GetSpacing(self):
            return (1.0, 1.0, 1.0)

    class Extractor:
        settings = {
            "resampledPixelSpacing": (1.0, 1.0, 1.0),
            "padDistance": 5,
        }

        def execute(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("resource guard must run before extraction")

    task = rp._RoiTask(
        source="Manual",
        rs_path=str(tmp_path / "RS.dcm"),
        roi_name="m3",
        course_dir=str(tmp_path),
        series_uid="series",
        mask_identity="mask",
        stable_roi_identifier="roi",
        decision=classify_ct_roi("Manual", "m3"),
        run_identifier="run",
        code_revision="revision",
        configured_parameter_hashes={
            "primary_resegmented": "configured-primary",
            "sensitivity_raw": "configured-raw",
        },
        effective_parameter_hashes={
            "primary_resegmented": "effective-primary",
            "sensitivity_raw": "effective-raw",
        },
        required=False,
    )
    config = SimpleNamespace(radiomics_max_resampled_bbox_voxels=1_000)
    rp._WORKER_STATE.clear()
    rp._WORKER_STATE.update(
        {
            "img": Image(),
            "extractor": Extractor(),
            "config": config,
            "skip_rois": set(),
            "min_voxels": 1,
            "max_voxels": 1_500_000_000,
            "base_timeout": 600,
        }
    )
    monkeypatch.setattr(rp, "_get_builder", lambda _path: Builder())

    records = rp._extract_one(task)

    assert len(records) == 2
    assert {record["extraction_status"] for record in records} == {"failed"}
    assert {record["extraction_failure_kind"] for record in records} == {
        "resource_limit"
    }
    assert {record["roi_structural_code"] for record in records} == {
        RESAMPLED_BBOX_LIMIT_CODE
    }
    assert {record["estimated_resampled_bbox_voxel_count"] for record in records} == {
        60_000
    }
    assert {record["max_resampled_bbox_voxel_count"] for record in records} == {
        1_000
    }

    rp._write_parallel_roi_ledger(tmp_path, [task], records, extracted=True)
    ledger = json.loads(
        (tmp_path / "metadata" / "radiomics_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    roi_row = ledger["course_roi"][0]
    assert roi_row["reason_code"] == FAILED_RADIOMICS_RESOURCE_LIMIT
    assert roi_row["detail_code"] == RESAMPLED_BBOX_LIMIT_CODE
    assert roi_row["estimated_resampled_bbox_voxel_count"] == 60_000
