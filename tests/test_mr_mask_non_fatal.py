"""An empty auto MR mask must not void a course's radiomics.

TotalSegmentator's MR model emits its full structure list regardless of field of
view, so a pelvic MR legitimately produces empty brain/lung/liver masks.
_collect_total_mr_masks used to raise on the first empty one, which aborted
radiomics for the whole course AFTER the CT parquet had been written, leaving
.radiomics_done unwritten and the pipeline unable to finish. The module already
classifies this source as not required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtpipeline.radiomics import MR_AUTO_SOURCE, _collect_total_mr_masks
from rtpipeline.radiomics_outcomes import roi_source_is_required


def _write_mask(path: Path, filled: bool) -> Path:
    sitk = pytest.importorskip("SimpleITK")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    if filled:
        arr[1:3, 1:3, 1:3] = 1
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))
    return path


def test_auto_mr_source_is_not_required() -> None:
    assert roi_source_is_required(MR_AUTO_SOURCE) is False


def test_empty_mr_mask_is_skipped_not_fatal(tmp_path: Path) -> None:
    seg = tmp_path / "Segmentation_TotalSegmentator"
    _write_mask(seg / "total_mr--brain.nii.gz", filled=False)
    _write_mask(seg / "total_mr--urinary_bladder.nii.gz", filled=True)
    failures: list[dict[str, str]] = []
    masks = _collect_total_mr_masks(tmp_path / "series", seg, failures)
    assert "urinary_bladder" in masks, "the populated mask must still be extracted"
    assert "brain" not in masks
    assert any(f["roi_name"].startswith("total_mr--brain") for f in failures), (
        "the skipped empty mask must be recorded, not silently dropped"
    )
    assert failures[0]["source"] == MR_AUTO_SOURCE


def test_all_empty_masks_yield_no_masks_without_raising(tmp_path: Path) -> None:
    seg = tmp_path / "Segmentation_TotalSegmentator"
    for organ in ("brain", "lung_left", "liver"):
        _write_mask(seg / f"total_mr--{organ}.nii.gz", filled=False)
    failures: list[dict[str, str]] = []
    masks = _collect_total_mr_masks(tmp_path / "series", seg, failures)
    assert masks == {}
    assert len(failures) == 3


def test_unreadable_mr_mask_is_recorded_not_fatal(tmp_path: Path) -> None:
    seg = tmp_path / "Segmentation_TotalSegmentator"
    seg.mkdir(parents=True, exist_ok=True)
    (seg / "total_mr--corrupt.nii.gz").write_bytes(b"not a nifti")
    failures: list[dict[str, str]] = []
    masks = _collect_total_mr_masks(tmp_path / "series", seg, failures)
    assert masks == {}
    assert failures and failures[0]["status"] == "failed"
