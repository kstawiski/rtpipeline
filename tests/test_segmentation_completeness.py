"""Unit tests for `_series_segmentation_ready`'s manifest-completeness check.

Mask-file presence alone used to be treated as "segmentation done" (see git history), which
let a course killed mid-write (some masks copied, not all) be silently accepted as complete
on resume. `manifest.json` is written last, after every mask for a model has been copied, so
it must be present, parseable, and internally consistent with what's on disk before a series
is considered ready to skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import rtpipeline.segmentation as segmentation


MODEL = "total"
BASE_NAME = "series1"


def _write_mask(base_dir: Path, name: str) -> None:
    (base_dir / f"{MODEL}--{name}.nii.gz").write_text("fake mask", encoding="utf-8")


def _write_manifest(base_dir: Path, masks: list[str], model: str = MODEL) -> None:
    manifest = {
        "source_nifti": f"{BASE_NAME}.nii.gz",
        "source_dicom": "/fake/dicom",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "models": [{"model": model, "rtstruct_ok": True, "rtstruct": "", "masks": masks}],
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_complete_course_is_ready(tmp_path):
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    _write_mask(base_dir, "liver")
    _write_mask(base_dir, "spleen")
    masks = sorted(p.name for p in base_dir.glob(f"{MODEL}--*.nii*"))
    _write_manifest(base_dir, masks)

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is True


def test_truncated_course_no_manifest_is_not_ready(tmp_path):
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    # Simulate a run killed mid-write: some masks landed on disk, but the process died
    # before reaching the manifest.json write (which happens last).
    _write_mask(base_dir, "liver")

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is False


def test_corrupt_manifest_is_not_ready(tmp_path):
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    _write_mask(base_dir, "liver")
    (base_dir / "manifest.json").write_text('{"models": [{"model": "total", "masks": ', encoding="utf-8")

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is False


def test_manifest_records_mask_absent_from_disk_is_not_ready(tmp_path):
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    _write_mask(base_dir, "liver")
    # Manifest claims a second mask that never made it to disk.
    _write_manifest(base_dir, [f"{MODEL}--liver.nii.gz", f"{MODEL}--spleen.nii.gz"])

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is False


def test_manifest_missing_this_model_entry_is_not_ready(tmp_path):
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    _write_mask(base_dir, "liver")
    _write_manifest(base_dir, [f"{MODEL}--liver.nii.gz"], model="total_mr")

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is False


def test_no_masks_at_all_is_not_ready_even_with_manifest(tmp_path):
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    _write_manifest(base_dir, [])

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is False


def test_manifest_empty_masks_with_stale_disk_mask_is_not_ready(tmp_path):
    """Vacuous-case regression + the failed-retry-stale-mask double-fault.

    Reproduces the reachable poison state: Run A leaves a partial `{model}--*.nii*` on disk with
    no manifest; Run B re-segments but TotalSegmentator FAILS (ok_nifti False) so masks are NOT
    re-captured (empty entry.masks) yet the rtstruct-only manifest is still written. The stale mask
    from Run A therefore coexists with a manifest whose mask list is empty. `_series_segmentation_ready`
    must treat this as NOT ready (empty mask list must never vacuously satisfy `all(...)`), so the
    course is retried instead of skipped-forever-incomplete. A later successful run's
    `_materialize_masks` (via `_clear_previous_masks`) then clears the stale mask.
    """
    base_dir = tmp_path / "seg"
    base_dir.mkdir()
    _write_mask(base_dir, "liver")  # stale partial mask on disk from a prior killed run
    _write_manifest(base_dir, [])   # failed retry recorded rtstruct-only: masks == []

    assert segmentation._series_segmentation_ready(base_dir, BASE_NAME, MODEL) is False


def test_write_manifest_atomic_leaves_no_partial_file(tmp_path):
    path = tmp_path / "manifest.json"
    data = {"models": [{"model": MODEL, "masks": ["total--liver.nii.gz"]}]}

    segmentation._write_manifest_atomic(path, data)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == data
    leftover_tmp_files = list(tmp_path.glob(".*.tmp"))
    assert leftover_tmp_files == []


def test_write_manifest_atomic_uses_unique_pid_temp(tmp_path):
    """The temp name is PID-qualified so concurrent same-dir writers cannot collide on a fixed name."""
    path = tmp_path / "manifest.json"
    captured: dict = {}
    real_replace = os.replace

    def spy_replace(src, dst):
        captured["src"] = str(src)
        return real_replace(src, dst)

    import unittest.mock as mock
    with mock.patch.object(segmentation.os, "replace", spy_replace):
        segmentation._write_manifest_atomic(path, {"models": []})

    assert str(os.getpid()) in Path(captured["src"]).name


def test_write_manifest_atomic_removes_temp_on_failure(tmp_path):
    """A crash between temp-write and rename leaves NO leftover temp and does NOT create the target."""
    path = tmp_path / "manifest.json"

    import unittest.mock as mock
    with mock.patch.object(segmentation.os, "replace", side_effect=OSError("simulated crash")):
        try:
            segmentation._write_manifest_atomic(path, {"models": []})
        except OSError:
            pass

    assert not path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob("*.tmp")) == []
