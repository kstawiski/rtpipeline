"""Regression test: segmentation materialization writes a per-series/model
TotalSegmentator version sidecar into the final Segmentation_TotalSegmentator
dir, so cohort-wide TS-version uniformity is auditable from the outputs rather
than assumed from the run environment. The write is fail-soft and must never
abort segmentation.
"""
import json

from rtpipeline.segmentation import (
    _materialize_masks,
    _totalseg_version,
    _write_ts_version_sidecar,
)


def test_version_helper_returns_string():
    v = _totalseg_version()
    assert isinstance(v, str) and v  # installed version or "unknown"


def test_materialize_masks_writes_version_sidecar(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    # A single loose mask the final copy loop will pick up.
    (source / "liver.nii.gz").write_bytes(b"\x1f\x8b\x08\x00")  # gzip magic; content irrelevant
    dest = tmp_path / "Segmentation_TotalSegmentator"

    _materialize_masks(source, dest, base_name="CT", model="total")

    sidecar = dest / "total--ts_version.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["model"] == "total"
    assert isinstance(data["totalsegmentator_version"], str) and data["totalsegmentator_version"]
    assert "written_utc" in data
    # The mask itself was materialized with the model-prefixed name.
    assert (dest / "total--liver.nii.gz").exists()


def test_sidecar_writer_is_fail_soft_on_bad_dest(tmp_path):
    # dest is a FILE, so mkdir/write would fail inside; the writer must swallow it.
    bad = tmp_path / "not_a_dir"
    bad.write_text("x")
    # Should not raise.
    _write_ts_version_sidecar(bad, "total")


def test_distinct_models_get_distinct_sidecars(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "brain.nii.gz").write_bytes(b"\x1f\x8b\x08\x00")
    dest = tmp_path / "seg"
    _materialize_masks(source, dest, base_name="MR", model="total")
    _materialize_masks(source, dest, base_name="MR", model="total_mr")
    assert (dest / "total--ts_version.json").exists()
    assert (dest / "total_mr--ts_version.json").exists()
