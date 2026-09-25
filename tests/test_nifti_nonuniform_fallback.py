"""Planning-CT NIfTI conversion of unsigned 16-bit CTs with non-uniform slice spacing.

dcm2niix equalizes non-uniform slice spacing only for signed 16-bit data, so
an unsigned CT with one 2 mm + 3 mm step inside 5 mm steps got no NIfTI and
its course was quarantined. These tests use synthetic series and the real
dcm2niix binary (``RTPIPELINE_TEST_DCM2NIIX`` or ``dcm2niix`` on PATH).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest

import rtpipeline.nonuniform_ct_conversion as nonuniform
import rtpipeline.segmentation as segmentation
from rtpipeline.config import PipelineConfig
from rtpipeline.course_contract import _ct_provenance
from rtpipeline.nifti_provenance import sidecar_is_complete
from synthetic_ct_series import (
    mixed_z_positions,
    synthetic_stored_values,
    uniform_z_positions,
    write_ct_series,
)

nib = pytest.importorskip("nibabel")


def _dcm2niix_binary() -> str | None:
    configured = os.environ.get("RTPIPELINE_TEST_DCM2NIIX")
    if configured:
        return configured if Path(configured).is_file() else None
    return shutil.which("dcm2niix")


@pytest.fixture
def dcm2niix() -> str:
    binary = _dcm2niix_binary()
    if binary is None:
        pytest.skip("dcm2niix executable unavailable")
    return binary


def _config(tmp_path: Path, binary: str) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
        dcm2niix_cmd=binary,
    )


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _direct_outputs(tmp_path: Path, config: PipelineConfig, series_dir: Path, name: str):
    out = tmp_path / name
    picked = segmentation.run_dcm2niix(config, series_dir, out, recursive_depth=0)
    volumes = sorted(p for p in out.iterdir() if p.name.endswith(".nii.gz"))
    return picked, volumes


def _require_unsigned_equalization_refused(tmp_path, config, series_dir):
    picked, _ = _direct_outputs(tmp_path, config, series_dir, "probe_unsigned")
    if picked is not None:
        pytest.skip("this dcm2niix equalizes unsigned data; the fallback is not reached")


def test_fallback_publishes_the_volume_dcm2niix_writes_for_signed_data(tmp_path, dcm2niix):
    config = _config(tmp_path, dcm2niix)
    z = mixed_z_positions()
    unsigned_dir = tmp_path / "unsigned_ct"
    signed_dir = tmp_path / "signed_ct"
    write_ct_series(unsigned_dir, z, signed=False)
    write_ct_series(signed_dir, z, signed=True)
    _require_unsigned_equalization_refused(tmp_path, config, unsigned_dir)
    source_before = _tree_digest(unsigned_dir)

    nifti_dir = tmp_path / "nifti"
    published = segmentation._ensure_ct_nifti(config, unsigned_dir, nifti_dir, dcm2niix_depth=0)

    assert published is not None and published.is_file()
    assert _tree_digest(unsigned_dir) == source_before
    assert not (nifti_dir / ".tmp_dcm2niix").exists()
    assert sorted(p.name for p in nifti_dir.iterdir()) == sorted(
        [published.name, published.name[:-7] + ".metadata.json"]
    )

    # Same stored values and geometry as signed data: dcm2niix's own equalized volume.
    _, signed_volumes = _direct_outputs(tmp_path, config, signed_dir, "signed_direct")
    signed_eq = [p for p in signed_volumes if "_Eq_" in p.name]
    assert len(signed_eq) == 1
    reference = nib.load(str(signed_eq[0]))
    result = nib.load(str(published))
    assert result.shape == reference.shape
    assert result.shape[2] > len(z)
    assert np.array_equal(result.affine, reference.affine)
    assert result.get_data_dtype() == np.dtype("int16")
    assert np.array_equal(np.asanyarray(result.dataobj), np.asanyarray(reference.dataobj))

    # Independent check of the rescale: the first and last equalized planes lie on
    # acquired slices and hold the stored value + RescaleIntercept.
    values = np.asanyarray(result.dataobj)
    rows, columns = 16, 20
    for plane, index in ((0, 0), (result.shape[2] - 1, len(z) - 1)):
        expected = synthetic_stored_values(rows, columns, index) - 1000
        assert np.array_equal(values[:, ::-1, plane], expected.T)

    sidecar = json.loads((nifti_dir / (published.name[:-7] + ".metadata.json")).read_text())
    assert sidecar_is_complete(sidecar)
    source = _ct_provenance(unsigned_dir)
    for key in ("series_instance_uid", "sop_hash", "geometry"):
        assert sidecar[key] == source[key]
    assert sidecar["nifti_sha256"] == hashlib.sha256(published.read_bytes()).hexdigest()
    conversion = sidecar["nifti_conversion"]
    assert conversion["method"] == nonuniform.CONVERSION_METHOD
    assert conversion["reason"] == nonuniform.CONVERSION_REASON
    assert conversion["slice_step_min_mm"] == 2.0
    assert conversion["slice_step_max_mm"] == 5.0


def test_fallback_record_survives_reuse_without_regeneration(tmp_path, dcm2niix):
    config = _config(tmp_path, dcm2niix)
    ct_dir = tmp_path / "ct"
    write_ct_series(ct_dir, mixed_z_positions(), signed=False)
    _require_unsigned_equalization_refused(tmp_path, config, ct_dir)
    nifti_dir = tmp_path / "nifti"
    first = segmentation._ensure_ct_nifti(config, ct_dir, nifti_dir, dcm2niix_depth=0)
    sidecar_path = nifti_dir / (first.name[:-7] + ".metadata.json")
    before = json.loads(sidecar_path.read_text())

    second = segmentation._ensure_ct_nifti(config, ct_dir, nifti_dir, dcm2niix_depth=0)

    assert second == first
    assert json.loads(sidecar_path.read_text()) == before


def test_fallback_fails_closed_when_a_stored_value_exceeds_signed_range(tmp_path, dcm2niix):
    config = _config(tmp_path, dcm2niix)
    ct_dir = tmp_path / "ct"
    high = synthetic_stored_values(16, 20, 3)
    high[4, 5] = 40000
    write_ct_series(ct_dir, mixed_z_positions(), signed=False, overrides={3: high})
    _require_unsigned_equalization_refused(tmp_path, config, ct_dir)
    nifti_dir = tmp_path / "nifti"

    assert segmentation._ensure_ct_nifti(config, ct_dir, nifti_dir, dcm2niix_depth=0) is None
    assert list(nifti_dir.iterdir()) == []


@pytest.mark.parametrize(
    "signed, z_positions",
    [
        (True, uniform_z_positions()),
        (False, uniform_z_positions()),
        (True, mixed_z_positions()),
    ],
    ids=["signed-uniform", "unsigned-uniform", "signed-mixed"],
)
def test_series_dcm2niix_converts_today_are_unchanged(tmp_path, dcm2niix, monkeypatch, signed, z_positions):
    config = _config(tmp_path, dcm2niix)
    ct_dir = tmp_path / "ct"
    write_ct_series(ct_dir, z_positions, signed=signed)
    picked, _ = _direct_outputs(tmp_path, config, ct_dir, "direct")
    assert picked is not None

    def fallback_must_not_run(*_args, **_kwargs):
        raise AssertionError("fallback used for a series dcm2niix converts")

    monkeypatch.setattr(segmentation, "convert_nonuniform_unsigned_ct", fallback_must_not_run)
    nifti_dir = tmp_path / "nifti"
    published = segmentation._ensure_ct_nifti(config, ct_dir, nifti_dir, dcm2niix_depth=0)

    assert published.read_bytes() == picked.read_bytes()
    sidecar = json.loads((nifti_dir / (published.name[:-7] + ".metadata.json")).read_text())
    assert "nifti_conversion" not in sidecar
    assert set(sidecar) == {
        "study_instance_uid",
        "series_instance_uid",
        "instances",
        "modality",
        "geometry",
        "instance_count",
        "sop_hash",
        "nifti_path",
        "source_directory",
        "nifti_sha256",
        "generated_at",
        "nifti_generated_at",
        "nifti_geometry",
    }
    assert not (nifti_dir / ".tmp_dcm2niix").exists()


def _never_called(*_args):
    raise AssertionError("converter must not run for an ineligible series")


@pytest.mark.parametrize(
    "signed, z_positions, reason",
    [
        (False, uniform_z_positions(), "slice spacing is uniform"),
        (True, mixed_z_positions(), "PixelRepresentation is not an unsigned"),
    ],
)
def test_ineligible_series_do_not_reach_the_converter(tmp_path, caplog, signed, z_positions, reason):
    ct_dir = tmp_path / "ct"
    write_ct_series(ct_dir, z_positions, signed=signed)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    assert nonuniform.convert_nonuniform_unsigned_ct(ct_dir, work_dir, _never_called) is None
    assert reason in caplog.text
    assert list(work_dir.iterdir()) == []


def test_staged_copy_changes_only_pixel_representation(tmp_path):
    ct_dir = tmp_path / "ct"
    write_ct_series(ct_dir, mixed_z_positions(), signed=False)
    files, _evidence = nonuniform._inspect_series(ct_dir)
    staging = tmp_path / "staged"

    nonuniform._stage_signed_copy(files, staging)

    for path, offset in files:
        original = path.read_bytes()
        staged = (staging / path.name).read_bytes()
        assert len(staged) == len(original)
        differing = [i for i, (a, b) in enumerate(zip(original, staged)) if a != b]
        assert differing == [offset]
        original_ds = pydicom.dcmread(str(path))
        staged_ds = pydicom.dcmread(str(staging / path.name))
        assert staged_ds.PixelRepresentation == 1
        assert np.array_equal(
            staged_ds.pixel_array.astype(np.int64), original_ds.pixel_array.astype(np.int64)
        )
        assert staged_ds.RescaleIntercept == original_ds.RescaleIntercept
        assert staged_ds.RescaleSlope == original_ds.RescaleSlope
