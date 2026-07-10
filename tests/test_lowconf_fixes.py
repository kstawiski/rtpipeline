"""Regression tests for the low-confidence bug-fix batch.

5 - dicom_copy.DicomCopyManager._try_hardlink: unlinked the destination before
    creating the hardlink, so a crash in that window destroyed an existing file
    (e.g. a dose file) with no recovery. Fixed by linking to a temp path in the
    same directory and publishing via os.replace() (atomic; dst is never absent).
1 - radiomics_parallel._prepare_radiomics_task: cached the per-image temp NRRD
    path under id(image), which CPython can reuse for an unrelated later object,
    risking a stale on-disk cache hit. Fixed by keying on a content hash instead.
6 - organize._sum_doses_with_resample: the reference dose grid's X/Y physical
    positions only used the axis-aligned direction-cosine term, dropping the
    cross term from the other in-plane index. An oblique-but-consistent grid
    (in-plane rotation) therefore resampled to the wrong physical position.
    Fixed by including both row- and column-index contributions to X and Y
    (mirroring the full-vector Z-axis treatment already in the file).
4 - custom_models._load_named_binary_masks: silently returned a partial mask
    set (fewer masks than requested labels) with no signal. Fixed by warning
    loudly and reporting the missing labels back to the caller.
2 - custom_models._structure_filename / _norm_label_token: two distinct label
    names can sanitize to the same on-disk token, so the second mask write
    would silently clobber the first. Fixed by failing closed at config/model
    load time via `_check_label_name_collisions`.
7 - nnunetv2_modelfolder.run_modelfolder_prediction: os.environ.update()
    mutated the real process environment with no restoration, leaking across
    sequential calls in a reused pool worker. Fixed via snapshot + restore in
    a finally (through a `_temporary_env` context manager).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, RTPlanStorage, generate_uid

from rtpipeline.dicom_copy import DicomCopyConfig, DicomCopyManager
from rtpipeline.radiomics_parallel import _prepare_radiomics_task
from rtpipeline.organize import _sum_doses_with_resample
from rtpipeline.custom_models import (
    _check_label_name_collisions,
    _load_named_binary_masks,
    _parse_custom_model,
)
from rtpipeline.segmenters.nnunetv2_modelfolder import _temporary_env


# ---------------------------------------------------------------------------
# 5 - dicom_copy._try_hardlink atomic replace
# ---------------------------------------------------------------------------

def _mk_copy_manager(tmp_path: Path) -> DicomCopyManager:
    return DicomCopyManager(DicomCopyConfig(use_hardlinks=True), tmp_path / "out")


def test_hardlink_into_existing_dst_never_leaves_it_absent(tmp_path: Path, monkeypatch) -> None:
    mgr = _mk_copy_manager(tmp_path)
    src = tmp_path / "src.dcm"
    src.write_bytes(b"NEW-DOSE-BYTES")
    dst = tmp_path / "RD.dcm"
    dst.write_bytes(b"OLD-DOSE-BYTES")  # simulates a pre-existing dose file

    replace_calls = []
    real_replace = os.replace

    def spy_replace(a, b):
        # At the moment the atomic publish happens, dst must still hold its
        # OLD content -- proving it was never unlinked ahead of time.
        replace_calls.append(dst.exists() and dst.read_bytes() == b"OLD-DOSE-BYTES")
        return real_replace(a, b)

    monkeypatch.setattr("rtpipeline.dicom_copy.os.replace", spy_replace)

    ok = mgr._try_hardlink(src, dst)

    assert ok is True
    assert replace_calls == [True]
    assert dst.exists()
    assert os.stat(dst).st_ino == os.stat(src).st_ino  # correctly linked to src
    leftovers = list(tmp_path.glob(".RD.dcm.*.tmp"))
    assert leftovers == []


def test_hardlink_crash_before_replace_leaves_dst_intact(tmp_path: Path, monkeypatch) -> None:
    """Simulates a failure at the latest possible point (the publish step).

    Under the pre-fix code, dst.unlink() happened up front, so any failure
    after that point (including an os.link failure) permanently destroyed
    dst. Under the fix, dst is untouched until the atomic replace, so even a
    failure there leaves the original file intact.
    """
    mgr = _mk_copy_manager(tmp_path)
    src = tmp_path / "src.dcm"
    src.write_bytes(b"NEW")
    dst = tmp_path / "RD.dcm"
    dst.write_bytes(b"OLD")

    def boom(_a, _b):
        raise OSError("simulated crash before publish")

    monkeypatch.setattr("rtpipeline.dicom_copy.os.replace", boom)

    ok = mgr._try_hardlink(src, dst)

    assert ok is False
    assert dst.exists()
    assert dst.read_bytes() == b"OLD"


def test_hardlink_falls_back_to_copy_on_link_failure(tmp_path: Path, monkeypatch) -> None:
    mgr = _mk_copy_manager(tmp_path)
    src = tmp_path / "src.dcm"
    src.write_bytes(b"CONTENT")
    dst = tmp_path / "RD.dcm"
    dst.write_bytes(b"PRIOR")

    def boom(_src, _dst):
        raise OSError(18, "Invalid cross-device link")  # EXDEV

    monkeypatch.setattr("rtpipeline.dicom_copy.os.link", boom)

    ok = mgr._try_hardlink(src, dst)

    assert ok is False
    # _try_hardlink itself must not have touched a pre-existing dst; the
    # caller (copy_dicom) is responsible for the actual copy fallback.
    assert dst.read_bytes() == b"PRIOR"


# ---------------------------------------------------------------------------
# 1 - radiomics_parallel image cache key
# ---------------------------------------------------------------------------

def test_prepare_radiomics_task_image_key_is_content_derived(tmp_path: Path) -> None:
    """Two distinct (different id()) SimpleITK images with identical content must
    reuse the same cache file, and the img_key must not simply be id(image)."""
    def mk_image(fill_value: int) -> sitk.Image:
        arr = np.full((4, 4, 4), fill_value, dtype=np.int16)
        return sitk.GetImageFromArray(arr)

    def mk_mask() -> sitk.Image:
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        return sitk.GetImageFromArray(arr)

    img_a = mk_image(7)
    img_b = mk_image(7)  # distinct object, identical content
    assert id(img_a) != id(img_b)

    course_dir = tmp_path / "P1" / "2024-01"
    course_dir.mkdir(parents=True)

    _, params_a = _prepare_radiomics_task(
        img_a, mk_mask(), None, "source_a", "ROI", course_dir, tmp_path, False
    )
    _, params_b = _prepare_radiomics_task(
        img_b, mk_mask(), None, "source_b", "ROI", course_dir, tmp_path, False
    )

    assert params_a["image_path"] == params_b["image_path"]
    expected_key = hashlib.sha1(
        sitk.GetArrayViewFromImage(img_a).tobytes()
    ).hexdigest()
    assert Path(params_a["image_path"]).name == f"img_{expected_key}.nrrd"


def test_prepare_radiomics_task_different_content_never_collides(tmp_path: Path) -> None:
    """Different-content images must always get distinct cache files, and each
    file must contain exactly its own image's data (no stale reuse)."""
    def mk_image(fill_value: int) -> sitk.Image:
        arr = np.full((3, 3, 3), fill_value, dtype=np.int16)
        return sitk.GetImageFromArray(arr)

    def mk_mask() -> sitk.Image:
        arr = np.ones((3, 3, 3), dtype=np.uint8)
        return sitk.GetImageFromArray(arr)

    course_dir = tmp_path / "P1" / "2024-01"
    course_dir.mkdir(parents=True)

    img1 = mk_image(1)
    img2 = mk_image(2)

    _, params1 = _prepare_radiomics_task(
        img1, mk_mask(), None, "s1", "ROI1", course_dir, tmp_path, False
    )
    _, params2 = _prepare_radiomics_task(
        img2, mk_mask(), None, "s2", "ROI2", course_dir, tmp_path, False
    )

    assert params1["image_path"] != params2["image_path"]
    read_back_1 = sitk.GetArrayFromImage(sitk.ReadImage(params1["image_path"]))
    read_back_2 = sitk.GetArrayFromImage(sitk.ReadImage(params2["image_path"]))
    assert (read_back_1 == 1).all()
    assert (read_back_2 == 2).all()


# ---------------------------------------------------------------------------
# 6 - organize._sum_doses_with_resample oblique in-plane cross terms
# ---------------------------------------------------------------------------

def _mk_dose_dataset(
    path: Path,
    *,
    rows: int,
    cols: int,
    frames: int,
    pixel_spacing: List[float],
    origin: List[float],
    iop: List[float],
    offsets: List[float],
    values: np.ndarray,
) -> Path:
    meta = FileMetaDataset()
    sop_uid = generate_uid()
    meta.MediaStorageSOPClassUID = RTDoseStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = sop_uid
    ds.PatientID = "P1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "RTDOSE"
    ds.FrameOfReferenceUID = "1.2.3"
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = frames
    ds.PixelSpacing = list(pixel_spacing)
    ds.ImagePositionPatient = list(origin)
    ds.ImageOrientationPatient = list(iop)
    ds.GridFrameOffsetVector = list(offsets)
    ds.DoseGridScaling = 1.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = values.astype(np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)
    return path


def test_sum_doses_with_resample_handles_oblique_reference_in_plane_rotation(tmp_path: Path) -> None:
    """Reference grid is in-plane rotated 90 degrees:
      row_cosines_ref (u) = [0, 1, 0]   -> column index increases along +Y
      col_cosines_ref (v) = [-1, 0, 0]  -> row index increases along -X
    So physical position(r, c) = (X=-r, Y=c, Z=frame).

    Under the pre-fix code (which drops the cross term), every reference voxel
    would resolve to the SAME constant (X=0, Y=0) regardless of (r, c), so the
    entire resampled frame would collapse to a single source voxel's value.
    """
    rows_ref = cols_ref = 3
    frames = 2
    ref_path = _mk_dose_dataset(
        tmp_path / "ref.dcm",
        rows=rows_ref, cols=cols_ref, frames=frames,
        pixel_spacing=[1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        iop=[0, 1, 0, -1, 0, 0],
        offsets=[0.0, 1.0],
        values=np.zeros((frames, rows_ref, cols_ref)),
    )

    # Axis-aligned source, offset so the oblique reference maps fully in-bounds.
    # value(frame, row, col) = 100*frame + 10*row + col (unique per voxel).
    src_rows = src_cols = 5
    src_vals = np.zeros((frames, src_rows, src_cols))
    for f in range(frames):
        for row in range(src_rows):
            for col in range(src_cols):
                src_vals[f, row, col] = 100 * f + 10 * row + col
    src_path = _mk_dose_dataset(
        tmp_path / "src.dcm",
        rows=src_rows, cols=src_cols, frames=frames,
        pixel_spacing=[1.0, 1.0],
        origin=[-2.0, 0.0, 0.0],
        iop=[1, 0, 0, 0, 1, 0],
        offsets=[0.0, 1.0],
        values=src_vals,
    )

    plan_sum = Dataset()
    plan_sum.SOPClassUID = RTPlanStorage
    plan_sum.SOPInstanceUID = generate_uid()

    new_ds, _, _ = _sum_doses_with_resample([ref_path, src_path], plan_sum, [])

    result = new_ds.pixel_array.astype("float64") * float(new_ds.DoseGridScaling)

    for f in range(frames):
        for r in range(rows_ref):
            for c in range(cols_ref):
                # source row = c, source col = 2 - r (see derivation above)
                expected = 100 * f + 10 * c + (2 - r)
                assert result[f, r, c] == pytest.approx(expected, abs=0.5), (f, r, c)


# ---------------------------------------------------------------------------
# 4 - custom_models._load_named_binary_masks partial-mask warning
# ---------------------------------------------------------------------------

def _write_nifti_mask(path: Path, value: int = 1) -> None:
    arr = np.full((2, 2, 2), value, dtype=np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


def test_load_named_binary_masks_full_match_unaffected(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_nifti_mask(out_dir / "bladder.nii.gz")
    _write_nifti_mask(out_dir / "rectum.nii.gz")

    masks, missing = _load_named_binary_masks(out_dir, ["bladder", "rectum"])

    assert set(masks) == {"bladder", "rectum"}
    assert missing == []


def test_load_named_binary_masks_partial_match_warns_and_reports_missing(
    tmp_path: Path, caplog
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_nifti_mask(out_dir / "bladder.nii.gz")
    # "rectum" is intentionally missing from output_dir.

    with caplog.at_level("WARNING"):
        masks, missing = _load_named_binary_masks(out_dir, ["bladder", "rectum"])

    assert set(masks) == {"bladder"}
    assert missing == ["rectum"]
    assert any("rectum" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 2 - custom_models label-name sanitized-filename collisions
# ---------------------------------------------------------------------------

def test_check_label_name_collisions_raises_on_output_filename_collision() -> None:
    # "Bladder Wall" and "Bladder-Wall" both sanitize to "Bladder_Wall".
    with pytest.raises(ValueError, match="normalize to the same"):
        _check_label_name_collisions("Network x", ["Bladder Wall", "Bladder-Wall"])


def test_check_label_name_collisions_raises_on_lookup_token_collision() -> None:
    # "Bladder_Wall" and "bladder_wall" differ only by case, which
    # _norm_label_token folds away (even though _structure_filename keeps them apart).
    with pytest.raises(ValueError, match="normalize to the same"):
        _check_label_name_collisions("Network x", ["Bladder_Wall", "bladder_wall"])


def test_check_label_name_collisions_allows_distinct_names() -> None:
    _check_label_name_collisions("Network x", ["Bladder", "Rectum", "Femur_L", "Femur_R"])


def test_parse_custom_model_fails_closed_on_colliding_network_labels(tmp_path: Path) -> None:
    data = {
        "name": "m",
        "nnunet": {
            "networks": [
                {
                    "id": "Dataset001",
                    "label_order": ["Bladder Wall", "Bladder-Wall"],
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="normalize to the same"):
        _parse_custom_model(tmp_path, data, "nnUNetv2_predict")


# ---------------------------------------------------------------------------
# 7 - nnunetv2_modelfolder os.environ leak
# ---------------------------------------------------------------------------

def test_temporary_env_restores_prior_values_and_absent_keys(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "original")
    monkeypatch.delenv("ABSENT_KEY", raising=False)

    with _temporary_env({"EXISTING_KEY": "overridden", "ABSENT_KEY": "temp"}):
        assert os.environ["EXISTING_KEY"] == "overridden"
        assert os.environ["ABSENT_KEY"] == "temp"

    assert os.environ["EXISTING_KEY"] == "original"
    assert "ABSENT_KEY" not in os.environ


def test_temporary_env_restores_on_exception(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "original")
    monkeypatch.delenv("ABSENT_KEY", raising=False)

    with pytest.raises(RuntimeError):
        with _temporary_env({"EXISTING_KEY": "overridden", "ABSENT_KEY": "temp"}):
            raise RuntimeError("boom")

    assert os.environ["EXISTING_KEY"] == "original"
    assert "ABSENT_KEY" not in os.environ
