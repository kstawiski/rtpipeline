"""B2 per-structure PET SUV tests. Pure-function + synthetic-orchestrator coverage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline.pet_structures import (
    MIN_VOLUME_VOXELS,
    QC_AMBIGUOUS_PETCT,
    QC_EMPTY_MASK,
    QC_MASKS_MISSING,
    QC_NO_PETCT,
    QC_SUV_MISSING,
    pair_petct_ct,
    per_structure_suv,
    read_total_ct_label_image,
    resample_mask_to_suv_grid,
    sample_patient_pet_suv,
    suvpeak_sphere,
    write_pet_suv_structures_csv,
)


def _img(arr: np.ndarray, *, origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0), dtype=np.float32) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(dtype))  # (z,y,x)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    return img


# ----------------------------------------------------- per-structure stats

def test_suvmax_mean_exact():
    suv = _img(np.array([[[0, 1, 2, 3, 4, 5]]], dtype=np.float32))
    mask = _img(np.ones((1, 1, 6), dtype=np.int16), dtype=np.int16)
    rows = per_structure_suv(suv, sitk.GetArrayFromImage(mask), sitk.GetArrayFromImage(mask),
                             {1: "roi"}, ct_voxel_vol_ml=0.008)
    r = rows[0]
    assert r["suvmax"] == pytest.approx(5.0)
    assert r["suvmean"] == pytest.approx(np.mean([0, 1, 2, 3, 4, 5]))
    assert r["n_suv_voxels"] == 6
    assert r["volume_ml"] == pytest.approx(6 * 0.008)
    assert r["qc_flag"] == "suvpeak_sphere_truncated"
    assert r["suvpeak"] is None


def test_suvpeak_uniform_field_equals_value():
    suv = _img(np.full((9, 9, 9), 4.0, dtype=np.float32))
    out = suvpeak_sphere(sitk.GetArrayFromImage(suv), suv, (4, 4, 4))
    assert out == pytest.approx(4.0)  # uniform field → sphere mean = value


def test_suvpeak_single_hot_voxel_is_sphere_mean_not_max():
    arr = np.zeros((9, 9, 9), dtype=np.float32)
    arr[4, 4, 4] = 100.0  # single hot voxel in a zero field
    suv = _img(arr)
    peak = suvpeak_sphere(arr, suv, (4, 4, 4))
    assert 0.0 < peak < 100.0  # PERCIST sphere averages neighbors (incl. zeros) → < max


def test_suvpeak_not_clipped_to_structure():
    # hot value present both in- and out-of-structure; sphere (the VOI) samples beyond the label.
    arr = np.full((9, 9, 9), 7.0, dtype=np.float32)
    suv = _img(arr)
    mask = np.zeros((9, 9, 9), dtype=np.int16)
    mask[4, 4, 4] = 1  # single-voxel structure
    rows = per_structure_suv(suv, mask, mask, {1: "roi"}, ct_voxel_vol_ml=0.008)
    assert rows[0]["suvpeak"] == pytest.approx(7.0)  # sphere extends past the 1-voxel mask → still 7
    assert rows[0]["min_volume_flag"] is True  # 1 < MIN_VOLUME_VOXELS


def test_empty_structure_qc():
    suv = _img(np.ones((4, 4, 4), dtype=np.float32))
    mask = np.zeros((4, 4, 4), dtype=np.int16)  # label 1 in map, no voxels
    rows = per_structure_suv(suv, mask, mask, {1: "roi"}, ct_voxel_vol_ml=0.008)
    assert rows[0]["qc_flag"] == QC_EMPTY_MASK and rows[0]["n_suv_voxels"] == 0
    assert 1 < MIN_VOLUME_VOXELS


def test_resample_mask_to_suv_grid_nn():
    ct_mask = _img(np.ones((20, 20, 20), dtype=np.int16), spacing=(1.0, 1.0, 1.0), dtype=np.int16)
    suv = _img(np.zeros((10, 10, 10), dtype=np.float32), spacing=(2.0, 2.0, 2.0))  # coarser, same FoR
    out = resample_mask_to_suv_grid(ct_mask, suv)
    assert out.GetSize() == (10, 10, 10)
    assert int(sitk.GetArrayFromImage(out).sum()) > 0  # label landed on the SUV grid


# ----------------------------------------------------- pairing

def test_pair_one():
    pt = {"study_uid": "S", "frame_of_reference_uid": "F"}
    cands = [{"series_uid": "C", "study_uid": "S", "frame_of_reference_uid": "F", "n_slices": 200}]
    chosen, basis, qc = pair_petct_ct(pt, cands)
    assert chosen["series_uid"] == "C" and qc is None


def test_pair_none():
    pt = {"study_uid": "S", "frame_of_reference_uid": "F"}
    chosen, basis, qc = pair_petct_ct(pt, [{"series_uid": "C", "study_uid": "OTHER", "frame_of_reference_uid": "F"}])
    assert chosen is None and qc == QC_NO_PETCT


def test_pair_ambiguous_tie():
    pt = {"study_uid": "S", "frame_of_reference_uid": "F"}
    cands = [{"series_uid": "A", "study_uid": "S", "frame_of_reference_uid": "F", "n_slices": 200},
             {"series_uid": "B", "study_uid": "S", "frame_of_reference_uid": "F", "n_slices": 200}]
    chosen, basis, qc = pair_petct_ct(pt, cands)
    assert chosen is None and qc == QC_AMBIGUOUS_PETCT  # tied → never silently pick


def test_pair_multiple_candidates_are_ambiguous_even_when_slice_counts_differ():
    pt = {"study_uid": "S", "frame_of_reference_uid": "F"}
    cands = [{"series_uid": "A", "study_uid": "S", "frame_of_reference_uid": "F", "n_slices": 100},
             {"series_uid": "B", "study_uid": "S", "frame_of_reference_uid": "F", "n_slices": 300}]
    chosen, basis, qc = pair_petct_ct(pt, cands)
    assert chosen is None and basis == "multiple_study_for" and qc == QC_AMBIGUOUS_PETCT


def test_pair_empty_for():
    pt = {"study_uid": "S", "frame_of_reference_uid": ""}  # missing FoR
    chosen, basis, qc = pair_petct_ct(pt, [{"series_uid": "C", "study_uid": "S", "frame_of_reference_uid": ""}])
    assert chosen is None and qc == QC_NO_PETCT


# ----------------------------------------------------- mask reader

def test_read_total_ct_label_image(tmp_path):
    arr = np.zeros((4, 4, 4), dtype=np.int16); arr[0:2] = 1; arr[2:4] = 2
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(tmp_path / "total--multilabel.nii.gz"))
    (tmp_path / "total--segmentations.json").write_text(json.dumps({"labels": {"1": "liver", "2": "spleen"}}))
    img, lm = read_total_ct_label_image(tmp_path)
    assert img is not None and lm == {1: "liver", 2: "spleen"}


# ----------------------------------------------------- orchestrator (integration)

def _build_pet_patient(out_root: Path, *, petct_for="F1", n_petct=1, tie=False):
    man_dir = out_root / "PT1" / "all_series" / "metadata"; man_dir.mkdir(parents=True)
    pt_dir = out_root / "PT1" / "all_series" / "pt" / "uidPT" / "DICOM"; pt_dir.mkdir(parents=True)
    series = [{"series_uid": "uidPT", "image_class": "pt", "series_description": "PET WB",
               "study_uid": "S1", "frame_of_reference_uid": "F1", "status": "suv_computed",
               "output_dir": str(pt_dir)}]
    for i in range(n_petct):
        ct_dir = out_root / "PT1" / "all_series" / "petct_ct" / f"uidCT{i}" / "DICOM"; ct_dir.mkdir(parents=True)
        (ct_dir.parent / "Segmentation_TotalSegmentator").mkdir()
        series.append({"series_uid": f"uidCT{i}", "image_class": "petct_ct", "series_description": "CTAC",
                       "study_uid": "S1", "frame_of_reference_uid": petct_for,
                       "n_slices": 200 if tie else (200 + i * 100), "output_dir": str(ct_dir)})
    (man_dir / "series_manifest.json").write_text(json.dumps({"series": series}))  # production key
    return out_root / "PT1" / "all_series"


def _synth_suv_mask():
    suv = _img(np.arange(1000, dtype=np.float32).reshape(10, 10, 10))
    mask = _img((np.indices((10, 10, 10)).sum(0) < 6).astype(np.int16), dtype=np.int16)
    return (lambda p: suv), (lambda d: (mask, {1: "roi"}))


def test_orchestrator_ok_path(tmp_path):
    out = tmp_path / "out"
    asr = _build_pet_patient(out)
    suv_fn, mask_fn = _synth_suv_mask()
    summary = sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    assert summary["n_pet"] == 1 and summary["n_sampled"] == 1
    sidecar = asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json"
    payload = json.loads(sidecar.read_text())
    assert payload["series_qc"] == "ok"
    assert payload["rows"][0]["petct_ct_series_uid"] == "uidCT0"
    assert payload["rows"][0]["suvmax"] is not None and payload["rows"][0]["volume_ml"] is not None


def test_orchestrator_no_petct(tmp_path):
    out = tmp_path / "out"
    asr = _build_pet_patient(out, petct_for="OTHER")  # CT FoR mismatch → no pair
    suv_fn, mask_fn = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    payload = json.loads((asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json").read_text())
    assert payload["series_qc"] == QC_NO_PETCT


def test_orchestrator_suv_missing(tmp_path):
    out = tmp_path / "out"
    asr = _build_pet_patient(out)
    _, mask_fn = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=lambda p: None, _mask_fn=mask_fn)  # SUV nifti absent
    payload = json.loads((asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json").read_text())
    assert payload["series_qc"] == QC_SUV_MISSING


def test_orchestrator_masks_missing(tmp_path):
    out = tmp_path / "out"
    asr = _build_pet_patient(out)
    suv_fn, _ = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=lambda d: (None, {}))
    payload = json.loads((asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json").read_text())
    assert payload["series_qc"] == QC_MASKS_MISSING


def test_orchestrator_ambiguous_petct(tmp_path):
    out = tmp_path / "out"
    asr = _build_pet_patient(out, n_petct=2, tie=True)  # two tied petct_ct
    suv_fn, mask_fn = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    payload = json.loads((asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json").read_text())
    assert payload["series_qc"] == QC_AMBIGUOUS_PETCT


def test_orchestrator_completeness_and_csv(tmp_path):
    out = tmp_path / "out"
    _build_pet_patient(out)
    suv_fn, mask_fn = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    csv_path = write_pet_suv_structures_csv(out)
    import csv as _csv
    with csv_path.open() as f:
        uids = {r["pet_series_uid"] for r in _csv.DictReader(f)}
    assert uids == {"uidPT"}  # every manifest PET series present in CSV (no silent drop)


def test_orchestrator_existing_sidecar_is_refreshed(tmp_path):
    out = tmp_path / "out"
    _build_pet_patient(out)
    suv_fn, mask_fn = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    s2 = sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    assert s2["n_sampled"] == 1


def test_csv_unreadable_logged(tmp_path, caplog):
    import logging
    out = tmp_path / "out"
    d = out / "PT1" / "all_series" / "NIFTI" / "SUV" / "uidPT"; d.mkdir(parents=True)
    (d / "pet_suv_structures.json").write_text("{ not valid json")
    with caplog.at_level(logging.ERROR):
        assert write_pet_suv_structures_csv(out) is None
    assert any("UNREADABLE" in r.message for r in caplog.records)


def test_config_default_off():
    import dataclasses
    from rtpipeline.config import PipelineConfig
    fld = {f.name: f for f in dataclasses.fields(PipelineConfig)}["pet_suv_structures"]
    assert fld.default is False


# ----------------------------------------------- impl-gate R1 remediation tests

def test_suvpeak_sphere_asymmetric_exact():
    # Non-cubic (z=4,y=6,x=8), anisotropic spacing, hot voxel at an asymmetric index. A z<->x
    # swap of the center would mis-map / go out of range. Tiny radius → sphere = center voxel.
    arr = np.zeros((4, 6, 8), dtype=np.float32)
    arr[3, 4, 7] = 42.0  # numpy (z=3,y=4,x=7)
    suv = _img(arr, spacing=(3.0, 2.0, 1.0))  # sitk (sx=3,sy=2,sz=1)
    assert suvpeak_sphere(arr, suv, (7, 4, 3), radius_mm=0.4) == pytest.approx(42.0)  # exact center
    assert suvpeak_sphere(arr, suv, (7, 4, 3)) is None  # default 1-cm3 sphere crosses image boundary


def test_per_structure_suv_hottest_voxel_index_ordering():
    # catches a numpy(z,y,x)->sitk(x,y,z) transposition: a swap would center the sphere away from
    # the hot voxel (→ peak ≈ background) or IndexError; correct code includes the 100 (peak>1).
    arr = np.ones((4, 6, 8), dtype=np.float32)
    arr[3, 4, 7] = 100.0
    suv = _img(arr, spacing=(2.0, 2.0, 2.0))
    mask = np.ones((4, 6, 8), dtype=np.int16)
    rows = per_structure_suv(suv, mask, mask, {1: "roi"}, ct_voxel_vol_ml=0.001)
    assert rows[0]["suvmax"] == pytest.approx(100.0)
    assert rows[0]["suvpeak"] is None
    assert rows[0]["qc_flag"] == "suvpeak_sphere_truncated"


def test_resample_mask_label_placement():
    # spatially-split labels (low-z=1, high-z=2) on a fine CT grid → coarse SUV grid (same FoR);
    # assert each label lands in the correct SUV half (all-ones masks can't prove placement).
    ct = np.zeros((20, 4, 4), dtype=np.int16)  # (z=20,y,x)
    ct[:10] = 1; ct[10:] = 2
    ct_img = _img(ct, spacing=(1.0, 1.0, 1.0), dtype=np.int16)
    suv = _img(np.zeros((10, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 2.0))  # coarse z, same extent/FoR
    oa = sitk.GetArrayFromImage(resample_mask_to_suv_grid(ct_img, suv))  # (z=10,y,x)
    assert (oa[0] == 1).all() and (oa[9] == 2).all()  # labels in the right halves, not transposed


def test_orchestrator_exact_value(tmp_path):
    out = tmp_path / "out"
    asr = _build_pet_patient(out)
    suv = _img(np.full((10, 10, 10), 3.0, dtype=np.float32))  # uniform SUV=3
    mask = _img(np.ones((10, 10, 10), dtype=np.int16), dtype=np.int16)
    sample_patient_pet_suv(out, "PT1", _suv_fn=lambda p: suv, _mask_fn=lambda d: (mask, {1: "roi"}))
    r = json.loads((asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json").read_text())["rows"][0]
    assert r["suvmax"] == pytest.approx(3.0) and r["suvmean"] == pytest.approx(3.0) and r["suvpeak"] == pytest.approx(3.0)


def test_orchestrator_real_mask_discovery(tmp_path):
    # No _mask_fn → exercises the real read_total_ct_label_image + seg_dir.glob("*")->base subdir walk.
    out = tmp_path / "out"
    asr = _build_pet_patient(out)
    ct_dir = out / "PT1" / "all_series" / "petct_ct" / "uidCT0" / "DICOM"
    seg_base = ct_dir.parent / "Segmentation_TotalSegmentator" / "base1"
    seg_base.mkdir(parents=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((10, 10, 10), dtype=np.int16)),
                    str(seg_base / "total--multilabel.nii.gz"))
    (seg_base / "total--segmentations.json").write_text(json.dumps({"labels": {"1": "liver"}}))
    suv = _img(np.full((10, 10, 10), 2.0, dtype=np.float32))
    summary = sample_patient_pet_suv(out, "PT1", _suv_fn=lambda p: suv)  # real mask reader/discovery
    assert summary["n_sampled"] == 1
    r = json.loads((asr / "NIFTI" / "SUV" / "uidPT" / "pet_suv_structures.json").read_text())["rows"][0]
    assert r["structure_name"] == "liver" and r["suvmax"] == pytest.approx(2.0)


def test_orchestrator_status_filter_excludes_non_eligible(tmp_path):
    # suv_excluded/suv_failed PET rows are upstream-excluded (no NIfTI) → NOT B2 scope, NOT
    # mislabeled suv_nifti_missing (Codex impl-gate MAJOR).
    out = tmp_path / "out"
    man = out / "PT1" / "all_series" / "metadata"; man.mkdir(parents=True)
    pt_ok = out / "PT1" / "all_series" / "pt" / "uidOK" / "DICOM"; pt_ok.mkdir(parents=True)
    series = [
        {"series_uid": "uidOK", "image_class": "pt", "status": "suv_computed",
         "study_uid": "S", "frame_of_reference_uid": "F", "output_dir": str(pt_ok)},
        {"series_uid": "uidEXCL", "image_class": "pt", "status": "suv_excluded",
         "study_uid": "S", "frame_of_reference_uid": "F", "output_dir": ""},
    ]
    (man / "series_manifest.json").write_text(json.dumps({"series": series}))
    suv_fn, mask_fn = _synth_suv_mask()
    summary = sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    assert summary["n_pet"] == 1  # only the suv_computed series is in B2 scope


def test_orchestrator_multi_pet_completeness(tmp_path):
    out = tmp_path / "out"
    man = out / "PT1" / "all_series" / "metadata"; man.mkdir(parents=True)
    pa = out / "PT1" / "all_series" / "pt" / "uidP1" / "DICOM"; pa.mkdir(parents=True)
    pb = out / "PT1" / "all_series" / "pt" / "uidP2" / "DICOM"; pb.mkdir(parents=True)
    ct = out / "PT1" / "all_series" / "petct_ct" / "uidC" / "DICOM"; ct.mkdir(parents=True)
    (ct.parent / "Segmentation_TotalSegmentator").mkdir()
    series = [
        {"series_uid": "uidP1", "image_class": "pt", "status": "suv_computed", "study_uid": "S",
         "frame_of_reference_uid": "F", "output_dir": str(pa)},
        {"series_uid": "uidP2", "image_class": "pt", "status": "suv_skipped_idempotent", "study_uid": "S",
         "frame_of_reference_uid": "F", "output_dir": str(pb)},
        {"series_uid": "uidC", "image_class": "petct_ct", "study_uid": "S",
         "frame_of_reference_uid": "F", "n_slices": 200, "output_dir": str(ct)},
    ]
    (man / "series_manifest.json").write_text(json.dumps({"series": series}))
    suv_fn, mask_fn = _synth_suv_mask()
    sample_patient_pet_suv(out, "PT1", _suv_fn=suv_fn, _mask_fn=mask_fn)
    csv_path = write_pet_suv_structures_csv(out)
    import csv as _csv
    with csv_path.open() as f:
        uids = {r["pet_series_uid"] for r in _csv.DictReader(f)}
    assert uids == {"uidP1", "uidP2"}  # both SUV-eligible PET series present (no silent drop)
