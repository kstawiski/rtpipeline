"""B3 functional-MR sampling tests (rev5 §5). Pure-function + synthetic/phantom coverage.

Run: PYTHONPATH=<worktree> <rtpipeline-python> -m pytest tests/test_mr_functional_b3.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import SimpleITK as sitk

from rtpipeline.mr_functional import (
    LOW_NATIVE_VOXELS,
    QC_EMPTY_MASK,
    QC_MIP,
    QC_NO_ANATOMIC,
    QC_RAW_SOURCE,
    QC_UNSUPPORTED,
    coverage_fraction,
    per_structure_stats,
    read_total_mr_label_image,
    resample_functional_to_anatomic,
    route_functional_subtype,
    select_anatomic_for_functional,
    write_mr_functional_structures_csv,
)


# ---------------------------------------------------------------- subtype routing

@pytest.mark.parametrize("desc,subtype,sampled,kind", [
    # ADC (substring incl. vendor-fused; spelled-out)
    ("ep2d_diff_tra_0-800_ADC", "adc", True, "percentiles"),
    ("IsoADC", "adc", True, "percentiles"),
    ("dADC", "adc", True, "percentiles"),
    ("Apparent Diffusion Coefficient", "adc", True, "percentiles"),
    # DWI (substring vendor-fused; diff/ep2d prefix)
    ("IsoDWI", "dwi", True, "percentiles"),
    ("cDWI.b=1000", "dwi", True, "percentiles"),
    ("DWI_b1500", "dwi", True, "percentiles"),
    ("DWIBS_SBC", "dwi", True, "percentiles"),
    # perfusion derived maps (DERIVED_MAP standalone tokens)
    ("TTP_ep2d_perf_p2", "perfusion", True, "mean_median"),
    ("RELCBV_LOCAL_ep2d_perf_p2", "perfusion", True, "mean_median"),
    ("t1_vibe_tra_perf_CM_WO", "perfusion", True, "mean_median"),
    ("t1_vibe_tra_perf_CM_PEI", "perfusion", True, "mean_median"),
    # subtraction
    ("t1_vibe_fs_tra_SUB", "subtraction", True, "mean_median"),
    ("t1_vibe-grasp_fs_tra_SUB", "subtraction", True, "mean_median"),
])
def test_route_sampled_subtypes(desc, subtype, sampled, kind):
    r = route_functional_subtype(desc)
    assert (r.subtype, r.sampled, r.stats_kind) == (subtype, sampled, kind), f"{desc!r} -> {r}"
    assert r.qc_reason is None


@pytest.mark.parametrize("desc", [
    "ep2d_perf_p2",           # raw EPI-perfusion acquisition — MUST defer, NOT dwi (Claude-r3)
    "t1_vibe_tra_perf_CM",    # raw DCE acquisition (bare perf)
])
def test_route_raw_perf_defers_before_dwi(desc):
    r = route_functional_subtype(desc)
    assert r.subtype == "raw_perf" and not r.sampled and r.qc_reason == QC_RAW_SOURCE, f"{desc!r} -> {r}"


@pytest.mark.parametrize("desc", [
    "t1_twist_tra_dyn_TT=7.0s",   # TWIST raw — MUST NOT route to perfusion via wi⊂twist (Claude-r3)
    "t1_vibe_tra_dynamic",        # dyn-prefix raw dynamic
    "t1_vibe-grasp_fs_tra",       # grasp raw (no sub)
])
def test_route_raw_dynamics_defer(desc):
    r = route_functional_subtype(desc)
    assert not r.sampled and r.qc_reason == QC_RAW_SOURCE, f"{desc!r} -> {r}"


def test_route_twist_is_not_perfusion_substring_guard():
    # The decisive substring-vs-token guard: 'wi' ⊂ 'twist' must not make it perfusion.
    r = route_functional_subtype("t1_twist_tra_dyn_TT=11.7s")
    assert r.subtype != "perfusion"


def test_route_ep2d_diff_is_dwi_not_deferred():
    r = route_functional_subtype("ep2d_diff_tra")
    assert r.subtype == "dwi" and r.sampled


@pytest.mark.parametrize("desc,image_types", [
    ("t1_vibe_fs_tra_SUB_MIP_COR", None),                 # MIP token → excluded at sampling
    ("ADC map", ["DERIVED", "SECONDARY", "MIP"]),         # MIP via ImageType
])
def test_route_mip_excluded(desc, image_types):
    r = route_functional_subtype(desc, image_types)
    assert not r.sampled and r.qc_reason == QC_MIP, f"{desc!r} -> {r}"


def test_route_unsupported_default_deny():
    r = route_functional_subtype("some_unknown_mr_thing")
    assert not r.sampled and r.qc_reason == QC_UNSUPPORTED


# ------------------------------------------------------------- geometry helpers

def _img(arr: np.ndarray, *, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.float32))  # arr is (z,y,x)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    return img


def test_direct_tier_when_same_geometry():
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    f = _img(arr)
    a = _img(np.zeros_like(arr))  # same geometry, content irrelevant for 'direct'
    res = resample_functional_to_anatomic(f, a, same_for=True)
    assert res.tier == "direct" and res.qc_reason is None
    np.testing.assert_allclose(sitk.GetArrayFromImage(res.image), arr)


def test_exact_percentiles_direct_tier():
    # 1x1x10 functional with known values; mask labels all 10 voxels as structure 1.
    vals = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]], dtype=np.float32)  # (z=1,y=1,x=10)
    f = _img(vals)
    a = _img(np.zeros_like(vals))
    res = resample_functional_to_anatomic(f, a, same_for=True)
    mask = np.ones_like(vals, dtype=np.int32)  # all voxels label 1
    rows = per_structure_stats(res.image, mask, {1: "roi"}, stats_kind="percentiles", tier="direct")
    row = rows[0]
    assert row["n_voxels"] == 10
    assert row["mean"] == pytest.approx(4.5)
    assert row["median"] == pytest.approx(4.5)
    assert row["p10"] == pytest.approx(np.percentile(np.arange(10), 10))
    assert row["p90"] == pytest.approx(np.percentile(np.arange(10), 90))
    assert row["qc_flag"] == "ok"


def test_same_for_different_grid_linear_resample():
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    arr[2:8, 2:8, 2:8] = 100.0
    f = _img(arr, spacing=(1.0, 1.0, 1.0))
    a = _img(np.zeros((20, 20, 20), dtype=np.float32), spacing=(0.5, 0.5, 0.5))  # finer grid, same extent
    res = resample_functional_to_anatomic(f, a, same_for=True)
    assert res.tier == "resample_samefor"
    out = sitk.GetArrayFromImage(res.image)
    assert out.shape == (20, 20, 20)
    assert np.nanmax(out) == pytest.approx(100.0, abs=1.0)  # linear interp preserves the peak


def test_empty_mask_qc():
    f = _img(np.ones((3, 3, 3), dtype=np.float32))
    a = _img(np.zeros((3, 3, 3), dtype=np.float32))
    res = resample_functional_to_anatomic(f, a, same_for=True)
    mask = np.zeros((3, 3, 3), dtype=np.int32)  # label 1 present in map but no voxels
    rows = per_structure_stats(res.image, mask, {1: "roi"}, stats_kind="mean_median", tier="direct")
    assert rows[0]["qc_flag"] == QC_EMPTY_MASK and rows[0]["n_voxels"] == 0


def test_low_native_voxels_and_deformable_flag():
    f = _img(np.ones((4, 4, 4), dtype=np.float32))
    a = _img(np.zeros((4, 4, 4), dtype=np.float32))
    res = resample_functional_to_anatomic(f, a, same_for=True)
    mask = np.zeros((4, 4, 4), dtype=np.int32)
    mask[0, 0, 0] = 1  # 1 voxel structure → below LOW_NATIVE_VOXELS
    rows = per_structure_stats(
        res.image, mask, {1: "urinary_bladder"}, stats_kind="mean_median",
        tier="rigid_mi", native_count={1: 1},
    )
    assert rows[0]["low_native_voxels"] is True
    assert rows[0]["deformable_flag"] is True  # bladder on rigid tier
    assert 1 < LOW_NATIVE_VOXELS


def test_coverage_fraction():
    arr = np.ones((1, 1, 10), dtype=np.float32)
    img = _img(arr)
    # set half the structure region to NaN to simulate out-of-FOV after resample
    a = sitk.GetArrayFromImage(img)
    a[0, 0, 5:] = np.nan
    img = _img(a)
    mask = np.ones((1, 1, 10), dtype=np.int32)
    assert coverage_fraction(img, mask, 1) == pytest.approx(0.5)


def test_rigid_registration_recovers_translation():
    # Phantom: bright blob on dark background; moving = same content, origin shifted in physical
    # space (different FoR). Rigid-MI must recover the shift and yield high coverage.
    rng = np.random.default_rng(0)
    base = rng.normal(20.0, 2.0, size=(32, 32, 32)).astype(np.float32)
    base[10:22, 10:22, 10:22] += 200.0  # clear cubic feature
    fixed = _img(base, origin=(0.0, 0.0, 0.0))
    shift = (6.0, 0.0, 0.0)  # physical mm shift along x
    moving = _img(base, origin=shift)  # same content, shifted origin → different FoR
    res = resample_functional_to_anatomic(moving, fixed, same_for=False)
    assert res.tier == "rigid_mi"
    assert res.qc_reason is None, f"registration rejected: {res.qc_reason}"
    assert res.reg_converged
    # coverage of a central ROI should be high (feature is in-FOV after registration)
    mask = np.zeros((32, 32, 32), dtype=np.int32)
    mask[12:20, 12:20, 12:20] = 1
    cov = coverage_fraction(res.image, mask, 1)
    assert cov > 0.9, f"coverage {cov}"


# ----------------------------------------------------- anatomic selection

def test_select_shared_for():
    func_for = "FOR-1"
    cands = [
        {"series_uid": "A", "frame_of_reference_uid": "FOR-2", "study_uid": "S", "n_slices": 30},
        {"series_uid": "B", "frame_of_reference_uid": "FOR-1", "study_uid": "S", "n_slices": 20},
    ]
    chosen, basis, qc = select_anatomic_for_functional(func_for, "S", cands)
    assert chosen["series_uid"] == "B" and basis == "shared_for" and qc is None


def test_select_no_candidates():
    chosen, basis, qc = select_anatomic_for_functional("FOR-1", "S", [])
    assert chosen is None and qc == QC_NO_ANATOMIC


def test_select_same_study_tiebreak_n_slices():
    cands = [
        {"series_uid": "A", "frame_of_reference_uid": "X", "study_uid": "S", "n_slices": 30},
        {"series_uid": "B", "frame_of_reference_uid": "Y", "study_uid": "S", "n_slices": 60},
    ]
    chosen, basis, qc = select_anatomic_for_functional("FOR-none", "S", cands)
    assert chosen["series_uid"] == "B" and "study" in basis and qc is None  # larger n_slices


# ----------------------------------------------------- mask reader + CSV

def test_read_total_mr_label_image(tmp_path):
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[0:2] = 1
    arr[2:4] = 2
    limg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(limg, str(tmp_path / "total_mr--multilabel.nii.gz"))
    (tmp_path / "total_mr--segmentations.json").write_text(
        json.dumps({"labels": {"1": "muscle", "2": "fat"}}), encoding="utf-8"
    )
    img, label_map = read_total_mr_label_image(tmp_path)
    assert img is not None
    assert label_map == {1: "muscle", 2: "fat"}


def test_csv_writer_flattens_sidecars(tmp_path):
    out_root = tmp_path / "out"
    sidecar_dir = out_root / "pt1" / "all_series" / "MR_functional" / "uid1"
    sidecar_dir.mkdir(parents=True)
    payload = {
        "rows": [
            {"patient_id": "pt1", "functional_series_uid": "uid1", "functional_subtype": "adc",
             "structure_name": "muscle", "qc_flag": "ok", "mean": 1.2, "n_voxels": 50},
            {"patient_id": "pt1", "functional_series_uid": "uid1", "functional_subtype": "adc",
             "structure_name": "fat", "qc_flag": "ok", "mean": 0.4, "n_voxels": 80},
        ]
    }
    (sidecar_dir / "mr_functional.json").write_text(json.dumps(payload), encoding="utf-8")
    csv_path = write_mr_functional_structures_csv(out_root)
    assert csv_path is not None and csv_path.exists()
    import csv as _csv
    with csv_path.open() as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2
    assert {r["structure_name"] for r in rows} == {"muscle", "fat"}
    assert rows[0]["functional_subtype"] == "adc"


def test_csv_writer_no_sidecars_returns_none(tmp_path):
    assert write_mr_functional_structures_csv(tmp_path) is None


# ----------------------------------------------------- orchestrator (integration)

from rtpipeline.mr_functional import sample_patient_mr_functional  # noqa: E402


def _build_patient(out_root: Path, func_desc: str, *, func_for="FOR1", anat_for="FOR1"):
    """Create a synthetic patient layout + manifest with one functional + one anatomic series."""
    man_dir = out_root / "PT1" / "all_series" / "metadata"
    man_dir.mkdir(parents=True)
    func_dir = out_root / "PT1" / "all_series" / "MR_functional" / "uidF" / "DICOM"
    func_dir.mkdir(parents=True)
    anat_dir = out_root / "PT1" / "all_series" / "MR" / "uidA" / "DICOM"
    anat_dir.mkdir(parents=True)
    (anat_dir.parent / "Segmentation_TotalSegmentator").mkdir()  # so the mask-dir loop runs
    manifest = {"series": [  # production manifest key (inventory.py:326), NOT "rows"
        {"series_uid": "uidF", "image_class": "mr_functional", "series_description": func_desc,
         "frame_of_reference_uid": func_for, "study_uid": "S1", "output_dir": str(func_dir)},
        {"series_uid": "uidA", "image_class": "mr_anatomic", "series_description": "T2W_TSE",
         "frame_of_reference_uid": anat_for, "study_uid": "S1", "n_slices": 30, "output_dir": str(anat_dir)},
    ]}
    (man_dir / "series_manifest.json").write_text(json.dumps(manifest))
    return func_dir, anat_dir


def _synthetic_io():
    f_arr = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    f_img = _img(f_arr)
    mask_arr = np.zeros((10, 10, 10), dtype=np.int16)
    mask_arr[0:5] = 1
    mask_img = sitk.GetImageFromArray(mask_arr)
    intensity = _img(np.ones((10, 10, 10), dtype=np.float32))  # anatomic intensity, same grid as mask
    return (lambda d: (f_img, 1, None)), (lambda d: (mask_img, {1: "roi"})), (lambda d: intensity)


def test_orchestrator_ok_path_writes_sidecar(tmp_path):
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "ep2d_diff_tra_0-800_ADC")
    load_fn, mask_fn, int_fn = _synthetic_io()
    summary = sample_patient_mr_functional(out, "PT1", _load_fn=load_fn, _mask_fn=mask_fn, _intensity_fn=int_fn)
    assert summary["n_functional"] == 1 and summary["n_sampled"] == 1
    sidecar = func_dir.parent / "mr_functional.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["series_qc"] == "ok"
    assert payload["registration"]["tier"] == "direct"
    assert payload["rows"][0]["functional_subtype"] == "adc"
    assert payload["rows"][0]["qc_flag"] == "ok"
    assert payload["rows"][0]["raw_unit"] and payload["rows"][0]["unit_source"]  # provenance recorded


def test_orchestrator_raw_dynamic_excluded(tmp_path):
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "t1_twist_tra_dyn_TT=7.0s")  # raw TWIST -> deferred
    load_fn, mask_fn, int_fn = _synthetic_io()
    summary = sample_patient_mr_functional(out, "PT1", _load_fn=load_fn, _mask_fn=mask_fn, _intensity_fn=int_fn)
    assert summary["n_sampled"] == 0
    payload = json.loads((func_dir.parent / "mr_functional.json").read_text())
    assert payload["series_qc"] == QC_RAW_SOURCE
    assert len(payload["rows"]) == 1 and payload["rows"][0]["qc_flag"] == QC_RAW_SOURCE


def test_orchestrator_anatomic_out_of_scope(tmp_path):
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "ep2d_diff_tra_ADC")
    load_fn, mask_fn, int_fn = _synthetic_io()
    summary = sample_patient_mr_functional(out, "PT1", anatomic_in_scope=False, _load_fn=load_fn, _mask_fn=mask_fn, _intensity_fn=int_fn)
    payload = json.loads((func_dir.parent / "mr_functional.json").read_text())
    assert payload["series_qc"] == "anatomic_out_of_scope"


def test_orchestrator_completeness_every_series_in_csv(tmp_path):
    # Two functional series (one sampled ADC, one deferred TWIST); CSV must contain BOTH UIDs.
    out = tmp_path / "out"
    man_dir = out / "PT1" / "all_series" / "metadata"; man_dir.mkdir(parents=True)
    fa = out / "PT1" / "all_series" / "MR_functional" / "uidADC" / "DICOM"; fa.mkdir(parents=True)
    ft = out / "PT1" / "all_series" / "MR_functional" / "uidTWIST" / "DICOM"; ft.mkdir(parents=True)
    anat = out / "PT1" / "all_series" / "MR" / "uidA" / "DICOM"; anat.mkdir(parents=True)
    (anat.parent / "Segmentation_TotalSegmentator").mkdir()
    manifest = {"series": [  # production manifest key
        {"series_uid": "uidADC", "image_class": "mr_functional", "series_description": "IsoADC",
         "frame_of_reference_uid": "FOR1", "study_uid": "S1", "output_dir": str(fa)},
        {"series_uid": "uidTWIST", "image_class": "mr_functional", "series_description": "t1_twist_tra_dyn",
         "frame_of_reference_uid": "FOR1", "study_uid": "S1", "output_dir": str(ft)},
        {"series_uid": "uidA", "image_class": "mr_anatomic", "series_description": "T2W_TSE",
         "frame_of_reference_uid": "FOR1", "study_uid": "S1", "n_slices": 30, "output_dir": str(anat)},
    ]}
    (man_dir / "series_manifest.json").write_text(json.dumps(manifest))
    load_fn, mask_fn, int_fn = _synthetic_io()
    sample_patient_mr_functional(out, "PT1", _load_fn=load_fn, _mask_fn=mask_fn, _intensity_fn=int_fn)
    csv_path = write_mr_functional_structures_csv(out)
    import csv as _csv
    with csv_path.open() as f:
        uids = {r["functional_series_uid"] for r in _csv.DictReader(f)}
    manifest_func_uids = {"uidADC", "uidTWIST"}
    assert uids == manifest_func_uids, "series-UID set-equality (no silent drop) violated"


def test_orchestrator_idempotent_skip(tmp_path):
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "IsoADC")
    load_fn, mask_fn, int_fn = _synthetic_io()
    sample_patient_mr_functional(out, "PT1", _load_fn=load_fn, _mask_fn=mask_fn, _intensity_fn=int_fn)
    # second call without force must skip (sidecar exists) -> n_sampled 0
    s2 = sample_patient_mr_functional(out, "PT1", _load_fn=load_fn, _mask_fn=mask_fn, _intensity_fn=int_fn)
    assert s2["n_sampled"] == 0


def test_config_default_off():
    import dataclasses
    from rtpipeline.config import PipelineConfig
    fld = {f.name: f for f in dataclasses.fields(PipelineConfig)}["mr_functional_sampling"]
    assert fld.default is False  # opt-in: default OFF, zero behavior change for existing callers


# ----------------------------------------------- impl-gate R1 remediation tests

def test_select_ambiguous_when_tied():
    from rtpipeline.mr_functional import QC_AMBIGUOUS_ANATOMIC
    cands = [
        {"series_uid": "A", "frame_of_reference_uid": "F1", "study_uid": "S", "n_slices": 30},
        {"series_uid": "B", "frame_of_reference_uid": "F1", "study_uid": "S", "n_slices": 30},
    ]
    chosen, basis, qc = select_anatomic_for_functional("F1", "S", cands)
    assert chosen is None and qc == QC_AMBIGUOUS_ANATOMIC  # tied same-FoR, equal n_slices, no affine


def test_select_affine_breaks_tie():
    cands = [
        {"series_uid": "A", "frame_of_reference_uid": "F1", "n_slices": 30},
        {"series_uid": "B", "frame_of_reference_uid": "F1", "n_slices": 30},
    ]
    sims = {"A": 0.9, "B": 0.3}
    chosen, basis, qc = select_anatomic_for_functional(
        "F1", "S", cands, affine_similarity=lambda r: sims[r["series_uid"]])
    assert chosen["series_uid"] == "A" and qc is None  # geometry similarity breaks the tie


def test_native_voxel_counts_identity():
    import rtpipeline.mr_functional as mrf
    mask = sitk.GetImageFromArray(np.array([[[1, 1, 0, 0]]], dtype=np.int16))
    f = _img(np.ones((1, 1, 4), dtype=np.float32))
    counts = mrf._native_voxel_counts(mask, f, sitk.Transform(3, sitk.sitkIdentity))
    assert counts.get(1) == 2  # mask NN-resampled onto F's native grid (identity) preserves the 2 voxels


def test_orchestrator_low_coverage_reg_failed(tmp_path):
    # functional FOV (3^3) covers only ~2.7% of the 10^3 anatomic mask region → union coverage
    # below MIN_COVERAGE_FRAC → reg_failed (acceptance gate enforced).
    from rtpipeline.mr_functional import QC_REG_FAILED
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "IsoADC")
    small_f = _img(np.ones((3, 3, 3), dtype=np.float32))
    big_geom = _img(np.zeros((10, 10, 10), dtype=np.float32))
    big_mask = sitk.GetImageFromArray(np.ones((10, 10, 10), dtype=np.int16))
    sample_patient_mr_functional(
        out, "PT1",
        _load_fn=lambda d: (small_f, 1, None),
        _mask_fn=lambda d: (big_mask, {1: "roi"}),
        _intensity_fn=lambda d: big_geom,
    )
    payload = json.loads((func_dir.parent / "mr_functional.json").read_text())
    assert payload["series_qc"] == QC_REG_FAILED
    assert payload["rows"][0]["reg_coverage_frac"] < 0.7


def test_orchestrator_registers_against_intensity_not_mask(tmp_path):
    # The intensity image (not the discrete mask) must be the resample/registration reference.
    # Use distinct intensity vs mask geometry-compatible images; OK path must still sample.
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "IsoADC")
    f_img = _img(np.arange(1000, dtype=np.float32).reshape(10, 10, 10))
    intensity = _img(np.ones((10, 10, 10), dtype=np.float32) * 50.0)  # anatomic intensity
    mask = sitk.GetImageFromArray((np.indices((10, 10, 10)).sum(0) < 8).astype(np.int16))
    called = {"intensity": 0}

    def int_fn(d):
        called["intensity"] += 1
        return intensity
    summary = sample_patient_mr_functional(
        out, "PT1",
        _load_fn=lambda d: (f_img, 1, None),
        _mask_fn=lambda d: (mask, {1: "roi"}),
        _intensity_fn=int_fn,
    )
    assert summary["n_sampled"] == 1
    assert called["intensity"] >= 1  # intensity loader was used (registration ref), not the mask


def test_load_functional_volume_3d_ok_and_multivolume(tmp_path, monkeypatch):
    import rtpipeline.mr_functional as mrf
    dicom = tmp_path / "DICOM"
    dicom.mkdir()
    (dicom / "a.dcm").write_text("x")

    def _stub(n):
        def fake(cmd, **kw):
            outdir = Path(cmd[cmd.index("-o") + 1])
            for i in range(n):
                sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.float32)),
                                str(outdir / f"func_{i}.nii.gz"))
            class R:  # noqa
                returncode = 0
            return R()
        return fake

    monkeypatch.setattr(mrf.subprocess, "run", _stub(1))
    img, n, qc = mrf.load_functional_volume(dicom)
    assert qc is None and img is not None and img.GetDimension() == 3  # single 3-D map -> ok

    monkeypatch.setattr(mrf.subprocess, "run", _stub(2))
    _, _, qc2 = mrf.load_functional_volume(dicom)
    assert qc2 == mrf.QC_MULTIVOLUME  # >1 output file -> 4-D/multivolume -> deferred


def test_load_functional_volume_not_materialized(tmp_path):
    import rtpipeline.mr_functional as mrf
    empty = tmp_path / "DICOM"
    empty.mkdir()
    img, n, qc = mrf.load_functional_volume(empty)
    assert img is None and qc == mrf.QC_NOT_MATERIALIZED


def test_units_provenance_convention_fallback(tmp_path):
    import rtpipeline.mr_functional as mrf
    raw_unit, src, applied = mrf._units_provenance("adc", tmp_path)  # no DICOM tags -> convention
    assert src == "convention" and raw_unit and applied is True


def test_csv_unreadable_sidecar_logged(tmp_path, caplog):
    import logging
    out = tmp_path / "out"
    d = out / "PT1" / "all_series" / "MR_functional" / "uid1"
    d.mkdir(parents=True)
    (d / "mr_functional.json").write_text("{ this is not valid json")
    with caplog.at_level(logging.ERROR):
        res = write_mr_functional_structures_csv(out)
    assert res is None  # nothing valid to write
    assert any("UNREADABLE" in r.message or "unreadable" in r.message.lower() for r in caplog.records)


def test_orchestrator_non_converged_rigid_reg_failed(tmp_path, monkeypatch):
    # The convergence half of the acceptance gate: a rigid result that did NOT converge is
    # rejected (reg_failed) even when coverage is full (closes the R2 convergence test gap).
    import rtpipeline.mr_functional as mrf
    from rtpipeline.mr_functional import ResampleResult, QC_REG_FAILED
    out = tmp_path / "out"
    func_dir, _ = _build_patient(out, "IsoADC")
    f_img = _img(np.ones((10, 10, 10), dtype=np.float32))
    intensity = _img(np.ones((10, 10, 10), dtype=np.float32))
    mask = sitk.GetImageFromArray(np.ones((10, 10, 10), dtype=np.int16))

    def fake_resample(functional, anatomic_geom, *, same_for, default_value=float("nan")):
        return ResampleResult(image=f_img, tier="rigid_mi", reg_converged=False,
                              coverage_frac=None, transform=sitk.Transform(3, sitk.sitkIdentity),
                              qc_reason=None)
    monkeypatch.setattr(mrf, "resample_functional_to_anatomic", fake_resample)
    mrf.sample_patient_mr_functional(
        out, "PT1", _load_fn=lambda d: (f_img, 1, None),
        _mask_fn=lambda d: (mask, {1: "roi"}), _intensity_fn=lambda d: intensity)
    payload = json.loads((func_dir.parent / "mr_functional.json").read_text())
    assert payload["series_qc"] == QC_REG_FAILED  # non-converged rigid rejected despite full coverage
