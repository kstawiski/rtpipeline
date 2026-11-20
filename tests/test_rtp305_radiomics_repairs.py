"""Regression tests for RTP305-RAD (CT radiomics, IBSI path, and NTCV robustness)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from rtpipeline import acquisition_scale, cli, radiomics as rm, radiomics_conda as rc
from rtpipeline import radiomics_resource_guard as rg, radiomics_robustness as rr
from rtpipeline import radiomics_robustness_aggregate as ra
from rtpipeline import radiomics_robustness_outcomes as rro
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS, classify_ct_roi
from robustness_prep_fixture import cylinder, write_ct, write_rtstruct

ROOT = Path(__file__).resolve().parents[1]


def _write_ct_series_custom(
    ct_dir: Path,
    *,
    rows: int = 16,
    columns: int = 16,
    slices: int = 8,
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    series_uid: str | None = None,
    rescale_type: str | None = "HU",
    include_rescale: bool = True,
    value_offset: int = 0,
    study_uid: str | None = None,
) -> str:
    ct_dir.mkdir(parents=True, exist_ok=True)
    suid = series_uid or generate_uid()
    stuid = study_uid or generate_uid()
    fuid = generate_uid()
    yy, xx = np.mgrid[0:rows, 0:columns]
    for idx in range(slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = ct_dir / f"ct_{suid[-6:]}_{idx:04d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientID = "SYNTH"
        ds.StudyInstanceUID = stuid
        ds.SeriesInstanceUID = suid
        ds.FrameOfReferenceUID = fuid
        ds.StudyDate = "20240101"
        ds.StudyTime = "093000"
        ds.SeriesNumber = 1
        ds.InstanceNumber = idx + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.Rows, ds.Columns = rows, columns
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, idx * spacing[2]]
        if include_rescale:
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = -1024.0
        if rescale_type is not None:
            ds.RescaleType = rescale_type
        pixels = (1024 + value_offset + ((xx * 5 + yy * 3 + idx * 7) % 80)).astype(np.int16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return suid


# ---------------------------------------------------------------------------
# A2: --stage radiomics config propagation + conda packaged params default
# ---------------------------------------------------------------------------


def test_a2_cli_stage_radiomics_carries_yaml_settings(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    outdir = tmp_path / "out"
    logs = tmp_path / "logs"
    dicom_root.mkdir()
    params = tmp_path / "custom_ct.yaml"
    params.write_text("setting:\n  binWidth: 25\n", encoding="utf-8")
    mr_params = tmp_path / "custom_mr.yaml"
    mr_params.write_text("setting:\n  binWidth: 5\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dicom_root": str(dicom_root),
                "output_dir": str(outdir),
                "logs_dir": str(logs),
                "radiomics": {
                    "params_file": str(params),
                    "mr_params_file": str(mr_params),
                    "min_voxels": 64,
                    "max_voxels": 500000,
                    "max_resampled_bbox_voxels": 123456,
                    "skip_rois": ["BODY", "couch*"],
                    "sequential": True,
                    "env_probe_timeout": 333,
                    "thread_limit": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    seen = {}
    monkeypatch.setattr(cli, "organize_and_merge", lambda *_a, **_k: [SimpleNamespace(patient_id="P1", course_id="C1")])
    import rtpipeline.radiomics as rad_mod

    monkeypatch.setattr(
        rad_mod,
        "run_radiomics",
        lambda config, *_a, **_k: seen.setdefault("config", config),
    )
    # The test checks configuration propagation; it must not depend on a radiomics
    # environment being installed on the test host (the minimum-Python suite has none).
    monkeypatch.setattr(rad_mod, "_have_pyradiomics", lambda: True)

    monkeypatch.setenv("RTPIPELINE_CONFIGFILE", str(cfg_path))
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_SEQUENTIAL", "")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT", "")
    rc_code = cli.main(
        [
            "--dicom-root", str(dicom_root),
            "--outdir", str(outdir),
            "--logs", str(logs),
            "--stage", "radiomics",
            "--no-metadata",
        ]
    )
    assert rc_code == 0
    cfg = seen["config"]
    assert Path(cfg.radiomics_params_file) == params
    assert Path(cfg.radiomics_params_file_mr) == mr_params
    assert cfg.radiomics_min_voxels == 64
    assert cfg.radiomics_max_voxels == 500000
    assert cfg.radiomics_max_resampled_bbox_voxels == 123456
    assert list(cfg.radiomics_skip_rois) == ["BODY", "couch*"]
    assert cfg.radiomics_sequential is True
    assert cfg.radiomics_env_probe_timeout == 333
    assert cfg.radiomics_thread_limit == 3


def test_a2_conda_make_extractor_uses_packaged_params_never_library_defaults(monkeypatch):
    seen = []

    class _FakePyRadExtractor:
        def __init__(self, params_path=None):
            seen.append(params_path)
            self.settings = {}
            self.enabledImagetypes = {"Original": {}}
            self.enabledFeatures = {"firstorder": ["Mean"]}

    import types

    fake_fe = types.ModuleType("radiomics.featureextractor")
    fake_fe.RadiomicsFeatureExtractor = _FakePyRadExtractor
    fake_rad = types.ModuleType("radiomics")
    fake_rad.featureextractor = fake_fe
    monkeypatch.setitem(sys.modules, "radiomics", fake_rad)
    monkeypatch.setitem(sys.modules, "radiomics.featureextractor", fake_fe)

    ext = rc.make_extractor(None, False)
    assert ext is not None
    assert len(seen) == 1
    assert seen[0] is not None
    assert Path(seen[0]).name == "radiomics_params.yaml"
    assert Path(seen[0]).is_file()


# ---------------------------------------------------------------------------
# A5: Snakefile LD_LIBRARY_PATH / fail-closed env + compute_icc_pingouin errors
# ---------------------------------------------------------------------------


def test_a5_snakefile_exports_ld_library_path_and_resolve_env_fails_closed(tmp_path, monkeypatch):
    snakefile_text = (ROOT / "Snakefile").read_text(encoding="utf-8")
    idx = snakefile_text.index("rule radiomics_robustness_course:")
    # Inspect the second (non-container) branch too
    second_idx = snakefile_text.index("rule radiomics_robustness_course:", idx + 1)
    next_rule = snakefile_text.index("rule aggregate_radiomics_robustness:", second_idx)
    container_rule = snakefile_text[idx:second_idx]
    assert "LD_LIBRARY_PATH" in container_rule
    assert "PYTHONPATH" in container_rule
    non_container_rule = snakefile_text[second_idx:next_rule]
    assert "LD_LIBRARY_PATH" in non_container_rule
    assert "PYTHONPATH" in non_container_rule

    # Extract _resolve_env_python from Snakefile AST and verify missing env raises
    start = snakefile_text.index("def _resolve_env_python(")
    end = snakefile_text.index("\n# Resolve python executables", start)
    ns = {"os": os, "sys": sys, "Path": Path, "config": {}, "_CONDA_ROOT": Path("/nonexistent/conda"), "_CONTAINER": False}
    exec(compile(snakefile_text[start:end], "Snakefile", "exec"), ns)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        ns["_resolve_env_python"]("missing-nonexistent-env-xyz")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_ENV_PREFIX", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Configured radiomics interpreter"):
        ns["_resolve_env_python"]("rtpipeline-radiomics")


def test_a5_compute_icc_pingouin_surfaces_import_and_runtime_errors(monkeypatch):
    import builtins

    df = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B"],
            "rater": ["r1", "r2", "r1", "r2"],
            "value": [1.0, 1.1, 2.0, 2.2],
        }
    )
    real_import = builtins.__import__

    def fail_pingouin(name, *args, **kwargs):
        if name == "pingouin":
            raise ImportError("synthetic missing pingouin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pingouin)
    with pytest.raises((ImportError, RuntimeError), match="pingouin"):
        rr.compute_icc_pingouin(df, rr.ICCConfig(icc_type="ICC3", ci=True))

    monkeypatch.undo()
    broken_df = pd.DataFrame({"subject": ["A", "A", "B", "B"], "rater": ["r1", "r2", "r1", "r2"], "value": ["bad", "data", "x", "y"]})
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        rr.compute_icc_pingouin(broken_df, rr.ICCConfig(icc_type="ICC3", ci=True))


# ---------------------------------------------------------------------------
# C1: Worker-identity mismatch stays in requested_condition_ids as technical_failure
# ---------------------------------------------------------------------------


def test_c1_worker_identity_mismatch_kept_as_technical_failure(tmp_path, monkeypatch):
    from test_robustness_nonmeasurements import course_fixture

    course, cfg, rob, real_features = course_fixture(tmp_path, monkeypatch, parallel=False)

    def mismatch_features(image, masks, *args, **kwargs):
        frame = real_features(image, masks, *args, **kwargs)
        bad_pid = sorted(masks)[0]
        frame.loc[frame["perturbation_id"] == bad_pid, "series_uid"] = "WRONG.SERIES.UID"
        return frame

    monkeypatch.setattr(rr, "extract_features_for_masks", mismatch_features)
    with pytest.raises(rr.RobustnessPartialExtractionError):
        rr.robustness_for_course(cfg, rob, course)

    assert not (course / "radiomics_robustness_ct.parquet").exists()
    evidence = course / "radiomics_robustness_ct.failed_evidence.parquet"
    assert evidence.is_file()
    df = pd.read_parquet(evidence)
    tech = df[df["robustness_status"] == "technical_failure"]
    assert len(tech) == len(CT_EXTRACTION_ARMS)
    # Must NOT drop the mismatched condition from requested_condition_ids or possible_condition_count
    assert all(len(json.loads(v)) == 3 for v in df["requested_condition_ids"])
    assert set(df["possible_condition_count"]) == {2}  # 3 requested - 1 translation_outside_image


# ---------------------------------------------------------------------------
# C2 & A4: Realized translation recorded + coarse slices refused
# ---------------------------------------------------------------------------


def test_c2_coarse_slices_refused_and_realized_translation_recorded():
    arr = np.zeros((7, 16, 16), dtype=np.uint8)
    arr[2:5, 5:11, 5:11] = 1
    coarse_mask = sitk.GetImageFromArray(arr)
    coarse_mask.SetSpacing((1.0, 1.0, 10.0))  # +/- 4 mm rounds to 0 voxels (0 mm shift!)
    coarse_img = sitk.GetImageFromArray(np.ones_like(arr, dtype=np.float32))
    coarse_img.SetSpacing((1.0, 1.0, 10.0))

    translated = rr.translate_mask(coarse_mask, (0.0, 0.0, 4.0))
    with pytest.raises(RuntimeError):
        rr._validate_translated_mask(coarse_mask, translated, (0.0, 0.0, 4.0))

    cfg = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=4.0,
        n_random_contour_realizations=0,
        noise_levels=[0.0],
    )
    masks, _ = rr.generate_ntcv_perturbations(coarse_mask, coarse_img, cfg, "ROI")
    assert isinstance(masks["ntcv_t0_0_4_v0"], rro.GeometricNonmeasurement)
    assert isinstance(masks["ntcv_t0_0_-4_v0"], rro.GeometricNonmeasurement)

    # Fine isotropic 2 mm spacing realizes exact 4.0 mm shift and records it
    fine_mask = sitk.GetImageFromArray(np.pad(arr, ((3, 3), (0, 0), (0, 0))))
    fine_mask.SetSpacing((1.0, 1.0, 2.0))
    shifted_2mm = rr.translate_mask(fine_mask, (0.0, 0.0, 4.0))
    realized = rr._validate_translated_mask(fine_mask, shifted_2mm, (0.0, 0.0, 4.0))
    assert realized is not None
    assert np.allclose(realized, (0.0, 0.0, 4.0), atol=1e-6)


def test_c2_subvoxel_translation_records_a_geometric_disposition():
    array = np.zeros((9, 9, 9), dtype=np.uint8)
    array[2:7, 2:7, 2:7] = 1
    mask = sitk.GetImageFromArray(array)
    translation = (0.0, 0.0, 1e-7)
    with pytest.raises(rro.GeometricNotExtractable) as caught:
        rr._validate_translated_mask(mask, rr.translate_mask(mask, translation), translation)
    outcome = caught.value.outcome
    assert outcome.reason_code == "translation_not_realizable"
    assert outcome.evidence["requested_translation_mm"] == list(translation)
    assert outcome.evidence["realized_translation_mm"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("shift", [4.0, 1e-5], ids=["clinical", "subvoxel"])
def test_c2_published_parquet_and_sidecar_record_requested_and_realized_translation(
    tmp_path, monkeypatch, shift
):
    from rasterizer_test_utils import stub_synthetic_rasterizer_columns

    stub_synthetic_rasterizer_columns(monkeypatch)
    monkeypatch.setattr("rtpipeline.radiomics_rasterizer.require_radiomics_rasterizer", lambda: None)

    def _run_spacing(tag: str, spacing_z: float):
        course = tmp_path / tag / "P1" / "C1"
        course.mkdir(parents=True)
        ct_dir = course / "CT"
        series_uid = write_ct(ct_dir, rows=21, columns=21, slices=13, spacing=(1.0, 1.0, spacing_z))
        arr = cylinder((21, 21, 13), (10, 10, 6), (5, 5, 2))
        rs_path = course / "RS.dcm"
        write_rtstruct(ct_dir, rs_path, [("ROI", arr)])
        identity = rr.RobustnessRoiIdentity.from_mapping(
            dict(
                patient_id="P1",
                course_id="C1",
                series_uid=series_uid,
                segmentation_source="Manual",
                mask_identity="mask-c2",
                roi_original_name="ROI",
                stable_roi_identifier="roi-c2",
            )
        )
        monkeypatch.setattr(
            rr,
            "load_course_contract",
            lambda _: SimpleNamespace(
                planning_ct_dir=ct_dir,
                planning_ct_nifti=None,
                authoritative_rtstruct_path=rs_path,
                authoritative_rtstruct_source="Manual",
                planning_ct={"series_instance_uid": series_uid},
            ),
        )
        monkeypatch.setattr(rr, "_load_main_ct_identity_catalog", lambda *a, **k: ({("Manual", "ROI"): identity}, {}))
        monkeypatch.setattr(rm, "_standard_rtstruct_sources", lambda *a: [("Manual", rs_path, None)])
        monkeypatch.setattr("rtpipeline.custom_models.list_custom_model_outputs", lambda *a: [])
        monkeypatch.setenv("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "1")
        monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")
        monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")

        def fake_extract(image, masks, *a, **k):
            out = []
            for pid, m in masks.items():
                for arm in CT_EXTRACTION_ARMS:
                    out.append(
                        {
                            **identity.as_dict(),
                            "structure": "ROI",
                            "modality": "CT",
                            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                            "perturbed_mask_identity": rr._perturbed_mask_identity(m),
                            "perturbation_id": pid,
                            "extraction_arm": arm,
                            "run_identifier": k["run_identifier"],
                            "feature_name": "original_firstorder_Mean",
                            "value": 10.0,
                        }
                    )
            return pd.DataFrame(out)

        monkeypatch.setattr(rr, "extract_features_for_masks", fake_extract)
        cfg = PipelineConfig(tmp_path / tag, tmp_path / tag, tmp_path / tag, max_workers_override=1)
        rob = rr.RobustnessConfig(
            enabled=True,
            perturbation=rr.PerturbationConfig(
                small_volume_changes=[0.0],
                max_translation_mm=shift,
                n_random_contour_realizations=0,
                noise_levels=[0.0],
                apply_to_structures=["ROI"],
            ),
        )
        out_pq = rr.robustness_for_course(cfg, rob, course)
        df = pd.read_parquet(out_pq)
        sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
        return df, sidecar

    if shift == 1e-5:
        frame, sidecar = _run_spacing("subvoxel", 5.0)
        for sign in (1, -1):
            requested = [0.0, 0.0, sign * shift]
            evidence = next(e for e in sidecar["translation_realizations"]
                            if e["requested_translation_mm"] == requested)
            key = evidence["translation_suffix"]
            rows = frame.loc[frame["perturbation_id"].eq(f"ntcv{key}_v0")]
            assert len(rows) == len(CT_EXTRACTION_ARMS)
            assert rows["reason_code"].eq("translation_not_realizable").all()
            assert rows["requested_translation_mm"].map(json.loads).tolist() == [requested] * len(rows)
            assert rows["realized_translation_mm"].map(json.loads).tolist() == [[0., 0., 0.]] * len(rows)
            assert evidence["realized_translation_mm"] == [0.0, 0.0, 0.0]
            assert key == ("_t0_0_0p00001" if sign == 1 else "_t0_0_-0p00001")
        return

    df_5mm, sc_5mm = _run_spacing("s5", 5.0)
    t_5mm = {e["translation_suffix"]: e for e in sc_5mm["translation_realizations"]}
    assert t_5mm["_t0_0_4"]["requested_translation_mm"] == [0.0, 0.0, 4.0]
    assert t_5mm["_t0_0_4"]["realized_translation_mm"] == [0.0, 0.0, 5.0]
    row_5mm = df_5mm[df_5mm["perturbation_id"] == "ntcv_t0_0_4_v0"].iloc[0]
    assert json.loads(row_5mm["requested_translation_mm"]) == [0.0, 0.0, 4.0]
    assert json.loads(row_5mm["realized_translation_mm"]) == [0.0, 0.0, 5.0]

    df_10mm, sc_10mm = _run_spacing("s10", 10.0)
    t_10mm = {e["translation_suffix"]: e for e in sc_10mm["translation_realizations"]}
    assert t_10mm["_t0_0_4"]["requested_translation_mm"] == [0.0, 0.0, 4.0]
    assert t_10mm["_t0_0_4"]["realized_translation_mm"] == [0.0, 0.0, 0.0]
    imp_10mm = df_10mm[df_10mm["perturbation_id"] == "ntcv_t0_0_4_v0"]
    assert (imp_10mm["robustness_status"] == "geometrically_impossible").all()
    assert (imp_10mm["reason_code"] == "translation_not_realizable").all()
    assert json.loads(imp_10mm.iloc[0]["requested_translation_mm"]) == [0.0, 0.0, 4.0]
    assert json.loads(imp_10mm.iloc[0]["realized_translation_mm"]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("value", [-4.0, 0.0, 1.25, -1e-5, 5e-324])
def test_r3_translation_key_accepts_every_fixed_point_form(value):
    encoded = rr._format_ntcv_scalar(value)
    assert "e" not in encoded
    key = f"_t{encoded}_0_{encoded}"
    assert rr._ntcv_translation_key_for_pert_id(f"ntcv_n10{key}_c2_v+15") == key
    assert rr._ntcv_translation_key_for_pert_id("ntcv_t0_0_-1p25e-05_v0") == "_t0_0_-1p25e-05"


# ---------------------------------------------------------------------------
# C3 & A4: Boundary clipping, 16-attempt exhaustion, max_translation_mm=0, float IDs
# ---------------------------------------------------------------------------


def test_c3_boundary_clipped_contour_and_max_trans_zero_and_float_ids():
    # Edge-touching ROI: outward randomization would clip against x=0 / z=0 boundary
    arr = np.zeros((9, 12, 12), dtype=np.uint8)
    arr[0:4, 0:4, 0:4] = 1
    mask = sitk.GetImageFromArray(arr)
    img = sitk.GetImageFromArray(np.ones_like(arr, dtype=np.float32))

    # Zero translation with default n_random_contour_realizations=2 and contour_randomization_mm=0.0 must not crash
    cfg_zero_t = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=0.0,
        contour_randomization_mm=0.0,
        n_random_contour_realizations=2,
        noise_levels=[0.0],
    )
    masks_edge, _ = rr.generate_ntcv_perturbations(mask, img, cfg_zero_t, "EDGE_ROI")
    assert len(masks_edge) == 3
    # Boundary-clipped contour realizations must be marked as GeometricNonmeasurement (not silently truncated)
    assert any(
        isinstance(v, rro.GeometricNonmeasurement)
        and (("clip" in v.reason_code) or ("outside" in v.reason_code))
        for k, v in masks_edge.items()
        if "_c" in k
    )

    # Fractional noise & translation values must not collide under int() truncation
    interior = np.zeros((15, 16, 16), dtype=np.uint8)
    interior[4:11, 4:12, 4:12] = 1
    interior_mask = sitk.GetImageFromArray(interior)
    interior_img = sitk.GetImageFromArray(np.ones_like(interior, dtype=np.float32))
    cfg_frac = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=1.5,
        n_random_contour_realizations=0,
        noise_levels=[0.0, 0.5, 0.8],
    )
    masks_frac, _ = rr.generate_ntcv_perturbations(interior_mask, interior_img, cfg_frac, "ROI")
    assert len(masks_frac) == 9
    assert rr._ntcv_noise_suffix(0.5000001) != rr._ntcv_noise_suffix(0.5000002)
    assert rr._ntcv_volume_suffix(0.15) == "_v+15"
    assert rr._ntcv_volume_suffix(-0.15) == "_v-15"
    assert rr._ntcv_volume_suffix(0.151) != rr._ntcv_volume_suffix(0.152)


def test_c3_unavailable_contour_after_16_attempts_gives_disposition_not_runtime_error(monkeypatch):
    # Real 4^3 cube on 3 mm isotropic spacing with contour_randomization_mm=2.0:
    # every inward/outward offset in [1.0, 2.0] mm < 3.0 mm leaves the binary mask
    # unchanged, exhausting 16 attempts cleanly without raising RuntimeError.
    cube = np.zeros((16, 16, 16), dtype=np.uint8)
    cube[6:10, 6:10, 6:10] = 1
    cube_mask = sitk.GetImageFromArray(cube)
    cube_mask.SetSpacing((3.0, 3.0, 3.0))
    cube_img = sitk.GetImageFromArray(np.ones_like(cube, dtype=np.float32))
    cube_img.SetSpacing((3.0, 3.0, 3.0))
    real_cfg = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=0.0,
        contour_randomization_mm=2.0,
        n_random_contour_realizations=2,
        noise_levels=[0.0],
    )
    real_masks, _ = rr.generate_ntcv_perturbations(cube_mask, cube_img, real_cfg, "CUBE4")
    assert len(real_masks) == 3
    real_c2 = [k for k in real_masks if "_c2_" in k]
    assert len(real_c2) == 1
    assert isinstance(real_masks[real_c2[0]], rro.GeometricNonmeasurement)
    assert real_masks[real_c2[0]].reason_code == "contour_realization_unavailable"

    arr = np.zeros((11, 11, 11), dtype=np.uint8)
    arr[3:8, 3:8, 3:8] = 1
    mask = sitk.GetImageFromArray(arr)
    img = sitk.GetImageFromArray(np.ones_like(arr, dtype=np.float32))
    # Force randomize_contour to always return identical mask so 16 attempts cannot find distinct realization
    monkeypatch.setattr(rr, "randomize_contour", lambda m, *_a, **_k: m)
    cfg = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=0.0,
        contour_randomization_mm=2.0,
        n_random_contour_realizations=1,
        noise_levels=[0.0],
    )
    masks, _ = rr.generate_ntcv_perturbations(mask, img, cfg, "ROI")
    assert len(masks) == 2
    c_keys = [k for k in masks if "_c1_" in k]
    assert len(c_keys) == 1
    assert isinstance(masks[c_keys[0]], rro.GeometricNonmeasurement)
    assert masks[c_keys[0]].reason_code == "contour_realization_unavailable"


# ---------------------------------------------------------------------------
# C4: Sidecar and admission bind CT pixel identity and baseline radiomics_ct.parquet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch_parameters", [False, True])
def test_c4_sidecar_binds_ct_pixels_and_baseline_publication(tmp_path, monkeypatch, switch_parameters):
    from rasterizer_test_utils import stub_synthetic_rasterizer_columns

    stub_synthetic_rasterizer_columns(monkeypatch)
    monkeypatch.setattr("rtpipeline.radiomics_rasterizer.require_radiomics_rasterizer", lambda: None)
    course = tmp_path / "P1" / "C1"
    course.mkdir(parents=True)
    ct_dir = course / "CT"
    series_uid = write_ct(ct_dir, rows=21, columns=21, slices=9, spacing=(1.0, 1.0, 2.0))
    arr = cylinder((21, 21, 9), (10, 10, 4), (5, 5, 2))
    rs_path = course / "RS.dcm"
    write_rtstruct(ct_dir, rs_path, [("PTV", arr)])

    identity = rr.RobustnessRoiIdentity.from_mapping(
        dict(
            patient_id="P1",
            course_id="C1",
            series_uid=series_uid,
            segmentation_source="Manual",
            mask_identity="mask-id",
            roi_original_name="PTV",
            stable_roi_identifier="roi-1",
        )
    )
    # Write real authoritative radiomics_ct.parquet on disk
    from rtpipeline.radiomics_ct_contract import (
        _feature_schema_metadata,
        PRIMARY_ARM,
        configured_parameter_hash,
        disposition_rows_for_arms,
        write_ct_publication_atomic,
    )
    from rasterizer_test_utils import PINNED_RASTERIZER_ROW

    custom_params = tmp_path / "custom_ct_params.yaml"
    original_params_text = (ROOT / "rtpipeline" / "radiomics_params.yaml").read_text(encoding="utf-8")
    custom_params.write_text(original_params_text, encoding="utf-8")
    decision = classify_ct_roi("Manual", "PTV")
    rows = disposition_rows_for_arms(
        {
            "modality": "CT",
            "segmentation_source": "Manual",
            "roi_name": "PTV",
            "roi_original_name": "PTV",
            "patient_id": "P1",
            "course_id": "C1",
            "series_uid": series_uid,
            "mask_identity": "mask-id",
            "stable_roi_identifier": "roi-1",
        },
        decision=decision,
        disposition="success",
        detail="ok",
        failure_kind="none",
        run_identifier="base-run",
        code_revision="rev",
        native_voxel_count=int(arr.sum()),
        required=True,
        configured_parameter_hashes={
            arm: configured_parameter_hash(
                custom_params, arm=arm,
                window=decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None,
                large_roi=False,
            )
            for arm in CT_EXTRACTION_ARMS
        },
        effective_hashes={a: f"eff-{a}" for a in CT_EXTRACTION_ARMS},
    )
    for r in rows:
        r.update(PINNED_RASTERIZER_ROW)
        r["original_firstorder_Mean"] = 10.0
        r.update(_feature_schema_metadata({"original_firstorder_Mean"}))
    baseline_pq = course / "radiomics_ct.parquet"
    baseline_xlsx = course / "radiomics_ct.xlsx"
    write_ct_publication_atomic(pd.DataFrame(rows), baseline_xlsx)
    done_file = course / ".radiomics_done"
    done_file.write_text(json.dumps({"status": "ok", "run": "base-run"}), encoding="utf-8")

    monkeypatch.setattr(
        rr,
        "load_course_contract",
        lambda _: SimpleNamespace(
            planning_ct_dir=ct_dir,
            planning_ct_nifti=None,
            authoritative_rtstruct_path=rs_path,
            authoritative_rtstruct_source="Manual",
            planning_ct={"series_instance_uid": series_uid},
        ),
    )
    monkeypatch.setattr(rm, "_standard_rtstruct_sources", lambda *a: [("Manual", rs_path, None)])
    monkeypatch.setattr("rtpipeline.custom_models.list_custom_model_outputs", lambda *a: [])
    monkeypatch.setenv("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")

    def fake_extract(image, masks, *a, **k):
        out = []
        for pid, m in masks.items():
            for arm in CT_EXTRACTION_ARMS:
                out.append(
                    {
                        **identity.as_dict(),
                        "structure": "PTV",
                        "modality": "CT",
                        "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                        "perturbed_mask_identity": rr._perturbed_mask_identity(m),
                        "perturbation_id": pid,
                        "extraction_arm": arm,
                        "run_identifier": k["run_identifier"],
                        "feature_name": "original_firstorder_Mean",
                        "value": 10.0,
                    }
                )
        return pd.DataFrame(out)

    monkeypatch.setattr(rr, "extract_features_for_masks", fake_extract)
    cfg = PipelineConfig(
        tmp_path,
        tmp_path,
        tmp_path,
        max_workers_override=1,
        radiomics_params_file=custom_params,
    )
    rob = rr.RobustnessConfig(
        enabled=True,
        perturbation=rr.PerturbationConfig(
            small_volume_changes=[-0.15, 0.0],
            max_translation_mm=0.0,
            n_random_contour_realizations=0,
            noise_levels=[0.0],
            apply_to_structures=["PTV"],
        ),
    )
    if switch_parameters:
        different = yaml.safe_load(original_params_text)
        different["setting"]["binWidth"] = 50
        cfg.radiomics_params_file = tmp_path / "different_ct_params.yaml"
        cfg.radiomics_params_file.write_text(yaml.safe_dump(different), encoding="utf-8")
        with pytest.raises(RuntimeError, match="configured_parameter_hash"):
            rr.robustness_for_course(cfg, rob, course)
        assert not (course / "radiomics_robustness_ct.parquet").exists()
        assert not rr.robustness_source_dispositions_path(course).exists()
        return

    table_path = rr.robustness_for_course(cfg, rob, course)
    sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    run_id = sidecar["robustness_run_identifier"]
    assert sidecar["ct_source_binding"]["series_uid"] == series_uid
    assert sidecar["ct_source_binding"]["file_count"] == 9
    assert sidecar["baseline_ct_publication"] == {
        "radiomics_ct.parquet": rr._file_sha256(baseline_pq),
        ".radiomics_done": rr._file_sha256(done_file),
    }
    assert rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob) is not None

    # Mutate one CT slice pixel while keeping SeriesInstanceUID & filename
    ct_slice = sorted(ct_dir.glob("*.dcm"))[0]
    original_ct_bytes = ct_slice.read_bytes()
    ds = pydicom.dcmread(ct_slice)
    px = ds.pixel_array.copy()
    px[0, 0] += 500
    ds.PixelData = px.astype(np.int16).tobytes()
    ds.save_as(str(ct_slice))
    with pytest.raises((ValueError, RuntimeError)):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)
    with pytest.raises((ValueError, RuntimeError)):
        rr._admit_robustness_aggregation_input(table_path, rob)

    # Restore CT slice, mutate custom radiomics_params_file -> admission refuses
    ct_slice.write_bytes(original_ct_bytes)
    assert rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob) is not None
    custom_params.write_text(original_params_text + "\n# mutated\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="parameter"):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)
    custom_params.write_text(original_params_text, encoding="utf-8")

    # Mutate .radiomics_done -> admission refuses
    done_bytes = done_file.read_bytes()
    done_file.write_text(json.dumps({"status": "ok", "run": "other-run"}), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="radiomics_done"):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)
    done_file.write_bytes(done_bytes)

    # Mutate baseline radiomics_ct.parquet -> admission refuses
    assert rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob) is not None
    df_base = pd.read_parquet(baseline_pq)
    df_base["original_firstorder_Mean"] = 999.0
    write_ct_publication_atomic(df_base, baseline_xlsx)
    with pytest.raises((ValueError, RuntimeError)):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


def test_r3_measured_sidecar_requires_parameter_binding(tmp_path, monkeypatch):
    from test_robustness_aggregation_admission import _build_cohort

    cohort = _build_cohort(tmp_path, monkeypatch, patients=("P1",))
    sidecar = rr.robustness_source_dispositions_path(cohort.courses[0])
    payload = json.loads(sidecar.read_text())
    assert payload.pop("extraction_parameters_binding")["sha256"]
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extraction parameters binding"):
        rr._admit_robustness_aggregation_input(cohort.tables[0], cohort.rob)


def test_r3_default_parameter_binding_uses_package(tmp_path):
    logs = tmp_path / "Logs"
    logs.mkdir()
    stale = logs / "radiomics_params.yaml"
    stale.write_text("setting: {binWidth: 50}")
    config = PipelineConfig(tmp_path, tmp_path, logs)
    packaged = ROOT / "rtpipeline" / "radiomics_params.yaml"

    assert rm._get_params_file(config, "CT") == packaged
    assert rr._compute_extraction_parameters_binding(config)["path"] == str(packaged)
    assert stale.read_text() == "setting: {binWidth: 50}"


@pytest.mark.parametrize("mode", ["inputs", "manifest"])
def test_r3_aggregate_refuses_different_configured_ct_parameters(tmp_path, monkeypatch, caplog, mode):
    from test_robustness_cohort_snapshot_contract import _fixture

    fixture, manifest, _, _, courses = _fixture(tmp_path, monkeypatch, ("measured",))
    output = tmp_path / "summary.xlsx"
    args = ["radiomics-robustness-aggregate", "--config", str(fixture.config_path),
            "--output", str(output)]
    if mode == "inputs":
        args += ["--inputs", str(courses[0].measured_output)]
    else:
        args += ["--manifest", str(manifest), "--output-root", str(fixture.output_root)]

    config = yaml.safe_load(fixture.config_path.read_text())
    params = fixture.config_path.parent / "active_params.yaml"
    packaged = ROOT / "rtpipeline" / "radiomics_params.yaml"
    params.write_bytes(packaged.read_bytes())
    config["radiomics"] = {"params_file": params.name}
    fixture.config_path.write_text(yaml.safe_dump(config))
    assert cli.main(args) == 0

    params.write_text("setting: {binWidth: 50}")
    assert cli.main(args) == 1
    assert "configured CT parameter file does not match" in caplog.text
    assert all(not path.exists() for path in rr.robustness_cohort_output_paths(output))


# ---------------------------------------------------------------------------
# C5: Refuse HU extraction & noise when DICOM RescaleType / units are not HU
# ---------------------------------------------------------------------------


def test_c5_non_hu_rescale_type_refuses_hu_extraction_and_noise(tmp_path):
    ct_dir = tmp_path / "CT_NON_HU"
    _write_ct_series_custom(ct_dir, rescale_type="US")
    desc = acquisition_scale.describe_planning_ct(ct_dir)
    assert desc["acq_provenance_status"] == "non_hu_rescale_units"
    assert desc["acq_scale_class"] == "unknown"
    for bounds in ("representable", "observed", "effective"):
        for endpoint in ("min", "max"):
            assert desc[f"acq_{bounds}_hu_{endpoint}"] is None

    img = rm._load_series_image(ct_dir)
    assert img is not None
    with pytest.raises((ValueError, RuntimeError), match="HU"):
        rr.add_noise_to_image(img, 10.0)


# ---------------------------------------------------------------------------
# C6: CT cache keyed by series_uid and content binding (no sids[0] fallback)
# ---------------------------------------------------------------------------


def test_c6_series_image_cache_keyed_by_uid_and_content(tmp_path):
    rm.clear_image_cache()
    mixed_dir = tmp_path / "MIXED_CT"
    uid1 = _write_ct_series_custom(mixed_dir, slices=4, series_uid="1.2.840.10008.1.1", value_offset=0)
    uid2 = _write_ct_series_custom(mixed_dir, slices=4, series_uid="1.2.840.10008.2.2", value_offset=300)

    img1 = rm._load_series_image(mixed_dir, series_uid=uid1, use_cache=True)
    img2 = rm._load_series_image(mixed_dir, series_uid=uid2, use_cache=True)
    assert img1 is not None and img2 is not None
    assert float(sitk.GetArrayViewFromImage(img2).mean()) > float(sitk.GetArrayViewFromImage(img1).mean()) + 200

    # Requesting a non-existent series_uid must NEVER silently fall back to sids[0]
    assert rm._load_series_image(mixed_dir, series_uid="1.2.840.99999.9", use_cache=True) is None

    # Mutating slice pixel data with preserved mtime must invalidate cache
    single_dir = tmp_path / "SINGLE_CT"
    uid_single = _write_ct_series_custom(single_dir, slices=3, series_uid="1.2.840.10008.3.3", value_offset=0)
    before = rm._load_series_image(single_dir, series_uid=uid_single, use_cache=True)
    mean_before = float(sitk.GetArrayViewFromImage(before).mean())

    for dcm in sorted(single_dir.glob("*.dcm")):
        st = dcm.stat()
        ds = pydicom.dcmread(dcm)
        ds.PixelData = (ds.pixel_array + 400).astype(np.int16).tobytes()
        ds.save_as(str(dcm))
        os.utime(dcm, ns=(st.st_atime_ns, st.st_mtime_ns))

    after = rm._load_series_image(single_dir, series_uid=uid_single, use_cache=True)
    mean_after = float(sitk.GetArrayViewFromImage(after).mean())
    assert mean_after > mean_before + 300


# ---------------------------------------------------------------------------
# C7 & A3: IBSI guard 0-axis preservation + NumPy-2 refusal + label normalization
# ---------------------------------------------------------------------------


def test_c7_a3_resource_guard_allows_zero_preserving_axis_and_ibsi_numpy2_guard(
    tmp_path, monkeypatch
):
    mask = np.ones((4, 6, 8), dtype=bool)
    est = rg.estimate_resampled_bounding_box(
        mask,
        native_spacing_xyz=(1.0, 1.0, 2.5),
        resampled_spacing_xyz=(2.0, 2.0, 0.0),
        array_axis_to_xyz=(2, 1, 0),
        pad_distance=5,
    )
    assert est.estimated_resampled_bbox_voxels > 0

    ibsi_yaml = tmp_path / "ibsi_ct.yaml"
    ibsi_yaml.write_text(
        "radiomics:\n  preprocessing: ibsi\nsetting:\n  binWidth: 25\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(np, "__version__", "2.0.0")
    with pytest.raises(ValueError, match=r"NumPy 1\.x/PyRadiomics"):
        rm.build_ct_extractor_from_params(ibsi_yaml)
    with pytest.raises(ValueError, match=r"NumPy 1\.x/PyRadiomics"):
        rc.make_extractor(ibsi_yaml)

    from rtpipeline import radiomics_parallel, radiomics_runtime

    def unexpected_driver(*args, **kwargs):
        pytest.fail("IBSI configuration reached the delegated course driver")

    monkeypatch.setattr(radiomics_runtime, "run_course_driver", unexpected_driver)
    config = PipelineConfig(tmp_path, tmp_path, tmp_path, radiomics_params_file=ibsi_yaml)
    for driver in (rm.radiomics_for_course, radiomics_parallel.parallel_radiomics_for_course):
        with pytest.raises(ValueError, match=r"NumPy 1\.x/PyRadiomics"):
            driver(config, tmp_path)
    with pytest.raises(ValueError, match=r"NumPy 1\.x/PyRadiomics"):
        rc.radiomics_for_course(tmp_path, config)

    # Native (non-IBSI) params with a top-level radiomics: block must strip radiomics:
    native_yaml = tmp_path / "native_with_radiomics_key.yaml"
    native_yaml.write_text(
        "radiomics:\n  preprocessing: native\nsetting:\n  binWidth: 30\nimageType:\n  Original: {}\nfeatureClass:\n  firstorder: [Mean]\n",
        encoding="utf-8",
    )
    applied_settings = {}

    class _RecorderExtractor:
        featureClassNames = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")

        def __init__(self):
            self.settings = {}
            self.enabledImagetypes = {}
            self.enabledFeatures = {}

        def disableAllImageTypes(self):
            self.enabledImagetypes.clear()

        def enableImageTypeByName(self, name, **kwargs):
            self.enabledImagetypes[name] = kwargs

        def disableAllFeatures(self):
            self.enabledFeatures.clear()

        def enableFeaturesByName(self, **kwargs):
            self.enabledFeatures.update(kwargs)

        def enableFeatureClassByName(self, name):
            self.enabledFeatures[name] = []

        def _setTolerance(self):
            pass

    ext = rm.build_ct_extractor_from_params(native_yaml, featureextractor_cls=_RecorderExtractor)
    applied_settings.update(ext.settings)
    assert applied_settings["binWidth"] == 30
    assert "radiomics" not in applied_settings
    assert "Original" in ext.enabledImagetypes
    assert "firstorder" in ext.enabledFeatures


def test_c7_a3_extraction_nonmeasurement_normalizes_ibsi_binary_label():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 7
    mask = sitk.GetImageFromArray(arr)
    image = sitk.GetImageFromArray(arr.astype(np.float32))

    class FakeIBSIExtractor:
        settings = {
            "label": 7,
            "rtpipelinePreprocessing": "ibsi",
            "minimumROISize": 64,
            "minimumROIDimensions": 2,
        }

        def loadImage(self, image, mask, *_args, **settings):
            binary = sitk.GetArrayFromImage(mask) == settings["label"]
            loaded_mask = sitk.GetImageFromArray(binary.astype(np.uint8))
            loaded_mask.CopyInformation(mask)
            return image, loaded_mask

    outcome = rro.extraction_nonmeasurement(image, mask, FakeIBSIExtractor)
    assert outcome is not None
    assert outcome.reason_code == "resampled_mask_below_minimum_voxels"
    assert outcome.evidence["resampled_voxels"] == 27


# ---------------------------------------------------------------------------
# A1 & C8: Aggregator dispositions for geometrically_impossible, NaN ICC -> not_evaluable, docs
# ---------------------------------------------------------------------------


def test_a1_aggregator_gives_geometrically_impossible_disposition_and_nan_icc_not_evaluable(
    tmp_path, monkeypatch
):
    from test_robustness_nonmeasurements import course_fixture

    # Two courses so aggregate_robustness_results / summarize_feature_stability runs on cohort
    c1, cfg1, rob, _ = course_fixture(tmp_path / "s1", monkeypatch)
    t1 = rr.robustness_for_course(cfg1, rob, c1)
    c2, cfg2, _, _ = course_fixture(tmp_path / "s2", monkeypatch)
    # Give P2 distinct patient_id so cohort has 2 subjects
    df1 = pd.read_parquet(t1)
    df2 = df1.copy()
    df2["patient_id"] = "P2"
    df2["stable_roi_identifier"] = "roi-2"

    # Must NOT raise ValueError; must emit an explicit disposition for the geometrically_impossible ROI/group
    summary = rr.summarize_feature_stability(pd.concat([df1, df2], ignore_index=True), rob)
    assert not summary.empty
    assert set(summary["robustness_label"]) <= {"not_evaluable", "geometrically_impossible"}

    # NaN ICC with complete CoV must be 'not_evaluable', NEVER 'poor'
    const_df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "course_id": ["C1"] * 6,
            "structure": ["ROI"] * 6,
            "segmentation_source": ["Manual"] * 6,
            "extraction_arm": [CT_EXTRACTION_ARMS[0]] * 6,
            "perturbation_id": ["a", "b"] * 3,
            "feature_name": ["f1"] * 6,
            "value": [42.0] * 6,
        }
    )
    const_config = rr.RobustnessConfig(perturbation=rr.PerturbationConfig(
        max_translation_mm=0.0, n_random_contour_realizations=0,
        small_volume_changes=[0.0], noise_levels=[0.0, 10.0],
    ))
    const_summary = rr.summarize_feature_stability(const_df, const_config)
    assert len(const_summary) == 1
    assert np.isnan(const_summary.loc[0, "icc"])
    assert const_summary.loc[0, "cov_status"] == "complete"
    assert const_summary.loc[0, "robustness_label"] == "not_evaluable"

    # Docs and docstrings explicitly clarify this package summary is NOT O7
    docs_text = (ROOT / "docs" / "RADIOMICS_ROBUSTNESS.md").read_text(encoding="utf-8")
    assert "O7" in docs_text
    assert "O7" in (rr.summarize_feature_stability.__doc__ or "")
    assert "O7" in (rr.aggregate_robustness_results.__doc__ or "")
    assert "O7" in (ra.aggregate_robustness_results.__doc__ or "")


# ---------------------------------------------------------------------------
# C9: Opt-in course-specific noise and contour seeds (default unchanged)
# ---------------------------------------------------------------------------


def test_c9_opt_in_course_specific_seeds_and_default_unchanged():
    arr = np.zeros((13, 15, 15), dtype=np.uint8)
    arr[3:10, 4:11, 4:11] = 1
    mask = sitk.GetImageFromArray(arr)
    img = sitk.GetImageFromArray(np.full_like(arr, 40.0, dtype=np.float32))

    # Default config keeps identical draws regardless of course context
    default_cfg = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=0.0,
        contour_randomization_mm=2.0,
        n_random_contour_realizations=1,
        noise_levels=[10.0],
    )
    assert getattr(default_cfg, "course_specific_seeds", False) is False
    # Check both default fixed-seed keys (with n10) and opt-in course_specific_seeds
    m_a, i_a = rr.generate_ntcv_perturbations(mask, img, default_cfg, "ROI")
    m_b, i_b = rr.generate_ntcv_perturbations(mask, img, default_cfg, "ROI")
    assert np.array_equal(sitk.GetArrayFromImage(m_a["ntcv_n10_c1_v0"]), sitk.GetArrayFromImage(m_b["ntcv_n10_c1_v0"]))
    assert np.array_equal(sitk.GetArrayFromImage(i_a["ntcv_n10_v0"]), sitk.GetArrayFromImage(i_b["ntcv_n10_v0"]))

    # Opt-in course_specific_seeds=True differentiates courses deterministically
    opt_cfg = rr.PerturbationConfig(
        small_volume_changes=[0.0],
        max_translation_mm=0.0,
        contour_randomization_mm=2.0,
        n_random_contour_realizations=1,
        noise_levels=[10.0],
        course_specific_seeds=True,
    )
    m_p1, i_p1 = rr.generate_ntcv_perturbations(mask, img, opt_cfg, "ROI", course_seed_key="P1/C1/1.2.3")
    m_p1_rep, i_p1_rep = rr.generate_ntcv_perturbations(mask, img, opt_cfg, "ROI", course_seed_key="P1/C1/1.2.3")
    m_p2, i_p2 = rr.generate_ntcv_perturbations(mask, img, opt_cfg, "ROI", course_seed_key="P2/C1/1.2.4")

    assert np.array_equal(sitk.GetArrayFromImage(i_p1["ntcv_n10_v0"]), sitk.GetArrayFromImage(i_p1_rep["ntcv_n10_v0"]))
    assert not np.array_equal(sitk.GetArrayFromImage(i_p1["ntcv_n10_v0"]), sitk.GetArrayFromImage(i_p2["ntcv_n10_v0"]))
    assert not np.array_equal(sitk.GetArrayFromImage(m_p1["ntcv_n10_c1_v0"]), sitk.GetArrayFromImage(m_p2["ntcv_n10_c1_v0"]))


@pytest.mark.parametrize("bin_settings", [{"binWidth": 25}, {}, {"binCount": 32}])
@pytest.mark.parametrize("normalize", [False, True])
def test_c5_dual_arm_ct_refuses_hu_texture_on_non_hu_rescale(monkeypatch, bin_settings, normalize):
    from rtpipeline.radiomics_ct_contract import extract_ct_roi_arms, PRIMARY_ARM, SENSITIVITY_ARM
    from test_radiomics_ct_dual_arm import _FakeExtractor, _common_identity, _qc, _versions

    monkeypatch.setattr("rtpipeline.radiomics_ct_contract.resampled_mask_qc", _qc)
    monkeypatch.setattr("rtpipeline.radiomics_ct_contract._runtime_versions", _versions)

    arr = np.ones((8, 8, 8), dtype=np.float32) * 50.0
    img = sitk.GetImageFromArray(arr)
    img.SetMetaData("rtpipeline_hu_verified", "0")
    img.SetMetaData("rtpipeline_hu_status", "non_hu_rescale_units")
    img.SetMetaData("rtpipeline_hu_detail", "RescaleType='US'")
    mask = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.uint8))

    def factory():
        extractor = _FakeExtractor()
        extractor.settings.pop("binWidth")
        extractor.settings.update(bin_settings, normalize=normalize)
        return extractor

    rows = extract_ct_roi_arms(
        img,
        mask,
        factory=factory,
        decision=classify_ct_roi("Manual", "PTV"),
        common_metadata=_common_identity(),
        run_identifier="run-non-hu",
        code_revision="rev",
        native_voxel_count=512,
        required=True,
        configured_parameter_hashes={PRIMARY_ARM: "cfg-p", SENSITIVITY_ARM: "cfg-s"},
    )
    by_arm = {r["extraction_arm"]: r for r in rows}
    assert by_arm[PRIMARY_ARM]["shape_disposition"] == "success"
    assert by_arm[PRIMARY_ARM]["intensity_texture_disposition"] == "non_hu_rescale_units"
    assert "original_shape_VoxelVolume" in by_arm[PRIMARY_ARM]
    assert "original_firstorder_Mean" not in by_arm[PRIMARY_ARM]
    raw = by_arm[SENSITIVITY_ARM]
    assert raw["shape_disposition"] == "success"
    assert raw["intensity_texture_disposition"] == ("success" if normalize else "non_hu_rescale_units")
    assert ("original_firstorder_Mean" in raw) == normalize


def test_c5_sequential_conda_robustness_propagates_ct_hu_metadata(monkeypatch):
    from rtpipeline.radiomics_ct_contract import PRIMARY_ARM, SENSITIVITY_ARM

    arr = np.ones((8, 8, 8), dtype=np.float32) * 50.0
    img = sitk.GetImageFromArray(arr)
    img.SetMetaData("rtpipeline_hu_verified", "0")
    img.SetMetaData("rtpipeline_hu_status", "non_hu_rescale_units")
    img.SetMetaData("rtpipeline_hu_detail", "RescaleType='US'")
    mask = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.uint8))

    identity = rr.RobustnessRoiIdentity.from_mapping(
        dict(
            patient_id="P1",
            course_id="C1",
            series_uid="1.2.3",
            segmentation_source="Manual",
            mask_identity="m1",
            roi_original_name="ROI",
            stable_roi_identifier="roi-1",
        )
    )
    seen_tasks = []

    def fake_conda_batch(tasks, params_path, timeout_per_roi=120):
        seen_tasks.extend(tasks)
        t = tasks[0]
        rec = {
            **t["metadata"],
            "structure": "ROI",
            "roi_name": "ROI",
            "patient_id": "P1",
            "course_id": "C1",
            "segmentation_source": "Manual",
            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
            "perturbation_id": t["robustness_perturbation_id"],
            "perturbed_mask_identity": rr._perturbed_mask_identity(mask),
            "run_identifier": t["run_identifier"],
            "code_revision": t["code_revision"],
            "native_voxel_count": t["native_voxel_count"],
            "roi_class": "target",
            "roi_map_version": "1",
            "roi_map_hash": "h",
            "roi_map_entry_source": "src",
            "roi_class_adjudication_status": "adjudicated",
            "effective_resegment_lower_hu": None,
            "effective_resegment_upper_hu": None,
            "configured_parameter_hash": "cfg",
            "effective_parameter_hash": "eff",
            "shape_disposition": "success",
            "intensity_texture_disposition": "non_hu_rescale_units",
            "feature_extraction_disposition": "partial_shape_only",
            "disposition_detail": "non-HU",
            "original_shape_VoxelVolume": 512.0,
        }
        return [
            {
                "__task_index__": 0,
                "__status__": "success",
                "__records__": [
                    {**rec, "extraction_arm": PRIMARY_ARM},
                    {**rec, "extraction_arm": SENSITIVITY_ARM},
                ],
            }
        ]

    monkeypatch.setattr(rm, "_extractor", lambda *a, **k: None)
    monkeypatch.setattr(rc, "extract_radiomics_batch_with_conda", fake_conda_batch)
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    df = rr.extract_features_for_masks(
        img,
        {"ntcv_v0": mask},
        SimpleNamespace(),
        modality="CT",
        structure_name="ROI",
        patient_id="P1",
        course_id="C1",
        segmentation_source="Manual",
        run_identifier="run-seq",
        source_identity=identity,
    )
    assert len(seen_tasks) == 1
    meta = seen_tasks[0]["metadata"]
    assert meta["_ct_hu_verified"] is False
    assert meta["_ct_hu_status"] == "non_hu_rescale_units"
    assert "RescaleType='US'" in meta["_ct_hu_detail"]
    assert not df.empty


@pytest.mark.parametrize("recorded_hash", ["", "stale-hash"])
def test_c4_admission_checks_declared_roi_map_identity(tmp_path, monkeypatch, recorded_hash):
    import test_robustness_aggregation_admission as admission
    from rtpipeline.radiomics_ct_contract import roi_class_map_identity

    cohort = admission._build_cohort(tmp_path, monkeypatch)
    path = cohort.tables[0]
    frame = pd.read_parquet(path)
    frame["roi_map_version"], _ = roi_class_map_identity()
    frame["roi_map_hash"] = recorded_hash
    frame.to_parquet(path, index=False)

    # Update the artifact digests so this probe reaches the ROI map check.
    sidecar_path = rr.robustness_source_dispositions_path(path.parent)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["measured_output"]["sha256"] = rr._file_sha256(path)
    sidecar["measured_output"]["size_bytes"] = path.stat().st_size
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    if recorded_hash:
        with pytest.raises(ValueError, match="stale ROI class map hash"):
            rr._admit_robustness_aggregation_input(path, cohort.rob)
    else:
        admitted = rr._admit_robustness_aggregation_input(path, cohort.rob)
        assert len(admitted.frame) == len(frame)


def test_a7_sequential_bound_item_invalidated_by_rob_config_and_packaged_yaml(
    tmp_path, monkeypatch
):
    from rtpipeline import extraction_progress as ep

    manifest = ep.code_manifest()
    assert "radiomics_params.yaml" in manifest
    assert "roi_class_map_v1.yaml" in manifest

    # Copy package files into a mutable scratch dir to verify YAML change invalidates BoundItem
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    for name, src in manifest.items():
        (pkg_dir / name).write_bytes(src.read_bytes())
    (pkg_dir / "radiomics_robustness.py").write_bytes((ROOT / "rtpipeline" / "radiomics_robustness.py").read_bytes())
    monkeypatch.setattr(ep, "__file__", str(pkg_dir / "extraction_progress.py"))

    arr = np.ones((8, 8, 8), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    mask = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.uint8))
    identity = rr.RobustnessRoiIdentity.from_mapping(
        dict(
            patient_id="P1",
            course_id="C1",
            series_uid="1.2.3",
            segmentation_source="Manual",
            mask_identity="m1",
            roi_original_name="ROI",
            stable_roi_identifier="roi-1",
        )
    )
    calls = []

    def fake_uncached(image, masks, config, modality="CT", structure_name="", patient_id="", course_id="", perturbed_images=None, segmentation_source="Manual", run_identifier="r1", source_identity=None, **_):
        calls.append(run_identifier)
        mask_id = rr._perturbed_mask_identity(next(iter(masks.values())))
        return pd.DataFrame(
            [
                {
                    **identity.as_dict(),
                    "structure": "ROI",
                    "modality": "CT",
                    "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                    "perturbed_mask_identity": mask_id,
                    "perturbation_id": "ntcv_v0",
                    "extraction_arm": arm,
                    "run_identifier": run_identifier,
                    "code_revision": "rev-a",
                    "feature_name": "original_firstorder_Mean",
                    "value": float(len(calls)),
                }
                for arm in CT_EXTRACTION_ARMS
            ]
        )

    monkeypatch.setattr(rr, "_extract_features_for_masks_uncached", fake_uncached)
    progress_root = tmp_path / "progress"
    cfg = PipelineConfig(tmp_path, tmp_path, tmp_path)
    rob_a = rr.effective_robustness_configuration(rr.RobustnessConfig(enabled=True), output_name="rob.parquet")
    rob_b = rr.effective_robustness_configuration(
        rr.RobustnessConfig(
            enabled=True,
            perturbation=rr.PerturbationConfig(course_specific_seeds=True),
        ),
        output_name="rob.parquet",
    )

    df1 = rr.extract_features_for_masks(
        img,
        {"ntcv_v0": mask},
        cfg,
        modality="CT",
        structure_name="ROI",
        patient_id="P1",
        course_id="C1",
        segmentation_source="Manual",
        run_identifier="run-1",
        source_identity=identity,
        progress_root=progress_root,
        progress_context={"configuration": rob_a},
    )
    assert len(calls) == 1
    assert df1["value"].iloc[0] == 1.0

    # Cache hit under unchanged context & YAML updates run_identifier
    df2 = rr.extract_features_for_masks(
        img,
        {"ntcv_v0": mask},
        cfg,
        modality="CT",
        structure_name="ROI",
        patient_id="P1",
        course_id="C1",
        segmentation_source="Manual",
        run_identifier="run-2",
        source_identity=identity,
        progress_root=progress_root,
        progress_context={"configuration": rob_a},
    )
    assert len(calls) == 1
    assert df2["run_identifier"].iloc[0] == "run-2"
    assert df2["code_revision"].iloc[0] != "rev-a"

    # Mutating robustness config misses cache
    df3 = rr.extract_features_for_masks(
        img,
        {"ntcv_v0": mask},
        cfg,
        modality="CT",
        structure_name="ROI",
        patient_id="P1",
        course_id="C1",
        segmentation_source="Manual",
        run_identifier="run-3",
        source_identity=identity,
        progress_root=progress_root,
        progress_context={"configuration": rob_b},
    )
    assert len(calls) == 2
    assert df3["value"].iloc[0] == 2.0

    # Mutating packaged radiomics_params.yaml misses cache even under rob_a
    (pkg_dir / "radiomics_params.yaml").write_text("# modified params\n", encoding="utf-8")
    df4 = rr.extract_features_for_masks(
        img,
        {"ntcv_v0": mask},
        cfg,
        modality="CT",
        structure_name="ROI",
        patient_id="P1",
        course_id="C1",
        segmentation_source="Manual",
        run_identifier="run-4",
        source_identity=identity,
        progress_root=progress_root,
        progress_context={"configuration": rob_a},
    )
    assert len(calls) == 3
    assert df4["value"].iloc[0] == 3.0


@pytest.mark.parametrize("setting, expected", [("271", 271), ("invalid", 180)])
def test_r3_materialized_arm_probe_uses_configured_timeout(monkeypatch, setting, expected):
    from dataclasses import asdict

    monkeypatch.setenv("RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT", setting)
    monkeypatch.setattr(rc, "_radiomics_env_command", lambda *args: list(args))
    seen = []

    def run(command, **kwargs):
        seen.append(kwargs["timeout"])
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({arm: f"hash-{arm}" for arm in CT_EXTRACTION_ARMS}),
        )

    monkeypatch.setattr(rc.subprocess, "run", run)
    rc._materialized_ct_arm_hashes.cache_clear()
    try:
        decision = json.dumps(asdict(classify_ct_roi("Manual", "PTV")))
        assert set(rc._materialized_ct_arm_hashes("", False, decision)) == set(CT_EXTRACTION_ARMS)
        assert seen == [expected]
    finally:
        rc._materialized_ct_arm_hashes.cache_clear()


@pytest.mark.parametrize("conditions, values", [
    (["ntcv_v0", "ntcv_n10_v0"], [1., 2., 3., 4.]),
    (["ntcv_v0", "ntcv_n10_v0", "ntcv_n20_v0"], [1., 2., np.nan, 3., 4., np.nan]),
    (["ntcv_v0", "ntcv_n10_v0", "ntcv_n20_v0"], [1., 2., pd.NA, 3., 4., pd.NA]),
])
def test_r3_icc_with_fewer_than_five_observations_is_not_evaluable(conditions, values):
    config = rr.RobustnessConfig(perturbation=rr.PerturbationConfig(
        max_translation_mm=0.0, n_random_contour_realizations=0,
        small_volume_changes=[0.0], noise_levels=[0., 10., 20.][:len(conditions)],
    ))
    frame = pd.DataFrame({
        "patient_id": np.repeat(["P1", "P2"], len(conditions)),
        "structure": "PTV",
        "perturbation_id": conditions * 2,
        "feature_name": "original_firstorder_Mean",
        "value": values,
    })
    row = rr.summarize_feature_stability(frame, config).iloc[0]
    assert row["n_subjects"] == 2
    assert row["n_perturbations"] == len(conditions)
    assert row["robustness_label"] == "not_evaluable"
    assert np.isnan(row["icc"])
    assert not row["pass_seg_perturb"]


@pytest.mark.parametrize("geometric", [False, True])
def test_r3_summary_refuses_a_condition_missing_from_every_subject(geometric):
    config = rr.RobustnessConfig(perturbation=rr.PerturbationConfig(
        max_translation_mm=0.0, n_random_contour_realizations=0,
        small_volume_changes=[0.0], noise_levels=[0., 10., 20.],
    ))
    frame = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
        "structure": "PTV",
        "perturbation_id": ["ntcv_v0", "ntcv_n10_v0"] * 3,
        "feature_name": "original_firstorder_Mean",
        "value": [1., 2., 3., 4.1, 5., 6.2],
    })
    if geometric:
        frame["robustness_status"] = "measured"
        frame.loc[frame["perturbation_id"].eq("ntcv_n10_v0"),
                  ["feature_name", "value", "robustness_status", "reason_code"]] = [
            None, np.nan, "geometrically_impossible", "contour_outside_image"
        ]
    with pytest.raises(ValueError, match="expected 3 NTCV conditions"):
        rr.summarize_feature_stability(frame, config)
