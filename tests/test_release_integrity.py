from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

import rtpipeline
import rtpipeline.organize as organize
import rtpipeline.radiomics_conda as radiomics_conda
import rtpipeline.radiomics_robustness as robustness
import rtpipeline.segmentation as segmentation
from rtpipeline.custom_models import _safe_extract_network_archive
from rtpipeline.radiomics_robustness import PerturbationConfig, RobustnessConfig


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_synchronized():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    version = project["project"]["version"]
    assert version == rtpipeline.__version__
    assert str(citation["version"]) == version
    assert str(citation["preferred-citation"]["version"]) == version
    assert f'LABEL version="{version}"' in dockerfile


def test_default_robustness_profile_is_complete_ntcv():
    direct = PerturbationConfig()
    parsed = RobustnessConfig.from_dict({"enabled": True}).perturbation
    configured = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))[
        "radiomics_robustness"
    ]["segmentation_perturbation"]

    for profile in (direct, parsed):
        assert profile.small_volume_changes == [-0.15, 0.0, 0.15]
        assert profile.max_translation_mm == 4.0
        assert profile.n_random_contour_realizations == 2
        assert profile.noise_levels == [0.0, 10.0, 20.0]
    assert configured["small_volume_changes"] == [-0.15, 0.0, 0.15]
    assert configured["max_translation_mm"] == 4.0
    assert configured["n_random_contour_realizations"] == 2
    assert configured["noise_levels"] == [0.0, 10.0, 20.0]


@pytest.mark.parametrize("intensity, expected", [("standard", 81), ("aggressive", 315)])
def test_ntcv_grid_has_documented_unique_count(intensity, expected, monkeypatch):
    image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.float32))
    mask = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8))
    monkeypatch.setattr(robustness, "add_noise_to_image", lambda value, _std, rng=None: value)
    monkeypatch.setattr(robustness, "translate_mask", lambda value, _translation: value)
    monkeypatch.setattr(robustness, "_validate_translated_mask", lambda *_args: None)

    contour_counter = iter(range(10_000))

    def unique_contour(value, _mm, rng=None):
        array = sitk.GetArrayFromImage(value).copy()
        array.ravel()[next(contour_counter) % array.size] = 0
        result = sitk.GetImageFromArray(array)
        result.CopyInformation(value)
        return result

    monkeypatch.setattr(robustness, "randomize_contour", unique_contour)
    monkeypatch.setattr(robustness, "volume_adapt_mask", lambda value, _tau: value)

    config = PerturbationConfig(intensity=intensity)
    masks, images = robustness.generate_ntcv_perturbations(mask, image, config, "ROI")
    assert len(masks) == expected
    assert set(masks) == set(images)
    assert robustness.expected_ntcv_perturbation_count(config) == expected


def test_standard_ntcv_is_an_unmocked_81_state_grid():
    image = sitk.GetImageFromArray(np.ones((19, 31, 31), dtype=np.float32))
    mask_array = np.zeros((19, 31, 31), dtype=np.uint8)
    mask_array[5:14, 8:23, 8:23] = 1
    mask = sitk.GetImageFromArray(mask_array)

    masks, images = robustness.generate_ntcv_perturbations(
        mask, image, PerturbationConfig(), "ROI"
    )

    assert len(masks) == 81
    assert set(masks) == set(images)


def test_ntcv_contour_and_noise_draws_are_factor_independent(monkeypatch):
    image = sitk.GetImageFromArray(np.ones((7, 9, 9), dtype=np.float32))
    mask = sitk.GetImageFromArray(np.ones((7, 9, 9), dtype=np.uint8))
    monkeypatch.setattr(robustness, "translate_mask", lambda value, _translation: value)
    monkeypatch.setattr(robustness, "_validate_translated_mask", lambda *_args: None)
    monkeypatch.setattr(robustness, "volume_adapt_mask", lambda value, _tau: value)

    contour_draws = []

    def tagged_contour(value, _mm, rng=None):
        tag = int(rng.integers(1, 2**31))
        contour_draws.append(tag)
        array = sitk.GetArrayFromImage(value).copy()
        array.ravel()[tag % array.size] = 0
        result = sitk.GetImageFromArray(array)
        result.CopyInformation(value)
        return result

    def tagged_noise(value, noise_std, rng=None):
        array = np.full(sitk.GetArrayFromImage(value).shape, noise_std, dtype=np.float32)
        result = sitk.GetImageFromArray(array)
        result.CopyInformation(value)
        return result

    monkeypatch.setattr(robustness, "randomize_contour", tagged_contour)
    monkeypatch.setattr(robustness, "add_noise_to_image", tagged_noise)

    masks, images = robustness.generate_ntcv_perturbations(
        mask, image, PerturbationConfig(), "ROI"
    )

    assert len(contour_draws) == 3 * 2
    for noise_suffix in ("", "_n10", "_n20"):
        key = f"ntcv{noise_suffix}_t0_0_4_c1_v0"
        assert np.array_equal(
            sitk.GetArrayFromImage(masks[key]),
            sitk.GetArrayFromImage(masks["ntcv_t0_0_4_c1_v0"]),
        )
    assert np.all(sitk.GetArrayFromImage(images["ntcv_n10_v0"]) == 10.0)
    assert np.all(sitk.GetArrayFromImage(images["ntcv_n20_t0_0_4_c1_v0"]) == 20.0)


def test_translation_realises_requested_physical_direction():
    array = np.zeros((17, 17, 17), dtype=np.uint8)
    array[6:11, 6:11, 6:11] = 1
    mask = sitk.GetImageFromArray(array)

    translated = robustness.translate_mask(mask, (0.0, 0.0, 4.0))
    robustness._validate_translated_mask(mask, translated, (0.0, 0.0, 4.0))

    before = np.argwhere(array > 0).mean(axis=0)
    after = np.argwhere(sitk.GetArrayFromImage(translated) > 0).mean(axis=0)
    assert after[0] - before[0] == pytest.approx(4.0)


def test_translation_fails_closed_on_boundary_clipping():
    array = np.zeros((17, 17, 17), dtype=np.uint8)
    array[0:5, 6:11, 6:11] = 1
    mask = sitk.GetImageFromArray(array)
    translated = robustness.translate_mask(mask, (0.0, 0.0, -4.0))

    with pytest.raises(RuntimeError, match="clipped"):
        robustness._validate_translated_mask(mask, translated, (0.0, 0.0, -4.0))


@pytest.mark.parametrize("tau", [-0.15, 0.15])
def test_volume_adaptation_hits_exact_rounded_target_on_anisotropic_grid(tau):
    array = np.zeros((9, 24, 24), dtype=np.uint8)
    array[2:7, 5:19, 5:19] = 1
    mask = sitk.GetImageFromArray(array)
    mask.SetSpacing((1.0, 1.0, 4.0))

    result = robustness.volume_adapt_mask(mask, tau)
    repeated = robustness.volume_adapt_mask(mask, tau)
    assert result is not None
    assert repeated is not None
    original_count = int(array.sum())
    expected_count = round(original_count * (1.0 + tau))
    assert int(sitk.GetArrayFromImage(result).sum()) == expected_count
    assert np.array_equal(
        sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(repeated)
    )
    assert result.GetSpacing() == mask.GetSpacing()


def test_contour_randomization_uses_physical_distance_and_is_reproducible():
    array = np.zeros((7, 24, 24), dtype=np.uint8)
    array[2:5, 6:18, 6:18] = 1
    mask = sitk.GetImageFromArray(array)
    mask.SetSpacing((1.0, 1.0, 5.0))

    first = robustness.randomize_contour(
        mask, 2.0, rng=np.random.Generator(np.random.PCG64(7))
    )
    second = robustness.randomize_contour(
        mask, 2.0, rng=np.random.Generator(np.random.PCG64(7))
    )
    first_array = sitk.GetArrayFromImage(first)
    assert np.array_equal(first_array, sitk.GetArrayFromImage(second))
    assert not np.array_equal(first_array, array)
    # A 2 mm offset must not jump through the 5 mm slice direction.
    assert np.flatnonzero(first_array.any(axis=(1, 2))).tolist() == [2, 3, 4]


def _stable_subject_frame() -> pd.DataFrame:
    rows = []
    for patient_id, value in (("P1", 1.0), ("P2", 100.0)):
        for perturbation_id in ("a", "b", "c", "d"):
            rows.append({
                "patient_id": patient_id,
                "course_id": "C1",
                "structure": "ROI",
                "segmentation_source": "manual",
                "perturbation_id": perturbation_id,
                "feature_name": "feature",
                "value": value,
            })
    return pd.DataFrame(rows)


def test_cov_is_computed_within_subject_not_across_subjects():
    summary = robustness.summarize_feature_stability(
        _stable_subject_frame(), RobustnessConfig(enabled=True)
    )
    assert len(summary) == 1
    assert summary.loc[0, "cov_pct"] == pytest.approx(0.0)
    assert summary.loc[0, "cov_pct_q1"] == pytest.approx(0.0)
    assert summary.loc[0, "cov_pct_q3"] == pytest.approx(0.0)
    assert summary.loc[0, "n_subjects_cov"] == 2
    assert summary.loc[0, "n_subjects_qcd"] == 2
    assert summary.loc[0, "cov_status"] == "complete"
    assert summary.loc[0, "qcd_status"] == "complete"
    assert summary.loc[0, "n_subjects_dropped"] == 0


@pytest.mark.parametrize("values", [[-4.0, -3.0, -2.0, -1.0], [-1.0, 1.0, 2.0, 3.0]])
def test_relative_dispersion_rejects_non_ratio_scale_values(values):
    array = np.asarray(values, dtype=float)
    assert np.isnan(robustness.compute_cov(array))
    assert np.isnan(robustness.compute_qcd(array))


def test_ineligible_relative_dispersion_is_not_misclassified_as_poor():
    frame = _stable_subject_frame()
    frame["value"] = -frame["value"]
    summary = robustness.summarize_feature_stability(
        frame, RobustnessConfig(enabled=True)
    )
    assert summary.loc[0, "n_subjects_cov"] == 0
    assert summary.loc[0, "n_subjects_qcd"] == 0
    assert summary.loc[0, "cov_status"] == "not_evaluable"
    assert summary.loc[0, "qcd_status"] == "not_evaluable"
    assert summary.loc[0, "robustness_label"] == "not_evaluable"
    assert not bool(summary.loc[0, "pass_seg_perturb"])


def test_incomplete_subject_grid_fails_closed():
    frame = _stable_subject_frame()
    frame = frame[
        ~((frame["patient_id"] == "P2") & (frame["perturbation_id"] == "d"))
    ]
    with pytest.raises(ValueError, match="incomplete perturbation grid"):
        robustness.summarize_feature_stability(frame, RobustnessConfig(enabled=True))


def test_aggregate_keeps_structure_source_rows_separate(tmp_path):
    first = _stable_subject_frame()
    second = first.copy()
    second["structure"] = "OTHER_ROI"
    second["value"] = second["value"] * 10
    input_path = tmp_path / "course.parquet"
    pd.concat([first, second], ignore_index=True).to_parquet(input_path, index=False)
    output_path = tmp_path / "summary.xlsx"

    robustness.aggregate_robustness_results(
        [input_path], output_path, RobustnessConfig(enabled=True)
    )

    summary = pd.read_excel(output_path, sheet_name="global_summary")
    assert set(summary["structure"]) == {"ROI", "OTHER_ROI"}
    assert len(summary) == 2
    assert (tmp_path / "summary_raw_values.parquet").exists()


def test_aggregate_rejects_missing_course_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        robustness.aggregate_robustness_results(
            [tmp_path / "missing.parquet"],
            tmp_path / "summary.xlsx",
            RobustnessConfig(enabled=True),
        )


def test_aggregate_rejects_feature_missing_from_one_subject(tmp_path):
    frame = _stable_subject_frame()
    extra = frame.copy()
    extra["feature_name"] = "second_feature"
    combined = pd.concat([frame, extra], ignore_index=True)
    combined = combined[
        ~(
            (combined["patient_id"] == "P2")
            & (combined["feature_name"] == "second_feature")
        )
    ]
    input_path = tmp_path / "course.parquet"
    combined.to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="inconsistent feature sets across subjects"):
        robustness.aggregate_robustness_results(
            [input_path],
            tmp_path / "summary.xlsx",
            RobustnessConfig(enabled=True),
        )


def test_incomplete_feature_extraction_fails_closed():
    frame = _stable_subject_frame().query("patient_id == 'P1'")
    with pytest.raises(RuntimeError, match="missing perturbations"):
        robustness._validate_extracted_feature_frame(
            frame,
            {"a", "b", "c", "d", "missing"},
            "manual/ROI",
        )


def test_supported_entrypoints_share_complete_standard_grid():
    web_manager = (ROOT / "webui" / "job_manager.py").read_text(encoding="utf-8")
    docker_setup = (ROOT / "setup_docker_project.sh").read_text(encoding="utf-8")
    web_template = (ROOT / "webui" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    for text in (web_manager, docker_setup):
        assert "max_translation_mm': 4.0" in text or "max_translation_mm: 4.0" in text
        assert "[0.0, 10.0, 20.0]" in text
    assert 'id="robustness-intensity"' in web_template
    assert "Standard (81 states)" in web_template


def test_robustness_workflow_failure_is_not_converted_to_success():
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    assert "Don't fail the pipeline for robustness" not in snakefile
    assert snakefile.count("Radiomics robustness failed; see") == 2
    assert snakefile.count("upstream radiomics failed") == 2
    assert "Non-success robustness sentinel encountered" in snakefile


def test_dual_environment_contract_is_enforced_in_packaging_and_receipt():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "radiomics" not in project["project"]["optional-dependencies"]
    assert "dcmseg" not in project["project"]["optional-dependencies"]
    assert "numpy>=1.23,<3" in project["project"]["dependencies"]
    main_env = yaml.safe_load(
        (ROOT / "envs" / "rtpipeline-local.yaml").read_text(encoding="utf-8")
    )
    assert "numpy>=2.0,<3" in main_env["dependencies"]
    installer = (ROOT / "scripts" / "install_local.sh").read_text(encoding="utf-8")
    assert installer.count("python -m pip freeze --all") == 2
    assert installer.count("python -m pip inspect --local") == 2
    assert "python -m pip uninstall \\" in installer
    assert "--yes rtpipeline" in installer
    assert '--no-deps --editable "$PROJECT_DIR"' not in installer
    assert installer.count('PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"') == 2
    assert "env update --yes" in installer
    assert "--prune" in installer


def test_public_docs_do_not_overclaim_or_mix_roi_sources():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    case_study = (ROOT / "docs" / "case_studies" / "index.md").read_text(
        encoding="utf-8"
    )
    assert "IBSI-compliant" not in readme
    assert "inspired by but not identical" in readme
    assert "radiomics_robustness_ct\n" not in case_study
    assert 'features["roi_original_name"] == "GTV_primary"' in case_study
    assert 'features["segmentation_source"] == "Custom"' in case_study


def test_related_series_metadata_helper_is_available_from_organize(tmp_path, monkeypatch):
    paths = [tmp_path / "one.dcm", tmp_path / "two.dcm"]
    for path in paths:
        path.touch()
    datasets = {
        "one.dcm": SimpleNamespace(
            StudyInstanceUID="1.2.3",
            SeriesInstanceUID="4.5.6",
            SOPInstanceUID="7.8.1",
            Modality="MR",
        ),
        "two.dcm": SimpleNamespace(
            StudyInstanceUID="1.2.3",
            SeriesInstanceUID="4.5.6",
            SOPInstanceUID="7.8.2",
            Modality="MR",
        ),
    }
    monkeypatch.setattr(
        segmentation.pydicom,
        "dcmread",
        lambda path, stop_before_pixels=True: datasets[Path(path).name],
    )

    metadata = organize._collect_series_metadata(tmp_path)
    assert metadata["study_instance_uid"] == "1.2.3"
    assert metadata["series_instance_uid"] == "4.5.6"
    assert metadata["modality"] == "MR"
    assert metadata["instance_count"] == 2
    expected = hashlib.sha256("7.8.17.8.2".encode()).hexdigest()
    assert metadata["sop_hash"] == expected


def test_model_archive_validation_is_fail_closed(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("valid/model.bin", b"weights")
        archive.writestr("../escape.bin", b"escape")
    payload.seek(0)

    destination = tmp_path / "weights"
    with zipfile.ZipFile(payload) as archive, pytest.raises(ValueError, match="Unsafe"):
        _safe_extract_network_archive(archive, destination)

    assert not (destination / "valid" / "model.bin").exists()
    assert not (tmp_path / "escape.bin").exists()


def test_model_archive_extracts_valid_members(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("Dataset001/model.bin", b"weights")
    payload.seek(0)

    with zipfile.ZipFile(payload) as archive:
        _safe_extract_network_archive(archive, tmp_path / "weights")
    assert (tmp_path / "weights" / "Dataset001" / "model.bin").read_bytes() == b"weights"


def test_model_archive_rejects_file_directory_collision_before_writing(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("Dataset001", b"not a directory")
        archive.writestr("Dataset001/model.bin", b"weights")
    payload.seek(0)

    destination = tmp_path / "weights"
    with zipfile.ZipFile(payload) as archive, pytest.raises(ValueError, match="collision"):
        _safe_extract_network_archive(archive, destination)
    assert not (destination / "Dataset001").exists()


def test_conda_resolver_accepts_micromamba(monkeypatch):
    monkeypatch.delenv("RTPIPELINE_CONDA_EXE", raising=False)
    monkeypatch.setattr(
        radiomics_conda.shutil,
        "which",
        lambda name: "/opt/micromamba" if name == "micromamba" else None,
    )
    assert radiomics_conda._conda_executable() == "/opt/micromamba"


@pytest.mark.parametrize(
    "command",
    [
        ["bash", "scripts/install_local.sh", "--plan", "--device", "cpu"],
        ["bash", "scripts/install_local.sh", "--help"],
        ["bash", "scripts/run_local.sh", "--help"],
    ],
)
def test_local_scripts_have_non_mutating_entrypoints(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_docker_runner_preserves_paths_with_spaces(tmp_path):
    input_dir = tmp_path / "input dicom"
    output_dir = tmp_path / "output results"
    log_dir = tmp_path / "pipeline logs"
    input_dir.mkdir()
    env = os.environ.copy()
    env["RTPIPELINE_LOG_DIR"] = str(log_dir)

    result = subprocess.run(
        [
            "bash",
            "run_rtpipeline.sh",
            "--dry-run",
            "--cpu-only",
            "--cores",
            "2",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert f"{input_dir!s}".replace(" ", "\\ ") + ":/data/input:ro" in result.stdout
    assert f"{output_dir!s}".replace(" ", "\\ ") + ":/data/output:rw" in result.stdout
    assert f"{log_dir!s}".replace(" ", "\\ ") + ":/data/logs:rw" in result.stdout
