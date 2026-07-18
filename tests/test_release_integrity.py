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
    monkeypatch.setattr(robustness, "randomize_contour", lambda value, _mm, rng=None: value)
    monkeypatch.setattr(robustness, "volume_adapt_mask", lambda value, _tau: value)

    config = PerturbationConfig(intensity=intensity)
    masks, images = robustness.generate_ntcv_perturbations(mask, image, config, "ROI")
    assert len(masks) == expected
    assert set(masks) == set(images)


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
