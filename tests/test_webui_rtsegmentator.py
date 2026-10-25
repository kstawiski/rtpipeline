"""Web UI jobs can request RTsegmentator models; only the operator sets the portal URL."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))
import job_manager  # noqa: E402


def test_request_without_operator_portal_is_refused(monkeypatch):
    monkeypatch.delenv(job_manager.RTSEGMENTATOR_URL_ENV, raising=False)
    assert job_manager.rtsegmentator_job_config({}) is None
    assert job_manager.rtsegmentator_job_config({"rtsegmentator": {"enabled": False}}) is None
    with pytest.raises(ValueError, match="not configured"):
        job_manager.rtsegmentator_job_config({"rtsegmentator": {"enabled": True, "models": ["total"]}})


def test_browser_cannot_choose_the_portal_url(monkeypatch):
    monkeypatch.setenv(job_manager.RTSEGMENTATOR_URL_ENV, "http://rtsegmentator:8080")
    monkeypatch.setenv(job_manager.RTSEGMENTATOR_MODELS_ENV, "total, lung_nodules")
    config = job_manager.rtsegmentator_job_config(
        {"rtsegmentator": {"enabled": True, "url": "http://attacker.example", "models": ["total", "total"]}})
    assert config == {"enabled": True, "url": "http://rtsegmentator:8080", "models": ["total"],
                      "headers_env": "RTSEG_CLIENT_HEADERS"}
    defaults = job_manager.rtsegmentator_job_config({"rtsegmentator": {"enabled": True}})
    assert defaults["models"] == ["total", "lung_nodules"]


@pytest.mark.parametrize("models", [["../total"], "total", [1], []])
def test_invalid_model_selection_is_refused(monkeypatch, models):
    monkeypatch.setenv(job_manager.RTSEGMENTATOR_URL_ENV, "http://rtsegmentator:8080")
    with pytest.raises(ValueError):
        job_manager.rtsegmentator_job_config({"rtsegmentator": {"enabled": True, "models": models}})


def test_job_config_and_start_validation(tmp_path, monkeypatch):
    monkeypatch.setenv(job_manager.RTSEGMENTATOR_URL_ENV, "http://rtsegmentator:8080")
    manager = job_manager.JobManager(tmp_path / "in", tmp_path / "out", tmp_path)
    job = {"dicom_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out"),
           "config": {"rtsegmentator": {"enabled": True, "models": ["total"]}}}
    manager._create_job_config(job, tmp_path / "job.yaml")
    written = yaml.safe_load((tmp_path / "job.yaml").read_text())
    assert written["rtsegmentator"]["url"] == "http://rtsegmentator:8080"
    monkeypatch.delenv(job_manager.RTSEGMENTATOR_URL_ENV)
    result = manager.start_job("missing", {"rtsegmentator": {"enabled": True, "models": ["total"]}})
    assert result == {"success": False, "error": "RTsegmentator is not configured on this server"}
