"""Prefix-path resolution for the radiomics env (no HOME-based name lookup).

``conda run -n <name>`` resolves the name under the current HOME, so any
isolated invocation (tests with a scratch HOME, services, containers)
could never find the env. These tests pin the prefix resolver and the
command builder that replaced the four ``conda run -n`` call sites.
"""
import os
from pathlib import Path

import pytest

from rtpipeline.radiomics_conda import (
    RADIOMICS_ENV,
    _radiomics_env_command,
    _radiomics_env_prefix,
)

_OPT_ROOTS = ("/opt/conda/envs", "/opt/micromamba/envs")


def _skip_if_shared_opt_env():
    # NB2: the fallback test assumes no system-wide env exists; skip
    # rather than fail on hosts where one is legitimately installed.
    if any((Path(p) / RADIOMICS_ENV / "bin" / "python").is_file() for p in _OPT_ROOTS):
        pytest.skip("system-wide radiomics env present under /opt")


def _make_fake_env(root: Path, name: str = RADIOMICS_ENV) -> Path:
    prefix = root / name
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return prefix


def test_override_prefix_wins(tmp_path, monkeypatch):
    prefix = _make_fake_env(tmp_path / "custom")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_ENV_PREFIX", str(prefix))
    assert _radiomics_env_prefix() == str(prefix)
    command = _radiomics_env_command("-c", "print('OK')")
    assert command[0] == str(prefix / "bin" / "python")
    assert command[1:] == ["-c", "print('OK')"]


def test_broken_override_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "RTPIPELINE_RADIOMICS_ENV_PREFIX", str(tmp_path / "does-not-exist")
    )
    assert _radiomics_env_prefix() is None
    with pytest.raises(ValueError, match="RTPIPELINE_RADIOMICS_ENV_PREFIX"):
        _radiomics_env_command("-c", "print('OK')")


def test_conda_prefix_sibling_discovery(tmp_path, monkeypatch):
    current = tmp_path / "prefixes" / "snakemake"
    (current / "bin").mkdir(parents=True)
    (current / "bin" / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target = _make_fake_env(tmp_path / "prefixes")
    assert target != current
    monkeypatch.setenv("CONDA_PREFIX", str(current))
    monkeypatch.delenv("RTPIPELINE_RADIOMICS_ENV_PREFIX", raising=False)
    # Isolated HOME so no user-level root can contribute.
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    found = _radiomics_env_prefix()
    assert found == str(target)


def test_fallback_keeps_conda_run_shape(tmp_path, monkeypatch):
    _skip_if_shared_opt_env()
    monkeypatch.delenv("RTPIPELINE_RADIOMICS_ENV_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("MAMBA_ROOT_PREFIX", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "emptyhome"))
    assert _radiomics_env_prefix() is None
    command = _radiomics_env_command("-c", "print('OK')")
    assert command[1:4] == ["run", "-n", RADIOMICS_ENV]
    assert command[4:] == ["python", "-c", "print('OK')"]


def test_broken_override_makes_probe_return_false(tmp_path, monkeypatch):
    """NB3: the probe folds a broken override into closed False, not a raise."""
    from rtpipeline import radiomics_conda as rc

    monkeypatch.setenv(
        "RTPIPELINE_RADIOMICS_ENV_PREFIX", str(tmp_path / "does-not-exist")
    )
    monkeypatch.setattr(rc, "_ENV_CHECK_OK", False)
    assert rc.check_radiomics_env(timeout=5, retries=0) is False


def test_probe_uses_prefix_python_not_conda_run(tmp_path, monkeypatch):
    """The probe command must not route through `conda run -n` when a prefix resolves."""
    prefix = _make_fake_env(tmp_path / "custom")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_ENV_PREFIX", str(prefix))
    command = _radiomics_env_command(
        "-c", "import radiomics; import numpy; print('OK')"
    )
    assert "run" not in command
    assert "-n" not in command
    assert os.path.basename(command[0]) == "python"
