"""Fail-closed regression tests for the Snakefile startup helpers.

The helpers ``_ensure_writable_dir`` and ``_materialize_effective_config``
run while the Snakemake workflow starts. Before the fail-closed repair, an
unwritable configured destination was silently redirected into a
``ROOT_DIR/*_fallback`` directory and a failed merged-config persistence
fell back to a single source config file, so scientific output placement
and subprocess-stage configuration could change without any error.

These tests isolate the two functions from the Snakefile source (no
Snakemake context, no Snakefile body execution, no real or clinical
config) and inject failures through monkeypatching, because the suite may
run as root, where chmod-based probes would not fail. All fixtures are
synthetic and confined to pytest tmp_path directories.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SNAKEFILE = ROOT / "Snakefile"
PROBE_PREFIX = ".rtpipeline-write-probe-"


def _extract_top_level_function(source: str, name: str) -> str:
    """Return the source of a top-level function without parsing the Snakefile.

    The Snakefile contains Snakemake-only directives, so the whole file is
    not parseable as plain Python; a top-level ``def`` block ends at the
    next non-blank, non-indented line.
    """
    lines = source.splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"def {name}("):
            start = index
            break
    assert start is not None, f"{name} not found at top level in Snakefile"
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace():
            end = index
            break
    return "".join(lines[start:end])


def _load_helper(source: str, name: str, namespace: dict):
    code = compile(_extract_top_level_function(source, name), "Snakefile", "exec")
    exec(code, namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def snakefile_source() -> str:
    return SNAKEFILE.read_text(encoding="utf-8")


def _ensure_dir_helper(snakefile_source: str, root_dir: Path):
    return _load_helper(
        snakefile_source,
        "_ensure_writable_dir",
        {
            "Path": Path,
            "os": os,
            "sys": sys,
            "tempfile": tempfile,
            "ROOT_DIR": root_dir,
        },
    )


def _materialize_helper(snakefile_source: str, tmp_path: Path, config: dict):
    return _load_helper(
        snakefile_source,
        "_materialize_effective_config",
        {
            "Path": Path,
            "os": os,
            "sys": sys,
            "ROOT_DIR": tmp_path,
            "LOGS_DIR": tmp_path / "logs",
            "config": config,
        },
    )


# --- _ensure_writable_dir -------------------------------------------------


def test_ensure_writable_dir_success_returns_configured_candidate(
    tmp_path: Path, snakefile_source: str
) -> None:
    """A writable configured destination is returned unchanged; the unique
    probe is cleaned up and no fallback directory is created."""
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "out" / "Data_Snakemake"

    result = helper(candidate)

    assert result == candidate
    assert candidate.is_dir()
    assert [path.name for path in candidate.iterdir()] == []
    assert not (tmp_path / "Data_Snakemake_fallback").exists()


def test_ensure_writable_dir_preserves_preexisting_write_test_file(
    tmp_path: Path, snakefile_source: str
) -> None:
    """A pre-existing user file named .write_test must never be clobbered by
    the startup probe."""
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "out"
    candidate.mkdir()
    user_probe = candidate / ".write_test"
    user_probe.write_text("user content\n", encoding="utf-8")

    result = helper(candidate)

    assert result == candidate
    assert user_probe.read_text(encoding="utf-8") == "user content\n"
    assert [path.name for path in candidate.iterdir()] == [".write_test"]
    assert not (tmp_path / "Data_Snakemake_fallback").exists()


def test_ensure_writable_dir_failed_mkdir_raises_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str
) -> None:
    """A failed mkdir raises with the original cause; no fallback directory
    is ever chosen."""
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "bad" / "Data_Snakemake"

    def _raise_mkdir(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "mkdir", _raise_mkdir)

    with pytest.raises(RuntimeError) as excinfo:
        helper(candidate)

    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert "Permission denied" in str(excinfo.value.__cause__)
    assert "Data_Snakemake_fallback" not in str(candidate)
    assert str(candidate) in str(excinfo.value)
    assert not candidate.exists()
    assert not (tmp_path / "Data_Snakemake_fallback").exists()
    assert not (tmp_path / "bad").exists()


def test_ensure_writable_dir_failed_probe_raises_and_cleans_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str
) -> None:
    """A failed probe write raises with the original cause, the unique probe
    file is removed, and no fallback directory is created."""
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "out"

    def _raise_write(fd, data):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(os, "write", _raise_write)

    with pytest.raises(RuntimeError) as excinfo:
        helper(candidate)

    assert isinstance(excinfo.value.__cause__, OSError)
    assert excinfo.value.__cause__.errno == 30
    assert str(candidate) in str(excinfo.value)
    assert candidate.is_dir()
    assert [path.name for path in candidate.iterdir()] == []
    assert not (tmp_path / "Data_Snakemake_fallback").exists()


def test_ensure_writable_dir_failed_cleanup_preserves_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str
) -> None:
    """If the probe write succeeds but probe cleanup also fails, the primary
    error is preserved as the cause and no fallback directory is created."""
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "out"

    def _raise_unlink(self, *args, **kwargs):
        raise OSError(16, "Device or resource busy", str(self))

    monkeypatch.setattr(Path, "unlink", _raise_unlink)

    with pytest.raises(RuntimeError) as excinfo:
        helper(candidate)

    assert isinstance(excinfo.value.__cause__, OSError)
    assert excinfo.value.__cause__.errno == 16
    assert str(candidate) in str(excinfo.value)
    assert candidate.is_dir()
    leftovers = [path.name for path in candidate.iterdir()]
    assert len(leftovers) == 1
    assert leftovers[0].startswith(PROBE_PREFIX)
    assert not (tmp_path / "Data_Snakemake_fallback").exists()


@pytest.mark.parametrize("bytes_written", [0, 1])
def test_ensure_writable_dir_rejects_short_probe_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str,
    bytes_written: int,
) -> None:
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "out"
    monkeypatch.setattr(os, "write", lambda _fd, _data: bytes_written)

    with pytest.raises(RuntimeError) as excinfo:
        helper(candidate)

    assert isinstance(excinfo.value.__cause__, OSError)
    assert "incomplete" in str(excinfo.value.__cause__).lower()
    assert list(candidate.iterdir()) == []


def test_ensure_writable_dir_write_error_survives_close_and_unlink_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str,
) -> None:
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    candidate = tmp_path / "out"
    write_error = OSError(28, "synthetic write failure")
    original_close = os.close

    def fail_write(_fd, _data):
        raise write_error

    def fail_close(fd):
        original_close(fd)
        raise OSError(5, "synthetic close failure")

    def fail_unlink(_path, *args, **kwargs):
        raise OSError(16, "synthetic unlink failure")

    monkeypatch.setattr(os, "write", fail_write)
    monkeypatch.setattr(os, "close", fail_close)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(RuntimeError) as excinfo:
        helper(candidate)

    assert excinfo.value.__cause__ is write_error
    leftovers = list(candidate.iterdir())
    assert len(leftovers) == 1
    assert leftovers[0].name.startswith(PROBE_PREFIX)


def test_ensure_writable_dir_real_file_parent_fails_without_fallback(
    tmp_path: Path, snakefile_source: str,
) -> None:
    helper = _ensure_dir_helper(snakefile_source, tmp_path)
    parent_file = tmp_path / "file-not-directory"
    parent_file.write_text("preserve this content", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        helper(parent_file / "output")

    assert isinstance(excinfo.value.__cause__, OSError)
    assert parent_file.read_text(encoding="utf-8") == "preserve this content"
    assert not (tmp_path / "Data_Snakemake_fallback").exists()


# --- _materialize_effective_config ----------------------------------------


def _merged_config() -> dict:
    return {
        "dicom_root": "Example_data",
        "output_dir": "Data_Snakemake",
        "logs_dir": "Logs_Snakemake",
        "radiomics": {"enabled": True},
        "environments": {"main": "rtpipeline", "radiomics": "rtpipeline-radiomics"},
    }


def test_materialize_effective_config_writes_merged_yaml(
    tmp_path: Path, snakefile_source: str
) -> None:
    """Successful persistence keeps the existing semantics: the merged
    config is written to LOGS_DIR/_workflow/effective_config.yaml and the
    resolved target is returned."""
    helper = _materialize_helper(snakefile_source, tmp_path, _merged_config())

    target = helper()

    assert target.is_file()
    assert Path(target) == (
        tmp_path / "logs" / "_workflow" / "effective_config.yaml"
    ).resolve()
    loaded = yaml.safe_load(Path(target).read_text(encoding="utf-8"))
    assert loaded == _merged_config()


def test_materialize_effective_config_missing_yaml_raises_without_source_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str
) -> None:
    """A missing yaml import raises with the original cause instead of
    returning a single source config file."""
    helper = _materialize_helper(snakefile_source, tmp_path, _merged_config())
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(RuntimeError) as excinfo:
        helper()

    assert isinstance(excinfo.value.__cause__, ImportError)
    assert list(tmp_path.rglob("effective_config.yaml")) == []
    assert not (tmp_path / "config.yaml").exists()


def test_materialize_effective_config_write_failure_raises_without_source_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snakefile_source: str
) -> None:
    """A failed write of the merged config raises with the original cause
    instead of silently falling back to a source config file."""
    helper = _materialize_helper(snakefile_source, tmp_path, _merged_config())

    def _raise_open(self, *args, **kwargs):
        raise OSError(28, "No space left on device", str(self))

    monkeypatch.setattr(Path, "open", _raise_open)

    with pytest.raises(RuntimeError) as excinfo:
        helper()

    assert isinstance(excinfo.value.__cause__, OSError)
    assert excinfo.value.__cause__.errno == 28
    assert list(tmp_path.rglob("effective_config.yaml")) == []
    assert not (tmp_path / "config.yaml").exists()


# --- Snakefile-level guards ------------------------------------------------


def test_snakefile_startup_helpers_are_fail_closed(snakefile_source: str) -> None:
    """The fail-open fallback machinery must be gone from the Snakefile,
    while both startup destinations still route through the helper."""
    materialize_source = _extract_top_level_function(
        snakefile_source, "_materialize_effective_config"
    )
    assert "_workflow_configfiles" not in materialize_source
    assert "return fallback" not in materialize_source
    assert "Using fallback" not in snakefile_source
    assert "Falling back to" not in snakefile_source
    assert 'probe = candidate / ".write_test"' not in snakefile_source
    assert 'OUTPUT_DIR = _ensure_writable_dir(' in snakefile_source
    assert 'LOGS_DIR = _ensure_writable_dir(' in snakefile_source
    assert "EFFECTIVE_CONFIGFILE = _materialize_effective_config()" in snakefile_source
