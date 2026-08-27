"""Preparing the bundled dcm2niix must be safe under parallel workers.

Every worker calls the preparation helper. It used to extract straight onto the
target path, so one worker rewrote the binary while another was executing it,
which POSIX refuses with ETXTBSY ("Text file busy"). CT to NIfTI conversion then
failed for that course.

Observed on a 154-patient cohort: 2,567 "Failed to prepare bundled dcm2niix:
[Errno 26] Text file busy" errors and 2,567 corresponding conversion failures,
out of 5,134 errors in a single organise log.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
import pytest

from rtpipeline import segmentation
from rtpipeline.config import PipelineConfig


@pytest.fixture()
def config(tmp_path):
    """A real config: the helper reads several fields beyond logs_root."""
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "Logs",
    )


def _plant_prepared_binary(logs_root: Path, name: str = "dcm2niix") -> Path:
    binary = logs_root / "bin" / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


@pytest.mark.skipif(sys.platform == "win32", reason="ETXTBSY is POSIX")
def test_an_already_prepared_binary_is_reused_without_rewriting(config):
    """The fast path must not touch a binary another worker may be executing."""
    planted = _plant_prepared_binary(config.logs_root)
    before = planted.stat().st_mtime_ns

    found = segmentation._ensure_local_dcm2niix(config)

    assert found is not None
    assert found.resolve() == planted.resolve()
    assert planted.stat().st_mtime_ns == before, "binary was rewritten in place"


@pytest.mark.skipif(sys.platform == "win32", reason="ETXTBSY is POSIX")
def test_preparation_does_not_fail_while_the_binary_is_executing(config):
    """The exact production failure: rewrite a running executable.

    Without the fast path this raises ETXTBSY, which the caller logged as
    'Failed to prepare bundled dcm2niix' and turned into a conversion failure.
    """
    binary = _plant_prepared_binary(config.logs_root)
    binary.write_text("#!/bin/sh\nsleep 2\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

    proc = subprocess.Popen([str(binary)])
    try:
        result = segmentation._ensure_local_dcm2niix(config)
        assert result is not None, "preparation failed while the binary was executing"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="ETXTBSY is POSIX")
def test_concurrent_preparation_all_succeed(config):
    """Many workers preparing at once must all get a usable path."""
    _plant_prepared_binary(config.logs_root)

    results: list[Path | None] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        try:
            r = segmentation._ensure_local_dcm2niix(config)
            with lock:
                results.append(r)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"preparation raised under concurrency: {errors[:3]}"
    assert len(results) == 12
    assert all(r is not None for r in results), "some workers got no binary"
    assert len({str(r) for r in results}) == 1, "workers disagreed on the binary path"


@pytest.mark.skipif(sys.platform == "win32", reason="ETXTBSY is POSIX")
def test_a_non_executable_candidate_is_not_treated_as_prepared(config):
    """A partially written file must not satisfy the fast path."""
    binary = config.logs_root / "bin" / "dcm2niix"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("partial")
    binary.chmod(0o644)

    result = segmentation._ensure_local_dcm2niix(config)

    # The fast path must not accept it. Preparation therefore re-extracts and
    # publishes a real, executable binary in its place.
    assert result is not None
    assert os.access(result, os.X_OK), "returned a non-executable binary"
    assert result.read_bytes() != b"partial", "partial file was accepted as prepared"


@pytest.mark.skipif(sys.platform == "win32", reason="ETXTBSY is POSIX")
def test_no_staging_directories_are_left_behind(config):
    _plant_prepared_binary(config.logs_root)
    segmentation._ensure_local_dcm2niix(config)
    leftovers = list((config.logs_root / "bin").glob(".stage-*"))
    assert leftovers == [], f"staging directories leaked: {leftovers}"
