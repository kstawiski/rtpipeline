from __future__ import annotations

import subprocess
import sys


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "rtpipeline.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--dicom-root" in result.stdout


def test_cli_doctor_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "rtpipeline.cli", "doctor"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "rtpipeline doctor" in combined
    assert "Bundled dcm2niix zips" in combined
