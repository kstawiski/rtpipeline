#!/usr/bin/env python3
"""Verify the supported headless substitution without editing rt-utils metadata."""
from __future__ import annotations

import subprocess
import sys

from rtpipeline.multiprocessing_utils import multiprocessing_subprocess_env
from rtpipeline.radiomics_rasterizer import (
    EXPECTED_RT_UTILS_VERSION,
    require_radiomics_rasterizer,
)


def dependency_errors(result: subprocess.CompletedProcess[str]) -> list[str]:
    # rt-utils declares the GUI distribution even though it imports only cv2.
    # Headless is deliberately installed instead. No other mismatch is waived.
    expected = f"rt-utils {EXPECTED_RT_UTILS_VERSION} requires opencv-python, which is not installed."
    lines = [line for line in result.stdout.splitlines() if line]
    errors = [line for line in lines if line != expected]
    if result.returncode == 0:
        return []
    if result.returncode == 1 and lines and not errors and not result.stderr.strip():
        return []
    return errors + [result.stderr.strip() or f"pip check exited {result.returncode}"]


def main() -> int:
    info = require_radiomics_rasterizer()
    if info["rt_utils_version"] != EXPECTED_RT_UTILS_VERSION:
        raise RuntimeError(f"rt-utils=={EXPECTED_RT_UTILS_VERSION} is required")
    checked = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True,
        env=multiprocessing_subprocess_env(), check=False,
    )
    errors = dependency_errors(checked)
    if errors:
        raise RuntimeError("\n".join(errors))
    print("Radiomics dependencies and the OpenCV 4.8.1 rasterizer verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
