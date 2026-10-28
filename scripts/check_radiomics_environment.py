#!/usr/bin/env python3
"""Verify the supported headless substitution without editing rt-utils metadata."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rtpipeline import __version__
from rtpipeline.multiprocessing_utils import multiprocessing_subprocess_env
from rtpipeline.radiomics_rasterizer import (
    EXPECTED_RT_UTILS_VERSION,
    RadiomicsRasterizerError,
    require_radiomics_rasterizer,
)


def dependency_errors(result: subprocess.CompletedProcess[str]) -> list[str]:
    # rt-utils declares the GUI distribution even though it imports only cv2.
    # Headless is deliberately installed instead. The editable package is
    # installed --no-deps in this helper; these main-runtime-only dependencies
    # are intentionally absent. Core radiomics dependencies and all version
    # conflicts remain errors. No installed distribution metadata is changed.
    expected = {f"rt-utils {EXPECTED_RT_UTILS_VERSION} requires opencv-python, which is not installed."}
    main_only = {
        "dicom2nifti", "dicompyler-core", "ipython", "nbformat", "nested-lookup",
        "opencv-python", "plotly", "totalsegmentator", "xgboost",
    }
    expected.update(
        f"rtpipeline {__version__} requires {name}, which is not installed."
        for name in main_only
    )
    lines = [line for line in result.stdout.splitlines() if line]
    errors = [line for line in lines if line not in expected]
    if result.returncode == 0:
        return []
    if result.returncode == 1 and lines and not errors and not result.stderr.strip():
        return []
    return errors + [result.stderr.strip() or f"pip check exited {result.returncode}"]


def main() -> int:
    try:
        require_radiomics_rasterizer()
    except RadiomicsRasterizerError as exc:
        # The guard includes exact repair commands for overlapping provider files.
        print(str(exc), file=sys.stderr)
        return 1
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
