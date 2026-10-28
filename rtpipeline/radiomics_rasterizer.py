"""The OpenCV convention used to turn RTSTRUCT polygons into radiomics masks."""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np

EXPECTED_OPENCV_VERSION = "4.8.1"
EXPECTED_OPENCV_DIST_VERSION = "4.8.1.78"
EXPECTED_RT_UTILS_VERSION = "1.2.7"
RASTERIZER_BACKEND = "rt-utils/cv2.fillPoly"
OPENCV_PROVIDERS = (
    "opencv-python-headless", "opencv-python",
    "opencv-contrib-python", "opencv-contrib-python-headless",
)
REFERENCE_TRIANGLE_VERTICES = ((8, 9), (8, 13), (14, 5))
EXPECTED_TRIANGLE_PIXELS = 23
EXPECTED_TRIANGLE_HEX = "000000000000000000000002000c001c007800f000e000e000c0008000000000"


class RadiomicsRasterizerError(RuntimeError):
    """The active rasterizer cannot supply the required measurement convention."""


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def reference_triangle_probe(cv2_module: Any) -> dict[str, Any]:
    """Check both count and exact locations on the 16 by 16 reference grid."""
    canvas = np.zeros((16, 16), dtype=np.uint8)
    cv2_module.fillPoly(canvas, np.asarray([REFERENCE_TRIANGLE_VERTICES], dtype=np.int32), 1)
    return {
        "triangle_pixels": int(np.count_nonzero(canvas)),
        "triangle_hex": np.packbits(canvas > 0, bitorder="big").tobytes().hex(),
    }


def active_radiomics_rasterizer(cv2_module: Any = None) -> dict[str, Any]:
    """Describe the imported binary, rather than infer it from wheel metadata."""
    info: dict[str, Any] = {
        "backend": RASTERIZER_BACKEND,
        "opencv_version": None,
        "opencv_path": None,
        "rt_utils_version": _dist_version("rt-utils"),
        "providers": {name: version for name in OPENCV_PROVIDERS
                      if (version := _dist_version(name)) is not None},
        "triangle_pixels": None,
        "triangle_hex": None,
        "error": None,
    }
    try:
        if cv2_module is None:
            import cv2 as cv2_module
        info["opencv_version"] = str(cv2_module.__version__)
        info["opencv_path"] = str(Path(cv2_module.__file__).resolve())
        info.update(reference_triangle_probe(cv2_module))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    info["pinned_convention"] = (
        info["error"] is None
        and info["opencv_version"] == EXPECTED_OPENCV_VERSION
        and info["triangle_pixels"] == EXPECTED_TRIANGLE_PIXELS
        and info["triangle_hex"] == EXPECTED_TRIANGLE_HEX
    )
    return info


def require_radiomics_rasterizer(*, cv2_module: Any = None) -> dict[str, Any]:
    """Fail before RTSTRUCT mask preparation on a different version or fill."""
    info = active_radiomics_rasterizer(cv2_module)
    providers_ok = info["providers"] == {
        "opencv-python-headless": EXPECTED_OPENCV_DIST_VERSION,
    }
    if (info["pinned_convention"] and providers_ok
            and info["rt_utils_version"] == EXPECTED_RT_UTILS_VERSION):
        return info
    raise RadiomicsRasterizerError(
        "RTSTRUCT rasterization requires a single opencv-python-headless==4.8.1.78 "
        "provider (cv2 4.8.1), rt-utils==1.2.7, and the exact 23-pixel reference triangle. "
        f"Imported cv2 {info['opencv_version']!r} at {info['opencv_path']!r}; "
        f"reference triangle filled {info['triangle_pixels']!r} pixels "
        f"(expected 23), mask={info['triangle_hex']!r}; "
        f"providers={info['providers']!r}; rt-utils={info['rt_utils_version']!r}; "
        f"probe error={info['error']!r}. "
        "In the isolated radiomics interpreter run:\n"
        "python -m pip uninstall -y opencv-python opencv-python-headless "
        "opencv-contrib-python opencv-contrib-python-headless\n"
        "python -m pip install --no-deps opencv-python-headless==4.8.1.78\n"
        "python -m pip install --no-deps rt-utils==1.2.7\n"
        "Launch Python with PYTHONNOUSERSITE=1."
    )


def mask_backend(record: Mapping[str, Any]) -> str:
    """Identify loaders that use stored voxels rather than polygon filling."""
    source = str(record.get("segmentation_source", "")).lower()
    path = str(record.get("mask_path_source", "")).lower()
    backend = str(record.get("rasterizer_backend", ""))
    if backend in {"nifti/no-polygon-fill", "dicom-seg/no-polygon-fill"}:
        return backend
    if source == "autots_total_nifti_fallback" or path.endswith((".nii", ".nii.gz")):
        return "nifti/no-polygon-fill"
    if source in {"dicom_seg", "dicom-seg"}:
        return "dicom-seg/no-polygon-fill"
    # Historical MR auto rows explicitly name the stored voxel source.
    if "totalsegmentator_total_mr_nifti" in path:
        return "nifti/no-polygon-fill"
    if "totalsegmentator_total_mr" in path and path.endswith(".dcm"):
        return "dicom-seg/no-polygon-fill"
    return RASTERIZER_BACKEND


def recorded_rasterizer_is_current(record: Mapping[str, Any]) -> bool:
    """Missing polygon provenance requires recomputation, never inference."""
    if mask_backend(record) != RASTERIZER_BACKEND:
        return True
    try:
        providers = json.loads(str(record.get("rasterizer_providers", "")))
    except (TypeError, ValueError):
        return False
    return (
        str(record.get("rasterizer_opencv_version")) == EXPECTED_OPENCV_VERSION
        and str(record.get("rasterizer_rt_utils_version")) == EXPECTED_RT_UTILS_VERSION
        and providers == {"opencv-python-headless": EXPECTED_OPENCV_DIST_VERSION}
    )


def require_recorded_rasterizer(record: Mapping[str, Any]) -> None:
    if not recorded_rasterizer_is_current(record):
        raise ValueError("recorded RTSTRUCT rasterizer identity is missing or stale")


def opencv_runtime_provenance() -> dict[str, Any]:
    """Record the imported main-runtime binary without imposing a radiomics pin."""
    info = active_radiomics_rasterizer()
    return {key: info[key] for key in ("opencv_version", "opencv_path", "providers", "error")}
