"""Convert a non-uniformly spaced unsigned 16-bit CT that dcm2niix refuses.

When slice spacing varies, dcm2niix writes the unequalized volume and an
equalized ``*_Eq_*`` volume resampled onto a uniform grid. Its equalization
only accepts signed 16-bit data, so an unsigned CT (PixelRepresentation 0)
with one irregular step exits 1 and the planning CT gets no NIfTI.

This fallback stages a temporary copy of the series in which only the
PixelRepresentation value is rewritten from 0 to 1, and only when every stored
pixel word fits the signed range for BitsStored. The stored values and the
rescale tags are untouched, so dcm2niix reads identical numbers and writes the
same rescale into the NIfTI header. The result is accepted only when the
unequalized volume from the staged copy has exactly the voxel values and
geometry of the unequalized volume dcm2niix wrote for the original series
before it failed. The published volume is the equalized ``*_Eq_*`` volume.
Anything else fails closed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

logger = logging.getLogger(__name__)

CONVERSION_METHOD = "dcm2niix_signed_staging_equalized"
CONVERSION_REASON = (
    "dcm2niix could not equalize non-uniform slice spacing of unsigned 16-bit CT data"
)

_CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
_NATIVE_LITTLE_ENDIAN = {str(ImplicitVRLittleEndian), str(ExplicitVRLittleEndian)}
_PIXEL_REPRESENTATION = 0x00280103
_PIXEL_DATA = 0x7FE00010
_SPACING_TOLERANCE_MM = 0.01
_ORIENTATION_TOLERANCE = 1e-4

Converter = Callable[[Path, Path], Optional[Path]]


class _Ineligible(Exception):
    """The series does not qualify for the signed staging fallback."""


def _is_nifti(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))


def _is_equalized(path: Path) -> bool:
    return "_Eq_" in path.name


def _inspect_series(ct_dir: Path) -> tuple[list[tuple[Path, int]], dict[str, Any]]:
    """Return (file, PixelRepresentation value offset) pairs and the spacing evidence."""
    entries = sorted(ct_dir.iterdir())
    if any(entry.is_dir() for entry in entries):
        raise _Ineligible("series directory contains subdirectories")
    files = [entry for entry in entries if entry.is_file()]
    if not files:
        raise _Ineligible("series directory has no files")

    series_uids: set[str] = set()
    orientation: Optional[np.ndarray] = None
    positions: list[float] = []
    staged: list[tuple[Path, int]] = []
    max_stored = 0
    bits_stored_values: set[int] = set()
    for path in files:
        try:
            ds = pydicom.dcmread(str(path))
        except Exception as exc:
            raise _Ineligible(f"unreadable file in series: {exc.__class__.__name__}") from exc
        # Read the raw elements before any attribute access converts them.
        raw_representation = ds.get_item(_PIXEL_REPRESENTATION)
        raw_pixels = ds.get_item(_PIXEL_DATA)
        transfer_syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
        if transfer_syntax not in _NATIVE_LITTLE_ENDIAN:
            raise _Ineligible("transfer syntax is not native little endian")
        if str(getattr(ds, "SOPClassUID", "")) != _CT_IMAGE_STORAGE:
            raise _Ineligible("not a single-frame CT Image Storage object")
        if str(getattr(ds, "Modality", "")).upper() != "CT":
            raise _Ineligible("modality is not CT")
        series_uids.add(str(getattr(ds, "SeriesInstanceUID", "")))
        if int(getattr(ds, "NumberOfFrames", 1) or 1) != 1:
            raise _Ineligible("multi-frame image")
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            raise _Ineligible("more than one sample per pixel")
        if int(getattr(ds, "BitsAllocated", 0)) != 16:
            raise _Ineligible("BitsAllocated is not 16")
        bits_stored = int(getattr(ds, "BitsStored", 0))
        if not 1 < bits_stored <= 16 or int(getattr(ds, "HighBit", -1)) != bits_stored - 1:
            raise _Ineligible("BitsStored/HighBit are not a low-aligned 16-bit layout")
        bits_stored_values.add(bits_stored)
        if (
            raw_representation is None
            or not hasattr(raw_representation, "value_tell")
            or raw_representation.length != 2
            or raw_representation.value != b"\x00\x00"
        ):
            raise _Ineligible("PixelRepresentation is not an unsigned (0) US element")
        if raw_pixels is None or not isinstance(raw_pixels.value, (bytes, bytearray)):
            raise _Ineligible("pixel data is missing")
        rows = int(getattr(ds, "Rows", 0))
        columns = int(getattr(ds, "Columns", 0))
        expected = rows * columns * 2
        if expected == 0 or len(raw_pixels.value) < expected:
            raise _Ineligible("pixel data is shorter than Rows x Columns")
        # Raw 16-bit words, including any bits above BitsStored, must be
        # non-negative in the signed range so the signed reading is identical.
        words = np.frombuffer(bytes(raw_pixels.value[:expected]), dtype="<u2")
        word_max = int(words.max())
        if word_max > (1 << (bits_stored - 1)) - 1:
            raise _Ineligible(
                f"stored value {word_max} does not fit signed {bits_stored}-bit data"
            )
        max_stored = max(max_stored, word_max)

        try:
            iop = np.asarray([float(v) for v in ds.ImageOrientationPatient], dtype=float)
            ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=float)
        except Exception as exc:
            raise _Ineligible("missing ImagePositionPatient/ImageOrientationPatient") from exc
        if iop.shape != (6,) or ipp.shape != (3,):
            raise _Ineligible("malformed ImagePositionPatient/ImageOrientationPatient")
        if orientation is None:
            orientation = iop
        elif np.max(np.abs(orientation - iop)) > _ORIENTATION_TOLERANCE:
            raise _Ineligible("slice orientation varies within the series")
        normal = np.cross(orientation[:3], orientation[3:])
        positions.append(float(np.dot(ipp, normal)))
        staged.append((path, int(raw_representation.value_tell)))

    if len(series_uids) != 1 or "" in series_uids:
        raise _Ineligible("files do not share one SeriesInstanceUID")
    ordered = np.sort(np.asarray(positions))
    steps = np.diff(ordered)
    if steps.size < 2 or np.min(steps) <= _SPACING_TOLERANCE_MM:
        raise _Ineligible("fewer than three distinct slice positions")
    if float(np.max(steps) - np.min(steps)) <= _SPACING_TOLERANCE_MM:
        raise _Ineligible("slice spacing is uniform")
    evidence = {
        "slice_count": len(files),
        "slice_step_min_mm": round(float(np.min(steps)), 4),
        "slice_step_max_mm": round(float(np.max(steps)), 4),
        "bits_stored": sorted(bits_stored_values),
        "max_stored_value": max_stored,
    }
    return staged, evidence


def _stage_signed_copy(files: list[tuple[Path, int]], staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=False)
    for path, offset in files:
        data = bytearray(path.read_bytes())
        if data[offset:offset + 2] != b"\x00\x00":
            raise _Ineligible("PixelRepresentation bytes changed while staging")
        data[offset:offset + 2] = b"\x01\x00"
        (staging_dir / path.name).write_bytes(bytes(data))


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise _Ineligible(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _same_volume(reference: Path, candidate: Path) -> bool:
    import nibabel as nib

    ref_img = nib.load(str(reference))
    cand_img = nib.load(str(candidate))
    if ref_img.shape != cand_img.shape:
        return False
    if not np.array_equal(ref_img.affine, cand_img.affine):
        return False
    return bool(
        np.array_equal(np.asanyarray(ref_img.dataobj), np.asanyarray(cand_img.dataobj))
    )


def convert_nonuniform_unsigned_ct(
    ct_dir: Path,
    work_dir: Path,
    converter: Converter,
) -> Optional[tuple[Path, dict[str, Any]]]:
    """Retry a failed dcm2niix conversion of ``ct_dir`` through a signed staged copy.

    ``work_dir`` is the failed conversion's temporary output directory; its
    unequalized volume is the reference for the equality check. ``converter``
    runs dcm2niix on (input directory, output directory). Returns the equalized
    volume inside ``work_dir`` and a provenance record, or ``None`` when the
    series does not qualify or any check fails. The caller removes ``work_dir``.
    """
    try:
        files, evidence = _inspect_series(ct_dir)
        original = _single(
            [p for p in sorted(work_dir.iterdir()) if _is_nifti(p) and not _is_equalized(p)],
            "unequalized volume from the failed conversion",
        )
        staging_dir = work_dir / "signed_staging_input"
        output_dir = work_dir / "signed_staging_output"
        _stage_signed_copy(files, staging_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        if converter(staging_dir, output_dir) is None:
            raise _Ineligible("dcm2niix failed on the signed staged copy")
        outputs = [p for p in sorted(output_dir.iterdir()) if _is_nifti(p)]
        equalized = _single([p for p in outputs if _is_equalized(p)], "equalized volume")
        unequalized = _single(
            [p for p in outputs if not _is_equalized(p)], "unequalized staged volume"
        )
        if not _same_volume(original, unequalized):
            raise _Ineligible(
                "staged unequalized volume differs from dcm2niix output for the original series"
            )
    except _Ineligible as exc:
        logger.warning("Signed staging fallback not applied to %s: %s", ct_dir, exc)
        return None
    except Exception as exc:
        logger.warning("Signed staging fallback failed for %s: %s", ct_dir, exc)
        return None

    provenance = {
        "method": CONVERSION_METHOD,
        "reason": CONVERSION_REASON,
        "staged_change": "PixelRepresentation 0 -> 1 in a temporary copy; stored pixel values unchanged",
        "verification": (
            "unequalized volume from the staged copy equals dcm2niix's unequalized volume "
            "of the original series (voxel values, shape, affine)"
        ),
        "published_volume": "dcm2niix equalized (_Eq_) volume",
        **evidence,
    }
    logger.warning(
        "Converted %s through the signed staging fallback (slice steps %.3f-%.3f mm)",
        ct_dir,
        evidence["slice_step_min_mm"],
        evidence["slice_step_max_mm"],
    )
    return equalized, provenance
