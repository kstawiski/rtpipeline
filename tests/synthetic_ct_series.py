"""Write small synthetic axial CT series for dcm2niix conversion tests.

Every identifier is generated; no clinical data is involved. The default
geometry mirrors the failure pattern seen at organize: 5 mm steps with one
step split into 2 mm + 3 mm, 16-bit stored pixels, rescale intercept -1000,
implicit VR little endian and no SliceThickness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


def mixed_z_positions(count: int = 12, step: float = 5.0, split_after: int = 5) -> list[float]:
    """Return ``count`` slice positions with one ``step`` split as 2 mm + 3 mm."""
    positions = [0.0]
    while len(positions) < count:
        if len(positions) == split_after + 1:
            positions.append(positions[-1] + 2.0)
            positions.append(positions[-1] + 3.0)
        else:
            positions.append(positions[-1] + step)
    return positions[:count]


def uniform_z_positions(count: int = 12, step: float = 5.0) -> list[float]:
    return [index * step for index in range(count)]


def synthetic_stored_values(rows: int, columns: int, slice_index: int, *, maximum: int = 3000) -> np.ndarray:
    """Deterministic, slice-dependent stored values in ``[0, maximum]``."""
    yy, xx = np.mgrid[0:rows, 0:columns]
    values = (xx * 37 + yy * 11 + slice_index * 101) % (maximum + 1)
    return values.astype(np.int64)


def write_ct_series(
    series_dir: Path,
    z_positions: Sequence[float],
    *,
    signed: bool,
    rows: int = 16,
    columns: int = 20,
    maximum: int = 3000,
    intercept: float = -1000.0,
    slope: float = 1.0,
    overrides: dict[int, np.ndarray] | None = None,
) -> list[Path]:
    """Write one single-series CT with the given slice positions.

    ``overrides`` maps a slice index to a stored-value array that replaces the
    default pattern for that slice.
    """
    series_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    dtype = np.int16 if signed else np.uint16
    paths: list[Path] = []
    for index, z in enumerate(z_positions):
        stored = synthetic_stored_values(rows, columns, index, maximum=maximum)
        if overrides and index in overrides:
            stored = np.asarray(overrides[index], dtype=np.int64)
        sop_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ImplicitVRLittleEndian
        path = series_dir / f"CT_{index + 1:04d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CT_IMAGE_STORAGE
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "CT"
        ds.PatientID = "SYNTHETIC"
        ds.PatientName = "Synthetic^Phantom"
        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.SeriesDescription = "SYNTH_CT"
        ds.SeriesNumber = 1
        ds.InstanceNumber = index + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImagePositionPatient = [-50.0, -40.0, float(z)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.Rows = rows
        ds.Columns = columns
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if signed else 0
        ds.RescaleIntercept = intercept
        ds.RescaleSlope = slope
        ds.PixelData = stored.astype(dtype).tobytes()
        ds.save_as(str(path), enforce_file_format=True)
        paths.append(path)
    return paths
