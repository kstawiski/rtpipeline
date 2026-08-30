"""Feature rows must record the acquisition scale they were extracted under.

A cohort mixing standard CT (about [-1000, 3071] HU) with Siemens extended-scale
iMAR reconstructions (beyond 8000 HU) discretises into very different numbers of
grey levels under fixed bin-size binning, so features are not comparable between
them. Without a recorded descriptor that confounder is invisible downstream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from rtpipeline.acquisition_scale import classify_scale, describe_planning_ct


def _ct(path: Path, *, slope: float, intercept: float, hi: int, desc: str,
        background: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesDescription = desc
    ds.Manufacturer = "Siemens Healthineers"
    ds.ConvolutionKernel = "Qr40f"
    ds.KVP = 120.0
    ds.SliceThickness = 3.0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.Rows = ds.Columns = 8
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    arr = np.full((8, 8), background, dtype=np.uint16)
    arr[0, 0] = hi
    ds.PixelData = arr.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_scale_class_boundaries() -> None:
    assert classify_scale(-1000, 3065) == "standard"
    assert classify_scale(-8192, 14023) == "extended"
    assert classify_scale(-8192, 3797) == "intermediate"
    assert classify_scale(None, None) == "unknown"


def test_extended_scale_is_detected_from_effective_hu(tmp_path: Path) -> None:
    """Intercept alone must not decide it; the effective mapping must."""
    ct = tmp_path / "DICOM" / "CT"
    _ct(ct / "s.dcm", slope=1.0, intercept=-8192.0, hi=16000, desc="MIEDNICA Qr40 S3 iMAR")
    d = describe_planning_ct(ct)
    assert d["acq_scale_class"] == "extended"
    assert d["acq_effective_hu_max"] == pytest.approx(16000 - 8192)
    assert d["acq_imar_present"] is True
    assert d["acq_rescale_intercept"] == pytest.approx(-8192.0)


def test_standard_scale_despite_unusual_intercept(tmp_path: Path) -> None:
    """A large intercept with a compensating slope is still a standard scan.

    slope=4, intercept=-10240: air (stored 2310) -> -1000 HU, and stored 3310 ->
    +3000 HU. The intercept looks alarming; the effective mapping is ordinary.
    """
    ct = tmp_path / "DICOM" / "CT"
    _ct(ct / "s.dcm", slope=4.0, intercept=-10240.0, hi=3310, desc="PELVIS", background=2310)
    d = describe_planning_ct(ct)
    assert d["acq_effective_hu_min"] == pytest.approx(-1000.0)
    assert d["acq_effective_hu_max"] == pytest.approx(3000.0)
    assert d["acq_scale_class"] == "standard"


def test_missing_series_fails_soft(tmp_path: Path) -> None:
    d = describe_planning_ct(tmp_path / "nope")
    assert d["acq_scale_class"] == "unknown"
    assert d["acq_manufacturer"] is None
