"""Related-series NIfTI sidecars must be contract-complete (F-D root cause).

Two quarantined Kopernik courses failed contract validation with
"planning CT NIfTI provenance sidecar is incomplete": their sidecars
carried series keys (series_instance_uid, sop_hash, geometry) but none
of the NIfTI-derived keys (nifti_geometry, nifti_sha256). The related-
series converter in organize.py wrote sidecars without calling
nifti_provenance.annotate — the third conversion path the earlier
provenance repair missed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline import organize
from rtpipeline.nifti_provenance import REQUIRED_SIDECAR_KEYS, sidecar_is_complete


def _ct_slice(path: Path, series_uid: str, instance_uid: str) -> None:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = instance_uid
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 4, 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = (np.arange(16, dtype=np.uint16)).tobytes()
    ds.save_as(str(path))


def test_related_conversion_writes_contract_complete_sidecar(tmp_path, monkeypatch):
    series = tmp_path / "series1"
    series.mkdir()
    series_uid = generate_uid()
    _ct_slice(series / "ct1.dcm", series_uid, generate_uid())
    _ct_slice(series / "ct2.dcm", series_uid, generate_uid())
    nifti_dir = tmp_path / "NIFTI"
    nifti_dir.mkdir()

    def _fake_dcm2niix(config, series_subdir, tmp_out):
        import SimpleITK as sitk

        image = sitk.GetImageFromArray(np.zeros((2, 4, 4), dtype=np.int16))
        out = Path(tmp_out) / "stub.nii.gz"
        sitk.WriteImage(image, str(out))
        return out

    monkeypatch.setattr(organize, "run_dcm2niix", _fake_dcm2niix)
    config = SimpleNamespace(resume=True)

    target = organize._convert_one_related_series(nifti_dir, series, config)
    assert target is not None and target.is_file()
    sidecars = list(nifti_dir.glob("*.metadata.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_SIDECAR_KEYS if key not in payload]
    assert not missing, f"sidecar missing contract keys: {missing}"
    assert sidecar_is_complete(payload)
    assert payload["series_instance_uid"] == series_uid
