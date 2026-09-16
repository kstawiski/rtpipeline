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


def _ct_slice(
    path: Path,
    series_uid: str,
    instance_uid: str,
    *,
    description: str = "",
    thickness: str = "",
    study_date: str = "",
) -> None:
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
    if description:
        ds.SeriesDescription = description
    if thickness:
        ds.SliceThickness = thickness
    if study_date:
        ds.StudyDate = study_date
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


def _stub_converter(monkeypatch):
    """Each series converts to a REAL, distinct NIfTI.

    Distinct voxel fills per series name, so an overwrite would show in
    bytes; a genuine image, so nifti_geometry/sha are truly computed
    rather than vacuously present.
    """

    def _fake_dcm2niix(config, series_subdir, tmp_out):
        import SimpleITK as sitk

        fill = sum(bytes(Path(series_subdir).name, encoding="utf-8")) % 200 + 1
        image = sitk.GetImageFromArray(
            np.full((2, 4, 4), fill, dtype=np.int16)
        )
        out = Path(tmp_out) / "stub.nii.gz"
        sitk.WriteImage(image, str(out))
        return out

    monkeypatch.setattr(organize, "run_dcm2niix", _fake_dcm2niix)


def _series(tmp_path: Path, name: str) -> tuple[Path, str]:
    series = tmp_path / name
    series.mkdir()
    series_uid = generate_uid()
    _ct_slice(series / "ct1.dcm", series_uid, generate_uid())
    _ct_slice(series / "ct2.dcm", series_uid, generate_uid())
    return series, series_uid


def test_related_conversion_never_overwrites_foreign_series(
    tmp_path, monkeypatch
):
    """A name collision with another series disambiguates with a suffix.

    Without this, the related converter unlinks the planning-CT NIfTI that
    owns the derived name and replaces it with foreign bytes, moving the
    course from an incompleteness quarantine to a stale-provenance one.
    """
    series_a, uid_a = _series(tmp_path, "seriesA")
    nifti_dir = tmp_path / "NIFTI"
    nifti_dir.mkdir()
    _stub_converter(monkeypatch)
    config = SimpleNamespace(resume=True)

    first = organize._convert_one_related_series(nifti_dir, series_a, config)
    assert first is not None and first.is_file()
    first_bytes = first.read_bytes()

    # A distinct series with the same description, thickness and study
    # date derives the identical NIfTI name: a genuine name collision.
    series_b = tmp_path / "seriesB"
    series_b.mkdir()
    uid_b = generate_uid()
    assert uid_b != uid_a
    _ct_slice(
        series_b / "ct1.dcm", uid_b, generate_uid(),
        description="MIEDNICA", thickness="3.0", study_date="20240403",
    )
    _ct_slice(
        series_b / "ct2.dcm", uid_b, generate_uid(),
        description="MIEDNICA", thickness="3.0", study_date="20240403",
    )
    for name in ("seriesA",):
        for i, p in enumerate(sorted((tmp_path / name).glob("*.dcm"))):
            ds = pydicom.dcmread(str(p))
            ds.SeriesDescription = "MIEDNICA"
            ds.SliceThickness = "3.0"
            ds.StudyDate = "20240403"
            ds.save_as(str(p))
    # Re-convert A under the colliding name so both sides share it.
    import shutil as _shutil

    _shutil.rmtree(nifti_dir)
    nifti_dir.mkdir()
    first = organize._convert_one_related_series(nifti_dir, series_a, config)
    assert first is not None and first.is_file()
    first_bytes = first.read_bytes()

    second = organize._convert_one_related_series(nifti_dir, series_b, config)
    assert second is not None and second.is_file()
    assert second != first
    # The planning-CT artifact is byte-identical; the newcomer is complete.
    assert first.read_bytes() == first_bytes
    payload = json.loads(
        (nifti_dir / f"{second.name.split('.nii')[0]}.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar_is_complete(payload)
    assert payload["series_instance_uid"] == uid_b


def test_related_conversion_honours_suffix_and_modality(
    tmp_path, monkeypatch
):
    """The MR-sibling parameters (name suffix, modality default) work."""
    series, uid = _series(tmp_path, "seriesA")
    nifti_dir = tmp_path / "NIFTI"
    nifti_dir.mkdir()
    _stub_converter(monkeypatch)
    config = SimpleNamespace(resume=True)
    target = organize._convert_one_related_series(
        nifti_dir, series, config, name_suffix="mrsuffix", default_modality="MR"
    )
    assert target is not None and target.is_file()
    assert "mrsuffix" in target.name
    payload = json.loads(
        (nifti_dir / f"{target.name.split('.nii')[0]}.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar_is_complete(payload)
    # The DICOM-declared modality wins over the default: these stub
    # slices declare CT, so the MR default must NOT overwrite it.
    assert payload["modality"] == "CT"
    assert payload["series_instance_uid"] == uid


def test_related_conversion_skip_returns_none_without_resume(
    tmp_path, monkeypatch
):
    import json as _json

    series, uid = _series(tmp_path, "seriesA")
    nifti_dir = tmp_path / "NIFTI"
    nifti_dir.mkdir()
    _stub_converter(monkeypatch)
    config = SimpleNamespace(resume=False)
    # A present sidecar for the SAME series with resume off skips
    # conversion entirely and reports no artifact (nothing was produced).
    name = organize._derive_nifti_name(series)
    (nifti_dir / f"{name}.metadata.json").write_text(
        _json.dumps({"series_instance_uid": uid}),
        encoding="utf-8",
    )
    calls: list = []
    monkeypatch.setattr(
        organize, "run_dcm2niix", lambda *a: calls.append(a) or None
    )
    assert organize._convert_one_related_series(nifti_dir, series, config) is None
    assert calls == []
