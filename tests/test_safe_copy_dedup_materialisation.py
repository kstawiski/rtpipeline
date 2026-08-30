"""SOP deduplication must not strip the authoritative flat course artifacts.

`_copy_into` places every source plan/dose/struct under ``DICOM/<modality>/``
first, which registers its SOPInstanceUID. The later `_safe_copy` of the
selected object to the flat course artifact (``RP.dcm``) then asks the copy
manager for a UID it has already seen. Deduplication answers with the nested
path and writes nothing, so the course loses the artifact its contract names.

These tests run with deduplication ENABLED, because that is the production
default and disabling it is what hid this defect.
"""
from __future__ import annotations

from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTPlanStorage, generate_uid

from rtpipeline.dicom_copy import DicomCopyConfig, DicomCopyManager
from rtpipeline import organize
from rtpipeline.organize import _copy_into, _safe_copy


def _write_plan(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTPlanStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTPLAN"
    ds.PatientID = "P1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.save_as(path, enforce_file_format=True)
    return str(ds.SOPInstanceUID)


def _manager(tmp_path: Path) -> DicomCopyManager:
    return DicomCopyManager(DicomCopyConfig(dedup_by_sop_uid=True), tmp_path / "out")


def test_flat_artifact_is_materialised_after_nested_copy(tmp_path: Path) -> None:
    src = tmp_path / "src" / "plan.dcm"
    sop = _write_plan(src)
    course = tmp_path / "course"
    manager = _manager(tmp_path)

    nested = _copy_into(src, course / "DICOM" / "RTPLAN", copy_manager=manager)
    assert nested.exists()

    flat = course / "RP.dcm"
    _safe_copy(src, flat, copy_manager=manager)

    assert flat.exists(), "flat course artifact was suppressed by SOP deduplication"
    assert str(pydicom.dcmread(str(flat), stop_before_pixels=True).SOPInstanceUID) == sop


def test_dose_and_struct_flat_artifacts_survive_dedup(tmp_path: Path) -> None:
    course = tmp_path / "course"
    manager = _manager(tmp_path)
    for name, subdir in (("RD.dcm", "RTDOSE"), ("RS.dcm", "RTSTRUCT")):
        src = tmp_path / "src" / f"{name}.src.dcm"
        sop = _write_plan(src)
        _copy_into(src, course / "DICOM" / subdir, copy_manager=manager)
        flat = course / name
        _safe_copy(src, flat, copy_manager=manager)
        assert flat.exists(), f"{name} was suppressed by SOP deduplication"
        assert str(pydicom.dcmread(str(flat), stop_before_pixels=True).SOPInstanceUID) == sop


def test_unwritten_destination_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If every write path silently no-ops, _safe_copy must raise, not return."""
    src = tmp_path / "src" / "plan.dcm"
    _write_plan(src)
    manager = _manager(tmp_path)
    dst = tmp_path / "course" / "RP.dcm"
    monkeypatch.setattr(
        manager, "copy_dicom", lambda s, d, skip_if_exists=False: (d, False)
    )
    monkeypatch.setattr(organize.os, "link", lambda a, b: None)
    monkeypatch.setattr(organize.shutil, "copy2", lambda a, b: None)
    with pytest.raises(OSError, match="not materialised"):
        _safe_copy(src, dst, copy_manager=manager)
    assert not dst.exists()


def test_stale_destination_from_earlier_run_is_replaced(tmp_path: Path) -> None:
    """A different artifact left at dst by an earlier run must not survive."""
    course = tmp_path / "course"
    manager = _manager(tmp_path)

    stale = course / "RP.dcm"
    stale_sop = _write_plan(stale)

    src = tmp_path / "src" / "plan.dcm"
    fresh_sop = _write_plan(src)
    assert stale_sop != fresh_sop

    _copy_into(src, course / "DICOM" / "RTPLAN", copy_manager=manager)
    _safe_copy(src, stale, copy_manager=manager)

    on_disk = str(pydicom.dcmread(str(stale), stop_before_pixels=True).SOPInstanceUID)
    assert on_disk == fresh_sop, "stale artifact from an earlier run was not replaced"


def test_copy_into_does_not_cite_another_courses_copy(tmp_path: Path) -> None:
    """A per-patient object must be materialised inside each course that cites it.

    RTRECORDs are copied into every course of a patient. SOP dedup answers the
    second course with the first course's path, and a course contract that then
    names it fails validation with "escapes the course directory".
    """
    manager = _manager(tmp_path)
    src = tmp_path / "src" / "record.dcm"
    sop = _write_plan(src)

    course_a = tmp_path / "patient" / "2018-05" / "DICOM_related" / "RTRECORD"
    course_b = tmp_path / "patient" / "2021-08" / "DICOM_related" / "RTRECORD"

    first = _copy_into(src, course_a, copy_manager=manager)
    second = _copy_into(src, course_b, copy_manager=manager)

    assert first.parent.resolve() == course_a.resolve()
    assert second.parent.resolve() == course_b.resolve(), (
        "second course cited the first course's copy"
    )
    assert second.is_file()
    assert str(pydicom.dcmread(str(second), stop_before_pixels=True).SOPInstanceUID) == sop
