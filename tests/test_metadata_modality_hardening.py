"""Regression tests for metadata export across DICOM naming conventions.

All inputs are synthetic DICOM. No production patient data is embedded here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTBeamsTreatmentRecordStorage,
    RTDoseStorage,
    RTPlanStorage,
    RTStructureSetStorage,
    UID,
    generate_uid,
)

from rtpipeline import meta
from rtpipeline.config import PipelineConfig


def _file_dataset(path: Path, sop_class_uid: str, modality: str, sop_uid: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(sop_class_uid)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_uid
    ds.Modality = modality
    ds.PatientID = "P1"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    return ds


def _write(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def _write_plan(path: Path, plan_uid: str, struct_uid: str) -> Path:
    ds = _file_dataset(path, RTPlanStorage, "RTPLAN", plan_uid)
    ds.RTPlanLabel = "clinical"
    ds.RTPlanDate = "20240101"
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTStructureSetStorage
    ref.ReferencedSOPInstanceUID = struct_uid
    ds.ReferencedStructureSetSequence = Sequence([ref])
    return _write(ds, path)


def _write_dose(path: Path, dose_uid: str, plan_uid: str) -> Path:
    ds = _file_dataset(path, RTDoseStorage, "RTDOSE", dose_uid)
    ds.DoseSummationType = "PLAN"
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTPlanStorage
    ref.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref])
    return _write(ds, path)


def _write_struct(path: Path, struct_uid: str) -> Path:
    ds = _file_dataset(path, RTStructureSetStorage, "RTSTRUCT", struct_uid)
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "PTV1"
    ds.StructureSetROISequence = Sequence([roi])
    return _write(ds, path)


def _write_record(path: Path, plan_uid: str) -> Path:
    ds = _file_dataset(path, RTBeamsTreatmentRecordStorage, "RTRECORD", generate_uid())
    ds.TreatmentDate = "20240102"
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTPlanStorage
    ref.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref])
    return _write(ds, path)


def _write_ct(path: Path) -> Path:
    ds = _file_dataset(path, CTImageStorage, "CT", generate_uid())
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    return _write(ds, path)


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
    )


def _write_export(root: Path, names: dict[str, str]) -> None:
    plan_uid = generate_uid()
    struct_uid = generate_uid()
    _write_plan(root / names["plan"], plan_uid, struct_uid)
    _write_dose(root / names["dose"], generate_uid(), plan_uid)
    _write_struct(root / names["struct"], struct_uid)
    _write_record(root / names["record"], plan_uid)
    _write_ct(root / names["ct"])


def test_non_aria_filenames_export_each_dicom_modality(tmp_path):
    """Kopernik RTPLAN_1, RTSTRUCT_1, and RTDOSE_1 files must populate their tables."""
    cfg = _config(tmp_path)
    _write_export(
        cfg.dicom_root,
        {
            "plan": "RTPLAN_1.dcm",
            "dose": "RTDOSE_1.dcm",
            "struct": "RTSTRUCT_1.dcm",
            "record": "RTRECORD_1.dcm",
            "ct": "CT_1.dcm",
        },
    )

    exported = meta.export_metadata(cfg)

    assert len(pd.read_excel(exported["plans"])) == 1
    assert len(pd.read_excel(exported["structures"])) == 1
    assert len(pd.read_excel(exported["doses"])) == 1
    assert len(pd.read_excel(exported["fractions"])) == 1
    assert len(pd.read_excel(exported["ct_images"])) == 1
    assert len(pd.read_excel(exported["metadata"])) == 1


def test_extensionless_dicom_export_populates_rt_tables(tmp_path):
    """Extensionless RTPLAN, RTSTRUCT, and RTDOSE exports must not disappear silently."""
    cfg = _config(tmp_path)
    _write_export(
        cfg.dicom_root,
        {
            "plan": "RTPLAN_1",
            "dose": "RTDOSE_1",
            "struct": "RTSTRUCT_1",
            "record": "RTRECORD_1",
            "ct": "CT_1",
        },
    )

    exported = meta.export_metadata(cfg)

    assert len(pd.read_excel(exported["plans"])) == 1
    assert len(pd.read_excel(exported["structures"])) == 1
    assert len(pd.read_excel(exported["doses"])) == 1
    assert len(pd.read_excel(exported["metadata"])) == 1


def test_plan_and_dose_associate_through_dicom_reference_without_a_filename_key():
    """A dose reference must link unrelated filenames without an ARIA core key."""
    plans = pd.DataFrame(
        {
            "file_path": ["/x/no-shared-plan-name.dcm"],
            "_sop_instance_uid": ["1.2.3.4"],
            "plan_value": ["plan"],
        }
    )
    doses = pd.DataFrame(
        {
            "file_path": ["/y/unrelated-dose-name.dcm"],
            "_referenced_plan_sop_uids": [("1.2.3.4",)],
            "dose_value": ["dose"],
        }
    )

    merged = meta._merge_plans_doses(plans, doses)

    assert len(merged) == 1
    assert merged.iloc[0]["plan_value"] == "plan"
    assert merged.iloc[0]["dose_value"] == "dose"


def test_aria_filenames_retain_the_legacy_export_shape(tmp_path):
    """ARIA RP, RS, RD, RT, and CT names must retain their populated tables and core key."""
    cfg = _config(tmp_path)
    _write_export(
        cfg.dicom_root,
        {
            "plan": "RP.100.Prostate.dcm",
            "dose": "RD.100.Prostate.dcm",
            "struct": "RS.100.Prostate.dcm",
            "record": "RT.100.Prostate.dcm",
            "ct": "CT.100.Prostate.dcm",
        },
    )

    exported = meta.export_metadata(cfg)
    plans = pd.read_excel(exported["plans"], keep_default_na=False)
    doses = pd.read_excel(exported["doses"], keep_default_na=False)
    structs = pd.read_excel(exported["structures"], keep_default_na=False)
    merged = pd.read_excel(exported["metadata"], keep_default_na=False)

    assert list(plans.columns) == [
        "file_path",
        "plan_name",
        "plan_date",
        "reference_dose_name",
        "approval",
        "CT_series",
        "CT_study",
        "patient_id",
        "patient_dob",
        "patient_gender",
        "patient_pesel",
    ]
    assert list(doses.columns) == ["file_path", "CT_series", "CT_study", "plan_id", "patient_id"]
    assert list(structs.columns) == [
        "file_path",
        "CT_series",
        "CT_study",
        "approval",
        "patient_id",
        "available_structures",
    ]
    assert len(merged) == 1
    assert merged.iloc[0]["core_key"] == "100.Prostate"
    assert merged.iloc[0]["patient_id_plans"] == "P1"


def test_detected_plans_that_yield_no_rows_fail_loudly(tmp_path, monkeypatch):
    """A detected RTPLAN must not disappear without plans.xlsx or an exception."""
    cfg = _config(tmp_path)
    plan = _write_plan(cfg.dicom_root / "RTPLAN_1.dcm", generate_uid(), generate_uid())
    original = pydicom.dcmread

    def fail_after_modality_detection(path, *args, **kwargs):
        if Path(path) == plan and kwargs.get("specific_tags") is None:
            raise InvalidDicomError("synthetic detailed-header failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", fail_after_modality_detection)

    with pytest.raises(meta.MetadataExportError, match="RTPLAN"):
        meta.export_metadata(cfg)


@pytest.mark.parametrize(
    ("modality", "writer"),
    [
        ("RTDOSE", lambda path: _write_dose(path, generate_uid(), generate_uid())),
        ("RTSTRUCT", lambda path: _write_struct(path, generate_uid())),
        ("RTRECORD", lambda path: _write_record(path, generate_uid())),
        ("CT", _write_ct),
    ],
)
def test_each_detected_modality_that_yields_no_rows_fails_loudly(
    tmp_path, monkeypatch, modality, writer
):
    cfg = _config(tmp_path)
    source = writer(cfg.dicom_root / f"{modality}.dcm")
    original = pydicom.dcmread

    def fail_after_modality_detection(path, *args, **kwargs):
        if Path(path) == source and kwargs.get("specific_tags") is None:
            raise InvalidDicomError("synthetic detailed-header failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", fail_after_modality_detection)

    with pytest.raises(meta.MetadataExportError, match=modality):
        meta.export_metadata(cfg)
