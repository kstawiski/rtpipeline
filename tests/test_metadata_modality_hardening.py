"""Regression tests for metadata export across DICOM naming conventions.

All inputs are synthetic DICOM. No production patient data is embedded here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pydicom
import pytest
from pydicom.dataelem import RawDataElement
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RTBeamsTreatmentRecordStorage,
    RTDoseStorage,
    RTPlanStorage,
    RTStructureSetStorage,
    UID,
    generate_uid,
)

from rtpipeline import meta
from rtpipeline.config import PipelineConfig
from rtpipeline.utils import ORGANIZE_DISCOVERY_TAGS


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


def _is_detailed_metadata_read(kwargs: dict) -> bool:
    tags = kwargs.get("specific_tags")
    return tags is None or set(tags) != {meta._MODALITY_TAG}


def _write_export(root: Path, names: dict[str, str]) -> None:
    plan_uid = generate_uid()
    struct_uid = generate_uid()
    _write_plan(root / names["plan"], plan_uid, struct_uid)
    _write_dose(root / names["dose"], generate_uid(), plan_uid)
    _write_struct(root / names["struct"], struct_uid)
    _write_record(root / names["record"], plan_uid)
    _write_ct(root / names["ct"])


def test_modality_index_uses_file_threads_not_process_workers(tmp_path, monkeypatch):
    """Header-only modality indexing must use the bounded I/O thread mapper."""
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    _write_ct(cfg.dicom_root / "CT_2.dcm")
    observed = {}

    def fake_parallel(paths, fn, workers):
        observed["workers"] = workers
        return (fn(path) for path in paths)

    def forbidden_process_pool(*args, **kwargs):
        raise AssertionError("I/O-bound modality indexing must not start processes")

    monkeypatch.setattr(meta, "parallel_map_files", fake_parallel)
    monkeypatch.setattr(
        meta,
        "run_tasks_with_adaptive_workers",
        forbidden_process_pool,
        raising=False,
    )

    indexed = meta._index_dicom_files_by_modality(
        cfg.dicom_root,
        max_workers=8,
    )

    assert observed["workers"] == 8
    assert indexed["CT"] == sorted(
        [cfg.dicom_root / "CT_1.dcm", cfg.dicom_root / "CT_2.dcm"],
        key=str,
    )


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


def test_snapshot_read_keeps_verified_modality_when_detailed_header_fails(
    tmp_path, monkeypatch
):
    """The organize snapshot must retain fail-closed modality accounting."""
    cfg = _config(tmp_path)
    plan = _write_plan(cfg.dicom_root / "RTPLAN_1.dcm", generate_uid(), generate_uid())
    original = pydicom.dcmread

    def fail_detailed_header(path, *args, **kwargs):
        if Path(path) == plan and _is_detailed_metadata_read(kwargs):
            raise InvalidDicomError("synthetic detailed-header failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", fail_detailed_header)

    source_read = meta._metadata_source_file(plan)

    assert source_read.source.result.modality == "RTPLAN"
    assert source_read.source.result.row is None
    assert source_read.source.result.extraction_error
    assert source_read.dataset is None


def test_snapshot_read_decodes_only_discovery_tags_and_drops_contour_payload(tmp_path):
    path = tmp_path / "dicom" / "large_struct.dcm"
    ds = _file_dataset(path, RTStructureSetStorage, "RTSTRUCT", generate_uid())
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "PTV1"
    ds.StructureSetROISequence = Sequence([roi])
    contour = Dataset()
    contour.ContourData = [float(index) for index in range(3000)]
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = Sequence([contour])
    ds.ROIContourSequence = Sequence([roi_contour])
    ds.add_new((0x7777, 0x0010), "OB", b"x" * 100_000)
    _write(ds, path)

    full = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    source_read = meta._metadata_source_file(path)

    assert source_read.dataset is not None
    assert set(source_read.dataset.keys()).issubset(set(ORGANIZE_DISCOVERY_TAGS))
    assert "StructureSetROISequence" in source_read.dataset
    assert "ROIContourSequence" not in source_read.dataset
    assert (0x7777, 0x0010) not in source_read.dataset
    assert source_read.source.result == meta._metadata_result_from_dataset(path, full)
    assert not hasattr(source_read.source, "dataset")


def test_metadata_scan_does_not_decode_unrequested_nested_leaf(tmp_path):
    path = tmp_path / "dicom" / "record.dcm"
    ds = _file_dataset(
        path,
        RTBeamsTreatmentRecordStorage,
        "RTRECORD",
        generate_uid(),
    )
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    machine = Dataset()
    machine.ContourData = [float(index) for index in range(3000)]
    machine.TreatmentMachineName = "LINAC_A"
    ds.TreatmentMachineSequence = Sequence([machine])
    _write(ds, path)

    source_read = meta._metadata_source_file(path)

    assert source_read.source.result.row is not None
    assert source_read.source.result.row["machine"] == "LINAC_A"
    assert source_read.dataset is not None
    child = source_read.dataset.TreatmentMachineSequence[0]
    raw_contour = child.get_item(Tag(0x3006, 0x0050))
    assert isinstance(raw_contour, RawDataElement)


def test_projected_dose_preserves_first_referenced_uid_order(tmp_path):
    """Projection must retain an earlier referenced-image UID used by legacy lookup."""
    path = tmp_path / "dicom" / "dose.dcm"
    image_uid = generate_uid()
    plan_uid = generate_uid()
    ds = _file_dataset(path, RTDoseStorage, "RTDOSE", generate_uid())
    image = Dataset()
    image.ReferencedSOPInstanceUID = image_uid
    ds.ReferencedImageSequence = Sequence([image])
    plan = Dataset()
    plan.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([plan])
    _write(ds, path)

    full = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    source_read = meta._metadata_source_file(path)

    assert source_read.source.result == meta._metadata_result_from_dataset(path, full)
    assert source_read.source.result.row is not None
    assert source_read.source.result.row["plan_id"] == image_uid


def test_streamed_inventory_digest_matches_v1_canonical_payload(tmp_path):
    root = tmp_path / "dicom"
    paths = [_write_ct(root / "CT_1.dcm"), _write_ct(root / "CT_2.dcm")]
    sources = [meta._metadata_source_file(path).source for path in paths]

    identity = meta._source_inventory_identity_from_files(root, ["P1"], sources)
    records = [
        [
            os.path.relpath(source.path, root),
            source.size,
            source.mtime_ns,
            source.ctime_ns,
            source.device,
            source.inode,
        ]
        for source in sources
    ]
    expected = meta._canonical_sha256(
        {
            "schema": "rtpipeline-source-inventory-v1",
            "root": str(root.resolve(strict=False)),
            "scope_sha256": identity.scope_digest,
            "files": records,
        }
    )

    assert identity.digest == expected
    assert identity.file_count == 2


def test_detected_plans_that_yield_no_rows_fail_loudly(tmp_path, monkeypatch):
    """A detected RTPLAN must not disappear without plans.xlsx or an exception."""
    cfg = _config(tmp_path)
    plan = _write_plan(cfg.dicom_root / "RTPLAN_1.dcm", generate_uid(), generate_uid())
    original = pydicom.dcmread

    def fail_after_modality_detection(path, *args, **kwargs):
        if Path(path) == plan and _is_detailed_metadata_read(kwargs):
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
        if Path(path) == source and _is_detailed_metadata_read(kwargs):
            raise InvalidDicomError("synthetic detailed-header failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", fail_after_modality_detection)

    with pytest.raises(meta.MetadataExportError, match=modality):
        meta.export_metadata(cfg)


def _export_frames(exported: dict[str, Path]) -> dict[str, pd.DataFrame | None]:
    return {
        name: pd.read_excel(path, keep_default_na=False) if path.exists() else None
        for name, path in exported.items()
    }


def test_metadata_rows_are_identical_across_worker_counts(tmp_path):
    input_root = tmp_path / "dicom"
    _write_export(
        input_root,
        {
            "plan": "RTPLAN_1.dcm",
            "dose": "RTDOSE_1.dcm",
            "struct": "RTSTRUCT_1.dcm",
            "record": "RTRECORD_1.dcm",
            "ct": "CT_1.dcm",
        },
    )
    cfg1 = PipelineConfig(
        dicom_root=input_root,
        output_root=tmp_path / "out1",
        logs_root=tmp_path / "logs1",
        max_workers_override=1,
    )
    cfg8 = PipelineConfig(
        dicom_root=input_root,
        output_root=tmp_path / "out8",
        logs_root=tmp_path / "logs8",
        max_workers_override=8,
    )

    serial = _export_frames(meta.export_metadata(cfg1))
    parallel = _export_frames(meta.export_metadata(cfg8))

    assert serial.keys() == parallel.keys()
    for name in serial:
        if serial[name] is None:
            assert parallel[name] is None
        else:
            pd.testing.assert_frame_equal(serial[name], parallel[name])


def test_metadata_cache_hit_skips_dicom_reads(tmp_path, monkeypatch):
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
    first = meta.export_metadata(cfg)
    before = {
        name: path.stat().st_mtime_ns for name, path in first.items() if path.exists()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("cache hit must not read DICOM headers")

    monkeypatch.setattr(meta.pydicom, "dcmread", forbidden)
    second = meta.export_metadata(cfg)

    assert second == first
    assert before == {
        name: path.stat().st_mtime_ns for name, path in second.items() if path.exists()
    }


def test_metadata_cache_invalidates_when_input_inventory_changes(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    source = _write_ct(cfg.dicom_root / "CT_1.dcm")
    meta.export_metadata(cfg)
    original_stat = source.stat()
    original_bytes = source.read_bytes()
    needle = b"P1"
    replacement = b"P2"
    assert needle in original_bytes and len(needle) == len(replacement)
    source.write_bytes(original_bytes.replace(needle, replacement, 1))
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    reads = 0
    original_read = meta.pydicom.dcmread

    def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", counted)
    exported = meta.export_metadata(cfg)

    assert reads > 0
    frame = pd.read_excel(exported["ct_images"], keep_default_na=False)
    assert frame["PatientID"].tolist() == ["P2"]


def test_metadata_cache_rejects_modified_output(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    exported = meta.export_metadata(cfg)
    exported["ct_images"].write_bytes(b"not an xlsx")

    reads = 0
    original_read = meta.pydicom.dcmread

    def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", counted)
    repaired = meta.export_metadata(cfg)

    assert reads > 0
    assert len(pd.read_excel(repaired["ct_images"])) == 1


def test_partial_supported_modality_failure_does_not_publish(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    good = _write_ct(cfg.dicom_root / "CT_1.dcm")
    bad = _write_ct(cfg.dicom_root / "CT_2.dcm")
    original = meta.pydicom.dcmread

    def fail_detailed_read(path, *args, **kwargs):
        if Path(path) == bad and _is_detailed_metadata_read(kwargs):
            raise InvalidDicomError("synthetic partial detailed-header failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", fail_detailed_read)

    with pytest.raises(meta.MetadataExportError, match="incomplete tables"):
        meta.export_metadata(cfg)
    assert good.exists()
    assert not (cfg.output_root / "Data" / "CT_images.xlsx").exists()
    assert not (cfg.output_root / "_CACHE" / "metadata_export.json").exists()


def test_metadata_cache_fails_closed_when_inventory_cannot_be_statted(
    tmp_path, monkeypatch
):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")

    def fail_stat(path):
        raise OSError("synthetic stat failure")

    monkeypatch.setattr(meta, "_inventory_stat", fail_stat)

    with pytest.raises(meta.MetadataExportError, match="complete metadata source inventory"):
        meta.export_metadata(cfg)
