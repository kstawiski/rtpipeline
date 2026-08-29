"""Regression tests for content-based segmentation resume decisions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import rtpipeline.auto_rtstruct as auto_rtstruct
import rtpipeline.custom_structures_rtstruct as custom_structures_rtstruct
import rtpipeline.segmentation as segmentation
from rtpipeline.config import PipelineConfig
from course_contract_test_utils import write_minimal_course_contract, write_synthetic_planning_ct


RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"


def _write_rtstruct(path: Path, series_uid: str) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = RTSTRUCT_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = RTSTRUCT_SOP_CLASS
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTSTRUCT"
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "liver"
    ds.StructureSetROISequence = Sequence([roi])
    series = Dataset()
    series.SeriesInstanceUID = series_uid
    study = Dataset()
    study.RTReferencedSeriesSequence = Sequence([series])
    ref_for = Dataset()
    ref_for.RTReferencedStudySequence = Sequence([study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def _make_course(tmp_path: Path) -> tuple[Path, object, Path, Path, str, str]:
    course = tmp_path / "patient" / "course"
    ct_dir = write_synthetic_planning_ct(course)
    metadata_path = write_minimal_course_contract(course, planning_ct_dir=ct_dir)
    contract = segmentation.load_course_contract(course)
    nifti = contract.planning_ct_nifti
    assert nifti is not None
    series_uid = str(contract.planning_ct.get("series_instance_uid"))
    sop_hash = str(contract.planning_ct["nifti_provenance"]["sop_hash"])
    seg_dir = course / "Segmentation_TotalSegmentator" / segmentation._strip_nifti_base(nifti)
    seg_dir.mkdir(parents=True, exist_ok=True)
    reference = sitk.ReadImage(str(nifti))
    for name in ("liver", "spleen"):
        image = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
        image.CopyInformation(reference)
        image[0, 0, 0] = 1
        sitk.WriteImage(image, str(seg_dir / f"total--{name}.nii.gz"))
    source = segmentation._segmentation_source_provenance(nifti, series_uid, sop_hash)
    masks = sorted(path.name for path in seg_dir.glob("total--*.nii.gz"))
    manifest = {
        **source,
        "generated_at": "2026-08-29T18:00:00+00:00",
        "models": [{"model": "total", "rtstruct": "", "masks": masks}],
    }
    (seg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        segmentation_temp_root=tmp_path / "seg-tmp",
    )
    return course, contract, nifti, seg_dir, series_uid, sop_hash


def _model_run_spy(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        output_dir = Path(args[2])
        output_type = args[3]
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_type == "nifti":
            reference = sitk.Image([2, 2, 2], sitk.sitkUInt8)
            reference[0, 0, 0] = 1
            sitk.WriteImage(reference, str(output_dir / "liver.nii.gz"))
        return True

    monkeypatch.setattr(segmentation, "run_totalsegmentator", run)
    return calls


def test_complete_current_masks_rebuild_rs_auto_without_model(monkeypatch, tmp_path):
    course, contract, nifti, seg_dir, series_uid, _ = _make_course(tmp_path)
    calls = _model_run_spy(monkeypatch)

    result = segmentation.segment_course(
        PipelineConfig(
            dicom_root=tmp_path / "input",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            segmentation_temp_root=tmp_path / "seg-tmp",
        ),
        course,
    )

    assert calls == []
    assert result["nifti_seg_dir"] == str(seg_dir)

    class StubRT:
        def __init__(self):
            self.names = []

        def add_roi(self, mask, name):
            self.names.append(name)

        def save(self, path):
            Path(path).write_bytes(b"RTSTRUCT")

    class StubBuilder:
        last = None

        @staticmethod
        def create_new(dicom_series_path):
            StubBuilder.last = StubRT()
            return StubBuilder.last

    (seg_dir / f"{seg_dir.name}--total.dcm").unlink(missing_ok=True)
    monkeypatch.setattr("rt_utils.RTStructBuilder", StubBuilder)
    monkeypatch.setattr(auto_rtstruct, "_load_ct_image", lambda _: sitk.ReadImage(str(nifti)))
    monkeypatch.setattr(auto_rtstruct, "sanitize_rtstruct", lambda _: None)
    monkeypatch.setattr(auto_rtstruct, "fix_rtstruct_rois", lambda *_: None)

    output = auto_rtstruct.build_auto_rtstruct(course)
    assert output == course / "RS_auto.dcm"
    assert calls == []
    assert StubBuilder.last is not None
    assert StubBuilder.last.names == ["liver", "spleen"]

    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["total"]["action"] == "reused"
    assert audit["decisions"]["RS_auto"]["action"] == "rebuilt"
    assert audit["source"]["planning_ct_series_instance_uid"] == series_uid



def test_missing_model_rtstruct_is_derived_without_rerunning_model(monkeypatch, tmp_path):
    course, _, nifti, seg_dir, _, _ = _make_course(tmp_path)
    calls = _model_run_spy(monkeypatch)

    class RecordingRT:
        def __init__(self):
            self.names = []

        def add_roi(self, mask, name):
            self.names.append(name)

        def save(self, path):
            Path(path).write_bytes(b"RTSTRUCT")

    class RecordingBuilder:
        instance = None

        @staticmethod
        def create_new(dicom_series_path):
            RecordingBuilder.instance = RecordingRT()
            return RecordingBuilder.instance

    monkeypatch.setattr("rt_utils.RTStructBuilder", RecordingBuilder)
    monkeypatch.setattr(auto_rtstruct, "_load_ct_image", lambda _: sitk.ReadImage(str(nifti)))
    result = segmentation.segment_course(
        PipelineConfig(
            dicom_root=tmp_path / "input",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            segmentation_temp_root=tmp_path / "seg-tmp",
        ),
        course,
    )

    assert calls == []
    assert result["dicom_seg"] == str(seg_dir / f"{seg_dir.name}--total.dcm")
    assert RecordingBuilder.instance is not None
    assert sorted(RecordingBuilder.instance.names) == ["liver", "spleen"]
    manifest = json.loads((seg_dir / "manifest.json").read_text(encoding="utf-8"))
    total = next(item for item in manifest["models"] if item["model"] == "total")
    assert total["rtstruct"] == f"{seg_dir.name}--total.dcm"
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["RS_auto"]["action"] == "rebuilt"
    assert audit["decisions"]["RS_auto"]["model_run"] is False



def test_masks_from_different_planning_ct_rerun_model(monkeypatch, tmp_path):
    course, _, _, seg_dir, series_uid, _ = _make_course(tmp_path)
    manifest_path = seg_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_series_instance_uid"] = generate_uid()
    manifest["planning_ct_series_instance_uid"] = manifest["source_series_instance_uid"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    calls = _model_run_spy(monkeypatch)

    segmentation.segment_course(
        PipelineConfig(
            dicom_root=tmp_path / "input",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            segmentation_temp_root=tmp_path / "seg-tmp",
        ),
        course,
    )

    assert len(calls) == 2
    assert all(call[1].get("task") is None for call in calls)
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["total"]["action"] == "rebuilt"
    assert audit["decisions"]["total"]["model_run"] is True


def test_incomplete_masks_rerun_model(monkeypatch, tmp_path):
    course, _, _, seg_dir, _, _ = _make_course(tmp_path)
    (seg_dir / "total--spleen.nii.gz").unlink()
    calls = _model_run_spy(monkeypatch)

    segmentation.segment_course(
        PipelineConfig(
            dicom_root=tmp_path / "input",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            segmentation_temp_root=tmp_path / "seg-tmp",
        ),
        course,
    )

    assert len(calls) == 2
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["total"]["action"] == "rebuilt"
    assert "inconsistent" in audit["decisions"]["total"]["reason"]


def test_current_rs_auto_and_rs_custom_are_reused_without_model_or_rebuild(monkeypatch, tmp_path):
    course, _, _, seg_dir, series_uid, _ = _make_course(tmp_path)
    rs_auto = course / "RS_auto.dcm"
    _write_rtstruct(rs_auto, series_uid)
    original_rs_auto = rs_auto.read_bytes()
    _write_rtstruct(course / "RS_custom.dcm", series_uid)
    calls = _model_run_spy(monkeypatch)

    segmentation.segment_course(
        PipelineConfig(
            dicom_root=tmp_path / "input",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            segmentation_temp_root=tmp_path / "seg-tmp",
        ),
        course,
    )

    assert calls == []
    assert auto_rtstruct.build_auto_rtstruct(course) == rs_auto
    assert rs_auto.read_bytes() == original_rs_auto
    assert calls == []
    assert custom_structures_rtstruct._is_rs_custom_stale(
        course / "RS_custom.dcm", None, None, None
    ) is False

    from rtpipeline.radiomics import _standard_rtstruct_sources

    sources = _standard_rtstruct_sources(segmentation.load_course_contract(course), course)
    assert ("AutoRTS_total", rs_auto, None) in sources
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["total"]["action"] == "reused"
    assert audit["decisions"]["RS_auto"]["action"] == "reused"
    assert audit["decisions"]["RS_custom"]["action"] == "reused"


def test_reuse_only_leaves_manifest_generation_time_unchanged(monkeypatch, tmp_path):
    course, _, _, seg_dir, series_uid, _ = _make_course(tmp_path)
    _write_rtstruct(seg_dir / f"{seg_dir.name}--total.dcm", series_uid)
    manifest_path = seg_dir / "manifest.json"
    before = manifest_path.read_bytes()
    calls = _model_run_spy(monkeypatch)
    segmentation.segment_course(
        PipelineConfig(
            dicom_root=tmp_path / "input",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            segmentation_temp_root=tmp_path / "seg-tmp",
        ),
        course,
    )
    assert calls == []
    assert manifest_path.read_bytes() == before


def test_legacy_reuse_fails_when_nifti_is_newer_than_manifest(tmp_path):
    course, _, nifti, seg_dir, series_uid, _ = _make_course(tmp_path)
    manifest_path = seg_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "source_series_instance_uid",
        "planning_ct_series_instance_uid",
        "source_nifti_sha256",
        "source_ct_sop_hash",
    ):
        manifest.pop(key, None)
    manifest["generated_at"] = "2026-08-20T10:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sidecar_path = nifti.with_name(f"{segmentation._strip_nifti_base(nifti)}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["series_instance_uid"] = series_uid
    sidecar["generated_at"] = "2026-08-29T19:00:00+00:00"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    import os
    os.utime(manifest_path, (1000, 1000))
    os.utime(nifti, (2000, 2000))

    current, reason = segmentation._series_masks_current(
        seg_dir,
        seg_dir.name,
        "total",
        source_nifti=nifti,
        planning_ct_series_uid=series_uid,
        source_ct_sop_hash=None,
    )
    assert current is False
    assert "legacy mask provenance" in reason


def test_legacy_masks_survive_sidecar_refresh_when_content_identity_matches(tmp_path):
    course, contract, nifti, seg_dir, series_uid, _ = _make_course(tmp_path)
    manifest_path = seg_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "source_series_instance_uid",
        "planning_ct_series_instance_uid",
        "source_nifti_sha256",
        "source_ct_sop_hash",
    ):
        manifest.pop(key, None)
    manifest["generated_at"] = "2026-08-20T10:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sidecar_path = nifti.with_name(f"{segmentation._strip_nifti_base(nifti)}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["series_instance_uid"] = series_uid
    sidecar["generated_at"] = "2026-08-29T19:00:00+00:00"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    current, reason = segmentation._series_masks_current(
        seg_dir,
        seg_dir.name,
        "total",
        source_nifti=nifti,
        planning_ct_series_uid=series_uid,
        source_ct_sop_hash=None,
    )
    assert current is True
    assert "contracted planning CT" in reason


def test_ensure_ct_nifti_preserves_sidecar_timestamp_without_regeneration(monkeypatch, tmp_path):
    course, _, nifti, _, _, _ = _make_course(tmp_path)
    ct_dir = course / "DICOM" / "CT"
    sidecar_path = nifti.with_name(f"{segmentation._strip_nifti_base(nifti)}.metadata.json")
    original = json.loads(sidecar_path.read_text(encoding="utf-8"))
    original["generated_at"] = "2026-08-20T10:00:00+00:00"
    sidecar_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
    monkeypatch.setattr(segmentation, "_derive_nifti_name", lambda _: segmentation._strip_nifti_base(nifti))
    monkeypatch.setattr(segmentation, "run_dcm2niix", lambda *args, **kwargs: pytest.fail("NIfTI was regenerated"))

    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        segmentation_temp_root=tmp_path / "seg-tmp",
    )
    assert segmentation._ensure_ct_nifti(config, ct_dir, nifti.parent) == nifti
    refreshed = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert refreshed["generated_at"] == original["generated_at"]


@pytest.mark.parametrize("layout", ["current", "legacy", "mixed"])
def test_binary_mask_layouts_produce_one_roi_per_input(monkeypatch, tmp_path, layout):
    course, _, nifti, seg_dir, _, _ = _make_course(tmp_path)
    for path in list(seg_dir.glob("*.dcm")):
        path.unlink()
    if layout in {"legacy", "mixed"}:
        for path in list(seg_dir.glob("total--*.nii.gz")):
            legacy = seg_dir / f"{seg_dir.name}--{path.name}"
            path.rename(legacy)
    if layout == "mixed":
        for path in sorted(seg_dir.glob(f"{seg_dir.name}--total--*.nii.gz")):
            current = seg_dir / path.name.replace(f"{seg_dir.name}--", "", 1)
            shutil.copy2(path, current)

    class RecordingRT:
        def __init__(self):
            self.names = []

        def add_roi(self, mask, name):
            self.names.append(name)

        def save(self, path):
            Path(path).write_bytes(b"RTSTRUCT")

    class RecordingBuilder:
        last = None

        @staticmethod
        def create_new(dicom_series_path):
            RecordingBuilder.last = RecordingRT()
            return RecordingBuilder.last

    monkeypatch.setattr("rt_utils.RTStructBuilder", RecordingBuilder)
    monkeypatch.setattr(auto_rtstruct, "_load_ct_image", lambda _: sitk.ReadImage(str(nifti)))
    monkeypatch.setattr(auto_rtstruct, "sanitize_rtstruct", lambda _: None)
    monkeypatch.setattr(auto_rtstruct, "fix_rtstruct_rois", lambda *_: None)

    output = auto_rtstruct.build_auto_rtstruct(course)
    assert output == course / "RS_auto.dcm"
    assert RecordingBuilder.last is not None
    assert sorted(RecordingBuilder.last.names) == ["liver", "spleen"]


def test_stale_rs_auto_is_removed_before_rebuild_and_a_failed_rebuild_leaves_none(
    monkeypatch, tmp_path
):
    course, _, _, seg_dir, series_uid, _ = _make_course(tmp_path)
    stale = course / "RS_auto.dcm"
    stale_series_uid = generate_uid()
    _write_rtstruct(stale, stale_series_uid)
    for path in seg_dir.glob("*.nii.gz"):
        path.unlink()

    load_attempted = False

    def fail_ct_load(_ct_dir):
        nonlocal load_attempted
        load_attempted = True
        assert not stale.exists(), "rejected RS_auto remained visible when rebuild began"
        return None

    monkeypatch.setattr(auto_rtstruct, "_load_ct_image", fail_ct_load)

    assert auto_rtstruct.build_auto_rtstruct(course) is None
    assert load_attempted is True
    assert not stale.exists()
    assert list(course.glob(".RS_auto.dcm.rejected.*")) == []
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    decision = audit["decisions"]["RS_auto"]
    assert decision["action"] == "failed"
    assert "planning CT could not be loaded" in decision["reason"]
    assert decision["rejected_artifact"] == {
        "action": "removed",
        "path": "RS_auto.dcm",
        "reason": (
            f"referenced planning CT series {stale_series_uid}, not the current "
            f"planning CT series {series_uid}"
        ),
    }


def test_rejected_rs_auto_audit_is_written_before_rebuild_exception(monkeypatch, tmp_path):
    course, _, nifti, seg_dir, series_uid, _ = _make_course(tmp_path)
    stale = course / "RS_auto.dcm"
    stale_series_uid = generate_uid()
    _write_rtstruct(stale, stale_series_uid)

    multilabel = sitk.Image([2, 2, 2], sitk.sitkUInt8)
    multilabel[0, 0, 0] = 1
    sitk.WriteImage(multilabel, str(seg_dir / "total--multilabel.nii.gz"))

    class StubRT:
        def add_roi(self, mask, name):
            return None

        def save(self, path):
            Path(path).write_bytes(b"RTSTRUCT")

    class StubBuilder:
        @staticmethod
        def create_new(dicom_series_path):
            return StubRT()

    def fail_resample(*_args, **_kwargs):
        assert not stale.exists()
        raise RuntimeError("synthetic resampling failure")

    monkeypatch.setattr("rt_utils.RTStructBuilder", StubBuilder)
    monkeypatch.setattr(auto_rtstruct, "_load_ct_image", lambda _: sitk.ReadImage(str(nifti)))
    monkeypatch.setattr(auto_rtstruct, "_resample_to_reference", fail_resample)

    with pytest.raises(RuntimeError, match="synthetic resampling failure"):
        auto_rtstruct.build_auto_rtstruct(course)

    assert not stale.exists()
    audit = json.loads(
        (course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8")
    )
    decision = audit["decisions"]["RS_auto"]
    assert decision["action"] == "rejected"
    assert decision["reason"].startswith("removed before rebuild:")
    assert decision["rejected_artifact"] == {
        "action": "removed",
        "path": "RS_auto.dcm",
        "reason": (
            f"referenced planning CT series {stale_series_uid}, not the current "
            f"planning CT series {series_uid}"
        ),
    }


@pytest.mark.parametrize(
    "provenance",
    ["different-series", "unverifiable", "authoritative-unverifiable"],
)
def test_standard_sources_exclude_rs_auto_without_current_planning_ct_provenance(
    tmp_path, provenance
):
    course, contract, _, _, _, _ = _make_course(tmp_path)
    rs_auto = course / "RS_auto.dcm"
    if provenance == "different-series":
        _write_rtstruct(rs_auto, generate_uid())
    else:
        rs_auto.write_bytes(b"not a verifiable RTSTRUCT")
    if provenance == "authoritative-unverifiable":
        contract = SimpleNamespace(
            planning_ct=getattr(contract, "planning_ct"),
            planning_ct_dir=getattr(contract, "planning_ct_dir"),
            planning_ct_nifti=getattr(contract, "planning_ct_nifti"),
            authoritative_rtstruct_path=rs_auto,
            authoritative_rtstruct_source="AutoRTS_total",
        )

    from rtpipeline.radiomics import _standard_rtstruct_sources

    sources = _standard_rtstruct_sources(contract, course)
    assert all(source != "AutoRTS_total" for source, _, _ in sources)
    assert rs_auto.exists(), "source validation must not mutate the course"


def test_qc_skipped_model_is_not_recorded_as_rebuilt(monkeypatch, tmp_path):
    course, _, _, _, _, _ = _make_course(tmp_path)
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        segmentation_temp_root=tmp_path / "seg-tmp",
        extra_seg_models=["lung_vessels"],
    )
    class QC:
        @staticmethod
        def save_body_region_qc(*args, **kwargs):
            return None

        @staticmethod
        def check_model_eligibility(*args, **kwargs):
            return False, "required body region absent"

    monkeypatch.setattr(segmentation, "_get_qc_functions", lambda: QC)
    calls = _model_run_spy(monkeypatch)
    result = segmentation.segment_course(config, course)
    assert calls == []
    assert result["skipped_models"]["lung_vessels"] == "required body region absent"
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["lung_vessels"]["action"] == "skipped"
    assert audit["decisions"]["lung_vessels"]["model_run"] is False


def test_failed_model_run_is_not_recorded_as_rebuilt(monkeypatch, tmp_path):
    course, _, _, seg_dir, _, _ = _make_course(tmp_path)
    (seg_dir / "total--spleen.nii.gz").unlink()
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        segmentation_temp_root=tmp_path / "seg-tmp",
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        return False

    monkeypatch.setattr(segmentation, "run_totalsegmentator", fail)
    segmentation.segment_course(config, course)
    assert len(calls) == 2
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    assert audit["decisions"]["total"]["action"] == "failed"
    assert audit["decisions"]["total"]["model_run"] is True
    assert audit["decisions"]["total"]["run_succeeded"] is False
