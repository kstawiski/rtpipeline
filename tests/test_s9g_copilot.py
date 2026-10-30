"""Synthetic regressions for RS_auto receipts, export validation, CT fallback reads, and dcm2niix skips."""
import hashlib
import inspect
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from rtpipeline import auto_rtstruct as auto, original_segmentation as original
from test_rs_auto_orientation import (
    ANISOTROPIC, _age, _decision, _geometry_course, _origin, _uniform_z,
)
from test_s9c_repairs import bind_export


@pytest.mark.parametrize("processor", ["real_sanitize_rtstruct", "sanitize_rtstruct", "fix_rtstruct_rois"])
def test_copied_receipt_binds_postprocessed_bytes_and_resume(tmp_path, monkeypatch, processor):
    from rtpipeline.rtstruct_geometry import geometry_adapter_output_current, record_geometry_adapter
    from rtpipeline.segmentation import _ensure_model_rtstruct_from_masks

    course, ct, masks = _geometry_course(tmp_path, _uniform_z(), ANISOTROPIC)
    source = _ensure_model_rtstruct_from_masks(ct, masks, "CT_series", "total")
    rewritten = []

    if processor == "real_sanitize_rtstruct":
        ds = pydicom.dcmread(source)
        degenerate = pydicom.Dataset()
        degenerate.ContourData = [0.0, 0.0, 0.0]
        degenerate.NumberOfContourPoints = 1
        ds.ROIContourSequence[0].ContourSequence.append(degenerate)
        ds.save_as(source)
        record_geometry_adapter(source, ct)
    else:
        process = getattr(auto, processor)

        def rewrite(*args):
            result = process(*args)
            path = args[0] if processor == "sanitize_rtstruct" else args[1]
            ds = pydicom.dcmread(path)
            ds.StructureSetLabel = "Synthetic"
            ds.save_as(path)
            rewritten.append(path)
            return result

        monkeypatch.setattr(auto, processor, rewrite)

    _age(ct, course / "NIFTI", masks)
    output = auto.build_auto_rtstruct(course)
    assert output is not None
    if processor != "real_sanitize_rtstruct":
        assert rewritten == [output]
    assert output.read_bytes() != source.read_bytes()
    assert _origin(course)["origin"] == auto.RS_AUTO_ORIGIN_COPY
    assert geometry_adapter_output_current(output, ct)
    receipt = json.loads(output.with_suffix(".geometry.json").read_text())
    assert receipt["rtstruct_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    before = output.read_bytes(), output.stat().st_mtime_ns
    monkeypatch.setattr(auto, "_write_rtstruct_atomic", lambda *a: pytest.fail("resume rebuilt RS_auto"))
    assert auto.build_auto_rtstruct(course) == output
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    assert _decision(course)["action"] == "reused"
    if processor != "real_sanitize_rtstruct":
        assert rewritten == [output]


def _export(course):
    directory = course / "Segmentation_Original/CT"
    directory.mkdir(parents=True)
    paths = [directory / f"ROI_{i}.nii.gz" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(f"synthetic mask {i}".encode())
    bind_export(course, paths)
    return directory, paths


@pytest.mark.parametrize("course_contract", [True, False])
@pytest.mark.parametrize("changed_source", ["CT.nii.gz", "RS.dcm"])
def test_export_reads_shared_sources_once_without_persistent_cache(
    tmp_path, monkeypatch, course_contract, changed_source,
):
    directory, masks = _export(tmp_path)
    if not course_contract:
        (tmp_path / "metadata/case_metadata.json").unlink()
    reads = Counter()
    texts = Counter()
    read_bytes, read_text = Path.read_bytes, Path.read_text

    def counted_bytes(path, *args, **kwargs):
        reads[path] += 1
        return read_bytes(path, *args, **kwargs)

    def counted_text(path, *args, **kwargs):
        texts[path] += 1
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counted_bytes)
    monkeypatch.setattr(Path, "read_text", counted_text)
    original.require_current_original_export(directory)
    assert reads == Counter({path: 1 for path in [*masks, tmp_path / "CT.nii.gz", tmp_path / "RS.dcm"]})
    assert texts[directory / "metadata.json"] == 1
    assert texts[directory / "provenance.json"] == 1
    original.require_current_original_export(directory)
    assert reads == Counter({path: 2 for path in [*masks, tmp_path / "CT.nii.gz", tmp_path / "RS.dcm"]})
    (tmp_path / changed_source).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source"):
        original.require_current_original_export(directory)
    with pytest.raises(ValueError, match="source"):
        original.require_current_original_mask(masks[0])


@pytest.mark.parametrize(
    "damage",
    ["missing_one", "missing_last", "missing_all", "extra", "changed_last", "duplicate"],
)
def test_incomplete_or_changed_export_is_quarantined_and_never_certified(tmp_path, damage):
    from rtpipeline.stage_completion import (
        _discover_outputs, _stale_original_exports, stage_definition,
        validate_stage_completion_sentinel, write_stage_completion_sentinel,
    )
    from test_stage_completion import _dependency

    course = tmp_path / "P/C"
    directory, masks = _export(course)
    original.require_current_original_export(directory)
    if damage == "missing_one":
        masks[0].unlink()
    elif damage == "missing_last":
        masks[-1].unlink()
    elif damage == "missing_all":
        for path in masks:
            path.unlink()
    elif damage == "extra":
        (directory / "extra.nii.gz").write_bytes(b"unlisted")
    elif damage == "changed_last":
        masks[-1].write_bytes(b"changed mask")
    else:
        manifest = json.loads((directory / "metadata.json").read_text())
        manifest["structures"].append(manifest["structures"][0])
        (directory / "metadata.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stale"):
        original.require_current_original_export(directory)
    definition = stage_definition("organize")
    assert _stale_original_exports(course, definition)
    with pytest.raises(ValueError, match="stale"):
        _discover_outputs(course, definition)
    dicom = course / "DICOM/CT/image.dcm"
    dicom.parent.mkdir(parents=True)
    dicom.write_bytes(b"synthetic DICOM")
    payload = write_stage_completion_sentinel(
        course,
        course / ".organized",
        stage="organize",
        status="ok",
        configuration_dependency=_dependency(tmp_path, "organize"),
    )
    assert not any(item["role"] == "original_segmentation" for item in payload["outputs"])
    validate_stage_completion_sentinel(course / ".organized", expected_stage="organize")
    assert not directory.exists()
    assert not list((course / "Segmentation_Original").rglob("*.nii*"))
    quarantined = list((course / "Segmentation_Quarantine").glob("*/export"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "metadata.json").exists()
    receipt = json.loads((quarantined[0].parent / "record.json").read_text())
    assert receipt["path"] == "Segmentation_Original/CT"
    assert "stale" in receipt["reason"]


@pytest.mark.parametrize("use_nifti", [True, False], ids=["nifti", "dicom"])
def test_fallback_loads_one_reference_per_series(tmp_path, monkeypatch, use_nifti):
    from course_contract_test_utils import write_minimal_course_contract, write_synthetic_planning_ct
    from rtpipeline import radiomics_conda as rc
    from rtpipeline.config import PipelineConfig

    course = tmp_path / "P/C"
    ct = write_synthetic_planning_ct(course)
    nifti = course / "NIFTI/CT.nii.gz"
    nifti.parent.mkdir(parents=True)
    reference = sitk.Image([5, 5, 5], sitk.sitkInt16)
    sitk.WriteImage(reference, str(nifti))
    write_minimal_course_contract(course, planning_ct_dir=ct, planning_ct_nifti=nifti)
    contract = rc.load_course_contract(course)
    if not use_nifti:
        contract.data["planning_ct"]["nifti_path"] = ""
    monkeypatch.setattr(rc, "load_course_contract", lambda _: contract)
    array = np.zeros((5, 5, 5), dtype=np.uint8)
    array[1:2, 1:3, 1:4] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(reference)
    masks = course / "Segmentation_TotalSegmentator/CT"
    masks.mkdir(parents=True)
    for roi in ["liver", "spleen", "kidney_left"]:
        sitk.WriteImage(sitk.Flip(mask, [True, False, False]), str(masks / f"total--{roi}.nii.gz"))
    dicom_executes = []

    def execute(*_args):
        dicom_executes.append(1)
        return reference

    monkeypatch.setattr(rc.sitk, "ImageSeriesReader", lambda: type("Reader", (), {
        "GetGDCMSeriesFileNames": lambda *a: ["synthetic"],
        "SetFileNames": lambda *a: None,
        "Execute": execute,
    })())
    reads = Counter()
    read_image = sitk.ReadImage

    def counted_image(path, *args, **kwargs):
        reads[Path(path)] += 1
        return read_image(path, *args, **kwargs)

    monkeypatch.setattr(rc.sitk, "ReadImage", counted_image)
    seen = []

    def batch(tasks, *args, **kwargs):
        assert len(tasks) == 3
        for task in tasks:
            # Use the unpatched reader so only preparation reads are counted.
            aligned = read_image(task["mask_path"])
            expected = read_image(task["image_path"])
            np.testing.assert_array_equal(sitk.GetArrayFromImage(aligned), array)
            assert aligned.GetOrigin() == expected.GetOrigin()
            assert aligned.GetSpacing() == expected.GetSpacing()
            assert aligned.GetDirection() == expected.GetDirection()
        seen.extend(tasks)
        raise RuntimeError("synthetic batch reached")

    monkeypatch.setattr(rc, "process_radiomics_batch", batch)
    config = PipelineConfig(
        dicom_root=tmp_path,
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
        radiomics_min_voxels=1,
    )
    with pytest.raises(RuntimeError, match="synthetic batch reached"):
        rc.radiomics_for_course_ct_nifti_fallback(course, config)
    assert seen
    reference_paths = {Path(task["image_path"]) for task in seen}
    assert len(reference_paths) == 1
    assert {path: reads[path] for path in reference_paths} == {path: 1 for path in reference_paths}
    if use_nifti:
        assert reads[nifti] == 1
        assert not dicom_executes
    else:
        assert len(dicom_executes) == 1
    assert all(reads[path] == 1 for path in masks.glob("*.nii.gz"))
    assert not any(path.exists() for path in reference_paths)
    assert all(not Path(task["mask_path"]).exists() for task in seen)


def test_branch_dcm2niix_tests_honor_override_and_skip_when_absent(tmp_path, monkeypatch):
    import test_mask_physical_space as mask_space
    import test_nifti_nonuniform_fallback as fallback
    import test_s9c_repairs as s9c

    clean_path = os.pathsep.join(
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not (Path(entry) / "dcm2niix").exists()
    )
    monkeypatch.setenv("PATH", clean_path)
    monkeypatch.delenv("RTPIPELINE_TEST_DCM2NIIX", raising=False)

    for module, fn in (
        (s9c, s9c.test_f2_real_512_oblique_dcm2niix),
        (mask_space, mask_space.test_real_dcm2niix),
    ):
        assert "dcm2niix" in inspect.signature(fn).parameters
        assert getattr(module, "dcm2niix", None) is fallback.dcm2niix

    with pytest.raises(pytest.skip.Exception, match="dcm2niix"):
        fallback.dcm2niix.__wrapped__()

    path_bin = tmp_path / "path_bin"
    path_bin.mkdir()
    path_exe = path_bin / "dcm2niix"
    path_exe.write_text("#!/bin/sh\nexit 0\n")
    path_exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{path_bin}{os.pathsep}{clean_path}")
    monkeypatch.setenv("RTPIPELINE_TEST_DCM2NIIX", str(tmp_path / "missing_dcm2niix"))
    with pytest.raises(pytest.skip.Exception, match="dcm2niix"):
        fallback.dcm2niix.__wrapped__()

    override_exe = tmp_path / "custom_dcm2niix"
    override_exe.write_text("#!/bin/sh\nexit 0\n")
    override_exe.chmod(0o755)
    monkeypatch.setenv("PATH", clean_path)
    monkeypatch.setenv("RTPIPELINE_TEST_DCM2NIIX", str(override_exe))
    resolved = fallback.dcm2niix.__wrapped__()
    assert resolved == str(override_exe)

    invoked = []

    def fake_run(cmd, *args, **kwargs):
        invoked.append(cmd[0])
        raise RuntimeError("converter invoked")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="converter invoked"):
        s9c.test_f2_real_512_oblique_dcm2niix(tmp_path / "s9c", resolved, None)
    fake_source = (tmp_path / "ct", tmp_path / "rs.dcm", None, None)
    mask_tmp = tmp_path / "mask"
    mask_tmp.mkdir()
    with pytest.raises(RuntimeError, match="converter invoked"):
        mask_space.test_real_dcm2niix(fake_source, mask_tmp, resolved)
    assert invoked == [str(override_exe), str(override_exe)]
