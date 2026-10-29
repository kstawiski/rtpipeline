"""Synthetic regressions for organize scheduling and course MR resume."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pydicom
import pytest
from pydicom.uid import MRImageStorage, generate_uid

from conftest import snakemake_executable
from course_contract_test_utils import write_minimal_course_contract
from rtpipeline import segmentation as seg
from rtpipeline.rtstruct_geometry import geometry_adapter_output_current
from rtpipeline.stage_completion import write_stage_completion_sentinel
from test_d1_all_series_original_segmentation import _cfg
from test_rs_auto_orientation import (
    ANISOTROPIC,
    _assert_physical_extent,
    _geometry_course,
    _uniform_z,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.requires_snakemake
@pytest.mark.parametrize("state", ["missing", "legacy", "changed", "valid"])
def test_manifest_courses_schedule_organize_only_for_invalid_completion(tmp_path, state):
    resources = tmp_path / "rtpipeline"
    resources.mkdir()
    for source in (ROOT / "rtpipeline").glob("*.yaml"):
        shutil.copyfile(source, resources / source.name)
    (tmp_path / "config.yaml").write_text(
        "dicom_root: input\noutput_dir: Output\nlogs_dir: Logs\n"
        "radiomics:\n  enabled: false\n",
        encoding="utf-8",
    )
    output = tmp_path / "Output"
    manifest = output / "_COURSES/manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    def dry_run(target):
        return subprocess.run(
            [snakemake_executable(), "--snakefile", str(ROOT / "Snakefile"),
             "--cores", "1", "--dry-run", str(target)],
            cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
        )

    startup = dry_run(manifest)
    assert startup.returncode == 0, startup.stdout + startup.stderr
    dependency, = tmp_path.glob(".snakemake/rtpipeline-config-dependencies/*/organize.json")
    records = []
    for name in ("C1", "C2"):
        course = output / "P1" / name
        contract = course / "metadata/case_metadata.json"
        contract.parent.mkdir(parents=True)
        contract.write_text('{"course_contract":{"version":4}}\n', encoding="utf-8")
        dicom = course / "DICOM/CT/image.dcm"
        dicom.parent.mkdir(parents=True)
        dicom.write_bytes(b"synthetic-organized-object")
        write_stage_completion_sentinel(
            course, course / ".organized", stage="organize", status="ok",
            configuration_dependency=dependency,
        )
        records.append({"patient": "P1", "course": name, "path": str(course)})
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"courses": records}), encoding="utf-8")
    # Require startup to advance the dependency beyond the manifest, even when
    # its current timestamp is substantially older than that publication.
    now = time.time_ns()
    os.utime(dependency, ns=(now - 10_000_000_000,) * 2)
    os.utime(manifest, ns=(now - 5_000_000_000,) * 2)
    marker_before = dependency.stat().st_mtime_ns
    sentinel = output / "P1/C2/.organized"
    if state == "missing":
        sentinel.unlink()
    elif state == "legacy":
        sentinel.write_text("ok\n", encoding="utf-8")
    elif state == "changed":
        (sentinel.parent / "metadata/case_metadata.json").write_text("{}\n", encoding="utf-8")

    result = dry_run(output / "P1/C2/.segmentation_done")
    transcript = result.stdout + result.stderr
    assert result.returncode == 0, transcript
    if state == "valid":
        assert "checkpoint organize_courses:" not in transcript
        assert dependency.stat().st_mtime_ns == marker_before
    else:
        assert "checkpoint organize_courses:" in transcript, transcript
        assert dependency.stat().st_mtime_ns > manifest.stat().st_mtime_ns


@pytest.mark.parametrize("with_ct", [False, True], ids=["mr_only", "ct_and_mr"])
@pytest.mark.parametrize("current", [False, True], ids=["stale", "current"])
def test_segment_course_mr_resume_uses_geometry_currentness(tmp_path, monkeypatch, with_ct, current):
    course, ct, ct_masks = _geometry_course(tmp_path, _uniform_z(), ANISOTROPIC)
    series = seg.build_course_dirs(course).dicom_mr / "T2"
    source = series / "DICOM"
    shutil.copytree(ct, source)
    mr_uid = generate_uid()
    for path in source.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.Modality = "MR"
        ds.SeriesDescription = "T2 anatomical"
        ds.SeriesInstanceUID = mr_uid
        ds.SOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.save_as(path, enforce_file_format=True)
    nifti = series / "NIFTI/MR.nii.gz"
    nifti.parent.mkdir()
    shutil.copyfile(seg.load_course_contract(course).planning_ct_nifti, nifti)
    (nifti.parent / "MR.metadata.json").write_text(json.dumps({
        "modality": "MR", "nifti_path": str(nifti), "source_directory": str(source),
    }), encoding="utf-8")
    masks = series / "Segmentation_TotalSegmentator"
    masks.mkdir()
    mask_paths = []
    for path in ct_masks.glob("total--*.nii.gz"):
        target = masks / path.name.replace("total--", "total_mr--")
        shutil.copyfile(path, target)
        mask_paths.append(target)
    target = masks / "MR--total_mr.dcm"
    manifest = masks / "manifest.json"
    manifest.write_text(json.dumps({"models": [{
        "model": "total_mr", "masks": [p.name for p in mask_paths], "rtstruct": target.name,
    }]}), encoding="utf-8")
    if current:
        assert seg._ensure_model_rtstruct_from_masks(source, masks, "MR", "total_mr") == target
    else:
        target.write_bytes(b"legacy RTSTRUCT without geometry evidence")
    assert geometry_adapter_output_current(target, source) is current
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [target, manifest, *mask_paths]}
    if not with_ct:
        shutil.rmtree(ct)
        write_minimal_course_contract(course)
    monkeypatch.setattr(seg, "_series_masks_current", lambda *a, **k: (True, "current masks"))
    monkeypatch.setattr(seg, "run_totalsegmentator", lambda *a, **k: pytest.fail("resume ran inference"))
    monkeypatch.setattr(seg, "run_dcm2niix", lambda *a, **k: pytest.fail("resume reconverted MR"))

    seg.segment_course(_cfg(tmp_path), course, force=False)

    assert geometry_adapter_output_current(target, source)
    _assert_physical_extent(target, _uniform_z(), ANISOTROPIC)
    for path, observed in before.items():
        if path != target or current:
            assert (path.read_bytes(), path.stat().st_mtime_ns) == observed
    # Every subsequent resume must reuse the geometry-scoped RTSTRUCT too.
    after = (target.read_bytes(), target.stat().st_mtime_ns)
    seg.segment_course(_cfg(tmp_path), course, force=False)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == after
