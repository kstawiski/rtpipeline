"""Exact bindings, reader verification and recovery on synthetic DVH courses."""
import json
import os
from pathlib import Path

import pandas as pd
import pydicom
import pytest

import dvh_plan_target_fixtures as fx
from rtpipeline import dvh, dvh_generation as gen
from rtpipeline.durable_progress import DurableProgressError, ProgressJournal


def course(root):
    return fx.build_course(root, rois=[("Organ", "ORGAN", fx.DOSED_LEFT_X)])


def events(root):
    return [r["body"] for r in ProgressJournal.read(root / ".dvh/progress.jsonl").records]


def measure(root, **kwargs):
    return dvh.dvh_for_course(root, parallel_workers=1, **kwargs)


def sampler(monkeypatch):
    calls = []
    original = dvh.dvhcalc.get_dvh
    def counted(*args, **kwargs):
        calls.append(args[2] if len(args) > 2 else kwargs.get("roi"))
        return original(*args, **kwargs)
    monkeypatch.setattr(dvh.dvhcalc, "get_dvh", counted)
    return calls


def test_exact_resume_is_a_noop_and_readback_binds_every_byte(tmp_path, monkeypatch):
    calls = sampler(monkeypatch)
    root = course(tmp_path)
    assert measure(root)
    marker = gen.current(root)
    before = {name: (root / name).read_bytes() for name, sha in marker["artifacts"].items()
              if sha and name != "ledger.json"}
    mtimes = {name: (root / name).stat().st_mtime_ns for name in before}
    count = len(calls)
    assert count > 0
    assert measure(root)
    assert len(calls) == count
    assert gen.current(root) == marker
    for name, raw in before.items():
        assert (root / name).read_bytes() == raw
        assert (root / name).stat().st_mtime_ns == mtimes[name]
    gen.verify(root / ".dvh/current", gen.binding(root, gen.effective_settings()))


@pytest.mark.parametrize("change", ["dose", "ct", "second_mask", "crop_setting"])
def test_preserved_mtime_binding_change_recomputes_only_affected_course(tmp_path, monkeypatch, change):
    calls = sampler(monkeypatch)
    one, two = course(tmp_path / "one"), course(tmp_path / "two")
    assert measure(one) and measure(two)
    second = gen.current(two)
    count = len(calls)
    kwargs = {}
    if change in {"dose", "ct"}:
        path = one / "DICOM/RTDOSE/dose.dcm" if change == "dose" else next((one / "DICOM/CT").rglob("*.dcm"))
        stat = path.stat()
        dataset = pydicom.dcmread(path)
        # Change an irrelevant text byte while retaining UIDs and geometry;
        # content binding must still detect this exact source change.
        dataset.SeriesDescription = "changed synthetic bytes"
        dataset.save_as(path, enforce_file_format=True)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "second_mask":
        import SimpleITK as sitk
        for index in (1, 2):
            path = one / "Segmentation_TotalSegmentator" / f"unused{index}.nii.gz"
            path.parent.mkdir(exist_ok=True)
            image = sitk.Image([3, 3, 3], sitk.sitkUInt8)
            sitk.WriteImage(image, str(path))
        # Establish the inventory once, then mutate only its second mask.
        assert measure(one)
        count = len(calls)
        path = one / "Segmentation_TotalSegmentator/unused2.nii.gz"
        stat = path.stat()
        image = sitk.ReadImage(str(path)); image[1, 1, 1] = 1
        sitk.WriteImage(image, str(path))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    else:
        kwargs["use_cropped"] = False
    assert measure(one, **kwargs)
    assert len(calls) > count
    count = len(calls)
    assert measure(two)
    assert len(calls) == count
    assert gen.current(two) == second
    reason = "DVH_BINDING_CONFIG_CHANGED" if change == "crop_setting" else "DVH_BINDING_INPUTS_CHANGED"
    assert any(e.get("reason_code") == reason for e in events(one))


def test_code_identity_is_rechecked_inside_same_process(tmp_path, monkeypatch):
    package = Path(gen.__file__).parent
    shadow = tmp_path / "package"
    import shutil
    shutil.copytree(package, shadow, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(package.parent / "Snakefile", tmp_path / "Snakefile")
    shutil.copytree(package.parent / "workflow", tmp_path / "workflow")
    # The shadow package is logically rooted at tmp_path/package rather than
    # tmp_path/rtpipeline; retain the wrapper's stable repository-relative name.
    (tmp_path / "rtpipeline").mkdir()
    shutil.copy2(package / "workflow_delegate.py", tmp_path / "rtpipeline/workflow_delegate.py")
    monkeypatch.setattr(gen, "__file__", str(shadow / "dvh_generation.py"))
    root = course(tmp_path / "fixture")
    assert measure(root)
    path = shadow / "dvh_support.py"
    stat = path.stat()
    path.write_bytes(path.read_bytes() + b"\n# binding mutation\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert measure(root)
    assert any(e.get("reason_code") == "DVH_BINDING_CODE_CHANGED" for e in events(root))


@pytest.mark.parametrize("artifact", ["dvh_metrics.parquet", "dvh_metrics.xlsx", "dvh_curves.json",
                                      "metadata/dvh_qc.json"])
def test_corrupt_committed_artifact_rebuilds_and_preserves_old_generation(tmp_path, artifact):
    root = course(tmp_path)
    assert measure(root)
    old = (root / ".dvh/current").resolve()
    (root / artifact).write_bytes(b"truncated")
    assert measure(root)
    assert old.is_dir() and (old / artifact).read_bytes() == b"truncated"
    assert (root / ".dvh/current").resolve() != old
    gen.current(root)
    assert any(e.get("reason_code") == "DVH_GENERATION_HASH_MISMATCH" for e in events(root))


def test_legacy_marker_recomputes_with_fixed_reason(tmp_path):
    root = course(tmp_path)
    (root / ".dvh_done").write_text("ok\n")
    assert measure(root)
    assert any(e.get("reason_code") == "DVH_LEGACY_UNBOUND" for e in events(root))


def test_structural_failure_needs_explicit_retry_or_changed_binding(tmp_path, monkeypatch):
    root = course(tmp_path)
    original = dvh._compute_dvh_for_course
    count = []
    def failed(*args, **kwargs):
        count.append(1)
        raise ValueError("synthetic structural input failure")
    monkeypatch.setattr(dvh, "_compute_dvh_for_course", failed)
    with pytest.raises(ValueError): measure(root)
    with pytest.raises(DurableProgressError, match="DVH_FAILED_RETRY_REQUIRES_EXPLICIT_RULE"):
        measure(root)
    assert len(count) == 1
    monkeypatch.setattr(dvh, "_compute_dvh_for_course", original)
    assert measure(root, retry_failed=True)
    assert json.loads((root / ".dvh/attempt.json").read_text())["state"] == "committed"


@pytest.mark.parametrize("point", ["published_before_commit", "committed_before_ledger"])
def test_publication_error_does_not_overwrite_previous_commit(tmp_path, monkeypatch, point):
    root = course(tmp_path)
    assert measure(root)
    old = (root / ".dvh/current").resolve()
    old_hashes = gen.verify(old)["artifacts"]
    def fail(name):
        if name == point: raise OSError("synthetic I/O fault")
    monkeypatch.setattr(gen, "_failpoint", fail)
    with pytest.raises(OSError): measure(root, use_cropped=False)
    assert gen.verify(old)["artifacts"] == old_hashes
    monkeypatch.setattr(gen, "_failpoint", lambda _: None)
    assert measure(root, use_cropped=False, retry_failed=True)
    gen.current(root)


def test_unreadable_deciding_source_fails_closed(tmp_path, monkeypatch):
    root = course(tmp_path)
    assert measure(root)
    monkeypatch.setattr(gen, "__file__", str(tmp_path / "absent/dvh_generation.py"))
    with pytest.raises(DurableProgressError, match="DVH_BINDING_UNREADABLE"):
        measure(root)


def test_measurement_bytes_match_unmodified_producer(tmp_path):
    """Run the pinned pre-change producer against exactly the same input bytes."""
    import subprocess
    import types
    import zipfile
    source = subprocess.check_output(["git", "show", "7e509c7:rtpipeline/dvh.py"],
                                     cwd=Path(__file__).resolve().parents[1], text=True)
    legacy = types.ModuleType("rtpipeline._dvh_before_generations")
    legacy.__package__ = "rtpipeline"
    legacy.__file__ = dvh.__file__
    # dataclasses inspects the defining module during import.
    import sys
    sys.modules[legacy.__name__] = legacy
    try:
        exec(compile(source, "7e509c7/rtpipeline/dvh.py", "exec"), legacy.__dict__)
        root = fx.build_course(tmp_path, rois=[("PTV1", "PTV", fx.DOSED_LEFT_X),
            ("PTV2", "PTV", fx.UNDOSED_RIGHT_X), ("Organ", "ORGAN", fx.WHOLE_X)])
        assert legacy.dvh_for_course(root, parallel_workers=1)
        parquet = (root / "dvh_metrics.parquet").read_bytes()
        curves = (root / "dvh_curves.json").read_bytes()
        frame = pd.read_parquet(root / "dvh_metrics.parquet")
        with zipfile.ZipFile(root / "dvh_metrics.xlsx") as archive:
            workbook = {name: archive.read(name) for name in archive.namelist() if name != "docProps/core.xml"}
        assert measure(root)
        assert (root / "dvh_metrics.parquet").read_bytes() == parquet
        assert (root / "dvh_curves.json").read_bytes() == curves
        pd.testing.assert_frame_equal(pd.read_parquet(root / "dvh_metrics.parquet"), frame, check_exact=True)
        with zipfile.ZipFile(root / "dvh_metrics.xlsx") as archive:
            assert {name: archive.read(name) for name in archive.namelist() if name != "docProps/core.xml"} == workbook
    finally:
        sys.modules.pop(legacy.__name__, None)


def test_committed_recovery_reconciles_attempt_state_without_sampling(tmp_path, monkeypatch):
    root = course(tmp_path)
    calls = sampler(monkeypatch)
    def stop(name):
        if name == "committed_before_ledger": raise RuntimeError("synthetic controller loss")
    monkeypatch.setattr(gen, "_failpoint", stop)
    with pytest.raises(RuntimeError): measure(root)
    count = len(calls)
    assert count > 0
    monkeypatch.setattr(gen, "_failpoint", lambda _: None)
    assert measure(root)
    assert len(calls) == count
    assert json.loads((root / ".dvh/attempt.json").read_text())["state"] == "committed"


def test_missing_or_replaced_alias_is_repaired_without_sampling(tmp_path, monkeypatch):
    root = course(tmp_path)
    calls = sampler(monkeypatch)
    assert measure(root)
    marker = gen.current(root)
    count = len(calls)
    path = root / "dvh_metrics.parquet"
    path.unlink()
    path.write_bytes(b"uncommitted fixed-path replacement")
    assert measure(root)
    assert len(calls) == count
    assert path.is_symlink()
    assert gen.current(root) == marker


def test_disabled_completion_does_not_modify_prior_immutable_receipt(tmp_path):
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    root = course(tmp_path)
    assert measure(root)
    old = (root / ".dvh/current").resolve()
    original = (old / ".dvh_done").read_bytes()
    dependency = materialize_stage_dependency(tmp_path / "config", "dvh", {})
    receipt = write_stage_completion_sentinel(root, root / ".dvh_done", stage="dvh", status="disabled",
                                             configuration_dependency=dependency)
    assert receipt["status"] == "disabled" and receipt["dvh_binding"]
    assert (old / ".dvh_done").read_bytes() == original
    gen.verify(old)


def test_direct_nifti_measurement_survives_generation_reassembly(tmp_path, monkeypatch):
    import numpy as np
    import SimpleITK as sitk
    from course_contract_test_utils import write_minimal_course_contract
    root = course(tmp_path)
    from synthetic_ct_series import write_ct_series
    ct_dir = root / "DICOM/CT"
    for path in ct_dir.glob("*.dcm"):
        path.unlink()  # Replace the helper's header-only CT with generated pixels.
    paths = write_ct_series(ct_dir, [0., 2., 4., 6., 8.], signed=False, rows=20, columns=20)
    for index, path in enumerate(paths):
        image = pydicom.dcmread(path)
        image.FrameOfReferenceUID = fx.FRAME_OF_REFERENCE_UID
        image.PatientID = root.parent.name
        image.ImagePositionPatient = [0., 0., index * 2.]
        image.PixelSpacing = [2., 2.]
        image.SliceThickness = 2.
        image.save_as(path, enforce_file_format=True)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in paths])
    reference = reader.Execute()
    nifti = root / "NIFTI/ct.nii.gz"
    sitk.WriteImage(reference, str(nifti))
    write_minimal_course_contract(root, selected_plans=[root / "DICOM/RTPLAN/plan.dcm"],
        selected_doses=[root / "DICOM/RTDOSE/dose.dcm"], authoritative_rtstruct=root / "DICOM/RTSTRUCT/rs.dcm",
        planning_ct_nifti=nifti)
    # Use the production provenance serializer for this pixel-bearing CT;
    # the minimal fixture helper is intended for header-only CTs.
    from rtpipeline.course_contract import _ct_provenance
    metadata = root / "metadata/case_metadata.json"
    payload = json.loads(metadata.read_text())
    provenance = payload["course_contract"]["planning_ct"]["nifti_provenance"]
    provenance["geometry"] = _ct_provenance(ct_dir)["geometry"]
    sidecar = root / provenance["sidecar_path"]
    publication = json.loads(sidecar.read_text())
    publication["geometry"] = provenance["geometry"]
    sidecar.write_text(json.dumps(publication))
    metadata.write_text(json.dumps(payload))
    fx.mark_fully_delivered(root)
    values = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.uint8)
    values[:, 1:3, 1:3] = 1
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(reference)
    source = root / "Segmentation_TotalSegmentator/extra.nii.gz"
    source.parent.mkdir()
    sitk.WriteImage(mask, str(source))
    calls = []
    original = dvh._compute_metrics_from_arrays
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(dvh, "_compute_metrics_from_arrays", counted)
    assert measure(root)
    frame = pd.read_parquet(root / "dvh_metrics.parquet")
    assert frame.Segmentation_Source.eq("NIfTI_Direct").sum() == 1
    assert calls
    count = len(calls)
    original_table = (root / "dvh_metrics.parquet").read_bytes()
    (root / "metadata/dvh_qc.json").write_bytes(b"truncated QC")
    assert measure(root)
    assert len(calls) == count
    assert (root / "dvh_metrics.parquet").read_bytes() == original_table


def test_degraded_qc_keeps_measurements_and_negative_disposition(tmp_path, monkeypatch):
    root = course(tmp_path)
    calls = sampler(monkeypatch)
    monkeypatch.setattr(dvh, "_expected_custom_structure_names", lambda _: ["Missing_required_structure"])
    assert measure(root)
    before = (root / "dvh_metrics.parquet").read_bytes()
    qc = json.loads((root / "metadata/dvh_qc.json").read_text())
    assert qc["status"] == "degraded"
    assert qc["custom_structures_missing"] == ["Missing_required_structure"]
    assert pd.read_parquet(root / "dvh_metrics.parquet").DmeanGy.notna().any()
    count = len(calls)
    assert measure(root)
    assert len(calls) == count
    assert (root / "dvh_metrics.parquet").read_bytes() == before
    assert json.loads((root / "metadata/dvh_qc.json").read_text()) == qc
    assert not dvh._is_dvh_up_to_date(root, root / "dvh_metrics.xlsx")
