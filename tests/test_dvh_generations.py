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
    # tmp_path/rtpipeline; retain the wrappers' stable repository-relative names.
    (tmp_path / "rtpipeline").mkdir()
    for name in ("workflow_delegate.py", "snakemake_delegate.py"):
        shutil.copy2(package / name, tmp_path / "rtpipeline" / name)
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


@pytest.fixture
def producer_string_inference():
    """Both producers use the validated pandas 3 text-inference policy.

    pandas 2 defaults to object strings; the current producer explicitly uses
    the validated Arrow string policy. Apply that same runtime policy to the
    unmodified producer before constructing either DataFrame.
    """
    with pd.option_context("future.infer_string", True, "mode.string_storage", "pyarrow"):
        yield


def test_measurement_bytes_match_unmodified_producer(tmp_path, producer_string_inference):
    """Run the pinned producer with the same inputs, packages and inference policy."""
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
        workbook_cells = pd.read_excel(root / "dvh_metrics.xlsx")
        assert measure(root)
        assert (root / "dvh_metrics.parquet").read_bytes() == parquet
        assert (root / "dvh_curves.json").read_bytes() == curves
        pd.testing.assert_frame_equal(pd.read_parquet(root / "dvh_metrics.parquet"), frame, check_exact=True)
        if int(pd.__version__.split(".")[0]) >= 3:
            # The validated production pandas: the pinned plain-header layout
            # equals the unmodified producer's workbook byte for byte.
            with zipfile.ZipFile(root / "dvh_metrics.xlsx") as archive:
                assert {name: archive.read(name) for name in archive.namelist() if name != "docProps/core.xml"} == workbook
        else:
            # pandas 2 (the only pandas on Python 3.10) bolds the unmodified
            # producer's default headers, while the producer pins the production
            # layout; the measured cells must still be identical.
            pd.testing.assert_frame_equal(pd.read_excel(root / "dvh_metrics.xlsx"), workbook_cells, check_exact=True)
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


@pytest.mark.parametrize("entrypoint", ["current", "verify", "preflight", "recovery"])
def test_all_acceptance_paths_reject_changed_derived_mask(tmp_path, monkeypatch, entrypoint):
    root, closure, custom, expected = derived_generation(tmp_path)
    mask = Path(json.loads(closure.read_text())["mask"]["path"])
    stat = mask.stat()
    mask.write_bytes(mask.read_bytes() + b"\x00")
    os.utime(mask, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = gen.binding(root, gen.effective_settings(), custom)
    assert changed.inputs_sha256 != expected.inputs_sha256
    assert changed.code_sha256 == expected.code_sha256
    rejected = (root / ".dvh/current").resolve()
    if entrypoint in {"current", "verify"}:
        action = (lambda: gen.current(root, expected)) if entrypoint == "current" else (lambda: gen.verify(rejected, expected))
        with pytest.raises(DurableProgressError, match="DVH_MASK_PROVENANCE_INVALID"):
            action()
    elif entrypoint == "preflight":
        config = tmp_path / "config.yaml"
        config.write_text("ct_cropping:\n  use_cropped_for_dvh: true\n")
        result = gen.preflight(root.parents[1], config, custom)["courses"]
        assert result == [{"course": str(root), "reusable": False,
                           "reason_code": "DVH_BINDING_INPUTS_CHANGED"}]
    else:
        # Force an unchanged binding as a second defense: recovery itself must
        # validate provenance, even when a source was missed by an inventory.
        monkeypatch.setattr(gen, "binding", lambda *args, **kwargs: expected)
        computes = []
        def compute():
            computes.append(1)
            write_synthetic_products(root, pd.DataFrame([{"ROI_Name": "Recomputed", "D95Gy": 1.0}]))
        assert gen.run(root, gen.effective_settings(), custom, compute)
        assert computes == [1]
        assert (root / ".dvh/current").resolve() != rejected
        assert not any(event["event"] == "recovered_commit" for event in events(root))
        assert gen.verify(root / ".dvh/current", expected)


def write_synthetic_products(root, frame):
    parquet = gen.output_path(root, "dvh_metrics.parquet")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet, index=False)
    frame.to_excel(gen.output_path(root, "dvh_metrics.xlsx"), index=False)
    qc = gen.output_path(root, "metadata/dvh_qc.json")
    qc.parent.mkdir(parents=True, exist_ok=True)
    qc.write_text(json.dumps({"status": "ok"}))
    gen.output_path(root, "dvh_curves.json").write_text("{}")


def derived_generation(tmp_path, *, materialize_during_compute=False, external_attempt=False):
    import numpy as np
    import SimpleITK as sitk
    from rtpipeline.dvh_support import publish_derived_mask
    from rtpipeline.durable_progress import file_sha256
    root = course(tmp_path / "Output")
    image = sitk.Image([4, 4, 4], sitk.sitkUInt8)
    image += 1
    component = root / "Segmentation_TotalSegmentator/organ.nii.gz"
    component.parent.mkdir()
    sitk.WriteImage(image, str(component))
    ct = tmp_path / "ct.bin"
    ct.write_bytes(b"synthetic CT source")
    custom = tmp_path / "custom.yaml"
    custom.write_text("structures: []\n")
    def derive():
        return publish_derived_mask(root / "metadata/dvh_derived_masks", "Union",
            sitk.GetArrayFromImage(image), image, [component], custom,
            [{"name": "Union", "operation": "union", "sources": ["organ"]}],
            {"Union": {"status": "ok"}}, [ct])
    closure = None if materialize_during_compute else derive()
    def compute():
        fresh = derive()
        frame = pd.DataFrame([{"ROI_Name": "Union", "D95Gy": 1.0,
            "structure_provenance_type": "DERIVED_NIFTI_MASK",
            "structure_provenance_path": str(fresh),
            "structure_provenance_sha256": file_sha256(fresh)}])
        write_synthetic_products(root, frame)
    if external_attempt:
        from workflow.scripts import campaign_ledger
        campaign_ledger.begin_attempt(root.parents[1], root.parent.name, root.name, "dvh",
                                     binding=gen.binding(root, gen.effective_settings(), custom).as_dict())
    assert gen.run(root, gen.effective_settings(), custom, compute)
    return root, closure or next((root / "metadata/dvh_derived_masks").glob("*.json")), custom, gen.binding(root, gen.effective_settings(), custom)


@pytest.mark.parametrize("changed", ["mask", "closure"])
def test_derived_dependency_change_recomputes_and_binds_fresh_bytes(tmp_path, changed):
    root, closure, custom, expected = derived_generation(tmp_path)
    previous = (root / ".dvh/current").resolve()
    target = closure if changed == "closure" else Path(json.loads(closure.read_text())["mask"]["path"])
    stat = target.stat()
    target.write_bytes(target.read_bytes() + (b" " if changed == "closure" else b"\x00"))
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    from rtpipeline.durable_progress import file_sha256
    from rtpipeline.dvh_support import publish_derived_mask
    import SimpleITK as sitk
    record = json.loads(closure.read_text())
    component = Path(record["components"][0]["path"])
    image = sitk.ReadImage(str(component))
    computes = []
    def compute():
        computes.append(1)
        fresh = publish_derived_mask(closure.parent, "Union", sitk.GetArrayFromImage(image), image,
            [component], custom, record["definitions"], record["outcomes"],
            [item["path"] for item in record["ct_sources"]])
        write_synthetic_products(root, pd.DataFrame([{"ROI_Name": "Union", "D95Gy": 1.0,
            "structure_provenance_type": "DERIVED_NIFTI_MASK",
            "structure_provenance_path": str(fresh), "structure_provenance_sha256": file_sha256(fresh)}]))
    assert gen.run(root, gen.effective_settings(), custom, compute)
    assert computes == [1]
    assert (root / ".dvh/current").resolve() != previous
    assert any(e.get("reason_code") == "DVH_BINDING_INPUTS_CHANGED" for e in events(root))
    final = gen.binding(root, gen.effective_settings(), custom)
    assert gen.current(root, final)["binding"] == final.as_dict()
    assert gen.run(root, gen.effective_settings(), custom, compute)
    assert computes == [1]


@pytest.mark.parametrize("point", [None, "committed_before_activation"])
def test_derived_materialization_binds_generation_and_both_attempt_projections(tmp_path, monkeypatch, point):
    if point:
        def stop(name):
            if name == point:
                raise RuntimeError("synthetic controller loss")
        monkeypatch.setattr(gen, "_failpoint", stop)
        with pytest.raises(RuntimeError, match="controller loss"):
            derived_generation(tmp_path, materialize_during_compute=True, external_attempt=True)
        monkeypatch.setattr(gen, "_failpoint", lambda _: None)
        root = tmp_path / "Output/SYNTH_P/SYNTH_C"
        custom = tmp_path / "custom.yaml"
        closure = next((root / "metadata/dvh_derived_masks").glob("*.json"))
        expected = gen.binding(root, gen.effective_settings(), custom)
        assert gen.run(root, gen.effective_settings(), custom, lambda: pytest.fail("recomputed valid commit"))
    else:
        root, closure, custom, expected = derived_generation(tmp_path, materialize_during_compute=True, external_attempt=True)
    assert closure.is_file()
    assert gen.current(root, expected)["binding"] == expected.as_dict()
    assert any(e["event"] == "derived_inputs_finalized" for e in events(root))
    for attempt_path in (root / ".dvh/attempt.json",
            root.parents[1] / "_campaign_ledger/attempts/SYNTH_P__SYNTH_C__dvh.json"):
        attempt = json.loads(attempt_path.read_text())
        assert attempt["binding"] == expected.as_dict()
        assert attempt["state"] == "committed"
    assert gen.run(root, gen.effective_settings(), custom, lambda: pytest.fail("recomputed valid derived generation"))


def test_valid_commit_before_activation_is_recovered_without_compute(tmp_path, monkeypatch):
    root = course(tmp_path)
    def stop(point):
        if point == "committed_before_activation":
            raise RuntimeError("synthetic controller loss")
    monkeypatch.setattr(gen, "_failpoint", stop)
    with pytest.raises(RuntimeError, match="controller loss"):
        measure(root)
    assert not (root / ".dvh/current").exists()
    monkeypatch.setattr(gen, "_failpoint", lambda _: None)
    monkeypatch.setattr(dvh, "_compute_dvh_for_course", lambda *a, **k: pytest.fail("recomputed valid commit"))
    assert measure(root)
    assert any(e["event"] == "recovered_commit" for e in events(root))


@pytest.mark.parametrize("entrypoint", ["run", "preflight"])
def test_torn_journal_tail_recovers_without_recomputing_valid_generation(tmp_path, monkeypatch, entrypoint):
    root = course(tmp_path / "Output")
    assert measure(root)
    marker = gen.current(root)
    path = root / ".dvh/progress.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"interrupted":')
    monkeypatch.setattr(dvh, "_compute_dvh_for_course", lambda *a, **k: pytest.fail("recomputed valid commit"))
    if entrypoint == "run":
        assert measure(root)
    else:
        config = tmp_path / "config.yaml"
        config.write_text("ct_cropping:\n  use_cropped_for_dvh: true\n")
        assert gen.preflight(root.parents[1], config)["courses"] == [{"course": str(root), "reusable": True}]
    assert gen.current(root) == marker
    assert any(e.get("reason_code") == "DURABLE_JOURNAL_TAIL_RECOVERED" for e in events(root))
    assert list(path.parent.glob("progress.jsonl.torn-*"))


@pytest.mark.parametrize("change", ["other_rule", "trailing_comment", "dvh_rule", "parameters"])
def test_binding_scopes_rule_text_and_effective_parameters(tmp_path, monkeypatch, change):
    import shutil
    shadow = tmp_path / "source"
    shadow.mkdir()
    for name in ("rtpipeline", "workflow"):
        shutil.copytree(Path(gen.__file__).parent.parent / name, shadow / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    snake = shadow / "Snakefile"
    shutil.copy2(Path(gen.__file__).parent.parent / "Snakefile", snake)
    monkeypatch.setattr(gen, "__file__", str(shadow / "rtpipeline/dvh_generation.py"))
    monkeypatch.delenv("RTPIPELINE_DVH_RULE_PARAMETERS", raising=False)
    root = course(tmp_path / "fixture")
    before = gen.binding(root, gen.effective_settings())
    original = snake.read_text()
    if change == "other_rule":
        snake.write_text(original.replace("rule segmentation_course:", "rule segmentation_course:\n    # unrelated rule edit"))
    elif change == "trailing_comment":
        snake.write_text(original + "\n# unrelated radiomics comment\n")
    elif change == "dvh_rule":
        snake.write_text(original.replace("rule dvh_course:", "rule dvh_course:\n    # changed DVH rule"))
    else:
        monkeypatch.setenv("RTPIPELINE_DVH_RULE_PARAMETERS", json.dumps({"extra_args": "--max-total-dose-gy 80"}))
    after = gen.binding(root, gen.effective_settings())
    if change in {"other_rule", "trailing_comment"}:
        assert after == before
    elif change == "dvh_rule":
        assert after.code_sha256 != before.code_sha256
        assert after.config_sha256 == before.config_sha256
    else:
        assert after.config_sha256 != before.config_sha256
        assert after.code_sha256 == before.code_sha256


def test_legacy_upstream_is_routed_and_cannot_be_trusted_on_repeated_runs(tmp_path):
    root = course(tmp_path)
    (root / ".organized").write_text('{"status":"ok"}')
    for _ in range(2):
        with pytest.raises(DurableProgressError, match="DVH_UPSTREAM_RECOMPUTE_REQUIRED"):
            gen.run(root, gen.effective_settings(), None, lambda: pytest.fail("legacy upstream trusted"))
    assert json.loads((root / ".organized").read_text()) == {"status": "ok"}
    assert json.loads((root / ".dvh/upstream-recompute.json").read_text())["upstream"] == [".organized"]
    assert any(e.get("reason_code") == "DVH_UPSTREAM_LEGACY_UNBOUND" for e in events(root))


def test_current_rejection_excludes_that_candidate_for_the_whole_decision(tmp_path, monkeypatch):
    root = course(tmp_path)
    assert measure(root)
    rejected = (root / ".dvh/current").resolve()
    def reject(*args):
        raise DurableProgressError("DVH_MASK_PROVENANCE_INVALID")
    monkeypatch.setattr(gen, "current", reject)
    real_verify = gen.verify
    def verify(directory, expected=None):
        assert Path(directory).resolve() != rejected
        return real_verify(directory, expected)
    monkeypatch.setattr(gen, "verify", verify)
    assert measure(root)
    assert (root / ".dvh/current").resolve() != rejected
    assert not any(e["event"] == "recovered_commit" for e in events(root))


def test_preflight_routes_legacy_upstream_before_any_dvh_state_exists(tmp_path):
    root = course(tmp_path / "Output")
    receipt = root / ".segmentation_done"
    receipt.write_text('{"status":"disabled"}')
    assert not (root / ".dvh").exists()
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    assert gen.preflight(root.parents[1], config)["courses"] == [{
        "course": str(root), "reusable": False,
        "reason_code": "DVH_UPSTREAM_RECOMPUTE_REQUIRED"}]
    assert not receipt.exists()
    assert json.loads((root / ".dvh/upstream-recompute.json").read_text())["upstream"] == [".segmentation_done"]


@pytest.mark.parametrize("failed", ["failed\n", '{"status":"failed"}\n'])
def test_preflight_keeps_failed_upstream_closed_without_automatic_retry(tmp_path, failed):
    root = course(tmp_path / "Output")
    receipt = root / ".segmentation_done"
    receipt.write_text(failed)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    assert gen.preflight(root.parents[1], config)["courses"] == []
    assert receipt.read_text() == failed
    assert not (root / ".dvh").exists()


@pytest.mark.parametrize("entrypoint", ["run", "preflight", "input_manifest"])
@pytest.mark.parametrize("corruption", ['{"upstream":', '{}', '[]', '{"upstream":null}',
                                         '{"upstream":["../outside"]}', 'io_error'])
def test_corrupt_pending_upstream_is_quarantined_and_never_empty(tmp_path, monkeypatch, entrypoint, corruption):
    import errno
    import hashlib
    root = tmp_path / "Output/SYNTH/COURSE"
    (root / ".dvh").mkdir(parents=True)
    pending = root / ".dvh/upstream-recompute.json"
    raw = corruption.encode()
    pending.write_bytes(raw)
    config = tmp_path / "config.yaml"
    config.write_text('{}')
    original = Path.read_text
    if corruption == "io_error":
        def read(path, *args, **kwargs):
            if path == pending:
                raise OSError(errno.EIO, "synthetic pending read error")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", read)
    action = {"run": lambda: gen.run(root, {}, None, lambda: pytest.fail("computed with corrupt pending")),
              "preflight": lambda: gen.preflight(root.parents[1], config),
              "input_manifest": lambda: gen.input_manifest(root)}[entrypoint]
    for _ in range(2):
        with pytest.raises(DurableProgressError, match="DVH_UPSTREAM_PENDING_CORRUPT"):
            action()
    assert pending.read_bytes() == raw
    digest = hashlib.sha256(raw).hexdigest()
    quarantine = pending.with_name(pending.name + ".corrupt-" + digest)
    assert quarantine.read_bytes() == raw
    evidence = json.loads(original(quarantine.with_name(quarantine.name + ".json")))
    assert evidence["sha256"] == digest and evidence["bytes"] == len(raw)
    assert events(root)[-1]["reason_code"] == "DVH_UPSTREAM_PENDING_CORRUPT"


def test_unreadable_pending_stays_closed_when_quarantine_cannot_read(tmp_path, monkeypatch):
    import errno
    root = tmp_path / "SYNTH/COURSE"
    (root / ".dvh").mkdir(parents=True)
    pending = root / ".dvh/upstream-recompute.json"
    pending.write_bytes(b"synthetic unreadable evidence")
    for method in ("read_text", "read_bytes"):
        original = getattr(Path, method)
        def read(path, *args, _original=original, **kwargs):
            if path == pending:
                raise OSError(errno.EACCES, "synthetic unreadable pending")
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, method, read)
    with pytest.raises(DurableProgressError, match="DVH_UPSTREAM_PENDING_CORRUPT"):
        gen.input_manifest(root)
    assert pending.is_file()
    assert events(root)[-1]["quarantine_reason_code"] == "DVH_UPSTREAM_QUARANTINE_FAILED"


@pytest.mark.parametrize("entrypoint", ["run_without_generation", "run_with_generation", "preflight"])
def test_corrupt_local_attempt_has_fixed_failure_projection(tmp_path, entrypoint):
    root = course(tmp_path / "Output")
    if entrypoint != "run_without_generation":
        assert measure(root)
    else:
        (root / ".dvh").mkdir()
    attempt = root / ".dvh/attempt.json"
    attempt.write_text('{"state":')
    if entrypoint == "preflight":
        config = tmp_path / "config.yaml"
        config.write_text('{}')
        assert gen.preflight(root.parents[1], config)["courses"][0]["reason_code"] == "DVH_ATTEMPT_CORRUPT"
    else:
        with pytest.raises(DurableProgressError, match="DVH_ATTEMPT_CORRUPT"):
            measure(root)
    failed = json.loads(attempt.read_text())
    assert failed["state"] == "failed" and failed["reason_code"] == "DVH_ATTEMPT_CORRUPT"


@pytest.mark.parametrize("change", ["release", "system", "python_patch", "machine", "implementation", "python_minor", "libc", "numpy", "pyarrow"])
def test_binding_environment_uses_only_deciding_runtime_fields(tmp_path, monkeypatch, change):
    import copy
    from rtpipeline import stage_completion
    original = stage_completion.environment_record(gen._DVH_ENVIRONMENT_PACKAGES)
    record = copy.deepcopy(original)
    monkeypatch.setattr(stage_completion, "environment_record", lambda _: record)
    root = tmp_path / "course"
    root.mkdir()
    before = gen.binding(root, gen.effective_settings())
    if change in {"numpy", "pyarrow"}:
        record["packages"][change] += ".changed"
    elif change in {"python_patch", "python_minor"}:
        parts = record["python"].split('.')
        index = 2 if change == "python_patch" else 1
        parts[index] = str(int(parts[index]) + 1)
        record["python"] = '.'.join(parts)
    else:
        record[change] = ["other-libc", "1.0"] if change == "libc" else "other-value"
    after = gen.binding(root, gen.effective_settings())
    if change in {"release", "system", "python_patch"}:
        assert after == before
    else:
        assert after.config_sha256 != before.config_sha256
        assert after.code_sha256 == before.code_sha256
    projected = gen.dvh_environment_record()
    assert set(projected) == {"python", "implementation", "machine", "libc", "packages"}


@pytest.mark.parametrize("field", ["python", "python_bin", "root_dir", "configfile", "dicom_root", "output_dir", "logs_dir", "custom_structures", "campaign_mode"])
def test_binding_ignores_workflow_locations_and_campaign_policy(tmp_path, monkeypatch, field):
    root = tmp_path / "course"
    root.mkdir()
    parameters = {"stage": "dvh", field: False if field == "campaign_mode" else "/old/checkout/location"}
    monkeypatch.setenv("RTPIPELINE_DVH_RULE_PARAMETERS", json.dumps(parameters))
    before = gen.binding(root, gen.effective_settings())
    parameters[field] = True if field == "campaign_mode" else "/moved/checkout/location"
    monkeypatch.setenv("RTPIPELINE_DVH_RULE_PARAMETERS", json.dumps(parameters))
    assert gen.binding(root, gen.effective_settings()) == before


def test_workflow_file_override_binds_content_and_deciding_override_changes(tmp_path, monkeypatch):
    one, two = tmp_path / "one.yaml", tmp_path / "two.yaml"
    one.write_text("structures: []")
    two.write_bytes(one.read_bytes())
    root = tmp_path / "course"
    root.mkdir()
    def bound(path, limit):
        monkeypatch.setenv("RTPIPELINE_DVH_RULE_PARAMETERS", json.dumps({"extra_args":
            f"--custom-structures {path} --max-total-dose-gy {limit}"}))
        return gen.binding(root, gen.effective_settings(), path)
    before = bound(one, 80)
    assert bound(two, 80) == before
    assert bound(two, 90).config_sha256 != before.config_sha256
    two.write_text("structures: [changed]")
    assert bound(two, 80).config_sha256 != before.config_sha256


def test_legacy_checker_uses_explicit_nondefault_caller_settings(tmp_path, monkeypatch):
    import errno
    from rtpipeline import durable_progress
    root = course(tmp_path)
    metadata = root / "metadata/case_metadata.json"
    contract = json.loads(metadata.read_text())
    contract["course_contract"]["dose_qc"]["threshold_gy"] = 80.0
    metadata.write_text(json.dumps(contract))
    assert measure(root, use_cropped=False, max_total_dose_gy=80.0)
    settings = gen.effective_settings(use_cropped=False, max_total_dose_gy=80.0)
    assert dvh._is_dvh_up_to_date(root, root / "dvh_metrics.xlsx", settings=settings)
    assert not dvh._is_dvh_up_to_date(root, root / "dvh_metrics.xlsx")
    original_hash = durable_progress.file_sha256
    def unreadable(path):
        if Path(path).name == "dvh.py":
            raise OSError(errno.EIO, "synthetic unreadable code identity")
        return original_hash(path)
    with monkeypatch.context() as patch:
        patch.setattr(durable_progress, "file_sha256", unreadable)
        assert not dvh._is_dvh_up_to_date(root, root / "dvh_metrics.xlsx", settings=settings)
    monkeypatch.setattr(dvh, "staging", lambda: False)
    def checked(course_dir, output, custom, *, settings):
        assert settings == gen.effective_settings(False, False, 42.0, 80.0)
        return True
    monkeypatch.setattr(dvh, "_is_dvh_up_to_date", checked)
    assert dvh._compute_dvh_for_course(root, use_cropped=False, rx_dose_gy=42.0,
                                       max_total_dose_gy=80.0) == root / "dvh_metrics.xlsx"


def test_moving_identical_checkout_preserves_the_whole_binding(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / "course"
    root.mkdir()
    before = gen.binding(root, gen.effective_settings())
    relocated = tmp_path / "moved-checkout"
    relocated.mkdir()
    repository = Path(gen.__file__).parent.parent
    for name in ("rtpipeline", "workflow"):
        shutil.copytree(repository / name, relocated / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(repository / "Snakefile", relocated / "Snakefile")
    monkeypatch.setattr(gen, "__file__", str(relocated / "rtpipeline/dvh_generation.py"))
    assert gen.binding(root, gen.effective_settings()) == before
