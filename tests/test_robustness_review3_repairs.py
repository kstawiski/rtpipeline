"""Concrete synthetic regressions for NEW-1 through NEW-8."""
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
import yaml

from rtpipeline import cli, radiomics_robustness as rr, robustness_completion as rc
from rtpipeline.radiomics_ct_contract import classify_ct_roi, load_custom_structure_provenance
from test_robustness_producer_equivalence import (
    _layout, _produce, _custom_metadata, _canonical, main_producer, one_itk_thread,  # noqa: F401
)
from test_robustness_consumer_integration import _build_courses, _run_course


def _active_admit(course, config):
    kwargs = {"custom_structures_config": config} if "custom_structures_config" in inspect.signature(rr.admit_robustness_cohort_course).parameters else {}
    return rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True), **kwargs)


def _main_run(main_producer, course, config, monkeypatch):
    main, completion = main_producer
    for name in ("load_course_contract", "_load_main_ct_identity_catalog", "extract_features_for_masks"):
        monkeypatch.setattr(main, name, getattr(rr, name))
    return _produce(main, completion, course, config)


def test_new_1_shipped_config_default_pelvic_helper(tmp_path, monkeypatch):
    course, config, contract, catalog = _layout(tmp_path, monkeypatch, "default_pelvic")
    # Selected helper is absent from main's publication, as the real classifier requires.
    ds = pydicom.dcmread(course / "RS.dcm")
    ds.StructureSetROISequence[0].ROIName = "ODBYTNICA"
    ds.save_as(course / "RS.dcm")
    (course / "RS_auto.dcm").unlink()
    for path in (course / "Segmentation_TotalSegmentator").glob("*.nii.gz"):
        path.unlink()
    catalog.clear()
    custom_ds = pydicom.dcmread(course / "RS_custom.dcm")
    custom_ds.StructureSetROISequence = [r for r in custom_ds.StructureSetROISequence if r.ROIName == "pelvic_bones_3mm"]
    keep = {r.ROINumber for r in custom_ds.StructureSetROISequence}
    custom_ds.ROIContourSequence = [r for r in custom_ds.ROIContourSequence if r.ReferencedROINumber in keep]
    custom_ds.RTROIObservationsSequence = [r for r in custom_ds.RTROIObservationsSequence if r.ReferencedROINumber in keep]
    custom_ds.save_as(course / "RS_custom.dcm")
    _custom_metadata(course, config.custom_structures_config, contract, monkeypatch)
    shipped = yaml.safe_load((Path(rr.__file__).parent.parent / "config.yaml").read_text())
    assert "custom_structures" not in shipped
    shipped["radiomics_robustness"] = {"enabled": True, "segmentation_perturbation": {"apply_to_structures": ["pelvic_bones_3mm"]}}
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(shipped))
    seen = []
    real_run = rr.run_robustness_course
    def run(active, *args, **kwargs):
        seen.append(active.custom_structures_config)
        return real_run(active, *args, **kwargs)
    monkeypatch.setattr(rr, "run_robustness_course", run)
    provenance = load_custom_structure_provenance(config.custom_structures_config)
    assert classify_ct_roi("Custom", "pelvic_bones_3mm", custom_provenance=provenance).feature_publication_policy == "inventory_only_no_features"
    assert cli.main(["radiomics-robustness", "--course-dir", str(course), "--config", str(cfg), "--output", str(course / "radiomics_robustness_ct.parquet")]) == 0
    assert seen == [config.custom_structures_config.resolve()]
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert payload["measurement_outcome"] == rr.ROBUSTNESS_NO_TARGET_OUTCOME
    assert payload["source_only_basis"]["unresolved_identity_count"] == 0
    assert payload["source_only_basis"]["missed_targets"] == []


@pytest.mark.parametrize("kind", ["nifti_ct", "fallback"])
@pytest.mark.parametrize("name,mode,selection,selected", [
    ("HR-CTV", "patterns", ["GTV*"], False),
    ("HR-CTV", "targets", ["GTV*"], True),
    ("GTV", "patterns", ["GTV*"], True),
    ("urinary_bladder", "patterns", ["urinary_bladder"], True),
    ("PTV1-poprzednio", "patterns", ["PTV1-poprzednio"], True),
])
def test_new_2_off_grid_voxel_presence(tmp_path, monkeypatch, kind, name, mode, selection, selected):
    course, config, _, catalog = _layout(tmp_path, monkeypatch, kind, missing_identity=True)
    for path in (course / "Segmentation_TotalSegmentator").glob("*.nii.gz"):
        path.unlink()
    if (course / "RS.dcm").exists():
        ds = pydicom.dcmread(course / "RS.dcm")
        ds.StructureSetROISequence[0].ROIName = "ODBYTNICA"
        ds.save_as(course / "RS.dcm")
    catalog.clear()
    path = course / f"Segmentation_CustomModels/M/{name}.nii.gz"
    path.parent.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.ones((3, 3, 3), np.uint8))
    image.SetOrigin((5000, 5000, 5000))
    sitk.WriteImage(image, str(path))
    cfg = rr.RobustnessConfig(enabled=True)
    cfg.target_selection = mode
    cfg.perturbation.apply_to_structures = selection
    result = rr.run_robustness_course(config, cfg, course, output_path=course / "radiomics_robustness_ct.parquet")
    publishable = rr.classify_robustness_roi("CustomModel:M", name)["in_publication_base"]
    assert result.measurement_outcome == (rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME if selected and publishable else rr.ROBUSTNESS_NO_TARGET_OUTCOME)
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    roi = next(r for r in payload["target_inventory"] if r["source_path"] == str(path))
    assert roi["volumetric"] is False
    assert payload["source_only_basis"].get("missed_targets", []) == []
    assert any(r["source_path"] == str(path) and r["structural_code"] == "NIFTI_EMPTY_ON_PLANNING_CT_GRID" for r in payload["rows"]) == publishable
    rr._read_robustness_source_dispositions(course, run_identifier=payload["robustness_run_identifier"], rob_config=cfg, output_name=payload["output_path"])
    if publishable:
        # Even a self-consistent byte binding cannot claim grid emptiness
        # after the mask occupies the planning CT. Recheck the actual grid.
        image.SetOrigin((0, 0, 0))
        sitk.WriteImage(image, str(path))
        for binding in payload["voxel_source_bindings"]:
            if binding["source_path"] == str(path):
                binding["sha256"] = rr._file_sha256(path)
        payload["target_inventory"] = rr._robustness_bound_inventory(
            payload["source_bindings"], payload["voxel_source_bindings"],
            configuration=payload["effective_configuration"], planning_image=rr._planning_robustness_image(course),
        )
        with pytest.raises(ValueError, match="empty-on-grid disposition differs"):
            rr._read_robustness_source_dispositions(
                course, run_identifier=payload["robustness_run_identifier"], rob_config=cfg,
                output_name=payload["output_path"], snapshot=json.dumps(payload).encode(),
            )


@pytest.mark.parametrize("custom", [False, True])
def test_new_3_active_custom_configuration_refuses_historical_receipt(tmp_path, monkeypatch, main_producer, custom):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "custom" if custom else "normal")
    _main_run(main_producer, course, config, monkeypatch)
    active = tmp_path / "active.yaml"
    active.write_text("custom_structures:\n  - name: GTV_union\n    operation: union\n    source_structures: [PTV]\n")
    with pytest.raises((ValueError, RuntimeError, rc.RobustnessCompletionError)):
        _active_admit(course, active)


def test_new_3_robustness_digest_mismatch_with_older_configuration(tmp_path, monkeypatch, main_producer):
    from rtpipeline import custom_structures_rtstruct as custom
    course, config, contract, _ = _layout(tmp_path, monkeypatch, "default_pelvic")
    active = tmp_path / "active.yaml"
    active.write_bytes(config.custom_structures_config.read_bytes())
    os.utime(active, ns=(1_000_000_000, 1_000_000_000))
    config.custom_structures_config = active
    historical = _main_run(main_producer, course, config, monkeypatch)
    current = rr._current_robustness_code_identity()
    assert rr._compatible_main_measured_code(historical[1], current, course_dir=course, custom_structures_config=active)
    active.write_bytes(active.read_bytes() + b"\n# changed content\n")
    os.utime(active, ns=(1_000_000_000, 1_000_000_000))
    args = (course / "RS_custom.dcm", active, contract.authoritative_rtstruct_path, course / "RS_auto.dcm")
    assert active.stat().st_mtime_ns < args[0].stat().st_mtime_ns
    check = getattr(rr, "_is_robustness_rs_custom_stale", custom._is_rs_custom_stale)
    assert check(*args)
    assert not rr._compatible_main_measured_code(historical[1], current, course_dir=course, custom_structures_config=active)
    with pytest.raises((ValueError, RuntimeError, rc.RobustnessCompletionError)):
        _active_admit(course, active)
    with pytest.raises((ValueError, RuntimeError), match="RS_custom.*stale"):
        _produce(rr, rc, course, config)
    # Main retains its mtime-only configuration check on this branch.
    assert not custom._is_rs_custom_stale(*args)


@pytest.mark.parametrize("newer", [False, True])
def test_new_3_robustness_without_digest_defers_to_shared_check(tmp_path, monkeypatch, newer):
    from rtpipeline import custom_structures_rtstruct as custom
    course, config, contract, _ = _layout(tmp_path, monkeypatch, "default_pelvic")
    active = tmp_path / "active.yaml"
    active.write_bytes(config.custom_structures_config.read_bytes() + b"\n# changed content\n")
    custom_path = course / "RS_custom.dcm"
    stamp = custom_path.stat().st_mtime_ns + 1_000_000_000 if newer else 1_000_000_000
    os.utime(active, ns=(stamp, stamp))
    meta = course / "metadata/rs_custom_meta.json"
    payload = json.loads(meta.read_text())
    payload.pop("custom_config_sha256")
    meta.write_text(json.dumps(payload))
    original = custom._is_rs_custom_stale
    calls = []
    def shared(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(custom, "_is_rs_custom_stale", shared)
    args = (custom_path, active, contract.authoritative_rtstruct_path, course / "RS_auto.dcm")
    check = getattr(rr, "_is_robustness_rs_custom_stale", shared)
    assert check(*args) is newer
    assert calls == [args]


def test_new_4_reader_tolerates_new_unreadable_unselected_model(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "benign_voxel_model")
    produced = _produce(rr, rc, course, config)
    assert "voxel_source_bindings" not in produced[1]
    (course / "Segmentation_CustomModels/M/spleen.nii.gz").write_bytes(b"unreadable synthetic mask")
    assert _active_admit(course, None).measured
    after = _produce(rr, rc, course, config)
    assert ("CustomModel:M", "spleen") in [(r["segmentation_source"], r["roi_name"]) for r in after[1]["tolerated_source_failures"]]


def test_new_4_certificate_unreadable_benign_model(tmp_path, monkeypatch, main_producer):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "benign_voxel_model")
    (course / "Segmentation_CustomModels/M/colon.nii.gz").write_bytes(b"unreadable synthetic mask")
    before = _main_run(main_producer, course, config, monkeypatch)
    assert _active_admit(course, None).measured
    after = _produce(rr, rc, course, config)
    pd.testing.assert_frame_equal(_canonical(before)[0], _canonical(after)[0], check_exact=True)
    assert after[1]["tolerated_source_failures"]
    print({"layout": "unreadable_benign_voxel_model", "rows": len(after[0]),
           "comparison": "equal measurements; tolerated inventory failure recorded", "reuse": True})


@pytest.mark.parametrize("broken", ["RS.dcm", "RS_auto.dcm", "contract"])
def test_new_5_campaign_failure_receipt_and_ledger(tmp_path, monkeypatch, broken):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured", ("P2", "C1"): "measured"})
    course = c.course_dir("P1")
    if broken == "contract":
        from rtpipeline.course_contract import load_course_contract
        (course / "metadata").mkdir(exist_ok=True)
        (course / "metadata/case_metadata.json").write_text("{invalid json")
        original = rr.load_course_contract
        def corrupt(directory):
            return load_course_contract(directory) if Path(directory) == course else original(directory)
        monkeypatch.setattr(rr, "load_course_contract", corrupt)
    else:
        (course / broken).write_bytes(b"truncated synthetic DICOM")
        if broken == "RS_auto.dcm":
            from rtpipeline import radiomics as rm
            monkeypatch.setattr(rm, "_standard_rtstruct_sources", lambda contract, directory: [
                (label, directory / name, None) for label, name in (("Manual", "RS.dcm"), ("AutoRTS_total", "RS_auto.dcm"))
                if (directory / name).exists()])
    assert _run_course(c, "P1", campaign=True) == 1
    receipt = rc.read_robustness_completion_sentinel(c.sentinel("P1"), course_dir=course)
    assert receipt.measurement_outcome == rr.ROBUSTNESS_FAILED_OUTCOME
    sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert sidecar["source_errors"]
    rr._read_robustness_source_dispositions(course, run_identifier=receipt.run_identifier, rob_config=None, output_name=receipt.output_name)
    command = [sys.executable, str(Path(rr.__file__).parent.parent / "workflow/scripts/campaign_ledger.py"),
               "close-robustness-failed", "--output-dir", str(c.output_root), "--patient", "P1",
               "--course", "C1", "--sentinel", str(c.sentinel("P1"))]
    environment = {**os.environ, "PYTHONPATH": str(Path(rr.__file__).parent.parent)}
    closure = subprocess.run(command, text=True, capture_output=True, env=environment)
    assert closure.returncode == 0, closure.stdout + closure.stderr
    records = [json.loads(p.read_text()) for p in (c.output_root / "_campaign_ledger/records").rglob("*.json")]
    assert records and records[0]["status"] == "failed"
    # A later course in the same campaign completes after failed closure.
    assert _run_course(c, "P2", campaign=True) == 0
    assert c.receipt("P2")["measurement_outcome"] == "measured"
    complete = subprocess.run([
        command[0], command[1], "close-robustness-complete", "--output-dir", str(c.output_root),
        "--patient", "P2", "--course", "C1", "--sentinel", str(c.sentinel("P2")),
    ], text=True, capture_output=True, env=environment)
    assert complete.returncode == 0, complete.stdout + complete.stderr
    records = [json.loads(p.read_text()) for p in (c.output_root / "_campaign_ledger/records").rglob("*.json")]
    assert sorted(record["status"] for record in records) == ["failed", "ok"]
    # Failure error records are validated, rather than accepted as free text.
    sidecar["source_errors"][0]["reason"] = "tampered"
    rr.robustness_source_dispositions_path(course).write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="failure source errors"):
        rr._read_robustness_source_dispositions(course, run_identifier=receipt.run_identifier, rob_config=None, output_name=receipt.output_name)


@pytest.mark.parametrize("source", ["AutoRTS_total", "CustomModel:M"])
@pytest.mark.parametrize("ct_kind", ["nifti_ct", "normal"])
def test_new_6_deduplicate_rtstruct_and_voxel_inventory(tmp_path, monkeypatch, source, ct_kind):
    course, config, _, catalog = _layout(tmp_path, monkeypatch, ct_kind, missing_identity=True)
    for path in (course / "Segmentation_TotalSegmentator").glob("*.nii.gz"):
        path.unlink()
    ds = pydicom.dcmread(course / "RS.dcm")
    ds.StructureSetROISequence[0].ROIName = "HR-CTV"
    rtpath = course / "RS_auto.dcm" if source == "AutoRTS_total" else course / "Segmentation_CustomModels/M/rtstruct.dcm"
    rtpath.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(rtpath)
    ds.StructureSetROISequence[0].ROIName = "ODBYTNICA"
    ds.save_as(course / "RS.dcm")
    path = course / ("Segmentation_TotalSegmentator/total--HR-CTV.nii.gz" if source == "AutoRTS_total" else "Segmentation_CustomModels/M/HR-CTV.nii.gz")
    from rtpipeline.radiomics import _load_series_image
    image = sitk.ReadImage(str(course / "ct.nii.gz")) if ct_kind == "nifti_ct" else _load_series_image(course / "CT")
    mask = sitk.GetImageFromArray(np.ones(sitk.GetArrayFromImage(image).shape, np.uint8))
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(path))
    catalog.clear()
    cfg = rr.RobustnessConfig(enabled=True)
    cfg.perturbation.apply_to_structures = ["GTV*"]
    assert rr.run_robustness_course(config, cfg, course, output_path=course / "radiomics_robustness_ct.parquet").measurement_outcome == rr.ROBUSTNESS_SELECTION_GAP_OUTCOME
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    for rows in (payload["target_inventory"], payload["source_only_basis"]["missed_targets"]):
        assert [(r["segmentation_source"], r["roi_name"]) for r in rows].count((source, "HR-CTV")) == 1
    assert str(rtpath) in [b["source_path"] for b in payload["source_bindings"]]
    assert str(path) in [b["source_path"] for b in payload["voxel_source_bindings"]]


def test_new_7_positive_custom_certificate_real_staleness(tmp_path, monkeypatch, main_producer):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline import custom_structures_rtstruct as custom
    from rtpipeline.course_contract import _ct_provenance, load_course_contract
    from rtpipeline.radiomics_ct_contract import (
        disposition_rows_for_arms, _feature_schema_metadata, write_ct_publication_atomic,
    )
    from rasterizer_test_utils import PINNED_RASTERIZER_ROW
    from rtpipeline import radiomics_ct_contract, radiomics_rasterizer, auto_rtstruct
    real_catalog_loader = rr._load_main_ct_identity_catalog
    real_extractor = rr.extract_features_for_masks
    real_columns = radiomics_ct_contract._rasterizer_columns
    real_runtime_guard = radiomics_rasterizer.require_radiomics_rasterizer
    real_auto_currency = auto_rtstruct._derived_rtstruct_dependencies_are_current
    course, config, _, catalog = _layout(tmp_path, monkeypatch, "default_pelvic")
    (course / "RS_auto.dcm").unlink()  # Certificate path needs no auto-currency substitute.
    metadata_path = write_minimal_course_contract(course, planning_ct_dir=course / "CT", authoritative_rtstruct=course / "RS.dcm")
    metadata = json.loads(metadata_path.read_text())
    provenance = metadata["course_contract"]["planning_ct"]["nifti_provenance"]
    provenance.update(_ct_provenance(course / "CT"))
    sidecar_path = course / provenance["sidecar_path"]
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.update(_ct_provenance(course / "CT"))
    sidecar_path.write_text(json.dumps(sidecar))
    metadata_path.write_text(json.dumps(metadata))
    contract = load_course_contract(course)
    _custom_metadata(course, config.custom_structures_config, contract, monkeypatch)
    monkeypatch.setattr(rr, "load_course_contract", load_course_contract)
    monkeypatch.setattr(custom, "load_course_contract", load_course_contract)
    monkeypatch.setattr(rr, "_load_main_ct_identity_catalog", real_catalog_loader)
    custom_provenance = load_custom_structure_provenance(config.custom_structures_config)
    records = []
    for key in (("Manual", "PTV"), ("Custom", "GTV")):
        identity = catalog[key].as_dict()
        records.extend(disposition_rows_for_arms(
            {**identity, "modality": "CT", "roi_name": key[1]},
            decision=classify_ct_roi(*key, custom_provenance=custom_provenance),
            disposition="success", detail="synthetic publication", failure_kind="none",
            run_identifier="synthetic-main", code_revision="synthetic",
            native_voxel_count=120, required=True,
            configured_parameter_hashes={"primary_resegmented": "primary", "sensitivity_raw": "raw"},
            effective_hashes={"primary_resegmented": "effective-primary", "sensitivity_raw": "effective-raw"},
        ))
    for record in records:
        record.update(PINNED_RASTERIZER_ROW)
        record["original_firstorder_Mean"] = 1.0
        record.update(_feature_schema_metadata({"original_firstorder_Mean"}))
    write_ct_publication_atomic(pd.DataFrame(records), course / "radiomics_ct.xlsx")
    # Contract loading, source enumeration, publication reading and staleness
    # validation are real on the entire certificate/admission path.
    result = _main_run(main_producer, course, config, monkeypatch)
    monkeypatch.setattr(rr, "extract_features_for_masks", real_extractor)
    monkeypatch.setattr(radiomics_ct_contract, "_rasterizer_columns", real_columns)
    monkeypatch.setattr(radiomics_rasterizer, "require_radiomics_rasterizer", real_runtime_guard)
    monkeypatch.setattr(auto_rtstruct, "_derived_rtstruct_dependencies_are_current", real_auto_currency)
    kwargs = {"custom_structures_config": config.custom_structures_config} if "custom_structures_config" in inspect.signature(rr._compatible_main_measured_code).parameters else {}
    assert rr._compatible_main_measured_code(result[1], rr._current_robustness_code_identity(), course_dir=course, **kwargs)
    assert _active_admit(course, config.custom_structures_config).measured

    # The same artifact is refused by the complete archived 4ad1c53 reader.
    from types import ModuleType
    baseline_dir = tmp_path / "baseline-4ad1c53"
    baseline_dir.mkdir()
    for relative in (*rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES, "robustness_main_compatibility.json"):
        (baseline_dir / relative).write_bytes(subprocess.check_output(
            ["git", "show", f"4ad1c53:rtpipeline/{relative}"], cwd=Path(rr.__file__).parent.parent,
        ))
    baseline = ModuleType("rtpipeline._s11g_baseline_reader")
    baseline.__file__ = str(baseline_dir / "radiomics_robustness.py")
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    exec(compile(Path(baseline.__file__).read_bytes(), baseline.__file__, "exec"), baseline.__dict__)
    assert not baseline._compatible_main_measured_code(
        result[1], baseline._current_robustness_code_identity(), course_dir=course,
    )

def test_new_8_custom_selected_uses_custom_classification(tmp_path, monkeypatch):
    course, config, _, catalog = _layout(tmp_path, monkeypatch)
    catalog[("Custom", "GTV_union")] = next(iter(catalog.values()))
    seen = []
    original = rr.robustness_selection_matches
    def selection(name, *args, **kwargs):
        if name == "GTV_union":
            seen.append(kwargs)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(rr, "robustness_selection_matches", selection)
    monkeypatch.setattr("rtpipeline.custom_structures_rtstruct._is_rs_custom_stale", lambda *a: False)
    _produce(rr, rc, course, config)
    assert seen and seen[0].get("segmentation_source") == "Custom"
    assert "custom_provenance" in seen[0]


def _new_8_aggregate_gate_dry_run(tmp_path):
    """Execute the shipped aggregate rule's real input/output/params declarations."""
    source = (Path(rr.__file__).parent.parent / "Snakefile").read_text()
    rule = source[source.index("rule aggregate_radiomics_robustness:"):]
    rule = rule[:rule.index("    shell:")]
    # Replace the expensive aggregation command only; scheduling uses the
    # shipped rule, parameters and Snakemake's persisted execution metadata.
    rule += '    shell: "touch {output.summary}"\n'
    sentinel = tmp_path / "P1/C1/.radiomics_robustness_done"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("synthetic receipt")
    for name in ("manifest.tsv", "dependency.json"):
        (tmp_path / name).write_text("synthetic scheduling input")
    environment = tmp_path / "envs/rtpipeline.yaml"
    environment.parent.mkdir()
    environment.write_text("channels: [conda-forge]\ndependencies: [python]\n")
    snake = tmp_path / "dag.smk"
    preamble = '''from pathlib import Path
ROOT_DIR = Path.cwd()
RESULTS_DIR = ROOT_DIR / "Results"
LOGS_DIR = ROOT_DIR / "Logs"
AGG_OUTPUTS = {}
_manifest_input = "manifest.tsv"
ROBUSTNESS_CONFIG_DEPENDENCY = "dependency.json"
AGG_THREADS_RESERVED = 1
EFFECTIVE_CONFIGFILE = "config.yaml"
ROBUSTNESS_ENABLED = True
CAMPAIGN_MODE = True
CAMPAIGN_REQUIRE_ALL_COURSES = False
CAMPAIGN_MIN_COMPLETION_FRACTION = float(config.get("gate", 0.5))
PYTHON_MAIN = "python"
PYTHON_MAIN_BIN = "/usr/bin"
def _per_course_sentinels(name):
    return ["P1/C1/" + name]
rule radiomics_robustness_course:
    output: "P1/C1/.radiomics_robustness_done"
    shell: "touch {output}"
'''
    snake.write_text(preamble + rule)
    command = [sys.executable, "-m", "snakemake", "-s", str(snake), "--directory", str(tmp_path), "--cores", "1", "--nolock", "aggregate_radiomics_robustness"]
    run = subprocess.run(command, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    dry = subprocess.run([*command, "-n", "-p", "--config", "gate=0.9"], text=True, capture_output=True)
    output = dry.stdout + dry.stderr
    assert dry.returncode == 0, output
    assert "rule aggregate_radiomics_robustness:" in output
    assert "rule radiomics_robustness_course:" not in output
    assert "Params have changed" in output or "params have changed" in output
    return output


def test_new_8_aggregate_gate_snakemake_dry_run(tmp_path):
    _new_8_aggregate_gate_dry_run(tmp_path)




@pytest.mark.parametrize("mode", ["inputs", "manifest"])
def test_new_3_aggregate_cli_enforces_active_custom_configuration(tmp_path, monkeypatch, main_producer, mode):
    from types import SimpleNamespace
    from test_robustness_consumer_integration import _write_manifest
    course, config, _, _ = _layout(tmp_path, monkeypatch)
    _main_run(main_producer, course, config, monkeypatch)
    active = tmp_path / "active.yaml"
    active.write_text("custom_structures:\n  - name: GTV_union\n    operation: union\n    source_structures: [PTV]\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"custom_structures": str(active), "radiomics_robustness": {"enabled": True}}))
    output = tmp_path / "summary.xlsx"
    arguments = ["radiomics-robustness-aggregate", "--config", str(cfg), "--output", str(output)]
    if mode == "inputs":
        arguments += ["--inputs", str(course / "radiomics_robustness_ct.parquet")]
    else:
        manifest = _write_manifest(SimpleNamespace(output_root=tmp_path, courses={("P1", "C1"): course}))
        arguments += ["--manifest", str(manifest), "--output-root", str(tmp_path)]
    assert cli.main(arguments) == 1
    assert not output.exists()
    assert not output.with_name("summary_raw_values.parquet").exists()


@pytest.mark.parametrize("name", ["GTV_margin", "HR_CTV_margin"])
def test_new_1_unmapped_margin_does_not_override_governed_fallback(tmp_path, name):
    """The review's broader margin-only claim is outside the governed crosswalk."""
    path = tmp_path / "custom.yaml"
    path.write_text(f"custom_structures:\n  - name: {name}\n    operation: union\n    source_structures: [PTV]\n    margin: 3\n")
    decision = classify_ct_roi("Custom", name, custom_provenance=load_custom_structure_provenance(path))
    assert decision.roi_class == "unresolved_mixed"
    assert decision.feature_publication_policy == "extract"
    assert rr.classify_robustness_roi("Custom", name, custom_provenance=load_custom_structure_provenance(path))["detector_rules"] == ["name_token"]


@pytest.mark.parametrize("command", ["radiomics-robustness", "radiomics-robustness-aggregate"])
def test_new_1_merged_config_keeps_main_cli_relative_path_semantics(tmp_path, monkeypatch, command):
    from rtpipeline import radiomics_robustness_aggregate as aggregate
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "custom.yaml"
    expected.write_text("custom_structures: []\n")
    nested = tmp_path / "Logs/_workflow"
    nested.mkdir(parents=True)
    (nested / "custom.yaml").write_text("custom_structures: []\n")
    config = nested / "effective_config.yaml"
    config.write_text(yaml.safe_dump({"custom_structures": "custom.yaml", "radiomics_robustness": {"enabled": True}}))
    seen = []
    course = tmp_path / "P1/C1"
    course.mkdir(parents=True)
    if command == "radiomics-robustness":
        def capture(active, *args, **kwargs):
            seen.append(active.custom_structures_config)
            raise RuntimeError("configuration routing probe")
        monkeypatch.setattr(rr, "run_robustness_course", capture)
        arguments = [command, "--course-dir", str(course), "--output", str(course / "radiomics_robustness_ct.parquet")]
    else:
        def capture(*args, **kwargs):
            seen.append(kwargs.get("custom_structures_config"))
            raise RuntimeError("configuration routing probe")
        monkeypatch.setattr(aggregate, "aggregate_robustness_results", capture)
        arguments = [command, "--inputs", str(course / "radiomics_robustness_ct.parquet"), "--output", str(tmp_path / "summary.xlsx")]
    assert cli.main([*arguments, "--config", str(config)]) == 1
    assert seen == [expected.resolve()]
