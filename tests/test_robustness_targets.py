"""Governed robustness target evidence, synthetic RTSTRUCTs and CLI receipts."""
import json
from dataclasses import replace

import pandas as pd
import pydicom
import pytest
import yaml

from rtpipeline import radiomics as rm
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from test_robustness_consumer_integration import (
    _build_courses, _run_course, _rob_config, _roi_array, _run_aggregate,
    _write_manifest, _summary_paths,
)
from test_robustness_selection_unmatched import _ledger

NO_TARGET = "not_applicable_no_target_structure"
GAP = "selection_gap_targets_unselected"


@pytest.mark.parametrize("token", ["GTV", "CTV", "PTV", "ITV", "gtv", "cTv"])
@pytest.mark.parametrize("prefix", ["", "HR-", "IR_", "LR.", "boost/", "risk "])
@pytest.mark.parametrize("suffix", ["", "1", "_45", "-ring", " boost", ".2", "/3"])
def test_target_tokens(token, prefix, suffix):
    assert rr.detect_robustness_target(prefix + token + suffix) == ("name_token",)


@pytest.mark.parametrize("name", [
    "ODBYTNICA", "PECHERZ", "PĘCHERZ", "CTVx", "GTVnode", "PTVolume", "ITValue",
    "xCTV", "1PTV", "(CTV)", "x+PTV", "x:CTV", "x\tCTV", "CTVą", "PTVé", "",
])
def test_non_target_names(name):
    assert rr.detect_robustness_target(name) == ()


@pytest.mark.parametrize("name", ["HR-CTV", "IR-CTV", "LR-CTV", "z_PTV_opt", "PTV-ring"])
def test_risk_prefixes_and_helpers_are_detected(name):
    assert rr.detect_robustness_target(name, "TREATED_VOLUME") == ("name_token",)


@pytest.mark.parametrize("kind", ["GTV", "CTV", "PTV"])
def test_interpreted_type(kind):
    assert rr.detect_robustness_target("Unnamed", kind) == (f"RTROIInterpretedType:{kind}",)
    assert rr.detect_robustness_target("HR-CTV", kind) == ("name_token", f"RTROIInterpretedType:{kind}")


@pytest.mark.parametrize("kind", ["TREATED_VOLUME", "ITV", "ORGAN", "", None])
def test_other_types_alone_do_not_establish_target(kind):
    assert rr.detect_robustness_target("PECHERZ", kind) == ()


@pytest.mark.parametrize("mode", ["typo", "TARGETS", "", None, 1, []])
def test_unknown_selection_mode_rejected(mode):
    with pytest.raises(ValueError, match="target_selection"):
        rr.RobustnessConfig.from_dict({"target_selection": mode})
    with pytest.raises(ValueError, match="target_selection"):
        rr.RobustnessConfig(target_selection=mode)


def _source(cohort, patient, name, kind="ORGAN"):
    path = cohort.course_dir(patient) / "RS.dcm"
    ds = pydicom.dcmread(path)
    ds.StructureSetROISequence[0].ROIName = name
    ds.RTROIObservationsSequence[0].RTROIInterpretedType = kind
    ds.save_as(path)


def _mode(cohort, mode):
    data = yaml.safe_load(cohort.config_path.read_text())
    data["radiomics_robustness"]["target_selection"] = mode
    cohort.config_path.write_text(yaml.safe_dump(data))


@pytest.mark.parametrize("name,kind,expected", [
    ("ODBYTNICA", "ORGAN", NO_TARGET),
    ("PECHERZ", "TREATED_VOLUME", NO_TARGET),
    ("HR-CTV", "TREATED_VOLUME", GAP),
    ("IR-CTV", "TREATED_VOLUME", GAP),
    ("Unnamed", "CTV", GAP),
    ("Unnamed", "PTV", GAP),
    ("z_PTV_opt", "ORGAN", GAP),
])
def test_cli_ledger_rollup_and_aggregation(tmp_path, monkeypatch, name, kind, expected):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"},
                            apply_to_structures=["GTV*", "CTV*", "PTV*"])
    _source(cohort, "P3", name, kind)
    # Helpers that begin PTV would match these configured patterns; z_PTV_opt does not.
    assert _run_course(cohort, "P3", sentinel=False, campaign=True) == 0
    receipt = cohort.receipt("P3")
    assert receipt["measurement_outcome"] == expected
    assert receipt["target_selection"] == "patterns"
    missed = receipt["missed_targets"]
    assert [r["roi_name"] for r in missed] == ([name] if expected == GAP else [])
    if missed:
        assert missed[0]["detector_rules"] == list(rr.detect_robustness_target(name, kind))
    assert not rc.read_robustness_completion_sentinel(cohort.sentinel("P3")).measured
    ledger = _ledger()
    record = ledger.close_robustness_completion(cohort.output_root, "P3", "C1", cohort.sentinel("P3"))
    status = "technically_missing" if expected == GAP else "not_applicable"
    assert json.loads(record.read_text())["status"] == status
    assert ledger.rollup(cohort.output_root)["stage_counts"]["radiomics_robustness"] == {status: 1}
    if expected == GAP:
        stale = json.loads(record.read_text())
        stale.update(status="not_applicable", detail="not_applicable_selection_unmatched")
        record.write_text(json.dumps(stale))
        assert ledger.rollup(cohort.output_root)["stage_counts"]["radiomics_robustness"] == {status: 1}
        rows = pd.read_csv(cohort.output_root / "_campaign_ledger/campaign_ledger.csv")
        assert rows.loc[rows.stage.eq("radiomics_robustness"), "detail"].item() == GAP
    record.unlink()
    assert ledger.rollup(cohort.output_root)["stage_counts"]["radiomics_robustness"] == {status: 1}
    rows = pd.read_csv(cohort.output_root / "_campaign_ledger/campaign_ledger.csv")
    assert rows.loc[rows.stage.eq("radiomics_robustness"), "detail"].item() == expected
    output, raw = _summary_paths(tmp_path)
    assert _run_aggregate(cohort, _write_manifest(cohort), output) == 0
    row = pd.read_excel(output, sheet_name="course_outcomes").iloc[0]
    assert row.measurement_outcome == expected and row.target_selection == "patterns"
    assert bool(row.technically_missing) == (expected == GAP)
    assert row.accounting_status == status
    assert row.measured_value_row_count == 0 and not row.contributes_measurements
    assert pd.read_parquet(raw).empty
    counts = pd.read_excel(output, sheet_name="course_counts").iloc[0]
    assert counts.technically_missing == int(expected == GAP)
    assert counts.not_applicable == int(expected == NO_TARGET)


@pytest.mark.parametrize("name,kind", [("HR-CTV", "TREATED_VOLUME"), ("IR-CTV", "TREATED_VOLUME"),
                                      ("Unnamed", "CTV"), ("Unnamed", "PTV")])
def test_targets_mode_measures_name_and_typed_targets(tmp_path, monkeypatch, name, kind):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"}, apply_to_structures=["GTV*"])
    _source(cohort, "P1", name, kind)
    _mode(cohort, "targets")
    identity = rr.RobustnessRoiIdentity.from_mapping(dict(
        patient_id="P1", course_id="C1", series_uid="1.2.3", segmentation_source="Manual",
        mask_identity="source-mask", roi_original_name=name, stable_roi_identifier="roi-1",
    ))
    monkeypatch.setattr(rr, "_load_main_ct_identity_catalog", lambda *a, **kw: ({("Manual", name): identity}, {}))
    monkeypatch.setattr(rm, "_rtstruct_masks", lambda *a, **kw: {name: _roi_array()} if kw["retain_mask"](name) else {})
    assert _run_course(cohort, "P1") == 0
    receipt = cohort.receipt("P1")
    assert receipt["measurement_outcome"] == "measured" and receipt["target_selection"] == "targets"
    table = pd.read_parquet(cohort.table("P1"))
    assert set(table.roi_original_name) == {name}
    assert set(table.target_selection) == {"targets"}
    cfg = _rob_config(cohort.config_path)
    assert rr.effective_robustness_configuration(cfg, output_name="x") != rr.effective_robustness_configuration(
        replace(cfg, target_selection="patterns"), output_name="x")
    admitted = rr.admit_robustness_cohort_course(cohort.course_dir("P1"), patient_id="P1", course_id="C1", rob_config=cfg)
    assert admitted.measured
    original_hashes = table.extraction_configured_parameter_hash.copy()
    bound_hashes = table.configured_parameter_hash.copy()
    payload = yaml.safe_load(cohort.config_path.read_text())
    payload["radiomics_robustness"]["target_selection"] = "patterns"
    payload["radiomics_robustness"]["segmentation_perturbation"]["apply_to_structures"] = [name]
    cohort.config_path.write_text(yaml.safe_dump(payload))
    assert _run_course(cohort, "P1") == 0
    patterns_table = pd.read_parquet(cohort.table("P1"))
    assert patterns_table.extraction_configured_parameter_hash.equals(original_hashes)
    assert not patterns_table.configured_parameter_hash.equals(bound_hashes)
    assert patterns_table.roi_original_name.equals(table.roi_original_name)


@pytest.mark.parametrize("change", ["mode", "rules", "missed", "old-outcome", "rebound-evidence"])
def test_gap_receipt_rejects_tampering(tmp_path, monkeypatch, change):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})
    _source(cohort, "P3", "HR-CTV", "TREATED_VOLUME")
    assert _run_course(cohort, "P3") == 0
    receipt = cohort.receipt("P3")
    if change == "mode":
        receipt["target_selection"] = "targets"
    elif change == "rules":
        receipt["missed_targets"][0]["detector_rules"] = []
    elif change == "missed":
        receipt["missed_targets"] = []
    elif change == "old-outcome":
        receipt["measurement_outcome"] = "not_applicable_selection_unmatched"
    else:
        sidecar = rr.robustness_source_dispositions_path(cohort.course_dir("P3"))
        payload = json.loads(sidecar.read_text())
        payload["source_only_basis"]["missed_targets"] = []
        sidecar.write_text(json.dumps(payload))
        receipt["missed_targets"] = []
        receipt["source_dispositions"]["sha256"] = rr._file_sha256(sidecar)
    cohort.write_receipt("P3", receipt)
    with pytest.raises(rc.RobustnessCompletionError):
        rc.read_robustness_completion_sentinel(cohort.sentinel("P3"))
    if change == "old-outcome":
        assert _run_course(cohort, "P3") == 0
        assert cohort.receipt("P3")["measurement_outcome"] == GAP
        assert cohort.receipt("P3")["robustness_run_identifier"] != receipt["robustness_run_identifier"]


@pytest.mark.parametrize("selection", [[], ["PTV*", "HR-CTV"], ["ctv?", "[PG]TV*"], ["*"], ["ODBYTNICA"]])
def test_patterns_mode_is_byte_identical_to_existing_fnmatch_selection(selection):
    from fnmatch import fnmatch
    names = ["HR-CTV", "IR-CTV", "CTV1", "CTVx", "PTV1", "gtv", "ODBYTNICA", "z_PTV_opt", "Unnamed"]
    expected = [name for name in names if any(fnmatch(name.upper(), p.upper()) for p in selection)]
    actual = [name for name in names if rr.robustness_selection_matches(name, selection, interpreted_types="PTV")]
    explicit = [name for name in names if rr.robustness_selection_matches(name, selection, target_selection="patterns")]
    assert json.dumps(actual).encode() == json.dumps(explicit).encode() == json.dumps(expected).encode()
    assert rr.robustness_selection_matches("ODBYTNICA", ["ODB*"], target_selection="targets")


def test_target_evidence_from_all_sources_and_typed_rules_are_source_specific(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})
    _source(cohort, "P3", "ODBYTNICA")
    second = cohort.course_dir("P3") / "second.dcm"
    ds = pydicom.dcmread(cohort.course_dir("P3") / "RS.dcm")
    ds.StructureSetROISequence[0].ROIName = "HR-CTV"
    ds.RTROIObservationsSequence[0].RTROIInterpretedType = "TREATED_VOLUME"
    ds.save_as(second)
    monkeypatch.setattr(rm, "_standard_rtstruct_sources", lambda *a: [
        ("Manual", cohort.course_dir("P3") / "RS.dcm", None), ("AutoRTS_total", second, None),
    ])
    # The real structural inventory is used; only feature/mask extraction is stubbed.
    monkeypatch.setattr(rm, "_rtstruct_masks", lambda *a, **kw: {})
    assert _run_course(cohort, "P3") == 0
    missed = cohort.receipt("P3")["missed_targets"]
    assert [(r["segmentation_source"], r["roi_name"]) for r in missed] == [("AutoRTS_total", "HR-CTV")]
    from rtpipeline.roi_requiredness import Requiredness
    first = cohort.course_dir("P3") / "RS.dcm"
    _source(cohort, "P3", "SameName", "PTV")
    ds.StructureSetROISequence[0].ROIName = "SameName"
    ds.RTROIObservationsSequence[0].RTROIInterpretedType = "ORGAN"
    ds.save_as(second)
    inventory = [roi for path in (first, second) for roi in rr._robustness_target_inventory(
        {"source_path": str(path), "segmentation_source": "Manual"})]
    assert rr._robustness_selection_requiredness(first, [], target_selection="targets", target_inventory=inventory)["SameName"] == Requiredness.ANALYSIS_REQUIRED
    assert rr._robustness_selection_requiredness(second, [], target_selection="targets", target_inventory=inventory)["SameName"] == Requiredness.INVENTORY_ONLY


def test_aggregate_keeps_selection_gap_in_denominator(tmp_path, monkeypatch):
    specs = {("P1", "C1"): "measured", ("P2", "C1"): "measured", ("P3", "C1"): "unmatched"}
    cohort = _build_courses(tmp_path, monkeypatch, specs)
    _source(cohort, "P3", "HR-CTV", "TREATED_VOLUME")
    for patient, _ in specs:
        assert _run_course(cohort, patient) == 0
    output, raw = _summary_paths(tmp_path)
    assert _run_aggregate(cohort, _write_manifest(cohort), output) == 0
    counts = pd.read_excel(output, sheet_name="course_counts").iloc[0]
    assert counts.courses == 3 and counts.measured == 2 and counts.technically_missing == 1
    assert counts.not_applicable == 0
    assert set(pd.read_parquet(raw).patient_id) == {"P1", "P2"}
    assert set(pd.read_excel(output, sheet_name="global_summary").n_subjects) == {2}


@pytest.mark.parametrize("mode", ["unknown", "TARGETS", None, 4])
def test_cli_rejects_unknown_selection_mode(tmp_path, monkeypatch, mode):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    _mode(cohort, mode)
    assert _run_course(cohort, "P1") == 1
    assert not cohort.sentinel("P1").exists()


@pytest.mark.parametrize("mode", ["patterns", "targets"])
def test_nifti_sources_use_detector_and_bind_source_bytes(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    import SimpleITK as sitk
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"}, apply_to_structures=["GTV*"])
    _mode(cohort, mode)
    course = cohort.course_dir("P1")
    ct = course / "ct.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(_roi_array().astype("float32")), str(ct))
    seg = course / "Segmentation_TotalSegmentator" / "HR-CTV.nii.gz"
    seg.parent.mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(_roi_array()), str(seg))
    monkeypatch.setattr(rm, "_load_series_image", lambda *a: None)
    monkeypatch.setattr(rr, "load_course_contract", lambda *a: SimpleNamespace(
        planning_ct_dir=None, planning_ct_nifti=ct, planning_ct={"series_instance_uid": "1.2.3"},
    ))
    identity = rr.RobustnessRoiIdentity.from_mapping(dict(
        patient_id="P1", course_id="C1", series_uid="1.2.3", segmentation_source="AutoRTS_total",
        mask_identity="source-mask", roi_original_name="HR-CTV", stable_roi_identifier="roi-1",
    ))
    monkeypatch.setattr(rr, "_load_main_ct_identity_catalog", lambda *a, **kw: ({("AutoRTS_total", "HR-CTV"): identity}, {}))
    assert _run_course(cohort, "P1") == 0
    receipt = cohort.receipt("P1")
    assert receipt["measurement_outcome"] == (GAP if mode == "patterns" else "measured")
    if mode == "patterns":
        assert [r["roi_name"] for r in receipt["missed_targets"]] == ["HR-CTV"]
        # A changed NIfTI cannot continue to certify target absence or a gap.
        seg.write_bytes(seg.read_bytes() + b"changed")
        with pytest.raises(rc.RobustnessCompletionError):
            rc.read_robustness_completion_sentinel(cohort.sentinel("P1"))


@pytest.mark.parametrize("token", ["GTV", "CTV", "PTV", "ITV"])
@pytest.mark.parametrize("prefix", ["", "risk-", "risk_", "risk.", "risk/", "risk "])
@pytest.mark.parametrize("letter", ["x", "N", "ą", "é"])
def test_letter_suffixes_never_form_target_tokens(token, prefix, letter):
    assert rr.detect_robustness_target(prefix + token + letter) == ()


@pytest.mark.parametrize("cached_dependency", [False, True])
def test_workflow_dependency_invalidates_old_no_match_receipt(tmp_path, cached_dependency):
    from rtpipeline.config_dependencies import materialize_stage_dependency, content_sha256, advance_dependency_past_unbound_outputs
    from test_robustness_consumer_integration import ROOT
    scientific_config = {"enabled": True}
    robustness_config = {"enabled": True}
    old_payload = {"radiomics_configuration_sha256": content_sha256(scientific_config),
                   "robustness": robustness_config}
    dependency = materialize_stage_dependency(tmp_path, "radiomics-robustness", old_payload)
    old_digest = json.loads(dependency.read_text())["sha256"]
    if not cached_dependency:
        dependency.unlink()
    output_root = tmp_path / "Output"
    receipt = output_root / "P" / "C" / ".radiomics_robustness_done"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "ok", "measurement_outcome": "not_applicable_selection_unmatched"}))
    text = (ROOT / "Snakefile").read_text()
    assignment = text[text.index("ROBUSTNESS_CONFIG_DEPENDENCY = _dependency_path("):]
    assignment = assignment[:assignment.index("\n\n# Adding a real input")]
    namespace = {
        "_dependency_path": lambda stage, payload, paths: materialize_stage_dependency(tmp_path, stage, payload),
        "content_sha256": content_sha256, "_radiomics_scientific_config": scientific_config,
        "ROBUSTNESS_CONFIG": robustness_config, "_radiomics_source_paths": [],
        "OUTPUT_DIR": output_root, "json": json, "Path": type(tmp_path),
        "advance_dependency_past_unbound_outputs": advance_dependency_past_unbound_outputs,
    }
    exec(compile(assignment, str(ROOT / "Snakefile"), "exec"), namespace)
    assert json.loads(dependency.read_text())["sha256"] != old_digest
    assert dependency.stat().st_mtime_ns > receipt.stat().st_mtime_ns
    # Once recomputed, identical parses preserve the dependency timestamp.
    receipt.write_text(json.dumps({"measurement_outcome": GAP}))
    stamp = dependency.stat().st_mtime_ns
    exec(compile(assignment, str(ROOT / "Snakefile"), "exec"), namespace)
    assert dependency.stat().st_mtime_ns == stamp


@pytest.mark.parametrize("field", ["target_selection", "robustness_configuration_sha256", "configured_parameter_hash"])
def test_table_selection_provenance_rejects_rebound_tampering(tmp_path, monkeypatch, field):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    frame = pd.read_parquet(cohort.table("P1"))
    frame[field] = "targets" if field == "target_selection" else "0" * 64
    frame.to_parquet(cohort.table("P1"), index=False)
    sidecar = rr.robustness_source_dispositions_path(cohort.course_dir("P1"))
    payload = json.loads(sidecar.read_text())
    payload["measured_output"]["sha256"] = rr._file_sha256(cohort.table("P1"))
    sidecar.write_text(json.dumps(payload))
    receipt = cohort.receipt("P1")
    receipt["measured_output"]["sha256"] = rr._file_sha256(cohort.table("P1"))
    receipt["source_dispositions"]["sha256"] = rr._file_sha256(sidecar)
    cohort.write_receipt("P1", receipt)
    with pytest.raises(ValueError, match="target_selection|configuration hash"):
        rr.admit_robustness_cohort_course(cohort.course_dir("P1"), patient_id="P1", course_id="C1", rob_config=_rob_config(cohort.config_path))
