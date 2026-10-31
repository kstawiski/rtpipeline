"""Selection no-match closes a stage and contributes no robustness measurements.

Synthetic sources use the real CLI, sidecar and receipt writers, admission and
aggregation. Workflow tests execute both Snakefile shell bodies with real
upstream receipts; only extraction is replaced by replay of a CLI-written
receipt. No clinical data or installed environment is changed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from rtpipeline import radiomics_robustness as rr
from rtpipeline import radiomics as rm
from rtpipeline import robustness_completion as rc
from rtpipeline.config_dependencies import materialize_stage_dependency
from rtpipeline.stage_completion import write_stage_completion_sentinel
from test_robustness_cli_failure_contract import _course_rules, _render, _shell_body
from test_robustness_consumer_integration import (
    ROOT, OUTPUT_NAME, CONDITIONS, CT_EXTRACTION_ARMS,
    _build_courses, _run_course, _rob_config, _run_aggregate,
    _write_manifest, _summary_paths, _roi_array,
)

OUTCOME = "not_applicable_no_target_structure"


def _ledger():
    spec = importlib.util.spec_from_file_location(
        "selection_unmatched_ledger", ROOT / "workflow/scripts/campaign_ledger.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unmatched_volumetric_source_has_no_anatomical_dispositions(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})
    monkeypatch.setattr(rm, "_rtstruct_masks", lambda *a, **kw: {"Unrequested1": _roi_array()})
    assert _run_course(cohort, "P3") == 0
    receipt = rc.read_robustness_completion_sentinel(cohort.sentinel("P3"))
    assert receipt.measurement_outcome == OUTCOME and not receipt.measured
    assert receipt.source_disposition_count == 0
    admitted = rr.admit_robustness_cohort_course(
        cohort.course_dir("P3"), patient_id="P3", course_id="C1",
        rob_config=_rob_config(cohort.config_path),
    )
    assert admitted.source_dispositions == [] and admitted.frame is None
    row = rr._robustness_course_outcome_rows([admitted]).iloc[0]
    assert "no source ROI matched" in row.nonmeasurement_reason


@pytest.mark.parametrize("outcome", ["measured", "source_only"])
def test_campaign_completion_keeps_matching_courses_unchanged(tmp_path, monkeypatch, outcome):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): outcome})
    assert _run_course(cohort, "P1", campaign=True) == 0
    ledger = _ledger()
    record = ledger.close_robustness_completion(
        cohort.output_root, "P1", "C1", cohort.sentinel("P1")
    )
    entry = json.loads(record.read_text())
    expected = rr.ROBUSTNESS_MEASURED_OUTCOME if outcome == "measured" else rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME
    assert entry["status"] == "ok" and entry["detail"] == expected
    assert rc.read_robustness_completion_sentinel(cohort.sentinel("P1")).measured == (outcome == "measured")


@pytest.mark.parametrize("with_measurements", [False, True])
def test_aggregation_accounts_for_unmatched_without_measurements(
    tmp_path, monkeypatch, with_measurements
):
    specs = {("P3", "C1"): "unmatched"}
    if with_measurements:
        specs.update({("P1", "C1"): "measured", ("P2", "C1"): "measured"})
    cohort = _build_courses(tmp_path, monkeypatch, specs)
    for patient, _ in specs:
        assert _run_course(cohort, patient) == 0
    output, raw = _summary_paths(tmp_path)
    assert _run_aggregate(cohort, _write_manifest(cohort), output) == 0
    outcomes = pd.read_excel(output, sheet_name="course_outcomes").set_index("patient_id")
    assert len(outcomes) == len(specs)
    row = outcomes.loc["P3"]
    assert row.measurement_outcome == OUTCOME
    assert "no source ROI matched" in row.nonmeasurement_reason
    assert not row.table_present and not row.contributes_measurements
    assert row.table_row_count == row.measured_value_row_count == 0
    values = pd.read_parquet(raw)
    assert "P3" not in set(values.patient_id)
    summary = pd.read_excel(output, sheet_name="global_summary")
    if with_measurements:
        assert set(values.patient_id) == {"P1", "P2"}
        assert len(values) == 2 * len(CONDITIONS) * len(CT_EXTRACTION_ARMS)
        assert set(summary.n_subjects) == {2}
    else:
        assert values.empty and summary.empty
    dispositions = pd.read_excel(output, sheet_name="source_dispositions")
    assert set(dispositions.course_measurement_outcome) == {OUTCOME}
    assert set(dispositions.roi_name) == {"Unrequested1"}


@pytest.mark.parametrize("change", [
    "source", "configuration", "receipt", "basis", "matched",
    "source-only-outcome", "failed-outcome", "run", "row-count", "digest", "code",
])
def test_unmatched_receipt_and_admission_reject_tampering_and_staleness(
    tmp_path, monkeypatch, change
):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})
    assert _run_course(cohort, "P3") == 0
    course = cohort.course_dir("P3")
    if change == "source":
        with (course / "RS.dcm").open("ab") as stream:
            stream.write(b"changed source")
    elif change == "configuration":
        config = _rob_config(cohort.config_path)
        config.perturbation.apply_to_structures = ["GTV*"]
        with pytest.raises(RuntimeError, match="effective configuration"):
            rr.admit_robustness_cohort_course(
                course, patient_id="P3", course_id="C1", rob_config=config
            )
        return
    elif change in {"receipt", "source-only-outcome", "failed-outcome", "run", "row-count", "digest"}:
        receipt = cohort.receipt("P3")
        if change == "receipt":
            receipt["requested_selection"] = ["GTV*"]
        elif change.endswith("outcome"):
            receipt["measurement_outcome"] = (
                "source_only_nonvolumetric" if change == "source-only-outcome" else "failed_extraction"
            )
        elif change == "run":
            receipt["robustness_run_identifier"] = "another-run"
        elif change == "row-count":
            receipt["source_dispositions"]["row_count"] += 1
        else:
            receipt["source_dispositions"]["sha256"] = "0" * 64
        cohort.write_receipt("P3", receipt)
    elif change == "code":
        identity = rr._current_robustness_code_identity()
        identity["sources_sha256"] = "0" * 64
        monkeypatch.setattr(rr, "_current_robustness_code_identity", lambda: identity)
    else:
        sidecar = rr.robustness_source_dispositions_path(course)
        payload = json.loads(sidecar.read_text())
        if change == "basis":
            payload["source_only_basis"] = None
        else:
            payload["rows"][0]["roi_name"] = "Marker1"
            payload["rows_sha256"] = rr._content_sha256(payload["rows"])
            payload["source_only_basis"]["selection_matched_nonvolumetric_count"] = 1
        sidecar.write_text(json.dumps(payload))
        # Rebinding the digest cannot make contradictory no-match evidence valid.
        receipt = cohort.receipt("P3")
        receipt["source_dispositions"]["sha256"] = rr._file_sha256(sidecar)
        cohort.write_receipt("P3", receipt)
    with pytest.raises(rc.RobustnessCompletionError):
        rc.read_robustness_completion_sentinel(cohort.sentinel("P3"))
    with pytest.raises((ValueError, RuntimeError)):
        rr.admit_robustness_cohort_course(
            course, patient_id="P3", course_id="C1", rob_config=_rob_config(cohort.config_path)
        )


def test_new_attempt_cannot_reuse_an_unmatched_receipt(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})
    assert _run_course(cohort, "P3") == 0
    monkeypatch.setattr(rr, "robustness_for_course", lambda *a, **kw: None)
    assert _run_course(cohort, "P3") == 1
    assert not cohort.sentinel("P3").exists()
    assert not rr.robustness_source_dispositions_path(cohort.course_dir("P3")).exists()


@pytest.mark.parametrize("branch", [0, 1], ids=["container-rule", "local-rule"])
@pytest.mark.parametrize("tamper", [False, True], ids=["valid", "stale-source"])
@pytest.mark.parametrize("target_present", [False, True], ids=["no-target", "selection-gap"])
def test_campaign_rule_closes_unmatched_with_real_receipts(
    tmp_path, monkeypatch, branch, tamper, target_present
):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})
    if target_present:
        import pydicom
        path = cohort.course_dir("P3") / "RS.dcm"
        ds = pydicom.dcmread(path)
        ds.StructureSetROISequence[0].ROIName = "HR-CTV"
        ds.RTROIObservationsSequence[0].RTROIInterpretedType = "TREATED_VOLUME"
        ds.save_as(path)
    expected_outcome = "selection_gap_targets_unselected" if target_present else OUTCOME
    expected_status = "technically_missing" if target_present else "not_applicable"
    assert _run_course(cohort, "P3", campaign=True) == 0
    course = cohort.course_dir("P3")
    saved = tmp_path / "producer-receipt.json"
    saved.write_bytes(cohort.sentinel("P3").read_bytes())
    # Extraction is covered above. Replay its writer-produced receipt after the
    # rule's initial invalidation; all shell gates and closure commands run.
    interpreter = tmp_path / "producer-replay"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:4] == ['-m', 'rtpipeline.cli', 'radiomics-robustness']:\n"
        "    assert '--campaign-mode' in sys.argv\n"
        f"    pathlib.Path({str(cohort.sentinel('P3'))!r}).write_bytes(pathlib.Path({str(saved)!r}).read_bytes())\n"
        "else:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    interpreter.chmod(0o700)
    dependency = materialize_stage_dependency(tmp_path / "deps", "dvh", {"enabled": False})
    write_stage_completion_sentinel(
        course, course / ".dvh_done", stage="dvh", status="disabled",
        configuration_dependency=dependency,
    )
    from rtpipeline.radiomics_ct_contract import write_completion_sentinel
    # The upstream radiomics shell gate reads the declared structured status.
    # A real main-radiomics writer receipt is supplied by the existing helper.
    from test_workflow_fail_closed import _write_ct_publication
    _write_ct_publication(course, patient_id="P3")
    write_completion_sentinel(course, course / ".radiomics_done")
    if tamper:
        with (course / "RS.dcm").open("ab") as stream:
            stream.write(b"changed source")
    values = {
        "log": str(tmp_path / "rule.log"), "threads": "1",
        "output.sentinel": str(cohort.sentinel("P3")),
        "input.dvh": str(course / ".dvh_done"),
        "input.radiomics": str(course / ".radiomics_done"),
        "params.enabled": "True", "params.campaign_mode": "True",
        "params.campaign_mode_flag": "--campaign-mode",
        "params.python": str(interpreter), "params.python_bin": str(interpreter.parent),
        "params.root_dir": str(ROOT), "params.output_dir": str(cohort.output_root),
        "params.course_dir": str(course), "params.parquet": str(cohort.table("P3")),
        "params.config": str(cohort.config_path),
        "wildcards.patient": "P3", "wildcards.course": "C1",
    }
    body = _render(_shell_body(_course_rules()[branch]), values)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", body], cwd=tmp_path,
        capture_output=True, text=True, env=env, timeout=60,
    )
    if tamper:
        assert result.returncode != 0, result.stdout + result.stderr
        assert not cohort.sentinel("P3").exists()
        assert not (cohort.output_root / "_campaign_ledger/records").exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((cohort.output_root / "_campaign_ledger/records/P3__C1__radiomics_robustness.json").read_text())
    assert record["status"] == expected_status
    assert record["detail"] == expected_outcome
    assert record["returncode"] == 0
    assert cohort.sentinel("P3").read_bytes() == saved.read_bytes()
    ledger = _ledger()
    summary = ledger.rollup(cohort.output_root)
    assert summary["stage_counts"]["radiomics_robustness"] == {expected_status: 1}
    # The receipt alone also preserves the reason if a rollup has no record.
    (cohort.output_root / "_campaign_ledger/records/P3__C1__radiomics_robustness.json").unlink()
    ledger.rollup(cohort.output_root)
    rows = pd.read_csv(cohort.output_root / "_campaign_ledger/campaign_ledger.csv")
    row = rows.loc[rows.stage == "radiomics_robustness"].iloc[0]
    assert row.status == expected_status and row.detail == expected_outcome
