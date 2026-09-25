"""A course with no planning CT closes radiomics and robustness as not applicable.

Synthetic courses only, written under pytest's tmp_path. No producer, service or
pipeline is launched: the stage wrapper's subprocess launch and the robustness
producer are replaced by functions that fail the test if they are reached.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import rtpipeline.segmentation as segmentation
import rtpipeline.snakemake_delegate as snakemake_delegate
from rtpipeline import cli
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.config_dependencies import materialize_stage_dependency
from rtpipeline.course_manifest import CURRENT_COURSE_MANIFEST_SCHEMA
from rtpipeline.radiomics_ct_contract import (
    NOT_APPLICABLE_STATUS,
    validate_completion_sentinel,
    validate_not_applicable_completion_sentinel,
    write_not_applicable_completion_sentinel,
)
from rtpipeline.stage_completion import write_stage_completion_sentinel

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
    write_synthetic_planning_ct,
)
from test_workflow_fail_closed import (
    _aggregate_snakemake,
    _write_course_inputs,
    _write_ct_publication,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_COURSE_STAGE = ROOT / "workflow" / "scripts" / "run_course_stage.py"
AGGREGATE_RESULTS = ROOT / "workflow" / "scripts" / "aggregate_results.py"
ROBUSTNESS_OUTPUT = "radiomics_robustness_ct.parquet"
NA_OUTCOME = "not_applicable_no_planning_ct"

sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import campaign_ledger  # noqa: E402


class _ProducerRan(BaseException):
    """A producer was reached; not an Exception so no handler can absorb it."""


@pytest.fixture(autouse=True)
def _in_process_segmentation_assessment(monkeypatch: pytest.MonkeyPatch) -> None:
    original = snakemake_delegate.invoke

    def invoke(**kwargs):
        if kwargs.get("operation") != "assess-segmentation":
            return original(**kwargs)
        arguments = list(kwargs["arguments"])
        course_dir = Path(arguments[arguments.index("--course-dir") + 1])
        return {
            "course_dir": str(course_dir.resolve(strict=False)),
            "outcome": segmentation.assess_course_segmentation(course_dir),
        }

    monkeypatch.setattr(snakemake_delegate, "invoke", invoke)


def _dependency(tmp_path: Path, stage: str) -> Path:
    return materialize_stage_dependency(
        tmp_path / "dependencies", stage, {"fixture": "not-applicable", "stage": stage}
    )


def _segmented_course(course_dir: Path, *, planning_ct: bool) -> Path:
    """A synthetic organized course with its segmentation stage completed."""
    course_dir.mkdir(parents=True, exist_ok=True)
    if planning_ct:
        write_synthetic_planning_ct(course_dir)
    plan, dose = write_synthetic_plan_and_dose(course_dir)
    write_minimal_course_contract(course_dir, selected_plans=[plan], selected_doses=[dose])
    # The real producer decides and records the segmentation status.
    outcome = segmentation.publish_course_segmentation_status(course_dir)
    tmp_root = course_dir.parents[2]
    for stage, sentinel in (
        ("segmentation", ".segmentation_done"),
        ("segmentation_custom", ".custom_models_done"),
        ("crop_ct", ".crop_ct_done"),
    ):
        configuration_stage = {
            "segmentation": "segmentation",
            "segmentation_custom": "custom-models",
            "crop_ct": "crop-ct",
        }[stage]
        write_stage_completion_sentinel(
            course_dir,
            course_dir / sentinel,
            stage=stage,
            status="disabled" if outcome["status"] == "disabled" else "ok",
            configuration_dependency=_dependency(tmp_root, configuration_stage),
        )
    return course_dir


def _stage_workflow(
    tmp_path: Path,
    course_dir: Path,
    *,
    stage: str,
    sentinel: str,
    configuration: Path,
    campaign_mode: bool = False,
) -> SimpleNamespace:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"courses": []}\n', encoding="utf-8")
    return SimpleNamespace(
        output=SimpleNamespace(sentinel=str(course_dir / sentinel)),
        log=[str(tmp_path / "logs" / f"{stage}.log")],
        input=SimpleNamespace(
            manifest=str(manifest),
            segmentation=str(course_dir / ".segmentation_done"),
            custom=str(course_dir / ".custom_models_done"),
            crop=str(course_dir / ".crop_ct_done"),
            configuration=str(configuration),
        ),
        params=SimpleNamespace(
            root_dir=str(ROOT),
            configfile=str(tmp_path / "config.yaml"),
            radiomics_env="rtpipeline-radiomics",
            python_bin=str(Path(sys.executable).parent),
            python=sys.executable,
            dicom_root=str(tmp_path / "dicom"),
            output_dir=str(course_dir.parents[1]),
            logs_dir=str(tmp_path / "logs"),
            stage=stage,
            custom_structures="",
            campaign_mode=campaign_mode,
        ),
        wildcards=SimpleNamespace(patient=course_dir.parent.name, course=course_dir.name),
        threads=1,
    )


_REAL_RUN = subprocess.run


def _no_stage_launch(command, *args, **kwargs):
    if "rtpipeline.cli" in [str(part) for part in command]:
        raise _ProducerRan("the stage CLI must not be launched for this course")
    return _REAL_RUN(command, *args, **kwargs)


def _run_wrapper(workflow: SimpleNamespace) -> None:
    runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})


def _robustness_config(tmp_path: Path, *, enabled: bool = True) -> Path:
    path = tmp_path / "robustness_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "radiomics_robustness": {
                    "enabled": enabled,
                    "modes": ["segmentation_perturbation"],
                    "segmentation_perturbation": {
                        "intensity": "standard",
                        "apply_to_structures": ["GTV*", "CTV*", "PTV*"],
                        "small_volume_changes": [-0.15, 0.0, 0.15],
                        "max_translation_mm": 4.0,
                        "n_random_contour_realizations": 2,
                        "noise_levels": [0.0, 10.0, 20.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_robustness(course_dir: Path, config: Path) -> int:
    return cli.main(
        [
            "radiomics-robustness",
            "--course-dir",
            str(course_dir),
            "--config",
            str(config),
            "--output",
            str(course_dir / ROBUSTNESS_OUTPUT),
            "--sentinel",
            str(course_dir / rc.ROBUSTNESS_COMPLETION_SENTINEL_NAME),
        ]
    )


def _ledger_record(course_dir: Path, stage: str) -> dict:
    records = campaign_ledger.records_dir(course_dir.parents[1])
    patient, course = course_dir.parent.name, course_dir.name
    return json.loads(
        (records / f"{patient}__{course}__{stage}.json").read_text(encoding="utf-8")
    )


def _no_ct_course_through_wrapper(tmp_path: Path, monkeypatch) -> Path:
    course_dir = _segmented_course(tmp_path / "output" / "P1" / "C1", planning_ct=False)
    monkeypatch.setattr(subprocess, "run", _no_stage_launch)
    for stage, sentinel, configuration_stage in (
        ("dvh", ".dvh_done", "dvh"),
        ("radiomics", ".radiomics_done", "radiomics"),
        ("qc", ".qc_done", "qc"),
    ):
        _run_wrapper(
            _stage_workflow(
                tmp_path,
                course_dir,
                stage=stage,
                sentinel=sentinel,
                configuration=_dependency(tmp_path, configuration_stage),
            )
        )
    monkeypatch.undo()
    return course_dir


# ---------------------------------------------------------------------------
# End to end through the wrapper, the robustness step and the consumers
# ---------------------------------------------------------------------------


def test_course_without_planning_ct_closes_radiomics_and_robustness_as_not_applicable(
    tmp_path, monkeypatch
):
    course_dir = _no_ct_course_through_wrapper(tmp_path, monkeypatch)
    assert json.loads(
        (course_dir / "metadata" / "segmentation_status.json").read_text(encoding="utf-8")
    )["status"] == "disabled"

    # DVH and QC: unchanged behaviour, a bound disabled generic completion.
    for sentinel in (".dvh_done", ".qc_done"):
        payload = json.loads((course_dir / sentinel).read_text(encoding="utf-8"))
        assert payload["schema"] == "rtpipeline-stage-completion-v1"
        assert payload["status"] == "disabled"
        assert payload["output_count"] == 0

    # Radiomics: its own sentinel, schema and validator, with the reason.
    radiomics = json.loads((course_dir / ".radiomics_done").read_text(encoding="utf-8"))
    assert radiomics["status"] == NOT_APPLICABLE_STATUS
    assert radiomics["schema"] == "rtpipeline-radiomics-completion-v2"
    assert radiomics["reason_code"] == "no_planning_ct"
    assert "no planning CT" in radiomics["reason"]
    assert radiomics["configuration_dependency_sha256"]
    assert validate_not_applicable_completion_sentinel(
        course_dir,
        configuration_dependency=_dependency(tmp_path, "radiomics"),
    ) == radiomics
    # The extraction validator still refuses it: it certifies no publication.
    with pytest.raises(Exception):
        validate_completion_sentinel(course_dir, course_dir / ".radiomics_done")
    assert not (course_dir / "radiomics_ct.parquet").exists()

    for stage in ("dvh", "radiomics", "qc"):
        record = _ledger_record(course_dir, stage)
        assert record["status"] == campaign_ledger.STATUS_NOT_APPLICABLE
        assert record["returncode"] == 0
        assert "no planning CT" in record["detail"]

    # Robustness: the step completes as not applicable without the producer.
    def _never(*_args, **_kwargs):
        raise _ProducerRan("robustness producer must not run for this course")

    monkeypatch.setattr(rr, "run_robustness_course", _never)
    monkeypatch.setattr(rr, "robustness_for_course", _never)
    assert _run_robustness(course_dir, _robustness_config(tmp_path)) == 0
    receipt = rc.read_robustness_completion_sentinel(
        course_dir / rc.ROBUSTNESS_COMPLETION_SENTINEL_NAME
    )
    assert receipt.measurement_outcome == NA_OUTCOME
    assert not receipt.measured
    assert receipt.source_disposition_count == 0
    assert not (course_dir / ROBUSTNESS_OUTPUT).exists()

    # Cohort robustness admission counts it as a not-applicable course.
    rob_config = rr.RobustnessConfig.from_dict(
        yaml.safe_load(_robustness_config(tmp_path).read_text(encoding="utf-8"))[
            "radiomics_robustness"
        ]
    )
    admitted = rr.admit_robustness_cohort_course(
        course_dir, patient_id="P1", course_id="C1", rob_config=rob_config
    )
    outcome_rows = rr._robustness_course_outcome_rows([admitted])
    assert outcome_rows.to_dict("records")[0]["measurement_outcome"] == NA_OUTCOME
    assert not bool(outcome_rows["contributes_measurements"].iloc[0])

    # Campaign ledger rollup reports not applicable, not failed or missing.
    summary = campaign_ledger.rollup(course_dir.parents[1])
    assert summary["failed_unit_count"] == 0
    for stage in ("radiomics", "radiomics_robustness"):
        assert summary["stage_counts"][stage] == {"not_applicable": 1}


def test_robustness_shell_gate_accepts_not_applicable_radiomics(tmp_path, monkeypatch):
    course_dir = _no_ct_course_through_wrapper(tmp_path, monkeypatch)
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    patterns = {
        line.split("grep -Eq '", 1)[1].split("'", 1)[0]
        for line in snakefile.splitlines()
        if "grep -Eq '" in line and "{input.radiomics}" in line
    }
    assert len(patterns) == 1, "both robustness rules must share one gate"
    pattern = patterns.pop().replace("{{", "{").replace("}}", "}")
    failed = tmp_path / "failed_token"
    failed.write_text("failed\n", encoding="utf-8")
    forged = tmp_path / "forged"
    forged.write_text('{"status": "not_applicable_forged"}\n', encoding="utf-8")

    def gate(path: Path) -> int:
        return subprocess.run(["grep", "-Eq", pattern, str(path)], check=False).returncode

    assert gate(course_dir / ".radiomics_done") == 0
    assert gate(failed) != 0
    assert gate(forged) != 0


def _write_not_applicable_aggregation_course(course_dir: Path) -> None:
    """A CT-less course closed the way the wrapper closes it, for aggregation."""
    _write_course_inputs(course_dir)
    (course_dir / "dvh_metrics.xlsx").unlink()
    segmentation.publish_course_segmentation_status(course_dir)
    dependency_root = course_dir.parents[1] / "stage-dependencies"
    for stage, sentinel, configuration_stage in (
        ("segmentation", ".segmentation_done", "segmentation"),
        ("dvh", ".dvh_done", "dvh"),
        ("qc", ".qc_done", "qc"),
    ):
        write_stage_completion_sentinel(
            course_dir,
            course_dir / sentinel,
            stage=stage,
            status="disabled",
            configuration_dependency=materialize_stage_dependency(
                dependency_root, configuration_stage, {"fixture": stage}
            ),
        )
    write_not_applicable_completion_sentinel(course_dir)


def _normal_aggregation_course(course_dir: Path, patient_id: str) -> None:
    _write_ct_publication(course_dir, patient_id=patient_id)
    _write_course_inputs(course_dir, radiomics_sentinel="ok\n")


@pytest.mark.parametrize("campaign_mode", [False, True])
def test_aggregation_counts_course_without_planning_ct_as_not_applicable(
    tmp_path, campaign_mode
):
    workflow, outputs = _aggregate_snakemake(tmp_path, radiomics_enabled=True)
    workflow.params.campaign_mode = campaign_mode
    output_dir = Path(workflow.params.output_dir)
    _normal_aggregation_course(output_dir / "P1" / "C1", "P1")
    _write_not_applicable_aggregation_course(output_dir / "P2" / "C1")

    runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    dvh = pd.read_excel(outputs["dvh"])
    na_rows = dvh.loc[dvh["patient_id"] == "P2"]
    assert len(na_rows) == 1
    assert na_rows["row_status"].iloc[0] == "not_computed"
    assert NA_OUTCOME in na_rows["failure_reason"].iloc[0]
    assert set(dvh.loc[dvh["patient_id"] == "P1", "row_status"]) == {"computed"}

    radiomics = pd.read_excel(outputs["radiomics"])
    assert set(radiomics["patient_id"]) == {"P1"}
    exclusions = json.loads(radiomics["radiomics_cohort_exclusions_json"].iloc[0])
    assert [(e["patient_id"], e["disposition_type"], e["source"]) for e in exclusions] == [
        ("P2", NA_OUTCOME, "radiomics_completion")
    ]
    assert int(radiomics["radiomics_cohort_validated_n"].iloc[0]) == 2
    assert int(radiomics["radiomics_cohort_extracted_n"].iloc[0]) == 1

    ledger = json.loads(
        (output_dir / "_RESULTS" / "radiomics_denominator_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    na_course = [row for row in ledger["course"] if row["patient_id"] == "P2"]
    assert na_course == [
        {
            "entity": "COURSE",
            "course_id": "C1",
            "patient_id": "P2",
            "screened": 1,
            "in_scope": 0,
            "out_of_scope": 1,
            "adequate_coverage": 0,
            "insufficient_coverage": 0,
            "valid_derivation": 0,
            "technical_exclusion": 0,
            "indeterminate": 0,
            "extracted": 0,
            "reason_code": NA_OUTCOME,
        }
    ]
    if campaign_mode:
        attrition = pd.read_csv(output_dir / "_RESULTS" / "campaign_attrition.csv")
        statuses = dict(zip(attrition["patient_id"], attrition["status"]))
        assert statuses == {"P1": "aggregated", "P2": "not_applicable"}


# ---------------------------------------------------------------------------
# Negative controls: fail closed exactly as before
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("campaign_mode", [False, True])
def test_course_with_planning_ct_whose_radiomics_fails_still_fails(
    tmp_path, monkeypatch, campaign_mode
):
    course_dir = tmp_path / "output" / "P1" / "C1"
    course_dir.mkdir(parents=True)
    write_synthetic_planning_ct(course_dir)
    plan, dose = write_synthetic_plan_and_dose(course_dir)
    write_minimal_course_contract(course_dir, selected_plans=[plan], selected_doses=[dose])
    for stage, sentinel, configuration_stage in (
        ("segmentation", ".segmentation_done", "segmentation"),
        ("segmentation_custom", ".custom_models_done", "custom-models"),
        ("crop_ct", ".crop_ct_done", "crop-ct"),
    ):
        write_stage_completion_sentinel(
            course_dir,
            course_dir / sentinel,
            stage=stage,
            status="disabled",
            configuration_dependency=_dependency(tmp_path, configuration_stage),
        )
    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {"status": "ok", "reasons": ["validated test fixture"]},
    )
    def _failing_stage(command, *args, **kwargs):
        if "rtpipeline.cli" in [str(part) for part in command]:
            return SimpleNamespace(returncode=1)
        return _REAL_RUN(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _failing_stage)
    workflow = _stage_workflow(
        tmp_path,
        course_dir,
        stage="radiomics",
        sentinel=".radiomics_done",
        configuration=_dependency(tmp_path, "radiomics"),
        campaign_mode=campaign_mode,
    )
    sentinel = course_dir / ".radiomics_done"
    if campaign_mode:
        with pytest.raises(SystemExit) as raised:
            _run_wrapper(workflow)
        assert raised.value.code == 0
        assert sentinel.read_text(encoding="utf-8") == "failed\n"
    else:
        with pytest.raises(SystemExit) as raised:
            _run_wrapper(workflow)
        assert raised.value.code == 1
        assert not sentinel.exists()
    assert _ledger_record(course_dir, "radiomics")["status"] == "failed"


def test_planning_ct_course_reported_disabled_never_gets_a_not_applicable_radiomics(
    tmp_path, monkeypatch
):
    """Even if segmentation claimed "disabled", a declared CT blocks the claim."""
    course_dir = tmp_path / "output" / "P1" / "C1"
    course_dir.mkdir(parents=True)
    write_synthetic_planning_ct(course_dir)
    plan, dose = write_synthetic_plan_and_dose(course_dir)
    write_minimal_course_contract(course_dir, selected_plans=[plan], selected_doses=[dose])
    segmentation.publish_course_segmentation_status(
        course_dir, {"status": "disabled", "reasons": ["forced by the test"]}
    )
    for stage, sentinel, configuration_stage in (
        ("segmentation", ".segmentation_done", "segmentation"),
        ("segmentation_custom", ".custom_models_done", "custom-models"),
        ("crop_ct", ".crop_ct_done", "crop-ct"),
    ):
        write_stage_completion_sentinel(
            course_dir,
            course_dir / sentinel,
            stage=stage,
            status="disabled",
            configuration_dependency=_dependency(tmp_path, configuration_stage),
        )
    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {"status": "disabled", "reasons": ["forced by the test"]},
    )
    monkeypatch.setattr(subprocess, "run", _no_stage_launch)

    with pytest.raises(RuntimeError, match="declares a planning CT"):
        _run_wrapper(
            _stage_workflow(
                tmp_path,
                course_dir,
                stage="radiomics",
                sentinel=".radiomics_done",
                configuration=_dependency(tmp_path, "radiomics"),
            )
        )
    assert not (course_dir / ".radiomics_done").exists()
    with pytest.raises(ValueError, match="declares a planning CT"):
        write_not_applicable_completion_sentinel(course_dir)
    with pytest.raises(RuntimeError, match="not-applicable"):
        rr.write_robustness_not_applicable_dispositions(
            course_dir,
            rob_config=rr.RobustnessConfig.from_dict({"enabled": True}),
            output_name=ROBUSTNESS_OUTPUT,
        )


def test_not_applicable_radiomics_refuses_inconsistent_evidence(tmp_path, monkeypatch):
    course_dir = _no_ct_course_through_wrapper(tmp_path, monkeypatch)
    sentinel = course_dir / ".radiomics_done"

    # A stale CT publication beside the claim.
    publication = course_dir / "radiomics_ct.parquet"
    publication.write_bytes(b"not a publication")
    with pytest.raises(ValueError, match="existing CT radiomics publication"):
        validate_not_applicable_completion_sentinel(course_dir)
    publication.unlink()

    # A segmentation completion that no longer says disabled.
    original = (course_dir / ".segmentation_done").read_bytes()
    (course_dir / ".segmentation_done").write_text("ok\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_not_applicable_completion_sentinel(course_dir)
    (course_dir / ".segmentation_done").write_bytes(original)
    validate_not_applicable_completion_sentinel(course_dir)

    # Extraction fields smuggled into the claim.
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    sentinel.write_text(json.dumps({**payload, "row_count": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="extraction fields"):
        validate_not_applicable_completion_sentinel(course_dir)
    sentinel.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    # A robustness receipt stops validating once its upstream claim is gone.
    assert _run_robustness(course_dir, _robustness_config(tmp_path)) == 0
    receipt_path = course_dir / rc.ROBUSTNESS_COMPLETION_SENTINEL_NAME
    rc.read_robustness_completion_sentinel(receipt_path)
    sentinel.write_text("failed\n", encoding="utf-8")
    with pytest.raises(rc.RobustnessCompletionError, match="not-applicable"):
        rc.read_robustness_completion_sentinel(receipt_path)


def test_robustness_for_failed_or_disabled_radiomics_keeps_todays_behaviour(
    tmp_path, monkeypatch
):
    course_dir = _no_ct_course_through_wrapper(tmp_path, monkeypatch)
    receipt = course_dir / rc.ROBUSTNESS_COMPLETION_SENTINEL_NAME
    calls = []

    def _producer(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("producer reached")

    monkeypatch.setattr(rr, "run_robustness_course", _producer)

    # An upstream radiomics that is not a not-applicable claim is not routed
    # here: the producer runs and fails, and no receipt is published.
    (course_dir / ".radiomics_done").write_text("failed\n", encoding="utf-8")
    assert _run_robustness(course_dir, _robustness_config(tmp_path)) == 1
    assert calls and not receipt.exists()

    # A robustness configuration switched off is refused before any routing.
    write_not_applicable_completion_sentinel(course_dir)
    calls.clear()
    assert _run_robustness(course_dir, _robustness_config(tmp_path, enabled=False)) == 1
    assert not calls and not receipt.exists()


def test_aggregation_refuses_mixed_or_forged_not_applicable_closures(tmp_path):
    workflow, outputs = _aggregate_snakemake(tmp_path, radiomics_enabled=True)
    output_dir = Path(workflow.params.output_dir)
    _normal_aggregation_course(output_dir / "P1" / "C1", "P1")
    mixed = output_dir / "P2" / "C1"
    _write_not_applicable_aggregation_course(mixed)
    # QC claims a success while DVH and radiomics claim not applicable.
    write_stage_completion_sentinel(
        mixed,
        mixed / ".qc_done",
        stage="qc",
        status="ok",
        configuration_dependency=materialize_stage_dependency(
            output_dir / "stage-dependencies", "qc", {"fixture": "mixed"}
        ),
    )
    with pytest.raises(RuntimeError, match=r"\.qc_done has status 'ok'; expected disabled"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})
    assert not any(path.exists() for path in outputs.values())


def test_aggregation_rejects_not_applicable_radiomics_on_a_planning_ct_course(tmp_path):
    workflow, outputs = _aggregate_snakemake(tmp_path, radiomics_enabled=True)
    output_dir = Path(workflow.params.output_dir)
    _normal_aggregation_course(output_dir / "P1" / "C1", "P1")
    course_dir = output_dir / "P2" / "C1"
    course_dir.mkdir(parents=True)
    write_synthetic_planning_ct(course_dir)
    _write_course_inputs(course_dir)
    (course_dir / ".radiomics_done").write_text(
        json.dumps({"status": "not_applicable", "reason_code": "no_planning_ct"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"\.radiomics_done has status 'not_applicable'; expected ok"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})
    assert not any(path.exists() for path in outputs.values())
