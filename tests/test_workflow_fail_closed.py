"""Regression tests for Snakemake course-stage and cohort publication failures."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import rtpipeline.segmentation as segmentation
import rtpipeline.snakemake_delegate as snakemake_delegate

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
)
from rtpipeline.dvh import dvh_for_course


ROOT = Path(__file__).resolve().parents[1]
RUN_COURSE_STAGE = ROOT / "workflow" / "scripts" / "run_course_stage.py"
AGGREGATE_RESULTS = ROOT / "workflow" / "scripts" / "aggregate_results.py"


@pytest.fixture(autouse=True)
def _delegate_parent_segmentation_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep wrapper unit tests able to replace segmentation assessment locally."""

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


def _write_ct_publication(course_dir: Path, *, patient_id: str) -> None:
    from rtpipeline.radiomics_ct_contract import (
        classify_ct_roi,
        disposition_rows_for_arms,
        write_ct_publication_atomic,
    )

    records = disposition_rows_for_arms(
        {
            "modality": "CT",
            "segmentation_source": "Manual",
            "roi_name": "PTV",
            "roi_original_name": "PTV",
            "patient_id": patient_id,
            "course_id": "C1",
            "series_uid": f"series-{patient_id}",
            "mask_identity": f"mask-{patient_id}",
            "stable_roi_identifier": "roi-PTV",
        },
        decision=classify_ct_roi("Manual", "PTV"),
        disposition="success",
        detail="test success",
        failure_kind="none",
        run_identifier="test-run",
        code_revision="test-revision",
        native_voxel_count=120,
        required=True,
        configured_parameter_hashes={
            "primary_resegmented": "test-primary",
            "sensitivity_raw": "test-sensitivity",
        },
        effective_hashes={
            "primary_resegmented": "effective-primary",
            "sensitivity_raw": "effective-sensitivity",
        },
    )
    for record in records:
        record["original_firstorder_Mean"] = 1.0
    write_ct_publication_atomic(
        pd.DataFrame(records), course_dir / "radiomics_ct.xlsx"
    )


def _course_stage_snakemake(tmp_path: Path) -> SimpleNamespace:
    course_dir = tmp_path / "output" / "P1" / "C1"
    segmentation = course_dir / ".segmentation_done"
    segmentation.parent.mkdir(parents=True)
    segmentation.write_text("ok\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"courses": []}\n', encoding="utf-8")
    return SimpleNamespace(
        output=SimpleNamespace(sentinel=str(course_dir / ".dvh_done")),
        log=[str(tmp_path / "logs" / "dvh.log")],
        input=SimpleNamespace(manifest=str(manifest), segmentation=str(segmentation)),
        params=SimpleNamespace(
            root_dir=str(ROOT),
            configfile=str(tmp_path / "config.yaml"),
            radiomics_env="rtpipeline-radiomics",
            python_bin=str(Path(sys.executable).parent),
            python=sys.executable,
            dicom_root=str(tmp_path / "dicom"),
            output_dir=str(tmp_path / "output"),
            logs_dir=str(tmp_path / "logs"),
            stage="dvh",
            custom_structures="",
        ),
        wildcards=SimpleNamespace(patient="P1", course="C1"),
        threads=1,
    )


def test_required_course_stage_failure_returns_nonzero_without_sentinel(
    tmp_path, monkeypatch
):
    """Pre-fix reproducer: a CLI failure wrote a DAG-satisfying failed sentinel."""
    workflow = _course_stage_snakemake(tmp_path)
    sentinel = Path(workflow.output.sentinel)
    sentinel.write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=17),
    )
    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {"status": "ok", "reasons": ["validated test fixture"]},
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})

    assert excinfo.value.code == 17
    assert not sentinel.exists()


def test_required_course_stage_success_publishes_ok_sentinel(tmp_path, monkeypatch):
    workflow = _course_stage_snakemake(tmp_path)
    sentinel = Path(workflow.output.sentinel)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {"status": "ok", "reasons": ["validated test fixture"]},
    )

    runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})

    assert sentinel.read_text(encoding="utf-8") == "ok\n"


def test_required_course_stage_rejects_failed_upstream_without_sentinel(tmp_path):
    workflow = _course_stage_snakemake(tmp_path)
    sentinel = Path(workflow.output.sentinel)
    sentinel.write_text("ok\n", encoding="utf-8")
    Path(workflow.input.segmentation).write_text(
        "failed: see log\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="not successful"):
        runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})

    assert not sentinel.exists()


def test_snakefile_required_shell_stages_do_not_write_failure_sentinels():
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    assert 'echo "failed: see log" > {output.sentinel}' not in snakefile
    assert "skipped: upstream segmentation failed" not in snakefile
    assert 'echo "disabled" > {output.sentinel}' in snakefile
    assert "Required course stage failed; see {log}" in snakefile


def _write_course_inputs(
    course_dir: Path,
    *,
    dvh_sentinel: str | None = "ok\n",
    radiomics_sentinel: str | None = None,
    malformed_dvh: bool = False,
    unreadable_dvh: bool = False,
    plan_only: bool = False,
) -> None:
    course_dir.mkdir(parents=True, exist_ok=True)
    plan, dose = write_synthetic_plan_and_dose(course_dir)
    selected_doses = [] if plan_only else [dose]
    if plan_only:
        dose.unlink()
    write_minimal_course_contract(
        course_dir, selected_plans=[plan], selected_doses=selected_doses
    )
    if dvh_sentinel is not None:
        (course_dir / ".dvh_done").write_text(dvh_sentinel, encoding="utf-8")
    (course_dir / ".qc_done").write_text("ok\n", encoding="utf-8")
    (course_dir / ".custom_models_done").write_text("disabled\n", encoding="utf-8")
    if radiomics_sentinel is not None:
        if radiomics_sentinel.strip() == "ok" and (
            course_dir / "radiomics_ct.parquet"
        ).exists():
            from rtpipeline.radiomics_ct_contract import write_completion_sentinel

            write_completion_sentinel(course_dir, course_dir / ".radiomics_done")
        else:
            (course_dir / ".radiomics_done").write_text(
                radiomics_sentinel, encoding="utf-8"
            )
    if plan_only:
        assert dvh_for_course(course_dir) is None
        if dvh_sentinel is not None:
            (course_dir / ".dvh_done").write_text(
                dvh_sentinel, encoding="utf-8"
            )
    elif unreadable_dvh:
        (course_dir / "dvh_metrics.xlsx").write_bytes(b"not an Excel workbook")
    elif malformed_dvh:
        pd.DataFrame([{"unexpected": 1}]).to_excel(
            course_dir / "dvh_metrics.xlsx", index=False
        )
    else:
        pd.DataFrame(
            [{"ROI_Name": "PTV", "DmeanGy": 42.0}]
        ).to_excel(course_dir / "dvh_metrics.xlsx", index=False)


def _aggregate_snakemake(
    tmp_path: Path, *, radiomics_enabled: bool = False
) -> tuple[SimpleNamespace, dict[str, Path]]:
    output_dir = tmp_path / "output"
    results_dir = output_dir / "_RESULTS"
    manifest = output_dir / "_COURSES" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    courses = [
        {"patient": "P1", "course": "C1", "path": str(output_dir / "P1" / "C1")},
        {"patient": "P2", "course": "C1", "path": str(output_dir / "P2" / "C1")},
    ]
    manifest.write_text(json.dumps({"courses": courses}), encoding="utf-8")

    output_paths = {
        "dvh": results_dir / "dvh_metrics.xlsx",
        "dvh_parquet": results_dir / "dvh_metrics.parquet",
        "fractions": results_dir / "fractions.xlsx",
        "metadata": results_dir / "case_metadata.xlsx",
        "qc": results_dir / "qc_reports.xlsx",
    }
    if radiomics_enabled:
        output_paths.update(
            radiomics=results_dir / "radiomics_ct.xlsx",
            radiomics_mr=results_dir / "radiomics_mr.xlsx",
        )

    log_path = tmp_path / "logs" / "aggregate.log"
    log_path.parent.mkdir(parents=True)
    workflow = SimpleNamespace(
        input=SimpleNamespace(manifest=str(manifest)),
        output=SimpleNamespace(**{key: str(value) for key, value in output_paths.items()}),
        log=[str(tmp_path / "logs" / "aggregate.log")],
        params=SimpleNamespace(
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            radiomics_enabled=radiomics_enabled,
            worker_budget=2,
            auto_worker_budget=2,
            aggregation_threads=1,
        ),
    )
    return workflow, output_paths


def test_snakefile_declares_typed_dvh_aggregate() -> None:
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")

    assert '"dvh_parquet": RESULTS_DIR / "dvh_metrics.parquet"' in snakefile
    assert '"dvh_parquet": str(AGG_OUTPUTS["dvh_parquet"])' in snakefile


@pytest.mark.parametrize(
    "broken_course",
    [
        "missing_sentinel",
        "failed_sentinel",
        "malformed_sentinel",
        "missing_workbook",
        "malformed_workbook",
        "unreadable_workbook",
    ],
)
def test_required_aggregation_rejects_incomplete_course_before_publication(
    tmp_path, broken_course
):
    """Pre-fix reproducer: one valid course was published while another was bad."""
    workflow, outputs = _aggregate_snakemake(tmp_path)
    output_dir = Path(workflow.params.output_dir)
    _write_course_inputs(output_dir / "P1" / "C1")

    kwargs = {}
    if broken_course == "missing_sentinel":
        kwargs["dvh_sentinel"] = None
    elif broken_course == "failed_sentinel":
        kwargs["dvh_sentinel"] = "failed: see log\n"
    elif broken_course == "malformed_sentinel":
        kwargs["dvh_sentinel"] = "success-ish\n"
    elif broken_course == "malformed_workbook":
        kwargs["malformed_dvh"] = True
    elif broken_course == "unreadable_workbook":
        kwargs["unreadable_dvh"] = True
    _write_course_inputs(output_dir / "P2" / "C1", **kwargs)
    if broken_course == "missing_workbook":
        (output_dir / "P2" / "C1" / "dvh_metrics.xlsx").unlink()

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale aggregate")

    with pytest.raises(RuntimeError, match="Required aggregation inputs are incomplete"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    assert not any(path.exists() for path in outputs.values())


def test_noncampaign_aggregation_accepts_plan_only_course_without_workbook(tmp_path):
    workflow, outputs = _aggregate_snakemake(tmp_path)
    output_dir = Path(workflow.params.output_dir)
    _write_course_inputs(output_dir / "P1" / "C1")
    _write_course_inputs(output_dir / "P2" / "C1", plan_only=True)

    runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    aggregate = pd.read_excel(outputs["dvh"])
    assert aggregate[["patient_id", "course_id"]].drop_duplicates().to_dict("records") == [
        {"patient_id": "P1", "course_id": "C1"},
        {"patient_id": "P2", "course_id": "C1"},
    ]
    failure = aggregate.loc[
        (aggregate["patient_id"] == "P2") & (aggregate["course_id"] == "C1")
    ].iloc[0]
    assert failure["row_status"] == "not_computed"
    assert "dose_grid" in failure["failure_reason"]
    assert not (output_dir / "P2" / "C1" / "dvh_metrics.xlsx").exists()


def test_enabled_radiomics_output_is_required_but_mr_remains_optional(tmp_path):
    workflow, outputs = _aggregate_snakemake(tmp_path, radiomics_enabled=True)
    output_dir = Path(workflow.params.output_dir)
    for patient in ("P1", "P2"):
        course_dir = output_dir / patient / "C1"
        _write_ct_publication(course_dir, patient_id=patient)
        _write_course_inputs(course_dir, radiomics_sentinel="ok\n")

    runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    assert outputs["radiomics"].exists()
    assert outputs["radiomics_mr"].exists()
    assert pd.read_excel(outputs["radiomics_mr"]).empty


def test_enabled_radiomics_missing_required_ct_output_blocks_publication(tmp_path):
    workflow, outputs = _aggregate_snakemake(tmp_path, radiomics_enabled=True)
    output_dir = Path(workflow.params.output_dir)
    for patient in ("P1", "P2"):
        course_dir = output_dir / patient / "C1"
        _write_course_inputs(course_dir, radiomics_sentinel="ok\n")
        if patient == "P1":
            _write_ct_publication(course_dir, patient_id=patient)

    with pytest.raises(RuntimeError, match="radiomics_ct.parquet is unreadable"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    assert not any(path.exists() for path in outputs.values())
