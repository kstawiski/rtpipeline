"""Campaign mode must finish the DAG without weakening per-course fail-closed.

A single-cohort run should abort on a failing course: the operator sees it at
once and nothing downstream consumes a partial result. Across thousands of
courses that same behaviour makes the run unfinishable, because one malformed
series stops every remaining patient and blocks aggregation.

Campaign mode changes only where the failure is recorded, never whether the
course is trusted. The failing course writes ``failed`` into its own sentinel
and exits zero so the DAG completes; the course stays closed because the
upstream-status check refuses to run any dependent stage for it, and
aggregation counts only ``ok`` courses and records every exclusion with its
reason.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import rtpipeline.segmentation as segmentation
import rtpipeline.snakemake_delegate as snakemake_delegate
from rtpipeline.config_dependencies import materialize_stage_dependency

import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_COURSE_STAGE = ROOT / "workflow" / "scripts" / "run_course_stage.py"

_MISSING_PYTHON = "/nonexistent/python-that-cannot-be-launched"


@pytest.fixture(autouse=True)
def _validated_segmentation_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep campaign-wrapper tests focused on stage launch and close semantics."""

    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {"status": "ok", "reasons": ["validated test fixture"]},
    )
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


def _stub_failing_python(tmp_path: Path) -> str:
    """A launchable interpreter stub that always exits non-zero.

    Exercises the "stage ran and failed" path deterministically, as distinct
    from the "stage could not be launched" path covered by _MISSING_PYTHON.
    """
    stub = tmp_path / "failing-python"
    stub.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    stub.chmod(0o755)
    return str(stub)


def _workflow(tmp_path: Path, *, campaign_mode: bool, python: str | None = None):
    course_dir = tmp_path / "PT001" / "COURSE_A"
    course_dir.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"courses": []}', encoding="utf-8")
    segmentation = course_dir / ".segmentation_done"
    segmentation.write_text("ok\n", encoding="utf-8")
    configuration = materialize_stage_dependency(
        tmp_path / "dependencies", "dvh", {"test": True}
    )

    return SimpleNamespace(
        input=SimpleNamespace(
            manifest=str(manifest),
            segmentation=str(segmentation),
            configuration=str(configuration),
        ),
        output=SimpleNamespace(sentinel=str(course_dir / ".dvh_done")),
        log=[str(logs / "dvh.log")],
        threads=1,
        wildcards=SimpleNamespace(patient="PT001", course="COURSE_A"),
        params=SimpleNamespace(
            stage="dvh",
            campaign_mode=campaign_mode,
            python=python or _stub_failing_python(tmp_path),
            root_dir=str(ROOT),
            configfile=str(tmp_path / "config.yaml"),
            radiomics_env="rtpipeline-radiomics",
            python_bin=str(tmp_path / "bin"),
            dicom_root=str(tmp_path / "dicom"),
            output_dir=str(tmp_path),
            logs_dir=str(logs),
            custom_structures="",
        ),
    )


def _run(workflow):
    return runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})


def _set_stage_configuration(workflow, tmp_path: Path, stage: str) -> None:
    configuration_stage = {
        "segmentation_custom": "custom-models",
        "crop_ct": "crop-ct",
    }.get(stage, stage)
    workflow.input.configuration = str(
        materialize_stage_dependency(
            tmp_path / "dependencies", configuration_stage, {"test": True}
        )
    )


def test_campaign_mode_contains_a_launch_failure(tmp_path):
    """A missing interpreter must close the course, not abort the campaign."""
    workflow = _workflow(tmp_path, campaign_mode=True, python=_MISSING_PYTHON)

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code == 0
    assert Path(workflow.output.sentinel).read_text(encoding="utf-8").strip() == "failed"


def test_strict_mode_still_aborts_on_a_launch_failure(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=False, python=_MISSING_PYTHON)

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code not in (0, None)
    assert not Path(workflow.output.sentinel).exists()


def test_strict_mode_still_aborts_and_leaves_no_sentinel(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=False)
    sentinel = Path(workflow.output.sentinel)

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code not in (0, None)
    assert not sentinel.exists(), "a failed course must not leave a success sentinel"


def test_campaign_mode_records_failure_and_lets_the_campaign_continue(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=True)
    sentinel = Path(workflow.output.sentinel)

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    # Exit zero so Snakemake proceeds to the remaining courses...
    assert excinfo.value.code == 0
    # ...but the course is explicitly marked failed, never ok.
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "failed"


def test_campaign_mode_never_writes_ok_for_a_failed_course(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=True)
    sentinel = Path(workflow.output.sentinel)

    with pytest.raises(SystemExit):
        _run(workflow)

    assert sentinel.read_text(encoding="utf-8").strip() != "ok"


def test_campaign_mode_writes_a_ledger_record(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=True)

    with pytest.raises(SystemExit):
        _run(workflow)

    records = list((tmp_path / "_campaign_ledger" / "records").glob("*.json"))
    assert records, "a closed course must leave a ledger record"
    import json

    entry = json.loads(records[0].read_text(encoding="utf-8"))
    assert entry["status"] == "failed"
    assert entry["patient"] == "PT001"
    assert entry["stage"] == "dvh"


def test_custom_segmentation_campaign_failure_is_isolated_per_course(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=True)
    workflow.params.stage = "segmentation_custom"
    _set_stage_configuration(workflow, tmp_path, "segmentation_custom")
    workflow.output.sentinel = str(
        tmp_path / "PT001" / "COURSE_A" / ".custom_models_done"
    )

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code == 0
    assert Path(workflow.output.sentinel).read_text(encoding="utf-8").strip() == "failed"
    records = list((tmp_path / "_campaign_ledger" / "records").glob("*.json"))
    import json

    entries = [json.loads(path.read_text(encoding="utf-8")) for path in records]
    entry = next(row for row in entries if row["stage"] == "segmentation_custom")
    assert entry["status"] == "failed"


def test_disabled_custom_segmentation_publishes_disabled_without_launch(tmp_path):
    workflow = _workflow(tmp_path, campaign_mode=True, python=_MISSING_PYTHON)
    workflow.params.stage = "segmentation_custom"
    _set_stage_configuration(workflow, tmp_path, "segmentation_custom")
    workflow.params.enabled = False
    workflow.output.sentinel = str(
        tmp_path / "PT001" / "COURSE_A" / ".custom_models_done"
    )

    _run(workflow)

    import json

    payload = json.loads(Path(workflow.output.sentinel).read_text(encoding="utf-8"))
    assert payload["status"] == "disabled"
    assert payload["schema"] == "rtpipeline-stage-completion-v1"


def test_disabled_crop_skips_rich_segmentation_validation(tmp_path, monkeypatch):
    """A disabled crop must not load a newer contract schema before its no-op."""

    workflow = _workflow(tmp_path, campaign_mode=True, python=_MISSING_PYTHON)
    workflow.params.stage = "crop_ct"
    _set_stage_configuration(workflow, tmp_path, "crop_ct")
    workflow.params.enabled = False
    workflow.output.sentinel = str(
        tmp_path / "PT001" / "COURSE_A" / ".crop_ct_done"
    )
    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {
            "status": "failed",
            "reasons": ["unsupported course contract version 3; expected 4"],
        },
    )

    _run(workflow)

    import json

    assert json.loads(Path(workflow.output.sentinel).read_text(encoding="utf-8"))[
        "status"
    ] == "disabled"
    log_path = Path(workflow.log[0])
    assert not log_path.exists() or log_path.stat().st_size == 0


def test_disabled_crop_still_refuses_failed_segmentation_sentinel(tmp_path):
    """The no-op may skip rich validation but cannot bypass a failed prerequisite."""

    workflow = _workflow(tmp_path, campaign_mode=True, python=_MISSING_PYTHON)
    workflow.params.stage = "crop_ct"
    _set_stage_configuration(workflow, tmp_path, "crop_ct")
    workflow.params.enabled = False
    workflow.output.sentinel = str(
        tmp_path / "PT001" / "COURSE_A" / ".crop_ct_done"
    )
    Path(workflow.input.segmentation).write_text("failed\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code == 0
    assert Path(workflow.output.sentinel).read_text(encoding="utf-8").strip() == "failed"


def test_no_planning_ct_propagates_as_not_applicable(tmp_path, monkeypatch):
    workflow = _workflow(tmp_path, campaign_mode=True, python=_MISSING_PYTHON)
    workflow.input.custom = str(
        tmp_path / "PT001" / "COURSE_A" / ".custom_models_done"
    )
    workflow.input.crop = str(tmp_path / "PT001" / "COURSE_A" / ".crop_ct_done")
    Path(workflow.input.segmentation).write_text("disabled\n", encoding="utf-8")
    Path(workflow.input.custom).write_text("disabled\n", encoding="utf-8")
    Path(workflow.input.crop).write_text("disabled\n", encoding="utf-8")
    monkeypatch.setattr(
        segmentation,
        "assess_course_segmentation",
        lambda _course_dir: {
            "status": "disabled",
            "reasons": ["authoritative course contract declares no planning CT"],
        },
    )

    _run(workflow)

    import json

    assert json.loads(Path(workflow.output.sentinel).read_text(encoding="utf-8"))[
        "status"
    ] == "disabled"


def test_snakefile_routes_custom_segmentation_through_campaign_wrapper():
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    start = snakefile.index("rule segmentation_custom_models:")
    end = snakefile.index("rule crop_ct_course:", start)
    section = snakefile[start:end]

    assert section.count('script:\n            "workflow/scripts/run_course_stage.py"') == 2
    assert section.count("campaign_mode=CAMPAIGN_MODE") == 2


def test_segmentation_rule_uses_structured_campaign_wrapper():
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")

    start = snakefile.index("rule segmentation_course:")
    end = snakefile.index("_custom_params_lambda", start)
    section = snakefile[start:end]
    assert 'script:\n        "workflow/scripts/run_course_stage.py"' in section
    assert 'stage="segmentation"' in section
    assert "grep -Eqx" not in section


def test_a_failed_course_still_closes_its_dependent_stages(tmp_path):
    """The point of fail-closed: downstream must refuse a failed upstream."""
    workflow = _workflow(tmp_path, campaign_mode=True)
    Path(workflow.input.segmentation).write_text("failed\n", encoding="utf-8")
    sentinel = Path(workflow.output.sentinel)

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code == 0  # campaign continues
    assert sentinel.read_text(encoding="utf-8").strip() == "failed"  # course does not


def test_radiomics_campaign_failure_is_isolated_and_preserves_cli_options(tmp_path):
    argv_path = tmp_path / "radiomics-argv.txt"
    env_path = tmp_path / "radiomics-env.txt"
    stub = tmp_path / "capturing-failing-python"
    stub.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARGV_PATH\"\n"
        "printf '%s\\n' \"$RTPIPELINE_MAX_WORKERS,$OMP_NUM_THREADS\" > \"$ENV_PATH\"\n"
        "exit 3\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    workflow = _workflow(tmp_path, campaign_mode=True, python=str(stub))
    workflow.params.stage = "radiomics"
    workflow.params.extra_args = '--radiomics-params "/tmp/params with space.yaml" --no-resample'
    workflow.threads = 4
    workflow.output.sentinel = str(tmp_path / "PT001" / "COURSE_A" / ".radiomics_done")
    workflow.input.custom = str(tmp_path / "PT001" / "COURSE_A" / ".segmentation_custom_done")
    workflow.input.crop = str(tmp_path / "PT001" / "COURSE_A" / ".crop_ct_done")
    Path(workflow.input.custom).write_text("disabled\n", encoding="utf-8")
    Path(workflow.input.crop).write_text("ok\n", encoding="utf-8")

    old_argv = os.environ.get("ARGV_PATH")
    old_env = os.environ.get("ENV_PATH")
    os.environ["ARGV_PATH"] = str(argv_path)
    os.environ["ENV_PATH"] = str(env_path)
    try:
        with pytest.raises(SystemExit) as excinfo:
            _run(workflow)
    finally:
        if old_argv is None:
            os.environ.pop("ARGV_PATH", None)
        else:
            os.environ["ARGV_PATH"] = old_argv
        if old_env is None:
            os.environ.pop("ENV_PATH", None)
        else:
            os.environ["ENV_PATH"] = old_env

    assert excinfo.value.code == 0
    assert Path(workflow.output.sentinel).read_text(encoding="utf-8").strip() == "failed"
    arguments = argv_path.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--radiomics-params") + 1] == "/tmp/params with space.yaml"
    assert "--no-resample" in arguments
    assert env_path.read_text(encoding="utf-8").strip() == "4,1"


def test_strict_mode_surfaces_the_upstream_error_unflattened(tmp_path):
    """Existing single-cohort behaviour is unchanged by campaign mode."""
    workflow = _workflow(tmp_path, campaign_mode=False)
    Path(workflow.input.segmentation).write_text("failed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not successful"):
        _run(workflow)

    assert not Path(workflow.output.sentinel).exists()
