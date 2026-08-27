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

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_COURSE_STAGE = ROOT / "workflow" / "scripts" / "run_course_stage.py"

_MISSING_PYTHON = "/nonexistent/python-that-cannot-be-launched"


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

    return SimpleNamespace(
        input=SimpleNamespace(manifest=str(manifest), segmentation=str(segmentation)),
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


def test_a_failed_course_still_closes_its_dependent_stages(tmp_path):
    """The point of fail-closed: downstream must refuse a failed upstream."""
    workflow = _workflow(tmp_path, campaign_mode=True)
    Path(workflow.input.segmentation).write_text("failed\n", encoding="utf-8")
    sentinel = Path(workflow.output.sentinel)

    with pytest.raises(SystemExit) as excinfo:
        _run(workflow)

    assert excinfo.value.code == 0  # campaign continues
    assert sentinel.read_text(encoding="utf-8").strip() == "failed"  # course does not


def test_strict_mode_surfaces_the_upstream_error_unflattened(tmp_path):
    """Existing single-cohort behaviour is unchanged by campaign mode."""
    workflow = _workflow(tmp_path, campaign_mode=False)
    Path(workflow.input.segmentation).write_text("failed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not successful"):
        _run(workflow)

    assert not Path(workflow.output.sentinel).exists()
