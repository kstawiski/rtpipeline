"""Synthetic regression for worker-budget changes during workflow resume."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAKEMAKE = "/home/konrad/rtpipeline_release/b29a324/envs/rtpipeline-runtime/bin/snakemake"


def _run(workdir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [SNAKEMAKE, "--cores", "8", "--configfile", "config.yaml", "--quiet"],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_fixture(workdir: Path, *, legacy_param: bool) -> None:
    params = '        workflow_threads=config["max_workers"]\n' if legacy_param else ""
    (workdir / "Snakefile").write_text(
        """rule all:
    input: "course.done"

rule radiomics_course:
    output: "course.done"
    threads: int(config["max_workers"])
    params:
"""
        + params
        + """    shell:
        "printf 'threads=%s\\n' {threads} > {output}"
""",
        encoding="utf-8",
    )


def _run_resume_case(tmp_path: Path, *, legacy_param: bool) -> tuple[str, str]:
    workdir = tmp_path / ("legacy" if legacy_param else "fixed")
    workdir.mkdir()
    _write_fixture(workdir, legacy_param=legacy_param)
    config = workdir / "config.yaml"
    config.write_text("max_workers: 2\n", encoding="utf-8")
    first = _run(workdir)
    assert first.returncode == 0, first.stderr
    output = workdir / "course.done"
    before = output.read_text(encoding="utf-8")
    config.write_text("max_workers: 4\n", encoding="utf-8")
    second = _run(workdir)
    assert second.returncode == 0, second.stderr
    return before, output.read_text(encoding="utf-8")


def test_worker_budget_change_does_not_invalidate_completed_course(tmp_path: Path) -> None:
    """Changing only the operational budget reuses a completed course.

    The legacy synthetic rule has the same ``params.workflow_threads`` field
    that caused the production reruns; Snakemake reruns it after 2 -> 4.
    The fixed rule keeps the real ``threads`` limit while omitting that
    parameter, so the completed output remains untouched.
    """
    legacy_before, legacy_after = _run_resume_case(tmp_path, legacy_param=True)
    fixed_before, fixed_after = _run_resume_case(tmp_path, legacy_param=False)

    assert legacy_before == "threads=2\n"
    assert legacy_after == "threads=4\n"
    assert fixed_before == "threads=2\n"
    assert fixed_after == fixed_before

    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    radiomics_rules = snakefile.split("rule radiomics_course:")[1:]
    assert len(radiomics_rules) == 2
    assert all("workflow_threads" not in rule for rule in radiomics_rules)
    stage_source = (ROOT / "workflow/scripts/run_course_stage.py").read_text(
        encoding="utf-8"
    )
    assert "int(workflow.threads)" in stage_source
