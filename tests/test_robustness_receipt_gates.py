"""Exercise the actual shell gates against writer-produced receipts."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rtpipeline.config_dependencies import materialize_stage_dependency
from rtpipeline.stage_completion import write_stage_completion_sentinel

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage,status", [("dvh", "ok"), ("dvh", "disabled"), ("radiomics", "ok"), ("radiomics", "not_applicable")])
@pytest.mark.parametrize("representation", ["structured", "legacy"])
def test_robustness_shell_gates(tmp_path, stage, status, representation):
    course = tmp_path / "p" / "c"
    course.mkdir(parents=True)
    receipt = course / f".{stage}_done"
    if representation == "legacy":
        receipt.write_text(status + "\n")
    elif stage == "dvh":
        if status == "ok":
            (course / "metadata").mkdir()
            (course / "metadata" / "dvh_qc.json").write_text('{"status":"ok"}')
            (course / "dvh_metrics.parquet").write_bytes(b"test-output")
        dependency = materialize_stage_dependency(tmp_path / "deps", stage, {"enabled": status == "ok"})
        write_stage_completion_sentinel(course, receipt, stage=stage, status=status, configuration_dependency=dependency)
        assert len(receipt.read_text().splitlines()) > 1
    else:
        receipt.write_text(json.dumps({"status": status}, indent=2))
    gates = [line.strip() for line in (ROOT / "Snakefile").read_text().splitlines()
             if line.strip().startswith("if ") and "{input." + stage + "}" in line]
    assert len(gates) == 2
    for gate in gates:
        rendered = gate.format(input=SimpleNamespace(**{stage: receipt}), params=SimpleNamespace(root_dir=ROOT, python=sys.executable))
        result = subprocess.run(["bash", "-c", rendered + "\nexit 41\nfi"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        receipt.write_text('{"status":"failed"}')
        rejected = subprocess.run(["bash", "-c", rendered + "\nexit 41\nfi"], capture_output=True, text=True)
        assert rejected.returncode == 41
        # Restore the original representation for the other workflow branch.
        if representation == "legacy":
            receipt.write_text(status + "\n")
        elif stage == "dvh":
            write_stage_completion_sentinel(course, receipt, stage=stage, status=status, configuration_dependency=dependency)
        else:
            receipt.write_text(json.dumps({"status": status}, indent=2))


@pytest.mark.parametrize("status", ["disabled", "legacy", "missing"])
def test_configured_disabled_segmentation_requires_bound_receipt(tmp_path, status):
    import importlib.util
    script = ROOT / "workflow" / "scripts" / "run_course_stage.py"
    spec = importlib.util.spec_from_file_location("course_stage_gate_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    course = tmp_path / "p" / "c"
    course.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text("segmentation:\n  enabled: false\n")
    workflow = SimpleNamespace(params=SimpleNamespace(configfile=config))
    receipt = course / ".segmentation_done"
    if status == "disabled":
        dependency = materialize_stage_dependency(tmp_path / "deps", "segmentation", {"enabled": False})
        write_stage_completion_sentinel(course, receipt, stage="segmentation", status="disabled", configuration_dependency=dependency)
        assert module._require_segmentation_content(workflow, course) == "configuration_disabled"
    else:
        if status == "legacy":
            receipt.write_text("disabled\n")
        with pytest.raises(ValueError):
            module._require_segmentation_content(workflow, course)
