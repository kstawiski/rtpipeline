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
            import pandas as pd
            from course_contract_test_utils import (write_synthetic_plan_and_dose,
                write_minimal_course_contract, write_bound_aggregation_sentinels)
            plan, dose = write_synthetic_plan_and_dose(course)
            write_minimal_course_contract(course, selected_plans=[plan], selected_doses=[dose])
            pd.DataFrame([{"ROI_Name": "PTV", "DmeanGy": 1.0}]).to_excel(course / "dvh_metrics.xlsx", index=False)
            write_bound_aggregation_sentinels(course, tmp_path / "upstream-deps")
        dependency = materialize_stage_dependency(tmp_path / "deps", stage, {"enabled": status == "ok"})
        write_stage_completion_sentinel(course, receipt, stage=stage, status=status, configuration_dependency=dependency)
        assert json.loads(receipt.read_text())["status"] == status
    else:
        receipt.write_text(json.dumps({"status": status}, indent=2))
    # Parse a copy: a status-gate test must not corrupt an immutable DVH
    # generation while testing rejection of an unrelated failed receipt.
    published = receipt.read_bytes()
    receipt = tmp_path / "gate-input.json"
    receipt.write_bytes(published)
    if representation == "structured":
        receipt.write_text(json.dumps(json.loads(published), indent=2))
        published = receipt.read_bytes()
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
        receipt.write_bytes(published)


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
