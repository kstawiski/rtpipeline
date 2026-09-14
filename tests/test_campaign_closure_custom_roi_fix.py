"""F-A/F-B regression tests for the campaign-closure custom-ROI fix.

Covers: indeterminate custom ROI is course-fatal only when
ANALYSIS_REQUIRED; robustness campaign-mode closure records a ledger row
with reason upstream_radiomics_failed and publishes a failed sentinel;
aggregate stage-alias maps robustness sentinels.
"""
from pathlib import Path
import subprocess
import sys

from rtpipeline.roi_requiredness import (
    Requiredness,
    indeterminate_custom_roi_fails_course,
)


def test_indeterminate_fails_only_when_required():
    assert indeterminate_custom_roi_fails_course(Requiredness.ANALYSIS_REQUIRED) is True
    assert indeterminate_custom_roi_fails_course(Requiredness.INVENTORY_ONLY) is False
    assert indeterminate_custom_roi_fails_course(Requiredness.ANALYSIS_OPTIONAL) is False


def test_robustness_alias_maps_sentinel():
    # workflow_aggregate is a Snakemake-parsed module (imports snakemake at
    # top level), so verify the mapping by source instead of import.
    source = (
        Path(__file__).resolve().parents[1] / "rtpipeline" / "workflow_aggregate.py"
    ).read_text(encoding="utf-8")
    assert '((".radiomics_robustness_done",), {"radiomics_robustness"})' in source


def test_close_robustness_upstream_writes_ledger_and_sentinel(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "campaign_ledger_under_test",
        str(Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "campaign_ledger.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_dir = tmp_path / "Output"
    sentinel = tmp_path / "P1" / "C1" / ".radiomics_robustness_done"
    record_path = module.close_robustness_upstream_failure(
        output_dir, "P1", "C1", sentinel, log_path=str(tmp_path / "course.log")
    )
    assert record_path.is_file()
    import json

    entry = json.loads(record_path.read_text(encoding="utf-8"))
    assert entry["stage"] == "radiomics_robustness"
    assert entry["status"] == "failed"
    assert entry["detail"] == module.UPSTREAM_RADIOMICS_FAILED
    assert sentinel.read_text(encoding="utf-8").strip() == "failed"


def test_close_robustness_cli_exit_zero(tmp_path):
    script = Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "campaign_ledger.py"
    output_dir = tmp_path / "Output"
    sentinel = tmp_path / "P1" / "C1" / ".radiomics_robustness_done"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "close-robustness-upstream",
            "--output-dir",
            str(output_dir),
            "--patient",
            "P1",
            "--course",
            "C1",
            "--sentinel",
            str(sentinel),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8").strip() == "failed"
