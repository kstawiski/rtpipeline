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


def _ledger_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "campaign_ledger_under_test",
        str(Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "campaign_ledger.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failed_receipt(tmp_path, patient="P1", course="C1"):
    """A revalidatable failed-extraction receipt via the real writer."""
    import json

    from rtpipeline.robustness_completion import (
        write_robustness_completion_sentinel,
    )

    course_dir = tmp_path / "Output" / patient / course
    dispositions = course_dir / "metadata" / "radiomics_robustness_source_dispositions.json"
    dispositions.parent.mkdir(parents=True, exist_ok=True)
    dispositions.write_text(
        json.dumps({"rows": [], "run": "failure-run-1"}), encoding="utf-8"
    )
    sentinel = course_dir / ".radiomics_robustness_done"
    write_robustness_completion_sentinel(
        sentinel,
        course_dir,
        patient_id=patient,
        course_id=course,
        run_identifier="failure-run-1",
        measurement_outcome="failed_extraction",
        output_name="radiomics_robustness_ct.parquet",
        dispositions_path=dispositions,
        measured_output=None,
        source_disposition_count=0,
        effective_configuration_sha256="0" * 64,
    )
    return sentinel


def test_close_robustness_failed_records_ledger_and_keeps_receipt(tmp_path):
    import json

    module = _ledger_module()
    sentinel = _failed_receipt(tmp_path)
    before = sentinel.read_bytes()
    record_path = module.close_robustness_failed_extraction(
        tmp_path / "Output", "P1", "C1", sentinel,
        log_path=str(tmp_path / "course.log"),
    )
    assert record_path.is_file()
    entry = json.loads(record_path.read_text(encoding="utf-8"))
    assert entry["stage"] == "radiomics_robustness"
    assert entry["status"] == "failed"
    assert entry["detail"] == module.ROBUSTNESS_EXTRACTION_FAILED
    # The receipt is aggregation evidence: the close must not overwrite it.
    assert sentinel.read_bytes() == before


def test_close_robustness_failed_refuses_missing_receipt(tmp_path):
    import pytest

    module = _ledger_module()
    with pytest.raises(RuntimeError, match="no revalidatable failure receipt"):
        module.close_robustness_failed_extraction(
            tmp_path / "Output", "P1", "C1",
            tmp_path / "Output" / "P1" / "C1" / ".radiomics_robustness_done",
        )


def test_close_robustness_failed_refuses_non_failure_receipt(tmp_path):
    """A revalidatable receipt with any other outcome is not a failure."""
    import json

    import pytest

    from rtpipeline.robustness_completion import (
        write_robustness_completion_sentinel,
    )

    module = _ledger_module()
    course_dir = tmp_path / "Output" / "P1" / "C1"
    dispositions = course_dir / "metadata" / "radiomics_robustness_source_dispositions.json"
    dispositions.parent.mkdir(parents=True, exist_ok=True)
    dispositions.write_text(json.dumps({"rows": []}), encoding="utf-8")
    sentinel = course_dir / ".radiomics_robustness_done"
    write_robustness_completion_sentinel(
        sentinel,
        course_dir,
        patient_id="P1",
        course_id="C1",
        run_identifier="source-only-run-1",
        measurement_outcome="source_only_nonvolumetric",
        output_name="radiomics_robustness_ct.parquet",
        dispositions_path=dispositions,
        measured_output=None,
        source_disposition_count=0,
        effective_configuration_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="not a recorded extraction failure"):
        module.close_robustness_failed_extraction(
            tmp_path / "Output", "P1", "C1", sentinel
        )


def test_close_robustness_failed_carries_scoped_detail(tmp_path):
    import json

    module = _ledger_module()
    sentinel = _failed_receipt(tmp_path)
    record_path = module.close_robustness_failed_extraction(
        tmp_path / "Output",
        "P1",
        "C1",
        sentinel,
        detail="robustness_extraction_failed:MemoryError",
    )
    entry = json.loads(record_path.read_text(encoding="utf-8"))
    assert entry["detail"] == "robustness_extraction_failed:MemoryError"


def test_close_robustness_failed_refuses_unscoped_detail(tmp_path):
    import pytest

    module = _ledger_module()
    sentinel = _failed_receipt(tmp_path)
    with pytest.raises(RuntimeError, match="unscoped detail"):
        module.close_robustness_failed_extraction(
            tmp_path / "Output", "P1", "C1", sentinel, detail="boom"
        )
    import pytest

    module = _ledger_module()
    sentinel = _failed_receipt(tmp_path, patient="P1", course="C1")
    with pytest.raises(RuntimeError, match="not P9/C9"):
        module.close_robustness_failed_extraction(
            tmp_path / "Output", "P9", "C9", sentinel
        )


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
