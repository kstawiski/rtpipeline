"""Attempt evidence remains independent of campaign DAG closure."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dvh_campaign_ledger", ROOT / "workflow/scripts/campaign_ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def test_start_without_commit_is_interrupted_not_completed(tmp_path):
    first = ledger.begin_attempt(tmp_path, "SYNTH", "COURSE", "dvh")
    assert first["state"] == "started" and first["attempt_number"] == 1
    assert not list(ledger.records_dir(tmp_path).glob("*.json"))
    second = ledger.begin_attempt(tmp_path, "SYNTH", "COURSE", "dvh")
    assert second["previous_state"] == "interrupted"
    assert second["previous_reason_code"] == "DVH_INTERRUPTED"
    assert second["attempt_number"] == 2
    ledger.finish_attempt(tmp_path, "SYNTH", "COURSE", "dvh", "committed")
    path = ledger._attempt_path(tmp_path, "SYNTH", "COURSE", "dvh")
    assert json.loads(path.read_text())["state"] == "committed"
    assert not list(ledger.records_dir(tmp_path).glob("*.json"))


def test_failure_has_fixed_code_without_a_success_marker(tmp_path):
    ledger.begin_attempt(tmp_path, "SYNTH", "COURSE", "dvh")
    ledger.finish_attempt(tmp_path, "SYNTH", "COURSE", "dvh", "failed", "DVH_CHILD_FAILED")
    payload = json.loads(ledger._attempt_path(tmp_path, "SYNTH", "COURSE", "dvh").read_text())
    assert payload["state"] == "failed" and payload["reason_code"] == "DVH_CHILD_FAILED"
    assert not list(tmp_path.rglob(".dvh_done"))


def test_durable_attempt_write_failure_propagates(tmp_path, monkeypatch):
    from rtpipeline import durable_progress
    def refused(*args, **kwargs):
        raise OSError("synthetic persistence failure")
    monkeypatch.setattr(durable_progress, "final_result_writer", refused)
    with pytest.raises(OSError):
        ledger.begin_attempt(tmp_path, "SYNTH", "COURSE", "dvh")
    assert not list(ledger.records_dir(tmp_path).glob("*.json"))


@pytest.mark.parametrize("legacy", ["ok\n", "disabled\n", "[]", "true", '{"status":"ok"}'])
def test_preflight_records_legacy_reason_before_dag_skip(tmp_path, legacy):
    from rtpipeline import dvh_generation as gen
    from rtpipeline.durable_progress import ProgressJournal
    course = tmp_path / "Output/SYNTH/COURSE"
    course.mkdir(parents=True)
    (course / ".dvh_done").write_text(legacy)
    config = tmp_path / "config.yaml"
    config.write_text("ct_cropping:\n  use_cropped_for_dvh: true\n")
    result = gen.preflight(tmp_path / "Output", config)
    assert result["courses"] == [{"course": str(course), "reusable": False, "reason_code": "DVH_LEGACY_UNBOUND"}]
    assert not (course / ".dvh_done").exists()
    entries = ProgressJournal.read(course / ".dvh/progress.jsonl").records
    assert entries[-1]["body"]["reason_code"] == "DVH_LEGACY_UNBOUND"
