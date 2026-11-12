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


@pytest.mark.parametrize("operation", ["begin", "finish"])
@pytest.mark.parametrize("corruption", ['{"state":', '[]', '{}',
                                         '{"state":"started","binding":null,"attempt_number":"bad"}',
                                         'io_error'])
def test_corrupt_campaign_attempt_fails_with_durable_fixed_code(tmp_path, monkeypatch, operation, corruption):
    import errno
    from rtpipeline.durable_progress import DurableProgressError
    path = ledger._attempt_path(tmp_path, "SYNTH", "COURSE", "dvh")
    path.parent.mkdir(parents=True)
    path.write_text(corruption)
    original = Path.read_text
    if corruption == "io_error":
        def read(target, *args, **kwargs):
            if target == path:
                raise OSError(errno.EIO, "synthetic attempt read error")
            return original(target, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(DurableProgressError, match="DVH_LEDGER_ATTEMPT_CORRUPT"):
        if operation == "begin":
            ledger.begin_attempt(tmp_path, "SYNTH", "COURSE", "dvh")
        else:
            ledger.finish_attempt(tmp_path, "SYNTH", "COURSE", "dvh", "committed")
    failed = json.loads(original(path))
    assert failed["state"] == "failed" and failed["reason_code"] == "DVH_LEDGER_ATTEMPT_CORRUPT"
    assert failed["patient"] == "SYNTH" and failed["course"] == "COURSE"
    assert not (tmp_path / "SYNTH/COURSE/.dvh_done").exists()


@pytest.mark.parametrize("filename", [".dvh/attempt.json", ".dvh_done"])
@pytest.mark.parametrize("corruption", ['{"state":', '[]', '{}', 'io_error'])
def test_record_never_accepts_corrupt_dvh_evidence(tmp_path, monkeypatch, filename, corruption):
    import errno
    from rtpipeline.durable_progress import DurableProgressError
    path = tmp_path / "SYNTH/COURSE" / filename
    path.parent.mkdir(parents=True)
    path.write_text(corruption)
    original = Path.read_text
    if corruption == "io_error":
        def read(target, *args, **kwargs):
            if target == path:
                raise OSError(errno.EACCES, "synthetic evidence read error")
            return original(target, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", read)
    if filename == ".dvh_done":
        with pytest.raises(DurableProgressError, match="DVH_LEDGER_EVIDENCE_UNREADABLE"):
            ledger.record(tmp_path, "SYNTH", "COURSE", "dvh", ledger.STATUS_OK)
        code = "DVH_LEDGER_EVIDENCE_UNREADABLE"
        attempt = ledger._attempt_path(tmp_path, "SYNTH", "COURSE", "dvh")
    else:
        ledger.record(tmp_path, "SYNTH", "COURSE", "dvh", ledger.STATUS_FAILED)
        code = "DVH_ATTEMPT_CORRUPT"
        attempt = path
    failed = json.loads(original(attempt))
    assert failed["state"] == "failed" and failed["reason_code"] == code
    target = ledger.records_dir(tmp_path) / "SYNTH__COURSE__dvh.json"
    entry = json.loads(original(target))
    assert entry["status"] == "failed" and entry["attempt_state"] == "failed"
    assert entry["reason_code"] == code


def test_missing_dvh_completion_evidence_is_not_recorded_as_success(tmp_path):
    from rtpipeline.durable_progress import DurableProgressError
    with pytest.raises(DurableProgressError, match="DVH_LEDGER_EVIDENCE_UNREADABLE"):
        ledger.record(tmp_path, "SYNTH", "COURSE", "dvh", ledger.STATUS_OK)
    entry = json.loads((ledger.records_dir(tmp_path) / "SYNTH__COURSE__dvh.json").read_text())
    assert entry["status"] == "failed"
