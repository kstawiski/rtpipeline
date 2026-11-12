"""Synthetic durable I/O tests; no clinical fixtures, network or pipeline calls."""
from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from rtpipeline import durable_progress as dp

ROOT = Path(__file__).resolve().parents[1]
BINDING = dp.Binding("a" * 64, "b" * 64, "c" * 64)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("network forbidden in synthetic durable progress tests")
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refused)


@pytest.fixture(autouse=True)
def remove_synthetic_outputs(tmp_path):
    yield
    shutil.rmtree(tmp_path)


def child_code(code):
    # Children are isolated, dependency-free interpreters with no bytecode writes.
    # They may only import the package and operate on the passed synthetic paths.
    return ("import sys\nsys.path.insert(0, " + repr(str(ROOT)) + ")\n"
            "def boundary(event, args):\n"
            "    if event.startswith('socket.'):\n"
            "        raise RuntimeError('NETWORK_FORBIDDEN')\n"
            "sys.addaudithook(boundary)\n" + code)


def run_child(code):
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", child_code(code)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def scripted_writer(monkeypatch):
    """Replace only the private writer command with a controllable subprocess."""
    real_popen = subprocess.Popen
    processes = []
    def install(code):
        def popen(command, **kwargs):
            process = real_popen([sys.executable, "-I", "-S", "-B", "-c", child_code(code)], **kwargs)
            processes.append(process)
            return process
        monkeypatch.setattr(dp.subprocess, "Popen", popen)
        return processes
    yield install
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()


def transaction(cache):
    directories = list(cache.root.iterdir())
    assert len(directories) == 1
    return directories[0]


def test_journal_round_trip_readback_reopen_and_fsync(tmp_path, monkeypatch):
    path = tmp_path / "progress.jsonl"
    syncs = []
    real = os.fsync
    def synced(fd):
        syncs.append(fd)
        return real(fd)
    monkeypatch.setattr(os, "fsync", synced)
    with dp.ProgressJournal(path) as journal:
        syncs.clear()
        first = journal.append({"item": "SYNTH_A", "value": "Łódź"})
        second = journal.append({"item": "SYNTH_B", "value": 2})
        assert len(syncs) == 2
        assert dp.ProgressJournal.read(path).records == [first, second]
    with dp.ProgressJournal(path) as journal:
        third = journal.append({"event": "done"})
    read = dp.ProgressJournal.read(path)
    assert read.records == [first, second, third]
    assert read.report == {"records": 3, "truncated_last_line": 0}
    assert [row["sequence"] for row in read.records] == [1, 2, 3]
    for row in read.records:
        assert row["schema_version"] == 1 and row["utc"].endswith("Z")
        body = {key: value for key, value in row.items() if key != "sha256"}
        assert row["sha256"] == hashlib.sha256(dp._json(body)).hexdigest()


@pytest.mark.skipif(os.name != "posix", reason="advisory journal locks require POSIX")
def test_journal_two_writers_refused_until_close_then_sequence_recovers(tmp_path):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as first:
        first.append({"writer": 1})
        code = f'''
from rtpipeline.durable_progress import DurableProgressError, ProgressJournal
try:
    ProgressJournal({str(path)!r})
except DurableProgressError as error:
    assert error.code == 'DURABLE_JOURNAL_LOCKED', error.code
else:
    raise AssertionError('second writer acquired an active journal')
print('LOCKED')
'''
        assert run_child(code).strip() == "LOCKED"
        # The same process also cannot open a second writing descriptor.
        with pytest.raises(dp.DurableProgressError, match="DURABLE_JOURNAL_LOCKED"):
            dp.ProgressJournal(path)
        assert first.append({"writer": 1})["sequence"] == 2
    first.close()  # Idempotent close must leave the lock released.
    code = f'''
from rtpipeline.durable_progress import ProgressJournal
with ProgressJournal({str(path)!r}) as journal:
    assert journal.append({{'writer': 2}})['sequence'] == 3
print('REOPENED')
'''
    assert run_child(code).strip() == "REOPENED"
    read = dp.ProgressJournal.read(path)
    assert [row["body"]["writer"] for row in read.records] == [1, 1, 2]
    assert [row["sequence"] for row in read.records] == [1, 2, 3]


@pytest.mark.parametrize("failure", ["recovery", "fsync"])
def test_journal_failed_initialization_releases_lock(tmp_path, monkeypatch, failure):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as journal:
        journal.append({"previous": True})
    with monkeypatch.context() as patch:
        if failure == "recovery":
            def fail(*args):
                raise dp.DurableProgressError("DURABLE_JOURNAL_CORRUPT")
            patch.setattr(dp.ProgressJournal, "read", fail)
            expected = "DURABLE_JOURNAL_CORRUPT"
        else:
            def fail(*args):
                raise OSError(errno.EIO, "synthetic fsync failure")
            patch.setattr(os, "fsync", fail)
            expected = dp.WRITE_FAILED
        with pytest.raises(dp.DurableProgressError, match=expected):
            dp.ProgressJournal(path)
    with dp.ProgressJournal(path) as journal:
        assert journal.append({"recovered": True})["sequence"] == 2


@pytest.mark.parametrize("error_number,code", [
    (errno.EACCES, "DURABLE_JOURNAL_LOCKED"),
    (errno.EAGAIN, "DURABLE_JOURNAL_LOCKED"),
    (errno.EIO, dp.WRITE_FAILED),
])
def test_journal_lock_failure_precedes_recovery_and_closes_descriptor(tmp_path, monkeypatch, error_number, code):
    path = tmp_path / "journal"
    descriptors = []
    with monkeypatch.context() as patch:
        def fail(fd, operation):
            descriptors.append(fd)
            raise OSError(error_number, "synthetic lock failure")
        def forbidden_read(*args):
            raise AssertionError("sequence recovery ran without the journal lock")
        patch.setattr(dp.fcntl, "flock", fail)
        patch.setattr(dp.ProgressJournal, "read", forbidden_read)
        with pytest.raises(dp.DurableProgressError) as caught:
            dp.ProgressJournal(path)
        assert caught.value.code == code
    assert len(descriptors) == 1
    with pytest.raises(OSError) as caught:
        os.fstat(descriptors[0])
    assert caught.value.errno == errno.EBADF
    with dp.ProgressJournal(path) as journal:
        assert journal.append({"recovered": True})["sequence"] == 1


@pytest.mark.parametrize("tail", [b'{"body":', b'{"complete_but_no_newline":true}'])
def test_truncated_last_line_is_reported_and_append_refused(tmp_path, tail):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as journal:
        committed = journal.append({"item": "SYNTH_A"})
    with path.open("ab") as stream:
        stream.write(tail)
    before = path.read_bytes()
    read = dp.ProgressJournal.read(path)
    assert read.records == [committed]
    assert read.report == {"records": 1, "truncated_last_line": 1}
    with pytest.raises(dp.DurableProgressError, match="DURABLE_JOURNAL_TRUNCATED"):
        dp.ProgressJournal(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("tail", [b'{"body":', b'not JSON\n', b'{"body":{}}\n'])
def test_explicit_tail_recovery_quarantines_hash_fsyncs_and_continues(tmp_path, monkeypatch, tail):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as journal:
        first = journal.append({"item": "SYNTH_A"})
    prefix = path.read_bytes()
    path.write_bytes(prefix + tail)
    syncs = []
    real_fsync = os.fsync
    def synced(fd):
        # At truncation durability, evidence must already exist on disk.
        if path.stat().st_size == len(prefix):
            assert quarantine.read_bytes() == tail
            assert json.loads(evidence.read_bytes())["sha256"] == digest
        syncs.append(os.fstat(fd).st_mode)
        real_fsync(fd)
    digest = hashlib.sha256(tail).hexdigest()
    quarantine = path.with_name(path.name + ".torn-" + digest)
    evidence = quarantine.with_name(quarantine.name + ".json")
    monkeypatch.setattr(os, "fsync", synced)
    with dp.ProgressJournal.recover(path) as journal:
        assert journal.append({"item": "SYNTH_B"})["sequence"] == 3
    assert quarantine.read_bytes() == tail
    assert json.loads(evidence.read_bytes()) == {
        "sha256": digest, "bytes": len(tail), "verified_bytes": len(prefix)}
    read = dp.ProgressJournal.read(path)
    assert read.records[0] == first
    recovered = read.records[1]
    assert recovered["body"]["reason_code"] == "DURABLE_JOURNAL_TAIL_RECOVERED"
    assert recovered["body"]["quarantine"] == quarantine.name
    assert [r["sequence"] for r in read.records] == [1, 2, 3]
    import stat
    assert any(stat.S_ISDIR(mode) for mode in syncs)
    assert any(stat.S_ISREG(mode) for mode in syncs)
    with dp.ProgressJournal.recover(path) as journal:
        assert journal.append({"done": True})["sequence"] == 4
    assert len(list(tmp_path.glob("journal.torn-*"))) == 2


def test_explicit_tail_recovery_refuses_midfile_corruption_without_mutation(tmp_path):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as journal:
        journal.append({"verified": True})
    raw = path.read_bytes() + b'not JSON\n' + b'{"partial":'
    path.write_bytes(raw)
    with pytest.raises(dp.DurableProgressError, match="DURABLE_JOURNAL_CORRUPT"):
        dp.ProgressJournal.recover(path)
    assert path.read_bytes() == raw
    assert not list(tmp_path.glob("journal.torn-*"))


def test_failed_quarantine_leaves_original_tail_and_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / "journal"
    raw = b'{"partial":'
    path.write_bytes(raw)
    with monkeypatch.context() as patch:
        def failed(*args):
            raise dp.DurableProgressError(dp.WRITE_FAILED)
        patch.setattr(dp, "_atomic_write", failed)
        with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
            dp.ProgressJournal.recover(path)
    assert path.read_bytes() == raw
    with dp.ProgressJournal.recover(path) as journal:
        assert journal.append({"done": True})["sequence"] == 2


@pytest.mark.parametrize("fault", ["hash", "sequence", "schema", "json", "duplicate", "utc", "nested"])
def test_complete_corrupt_journal_line_is_never_accepted(tmp_path, fault):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as journal:
        row = journal.append({"value": 1})
    if fault == "hash":
        row["body"]["value"] = 9
    elif fault == "sequence":
        row["sequence"] = 2
    elif fault == "schema":
        row["schema_version"] = True
    elif fault == "utc":
        row["utc"] = "yesterday"
    if fault not in {"hash", "json", "duplicate"}:
        row["sha256"] = dp._sha(dp._json({k: v for k, v in row.items() if k != "sha256"}))
    raw = dp._json(row)
    if fault == "json":
        raw = b"not JSON"
    elif fault == "duplicate":
        raw = raw[:-1] + b',"sequence":1}'
    elif fault == "nested":
        raw = b"[" * 2000 + b"0" + b"]" * 2000
    path.write_bytes(raw + b"\n")
    with pytest.raises(dp.DurableProgressError, match="DURABLE_JOURNAL_CORRUPT"):
        dp.ProgressJournal.read(path)


def test_cache_roundtrip_two_phase_and_readback(tmp_path, monkeypatch):
    cache = dp.ResultCache(tmp_path / "cache")
    real = dp._atomic_write
    phases = []
    def write(path, raw):
        phases.append(path.name)
        if path.name == "commit.json":
            assert (path.parent / "payload.json").read_bytes() == dp._json({"value": 42})
            assert cache.load_valid(BINDING).report["incomplete"] == 1
        return real(path, raw)
    monkeypatch.setattr(dp, "_atomic_write", write)
    marker = cache.put("SYNTH_A", {"value": 42}, BINDING)
    directory = transaction(cache)
    assert phases == ["payload.json", "commit.json"]
    payload = (directory / "payload.json").read_bytes()
    assert json.loads(payload) == {"value": 42}
    assert json.loads((directory / "commit.json").read_bytes()) == marker
    assert marker["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    loaded = dp.ResultCache(cache.root).load_valid(BINDING.as_dict())
    assert loaded.entries == {"SYNTH_A": {"value": 42}}
    assert loaded.report == {"total": 1, "valid": 1, "stale": 0, "incomplete": 0,
                             "corrupt": 0, "conflict": 0}


def test_directory_commit_marker_is_corrupt_and_valid_cache_entries_are_reused(tmp_path):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_A", {"value": 42}, BINDING)
    malformed = cache.root / "malformed"
    (malformed / "commit.json").mkdir(parents=True)
    reuse, compute, report = dp.resume_plan(["SYNTH_A", "SYNTH_B"], cache, BINDING)
    assert reuse == {"SYNTH_A": {"value": 42}}
    assert compute == ["SYNTH_B"]
    assert report == {"items": 2, "reuse": 1, "compute": 1,
                      "cache_total": 2, "cache_valid": 1, "cache_corrupt": 1,
                      "cache_stale": 0, "cache_incomplete": 0, "cache_conflict": 0}
    assert (malformed / "commit.json").is_dir()


@pytest.mark.parametrize("field", ["code_sha256", "config_sha256", "inputs_sha256"])
def test_changed_binding_is_stale_and_retained(tmp_path, field):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_A", [1, 2], BINDING)
    changed = replace(BINDING, **{field: "d" * 64})
    read = cache.load_valid(changed)
    assert read.entries == {} and read.report["stale"] == 1
    assert cache.load_valid(BINDING).entries == {"SYNTH_A": [1, 2]}
    assert (transaction(cache) / "commit.json").exists()


@pytest.mark.parametrize("fault", ["payload", "marker", "identity", "duplicate", "nonfinite", "nested"])
def test_corrupt_cache_excluded_without_deletion(tmp_path, fault):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_A", {"value": 1}, BINDING)
    directory = transaction(cache)
    marker_path = directory / "commit.json"
    if fault == "payload":
        (directory / "payload.json").write_bytes(b'{"value":2}')
    elif fault == "marker":
        marker_path.write_bytes(b"broken")
    elif fault == "nested":
        marker_path.write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
    elif fault == "identity":
        marker = json.loads(marker_path.read_bytes())
        marker["item_key"] = "SYNTH_B"
        marker_path.write_bytes(dp._json(marker))
    elif fault == "duplicate":
        raw = marker_path.read_bytes().rstrip()
        marker_path.write_bytes(raw[:-1] + b',"schema_version":1}')
    elif fault == "nonfinite":
        raw = b'{"value":NaN}'
        (directory / "payload.json").write_bytes(raw)
        marker = json.loads(marker_path.read_bytes())
        marker["payload_sha256"] = dp._sha(raw)
        marker_path.write_bytes(dp._json(marker))
    read = cache.load_valid(BINDING)
    assert read.entries == {} and read.report["corrupt"] == 1
    assert directory.exists() and marker_path.exists()


@pytest.mark.parametrize("missing", ["commit.json", "payload.json"])
def test_incomplete_entry_excluded_and_retained(tmp_path, missing):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_A", 1, BINDING)
    directory = transaction(cache)
    (directory / missing).unlink()
    read = cache.load_valid(BINDING)
    assert read.entries == {} and read.report["incomplete"] == 1
    assert directory.exists()


def test_exact_code_bytes_config_and_frozen_inputs_bind(tmp_path):
    first, helper = tmp_path / "checker.py", tmp_path / "helper.py"
    first.write_bytes(b"# exact bytes\r\nvalue = 1\n")
    helper.write_bytes(b"helper = 2\n")
    files = {"checker.py": first, "helper.py": helper}
    config, inputs = {"gate": 3}, {"SYNTH_A": "frozen-hash"}
    initial = dp.make_binding(files, config, inputs)
    assert initial == dp.make_binding(dict(reversed(list(files.items()))), config, inputs)
    first.write_bytes(first.read_bytes() + b"# comment\n")
    assert dp.make_binding(files, config, inputs).code_sha256 != initial.code_sha256
    first.write_bytes(b"# exact bytes\r\nvalue = 1\n")
    helper.write_bytes(b"helper = 2\n ")
    assert dp.make_binding(files, config, inputs).code_sha256 != initial.code_sha256
    assert dp.make_binding(files, {"gate": 4}, inputs).config_sha256 != initial.config_sha256
    assert dp.make_binding(files, config, {"SYNTH_A": "changed"}).inputs_sha256 != initial.inputs_sha256
    assert dp.make_binding(files, b"a\r\n", b"manifest").config_sha256 == dp._sha(b"a\r\n")
    assert dp.file_sha256(first) == hashlib.sha256(first.read_bytes()).hexdigest()


@pytest.mark.parametrize("binding", [{}, {**BINDING.as_dict(), "extra": 1},
                                    {**BINDING.as_dict(), "code_sha256": "branch-name"}])
def test_invalid_binding_is_refused(tmp_path, binding):
    with pytest.raises(dp.DurableProgressError, match="DURABLE_INVALID_BINDING"):
        dp.ResultCache(tmp_path / "cache").load_valid(binding)


def test_resume_partitions_requested_items_and_counts_only(tmp_path):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_B", 2, BINDING)
    cache.put("UNREQUESTED", 99, BINDING)
    reuse, compute, report = dp.resume_plan(["SYNTH_A", "SYNTH_B", "SYNTH_C"], cache, BINDING)
    assert reuse == {"SYNTH_B": 2} and compute == ["SYNTH_A", "SYNTH_C"]
    assert (report["items"], report["reuse"], report["compute"]) == (3, 1, 2)
    assert all(type(value) is int for value in report.values())
    with pytest.raises(dp.DurableProgressError, match="DURABLE_INVALID_ITEMS"):
        dp.resume_plan(["SYNTH_A", "SYNTH_A"], cache, BINDING)


def test_conflicting_commits_excluded_and_duplicates_counted(tmp_path):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_A", 1, BINDING)
    cache.put("SYNTH_A", 1, BINDING)
    read = cache.load_valid(BINDING)
    assert read.entries == {"SYNTH_A": 1} and read.report["valid"] == 2
    cache.put("SYNTH_A", 2, BINDING)
    read = cache.load_valid(BINDING)
    assert read.entries == {} and read.report["conflict"] == 3
    assert read.report["valid"] == 0 and len(list(cache.root.iterdir())) == 3


def test_seal_runs_before_disk_write_and_hashes_stored_bytes(tmp_path):
    cache = dp.ResultCache(tmp_path / "cache", seal=lambda raw: b"SYNTHETIC-SEALED:" + raw[::-1])
    marker = cache.put("SYNTH_A", {"value": 1}, BINDING)
    directory = transaction(cache)
    assert not (directory / "payload.json").exists()
    sealed = (directory / "payload.age").read_bytes()  # synthetic bytes only
    assert marker["sealed"] is True and marker["payload_sha256"] == dp._sha(sealed)
    assert cache.load_valid(BINDING).entries == {"SYNTH_A": sealed}
    reader = dp.ResultCache(cache.root, unseal=lambda raw: raw.split(b":", 1)[1][::-1])
    assert reader.load_valid(BINDING).entries == {"SYNTH_A": {"value": 1}}


@pytest.mark.parametrize("committed", [0, 7, 19])
@pytest.mark.skipif(os.name != "posix", reason="SIGKILL test requires POSIX")
def test_sigkill_and_resume_reuses_exact_committed_items(tmp_path, committed):
    root = tmp_path / "cache"
    journal = tmp_path / "journal"
    code = f"""
from pathlib import Path
from rtpipeline import durable_progress as dp
binding = dp.Binding(**{BINDING.as_dict()!r})
cache = dp.ResultCache({str(root)!r})
with dp.ProgressJournal({str(journal)!r}) as journal:
    for index in range({committed}):
        cache.put(f'SYNTH_{{index:02}}', {{'square': index * index}}, binding)
        journal.append({{'event': 'cache_commit', 'item_key': f'SYNTH_{{index:02}}'}})
        print('COMMITTED', flush=True)
    real_write = dp._atomic_write
    def pause_before_marker(path, raw):
        if path.name == 'commit.json':
            assert (path.parent / 'payload.json').is_file()
            assert not path.exists()
            print('PAYLOAD_PUBLISHED', flush=True)
            # Test-only synchronization: block inside the active transaction.
            # The parent kills us here; production has no hook or delay.
            sys.stdin.buffer.read(1)
            raise AssertionError('active transaction was released instead of killed')
        return real_write(path, raw)
    dp._atomic_write = pause_before_marker
    index = {committed}
    cache.put(f'SYNTH_{{index:02}}', {{'square': index * index}}, binding)
raise AssertionError('active transaction unexpectedly completed')
"""
    process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", child_code(code)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            for expected in [b"COMMITTED"] * committed + [b"PAYLOAD_PUBLISHED"]:
                assert selector.select(10), "computing child stopped acknowledging commits"
                assert process.stdout.readline().strip() == expected
        # Observe the on-disk boundary before SIGKILL, rather than racing a delay.
        cache = dp.ResultCache(root)
        interrupted = [path for path in root.iterdir() if not (path / "commit.json").exists()]
        assert len(interrupted) == 1
        assert json.loads((interrupted[0] / "payload.json").read_bytes()) == {
            "square": committed * committed}
        assert cache.load_valid(BINDING).report["incomplete"] == 1
        process.kill()
        assert process.wait(timeout=10) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
    items = [f"SYNTH_{index:02}" for index in range(20)]
    cache = dp.ResultCache(root)
    reuse, compute, report = dp.resume_plan(items, cache, BINDING)
    assert list(reuse) == items[:committed]
    assert compute == items[committed:]
    assert report["reuse"] == committed and report["compute"] == 20 - committed
    assert report["cache_valid"] == committed and report["cache_incomplete"] == 1
    with dp.GuardSafeSink(journal, cache=cache) as sink:
        for item in compute:
            index = int(item.rsplit("_", 1)[1])
            sink.store(item, {"square": index * index}, BINDING)
    expected = {item: {"square": index * index} for index, item in enumerate(items)}
    uninterrupted = dp.ResultCache(tmp_path / "uninterrupted")
    for item, result in expected.items():
        uninterrupted.put(item, result, BINDING)
    assert cache.load_valid(BINDING).entries == uninterrupted.load_valid(BINDING).entries == expected
    assert cache.load_valid(BINDING).report["incomplete"] == 1
    assert not (interrupted[0] / "commit.json").exists()
    assert dp.ProgressJournal.read(journal).report == {"records": 20, "truncated_last_line": 0}
    final = tmp_path / "final.json"
    assert dp.final_result_writer(final, expected)["verified"] is True
    assert json.loads(final.read_bytes()) == expected


@pytest.mark.parametrize("operation", ["journal", "cache"])
@pytest.mark.skipif(os.name != "posix", reason="SIGKILL test requires POSIX")
def test_writer_drains_complete_queued_requests_after_computing_sigkill(tmp_path, operation):
    journal = tmp_path / "journal"
    cache = dp.ResultCache(tmp_path / "cache")
    sink = dp.GuardSafeSink(journal, cache=cache)
    process = None
    count = 4
    requests = []
    for index in range(count):
        if operation == "journal":
            request = {"op": "journal", "record": {"index": index}}
        else:
            request = {"op": "cache", "item_key": f"SYNTH_{index:02}",
                       "binding": BINDING.as_dict(), "sealed": False,
                       "payload": dp.base64.b64encode(dp._json({"square": index * index})).decode("ascii")}
        requests.append(dp._json(request) + b"\n")
    queued = b"".join(requests)
    writer_fd = sink._process.stdin.fileno()
    assert len(queued) <= os.fpathconf(writer_fd, "PC_PIPE_BUF")
    code = f"""
import os, time
assert os.write({writer_fd}, {queued!r}) == {len(queued)}
print('QUEUED', flush=True)
time.sleep(30)
"""
    try:
        # Stop the writer to prove all requests are still queued when the
        # computing process dies. Keeping the writer as our child lets us reap
        # it and check its exit status instead of leaving an orphan to poll.
        os.kill(sink._process.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            pid, status = os.waitpid(sink._process.pid, os.WUNTRACED | os.WNOHANG)
            if pid:
                assert os.WIFSTOPPED(status)
                break
            time.sleep(0.01)
        else:
            pytest.fail("writer did not reach the stopped synchronization point")
        process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", child_code(code)],
                                   pass_fds=(writer_fd,), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        sink._process.stdin.close()  # Child now owns the only request descriptor.
        sink._process.stdout.close()  # Writer must handle a broken ACK pipe too.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(10)
        assert process.stdout.readline().strip() == "QUEUED"
        process.kill()
        assert process.wait(timeout=10) == -signal.SIGKILL
        assert dp.ProgressJournal.read(journal).report["records"] == 0
        assert cache.load_valid(BINDING).report["total"] == 0
        os.kill(sink._process.pid, signal.SIGCONT)
        sink.close()
        assert sink._process.returncode == 0
        read = dp.ProgressJournal.read(journal)
        assert read.report == {"records": count, "truncated_last_line": 0}
        if operation == "journal":
            assert [row["body"]["index"] for row in read.records] == list(range(count))
        else:
            loaded = cache.load_valid(BINDING)
            assert loaded.entries == {f"SYNTH_{i:02}": {"square": i * i} for i in range(count)}
            assert loaded.report == {"total": count, "valid": count, "stale": 0,
                                     "incomplete": 0, "corrupt": 0, "conflict": 0}
            assert [row["body"]["item_key"] for row in read.records] == [
                f"SYNTH_{i:02}" for i in range(count)]
            assert all(row["body"]["event"] == "cache_commit" for row in read.records)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if process is not None:
            process.stdout.close()
            process.stderr.close()
        if sink._process.poll() is None:
            sink._process.kill()
        sink.close(check=False)


@pytest.mark.parametrize("operation", ["journal", "cache"])
@pytest.mark.parametrize("valid_json", [False, True])
def test_writer_ignores_truncated_request_without_newline_on_eof(tmp_path, operation, valid_json):
    journal = tmp_path / "journal"
    cache = dp.ResultCache(tmp_path / "cache")
    with dp.GuardSafeSink(journal, cache=cache) as sink:
        sink.send({"event": "prior_committed"})
        sink.store("SYNTH_PRIOR", {"value": 1}, BINDING)
        if operation == "journal":
            request = {"op": "journal", "record": {"event": "must_not_commit"}}
        else:
            request = {"op": "cache", "item_key": "SYNTH_PARTIAL",
                       "binding": BINDING.as_dict(), "sealed": False, "payload": "Mg=="}
        raw = dp._json(request)
        if not valid_json:
            raw = raw[:-3]
        dp._write_all(sink._process.stdin, raw)  # No newline; close supplies EOF.
    assert sink._process.returncode == 0
    read = dp.ProgressJournal.read(journal)
    assert read.report == {"records": 2, "truncated_last_line": 0}
    assert [row["body"]["event"] for row in read.records] == ["prior_committed", "cache_commit"]
    loaded = cache.load_valid(BINDING)
    assert loaded.entries == {"SYNTH_PRIOR": {"value": 1}}
    assert loaded.report == {"total": 1, "valid": 1, "stale": 0,
                             "incomplete": 0, "corrupt": 0, "conflict": 0}


GUARD = '''
import os, stat, sys
class Refusal(Exception):
    pass
def require(ok, code):
    if not ok:
        raise Refusal(code)
def read_only_guard():
    def audit(event, args):
        if event == 'open':
            mode, flags = args[1:3]
            null_sink = args[0] == '/dev/null' and stat.S_ISCHR(os.stat('/dev/null').st_mode) and os.stat('/dev/null').st_rdev == os.makedev(1,3)
            pipe = isinstance(args[0], int) and stat.S_ISFIFO(os.fstat(args[0]).st_mode)
            require(pipe or null_sink or not (isinstance(mode, str) and any(x in mode for x in 'wax+')), 'READ_ONLY_VIOLATION')
            require(pipe or null_sink or not (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)), 'READ_ONLY_VIOLATION')
        if event in {'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.utime', 'os.link', 'os.symlink', 'os.truncate'}:
            raise Refusal('READ_ONLY_VIOLATION')
    sys.addaudithook(audit)
'''


@pytest.mark.skipif(os.name != "posix", reason="checker audit semantics are POSIX")
def test_read_only_guard_persists_journal_cache_and_verified_final(tmp_path):
    code = GUARD + f'''
from pathlib import Path
from rtpipeline.durable_progress import Binding, GuardSafeSink, ResultCache, final_result_writer
root = Path({str(tmp_path)!r})
cache = ResultCache(root / 'cache')
regular_fd = os.open(root / 'preopened', os.O_WRONLY | os.O_CREAT, 0o600)
sink = GuardSafeSink(root / 'journal', cache=cache)
read_only_guard()
refused = 0
for action in (lambda: open(root / 'refused', 'w'),
               lambda: os.open(root / 'refused', os.O_CREAT | os.O_WRONLY),
               lambda: open(regular_fd, 'w', closefd=False),
               lambda: (root / 'refused-dir').mkdir(),
               lambda: cache.put('SYNTH_REFUSED', {{'value': 0}}, Binding(**{BINDING.as_dict()!r})),
               lambda: final_result_writer(root / 'refused-final.json', {{'status': 'refused'}}, sink=None)):
    try:
        action()
    except Refusal:
        refused += 1
assert refused == 6
with open('/dev/null', 'w') as null:
    null.write('allowed')
read_fd, write_fd = os.pipe()
with open(write_fd, 'wb') as pipe:
    pipe.write(b'allowed')
assert os.read(read_fd, 7) == b'allowed'
os.close(read_fd)
os.close(regular_fd)
sink.send({{'event': 'guard-active'}})
sink.store('SYNTH_A', {{'value': 7}}, Binding(**{BINDING.as_dict()!r}))
receipt = final_result_writer(root / 'final.json', {{'status': 'ok'}}, sink=sink)
assert receipt['verified'] and receipt['bytes'] == len((root / 'final.json').read_bytes())
sink.close()
print('GUARD_OK')
'''
    assert run_child(code).strip() == "GUARD_OK"
    assert [row["body"]["event"] for row in dp.ProgressJournal.read(tmp_path / "journal").records] == [
        "guard-active", "cache_commit"]
    assert dp.ResultCache(tmp_path / "cache").load_valid(BINDING).entries == {"SYNTH_A": {"value": 7}}
    assert json.loads((tmp_path / "final.json").read_bytes()) == {"status": "ok"}
    assert not (tmp_path / "refused").exists()
    assert not (tmp_path / "refused-dir").exists()
    assert not (tmp_path / "refused-final.json").exists()


def test_computation_exception_preserves_commits_and_failure_aggregate(tmp_path):
    cache = dp.ResultCache(tmp_path / "cache")
    final = tmp_path / "final.json"
    with pytest.raises(RuntimeError, match="synthetic computation failure"):
        with dp.GuardSafeSink(tmp_path / "journal", cache=cache) as sink:
            try:
                for item in ("SYNTH_A", "SYNTH_B"):
                    sink.store(item, {"done": True}, BINDING)
                raise RuntimeError("synthetic computation failure")
            except RuntimeError:
                dp.final_result_writer(final, {"status": "failed", "error_code": "SYNTHETIC_FAILURE",
                                               "committed_items": 2}, sink=sink)
                raise
    assert cache.load_valid(BINDING).entries == {"SYNTH_A": {"done": True}, "SYNTH_B": {"done": True}}
    assert json.loads(final.read_bytes()) == {"status": "failed", "error_code": "SYNTHETIC_FAILURE",
                                            "committed_items": 2}
    assert dp.ProgressJournal.read(tmp_path / "journal").records[-1]["body"] == {
        "event": "exception", "exception_type": "RuntimeError"}


@pytest.mark.parametrize("phase", ["payload", "marker"])
def test_injected_disk_full_preserves_previous_valid_cache(tmp_path, monkeypatch, phase):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_A", {"value": 1}, BINDING)
    original = {path: path.read_bytes() for path in cache.root.rglob("*") if path.is_file()}
    real = dp._write_all
    def fail(stream, raw):
        marker = b'"payload_sha256"' in bytes(raw)
        if marker == (phase == "marker"):
            stream.write(raw[:3])
            raise OSError(errno.ENOSPC, "synthetic full disk")
        return real(stream, raw)
    monkeypatch.setattr(dp, "_write_all", fail)
    with pytest.raises(dp.DurableProgressError) as caught:
        cache.put("SYNTH_A", {"value": 2}, BINDING)
    assert caught.value.code == dp.WRITE_FAILED
    assert {path: path.read_bytes() for path in original} == original
    read = cache.load_valid(BINDING)
    assert read.entries == {"SYNTH_A": {"value": 1}} and read.report["incomplete"] == 1


@pytest.mark.parametrize("stage", ["write", "fsync", "replace"])
def test_final_write_failure_preserves_old_aggregate(tmp_path, monkeypatch, stage):
    path = tmp_path / "final.json"
    dp.final_result_writer(path, {"status": "previous"})
    before = path.read_bytes()
    def fail(*args):
        raise OSError(errno.ENOSPC, "synthetic full disk")
    if stage == "write":
        monkeypatch.setattr(dp, "_write_all", fail)
    else:
        monkeypatch.setattr(os, stage, fail)
    with pytest.raises(dp.DurableProgressError) as caught:
        dp.final_result_writer(path, {"status": "new"})
    assert caught.value.code == dp.WRITE_FAILED and path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_final_result_verifies_before_and_after_rename(tmp_path, monkeypatch):
    path = tmp_path / "final"
    path.write_bytes(b"previous")
    def corrupt(stream, raw):
        stream.write(b"wrong")
    monkeypatch.setattr(dp, "_write_all", corrupt)
    with pytest.raises(dp.DurableProgressError, match=dp.VERIFY_FAILED):
        dp.final_result_writer(path, b"expected")
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_final_destination_readback_mismatch_never_reports_success(tmp_path, monkeypatch):
    path = tmp_path / "final"
    real = Path.read_bytes
    def read(target):
        return b"wrong read-back" if target == path else real(target)
    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(dp.DurableProgressError, match=dp.VERIFY_FAILED):
        dp.final_result_writer(path, b"expected")
    assert real(path) == b"expected"


def test_sink_writer_error_propagates_fixed_code_and_bad_startup_is_reaped(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"synthetic")
    with pytest.raises(dp.DurableProgressError, match=dp.READ_FAILED):
        dp.GuardSafeSink(blocked / "journal")
    with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
        dp.GuardSafeSink(tmp_path / "missing" / "journal")
    sink = dp.GuardSafeSink(tmp_path / "journal", cache=dp.ResultCache(blocked))
    with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
        sink.store("SYNTH_A", 1, BINDING)
    with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
        sink.close()
    assert sink._process.poll() is not None
    assert blocked.read_bytes() == b"synthetic"


@pytest.mark.parametrize("partial_reply", [False, True])
def test_sink_hung_startup_has_bounded_handshake_and_reaps_writer(tmp_path, scripted_writer, partial_reply):
    code = "import sys, time\n"
    if partial_reply:
        code += "sys.stdout.write('{\"ok\":true'); sys.stdout.flush()\n"
    code += "time.sleep(30)\n"
    processes = scripted_writer(code)
    started = time.monotonic()
    with pytest.raises(dp.DurableProgressError) as caught:
        dp.GuardSafeSink(tmp_path / "journal", timeout=0.25)
    assert caught.value.code == "DURABLE_WRITER_HANDSHAKE_TIMEOUT"
    assert time.monotonic() - started < 5
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


@pytest.mark.parametrize("partial_reply", [False, True])
def test_sink_hung_ack_times_out_and_refuses_late_reply_reuse(tmp_path, scripted_writer, partial_reply):
    code = '''
import sys, time
print('{"ok":true,"result":{"ready":true}}', flush=True)
sys.stdin.buffer.readline()
'''
    if partial_reply:
        code += "sys.stdout.write('{\"ok\":true'); sys.stdout.flush()\n"
    code += "time.sleep(30)\n"
    scripted_writer(code)
    sink = dp.GuardSafeSink(tmp_path / "journal", timeout=0.25)
    try:
        started = time.monotonic()
        with pytest.raises(dp.DurableProgressError) as caught:
            sink.send({"event": "hung"})
        assert caught.value.code == "DURABLE_WRITER_ACK_TIMEOUT"
        assert time.monotonic() - started < 5
        with pytest.raises(dp.DurableProgressError, match="DURABLE_SINK_CLOSED"):
            sink.send({"event": "cannot_consume_a_late_ack"})
    finally:
        sink.close(check=False)
    assert sink._process.poll() is not None
    assert sink._process.stdin.closed and sink._process.stdout.closed


def test_sink_hung_close_is_bounded_kills_writer_and_is_idempotent(tmp_path, scripted_writer):
    scripted_writer('''
import sys, time
print('{"ok":true,"result":{"ready":true}}', flush=True)
sys.stdin.buffer.read()
time.sleep(30)
''')
    sink = dp.GuardSafeSink(tmp_path / "journal", timeout=0.25)
    started = time.monotonic()
    with pytest.raises(dp.DurableProgressError) as caught:
        sink.close()
    assert caught.value.code == "DURABLE_WRITER_CLOSE_TIMEOUT"
    assert time.monotonic() - started < 5
    assert sink._process.poll() is not None
    assert sink._process.stdin.closed and sink._process.stdout.closed
    sink.close()


def test_sink_hung_writer_cannot_block_large_request_write(tmp_path, scripted_writer):
    scripted_writer('''
import sys, time
print('{"ok":true,"result":{"ready":true}}', flush=True)
time.sleep(30)
''')
    sink = dp.GuardSafeSink(tmp_path / "journal", timeout=0.25)
    try:
        started = time.monotonic()
        with pytest.raises(dp.DurableProgressError) as caught:
            sink.send({"large": "x" * (1024 * 1024)})
        assert caught.value.code == "DURABLE_WRITER_SEND_TIMEOUT"
        assert time.monotonic() - started < 5
        with pytest.raises(dp.DurableProgressError, match="DURABLE_SINK_CLOSED"):
            sink.send({"cannot_follow_partial_request": True})
    finally:
        sink.close(check=False)
    assert sink._process.poll() is not None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_sink_refuses_timeout_that_cannot_bound_waits(tmp_path, timeout):
    with pytest.raises(ValueError, match="positive finite"):
        dp.GuardSafeSink(tmp_path / "journal", timeout=timeout)
    assert not (tmp_path / "journal").exists()


def test_sink_final_error_and_context_preserve_computation_exception(tmp_path):
    sink = dp.GuardSafeSink(tmp_path / "journal")
    with pytest.raises(RuntimeError, match="computation"):
        with sink:
            with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
                dp.final_result_writer(tmp_path / "missing" / "final", {}, sink=sink)
            raise RuntimeError("computation")
    assert sink._process.poll() is not None


def test_exception_exposes_secondary_persistence_failure(tmp_path):
    sink = dp.GuardSafeSink(tmp_path / "journal")
    sink._process.kill()
    sink._process.wait(timeout=10)
    with pytest.raises(RuntimeError, match="computation") as caught:
        with sink:
            raise RuntimeError("computation")
    assert caught.value.durable_progress_error.code == dp.WRITE_FAILED


def test_journal_partial_write_poisoned_until_explicit_recovery(tmp_path, monkeypatch):
    path = tmp_path / "journal"
    with dp.ProgressJournal(path) as journal:
        journal.append({"committed": True})
        def fail(stream, raw):
            stream.write(raw[:5])
            raise OSError(errno.ENOSPC, "synthetic")
        monkeypatch.setattr(dp, "_write_all", fail)
        with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
            journal.append({"partial": True})
        with pytest.raises(dp.DurableProgressError, match="DURABLE_JOURNAL_CLOSED"):
            journal.append({"must_not_follow_partial": True})
    read = dp.ProgressJournal.read(path)
    assert read.report == {"records": 1, "truncated_last_line": 1}


def test_threaded_sink_serialization_and_explicit_idempotent_close(tmp_path):
    sink = dp.GuardSafeSink(tmp_path / "journal")
    errors = []
    def send(index):
        try:
            sink.send({"index": index})
        except Exception as error:
            errors.append(error)
    threads = [threading.Thread(target=send, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    sink.close()
    sink.close()
    assert not errors
    records = dp.ProgressJournal.read(tmp_path / "journal").records
    assert sorted(row["body"]["index"] for row in records) == list(range(12))
    with pytest.raises(dp.DurableProgressError, match="DURABLE_SINK_CLOSED"):
        sink.send({"closed": True})


def test_sink_large_request_and_ack_round_trip(tmp_path):
    journal = tmp_path / "journal"
    cache = dp.ResultCache(tmp_path / "cache")
    record = {"value": "Łódź" * 32768}
    assert len(dp._json(record)) > 65536
    with dp.GuardSafeSink(journal, cache=cache) as sink:
        acknowledged = sink.send(record)
        assert acknowledged["body"] == record
        sink.store("SYNTH_LARGE", record, BINDING)
    assert dp.ProgressJournal.read(journal).records[0] == acknowledged
    assert cache.load_valid(BINDING).entries == {"SYNTH_LARGE": record}


def test_module_import_and_writer_need_only_standard_library(tmp_path):
    code = f'''
from pathlib import Path
from rtpipeline.durable_progress import GuardSafeSink
with GuardSafeSink(Path({str(tmp_path)!r}) / 'journal') as sink:
    sink.send({{'stdlib': True}})
assert not any(name in sys.modules for name in ('numpy', 'pandas', 'pydicom', 'torch'))
print('STDLIB_OK')
'''
    assert run_child(code).strip() == "STDLIB_OK"


@pytest.mark.parametrize("error_number", [errno.EIO, errno.EACCES])
@pytest.mark.parametrize("filename", ["payload.json", "commit.json"])
def test_per_entry_io_error_excludes_only_that_transaction(tmp_path, monkeypatch, error_number, filename):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_BAD", {"value": 1}, BINDING)
    bad = transaction(cache)
    cache.put("SYNTH_GOOD", {"value": 2}, BINDING)
    original = Path.read_bytes
    def read(path):
        if path == bad / filename:
            raise OSError(error_number, "synthetic per-entry read failure")
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read)
        reused, pending, report = dp.resume_plan(["SYNTH_BAD", "SYNTH_GOOD"], cache, BINDING)
    assert reused == {"SYNTH_GOOD": {"value": 2}} and pending == ["SYNTH_BAD"]
    assert report["cache_corrupt"] == 1 and report["cache_valid"] == 1
    assert (bad / filename).is_file()
    assert cache.load_valid(BINDING).report["valid"] == 2


def test_cache_root_listing_error_keeps_fixed_read_failure(tmp_path, monkeypatch):
    cache = dp.ResultCache(tmp_path / "cache")
    cache.put("SYNTH_GOOD", 1, BINDING)
    original = Path.iterdir
    def listed(path):
        if path == cache.root:
            raise OSError(errno.EIO, "synthetic root listing failure")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", listed)
    with pytest.raises(dp.DurableProgressError, match=dp.READ_FAILED):
        cache.load_valid(BINDING)
