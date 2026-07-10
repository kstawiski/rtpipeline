"""Regression tests for four "batch B" reliability bugs.

Bug 1 (utils.py) / Bug 2 (radiomics_parallel.py) — eager-submission timing bug:
    Both ``run_tasks_with_adaptive_workers`` and ``parallel_radiomics_for_course``
    submitted an entire batch to the executor up front, then stamped every
    future's "start time" immediately afterwards -- even though only ``workers``
    slots exist, so excess tasks sit queued behind the active ones. Measuring
    "task took Ns" (and, where a timeout is configured, killing on timeout) from
    that stamp therefore counted queue-wait time as if it were execution time,
    producing false "slow" warnings and possible false timeout-kills of tasks
    that were still queued, not actually running long. The fix submits lazily
    (at most ``workers``/``pool_size`` futures in flight) and stamps each
    future's start time at the moment it is actually submitted.

Bug 3 (radiomics_conda.py) — positional zip misattribution:
    ``_process_batch`` paired ``batch`` tasks with ``batch_results`` via
    positional ``zip()``. If a subprocess output line is dropped or unparseable
    (caught + only debug-logged upstream), ``batch_results`` is shorter than
    ``batch`` and every pairing after the gap shifts by one, misattributing ROI
    feature rows (and silently dropping the last task in the batch). The fix
    matches results back to tasks by the ``__roi_name__`` field the subprocess
    already echoes back, mirroring the pattern in radiomics_robustness.py.

Bug 4 (radiomics_conda.py) — non-atomic checkpoint write:
    ``RadiomicsCheckpoint._flush_buffer`` wrote the combined DataFrame straight
    to ``self.checkpoint_path`` via ``to_parquet``. A process killed mid-write
    truncates/corrupts that file, and ``_load_existing`` silently discards all
    prior checkpointed results on the next run. The fix writes to a temp file
    in the same directory and publishes it via ``os.replace()`` (atomic on
    POSIX), mirroring the tmp+replace pattern already used in
    ``auto_rtstruct.py``.

Bug 5 (utils.py) / Bug 6 (radiomics_parallel.py) — CRITICAL backfill regression:
    The lazy-submit fix for Bug 1/2 above submits the next task into a freed
    slot ("backfill") by calling ``executor.submit()`` directly inside the
    watchdog loop's ``try:``, with no ``except`` around that specific call. If a
    worker dies (OOM-killed, segfault) the pool becomes broken and
    ``executor.submit()`` raises (``BrokenProcessPool``) from that backfill
    call, which propagates uncaught out of the whole function -- silently
    losing every remaining/never-submitted task in the batch (the OLD eager
    upfront-submit code did not have this hole: it rode worker deaths through
    the futures' own ``result()`` calls). The fix guards the backfill submit:
    on a submit exception, stop backfilling and take the existing
    restart-with-a-fresh-pool path (unfinished/never-submitted tasks are
    requeued), exactly as the timeout path already does.
"""

import inspect
import logging
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import rtpipeline.radiomics as radiomics_mod
import rtpipeline.radiomics_conda as rc
import rtpipeline.radiomics_parallel as rp
import rtpipeline.utils as ru
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs
from rtpipeline.radiomics_conda import RadiomicsCheckpoint

_RTSTRUCT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.3"


# ---- Bug 1: run_tasks_with_adaptive_workers lazy-submit timing ----------------------------

def _scripted_clock(values):
    """A fake perf_counter() that returns `values` in order (clamping to the last
    value if called more times than scripted). Since only the function under test
    calls this clock (the fake `func` below never touches it), the call sequence
    is fully deterministic -- no real-clock/thread races involved."""
    state = {"i": 0}

    def _clock() -> float:
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        return values[i]

    return _clock


def test_adaptive_workers_measures_own_runtime_not_queue_wait(monkeypatch, caplog):
    """REGRESSION: with more tasks than workers, a task's measured duration must
    reflect its own execution time, not the time it spent queued behind an
    earlier, slower task. Under the pre-fix eager submission, BOTH futures were
    stamped with (almost) the same start time before either began running, so
    the instant second task's measured duration included the first task's
    entire runtime -- a false "slow" warning.

    The perf_counter() call sequence for this scenario (2 items, 1 worker, no
    timeout, use_processes=False) is: start, task0's submit-stamp, last_heartbeat,
    "now" once task0 is observed done, task1's backfill submit-stamp, "now" once
    task1 is observed done -- verified empirically against the current
    implementation. Scripting these 6 values makes task0 "run" for 500s (call 4 -
    call 2) while task1 is backfilled only once task0 finishes (call 5 == call 4)
    and itself takes ~0s (call 6 - call 5).
    """
    clock_values = [0.0, 0.0, 0.0, 500.0, 500.0, 500.0]
    monkeypatch.setattr(ru, "perf_counter", _scripted_clock(clock_values))

    def func(idx):
        return "slow-done" if idx == 0 else "instant-done"

    with caplog.at_level(logging.INFO, logger="rtpipeline.utils"):
        results = ru.run_tasks_with_adaptive_workers(
            "batch-b-timing-test", [0, 1], func,
            max_workers=1, min_workers=1, use_processes=False,
        )

    assert results == ["slow-done", "instant-done"]

    slow_messages = [
        r.getMessage() for r in caplog.records
        if "took" in r.getMessage() and "(slow)" in r.getMessage()
    ]
    assert any("task #1 " in m for m in slow_messages), (
        "sanity check: task #1's genuine 500s runtime should still be reported as slow"
    )
    assert not any("task #2 " in m for m in slow_messages), (
        "task #2 (instant) must not be reported slow -- its measured duration must "
        "reflect its own runtime, not task #1's queue-wait/execution time"
    )


def test_adaptive_workers_lazy_submit_source_guard():
    """Regression guard: the eager `{fut: ex.submit(...) for idx in current_indices}`
    dict comprehension (which stamps every future's start time immediately after
    submitting the whole batch) must not reappear."""
    src = inspect.getsource(ru.run_tasks_with_adaptive_workers)
    assert "future_to_idx = {ex.submit(func, seq[idx]): idx for idx in current_indices}" not in src
    assert "_submit_next(" in src


# ---- Bug 2: parallel_radiomics_for_course lazy-submit timing ------------------------------

class _FakeFuture:
    def __init__(self, task):
        self.task = task


class _FakeExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, task):
        self.submitted.append((fn, task))
        return _FakeFuture(task)


def test_submit_lazy_stamps_start_time_at_submission_not_upfront(monkeypatch):
    """Focused test of the submission/timing helper introduced for the fix: each
    future's start time must be recorded when IT is submitted, not when the
    whole batch is enumerated (the anti-pattern this replaces)."""
    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(rp.time, "monotonic", fake_monotonic)

    executor = _FakeExecutor()
    pending = iter(["a", "b", "c"])
    futures = {}
    task_start = {}

    first = rp._submit_lazy(executor, pending, str, futures, task_start, 2)
    assert len(first) == 2
    assert futures[first[0]] == "a"
    assert futures[first[1]] == "b"
    assert task_start[first[0]] == 1.0
    assert task_start[first[1]] == 2.0

    # Simulate a slot freeing up much later: the backfilled future's start time
    # must reflect THAT later moment, not the initial submission batch.
    clock["t"] = 500.0
    second = rp._submit_lazy(executor, pending, str, futures, task_start, 5)
    assert len(second) == 1, "only one task ('c') remains in `pending`"
    assert futures[second[0]] == "c"
    assert task_start[second[0]] == 501.0
    assert executor.submitted == [(str, "a"), (str, "b"), (str, "c")]


def test_parallel_radiomics_uses_lazy_submit_helper_source_guard():
    """Regression guard: parallel_radiomics_for_course must submit ROI tasks via
    _submit_lazy, not the eager `for task in pending_tasks: futures[executor.submit(...)]`
    loop that stamped every future's start time before slots actually freed up."""
    src = inspect.getsource(rp.parallel_radiomics_for_course)
    assert "_submit_lazy(" in src
    assert "task_start = {fut: time.monotonic() for fut in futures}" not in src


# ---- Bug 3: _process_batch task-index matching (not positional zip / bare roi_name) -------
#
# H remediation: matching purely by __roi_name__ (the original Bug 3 fix) is still
# ambiguous when a batch has two same-named ROIs (the module documents roi_name is not
# guaranteed unique). extract_radiomics_batch_with_conda now injects a unique
# `__task_index__` per task and the subprocess echoes it back; _process_batch (and the
# sequential batch-mode branch of process_radiomics_batch, which now delegates to
# _process_batch instead of its own positional zip) matches by that index instead.

def test_process_batch_matches_by_roi_name_not_position(monkeypatch):
    """REGRESSION: a dropped/unparseable subprocess output line must not shift
    every subsequent (task, features) pairing. The middle task's result is
    missing here; the fix must still attribute the surviving results correctly
    and leave the missing one absent (not mislabeled)."""
    batch = [
        {"roi_name": "CTV", "params_file": None},
        {"roi_name": "BLADDER", "params_file": None},
        {"roi_name": "RECTUM", "params_file": None},
    ]
    # BLADDER's (index 1) output line was dropped/unparseable upstream.
    incomplete_results = [
        {"__status__": "success", "__roi_name__": "CTV", "__task_index__": 0, "feat": 1.0},
        {"__status__": "success", "__roi_name__": "RECTUM", "__task_index__": 2, "feat": 3.0},
    ]
    monkeypatch.setattr(
        rc, "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: incomplete_results,
    )

    pairs = rc._process_batch(batch)

    assert len(pairs) == 3, "every input task must produce one (task, features) pair"
    by_roi = {task["roi_name"]: features for task, features in pairs}
    assert by_roi["CTV"] is not None and by_roi["CTV"]["feat"] == 1.0
    assert by_roi["RECTUM"] is not None and by_roi["RECTUM"]["feat"] == 3.0
    assert by_roi["BLADDER"] is None, "missing result must be absent, not another ROI's data"


def test_process_batch_preserves_order_when_nothing_dropped(monkeypatch):
    """Sanity check: with a complete result set, matching-by-task-index reproduces
    the same (task, features) pairing the old positional zip would have."""
    batch = [
        {"roi_name": "A", "params_file": None},
        {"roi_name": "B", "params_file": None},
    ]
    full_results = [
        {"__status__": "success", "__roi_name__": "A", "__task_index__": 0, "feat": 10.0},
        {"__status__": "success", "__roi_name__": "B", "__task_index__": 1, "feat": 20.0},
    ]
    monkeypatch.setattr(
        rc, "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: full_results,
    )

    pairs = rc._process_batch(batch)
    assert [(t["roi_name"], f["feat"]) for t, f in pairs] == [("A", 10.0), ("B", 20.0)]


def test_process_batch_duplicate_roi_names_do_not_cross_assign(monkeypatch):
    """H: roi_name is documented as not guaranteed unique within a batch. Two tasks
    share a name; the FIRST occurrence's (index 0) line is the one dropped, and only
    the SECOND occurrence's (index 1) result survives. Name-based FIFO matching would
    hand index 1's real result to index 0 (the wrong task) and then report index 1 as
    missing (also wrong) -- __task_index__ matching must keep them distinct regardless
    of which occurrence of the shared name actually produced a result."""
    batch = [
        {"roi_name": "GTV", "params_file": None},   # index 0 (dropped upstream)
        {"roi_name": "GTV", "params_file": None},   # index 1 (duplicate name, survives)
        {"roi_name": "BODY", "params_file": None},  # index 2
    ]
    results = [
        {"__status__": "success", "__roi_name__": "GTV", "__task_index__": 1, "feat": 2.0},
        {"__status__": "success", "__roi_name__": "BODY", "__task_index__": 2, "feat": 3.0},
    ]
    monkeypatch.setattr(
        rc, "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: results,
    )

    pairs = rc._process_batch(batch)

    assert len(pairs) == 3
    assert pairs[0][1] is None, "first GTV's dropped result must stay missing, not borrow the second GTV's"
    assert pairs[1][1] is not None and pairs[1][1]["feat"] == 2.0, "second GTV keeps its own result"
    assert pairs[2][1] is not None and pairs[2][1]["feat"] == 3.0


def test_process_radiomics_batch_sequential_uses_process_batch_source_guard():
    """Regression guard: the sequential/one-worker batch-mode branch must delegate to
    _process_batch's index-based matching, not its own `for task, features in
    zip(tasks_list, batch_results)` (which is positionally vulnerable exactly like the
    already-fixed _process_batch used to be, and does not exist for duplicate roi_names)."""
    src = inspect.getsource(rc.process_radiomics_batch)
    assert "for task, features in zip(tasks_list, batch_results)" not in src
    assert "_process_batch(tasks_list)" in src


def test_process_radiomics_batch_sequential_dropped_middle_result_attributes_correctly(tmp_path, monkeypatch):
    """H TEST: end-to-end through the sequential/one-worker path (sequential=True) with
    a dropped middle result -- the surviving ROIs must attribute correctly and the
    missing one must be absent from the output workbook, not misattributed."""
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    monkeypatch.setattr(
        rc, "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: [
            # BLADDER (index 1) output line dropped/unparseable upstream.
            {"__status__": "success", "__roi_name__": "CTV", "__task_index__": 0, "original_firstorder_Mean": 1.0},
            {"__status__": "success", "__roi_name__": "RECTUM", "__task_index__": 2, "original_firstorder_Mean": 3.0},
        ],
    )

    tasks = [
        {"image_path": "i", "mask_path": "m", "roi_name": "CTV", "cleanup": False, "metadata": {}},
        {"image_path": "i", "mask_path": "m", "roi_name": "BLADDER", "cleanup": False, "metadata": {}},
        {"image_path": "i", "mask_path": "m", "roi_name": "RECTUM", "cleanup": False, "metadata": {}},
    ]
    output_path = tmp_path / "radiomics_ct.xlsx"

    result = rc.process_radiomics_batch(
        tasks, str(output_path), sequential=True, max_workers=1, enable_heartbeat=False,
    )

    assert result is not None
    import pandas as pd  # type: ignore

    df = pd.read_excel(output_path)
    written = {row["roi_original_name"]: row["original_firstorder_Mean"] for _, row in df.iterrows()}
    assert written == {"CTV": 1.0, "RECTUM": 3.0}, (
        f"BLADDER's dropped result must be absent, CTV/RECTUM must keep their own values; got {written}"
    )


# ---- Bug 4: RadiomicsCheckpoint atomic flush ----------------------------------------------

def test_checkpoint_flush_writes_via_temp_and_replace_source_guard():
    src = inspect.getsource(RadiomicsCheckpoint._flush_buffer)
    assert "os.replace(" in src, "_flush_buffer must publish via os.replace(), not write in place"
    assert "combined_df.to_parquet(self.checkpoint_path" not in src, (
        "must not write the final parquet path directly (non-atomic; truncates on a kill mid-write)"
    )


def test_checkpoint_flush_leaves_prior_checkpoint_intact_on_simulated_crash(tmp_path, monkeypatch):
    """A pre-existing valid checkpoint must never be left truncated/corrupted by a
    write that fails partway through publishing (simulated kill between writing
    the temp file and the atomic rename)."""
    cp_path = tmp_path / "checkpoint.parquet"
    cp = RadiomicsCheckpoint(cp_path, buffer_size=1)
    cp.add_result({"roi_name": "CTV", "value": 1.0})
    cp.flush()
    assert cp_path.exists()
    original_bytes = cp_path.read_bytes()

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash mid-publish")

    monkeypatch.setattr(rc.os, "replace", boom)

    cp.add_result({"roi_name": "BLADDER", "value": 2.0})
    cp.flush()  # errors are caught and logged, not raised out of flush()

    assert cp_path.read_bytes() == original_bytes, (
        "the previously-published checkpoint must be untouched by the failed flush"
    )
    leftovers = list(tmp_path.glob(f".{cp_path.name}.*.tmp"))
    assert leftovers == [], "no temp file should remain after the failed publish"


def test_checkpoint_flush_leaves_no_temp_file_on_success(tmp_path):
    """Happy path: after a successful flush, only the final checkpoint file exists
    (the temp file was renamed away, not left behind)."""
    cp_path = tmp_path / "checkpoint.parquet"
    cp = RadiomicsCheckpoint(cp_path, buffer_size=1)
    cp.add_result({"roi_name": "CTV", "value": 1.0})
    cp.flush()

    assert cp_path.exists()
    leftovers = list(tmp_path.glob(f".{cp_path.name}.*.tmp"))
    assert leftovers == []


# ---- Bug 5/6: CRITICAL backfill regression -- a dead worker must not lose the batch -------

def _identity(x):
    """Module-level (picklable) stand-in task function for run_tasks_with_adaptive_workers."""
    return x


class _SyncExecutor:
    """Fake process-pool-like executor: submit() completes synchronously and returns a
    real, already-done concurrent.futures.Future (so concurrent.futures.wait() behaves
    correctly), with no actual subprocess/threading involved -- fully deterministic.
    After `break_after` successful submits, submit() raises BrokenProcessPool, simulating
    a worker dying (OOM-killed/segfault) partway through a batch.

    If `result_fn` is given, it computes the future's result instead of the submitted
    `fn` -- needed when the real submitted callable (e.g. radiomics_parallel._extract_one)
    only works inside a worker process set up via `initializer=`, which this fake never
    runs; a plain `fn` like `_identity` above can just be called directly (result_fn=None).
    """

    def __init__(self, break_after=None, result_fn=None, **_kwargs):
        self._break_after = break_after
        self._result_fn = result_fn
        self.calls = 0

    def submit(self, fn, item):
        self.calls += 1
        if self._break_after is not None and self.calls > self._break_after:
            raise BrokenProcessPool("simulated dead worker")
        fut = Future()
        compute = self._result_fn if self._result_fn is not None else fn
        try:
            fut.set_result(compute(item))
        except Exception as exc:  # pragma: no cover - not exercised by these tests
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class _RecoverableBrokenPoolFactory:
    """Stands in for ProcessPoolExecutor: the FIRST pool created is flaky (breaks after
    `break_after` submits); every pool created after that (i.e. the fresh pool from a
    restart) works normally, simulating a successful recovery."""

    def __init__(self, break_after: int, result_fn=None):
        self._break_after = break_after
        self._result_fn = result_fn
        self.created = 0

    def __call__(self, max_workers=None, mp_context=None, **_kwargs):
        self.created += 1
        break_after = self._break_after if self.created == 1 else None
        return _SyncExecutor(break_after=break_after, result_fn=self._result_fn)


def test_adaptive_workers_recovers_from_broken_pool_during_backfill(monkeypatch):
    """CRITICAL regression: a worker dying mid-batch makes the NEXT executor.submit()
    (the backfill call) raise BrokenProcessPool. Pre-fix this propagated straight out of
    run_tasks_with_adaptive_workers (uncaught -- the backfill submit had a `try` with no
    `except`, only a `finally`), silently losing every remaining/never-submitted task.
    The fix must catch it, restart with a fresh pool, and requeue everything unfinished
    so no task is lost. This test fails on the un-remediated code (raises BrokenProcessPool
    out of the function instead of returning results)."""
    factory = _RecoverableBrokenPoolFactory(break_after=2)  # both initial submits succeed
    monkeypatch.setattr(ru, "ProcessPoolExecutor", factory)

    results = ru.run_tasks_with_adaptive_workers(
        "backfill-broken-pool-test", [0, 1, 2, 3], _identity,
        max_workers=2, min_workers=1, use_processes=True,
    )

    assert results == [0, 1, 2, 3], "no task may be silently lost when the pool breaks mid-backfill"
    assert factory.created == 2, "must have restarted with a fresh pool after the break"


def test_adaptive_workers_backfill_submit_guarded_source_guard():
    """Regression guard: the backfill submit loop must not be a bare
    `for _ in range(len(done)): new_fut = _submit_next()` with no exception guard -- that
    is exactly the shape that let a BrokenProcessPool escape uncaught."""
    src = inspect.getsource(ru.run_tasks_with_adaptive_workers)
    assert "except Exception as exc:" in src
    assert "restart_due_to_pool_failure" in src


def _write_rtstruct_with_rois(path: Path, roi_names: list[str]) -> Path:
    meta = FileMetaDataset()
    sop_uid = generate_uid()
    meta.MediaStorageSOPClassUID = _RTSTRUCT_SOP_CLASS_UID
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = _RTSTRUCT_SOP_CLASS_UID
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "RTSTRUCT"
    rois = Sequence()
    for i, name in enumerate(roi_names, start=1):
        roi = Dataset()
        roi.ROINumber = i
        roi.ROIName = name
        rois.append(roi)
    ds.StructureSetROISequence = rois
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def test_parallel_radiomics_recovers_from_broken_pool_during_backfill(tmp_path, monkeypatch):
    """Same CRITICAL regression, end-to-end through parallel_radiomics_for_course: a
    worker dying mid-batch must not lose ROI tasks. Fails on the un-remediated code
    (BrokenProcessPool propagates out of parallel_radiomics_for_course)."""
    course_dir = tmp_path / "course"
    course_dirs = build_course_dirs(course_dir)
    course_dirs.dicom_ct.mkdir(parents=True, exist_ok=True)
    roi_names = ["PTV", "BLADDER", "RECTUM", "FEMUR_L"]
    _write_rtstruct_with_rois(course_dir / "RS_auto.dcm", roi_names)

    config = PipelineConfig(dicom_root=tmp_path / "dicom", output_root=tmp_path / "out", logs_root=tmp_path / "logs")

    # Bypass the real PyRadiomics extractor construction / NumPy-version routing -- this
    # test targets the submit/recovery control flow, not feature extraction itself.
    monkeypatch.setattr(radiomics_mod, "_extractor", lambda *a, **kw: object())

    # result_fn stands in for the real _extract_one (which only works inside a worker
    # process initialized via initializer=); it just echoes back a minimal, valid row.
    factory = _RecoverableBrokenPoolFactory(
        break_after=2,
        result_fn=lambda task: {
            "segmentation_source": task.source,
            "roi_original_name": task.roi_name,
            "value": 1.0,
        },
    )
    monkeypatch.setattr(rp, "ProcessPoolExecutor", factory)

    result_path = rp.parallel_radiomics_for_course(config, course_dir, max_workers=2)

    assert factory.created == 2, "must have restarted with a fresh pool after the break"
    assert result_path is not None and result_path.exists()

    import pandas as pd  # type: ignore

    df = pd.read_excel(result_path, engine="openpyxl")
    assert sorted(df["roi_original_name"].tolist()) == sorted(roi_names), (
        "every ROI task must be recovered after the broken pool -- none silently lost"
    )


def test_parallel_radiomics_backfill_submit_guarded_source_guard():
    """Regression guard: the backfill _submit_lazy call in parallel_radiomics_for_course
    must be guarded against a submit exception, and the restart reconstruction must not
    depend on `remaining`/`task_iter` state alone (which loses a task if the pool breaks
    mid-_submit_lazy) -- it must use the fixed per-round task snapshot instead."""
    src = inspect.getsource(rp.parallel_radiomics_for_course)
    assert "round_tasks" in src
    assert "except Exception as exc:" in src
