"""Regression tests for the radiomics fork-after-threads deadlock fix.

Root cause: the radiomics ``ProcessPoolExecutor`` pools are created from *inside* the
course-level ``ThreadPoolExecutor`` workers. With the POSIX default ``fork`` start method,
a pool worker forked from the multi-threaded parent inherits a COPY of every lock in
whatever state it had at fork time; a lock held by a sibling thread is copied locked with
no holder in the child, so the worker deadlocks forever acquiring it (observed live as
workers blocked in ``synchronize.py __enter__`` / ``queues.py get`` while ``as_completed``
waits indefinitely). The fix routes both pools through ``utils.radiomics_mp_context()``,
which returns a ``forkserver`` (default) or ``spawn`` context — workers are no longer
forked from the multi-threaded parent.
"""

import inspect
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pytest

import rtpipeline.radiomics_conda as rc
import rtpipeline.radiomics_parallel as rp
from rtpipeline.utils import radiomics_mp_context


# ---- module-level workers (must be picklable for spawn/forkserver) ----------------------

def _square(x):
    return x * x


def _nested_course_worker(n):
    """Mimic radiomics: a ProcessPoolExecutor created INSIDE a worker thread.

    Bounded by ``as_completed(..., timeout=...)`` so a broken context fails with a
    TimeoutError instead of hanging the suite.
    """
    ctx = radiomics_mp_context()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        futs = [ex.submit(_square, i) for i in range(n)]
        total = 0
        for fut in as_completed(futs, timeout=120):
            total += fut.result()
        return total


# ---- helper correctness -----------------------------------------------------------------

def test_radiomics_mp_context_is_never_fork():
    ctx = radiomics_mp_context()
    assert ctx.get_start_method() != "fork"
    assert ctx.get_start_method() in ("forkserver", "spawn")


def test_radiomics_mp_context_default_is_forkserver():
    # No env override → forkserver (warm import, cheaper than spawn for many short pools).
    assert radiomics_mp_context().get_start_method() == "forkserver"


def test_radiomics_mp_context_env_override_spawn(monkeypatch):
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_MP_START", "spawn")
    assert radiomics_mp_context().get_start_method() == "spawn"


def test_radiomics_mp_context_invalid_env_falls_back_to_forkserver(monkeypatch):
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_MP_START", "garbage-not-a-method")
    assert radiomics_mp_context().get_start_method() == "forkserver"


def test_radiomics_mp_context_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_MP_START", "forkserver")
    assert radiomics_mp_context("spawn").get_start_method() == "spawn"


# ---- functional: the real nesting pattern must complete (not deadlock) ------------------

def test_nested_threadpool_processpool_completes_forkserver():
    """Outer ThreadPoolExecutor → inner ProcessPoolExecutor(mp_context), the exact
    structure that deadlocked under the default fork context. Must complete."""
    expected = sum(i * i for i in range(5))
    with ThreadPoolExecutor(max_workers=4) as tex:
        course_futs = [tex.submit(_nested_course_worker, 5) for _ in range(4)]
        totals = [cf.result(timeout=180) for cf in course_futs]
    assert totals == [expected] * 4


def test_nested_threadpool_processpool_completes_spawn(monkeypatch):
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_MP_START", "spawn")
    expected = sum(i * i for i in range(5))
    with ThreadPoolExecutor(max_workers=4) as tex:
        course_futs = [tex.submit(_nested_course_worker, 5) for _ in range(4)]
        totals = [cf.result(timeout=180) for cf in course_futs]
    assert totals == [expected] * 4


# ---- source guards: both real pool sites must carry mp_context (anti-regression) --------

def test_conda_pool_site_uses_mp_context():
    src = inspect.getsource(rc.process_radiomics_batch)
    assert "ProcessPoolExecutor(" in src, "expected a ProcessPoolExecutor in process_radiomics_batch"
    assert "mp_context=radiomics_mp_context()" in src, (
        "process_radiomics_batch must create its ProcessPoolExecutor with a fork-safe "
        "mp_context (radiomics_mp_context()); a bare fork pool re-introduces the deadlock"
    )


def test_parallel_pool_site_uses_mp_context():
    src = inspect.getsource(rp.parallel_radiomics_for_course)
    assert "ProcessPoolExecutor(" in src, "expected a ProcessPoolExecutor in parallel_radiomics_for_course"
    assert "mp_context=radiomics_mp_context()" in src, (
        "parallel_radiomics_for_course must create its ProcessPoolExecutor with a fork-safe "
        "mp_context (radiomics_mp_context())"
    )


# ---- deterministic root-cause documentation ---------------------------------------------

@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork semantics only")
def test_fork_after_threads_lock_inheritance_is_the_hazard():
    """A threading.Lock held at fork time is copied LOCKED into the child, whose holder
    thread does not exist there, so the child can never acquire it. This is precisely why a
    fork-context pool nested inside threads deadlocks; forkserver/spawn avoid it by not
    forking workers from the multi-threaded parent.
    """
    lock = threading.Lock()
    lock.acquire()  # held by this (main) thread at fork time
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: only async-safe ops, then hard-exit (no pytest teardown)
        got = lock.acquire(timeout=1.0)
        os.write(w, b"1" if got else b"0")
        os._exit(0)
    os.close(w)
    out = os.read(r, 1)
    os.close(r)
    os.waitpid(pid, 0)
    lock.release()
    assert out == b"0", "child unexpectedly acquired the inherited-locked copy"
