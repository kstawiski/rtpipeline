"""Bounded workers with computational progress, exact task identity and loud failures.

A completed image/filter batch is progress. A timer heartbeat or CPU usage is
not: either can conceal a livelock. No feature settings are changed here.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import json
import logging
from multiprocessing.connection import wait
import os
import resource
import time
import traceback

logger = logging.getLogger(__name__)
_sender = None


def report_progress(stage):
    if _sender is not None:
        _sender(str(stage))


def observed_extractor(extractor):
    """Instrument only real completed work; preserve the extractor and settings."""
    if _sender is None:
        return extractor
    for name in ("loadImage", "computeShape", "computeFeatures"):
        original = getattr(extractor, name)
        def completed(*args, _original=original, _name=name, **kwargs):
            value = _original(*args, **kwargs)
            label = str(args[2]) if _name == "computeFeatures" and len(args) > 2 else ""
            report_progress(f"{_name}:{label}")
            return value
        setattr(extractor, name, completed)
    return extractor


def _worker(connection, function):
    global _sender
    try:
        while True:
            item = connection.recv()
            if item is None:
                return
            index, task = item
            started = time.monotonic()
            def progress(stage):
                connection.send(("progress", index, {
                    "stage": stage, "elapsed_s": time.monotonic() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }))
            _sender = progress
            try:
                progress("task_started")
                value = function(task)
                if not isinstance(value, dict) or not value:
                    raise RuntimeError("worker returned no identified result")
                connection.send(("result", index, value))
            except Exception as exc:
                connection.send(("error", index, {
                    "exception_type": type(exc).__name__, "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }))
            finally:
                _sender = None
    finally:
        connection.close()


class SupervisedResults:
    """One in-flight task per worker. A failed worker cannot destroy its peers.

    The course budget retains its historical explicit/default wall-clock limit.
    The stall budget is measured since the last completed computational step,
    per active task, never since the last whole-condition return or submission
    of a task that is merely queued. Errors are returned with exact task index
    for durable publication; the caller MUST raise after publishing failures.
    """
    def __init__(self, context, function, tasks, workers, *, course_timeout, progress_timeout):
        if course_timeout <= 0 or progress_timeout <= 0:
            raise RuntimeError("incomplete robustness extraction: watchdog budgets must be positive")
        self.context, self.function, self.tasks = context, function, list(tasks)
        self.workers = min(max(1, workers), len(self.tasks))
        self.course_timeout, self.progress_timeout = course_timeout, progress_timeout
        self.pending = deque(range(len(self.tasks)))
        self.slots, self.ready = [], deque()
        self.started = time.monotonic()
        self.returned = 0
        self.deadline_reached = False

    def __enter__(self):
        return self

    @staticmethod
    def _stop(slot):
        process = slot['process']
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.is_alive():
            raise RuntimeError(f"could not stop owned robustness worker {process.pid}")
        slot['connection'].close()

    def __exit__(self, *args):
        for slot in self.slots:
            self._stop(slot)
        self.slots.clear()

    def _failure(self, index, reason, slot=None, **evidence):
        task = self.tasks[index][1]
        details = dict(evidence)
        if slot is not None:
            details.update(pid=slot['process'].pid, exitcode=slot['process'].exitcode,
                           elapsed_s=time.monotonic()-slot['started'],
                           no_progress_s=time.monotonic()-slot['progress'],
                           last_progress=slot['last'])
        logger.error("Robustness technical failure task=%d reason=%s evidence=%s", index, reason, details)
        return {"__technical_failure__": {"reason_code": reason, "evidence": details},
                "__task_index__": index, "roi_name": task.get('roi_name', ''),
                "segmentation_source": task.get('segmentation_source', ''),
                **task.get('extra_metadata', {})}

    def _backfill(self):
        while self.pending and len(self.slots) < self.workers:
            parent, child = self.context.Pipe()
            process = self.context.Process(target=_worker, args=(child, self.function))
            process.start()
            child.close()
            self.slots.append(dict(process=process, connection=parent, index=None))
        for slot in self.slots:
            if slot['index'] is None and self.pending:
                index = self.pending.popleft()
                slot.update(index=index, started=time.monotonic(), progress=time.monotonic(), last={'stage': 'submitted'})
                slot['connection'].send((index, self.tasks[index]))

    def next(self, timeout=10):
        from multiprocessing import TimeoutError
        end = time.monotonic() + timeout
        while True:
            if self.ready:
                self.returned += 1
                return self.ready.popleft()
            if self.returned == len(self.tasks):
                raise StopIteration
            if not self.deadline_reached and time.monotonic()-self.started <= self.course_timeout:
                self._backfill()
            # Drain buffered results before acting on a deadline or process death.
            connections = [s['connection'] for s in self.slots if s['index'] is not None]
            readable = wait(connections, timeout=0) if connections else []
            for slot in list(self.slots):
                index = slot['index']
                if index is None:
                    continue
                if slot['connection'] in readable:
                    try:
                        kind, received_index, value = slot['connection'].recv()
                        if received_index != index:
                            raise RuntimeError("worker task identity mismatch")
                        if kind == 'progress':
                            slot.update(progress=time.monotonic(), last=value)
                            continue
                        if kind == 'result':
                            value['__task_index__'] = index
                            value['robustness_elapsed_s'] = time.monotonic()-slot['started']
                            self.ready.append(value)
                        elif kind == 'error':
                            self.ready.append(self._failure(index, 'worker_exception', slot, **value))
                        else:
                            raise RuntimeError(f"unknown worker message {kind}")
                        slot['index'] = None
                        continue
                    except (EOFError, OSError) as exc:
                        self._stop(slot)
                        self.ready.append(self._failure(index, 'worker_died', slot, detail=str(exc)))
                        self.slots.remove(slot)
                        continue
                if not slot['process'].is_alive():
                    self._stop(slot)
                    self.ready.append(self._failure(index, 'worker_died', slot))
                    self.slots.remove(slot)
                elif time.monotonic()-slot['progress'] > self.progress_timeout:
                    self._stop(slot)
                    self.ready.append(self._failure(index, 'watchdog_stall', slot,
                                                    progress_budget_s=self.progress_timeout))
                    self.slots.remove(slot)
            if self.ready or readable:
                # A progress event can precede an already-buffered final result.
                # Drain the pipe fully before declaring any outstanding work.
                continue
            if not self.deadline_reached and time.monotonic()-self.started > self.course_timeout:
                self.deadline_reached = True
                for slot in self.slots:
                    self._stop(slot)
                    if slot['index'] is not None:
                        self.ready.append(self._failure(slot['index'], 'course_deadline', slot,
                                                        task_state='running', course_budget_s=self.course_timeout))
                self.slots.clear()
                for index in self.pending:
                    self.ready.append(self._failure(index, 'course_deadline', task_state='not_started',
                                                    course_budget_s=self.course_timeout))
                self.pending.clear()
                continue
            if time.monotonic() >= end:
                raise TimeoutError
            wait([s['connection'] for s in self.slots], timeout=min(0.05, max(0, end-time.monotonic())))


def technical_rows(result, identity, run_identifier, mask_identity, task_params=None):
    from .radiomics_ct_contract import CT_EXTRACTION_ARMS
    from .radiomics_robustness import ROBUSTNESS_MEASUREMENT_TYPE
    failure = result['__technical_failure__']
    task_params = task_params or {}
    return [{**identity, 'roi_name': result['roi_name'],
             'perturbation_id': result['perturbation_id'], 'extraction_arm': arm,
             'measurement_type': ROBUSTNESS_MEASUREMENT_TYPE, 'modality': 'CT',
             'run_identifier': run_identifier, 'perturbed_mask_identity': mask_identity,
             'configured_parameter_hash': task_params.get('configured_parameter_hashes', {}).get(arm),
             'code_revision': task_params.get('code_revision'),
             'robustness_status': 'technical_failure', 'reason_code': failure['reason_code'],
             'technical_evidence': json.dumps(failure['evidence'], sort_keys=True),
             'feature_name': None, 'value': None} for arm in CT_EXTRACTION_ARMS]


def validate_technical_frame(frame, expected_ids, context, identity=None):
    """Technical non-measurements reconcile inventory, never satisfy success."""
    from .radiomics_ct_contract import CT_EXTRACTION_ARMS
    from .radiomics_robustness import ROBUSTNESS_MEASUREMENT_TYPE
    allowed = {'worker_exception', 'worker_died', 'watchdog_stall', 'course_deadline'}
    ids = set(frame['perturbation_id'].astype(str))
    if not ids <= expected_ids:
        raise RuntimeError(f"unexpected technical conditions for {context}")
    for pid, group in frame.groupby('perturbation_id'):
        if len(group) != len(CT_EXTRACTION_ARMS) or set(group.extraction_arm) != set(CT_EXTRACTION_ARMS):
            raise RuntimeError(f"incomplete technical non-measurement arms for {context}/{pid}")
        if group.value.notna().any() or group.feature_name.notna().any():
            raise RuntimeError("technical failure contains numeric measurements")
        if group.reason_code.nunique() != 1 or group.technical_evidence.nunique() != 1:
            raise RuntimeError("discordant technical non-measurement arms")
        for row in group.to_dict('records'):
            if row['reason_code'] not in allowed or not json.loads(row['technical_evidence']):
                raise RuntimeError("unexplained technical non-measurement")
            if row.get('measurement_type') != ROBUSTNESS_MEASUREMENT_TYPE:
                raise RuntimeError("invalid technical measurement type")
            if identity is not None and any(str(row.get(k,'')) != v for k,v in identity.as_dict().items()):
                raise RuntimeError("technical non-measurement identity mismatch")
    return ids
