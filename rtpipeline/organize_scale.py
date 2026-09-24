"""Bounded process work for organize; publication remains in the parent."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import wraps
from dataclasses import dataclass
from itertools import islice
import logging
import multiprocessing
import os
import warnings

from . import utils


def process_count(variable="RTPIPELINE_INDEX_PROCESSES"):
    default = max(1, min(16, (os.cpu_count() or 1) // 4))
    try:
        return max(1, min(64, int(os.environ.get(variable, default))))
    except ValueError:
        return default


@dataclass
class Outcome:
    value: object
    error: Exception | None
    logs: list
    notices: list
    worker_pid: int | None = None

    def result(self):
        for record in self.logs:
            logging.getLogger(record.name).handle(record)
        for message, category in self.notices:
            warnings.warn(message, category)
        if self.error is not None:
            raise self.error
        return self.value


def _invoke(fn, arg, level):
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            # Format before pickling: args and traceback objects need not pickle.
            record.msg, record.args = record.getMessage(), ()
            record.exc_info = None
            records.append(record)
    root = logging.getLogger()
    root.handlers = [Capture()]
    root.setLevel(level)
    with warnings.catch_warnings(record=True) as notices:
        try:
            value, error = fn(arg), None
        except Exception as exc:
            value, error = None, exc
    return Outcome(value, error, records,
                   [(str(w.message), w.category) for w in notices], os.getpid())


def _ready():
    return os.getpid()


@contextmanager
def ordered_process_results(items, fn, workers, *, fallback=None):
    """At most two jobs per process, in input order; close even on consumer error.

    Only pool startup failure falls back. A worker failure after side effects
    begin must propagate, never silently repeat a mask write.
    """
    stream = iter(items)
    executor = None
    if workers > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            executor.submit(_ready).result()
        except (OSError, RuntimeError, BrokenProcessPool):
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            executor = None
    def results():
        if executor is None:
            for item in stream:
                try:
                    yield Outcome(fn(item), None, [], [])
                except Exception as exc:
                    yield Outcome(None, exc, [], [])
            return
        pending = deque()
        level = logging.getLogger().getEffectiveLevel()
        for item in islice(stream, 2 * workers):
            pending.append(executor.submit(_invoke, fn, item, level))
        while pending:
            yield pending.popleft().result()
            for item in islice(stream, 1):
                pending.append(executor.submit(_invoke, fn, item, level))
    try:
        yield fallback() if executor is None and fallback is not None else results()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _discover_chunk(args):
    from .meta import _metadata_source_file
    paths, threads, use_cache, source_reads = args
    cache = utils.OrganizeReadCache() if use_cache else None
    token = utils._organize_reads.set(cache)
    try:
        # Return successful prefixes too, so a later exception cannot suppress
        # the callbacks or inventory entries for earlier inputs.
        def read(path):
            try:
                return Outcome((_metadata_source_file(path) if source_reads else
                                utils.read_organize_discovery_dicom(path)), None, [], [])
            except Exception as exc:
                return Outcome(None, exc, [], [])
        rows = list(utils.parallel_map_files(paths, read, threads))
        return rows, cache.identities if cache else {}, cache.records if cache else {}
    finally:
        utils._organize_reads.reset(token)


def discover_sources(paths, threads, *, source_reads=True):
    from .meta import _metadata_source_file
    workers = process_count()
    reader = _metadata_source_file if source_reads else utils.read_organize_discovery_dicom
    if workers == 1 or (len(paths) < 64 and "RTPIPELINE_INDEX_PROCESSES" not in os.environ):
        yield from utils.parallel_map_files(paths, reader, threads)
        return
    cache = utils._organize_reads.get()
    stream = iter(paths)
    def chunks():
        while chunk := list(islice(stream, 32)):
            yield chunk, max(1, (threads + workers - 1) // workers), cache is not None, source_reads
    def fallback():
        # Retain the original thread count, run cache, and streaming boundary.
        for row in utils.parallel_map_files(paths, reader, threads):
            yield Outcome(([Outcome(row, None, [], [])], {}, {}), None, [], [])
    with ordered_process_results(chunks(), _discover_chunk, workers,
                                 fallback=fallback) as batches:
        for batch in batches:
            rows, identities, records = batch.result()
            if cache is not None:
                cache.identities.update(identities)
                cache.records.update(records)
            for row in rows:
                yield row.result()


def export_mask(args):
    from .organize import _export_original_segmentation_from_paths
    return _export_original_segmentation_from_paths(**args)


@contextmanager
def mask_exports(courses, overwrite):
    def args():
        for co in courses:
            # Revoke publication in the parent BEFORE any worker can write.
            # This is the same preparation that preceded the serial export.
            co.dirs.metadata.mkdir(parents=True, exist_ok=True)
            (co.dirs.root / ".organized").unlink(missing_ok=True)
            yield dict(rs_path=co.rs_path, primary_nifti=co.primary_nifti,
                       dicom_ct_dir=co.dirs.dicom_ct,
                       segmentation_original_dir=co.dirs.segmentation_original,
                       log_root=co.dirs.root, overwrite=overwrite)
    workers = min(len(courses), process_count("RTPIPELINE_MASK_PROCESSES"))
    with ordered_process_results(args(), export_mask, workers) as results:
        yield results


_resources = ContextVar("organize_process_resources", default=None)


def organize_process_resources(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        with ExitStack() as stack:
            token = _resources.set(stack)
            try:
                return fn(*args, **kwargs)
            finally:
                _resources.reset(token)
    return run


def prepare_mask_exports(courses, overwrite):
    return _resources.get().enter_context(mask_exports(courses, overwrite))
