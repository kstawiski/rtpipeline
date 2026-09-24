"""Opt-in synthetic latency profile (not collected by pytest).

Run with the prescribed interpreter and PYTHONNOUSERSITE/PYTHONDONTWRITEBYTECODE.
Counts Python read/readinto bytes plus sendfile bytes, NOT kernel/NFS readahead.
The injected delay applies to read opens beneath the disposable fixture root.
"""
import builtins
import cProfile
import collections
import io
import json
import logging
import os
import pstats
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organize_io_fixture import baseline, synthetic
from rtpipeline import organize, segmentation
from rtpipeline.config import PipelineConfig


class Meter:
    def __init__(self, root, delay):
        self.root = str(root) + os.sep
        self.delay = delay
        self.rows = collections.defaultdict(lambda: dict(seconds=0., opens=0, bytes_read=0))
        self.phase = 'other'
        self.started = time.perf_counter()
        self.lock = threading.Lock()

    def switch(self, phase):
        now = time.perf_counter()
        self.rows[self.phase]['seconds'] += now - self.started
        self.phase, self.started = phase, now

    def add(self, name, count):
        with self.lock:
            self.rows[self.phase][name] += count

    def wrap(self, fn):
        meter = self
        class Stream:
            def __init__(self, stream):
                self.stream = stream
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def read(self, *args):
                value = self.stream.read(*args)
                meter.add('bytes_read', len(value))
                return value
            def readinto(self, *args):
                value = self.stream.readinto(*args)
                meter.add('bytes_read', value or 0)
                return value
        def opened(path, mode='r', *args, **kwargs):
            counted = isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).startswith(meter.root) and 'r' in mode and 'b' in mode
            if counted:
                meter.add('opens', 1)
                time.sleep(meter.delay)
            stream = fn(path, mode, *args, **kwargs)
            return Stream(stream) if counted else stream
        return opened

    def phase_wrapper(self, fn, phase):
        def run(*args, **kwargs):
            previous = self.phase
            self.switch(phase)
            try:
                return fn(*args, **kwargs)
            finally:
                self.switch(previous)
        return run


@contextmanager
def instrument(module, meter):
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch('builtins.open', meter.wrap(builtins.open)))
        stack.enter_context(patch('io.open', meter.wrap(io.open)))
        if hasattr(os, 'sendfile'):
            sendfile = os.sendfile
            def counted_sendfile(*args, **kwargs):
                count = sendfile(*args, **kwargs)
                meter.add('bytes_read', count)
                return count
            stack.enter_context(patch('os.sendfile', counted_sendfile))
        for name, phase in (
            ('extract_rt_with_records', 'discovery'),
            ('_delivery_reference_audit', 'reference_audits'),
            ('_record_delivery_evidence', 'delivery_evidence'),
            ('_safe_copy', 'copy'), ('_copy_into', 'copy'),
            ('copy_ct_series', 'copy'),
            ('build_source_plan_dispositions', 'source_dispositions'),
        ):
            stack.enter_context(patch.object(module, name, meter.phase_wrapper(getattr(module, name), phase)))
        # This fixture tests organize I/O, not external image conversion.
        stack.enter_context(patch.object(module, 'run_dcm2niix', lambda *a, **k: None))
        stack.enter_context(patch.object(segmentation, 'run_dcm2niix', lambda *a, **k: None))
        yield


def main():
    logging.disable(logging.CRITICAL)
    old = baseline()['organize']
    results = {}
    with tempfile.TemporaryDirectory(prefix='.organize-benchmark-', dir=ROOT) as scratch:
        scratch = Path(scratch)
        root, output = scratch / 'input', scratch / 'output'
        records = synthetic(root, records_per_course=16, ct_slices=32, ct_size=512)
        config = PipelineConfig(root, output, scratch / 'logs', max_workers_override=1,
                                dicom_copy_use_hardlinks=False)
        for mode, module in [('before', old), ('after', organize)]:
            meter = Meter(scratch, .025)
            profile = cProfile.Profile()
            start = time.perf_counter()
            with instrument(module, meter):
                profile.enable()
                module.organize_and_merge(config, metadata_snapshot={})
                profile.disable()
            meter.switch('finished')
            elapsed = time.perf_counter() - start
            stats = pstats.Stats(profile)
            hot = sorted((dict(function=name, calls=nc, cumulative_seconds=ct)
                          for (_, _, name), (_, nc, _, ct, _) in stats.stats.items()),
                         key=lambda row: row['cumulative_seconds'], reverse=True)[:20]
            results[mode] = dict(wall_seconds=elapsed, phases=dict(meter.rows), profile=hot)
            shutil.rmtree(output)
        results['fixture'] = dict(patients=2, courses=4, records=len(records),
                                  input_files=len(list(root.rglob('*.dcm'))),
                                  per_read_open_delay_seconds=.025,
                                  cpu_workers=1, after_io_workers=organize.DEFAULT_INDEX_WORKERS)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
