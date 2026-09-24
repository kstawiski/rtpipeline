"""Synthetic phase benchmark; run directly, never collected as a test.

Measures parent plus descendant CPU time (100% = one occupied CPU). Injects
latency only for fixture read opens/closes and stats, including spawned workers.
"""
import argparse
import builtins
from contextlib import contextmanager
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from functools import partial
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organize_io_fixture import baseline, synthetic
from rtpipeline import meta, organize, organize_scale as scale, plan_disposition, utils


class DelayedReader(io.BufferedReader):
    """Remain a real BufferedReader so pydicom does not retain a stream buffer."""
    def close(self):
        if not self.closed:
            time.sleep(self.latency)
        super().close()


def install_latency(root, delay):
    import pydicom.dataset
    # FileDataset stores the open constructor for deferred reads. Keep that
    # constructor picklable when builtins.open is temporarily instrumented.
    pydicom.dataset.open = builtins.open
    root = str(root) + os.sep
    def wrap(fn):
        def opened(path, mode='r', *args, **kwargs):
            active = isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).startswith(root) and 'r' in mode
            if active:
                time.sleep(delay)
            stream = fn(path, mode, *args, **kwargs)
            if not active:
                return stream
            if isinstance(stream, io.BufferedReader):
                buffering = kwargs.get('buffering', args[0] if args else -1)
                buffered = DelayedReader(stream.detach(), buffer_size=(
                    buffering if buffering > 0 else io.DEFAULT_BUFFER_SIZE))
                buffered.latency = delay
                return buffered
            class Stream:
                def __getattr__(self, name):
                    return getattr(stream, name)
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    time.sleep(delay)
                    return stream.__exit__(*args)
            return Stream()
        return opened
    builtins.open = wrap(builtins.open)
    Path.open = wrap(Path.open)
    stat = os.stat
    def delayed_stat(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).startswith(root):
            time.sleep(delay / 3)
        return stat(path, *args, **kwargs)
    os.stat = delayed_stat


@contextmanager
def latency(root, delay, modules=()):
    import pydicom.dataset
    dataset_open = getattr(pydicom.dataset, 'open', None)
    originals = builtins.open, Path.open, os.stat
    install_latency(root, delay)
    # git-show modules have a private copy of builtins for redirected imports.
    # Patch their captured open too, or baseline header reads miss the latency.
    captured = [(module.__dict__['__builtins__'], module.__dict__['__builtins__']['open'])
                for module in modules]
    for namespace, _ in captured:
        namespace['open'] = builtins.open
    try:
        yield
    finally:
        for namespace, original_open in captured:
            namespace['open'] = original_open
        builtins.open, Path.open, os.stat = originals
        if dataset_open is None:
            del pydicom.dataset.open
        else:
            pydicom.dataset.open = dataset_open


def measure(fn):
    import psutil
    parent = psutil.Process()
    cpu = {}
    stop = threading.Event()
    def sample():
        for process in [parent, *parent.children(recursive=True)]:
            try:
                times = process.cpu_times()
                cpu[process.pid] = times.user + times.system
            except psutil.Error:
                pass
    sample()
    initial = dict(cpu)
    def poll():
        while not stop.wait(.02):
            sample()
    thread = threading.Thread(target=poll)
    start = time.perf_counter()
    thread.start()
    try:
        value = fn()
    finally:
        sample()
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
    seconds = sum(value - initial.get(pid, 0) for pid, value in cpu.items())
    return value, dict(wall_seconds=elapsed, cpu_seconds=seconds,
                       cpu_percent=100 * seconds / elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slices', type=int, default=256)
    parser.add_argument('--records', type=int, default=256)
    parser.add_argument('--delay', type=float, default=.005)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    before = baseline()
    from concurrent.futures import ProcessPoolExecutor
    available = True
    try:
        with ProcessPoolExecutor(2) as pool:
            pool.submit(abs, -1).result()
    except (OSError, RuntimeError) as exc:
        available = False
        failure = repr(exc)
    report = {'process_pool_available': available}
    if not available:
        report['process_pool_failure'] = failure
    with tempfile.TemporaryDirectory(prefix='.scale-benchmark-', dir=ROOT) as scratch:
        scratch = Path(scratch)
        root = scratch / 'input'
        synthetic(root, ct_slices=args.slices, ct_size=64,
                  records_per_course=args.records,
                  roi_names=('BODY', *[f'PTV{i}' for i in range(7)]), all_slices=True)
        paths = sorted(root.rglob('*.dcm'))
        report['fixture'] = dict(files=len(paths), courses=4, rois_per_course=8,
                                 slices_per_course=args.slices,
                                 open_close_latency_seconds=args.delay,
                                 stat_latency_seconds=args.delay / 3)
        mask_args = []
        import SimpleITK as sitk
        for i, folder in enumerate(sorted(root.glob('*/*'))):
            if not folder.is_dir():
                continue
            ct = scratch / f'ct{i}'
            ct.mkdir()
            for path in folder.glob('ct*.dcm'):
                shutil.copyfile(path, ct / path.name)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([str(p) for p in sorted(ct.glob('*.dcm'), key=lambda p: int(p.stem[2:]))])
            nifti = scratch / f'ct{i}.nii.gz'
            sitk.WriteImage(reader.Execute(), str(nifti))
            mask_args.append(dict(rs_path=folder / 'struct.dcm', primary_nifti=nifti,
                                  dicom_ct_dir=ct, segmentation_original_dir=scratch / f'masks{i}',
                                  log_root=scratch, overwrite=True))
        with patch.object(scale, 'ProcessPoolExecutor', partial(ProcessPoolExecutor,
                         initializer=install_latency, initargs=(root, args.delay))):
            for label in ('before', 'after'):
                result = {}
                u = before['utils'] if label == 'before' else utils
                fingerprint = before['plan_disposition'].source_scope_fingerprint if label == 'before' else plan_disposition.source_scope_fingerprint
                with latency(root, args.delay, before.values()):
                    _, result['fingerprint'] = measure(lambda: fingerprint(root))
                    @u.organize_read_cache
                    def discovery():
                        values = (u.parallel_map_files(paths, before['meta']._metadata_source_file, 64)
                                  if label == 'before' else scale.discover_sources(paths, 64))
                        return sum(1 for _ in values)
                    _, result['discovery'] = measure(discovery)
                    def masks():
                        if label == 'before':
                            return [before['organize']._export_original_segmentation_from_paths(**a) for a in mask_args]
                        with scale.ordered_process_results(mask_args, scale.export_mask, min(4, scale.process_count())) as values:
                            return [v.result() for v in values]
                    manifests, result['mask_export'] = measure(masks)
                    assert all(m and len(m['structures']) == 8 for m in manifests)
                report[label] = result
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
