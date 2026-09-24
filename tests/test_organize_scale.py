"""Synthetic contracts for bounded organize workers."""
from dataclasses import asdict
import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from rtpipeline import meta, organize_scale as scale, plan_disposition, utils
from organize_io_fixture import baseline, synthetic, require_process_pool


def test_fingerprint_order_and_failures(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ('z', 'a', 'b')]
    for p in paths:
        p.write_bytes(p.name.encode())
    expected = hashlib.sha256(json.dumps(sorted(
        (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths)).encode()).hexdigest()
    assert plan_disposition.source_scope_fingerprint(tmp_path) == expected
    assert plan_disposition.source_scope_fingerprint(tmp_path / 'absent') is None
    original = Path.stat
    def fail(p, *args, **kwargs):
        if p.name in ('z', 'a'):
            raise PermissionError(p.name)
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', fail)
    monkeypatch.setattr(plan_disposition, '_scoped_walk', lambda *a, **k: iter([(tmp_path, [], ['z', 'a', 'b'])]))
    with pytest.raises(PermissionError, match='z'):
        plan_disposition.source_scope_fingerprint(tmp_path)


def test_inventory_uses_two_snapshots(tmp_path, monkeypatch):
    from synthetic_rt_fixtures import make_record
    from pydicom.uid import generate_uid
    path = make_record(tmp_path / 'record.dcm', generate_uid())
    original = Path.stat
    calls = []
    def counted(p, *args, **kwargs):
        calls.append(p)
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', counted)
    meta._metadata_source_file(path)
    assert calls == [path, path]
    read = meta.read_discovery_header
    def changed(p):
        ds = read(p)
        os.utime(p, ns=(1, 1))
        return ds
    monkeypatch.setattr(meta, 'read_discovery_header', changed)
    with pytest.raises(meta.MetadataExportError, match='changed while'):
        meta._metadata_source_file(path)


@pytest.mark.parametrize('processes', [False, True])
def test_discovery_roundtrip_order_cache_errors_and_fallback(tmp_path, monkeypatch, processes):
    synthetic(tmp_path)
    paths = sorted(tmp_path.rglob('*.dcm'), reverse=True)
    paths += [tmp_path / 'missing.dcm']
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '2')
    if processes:
        require_process_pool()
    else:
        def denied(*args, **kwargs):
            raise PermissionError('synthetic pool denial')
        monkeypatch.setattr(scale, 'ProcessPoolExecutor', denied)
    if processes:
        from contextlib import contextmanager
        original_results = scale.ordered_process_results
        @contextmanager
        def observed(*args, **kwargs):
            with original_results(*args, **kwargs) as results:
                def checked():
                    for outcome in results:
                        assert outcome.worker_pid not in (None, os.getpid())
                        yield outcome
                yield checked()
        monkeypatch.setattr(scale, 'ordered_process_results', observed)
    old = baseline()
    @old['utils'].organize_read_cache
    def expected():
        rows = [old['meta']._metadata_source_file(p) for p in paths[:-1]]
        cache = old['utils']._organize_reads.get()
        return rows, cache
    rows, cache = expected()
    @utils.organize_read_cache
    def actual():
        stream = scale.discover_sources(paths, 4)
        values = [next(stream) for _ in rows]
        with pytest.raises(FileNotFoundError):
            next(stream)
        current = utils._organize_reads.get()
        assert current.identities == cache.identities
        assert current.records == cache.records
        for key in cache.identities:
            assert current.identities[key].sop_instance_uid == cache.identities[key].sop_instance_uid
        for before, after in zip(rows, values):
            assert asdict(before.source) == asdict(after.source)
            assert before.dataset == after.dataset
            assert before.dataset.file_meta == after.dataset.file_meta
            assert before.dataset.filename == after.dataset.filename
            assert before.dataset.preamble == after.dataset.preamble
            assert pickle.loads(pickle.dumps(after.dataset)) == before.dataset
    actual()


def test_pool_closes_on_consumer_error(monkeypatch):
    events = []
    class Future:
        def result(self):
            return scale.Outcome(1, None, [], [])
    class Pool:
        def __init__(self, **kwargs):
            pass
        def submit(self, *args):
            events.append('submit')
            return Future()
        def shutdown(self, **kwargs):
            events.append(('shutdown', kwargs))
    monkeypatch.setattr(scale, 'ProcessPoolExecutor', Pool)
    with pytest.raises(ValueError, match='consumer'):
        with scale.ordered_process_results(range(100), abs, 2) as results:
            assert next(results).result() == 1
            raise ValueError('consumer')
    assert events.count('submit') == 5  # startup + bounded four-job window
    assert events[-1] == ('shutdown', dict(wait=True, cancel_futures=True))


def test_mask_exception_retains_course_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    course = SimpleNamespace(rs_path=None, primary_nifti=None,
        dirs=SimpleNamespace(dicom_ct=tmp_path, segmentation_original=tmp_path, root=tmp_path,
                             metadata=tmp_path / "metadata"))
    def failed(args):
        raise ValueError('synthetic export failure')
    monkeypatch.setattr(scale, 'export_mask', failed)
    with scale.mask_exports([course], True) as results:
        result = next(results)
        with pytest.raises(ValueError, match='synthetic export failure'):
            result.result()


@pytest.mark.parametrize('processes', [False, True])
def test_mask_nifti_bytes(tmp_path, monkeypatch, processes):
    import SimpleITK as sitk
    from rtpipeline import organize
    synthetic(tmp_path / 'input')
    folder = tmp_path / 'input/SYNTH_A/0'
    ct = tmp_path / 'ct'
    ct.mkdir()
    import shutil
    for path in folder.glob('ct*.dcm'):
        shutil.copyfile(path, ct / path.name)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in sorted(ct.glob('*.dcm'))])
    nifti = tmp_path / 'ct.nii.gz'
    sitk.WriteImage(reader.Execute(), str(nifti))
    args = dict(rs_path=folder / 'struct.dcm', primary_nifti=nifti,
                dicom_ct_dir=ct, segmentation_original_dir=tmp_path / 'masks',
                log_root=tmp_path, overwrite=True)
    expected_manifest = baseline()['organize']._export_original_segmentation_from_paths(**args)
    assert expected_manifest and len(expected_manifest['structures']) == 2
    expected = {p.name: p.read_bytes() for p in (tmp_path / 'masks').rglob('*.nii.gz')}
    assert expected
    if processes:
        require_process_pool()
    else:
        monkeypatch.setattr(scale, 'ProcessPoolExecutor', lambda **k: (_ for _ in ()).throw(PermissionError()))
    with scale.ordered_process_results([args], scale.export_mask, 2) as results:
        outcome = next(results)
        if processes:
            assert outcome.worker_pid not in (None, os.getpid())
        assert outcome.result() == expected_manifest
    assert {p.name: p.read_bytes() for p in (tmp_path / 'masks').rglob('*.nii.gz')} == expected


def test_discovery_transport_pickle_preserves_dataset_and_cache(tmp_path):
    synthetic(tmp_path)
    paths = sorted(tmp_path.rglob('*.dcm'))
    expected = scale._discover_chunk((paths, 4, True, True))
    actual = pickle.loads(pickle.dumps(expected))
    for before, after in zip(expected[0], actual[0]):
        assert asdict(before.value.source) == asdict(after.value.source)
        x, y = before.value.dataset, after.value.dataset
        assert x == y and x.file_meta == y.file_meta
        assert x.filename == y.filename and x.preamble == y.preamble
        assert (x.is_little_endian, x.is_implicit_VR) == (y.is_little_endian, y.is_implicit_VR)
    assert expected[1:] == actual[1:]
    for key, identity in expected[1].items():
        assert actual[1][key].sop_instance_uid == identity.sop_instance_uid


def test_mask_worker_error_quarantines_course(tmp_path, monkeypatch):
    from rtpipeline import organize, segmentation
    from rtpipeline.config import PipelineConfig
    synthetic(tmp_path / 'input')
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '1')
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    monkeypatch.setattr(organize, 'run_dcm2niix', lambda *a, **k: None)
    monkeypatch.setattr(segmentation, 'run_dcm2niix', lambda *a, **k: None)
    def failed(args):
        raise ValueError('synthetic mask failure')
    monkeypatch.setattr(scale, 'export_mask', failed)
    cfg = PipelineConfig(tmp_path / 'input', tmp_path / 'output', tmp_path / 'logs',
                         max_workers_override=2, dicom_copy_use_hardlinks=False)
    assert organize.organize_and_merge(cfg, metadata_snapshot={}) == []
    ledger = json.loads((cfg.output_root / '_COURSES/organize_ledger.json').read_text())
    assert ledger['technical_quarantine_count'] == 4
    assert ledger['validated_course_count'] == 0


def test_mask_export_revokes_publication_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    marker = tmp_path / '.organized'
    marker.touch()
    course = SimpleNamespace(rs_path=None, primary_nifti=None,
        dirs=SimpleNamespace(dicom_ct=tmp_path, segmentation_original=tmp_path,
                             root=tmp_path, metadata=tmp_path / 'metadata'))
    def export(args):
        assert not marker.exists()
        assert course.dirs.metadata.is_dir()
        return {'prepared': True}
    monkeypatch.setattr(scale, 'export_mask', export)
    with scale.mask_exports([course], True) as results:
        assert next(results).result() == {'prepared': True}
