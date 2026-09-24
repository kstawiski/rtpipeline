"""Performance contracts, using synthetic files only."""
import collections
from dataclasses import asdict
import os
import shutil
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline import meta, organize, segmentation, utils
from rtpipeline.config import PipelineConfig
from rtpipeline.dicom_copy import DicomCopyConfig, DicomCopyManager
from organize_io_fixture import baseline, synthetic, tree_bytes, publish_manifest


@pytest.mark.parametrize('modality', ['CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE', 'RTRECORD'])
@pytest.mark.parametrize('projection', [None, ['SOPInstanceUID', 'Modality']])
@pytest.mark.parametrize('preamble', [None, b'\0' * 128])
def test_header_reader_matches_path(tmp_path, modality, projection, preamble):
    path = tmp_path / 'synthetic.dcm'
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=preamble)
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = modality
    ds.PatientID = 'SYNTH'
    nested = Dataset()
    nested.ReferencedSOPInstanceUID = generate_uid()
    ds.ReferencedRTPlanSequence = [nested]
    ds.PixelData = b'\0' * (256 * 1024)
    ds.save_as(path)
    kwargs = dict(stop_before_pixels=True, force=preamble is None, specific_tags=projection)
    old = pydicom.dcmread(str(path), **kwargs)
    new = utils.read_dicom_header(path, **kwargs)
    assert new == old
    assert new.file_meta == old.file_meta
    assert new.preamble == old.preamble
    assert new.filename == old.filename == str(path)
    assert isinstance(new.filename, str)
    assert (new.is_little_endian, new.is_implicit_VR) == (old.is_little_endian, old.is_implicit_VR)
    assert 'PixelData' not in new


@pytest.mark.parametrize('content', [b'', b'not dicom', b'\0' * 128 + b'DICM\x08\0\x18\0UI\xff\xff'])
@pytest.mark.parametrize('force', [False, True])
def test_header_reader_preserves_corrupt_behavior(tmp_path, content, force):
    path = tmp_path / 'bad.dcm'
    path.write_bytes(content)
    try:
        old = pydicom.dcmread(str(path), stop_before_pixels=True, force=force)
    except Exception as exc:
        with pytest.raises(type(exc)) as raised:
            utils.read_dicom_header(path, stop_before_pixels=True, force=force)
        assert str(raised.value) == str(exc)
    else:
        assert utils.read_dicom_header(path, stop_before_pixels=True, force=force) == old


def test_advice_failure_is_only_a_performance_fallback(tmp_path, monkeypatch):
    from synthetic_rt_fixtures import make_record
    path = make_record(tmp_path / 'record.dcm', generate_uid())
    calls = []
    def advice(*args):
        calls.append(args)
        raise OSError('unsupported synthetic filesystem')
    monkeypatch.setattr(os, 'posix_fadvise', advice, raising=False)
    assert utils.read_dicom_header(path, stop_before_pixels=True) == pydicom.dcmread(path, stop_before_pixels=True)
    assert len(calls) == 1
    assert calls[0][1:] == (0, 0, os.POSIX_FADV_RANDOM)
    monkeypatch.delattr(os, 'posix_fadvise')
    assert utils.read_dicom_header(path, stop_before_pixels=True) == pydicom.dcmread(path, stop_before_pixels=True)


def test_cache_invalidates_and_is_scoped(tmp_path, monkeypatch):
    from synthetic_rt_fixtures import make_record
    path = make_record(tmp_path / 'record.dcm', generate_uid())
    original = utils.read_dicom_header
    reads = []
    def observed(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(utils, 'read_dicom_header', observed)
    @utils.organize_read_cache
    def run():
        utils.read_discovery_header(path)
        manager = DicomCopyManager(DicomCopyConfig(), tmp_path / 'out')
        uid = manager._get_sop_uid(path)
        assert uid == organize._sop_instance_uid(path)
        assert utils.read_record_header(path).SOPInstanceUID == uid
        assert len(reads) == 1
        previous_mtime = path.stat().st_mtime_ns
        ds = pydicom.dcmread(path)
        ds.SOPInstanceUID = generate_uid()
        ds.save_as(path)
        os.utime(path, ns=(previous_mtime + 1000000000, previous_mtime + 1000000000))
        assert manager._get_sop_uid(path) == ds.SOPInstanceUID
        assert utils.read_record_header(path).SOPInstanceUID == ds.SOPInstanceUID
        assert len(reads) == 3
    run()
    assert utils._organize_reads.get() is None


# More than one course worker is scheduling-dependent in the baseline itself (cache key order,
# SOP-registry owner, related-file order); cross-worker determinism is tested separately.
@pytest.mark.parametrize('course_workers,processes', [(1, 1), (1, 2)])
def test_organize_tree_matches_pinned_baseline(tmp_path, monkeypatch, caplog, course_workers, processes):
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', str(processes))
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', str(processes))
    if processes > 1:
        from organize_io_fixture import require_process_pool
        require_process_pool()
    root, out = tmp_path / 'input', tmp_path / 'output'
    records = synthetic(root)
    # XLSX creation times also feed the metadata-cache content hashes. Freeze
    # that packaging clock so the cache manifest itself is byte comparable.
    import datetime
    import xlsxwriter.core
    class FixedWorkbookTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, tzinfo=tz)
    monkeypatch.setattr(xlsxwriter.core, 'datetime', FixedWorkbookTime)
    before = baseline()
    config = PipelineConfig(root, out, tmp_path / 'logs', max_workers_override=course_workers,
                            dicom_copy_use_hardlinks=False)
    monkeypatch.setattr(before['organize'], 'run_dcm2niix', lambda *args, **kwargs: None)
    monkeypatch.setattr(organize, 'run_dcm2niix', lambda *args, **kwargs: None)
    monkeypatch.setattr(segmentation, 'run_dcm2niix', lambda *args, **kwargs: None)
    snapshot_before = {}
    courses_before = before['organize'].organize_and_merge(config, metadata_snapshot=snapshot_before)
    before['meta'].export_metadata(config, source_snapshot=snapshot_before)
    publish_manifest(courses_before, out)
    expected = tree_bytes(out)
    expected_workbooks = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*.xlsx')}
    expected_warnings = [(r.levelno, r.getMessage()) for r in caplog.records]
    caplog.clear()
    assert len(courses_before) == 4
    assert any(name.endswith('case_metadata.json') for name in expected)
    assert any(name.endswith('.done.json') for name in expected)
    shutil.rmtree(out)
    reads = collections.Counter()
    reuse = collections.Counter()
    original = pydicom.dcmread
    def counted(path, *args, **kwargs):
        filename = Path(path.name if hasattr(path, 'read') else path)
        if filename in records:
            reads[filename] += 1
        if hasattr(path, 'read'):
            reuse['helper'] += 1
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pydicom, 'dcmread', counted)
    identity = utils.cached_source_identity
    def cached(path):
        value = identity(path)
        if value is not None:
            reuse['identity'] += 1
        return value
    monkeypatch.setattr(organize, 'cached_source_identity', cached)
    import rtpipeline.dicom_copy as copying
    monkeypatch.setattr(copying, 'cached_source_identity', cached)
    scan_workers = []
    original_scan = organize.extract_rt_with_records
    def observed_scan(*args, **kwargs):
        scan_workers.append(kwargs['max_workers'])
        return original_scan(*args, **kwargs)
    monkeypatch.setattr(organize, 'extract_rt_with_records', observed_scan)
    snapshot_after = {}
    courses_after = organize.organize_and_merge(config, metadata_snapshot=snapshot_after)
    meta.export_metadata(config, source_snapshot=snapshot_after)
    publish_manifest(courses_after, out)
    actual = tree_bytes(out)
    # The fixed workbook clock permits an additional exact ZIP-byte check.
    assert {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*.xlsx')} == expected_workbooks
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name] == expected[name], name
    assert [asdict(course) for course in courses_after] == [asdict(course) for course in courses_before]
    assert asdict(snapshot_before['identity']) == asdict(snapshot_after['identity'])
    assert [asdict(row) for row in snapshot_before['results']] == [asdict(row) for row in snapshot_after['results']]
    assert snapshot_before['candidates'] == snapshot_after['candidates']
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == expected_warnings
    assert reads == ({path: 1 for path in records} if processes == 1 else {})
    assert reuse['identity'] > 0
    if processes == 1:
        assert reuse['helper'] > 0
    assert config.effective_workers() == course_workers
    assert scan_workers == [utils.DEFAULT_INDEX_WORKERS]


@pytest.mark.parametrize('summary', [False, True])
@pytest.mark.parametrize('preamble', [False, True])
def test_record_projection_preserves_all_delivery_evidence(tmp_path, monkeypatch, caplog, summary, preamble):
    from synthetic_rt_fixtures import make_record
    from pydicom.sequence import Sequence
    path = make_record(tmp_path / 'record.dcm', generate_uid())
    ds = pydicom.dcmread(path)
    item = Dataset()
    item.ReferencedDoseReferenceNumber = 1
    item.CalculatedDoseReferenceDoseValue = 2.0
    ds.ReferencedCalculatedDoseReferenceSequence = Sequence([item])
    ds.TreatmentSessionBeamSequence[0].ReferencedCalculatedDoseReferenceSequence = Sequence([item])
    ds.CurrentFractionNumber = 2
    ds.SeriesDate = '20240101'
    if summary:
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.7'
        cumulative = Dataset()
        cumulative.ReferencedDoseReferenceNumber = 1
        cumulative.CumulativeDoseToDoseReference = 6.0
        ds.TreatmentSummaryCalculatedDoseReferenceSequence = Sequence([cumulative])
    if not preamble:
        ds.preamble = None
    ds.save_as(path)
    old = baseline()['organize']
    expected = old._record_delivery_evidence([path, path])
    expected_audit = old._delivery_reference_audit([path, path], [])
    expected_logs = [r.getMessage() for r in caplog.records]
    caplog.clear()
    original = utils.read_dicom_header
    reads = []
    def observed(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(utils, 'read_dicom_header', observed)
    @utils.organize_read_cache
    def run():
        projected = utils.read_discovery_header(path)
        assert projected == pydicom.dcmread(path, force=True, stop_before_pixels=True,
                                          specific_tags=utils.ORGANIZE_DISCOVERY_TAGS)
        assert organize._record_delivery_evidence([path, path]) == expected
        assert organize._delivery_reference_audit([path, path], []) == expected_audit
        assert organize._record_delivery_evidence([path]) == expected
        assert [r.getMessage() for r in caplog.records] == expected_logs
        if not preamble:
            with pytest.raises(pydicom.errors.InvalidDicomError):
                utils.read_record_header(path, force=False)
        assert len(reads) == 1
    run()


def test_parallel_record_aliases_share_a_single_read(tmp_path, monkeypatch):
    from synthetic_rt_fixtures import make_record
    path = make_record(tmp_path / 'record.dcm', generate_uid())
    alias = tmp_path / 'alias.dcm'
    alias.symlink_to(path)
    original = utils.read_dicom_header
    reads = []
    def observed(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(utils, 'read_dicom_header', observed)
    @utils.organize_read_cache
    def run():
        result = list(utils.parallel_map_files([path, alias] * 4, utils.read_discovery_header, 4))
        assert len(result) == 8
        assert [dataset.filename for dataset in result] == [str(p) for p in [path, alias] * 4]
        assert len(reads) == 1
    run()


def test_run_cache_is_released_on_exception():
    @utils.organize_read_cache
    def fail():
        assert utils._organize_reads.get() is not None
        raise RuntimeError('synthetic failure')
    with pytest.raises(RuntimeError, match='synthetic failure'):
        fail()
    assert utils._organize_reads.get() is None


@pytest.mark.parametrize('missing', [False, True])
def test_discovery_reuses_missing_uid_without_changing_copy_headers(tmp_path, monkeypatch, missing):
    from synthetic_rt_fixtures import make_record
    path = make_record(tmp_path / 'record.dcm', generate_uid())
    ds = pydicom.dcmread(path)
    if missing:
        del ds.SOPInstanceUID
    else:
        ds.SOPInstanceUID = None
    ds.save_as(path)
    old = baseline()
    old_manager = old['dicom_copy'].DicomCopyManager(DicomCopyConfig(), tmp_path / 'old')
    expected = old_manager._get_sop_uid(path)
    expected_safe_uid = old['organize']._sop_instance_uid(path)
    original = utils.read_dicom_header
    reads = []
    def observed(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(utils, 'read_dicom_header', observed)
    @utils.organize_read_cache
    def run():
        utils.read_discovery_header(path)
        manager = DicomCopyManager(DicomCopyConfig(), tmp_path / 'new')
        assert manager._get_sop_uid(path) == expected
        assert manager._header_cache == old_manager._header_cache
        assert organize._sop_instance_uid(path) == expected_safe_uid
        assert len(reads) == 1
    run()
