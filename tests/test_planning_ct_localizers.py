"""RF12: synthetic planning-series publication and repair only."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pydicom
import pytest
import SimpleITK as sitk

from rtpipeline import organize, segmentation
from rtpipeline.course_contract import load_course_contract
from rtpipeline.planning_ct_localizers import PlanningCTLocalizerError, select_localizers
from test_ct_series_hygiene import _localizer, _rtstruct
from test_nifti_nonuniform_fallback import _config, _tree_digest, dcm2niix
from synthetic_ct_series import write_ct_series, uniform_z_positions


def _load(ct):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct)))
    return reader.Execute()


def test_organize_localizer_publication_and_reference_refusal(tmp_path, monkeypatch, dcm2niix):
    from organize_io_fixture import synthetic
    root = tmp_path / 'input'
    synthetic(root, ct_slices=12, all_slices=True)
    for number, folder in enumerate(sorted(root.glob('*/*'))):
        if not folder.is_dir():
            continue
        paths = [folder / f'ct{i}.dcm' for i in range(12)]
        scout, uid = _localizer(paths)
        if number == 0:
            # The failure must not depend on a localizer sorting last.
            ds = pydicom.dcmread(scout)
            ds.InstanceNumber = 0
            ds.ImageType = ['ORIGINAL', 'PRIMARY', 'localizer']
            ds.save_as(scout, enforce_file_format=True)
        if number == 1:
            rs = pydicom.dcmread(folder / 'struct.dcm')
            ref = pydicom.Dataset()
            ref.ReferencedSOPClassUID = pydicom.dcmread(scout).SOPClassUID
            ref.ReferencedSOPInstanceUID = uid
            rs.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(ref)
            rs.save_as(folder / 'struct.dcm', enforce_file_format=True)
    before = _tree_digest(root)
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '1')
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    config = _config(tmp_path, dcm2niix)
    config.dicom_root = root
    config.max_workers_override = 1
    config.dicom_copy_use_hardlinks = True
    courses = organize.organize_and_merge(config, metadata_snapshot={})
    assert len(courses) == 3
    ledger = json.loads((config.output_root / '_COURSES/organize_ledger.json').read_text())
    refused = ledger['technical_quarantines']
    assert len(refused) == 1
    assert refused[0]['planning_ct_conversion']['reason_code'] == 'ct_localizer_referenced_by_rtstruct'
    for course in courses:
        contract = load_course_contract(course.dirs.root)
        selection = contract.planning_ct['nifti_provenance']['instance_selection']
        assert selection['excluded_instance_count'] == 1
        assert selection['reason'] == 'localizer_image_type'
        assert _load(course.dirs.dicom_ct).GetSize() == (8, 8, 12)
        assert len(list(course.dirs.dicom_ct.iterdir())) == 12
    assert _tree_digest(root) == before


@pytest.mark.parametrize('case,reason', [
    ('referenced', 'ct_localizer_referenced_by_rtstruct'),
    ('no_rtstruct', 'ct_localizer_rtstruct_references_missing'),
    ('too_few', 'ct_localizer_insufficient_volume'),
    ('mixed', 'ct_localizer_inconsistent_geometry'),
    ('duplicate', 'ct_localizer_inconsistent_volume_positions'),
])
def test_ambiguous_localizer_selection_refused(tmp_path, case, reason):
    paths = write_ct_series(tmp_path / 'ct', uniform_z_positions(), signed=False)
    _, uid = _localizer(paths)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths, extra=[uid] if case == 'referenced' else [])
    if case == 'too_few':
        for path in paths[2:]:
            path.unlink()
        rs = _rtstruct(rs, paths[:2])
    if case in ('mixed', 'duplicate'):
        ds = pydicom.dcmread(paths[-1])
        if case == 'mixed':
            ds.PixelSpacing = [2., 2.]
        else:
            ds.ImagePositionPatient = pydicom.dcmread(paths[0]).ImagePositionPatient
        ds.save_as(paths[-1], enforce_file_format=True)
    before = _tree_digest(tmp_path)
    with pytest.raises(PlanningCTLocalizerError, match=reason):
        select_localizers(tmp_path / 'ct', None if case == 'no_rtstruct' else rs)
    assert _tree_digest(tmp_path) == before


def test_clean_organize_outputs_byte_identical_to_879f225(tmp_path, monkeypatch, dcm2niix):
    """Compare every output byte at one path with clocks fixed, no normalization."""
    import datetime
    import subprocess
    import sys
    import types
    import zipfile
    import xlsxwriter.core
    from organize_io_fixture import synthetic

    class FixedTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, tzinfo=tz)
        @classmethod
        def utcnow(cls):
            return cls(2024, 1, 1)
    monkeypatch.setattr(datetime, 'datetime', FixedTime)
    monkeypatch.setattr(xlsxwriter.core, 'datetime', FixedTime)
    fixed_zip_time = zipfile.time.gmtime(1704067200)
    monkeypatch.setattr(zipfile.time, 'localtime', lambda *args: fixed_zip_time)
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '1')
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    source = subprocess.check_output(['git', 'show', '879f225:rtpipeline/organize.py'], text=True)
    baseline = types.ModuleType('rtpipeline._rf12_baseline_organize')
    baseline.__package__ = 'rtpipeline'
    baseline.__file__ = organize.__file__
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    exec(compile(source, baseline.__file__, 'exec'), baseline.__dict__)
    root = tmp_path / 'input'
    synthetic(root, ct_slices=12, all_slices=True)
    cfg = _config(tmp_path, dcm2niix)
    cfg.dicom_root = root
    cfg.max_workers_override = 1
    cfg.dicom_copy_use_hardlinks = False
    assert len(baseline.organize_and_merge(cfg, metadata_snapshot={})) == 4
    expected = _tree_digest(cfg.output_root)
    shutil.rmtree(cfg.output_root)
    assert len(organize.organize_and_merge(cfg, metadata_snapshot={})) == 4
    actual = _tree_digest(cfg.output_root)
    assert actual.keys() == expected.keys()
    assert [name for name in expected if expected[name] != actual[name]] == []
