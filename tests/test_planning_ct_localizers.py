"""RF12: synthetic planning-series publication and repair only."""
from __future__ import annotations

import hashlib
import os
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
    assert refused[0]['reason'] == 'ct_localizer_referenced_by_rtstruct'
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


def _organized_course(root, dcm2niix, name='course_a', localizer=True):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline.course_contract import _ct_provenance
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.organize_ledger import write_organize_ledger, read_organize_ledger
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    import pandas as pd

    course = root / 'SYNTH' / name
    ct = course / 'DICOM/CT'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    rs = _rtstruct(course / 'RS.dcm', paths)
    nifti = segmentation._ensure_ct_nifti(_config(root, dcm2niix), ct, course / 'NIFTI')
    sidecar = nifti.with_name(nifti.name[:-7] + '.metadata.json')
    original_meta = json.loads(sidecar.read_text())
    for index in range(int(localizer)):
        scout, _ = _localizer(paths)
        # Put localizer first to exercise re-derivation of first-slice fields.
        scout.rename(ct / f'000_localizer{index}.dcm')
    metadata_path = write_minimal_course_contract(course, authoritative_rtstruct=rs,
                                                 planning_ct_dir=ct, planning_ct_nifti=nifti)
    case = json.loads(metadata_path.read_text())
    original_meta.update(segmentation._collect_series_metadata(ct))
    original_meta.update(_ct_provenance(ct))
    sidecar.write_text(json.dumps(original_meta, indent=2))
    case['course_contract']['planning_ct']['nifti_provenance'] = {
        **original_meta, 'sidecar_path': str(sidecar.relative_to(course))}
    case.update(organize._planning_ct_summary(ct))
    metadata_path.write_text(json.dumps(case, indent=2))
    pd.DataFrame([case]).to_excel(course / 'metadata/case_metadata.xlsx', index=False)
    load_course_contract(course)
    try:
        entries = read_organize_ledger(root)['courses']
    except Exception:
        entries = []
    entries.append({'patient': 'SYNTH', 'course': name, 'status': 'validated'})
    write_organize_ledger(root, entries)
    dependency = materialize_stage_dependency(root / '_CONFIG', 'organize', {'synthetic': True})
    write_stage_completion_sentinel(course, course / '.organized', stage='organize', status='ok',
                                    configuration_dependency=dependency)
    return course


def _files_and_mtimes(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_repair_atomic_idempotent_and_course_scoped(tmp_path, dcm2niix):
    from rtpipeline.repair_planning_ct_localizers import repair_output
    from rtpipeline.stage_completion import validate_stage_completion_sentinel
    root = tmp_path / 'Output'
    affected = _organized_course(root, dcm2niix)
    clean = _organized_course(root, dcm2niix, name='course_b', localizer=False)
    # A second affected course must also remain untouched when not selected.
    other = _organized_course(root, dcm2niix, name='course_c')
    for course in (affected, clean, other):
        (course / '.segmentation_done').write_text('downstream placeholder')
    (root / '_COURSES/manifest.json').write_text(json.dumps({'courses': [
        {'patient': 'SYNTH', 'course': name} for name in ('course_a', 'course_b', 'course_c')]}))
    shared_before = _files_and_mtimes(root / '_COURSES')
    clean_before, other_before = _files_and_mtimes(clean), _files_and_mtimes(other)
    nifti = load_course_contract(affected).planning_ct_nifti
    nifti_before = nifti.read_bytes()
    sentinel_before = (affected / '.organized').stat().st_mtime_ns
    before = _files_and_mtimes(root)
    dry = repair_output(root, courses=['SYNTH/course_a'], dry_run=True)
    assert dry['would_repair'] == 1 and dry['refused'] == 0
    assert _files_and_mtimes(root) == before
    result = repair_output(root, courses=['SYNTH/course_a'])
    assert result['repaired'] == 1 and result['refused'] == 0
    assert result['localizers_excluded'] == 1
    contract = load_course_contract(affected)
    assert contract.planning_ct['nifti_provenance']['instance_selection']['excluded_instance_count'] == 1
    assert _load(contract.planning_ct_dir).GetSize() == (20, 16, 12)
    assert nifti.read_bytes() == nifti_before
    assert (affected / '.organized').stat().st_mtime_ns > sentinel_before
    validate_stage_completion_sentinel(affected / '.organized', expected_stage='organize')
    assert _files_and_mtimes(clean) == clean_before
    assert _files_and_mtimes(other) == other_before
    assert _files_and_mtimes(root / '_COURSES') == shared_before
    before = _files_and_mtimes(root)
    result = repair_output(root, courses=['SYNTH/course_a', 'SYNTH/course_b'])
    assert result['unchanged'] == 2 and result['repaired'] == result['refused'] == 0
    assert _files_and_mtimes(root) == before


@pytest.mark.parametrize('damage,reason', [
    ('sidecar', 'ct_localizer_inconsistent_evidence'),
    ('sentinel', 'ct_localizer_inconsistent_evidence'),
    ('pixels', 'ct_localizer_nifti_volume_mismatch'),
    ('publish', 'ct_localizer_atomic_publish_failed'),
])
def test_repair_refusal_leaves_every_byte_and_mtime_unchanged(tmp_path, dcm2niix, monkeypatch, damage, reason):
    from rtpipeline import repair_planning_ct_localizers as repair
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    contract = load_course_contract(course)
    sidecar = course / contract.planning_ct['nifti_provenance']['sidecar_path']
    if damage == 'sidecar':
        data = json.loads(sidecar.read_text())
        data['sop_hash'] = '0' * 64
        sidecar.write_text(json.dumps(data))
    elif damage == 'sentinel':
        (course / '.organized').write_text('{}')
    elif damage == 'pixels':
        # A header-valid but wrong CT cannot be silently relabelled as the old NIfTI.
        path = sorted((course / 'DICOM/CT').glob('*.dcm'))[-1]
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array.copy()
        pixels[0, 0] += 1
        ds.PixelData = pixels.tobytes()
        ds.save_as(path, enforce_file_format=True)
    elif damage == 'publish':
        def fail(*args):
            raise PlanningCTLocalizerError('ct_localizer_atomic_publish_failed')
        monkeypatch.setattr(repair, '_exchange', fail)
    before = _files_and_mtimes(root)
    result = repair.repair_output(root)
    assert result['refused'] == 1
    assert result['reason_codes'] == {reason: 1}
    assert _files_and_mtimes(root) == before


def test_repair_validates_staged_absolute_artifact_paths(tmp_path, dcm2niix):
    from rtpipeline.repair_planning_ct_localizers import repair_output
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    path = course / 'metadata/case_metadata.json'
    case = json.loads(path.read_text())
    case['course_contract']['authoritative_rtstruct']['path'] = str(course / 'RS.dcm')
    path.write_text(json.dumps(case))
    completion = json.loads((course / '.organized').read_text())
    dependency = tmp_path / 'configuration.json'
    dependency.write_text(json.dumps(completion['configuration_dependency']))
    write_stage_completion_sentinel(course, course / '.organized', stage='organize', status='ok',
                                    configuration_dependency=dependency)
    result = repair_output(root)
    assert result['repaired'] == 1 and result['refused'] == 0
    assert load_course_contract(course).authoritative_rtstruct_path == course / 'RS.dcm'
    assert json.loads(path.read_text())['course_contract']['authoritative_rtstruct']['path'] == str(course / 'RS.dcm')


def test_repair_cli_prints_only_identifier_free_json(tmp_path, dcm2niix):
    import subprocess
    import sys
    root = tmp_path / 'Output'
    _organized_course(root, dcm2niix)
    command = [sys.executable, '-m', 'rtpipeline.cli', 'repair-planning-ct-localizers',
               '--output-dir', str(root), '--course', 'SYNTH/course_a', '--dry-run']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)['would_repair'] == 1
    assert result.stderr == ''
    assert 'SYNTH' not in result.stdout and str(root) not in result.stdout
    result = subprocess.run(command[:-2] + ['unvalidated/course'], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['reason_codes'] == {'ct_localizer_course_not_validated': 1}
    assert result.stderr == ''


def test_contract_rejects_tampered_exclusion_evidence(tmp_path, dcm2niix):
    from rtpipeline.repair_planning_ct_localizers import repair_output
    from rtpipeline.course_contract import CourseContractError
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    assert repair_output(root)['repaired'] == 1
    contract = load_course_contract(course)
    sidecar = course / contract.planning_ct['nifti_provenance']['sidecar_path']
    case_path = course / 'metadata/case_metadata.json'
    case = json.loads(case_path.read_text())
    metadata = json.loads(sidecar.read_text())
    # Even a matching contract and sidecar cannot claim an excluded image is
    # still in DICOM/CT, or change the authoritative reference evidence.
    evidence = metadata['instance_selection']
    evidence['excluded_instances'][0]['sop_instance_uid'] = evidence['kept_sop_instance_uids'][0]
    case['course_contract']['planning_ct']['nifti_provenance']['instance_selection'] = evidence
    sidecar.write_text(json.dumps(metadata))
    case_path.write_text(json.dumps(case))
    with pytest.raises(CourseContractError, match='ct_localizer_stale_selection_evidence'):
        load_course_contract(course)


def test_repair_two_localizers_and_continue_after_course_refusal(tmp_path, dcm2niix):
    from rtpipeline.repair_planning_ct_localizers import repair_output
    root = tmp_path / 'Output'
    broken = _organized_course(root, dcm2niix)
    affected = _organized_course(root, dcm2niix, name='course_b', localizer=2)
    (broken / '.organized').write_text('{}')
    broken_before = _files_and_mtimes(broken)
    result = repair_output(root)
    assert result['repaired'] == 1 and result['refused'] == 1
    assert result['localizers_excluded'] == 2
    assert _files_and_mtimes(broken) == broken_before
    contract = load_course_contract(affected)
    assert contract.planning_ct['nifti_provenance']['instance_selection']['excluded_instance_count'] == 2
    assert _load(contract.planning_ct_dir).GetSize() == (20, 16, 12)


def test_repair_retains_and_rebinds_original_mask_provenance(tmp_path, dcm2niix):
    from rtpipeline.repair_planning_ct_localizers import repair_output
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    contract = load_course_contract(course)
    original = course / 'Segmentation_Original/rtstruct'
    original.mkdir(parents=True)
    mask = original / 'mask.nii.gz'
    sitk.WriteImage(sitk.Image([20, 16, 12], sitk.sitkUInt8), str(mask))
    before = mask.read_bytes()
    evidence = organize._original_segmentation_provenance(contract.authoritative_rtstruct_path, contract.planning_ct_nifti)
    (original / 'provenance.json').write_text(json.dumps(evidence))
    completion = json.loads((course / '.organized').read_text())
    dependency = tmp_path / 'configuration.json'
    dependency.write_text(json.dumps(completion['configuration_dependency']))
    write_stage_completion_sentinel(course, course / '.organized', stage='organize', status='ok',
                                    configuration_dependency=dependency)
    assert repair_output(root)['repaired'] == 1
    after = json.loads((original / 'provenance.json').read_text())
    assert after['source_ct_sop_hash'] != evidence['source_ct_sop_hash']
    assert {k: v for k, v in after.items() if k != 'source_ct_sop_hash'} == {k: v for k, v in evidence.items() if k != 'source_ct_sop_hash'}
    assert mask.read_bytes() == before


def test_cli_suppresses_library_identifier_warnings(monkeypatch, capsys, tmp_path):
    import warnings
    import logging
    from rtpipeline import repair_planning_ct_localizers as repair
    def noisy(*args, **kwargs):
        warnings.warn('synthetic identifying text')
        logging.error('synthetic identifying text')
        return {'refused': 1, 'reason_codes': {'ct_localizer_inconsistent_evidence': 1}}
    monkeypatch.setattr(repair, 'repair_output', noisy)
    assert repair.main(['--output-dir', str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.err == ''
    assert 'synthetic identifying text' not in output.out
    assert json.loads(output.out)['refused'] == 1


def _exchange_unavailable(*args):
    raise PlanningCTLocalizerError('ct_localizer_atomic_publish_unavailable')


def test_repair_two_step_publish_only_when_allowed(tmp_path, dcm2niix, monkeypatch):
    """NFS clients have no RENAME_EXCHANGE: refuse by default, publish by two renames on opt-in."""
    from rtpipeline import repair_planning_ct_localizers as repair
    from rtpipeline.stage_completion import validate_stage_completion_sentinel
    root = tmp_path / 'Output'
    affected = _organized_course(root, dcm2niix)
    clean = _organized_course(root, dcm2niix, name='course_b', localizer=False)
    monkeypatch.setattr(repair, '_exchange', _exchange_unavailable)
    shared_before = _files_and_mtimes(root / '_COURSES')
    clean_before = _files_and_mtimes(clean)
    nifti = load_course_contract(affected).planning_ct_nifti
    nifti_before = nifti.read_bytes()
    before = _files_and_mtimes(root)
    refused = repair.repair_output(root)
    assert refused['refused'] == 1
    assert refused['reason_codes'] == {'ct_localizer_atomic_publish_unavailable': 1}
    assert _files_and_mtimes(root) == before
    result = repair.repair_output(root, two_step=True)
    assert result['repaired'] == 1 and result['unchanged'] == 1 and result['refused'] == 0
    contract = load_course_contract(affected)
    assert contract.planning_ct['nifti_provenance']['instance_selection']['excluded_instance_count'] == 1
    assert _load(contract.planning_ct_dir).GetSize() == (20, 16, 12)
    assert nifti.read_bytes() == nifti_before
    validate_stage_completion_sentinel(affected / '.organized', expected_stage='organize')
    assert _files_and_mtimes(clean) == clean_before
    assert _files_and_mtimes(root / '_COURSES') == shared_before
    assert not [p for p in (root / '_COURSES').iterdir() if p.name.startswith('.localizer-repair-')]
    again = _files_and_mtimes(root)
    assert repair.repair_output(root, two_step=True)['unchanged'] == 2
    assert _files_and_mtimes(root) == again


def test_repair_two_step_publish_rolls_back_a_failed_second_rename(tmp_path, dcm2niix, monkeypatch):
    from rtpipeline import repair_planning_ct_localizers as repair
    root = tmp_path / 'Output'
    _organized_course(root, dcm2niix)
    monkeypatch.setattr(repair, '_exchange', _exchange_unavailable)
    real_rename = os.rename
    calls = []

    def flaky_rename(src, dst):
        calls.append(1)
        if len(calls) == 2:
            raise OSError('injected failure of the second rename')
        return real_rename(src, dst)

    monkeypatch.setattr(repair.os, 'rename', flaky_rename)
    before = _files_and_mtimes(root)
    result = repair.repair_output(root, two_step=True)
    assert result['refused'] == 1
    assert _files_and_mtimes(root) == before
    assert not [p for p in (root / '_COURSES').iterdir() if p.name.startswith('.localizer-repair-')]
