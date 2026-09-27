"""RF13 publication and repair: synthetic DICOM, real dcm2niix only."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.uid import generate_uid

from rtpipeline import organize, segmentation
from rtpipeline.course_contract import CourseContractError, _ct_provenance, load_course_contract
from rtpipeline.nifti_provenance import annotate
from rtpipeline.planning_ct_localizers import PlanningCTLocalizerError, select_localizers, verify_volume
from rtpipeline.repair_planning_ct_localizers import repair_output
from rtpipeline.stage_completion import validate_stage_completion_sentinel, write_stage_completion_sentinel
from test_ct_series_hygiene import _duplicate, _localizer, _rtstruct
from test_nifti_nonuniform_fallback import _config, _tree_digest, dcm2niix
from test_planning_ct_localizers import _files_and_mtimes, _load, _organized_course
from synthetic_ct_series import write_ct_series, uniform_z_positions


def _refresh(course):
    """Bind intentional synthetic input changes before exercising repair."""
    case_path = course / 'metadata/case_metadata.json'
    case = json.loads(case_path.read_text())
    case['course_contract']['authoritative_rtstruct']['sop_instance_uid'] = str(
        pydicom.dcmread(course / 'RS.dcm', stop_before_pixels=True).SOPInstanceUID)
    provenance = case['course_contract']['planning_ct']['nifti_provenance']
    sidecar = course / provenance['sidecar_path']
    metadata = json.loads(sidecar.read_text())
    ct = course / 'DICOM/CT'
    nifti = next((course / 'NIFTI').glob('*.nii*'))
    metadata.update(segmentation._collect_series_metadata(ct))
    metadata.update(_ct_provenance(ct))
    annotate(metadata, nifti, ct, regenerated=False, existing_sidecar=metadata.copy())
    sidecar.write_text(json.dumps(metadata))
    provenance.update(metadata)
    case_path.write_text(json.dumps(case))
    completion = json.loads((course / '.organized').read_text())
    dependency = course.parent.parent / '_COURSES/synthetic-config.json'
    dependency.write_text(json.dumps(completion['configuration_dependency']))
    write_stage_completion_sentinel(course, course / '.organized', stage='organize', status='ok',
                                    configuration_dependency=dependency)
    return load_course_contract(course)


def _copies(course, copies=2, *, differing=False, rescaled=False):
    paths = sorted((course / 'DICOM/CT').glob('CT_*.dcm'))
    assert len(paths) == 12
    added = []
    for copy in range(1, copies):
        for index, path in enumerate(paths):
            ds = pydicom.dcmread(path)
            ds.SOPInstanceUID = generate_uid()
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.InstanceNumber = 100 * copy + index
            if differing:
                pixels = ds.pixel_array.copy()
                pixels[0, 0] += 1
                ds.PixelData = pixels.tobytes()
            if rescaled:
                ds.PixelData = (ds.pixel_array.astype(np.uint16) * 2).tobytes()
                ds.RescaleSlope = 0.5
            target = path.with_name(f'copy{copy}_{index}.dcm')
            ds.save_as(target, enforce_file_format=True)
            added.append(target)
    return paths, added


@pytest.mark.parametrize('localizers,copies,rescaled', [(0, 2, False), (1, 2, True), (2, 3, False), (1, 4, False)])
def test_repair_identical_copies_preserves_nifti(tmp_path, dcm2niix, localizers, copies, rescaled):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix, localizer=localizers)
    _copies(course, copies, rescaled=rescaled)
    original = _refresh(course)
    before = _files_and_mtimes(root)
    nifti_bytes = original.planning_ct_nifti.read_bytes()
    source_bytes = _tree_digest(original.planning_ct_dir)
    dry = repair_output(root, dry_run=True)
    assert dry['would_repair'] == 1 and dry['refused'] == 0
    assert dry['duplicates_would_exclude'] == 12 * (copies - 1)
    assert _files_and_mtimes(root) == before
    result = repair_output(root)
    assert result['repaired'] == 1 and result['refused'] == 0
    assert result['duplicates_excluded'] == 12 * (copies - 1)
    assert result['localizers_excluded'] == localizers
    assert result['niftis_rederived'] == 0
    contract = load_course_contract(course)
    selection = contract.planning_ct['nifti_provenance']['instance_selection']
    assert selection['kept_instance_count'] == 12
    for excluded in selection['excluded_instances']:
        if excluded['reason_code'] == 'identical_rescaled_duplicate_position':
            assert excluded['retained_sop_instance_uid'] in selection['kept_sop_instance_uids']
    assert contract.planning_ct_nifti.read_bytes() == nifti_bytes
    assert _load(contract.planning_ct_dir).GetSize() == (20, 16, 12)
    assert all(source_bytes[k] == v for k, v in _tree_digest(contract.planning_ct_dir).items())
    validate_stage_completion_sentinel(course / '.organized', expected_stage='organize')
    before = _files_and_mtimes(root)
    assert repair_output(root)['unchanged'] == 1
    assert _files_and_mtimes(root) == before


@pytest.mark.parametrize('problem,reason', [
    ('pixels', 'ct_hygiene_duplicate_position_reference_coverage'),
    ('references', 'ct_hygiene_referenced_duplicate_would_be_excluded'),
    ('interleaved', 'ct_localizer_inconsistent_volume_positions'),
    ('near_position', 'ct_localizer_inconsistent_volume_positions'),
    ('acquisition', 'ct_localizer_inconsistent_geometry'),
])
def test_unsafe_duplicate_repair_refuses_without_changes(tmp_path, dcm2niix, problem, reason):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    paths, added = _copies(course, differing=problem == 'pixels')
    if problem == 'pixels':
        _rtstruct(course / 'RS.dcm', paths[:-1])
    if problem == 'references':
        _rtstruct(course / 'RS.dcm', paths + added)
    if problem == 'interleaved':
        for path in paths + added:
            ds = pydicom.dcmread(path)
            ds.ImagePositionPatient[2] /= 5.0  # 1 mm grid, then 0.412/0.588 mm interleaving.
            ds.save_as(path, enforce_file_format=True)
    if problem in ('interleaved', 'near_position', 'acquisition'):
        for path in added:
            ds = pydicom.dcmread(path)
            if problem == 'acquisition':
                ds.AcquisitionNumber = 2
            else:
                ds.ImagePositionPatient[2] += 0.412 if problem == 'interleaved' else 0.001
            ds.save_as(path, enforce_file_format=True)
    _refresh(course)
    before = _files_and_mtimes(root)
    for dry in (True, False):
        result = repair_output(root, dry_run=dry)
        assert result['refused'] == 1
        assert result['reason_codes'] == {reason: 1}
        assert _files_and_mtimes(root) == before


@pytest.mark.parametrize('kind', ['4d', 'wrong_pixels'])
def test_mismatched_nifti_rederived_and_old_masks_stay_stale(tmp_path, dcm2niix, kind):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    _copies(course)
    original = _refresh(course)
    nifti = original.planning_ct_nifti
    image = nib.load(nifti)
    values = np.asanyarray(image.dataobj).copy()
    values = np.stack([values, values], axis=-1) if kind == '4d' else values + 1
    nib.save(nib.Nifti1Image(values, image.affine), nifti)
    contract = _refresh(course)
    masks = course / 'Segmentation_Original/rtstruct'
    masks.mkdir(parents=True)
    sitk.WriteImage(sitk.Image([20, 16, 12], sitk.sitkUInt8), str(masks / 'mask.nii.gz'))
    (masks / 'provenance.json').write_text(json.dumps(organize._original_segmentation_provenance(
        contract.authoritative_rtstruct_path, contract.planning_ct_nifti)))
    (course / '.segmentation_done').write_text('stale synthetic segmentation marker')
    _refresh(course)
    mask_before = _files_and_mtimes(masks)
    marker_before = (course / '.segmentation_done').read_bytes()
    before = _files_and_mtimes(root)
    dry = repair_output(root, dry_run=True)
    assert dry['niftis_would_rederive'] == 1 and dry['would_repair'] == 1
    assert _files_and_mtimes(root) == before
    result = repair_output(root)
    assert result['niftis_rederived'] == result['repaired'] == 1 and result['refused'] == 0
    assert result['segmentation_outputs_stale_courses'] == 1
    contract = load_course_contract(course)
    verify_volume(contract.planning_ct_dir, contract.planning_ct_nifti)
    assert sitk.ReadImage(str(nifti)).GetDimension() == 3
    assert _files_and_mtimes(masks) == mask_before
    assert (course / '.segmentation_done').read_bytes() == marker_before
    evidence = contract.planning_ct['nifti_provenance']['instance_selection']['nifti_repair']
    assert evidence['status'] == 'rederived'
    assert evidence['segmentation_outputs'] == 'stale_left_in_place'
    assert evidence['previous_nifti_sha256'] != contract.planning_ct['nifti_provenance']['nifti_sha256']
    validate_stage_completion_sentinel(course / '.organized', expected_stage='organize')
    before = _files_and_mtimes(root)
    assert repair_output(root)['unchanged'] == 1
    assert _files_and_mtimes(root) == before


@pytest.mark.parametrize('localizer', [False, True])
def test_organize_publishes_one_volume_from_duplicate_positions(tmp_path, monkeypatch, dcm2niix, localizer):
    from organize_io_fixture import synthetic
    root = tmp_path / 'input'
    synthetic(root, ct_slices=12, all_slices=True)
    for folder in root.glob('*/*'):
        if not folder.is_dir():
            continue
        paths = [folder / f'ct{i}.dcm' for i in range(12)]
        _duplicate(paths)
        if localizer:
            _localizer(paths)
    before = _tree_digest(root)
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '1')
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    config = _config(tmp_path, dcm2niix)
    config.dicom_root = root
    config.max_workers_override = 1
    config.dicom_copy_use_hardlinks = True
    courses = organize.organize_and_merge(config, metadata_snapshot={})
    assert len(courses) == 4
    for course in courses:
        contract = load_course_contract(course.dirs.root)
        evidence = contract.planning_ct['nifti_provenance']['instance_selection']
        assert evidence['excluded_instance_count'] == 1 + int(localizer)
        assert evidence['kept_instance_count'] == 12
        verify_volume(contract.planning_ct_dir, contract.planning_ct_nifti)
    assert _tree_digest(root) == before


@pytest.mark.parametrize('damage', ['retained_uid', 'wrong_retained_slice', 'pixel_hash', 'reason', 'repair_status'])
def test_contract_validates_duplicate_evidence(tmp_path, dcm2niix, damage):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix, localizer=False)
    _copies(course)
    _refresh(course)
    assert repair_output(root)['repaired'] == 1
    contract = load_course_contract(course)
    sidecar = course / contract.planning_ct['nifti_provenance']['sidecar_path']
    metadata = json.loads(sidecar.read_text())
    selection = metadata['instance_selection']
    if damage == 'retained_uid':
        selection['excluded_instances'][0]['retained_sop_instance_uid'] = generate_uid()
    elif damage == 'wrong_retained_slice':
        item = selection['excluded_instances'][0]
        item['retained_sop_instance_uid'] = next(
            uid for uid in selection['kept_sop_instance_uids'] if uid != item['retained_sop_instance_uid'])
    elif damage == 'pixel_hash':
        selection['excluded_instances'][0]['rescaled_pixels_sha256'] = '0' * 64
    elif damage == 'reason':
        selection['excluded_instances'][0]['reason_code'] = 'unreferenced_geometry_partition'
    else:
        selection['nifti_repair']['segmentation_outputs'] = 'stale_left_in_place'
    sidecar.write_text(json.dumps(metadata))
    case_path = course / 'metadata/case_metadata.json'
    case = json.loads(case_path.read_text())
    case['course_contract']['planning_ct']['nifti_provenance']['instance_selection'] = selection
    case_path.write_text(json.dumps(case))
    with pytest.raises(CourseContractError, match='ct_localizer_stale_selection_evidence'):
        load_course_contract(course)


def test_selection_prefers_referenced_copy_then_instance_number_and_uid(tmp_path):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    duplicate, uid = _duplicate(paths)
    # Only the high-numbered copy is referenced at this position.
    rs = _rtstruct(tmp_path / 'RS.dcm', [p for p in paths if p != paths[4]] + [duplicate])
    excluded, evidence = select_localizers(ct, rs)
    assert excluded == [paths[4]]
    assert evidence['excluded_instances'][0]['retained_sop_instance_uid'] == uid
    # Neither equivalent image is referenced: the lowest number wins.
    rs = _rtstruct(rs, paths[:2])
    excluded, evidence = select_localizers(ct, rs)
    assert excluded == [duplicate]
    ds = pydicom.dcmread(duplicate)
    ds.InstanceNumber = pydicom.dcmread(paths[4]).InstanceNumber
    ds.save_as(duplicate, enforce_file_format=True)
    _, evidence = select_localizers(ct, rs)
    expected = min(uid, str(pydicom.dcmread(paths[4]).SOPInstanceUID))
    assert evidence['excluded_instances'][0]['retained_sop_instance_uid'] == expected


def _make_4d(course):
    contract = load_course_contract(course)
    image = nib.load(contract.planning_ct_nifti)
    values = np.asanyarray(image.dataobj).copy()
    nib.save(nib.Nifti1Image(np.stack([values, values], axis=-1), image.affine), contract.planning_ct_nifti)
    return _refresh(course)


@pytest.mark.parametrize('failure,reason', [
    ('no_output', 'ct_localizer_volume_conversion_failed'),
    ('wrong_output', 'ct_localizer_nifti_volume_mismatch'),
    ('publish', 'ct_localizer_atomic_publish_failed'),
])
def test_failed_rederivation_or_publication_leaves_course_unchanged(tmp_path, monkeypatch, dcm2niix, failure, reason):
    from rtpipeline import repair_planning_ct_localizers as repair
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    _copies(course)
    _refresh(course)
    contract = _make_4d(course)
    if failure == 'no_output':
        monkeypatch.setattr(segmentation, '_ensure_ct_nifti', lambda *a, **kw: None)
    elif failure == 'wrong_output':
        monkeypatch.setattr(segmentation, '_ensure_ct_nifti', lambda *a, **kw: contract.planning_ct_nifti)
    else:
        def fail(*args):
            raise PlanningCTLocalizerError(reason)
        monkeypatch.setattr(repair, '_exchange', fail)
    before = _files_and_mtimes(root)
    result = repair_output(root)
    assert result['reason_codes'] == {reason: 1}
    assert result['repaired'] == 0
    assert _files_and_mtimes(root) == before


def test_cli_rederivation_prints_only_identifier_free_json(tmp_path, dcm2niix):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    _copies(course)
    _refresh(course)
    _make_4d(course)
    command = [sys.executable, '-m', 'rtpipeline.cli', 'repair-planning-ct-localizers',
               '--output-dir', str(root)]
    before = _files_and_mtimes(root)
    for args, key in [(['--dry-run'], 'niftis_would_rederive'), ([], 'niftis_rederived')]:
        completed = subprocess.run(command + args, capture_output=True, text=True)
        assert completed.returncode == 0
        assert completed.stderr == ''
        assert json.loads(completed.stdout)[key] == 1
        assert str(root) not in completed.stdout and 'SYNTH' not in completed.stdout
        if args:
            assert _files_and_mtimes(root) == before


def test_exact_position_comparison_does_not_round_small_offsets_to_zero(tmp_path):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _localizer(paths)
    ds = pydicom.dcmread(paths[0])
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    # Squaring this nonzero distance in a Euclidean norm underflows to zero.
    # Exact-position publication must retain it and refuse the irregular volume.
    ds.ImagePositionPatient[2] = 1e-200
    ds.save_as(ct / 'near.dcm', enforce_file_format=True)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths)
    with pytest.raises(PlanningCTLocalizerError, match='ct_localizer_inconsistent_volume_positions'):
        select_localizers(ct, rs)


@pytest.mark.parametrize('copies,localizers,referenced_copy', [(2, 0, False), (3, 1, True), (4, 2, False)])
def test_repair_differing_duplicates_from_complete_references(tmp_path, dcm2niix, copies, localizers, referenced_copy):
    from rtpipeline.ct_series_hygiene import _pixels
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix, localizer=localizers)
    paths, added = _copies(course, copies, differing=True)
    refs = added[:12] if referenced_copy else paths
    _rtstruct(course / 'RS.dcm', refs)
    _refresh(course)
    expected_pixels = {str(pydicom.dcmread(p).SOPInstanceUID): hashlib.sha256(_pixels(p)).hexdigest()
                       for p in paths + added}
    before = _files_and_mtimes(root)
    dry = repair_output(root, dry_run=True)
    assert dry['would_repair'] == 1 and dry['refused'] == 0
    assert _files_and_mtimes(root) == before
    result = repair_output(root)
    assert result['repaired'] == 1 and result['refused'] == 0
    assert result['duplicates_excluded'] == 12 * (copies - 1)
    assert result['niftis_rederived'] == int(referenced_copy)
    contract = load_course_contract(course)
    evidence = contract.planning_ct['nifti_provenance']['instance_selection']
    assert set(evidence['kept_sop_instance_uids']) == {str(pydicom.dcmread(p).SOPInstanceUID) for p in refs}
    different = [e for e in evidence['excluded_instances'] if e['reason_code'] == 'unreferenced_duplicate_position']
    assert different
    for item in different:
        assert item['excluded_rescaled_pixels_sha256'] == expected_pixels[item['sop_instance_uid']]
        assert item['retained_rescaled_pixels_sha256'] == expected_pixels[item['retained_sop_instance_uid']]
        assert item['excluded_rescaled_pixels_sha256'] != item['retained_rescaled_pixels_sha256']
    verify_volume(contract.planning_ct_dir, contract.planning_ct_nifti)
    assert _load(contract.planning_ct_dir).GetSize() == (20, 16, 12)
    validate_stage_completion_sentinel(course / '.organized', expected_stage='organize')
    before = _files_and_mtimes(root)
    assert repair_output(root)['unchanged'] == 1
    assert _files_and_mtimes(root) == before


@pytest.mark.parametrize('mode,reason', [
    ('two', 'ct_hygiene_referenced_duplicate_would_be_excluded'),
    ('missing', 'ct_hygiene_duplicate_position_reference_coverage'),
    ('unknown', 'ct_localizer_rtstruct_references_unresolvable'),
    ('none', 'ct_localizer_rtstruct_references_missing'),
    ('irregular', 'ct_localizer_inconsistent_volume_positions'),
])
def test_differing_duplicates_refuse_incomplete_or_ambiguous_authority(tmp_path, dcm2niix, mode, reason):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix, localizer=False)
    paths, added = _copies(course, differing=True)
    refs = paths + added[:1] if mode == 'two' else paths[:-1] if mode == 'missing' else paths
    _rtstruct(course / 'RS.dcm', refs, extra=[generate_uid()] if mode == 'unknown' else [])
    if mode == 'none':
        from course_contract_test_utils import write_synthetic_rtstruct
        write_synthetic_rtstruct(course / 'RS.dcm', referenced_series_uid=str(pydicom.dcmread(paths[0]).SeriesInstanceUID))
    if mode == 'irregular':
        for path in (paths[-1], added[-1]):
            ds = pydicom.dcmread(path)
            ds.ImagePositionPatient[2] += 1
            ds.save_as(path, enforce_file_format=True)
    _refresh(course)
    before = _files_and_mtimes(root)
    for dry in (True, False):
        result = repair_output(root, dry_run=dry)
        assert result['reason_codes'] == {reason: 1}
        assert result['refused'] == 1
        assert _files_and_mtimes(root) == before


def _orientation_partition(paths):
    added = []
    for index, path in enumerate(paths):
        ds = pydicom.dcmread(path)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.InstanceNumber += 100
        ds.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
        ds.ImagePositionPatient[2] += 0.412
        target = path.with_name(f'partition_{index}.dcm')
        ds.save_as(target, enforce_file_format=True)
        added.append(target)
    return added


@pytest.mark.parametrize('spanning', [False, True])
@pytest.mark.parametrize('duplicates', [False, True])
def test_orientation_partition_selection_and_reference_refusal(tmp_path, dcm2niix, spanning, duplicates):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix, localizer=False)
    paths = sorted((course / 'DICOM/CT').glob('CT_*.dcm'))
    added = _orientation_partition(paths)
    if duplicates:
        _copies(course, differing=True)
    _rtstruct(course / 'RS.dcm', paths + added[:1] if spanning else paths)
    _refresh(course)
    before = _files_and_mtimes(root)
    dry = repair_output(root, dry_run=True)
    assert _files_and_mtimes(root) == before
    result = repair_output(root)
    if spanning:
        assert dry['refused'] == result['refused'] == 1
        assert result['reason_codes'] == {'ct_hygiene_references_span_geometry_partitions': 1}
        assert _files_and_mtimes(root) == before
    else:
        assert dry['would_repair'] == result['repaired'] == 1
        assert result['geometry_partition_instances_excluded'] == 12
        contract = load_course_contract(course)
        verify_volume(contract.planning_ct_dir, contract.planning_ct_nifti)
        assert _load(contract.planning_ct_dir).GetSize() == (20, 16, 12)
        before = _files_and_mtimes(root)
        assert repair_output(root)['unchanged'] == 1
        assert _files_and_mtimes(root) == before


@pytest.mark.parametrize('damage', ['retained_hash', 'excluded_hash', 'equal_hash', 'position', 'coverage', 'geometry', 'excluded_referenced'])
def test_contract_reverifies_reference_authorized_exclusions(tmp_path, dcm2niix, damage):
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix, localizer=False)
    paths, _ = _copies(course, differing=True)
    _orientation_partition(paths)
    _refresh(course)
    assert repair_output(root)['repaired'] == 1
    contract = load_course_contract(course)
    sidecar = course / contract.planning_ct['nifti_provenance']['sidecar_path']
    metadata = json.loads(sidecar.read_text())
    selection = metadata['instance_selection']
    item = next(e for e in selection['excluded_instances'] if e['reason_code'] == 'unreferenced_duplicate_position')
    if damage == 'retained_hash':
        item['retained_rescaled_pixels_sha256'] = '0' * 64
    elif damage == 'excluded_hash':
        item['excluded_rescaled_pixels_sha256'] = 'invalid'
    elif damage == 'equal_hash':
        item['excluded_rescaled_pixels_sha256'] = item['retained_rescaled_pixels_sha256']
    elif damage == 'position':
        item['image_position_patient'][2] += 5
    elif damage == 'coverage':
        _rtstruct(course / 'RS.dcm', paths[:-1])
        selection['authoritative_rtstruct_sha256'] = hashlib.sha256((course / 'RS.dcm').read_bytes()).hexdigest()
        selection['rtstruct_referenced_instance_count'] -= 1
    elif damage == 'excluded_referenced':
        item['sop_instance_uid'] = selection['kept_sop_instance_uids'][0]
    else:
        geometry_item = next(e for e in selection['excluded_instances'] if e['reason_code'] == 'unreferenced_geometry_partition')
        geometry_item['excluded_geometry'] = {k: v for k, v in selection['selected_geometry'].items() if k != 'slice_step_mm'}
    sidecar.write_text(json.dumps(metadata))
    case_path = course / 'metadata/case_metadata.json'
    case = json.loads(case_path.read_text())
    case['course_contract']['planning_ct']['nifti_provenance']['instance_selection'] = selection
    if damage == 'coverage':
        case['course_contract']['authoritative_rtstruct']['sop_instance_uid'] = str(pydicom.dcmread(course / 'RS.dcm').SOPInstanceUID)
    case_path.write_text(json.dumps(case))
    with pytest.raises(CourseContractError, match='ct_localizer_stale_selection_evidence'):
        load_course_contract(course)


@pytest.mark.parametrize('defect', ['irregular', 'too_few', 'missing_references'])
def test_selected_geometry_partition_must_be_regular_and_referenced(tmp_path, defect):
    paths = write_ct_series(tmp_path / 'ct', uniform_z_positions(), signed=False)
    _orientation_partition(paths)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths)
    if defect == 'irregular':
        ds = pydicom.dcmread(paths[-1])
        ds.ImagePositionPatient[2] += 1
        ds.save_as(paths[-1], enforce_file_format=True)
        reason = 'ct_localizer_inconsistent_volume_positions'
    elif defect == 'too_few':
        for path in paths[2:]:
            path.unlink()
        _rtstruct(rs, paths[:2])
        reason = 'ct_localizer_insufficient_volume'
    else:
        rs = None
        reason = 'ct_localizer_rtstruct_references_missing'
    before = _tree_digest(tmp_path)
    with pytest.raises(PlanningCTLocalizerError, match=reason):
        select_localizers(tmp_path / 'ct', rs)
    assert _tree_digest(tmp_path) == before


def test_differing_duplicate_requires_reference_at_positions_without_duplicates(tmp_path):
    paths = write_ct_series(tmp_path / 'ct', uniform_z_positions(), signed=False)
    _duplicate(paths, differing=True)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths[1:])
    with pytest.raises(PlanningCTLocalizerError, match='ct_hygiene_duplicate_position_reference_coverage'):
        select_localizers(tmp_path / 'ct', rs)
