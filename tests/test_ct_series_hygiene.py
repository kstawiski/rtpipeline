"""Synthetic, failure-only CT hygiene and contract/ledger regression tests."""
from __future__ import annotations

import copy
import json

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

import rtpipeline.segmentation as segmentation
from rtpipeline.ct_series_hygiene import (
    CTSeriesHygieneError, METHOD, select_ct_instances,
)
from rtpipeline.course_contract import (
    CourseContract, CourseContractError, _validate_nifti_provenance,
)
from course_contract_test_utils import write_synthetic_rtstruct
from synthetic_ct_series import write_ct_series, uniform_z_positions, mixed_z_positions
from test_nifti_nonuniform_fallback import _config, _tree_digest, dcm2niix


def _duplicate(paths, *, differing=False, rescaled=False):
    ds = pydicom.dcmread(paths[4])
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.InstanceNumber = 100
    if differing:
        pixels = ds.pixel_array.copy()
        pixels[0, 0] += 1
        ds.PixelData = pixels.tobytes()
    if rescaled:
        ds.PixelData = (ds.pixel_array.astype(np.uint16) * 2).tobytes()
        ds.RescaleSlope = 0.5
    path = paths[0].parent / 'duplicate.dcm'
    ds.save_as(path, enforce_file_format=True)
    return path, str(ds.SOPInstanceUID)


def _localizer(paths):
    ds = pydicom.dcmread(paths[0])
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.InstanceNumber = 200
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'LOCALIZER']
    ds.ImageOrientationPatient = [1, 0, 0, 0, 0, 1]
    ds.Rows = 24
    ds.Columns = 32
    ds.PixelData = np.zeros((24, 32), dtype=np.uint16).tobytes()
    path = paths[0].parent / 'localizer.dcm'
    ds.save_as(path, enforce_file_format=True)
    return path, str(ds.SOPInstanceUID)


def _rtstruct(path, paths, *, extra=None, per_contour=False):
    first = pydicom.dcmread(paths[0], stop_before_pixels=True)
    write_synthetic_rtstruct(path, referenced_series_uid=str(first.SeriesInstanceUID))
    ds = pydicom.dcmread(path)
    refs = []
    for uid in [str(pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID) for p in paths] + (extra or []):
        item = Dataset()
        item.ReferencedSOPClassUID = first.SOPClassUID
        item.ReferencedSOPInstanceUID = uid
        refs.append(item)
    if per_contour:
        contour = Dataset()
        contour.ContourImageSequence = Sequence(refs)
        roi = Dataset()
        roi.ContourSequence = Sequence([contour])
        ds.ROIContourSequence = Sequence([roi])
    else:
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence = Sequence(refs)
    ds.save_as(path, enforce_file_format=True)
    return path


def _failed_native(monkeypatch, ct_dir):
    """Some dcm2niix versions tolerate duplicates. Exercise the failure gate."""
    original = segmentation.run_dcm2niix
    def run(config, source, output, recursive_depth=None):
        if source == ct_dir:
            return None
        return original(config, source, output, recursive_depth)
    monkeypatch.setattr(segmentation, 'run_dcm2niix', run)


def _evidence(output):
    return json.loads(output.with_name(output.name[:-7] + '.metadata.json').read_text())


@pytest.mark.parametrize('rescaled', [False, True])
def test_duplicate_identical_converted_with_exclusion(tmp_path, dcm2niix, monkeypatch, rescaled):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    dropped_path, dropped_uid = _duplicate(paths, rescaled=rescaled)
    before = _tree_digest(ct)
    _failed_native(monkeypatch, ct)
    evidence = {}
    result = segmentation._ensure_ct_nifti(_config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', conversion_evidence=evidence)
    assert result is not None
    volume = nib.load(result)
    assert volume.shape == (20, 16, 12)
    values = np.asanyarray(volume.dataobj)
    for index, path in enumerate(paths):
        ds = pydicom.dcmread(path)
        expected = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert np.array_equal(values[:, ::-1, index], expected.T)
    assert _tree_digest(ct) == before
    assert evidence == _evidence(result)['nifti_conversion']
    assert evidence['excluded_instances'] == [{
        'sop_instance_uid': dropped_uid,
        'reason_code': 'identical_rescaled_duplicate_position',
        'retained_sop_instance_uid': str(pydicom.dcmread(paths[4]).SOPInstanceUID),
    }]
    assert evidence['kept_instance_count'] == 12
    assert evidence['excluded_instance_count'] == 1
    assert not (tmp_path / 'nifti' / '.tmp_dcm2niix').exists()


def test_duplicate_differing_refused_with_reason(tmp_path, dcm2niix, monkeypatch):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _duplicate(paths, differing=True)
    _failed_native(monkeypatch, ct)
    evidence = {}
    assert segmentation._ensure_ct_nifti(_config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', conversion_evidence=evidence) is None
    assert evidence['reason_code'] == 'ct_hygiene_duplicate_position_pixels_differ'
    assert not list((tmp_path / 'nifti').iterdir())


@pytest.mark.parametrize('per_contour', [False, True])
def test_mixed_geometry_only_referenced_axial_converted(tmp_path, dcm2niix, monkeypatch, per_contour):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _, excluded_uid = _localizer(paths)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths, per_contour=per_contour)
    _failed_native(monkeypatch, ct)
    result = segmentation._ensure_ct_nifti(_config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs)
    assert result is not None
    record = _evidence(result)['nifti_conversion']
    assert record['kept_instance_count'] == 12
    assert record['excluded_instances'] == [{'sop_instance_uid': excluded_uid, 'reason_code': 'unreferenced_geometry_partition'}]
    assert record['rtstruct_references_all_kept'] is True
    assert nib.load(result).shape == (20, 16, 12)
    assert nib.load(result).header.get_zooms() == (1, 1, 5)


@pytest.mark.parametrize('reference_mode,reason', [
    ('both', 'ct_hygiene_references_span_geometry_partitions'),
    ('missing', 'ct_hygiene_rtstruct_references_unresolvable'),
    ('none', 'ct_hygiene_mixed_geometry_without_image_references'),
    ('series_only', 'ct_hygiene_mixed_geometry_without_image_references'),
])
def test_mixed_ambiguous_references_refused(tmp_path, dcm2niix, monkeypatch, reference_mode, reason):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _, scout_uid = _localizer(paths)
    rs = None
    if reference_mode in ('both', 'missing'):
        rs = _rtstruct(tmp_path / 'RS.dcm', paths, extra=[scout_uid if reference_mode == 'both' else generate_uid()])
    if reference_mode == 'series_only':
        rs = write_synthetic_rtstruct(tmp_path / 'RS.dcm', referenced_series_uid=str(pydicom.dcmread(paths[0]).SeriesInstanceUID))
    _failed_native(monkeypatch, ct)
    evidence = {}
    assert segmentation._ensure_ct_nifti(_config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence) is None
    assert evidence['reason_code'] == reason


def test_duplicate_with_uneven_spacing_uses_existing_signed_fallback(tmp_path, dcm2niix, monkeypatch):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, mixed_z_positions(), signed=False)
    _duplicate(paths)
    _failed_native(monkeypatch, ct)
    result = segmentation._ensure_ct_nifti(_config(tmp_path, dcm2niix), ct, tmp_path / 'nifti')
    assert result is not None
    assert _evidence(result)['nifti_conversion']['downstream_conversion']['method'] == 'dcm2niix_signed_staging_equalized'


@pytest.mark.parametrize('signed,z', [(False, uniform_z_positions()), (True, uniform_z_positions()), (True, mixed_z_positions()), (False, mixed_z_positions())])
def test_existing_conversions_byte_identical(tmp_path, dcm2niix, monkeypatch, signed, z):
    ct = tmp_path / 'ct'
    write_ct_series(ct, z, signed=signed)
    config = _config(tmp_path, dcm2niix)
    before = segmentation._ensure_ct_nifti(config, ct, tmp_path / 'before')
    def unexpected(*args, **kwargs):
        raise AssertionError('hygiene changed an existing conversion')
    monkeypatch.setattr(segmentation, 'convert_ct_series_hygiene', unexpected)
    after = segmentation._ensure_ct_nifti(config, ct, tmp_path / 'after')
    assert before.read_bytes() == after.read_bytes()


def test_contract_binds_selection_and_authoritative_rtstruct(tmp_path, dcm2niix, monkeypatch):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _localizer(paths)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths)
    _failed_native(monkeypatch, ct)
    result = segmentation._ensure_ct_nifti(_config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs)
    meta = _evidence(result)
    provenance = {key: meta[key] for key in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')}
    sidecar = result.with_name(result.name[:-7] + '.metadata.json')
    provenance['sidecar_path'] = str(sidecar.relative_to(tmp_path))
    contract = CourseContract(tmp_path, tmp_path / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
    planning = {'nifti_provenance': provenance}
    _validate_nifti_provenance(contract, planning, ct, result, meta['series_instance_uid'])
    altered = copy.deepcopy(planning)
    altered['nifti_provenance']['nifti_conversion']['excluded_instances'] = []
    with pytest.raises(CourseContractError, match='sidecar differs'):
        _validate_nifti_provenance(contract, altered, ct, result, meta['series_instance_uid'])
    _rtstruct(rs, paths, extra=[generate_uid()])
    with pytest.raises(CourseContractError, match='ct_hygiene_rtstruct_references_unresolvable'):
        _validate_nifti_provenance(contract, planning, ct, result, meta['series_instance_uid'])


def test_organize_ledger_records_validated_exclusions_and_refusal_codes(tmp_path, dcm2niix, monkeypatch):
    from rtpipeline import organize
    from organize_io_fixture import synthetic

    root = tmp_path / 'input'
    synthetic(root, ct_slices=12, all_slices=True)
    folders = [root / patient / course for patient in ('SYNTH_A', 'SYNTH_B') for course in ('0', '1')]
    for index, folder in enumerate(folders):
        paths = [folder / f'ct{i}.dcm' for i in range(12)]
        if index < 2:
            _duplicate(paths, differing=index == 1)
        else:
            scout_path, scout_uid = _localizer(paths)
            # These inputs reached planning conversion in the reported failure.
            # Explicit LOCALIZER tags are excluded earlier by existing policy.
            scout = pydicom.dcmread(scout_path)
            del scout.ImageType
            # RF13b now handles this unreferenced geometry at publication,
            # even without exact-position duplicates.
            scout.ImagePositionPatient[2] = -20
            scout.save_as(scout_path, enforce_file_format=True)
            if index == 3:
                rs_path = folder / 'struct.dcm'
                rs = pydicom.dcmread(rs_path)
                item = Dataset()
                item.ReferencedSOPClassUID = pydicom.dcmread(paths[0]).SOPClassUID
                item.ReferencedSOPInstanceUID = scout_uid
                rs.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(item)
                rs.save_as(rs_path, enforce_file_format=True)
    original = segmentation.run_dcm2niix
    def run(config, source, output, recursive_depth=None):
        if source.name == 'CT' and len(list(source.iterdir())) > 12:
            return None
        return original(config, source, output, recursive_depth)
    monkeypatch.setattr(segmentation, 'run_dcm2niix', run)
    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '1')
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    config = _config(tmp_path, dcm2niix)
    config.dicom_root = root
    config.max_workers_override = 1
    config.dicom_copy_use_hardlinks = False
    courses = organize.organize_and_merge(config, metadata_snapshot={})
    ledger = json.loads((config.output_root / '_COURSES/organize_ledger.json').read_text())
    assert len(courses) == 3
    assert ledger['validated_course_count'] == 3
    assert ledger['technical_quarantine_count'] == 1
    for entry in ledger['courses']:
        evidence = entry.get('planning_ct_conversion')
        if entry['status'] == 'validated':
            if not evidence:
                # Reference-authorized exclusions precede native conversion.
                from rtpipeline.course_contract import load_course_contract
                contract = load_course_contract(config.output_root / entry['patient'] / entry['course'])
                evidence = contract.planning_ct['nifti_provenance']['instance_selection']
                assert evidence['reason'] in {'identical_rescaled_duplicate_position',
                                              'rtstruct_referenced_volume_selection'}
            else:
                assert evidence['method'] == METHOD
            assert evidence['excluded_instance_count'] == 1
            assert evidence['kept_instance_count'] == 12
            assert evidence['rtstruct_references_all_kept'] is True
        else:
            assert evidence['reason_code'] == 'ct_hygiene_references_span_geometry_partitions'
            assert evidence['reason_code'] in entry['reason']


@pytest.mark.parametrize('differing', [False, True])
def test_real_converter_duplicate_failure_and_recovery(tmp_path, dcm2niix, differing):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _duplicate(paths, differing=differing)
    config = _config(tmp_path, dcm2niix)
    # Do not alter converter execution in this regression of the actual rc=1.
    if segmentation.run_dcm2niix(config, ct, tmp_path / 'probe', recursive_depth=0) is not None:
        pytest.skip('this converter accepts duplicate positions; existing output is preserved')
    evidence = {}
    result = segmentation._ensure_ct_nifti(config, ct, tmp_path / 'nifti', conversion_evidence=evidence)
    if differing:
        assert result is None
        assert evidence['reason_code'] == 'ct_hygiene_duplicate_position_pixels_differ'
    else:
        assert result is not None
        assert evidence['excluded_instance_count'] == 1


def test_existing_native_mixed_conversion_is_unchanged(tmp_path, dcm2niix, monkeypatch):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _localizer(paths)
    config = _config(tmp_path, dcm2niix)
    native = segmentation.run_dcm2niix(config, ct, tmp_path / 'native', recursive_depth=0)
    if native is None:
        pytest.skip('this converter refuses the mixed fixture')
    def unexpected(*args, **kwargs):
        raise AssertionError('hygiene must not change a successful native conversion')
    monkeypatch.setattr(segmentation, 'convert_ct_series_hygiene', unexpected)
    result = segmentation._ensure_ct_nifti(config, ct, tmp_path / 'nifti', dcm2niix_depth=0)
    assert result.read_bytes() == native.read_bytes()


def test_same_stored_pixels_with_different_rescale_are_refused(tmp_path):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    duplicate, _ = _duplicate(paths)
    ds = pydicom.dcmread(duplicate)
    ds.RescaleIntercept = -999
    ds.save_as(duplicate, enforce_file_format=True)
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_duplicate_position_pixels_differ'):
        select_ct_instances(ct)


def test_duplicate_choice_retains_the_referenced_equivalent(tmp_path):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    duplicate, uid = _duplicate(paths)
    rs = _rtstruct(tmp_path / 'RS.dcm', [duplicate])
    kept, evidence = select_ct_instances(ct, rs)
    assert duplicate in kept
    assert paths[4] not in kept
    assert evidence['rtstruct_references_all_kept'] is True
    assert evidence['excluded_instances'][0]['retained_sop_instance_uid'] == uid


def test_mixed_nonaxial_referenced_partition_refused(tmp_path):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    scout, _ = _localizer(paths)
    rs = _rtstruct(tmp_path / 'RS.dcm', [scout])
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_not_consistent_axial_volume'):
        select_ct_instances(ct, rs)


def test_near_duplicate_position_tolerance_and_source_immutability(tmp_path):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    duplicate, _ = _duplicate(paths)
    ds = pydicom.dcmread(duplicate)
    ds.ImagePositionPatient[2] = float(ds.ImagePositionPatient[2]) + 0.005
    ds.save_as(duplicate, enforce_file_format=True)
    before = _tree_digest(ct)
    kept, evidence = select_ct_instances(ct)
    assert len(kept) == 12
    assert evidence['position_tolerance_mm'] == 0.01
    assert _tree_digest(ct) == before


def test_selected_geometry_with_multiple_converter_volumes_refused(tmp_path):
    from rtpipeline.ct_series_hygiene import convert_ct_series_hygiene
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _duplicate(paths)
    work = tmp_path / 'work'
    work.mkdir()
    def ambiguous(source, output):
        for name in ('first.nii.gz', 'second.nii.gz'):
            nib.save(nib.Nifti1Image(np.zeros((20, 16, 12)), np.eye(4)), output / name)
        return output / 'first.nii.gz'
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_ambiguous_converter_outputs'):
        convert_ct_series_hygiene(ct, work, ambiguous)


def test_hygiene_evidence_survives_reuse(tmp_path, dcm2niix, monkeypatch):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _duplicate(paths)
    _failed_native(monkeypatch, ct)
    config = _config(tmp_path, dcm2niix)
    first = segmentation._ensure_ct_nifti(config, ct, tmp_path / 'nifti')
    before = _evidence(first)
    second = segmentation._ensure_ct_nifti(config, ct, tmp_path / 'nifti')
    assert second == first
    assert _evidence(second) == before


def test_converter_cannot_silently_drop_a_selected_slice(tmp_path):
    from rtpipeline.ct_series_hygiene import convert_ct_series_hygiene
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _duplicate(paths)
    work = tmp_path / 'work'
    work.mkdir()
    def incomplete(source, output):
        result = output / 'volume.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((20, 16, 11)), np.eye(4)), result)
        return result
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_converter_did_not_preserve_selected_extent'):
        convert_ct_series_hygiene(ct, work, incomplete)
