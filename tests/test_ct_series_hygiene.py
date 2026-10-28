"""Synthetic, failure-only CT hygiene and contract/ledger regression tests."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

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


def _rtstruct_dual(path, frame_paths, contour_paths=None, *, frame_extra=None, contour_extra=None):
    """Write a synthetic RTSTRUCT with independent frame-level and contour-level image references."""
    if contour_paths is None:
        contour_paths = frame_paths
    ref_paths = frame_paths or contour_paths
    first = pydicom.dcmread(ref_paths[0], stop_before_pixels=True)
    write_synthetic_rtstruct(path, referenced_series_uid=str(first.SeriesInstanceUID))
    ds = pydicom.dcmread(path)

    def _items(paths_list, extra_list):
        items = []
        for uid in [str(pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID) for p in paths_list] + (extra_list or []):
            item = Dataset()
            item.ReferencedSOPClassUID = first.SOPClassUID
            item.ReferencedSOPInstanceUID = uid
            items.append(item)
        return items

    frame_items = _items(frame_paths, frame_extra)
    ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence = Sequence(frame_items)

    contour_items = _items(contour_paths, contour_extra)
    if contour_items:
        contours = []
        for item in contour_items:
            contour = Dataset()
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = 3
            contour.ContourData = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            contour.ContourImageSequence = Sequence([item])
            contours.append(contour)
        roi = Dataset()
        roi.ReferencedROINumber = 1
        roi.ContourSequence = Sequence(contours)
        ds.ROIContourSequence = Sequence([roi])
    else:
        ds.ROIContourSequence = Sequence([])
    ds.save_as(path, enforce_file_format=True)
    return path


def test_pattern_e_mixed_image_type_harmonized_and_verified(tmp_path, dcm2niix):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(16), signed=False)
    for p in paths[5:8]:
        ds = pydicom.dcmread(p)
        ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.save_as(p, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[2:14])
    before = _tree_digest(ct)
    config = _config(tmp_path, dcm2niix)
    assert segmentation.run_dcm2niix(config, ct, tmp_path / 'probe', recursive_depth=0) is None
    evidence = {}
    result = segmentation._ensure_ct_nifti(
        config, ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    )
    assert result is not None
    assert _tree_digest(ct) == before
    assert evidence['method'] == METHOD
    assert evidence['status'] == 'converted'
    assert evidence['reason_code'] == 'mixed_image_type_harmonized'
    assert evidence['selection_basis'] == 'rtstruct_complete_references_image_type_harmonized'
    assert evidence['kept_instance_count'] == 16
    assert evidence['excluded_instance_count'] == 0
    assert evidence['original_image_type_counts'] == {
        'DERIVED\\SECONDARY\\AXIAL': 3,
        'ORIGINAL\\PRIMARY\\AXIAL': 13,
    }
    assert evidence['harmonized_image_type'] == ['ORIGINAL', 'PRIMARY', 'AXIAL']
    volume = nib.load(result)
    assert volume.shape == (20, 16, 16)
    assert '_Eq_' not in result.name
    values = np.asanyarray(volume.dataobj)
    for index, path in enumerate(paths):
        ds = pydicom.dcmread(path)
        expected = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert np.array_equal(values[:, ::-1, index], expected.T)
    meta = _evidence(result)
    provenance = {key: meta[key] for key in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')}
    sidecar = result.with_name(result.name[:-7] + '.metadata.json')
    provenance['sidecar_path'] = str(sidecar.relative_to(tmp_path))
    contract = CourseContract(tmp_path, tmp_path / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
    _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, result, meta['series_instance_uid'])


@pytest.mark.parametrize('defect,expected_reason', [
    ('rescale_slope', 'ct_hygiene_mixed_image_type_refused'),
    ('rescale_intercept', 'ct_hygiene_mixed_image_type_refused'),
    ('geometry', 'ct_hygiene_mixed_image_type_refused'),
    ('acquisition', 'ct_hygiene_mixed_image_type_refused'),
    ('incomplete_references', 'ct_hygiene_incomplete_rtstruct_references'),
    ('tampered_converter_pixels', 'ct_hygiene_mixed_image_type_refused'),
])
def test_pattern_e_negative_controls_refused(tmp_path, dcm2niix, monkeypatch, defect, expected_reason):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    for p in paths[4:7]:
        ds = pydicom.dcmread(p)
        ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        if defect == 'rescale_slope' and p == paths[5]:
            ds.RescaleSlope = 2.0
        elif defect == 'rescale_intercept' and p == paths[5]:
            ds.RescaleIntercept = -500.0
        elif defect == 'geometry' and p == paths[5]:
            ds.PixelSpacing = [2.0, 2.0]
        elif defect == 'acquisition' and p == paths[5]:
            ds.AcquisitionNumber = 2
        ds.save_as(p, enforce_file_format=True)
    frame_refs = paths[:-1] if defect == 'incomplete_references' else paths
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', frame_refs, frame_refs[:6])
    config = _config(tmp_path, dcm2niix)
    if defect == 'tampered_converter_pixels':
        original_run = segmentation.run_dcm2niix
        def tampered_run(cfg, source, output, recursive_depth=None):
            if source == ct:
                return None
            out = original_run(cfg, source, output, recursive_depth)
            if out is not None:
                img = nib.load(out)
                arr = np.asanyarray(img.dataobj).copy()
                arr[0, 0, 0] += 1
                nib.save(nib.Nifti1Image(arr, img.affine, img.header), out)
            return out
        monkeypatch.setattr(segmentation, 'run_dcm2niix', tampered_run)
    else:
        _failed_native(monkeypatch, ct)
    evidence = {}
    assert segmentation._ensure_ct_nifti(
        config, ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    ) is None
    assert evidence['reason_code'] == expected_reason


def test_pattern_b_referenced_duplicate_differing_pixels_converted(tmp_path, dcm2niix):
    from rtpipeline.ct_series_hygiene import _pixels
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    for p in paths:
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 2
        ds.save_as(p, enforce_file_format=True)
    dup_path, dup_uid = _duplicate(paths, differing=True)
    dup_ds = pydicom.dcmread(dup_path)
    dup_ds.AcquisitionNumber = 3
    dup_ds.save_as(dup_path, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[3:8])
    before = _tree_digest(ct)
    config = _config(tmp_path, dcm2niix)
    evidence = {}
    result = segmentation._ensure_ct_nifti(
        config, ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    )
    assert result is not None
    assert _tree_digest(ct) == before
    assert evidence['method'] == METHOD
    assert evidence['reason_code'] == 'unreferenced_duplicate_position'
    assert evidence['selection_basis'] == 'rtstruct_frame_and_contour_complete_references'
    assert evidence['kept_instance_count'] == 12
    assert evidence['excluded_instance_count'] == 1
    retained_uid = str(pydicom.dcmread(paths[4]).SOPInstanceUID)
    expected_retained_hash = hashlib.sha256(_pixels(paths[4])).hexdigest()
    expected_excluded_hash = hashlib.sha256(_pixels(dup_path)).hexdigest()
    assert expected_retained_hash != expected_excluded_hash
    assert evidence['excluded_instances'] == [{
        'sop_instance_uid': dup_uid,
        'reason_code': 'unreferenced_duplicate_position',
        'retained_sop_instance_uid': retained_uid,
        'image_position_patient': [float(v) for v in pydicom.dcmread(paths[4]).ImagePositionPatient],
        'retained_rescaled_pixels_sha256': expected_retained_hash,
        'excluded_rescaled_pixels_sha256': expected_excluded_hash,
    }]
    meta = _evidence(result)
    provenance = {key: meta[key] for key in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')}
    sidecar = result.with_name(result.name[:-7] + '.metadata.json')
    provenance['sidecar_path'] = str(sidecar.relative_to(tmp_path))
    contract = CourseContract(tmp_path, tmp_path / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
    _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, result, meta['series_instance_uid'])


@pytest.mark.parametrize('mode,expected_reason', [
    ('missing_contour_at_dup', 'ct_hygiene_duplicate_position_reference_coverage'),
    ('frame_only', 'ct_hygiene_duplicate_position_reference_coverage'),
    ('contour_only', 'ct_hygiene_duplicate_position_reference_coverage'),
    ('incomplete_frame_refs', 'ct_hygiene_duplicate_position_reference_coverage'),
    ('both_duplicates_referenced', 'ct_hygiene_referenced_duplicate_would_be_excluded'),
    ('external_reference', 'ct_hygiene_rtstruct_references_unresolvable'),
])
def test_pattern_b_negative_controls_refused(tmp_path, dcm2niix, monkeypatch, mode, expected_reason):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    dup_path, _ = _duplicate(paths, differing=True)
    dup_ds = pydicom.dcmread(dup_path)
    dup_ds.AcquisitionNumber = 3
    dup_ds.save_as(dup_path, enforce_file_format=True)
    if mode == 'missing_contour_at_dup':
        rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, [paths[0], paths[1]])
    elif mode == 'frame_only':
        rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, [])
    elif mode == 'contour_only':
        rs = _rtstruct_dual(tmp_path / 'RS.dcm', [], paths)
    elif mode == 'incomplete_frame_refs':
        rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths[:-1], paths[:6])
    elif mode == 'both_duplicates_referenced':
        rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths + [dup_path], paths)
    else:
        rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths, frame_extra=[generate_uid()])
    _failed_native(monkeypatch, ct)
    evidence = {}
    assert segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    ) is None
    assert evidence['reason_code'] == expected_reason


def test_pattern_a_referenced_uniform_subset_preferred_before_equalization(tmp_path, dcm2niix):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    stray = pydicom.dcmread(paths[3])
    stray_uid = str(generate_uid())
    stray.SOPInstanceUID = stray_uid
    stray.file_meta.MediaStorageSOPInstanceUID = stray_uid
    stray.InstanceNumber = 99
    stray.AcquisitionNumber = 2
    stray.ImagePositionPatient = [-10.0, -20.0, 17.0]
    stray_path = ct / 'stray_slice.dcm'
    stray.save_as(stray_path, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[2:8])
    before = _tree_digest(ct)
    config = _config(tmp_path, dcm2niix)
    assert segmentation.run_dcm2niix(config, ct, tmp_path / 'probe', recursive_depth=0) is None
    evidence = {}
    result = segmentation._ensure_ct_nifti(
        config, ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    )
    assert result is not None
    assert _tree_digest(ct) == before
    assert evidence['method'] == METHOD
    assert evidence['reason_code'] == 'unreferenced_slice_position'
    assert evidence['selection_basis'] == 'rtstruct_complete_referenced_volume'
    assert 'downstream_conversion' not in evidence
    assert evidence['kept_instance_count'] == 12
    assert evidence['excluded_instance_count'] == 1
    assert evidence['excluded_instances'] == [{
        'sop_instance_uid': stray_uid,
        'reason_code': 'unreferenced_slice_position',
    }]
    volume = nib.load(result)
    assert volume.shape == (20, 16, 12)
    assert volume.header.get_zooms() == (1.0, 1.0, 5.0)
    assert '_Eq_' not in result.name
    values = np.asanyarray(volume.dataobj)
    for index, path in enumerate(paths):
        ds = pydicom.dcmread(path)
        expected = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert np.array_equal(values[:, ::-1, index], expected.T)


def test_pattern_a_nonuniform_referenced_subset_uses_signed_staging_on_subset(tmp_path, dcm2niix):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, mixed_z_positions(), signed=False)
    stray = pydicom.dcmread(paths[2])
    stray_uid = str(generate_uid())
    stray.SOPInstanceUID = stray_uid
    stray.file_meta.MediaStorageSOPInstanceUID = stray_uid
    stray.InstanceNumber = 99
    stray.AcquisitionNumber = 2
    stray.ImagePositionPatient = [-10.0, -20.0, 12.0]
    stray.save_as(ct / 'stray_slice.dcm', enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[:6])
    evidence = {}
    result = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    )
    assert result is not None
    assert evidence['method'] == METHOD
    assert evidence['reason_code'] == 'unreferenced_slice_position'
    assert evidence['kept_instance_count'] == 12
    assert evidence['excluded_instance_count'] == 1
    assert evidence['downstream_conversion']['method'] == 'dcm2niix_signed_staging_equalized'


def test_pattern_a_negative_control_incomplete_references_refused(tmp_path, dcm2niix):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    stray = pydicom.dcmread(paths[3])
    stray_uid = str(generate_uid())
    stray.SOPInstanceUID = stray_uid
    stray.file_meta.MediaStorageSOPInstanceUID = stray_uid
    stray.InstanceNumber = 99
    stray.AcquisitionNumber = 2
    stray.ImagePositionPatient = [-10.0, -20.0, 17.0]
    stray.save_as(ct / 'stray_slice.dcm', enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths[:-1], paths[:6])
    evidence = {}
    assert segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs, conversion_evidence=evidence
    ) is None
    assert evidence['reason_code'] == 'ct_hygiene_incomplete_rtstruct_references'


def _build_pattern_d_series(ct_dir, rs_path, slices=12):
    stack1 = write_ct_series(ct_dir, uniform_z_positions(slices), signed=False)
    stack2 = []
    for idx, p in enumerate(stack1):
        ds1 = pydicom.dcmread(p)
        ds1.ScanOptions = 'HELIX'
        ds1.ConvolutionKernel = 'SHARP'
        ds1.ReconstructionDiameter = 300.0
        ds1.WindowCenter = 40.0
        ds1.Exposure = 200
        ds1.save_as(p, enforce_file_format=True)

        ds2 = pydicom.dcmread(p)
        uid2 = str(generate_uid())
        ds2.SOPInstanceUID = uid2
        ds2.file_meta.MediaStorageSOPInstanceUID = uid2
        ds2.InstanceNumber = 100 + idx + 1
        ds2.ScanOptions = 'SURVIEW'
        ds2.ConvolutionKernel = 'SMOOTH'
        ds2.ReconstructionDiameter = 280.0
        ds2.WindowCenter = 50.0
        ds2.Exposure = 180
        pixels = ds2.pixel_array.copy()
        pixels[0, 0] += 15
        ds2.PixelData = pixels.tobytes()
        p2 = ct_dir / f'stack2_{idx:03d}.dcm'
        ds2.save_as(p2, enforce_file_format=True)
        stack2.append(p2)

    loc1, _ = _localizer(stack1)
    loc2 = ct_dir / 'localizer_2.dcm'
    ds_loc2 = pydicom.dcmread(loc1)
    uid_loc2 = str(generate_uid())
    ds_loc2.SOPInstanceUID = uid_loc2
    ds_loc2.file_meta.MediaStorageSOPInstanceUID = uid_loc2
    ds_loc2.InstanceNumber = 201
    ds_loc2.save_as(loc2, enforce_file_format=True)

    _rtstruct_dual(rs_path, stack1 + stack2, stack1[:slices - 2] + stack2[:2])
    return stack1, stack2, [loc1, loc2]


def test_pattern_d_spanning_reconstruction_stacks_refused_and_governed_override(tmp_path, dcm2niix):
    from rtpipeline.cli import _apply_dicom_copy_yaml_config
    ct = tmp_path / 'SYNTH' / 'course_0' / 'DICOM' / 'CT'
    rs = tmp_path / 'SYNTH' / 'course_0' / 'RS.dcm'
    stack1, stack2, localizers = _build_pattern_d_series(ct, rs, slices=12)
    config = _config(tmp_path, dcm2niix)

    # 1. Without override: fails closed with ct_hygiene_references_span_reconstruction_stacks
    evidence = {}
    assert segmentation._ensure_ct_nifti(
        config,
        ct,
        tmp_path / 'SYNTH' / 'course_0' / 'NIFTI',
        authoritative_rtstruct=rs,
        conversion_evidence=evidence,
        require_hygiene=True,
    ) is None
    assert evidence['reason_code'] == 'ct_hygiene_references_span_reconstruction_stacks'
    candidates = evidence.get('candidate_stacks')
    assert isinstance(candidates, list) and len(candidates) == 2
    serialized_candidates = json.dumps(candidates)
    for p in stack1 + stack2 + localizers:
        uid = str(pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID)
        assert uid not in serialized_candidates
    assert str(tmp_path) not in serialized_candidates

    sharp_stack = next(c for c in candidates if c['ConvolutionKernel'] == 'SHARP')
    smooth_stack = next(c for c in candidates if c['ConvolutionKernel'] == 'SMOOTH')
    assert sharp_stack['instance_count'] == 12
    assert sharp_stack['contour_reference_count'] == 10
    assert sharp_stack['frame_reference_count'] == 12
    assert smooth_stack['instance_count'] == 12
    assert smooth_stack['contour_reference_count'] == 2
    assert smooth_stack['frame_reference_count'] == 12

    # 2. Missing adjudication_record_id or mismatching stack_signature is refused
    for bad_override, expected_code in [
        ({'reason': 'missing id', 'stack_signature': sharp_stack['stack_signature']}, 'ct_hygiene_stack_override_invalid_governance'),
        ({'adjudication_record_id': 'ADJ-1', 'stack_signature': sharp_stack['stack_signature']}, 'ct_hygiene_stack_override_invalid_governance'),
        ({'adjudication_record_id': 'ADJ-1', 'reason': 'wrong kernel', 'stack_signature': {**sharp_stack['stack_signature'], 'ConvolutionKernel': 'BONE'}}, 'ct_hygiene_stack_override_signature_mismatch'),
    ]:
        bad_ev = {}
        assert segmentation._ensure_ct_nifti(
            config,
            ct,
            tmp_path / 'SYNTH' / 'course_0' / 'NIFTI',
            authoritative_rtstruct=rs,
            conversion_evidence=bad_ev,
            stack_override=bad_override,
        ) is None
        assert bad_ev['reason_code'] == expected_code

    # 3. With matching governed override via YAML config parsing: converts selected stack and validates contract
    _apply_dicom_copy_yaml_config(config, {
        'planning_ct_stack_overrides': {
            'SYNTH/course_0': {
                'adjudication_record_id': 'WS201-ADJ-001',
                'reason': 'Select primary HELIX/SHARP planning stack (10/12 contour references)',
                'stack_signature': sharp_stack['stack_signature'],
            }
        }
    })
    ok_ev = {}
    nifti_dir = tmp_path / 'SYNTH' / 'course_0' / 'NIFTI'
    result = segmentation._ensure_ct_nifti(
        config,
        ct,
        nifti_dir,
        authoritative_rtstruct=rs,
        conversion_evidence=ok_ev,
    )
    assert result is not None
    assert ok_ev['method'] == METHOD
    assert ok_ev['reason_code'] == 'governed_stack_override_selected'
    assert ok_ev['selection_basis'] == 'governed_stack_override'
    assert ok_ev['kept_instance_count'] == 12
    assert ok_ev['excluded_instance_count'] == 14
    assert ok_ev['stack_override']['adjudication_record_id'] == 'WS201-ADJ-001'
    meta = _evidence(result)
    provenance = {key: meta[key] for key in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')}
    sidecar = result.with_name(result.name[:-7] + '.metadata.json')
    course_root = tmp_path / 'SYNTH' / 'course_0'
    provenance['sidecar_path'] = str(sidecar.relative_to(course_root))
    contract = CourseContract(course_root, course_root / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
    _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, result, meta['series_instance_uid'])


def test_organize_publishes_contracts_for_patterns_a_b_d_e(tmp_path, dcm2niix, monkeypatch):
    from rtpipeline import organize
    from rtpipeline.course_contract import load_course_contract
    from organize_io_fixture import synthetic

    root = tmp_path / 'input'
    synthetic(root, ct_slices=12, all_slices=True)

    # Course 1 (SYNTH_A/0): Pattern E - mixed ImageType within one uniform stack
    folder_e = root / 'SYNTH_A' / '0'
    for idx in range(12):
        p = folder_e / f'ct{idx}.dcm'
        ds = pydicom.dcmread(p)
        ds.ImageType = (
            ['DERIVED', 'SECONDARY', 'AXIAL']
            if idx in (4, 5, 6)
            else ['ORIGINAL', 'PRIMARY', 'AXIAL']
        )
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.save_as(p, enforce_file_format=True)

    # Course 2 (SYNTH_A/1): Pattern B - duplicate slice at one position from second AcquisitionNumber with differing pixels
    folder_b = root / 'SYNTH_A' / '1'
    paths_b = [folder_b / f'ct{i}.dcm' for i in range(12)]
    for p in paths_b:
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 2
        ds.save_as(p, enforce_file_format=True)
    dup_b, _ = _duplicate(paths_b, differing=True)
    ds_dup_b = pydicom.dcmread(dup_b)
    ds_dup_b.AcquisitionNumber = 3
    ds_dup_b.save_as(dup_b, enforce_file_format=True)

    # Course 3 (SYNTH_B/0): Pattern A - unreferenced stray slice at intermediate z from second AcquisitionNumber
    folder_a = root / 'SYNTH_B' / '0'
    paths_a = [folder_a / f'ct{i}.dcm' for i in range(12)]
    for p in paths_a:
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 1
        ds.save_as(p, enforce_file_format=True)
    stray_a = pydicom.dcmread(paths_a[3])
    stray_uid = str(generate_uid())
    stray_a.SOPInstanceUID = stray_uid
    stray_a.file_meta.MediaStorageSOPInstanceUID = stray_uid
    stray_a.InstanceNumber = 99
    stray_a.AcquisitionNumber = 2
    stray_a.ImagePositionPatient = [-10.0, -10.0, 17.0]
    stray_a.save_as(folder_a / 'ct_stray.dcm', enforce_file_format=True)

    # Course 4 (SYNTH_B/1): Pattern D - two complete reconstruction stacks + 2 localizers with spanning references
    folder_d = root / 'SYNTH_B' / '1'
    paths_d = [folder_d / f'ct{i}.dcm' for i in range(12)]
    stack2_uids = []
    for idx, p in enumerate(paths_d):
        ds1 = pydicom.dcmread(p)
        ds1.ScanOptions = 'HELIX'
        ds1.ConvolutionKernel = 'SHARP'
        ds1.ReconstructionDiameter = 300.0
        ds1.save_as(p, enforce_file_format=True)

        ds2 = pydicom.dcmread(p)
        uid2 = str(generate_uid())
        ds2.SOPInstanceUID = uid2
        ds2.file_meta.MediaStorageSOPInstanceUID = uid2
        ds2.InstanceNumber = 100 + idx + 1
        ds2.ScanOptions = 'SURVIEW'
        ds2.ConvolutionKernel = 'SMOOTH'
        ds2.ReconstructionDiameter = 280.0
        arr = bytearray(ds2.PixelData)
        arr[0] = (arr[0] + 5) % 256
        ds2.PixelData = bytes(arr)
        ds2.save_as(folder_d / f'ct_stack2_{idx}.dcm', enforce_file_format=True)
        stack2_uids.append(uid2)

    loc1, _ = _localizer(paths_d)
    loc2 = folder_d / 'localizer2.dcm'
    ds_loc2 = pydicom.dcmread(loc1)
    uid_loc2 = str(generate_uid())
    ds_loc2.SOPInstanceUID = uid_loc2
    ds_loc2.file_meta.MediaStorageSOPInstanceUID = uid_loc2
    ds_loc2.InstanceNumber = 201
    ds_loc2.save_as(loc2, enforce_file_format=True)

    rs_d_path = folder_d / 'struct.dcm'
    rs_d = pydicom.dcmread(rs_d_path)
    first_sop_class = pydicom.dcmread(paths_d[0], stop_before_pixels=True).SOPClassUID
    for uid2 in stack2_uids:
        item = Dataset()
        item.ReferencedSOPClassUID = first_sop_class
        item.ReferencedSOPInstanceUID = uid2
        rs_d.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(item)
    # Switch 2 contours in ROI 1 to reference stack 2
    for c_idx in (0, 1):
        rs_d.ROIContourSequence[0].ContourSequence[c_idx].ContourImageSequence[0].ReferencedSOPInstanceUID = stack2_uids[c_idx]
    rs_d.save_as(rs_d_path, enforce_file_format=True)

    monkeypatch.setenv('RTPIPELINE_INDEX_PROCESSES', '1')
    monkeypatch.setenv('RTPIPELINE_MASK_PROCESSES', '1')
    config = _config(tmp_path / 'run1', dcm2niix)
    config.dicom_root = root
    config.max_workers_override = 1
    config.dicom_copy_use_hardlinks = False

    # Run 1: no override for SYNTH_B/1 -> 3 courses validated (Patterns E, B, A), 1 quarantined (Pattern D)
    courses = organize.organize_and_merge(config, metadata_snapshot={})
    ledger = json.loads((config.output_root / '_COURSES/organize_ledger.json').read_text())
    assert len(courses) == 3
    assert ledger['validated_course_count'] == 3
    assert ledger['technical_quarantine_count'] == 1

    quarantined = ledger['technical_quarantines'][0]
    assert quarantined['patient'] == 'SYNTH_B' and quarantined['course'] == '2024-02'
    assert quarantined['planning_ct_conversion']['reason_code'] == 'ct_hygiene_references_span_reconstruction_stacks'
    cand_stacks = quarantined['planning_ct_conversion']['candidate_stacks']
    assert len(cand_stacks) == 2
    sharp_sig = next(c['stack_signature'] for c in cand_stacks if c['ConvolutionKernel'] == 'SHARP')

    for patient, course_id, expected_reason in [
        ('SYNTH_A', '2024-01', 'mixed_image_type_harmonized'),
        ('SYNTH_A', '2024-02', 'unreferenced_duplicate_position'),
        ('SYNTH_B', '2024-01', 'unreferenced_slice_position'),
    ]:
        contract = load_course_contract(config.output_root / patient / course_id)
        conv = contract.planning_ct['nifti_provenance']['nifti_conversion']
        assert conv['method'] == METHOD
        assert conv['reason_code'] == expected_reason

    # Run 2: with governed stack override for SYNTH_B/2024-02 -> all 4 courses validated
    config2 = _config(tmp_path / 'run2', dcm2niix)
    config2.dicom_root = root
    config2.max_workers_override = 1
    config2.dicom_copy_use_hardlinks = False
    config2.planning_ct_stack_overrides = {
        'SYNTH_B/2024-02': {
            'adjudication_record_id': 'WS201-SYNTH-B1',
            'reason': 'Select SHARP stack with 10/12 contour references',
            'stack_signature': sharp_sig,
        }
    }
    courses2 = organize.organize_and_merge(config2, metadata_snapshot={})
    assert len(courses2) == 4
    contract_d = load_course_contract(config2.output_root / 'SYNTH_B' / '2024-02')
    conv_d = contract_d.planning_ct['nifti_provenance']['nifti_conversion']
    assert conv_d['method'] == METHOD
    assert conv_d['reason_code'] == 'governed_stack_override_selected'
    assert conv_d['stack_override']['adjudication_record_id'] == 'WS201-SYNTH-B1'


@pytest.mark.parametrize('syntax_uid', ['1.2.840.10008.1.2', '1.2.840.10008.1.2.1'])
def test_pattern_e_only_imagetype_bytes_change(tmp_path, syntax_uid):
    from pydicom.uid import UID
    from rtpipeline import ct_series_hygiene as hygiene

    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    for idx, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 1
        if idx in (3, 4, 5):
            ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.file_meta.TransferSyntaxUID = UID(syntax_uid)
        ds.save_as(p, enforce_file_format=True)
    originals = {p: p.read_bytes() for p in paths}
    stage = tmp_path / 'staging'
    stage.mkdir()
    hygiene._stage_harmonized_image_type(paths, stage, ['ORIGINAL', 'PRIMARY', 'AXIAL'])
    for p in paths:
        staged = stage / p.name
        ds1 = pydicom.dcmread(p)
        ds2 = pydicom.dcmread(staged)
        raw1 = ds1.get_item(0x00080008)
        raw2 = ds2.get_item(0x00080008)
        off1, end1 = raw1.value_tell, raw1.value_tell + raw1.length
        off2, end2 = raw2.value_tell, raw2.value_tell + raw2.length
        assert originals[p] == p.read_bytes()
        assert originals[p][:off1 - 8] == staged.read_bytes()[:off2 - 8]
        assert originals[p][end1:] == staged.read_bytes()[end2:]
        del ds1.ImageType
        del ds2.ImageType
        assert ds1 == ds2


@pytest.mark.parametrize('orient', [
    [1, 0, 0, 0, 1, 0],
    [-1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, -1, 0, 0],
    [0.8660254038, 0.5, 0, -0.5, 0.8660254038, 0],
])
def test_pattern_e_real_conversion_axis_permutation_and_flips(tmp_path, dcm2niix, orient):
    from rtpipeline import ct_series_hygiene as hygiene

    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    for idx, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 1
        ds.ImageOrientationPatient = orient
        if idx in (3, 4, 5):
            ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.save_as(p, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[1:8])
    out = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs
    )
    assert out is not None
    meta = _evidence(out)
    assert meta['nifti_conversion']['reason_code'] == 'mixed_image_type_harmonized'
    hygiene._verify_slice_by_slice_volume(paths, out)
    provenance = {
        k: meta[k]
        for k in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')
    }
    sidecar = out.with_name(out.name[:-7] + '.metadata.json')
    provenance['sidecar_path'] = str(sidecar.relative_to(tmp_path))
    contract = CourseContract(tmp_path, tmp_path / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
    _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, out, meta['series_instance_uid'])


@pytest.mark.parametrize('index,delta', [(0, 1.0), (5, 1.0), (11, 1.0), (5, 1.0e-7)])
def test_pattern_e_each_slice_and_subunit_voxel_mismatch_refused(tmp_path, dcm2niix, index, delta):
    import SimpleITK as sitk
    from rtpipeline import ct_series_hygiene as hygiene

    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    for idx, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 1
        if idx in (3, 4, 5):
            ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.save_as(p, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[1:8])
    out = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs
    )
    assert out is not None
    image = sitk.DICOMOrient(sitk.ReadImage(str(out)), 'LPS')
    data = sitk.GetArrayFromImage(image).astype(np.float64)
    data[index, 3, 4] += delta
    changed = sitk.GetImageFromArray(data)
    changed.CopyInformation(image)
    target = tmp_path / 'tampered.nii.gz'
    sitk.WriteImage(changed, str(target))
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_mixed_image_type_refused'):
        hygiene._verify_slice_by_slice_volume(paths, target)


@pytest.mark.parametrize('defect', ['origin', 'spacing', 'direction'])
def test_pattern_e_any_nifti_geometry_mismatch_must_be_refused(tmp_path, dcm2niix, defect):
    import SimpleITK as sitk
    from rtpipeline import ct_series_hygiene as hygiene

    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    for idx, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        ds.AcquisitionNumber = 1
        if idx in (3, 4, 5):
            ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.save_as(p, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths[1:8])
    out = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs
    )
    assert out is not None
    image = sitk.DICOMOrient(sitk.ReadImage(str(out)), 'LPS')
    if defect == 'origin':
        origin = list(image.GetOrigin())
        origin[0] += 0.0005
        image.SetOrigin(origin)
    elif defect == 'spacing':
        spacing = list(image.GetSpacing())
        spacing[0] += 0.00005
        image.SetSpacing(spacing)
    else:
        angle = 0.000005
        image.SetDirection([np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1])
    target = tmp_path / 'geometry_mismatch.nii.gz'
    sitk.WriteImage(image, str(target))
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_mixed_image_type_refused'):
        hygiene._verify_slice_by_slice_volume(paths, target)


@pytest.mark.parametrize('mode', [
    'frame_only',
    'contour_only',
    'frame_incomplete',
    'contour_missing_duplicate',
    'both_duplicates',
    'outside',
])
def test_pattern_b_incomplete_dual_authority_refused_in_segmentation_and_organize(
    tmp_path, dcm2niix, mode
):
    from rtpipeline import organize

    for entry in ('segmentation', 'organize'):
        sub = tmp_path / entry
        ct = sub / 'ct'
        paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
        for p in paths:
            ds = pydicom.dcmread(p)
            ds.AcquisitionNumber = 1
            ds.save_as(p, enforce_file_format=True)
        dup_path, _ = _duplicate(paths, differing=True)
        dup_ds = pydicom.dcmread(dup_path)
        dup_ds.AcquisitionNumber = 1
        dup_ds.save_as(dup_path, enforce_file_format=True)
        frame, contour, extra = paths, paths, None
        if mode == 'frame_only':
            contour = []
        elif mode == 'contour_only':
            frame = []
        elif mode == 'frame_incomplete':
            frame = paths[:-1]
        elif mode == 'contour_missing_duplicate':
            contour = [p for p in paths if p != paths[4]]
        elif mode == 'both_duplicates':
            frame = paths + [dup_path]
        elif mode == 'outside':
            extra = [str(generate_uid())]
        rs = _rtstruct_dual(sub / 'RS.dcm', frame, contour, frame_extra=extra)
        cfg = _config(sub, dcm2niix)
        ev = {}
        fn = organize._ensure_planning_ct_nifti if entry == 'organize' else segmentation._ensure_ct_nifti
        out = fn(cfg, ct, sub / 'nifti', authoritative_rtstruct=rs, conversion_evidence=ev, require_hygiene=True)
        assert out is None
        assert ev.get('status') == 'refused'


@pytest.mark.parametrize('field,value', [
    ('ReconstructionDiameter', float('nan')),
    ('WindowCenter', float('nan')),
    ('Exposure', float('nan')),
    ('ReconstructionDiameter', float('inf')),
    ('WindowCenter', [float('nan')]),
    ('WindowCenter', []),
    ('ScanOptions', 123),
    ('ConvolutionKernel', 456),
    ('Exposure', True),
    ('ReconstructionDiameter', 300.0000005),
])
def test_pattern_d_exact_signature_mismatch_and_malformed_values_refused(tmp_path, field, value):
    ct = tmp_path / 'ct'
    rs = tmp_path / 'RS.dcm'
    _build_pattern_d_series(ct, rs, slices=12)
    with pytest.raises(CTSeriesHygieneError) as exc_info:
        select_ct_instances(ct, rs)
    candidates = exc_info.value.evidence['candidate_stacks']
    sharp_sig = copy.deepcopy(next(c['stack_signature'] for c in candidates if c['ConvolutionKernel'] == 'SHARP'))
    sharp_sig[field] = value
    ov = {
        'adjudication_record_id': 'SYNTH-WS210-1',
        'reason': 'Synthetic adjudication for independent verification',
        'stack_signature': sharp_sig,
    }
    with pytest.raises(CTSeriesHygieneError):
        select_ct_instances(ct, rs, stack_override=ov)


@pytest.mark.parametrize('mode', ['mismatch', 'invalid_governance', 'different_valid_stack', 'removed_override',
                                  'different_adjudication', 'different_reason', 'changed_selection_basis'])
def test_pattern_d_cached_nifti_honors_current_override_in_segmentation_and_organize(
    tmp_path, dcm2niix, mode
):
    from rtpipeline import organize

    for entry in ('segmentation', 'organize'):
        sub = tmp_path / entry
        ct = sub / 'ct'
        rs = sub / 'RS.dcm'
        _build_pattern_d_series(ct, rs, slices=12)
        with pytest.raises(CTSeriesHygieneError) as exc_info:
            select_ct_instances(ct, rs)
        candidates = exc_info.value.evidence['candidate_stacks']
        sharp = next(c['stack_signature'] for c in candidates if c['ConvolutionKernel'] == 'SHARP')
        smooth = next(c['stack_signature'] for c in candidates if c['ConvolutionKernel'] == 'SMOOTH')

        def _ov(sig):
            return {
                'adjudication_record_id': 'SYNTH-WS210-1',
                'reason': 'Synthetic adjudication for independent verification',
                'stack_signature': copy.deepcopy(sig),
            }

        cfg = _config(sub, dcm2niix)
        if entry == 'organize':
            cfg.planning_ct_stack_overrides = {'SYNTH/course': _ov(sharp)}
            first = organize._ensure_planning_ct_nifti(
                cfg, ct, sub / 'nifti', authoritative_rtstruct=rs, course_key='SYNTH/course'
            )
        else:
            first = segmentation._ensure_ct_nifti(
                cfg, ct, sub / 'nifti', authoritative_rtstruct=rs, stack_override=_ov(sharp)
            )
        assert first is not None
        assert _evidence(first)['nifti_conversion']['stack_override']['stack_signature'] == sharp

        if mode == 'different_valid_stack':
            next_ov = _ov(smooth)
        elif mode == 'mismatch':
            next_ov = _ov(sharp)
            next_ov['stack_signature']['sop_uids_sha256'] = '0' * 64
        elif mode == 'invalid_governance':
            next_ov = _ov(sharp)
            next_ov['reason'] = ''
        else:
            next_ov = None

        if mode in ('different_adjudication', 'different_reason', 'changed_selection_basis'):
            next_ov = _ov(sharp)
            if mode == 'different_adjudication':
                next_ov['adjudication_record_id'] = 'SYNTH-WS210-2'
            elif mode == 'different_reason':
                next_ov['reason'] = 'A second synthetic adjudication reason'
            else:
                meta = _evidence(first)
                meta['nifti_conversion']['selection_basis'] = 'stale_basis'
                first.with_name(first.name[:-7] + '.metadata.json').write_text(json.dumps(meta))

        ev = {}
        if entry == 'organize':
            cfg.planning_ct_stack_overrides = {'SYNTH/course': next_ov} if next_ov is not None else {}
            second = organize._ensure_planning_ct_nifti(
                cfg,
                ct,
                sub / 'nifti',
                authoritative_rtstruct=rs,
                course_key='SYNTH/course',
                conversion_evidence=ev,
            )
        else:
            second = segmentation._ensure_ct_nifti(
                cfg,
                ct,
                sub / 'nifti',
                authoritative_rtstruct=rs,
                stack_override=next_ov,
                conversion_evidence=ev,
            )

        if mode in ('different_valid_stack', 'different_adjudication', 'different_reason', 'changed_selection_basis'):
            assert second is not None
            meta = _evidence(second)
            assert meta['nifti_conversion']['stack_override'] == next_ov
            assert meta['nifti_conversion']['selection_basis'] == 'governed_stack_override'
            provenance = {
                k: meta[k]
                for k in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')
            }
            sidecar = second.with_name(second.name[:-7] + '.metadata.json')
            provenance['sidecar_path'] = str(sidecar.relative_to(sub))
            contract = CourseContract(sub, sub / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
            _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, second, meta['series_instance_uid'])
        else:
            assert second is None
            assert ev.get('status') == 'refused'


@pytest.mark.parametrize('pattern', ['A', 'B', 'D', 'E'])
def test_truncated_voxel_payload_must_not_be_published(tmp_path, dcm2niix, monkeypatch, pattern):
    import gzip

    ct = tmp_path / 'ct'
    rs = tmp_path / 'RS.dcm'
    ov = None
    if pattern == 'E':
        paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
        for idx, p in enumerate(paths):
            ds = pydicom.dcmread(p)
            ds.AcquisitionNumber = 1
            if idx in (3, 4, 5):
                ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
            ds.save_as(p, enforce_file_format=True)
        _rtstruct_dual(rs, paths, paths[1:8])
    elif pattern == 'D':
        _build_pattern_d_series(ct, rs, slices=12)
        with pytest.raises(CTSeriesHygieneError) as exc_info:
            select_ct_instances(ct, rs)
        sharp = next(c['stack_signature'] for c in exc_info.value.evidence['candidate_stacks'] if c['ConvolutionKernel'] == 'SHARP')
        ov = {
            'adjudication_record_id': 'SYNTH-WS210-1',
            'reason': 'Synthetic adjudication for independent verification',
            'stack_signature': copy.deepcopy(sharp),
        }
    elif pattern == 'B':
        paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
        for p in paths:
            ds = pydicom.dcmread(p)
            ds.AcquisitionNumber = 1
            ds.save_as(p, enforce_file_format=True)
        dup_path, _ = _duplicate(paths, differing=True)
        dup_ds = pydicom.dcmread(dup_path)
        dup_ds.AcquisitionNumber = 2
        dup_ds.save_as(dup_path, enforce_file_format=True)
        _rtstruct_dual(rs, paths, [paths[4]])
    else:
        paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
        for p in paths:
            ds = pydicom.dcmread(p)
            ds.AcquisitionNumber = 1
            ds.save_as(p, enforce_file_format=True)
        stray = pydicom.dcmread(paths[3])
        stray_uid = str(generate_uid())
        stray.SOPInstanceUID = stray_uid
        stray.file_meta.MediaStorageSOPInstanceUID = stray_uid
        stray.InstanceNumber = 99
        stray.AcquisitionNumber = 2
        stray.ImagePositionPatient = [-10.0, -20.0, 17.0]
        stray.save_as(ct / 'stray_slice.dcm', enforce_file_format=True)
        _rtstruct_dual(rs, paths, paths[2:8])

    real_run = segmentation.run_dcm2niix

    def truncate(*args, **kwargs):
        result = real_run(*args, **kwargs)
        if result is not None:
            raw = gzip.decompress(result.read_bytes())
            result.write_bytes(gzip.compress(raw[:352], mtime=0))
            assert nib.load(result).shape == (20, 16, 12)
            with pytest.raises(Exception):
                nib.load(result).get_fdata()
        return result

    monkeypatch.setattr(segmentation, 'run_dcm2niix', truncate)
    ev = {}
    out = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix),
        ct,
        tmp_path / 'nifti',
        authoritative_rtstruct=rs,
        conversion_evidence=ev,
        require_hygiene=True,
        stack_override=ov,
    )
    assert out is None
    assert ev['status'] == 'refused'
    assert ev['reason_code'] == (
        'ct_hygiene_mixed_image_type_refused'
        if pattern == 'E'
        else 'ct_hygiene_unreadable_converter_output'
    )


@pytest.mark.parametrize("reference_both", [False, True])
def test_parent_hygiene_contract_remains_loadable_and_detects_tampering(tmp_path, dcm2niix, reference_both):
    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(12), signed=False)
    dup_path, _ = _duplicate(paths, differing=False)
    rs = _rtstruct_dual(
        tmp_path / 'RS.dcm', paths + ([dup_path] if reference_both else []), paths[:6]
    )
    # Generate the sidecar with the actual release parent, including its selector,
    # converter and provenance writer. No legacy evidence is fabricated here.
    archive = subprocess.run(
        ['git', 'archive', '1aeb0c528ad3d2565d0f4cdd379f318c5c2e4d95', 'rtpipeline'],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True,
    ).stdout
    parent_tree = tmp_path / 'parent'
    parent_tree.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(parent_tree, filter='data')
    script = """
import sys
from pathlib import Path
from rtpipeline.config import PipelineConfig
from rtpipeline.segmentation import _ensure_ct_nifti
root, binary = Path(sys.argv[1]), sys.argv[2]
cfg = PipelineConfig(dicom_root=root/'input', output_root=root/'out', logs_root=root/'logs', dcm2niix_cmd=binary)
result = _ensure_ct_nifti(cfg, root/'ct', root/'nifti', authoritative_rtstruct=root/'RS.dcm')
if result is None:
    raise RuntimeError('parent synthetic conversion failed')
"""
    subprocess.run(
        [sys.executable, '-B', '-c', script, str(tmp_path), str(dcm2niix)],
        cwd=parent_tree,
        env={**os.environ, 'PYTHONPATH': str(parent_tree), 'PYTHONDONTWRITEBYTECODE': '1'},
        check=True, capture_output=True, text=True, timeout=120,
    )
    out = next((tmp_path / 'nifti').glob('*.nii.gz'))
    sidecar = out.with_name(out.name[:-7] + '.metadata.json')
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    assert meta['nifti_conversion']['method'] == METHOD
    assert 'reason_code' not in meta['nifti_conversion']
    assert 'selection_basis' not in meta['nifti_conversion']
    original_bytes = out.read_bytes()
    original_sidecar = sidecar.read_bytes()
    reused = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', authoritative_rtstruct=rs
    )
    assert reused == out
    assert out.read_bytes() == original_bytes
    assert sidecar.read_bytes() == original_sidecar
    provenance = {
        k: meta[k]
        for k in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'nifti_conversion')
    }
    provenance['sidecar_path'] = str(sidecar.relative_to(tmp_path))
    contract = CourseContract(tmp_path, tmp_path / 'metadata.json', {'authoritative_rtstruct': {'path': 'RS.dcm'}})
    _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, out, meta['series_instance_uid'])

    # Also verify single-partition identical duplicate with both copies referenced in RTSTRUCT succeeds like 1aeb0c5
    rs_both = _rtstruct_dual(tmp_path / 'RS_both.dcm', paths + [dup_path], paths[:6])
    kept_both, ev_both = select_ct_instances(ct, rs_both)
    assert len(kept_both) == 12
    assert ev_both['excluded_instance_count'] == 1
    assert ev_both['excluded_instances'][0]['reason_code'] == 'identical_rescaled_duplicate_position'

    # Tampered kept_sop_instance_uids in legacy schema must still fail contract validation
    tampered_meta = copy.deepcopy(meta)
    tampered_meta['nifti_conversion']['kept_sop_instance_uids'] = tampered_meta['nifti_conversion']['kept_sop_instance_uids'][:-1]
    sidecar.write_text(json.dumps(tampered_meta, indent=2), encoding='utf-8')
    tampered_prov = copy.deepcopy(provenance)
    tampered_prov['nifti_conversion'] = tampered_meta['nifti_conversion']
    with pytest.raises(CourseContractError, match='source selection differs'):
        _validate_nifti_provenance(contract, {'nifti_provenance': tampered_prov}, ct, out, meta['series_instance_uid'])




@pytest.mark.parametrize('field', ['ReconstructionDiameter', 'WindowCenter'])
def test_stack_signature_keeps_exact_stored_decimal(tmp_path, field):
    from rtpipeline import ct_series_hygiene as hygiene

    paths = write_ct_series(tmp_path / 'ct', uniform_z_positions(3), signed=False)
    ds = pydicom.dcmread(paths[0])
    # These distinct legal decimal strings round to the same binary64 float.
    exact = '9007199254740993'
    rounded = '9007199254740992'
    setattr(ds, field, exact)
    ds.save_as(paths[0], enforce_file_format=True)
    stored = pydicom.dcmread(paths[0], stop_before_pixels=True)
    value = hygiene._stack_attr_value(stored, field)
    assert value == exact
    attrs = {name: None for name in hygiene.STACK_DISTINGUISHING_ATTRIBUTES}
    attrs[field] = value
    candidate = hygiene.compute_stack_signature(attrs, 3, 'a' * 64)
    requested = copy.deepcopy(candidate)
    requested[field] = rounded
    assert not hygiene._signatures_match(candidate, requested)
    assert hygiene._signatures_match(candidate, candidate)


def test_contract_and_cache_require_complete_hygiene_payload(tmp_path, dcm2niix):
    import gzip

    ct = tmp_path / 'ct'
    paths = write_ct_series(ct, uniform_z_positions(), signed=False)
    _duplicate(paths)
    out = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', require_hygiene=True
    )
    assert out is not None
    original = out.read_bytes()
    out.write_bytes(gzip.compress(gzip.decompress(original)[:352], mtime=0))
    sidecar = out.with_name(out.name[:-7] + '.metadata.json')
    meta = json.loads(sidecar.read_text())
    # Bind the truncated file's hash so that this specifically tests the payload
    # gate, rather than stopping at an ordinary content-hash mismatch.
    meta['nifti_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(meta))
    provenance = {k:meta[k] for k in ('series_instance_uid', 'sop_hash', 'geometry',
                                    'nifti_geometry', 'nifti_sha256', 'nifti_conversion')}
    provenance['sidecar_path'] = str(sidecar.relative_to(tmp_path))
    contract = CourseContract(tmp_path, tmp_path / 'metadata.json', {})
    with pytest.raises(CourseContractError, match='ct_hygiene_unreadable_converter_output'):
        _validate_nifti_provenance(contract, {'nifti_provenance': provenance}, ct, out, meta['series_instance_uid'])
    reused = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), ct, tmp_path / 'nifti', require_hygiene=True
    )
    assert reused == out
    assert reused.read_bytes() == original
    assert nib.load(reused).get_fdata().shape == (20, 16, 12)


@pytest.mark.parametrize('differing', [False, True])
def test_localizer_dual_authority_guard_only_changes_differing_duplicates(tmp_path, differing):
    from rtpipeline.planning_ct_localizers import PlanningCTLocalizerError, select_localizers

    paths = write_ct_series(tmp_path / 'ct', uniform_z_positions(), signed=False)
    _duplicate(paths, differing=differing)
    rs = _rtstruct(tmp_path / 'RS.dcm', paths)
    if differing:
        with pytest.raises(PlanningCTLocalizerError, match='ct_hygiene_duplicate_position_reference_coverage'):
            select_localizers(tmp_path / 'ct', rs)
    else:
        excluded, evidence = select_localizers(tmp_path / 'ct', rs)
        assert len(excluded) == 1
        assert evidence['reason'] == 'identical_rescaled_duplicate_position'
    _rtstruct_dual(rs, paths, [paths[4]])
    excluded, evidence = select_localizers(tmp_path / 'ct', rs)
    assert len(excluded) == 1
    assert evidence['kept_instance_count'] == 12


@pytest.mark.parametrize('spacing,origin_y', [(0.9765625, -14.1234567), (0.7234, -220.1234567)])
def test_pattern_e_float32_source_geometry_representation(tmp_path, dcm2niix, spacing, origin_y):
    from rtpipeline import ct_series_hygiene as hygiene

    paths = write_ct_series(tmp_path / 'ct', uniform_z_positions(), signed=False)
    for idx, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        ds.PixelSpacing = [spacing, spacing]
        ds.ImagePositionPatient = [-50.1234567, origin_y, float(ds.ImagePositionPatient[2]) + 0.1234567]
        ds.AcquisitionNumber = 1
        if idx in (3, 4, 5):
            ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.save_as(p, enforce_file_format=True)
    rs = _rtstruct_dual(tmp_path / 'RS.dcm', paths, paths)
    out = segmentation._ensure_ct_nifti(
        _config(tmp_path, dcm2niix), tmp_path / 'ct', tmp_path / 'nifti', authoritative_rtstruct=rs
    )
    assert out is not None
    hygiene._verify_slice_by_slice_volume(paths, out)
    # A real geometry change still refuses at this non-integer origin.
    image = nib.load(out)
    changed_affine = image.affine.copy()
    changed_affine[0, 3] += 0.0005
    altered = out.parent / 'shifted.nii.gz'
    header = image.header.copy()
    header.set_sform(changed_affine, code=1)
    header.set_qform(changed_affine, code=1)
    nib.save(nib.Nifti1Image(image.get_fdata(), changed_affine, header), altered)
    with pytest.raises(CTSeriesHygieneError, match='ct_hygiene_mixed_image_type_refused'):
        hygiene._verify_slice_by_slice_volume(paths, altered)
