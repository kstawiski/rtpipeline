"""DICOM/NIfTI boundaries preserve asymmetric voxels in physical LPS space."""
import copy
import json
import shutil
import subprocess

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from rt_utils import RTStructBuilder

from robustness_prep_fixture import write_ct
from rtpipeline import organize
from rtpipeline.rtstruct_geometry import (
    image_array_for_rtstruct, rtstruct_mask_to_image, create_scoped_rtstruct,
    place_added_rois_on_planes,
)


@pytest.fixture(params=[False, True], ids=['axial', 'oblique'])
def source(tmp_path, request):
    ct = tmp_path / 'CT'
    write_ct(ct, rows=23, columns=23, slices=7, spacing=(1.2, 1.7, 2.3))
    if request.param:
        angle = 0.37
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        for path in ct.glob('*.dcm'):
            ds = pydicom.dcmread(path)
            ds.ImageOrientationPatient = [*rotation[:, 0], *rotation[:, 1]]
            ds.ImagePositionPatient = list(rotation @ ds.ImagePositionPatient)
            ds.save_as(path, enforce_file_format=True)
    rt = RTStructBuilder.create_new(str(ct))
    mask = np.zeros((23, 23, 7), dtype=bool)
    mask[2:7, 3:8, 1:4] = True
    mask[5:9, 3:5, 1:4] = True
    rt.add_roi(mask=mask, name='Asymmetric')
    place_added_rois_on_planes(rt.ds, rt.series_data)
    rs = tmp_path / 'RS.dcm'
    rt.save(str(rs))
    rt = create_scoped_rtstruct(ct, rs)
    np.testing.assert_array_equal(rt.get_roi_mask_by_name('Asymmetric'), mask)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct)))
    return ct, rs, rt, reader.Execute()


def physical_voxels(image):
    zyx = np.argwhere(sitk.GetArrayFromImage(image) > 0)
    points = np.array([image.TransformIndexToPhysicalPoint(tuple(map(int, p[::-1]))) for p in zyx])
    return {tuple(p) for p in np.round(points, 4)}, points.mean(axis=0)


def dicom_voxels(mask, slices):
    points = []
    for row, col, k in np.argwhere(mask):
        ds = slices[k]
        o = np.asarray(ds.ImageOrientationPatient).reshape(2, 3)
        points.append(np.asarray(ds.ImagePositionPatient) + row * float(ds.PixelSpacing[0]) * o[1]
                      + col * float(ds.PixelSpacing[1]) * o[0])
    points = np.asarray(points)
    return {tuple(p) for p in np.round(points, 4)}, points.mean(axis=0)


@pytest.mark.parametrize('perm,flip', [([0, 1, 2], [False]*3),
    ([0, 1, 2], [False, True, False]), ([2, 0, 1], [True, False, True])])
def test_export_voxels_and_centroid_in_lps(source, tmp_path, perm, flip):
    ct, rs, rt, ref = source
    ref = sitk.Flip(sitk.PermuteAxes(ref, perm), flip)
    nifti = tmp_path / 'CT_reference.nii.gz'
    sitk.WriteImage(ref, str(nifti))
    manifest = organize._export_original_segmentation_from_paths(
        rs_path=rs, primary_nifti=nifti, dicom_ct_dir=ct,
        segmentation_original_dir=tmp_path/'Segmentation_Original', log_root=tmp_path, overwrite=False)
    assert manifest['geometry_status'] == 'exact_physical_mapping'
    image = sitk.ReadImage(str(tmp_path/'Segmentation_Original'/manifest['structures'][0]['mask']))
    expected, center = dicom_voxels(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data)
    actual, observed = physical_voxels(image)
    assert actual == expected
    np.testing.assert_allclose(observed, center, atol=1e-5)
    np.testing.assert_array_equal(image_array_for_rtstruct(image, rt.series_data)>0,
                                  rt.get_roi_mask_by_name('Asymmetric'))


@pytest.mark.parametrize('defect', ['shift', 'spacing', 'irregular', 'orientation', 'shape', 'missing'])
def test_mapping_fails_closed(source, defect):
    _, _, rt, ref = source
    slices = copy.deepcopy(rt.series_data)
    mask = rt.get_roi_mask_by_name('Asymmetric')
    if defect == 'shift':
        ref.SetOrigin(tuple(np.asarray(ref.GetOrigin()) + [0.1, 0, 0]))
    elif defect == 'spacing':
        ref.SetSpacing((1.3, 1.7, 2.3))
    elif defect == 'irregular':
        slices[3].ImagePositionPatient[2] += 0.2
    elif defect == 'orientation':
        slices[3].ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
    elif defect == 'shape':
        mask = mask[:-1]
    else:
        slices = []
    with pytest.raises(ValueError, match='MASK_GRID_NOT_EQUIVALENT'):
        rtstruct_mask_to_image(mask, slices, ref)


def test_stale_export_is_removed_on_geometry_failure(source, tmp_path):
    ct, rs, _, ref = source
    nifti = tmp_path/'CT_reference.nii.gz'
    ref.SetOrigin(tuple(np.asarray(ref.GetOrigin()) + [0.2, 0, 0]))
    sitk.WriteImage(ref, str(nifti))
    target = tmp_path/'Segmentation_Original/CT_reference'
    target.mkdir(parents=True)
    (target/'obsolete.nii.gz').write_bytes(b'old')
    (target/'metadata.json').write_text('{}')
    result = organize._export_original_segmentation_from_paths(rs_path=rs, primary_nifti=nifti,
        dicom_ct_dir=ct, segmentation_original_dir=target.parent, log_root=tmp_path, overwrite=False)
    assert result is None
    assert not list(target.glob('*.nii*'))
    assert json.loads((target/'metadata.json').read_text())['geometry_status'] == 'MASK_GRID_NOT_EQUIVALENT'
    assert not (target/'provenance.json').exists()


@pytest.mark.integration
def test_real_dcm2niix(source, tmp_path):
    executable = shutil.which('dcm2niix') or '/home/konrad/micromamba/envs/rtpipeline/bin/dcm2niix'
    from pathlib import Path
    if not Path(executable).is_file():
        pytest.skip('dcm2niix unavailable')
    ct, _, rt, _ = source
    out = tmp_path/'converted'
    out.mkdir()
    subprocess.run([executable, '-z', 'y', '-f', 'reference', '-o', str(out), str(ct)],
                   check=True, capture_output=True)
    ref = sitk.ReadImage(str(next(out.glob('*.nii.gz'))))
    image = rtstruct_mask_to_image(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data, ref)
    expected, center = dicom_voxels(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data)
    actual, observed = physical_voxels(image)
    assert actual == expected
    np.testing.assert_allclose(observed, center, atol=1e-5)


def test_legacy_export_is_recomputed_and_stage_receipt_rejects_it(source, tmp_path):
    from rtpipeline.stage_completion import _STAGE_DEFINITIONS, _validate_stage_specific_outcome

    ct, rs, rt, ref = source
    ref = sitk.Flip(ref, [False, True, False])
    nifti = tmp_path/'CT_reference.nii.gz'
    sitk.WriteImage(ref, str(nifti))
    (tmp_path/'CT_reference.metadata.json').write_text(json.dumps({
        'series_instance_uid': 'synthetic', 'sop_hash': 'synthetic',
        'nifti_geometry': {'size': list(ref.GetSize())}}))
    target = tmp_path/'Segmentation_Original/CT_reference'
    target.mkdir(parents=True)
    mask = rt.get_roi_mask_by_name('Asymmetric')
    old = sitk.GetImageFromArray(np.moveaxis(mask.astype(np.uint8), -1, 0))
    old.CopyInformation(ref)
    sitk.WriteImage(old, str(target/'Asymmetric.nii.gz'))
    (target/'metadata.json').write_text(json.dumps({'structures': [{'mask': 'CT_reference/Asymmetric.nii.gz'}]}))
    provenance = organize._original_segmentation_provenance(rs, nifti)
    provenance['schema'] = 'rtpipeline-original-segmentation-provenance-v1'
    (target/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='stale'):
        _validate_stage_specific_outcome(tmp_path, _STAGE_DEFINITIONS['organize'], 'ok', [])
    result = organize._export_original_segmentation_from_paths(rs_path=rs, primary_nifti=nifti,
        dicom_ct_dir=ct, segmentation_original_dir=target.parent, log_root=tmp_path, overwrite=False)
    new = sitk.ReadImage(str(target/'Asymmetric.nii.gz'))
    assert physical_voxels(new)[0] == dicom_voxels(mask, rt.series_data)[0]
    _validate_stage_specific_outcome(tmp_path, _STAGE_DEFINITIONS['organize'], 'ok', [])
    assert organize._original_masks_reusable(target, target.parent, result,
        organize._original_segmentation_provenance(rs, nifti))[0]


def test_rectangular_reader_mask_is_rejected(tmp_path):
    ct = tmp_path/'CT'
    write_ct(ct, rows=19, columns=23, slices=7)
    rt = RTStructBuilder.create_new(str(ct))
    mask = np.zeros((19, 23, 7), bool)
    mask[2:5, 3:7, 1:3] = True
    rt.add_roi(mask=mask, name='Asymmetric')
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct)))
    ref = reader.Execute()
    # The helper itself supports rectangular grids; rt-utils 1.2.7's reader
    # swaps the allocation dimensions and must never be guessed back into place.
    actual = rtstruct_mask_to_image(mask, rt.series_data, ref)
    assert physical_voxels(actual)[0] == dicom_voxels(mask, rt.series_data)[0]
    with pytest.raises(ValueError, match='MASK_GRID_NOT_EQUIVALENT'):
        rtstruct_mask_to_image(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data, ref)


def test_all_axis_permutations_and_flips(source):
    from itertools import permutations, product
    _, _, rt, native = source
    mask = rt.get_roi_mask_by_name('Asymmetric')
    expected, center = dicom_voxels(mask, rt.series_data)
    for permutation in permutations(range(3)):
        for flips in product((False, True), repeat=3):
            reference = sitk.Flip(sitk.PermuteAxes(native, permutation), flips)
            image = rtstruct_mask_to_image(mask, rt.series_data, reference)
            actual, observed = physical_voxels(image)
            assert actual == expected, (permutation, flips)
            np.testing.assert_allclose(observed, center, atol=1e-5)
    # The mapping follows the supplied rt-utils slice order, not a z guess.
    reversed_image = rtstruct_mask_to_image(mask[:, :, ::-1], rt.series_data[::-1], native)
    assert physical_voxels(reversed_image)[0] == expected
