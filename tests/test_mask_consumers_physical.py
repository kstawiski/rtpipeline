"""Exercise mask consumers with asymmetric physical geometry."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from rt_utils import RTStructBuilder

from test_mask_physical_space import source, dicom_voxels, physical_voxels  # noqa: F401
from rtpipeline.rtstruct_geometry import rtstruct_mask_to_image, create_scoped_rtstruct


def mask_image(rt, ref):
    return rtstruct_mask_to_image(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data, ref)


def assert_physical(image, rt):
    expected, center = dicom_voxels(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data)
    actual, observed = physical_voxels(image)
    assert actual == expected
    np.testing.assert_allclose(observed, center, atol=1e-5)


def test_radiomics_reader_and_nifti_robustness(source, rasterizer_guard_stub):
    from rtpipeline.radiomics import _mask_from_array_like, _rtstruct_masks
    from rtpipeline.radiomics_robustness import _nifti_mask_in_reference_order, _ImageGeometry

    ct, rs, rt, native = source
    masks = _rtstruct_masks(ct, rs)
    assert_physical(_mask_from_array_like(native, masks['Asymmetric']), rt)
    flipped = sitk.Flip(native, [False, True, False])
    assert_physical(_mask_from_array_like(flipped, rt.get_roi_mask_by_name('Asymmetric'),
                                         series_data=rt.series_data), rt)
    # Same dimensions but different direction/origin; both NIfTI-only and
    # DICOM-backed robustness use this production loader and preparation order.
    image = mask_image(rt, flipped)
    array = _nifti_mask_in_reference_order(image, native)
    geometry = _ImageGeometry(native.GetSize(), native.GetSpacing(), native.GetDirection(), native.GetOrigin())
    assert_physical(_mask_from_array_like(geometry, array), rt)
    array = _nifti_mask_in_reference_order(mask_image(rt, native), flipped)
    assert_physical(_mask_from_array_like(flipped, array), rt)


def test_cubic_layout_is_explicit():
    from rtpipeline.radiomics import _mask_from_array_like
    ref = sitk.Image(5, 5, 5, sitk.sitkUInt8)
    mask = np.zeros((5, 5, 5), bool)
    mask[1, 3, 2] = True  # row=1, column=3, slice=2
    image = _mask_from_array_like(ref, mask)
    assert physical_voxels(image)[0] == {(3., 1., 2.)}
    zyx = np.moveaxis(mask, -1, 0)
    assert physical_voxels(_mask_from_array_like(ref, zyx, array_order='zyx'))[0] == {(3., 1., 2.)}


def test_mr_nifti_fallback(source, tmp_path):
    from rtpipeline.radiomics import _collect_total_mr_masks, _mask_from_array_like
    ct, _, rt, native = source
    seg = tmp_path/'mr_masks'
    seg.mkdir()
    sitk.WriteImage(mask_image(rt, sitk.Flip(native, [False, True, False])),
                    str(seg/'total_mr--Asymmetric.nii.gz'))
    backend = {}
    masks = _collect_total_mr_masks(ct, seg, backend_out=backend)
    assert backend['backend'] == 'nifti/no-polygon-fill'
    assert_physical(_mask_from_array_like(native, masks['Asymmetric']), rt)


def test_conda_mask_serialization(source, tmp_path):
    from rtpipeline.radiomics_conda import _write_mask_to_file
    ct, _, rt, native = source
    path = tmp_path/'mask.nrrd'
    _write_mask_to_file(rt.get_roi_mask_by_name('Asymmetric'), str(path),
        dict(spacing=native.GetSpacing(), origin=native.GetOrigin(), direction=native.GetDirection()))
    assert_physical(sitk.ReadImage(str(path)), rt)


def test_custom_anchor_and_dice_use_physical_coordinates(source, tmp_path):
    from rtpipeline.segmenters.postprocessing import derive_anchor_from_reference
    from rtpipeline.eval.geometry import dice
    _, _, rt, native = source
    target = tmp_path/'Segmentation_Original/CT_reference'
    target.mkdir(parents=True)
    image = mask_image(rt, sitk.Flip(native, [False, True, False]))
    path = target/'GTV-1vis-1.nii.gz'
    sitk.WriteImage(image, str(path))
    with pytest.raises(ValueError, match='stale'):
        derive_anchor_from_reference(str(path))
    from test_s9c_repairs import bind_export
    bind_export(tmp_path, [path])
    anchor = derive_anchor_from_reference(str(path))
    center = anchor['affine'] @ np.array([*anchor['centroid_zyx'], 1])
    np.testing.assert_allclose(center[:3], dicom_voxels(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data)[1], atol=1e-5)
    assert dice(image, mask_image(rt, native)) == pytest.approx(1.)


def test_cropped_and_custom_rtstruct_writers(source, tmp_path, monkeypatch):
    from rtpipeline import anatomical_cropping as ac, custom_models as cm
    from rtpipeline.layout import build_course_dirs
    ct, _, rt, native = source
    monkeypatch.setattr(ac, 'load_course_contract', lambda _:SimpleNamespace(planning_ct_dir=ct))
    dirs = build_course_dirs(tmp_path)
    seg = dirs.segmentation_totalseg/'CT_reference'
    seg.mkdir(parents=True)
    full = mask_image(rt, sitk.Flip(native, [False, True, False]))
    cropped = full[1:20, 1:22, 1:6]
    path = seg/'total--Asymmetric_cropped.nii.gz'
    sitk.WriteImage(cropped, str(path))
    result = ac._create_rtstruct_from_cropped_masks(tmp_path, dirs, '_cropped')
    assert result is not None
    actual = create_scoped_rtstruct(ct, result).get_roi_mask_by_name('Asymmetric')
    np.testing.assert_array_equal(actual, rt.get_roi_mask_by_name('Asymmetric'))
    out = tmp_path/'custom_model'
    out.mkdir()
    result = cm._build_rtstruct(ct, {'Asymmetric':full}, out, {'Asymmetric':path})
    actual = create_scoped_rtstruct(ct, result).get_roi_mask_by_name('Asymmetric')
    np.testing.assert_array_equal(actual, rt.get_roi_mask_by_name('Asymmetric'))


def test_landmarks_and_direction_aware_crop(source):
    from rtpipeline.anatomical_cropping import _mask_z_extent, crop_image_to_boundaries
    _, _, rt, native = source
    full = mask_image(rt, sitk.Flip(native, [False, True, True]))
    expected, _ = dicom_voxels(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data)
    high, low = _mask_z_extent(full, sitk.GetArrayFromImage(full))
    assert high == pytest.approx(max(p[2] for p in expected), abs=1e-4)
    assert low == pytest.approx(min(p[2] for p in expected), abs=1e-4)
    if abs(native.GetDirection()[6]) > 1e-6:
        with pytest.raises(ValueError, match='axis-aligned'):
            crop_image_to_boundaries(full, high, low)
    else:
        for image in [full, sitk.PermuteAxes(full, [2, 0, 1])]:
            cropped, _ = crop_image_to_boundaries(image, high, low)
            assert_physical(cropped, rt)


def test_rs_auto_and_dvh_nifti_boundary(source):
    from rtpipeline.auto_rtstruct import _image_array_for_rtstruct_builder
    from rtpipeline.dvh import _resample_mask_to_ct_if_equivalent
    _, _, rt, native = source
    image = mask_image(rt, sitk.Flip(native, [False, True, False]))
    np.testing.assert_array_equal(_image_array_for_rtstruct_builder(image, native, rt)>0,
                                  rt.get_roi_mask_by_name('Asymmetric'))
    assert_physical(_resample_mask_to_ct_if_equivalent(image, native), rt)


@pytest.mark.parametrize('source_name', ['manual', 'auto', 'custom'])
def test_axial_review_overlay(source, tmp_path, monkeypatch, source_name):
    import shutil
    import re
    from rtpipeline import visualize
    ct, rs, rt, native = source
    contract_rs = rs if source_name == 'manual' else None
    if source_name != 'manual':
        shutil.copy2(rs, tmp_path/f'RS_{source_name}.dcm')
    flipped = sitk.Flip(native, [False, True, False])
    monkeypatch.setattr(visualize, 'load_course_contract', lambda _:SimpleNamespace(
        planning_ct_dir=ct, authoritative_rtstruct_path=contract_rs))
    monkeypatch.setattr(visualize, '_load_ct_sitk', lambda _:flipped)
    monkeypatch.setattr(visualize, '_png_b64', lambda _: 'CT')
    planes = []
    def capture(mask, color):
        planes.append(mask.copy())
        return str(len(planes)-1)
    monkeypatch.setattr(visualize, '_rgba_mask_b64', capture)
    result = visualize.generate_axial_review(tmp_path)
    assert result is not None
    overlays = json.loads(re.search(r'const OVERLAYS_'+source_name.upper()+r' = (.*);', result.read_text())[1])
    array = np.zeros(tuple(reversed(flipped.GetSize())), np.uint8)
    for z, index in overlays['Asymmetric'].items():
        array[int(z)] = planes[int(index)]
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(flipped)
    assert_physical(image, rt)


def test_qc_cubic_dicom_axes_are_unambiguous(tmp_path):
    from robustness_prep_fixture import write_ct
    from rtpipeline.quality_control import DICOMValidator
    ct = tmp_path/'CT'
    write_ct(ct, rows=7, columns=7, slices=7)
    rt = RTStructBuilder.create_new(str(ct))
    mask = np.zeros((7, 7, 7), bool)
    mask[2:5, 2:5, 5:] = True
    rt.add_roi(mask=mask, name='Asymmetric')
    rs = tmp_path/'RS.dcm'
    rt.save(str(rs))
    validator = DICOMValidator.__new__(DICOMValidator)
    validator.course_dir, validator.ct_dir = tmp_path, ct
    validator.rs_path, validator.rs_auto_path = rs, tmp_path/'absent.dcm'
    result = validator._segmentation_cropping()
    assert result['status'] == 'INFO'
    assert result['structures'][0]['cropped_sides'] == ['z_max']
    assert not result['structures'][0]['inplane_cropped']
    # The terminal slice is physically superior in this synthetic DICOM series.
    _, center = dicom_voxels(rt.get_roi_mask_by_name('Asymmetric'), rt.series_data)
    assert center[2] == pytest.approx(16.5)


def test_custom_selection_and_medsam_anchor_cross_grid(source, tmp_path):
    from rtpipeline.segmenters.postprocessing import _zyx_affine_from_sitk, anchor_in_reference_geometry
    from rtpipeline.segmenters.medsam_boxprompt import _prompt_from_anchor_bbox
    from rtpipeline.custom_models import _postprocess_structures, NetworkDefinition
    _, _, rt, native = source
    flipped = mask_image(rt, sitk.Flip(native, [False, True, False]))
    indices = np.argwhere(sitk.GetArrayFromImage(flipped)>0)
    anchor = {'centroid_zyx':indices.mean(axis=0), 'bbox_zyx': [indices.min(axis=0), indices.max(axis=0)],
              'affine':_zyx_affine_from_sitk(flipped)}
    image = mask_image(rt, native)
    network = NetworkDefinition(network_id='test', alias='test', label_names=['Asymmetric'],
        archive_path=None, checkpoint_path=None, dataset_dir='',
        anchor_source='specialist_reference', anchor_max_distance_cm=0.01, min_component_volume_cm3=0.)
    result, _ = _postprocess_structures({'Asymmetric':image}, tmp_path, network, anchor, {})
    assert_physical(result['Asymmetric'], rt)
    local = anchor_in_reference_geometry(anchor, native)
    prompt = _prompt_from_anchor_bbox(local, tuple(reversed(native.GetSize())))
    native_indices = np.argwhere(sitk.GetArrayFromImage(image)>0)
    np.testing.assert_array_equal(np.argwhere(prompt).min(axis=0), native_indices.min(axis=0))
    np.testing.assert_array_equal(np.argwhere(prompt).max(axis=0), native_indices.max(axis=0))


def test_model_rtstruct_rebuild_from_flipped_nifti(source, tmp_path):
    from rtpipeline.segmentation import _ensure_model_rtstruct_from_masks
    ct, _, rt, native = source
    seg = tmp_path/'derived_model'
    seg.mkdir()
    image = mask_image(rt, sitk.Flip(native, [False, True, False]))
    sitk.WriteImage(image, str(seg/'total--asymmetric.nii.gz'))
    result = _ensure_model_rtstruct_from_masks(ct, seg, 'CT_reference', 'total')
    assert result is not None
    builder = create_scoped_rtstruct(ct, result)
    actual = builder.get_roi_mask_by_name(builder.get_roi_names()[0])
    np.testing.assert_array_equal(actual, rt.get_roi_mask_by_name('Asymmetric'))


def test_comparison_rejects_legacy_original_path(source, tmp_path):
    from rtpipeline.eval.geometry import compare_masks
    _, _, rt, native = source
    image = mask_image(rt, native)
    target = tmp_path/'Segmentation_Original/CT_reference'
    target.mkdir(parents=True)
    path = target/'Asymmetric.nii.gz'
    sitk.WriteImage(image, str(path))
    result = compare_masks(path, image)
    assert result.status == 'INCOMPATIBLE'
    assert result.dice is None
    assert 'stale' in result.reason


def test_figure_overlay_aligns_equal_sized_flipped_grid(source):
    from rtpipeline.visualize import _overlay_fig
    import matplotlib.pyplot as plt
    _, _, rt, native = source
    flipped = mask_image(rt, sitk.Flip(native, [False, True, False]))
    expected = sitk.GetArrayFromImage(mask_image(rt, native))
    figure = _overlay_fig(native, flipped, 'synthetic')
    try:
        actual = np.ma.filled(figure.axes[0].images[1].get_array(), 0)
        np.testing.assert_array_equal(actual, expected[native.GetSize()[2]//2])
    finally:
        plt.close(figure)


@pytest.mark.parametrize('columns,spacing', [(20, (1., 1., 2.)), (24, (1.2, 1.7, 2.)),
                                             (20, (1.2, 1.7, 2.))])
def test_rectangular_and_unequal_spacing_readers_preserve_dicom_voxels(tmp_path, columns, spacing):
    from robustness_prep_fixture import write_ct
    from rtpipeline.rtstruct_geometry import rtstruct_roi_mask, place_added_rois_on_planes
    from rtpipeline.radiomics import _rt_utils_indexed_roi_mask
    from rtpipeline.roi_fixer import _rt_utils_roi_mask
    from rtpipeline.organize import _export_original_segmentation_from_paths

    ct = tmp_path/'CT'
    write_ct(ct, rows=24, columns=columns, slices=8, spacing=spacing)
    builder = RTStructBuilder.create_new(str(ct))
    expected = np.zeros((24, columns, 8), bool)
    expected[3:9, 12:17, 2:5] = True
    builder.add_roi(mask=expected, name='Asymmetric')
    place_added_rois_on_planes(builder.ds, builder.series_data)
    rs = tmp_path/'RS.dcm'
    builder.save(str(rs))
    for actual in (rtstruct_roi_mask(builder, 'Asymmetric'),
                   _rt_utils_indexed_roi_mask(builder, 'Asymmetric', retain=True),
                   _rt_utils_roi_mask(builder, 'Asymmetric')):
        np.testing.assert_array_equal(actual, expected)
    assert _rt_utils_indexed_roi_mask(builder, 'Asymmetric', retain=False)
    native = sitk.ImageSeriesReader()
    native.SetFileNames(native.GetGDCMSeriesFileNames(str(ct)))
    reference = sitk.Flip(native.Execute(), [False, True, False])
    path = tmp_path/'reference.nii.gz'
    sitk.WriteImage(reference, str(path))
    from test_s9c_repairs import write_sidecar
    write_sidecar(path)
    manifest = _export_original_segmentation_from_paths(rs_path=rs, primary_nifti=path,
        dicom_ct_dir=ct, segmentation_original_dir=tmp_path/'Segmentation_Original',
        log_root=tmp_path, overwrite=False)
    image = sitk.ReadImage(str(tmp_path/'Segmentation_Original'/manifest['structures'][0]['mask']))
    expected_points, expected_center = dicom_voxels(expected, builder.series_data)
    actual_points, actual_center = physical_voxels(image)
    assert actual_points == expected_points
    np.testing.assert_allclose(actual_center, expected_center, atol=1e-5)
