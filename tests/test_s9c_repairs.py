"""Synthetic regressions for the S9c geometry and export contract."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline import organize
from rtpipeline.original_segmentation import require_current_original_mask
from rtpipeline.rtstruct_geometry import rtstruct_mask_to_image, rtstruct_contours_to_mask
from test_nifti_nonuniform_fallback import dcm2niix  # noqa: F401


def bind_export(root, paths):
    """Write a complete contract with content identities for synthetic inputs."""
    root.mkdir(parents=True, exist_ok=True)
    rs = root / 'RS.dcm'
    rs.write_bytes(b'synthetic RTSTRUCT')
    nifti = root / 'CT.nii.gz'
    nifti.write_bytes(b'synthetic NIfTI')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    current = dict(series_instance_uid='1.2.3', sop_hash='synthetic-sops',
                   nifti_geometry={'size': [8, 8, 1]}, nifti_sha256=digest(nifti))
    (root / 'CT.metadata.json').write_text(json.dumps(current))
    metadata_dir = root / 'metadata'
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / 'case_metadata.json').write_text(json.dumps({'course_contract': {
        'planning_ct': {'nifti_provenance': current, 'nifti': 'CT.nii.gz'},
        'authoritative_rtstruct': {'path': 'RS.dcm'}}}))
    provenance = organize._original_segmentation_provenance(rs, nifti)
    for parent in {p.parent for p in paths}:
        (parent / 'metadata.json').write_text(json.dumps({
            'schema': organize.ORIGINAL_SEGMENTATION_SCHEMA,
            'geometry_status': 'exact_physical_mapping',
            'source_rtstruct': str(rs), 'source_nifti': str(nifti),
            'structures': [{'mask': f'{p.parent.name}/{p.name}', 'sha256': digest(p),
                            'size_bytes': p.stat().st_size} for p in paths if p.parent == parent]}))
        (parent / 'provenance.json').write_text(json.dumps(provenance))


def slices(rows=8, columns=8, count=1, spacing=(1., 1.), thickness=2.):
    return [SimpleNamespace(Rows=rows, Columns=columns, PixelSpacing=spacing,
        ImageOrientationPatient=[1., 0., 0., 0., 1., 0.],
        ImagePositionPatient=[0., 0., k*thickness], SliceThickness=thickness,
        SOPInstanceUID=f'1.2.3.{k+1}') for k in range(count)]


@pytest.mark.parametrize('spacing', [3.5, 1.5])
def test_f2_single_slice_rejects_wrong_step(spacing):
    ref = sitk.Image([8, 8, 1], sitk.sitkUInt8)
    ref.SetSpacing((1., 1., spacing))
    with pytest.raises(ValueError, match='MASK_GRID_NOT_EQUIVALENT'):
        rtstruct_mask_to_image(np.ones((8, 8, 1)), slices(), ref)


def test_f2_single_slice_rejects_tilt():
    ref = sitk.Image([8, 8, 1], sitk.sitkUInt8)
    ref.SetSpacing((1., 1., 2.))
    ref.SetDirection((1., 0., .17, 0., 1., 0., 0., 0., .985))
    with pytest.raises(ValueError, match='MASK_GRID_NOT_EQUIVALENT'):
        rtstruct_mask_to_image(np.ones((8, 8, 1)), slices(), ref)


def test_f2_single_slice_accepts_gdcm_unit_step():
    ref = sitk.Image([8, 8, 1], sitk.sitkUInt8)
    assert rtstruct_mask_to_image(np.ones((8, 8, 1)), slices(thickness=5.), ref).GetSpacing() == (1., 1., 1.)


@pytest.mark.parametrize('decimals', [5, 6])
def test_f2_rounded_oblique_512_grid(decimals):
    x, y = .25, .37
    rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rotation = ry @ rx
    series = slices(rows=512, columns=512, count=3, spacing=(.931,.931), thickness=3.)
    for k, ds in enumerate(series):
        ds.ImageOrientationPatient = np.round(rotation[:, :2].T, decimals).ravel().tolist()
        ds.ImagePositionPatient = (rotation[:,2]*k*3.).tolist()
    ref = sitk.Image([512,512,3], sitk.sitkUInt8)
    ref.SetSpacing((.931,.931,3.))
    ref.SetDirection(rotation.ravel())
    out = rtstruct_mask_to_image(np.ones((512,512,3)), series, ref)
    assert sitk.GetArrayFromImage(out).sum() == 512*512*3
    ref.SetOrigin((.02,0.,0.))
    with pytest.raises(ValueError, match='MASK_GRID_NOT_EQUIVALENT'):
        rtstruct_mask_to_image(np.ones((512,512,3)), series, ref)


@pytest.mark.parametrize('data', [[], [1,2,0], [1,2,0,3,4,0], [1,2]])
def test_f8_empty_short_contours(monkeypatch, data):
    from rt_utils import image_helper
    monkeypatch.setattr(image_helper, 'get_slice_contour_data', lambda *_: [data])
    assert not rtstruct_contours_to_mask(slices(columns=9), []).any()


@pytest.mark.parametrize('mutation', ['extra', 'bytes', 'missing_contract', 'source', 'geometry_status'])
def test_f4_mask_is_bound(tmp_path, mutation):
    parent = tmp_path / 'Segmentation_Original/CT_1'
    parent.mkdir(parents=True)
    path = parent / 'GTV-1vis-1.nii.gz'
    path.write_bytes(b'mask')
    bind_export(tmp_path, [path])
    require_current_original_mask(path)
    match = 'stale'
    if mutation == 'extra':
        path = parent / 'extra.nii.gz'
        path.write_bytes(b'mask')
    elif mutation == 'bytes':
        path.write_bytes(b'changed')
    elif mutation == 'missing_contract':
        (parent / 'provenance.json').write_text(json.dumps({'schema': organize.ORIGINAL_PROVENANCE_SCHEMA}))
        (tmp_path / 'metadata/case_metadata.json').unlink()
    elif mutation == 'source':
        (tmp_path / 'CT.nii.gz').write_bytes(b'changed')
    else:
        meta = json.loads((parent/'metadata.json').read_text())
        meta['geometry_status'] = 'MASK_GRID_NOT_EQUIVALENT'
        (parent/'metadata.json').write_text(json.dumps(meta))
        (parent/'provenance.json').unlink()
        match = 'MASK_GRID_NOT_EQUIVALENT'
    with pytest.raises(ValueError, match=match):
        require_current_original_mask(path)


def test_f1_f3_producer_quarantines_siblings_before_reuse(tmp_path, monkeypatch):
    from test_d1_all_series_original_segmentation import _patch_manual_export_dependencies
    from rtpipeline.segmenters.postprocessing import _resolve_reference_paths
    from rtpipeline.stage_completion import _validate_stage_specific_outcome, stage_definition
    calls = []
    _patch_manual_export_dependencies(monkeypatch, calls)
    parent = tmp_path / 'Segmentation_Original/CT_1'
    parent.mkdir(parents=True)
    mask = parent / 'GTV-1vis-1.nii.gz'
    mask.write_bytes(b'mask')
    bind_export(tmp_path, [mask])
    import rt_utils
    monkeypatch.setattr(rt_utils.RTStructBuilder, 'get_roi_names', lambda _: ['GTV-1vis-1'])
    (tmp_path/'CT.nii.gz').rename(tmp_path/'CT_new.nii.gz')
    (tmp_path/'CT.metadata.json').rename(tmp_path/'CT_new.metadata.json')
    contract_path=tmp_path/'metadata/case_metadata.json'
    contract=json.loads(contract_path.read_text())
    contract['course_contract']['planning_ct']['nifti']='CT_new.nii.gz'
    contract_path.write_text(json.dumps(contract))
    require_current_original_mask(mask)
    old = parent.parent / 'CT_old'
    old.mkdir()
    (old / mask.name).write_bytes(b'stale')
    organize._export_original_segmentation_from_paths(rs_path=tmp_path/'RS.dcm',
        primary_nifti=tmp_path/'CT_new.nii.gz', dicom_ct_dir=tmp_path/'DICOM',
        segmentation_original_dir=parent.parent, log_root=tmp_path, overwrite=False)
    _validate_stage_specific_outcome(tmp_path, stage_definition('segmentation'), 'ok', [])
    assert not old.exists()
    assert not parent.exists()  # A valid old stem is superseded too.
    assert _resolve_reference_paths(str(parent.parent/'CT_*/GTV-1*-*.nii.gz')) == [parent.parent/'CT_new/GTV-1vis-1.nii.gz']
    assert not _resolve_reference_paths(str(parent.parent/'CT_old/*.nii.gz'))
    records = list((tmp_path/'Segmentation_Quarantine').rglob('record.json'))
    assert records
    record = json.loads(records[0].read_text())
    assert record['reason'] and record['time'] and record['path'].startswith('Segmentation_Original/')


def test_f9_export_without_sidecar_fails_closed(tmp_path, monkeypatch):
    from test_d1_all_series_original_segmentation import _patch_manual_export_dependencies
    _patch_manual_export_dependencies(monkeypatch, [])
    for name in ['RS.dcm','CT.nii.gz']:
        (tmp_path/name).write_bytes(b'input')
    result = organize._export_original_segmentation_from_paths(rs_path=tmp_path/'RS.dcm',
        primary_nifti=tmp_path/'CT.nii.gz', dicom_ct_dir=tmp_path/'DICOM',
        segmentation_original_dir=tmp_path/'Segmentation_Original', log_root=tmp_path, overwrite=False)
    assert result is None
    assert not list((tmp_path/'Segmentation_Original').rglob('*.nii.gz'))


def test_f7_square_v2_rs_auto_is_current(tmp_path, monkeypatch):
    from rtpipeline import auto_rtstruct as auto
    monkeypatch.setattr(auto, '_read_rs_auto_origin', lambda *_: {
        'origin': auto.RS_AUTO_ORIGIN_MASKS, 'mask_layout': 'rows-columns-v2'})
    assert auto._rs_auto_origin_rejection(tmp_path, tmp_path/'RS_auto.dcm', tmp_path/'CT', '', tmp_path/'seg') == ''


@pytest.mark.parametrize('vary', [False, True])
def test_f6_writer_first_slice_inverse_on_tilted_stack(tmp_path, vary):
    import pydicom
    from rt_utils import RTStructBuilder
    from robustness_prep_fixture import write_ct
    from rtpipeline.rtstruct_geometry import place_added_rois_on_planes, rtstruct_roi_mask
    ct = tmp_path/'CT'
    write_ct(ct, rows=20, columns=20, slices=4, spacing=(1., 1. if vary else 2., 3.))
    for k, path in enumerate(sorted(ct.glob('*.dcm'))):
        ds = pydicom.dcmread(path)
        ds.ImagePositionPatient = [0., k*.7, k*3.]
        if vary and k:
            ds.PixelSpacing = [1.,2.]
        ds.save_as(path, enforce_file_format=True)
    builder = RTStructBuilder.create_new(str(ct))
    mask = np.zeros((20,20,4), dtype=bool)
    mask[3:8,4:10,1:3] = True
    builder.add_roi(mask=mask, name='ROI')
    place_added_rois_on_planes(builder.ds, builder.series_data)
    np.testing.assert_array_equal(rtstruct_roi_mask(builder, 'ROI'), mask)


def test_f5_conda_fallback_aligns_mask_before_batch(tmp_path, monkeypatch):
    from rtpipeline import radiomics_conda as rc
    from rtpipeline.config import PipelineConfig
    from course_contract_test_utils import write_minimal_course_contract
    from robustness_prep_fixture import write_ct
    course = tmp_path/'P/C'
    ct = course/'DICOM/CT'
    write_ct(ct, rows=8, columns=8, slices=4)
    case_path = write_minimal_course_contract(course, planning_ct_dir=ct)
    from rtpipeline.course_contract import _ct_provenance
    case = json.loads(case_path.read_text())
    current = case['course_contract']['planning_ct']['nifti_provenance']
    current.update(_ct_provenance(ct))
    sidecar = course/current['sidecar_path']
    sidecar_data = json.loads(sidecar.read_text())
    sidecar_data.update(_ct_provenance(ct))
    sidecar.write_text(json.dumps(sidecar_data))
    case_path.write_text(json.dumps(case))
    contract = rc.load_course_contract(course)
    # The standard contract requires NIfTI. Inject its absent-image state to
    # exercise the defensive DICOM fallback with real DICOM pixels and masks.
    contract.data['planning_ct']['nifti_path'] = ''
    monkeypatch.setattr(rc, 'load_course_contract', lambda _: contract)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct)))
    native = reader.Execute()
    arr = np.zeros(tuple(reversed(native.GetSize())), dtype=np.uint8)
    arr[:, :1, :1] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(native)
    flipped = sitk.Flip(mask, [False,True,False])
    seg = course/'Segmentation_TotalSegmentator/CT'
    seg.mkdir(parents=True)
    sitk.WriteImage(flipped, str(seg/'total--liver.nii.gz'))
    seen = []
    def batch(tasks, *args, **kwargs):
        for task in tasks:
            image = sitk.ReadImage(task['image_path'])
            observed = sitk.ReadImage(task['mask_path'])
            seen.append(observed)
            assert observed.GetDirection() == image.GetDirection()
            np.testing.assert_array_equal(sitk.GetArrayFromImage(observed), arr)
        return []
    monkeypatch.setattr(rc, 'process_radiomics_batch', batch)
    monkeypatch.setattr(rc, '_materialized_ct_arm_hashes', lambda *_a, **_k: {
        'primary_resegmented':'p', 'sensitivity_raw':'r'})
    cfg = PipelineConfig(dicom_root=tmp_path, output_root=tmp_path/'out', logs_root=tmp_path/'logs', radiomics_min_voxels=1)
    try:
        rc.radiomics_for_course_ct_nifti_fallback(course,cfg)
    except RuntimeError as exc:
        assert seen, str(exc)  # Empty batch result prevents artifact publication.
    assert seen


def test_f10_parallel_guards_receive_aligned_array(tmp_path, monkeypatch):
    from rtpipeline import radiomics_parallel as rp
    from rtpipeline.radiomics_ct_contract import classify_ct_roi
    series = slices(rows=8, columns=9, count=4, spacing=(2.,3.), thickness=4.)
    native = sitk.Image([9,8,4],sitk.sitkInt16)
    native.SetSpacing((3.,2.,4.))
    image = sitk.PermuteAxes(native,[2,0,1])
    mask = np.zeros((8,9,4),dtype=bool)
    mask[1:5,2:7,1:3] = True
    builder = SimpleNamespace(series_data=series, get_roi_mask_by_name=lambda _:mask)
    monkeypatch.setattr(rp, '_get_builder', lambda _:builder)
    expected = np.moveaxis(sitk.GetArrayFromImage(rtstruct_mask_to_image(mask,series,image)),0,-1)>0
    def guard(observed, **kwargs):
        np.testing.assert_array_equal(observed,expected)
        raise RuntimeError('guard observed physical alignment')
    monkeypatch.setattr(rp,'estimate_resampled_bounding_box',guard)
    monkeypatch.setattr(rp,'_WORKER_STATE',dict(img=image,extractor=SimpleNamespace(settings={}),
        min_voxels=1,max_voxels=10000,skip_rois=set(),config=SimpleNamespace()))
    task = rp._RoiTask(source='Manual', rs_path=str(tmp_path/'RS.dcm'),roi_name='PTV',
        course_dir=str(tmp_path),series_uid='1.2.3', mask_identity='mask',stable_roi_identifier='roi',
        decision=classify_ct_roi('Manual','PTV'),run_identifier='run',code_revision='revision',
        configured_parameter_hashes={},effective_parameter_hashes={},required=False)
    with pytest.raises(RuntimeError,match='guard observed physical alignment'):
        rp._extract_one(task)


def write_sidecar(nifti):
    data = dict(series_instance_uid='1.2.3', sop_hash='synthetic-sops',
                nifti_geometry={'size':[2,2,1]},
                nifti_sha256=hashlib.sha256(nifti.read_bytes()).hexdigest())
    nifti.with_name(nifti.name.removesuffix('.nii.gz')+'.metadata.json').write_text(json.dumps(data))


@pytest.mark.parametrize('columns,spacing,adapter', [(8,(1.,1.,2.),False), (9,(1.,1.,2.),False), (8,(1.,2.,2.),True)])
def test_f7_model_reuse_is_geometry_scoped(tmp_path, columns, spacing, adapter):
    from robustness_prep_fixture import write_ct
    from rtpipeline.segmentation import _ensure_model_rtstruct_from_masks
    from rtpipeline.custom_models import _model_outputs_ready
    ct = tmp_path/'CT'
    write_ct(ct, rows=8, columns=columns, slices=3, spacing=spacing)
    seg = tmp_path/'seg'
    seg.mkdir()
    target = seg/'CT--total.dcm'
    target.write_bytes(b'existing model RTSTRUCT')
    custom = tmp_path/'Segmentation_CustomModels/model'
    custom.mkdir(parents=True)
    custom_rs = custom/'rtstruct.dcm'
    custom_rs.write_bytes(b'existing custom RTSTRUCT')
    model = SimpleNamespace(name='model', expected_structures=lambda:[])
    result = _ensure_model_rtstruct_from_masks(ct,seg,'CT','total')
    assert (result == target) is (not adapter)
    import inspect
    readiness_kwargs = {'ct_dir':ct} if 'ct_dir' in inspect.signature(_model_outputs_ready).parameters else {}
    assert _model_outputs_ready(tmp_path,model,**readiness_kwargs) is (not adapter)
    from rtpipeline.rtstruct_geometry import record_geometry_adapter
    target.write_bytes(b'current model RTSTRUCT')
    record_geometry_adapter(target,ct)
    record_geometry_adapter(custom_rs,ct)
    assert _ensure_model_rtstruct_from_masks(ct,seg,'CT','total') == target
    assert _model_outputs_ready(tmp_path,model,ct_dir=ct)
    if adapter:
        custom_rs.write_bytes(b'changed')
        assert not _model_outputs_ready(tmp_path,model,ct_dir=ct)


@pytest.mark.parametrize('columns,spacing', [(9,(1.,1.,2.)), (8,(1.,2.,2.))])
def test_f7_writer_adapter_scopes_v2_rs_auto_currentness(tmp_path, monkeypatch, columns, spacing):
    from robustness_prep_fixture import write_ct
    from rtpipeline import auto_rtstruct as auto
    ct=tmp_path/'CT'
    write_ct(ct,rows=8,columns=columns,slices=3,spacing=spacing)
    monkeypatch.setattr(auto,'_read_rs_auto_origin',lambda *_:{
        'origin':auto.RS_AUTO_ORIGIN_MASKS,'mask_layout':'rows-columns-v2'})
    assert bool(auto._rs_auto_origin_rejection(tmp_path,tmp_path/'RS_auto.dcm',ct,'',tmp_path/'seg')) == (spacing[0] != spacing[1])


def test_f10_conda_guards_receive_aligned_array(tmp_path, monkeypatch, rasterizer_guard_stub):
    from rt_utils import RTStructBuilder
    from robustness_prep_fixture import write_ct
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline import radiomics_conda as rc
    from rtpipeline.config import PipelineConfig
    from rtpipeline.course_contract import _ct_provenance
    from rtpipeline.rtstruct_geometry import place_added_rois_on_planes
    course=tmp_path/'P/C'
    ct=course/'DICOM/CT'
    write_ct(ct,rows=8,columns=9,slices=4,spacing=(2.,3.,4.))
    builder=RTStructBuilder.create_new(str(ct))
    mask=np.zeros((8,9,4),dtype=bool)
    mask[1:5,2:7,1:3]=True
    builder.add_roi(mask=mask,name='PTV')
    place_added_rois_on_planes(builder.ds,builder.series_data)
    rs=course/'RS.dcm'
    builder.save(str(rs))
    reader=sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct)))
    native=reader.Execute()
    nifti=course/'CT.nii.gz'
    sitk.WriteImage(native,str(nifti))
    case_path=write_minimal_course_contract(course,planning_ct_dir=ct,
        planning_ct_nifti=nifti,authoritative_rtstruct=rs)
    case=json.loads(case_path.read_text())
    current=case['course_contract']['planning_ct']['nifti_provenance']
    current.update(_ct_provenance(ct))
    sidecar=course/current['sidecar_path']
    data=json.loads(sidecar.read_text())
    data.update(_ct_provenance(ct))
    sidecar.write_text(json.dumps(data))
    case_path.write_text(json.dumps(case))
    image=sitk.PermuteAxes(native,[2,0,1])
    expected=np.moveaxis(sitk.GetArrayFromImage(rtstruct_mask_to_image(mask,builder.series_data,image)),0,-1)>0
    monkeypatch.setattr(rc.sitk,'ImageSeriesReader',lambda:SimpleNamespace(
        GetGDCMSeriesFileNames=lambda _:['synthetic'],SetFileNames=lambda _:None,Execute=lambda:image))
    seen=[]
    def guard(observed,**kwargs):
        np.testing.assert_array_equal(observed,expected)
        seen.append(True)
        raise RuntimeError('aligned guard reached')
    monkeypatch.setattr(rc,'estimate_resampled_bounding_box',guard)
    cfg=PipelineConfig(dicom_root=tmp_path,output_root=tmp_path/'out',logs_root=tmp_path/'logs',radiomics_min_voxels=1)
    with pytest.raises(RuntimeError,match='aligned guard reached'):
        rc.radiomics_for_course(course,cfg)
    assert seen


@pytest.mark.parametrize('field', ['planning_ct_series_instance_uid', 'source_ct_sop_hash',
    'source_nifti_geometry', 'source_nifti_sha256', 'source_rtstruct_sha256'])
def test_f4_every_source_field_is_bound(tmp_path, field):
    parent=tmp_path/'Segmentation_Original/CT_1'
    parent.mkdir(parents=True)
    path=parent/'ROI.nii.gz'
    path.write_bytes(b'mask')
    bind_export(tmp_path,[path])
    record=json.loads((parent/'provenance.json').read_text())
    record[field]='changed'
    (parent/'provenance.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='source inputs changed'):
        require_current_original_mask(path)


@pytest.mark.parametrize('ipp_decimals', [None, 3])
def test_f2_real_512_oblique_dcm2niix(tmp_path, dcm2niix, ipp_decimals):
    import subprocess
    import pydicom
    from robustness_prep_fixture import write_ct
    from rt_utils import RTStructBuilder
    ct=tmp_path/'CT'
    write_ct(ct,rows=512,columns=512,slices=3,spacing=(.931,.931,3.))
    x,y=.25,.37
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rotation=ry@rx
    for path in ct.glob('*.dcm'):
        ds=pydicom.dcmread(path)
        ds.ImageOrientationPatient=[f'{v:.6f}' for v in rotation[:,:2].T.ravel()]
        position=rotation@np.asarray(ds.ImagePositionPatient)
        ds.ImagePositionPatient=(position if ipp_decimals is None else np.round(position,ipp_decimals)).tolist()
        ds.save_as(path,enforce_file_format=True)
    output=tmp_path/'converted'
    output.mkdir()
    subprocess.run([dcm2niix,'-z','y','-f','reference','-o',str(output),str(ct)],check=True,capture_output=True)
    reference=sitk.ReadImage(str(next(output.glob('*.nii.gz'))))
    series=RTStructBuilder.create_new(str(ct)).series_data
    mask=np.zeros((512,512,3),dtype=bool)
    mask[90:120,200:240,1]=True
    result=rtstruct_mask_to_image(mask,series,reference)
    assert sitk.GetArrayFromImage(result).sum()==mask.sum()
    from rtpipeline.rtstruct_geometry import image_array_for_rtstruct
    np.testing.assert_array_equal(image_array_for_rtstruct(result,series)>0,mask)


def test_f2_single_row_rejects_unsampled_axis_scale():
    ref=sitk.Image([8,1,3],sitk.sitkUInt8)
    ref.SetSpacing((1.,1.5,2.))
    with pytest.raises(ValueError,match='MASK_GRID_NOT_EQUIVALENT'):
        rtstruct_mask_to_image(np.ones((1,8,3)),slices(rows=1,count=3),ref)


def test_f2_native_single_slice_reader_with_thickness(tmp_path):
    import pydicom
    from robustness_prep_fixture import write_ct
    ct=tmp_path/'CT'
    write_ct(ct,rows=8,columns=8,slices=1,spacing=(1.,1.,5.))
    reader=sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct)))
    reference=reader.Execute()
    assert reference.GetSpacing()[2]==1.
    series=[pydicom.dcmread(p) for p in ct.glob('*.dcm')]
    assert float(series[0].SliceThickness)==5.
    out=rtstruct_mask_to_image(np.ones((8,8,1)),series,reference)
    assert sitk.GetArrayFromImage(out).sum()==64


def test_f4_completion_binds_manifest_and_mask_together(tmp_path):
    from test_stage_completion import _dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel, validate_stage_completion_sentinel
    parent=tmp_path/'P/C/Segmentation_Original/CT_1'
    parent.mkdir(parents=True)
    course=parent.parent.parent
    path=parent/'ROI.nii.gz'
    path.write_bytes(b'mask')
    bind_export(course,[path])
    dicom=course/'DICOM/CT/image.dcm'
    dicom.parent.mkdir(parents=True)
    dicom.write_bytes(b'synthetic')
    write_stage_completion_sentinel(course,course/'.organized',stage='organize',status='ok',
        configuration_dependency=_dependency(tmp_path,'organize'))
    path.write_bytes(b'MASK')
    manifest=json.loads((parent/'metadata.json').read_text())
    manifest['structures'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (parent/'metadata.json').write_text(json.dumps(manifest))
    require_current_original_mask(path)
    with pytest.raises(ValueError,match='bytes or inventory changed'):
        validate_stage_completion_sentinel(course/'.organized',expected_stage='organize')
