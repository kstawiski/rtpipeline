"""Synthetic regressions at the producer and CLI boundaries for S9d."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline import organize, segmentation as seg
from rtpipeline.original_segmentation import quarantine_stale_original_exports
from rtpipeline.rtstruct_geometry import geometry_adapter_output_current, rtstruct_mask_to_image
from test_s9c_repairs import bind_export, slices
from test_rs_auto_orientation import (
    ANISOTROPIC, NON_SQUARE, SQUARE, _geometry_course, _uniform_z,
    _assert_physical_extent, _age, _origin,
)
from test_d1_all_series_original_segmentation import _cfg
from test_planning_ct_localizers import _organized_course, dcm2niix  # noqa: F401


def test_n1_rejection_survives_repeated_organize_resumes(tmp_path, monkeypatch):
    from rt_utils import RTStructBuilder
    from rtpipeline import rtstruct_geometry as geo
    from rtpipeline.stage_completion import _validate_stage_specific_outcome, stage_definition, _stale_original_exports
    root = tmp_path / 'Segmentation_Original'
    bind_export(tmp_path, [])
    sitk.WriteImage(sitk.Image([8, 8, 1], sitk.sitkUInt8), str(tmp_path / 'CT.nii.gz'))
    sidecar = json.loads((tmp_path / 'CT.metadata.json').read_text())
    import hashlib
    sidecar['nifti_sha256'] = hashlib.sha256((tmp_path / 'CT.nii.gz').read_bytes()).hexdigest()
    (tmp_path / 'CT.metadata.json').write_text(json.dumps(sidecar))
    case_path = tmp_path / 'metadata/case_metadata.json'
    case = json.loads(case_path.read_text())
    case['course_contract']['planning_ct']['nifti_provenance'] = sidecar
    case_path.write_text(json.dumps(case))
    dicom = tmp_path / 'DICOM/CT/image.dcm'
    dicom.parent.mkdir(parents=True)
    dicom.write_bytes(b'synthetic DICOM')
    from test_stage_completion import _dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel, validate_stage_completion_sentinel
    dependency = _dependency(tmp_path, 'organize')
    calls = []
    def builder(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_roi_names=lambda: ['ROI'], series_data=slices())
    monkeypatch.setattr(RTStructBuilder, 'create_from', builder)
    monkeypatch.setattr(geo, 'rtstruct_roi_mask', lambda *_: np.ones((8, 8, 1)))
    def reject(*args):
        raise ValueError('MASK_GRID_NOT_EQUIVALENT: synthetic mismatch')
    monkeypatch.setattr(geo, 'rtstruct_mask_to_image', reject)
    for _ in range(3):
        organize._export_original_segmentation_from_paths(rs_path=tmp_path / 'RS.dcm',
            primary_nifti=tmp_path / 'CT.nii.gz', dicom_ct_dir=tmp_path / 'CT',
            segmentation_original_dir=root, log_root=tmp_path, overwrite=False)
        write_stage_completion_sentinel(tmp_path, tmp_path / '.organized', stage='organize',
                                        status='ok', configuration_dependency=dependency)
        validate_stage_completion_sentinel(tmp_path / '.organized', expected_stage='organize')
        assert json.loads((root / 'CT/metadata.json').read_text())['geometry_status'] == 'MASK_GRID_NOT_EQUIVALENT'
        assert not list(root.rglob('*.nii*'))
        assert not _stale_original_exports(tmp_path, stage_definition('organize'))
        with pytest.raises(ValueError, match='geometry was rejected'):
            _validate_stage_specific_outcome(tmp_path, stage_definition('segmentation'), 'ok', [])
    assert len(calls) == 1
    assert not (tmp_path / 'Segmentation_Quarantine').exists()
    (tmp_path / 'RS.dcm').write_bytes(b'changed RTSTRUCT')
    organize._export_original_segmentation_from_paths(rs_path=tmp_path / 'RS.dcm',
        primary_nifti=tmp_path / 'CT.nii.gz', dicom_ct_dir=tmp_path / 'CT',
        segmentation_original_dir=root, log_root=tmp_path, overwrite=False)
    assert len(calls) == 2


def test_n3_loose_files_never_move_root_or_current_exports(tmp_path):
    root = tmp_path / 'Segmentation_Original'
    (root / 'CT').mkdir(parents=True)
    mask = root / 'CT/roi.nii.gz'
    mask.write_bytes(b'mask')
    bind_export(tmp_path, [mask])
    (root / 'README.txt').write_text('loose file')
    inode = root.stat().st_ino
    quarantine_stale_original_exports(root)
    assert root.is_dir() and root.stat().st_ino == inode
    assert mask.read_bytes() == b'mask'
    receipts = list((tmp_path / 'Segmentation_Quarantine').rglob('record.json'))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())['path'] == 'Segmentation_Original/README.txt'
    quarantine_stale_original_exports(root)
    assert len(list((tmp_path / 'Segmentation_Quarantine').rglob('record.json'))) == 1


@pytest.mark.parametrize('rederived', [False, True])
def test_n2_localizer_invalidates_and_direct_cli_fails_closed(tmp_path, monkeypatch, dcm2niix, rederived):
    from rtpipeline import repair_planning_ct_localizers as repair, cli, auto_rtstruct
    root = tmp_path / 'Output'
    course = _organized_course(root, dcm2niix)
    original = course / 'Segmentation_Original/legacy'
    original.mkdir(parents=True)
    (original / 'roi.nii.gz').write_bytes(b'legacy clinical mask')
    (original / 'metadata.json').write_text('{"schema":"v1"}')
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    dependency = tmp_path / 'config.json'
    dependency.write_text(json.dumps(json.loads((course / '.organized').read_text())['configuration_dependency']))
    write_stage_completion_sentinel(course, course / '.organized', stage='organize', status='ok',
                                    configuration_dependency=dependency)
    if rederived:
        real_verify = repair.verify_volume
        calls = []
        def fail_once(*args):
            calls.append(1)
            if len(calls) == 1:
                raise repair.PlanningCTLocalizerError('synthetic_rederive')
            return real_verify(*args)
        monkeypatch.setattr(repair, 'verify_volume', fail_once)
    result = repair.repair_output(root)
    assert result['repaired'] == 1
    assert result['niftis_rederived'] == int(rederived)
    assert not (course / '.organized').exists()
    assert repair.repair_output(root)['unchanged'] == 1
    monkeypatch.setattr(seg, '_series_masks_current', lambda *a, **k: (True, 'current'))
    inference = []
    monkeypatch.setattr(seg, 'run_totalsegmentator', lambda *a, **k: inference.append(1))
    auto_calls = []
    monkeypatch.setattr(auto_rtstruct, 'build_auto_rtstruct', lambda *a: auto_calls.append(1))
    task = cli._SegmentTask(cfg=_cfg(tmp_path), course=SimpleNamespace(dirs=SimpleNamespace(root=course)), force_segmentation=False)
    assert cli._execute_segment_task(task) is False
    assert inference == [] and auto_calls == []
    report = json.loads((course / seg.SEGMENTATION_STATUS_RELATIVE_PATH).read_text())
    assert 'organize' in ' '.join(report['reasons']).lower()


@pytest.mark.parametrize('caller', ['course', 'all_series'])
@pytest.mark.parametrize('geometry', [SQUARE, ANISOTROPIC], ids=['square', 'unequal'])
def test_n4_production_resume_validates_existing_model_rtstruct(tmp_path, monkeypatch, caller, geometry):
    course, ct, masks = _geometry_course(tmp_path, _uniform_z(), geometry)
    cfg = _cfg(tmp_path)
    cfg.extra_seg_models = ['test_extra']
    if caller == 'course':
        contract = seg.load_course_contract(course)
        # Use the actual course stem while retaining the physical mask fixture.
        base = seg._strip_nifti_base(contract.planning_ct_nifti)
        target_dir = masks.parent / base
        if target_dir != masks:
            masks.rename(target_dir)
            masks = target_dir
    else:
        base = 'CT_series'
        monkeypatch.setattr(seg, '_ensure_ct_nifti', lambda *a, **k: next((course / 'NIFTI').glob('*.nii.gz')))
        # The all-series producer resolves its own output root.
        monkeypatch.setattr(seg, '_strip_nifti_base', lambda *_: base)
        monkeypatch.setattr(seg, '_series_artifact_dirs', lambda *_: (course / 'NIFTI', masks.parent))
        monkeypatch.setattr(seg, 'ts_tasks_for_image_class', lambda *a: ['total', 'test_extra'])
        manifest = cfg.output_root / 'P1/all_series/metadata/series_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'series': [{'ts_task': 'total', 'status': 'materialized',
            'image_class': 'planning_ct', 'series_uid': '1.2.3', 'output_dir': str(ct)}]}))
    for path in list(masks.glob('total--*.nii.gz')):
        (masks / path.name.replace('total--', 'test_extra--')).write_bytes(path.read_bytes())
    targets = [masks / f'{base}--{model}.dcm' for model in ('total', 'test_extra')]
    for target in targets:
        target.write_bytes(b'existing writer output')
    monkeypatch.setattr(seg, '_series_masks_current', lambda *a, **k: (True, 'current synthetic masks'))
    monkeypatch.setattr(seg, 'run_totalsegmentator', lambda *a, **k: pytest.fail('resume ran inference'))
    if caller == 'course':
        seg.segment_course(cfg, course)
    else:
        seg.segment_all_series_for_patient(cfg, 'P1')
    for target in targets:
        if geometry == SQUARE:
            assert target.read_bytes() == b'existing writer output'
        else:
            assert geometry_adapter_output_current(target, ct)
            _assert_physical_extent(target, _uniform_z(), geometry)


@pytest.mark.parametrize('geometry', [NON_SQUARE, ANISOTROPIC], ids=['rect_equal', 'unequal'])
def test_n5_current_writer_can_be_copied(tmp_path, geometry):
    from rtpipeline import auto_rtstruct as ar
    course, ct, masks = _geometry_course(tmp_path, _uniform_z(), geometry)
    target = seg._ensure_model_rtstruct_from_masks(ct, masks, 'CT_series', 'total')
    if geometry == NON_SQUARE:
        target.with_suffix('.geometry.json').unlink()
    assert geometry_adapter_output_current(target, ct)
    _age(ct, course / 'NIFTI', masks)
    output = ar.build_auto_rtstruct(course)
    assert _origin(course)['origin'] == ar.RS_AUTO_ORIGIN_COPY
    assert output.read_bytes() == target.read_bytes()
    _assert_physical_extent(output, _uniform_z(), geometry)
    before = output.stat().st_mtime_ns
    assert ar.build_auto_rtstruct(course).stat().st_mtime_ns == before


@pytest.mark.parametrize('raw', [None, '', ' ', -3., 0., 'invalid', float('nan')])
def test_n6_single_slice_spacing_fallback(raw):
    series = slices(thickness=3.)
    series[0].SpacingBetweenSlices = raw
    ref = sitk.Image([8, 8, 1], sitk.sitkUInt8)
    ref.SetSpacing((1., 1., 3.))
    result = rtstruct_mask_to_image(np.ones((8, 8, 1)), series, ref)
    assert result.GetDirection() == ref.GetDirection()
    assert sitk.GetArrayFromImage(result).sum() == 64


def test_n4_custom_model_rebuilds_from_existing_masks_without_inference(tmp_path, monkeypatch):
    from rtpipeline import custom_models as cm
    from rtpipeline.layout import build_course_dirs
    course, ct, masks = _geometry_course(tmp_path, _uniform_z(), ANISOTROPIC)
    model = SimpleNamespace(name='synthetic', expected_structures=lambda: ['wide_box', 'tall_box'])
    output = course / 'Segmentation_CustomModels/synthetic'
    output.mkdir(parents=True)
    structures = []
    for name in model.expected_structures():
        (output / f'{name}.nii.gz').write_bytes((masks / f'total--{name}.nii.gz').read_bytes())
        structures.append({'name': name, 'file': f'{name}.nii.gz'})
    (output / 'rtstruct.dcm').write_bytes(b'stale writer')
    (output / 'manifest.json').write_text(json.dumps({'structures': structures}))
    monkeypatch.setattr(cm, '_check_custom_model_eligibility', lambda *a: (True, 'synthetic'))
    monkeypatch.setattr(cm, '_run_single_model', lambda *a: pytest.fail('RTSTRUCT-only repair reran inference'))
    cm.run_custom_models_for_course(_cfg(tmp_path), SimpleNamespace(dirs=build_course_dirs(course),
        patient_id='synthetic', course_id='course'), [model])
    assert geometry_adapter_output_current(output / 'rtstruct.dcm', ct)
    _assert_physical_extent(output / 'rtstruct.dcm', _uniform_z(), ANISOTROPIC)
