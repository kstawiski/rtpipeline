"""Synthetic boundary tests; no patient data or installed-environment edits."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline import radiomics, radiomics_conda, radiomics_runtime as runtime
from rtpipeline import radiomics_rasterizer as rasterizer
from rtpipeline import radiomics_robustness as robustness
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rtpipeline.radiomics_ct_contract import validate_ct_publication, sentinel_payload, validate_completion_sentinel
from test_rasterizer_review_repairs import PIN, VARIANTS, altered
import test_radiomics_fail_closed as fixtures

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant', VARIANTS)
def test_ct_publication_and_sentinel_require_pin(tmp_path, variant, rasterizer_guard_stub):
    fixtures._write_test_ct_publication(tmp_path / 'radiomics_ct.xlsx', [
        {'segmentation_source': 'Manual', 'roi_name': 'PTV', 'original_firstorder_Mean': 7.5}])
    table = tmp_path / 'radiomics_ct.parquet'
    frame = pd.DataFrame([altered(row, variant) for row in pd.read_parquet(table).to_dict('records')])
    from rtpipeline.radiomics_ct_contract import _feature_schema_metadata
    for key, value in _feature_schema_metadata(["original_firstorder_Mean"]).items():
        frame[key] = value
    if variant == 'matching':
        validate_ct_publication(frame)
        frame.to_parquet(table)
        payload = sentinel_payload(frame, table)
        (tmp_path / '.radiomics_done').write_text(json.dumps(payload))
        validate_completion_sentinel(tmp_path)
    else:
        with pytest.raises(ValueError, match='rasterizer identity'):
            validate_ct_publication(frame)
        with pytest.raises(ValueError, match='rasterizer identity'):
            sentinel_payload(frame, table)
        frame.to_parquet(table)
        (tmp_path / '.radiomics_done').write_text('{}')
        with pytest.raises(ValueError, match='rasterizer identity'):
            validate_completion_sentinel(tmp_path)


@pytest.mark.parametrize('backend', ['nifti/no-polygon-fill', 'dicom-seg/no-polygon-fill'])
def test_voxel_rows_remain_exempt(backend):
    rasterizer.require_table_rasterizer(pd.DataFrame([{'rasterizer_backend': backend}]))


@pytest.mark.parametrize('path', [
    '/data/totalsegmentator_total_mr_cohort/P/RS.dcm',
    '/data/totalsegmentator_total_mr_nifti_study/P/RS_auto.dcm',
    '/data/TotalSegmentator_total_mr_NIFTI/RS_custom.dcm',
    '/data/scan--total_mr.dcm',
])
def test_directory_names_do_not_exempt_rtstruct(path):
    assert rasterizer.mask_backend({'mask_path_source': path}) == rasterizer.RASTERIZER_BACKEND
    assert not rasterizer.recorded_rasterizer_is_current({'mask_path_source': path})


def test_explicit_polygon_backend_wins_over_voxel_path():
    record = {'mask_path_source': '/data/TotalSegmentator_total_mr_NIFTI/liver.nii.gz',
              'rasterizer_backend': rasterizer.RASTERIZER_BACKEND}
    assert rasterizer.mask_backend(record) == rasterizer.RASTERIZER_BACKEND
    assert not rasterizer.recorded_rasterizer_is_current(record)
    record.pop('rasterizer_backend')
    assert rasterizer.mask_backend(record) == 'nifti/no-polygon-fill'


@pytest.mark.parametrize('sop,expected', [('1.2.840.10008.5.1.4.1.1.481.3', rasterizer.RASTERIZER_BACKEND),
                                         ('1.2.840.10008.5.1.4.1.1.66.4', 'dicom-seg/no-polygon-fill')])
def test_legacy_mr_dicom_artifact_is_classified_by_sop(tmp_path, monkeypatch, sop, expected):
    path = tmp_path / 'TotalSegmentator_total_mr_DICOM' / 'segmentations.dcm'
    path.parent.mkdir()
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = sop
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = sop
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.save_as(path, enforce_file_format=True)
    assert rasterizer.mask_backend({'mask_path_source': str(path)}) == expected
    monkeypatch.setattr(radiomics, '_find_mr_manual_rs', lambda *_a: [])
    config, series = SimpleNamespace(dicom_root=tmp_path), SimpleNamespace(patient_id='P')
    if expected == rasterizer.RASTERIZER_BACKEND:
        with pytest.raises(RadiomicsCourseExtractionError, match='not DICOM-SEG'):
            radiomics._mr_mask_sources(config, series, tmp_path, for_uid='synthetic')
    else:
        sources = radiomics._mr_mask_sources(config, series, tmp_path, for_uid='synthetic')
        assert len(sources) == 1 and sources[0].kind == 'auto_dicom_seg'


def test_course_mr_rtstruct_guard_is_not_swallowed(tmp_path, monkeypatch):
    seg = tmp_path / 'seg'
    seg.mkdir()
    (seg / 'scan--total_mr.dcm').touch()
    monkeypatch.setattr(rasterizer, 'require_radiomics_rasterizer',
                        lambda: (_ for _ in ()).throw(rasterizer.RadiomicsRasterizerError('pin required')))
    with pytest.raises(rasterizer.RadiomicsRasterizerError, match='pin required'):
        radiomics._collect_total_mr_masks(tmp_path, seg)


def test_rtstruct_guard_precedes_empty_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(rasterizer, 'require_radiomics_rasterizer',
                        lambda: (_ for _ in ()).throw(rasterizer.RadiomicsRasterizerError('pin required')))
    with pytest.raises(rasterizer.RadiomicsRasterizerError, match='pin required'):
        radiomics._rtstruct_masks(tmp_path, tmp_path / 'empty.dcm',
                                  structural_inventory=SimpleNamespace(named_rois=()))


def test_mr_nifti_fallback_records_loader(tmp_path):
    import SimpleITK as sitk
    from robustness_prep_fixture import write_ct

    series = tmp_path / 'series'
    write_ct(series, rows=4, columns=5, slices=3)
    reference = radiomics._load_series_image(series)
    expected = np.zeros((3, 4, 5), dtype=np.uint8)
    expected[1, 0:2, 2:4] = 1
    mask = sitk.GetImageFromArray(expected)
    mask.CopyInformation(reference)
    seg = tmp_path / 'seg'
    seg.mkdir()
    sitk.WriteImage(sitk.Flip(mask, [False, True, False]), str(seg / 'total_mr--liver.nii.gz'))
    backend = {}
    masks = radiomics._collect_total_mr_masks(series, seg, backend_out=backend)
    np.testing.assert_array_equal(masks['liver'], np.moveaxis(expected, 0, -1) > 0)
    assert backend == {'backend': 'nifti/no-polygon-fill'}


def test_course_driver_reexec_transfers_configuration_only(tmp_path, monkeypatch):
    monkeypatch.setattr(radiomics_conda, '_radiomics_env_prefix', lambda: '/isolated/radiomics')
    calls = []
    def execute(command, *, env, check):
        calls.append(command)
        driver, args, kwargs = pickle.loads(Path(command[-2]).read_bytes())
        assert driver == 'ct' and args == (tmp_path, {'resume': True})
        assert kwargs == {'allow_all_series_temp': True}
        assert command[0] == '/isolated/radiomics/bin/python'
        assert env['PYTHONNOUSERSITE'] == '1'
        assert env['PYTHONPATH'].split(os.pathsep)[0] == str(ROOT)
        Path(command[-1]).write_bytes(pickle.dumps({"ok": True, "result": None}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runtime.subprocess, 'run', execute)
    assert runtime.run_course_driver('ct', tmp_path, {'resume': True}, allow_all_series_temp=True) is None
    assert not Path(calls[0][-2]).exists()


@pytest.mark.parametrize('entry', ['serial', 'parallel', 'conda'])
def test_missing_interpreter_fails_before_course_io(tmp_path, monkeypatch, entry):
    from rtpipeline.radiomics_parallel import parallel_radiomics_for_course
    monkeypatch.setattr(runtime, 'needs_radiomics_process', lambda: True)
    monkeypatch.setattr(radiomics_conda, '_radiomics_env_prefix', lambda: None)
    calls = {
        'serial': lambda: radiomics.radiomics_for_course(None, tmp_path / 'absent'),
        'parallel': lambda: parallel_radiomics_for_course(None, tmp_path / 'absent'),
        'conda': lambda: radiomics_conda.radiomics_for_course(tmp_path / 'absent', None),
    }
    with pytest.raises(RadiomicsCourseExtractionError, match='RTPIPELINE_RADIOMICS_ENV_PREFIX'):
        calls[entry]()
    assert not list(tmp_path.iterdir())


def test_check_script_self_locates_without_pythonpath(tmp_path):
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_radiomics_environment.py')],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert 'ImportError' not in result.stderr and 'ModuleNotFoundError' not in result.stderr
    if result.returncode:
        assert 'RTSTRUCT rasterization requires' in result.stderr
    else:
        assert 'rasterizer verified' in result.stdout


def test_robustness_admission_rejects_unrecorded_rows_first(tmp_path, monkeypatch):
    monkeypatch.setattr(robustness, '_read_robustness_input_bytes', lambda _: ('digest', pd.DataFrame([{'roi_name': 'liver'}])))
    with pytest.raises(ValueError, match='rasterizer identity'):
        robustness._admit_robustness_aggregation_input(tmp_path / 'table.parquet', None)
    assert 'radiomics_rasterizer.py' in robustness.ROBUSTNESS_DISPOSITION_CODE_SOURCES


def test_workflow_orders_custom_builder_before_both_consumers():
    from test_robustness_snakefile_parse import parse_snakefile_source
    compiled, _, _ = parse_snakefile_source((ROOT / 'Snakefile').read_text())
    tree = ast.parse(compiled)
    inputs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                                               {'_radiomics_input_dict', '_robustness_input_dict'} for t in node.targets):
            inputs[node.targets[0].id] = dict(zip([key.value for key in node.value.keys], node.value.values))
    assert inputs['_radiomics_input_dict']['dvh'].id == 'DVH_SENTINEL_PATTERN'
    assert inputs['_robustness_input_dict']['radiomics'].id == 'RADIOMICS_SENTINEL_PATTERN'


def test_container_cleans_overlapping_wheels_and_checks_both_pins():
    docker = (ROOT / 'Dockerfile').read_text()
    assert '/rtpipeline/bin/pip uninstall -y opencv-python-headless opencv-contrib-python opencv-contrib-python-headless' in docker
    assert '/rtpipeline-radiomics/bin/pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless' in docker
    assert "p['providers'] == {'opencv-python': '5.0.0.93'}" in docker
    assert '/rtpipeline-radiomics/bin/python /app/scripts/check_radiomics_environment.py' in docker


def test_numpy2_cannot_pass_even_with_pinned_opencv(monkeypatch):
    from test_radiomics_rasterizer import fake_cv2, pinned_version
    monkeypatch.setattr(rasterizer.np, "__version__", "2.0.0")
    monkeypatch.setattr(rasterizer, "_dist_version", pinned_version)
    with pytest.raises(rasterizer.RadiomicsRasterizerError, match="NumPy 1.x"):
        rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())


def test_builder_rejects_non_main_runtime_before_derivation(tmp_path, monkeypatch):
    from rtpipeline import custom_structures_rtstruct as custom
    monkeypatch.setattr(rasterizer, "opencv_runtime_provenance", lambda: {"opencv_version": "4.8.1"})
    monkeypatch.setattr(custom, "_create_custom_structures_rtstruct_unlocked",
                        lambda *_a: pytest.fail("wrong interpreter reached custom derivation"))
    with pytest.raises(RuntimeError, match="main NumPy 2 runtime"):
        custom._create_custom_structures_rtstruct(tmp_path)
    assert not (tmp_path / "RS_custom.dcm").exists()


def test_real_child_driver_no_ct(tmp_path, monkeypatch):
    prefix = radiomics_conda._radiomics_env_prefix()
    if prefix is None:
        pytest.skip('radiomics interpreter absent')
    monkeypatch.setenv('RTPIPELINE_RADIOMICS_ENV_PREFIX', prefix)
    course = tmp_path / 'P1' / 'C1'
    fixtures._write_contract(course, no_ct=True)
    assert runtime.run_course_driver('ct', course, fixtures._config(tmp_path)) is None


def test_robustness_rejects_stale_custom_before_loading_it(tmp_path, monkeypatch):
    from test_robustness_roi_structural_status import _course_with_custom
    from rtpipeline import custom_structures_rtstruct as custom
    course, config, rob, _, _ = _course_with_custom(tmp_path, monkeypatch, selection=['ROI'])
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *_a: True)
    with pytest.raises(RuntimeError, match='RS_custom is stale'):
        robustness.robustness_for_course(config, rob, course)
    assert not (course / 'radiomics_robustness_ct.parquet').exists()


def test_robustness_second_nifti_fallback_labels_rows(tmp_path, monkeypatch):
    import SimpleITK as sitk
    from test_robustness_nonmeasurements import course_fixture
    course, config, rob, features = course_fixture(tmp_path, monkeypatch)
    identity = robustness.RobustnessRoiIdentity.from_mapping(dict(
        patient_id='P', course_id='C', series_uid='1.2.3', segmentation_source='AutoRTS_total',
        mask_identity='source-mask', roi_original_name='ROI', stable_roi_identifier='roi-1'))
    monkeypatch.setattr(robustness, '_load_main_ct_identity_catalog',
                        lambda *_a, **_k: ({('AutoRTS_total', 'ROI'): identity}, {}))
    monkeypatch.setattr(radiomics, '_standard_rtstruct_sources', lambda *_a: [])
    def voxel_features(*args, **kwargs):
        frame = features(*args, **kwargs)
        frame['segmentation_source'] = 'AutoRTS_total'
        return frame
    monkeypatch.setattr(robustness, 'extract_features_for_masks', voxel_features)
    seg = course / 'Segmentation_TotalSegmentator'
    seg.mkdir()
    array = np.zeros((17, 17, 17), dtype=np.uint8)
    array[5:12, 5:12, 5:12] = 1
    sitk.WriteImage(sitk.GetImageFromArray(array), str(seg / 'ROI.nii.gz'))
    table = robustness.robustness_for_course(config, rob, course)
    assert set(pd.read_parquet(table).rasterizer_backend) == {'nifti/no-polygon-fill'}


@pytest.mark.parametrize('variant', ['matching', 'missing', 'opencv'])
def test_workflow_mr_aggregation_enforces_identity(tmp_path, monkeypatch, variant, rasterizer_guard_stub):
    # Load the shipped functions without executing the workflow's top-level job.
    path = ROOT / 'rtpipeline' / 'workflow_aggregate.py'
    parsed = ast.parse(path.read_text())
    names = {'_load_course', '_load_body_region_data', '_add_course_ids', '_read_prefer_parquet'}
    subset = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'Path': Path, 'pd': pd, 'json': json, 'RADIOMICS_ENABLED': True}
    exec(compile(subset, str(path), 'exec'), namespace)
    fixtures._write_test_ct_publication(tmp_path / 'radiomics_ct.xlsx', [
        {'segmentation_source': 'Manual', 'roi_name': 'PTV', 'original_firstorder_Mean': 7.5}])
    mr = tmp_path / 'MR'
    mr.mkdir()
    pd.DataFrame([altered({'roi_name': 'liver'}, variant)]).to_parquet(mr / 'radiomics_mr.parquet')
    frames, errors = namespace['_load_course'](('P1', 'C1', tmp_path), {
        (tmp_path, 'dvh'): pd.DataFrame([{'ROI_Name': 'PTV'}]),
        (tmp_path, 'radiomics'): pd.read_parquet(tmp_path / 'radiomics_ct.parquet'),
    })
    if variant == 'matching':
        assert 'radiomics_mr' in frames and not errors
    else:
        assert 'radiomics_mr' not in frames
        assert any('rasterizer identity' in error for error in errors)


def test_native_mr_publication_keeps_nifti_loader_label(tmp_path, monkeypatch):
    import SimpleITK as sitk
    from rtpipeline.layout import build_course_dirs
    course = tmp_path / 'P1' / 'C1'
    dirs = build_course_dirs(course)
    series = dirs.dicom_mr / 'series'
    nifti, dicom, seg = series / 'NIFTI', series / 'DICOM', series / 'Segmentation_TotalSegmentator'
    for directory in (nifti, dicom, seg):
        directory.mkdir(parents=True)
    (dicom / 'slice.dcm').touch()
    image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.float32))
    image_path = nifti / 'image.nii.gz'
    sitk.WriteImage(image, str(image_path))
    (nifti / 'image.metadata.json').write_text(json.dumps({
        'modality': 'MR', 'nifti_path': str(image_path), 'source_directory': str(dicom),
        'series_instance_uid': 'synthetic',
    }))
    sitk.WriteImage(sitk.Cast(image, sitk.sitkUInt8), str(seg / 'total_mr--liver.nii.gz'))
    monkeypatch.setattr(radiomics.np, '__version__', '1.26.4')
    monkeypatch.setattr(radiomics, '_infer_mr_weighting', lambda *_a: 'T1')
    monkeypatch.setattr(radiomics, '_load_series_image', lambda *_a: image)
    monkeypatch.setattr(radiomics, '_extractor', lambda *_a, **_k: fixtures._FakeExtractor())
    result = radiomics.radiomics_for_course_mr(fixtures._config(tmp_path), SimpleNamespace(dirs=dirs))
    frame = pd.read_parquet(result.with_suffix('.parquet'))
    assert set(frame.rasterizer_backend) == {'nifti/no-polygon-fill'}
    rasterizer.require_table_rasterizer(frame)


@pytest.mark.parametrize('variant', ['missing', 'opencv'])
def test_partial_refresh_refuses_historical_polygon_rows(tmp_path, monkeypatch, variant, rasterizer_guard_stub):
    data = tmp_path / 'Data'
    data.mkdir()
    output = data / 'radiomics_all_series.xlsx'
    fixtures._write_test_ct_publication(output, [
        {'patient_id': 'KEEP', 'segmentation_source': 'Manual', 'roi_name': 'PTV', 'original_firstorder_Mean': 7.5}])
    table = output.with_suffix('.parquet')
    frame = pd.DataFrame([altered(row, variant) for row in pd.read_parquet(table).to_dict('records')])
    frame.to_parquet(table)
    before = table.read_bytes()
    monkeypatch.setattr(radiomics, '_have_pyradiomics', lambda: True)
    config = fixtures._config(tmp_path, output_root=tmp_path)
    with pytest.raises(RuntimeError, match='rasterizer identity'):
        radiomics.run_radiomics_all_series(config, ['REFRESH'])
    assert table.read_bytes() == before
    assert not (tmp_path / '.all_series_radiomics').exists()
