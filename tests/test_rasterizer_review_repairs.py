"""Runtime-enabled regressions for publication, resume and checkpoint boundaries."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pandas as pd
import pytest

from rtpipeline import radiomics, radiomics_conda as conda, radiomics_rasterizer as rasterizer
from rtpipeline import custom_structures_rtstruct as custom
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError, invalidate_radiomics_outputs
from rtpipeline.radiomics_ct_contract import _rasterizer_columns, write_ct_publication_atomic
import test_radiomics_fail_closed as fixtures

PIN = {
    'rasterizer_backend': 'rt-utils/cv2.fillPoly',
    'rasterizer_opencv_version': '4.8.1',
    'rasterizer_rt_utils_version': '1.2.7',
    'rasterizer_providers': json.dumps({'opencv-python-headless': '4.8.1.78'}),
}
VARIANTS = ('matching', 'missing', 'opencv', 'rt_utils', 'provider')


def altered(record, variant):
    record = {**record, **PIN}
    if variant == 'missing':
        for key in PIN:
            record.pop(key, None)
    elif variant == 'opencv':
        record['rasterizer_opencv_version'] = '4.11.0'
    elif variant == 'rt_utils':
        record['rasterizer_rt_utils_version'] = '1.2.6'
    elif variant == 'provider':
        record['rasterizer_providers'] = json.dumps({'opencv-python': '4.8.1.78'})
    return record


@pytest.mark.parametrize('backend', ['parallel', 'conda', 'parallel_delegated'])
def test_rebuild_guard_keeps_previous_file_quarantined(tmp_path, monkeypatch, backend):
    import rtpipeline.radiomics_parallel as parallel
    course = tmp_path / 'P1' / 'C1'
    fixtures._write_contract(course)
    fixtures._write_current_auto_rtstruct(course, roi_names=('PTV',))
    rs = course / 'RS_custom.dcm'
    fixtures.write_synthetic_rtstruct(rs, roi_names=('bowel_bag',))
    previous = rs.read_bytes()
    config_file = tmp_path / 'custom.yaml'
    config_file.write_text('custom_structures:\n  - name: bowel_bag\n    operation: union\n    source_structures: [PTV]\n')
    config = fixtures._config(tmp_path, custom_structures_config=config_file)
    monkeypatch.setattr(radiomics, '_load_series_image', lambda *_a: object())
    monkeypatch.setattr(radiomics, '_extractor', lambda *_a: (
        None if backend == 'parallel_delegated' else fixtures._FakeExtractor()))
    monkeypatch.setattr(parallel, '_is_rs_custom_stale', lambda *_a: True)
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *_a: True)
    writer = Mock(side_effect=lambda *_a: rs.write_bytes(b'WRONG NEW FILE'))
    monkeypatch.setattr(parallel, '_create_custom_structures_rtstruct', writer)
    monkeypatch.setattr(custom, '_create_custom_structures_rtstruct', writer)
    guard = Mock(side_effect=rasterizer.RadiomicsRasterizerError('unsupported rasterizer'))
    monkeypatch.setattr(rasterizer, 'require_radiomics_rasterizer', guard)
    checkpoint = course / 'metadata' / 'radiomics_ct_checkpoint.parquet'
    checkpoint.write_bytes(b'stale checkpoint')
    with pytest.raises(RadiomicsCourseExtractionError, match='unsupported rasterizer'):
        if backend == 'conda':
            conda.radiomics_for_course(course, config)
        else:
            parallel.parallel_radiomics_for_course(config, course)
    guard.assert_called_once()
    writer.assert_not_called()
    assert not rs.exists()
    assert not checkpoint.exists()
    quarantined = list(course.glob('.RS_custom.dcm.rejected*'))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == previous
    audit = json.loads((course / 'metadata' / 'segmentation_resume.json').read_text())
    assert audit['decisions']['RS_custom']['action'] == 'failed'


@pytest.mark.parametrize('variant', VARIANTS)
def test_parallel_resume_checks_rasterizer_before_reuse(tmp_path, monkeypatch, variant):
    # Synthetic features exercise real course publication and resume control flow.
    course, parallel = fixtures._parallel_course_with_fake_roi_results(
        tmp_path, monkeypatch, {'Manual': [fixtures._success_record],
                               'AutoRTS_total': [fixtures._success_record, fixtures._success_record]})
    monkeypatch.setattr(rasterizer, 'require_radiomics_rasterizer', lambda: None)
    config = fixtures._config(tmp_path, resume=True)
    parallel.parallel_radiomics_for_course(config, course, max_workers=1)
    output = course / 'radiomics_ct.xlsx'
    rows = pd.read_parquet(output.with_suffix('.parquet')).to_dict('records')
    write_ct_publication_atomic(pd.DataFrame([altered(row, variant) for row in rows]), output)
    before = output.with_suffix('.parquet').read_bytes()
    guard = Mock(side_effect=rasterizer.RadiomicsRasterizerError('recomputation required'))
    monkeypatch.setattr(rasterizer, 'require_radiomics_rasterizer', guard)
    if variant == 'matching':
        parallel.parallel_radiomics_for_course(config, course, max_workers=1)
        guard.assert_not_called()
        assert output.with_suffix('.parquet').read_bytes() == before
    else:
        with pytest.raises(rasterizer.RadiomicsRasterizerError, match='recomputation required'):
            parallel.parallel_radiomics_for_course(config, course, max_workers=1)
        guard.assert_called_once()
        assert not output.exists()


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('source', ['rtstruct', 'nifti', 'dicom-seg'])
def test_mr_resume_identity(variant, source):
    row = altered({'extraction_arm': 'mr_configured',
                   'configured_parameter_hash': radiomics.configured_parameter_hash(
                       None, arm='mr_configured', window=None, large_roi=False),
                   'effective_parameter_hash': 'effective', 'run_identifier': 'run',
                   'code_revision': 'same-revision', 'segmentation_source': 'Manual'}, variant)
    if source != 'rtstruct':
        row['rasterizer_backend'] = f'{source}/no-polygon-fill'
    assert radiomics._mr_resume_parameter_provenance_is_current(pd.DataFrame([row]), None) == (
        variant == 'matching' or source != 'rtstruct')


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('source', ['rtstruct', 'nifti'])
def test_ct_checkpoint_identity(tmp_path, variant, source):
    path = tmp_path / 'metadata' / 'radiomics_ct_checkpoint.parquet'
    path.parent.mkdir()
    fixtures._write_test_ct_publication(tmp_path / 'radiomics_ct.xlsx', [
        {'segmentation_source': 'Manual', 'roi_name': 'PTV', 'original_firstorder_Mean': 7.5}])
    rows = pd.read_parquet(tmp_path / 'radiomics_ct.parquet').to_dict('records')
    rows = [altered(row, variant) for row in rows]
    if source == 'nifti':
        for row in rows:
            row['mask_path_source'] = '/synthetic/liver.nii.gz'
            row['rasterizer_backend'] = 'nifti/no-polygon-fill'
    pd.DataFrame(rows).to_parquet(path, index=False)
    keys = {conda._roi_instance_key(row) for row in rows}
    hashes = {conda._roi_instance_key(row): row['configured_parameter_hash'] for row in rows}
    cp = conda.RadiomicsCheckpoint(path, expected_keys=keys, expected_configured_hashes=hashes)
    accepted = variant == 'matching' or source == 'nifti'
    assert path.exists() == accepted
    assert cp.get_completed_count() == (len(rows) if accepted else 0)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('source', ['rtstruct', 'nifti'])
def test_mr_checkpoint_identity(tmp_path, variant, source):
    path = tmp_path / 'radiomics_mr_checkpoint.parquet'
    row = altered({'segmentation_source': 'Manual', 'roi_name': 'liver',
                   'series_uid': 'series', 'original_firstorder_Mean': 4.0,
                   'extraction_arm': 'mr_configured', 'patient_id': 'P1',
                   'course_id': 'C1', 'mask_identity': 'mask',
                   'roi_original_name': 'liver', 'stable_roi_identifier': 'liver'}, variant)
    if source == 'nifti':
        row['mask_path_source'] = '/synthetic/liver.nii.gz'
        row['rasterizer_backend'] = 'nifti/no-polygon-fill'
    task = {'metadata': {**row, **PIN}}
    if source == 'nifti':
        task['metadata']['rasterizer_backend'] = 'nifti/no-polygon-fill'
    pd.DataFrame([row]).to_parquet(path, index=False)
    conda._mr_reject_stale_checkpoint(path, [task], tmp_path / 'radiomics_mr.xlsx')
    accepted = variant == 'matching' or source == 'nifti'
    assert path.exists() == accepted
    checkpoint = conda.RadiomicsCheckpoint(path, expected_keys={conda._roi_instance_key(row)})
    assert checkpoint.get_completed_count() == (1 if accepted else 0)
    assert path.exists() == accepted


@pytest.mark.parametrize('location', ['metadata/radiomics_ct_checkpoint.parquet',
                                      'metadata/radiomics_mr_checkpoint.parquet',
                                      'radiomics_mr_checkpoint.parquet'])
def test_invalidation_removes_checkpoints(tmp_path, location):
    path = tmp_path / location
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b'old checkpoint')
    invalidate_radiomics_outputs(tmp_path / 'radiomics_ct.xlsx')
    assert not path.exists()


def test_probe_failure_is_a_row_field(monkeypatch):
    info = rasterizer.active_radiomics_rasterizer()
    info['error'] = 'ImportError: cv2 unavailable'
    monkeypatch.setattr(rasterizer, 'active_radiomics_rasterizer', lambda: info)
    assert _rasterizer_columns()['rasterizer_probe_error'] == info['error']


def test_apptainer_fails_closed():
    definition = (Path(__file__).resolve().parents[1] / 'rtpipeline.def').read_text()
    post = definition.split('%post\n')[1].split('%environment')[0]
    assert 'set -eu' in post
    assert '/opt/conda/envs/rtpipeline-radiomics/bin/python /app/scripts/check_radiomics_environment.py' in post
    assert '|| true' not in post
    assert 'Dockerfile' in definition


@pytest.mark.parametrize('stage', ['organize', 'segmentation', 'segmentation_custom', 'crop_ct', 'dvh', 'qc'])
def test_main_stage_completion_records_imported_cv2(tmp_path, stage):
    from rtpipeline import stage_completion as sc
    from test_stage_completion import _dependency
    course = tmp_path / 'P1' / 'C1'
    course.mkdir(parents=True)
    definition = sc.stage_definition(stage)
    payload = sc.stage_completion_payload(
        course, stage=stage, status='disabled',
        configuration_dependency=_dependency(tmp_path, definition.configuration_stage))
    info = payload['opencv_runtime']
    import cv2
    assert info['opencv_version'] == cv2.__version__
    assert info['opencv_path'] == str(Path(cv2.__file__).resolve())
    assert info['providers']
    assert info['error'] is None


def test_main_origin_manifest_and_repair_summary_use_existing_records(tmp_path):
    from rtpipeline import auto_rtstruct, segmentation, roi_fixer
    course = tmp_path / 'P1' / 'C1'
    course.mkdir(parents=True)
    rs = course / 'RS_auto.dcm'
    rs.write_bytes(b'synthetic rtstruct')
    auto_rtstruct._record_rs_auto_origin(course, rs, auto_rtstruct.RS_AUTO_ORIGIN_MASKS)
    origin = json.loads((course / 'metadata' / 'rs_auto_origin.json').read_text())
    manifest = course / 'manifest.json'
    segmentation._write_manifest_atomic(manifest, {'structures': []})
    summary = roi_fixer.FixSummary(False, [], [], rs)
    expected = rasterizer.opencv_runtime_provenance()
    assert origin['opencv_runtime'] == expected
    assert json.loads(manifest.read_text())['opencv_runtime'] == expected
    assert summary.opencv_runtime == expected


@pytest.mark.parametrize('module,record', [
    ('custom_models', 'rtstruct_manifest.json'),
    ('custom_structures_rtstruct', 'rs_custom_meta.json'),
    ('anatomical_cropping', 'cropping_metadata.json'),
    ('dvh', 'dvh_qc.json'),
    ('quality_control', 'opencv_runtime'),
    ('visualize', 'OPENCV_RUNTIME'),
])
def test_main_writer_records_include_runtime_provenance(module, record):
    # The existing synthetic writer/geometry tests execute these write paths;
    # this wiring assertion protects their provenance without a second writer.
    import importlib
    import inspect
    source = inspect.getsource(importlib.import_module(f'rtpipeline.{module}'))
    assert record in source
    assert 'opencv_runtime_provenance()' in source
    assert 'require_radiomics_rasterizer' not in source


@pytest.mark.parametrize('backend', ['nifti/no-polygon-fill', 'dicom-seg/no-polygon-fill'])
def test_nonpolygon_backend_columns(backend):
    cols = _rasterizer_columns(backend)
    assert cols['rasterizer_backend'] == backend
    assert rasterizer.recorded_rasterizer_is_current(cols)


@pytest.mark.parametrize('source', ['CustomModel:nifti_model', 'CustomModel:dicom_seg_model'])
def test_model_names_do_not_exempt_polygon_rows(source):
    assert not rasterizer.recorded_rasterizer_is_current({'segmentation_source': source})


def test_nullable_identity_is_rejected():
    row = {**PIN, 'rasterizer_rt_utils_version': pd.NA}
    assert not rasterizer.recorded_rasterizer_is_current(row)
