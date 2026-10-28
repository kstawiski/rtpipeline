"""Synthetic regressions for DVH ownership and isolated radiomics handoff."""
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rtpipeline import custom_structures_rtstruct as custom, dvh
from rtpipeline import radiomics as serial, radiomics_conda as conda
from rtpipeline import radiomics_parallel as parallel, radiomics_runtime as runtime
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
import test_radiomics_fail_closed as fx


def custom_config(tmp_path):
    path = tmp_path / 'custom.yaml'
    path.write_text('custom_structures:\n  - name: bowel_bag\n    operation: union\n    source_structures: [PTV]\n')
    return path


@pytest.mark.parametrize('cause', ['stale', 'configuration'])
def test_dvh_cache_checks_custom_currency_and_configuration(tmp_path, monkeypatch, cause):
    from test_dvh_cache_code_identity import _measured_course
    cfg = custom_config(tmp_path)
    workbook = _measured_course(tmp_path, monkeypatch,
        code_sources_sha256=dvh._current_dvh_measurement_code_sha256())
    monkeypatch.setattr(dvh, '_dvh_has_expected_custom_structures', lambda *a: True)
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *a: cause == 'stale')
    if cause == 'configuration':
        later = workbook.stat().st_mtime + 10
        os.utime(cfg, (later, later))
    assert not dvh._is_dvh_up_to_date(tmp_path, workbook, cfg)


@pytest.mark.parametrize('dose_present', [False, True])
@pytest.mark.parametrize('failure', ['exception', 'no_output'])
def test_custom_build_failure_fails_dvh(tmp_path, monkeypatch, dose_present, failure):
    import dvh_plan_target_fixtures as dose_fx
    course = dose_fx.build_course(tmp_path)
    cfg = custom_config(tmp_path)
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *a: True)
    def build(*a):
        if failure == 'exception':
            raise RuntimeError('custom builder failed')
        return None
    monkeypatch.setattr(custom, '_create_custom_structures_rtstruct', build)
    with pytest.raises(RuntimeError, match='custom|RS_custom'):
        if dose_present:
            dvh.dvh_for_course(course, cfg, parallel_workers=1)
        else:
            dvh._publish_skipped_course_rs_custom(course, cfg,
                dvh.load_course_contract(course).authoritative_rtstruct_path)
    assert not (course / 'dvh_metrics.xlsx').exists()


def test_cropped_dvh_builds_custom_from_uncropped_auto(tmp_path, monkeypatch):
    import dvh_plan_target_fixtures as dose_fx
    course = dose_fx.build_course(tmp_path)
    cfg = custom_config(tmp_path)
    manual = dvh.load_course_contract(course).authoritative_rtstruct_path
    for name in ('RS_auto.dcm', 'RS_auto_cropped.dcm'):
        (course / name).write_bytes(manual.read_bytes())
    (course / 'cropping_metadata.json').write_text('{}')
    seen = []
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *a: True)
    def build(root, config, rs_manual, rs_auto):
        seen.append(rs_auto)
        output = root / 'RS_custom.dcm'
        output.write_bytes(manual.read_bytes())
        return output
    monkeypatch.setattr(custom, '_create_custom_structures_rtstruct', build)
    class ReachedMeasurement(Exception):
        pass
    def resolve(**kwargs):
        assert kwargs['rs_auto'] == course / 'RS_auto_cropped.dcm'
        raise ReachedMeasurement
    monkeypatch.setattr(dvh, '_resolve_dvh_structures', resolve)
    with pytest.raises(ReachedMeasurement):
        dvh.dvh_for_course(course, cfg, parallel_workers=1, use_cropped=True)
    assert seen == [course / 'RS_auto.dcm']


def consumer_fixture(tmp_path, monkeypatch):
    course = tmp_path / 'P1' / 'C1'
    fx._write_contract(course)
    fx._write_current_auto_rtstruct(course, roi_names=('PTV',))
    fx.write_synthetic_rtstruct(course / 'RS_custom.dcm', roi_names=('bowel_bag',))
    cfg = fx._config(tmp_path, custom_structures_config=custom_config(tmp_path))
    monkeypatch.setattr(serial, '_load_series_image', lambda *a: object())
    monkeypatch.setattr(serial, '_extractor', lambda *a: fx._FakeExtractor())
    monkeypatch.setattr(serial, '_rtstruct_masks', lambda *a, **k: {'PTV': np.ones((2, 2, 2), dtype=bool)})
    return course, cfg


def call_consumer(backend, course, cfg):
    if backend == 'serial':
        return serial.radiomics_for_course(cfg, course)
    if backend == 'parallel':
        return parallel.parallel_radiomics_for_course(cfg, course)
    return conda.radiomics_for_course(course, cfg)


@pytest.mark.parametrize('backend', ['serial', 'parallel', 'conda'])
def test_stale_custom_never_inspected(tmp_path, monkeypatch, rasterizer_guard_stub, backend):
    from rtpipeline import roi_requiredness as requiredness
    course, cfg = consumer_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *a: True)
    # Modules cache different imports of the inventory function.
    inspected = []
    original = requiredness.inspect_rtstruct
    def inspect(path, *a, **kw):
        if Path(path).name == 'RS_custom.dcm':
            inspected.append(path)
            return SimpleNamespace(named_rois=[SimpleNamespace(name='bowel_bag', has_readable_contour=False)])
        return original(path, *a, **kw)
    for module in (serial, parallel, requiredness):
        monkeypatch.setattr(module, 'inspect_rtstruct', inspect)
    error = None
    try:
        call_consumer(backend, course, cfg)
    except Exception as exc:
        error = exc
    assert inspected == []
    assert isinstance(error, RadiomicsCourseExtractionError)
    assert 'RS_custom' in str(error) and 'stale' in str(error)


@pytest.mark.parametrize('backend', ['serial', 'parallel', 'conda'])
def test_current_unreadable_required_custom_fails(tmp_path, monkeypatch, rasterizer_guard_stub, backend):
    from rtpipeline import roi_requiredness as requiredness
    course, cfg = consumer_fixture(tmp_path, monkeypatch)
    cfg.radiomics_analysis_contract = {'CT': {'required_rois': [
        {'canonical_name': 'bowel_bag', 'source': 'Custom'}]}}
    for module in (custom, serial, parallel):
        monkeypatch.setattr(module, '_is_rs_custom_stale', lambda *a: False)
    original = requiredness.inspect_rtstruct
    def inspect(path, *a, **kw):
        if Path(path).name == 'RS_custom.dcm':
            return SimpleNamespace(named_rois=[SimpleNamespace(name='bowel_bag', has_readable_contour=False)])
        return original(path, *a, **kw)
    for module in (serial, parallel, requiredness):
        monkeypatch.setattr(module, 'inspect_rtstruct', inspect)
    with pytest.raises(RadiomicsCourseExtractionError, match='failed_custom_read'):
        call_consumer(backend, course, cfg)
    assert not (course / 'radiomics_ct.parquet').exists()


@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('artifact', ['missing', 'stale'])
def test_robustness_custom_required_before_reading(tmp_path, monkeypatch, configured, artifact):
    from rtpipeline import radiomics_robustness as rr
    from test_robustness_nonmeasurements import course_fixture
    course, cfg, rob, _ = course_fixture(tmp_path, monkeypatch)
    contract = rr.load_course_contract(course)
    contract.authoritative_rtstruct_path = None
    monkeypatch.setattr(rr, "load_course_contract", lambda _: contract)
    if configured:
        cfg.custom_structures_config = custom_config(tmp_path)
    else:
        monkeypatch.setattr(rr, '_load_main_ct_identity_catalog', lambda *a, **k: ({('Custom', 'ROI'): object()}, {}))
    if artifact == 'stale':
        (course / 'RS_custom.dcm').write_bytes(b'legacy')
    monkeypatch.setattr(custom, '_is_rs_custom_stale', lambda *a, **k: True)
    with pytest.raises(RuntimeError, match='RS_custom.*(missing|stale)'):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / 'radiomics_robustness_ct.parquet').exists()


def test_dvh_failure_closes_radiomics_before_launch(tmp_path, monkeypatch):
    from test_campaign_mode import _workflow, _run, _set_stage_configuration
    workflow = _workflow(tmp_path, campaign_mode=True)
    workflow.params.stage = 'radiomics'
    _set_stage_configuration(workflow, tmp_path, 'radiomics')
    workflow.output.sentinel = str(tmp_path / 'PT001/COURSE_A/.radiomics_done')
    upstream = tmp_path / 'PT001/COURSE_A/.dvh_done'
    upstream.write_text('failed\n')
    workflow.input.dvh = str(upstream)
    launch = Mock(side_effect=AssertionError('must not launch after failed DVH'))
    monkeypatch.setattr('subprocess.run', launch)
    with pytest.raises(SystemExit) as exc:
        _run(workflow)
    assert exc.value.code == 0
    launch.assert_not_called()
    assert Path(workflow.output.sentinel).read_text().strip() == 'failed'
    record = json.loads(next((tmp_path / '_campaign_ledger/records').glob('*.json')).read_text())
    assert 'upstream_dvh_failed' in record['detail']


@pytest.mark.parametrize('error_kind', ['timeout', 'extraction'])
def test_child_errors_keep_type_message_and_code(tmp_path, monkeypatch, error_kind):
    error = (conda.RadiomicsEnvironmentProbeTimeout(['python', 'probe'], 7, 2)
             if error_kind == 'timeout' else RadiomicsCourseExtractionError('[REQUIRED_ROI_NOT_DECLARED] bowel_bag'))
    monkeypatch.setattr(conda, '_radiomics_env_prefix', lambda: str(tmp_path))
    monkeypatch.setattr(runtime, 'needs_radiomics_process', lambda: False)
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(conda, 'radiomics_for_course', fail)
    payloads = []
    def child(command, **kwargs):
        monkeypatch.setattr(runtime.sys, 'argv', command[2:])
        try:
            returncode = runtime.main()
        except SystemExit as exc:
            returncode = exc.code
        except Exception:
            returncode = 1
        result = Path(command[-1])
        if result.exists():
            payloads.append(pickle.loads(result.read_bytes()))
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(runtime.subprocess, 'run', child)
    with pytest.raises(type(error)) as caught:
        runtime.run_course_driver('ct', tmp_path, None)
    assert str(caught.value) == str(error)
    assert getattr(caught.value, 'code', None) == getattr(error, 'code', None)
    assert payloads[0]['ok'] is False
    assert payloads[0]['type'] == type(error).__name__
    assert payloads[0]['message'] == str(error)
    assert payloads[0]['code'] == getattr(error, 'code', None)


def test_parallel_delegation_forwards_worker_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'needs_radiomics_process', lambda: True)
    driver = Mock(return_value=None)
    monkeypatch.setattr(runtime, 'run_course_driver', driver)
    parallel.parallel_radiomics_for_course(fx._config(tmp_path), tmp_path, max_workers=2)
    assert driver.call_args.kwargs['max_workers'] == 2


def test_conda_unreadable_ct_records_ledger(tmp_path, monkeypatch, rasterizer_guard_stub):
    course = tmp_path / 'P1/C1'
    fx._write_contract(course)
    fx._write_current_auto_rtstruct(course)
    def unreadable(*a):
        raise RuntimeError('synthetic CT read failure')
    monkeypatch.setattr(conda.sitk, 'ImageSeriesReader', unreadable)
    ledger = Mock(wraps=conda._write_conda_roi_ledger)
    monkeypatch.setattr(conda, '_write_conda_roi_ledger', ledger)
    with pytest.raises(RadiomicsCourseExtractionError, match='CT series is present but unreadable'):
        conda.radiomics_for_course(course, fx._config(tmp_path))
    ledger.assert_called_once()
    assert ledger.call_args.kwargs['missing_reason'] == 'failed_source_read'
    payload = json.loads((course / 'metadata/radiomics_ct_roi_ledger.json').read_text())
    assert payload['course'][0]['reason_code'] == 'failed_source_read'


def test_env_lookup_uses_interpreter_sibling(tmp_path, monkeypatch):
    for name in ('CONDA_PREFIX', 'MAMBA_ROOT_PREFIX', 'RTPIPELINE_RADIOMICS_ENV_PREFIX'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path / 'service-home'))
    prefix = tmp_path / 'envs' / conda.RADIOMICS_ENV
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'bin/python').touch()
    monkeypatch.setattr(runtime.sys, 'prefix', str(tmp_path / 'envs/main'))
    assert conda._radiomics_env_prefix() == str(prefix)


@pytest.mark.parametrize('rule_index', [0, 1])
@pytest.mark.parametrize('status', ['failed', 'missing', 'ok', 'disabled'])
def test_robustness_shell_requires_successful_dvh(tmp_path, rule_index, status):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    snakefile = (root / 'Snakefile').read_text()
    rule = snakefile.split('rule radiomics_robustness_course:')[rule_index + 1]
    shell = rule.split('shell:', 1)[1].split('"""')[1]
    gate = shell.split('            # Export worker budget', 1)[0]
    course = tmp_path / 'P1/C1'
    course.mkdir(parents=True)
    dvh_sentinel = course / '.dvh_done'
    if status != 'missing':
        dvh_sentinel.write_text(status + '\n')
    radiomics_sentinel = course / '.radiomics_done'
    radiomics_sentinel.write_text('ok\n')
    output = course / '.radiomics_robustness_done'
    marker = course / 'launched'
    command = gate.format(
        params=SimpleNamespace(enabled='True', campaign_mode='True', root_dir=root,
                               python=sys.executable, output_dir=tmp_path),
        input=SimpleNamespace(dvh=dvh_sentinel, radiomics=radiomics_sentinel),
        output=SimpleNamespace(sentinel=output), log=tmp_path / 'robustness.log',
        wildcards=SimpleNamespace(patient='P1', course='C1'),
    ) + f'\ntouch "{marker}"\n'
    result = subprocess.run(['bash', '-euc', command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if status in {'ok', 'disabled'}:
        assert marker.exists()
    else:
        assert not marker.exists()
        assert output.read_text().strip() == 'failed'
        record = json.loads(next((tmp_path / '_campaign_ledger/records').glob('*.json')).read_text())
        assert record['detail'] == 'upstream_dvh_failed'


def test_conda_worker_cap_reaches_extraction(tmp_path, monkeypatch, rasterizer_guard_stub):
    course, _ = fx._prepare_conda_rtstruct_course(tmp_path, monkeypatch)
    # A failed source is represented as a task; no actual mask reader is needed
    # to observe the budget passed to the batch extractor.
    fx._install_rt_utils(monkeypatch, Mock(side_effect=ValueError('synthetic source failure')))
    class ReachedBatch(Exception):
        pass
    calls = []
    def batch(*args, **kwargs):
        calls.append(kwargs['max_workers'])
        raise ReachedBatch
    monkeypatch.setattr(conda, 'process_radiomics_batch', batch)
    monkeypatch.setenv('RTPIPELINE_MAX_WORKERS', '8')
    with pytest.raises(ReachedBatch):
        conda.radiomics_for_course(course, fx._config(tmp_path), max_workers=2)
    assert calls == [2]


def test_delegated_timeout_keeps_cli_exit_three(tmp_path, monkeypatch, capsys):
    from rtpipeline import cli
    monkeypatch.setattr(conda, '_radiomics_env_prefix', lambda: str(tmp_path))
    error = conda.RadiomicsEnvironmentProbeTimeout(['python', 'probe'], 7, 2)
    def child(command, **kwargs):
        Path(command[-1]).write_bytes(pickle.dumps({
            'ok': False, 'type': type(error).__name__, 'message': str(error), 'code': error.code,
            'details': {'command': list(error.command), 'timeout': error.timeout, 'attempts': error.attempts},
        }))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runtime.subprocess, 'run', child)
    monkeypatch.setattr(cli, 'main', lambda argv: runtime.run_course_driver('ct', tmp_path, None))
    assert cli.console_main([]) == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload == {'code': error.code, 'message': str(error)}


def test_real_child_preserves_extraction_diagnosis(tmp_path, monkeypatch):
    prefix = conda._radiomics_env_prefix()
    if prefix is None:
        pytest.skip('radiomics interpreter absent')
    monkeypatch.setenv('RTPIPELINE_RADIOMICS_ENV_PREFIX', prefix)
    course = tmp_path / 'P1/C1'
    fx._write_contract(course, no_ct=True)
    with pytest.raises(RadiomicsCourseExtractionError,
                       match='all-series dispatcher requires an all-series temporary contract'):
        runtime.run_course_driver('ct', course, fx._config(tmp_path), allow_all_series_temp=True)


def test_parallel_native_fallback_forwards_worker_cap(tmp_path, monkeypatch, rasterizer_guard_stub):
    course = tmp_path / 'P1/C1'
    fx._write_contract(course)
    monkeypatch.setattr(serial, '_load_series_image', lambda *a: object())
    monkeypatch.setattr(serial, '_extractor', lambda *a: None)
    driver = Mock(return_value=None)
    monkeypatch.setattr(conda, 'radiomics_for_course', driver)
    parallel.parallel_radiomics_for_course(fx._config(tmp_path), course, max_workers=2)
    assert driver.call_args.kwargs['max_workers'] == 2


def test_conda_voxel_fallback_forwards_worker_cap(tmp_path, monkeypatch, rasterizer_guard_stub):
    course = tmp_path / 'P1/C1'
    fx._write_contract(course)
    driver = Mock(return_value=None)
    monkeypatch.setattr(conda, 'radiomics_for_course_ct_nifti_fallback', driver)
    conda.radiomics_for_course(course, fx._config(tmp_path), max_workers=2)
    assert driver.call_args.kwargs['max_workers'] == 2
