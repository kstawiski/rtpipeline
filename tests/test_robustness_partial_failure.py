"""Partial synthetic extraction evidence closes as failure, never measurement."""
import json

import pandas as pd
import pytest
import yaml

from rtpipeline import cli, radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from workflow.scripts import campaign_ledger
from test_robustness_nonmeasurements import _real_mixed_course


def _failed_course(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(
        tmp_path, monkeypatch, technical_failure_conditions=1)
    extract = rr.extract_features_for_masks

    def features(*args, **kwargs):
        frame = extract(*args, **kwargs)
        frame['run_identifier'] = kwargs['run_identifier']
        return frame

    monkeypatch.setattr(rr, 'extract_features_for_masks', features)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'radiomics_robustness': {
        'enabled': True,
        'segmentation_perturbation': {
            'apply_to_structures': ['ROI'],
            'small_volume_changes': [0., .15],
            'max_translation_mm': 0.,
            'n_random_contour_realizations': 0,
            'noise_levels': [0.],
        },
    }}))
    output = course / 'radiomics_robustness_ct.parquet'
    sentinel = rc.robustness_completion_sentinel_path(course)
    assert cli.main(['radiomics-robustness', '--course-dir', str(course),
                     '--config', str(config), '--output', str(output),
                     '--sentinel', str(sentinel), '--campaign-mode']) == 1
    assert sentinel.is_file(), 'technical failure must leave a failure receipt'
    return course, cfg, rob, output, sentinel


def test_partial_technical_failure_closes_campaign_without_measurement(tmp_path, monkeypatch):
    course, _, rob, output, sentinel = _failed_course(tmp_path, monkeypatch)
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    assert receipt.measurement_outcome == rr.ROBUSTNESS_FAILED_OUTCOME
    assert not receipt.measured and receipt.measured_output is None
    assert not output.exists()
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    evidence = payload['failed_evidence']
    partial = course / evidence['path']
    assert partial.name == 'radiomics_robustness_ct.failed_evidence.parquet'
    assert evidence['sha256'] == rr._file_sha256(partial)
    assert evidence['size_bytes'] == partial.stat().st_size
    frame = pd.read_parquet(partial)
    assert set(frame.robustness_status) == {'measured', 'technical_failure'}
    assert frame.loc[frame.robustness_status.eq('technical_failure'), 'value'].isna().all()
    assert set(frame.run_identifier) == {receipt.run_identifier}
    ledger = campaign_ledger.close_robustness_failed_extraction(
        course.parent.parent, 'P', 'C', sentinel)
    assert json.loads(ledger.read_text())['status'] == 'failed'
    admitted = rr.admit_robustness_cohort_course(
        course, patient_id='P', course_id='C', rob_config=rob)
    assert admitted.frame is None and admitted.measured_output_sha256 is None
    counts = rr._robustness_course_outcome_rows([admitted]).iloc[0]
    assert counts['table_row_count'] == counts['measured_value_row_count'] == 0
    with pytest.raises((ValueError, RuntimeError)):
        rr._admit_robustness_aggregation_input(partial, rob)


@pytest.mark.parametrize('corruption', ['missing', 'changed', 'symlink', 'relabelled'])
def test_partial_evidence_corruption_blocks_receipt_and_ledger(tmp_path, monkeypatch, corruption):
    course, _, rob, output, sentinel = _failed_course(tmp_path, monkeypatch)
    partial = course / 'radiomics_robustness_ct.failed_evidence.parquet'
    if corruption == 'missing':
        partial.unlink()
    elif corruption == 'changed':
        partial.write_bytes(b'changed synthetic evidence')
    elif corruption == 'symlink':
        target = course / 'other.parquet'
        partial.rename(target)
        partial.symlink_to(target)
    else:
        # Even copying failed evidence into the nominal measurement slot
        # cannot turn this course into a measurement.
        output.write_bytes(partial.read_bytes())
    with pytest.raises((ValueError, RuntimeError)):
        rc.read_robustness_completion_sentinel(sentinel)
    with pytest.raises((ValueError, RuntimeError)):
        campaign_ledger.close_robustness_failed_extraction(
            course.parent.parent, 'P', 'C', sentinel)
    with pytest.raises((ValueError, RuntimeError)):
        rr.admit_robustness_cohort_course(
            course, patient_id='P', course_id='C', rob_config=rob)


def test_failed_course_aggregate_contains_no_partial_measurements(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_robustness_consumer_integration import _write_manifest, _run_aggregate

    course, _, _, _, _ = _failed_course(tmp_path, monkeypatch)
    cohort = SimpleNamespace(output_root=course.parent.parent,
                             courses={('P', 'C'): course},
                             config_path=tmp_path / 'config.yaml')
    manifest = _write_manifest(cohort)
    summary = tmp_path / 'summary.xlsx'
    assert _run_aggregate(cohort, manifest, summary) == 0
    raw = pd.read_parquet(tmp_path / 'summary_raw_values.parquet')
    assert raw.empty
    outcomes = pd.read_excel(summary, sheet_name='course_outcomes')
    assert outcomes.loc[0, 'measurement_outcome'] == rr.ROBUSTNESS_FAILED_OUTCOME
    assert outcomes.loc[0, 'measured_value_row_count'] == 0


def test_rerun_invalidates_old_partial_evidence_before_loading_contract(tmp_path, monkeypatch):
    course, cfg, rob, output, _ = _failed_course(tmp_path, monkeypatch)
    partial = course / 'radiomics_robustness_ct.failed_evidence.parquet'
    assert partial.exists()

    def fail_contract(*args, **kwargs):
        raise RuntimeError('synthetic contract failure')

    monkeypatch.setattr(rr, 'load_course_contract', fail_contract)
    with pytest.raises(RuntimeError, match='synthetic contract failure'):
        rr.robustness_for_course(cfg, rob, course)
    assert not partial.exists() and not output.exists()
    assert not rr.robustness_source_dispositions_path(course).exists()


def test_failure_writer_still_refuses_nominal_measurement_table(tmp_path, monkeypatch):
    course, _, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    output = course / 'radiomics_robustness_ct.parquet'
    output.write_bytes(b'synthetic existing table')
    with pytest.raises(RuntimeError, match='existing measurement table'):
        rr.write_robustness_failure_dispositions(
            course, rob_config=rob, output_name=output.name)
    assert output.read_bytes() == b'synthetic existing table'
