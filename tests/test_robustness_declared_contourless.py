"""Synthetic contourless declarations survive publication without measurements."""
import json

import pandas as pd
import pydicom
import pytest
import yaml

from rtpipeline import cli, radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from test_robustness_nonmeasurements import _real_mixed_course, _rewrite


def _course(tmp_path, monkeypatch, *, empty_sequence):
    course, _, rob, source = _real_mixed_course(
        tmp_path, monkeypatch, apply_to_structures=('ROI', 'Marker1'))
    dataset = pydicom.dcmread(source)
    marker = dataset.ROIContourSequence[1]
    if empty_sequence:
        marker.ContourSequence = []
    else:
        dataset.ROIContourSequence.remove(marker)
    dataset.save_as(source)
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
            'apply_to_structures': ['ROI', 'Marker1'],
            'small_volume_changes': [0., .15],
            'max_translation_mm': 0.,
            'n_random_contour_realizations': 0,
            'noise_levels': [0.],
        },
    }}))
    return course, rob, config


@pytest.mark.parametrize('empty_sequence,code', [
    (False, 'ROI_DECLARED_NO_CONTOUR_ITEM'),
    (True, 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'),
])
def test_declared_contourless_roi_completes_and_is_admitted(
        tmp_path, monkeypatch, empty_sequence, code):
    course, rob, config = _course(tmp_path, monkeypatch, empty_sequence=empty_sequence)
    output = course / 'radiomics_robustness_ct.parquet'
    sentinel = rc.robustness_completion_sentinel_path(course)
    assert cli.main(['radiomics-robustness', '--course-dir', str(course),
                     '--config', str(config), '--output', str(output),
                     '--sentinel', str(sentinel)]) == 0
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    assert receipt.measured
    admitted = rr.admit_robustness_cohort_course(
        course, patient_id='P', course_id='C', rob_config=rob)
    assert set(admitted.frame.structure) == {'ROI'}
    row, = admitted.source_dispositions
    assert (row['status'], row['failure_kind'], row['structural_code']) == (
        'structural_nonmeasurement', 'declared_without_contour_data', code)
    assert row['roi_name'] == 'Marker1'
    assert not {'feature_name', 'value', 'voxels'} & row.keys()
    assert rr._robustness_source_disposition_rows([admitted]).iloc[0].structural_code == code
    # The aggregate input path independently revalidates the sidecar.
    assert rr._admit_robustness_aggregation_input(output, rob).source_dispositions == [row]
    assert set(pd.read_parquet(output).structure) == {'ROI'}


@pytest.mark.parametrize('field,value', [
    ('status', 'failed'),
    ('failure_kind', 'extractor_error'),
    ('structural_code', 'ROI_CONTOUR_PARTIALLY_UNPARSEABLE'),
    ('structural_code', 'ROI_EXTRACTION_FAILED'),
    ('structural_code', 'ROI_NONVOLUMETRIC_POINT'),
    ('structural_code', 'UNKNOWN'),
    ('status', 'nonvolumetric_nonmeasurement'),
])
def test_contourless_disposition_rejects_other_combinations(
        tmp_path, monkeypatch, field, value):
    course, rob, _ = _course(tmp_path, monkeypatch, empty_sequence=True)
    # Use the fixture's normal config via the CLI, as in the positive case.
    config = tmp_path / 'config.yaml'
    assert cli.main(['radiomics-robustness', '--course-dir', str(course),
                     '--config', str(config), '--output',
                     str(course / 'radiomics_robustness_ct.parquet')]) == 0
    path = rr.robustness_source_dispositions_path(course)
    payload = json.loads(path.read_text())
    payload['rows'][0][field] = value
    _rewrite(course, payload)  # repaired digest cannot bypass semantic checks
    with pytest.raises(ValueError):
        rr.load_robustness_source_dispositions(
            course, run_identifier=payload['robustness_run_identifier'], rob_config=rob)
    with pytest.raises(RuntimeError):
        rr._validate_source_disposition_rows(
            payload['rows'], bindings=payload['source_bindings'], error=RuntimeError)


def test_invalid_disposition_fails_before_feature_extraction(tmp_path, monkeypatch):
    course, _, config = _course(tmp_path, monkeypatch, empty_sequence=True)
    from rtpipeline import radiomics
    masks = radiomics._rtstruct_masks
    calls = []

    def invalid(*args, **kwargs):
        result = masks(*args, **kwargs)
        kwargs['failure_outcomes'][0]['failure_kind'] = 'extractor_error'
        return result

    monkeypatch.setattr(radiomics, '_rtstruct_masks', invalid)
    monkeypatch.setattr(rr, 'extract_features_for_masks', lambda *a, **k: calls.append(True))
    assert cli.main(['radiomics-robustness', '--course-dir', str(course),
                     '--config', str(config), '--output',
                     str(course / 'radiomics_robustness_ct.parquet')]) == 1
    assert not calls
    assert not rr.robustness_source_dispositions_path(course).exists()
