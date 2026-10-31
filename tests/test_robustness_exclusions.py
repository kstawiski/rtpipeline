"""Configuration exclusions with synthetic source bytes and real admission."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pydicom
import pytest
import yaml

from rtpipeline import cli
from rtpipeline import radiomics as rm
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.roi_requiredness import Requiredness
from test_robustness_consumer_integration import (
    _build_courses, _rob_config, _roi_array, _run_aggregate, _run_course,
    _summary_paths, _write_manifest,
)
from test_robustness_selection_unmatched import _ledger
from test_robustness_targets import _source


def _configure(cohort, mode, exclusions):
    data = yaml.safe_load(cohort.config_path.read_text())
    section = data['radiomics_robustness']
    section['target_selection'] = mode
    section['segmentation_perturbation']['exclude_structures'] = exclusions
    cohort.config_path.write_text(yaml.safe_dump(data))


def _append_roi(path, name, kind, number=2):
    ds = pydicom.dcmread(path)
    roi = copy.deepcopy(ds.StructureSetROISequence[0])
    roi.ROINumber = number
    roi.ROIName = name
    ds.StructureSetROISequence.append(roi)
    contour = copy.deepcopy(ds.ROIContourSequence[0])
    contour.ReferencedROINumber = number
    ds.ROIContourSequence.append(contour)
    observation = copy.deepcopy(ds.RTROIObservationsSequence[0])
    observation.ObservationNumber = number
    observation.ReferencedROINumber = number
    observation.RTROIInterpretedType = kind
    ds.RTROIObservationsSequence.append(observation)
    ds.save_as(path)


@pytest.mark.parametrize('value', [None, '', 'RING', ('RING',), {}, 1, True, [''], ['  '], [None], [2], ['RING', False]])
def test_invalid_exclusions_rejected_at_configuration_and_cli_boundaries(value):
    with pytest.raises(ValueError, match='exclude_structures'):
        rr.PerturbationConfig(exclude_structures=value)
    section = {'segmentation_perturbation': {'exclude_structures': value}}
    with pytest.raises(ValueError, match='exclude_structures'):
        rr.RobustnessConfig.from_dict(section)
    with pytest.raises(cli._RobustnessRequestError, match='exclude_structures'):
        cli._validate_robustness_section(section)


@pytest.mark.parametrize('mode', ['patterns', 'targets'])
@pytest.mark.parametrize('name,pattern,excluded', [
    ('RING', 'ring', True), ('Ring', 'R?NG', True), ('RING', '[RP]ING', True),
    ('RING', 'RIN', False), ('RING_extra', 'ring', False), ('xRING', 'ring', False),
    ('z_PTV_opt', 'z_ptv_*', True), ('PTV-ring', '*-RING', True),
    ('PTV1', 'PTV?', True), (' RING', 'RING', False),
])
def test_exclusions_follow_selection_with_whole_name_fnmatch(mode, name, pattern, excluded):
    assert rr.robustness_selection_matches(name, ['*'], target_selection=mode, interpreted_types=['PTV'])
    assert rr.robustness_selection_matches(
        name, ['*'], target_selection=mode, interpreted_types=['PTV'], exclude_structures=[pattern],
    ) is not excluded
    # Exclusion cannot add an otherwise unselected ROI.
    assert not rr.robustness_selection_matches('ORGAN', ['PTV*'], target_selection=mode, exclude_structures=[pattern])


def test_empty_exclusions_preserve_main_configuration_and_parameter_hash_bytes():
    frozen = json.loads((Path(__file__).parent / 'fixtures/robustness_main_default_identity.json').read_text())
    assert frozen['revision'] == '072b146'
    for case in frozen['cases']:
        # An absent robustness section has its own pre-existing enabled default.
        for explicit_empty in ([False, True] if case['input'] else [False]):
            data = copy.deepcopy(case['input'])
            if explicit_empty:
                data.setdefault('segmentation_perturbation', {})['exclude_structures'] = []
            cfg = rr.RobustnessConfig.from_dict(data)
            assert cfg.perturbation.exclude_structures == []
            effective = rr.effective_robustness_configuration(cfg, output_name='radiomics_robustness_ct.parquet')
            assert json.dumps(effective, sort_keys=True).encode() == case['effective_configuration_json'].encode()
            assert rr._content_sha256(effective).encode() == case['effective_configuration_sha256'].encode()
            assert rr._robustness_configured_parameter_hash('synthetic-extractor', cfg.target_selection, []).encode() == case['configured_parameter_hash'].encode()
            selected = [name for name in frozen['names'] if rr.robustness_selection_matches(
                name, cfg.perturbation.apply_to_structures, target_selection=cfg.target_selection,
                interpreted_types=['PTV'] if name == 'RING' else [], exclude_structures=cfg.perturbation.exclude_structures,
            )]
            assert json.dumps(selected).encode() == case['selected_json'].encode()
    default = rr.RobustnessConfig(perturbation=rr.PerturbationConfig(exclude_structures=[]))
    frozen_default = next(case for case in frozen['cases'] if case['input'] == {})
    assert json.dumps(rr.effective_robustness_configuration(default, output_name='radiomics_robustness_ct.parquet'), sort_keys=True).encode() == frozen_default['effective_configuration_json'].encode()
    first, second = rr.PerturbationConfig(), rr.PerturbationConfig()
    first.exclude_structures.append('RING')
    assert second.exclude_structures == []


@pytest.mark.parametrize('mode', ['patterns', 'targets'])
@pytest.mark.parametrize('name,kind', [('RING', 'PTV'), ('z_PTV_opt', 'ORGAN'), ('PTV-ring', 'PTV')])
def test_all_excluded_course_is_not_applicable_with_source_bound_receipt(tmp_path, monkeypatch, mode, name, kind):
    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['*'])
    _source(cohort, 'P1', name, kind)
    _configure(cohort, mode, [name.lower()])

    def masks(*args, **kwargs):
        assert kwargs['requiredness_by_roi'][name] == Requiredness.INVENTORY_ONLY
        assert not kwargs['retain_mask'](name)
        return {}

    monkeypatch.setattr(rm, '_rtstruct_masks', masks)
    assert _run_course(cohort, 'P1', sentinel=False, campaign=True) == 0
    receipt = cohort.receipt('P1')
    assert receipt['measurement_outcome'] == rc.ROBUSTNESS_NO_TARGET_OUTCOME
    assert receipt['exclude_structures'] == [name.lower()]
    assert receipt['missed_targets'] == []
    excluded = receipt['excluded_by_configuration']
    assert len(excluded) == 1
    assert excluded[0]['roi_name'] == name and excluded[0]['segmentation_source'] == 'Manual'
    assert excluded[0]['roi_number'] == '1'
    assert excluded[0]['detector_rules'] == list(rr.detect_robustness_target(name, kind))
    assert excluded[0]['source_path'] == str(cohort.course_dir('P1') / 'RS.dcm')
    sidecar = json.loads(rr.robustness_source_dispositions_path(cohort.course_dir('P1')).read_text())
    assert sidecar['excluded_by_configuration'] == excluded
    assert sidecar['exclude_structures'] == [name.lower()]
    assert sidecar['source_only_basis']['selected_structure_count'] == 0
    assert not cohort.table('P1').exists()
    validated = rc.read_robustness_completion_sentinel(cohort.sentinel('P1'))
    assert validated.exclude_structures == (name.lower(),)
    cfg = _rob_config(cohort.config_path)
    admitted = rr.admit_robustness_cohort_course(cohort.course_dir('P1'), patient_id='P1', course_id='C1', rob_config=cfg)
    assert not admitted.measured
    ledger = _ledger()
    record = ledger.close_robustness_completion(cohort.output_root, 'P1', 'C1', cohort.sentinel('P1'))
    assert json.loads(record.read_text())['status'] == 'not_applicable'
    output, raw = _summary_paths(tmp_path)
    assert _run_aggregate(cohort, _write_manifest(cohort), output) == 0
    counts = pd.read_excel(output, sheet_name='course_counts').iloc[0]
    assert counts.courses == 1 and counts.not_applicable == 1 and counts.technically_missing == 0
    assert pd.read_parquet(raw).empty
    changed = replace(cfg, perturbation=replace(cfg.perturbation, exclude_structures=[]))
    with pytest.raises(RuntimeError, match='configuration'):
        rr.admit_robustness_cohort_course(cohort.course_dir('P1'), patient_id='P1', course_id='C1', rob_config=changed)


def test_excluded_target_does_not_hide_unexcluded_unselected_target(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['RING'])
    _source(cohort, 'P1', 'RING', 'PTV')
    _append_roi(cohort.course_dir('P1') / 'RS.dcm', 'HR-CTV', 'TREATED_VOLUME')
    _configure(cohort, 'patterns', ['ring'])
    monkeypatch.setattr(rm, '_rtstruct_masks', lambda *a, **kw: {})
    assert _run_course(cohort, 'P1') == 0
    receipt = cohort.receipt('P1')
    assert receipt['measurement_outcome'] == rc.ROBUSTNESS_SELECTION_GAP_OUTCOME
    assert [r['roi_name'] for r in receipt['missed_targets']] == ['HR-CTV']
    assert [r['roi_name'] for r in receipt['excluded_by_configuration']] == ['RING']
    assert rc.read_robustness_completion_sentinel(cohort.sentinel('P1')).measurement_outcome == rc.ROBUSTNESS_SELECTION_GAP_OUTCOME


@pytest.mark.parametrize('mode', ['patterns', 'targets'])
def test_partial_exclusion_measured_provenance_hashes_and_admission(tmp_path, monkeypatch, mode):
    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['ROI', 'RING'])
    _source(cohort, 'P1', 'ROI', 'ORGAN')
    _append_roi(cohort.course_dir('P1') / 'RS.dcm', 'RING', 'PTV')
    _configure(cohort, mode, ['ring'])
    seen = []

    def masks(*args, **kwargs):
        seen.append(kwargs['requiredness_by_roi'])
        assert kwargs['retain_mask']('ROI')
        assert not kwargs['retain_mask']('RING')
        return {'ROI': _roi_array()}

    monkeypatch.setattr(rm, '_rtstruct_masks', masks)
    assert _run_course(cohort, 'P1') == 0
    assert seen == [{'ROI': Requiredness.ANALYSIS_REQUIRED, 'RING': Requiredness.INVENTORY_ONLY}]
    frame = pd.read_parquet(cohort.table('P1'))
    cfg = _rob_config(cohort.config_path)
    assert set(frame.roi_original_name) == {'ROI'}
    assert frame.exclude_structures.eq('["ring"]').all()
    expected = rr._selection_bound_parameter_hashes(frame.extraction_configured_parameter_hash, mode, ['ring'])
    assert frame.configured_parameter_hash.eq(expected).all()
    assert not frame.configured_parameter_hash.eq(rr._selection_bound_parameter_hashes(frame.extraction_configured_parameter_hash, mode)).any()
    receipt = cohort.receipt('P1')
    assert receipt['exclude_structures'] == ['ring']
    assert [r['roi_name'] for r in receipt['excluded_by_configuration']] == ['RING']
    assert rr.admit_robustness_cohort_course(cohort.course_dir('P1'), patient_id='P1', course_id='C1', rob_config=cfg).measured
    changed = replace(cfg, perturbation=replace(cfg.perturbation, exclude_structures=['z_*']))
    assert rr.effective_robustness_configuration(cfg, output_name='x') != rr.effective_robustness_configuration(changed, output_name='x')
    with pytest.raises(ValueError, match='configuration'):
        rr.load_robustness_source_dispositions(cohort.course_dir('P1'), run_identifier=receipt['robustness_run_identifier'], rob_config=changed)
    with pytest.raises(RuntimeError, match='configuration'):
        rr.admit_robustness_cohort_course(cohort.course_dir('P1'), patient_id='P1', course_id='C1', rob_config=changed)
    output, raw = _summary_paths(tmp_path)
    assert _run_aggregate(cohort, _write_manifest(cohort), output) == 0
    assert pd.read_parquet(raw).exclude_structures.eq('["ring"]').all()


@pytest.mark.parametrize('change', ['receipt-patterns', 'receipt-evidence', 'sidecar-patterns', 'sidecar-evidence'])
def test_no_match_receipt_rejects_rebound_exclusion_tampering(tmp_path, monkeypatch, change):
    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['*'])
    _source(cohort, 'P1', 'RING', 'PTV')
    _configure(cohort, 'targets', ['ring'])
    monkeypatch.setattr(rm, '_rtstruct_masks', lambda *a, **kw: {})
    assert _run_course(cohort, 'P1') == 0
    receipt = cohort.receipt('P1')
    field = 'exclude_structures' if change.endswith('patterns') else 'excluded_by_configuration'
    if change.startswith('receipt'):
        receipt[field] = []
    else:
        path = rr.robustness_source_dispositions_path(cohort.course_dir('P1'))
        sidecar = json.loads(path.read_text())
        sidecar[field] = []
        path.write_text(json.dumps(sidecar))
        receipt[field] = []
        receipt['source_dispositions']['sha256'] = rr._file_sha256(path)
    cohort.write_receipt('P1', receipt)
    with pytest.raises(rc.RobustnessCompletionError, match='exclude'):
        rc.read_robustness_completion_sentinel(cohort.sentinel('P1'))


@pytest.mark.parametrize('field,tamper', [
    (field, tamper)
    for field in ['exclude_structures', 'configured_parameter_hash', 'robustness_configuration_sha256', 'target_selection']
    for tamper in ['foreign', 'null']
] + [('exclude_structures', 'missing')])
def test_rebound_measured_row_exclusion_tampering_rejected(tmp_path, monkeypatch, field, tamper):
    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'})
    _configure(cohort, 'patterns', ['ring'])
    assert _run_course(cohort, 'P1') == 0
    frame = pd.read_parquet(cohort.table('P1'))
    if tamper == 'missing':
        frame = frame.drop(columns=[field])
    else:
        frame[field] = (pd.Series(pd.NA, index=frame.index, dtype='string') if tamper == 'null'
                        else '[]' if field == 'exclude_structures' else '0' * 64)
    frame.to_parquet(cohort.table('P1'), index=False)
    path = rr.robustness_source_dispositions_path(cohort.course_dir('P1'))
    sidecar = json.loads(path.read_text())
    sidecar['measured_output']['sha256'] = rr._file_sha256(cohort.table('P1'))
    sidecar['measured_output']['size_bytes'] = cohort.table('P1').stat().st_size
    path.write_text(json.dumps(sidecar))
    receipt = cohort.receipt('P1')
    receipt['source_dispositions']['sha256'] = rr._file_sha256(path)
    receipt['measured_output'] = sidecar['measured_output']
    cohort.write_receipt('P1', receipt)
    with pytest.raises(ValueError, match='exclude_structures|configuration hash|target_selection'):
        rr.admit_robustness_cohort_course(cohort.course_dir('P1'), patient_id='P1', course_id='C1', rob_config=_rob_config(cohort.config_path))


@pytest.mark.parametrize('mode', ['patterns', 'targets'])
def test_exclusions_and_evidence_cover_each_source_and_roi(tmp_path, monkeypatch, mode):
    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['*'])
    _source(cohort, 'P1', 'RING', 'PTV')
    _append_roi(cohort.course_dir('P1') / 'RS.dcm', 'RING_extra', 'ORGAN')
    second = cohort.course_dir('P1') / 'second.dcm'
    ds = pydicom.dcmread(cohort.course_dir('P1') / 'RS.dcm')
    ds.RTROIObservationsSequence[0].RTROIInterpretedType = 'ORGAN'
    ds.save_as(second)
    _configure(cohort, mode, ['ring'])
    inventory = [roi for source, path in [('Manual', cohort.course_dir('P1') / 'RS.dcm'), ('AutoRTS_total', second)]
                 for roi in rr._robustness_target_inventory({'segmentation_source': source, 'source_path': str(path)})]
    for path in [cohort.course_dir('P1') / 'RS.dcm', second]:
        required = rr._robustness_selection_requiredness(path, ['*'], target_selection=mode,
                                                       target_inventory=inventory, exclude_structures=['ring'])
        assert required == {'RING': Requiredness.INVENTORY_ONLY, 'RING_extra': Requiredness.ANALYSIS_REQUIRED}
    # Retain both independent source identities in the exclusion evidence.
    monkeypatch.setattr(rm, '_standard_rtstruct_sources', lambda *a: [
        ('Manual', cohort.course_dir('P1') / 'RS.dcm', None), ('AutoRTS_total', second, None),
    ])
    # RING_extra is not selected in patterns mode with the narrower request;
    # in targets mode it is not a target, so the result is also no target.
    data = yaml.safe_load(cohort.config_path.read_text())
    data['radiomics_robustness']['segmentation_perturbation']['apply_to_structures'] = ['RING']
    cohort.config_path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(rm, '_rtstruct_masks', lambda *a, **kw: {})
    assert _run_course(cohort, 'P1') == 0
    receipt = cohort.receipt('P1')
    assert receipt['measurement_outcome'] == rc.ROBUSTNESS_NO_TARGET_OUTCOME
    assert [(r['segmentation_source'], r['roi_name'], r['roi_number']) for r in receipt['excluded_by_configuration']] == [
        ('Manual', 'RING', '1'), ('AutoRTS_total', 'RING', '1'),
    ]
    assert receipt['excluded_by_configuration'][0]['detector_rules'] == ['RTROIInterpretedType:PTV']
    assert receipt['excluded_by_configuration'][1]['detector_rules'] == []


@pytest.mark.parametrize('mode', ['patterns', 'targets'])
def test_nifti_exclusions_close_no_target_and_revalidate_source_bytes(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    import SimpleITK as sitk

    cohort = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['HR-CTV'])
    _configure(cohort, mode, ['hr-ctv'])
    course = cohort.course_dir('P1')
    ct = course / 'ct.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(_roi_array().astype('float32')), str(ct))
    seg = course / 'Segmentation_TotalSegmentator' / 'total--HR-CTV.nii.gz'
    seg.parent.mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(_roi_array()), str(seg))
    monkeypatch.setattr(rm, '_load_series_image', lambda *a: None)
    monkeypatch.setattr(rr, 'load_course_contract', lambda *a: SimpleNamespace(
        planning_ct_dir=None, planning_ct_nifti=ct, planning_ct={'series_instance_uid': '1.2.3'},
    ))
    assert _run_course(cohort, 'P1') == 0
    receipt = cohort.receipt('P1')
    assert receipt['measurement_outcome'] == rc.ROBUSTNESS_NO_TARGET_OUTCOME
    assert receipt['missed_targets'] == []
    evidence = receipt['excluded_by_configuration']
    assert len(evidence) == 1
    assert evidence[0]['roi_name'] == 'HR-CTV'
    assert evidence[0]['source_path'] == str(seg)
    assert evidence[0]['interpreted_types'] == []
    assert evidence[0]['detector_rules'] == ['name_token']
    cfg = _rob_config(cohort.config_path)
    rr.admit_robustness_cohort_course(course, patient_id='P1', course_id='C1', rob_config=cfg)
    seg.write_bytes(seg.read_bytes() + b'changed')
    with pytest.raises(rc.RobustnessCompletionError, match='stale'):
        rc.read_robustness_completion_sentinel(cohort.sentinel('P1'))
