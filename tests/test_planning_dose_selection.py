"""Synthetic courses only; public tests contain no campaign identifiers."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import dvh_plan_target_fixtures as fx
from course_contract_test_utils import write_minimal_course_contract
from rtpipeline import dvh
from rtpipeline.course_contract import load_course_contract, classify_course_dose_completeness
from rtpipeline.planning_dose_selection import BASIS, SIDECAR, select_planning_dose, load_planning_dose_sidecar
from rtpipeline.repair_planning_dose_selection import repair_course, main


def edit_contract(course, change):
    path = course/'metadata/case_metadata.json'
    payload = json.loads(path.read_text())
    change(payload['course_contract'])
    path.write_text(json.dumps(payload, indent=2))


def edit_dicom(path, **values):
    ds = pydicom.dcmread(path)
    for key, value in values.items():
        if value is None:
            delattr(ds, key)
        else:
            setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


def make_course(tmp_path):
    course = fx.build_course(tmp_path, delivered=False)
    edit_dicom(course/'DICOM/CT/ct_000.dcm', FrameOfReferenceUID=fx.FRAME_OF_REFERENCE_UID)
    write_minimal_course_contract(course, selected_doses=[],
                                  authoritative_rtstruct=course/"DICOM/RTSTRUCT/rs.dcm")
    def update(contract):
        contract['dose_classification'] = {
            'classification': 'no_delivered_plan_dose', 'should_sum': False,
            # Historical archive paths are not opened or trusted as live input.
            'excluded_doses': ['/unavailable/archive/dose.dcm'],
            'warnings': ['Excluded every plan because RTRECORD objects exist but none references a course plan'],
        }
        contract['dose_completeness'] = classify_course_dose_completeness(
            selected_plans=contract['selected_plans'], selected_doses=[],
            dose_classification=contract['dose_classification'], dose_grid=None,
            per_plan_delivery=contract['delivery']['per_plan'],
            delivery_status=contract['delivery']['status'])
    edit_contract(course, update)
    return course


def inventory(course):
    return {str(p.relative_to(course)): p.read_bytes() for p in course.rglob('*') if p.is_file()}


def test_selection_is_read_only_and_repair_only_writes_sidecar(tmp_path, capsys):
    course = make_course(tmp_path)
    before = inventory(course)
    selected = select_planning_dose(course)
    assert selected.accepted, selected.reason_code
    assert selected.dose_path == course/'DICOM/RTDOSE/dose.dcm'
    assert main(['--course', str(course)]) == 0
    assert json.loads(capsys.readouterr().out)['outcomes'] == {'accepted': 1}
    assert inventory(course) == before
    assert main(['--course', str(course), '--apply']) == 0
    assert json.loads(capsys.readouterr().out)['outcomes'] == {'applied': 1}
    after = inventory(course)
    sidecar = json.loads(after.pop(SIDECAR))
    assert after == before
    assert sidecar['basis'] == BASIS and sidecar['dose_response_eligible'] is False
    assert sidecar['selected_dose_path'] == 'DICOM/RTDOSE/dose.dcm'
    assert len(sidecar['plan_uid_sha256']) == 64 and sidecar['code_revision']
    assert fx.PLAN_UID not in json.dumps(sidecar)
    assert repair_course(course, apply=True) == 'unchanged'
    assert load_planning_dose_sidecar(course).accepted


@pytest.mark.parametrize('kind', ['PLAN', 'PLAN_SUM', 'MULTI_PLAN'])
def test_whole_plan_types_accepted_with_exact_single_plan_coverage(tmp_path, kind):
    course = make_course(tmp_path)
    edit_dicom(course/'DICOM/RTDOSE/dose.dcm', DoseSummationType=kind)
    assert select_planning_dose(course).accepted


@pytest.mark.parametrize('classification', [
    'no_approved_plan_unapproved_only', 'no_approved_plan_rejected_only',
    'no_approved_plan_mixed_status', 'independent_delivered_plans_unreconciled',
    'no_doses', 'single_dose', '',
])
def test_ineligible_classifications(tmp_path, classification):
    course = make_course(tmp_path)
    edit_contract(course, lambda c: c['dose_classification'].update(classification=classification))
    assert select_planning_dose(course).reason_code == 'classification_not_eligible'


@pytest.mark.parametrize('count', [0, 2])
def test_selected_plan_count(tmp_path, count):
    course = make_course(tmp_path)
    edit_contract(course, lambda c: c.update(selected_plans=c['selected_plans']*count))
    assert select_planning_dose(course).reason_code == 'selected_plan_count_not_one'


@pytest.mark.parametrize('status', ['REJECTED', 'UNAPPROVED', 'approved', '', None, 'UNKNOWN'])
def test_exact_approval_required(tmp_path, status):
    course = make_course(tmp_path)
    edit_dicom(course/'DICOM/RTPLAN/plan.dcm', ApprovalStatus=status)
    assert select_planning_dose(course).reason_code == 'plan_not_approved'


@pytest.mark.parametrize('case, expected', [
    ('contract_selected', 'contract_selected_dose'),
    ('bad_contract', 'invalid_course_contract'),
    ('plan_unreadable', 'plan_unreadable'),
    ('plan_uid', 'plan_identity_mismatch'),
    ('qc_failed', 'organize_dose_qc_failed'),
    ('qc_inconsistent', 'invalid_course_contract'),
    ('ct_absent', 'planning_ct_unavailable'),
    ('ct_frame_missing', 'planning_ct_frame_unresolved'),
    ('ct_frames_mixed', 'planning_ct_frame_unresolved'),
    ('no_dose', 'no_available_dose'),
    ('dose_unreadable', 'dose_inventory_unreadable'),
    ('dose_wrong_class', 'dose_identity_invalid'),
    ('dose_uid_missing', 'dose_identity_invalid'),
    ('no_reference', 'no_exact_plan_reference'),
    ('wrong_reference', 'no_exact_plan_reference'),
    ('composite_reference', 'no_exact_plan_reference'),
    ('beam_only', 'no_plan_level_dose'),
    ('two_doses', 'ambiguous_plan_level_doses'),
    ('frame_mismatch', 'dose_ct_frame_mismatch'),
    ('geometry', 'dose_geometry_qc_failed'),
    ('units', 'dose_geometry_qc_failed'),
    ('scaling', 'dose_geometry_qc_failed'),
    ('pixels', 'dose_pixels_invalid'),
    ('unsafe_plan', 'unsafe_course_path'),
    ('unsafe_dose', 'unsafe_course_path'),
    ('symlink', 'unsafe_course_path'),
])
def test_refusals(tmp_path, case, expected):
    course = make_course(tmp_path)
    dose = course/'DICOM/RTDOSE/dose.dcm'
    plan = course/'DICOM/RTPLAN/plan.dcm'
    ct = course/'DICOM/CT/ct_000.dcm'
    if case == 'contract_selected':
        write_minimal_course_contract(course)
    elif case == 'bad_contract':
        (course/'metadata/case_metadata.json').write_text('{}')
    elif case == 'plan_unreadable':
        plan.write_bytes(b'bad')
    elif case == 'plan_uid':
        edit_dicom(plan, SOPInstanceUID='2.25.987')
    elif case == 'qc_failed':
        edit_contract(course, lambda c: c['dose_qc'].update(status='fail', **{'pass': False}))
    elif case == 'qc_inconsistent':
        edit_contract(course, lambda c: c['dose_qc'].update(threshold_gy=1.))
    elif case == 'ct_absent':
        edit_contract(course, lambda c: c.update(planning_ct={
            'status': 'missing_reference', 'series_instance_uid': '',
            'referenced_series_uids': [], 'dicom_dir': '', 'nifti_path': '', 'nifti_provenance': None}))
    elif case == 'ct_frame_missing':
        edit_dicom(ct, FrameOfReferenceUID=None)
    elif case == 'ct_frames_mixed':
        extra = ct.with_name('ct_001.dcm')
        shutil.copy2(ct, extra)
        edit_dicom(extra, FrameOfReferenceUID='2.25.987', SOPInstanceUID='2.25.988')
        original_classification = json.loads((course/'metadata/case_metadata.json').read_text())['course_contract']['dose_classification']
        write_minimal_course_contract(course, selected_doses=[],
                                      authoritative_rtstruct=course/'DICOM/RTSTRUCT/rs.dcm')
        edit_contract(course, lambda c: c.update(dose_classification=original_classification))
    elif case == 'no_dose':
        dose.unlink()
    elif case == 'dose_unreadable':
        dose.write_bytes(b'bad')
    elif case == 'dose_wrong_class':
        edit_dicom(dose, SOPClassUID='2.25.987')
    elif case == 'dose_uid_missing':
        edit_dicom(dose, SOPInstanceUID='')
    elif case == 'no_reference':
        edit_dicom(dose, ReferencedRTPlanSequence=Sequence([]))
    elif case in ('wrong_reference', 'composite_reference'):
        ref = Dataset(); ref.ReferencedSOPInstanceUID = '2.25.987'
        refs = [ref]
        if case == 'composite_reference':
            refs += list(pydicom.dcmread(dose).ReferencedRTPlanSequence)
        edit_dicom(dose, ReferencedRTPlanSequence=Sequence(refs))
    elif case == 'beam_only':
        edit_dicom(dose, DoseSummationType='BEAM')
    elif case == 'two_doses':
        shutil.copy2(dose, dose.with_name('second.dcm'))
    elif case == 'frame_mismatch':
        edit_dicom(dose, FrameOfReferenceUID='2.25.987')
    elif case == 'geometry':
        edit_dicom(dose, PixelSpacing=[0., 2.])
    elif case == 'units':
        edit_dicom(dose, DoseUnits='RELATIVE')
    elif case == 'scaling':
        edit_dicom(dose, DoseGridScaling=0.)
    elif case == 'pixels':
        edit_dicom(dose, PixelData=b'bad')
    elif case == 'unsafe_plan':
        edit_contract(course, lambda c: c['selected_plans'][0].update(path='../elsewhere.dcm'))
    elif case == 'unsafe_dose':
        edit_contract(course, lambda c: c['dose_classification'].update(excluded_doses=['../elsewhere.dcm']))
    elif case == 'symlink':
        saved = course/'saved.dcm'; dose.rename(saved); dose.symlink_to(saved)
    before = inventory(course)
    assert select_planning_dose(course).reason_code == expected
    assert repair_course(course, apply=True) == expected
    assert inventory(course) == before


def test_beam_doses_and_other_plans_do_not_replace_unique_plan_grid(tmp_path):
    course = make_course(tmp_path)
    dose = course/'DICOM/RTDOSE/dose.dcm'
    beam = dose.with_name('beam.dcm'); shutil.copy2(dose, beam)
    edit_dicom(beam, DoseSummationType='BEAM')
    other = dose.with_name('other.dcm'); shutil.copy2(dose, other)
    ref = Dataset(); ref.ReferencedSOPInstanceUID = '2.25.987'
    edit_dicom(other, ReferencedRTPlanSequence=Sequence([ref]))
    assert select_planning_dose(course).dose_path == dose


@pytest.mark.parametrize('change', ['dose', 'plan', 'contract', 'path', 'eligibility', 'revision', 'json'])
def test_stale_sidecars_refused(tmp_path, change):
    course = make_course(tmp_path)
    assert repair_course(course, apply=True) == 'applied'
    if change == 'dose':
        edit_dicom(course/'DICOM/RTDOSE/dose.dcm', DoseGridScaling=.0002)
    elif change == 'plan':
        edit_dicom(course/'DICOM/RTPLAN/plan.dcm', RTPlanLabel='Changed')
    elif change == 'contract':
        edit_contract(course, lambda c: c['dose_classification'].update(reason='changed'))
    elif change == 'json':
        (course/SIDECAR).write_text('[]')
    else:
        path = course/SIDECAR; data = json.loads(path.read_text())
        data[{'path':'selected_dose_path', 'eligibility':'dose_response_eligible', 'revision':'code_revision'}[change]] = {
            'path':'../outside.dcm', 'eligibility':True, 'revision':''}[change]
        path.write_text(json.dumps(data))
    assert not load_planning_dose_sidecar(course).accepted
    assert repair_course(course, apply=True) == 'sidecar_conflict'


def test_missing_sidecar_does_not_select_implicitly(tmp_path):
    course = make_course(tmp_path)
    assert load_planning_dose_sidecar(course).reason_code == 'sidecar_missing'
    assert dvh.dvh_for_course(course, parallel_workers=1) is None
    assert not (course/'dvh_metrics.parquet').exists()


def test_planning_dvh_flags_cache_and_stale_refusal(tmp_path):
    course = make_course(tmp_path)
    contract_bytes = (course/'metadata/case_metadata.json').read_bytes()
    assert repair_course(course, apply=True) == 'applied'
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course/'dvh_metrics.parquet')
    assert set(frame.dose_basis) == {BASIS} and set(frame.planning_dose_basis) == {BASIS}
    assert not frame.dose_response_eligible.any()
    assert not frame.Dose_Response_Eligible.any()
    assert not frame.dose_metric_usable_for_dose_response.any()
    assert frame.DmeanGy.notna().any() and frame.D95Gy.notna().any()
    assert (course/'dvh_curves.json').is_file()
    qc = json.loads((course/'metadata/dvh_qc.json').read_text())
    assert qc['dose_basis'] == qc['planning_dose_basis'] == BASIS
    assert qc['dose_response_eligible'] is False
    assert qc['dose_resolution']['dose_response_eligible'] is False
    assert (course/'metadata/case_metadata.json').read_bytes() == contract_bytes
    before = inventory(course)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    assert inventory(course) == before
    edit_dicom(course/'DICOM/RTDOSE/dose.dcm', DoseGridScaling=.0002)
    assert dvh.dvh_for_course(course, parallel_workers=1) is None
    assert not (course/'dvh_metrics.parquet').exists()
    assert not (course/'dvh_curves.json').exists()
    qc = json.loads((course/'metadata/dvh_qc.json').read_text())
    assert qc['planning_dose_refusal_reason'] == 'sidecar_stale_or_mismatched'


def historical_dvh(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(['git', 'show', '80d5d65:rtpipeline/dvh.py'], cwd=root).decode()
    module = types.ModuleType('rtpipeline._rf16_baseline_dvh')
    sys.modules[module.__name__] = module
    shadow = tmp_path/'baseline_package'; shadow.mkdir()
    module.__file__ = str(shadow/'dvh.py')
    exec(compile(code, module.__file__, 'exec'), module.__dict__)
    for name in module.DVH_MEASUREMENT_CODE_SOURCES:
        source = subprocess.check_output(['git', 'show', f'80d5d65:rtpipeline/{name}'], cwd=root)
        (shadow/name).write_bytes(source)
    return module


@pytest.mark.parametrize('delivered', [False, True])
@pytest.mark.parametrize('sidecar', [False, True])
def test_contract_selected_bytes_against_80d5d65(tmp_path, monkeypatch, delivered, sidecar):
    """Exact output bytes except the truthful producing-code digest in QC."""
    import xlsxwriter.core
    class FixedTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, tzinfo=tz)
    monkeypatch.setattr(xlsxwriter.core, 'datetime', FixedTime)
    course = fx.build_course(tmp_path, delivered=delivered, rois=fx.NO_NEAR_ZERO_ROIS)
    if sidecar:
        (course/SIDECAR).write_text('{invalid sidecar ignored for a contracted grid')
    baseline = historical_dvh(tmp_path)
    names = ['dvh_metrics.parquet', 'dvh_metrics.xlsx', 'dvh_curves.json', 'metadata/dvh_qc.json']
    def run(module):
        module._invalidate_dvh_outputs(course)
        assert module.dvh_for_course(course, parallel_workers=1)
        return {name:(course/name).read_bytes() for name in names}
    old, new = run(baseline), run(dvh)
    for name in names[:-1]:
        assert old[name] == new[name], name
    old_digest = json.loads(old[names[-1]])['code_sources_sha256']
    new_digest = json.loads(new[names[-1]])['code_sources_sha256']
    assert old_digest != new_digest
    assert old[names[-1]].replace(old_digest.encode(), new_digest.encode()) == new[names[-1]]
    assert 'planning_dose_basis' not in pd.read_parquet(course/'dvh_metrics.parquet')


def test_atomic_publish_failure_leaves_inputs_untouched(tmp_path, monkeypatch):
    from rtpipeline import repair_planning_dose_selection as repair
    course = make_course(tmp_path)
    before = inventory(course)
    def fail(*args):
        raise OSError('simulated filesystem failure')
    monkeypatch.setattr(repair.os, 'replace', fail)
    assert repair_course(course, apply=True) == 'publish_failed'
    assert inventory(course) == before


def test_dvh_cli_accepts_valid_planning_sidecar(tmp_path):
    from rtpipeline.cli import _execute_dvh_task
    course = make_course(tmp_path)
    assert repair_course(course, apply=True) == 'applied'
    task = types.SimpleNamespace(course=types.SimpleNamespace(dirs=types.SimpleNamespace(root=course)),
                                 custom_structures=None, parallel_workers=1, use_cropped=True,
                                 max_total_dose_gy=100.)
    outcome = _execute_dvh_task(task)
    assert outcome is not None and outcome.status == 'computed'
    assert load_course_contract(course).data['dvh']['metrics_status'] == 'not_computed'


def test_sidecar_removal_invalidates_planning_outputs(tmp_path):
    course = make_course(tmp_path)
    assert repair_course(course, apply=True) == 'applied'
    assert dvh.dvh_for_course(course, parallel_workers=1)
    (course/SIDECAR).unlink()
    assert dvh.dvh_for_course(course, parallel_workers=1) is None
    for name in ('dvh_metrics.parquet', 'dvh_metrics.xlsx', 'dvh_curves.json'):
        assert not (course/name).exists()


def test_planning_dose_uses_existing_rotated_grid_path(tmp_path):
    from test_dvh_rotation import analytic_dose, box_structure
    course = make_course(tmp_path)
    analytic_dose(course/'DICOM/RTDOSE/dose.dcm', .1)
    box_structure(course/'DICOM/RTSTRUCT/rs.dcm')
    assert repair_course(course, apply=True) == 'applied'
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course/'dvh_metrics.parquet')
    assert set(frame.planning_dose_basis) == {BASIS}
    assert set(frame.dose_grid_coverage_status) == {'fully_covered'}
    assert frame.DmeanGy.tolist() == pytest.approx([10., 10.], abs=.12)
    assert not frame.dose_response_eligible.any()


def test_concurrent_input_change_refuses_atomic_publication(tmp_path, monkeypatch):
    from rtpipeline import repair_planning_dose_selection as repair
    course = make_course(tmp_path)
    original = repair.select_planning_dose
    calls = 0
    def change(root):
        nonlocal calls
        calls += 1
        if calls == 3:
            edit_dicom(course/'DICOM/RTDOSE/dose.dcm', DoseGridScaling=.0002)
        return original(root)
    monkeypatch.setattr(repair, 'select_planning_dose', change)
    assert repair_course(course, apply=True) == 'inputs_changed'
    assert not (course/SIDECAR).exists()
    assert not list((course/'metadata').glob('.planning-dose-*'))


def test_cli_restores_logging_state_and_counts_refusals(tmp_path, capsys):
    import logging
    previous = logging.root.manager.disable
    assert main(['--course', str(tmp_path/'missing')]) == 1
    assert logging.root.manager.disable == previous
    output = capsys.readouterr().out
    assert str(tmp_path) not in output
    assert json.loads(output)['outcomes'] == {'invalid_course_contract': 1}


@pytest.mark.parametrize('kind', ['directory', 'broken'])
def test_dose_inventory_refuses_hidden_symlinks(tmp_path, kind):
    course = make_course(tmp_path)
    target = course/'other' if kind == 'directory' else course/'absent'
    if kind == 'directory':
        target.mkdir()
    (course/'DICOM/RTDOSE/linked').symlink_to(target, target_is_directory=kind == 'directory')
    assert select_planning_dose(course).reason_code == 'unsafe_course_path'


def test_completion_binds_planning_sidecar_content(tmp_path):
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel, validate_stage_completion_sentinel
    course = make_course(tmp_path)
    assert repair_course(course, apply=True) == 'applied'
    assert dvh.dvh_for_course(course, parallel_workers=1)
    dependency = materialize_stage_dependency(tmp_path/'configuration', 'dvh', {'enabled':True})
    sentinel = course/'.dvh_done'
    receipt = write_stage_completion_sentinel(course, sentinel, stage='dvh', status='ok',
                                               configuration_dependency=dependency)
    bound = [item for item in receipt['outputs'] if item['role'] == 'planning_dose_selection']
    assert len(bound) == 1 and bound[0]['path'] == SIDECAR
    assert validate_stage_completion_sentinel(sentinel, expected_stage='dvh') == receipt
    (course/SIDECAR).unlink()
    with pytest.raises(ValueError, match='absent'):
        validate_stage_completion_sentinel(sentinel, expected_stage='dvh')
