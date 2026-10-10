"""RF19 composite course dose: synthetic two-phase course, scaling and refusals.

Every UID, date and dose is invented. The base fixture is one 5-fraction plan
whose PLAN dose is 20 Gy on the left half of the grid. A second plan and dose
are derived from it; organize's unreconciled multi-plan contract is emulated.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import RTBeamsTreatmentRecordStorage

import dvh_plan_target_fixtures as fx
from course_contract_test_utils import write_minimal_course_contract
from rtpipeline import dvh
from rtpipeline.course_contract import build_dvh_decision, classify_course_dose_completeness
from rtpipeline.composite_course_dose import (
    BASIS, OUTPUT, OUTPUT_PLAN, SIDECAR, load_composite_sidecar, select_composite_course_dose,
)
from rtpipeline.repair_composite_course_dose import main, repair_course

PLAN2_UID = f"{fx.UID_ROOT}.30"
DOSE2_UID = f"{fx.UID_ROOT}.31"


def edit_dicom(path, **values):
    ds = pydicom.dcmread(path)
    for key, value in values.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


def edit_contract(course, change):
    path = course / 'metadata/case_metadata.json'
    payload = json.loads(path.read_text())
    change(payload['course_contract'])
    path.write_text(json.dumps(payload, indent=2))


def dates(first, count):
    return [f"202001{day:02d}" if day <= 31 else f"202002{day - 31:02d}" for day in range(first, first + count)]


def write_records(course, plan_uid, tag, treatment_dates):
    """One synthetic NORMAL RTRECORD session per delivered fraction."""
    paths = []
    for fraction, date in enumerate(treatment_dates, start=1):
        uid = f"{fx.UID_ROOT}.40.{tag}.{fraction}"
        path = course / 'DICOM_related' / 'RTRECORD' / f'record_{tag}_{fraction}.dcm'
        record = fx._file_dataset(path, RTBeamsTreatmentRecordStorage, uid)
        record.Modality = 'RTRECORD'
        record.TreatmentDate = date
        reference = Dataset()
        reference.ReferencedSOPInstanceUID = plan_uid
        record.ReferencedRTPlanSequence = Sequence([reference])
        record.CurrentFractionNumber = fraction
        session = Dataset()
        session.TreatmentDeliveryType = 'TREATMENT'
        session.TreatmentTerminationStatus = 'NORMAL'
        record.TreatmentSessionBeamSequence = Sequence([session])
        path.parent.mkdir(parents=True, exist_ok=True)
        record.save_as(str(path), enforce_file_format=True)
        paths.append(path.relative_to(course).as_posix())
    return paths


def make_course(tmp_path, *, plan2_fractions=2, plan2_gy=10.0, plan2_dose_gy=10.0,
                delivered=(5, 2), treatment=((1, 5), (3, 2)), shift_mm=0.0, isocenters=None):
    course = fx.build_course(tmp_path, delivered=False)
    edit_dicom(course / 'DICOM/CT/ct_000.dcm', FrameOfReferenceUID=fx.FRAME_OF_REFERENCE_UID)
    plan1 = course / 'DICOM/RTPLAN/plan.dcm'
    plan2 = course / 'DICOM/RTPLAN/plan2.dcm'
    shutil.copy(plan1, plan2)
    ds = pydicom.dcmread(plan2)
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = PLAN2_UID
    ds.SeriesInstanceUID = f"{PLAN2_UID}.1"
    ds.DoseReferenceSequence[0].TargetPrescriptionDose = plan2_gy
    ds.FractionGroupSequence[0].NumberOfFractionsPlanned = plan2_fractions
    ds.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose = plan2_gy / plan2_fractions
    ds.save_as(plan2, enforce_file_format=True)
    dose2 = course / 'DICOM/RTDOSE/dose2.dcm'
    shutil.copy(course / 'DICOM/RTDOSE/dose.dcm', dose2)
    ds = pydicom.dcmread(dose2)
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = DOSE2_UID
    ds.SeriesInstanceUID = f"{DOSE2_UID}.1"
    ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = PLAN2_UID
    ds.ImagePositionPatient = [shift_mm, 0.0, 0.0]
    values = np.full((fx.GRID_PLANES, fx.GRID_SIZE, fx.GRID_SIZE), plan2_dose_gy)
    ds.PixelData = np.round(values / float(ds.DoseGridScaling)).astype('<u4').tobytes()
    ds.save_as(dose2, enforce_file_format=True)
    for path, point in zip((plan1, plan2), isocenters or ()):
        ds = pydicom.dcmread(path)
        control_point = Dataset()
        control_point.ControlPointIndex = 0
        control_point.IsocenterPosition = list(point)
        ds.BeamSequence[0].ControlPointSequence = Sequence([control_point])
        ds.save_as(path, enforce_file_format=True)
    write_minimal_course_contract(course, selected_plans=[plan1, plan2], selected_doses=[],
                                  authoritative_rtstruct=course / 'DICOM/RTSTRUCT/rs.dcm')
    # Organize copies one intent plan as the artifact of an unreconciled course.
    shutil.copy(plan1, course / 'RP.dcm')

    def update(contract):
        for index, item in enumerate(contract['selected_plans']):
            first, count = treatment[index]
            record_dates = dates(first, count)[:delivered[index]]
            records = write_records(course, item['sop_instance_uid'], index + 1, record_dates)
            evidence = {'delivered_fraction_count': delivered[index],
                        'delivered_record_count': delivered[index],
                        'planned_fraction_count': (5, plan2_fractions)[index],
                        'treatment_dates': record_dates}
            item.update(evidence)
            contract['delivery']['per_plan'][index].update(
                {**evidence, 'zero_delivery_records': delivered[index] == 0,
                 'record_paths': records, 'status': 'delivery_unresolved'})
        contract['delivery']['status'] = 'delivery_unresolved'
        contract['plan_artifact'] = {'path': 'RP.dcm', 'sop_instance_uid': fx.PLAN_UID,
                                     'source_plan_uids': [fx.PLAN_UID]}
        contract['dose_classification'] = {
            'classification': 'independent_delivered_plans_unreconciled', 'should_sum': False,
            'excluded_doses': ['DICOM/RTDOSE/dose.dcm', 'DICOM/RTDOSE/dose2.dcm'],
            'warnings': ['Retained 2 independently delivered plan(s) as course membership'],
        }
        contract['dose_completeness'] = classify_course_dose_completeness(
            selected_plans=contract['selected_plans'], selected_doses=[],
            dose_classification=contract['dose_classification'], dose_grid=None,
            per_plan_delivery=contract['delivery']['per_plan'],
            delivery_status=contract['delivery']['status'])
        contract['dvh'] = build_dvh_decision(
            2, 0, contract['delivery']['status'], dose_response_eligible=False,
            dose_completeness=contract['dose_completeness'])
    edit_contract(course, update)
    return course


def inventory(course):
    return {str(p.relative_to(course)): p.read_bytes() for p in course.rglob('*') if p.is_file()}


def composite_values(course):
    ds = pydicom.dcmread(course / OUTPUT)
    return ds, ds.pixel_array.astype(float) * float(ds.DoseGridScaling)


def test_two_phase_dry_run_is_read_only_and_apply_publishes_bound_composite(tmp_path, capsys):
    course = make_course(tmp_path)
    before = inventory(course)
    selection = select_composite_course_dose(course)
    assert selection.accepted, selection.reason_code
    assert selection.weights == (1.0, 1.0)
    assert main(['--course', str(course)]) == 0
    assert json.loads(capsys.readouterr().out)['outcomes'] == {'accepted': 1}
    assert inventory(course) == before
    assert main(['--course', str(course), '--apply']) == 0
    assert json.loads(capsys.readouterr().out)['outcomes'] == {'applied': 1}
    after = inventory(course)
    sidecar = json.loads(after.pop(SIDECAR))
    after.pop(OUTPUT)
    after.pop(OUTPUT_PLAN)
    plan_sum = pydicom.dcmread(course / OUTPUT_PLAN)
    assert plan_sum.ApprovalStatus == 'UNAPPROVED'
    assert {str(r.ReferencedSOPInstanceUID) for r in plan_sum.ReferencedRTPlanSequence} == {fx.PLAN_UID, PLAN2_UID}
    assert after == before
    assert sidecar['basis'] == BASIS and sidecar['dose_response_eligible'] is False
    assert [i['weight'] for i in sidecar['inputs']] == [1.0, 1.0]
    assert sidecar['interpolation']['method'] == 'trilinear'
    assert sidecar['date_relations'] == {'contained': 1}
    assert sidecar['output_dose_summation_type'] == 'MULTI_PLAN'
    assert sidecar['output_grid']['max_dose_gy'] == pytest.approx(30.0, abs=1e-3)
    text = json.dumps(sidecar)
    assert fx.PLAN_UID not in text and PLAN2_UID not in text and '202001' not in text
    ds, values = composite_values(course)
    assert ds.DoseSummationType == 'MULTI_PLAN'
    assert {str(r.ReferencedSOPInstanceUID) for r in ds.ReferencedRTPlanSequence} == {fx.PLAN_UID, PLAN2_UID}
    left, right = values[:, :, : fx.GRID_SIZE // 2], values[:, :, fx.GRID_SIZE // 2:]
    assert np.allclose(left, 30.0, atol=1e-3) and np.allclose(right, 10.005, atol=1e-3)
    assert load_composite_sidecar(course).accepted
    assert repair_course(course, apply=True) == 'unchanged'


def test_partially_delivered_concurrent_plan_is_weighted_by_delivered_over_planned(tmp_path):
    course = make_course(tmp_path, delivered=(5, 1), treatment=((1, 5), (3, 1)))
    selection = select_composite_course_dose(course)
    assert selection.accepted, selection.reason_code
    assert selection.weights == (1.0, 0.5)
    assert repair_course(course, apply=True) == 'applied'
    _, values = composite_values(course)
    assert np.allclose(values[:, :, : fx.GRID_SIZE // 2], 25.0, atol=1e-3)


def test_sequential_full_phases_with_offset_grids_are_resampled(tmp_path):
    course = make_course(tmp_path, treatment=((1, 5), (6, 2)), shift_mm=1.0)
    assert repair_course(course, apply=True) == 'applied'
    sidecar = json.loads((course / SIDECAR).read_text())
    assert sidecar['date_relations'] == {'disjoint': 1}
    assert sidecar['output_grid']['cols'] == fx.GRID_SIZE + 1
    _, values = composite_values(course)
    # Interior of the left half: 20 Gy plan 1 plus 10 Gy plan 2 after shift.
    assert np.allclose(values[:, 2:18, 2:8], 30.0, atol=1e-3)


def test_concurrent_equal_signature_plans_need_distinct_isocenters(tmp_path):
    # Two-lesion SBRT: equal fractionation, same dates, separate isocenters.
    same = dict(plan2_fractions=5, plan2_gy=20.0, delivered=(5, 5), treatment=((1, 5), (1, 5)))
    course = make_course(tmp_path / 'a', isocenters=((10.0, 18.0, 4.0), (30.0, 18.0, 4.0)), **same)
    selection = select_composite_course_dose(course)
    assert selection.accepted, selection.reason_code
    assert selection.evidence['date_relations'] == {'identical': 1}
    assert selection.evidence['equal_signature_pairs'] == 1
    course = make_course(tmp_path / 'b', isocenters=((10.0, 18.0, 4.0), (12.0, 18.0, 4.0)), **same)
    assert select_composite_course_dose(course).reason_code == 'equal_signature_shared_isocenter_unproven'
    course = make_course(tmp_path / 'c', **same)
    assert select_composite_course_dose(course).reason_code == 'equal_signature_shared_isocenter_unproven'


def test_replan_replacing_an_interrupted_plan_is_refused(tmp_path):
    # Same prescription signature: plan 1 stopped after 3 of 5, plan 2 finished.
    course = make_course(tmp_path, plan2_fractions=5, plan2_gy=20.0, delivered=(3, 2),
                         treatment=((1, 3), (4, 2)))
    assert select_composite_course_dose(course).reason_code == 'replacement_plan_chain_not_summed'
    before = inventory(course)
    assert repair_course(course, apply=True) == 'replacement_plan_chain_not_summed'
    assert inventory(course) == before


@pytest.mark.parametrize('case, expected', [
    ('not_delivered', 'plan_without_recorded_delivery'),
    ('over_delivered', 'delivered_exceeds_planned'),
    ('separate_course', 'separate_course_suspected'),
    ('classification', 'classification_not_eligible'),
    ('dose_frame', 'dose_ct_frame_mismatch'),
    ('effective_dose', 'dose_not_physical_gy'),
    ('beam_dose', 'no_plan_level_dose'),
    ('no_dates', 'treatment_dates_unavailable'),
    ('interrupted_then_later_plan', 'replacement_plan_chain_not_summed'),
    ('brachy', 'brachytherapy_plan_not_supported'),
])
def test_governed_refusals(tmp_path, case, expected):
    kwargs = {
        'not_delivered': dict(delivered=(5, 0)),
        'over_delivered': dict(delivered=(5, 3), treatment=((1, 5), (3, 3))),
        'separate_course': dict(treatment=((1, 5), (40, 2))),
        'interrupted_then_later_plan': dict(delivered=(3, 2), treatment=((1, 3), (6, 2))),
    }.get(case, {})
    course = make_course(tmp_path, **kwargs)
    dose2 = course / 'DICOM/RTDOSE/dose2.dcm'
    if case == 'classification':
        edit_contract(course, lambda c: c['dose_classification'].update(classification='single_dose'))
    elif case == 'dose_frame':
        edit_dicom(dose2, FrameOfReferenceUID=f"{fx.UID_ROOT}.99")
    elif case == 'effective_dose':
        edit_dicom(dose2, DoseType='EFFECTIVE')
    elif case == 'beam_dose':
        edit_dicom(dose2, DoseSummationType='BEAM')
    elif case == 'no_dates':
        edit_contract(course, lambda c: [c['selected_plans'][1].update(treatment_dates=['bad']),
                                         c['delivery']['per_plan'][1].update(treatment_dates=['bad'])])
    elif case == 'brachy':
        edit_dicom(course / 'DICOM/RTPLAN/plan2.dcm', BrachyTreatmentType='HDR')
    before = inventory(course)
    assert select_composite_course_dose(course).reason_code == expected
    assert repair_course(course, apply=True) == expected
    assert inventory(course) == before


def test_stale_inputs_or_output_invalidate_sidecar(tmp_path):
    course = make_course(tmp_path)
    assert repair_course(course, apply=True) == 'applied'
    edit_dicom(course / 'DICOM/RTDOSE/dose2.dcm', DoseGridScaling=0.0002)
    assert load_composite_sidecar(course).reason_code == 'sidecar_stale_or_mismatched'
    course = make_course(tmp_path / 'b')
    assert repair_course(course, apply=True) == 'applied'
    edit_dicom(course / OUTPUT, SeriesDescription='tampered')
    assert load_composite_sidecar(course).reason_code == 'composite_output_stale_or_missing'


def test_missing_sidecar_never_selects_implicitly(tmp_path):
    course = make_course(tmp_path)
    assert load_composite_sidecar(course).reason_code == 'sidecar_missing'
    assert dvh.dvh_for_course(course, parallel_workers=1) is None
    assert not (course / 'dvh_metrics.parquet').exists()


def test_dvh_marks_composite_provenance_and_keeps_dose_response_ineligible(tmp_path):
    course = make_course(tmp_path)
    contract_bytes = (course / 'metadata/case_metadata.json').read_bytes()
    assert repair_course(course, apply=True) == 'applied'
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course / 'dvh_metrics.parquet')
    assert set(frame.dose_basis) == {BASIS} and set(frame.composite_dose_basis) == {BASIS}
    assert set(frame.dose_provenance) == {'composite'}
    assert not frame.dose_response_eligible.any()
    assert not frame.Dose_Response_Eligible.any()
    assert not frame.dose_metric_usable_for_dose_response.any()
    ptv = frame[frame.ROI_Name == 'PTV1'].iloc[0]
    assert ptv.DmeanGy == pytest.approx(30.0, abs=0.5)
    qc = json.loads((course / 'metadata/dvh_qc.json').read_text())
    assert qc['composite_dose_basis'] == BASIS and qc['dose_provenance'] == 'composite'
    assert qc['dose_response_eligible'] is False
    assert (course / 'metadata/case_metadata.json').read_bytes() == contract_bytes
    before = inventory(course)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    assert inventory(course) == before
    edit_dicom(course / 'DICOM/RTDOSE/dose2.dcm', DoseGridScaling=0.0002)
    assert dvh.dvh_for_course(course, parallel_workers=1) is None
    assert not (course / 'dvh_metrics.parquet').exists()
    qc = json.loads((course / 'metadata/dvh_qc.json').read_text())
    assert qc['composite_dose_refusal_reason'] == 'sidecar_stale_or_mismatched'




def test_weighted_prescription_sum_at_threshold_is_not_a_float_refusal(tmp_path):
    # 20 Gy + (1/3) x 4.4 Gy evaluates to 21.46666666666667 in binary floating
    # point, above the float of the exact value 322/15 = 21.466666666666665.
    course = make_course(tmp_path, plan2_fractions=3, plan2_gy=4.4, delivered=(5, 1),
                         treatment=((1, 5), (3, 1)))
    edit_contract(course, lambda c: c['dose_qc'].update(threshold_gy=322 / 15))
    selection = select_composite_course_dose(course)
    assert selection.accepted, selection.reason_code
    assert sum(i['weighted_prescribed_gy'] for i in selection.evidence['inputs']) > 322 / 15
    edit_contract(course, lambda c: c['dose_qc'].update(threshold_gy=21.4))
    assert select_composite_course_dose(course).reason_code == 'composite_prescription_implausible'
