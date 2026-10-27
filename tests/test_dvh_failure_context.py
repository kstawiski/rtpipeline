"""RF17: real synthetic course publications retain refusal context and schema."""
import json

import pandas as pd
import pytest

import dvh_plan_target_fixtures as fx
from rtpipeline import dvh
from rtpipeline.course_contract import DOSE_RESPONSE_ELIGIBILITY_BASIS, load_course_contract

QUANTILES = ('D98Gy', 'D95Gy', 'D50Gy', 'D2Gy')
CORE = ('DmeanGy', 'DminGy', 'DmaxGy', 'D0.03ccGy', *QUANTILES)


def mark_ineligible(course):
    path = course / 'metadata/case_metadata.json'
    payload = json.loads(path.read_text())
    payload['course_contract']['delivery'].update(
        dose_response_eligible=False,
        dose_response_eligibility_basis=DOSE_RESPONSE_ELIGIBILITY_BASIS,
        dose_response_ineligibility_reason_code='delivery_unresolved',
        dose_response_ineligibility_reason='Synthetic course has no delivery evidence.',
    )
    path.write_text(json.dumps(payload))


def fail_histogram(*args, **kwargs):
    raise ValueError('synthetic histogram refusal')


@pytest.mark.parametrize('eligible', [False, True])
@pytest.mark.parametrize('coverage', ['fully_covered', 'partial_grid', 'coverage_unresolved', 'outside_grid'])
def test_failed_course_preserves_context_and_null_schema(tmp_path, monkeypatch, eligible, coverage):
    # The neutral name requires the interpreted type to identify this target.
    course = fx.build_course(tmp_path, delivered=eligible,
                             rois=[('Focus', 'PTV', fx.DOSED_LEFT_X)])
    if not eligible:
        mark_ineligible(course)
    contract = load_course_contract(course)
    assert dvh._contract_dose_response_eligible(contract) is eligible
    calls = []
    annotate = dvh.annotate_dvh_metrics

    def observe_context(metrics, **kwargs):
        calls.append(kwargs)
        return annotate(metrics, **kwargs)

    monkeypatch.setattr(dvh, 'annotate_dvh_metrics', observe_context)
    monkeypatch.setattr(dvh.dvhcalc, 'get_dvh', fail_histogram)
    monkeypatch.setattr(dvh, 'rtstruct_grid_coverage', lambda *a: {
        'status': coverage, 'fraction': {'fully_covered': 1., 'partial_grid': .5,
                                        'outside_grid': 0.}.get(coverage),
        'method': 'synthetic_geometry',
    })
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course / 'dvh_metrics.parquet')
    assert len(frame) == len(calls) == 1
    assert calls[0]['prescription_resolved'] is eligible
    assert calls[0]['dose_response_ineligibility_reason'] == (
        contract.delivery.get('dose_response_ineligibility_reason') or ''
    )
    assert frame.dose_response_eligible.tolist() == [eligible]
    assert frame.Dose_Response_Eligible.tolist() == [eligible]
    assert frame.dose_metric_usable_for_dose_response.tolist() == [False]
    assert frame.target_like.tolist() == [True]
    assert frame.ROI_Interpreted_Type.tolist() == ['PTV']
    expected = 'dvh_computation_failed' if coverage == 'fully_covered' else coverage
    assert frame.dose_metric_status.tolist() == [expected]
    assert frame.dvh_computation_error.tolist() == ['synthetic histogram refusal']
    assert set(CORE) <= set(frame.columns)
    assert frame[list(CORE)].isna().all().all()
    assert frame[list(dvh.RELATIVE_DVH_METRIC_COLUMNS)].isna().all().all()
    assert frame['Volume (cm³)'].isna().all()
    workbook = pd.read_excel(course / 'dvh_metrics.xlsx')
    assert set(QUANTILES) <= set(workbook.columns)
    assert workbook[list(QUANTILES)].isna().all().all()
    assert not (course / 'dvh_curves.json').exists()


def test_mixed_publication_keeps_successful_measurements(tmp_path, monkeypatch):
    course = fx.build_course(tmp_path, rois=fx.NO_NEAR_ZERO_ROIS)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    before = pd.read_parquet(course / 'dvh_metrics.parquet')
    original = dvh.dvhcalc.get_dvh

    def fail_one(rs, dose, number, **kwargs):
        if number == 1:
            return fail_histogram()
        return original(rs, dose, number, **kwargs)

    monkeypatch.setattr(dvh.dvhcalc, 'get_dvh', fail_one)
    monkeypatch.setattr(dvh, '_is_dvh_up_to_date', lambda *a: False)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    after = pd.read_parquet(course / 'dvh_metrics.parquet')
    columns = list(before.columns)
    pd.testing.assert_frame_equal(
        before.loc[before.ROI_Number != 1, columns].reset_index(drop=True),
        after.loc[after.ROI_Number != 1, columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert after.loc[after.ROI_Number == 1, list(CORE)].isna().all().all()


def test_rf14_orientation_refusal_keeps_ineligible_context(tmp_path):
    from test_dvh_rotation import analytic_dose, box_structure

    course = fx.build_course(tmp_path, delivered=False)
    mark_ineligible(course)
    analytic_dose(course / 'DICOM/RTDOSE/dose.dcm', 1.1)
    box_structure(course / 'DICOM/RTSTRUCT/rs.dcm')
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course / 'dvh_metrics.parquet')
    assert not frame.dose_response_eligible.any()
    assert not frame.dose_metric_usable_for_dose_response.any()
    assert frame[list(CORE)].isna().all().all()
    assert set(frame.dvh_computation_error) == {'rotated_dose_rotation_exceeds_1_degree'}


def test_rf16_planning_failure_keeps_sidecar_and_ineligible_context(tmp_path, monkeypatch):
    from test_planning_dose_selection import make_course
    from rtpipeline.planning_dose_selection import BASIS
    from rtpipeline.repair_planning_dose_selection import repair_course

    course = make_course(tmp_path)
    assert repair_course(course, apply=True) == 'applied'
    monkeypatch.setattr(dvh.dvhcalc, 'get_dvh', fail_histogram)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course / 'dvh_metrics.parquet')
    assert not frame.dose_response_eligible.any()
    assert not frame.dose_metric_usable_for_dose_response.any()
    assert frame[list(CORE)].isna().all().all()
    qc = json.loads((course / 'metadata/dvh_qc.json').read_text())
    assert qc['planning_dose_basis'] == BASIS
    assert qc['planning_dose_selection_sha256']
