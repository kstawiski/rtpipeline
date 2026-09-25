"""Synthetic late-quarantine regression; no DICOM or clinical inputs."""
import pytest

from rtpipeline.dvh import RELATIVE_DVH_METRIC_COLUMNS, apply_course_dose_quarantine


@pytest.mark.parametrize('status', ['computed', 'excluded_dose_response_ineligible',
                                   'unavailable_partial_grid',
                                   'quarantined_near_zero_requires_reconciliation',
                                   'suppressed_non_ebrt',
                                   'excluded_target_not_bound_to_course_plan'])
def test_course_quarantine_suppresses_relative_values_and_preserves_absolute_audit(status):
    row = {column: 1.0 for column in RELATIVE_DVH_METRIC_COLUMNS}
    row.update(DmeanGy=2.0, relative_metric_status=status,
               relative_metric_reason='Original reason', HI_status='computed',
               dose_response_eligible=True, dose_metric_usable_for_dose_response=True)
    apply_course_dose_quarantine([row], 'Synthetic pending reconciliation')
    assert all(row[c] is None for c in RELATIVE_DVH_METRIC_COLUMNS)
    assert row['DmeanGy'] == 2.0
    assert row['dose_response_eligible'] is False
    assert row['dose_metric_usable_for_dose_response'] is False
    assert row['dose_response_quarantine_status'] == 'pending_plan_target_reconciliation'
    assert row['dose_response_quarantine_reason'] == 'Synthetic pending reconciliation'
    if status in {'computed', 'excluded_dose_response_ineligible'}:
        assert row['relative_metric_status'] == 'excluded_dose_response_ineligible'
        assert row['relative_metric_reason'] == 'Synthetic pending reconciliation'
    else:
        assert row['relative_metric_status'] == status
        assert row['relative_metric_reason'] == 'Original reason'
    assert row['HI_status'] == 'excluded_dose_response_ineligible'
