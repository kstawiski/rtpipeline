"""Closed-form hottest-2 cm³ doses for the production metric backends."""
from types import SimpleNamespace

import numpy as np
import pytest

from rtpipeline import dvh


@pytest.mark.parametrize('backend', ['histogram', 'arrays'])
@pytest.mark.parametrize('field', ['uniform', 'linear', 'linear_from_zero'])
@pytest.mark.parametrize('volume', [8., 2., np.nextafter(2., 0.), 1.999])
def test_d2cc_analytic_fields_and_exact_volume_boundary(backend, field, volume):
    # Linear dose is uniform in volume on [3, 7] Gy:
    # D2cc = 7 - (7 - 3) * 2 / volume, provided volume >= 2 cm³.
    minimum, maximum = {'uniform': (5., 5.), 'linear': (3., 7.),
                        'linear_from_zero': (0., 4.)}[field]
    if backend == 'histogram':
        histogram = SimpleNamespace(
            bincenters=np.array([minimum, maximum]),
            counts=np.array([volume, 0.]), mean=(minimum + maximum) / 2,
            min=minimum, max=maximum)
        result = dvh._compute_metrics(histogram, None)
        tolerance = 1e-12
    else:
        doses = (np.full(8000, minimum) if field == 'uniform'
                 else np.linspace(minimum, maximum, 8000))
        result = dvh._compute_metrics_from_arrays(doses, volume / len(doses), maximum)
        # Same 0.01 Gy bin convention as the existing backend-agreement test.
        tolerance = .01
    if volume < 2.:
        assert result['D2ccGy'] is None
        assert result['D2cc_status'] == 'roi_below_2cc'
    else:
        expected = maximum - (maximum - minimum) * 2. / volume
        assert result['D2ccGy'] == pytest.approx(expected, abs=tolerance)
        assert result['D2cc_status'] == 'available'
        assert result['DminGy'] <= result['D2ccGy'] <= result['DmaxGy']
        if volume == 2.:
            assert result['D2ccGy'] == result['DminGy']


@pytest.mark.parametrize('status', ['partial_grid', 'outside_grid', 'coverage_unresolved'])
def test_d2cc_propagates_grid_coverage_and_keeps_covered_diagnostic(status):
    metrics = dvh._compute_metrics_from_arrays(np.full(4000, 5.), .001, 5.)
    row = dvh.annotate_dvh_metrics(
        metrics, technique='EBRT', structure_name='Bladder',
        rtstruct_sop_instance_uid='1.2.3', rtstruct_path='RS.dcm',
        grid_coverage={'status': status, 'fraction': None},
        covered_metrics=metrics if status == 'partial_grid' else None)
    assert row['D2ccGy'] is None
    assert row['D2cc_status'] == status
    assert row['diagnostic_legacy_D2ccGy'] == 5.
    if status == 'partial_grid':
        assert row['covered_D2ccGy'] == 5.
        assert row['covered_D2cc_status'] == 'available'
