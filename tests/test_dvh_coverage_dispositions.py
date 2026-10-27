"""Synthetic geometry and publication regressions for WS105."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest

from rtpipeline import dvh, dvh_support
from rtpipeline.dvh_rotation import RotatedDoseGrid
from test_dvh_grid_support import boxes


def contour(points, kind='CLOSED_PLANAR'):
    points = np.asarray(points, float).reshape(-1, 3)
    return SimpleNamespace(ContourData=points.ravel().tolist(),
                           NumberOfContourPoints=len(points), ContourGeometricType=kind)


def structure(contours):
    return SimpleNamespace(ROIContourSequence=[SimpleNamespace(
        ReferencedROINumber=1, ContourSequence=contours)])


def both_coverages(rs):
    _, dose = boxes(2, 4)
    # Header-only rotated adapter: dispositions must not depend on pixel values.
    grid = object.__new__(RotatedDoseGrid)
    grid.origin = np.zeros(3)
    grid.shape = np.array([10, 10, 3])
    grid.step = np.ones(3)
    grid.inverse_basis = np.eye(3)
    return dvh_support.rtstruct_grid_coverage(rs, 1, dose), grid.coverage(rs, 1)


@pytest.mark.parametrize('items,status,code', [
    ([], 'declared_empty', 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'),
    ([contour([[1, 1, 1]], 'POINT')], 'non_volumetric', 'ROI_NONVOLUMETRIC_CONTOURS'),
    ([contour([[1, 1, 1], [2, 2, 1]], 'OPEN_PLANAR')], 'non_volumetric', 'ROI_NONVOLUMETRIC_CONTOURS'),
    ([contour([[1, 1, 0], [2, 2, 1], [3, 1, 2]], 'OPEN_NONPLANAR')], 'non_volumetric', 'ROI_NONVOLUMETRIC_CONTOURS'),
    ([contour([[1, 1, 1], [2, 1, 1]])], 'degenerate_contours', 'ROI_NO_AREA_BEARING_CONTOURS'),
    ([contour([[1, 1, 1], [2, 1, 1], [1, 1, 1], [2, 1, 1]])], 'degenerate_contours', 'ROI_NO_AREA_BEARING_CONTOURS'),
])
def test_identical_nonvolume_dispositions(items, status, code):
    first, second = both_coverages(structure(items))
    assert first == second
    assert first['status'] == status
    assert first['fraction'] is None
    assert first['reason'] and first['reason_code'] == code


def test_missing_roi_is_not_declared_empty():
    for result in both_coverages(SimpleNamespace(ROIContourSequence=[])):
        assert result['status'] == 'coverage_unresolved'
        assert result['fraction'] is None


@pytest.mark.parametrize('defect', ['short', 'nan', 'infinity', 'count', 'absent', 'unknown', 'mixed_open', 'mixed_point'])
def test_malformed_or_mixed_geometry_remains_unresolved(defect):
    good = contour([[1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1]])
    bad = contour([[1, 1, 1], [2, 1, 1], [2, 2, 1]])
    if defect == 'short': bad.ContourData.pop()
    if defect == 'nan': bad.ContourData[0] = float('nan')
    if defect == 'infinity': bad.ContourData[0] = float('inf')
    if defect == 'count': bad.NumberOfContourPoints += 1
    if defect == 'absent': del bad.ContourData
    if defect == 'unknown': bad.ContourGeometricType = 'UNKNOWN'
    if defect == 'mixed_open': bad.ContourGeometricType = 'OPEN_PLANAR'
    if defect == 'mixed_point': bad = contour([[1, 1, 1]], 'POINT')
    for result in both_coverages(structure([good, bad])):
        assert result['status'] == 'coverage_unresolved'
        assert result['fraction'] is None


@pytest.mark.parametrize('repeat', [False, True])
def test_degenerate_mixed_with_valid_contours_does_not_change_coverage(repeat):
    rs, dose = boxes(-2, 4)
    before = dvh_support.rtstruct_grid_coverage(rs, 1, dose)
    degenerate = contour([[20, 1, 1], [21, 1, 1]] * (2 if repeat else 1))
    rs = structure([*rs.ROIContourSequence[0].ContourSequence, degenerate])
    assert dvh_support.rtstruct_grid_coverage(rs, 1, dose) == before


@pytest.mark.parametrize('zero_plane', [0, 1, 2])
def test_zero_area_plane_retains_trapezoid_volume(zero_plane):
    rs, dose = boxes(-2, 4)
    square = np.array([[-2, 2], [4, 2], [4, 4], [-2, 4]])
    # Three distinct non-collinear vertices, with every edge retraced.
    trace = np.array([[-2, 2], [4, 2], [4, 4], [4, 2]])
    rs = structure([contour(np.c_[trace if z == zero_plane else square, np.full(4, z)])
                    for z in range(3)])
    result = dvh_support.rtstruct_grid_coverage(rs, 1, dose)
    assert result['status'] == 'partial_grid'
    assert result['fraction'] == pytest.approx(2 / 3)
    assert result['roi_volume_cm3'] == pytest.approx((12 if zero_plane == 1 else 18) / 1000)


def test_zero_integrated_volume_is_unresolved():
    trace = [[-2, 2, 0], [4, 2, 0], [4, 4, 0], [4, 2, 0]]
    rs = structure([contour(trace), contour(np.asarray(trace) + [0, 0, 2])])
    assert both_coverages(rs)[0]['status'] == 'coverage_unresolved'


def test_no_extrapolation_beyond_contour_planes():
    rs, dose = boxes(-2, 4, 0, 1)
    result = dvh_support.rtstruct_grid_coverage(rs, 1, dose)
    assert result['roi_volume_cm3'] == pytest.approx(.012)
    assert result['covered_volume_cm3'] == pytest.approx(.008)


def test_no_grid_skip_publishes_governed_receipt(tmp_path):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel, validate_stage_completion_sentinel
    course = tmp_path / 'SYNTH_P' / 'SYNTH_C'
    write_minimal_course_contract(course, selected_plans=[], selected_doses=[])
    assert dvh.dvh_for_course(course) is None
    dependency = materialize_stage_dependency(tmp_path / 'config', 'dvh', {'enabled': True})
    receipt = write_stage_completion_sentinel(course, course / '.dvh_done', stage='dvh',
                                              status='ok', configuration_dependency=dependency)
    assert receipt['measurement_outcome'] == 'not_measured'
    assert validate_stage_completion_sentinel(course / '.dvh_done', expected_stage='dvh') == receipt
    assert not any((course / name).exists() for name in ['dvh_metrics.parquet', 'dvh_metrics.xlsx', 'dvh_curves.json'])


def ray_crossing(points, rings):
    """Independent horizontal-ray fill, with no area-helper calls."""
    x, y = points.T
    inside = np.zeros(len(points), dtype=bool)
    for ring in rings:
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            if a[1] == b[1]:
                continue
            crosses = ((a[1] > y) != (b[1] > y))
            intercept = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            inside ^= crosses & (x < intercept)
    return inside


SQUARE = np.array([[-1., -1.], [3., -1.], [3., 3.], [-1., 3.]])
RINGS = [
    [SQUARE],
    [SQUARE, np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])],  # hole
    [SQUARE, SQUARE.copy()],  # duplicate rings cancel
    [SQUARE, SQUARE + [2., 0.]],  # overlapping XOR
    [np.array([[-1., -1.], [3., 3.], [-1., 3.], [3., -1.]])],  # bow tie
    [np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.],
               [-2., 0.], [-2., -2.], [0., -2.]])],  # touching lobes
    [np.array([[-1., -1.], [3., -1.], [3., 3.], [3., -1.]])],  # retraced
    [SQUARE, SQUARE + [4., 0.]],  # shared edge
]


@pytest.mark.parametrize('rings', RINGS)
@pytest.mark.parametrize('reverse', [False, True])
def test_planar_parity_matches_independent_ray_fill(rings, reverse):
    rings = [p[::-1] if reverse else p for p in rings]
    # Regular midpoint quadrature with unequal x/y phase avoids edge samples.
    coords = np.concatenate(rings)
    lo, hi = coords.min(axis=0) - .123, coords.max(axis=0) + .137
    x = lo[0] + (np.arange(801) + .5) * (hi[0] - lo[0]) / 801
    y = lo[1] + (np.arange(803) + .5) * (hi[1] - lo[1]) / 803
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel()]
    filled = ray_crossing(points, rings)
    cell_area = np.prod(hi - lo) / len(points)
    total, covered = dvh_support._parity_plane_areas(rings, 2., 2.)
    clipped = filled & (points[:, 0] > 0) & (points[:, 0] < 2) & (points[:, 1] > 0) & (points[:, 1] < 2)
    assert total == pytest.approx(filled.sum() * cell_area, abs=.04)
    assert covered == pytest.approx(clipped.sum() * cell_area, abs=.02)
    # Compare fill locally too: tiny boxes centred away from ring boundaries.
    rng = np.random.default_rng(105)
    probes = rng.uniform(lo, hi, (100, 2))
    expected = ray_crossing(probes, rings)
    half = 1e-7
    for point, interior in zip(probes, expected):
        moved = [p - point + half for p in rings]
        _, area = dvh_support._parity_plane_areas(moved, 2 * half, 2 * half)
        assert area / (4 * half**2) == pytest.approx(float(interior), abs=1e-7)


@pytest.mark.parametrize('rings,total,covered', [
    ([SQUARE], 16., 4.),
    ([SQUARE, SQUARE], 0., 0.),
    ([SQUARE, SQUARE + [2., 0.]], 16., 2.),
    ([np.array([[-1., -1.], [3., 3.], [-1., 3.], [3., -1.]])], 8., 2.),
    ([np.array([[-1., -1.], [3., -1.], [3., 3.], [3., -1.]])], 0., 0.),
])
def test_analytic_planar_area_and_axis_clipping(rings, total, covered):
    assert dvh_support._parity_plane_areas(rings, 2., 2.) == pytest.approx((total, covered), abs=1e-12)


def test_nonfinite_planar_coordinates_rejected():
    with pytest.raises(ValueError, match='Nonfinite'):
        dvh_support._parity_plane_areas([np.array([[0, 0], [1, 1], [np.nan, 2]])], 2, 2)


def test_reversed_decimal_ring_cancels_without_residual_volume():
    ring = np.array([[-.13, .29], [2.17, -.11], [3.03, 2.71], [-1.07, 3.19]])
    assert dvh_support._parity_plane_areas([ring, ring[::-1]], 2., 2.) == (0., 0.)
    _, dose = boxes(2, 4)
    rs = structure([contour(np.c_[p, np.full(len(p), z)])
                    for z in [0, 2] for p in [ring, ring[::-1]]])
    result = dvh_support.rtstruct_grid_coverage(rs, 1, dose)
    assert result['status'] == 'coverage_unresolved'
    assert result['fraction'] is None


@pytest.mark.parametrize('kind,status,code', [
    ('empty', 'declared_empty', 'ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE'),
    ('OPEN_PLANAR', 'non_volumetric', 'ROI_NONVOLUMETRIC_CONTOURS'),
    ('CLOSED_PLANAR', 'degenerate_contours', 'ROI_NO_AREA_BEARING_CONTOURS'),
])
@pytest.mark.parametrize('rotated', [False, True])
def test_nonmeasurement_rows_and_reasons_reach_publication(tmp_path, monkeypatch, kind, status, code, rotated):
    import dvh_plan_target_fixtures as fx
    from rtpipeline.dvh_aggregate import build_dvh_aggregate
    from test_dvh_rotation import analytic_dose
    course = fx.build_course(tmp_path, rois=[('Focus', 'PTV', fx.DOSED_LEFT_X)])
    if rotated:
        analytic_dose(course / 'DICOM/RTDOSE/dose.dcm', .1)
    path = course / 'DICOM/RTSTRUCT/rs.dcm'
    rs = pydicom.dcmread(path)
    if kind == 'empty':
        rs.ROIContourSequence[0].ContourSequence = []
    else:
        for item in rs.ROIContourSequence[0].ContourSequence:
            item.ContourGeometricType = kind
            item.ContourData = item.ContourData[:6]
            item.NumberOfContourPoints = 2
    rs.save_as(path, enforce_file_format=True)

    def forbidden(*args, **kwargs):
        pytest.fail('Nonvolume geometry reached a DVH sampler')
    monkeypatch.setattr(dvh.dvhcalc, 'get_dvh', forbidden)
    monkeypatch.setattr(RotatedDoseGrid, 'get_dvh', forbidden)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course / 'dvh_metrics.parquet')
    for table in [frame, pd.read_excel(course / 'dvh_metrics.xlsx'),
                  build_dvh_aggregate([frame], [('SYNTH_P', 'SYNTH_C', course)])]:
        assert len(table) == 1
        row = table.iloc[0]
        assert row['row_status'] == 'not_computed'
        assert row['dvh_nonmeasurement_class'] == status
        assert row['dose_grid_coverage_reason_code'] == code
        assert row['dose_grid_coverage_reason']
        assert row['dose_grid_coverage_status'] == status
        assert pd.isna(row['dose_grid_coverage_fraction'])
        assert table[['DmeanGy', 'DmaxGy', 'DminGy', 'D95Gy', 'D98Gy', 'D0.03ccGy',
                      'D90Gy', 'D2ccGy', 'Volume (cm³)']].isna().all().all()
        assert not row['dose_metric_usable_for_dose_response']
    assert not (course / 'dvh_curves.json').exists()


@pytest.mark.parametrize('seed', [19, 105, 772])
def test_irregular_self_crossing_rings_match_ray_crossing(seed):
    rng = np.random.default_rng(seed)
    rings = [rng.uniform(-2, 4, (9, 2)), rng.uniform(-1, 3, (7, 2))]
    coords = (np.arange(1001) + .5) * .006 - 2
    xx, yy = np.meshgrid(coords, coords + .00017)
    points = np.c_[xx.ravel(), yy.ravel()]
    fill = ray_crossing(points, rings)
    clipped = fill & (points[:, 0] > 0) & (points[:, 0] < 2) & (points[:, 1] > 0) & (points[:, 1] < 2)
    area, covered = dvh_support._parity_plane_areas(rings, 2, 2)
    assert area == pytest.approx(fill.sum() * .006**2, abs=.015)
    assert covered == pytest.approx(clipped.sum() * .006**2, abs=.015)


@pytest.mark.parametrize('defect', ['technical_failure', 'decision_mismatch', 'stale_contract', 'selected_dose'])
def test_no_grid_closure_rejects_unbound_or_technical_skip(tmp_path, defect):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel, file_sha256
    course = tmp_path / 'SYNTH_P' / 'SYNTH_C'
    write_minimal_course_contract(course, selected_plans=[], selected_doses=[])
    assert dvh.dvh_for_course(course) is None
    qc_path = course / 'metadata/dvh_qc.json'
    qc = json.loads(qc_path.read_text())
    if defect == 'technical_failure': qc['dose_resolution']['dose_qc_pass'] = False
    if defect == 'decision_mismatch': qc['dose_resolution']['dvh']['reason_code'] = 'different'
    if defect == 'stale_contract': qc['course_contract_sha256'] = '0' * 64
    if defect == 'selected_dose':
        contract_path = course / 'metadata/case_metadata.json'
        contract = json.loads(contract_path.read_text())
        contract['course_contract']['selected_doses'] = [{'path': 'synthetic.dcm'}]
        contract_path.write_text(json.dumps(contract))
        qc['course_contract_sha256'] = file_sha256(contract_path)
    qc_path.write_text(json.dumps(qc))
    dependency = materialize_stage_dependency(tmp_path / 'config', 'dvh', {'enabled': True})
    with pytest.raises(ValueError, match='authoritative not-computed'):
        write_stage_completion_sentinel(course, course / '.dvh_done', stage='dvh', status='ok',
                                        configuration_dependency=dependency)
    assert not (course / '.dvh_done').exists()


def test_missing_or_forged_receipt_outcome_is_rejected(tmp_path):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel, validate_stage_completion_sentinel
    course = tmp_path / 'SYNTH_P' / 'SYNTH_C'
    write_minimal_course_contract(course, selected_plans=[], selected_doses=[])
    assert dvh.dvh_for_course(course) is None
    dependency = materialize_stage_dependency(tmp_path / 'config', 'dvh', {'enabled': True})
    path = course / '.dvh_done'
    receipt = write_stage_completion_sentinel(course, path, stage='dvh', status='ok',
                                              configuration_dependency=dependency)
    receipt.pop('measurement_outcome')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='measurement outcome'):
        validate_stage_completion_sentinel(path, expected_stage='dvh')
    (course / 'dvh_metrics.parquet').write_bytes(b'synthetic table presence')
    measured = write_stage_completion_sentinel(course, path, stage='dvh', status='ok',
                                               configuration_dependency=dependency)
    assert 'measurement_outcome' not in measured
    measured['measurement_outcome'] = 'not_measured'
    path.write_text(json.dumps(measured))
    with pytest.raises(ValueError, match='measurement outcome'):
        validate_stage_completion_sentinel(path, expected_stage='dvh')


def test_mixed_nonmeasurement_publication_keeps_measured_values(tmp_path):
    import dvh_plan_target_fixtures as fx
    course = fx.build_course(tmp_path, rois=fx.NO_NEAR_ZERO_ROIS)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    before = pd.read_parquet(course / 'dvh_metrics.parquet')
    path = course / 'DICOM/RTSTRUCT/rs.dcm'
    rs = pydicom.dcmread(path)
    rs.ROIContourSequence[-1].ContourSequence = []
    rs.save_as(path, enforce_file_format=True)
    dvh._invalidate_dvh_outputs(course)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    after = pd.read_parquet(course / 'dvh_metrics.parquet')
    measured = after.ROI_Number != len(rs.ROIContourSequence)
    pd.testing.assert_frame_equal(before.loc[measured], after.loc[measured, before.columns],
                                  check_exact=True, check_dtype=False)
    assert after.loc[~measured, 'row_status'].tolist() == ['not_computed']
