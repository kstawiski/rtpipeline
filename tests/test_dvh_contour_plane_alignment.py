"""Synthetic axial plane alignment and production receipt regression tests."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import dvh_plan_target_fixtures as fx
from rtpipeline import dvh
from rtpipeline.dvh_support import rtstruct_grid_coverage
from rtpipeline.utils import (
    _axial_dose_plane_positions, dose_on_contour_planes, snap_rtstruct_to_dose_grid,
)
from dicompylercore import dicomparser


def dose_grid(path, step=2.5, start=0., stop=60., normal=1, descending=False):
    dose = pydicom.dcmread(fx.write_dose(path))
    zs = np.arange(start, stop + 1e-6, step)
    if descending:
        zs = zs[::-1]
    dose.Rows = dose.Columns = 32
    dose.NumberOfFrames = len(zs)
    dose.PixelSpacing = [1., 1.]
    dose.ImagePositionPatient = [0., 0., float(zs[0])]
    dose.ImageOrientationPatient = [1., 0., 0., 0., float(normal), 0.]
    dose.GridFrameOffsetVector = ((zs - zs[0]) / normal).tolist()
    dose.DoseGridScaling = 1e-6
    values = np.broadcast_to((20. + .1 * zs)[:, None, None], (len(zs), 32, 32))
    dose.PixelData = np.rint(values / 1e-6).astype('<u4').tobytes()
    return dose


def structure(path, plane_sets):
    rs = pydicom.dcmread(fx.write_rtstruct(path, [
        (f'Organ{k}', 'ORGAN', (4.1, 24.1)) for k in range(len(plane_sets))]))
    for roi, zs in zip(rs.ROIContourSequence, plane_sets):
        items = []
        for z in zs:
            c = Dataset()
            c.ContourGeometricType = 'CLOSED_PLANAR'
            c.NumberOfContourPoints = 4
            c.ContourData = [4.1,4.1,z,24.1,4.1,z,24.1,24.1,z,4.1,24.1,z]
            items.append(c)
        roi.ContourSequence = Sequence(items)
    return rs


@pytest.mark.parametrize('ct,step', [(1.5,3.), (3.,3.), (3.,1.5), (5.,2.5)])
def test_commensurate_results_match_native(tmp_path, ct, step):
    dose = dose_grid(tmp_path/'dose.dcm', step)
    zs = 2 * step + np.arange(7) * ct
    rs = structure(tmp_path/'rs.dcm', [zs])
    working, method = dose_on_contour_planes(rs, dose, 1)
    before = dvh.dvhcalc.get_dvh(rs, dose, 1)
    after = dvh.dvhcalc.get_dvh(rs, working, 1)
    assert after.volume == pytest.approx(400 * len(zs) * ct / 1000, abs=1e-12)
    assert after.volume == pytest.approx(before.volume, abs=1e-12)
    a, b = dvh._compute_metrics(after, 25.), dvh._compute_metrics(before, 25.)
    for key in ('DmeanGy', 'DminGy', 'DmaxGy', 'D95Gy', 'D2Gy'):
        assert a[key] == pytest.approx(b[key], abs=1e-12)
    assert method == ('DOSE_Z_INTERPOLATED' if ct == 1.5 else 'EXACT')


@pytest.mark.parametrize('ct', [1., 3., 2.])
def test_noncommensurate_volume_and_linear_dose(tmp_path, ct):
    dose = dose_grid(tmp_path/'dose.dcm')
    zs = 5. + np.arange(11) * ct
    rs = structure(tmp_path/'rs.dcm', [zs])
    original = rs.to_json()
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    assert rs.to_json() == original
    parser = dicomparser.DicomParser(working)
    for z in zs:
        np.testing.assert_allclose(parser.GetDoseGrid(z) * float(dose.DoseGridScaling),
                                   20. + .1 * z, atol=1e-12, rtol=0)
    planes = dicomparser.DicomParser(rs).GetStructureCoordinates(1)
    assert len(planes) == len(zs)
    histogram = dvh.dvhcalc.get_dvh(rs, working, 1)
    assert histogram.volume == pytest.approx(400 * len(zs) * ct / 1000, abs=1e-12)
    metrics = dvh._compute_metrics(histogram, None)
    expected = dvh._compute_metrics_from_arrays(20. + .1 * zs, .4 * ct, 30.)
    for key in ('DmeanGy', 'D95Gy', 'D2Gy'):
        assert metrics[key] == pytest.approx(expected[key], abs=.011)


def test_partial_support_keeps_full_and_covered_diagnostics(tmp_path):
    dose = dose_grid(tmp_path/'dose.dcm', start=5., stop=15.)
    zs = np.arange(-1., 21., 3.)
    rs = structure(tmp_path/'rs.dcm', [zs])
    coverage = rtstruct_grid_coverage(rs, 1, dose)
    assert coverage['status'] == 'partial_grid'
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    parser = dicomparser.DicomParser(working)
    assert parser.GetDoseGrid(-1.).size == parser.GetDoseGrid(20.).size == 0
    full = dvh.dvhcalc.get_dvh(rs, working, 1)
    covered = dvh.dvhcalc.get_dvh(rs, working, 1, calculate_full_volume=False)
    assert full.volume == pytest.approx(.4 * 3 * len(zs))
    assert covered.volume == pytest.approx(.4 * 3 * 4)
    row = dvh.annotate_dvh_metrics(dvh._compute_metrics(full, None),
        technique='EBRT', structure_name='Organ', rtstruct_sop_instance_uid=str(rs.SOPInstanceUID),
        rtstruct_path=tmp_path/'rs.dcm', grid_coverage=coverage,
        covered_metrics=dvh._compute_metrics(covered, None))
    assert row['DmeanGy'] is None
    assert row['dose_metric_status'] == 'partial_grid'
    assert row['covered_DmeanGy'] == pytest.approx(20.95, abs=.011)


def test_no_input_mutation_between_rois_threads_or_calls(tmp_path):
    dose = dose_grid(tmp_path/'dose.dcm')
    rs = structure(tmp_path/'rs.dcm', [np.arange(5., 16.), np.arange(6., 22., 3.)])
    original_rs, original_dose = rs.to_json(), dose.to_json()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda roi: dose_on_contour_planes(rs, dose, roi), [1,2,1,2]))
    assert rs.to_json() == original_rs
    assert dose.to_json() == original_dose
    assert results[0][0] is not results[2][0]
    np.testing.assert_array_equal(results[0][0].pixel_array, results[2][0].pixel_array)
    assert list(results[0][0].GridFrameOffsetVector) != list(results[1][0].GridFrameOffsetVector)
    results[0][0].GridFrameOffsetVector[1] += .25
    assert list(results[0][0].GridFrameOffsetVector) != list(results[2][0].GridFrameOffsetVector)
    with pytest.raises(ValueError):
        results[0][0].pixel_array[0,0,0] = 0
    assert list(dose_on_contour_planes(rs, dose, 1)[0].GridFrameOffsetVector) == list(results[2][0].GridFrameOffsetVector)


@pytest.mark.parametrize('normal,descending', [(1,False), (1,True), (-1,False), (-1,True)])
def test_signed_axial_and_descending_planes_interpolate_brackets(tmp_path, normal, descending):
    dose = dose_grid(tmp_path/'dose.dcm', normal=normal, descending=descending)
    rs = structure(tmp_path/'rs.dcm', [[6., 9., 12.]])
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    for z in [6., 9., 12.]:
        np.testing.assert_allclose(dicomparser.DicomParser(working).GetDoseGrid(z) * 1e-6,
                                   20. + .1 * z, atol=1e-12, rtol=0)
    assert working.ImageOrientationPatient == dose.ImageOrientationPatient
    assert working.PixelSpacing == dose.PixelSpacing
    assert working.ImagePositionPatient[:2] == dose.ImagePositionPatient[:2]


def test_absolute_frame_offsets_and_no_integer_requantization(tmp_path):
    dose = dose_grid(tmp_path/'dose.dcm', start=5., stop=20.)
    dose.GridFrameOffsetVector = np.arange(5., 20.1, 2.5).tolist()
    # One stored unit per 2.5 mm gives fractional stored values at z=6.
    values = np.arange(7)[:, None, None] + np.arange(32)[None, None, :]
    dose.PixelData = np.broadcast_to(values, (7,32,32)).astype('<u4').tobytes()
    rs = structure(tmp_path/'rs.dcm', [[6., 9., 12.]])
    working, _ = dose_on_contour_planes(rs, dose, 1)
    expected = np.broadcast_to(.4 + np.arange(32), (32,32))
    np.testing.assert_allclose(dicomparser.DicomParser(working).GetDoseGrid(6.), expected, atol=1e-14)


def test_nonuniform_source_grid_uses_bracketing_planes(tmp_path):
    dose = dose_grid(tmp_path/'dose.dcm', stop=10.)
    source_z = np.array([0., 1., 2., 5., 10.])
    dose.GridFrameOffsetVector = source_z.tolist()
    values = source_z[:,None,None]**2 + np.arange(32)[None,None,:]
    dose.PixelData = np.broadcast_to(values, (5,32,32)).astype('<u4').tobytes()
    rs = structure(tmp_path/'rs.dcm', [[2.2,3.2,4.2]])
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    # At z=2.2 the two nearest nodes (z=2 and z=1) are on the same side.
    # The correct bracketing nodes are z=2 (value 4) and z=5 (value 25).
    expected = np.broadcast_to(5.4 + np.arange(32), (32,32))
    np.testing.assert_allclose(dicomparser.DicomParser(working).GetDoseGrid(2.2), expected, atol=1e-14)


@pytest.mark.parametrize('first', [-.001, -.0004, .0004, .001])
@pytest.mark.parametrize('normal,descending', [(1,False), (1,True), (-1,False), (-1,True)])
def test_relative_offset_rounding_is_normalized(tmp_path, first, normal, descending):
    dose = dose_grid(tmp_path/'dose.dcm', start=1000., stop=1060.,
                     normal=normal, descending=descending)
    expected_z = _axial_dose_plane_positions(dose)
    dose.GridFrameOffsetVector = (np.asarray(dose.GridFrameOffsetVector) + first).tolist()
    np.testing.assert_allclose(_axial_dose_plane_positions(dose), expected_z, atol=1e-12, rtol=0)
    rs = structure(tmp_path/'rs.dcm', [[1006., 1009.]])
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    np.testing.assert_allclose(dicomparser.DicomParser(working).GetDoseGrid(1006.) * 1e-6,
                               120.6, atol=1e-12, rtol=0)


@pytest.mark.parametrize('shift', [-.001, -.0004, .0004, .001])
def test_absolute_offset_rounding_is_accepted(tmp_path, shift):
    dose = dose_grid(tmp_path/'dose.dcm', start=1000., stop=1060.)
    expected_z = np.arange(1000., 1060.1, 2.5) + shift
    dose.GridFrameOffsetVector = expected_z.tolist()
    np.testing.assert_array_equal(_axial_dose_plane_positions(dose), expected_z)
    rs = structure(tmp_path/'rs.dcm', [[1006.+shift, 1009.+shift]])
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    np.testing.assert_allclose(dicomparser.DicomParser(working).GetDoseGrid(1006.+shift) * 1e-6,
                               120.6, atol=1e-12, rtol=0)


@pytest.mark.parametrize('first,normal', [(.0011,1), (-.0011,1), (1000.005,1),
                                        (999.995,1), (1000.,-1)])
def test_offset_mode_rejects_other_encodings(tmp_path, first, normal):
    dose = dose_grid(tmp_path/'dose.dcm', start=1000., stop=1060., normal=normal)
    dose.GridFrameOffsetVector = (np.asarray(dose.GridFrameOffsetVector) + first).tolist()
    with pytest.raises(ValueError, match='Invalid RTDOSE frame offsets'):
        dose_on_contour_planes(structure(tmp_path/'rs.dcm', [[1006.]]), dose, 1)


@pytest.mark.parametrize('spread', [.0015, .008, .01])
def test_contour_z_rounding_uses_mean_without_mutation(tmp_path, spread):
    from rtpipeline.dvh_rotation import MAX_CONTOUR_PROJECTION_MM
    assert spread <= MAX_CONTOUR_PROJECTION_MM
    dose = dose_grid(tmp_path/'dose.dcm')
    rs = structure(tmp_path/'rs.dcm', [[0.]])
    zs = np.array([0., spread, spread, spread])
    rs.ROIContourSequence[0].ContourSequence[0].ContourData[2::3] = zs.tolist()
    original = rs.to_json()
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert method == 'DOSE_Z_INTERPOLATED'
    np.testing.assert_allclose(_axial_dose_plane_positions(working), [0., zs.mean(), 60.],
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(working.pixel_array[1] * 1e-6, 20. + .1 * zs.mean(),
                               atol=1e-12, rtol=0)
    assert rs.to_json() == original


@pytest.mark.parametrize('value', [.010001, float('nan'), float('inf'), -float('inf')])
def test_contour_z_above_ceiling_or_nonfinite_is_rejected(tmp_path, value):
    dose = dose_grid(tmp_path/'dose.dcm')
    rs = structure(tmp_path/'rs.dcm', [[0.]])
    rs.ROIContourSequence[0].ContourSequence[0].ContourData[5] = value
    with pytest.raises(ValueError, match='finite axial contours'):
        dose_on_contour_planes(rs, dose, 1)


@pytest.mark.parametrize('stop,zs', [(60., [-1., 61.]), (0., [.1]), (0., [-.1]),
                                   (60., [])])
def test_no_in_range_planes_preserves_original_dose(tmp_path, stop, zs):
    dose = dose_grid(tmp_path/'dose.dcm', stop=stop)
    rs = structure(tmp_path/'rs.dcm', [zs])
    working, method = dose_on_contour_planes(rs, dose, 1)
    assert working is dose
    assert method == 'NO_IN_RANGE_PLANE'


def test_single_coincident_plane_is_exact(tmp_path):
    dose = dose_grid(tmp_path/'dose.dcm', stop=0.)
    working, method = dose_on_contour_planes(structure(tmp_path/'rs.dcm', [[0.]]), dose, 1)
    assert working is dose
    assert method == 'EXACT'


@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('zs,method', [([5., 10., 20.], 'EXACT'),
                                    ([5., 10.0005, 20.], 'DOSE_Z_INTERPOLATED')])
def test_absolute_coincident_grid_rebuilds_relative_samples(tmp_path, descending, zs, method):
    dose = dose_grid(tmp_path/'dose.dcm', start=5., stop=20., descending=descending)
    source_z = _axial_dose_plane_positions(dose)
    dose.GridFrameOffsetVector = source_z.tolist()
    original = dose.to_json()
    working, actual = dose_on_contour_planes(structure(tmp_path/'rs.dcm', [zs]), dose, 1)
    assert actual == method
    assert working is not dose
    assert working.GridFrameOffsetVector[0] == 0.
    assert working.ImagePositionPatient[2] == 5.
    assert dose.to_json() == original
    parser = dicomparser.DicomParser(working)
    for z in zs:
        np.testing.assert_allclose(parser.GetDoseGrid(z) * 1e-6, 20. + .1 * z, atol=1e-12, rtol=0)
    if method == 'EXACT':
        for z in zs:
            np.testing.assert_array_equal(parser.GetDoseGrid(z), dose.pixel_array[list(source_z).index(z)])


def test_memory_error_reaches_adaptive_worker_backoff(tmp_path, monkeypatch, caplog):
    course = fx.build_course(tmp_path, delivered=False, rois=fx.NO_NEAR_ZERO_ROIS)
    original_align = dvh.dose_on_contour_planes
    attempts = []

    def allocate(rs, dose, roi):
        attempts.append(int(roi))
        if int(roi) == 1 and attempts.count(1) == 1:
            raise MemoryError('synthetic per-ROI allocation failure')
        return original_align(rs, dose, roi)

    monkeypatch.setattr(dvh, 'dose_on_contour_planes', allocate)
    assert dvh.dvh_for_course(course, parallel_workers=2)
    assert attempts.count(1) == 2
    assert 'retrying 1 task(s) with 1 worker(s) after memory pressure' in caplog.text
    table = pd.read_parquet(course/'dvh_metrics.parquet')
    assert len(table) == len(fx.NO_NEAR_ZERO_ROIS)
    assert table.DmeanGy.notna().all()
    assert not table.dose_metric_status.eq('dvh_computation_failed').any()


def test_geometry_error_remains_explicit_roi_failure(tmp_path, monkeypatch):
    course = fx.build_course(tmp_path, delivered=False)

    def invalid_geometry(*args):
        raise ValueError('synthetic invalid geometry')

    monkeypatch.setattr(dvh, 'dose_on_contour_planes', invalid_geometry)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    table = pd.read_parquet(course/'dvh_metrics.parquet')
    assert table.dose_metric_status.eq('dvh_computation_failed').all()
    assert table.dvh_computation_error.eq('synthetic invalid geometry').all()


@pytest.mark.parametrize('methods,expected', [(['NO_IN_RANGE_PLANE'], 'NO_IN_RANGE_PLANE'),
    (['EXACT', 'NO_IN_RANGE_PLANE'], 'EXACT'),
    (['EXACT', 'DOSE_Z_INTERPOLATED'], 'DOSE_Z_INTERPOLATED'), (['SUBMM_SNAP'], None)])
def test_receipt_uses_production_alignment_vocabulary(tmp_path, methods, expected):
    table = pd.DataFrame({'ROI_Name': ['Organ'] * len(methods), 'dose_plane_alignment': methods})
    dvh._write_dvh_qc(tmp_path, table, [], 'none')
    receipt = json.loads((tmp_path/'metadata/dvh_qc.json').read_text())
    assert receipt['dose_plane_alignment'] == expected
    assert receipt['dose_plane_alignment_counts'] == table.dose_plane_alignment.value_counts().to_dict()


@pytest.mark.parametrize('shift,method', [(.0005,'EXACT'), (.0015,'DOSE_Z_INTERPOLATED'),
                                         (.05,'DOSE_Z_INTERPOLATED'), (.1,'DOSE_Z_INTERPOLATED')])
def test_alignment_tolerance_uses_absolute_distance(tmp_path, shift, method):
    dose = dose_grid(tmp_path/'dose.dcm', start=1000., stop=1060.)
    rs = structure(tmp_path/'rs.dcm', [[1005.+shift, 1010.+shift]])
    before = rs.to_json()
    working, actual = dose_on_contour_planes(rs, dose, 1)
    assert actual == method
    assert rs.to_json() == before
    if method == 'EXACT':
        assert working is dose


@pytest.mark.parametrize('zs,changed', [([5.05,7.55],True), ([5.05,5.08],False),
                                      ([5.,5.05],False), ([6.,9.],False), ([5.1,7.6],False)])
def test_optional_snap_is_below_point_one_mm_and_injective(tmp_path, zs, changed):
    dose = dose_grid(tmp_path/'dose.dcm')
    rs = structure(tmp_path/'rs.dcm', [zs])
    assert snap_rtstruct_to_dose_grid(rs, dose, tolerance_mm=100) is changed
    actual = [float(c.ContourData[2]) for c in rs.ROIContourSequence[0].ContourSequence]
    assert len(set(actual)) == len(zs)
    assert actual == ([5.,7.5] if changed else zs)


@pytest.mark.parametrize('spacing', [1., 2.5])
def test_course_publishes_alignment_per_roi_and_in_receipt(tmp_path, spacing):
    course = fx.build_course(tmp_path, delivered=False)
    dose = dose_grid(course/'DICOM/RTDOSE/dose.dcm', step=spacing)
    dose.save_as(course/'DICOM/RTDOSE/dose.dcm', enforce_file_format=True)
    rs = structure(course/'DICOM/RTSTRUCT/rs.dcm', [np.arange(5., 16.), np.arange(5., 20.1, 2.5)])
    rs.save_as(course/'DICOM/RTSTRUCT/rs.dcm', enforce_file_format=True)
    original = (course/'DICOM/RTSTRUCT/rs.dcm').read_bytes()
    assert dvh.dvh_for_course(course, parallel_workers=2)
    table = pd.read_parquet(course/'dvh_metrics.parquet')
    expected = ['EXACT','DOSE_Z_INTERPOLATED'] if spacing == 1. else ['DOSE_Z_INTERPOLATED','EXACT']
    assert table.sort_values('ROI_Number').dose_plane_alignment.tolist() == expected
    qc = json.loads((course/'metadata/dvh_qc.json').read_text())
    assert qc['dose_plane_alignment_counts'] == {'EXACT':1, 'DOSE_Z_INTERPOLATED':1}
    assert qc['dose_plane_alignment'] == 'DOSE_Z_INTERPOLATED'
    curves = json.loads((course/'dvh_curves.json').read_text())['points']
    assert {c['dose_plane_alignment'] for c in curves} == set(expected)
    assert (course/'DICOM/RTSTRUCT/rs.dcm').read_bytes() == original
    assert dvh.dvh_for_course(course, parallel_workers=1)
    pd.testing.assert_frame_equal(table, pd.read_parquet(course/'dvh_metrics.parquet'))


def test_alignment_helper_participates_in_measurement_identity():
    assert 'utils.py' in dvh.DVH_MEASUREMENT_CODE_SOURCES


@pytest.mark.parametrize('quarantine', [False, True])
def test_existing_table_columns_match_stored_baseline_with_additive_alignment(tmp_path, quarantine):
    """The required extra column preserves every existing value and dtype."""
    from pathlib import Path
    kwargs = ({'references':[fx.dose_reference(number=1, description='PTV1'),
                            fx.dose_reference(number=2, description='ptv2 ')]} if quarantine
              else {'rois':fx.NO_NEAR_ZERO_ROIS})
    course = fx.build_course(tmp_path, **kwargs)
    assert dvh.dvh_for_course(course, parallel_workers=1, max_total_dose_gy=100.)
    table = pd.read_parquet(course/'dvh_metrics.parquet')
    filename = ('ae60f01_near_zero_course_quarantine.json' if quarantine
                else 'ae60f01_no_near_zero_targets.json')
    baseline = json.loads((Path(fx.__file__).parent/'synthetic_dvh_plan_target'/filename).read_text())
    assert set(table) - set(baseline['columns']) == {'dose_plane_alignment'}
    projected = table[baseline['columns']]
    assert projected.dtypes.astype(str).tolist() == baseline['dtypes']

    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (np.floating, float)):
            return repr(float(value))
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return str(value).replace(str(course), '<COURSE>')

    assert [[cell(v) for v in row] for row in projected.itertuples(index=False)] == baseline['rows']
