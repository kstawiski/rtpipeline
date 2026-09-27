"""Synthetic patient-coordinate dose tests; no campaign input is used."""
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import dvh_plan_target_fixtures as fx
from rtpipeline import dvh
from rtpipeline.dvh_rotation import DoseOrientationError, METHOD, prepare_rotated_dose, add_brachy_metrics


def rotate(dose, angle, base=None):
    a = np.radians(angle)
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    basis = rotation @ (np.eye(3) if base is None else base)
    dose.ImageOrientationPatient = basis[:, :2].T.ravel().tolist()
    return basis


def analytic_dose(path, angle, base=None, descending=False):
    dose = pydicom.dcmread(fx.write_dose(path))
    dose.Rows = dose.Columns = dose.NumberOfFrames = 81
    dose.PixelSpacing = [.5, .5]
    dose.GridFrameOffsetVector = (np.arange(81)*(-.5 if descending else .5)).tolist()
    basis = rotate(dose, angle, base)
    origin = -basis @ np.array([20., 20., -20. if descending else 20.])
    dose.ImagePositionPatient = origin.tolist()
    zz, yy, xx = np.meshgrid(np.array(dose.GridFrameOffsetVector), np.arange(81)*.5,
                             np.arange(81)*.5, indexing='ij')
    coords = np.stack((xx, yy, zz), axis=-1) @ basis.T + origin
    values = 10. + .1*coords[..., 0]
    dose.DoseGridScaling = 1e-6
    dose.PixelData = np.rint(values/1e-6).astype('<u4').tobytes()
    dose.save_as(path, enforce_file_format=True)
    return dose


def box_structure(path, x=(-10., 10.), y=(-10., 10.), zs=range(-10, 11)):
    fx.write_rtstruct(path, [('HR-CTV', 'CTV', x), ('Bladder', 'ORGAN', x)])
    rs = pydicom.dcmread(path)
    for roi in rs.ROIContourSequence:
        items = []
        for z in zs:
            item = Dataset()
            item.ContourGeometricType = 'CLOSED_PLANAR'
            item.NumberOfContourPoints = 4
            item.ContourData = [x[0],y[0],z,x[1],y[0],z,x[1],y[1],z,x[0],y[1],z]
            items.append(item)
        roi.ContourSequence = Sequence(items)
    rs.save_as(path, enforce_file_format=True)
    return rs


@pytest.mark.parametrize('angle', [.025, .1, .5, -.1])
def test_trilinear_analytic_field_and_native_node_preservation(tmp_path, angle):
    dose = analytic_dose(tmp_path/'dose.dcm', angle)
    grid = prepare_rotated_dose(dose)
    points = np.random.default_rng(42).uniform(-15, 15, (1000, 3))
    sampled = grid.sample(points)*float(dose.DoseGridScaling)
    assert not np.ma.getmaskarray(sampled).any()
    np.testing.assert_allclose(sampled, 10+.1*points[:, 0], atol=5.1e-7, rtol=0)
    indices = np.random.default_rng(123).integers(0, 81, (1000, 3))
    nodes = grid.origin + (indices*grid.step) @ grid.basis.T
    np.testing.assert_allclose(grid.sample(nodes), grid.values[tuple(indices[:, ::-1].T)], atol=1e-7, rtol=0)
    assert grid.provenance['rotation_angle_degrees'] == pytest.approx(abs(angle), abs=1e-8)
    # Bounding-box corners outside the tilted original grid remain unsupported.
    corners = np.array([[grid.lower[0], grid.lower[1], grid.lower[2]],
                        [grid.upper[0], grid.upper[1], grid.upper[2]]])
    assert np.ma.getmaskarray(grid.sample(corners)).all()


@pytest.mark.parametrize('angle', [.025, .1, .5])
def test_analytic_box_dvh_and_course_curves(tmp_path, angle):
    course = fx.build_course(tmp_path, delivered=False,
                             references=[fx.dose_reference(description='HR-CTV')])
    dose = analytic_dose(course/'DICOM/RTDOSE/dose.dcm', angle)
    rs = box_structure(course/'DICOM/RTSTRUCT/rs.dcm')
    grid = prepare_rotated_dose(dose)
    histogram = grid.get_dvh(rs, 1)
    result = add_brachy_metrics(histogram, dvh._compute_metrics(histogram, None))
    # Uniform X dose distribution on [-10,10] mm: D90=9.2 Gy,
    # Dmean=10 Gy. dicompyler counts 21 one-mm contour slabs, giving
    # an 8.4 cc continuous box; hottest 2 cc => 11 - .1*2000/(20*21).
    # 0.12 Gy includes raster boundary/node phase and 0.01 Gy histogram bins.
    expected = {'D90Gy': 9.2, 'DmeanGy': 10., 'D2ccGy': 11.-.1*2000/(20*21)}
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=.12)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    rows = pd.read_parquet(course/'dvh_metrics.parquet')
    assert set(rows.dose_grid_resampling_status) == {METHOD}
    assert set(rows.dose_grid_coverage_status) == {'fully_covered'}
    assert rows.D90Gy.notna().all() and rows.D2ccGy.notna().all()
    qc = json.loads((course/'metadata/dvh_qc.json').read_text())
    assert qc['dose_grid_resampling']['method'] == METHOD
    curves = json.loads((course/'dvh_curves.json').read_text())['points']
    assert len(curves) == 2 and all(c['data'] for c in curves)


@pytest.mark.parametrize('base', [np.diag([-1.,-1.,1.]), np.diag([-1.,1.,-1.]),
                                 np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])])
@pytest.mark.parametrize('descending', [False, True])
def test_signed_axial_bases_and_descending_offsets(tmp_path, base, descending):
    dose = analytic_dose(tmp_path/'dose.dcm', .1, base, descending)
    grid = prepare_rotated_dose(dose)
    points = np.array([[1.,2.,3.],[-4.,-5.,-6.]])
    np.testing.assert_allclose(grid.sample(points)*float(dose.DoseGridScaling),
                               10+.1*points[:,0], atol=5.1e-7, rtol=0)


@pytest.mark.parametrize('angle', [1.01, 5, 45])
def test_large_rotation_fails_explicitly(tmp_path, angle):
    dose = analytic_dose(tmp_path/'dose.dcm', angle)
    with pytest.raises(DoseOrientationError, match='rotation_exceeds_1_degree'):
        prepare_rotated_dose(dose)


def test_nonorthonormal_and_nonuniform_fail_closed(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .1)
    dose.ImageOrientationPatient[3] = .001
    with pytest.raises(DoseOrientationError, match='non_orthonormal'):
        prepare_rotated_dose(dose)
    rotate(dose, .1)
    dose.GridFrameOffsetVector[10] += .1
    with pytest.raises(DoseOrientationError, match='invalid_or_nonuniform_grid'):
        prepare_rotated_dose(dose)


def test_original_support_partial_and_outside_with_true_zero(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .5)
    dose.PixelData = np.zeros_like(dose.pixel_array).tobytes()
    grid = prepare_rotated_dose(dose)
    rs = box_structure(tmp_path/'rs.dcm', x=(-25., 10.))
    assert grid.coverage(rs, 1)['status'] == 'partial_grid'
    covered = grid.get_dvh(rs, 1, calculate_full_volume=False)
    assert covered.volume > 0
    assert covered.mean < .01
    rs = box_structure(tmp_path/'rs.dcm', x=(-35., -30.))
    assert grid.coverage(rs, 1)['status'] == 'outside_grid'
    assert grid.get_dvh(rs, 1).volume == 0


def test_course_above_tolerance_records_refusal(tmp_path):
    course = fx.build_course(tmp_path, delivered=False)
    analytic_dose(course/'DICOM/RTDOSE/dose.dcm', 1.1)
    box_structure(course/'DICOM/RTSTRUCT/rs.dcm')
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course/'dvh_metrics.parquet')
    assert set(frame.dvh_computation_error) == {'rotated_dose_rotation_exceeds_1_degree'}
    assert frame.DmeanGy.isna().all()
    qc = json.loads((course/'metadata/dvh_qc.json').read_text())
    assert qc['dose_grid_resampling']['status'] == 'refused'


@pytest.mark.parametrize('orientation', [[1,0,0,0,1,0],[-1,0,0,0,-1,0],[-1,0,0,0,1,0],
                                       [0,1,0,-1,0,0],[0,-1,0,1,0,0],
                                       [1,0,0,0,-1,0], [0,1,0,1,0,0],[0,-1,0,-1,0,0]])
def test_exact_orientations_bypass_resampling(tmp_path, orientation):
    dose = pydicom.dcmread(fx.write_dose(tmp_path/'dose.dcm'))
    dose.ImageOrientationPatient = orientation
    assert prepare_rotated_dose(dose) is None
