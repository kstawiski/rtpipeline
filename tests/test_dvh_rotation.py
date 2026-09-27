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
    assert qc['dose_grid_resampling']['rotation_angle_degrees'] == pytest.approx(1.1)


@pytest.mark.parametrize('orientation', [[1,0,0,0,1,0],[-1,0,0,0,-1,0],[-1,0,0,0,1,0],
                                       [0,1,0,-1,0,0],[0,-1,0,1,0,0],
                                       [1,0,0,0,-1,0], [0,1,0,1,0,0],[0,-1,0,-1,0,0]])
def test_exact_orientations_bypass_resampling(tmp_path, orientation):
    dose = pydicom.dcmread(fx.write_dose(tmp_path/'dose.dcm'))
    dose.ImageOrientationPatient = orientation
    assert prepare_rotated_dose(dose) is None


def historical_dvh(tmp_path):
    """Load actual a61b96a code and its original on-disk identity inputs."""
    root = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(['git', 'show', 'a61b96a:rtpipeline/dvh.py'], cwd=root).decode()
    module = types.ModuleType('rtpipeline._rf14_baseline_dvh')
    sys.modules[module.__name__] = module
    shadow = tmp_path/'baseline_package'
    shadow.mkdir()
    module.__file__ = str(shadow/'dvh.py')
    exec(compile(code, module.__file__, 'exec'), module.__dict__)
    for name in module.DVH_MEASUREMENT_CODE_SOURCES:
        source = subprocess.check_output(['git', 'show', f'a61b96a:rtpipeline/{name}'], cwd=root)
        (shadow/name).write_bytes(source)
    return module


@pytest.mark.parametrize('orientation', [[1,0,0,0,1,0], [-1,0,0,0,-1,0], [-1,0,0,0,1,0],
                                       [0,1,0,-1,0,0], [0,-1,0,1,0,0],
                                       [1,0,0,0,-1,0], [0,1,0,1,0,0], [0,-1,0,-1,0,0]])
def test_axis_aligned_publication_bytes_against_git_a61b96a(tmp_path, monkeypatch, orientation):
    """Exact publication bytes except truthful code identity and its QC hash.

    The production QC source hash cannot be byte-identical after a source edit.
    We assert it DOES change, then compare every other QC byte and receipt
    content field. Workbook packaging time is fixed for both implementations.
    """
    import xlsxwriter.core
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel

    class FixedWorkbookTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, tzinfo=tz)
    monkeypatch.setattr(xlsxwriter.core, 'datetime', FixedWorkbookTime)
    baseline = historical_dvh(tmp_path)
    course = fx.build_course(tmp_path, rois=fx.NO_NEAR_ZERO_ROIS)
    path = course/'DICOM/RTDOSE/dose.dcm'
    dose = pydicom.dcmread(path)
    dose.ImageOrientationPatient = orientation
    basis = np.column_stack((orientation[:3], orientation[3:],
                             np.cross(orientation[:3], orientation[3:])))
    corners = np.array([[x,y,z] for x in (0.,38.) for y in (0.,38.) for z in (0.,8.)])
    dose.ImagePositionPatient = (-(corners @ basis.T).min(axis=0)).tolist()
    dose.save_as(path, enforce_file_format=True)
    dependency = materialize_stage_dependency(tmp_path/'configuration', 'dvh', {'enabled':True})

    def run(module):
        module._invalidate_dvh_outputs(course)
        assert module.dvh_for_course(course, parallel_workers=1)
        artifacts = {name: (course/name).read_bytes() for name in
                     ('dvh_metrics.parquet', 'dvh_metrics.xlsx', 'dvh_curves.json', 'metadata/dvh_qc.json')}
        receipt = write_stage_completion_sentinel(course, course/'.dvh_done', stage='dvh',
                                                  status='ok', configuration_dependency=dependency)
        return artifacts, receipt

    old, old_receipt = run(baseline)
    new, new_receipt = run(dvh)
    for name in ('dvh_metrics.parquet', 'dvh_metrics.xlsx', 'dvh_curves.json'):
        assert old[name] == new[name], name
    old_qc = json.loads(old['metadata/dvh_qc.json'])
    new_qc = json.loads(new['metadata/dvh_qc.json'])
    old_digest = old_qc['code_sources_sha256']
    new_digest = new_qc['code_sources_sha256']
    assert old_digest != new_digest
    assert old['metadata/dvh_qc.json'].replace(old_digest.encode(), new_digest.encode()) == new['metadata/dvh_qc.json']
    old_content = [{k: row[k] for k in ('path','role','binding','sha256','size_bytes') if k in row}
                   for row in old_receipt['outputs'] if row['role'] != 'dvh_qc']
    new_content = [{k: row[k] for k in ('path','role','binding','sha256','size_bytes') if k in row}
                   for row in new_receipt['outputs'] if row['role'] != 'dvh_qc']
    assert old_content == new_content
    assert old_receipt['content_closure_sha256'] != new_receipt['content_closure_sha256']
    assert 'dose_grid_resampling' not in new_qc
    assert 'dose_grid_resampling_status' not in pd.read_parquet(course/'dvh_metrics.parquet').columns


def test_partial_course_never_exports_whole_roi_values_or_curve(tmp_path):
    course = fx.build_course(tmp_path, delivered=False)
    analytic_dose(course/'DICOM/RTDOSE/dose.dcm', .5)
    box_structure(course/'DICOM/RTSTRUCT/rs.dcm', x=(-25., 10.))
    assert dvh.dvh_for_course(course, parallel_workers=1)
    rows = pd.read_parquet(course/'dvh_metrics.parquet')
    assert set(rows.dose_grid_coverage_status) == {'partial_grid'}
    assert rows[['D90Gy','D2ccGy','DmeanGy','D0.03ccGy','D0.1ccGy','D1ccGy']].isna().all().all()
    assert set(rows.D2cc_status) == {'partial_grid'}
    assert rows[['covered_D90Gy','covered_D2ccGy','covered_DmeanGy']].notna().all().all()
    assert not (course/'dvh_curves.json').exists()


def test_small_roi_has_no_d2cc(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .1)
    rs = box_structure(tmp_path/'rs.dcm', x=(-2.,2.), y=(-2.,2.), zs=[-1,0,1])
    histogram = prepare_rotated_dose(dose).get_dvh(rs, 1)
    metrics = add_brachy_metrics(histogram, dvh._compute_metrics(histogram, None))
    assert metrics['D2ccGy'] is None
    assert metrics['D2cc_status'] == 'roi_below_2cc'


def test_direct_mask_sampler_preserves_patient_coordinates_and_support(tmp_path):
    import SimpleITK as sitk
    dose = analytic_dose(tmp_path/'dose.dcm', .5)
    grid = prepare_rotated_dose(dose)
    ct = sitk.Image([50, 3, 3], sitk.sitkFloat32)
    ct.SetOrigin([-25., -1., -1.])
    values, support = grid.sample_image(ct)
    points = np.array([ct.TransformIndexToPhysicalPoint((x,1,1)) for x in range(50)])
    expected = grid.sample(points)
    np.testing.assert_array_equal(support[1,1], ~np.ma.getmaskarray(expected))
    np.testing.assert_allclose(values[1,1], expected.filled(0)*float(dose.DoseGridScaling))
    assert values[1,1,0] == 0 and not support[1,1,0]


def test_contour_planes_are_sampled_without_snap_or_centimm_rounding(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .5)
    grid = prepare_rotated_dose(dose)
    # Use a field with a z gradient so any plane shift changes the answer.
    zz, yy, xx = np.meshgrid(np.arange(81)*.5, np.arange(81)*.5, np.arange(81)*.5, indexing='ij')
    points = np.stack((xx,yy,zz), axis=-1) @ grid.basis.T + grid.origin
    dose.PixelData = np.rint((10+.4*points[...,2])/float(dose.DoseGridScaling)).astype('<u4').tobytes()
    grid = prepare_rotated_dose(dose)
    rs = box_structure(tmp_path/'rs.dcm', zs=[-1.1234, .8766, 2.8766])
    histogram = grid.get_dvh(rs, 1)
    assert histogram.mean == pytest.approx(10+.4*.8766, abs=.006)
    # Inspect physical sampler calls to pin the absence of 0.01 mm rounding.
    seen = []
    plane = grid.plane
    grid.plane = lambda z: (seen.append(float(z)) or plane(z))
    grid.get_dvh(rs, 1)
    assert sorted(seen) == [-1.1234, .8766, 2.8766]


@pytest.mark.parametrize('angle', [.025, .1, .5])
def test_brachy_like_radial_analytic_field(tmp_path, angle):
    """Smooth radial fall-off: 50/(1+r²/25) Gy inside a 10 mm sphere."""
    dose = analytic_dose(tmp_path/'dose.dcm', angle)
    grid = prepare_rotated_dose(dose)
    zz, yy, xx = np.meshgrid(np.arange(81)*.5, np.arange(81)*.5, np.arange(81)*.5, indexing='ij')
    points = np.stack((xx,yy,zz), axis=-1) @ grid.basis.T + grid.origin
    radius_squared = (points**2).sum(axis=-1)
    dose.PixelData = np.rint((50/(1+radius_squared/25))/float(dose.DoseGridScaling)).astype('<u4').tobytes()
    grid = prepare_rotated_dose(dose)
    rs = box_structure(tmp_path/'rs.dcm')
    angles = np.linspace(0, 2*np.pi, 256, endpoint=False)
    for roi in rs.ROIContourSequence:
        contours = []
        for z in np.arange(-9.5, 10., .5):
            r = np.sqrt(100-z*z)
            item = Dataset()
            item.ContourGeometricType = 'CLOSED_PLANAR'
            item.NumberOfContourPoints = len(angles)
            item.ContourData = np.column_stack((r*np.cos(angles), r*np.sin(angles), np.full_like(angles,z))).ravel().tolist()
            contours.append(item)
        roi.ContourSequence = Sequence(contours)
    hist = grid.get_dvh(rs, 1)
    metrics = add_brachy_metrics(hist, dvh._compute_metrics(hist, None))
    expected = {'D90Gy': 50/(1+(10*.9**(1/3))**2/25),
                'D2ccGy': 50/(1+(3*2000/(4*np.pi))**(2/3)/25),
                'DmeanGy': 3*50*25/1000*(10-5*np.arctan(2))}
    # Includes 0.5 mm spherical rasterization, trilinear interpolation, and
    # dicompyler's 0.01 Gy histogram bins; not a universal patient-dose bound.
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, abs=.25)


def test_decimal_frame_position_rounding_preserves_original_nodes(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .1)
    dose.GridFrameOffsetVector[10] += 1e-7
    grid = prepare_rotated_dose(dose)
    local = np.array([[3., 4., float(dose.GridFrameOffsetVector[10])]])
    point = local @ grid.basis.T + grid.origin
    assert float(grid.sample(point)[0]) == pytest.approx(grid.values[10, 8, 6], abs=1e-7)


def test_tolerance_boundary_and_nonaxial_contours(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', 1.)
    grid = prepare_rotated_dose(dose)
    assert grid is not None
    rs = box_structure(tmp_path/'rs.dcm')
    rs.ROIContourSequence[0].ContourSequence[0].ContourData[2] += .1
    with pytest.raises(DoseOrientationError, match='nonparallel_or_nonplanar_contours'):
        grid.get_dvh(rs, 1)


def test_nonuniform_contour_planes_use_physical_slab_volumes(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .1)
    grid = prepare_rotated_dose(dose)
    uniform = box_structure(tmp_path/'uniform.dcm', zs=[-10., -5., 0., 5., 10.])
    # Adding an almost duplicate internal plane must not shrink the entire ROI
    # by 500x, as dicompyler's global minimum-thickness rule would do.
    nonuniform = box_structure(tmp_path/'nonuniform.dcm', zs=[-10., -5., 0., .01, 5., 10.])
    first, second = grid.get_dvh(uniform, 1), grid.get_dvh(nonuniform, 1)
    assert second.volume == pytest.approx(first.volume, rel=1e-12)
    assert second.mean == pytest.approx(first.mean, abs=1e-12)
    assert first.volume > 9.


@pytest.mark.parametrize('angle', [.025, .1, .5])
def test_tilted_rtstruct_contours_keep_their_patient_geometry(tmp_path, angle):
    course = fx.build_course(tmp_path, delivered=False)
    dose = analytic_dose(course/'DICOM/RTDOSE/dose.dcm', angle)
    grid = prepare_rotated_dose(dose)
    rs = box_structure(course/'DICOM/RTSTRUCT/rs.dcm')
    for roi in rs.ROIContourSequence:
        for contour in roi.ContourSequence:
            points = np.array(contour.ContourData).reshape(-1,3) @ grid.basis.T
            contour.ContourData = points.ravel().tolist()
    rs.save_as(course/'DICOM/RTSTRUCT/rs.dcm', enforce_file_format=True)
    hist = grid.get_dvh(rs, 1)
    metrics = add_brachy_metrics(hist, dvh._compute_metrics(hist, None))
    assert metrics['DmeanGy'] == pytest.approx(10., abs=.06)
    assert metrics['D90Gy'] == pytest.approx(9.2, abs=.12)
    assert metrics['D2ccGy'] == pytest.approx(11.-.1*2000/(20*21), abs=.12)
    assert metrics['dose_grid_contour_plane_tilt_degrees'] == pytest.approx(angle, abs=1e-7)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course/'dvh_metrics.parquet')
    assert set(frame.dose_metric_status) == {'computed'}
    assert (course/'dvh_curves.json').is_file()


def test_refused_rebuild_removes_stale_curves(tmp_path):
    course = fx.build_course(tmp_path, delivered=False)
    analytic_dose(course/'DICOM/RTDOSE/dose.dcm', .1)
    box_structure(course/'DICOM/RTSTRUCT/rs.dcm')
    assert dvh.dvh_for_course(course, parallel_workers=1)
    assert (course/'dvh_curves.json').is_file()
    analytic_dose(course/'DICOM/RTDOSE/dose.dcm', 1.1)
    # Explicit rebuild isolates publication behavior from filesystem timestamp
    # precision and keeps this test focused on stale-curve invalidation.
    dvh._invalidate_dvh_outputs(course)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    assert not (course/'dvh_curves.json').exists()


def test_rounded_tilted_contours_have_bounded_recorded_projection(tmp_path):
    dose = analytic_dose(tmp_path/'dose.dcm', .1)
    grid = prepare_rotated_dose(dose)
    rs = box_structure(tmp_path/'rs.dcm')
    for roi in rs.ROIContourSequence:
        for contour in roi.ContourSequence:
            points = np.array(contour.ContourData).reshape(-1,3) @ grid.basis.T
            contour.ContourData = np.round(points, 2).ravel().tolist()
    hist = grid.get_dvh(rs, 1)
    metrics = add_brachy_metrics(hist, dvh._compute_metrics(hist, None))
    assert 0 < metrics['dose_grid_contour_max_projection_mm'] <= .01
    assert metrics['DmeanGy'] == pytest.approx(10., abs=.06)
    # An extra nonplanar displacement is refused rather than absorbed by fit.
    rs.ROIContourSequence[0].ContourSequence[0].ContourData[2] += .1
    with pytest.raises(DoseOrientationError, match='nonparallel_or_nonplanar'):
        grid.get_dvh(rs, 1)
