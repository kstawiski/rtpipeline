"""Synthetic producer controls for automatic slivers and sub-centigray dose."""
import copy
import ast
import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest

import dvh_plan_target_fixtures as fx
from rtpipeline import dvh, dvh_support
from rtpipeline.auto_rtstruct import _record_rs_auto_origin, RS_AUTO_ORIGIN_COPY
from rtpipeline.dvh_rotation import RotatedDoseGrid, prepare_rotated_dose
from test_dvh_coverage_dispositions import contour, structure
from test_dvh_grid_support import boxes
from test_dvh_rotation import analytic_dose, box_structure


AUTO = {'source': 'AutoRTS', 'origin': RS_AUTO_ORIGIN_COPY, 'rs_auto_sha256': 'a' * 64}


@pytest.mark.parametrize('source,status', [(AUTO, 'nonmeasurable_auto_sliver'),
                                         (None, 'degenerate_contours')])
@pytest.mark.parametrize('count', [1, 2, 3])
def test_sliver_precedes_grid_comparison(source, status, count):
    rs = structure([contour([[1, 1, 1], [2, 1, 1], [1, 1, 1]])] * count)
    # No usable RTDOSE geometry: the disposition must precede grid comparison.
    expected = dvh_support.roi_nonvolume_coverage(rs, 1, automatic_source=source)
    assert expected['status'] == status
    assert expected['evidence'] == {'contour_count': count, 'distinct_point_counts': [2] * count,
                                    'automatic_source': source}
    assert dvh_support.rtstruct_grid_coverage(rs, 1, object(), automatic_source=source) == expected
    assert dvh.classify_zero_dose_roi_geometry(rs, 1, object(), automatic_source=source) == expected
    assert object.__new__(RotatedDoseGrid).coverage(rs, 1, automatic_source=source) == expected


@pytest.mark.parametrize('x0,x1,status', [(2, 4, 'fully_covered'), (-2, 4, 'partial_grid'),
                                       (20, 22, 'outside_grid')])
def test_automatic_polygon_with_stray_sliver_keeps_polygon_coverage(x0, x1, status):
    rs, dose = boxes(x0, x1)
    expected = dvh_support.rtstruct_grid_coverage(rs, 1, dose)
    rs.ROIContourSequence[0].ContourSequence.append(copy.deepcopy(rs.ROIContourSequence[0].ContourSequence[0]))
    item = rs.ROIContourSequence[0].ContourSequence[-1]
    item.ContourData = [100, 100, 1, 101, 100, 1]
    item.NumberOfContourPoints = 2
    result = dvh_support.rtstruct_grid_coverage(rs, 1, dose, automatic_source=AUTO)
    assert result == expected
    assert result['status'] == status


@pytest.mark.parametrize('provenance', ['matching', 'missing', 'stale', 'manual', 'custom'])
@pytest.mark.parametrize('rotated', [False, True])
def test_sliver_publication_retains_evidence_without_dvh(tmp_path, monkeypatch, provenance, rotated):
    course = fx.build_course(tmp_path, rois=[('Organ', 'ORGAN', fx.DOSED_LEFT_X)])
    if rotated:
        analytic_dose(course / 'DICOM/RTDOSE/dose.dcm', .1)
    path = course / 'DICOM/RTSTRUCT/rs.dcm'
    rs = pydicom.dcmread(path)
    for item in rs.ROIContourSequence[0].ContourSequence:
        item.ContourData = item.ContourData[:6]
        item.NumberOfContourPoints = 2
    if provenance != 'manual':
        # Distinct identity keeps the original manual ROI as a separate row.
        rs.SOPInstanceUID = rs.file_meta.MediaStorageSOPInstanceUID = fx.OTHER_RTSTRUCT_UID
        path = course / 'RS_auto.dcm'
        if provenance == 'custom':
            path = course / 'Toy' / 'rtstruct.dcm'
            path.parent.mkdir()
            monkeypatch.setattr(dvh, 'list_custom_model_outputs', lambda *_: [('Toy', path.parent)])
            # This fixture's CT has an unrelated invented frame. Isolate the
            # source-disposition test from CT/NIfTI source-selection gates.
            monkeypatch.setattr(dvh, '_ct_frame_of_reference_uids', lambda *_: {fx.FRAME_OF_REFERENCE_UID})
    rs.save_as(path, enforce_file_format=True)
    if provenance in {'matching', 'stale'}:
        _record_rs_auto_origin(course, path, RS_AUTO_ORIGIN_COPY)
    if provenance == 'stale':
        rs.StructureSetLabel = 'Changed'
        rs.save_as(path, enforce_file_format=True)

    original = dvh.dvhcalc.get_dvh
    rotated_original = RotatedDoseGrid.get_dvh

    def guarded(call):
        def wrapped(rs, *args, **kwargs):
            if str(rs.SOPInstanceUID) == str(rs_auto_uid):
                pytest.fail('Area-less ROI reached the DVH sampler')
            return call(rs, *args, **kwargs)
        return wrapped
    rs_auto_uid = rs.SOPInstanceUID
    monkeypatch.setattr(dvh.dvhcalc, 'get_dvh', guarded(original))
    def guarded_rotated(self, rs, *args, **kwargs):
        return guarded(lambda rs, *a, **kw: rotated_original(self, rs, *a, **kw))(rs, *args, **kwargs)
    monkeypatch.setattr(RotatedDoseGrid, 'get_dvh', guarded_rotated)
    assert dvh.dvh_for_course(course, parallel_workers=1)
    status = 'nonmeasurable_auto_sliver' if provenance in {'matching', 'custom'} else 'degenerate_contours'
    source_label = 'Manual' if provenance == 'manual' else 'CustomModel:Toy' if provenance == 'custom' else 'AutoRTS'
    for table in (pd.read_parquet(course / 'dvh_metrics.parquet'), pd.read_excel(course / 'dvh_metrics.xlsx')):
        row = table.loc[table.Segmentation_Source.eq(source_label)].iloc[0]
        assert row.dvh_nonmeasurement_class == row.dose_metric_status == status
        assert row.row_status == 'not_computed'
        assert pd.isna(row.DmeanGy) and pd.isna(row.DmaxGy)
        evidence = json.loads(row.dvh_nonmeasurement_evidence)
        assert evidence['contour_count'] == 3
        assert evidence['distinct_point_counts'] == [2, 2, 2]
        if provenance == 'matching':
            assert evidence['automatic_source']['rs_auto_sha256'] == dvh_support.sha256_file(path)
            assert evidence['automatic_source']['source'] == 'AutoRTS'
        elif provenance == 'custom':
            assert evidence['automatic_source']['rtstruct_sha256'] == dvh_support.sha256_file(path)
            assert evidence['automatic_source']['source'] == source_label
        else:
            assert evidence['automatic_source'] is None
    if provenance == 'matching':
        assert dvh._is_dvh_up_to_date(course, course / 'dvh_metrics.xlsx')
        (course / 'metadata/rs_auto_origin.json').unlink()
        assert not dvh._is_dvh_up_to_date(course, course / 'dvh_metrics.xlsx')


@pytest.mark.parametrize('level', [0., .002, .009, .01, .02])
@pytest.mark.parametrize('aligned', [False, True])
def test_axial_summary_uses_working_mask_samples(tmp_path, level, aligned):
    course = fx.build_course(tmp_path, rois=[('Organ', 'ORGAN', fx.DOSED_LEFT_X)])
    dose_path = course / 'DICOM/RTDOSE/dose.dcm'
    dose = pydicom.dcmread(dose_path)
    dose.DoseGridScaling = 1e-6
    values = np.full(dose.pixel_array.shape, .5)  # outside-mask dose must not enter the summary
    # Dicompyler's rectangle mask includes x=4..16, y=8..30, z=2,4,6.
    inside = np.linspace(level / 4, level, 12 * 7).reshape(12, 7)
    values[1:4, 4:16, 2:9] = inside
    dose.PixelData = np.rint(values / 1e-6).astype('<u4').tobytes()
    dose.save_as(dose_path, enforce_file_format=True)
    if aligned:
        rs_path = course / 'DICOM/RTSTRUCT/rs.dcm'
        rs = pydicom.dcmread(rs_path)
        for item in rs.ROIContourSequence[0].ContourSequence:
            points = np.asarray(item.ContourData).reshape(-1, 3)
            points[:, 2] += .5
            item.ContourData = points.ravel().tolist()
        rs.save_as(rs_path, enforce_file_format=True)
    rs = pydicom.dcmread(course / 'DICOM/RTSTRUCT/rs.dcm')
    working, _ = dvh.dose_on_contour_planes(rs, dose, 1)
    histogram = dvh.dvhcalc.get_dvh(rs, working, 1)
    # Independent rectangular sample selection, with native z interpolation.
    parser = dvh._dicomparser.DicomParser(working)
    samples = np.concatenate([parser.GetDoseGrid(z)[4:16, 2:9].ravel() * 1e-6
                              for z in ([2.5, 4.5, 6.5] if aligned else [2, 4, 6])])
    result = dvh._compute_metrics(histogram, 20., lambda: dvh._rtstruct_dose_summary(rs, working, 1))
    if samples.max() < .01:
        assert result['DmeanGy'] == pytest.approx(samples.mean(), abs=1e-15)
        assert result['DmaxGy'] == samples.max()
        assert result['dose_summary_basis'] == 'unbinned_subcentigray'
        assert result['Dmean%'] == result['DmeanGy'] / 20 * 100
        assert result['Dmax%'] == result['DmaxGy'] / 20 * 100
        assert result['IntegralDose_Gycm3'] == result['DmeanGy'] * histogram.volume
    else:
        assert result['DmeanGy'] == histogram.mean
        assert result['DmaxGy'] == histogram.max
        assert result['dose_summary_basis'] == 'binned'
    assert dvh.dvh_for_course(course, parallel_workers=1)
    row = pd.read_parquet(course / 'dvh_metrics.parquet').iloc[0]
    assert row.DmeanGy == result['DmeanGy']
    assert row.DmaxGy == result['DmaxGy']
    assert row.dose_summary_basis == result['dose_summary_basis']


@pytest.mark.parametrize('level', [.01, .02, 20.])
def test_multibin_summary_is_lazy_and_metric_bytes_stay_exact(tmp_path, level):
    course = fx.build_course(tmp_path, rois=[('Organ', 'ORGAN', fx.WHOLE_X)])
    dose_path = course / 'DICOM/RTDOSE/dose.dcm'
    dose = pydicom.dcmread(dose_path)
    values = np.zeros(dose.pixel_array.shape, dtype='<u4')
    values[:, :, :fx.GRID_SIZE // 2] = round(level / float(dose.DoseGridScaling))
    dose.PixelData = values.tobytes()
    dose.save_as(dose_path, enforce_file_format=True)
    histogram = dvh.dvhcalc.get_dvh(str(course / 'DICOM/RTSTRUCT/rs.dcm'),
                                    str(dose_path), 1)
    assert histogram.counts.size > 1
    result = dvh._compute_metrics(histogram, 20., lambda: pytest.fail('Multibin summary was sampled'))
    # Execute the merge-head metric function against exactly the same histogram.
    source = subprocess.check_output(
        ['git', 'show', '2875df2:rtpipeline/dvh.py'], cwd=Path(dvh.__file__).parents[1], text=True)
    original = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_compute_metrics')
    namespace = dict(vars(dvh))
    exec(compile(ast.Module(body=[original], type_ignores=[]), '<merge-head-metrics>', 'exec'), namespace)
    expected = namespace['_compute_metrics'](histogram, 20.)
    assert result.pop('dose_summary_basis') == 'binned'
    assert result.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            assert struct.pack('!d', result[key]) == struct.pack('!d', value), key
        else:
            assert result[key] == value, key


@pytest.mark.parametrize('bound', [False, True])
def test_positive_subcentigray_targets_keep_plan_binding_disposition(tmp_path, bound):
    references = [fx.dose_reference(number=1, description='PTV1')]
    if bound:
        references.append(fx.dose_reference(number=2, description='PTV2'))
    course = fx.build_course(tmp_path, references=references)  # original 0.005-Gy fixture
    assert dvh.dvh_for_course(course, parallel_workers=1)
    frame = pd.read_parquet(course / 'dvh_metrics.parquet').set_index('ROI_OriginalName')
    for name in ('PTV2', 'CTV2'):
        assert frame.loc[name, 'DmeanGy'] == pytest.approx(.005, abs=1e-15)
        assert frame.loc[name, 'DmaxGy'] == .005
        assert frame.loc[name, 'dose_summary_basis'] == 'unbinned_subcentigray'
    qc = json.loads((course / 'metadata/dvh_qc.json').read_text())
    expected = ('course_quarantined_pending_plan_target_reconciliation' if bound
                else 'unbound_near_zero_targets_excluded')
    assert qc['plan_target_reconciliation']['decision'] == expected
    if bound:
        assert 'plan_bound_target_near_zero' in qc['plan_target_reconciliation']['quarantine_rules']
    else:
        assert frame.loc['PTV2', 'dose_metric_status'] == dvh.EXCLUDED_TARGET_NOT_BOUND_STATUS
        assert frame.loc['CTV2', 'dose_metric_status'] == dvh.EXCLUDED_TARGET_NOT_BOUND_STATUS


def test_zero_binned_maximum_uses_available_samples_even_with_multiple_bins():
    histogram = SimpleNamespace(bincenters=np.array([.005, .015]), counts=np.array([1., 0.]),
                                mean=0., max=0., min=0.)
    result = dvh._compute_metrics(histogram, None, {'mean': .001, 'max': .009})
    assert result['DmeanGy'] == .001
    assert result['DmaxGy'] == .009
    assert result['dose_summary_basis'] == 'unbinned_subcentigray'


def test_collapsed_histogram_cannot_claim_zero_without_samples():
    histogram = SimpleNamespace(bincenters=np.array([.5]), counts=np.array([1.]),
                                mean=0., max=0., min=0.)
    with pytest.raises(ValueError, match='unbinned in-mask dose samples'):
        dvh._compute_metrics(histogram, None)


def test_unbinned_summary_preserves_xor_holes(tmp_path):
    course = fx.build_course(tmp_path, rois=[('Organ', 'ORGAN', fx.DOSED_LEFT_X)])
    rs = pydicom.dcmread(course / 'DICOM/RTSTRUCT/rs.dcm')
    for outer in list(rs.ROIContourSequence[0].ContourSequence):
        hole = copy.deepcopy(outer)
        z = float(outer.ContourData[2])
        hole.ContourData = [8, 10, z, 12, 10, z, 12, 22, z, 8, 22, z]
        rs.ROIContourSequence[0].ContourSequence.append(hole)
    dose = pydicom.dcmread(course / 'DICOM/RTDOSE/dose.dcm')
    dose.DoseGridScaling = 1e-6
    values = np.full(dose.pixel_array.shape, .003)
    # Bright samples in the hole must not contribute to the in-mask dose.
    values[:, 6:12, 4:7] = .009
    dose.PixelData = np.rint(values / 1e-6).astype('<u4').tobytes()
    histogram = dvh.dvhcalc.get_dvh(rs, dose, 1)
    result = dvh._compute_metrics(histogram, None, lambda: dvh._rtstruct_dose_summary(rs, dose, 1))
    assert result['DmeanGy'] == pytest.approx(.003, abs=1e-15)
    assert result['DmaxGy'] == .003


@pytest.mark.parametrize('level', [0., .007])
def test_rotated_unbinned_summary_retains_support_and_slab_weights(tmp_path, level):
    course = fx.build_course(tmp_path, rois=[('Organ', 'ORGAN', fx.DOSED_LEFT_X)])
    path = course / 'DICOM/RTDOSE/dose.dcm'
    dose = analytic_dose(path, .1)
    # Uniform supported dose; unsupported exterior samples are excluded.
    dose.DoseGridScaling = 1e-6
    dose.PixelData = np.full(dose.pixel_array.shape, round(level / 1e-6), dtype='<u4').tobytes()
    dose.save_as(path, enforce_file_format=True)
    rs = pydicom.dcmread(course / 'DICOM/RTSTRUCT/rs.dcm')
    histogram = prepare_rotated_dose(dose).get_dvh(rs, 1)
    result = dvh._compute_metrics(histogram, None)
    assert result['DmeanGy'] == pytest.approx(level, abs=1e-15)
    assert result['DmaxGy'] == pytest.approx(level, abs=1e-15)
    assert result['dose_summary_basis'] == 'unbinned_subcentigray'


def test_rotated_subcentigray_mean_uses_physical_slab_weights(tmp_path):
    dose = analytic_dose(tmp_path / 'dose.dcm', .1)
    dose.DoseGridScaling = 1e-8
    # A linear field in z makes unequal contour-plane weights observable.
    from test_dvh_rotation import rotate
    basis = rotate(dose, .1)
    zz, yy, xx = np.meshgrid(np.array(dose.GridFrameOffsetVector), np.arange(81) * .5,
                             np.arange(81) * .5, indexing='ij')
    physical = np.stack((xx, yy, zz), axis=-1) @ basis.T + np.asarray(dose.ImagePositionPatient)
    values = .005 + physical[..., 2] * .0001
    dose.PixelData = np.rint(values / 1e-8).astype('<u4').tobytes()
    rs = box_structure(tmp_path / 'rs.dcm', zs=[-4, -2, 4])
    grid = prepare_rotated_dose(dose)
    histogram = grid.get_dvh(rs, 1)
    result = dvh._compute_metrics(histogram, None)
    # Equal-area planes at z=-4,-2,4 have slab weights 2,4,6 mm.
    expected = np.average(.005 + np.array([-4, -2, 4]) * .0001, weights=[2, 4, 6])
    assert result['DmeanGy'] == pytest.approx(expected, abs=1e-8)
    assert result['DmaxGy'] == pytest.approx(.0054, abs=1e-8)
    assert abs(result['DmeanGy'] - np.mean(.005 + np.array([-4, -2, 4]) * .0001)) > .0001
