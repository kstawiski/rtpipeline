"""Synthetic CT quantization fallback and baseline byte-equivalence checks."""
import copy
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from rt_utils import image_helper

from rtpipeline import rtstruct_geometry as geometry
from test_scope_resolution_degenerate_contours import _ct_slice, _contour, _rtstruct


@pytest.fixture(scope='module')
def baseline():
    source = subprocess.check_output(
        ['git', 'show', '939ad18:rtpipeline/rtstruct_geometry.py'],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    module = types.ModuleType('rtpipeline._geometry_939ad18')
    sys.modules[module.__name__] = module
    exec(compile(source, '939ad18:rtpipeline/rtstruct_geometry.py', 'exec'), module.__dict__)
    return module


def fixture(angle, rounded=True, kind='CLOSED_PLANAR', references=True):
    theta = np.deg2rad(angle)
    row, column = np.array([1., 0, 0]), np.array([0., np.cos(theta), np.sin(theta)])
    normal = np.cross(row, column)
    images, contours = [], []
    # Irregular, non-boundary vertices exercise fitted-plane as well as CT offset.
    xy = np.array([[8.2, 9.2], [30.2, 8.2], [37.2, 20.2], [29.2, 39.2], [10.2, 33.2]])
    for z in range(3):
        image = _ct_slice(z)
        image.ImageOrientationPatient = [*row, *column]
        image.ImagePositionPatient = (normal * (z * 2 + 0.003)).tolist()
        image.SliceThickness = 2
        images.append(image)
        points = np.array(image.ImagePositionPatient) + xy[:, :1] * row + xy[:, 1:] * column
        contour = _contour(np.round(points, 2) if rounded else points, kind)
        if references:
            ref = Dataset()
            ref.ReferencedSOPClassUID = image.SOPClassUID
            ref.ReferencedSOPInstanceUID = image.SOPInstanceUID
            contour.ContourImageSequence = Sequence([ref])
        contours.append(contour)
    ds = _rtstruct(contours)
    study, frame = Dataset(), Dataset()
    frame.FrameOfReferenceUID = images[0].FrameOfReferenceUID
    study.RTReferencedSeriesSequence = Sequence([])
    frame.RTReferencedStudySequence = Sequence([study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([frame])
    return ds, images


def mask(module, ds, images):
    # rt-utils needs no pixel bytes to rasterize; shape comes from pixel_array.
    # Supply an in-memory synthetic pixel array through its shape helper.
    return module.ScopedRTStruct(ds, images).get_roi_mask_by_name('hip_left')


@pytest.fixture(autouse=True)
def synthetic_shape(monkeypatch):
    monkeypatch.setattr(image_helper, 'create_empty_series_mask',
                        lambda series: np.zeros((64, 64, len(series)), dtype=bool))


@pytest.mark.parametrize('angle', [0.025, 0.1, 0.25])
@pytest.mark.parametrize('kind', ['CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'])
@pytest.mark.parametrize('references', [True, False])
def test_rounded_tilted_contours_are_projected_only_in_copy(angle, kind, references, baseline):
    ds, images = fixture(angle, kind=kind, references=references)
    exact, _ = fixture(angle, rounded=False, kind=kind, references=False)
    before = copy.deepcopy(ds)
    assert baseline.resolve_roi_scopes(ds, images)[1].code is not None
    result = geometry.resolve_roi_scopes(ds, images)[1]
    assert result.code is None
    assert 0.001 < result.contour_quantization_projection_mm <= 0.01
    assert result.projection_metadata['geometry_basis'] == 'quantized_plane_projection'
    assert ds == before
    for contour, image in zip(result.contours, images):
        assert geometry.plane_offset_mm(np.array(contour.ContourData).reshape(-1, 3), image) < 1e-10
    assert mask(geometry, ds, images).tobytes() == mask(baseline, exact, images).tobytes()


@pytest.mark.parametrize('offset', [0., 0.0009])
def test_strict_axial_masks_and_contour_bytes_are_baseline_identical(offset, baseline):
    ds, images = fixture(0., rounded=False)
    for item in ds.ROIContourSequence[0].ContourSequence:
        values = np.array(item.ContourData).reshape(-1, 3)
        values[:, 2] += offset
        item.ContourData = values.ravel().tolist()
    old = baseline.resolve_roi_scopes(ds, images)[1]
    new = geometry.resolve_roi_scopes(ds, images)[1]
    assert old.code is new.code is None
    assert new.projection_metadata == {}
    assert old.contours == new.contours
    assert mask(baseline, ds, images).tobytes() == mask(geometry, ds, images).tobytes()


@pytest.mark.parametrize('offset', [0.010001, 0.02])
def test_excess_ct_plane_offset_keeps_original_code(offset, baseline):
    ds, images = fixture(0., rounded=False)
    for contour in ds.ROIContourSequence[0].ContourSequence:
        points = np.array(contour.ContourData).reshape(-1, 3)
        points[:, 2] += offset
        contour.ContourData = points.ravel().tolist()
    old = baseline.resolve_roi_scopes(ds, images)[1]
    new = geometry.resolve_roi_scopes(ds, images)[1]
    assert new.code == old.code == 'ROI_UNRESOLVED_SOURCE_SCOPE'
    assert new.detail == old.detail
    assert new.projection_metadata == {}


def test_excess_fitted_plane_residual_keeps_original_code(baseline):
    ds, images = fixture(0., rounded=False)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    points = np.array(contour.ContourData).reshape(-1, 3)
    points[2, 2] += 0.08
    contour.ContourData = points.ravel().tolist()
    _, _, vt = np.linalg.svd(points - points[0], full_matrices=False)
    assert np.max(np.abs((points-points[0]) @ vt[-1])) > 0.02
    old = baseline.resolve_roi_scopes(ds, images)[1]
    new = geometry.resolve_roi_scopes(ds, images)[1]
    assert new.code == old.code == 'ROI_CONTOUR_PARTIALLY_UNPARSEABLE'
    assert new.projection_metadata == {}


@pytest.mark.parametrize('points', [[(10, 10, 0)], [(10, 10, 0), (11, 10, 0)], [(10, 10, 0)] * 4])
def test_arealess_slivers_keep_original_codes(points, baseline):
    ds = _rtstruct([_contour(points)])
    images = [_ct_slice(0)]
    assert geometry.resolve_roi_scopes(ds, images)[1].code == baseline.resolve_roi_scopes(ds, images)[1].code == 'ROI_CONTOUR_UNPARSEABLE'


@pytest.mark.parametrize('defect', ['missing_reference', 'wrong_frame', 'ambiguous_plane', 'out_of_bounds', 'declared_count', 'open'])
def test_fallback_cannot_relax_other_geometry_constraints(defect, baseline):
    ds, images = fixture(.1)
    contours = ds.ROIContourSequence[0].ContourSequence
    if defect == 'missing_reference':
        contours[0].ContourImageSequence[0].ReferencedSOPInstanceUID = '1.2.3.999'
    elif defect == 'wrong_frame':
        ds.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = '1.2.3.999'
    elif defect == 'ambiguous_plane':
        del contours[0].ContourImageSequence
        duplicate = copy.deepcopy(images[0])
        duplicate.SOPInstanceUID = '1.2.3.999'
        images.append(duplicate)
    elif defect == 'out_of_bounds':
        values = np.array(contours[0].ContourData).reshape(-1, 3)
        values[:, 0] += 100
        contours[0].ContourData = values.ravel().tolist()
    elif defect == 'declared_count':
        contours[0].NumberOfContourPoints = 90
    elif defect == 'open':
        contours[0].ContourGeometricType = 'OPEN_PLANAR'
    old = baseline.resolve_roi_scopes(ds, images)[1]
    new = geometry.resolve_roi_scopes(ds, images)[1]
    assert new.code == old.code
    assert new.code is not None
    assert new.projection_metadata == {}


def test_source_ledger_carries_projection_only_for_affected_identity():
    from rtpipeline.radiomics_source_inventory import source_ledger_rows
    ds, images = fixture(.1)
    path = Path('synthetic.dcm')
    fields = geometry.resolve_roi_scopes(ds, images)[1].projection_metadata
    task = types.SimpleNamespace(rs_path=path, roi_name='hip_left',
                                 stable_roi_identifier='rtstruct_roi_number:1',
                                 source='Manual', mask_identity='synthetic-mask')
    row = dict(segmentation_source=task.source, mask_identity=task.mask_identity,
               stable_roi_identifier=task.stable_roi_identifier,
               roi_original_name=task.roi_name, extraction_arm='primary', extraction_status='success', **fields)
    result = source_ledger_rows([path], [task], [row], datasets={path:ds})[0]
    assert result['disposition'] == 'extracted'
    assert result['geometry_basis'] == fields['geometry_basis']
    assert result['contour_quantization_projection_mm'] == fields['contour_quantization_projection_mm']


def test_parallel_rows_and_both_ledgers_carry_actual_scope_metadata(tmp_path, monkeypatch):
    from rtpipeline import radiomics_parallel as parallel
    import json
    ds, images = fixture(.1)
    scope = geometry.resolve_roi_scopes(ds, images)[1]
    source = tmp_path/'synthetic.dcm'
    task = types.SimpleNamespace(rs_path=str(source), roi_name='hip_left')
    cache = {(str(source), 123, 456): types.SimpleNamespace(by_name={'hip_left':scope})}
    monkeypatch.setitem(parallel._WORKER_STATE, 'builders', cache)
    monkeypatch.setattr(parallel, '_extract_one_with_geometry', lambda task: [
        {'roi_name':'hip_left', 'extraction_status':'success'}])
    rows = parallel._extract_one(task)
    assert rows[0]['geometry_basis'] == 'quantized_plane_projection'
    assert rows[0]['contour_quantization_projection_mm'] == scope.contour_quantization_projection_mm
    course = tmp_path/'synthetic-patient'/'synthetic-course'
    parallel._write_parallel_roi_ledger(course, [], rows, extracted=True)
    for filename in ('radiomics_ct_roi_ledger.json', 'radiomics_roi_ledger.json'):
        row = json.loads((course/'metadata'/filename).read_text())['course_roi'][0]
        assert row['geometry_basis'] == 'quantized_plane_projection'
        assert row['contour_quantization_projection_mm'] == scope.contour_quantization_projection_mm
    strict = types.SimpleNamespace(code=None, projection_metadata={})
    cache[(str(source),123,456)].by_name['hip_left'] = strict
    assert parallel._extract_one(task) == [{'roi_name':'hip_left','extraction_status':'success'}]


def test_unscoped_inventory_remains_strict_for_quantized_contours():
    from rtpipeline.roi_requiredness import inspect_rtstruct
    ds, images = fixture(0., rounded=False)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    points = np.array(contour.ContourData).reshape(-1,3)
    points[:,2] += np.array([.004,-.004,.004,-.004,.004])
    contour.ContourData = points.ravel().tolist()
    assert inspect_rtstruct(None, ds).named_rois[0].structural_code is not None
    assert geometry.resolve_roi_scopes(ds, images)[1].code is None


def test_exact_tolerance_boundary_and_above_it(baseline):
    image = _ct_slice(0.)
    for offset, accepted in [(0.01, True), (np.nextafter(0.01, np.inf), False)]:
        ds = _rtstruct([_contour([(5,5,offset),(20,5,offset),(20,20,offset),(5,20,offset)])])
        result = geometry.resolve_roi_scopes(ds, [image])[1]
        assert (result.code is None) == accepted
        if accepted:
            assert result.contour_quantization_projection_mm == 0.01
        else:
            assert result.code == baseline.resolve_roi_scopes(ds, [image])[1].code


def test_mixed_roi_leaves_strict_contour_coordinates_untouched():
    ds, images = fixture(.1)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    points = np.array(contour.ContourData).reshape(-1,3)
    normal = np.cross(np.array(images[0].ImageOrientationPatient[:3]),
                      np.array(images[0].ImageOrientationPatient[3:]))
    distances = (points-np.array(images[0].ImagePositionPatient))@normal
    contour.ContourData = (points-distances[:,None]*normal).ravel().tolist()
    result = geometry.resolve_roi_scopes(ds, images)[1]
    assert result.code is None
    assert result.contours[0].ContourData == contour.ContourData
    assert result.projection_metadata
