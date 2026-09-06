"""Production-shaped synthetic regressions for D07, D03 and D04."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from rt_utils import RTStructBuilder

from test_science_batch_c import _build_real_rtstruct
from rtpipeline import custom_structures_rtstruct as custom
from rtpipeline import radiomics_parallel as parallel
from rtpipeline.roi_requiredness import inspect_rtstruct
from rtpipeline.rtstruct_geometry import (
    NONVOLUMETRIC_CODES, ROIContourDisposition, create_scoped_rtstruct,
    resolve_roi_scopes, contour_geometry,
)
from rtpipeline.rtstruct_identity import (
    assign_derived_identity, require_rtstruct_identity, RTStructIdentityError,
)
from rtpipeline.radiomics_ct_contract import (
    classify_ct_roi, stable_rtstruct_roi_identity, CT_EXTRACTION_ARMS,
)


def fixture(tmp_path):
    ct = tmp_path/'ct'
    ct.mkdir()
    rt = _build_real_rtstruct(ct, side=12)
    source = tmp_path/'RS.dcm'
    rt.save(str(source))
    return ct, source, rt


def task(source, name, number=1):
    ds = pydicom.dcmread(source)
    return parallel._RoiTask(
        source='Manual', rs_path=str(source), roi_name=name,
        course_dir=str(source.parent), series_uid=str(ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID),
        mask_identity=str(ds.SOPInstanceUID), stable_roi_identifier=f'rtstruct_roi_number:{number}',
        decision=classify_ct_roi('Manual', name), run_identifier='synthetic', code_revision='test',
        configured_parameter_hashes={arm:'configured' for arm in CT_EXTRACTION_ARMS},
        effective_parameter_hashes={arm:'effective' for arm in CT_EXTRACTION_ARMS}, required=False,
    )


def test_real_custom_publication_new_content_uid_parent_and_determinism(tmp_path, monkeypatch):
    ct, source, rt = fixture(tmp_path)
    before = source.read_bytes()
    contract = SimpleNamespace(authoritative_rtstruct_path=source, planning_ct_dir=ct,
        planning_ct={'series_instance_uid':str(rt.series_data[0].SeriesInstanceUID)})
    monkeypatch.setattr(custom, 'load_course_contract', lambda _:contract)
    config = tmp_path/'custom.yaml'
    config.write_text('custom_structures:\n  - name: target_copy\n    source_structures: [PTV]\n    operation: union\n')
    out = custom._create_custom_structures_rtstruct_unlocked(tmp_path, config, source)
    first = out.read_bytes()
    published = pydicom.dcmread(out)
    assert source.read_bytes() == before
    assert first != before
    assert published.SOPInstanceUID != rt.ds.SOPInstanceUID
    assert published.file_meta.MediaStorageSOPInstanceUID == published.SOPInstanceUID
    assert published.PredecessorStructureSetSequence[0].ReferencedSOPInstanceUID == rt.ds.SOPInstanceUID
    require_rtstruct_identity(out, require_derived=True)
    assert 'target_copy' in [r.ROIName for r in published.StructureSetROISequence]
    assert np.array_equal(RTStructBuilder.create_from(str(ct),str(out)).get_roi_mask_by_name('PTV'), rt.get_roi_mask_by_name('PTV'))
    meta = json.loads((tmp_path/'metadata'/'rs_custom_meta.json').read_text())
    assert meta['parent_sop_instance_uid'] == str(rt.ds.SOPInstanceUID)
    assert meta['rs_custom_sop_instance_uid'] == str(published.SOPInstanceUID)
    out2 = custom._create_custom_structures_rtstruct_unlocked(tmp_path, config, source)
    assert out2.read_bytes() == first
    assert pydicom.dcmread(out2).SOPInstanceUID == published.SOPInstanceUID
    with pytest.raises(RTStructIdentityError, match='STALE_UID'):
        require_rtstruct_identity(out, str(rt.ds.SOPInstanceUID))
    stale_task = replace(task(out, 'PTV'), mask_identity=str(rt.ds.SOPInstanceUID))
    with pytest.raises(RTStructIdentityError, match='STALE_UID'):
        parallel._extract_one(stale_task)
    changed = pydicom.dcmread(out)
    changed.StructureSetLabel = 'tampered'
    changed.save_as(out)
    with pytest.raises(RTStructIdentityError, match='CONTENT_UID'):
        stable_rtstruct_roi_identity(out, 'PTV')


def multiseries_fixture(tmp_path):
    ct, source, rt = fixture(tmp_path)
    ds = rt.ds
    roi0, contour0, obs0 = [copy.deepcopy(sequence[0]) for sequence in (ds.StructureSetROISequence, ds.ROIContourSequence, ds.RTROIObservationsSequence)]
    ds.StructureSetROISequence = Sequence([])
    ds.ROIContourSequence = Sequence([])
    ds.RTROIObservationsSequence = Sequence([])
    for i in range(15):
        roi, item, obs = map(copy.deepcopy, (roi0, contour0, obs0))
        roi.ROINumber = item.ReferencedROINumber = obs.ReferencedROINumber = obs.ObservationNumber = i+1
        roi.ROIName = f'bound_{i}' if i < 4 else f'unresolved_target_{i}'
        if i >= 4:
            outside = copy.deepcopy(item.ContourSequence[0])
            del outside.ContourImageSequence
            points = np.asarray(outside.ContourData).reshape(-1,3)
            points[:,2] += 30
            outside.ContourData = points.ravel().tolist()
            item.ContourSequence.append(outside)
        ds.StructureSetROISequence.append(roi)
        ds.ROIContourSequence.append(item)
        ds.RTROIObservationsSequence.append(obs)
    refs = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence
    stale = copy.deepcopy(refs[0])
    stale.SeriesInstanceUID = generate_uid()
    stale.ContourImageSequence[0].ReferencedSOPInstanceUID = generate_uid()
    refs.append(stale)
    ds.save_as(source)
    return ct, source, rt


def test_four_complete_rois_recover_eleven_remain_unresolved_no_clipping(tmp_path, monkeypatch):
    ct, source, rt = multiseries_fixture(tmp_path)
    before = source.read_bytes()
    with pytest.raises(Exception):
        RTStructBuilder.create_from(str(ct), str(source))
    scoped = create_scoped_rtstruct(ct, source)
    assert len(scoped.ds.StructureSetROISequence) == 15
    assert len(scoped.builder.ds.StructureSetROISequence) == 4
    assert sum(r.code is None for r in scoped.scopes.values()) == 4
    assert sum(r.code == 'ROI_UNRESOLVED_SOURCE_SCOPE' for r in scoped.scopes.values()) == 11
    for i in range(4):
        assert scoped.get_roi_mask_by_name(f'bound_{i}').sum() > 0
    for i in range(4,15):
        with pytest.raises(ROIContourDisposition, match='UNRESOLVED_SOURCE_SCOPE'):
            scoped.get_roi_mask_by_name(f'unresolved_target_{i}')
    monkeypatch.setattr(parallel, '_WORKER_STATE', {'ct_dir':ct, 'img':object(), 'extractor':object()})
    for i in range(4,15):
        rows = parallel._extract_one(task(source, f'unresolved_target_{i}', i+1))
        assert len(rows) == 2
        assert {r['roi_structural_code'] for r in rows} == {'ROI_UNRESOLVED_SOURCE_SCOPE'}
        assert all(r['native_mask_voxel_count'] is None for r in rows)
    assert source.read_bytes() == before


def test_unreferenced_contour_binds_by_geometry_and_bad_reference_never_snaps(tmp_path):
    ct, source, rt = fixture(tmp_path)
    expected = rt.get_roi_mask_by_name('PTV')
    del rt.ds.ROIContourSequence[0].ContourSequence[0].ContourImageSequence
    rt.ds.save_as(source)
    scoped = create_scoped_rtstruct(ct, source)
    assert np.array_equal(scoped.get_roi_mask_by_name('PTV'), expected)
    rt.ds.ROIContourSequence[0].ContourSequence[0].ContourData[2] += 0.1
    rt.ds.save_as(source)
    scoped = create_scoped_rtstruct(ct, source)
    assert scoped.scopes[1].code in {'ROI_CONTOUR_PARTIALLY_UNPARSEABLE', 'ROI_UNRESOLVED_SOURCE_SCOPE'}


def test_same_plane_in_distinct_series_is_ambiguous_not_merged(tmp_path):
    ct, source, rt = fixture(tmp_path)
    images = copy.deepcopy(rt.series_data)
    duplicate_series = copy.deepcopy(images)
    for image in duplicate_series:
        image.SeriesInstanceUID = '1.2.3.4'
        image.SOPInstanceUID = generate_uid()
    for contour in rt.ds.ROIContourSequence[0].ContourSequence:
        del contour.ContourImageSequence
    result = resolve_roi_scopes(rt.ds, images+duplicate_series)[1]
    assert result.code == 'ROI_UNRESOLVED_SOURCE_SCOPE'


def nonvolume_fixture(tmp_path):
    ct, source, rt = fixture(tmp_path)
    for i in range(11):
        roi = Dataset()
        roi.ROINumber = i+2
        roi.ROIName = f'Applicator{i}' if i < 8 else f'Marker{i}'
        roi.ReferencedFrameOfReferenceUID = rt.series_data[0].FrameOfReferenceUID
        item, contour = Dataset(), Dataset()
        item.ReferencedROINumber = i+2
        contour.ContourGeometricType = 'OPEN_NONPLANAR' if i < 8 else 'POINT'
        contour.ContourData = [2.,2.,1.,3.,4.,2.,4.,2.,3.] if i < 8 else [2.,2.,1.]
        contour.NumberOfContourPoints = len(contour.ContourData)//3
        item.ContourSequence = Sequence([contour])
        rt.ds.StructureSetROISequence.append(roi)
        rt.ds.ROIContourSequence.append(item)
    rt.ds.save_as(source)
    return ct, source, rt


def test_eight_applicators_three_points_have_explicit_identity_bound_nonmeasurements(tmp_path, monkeypatch):
    ct, source, rt = nonvolume_fixture(tmp_path)
    inventory = inspect_rtstruct(source)
    assert sum(r.structural_code == 'ROI_NONVOLUMETRIC_OPEN_NONPLANAR' for r in inventory.rois) == 8
    assert sum(r.structural_code == 'ROI_NONVOLUMETRIC_POINT' for r in inventory.rois) == 3
    monkeypatch.setattr(parallel, '_get_builder', lambda _:pytest.fail('nonvolumetric ROI reached mask builder'))
    monkeypatch.setattr(parallel, '_WORKER_STATE', {})
    rows = []
    for observation in inventory.rois[1:]:
        rows.extend(parallel._extract_one(task(source, observation.name, observation.roi_number)))
    assert len(rows) == 22
    assert {r['extraction_status'] for r in rows} == {'nonvolumetric_nonmeasurement'}
    assert all(r['roi_structural_code'] in NONVOLUMETRIC_CODES for r in rows)
    assert all(r['rtstruct_sop_instance_uid'] == str(rt.ds.SOPInstanceUID) for r in rows)
    from rtpipeline.utils import sanitize_rtstruct
    before = source.read_bytes()
    assert not sanitize_rtstruct(source)
    assert source.read_bytes() == before


@pytest.mark.parametrize('points', [[0.,0.,0.], [0.,0.,0.,1.,1.,1.,2.,2.,2.], [0.,0.,0.,1.,0.,0.,1.,1.,1.,0.,1.,0.]])
def test_invalid_closed_planar_geometry_stays_distinct(tmp_path, points, monkeypatch):
    ct, source, rt = fixture(tmp_path)
    contour = rt.ds.ROIContourSequence[0].ContourSequence[0]
    contour.ContourData = points
    contour.NumberOfContourPoints = len(points)//3
    rt.ds.ROIContourSequence[0].ContourSequence = Sequence([contour])
    rt.ds.save_as(source)
    assert not contour_geometry(contour)[1]
    observation = inspect_rtstruct(source).rois[0]
    assert observation.structural_code == 'ROI_CONTOUR_UNPARSEABLE'
    monkeypatch.setattr(parallel, '_get_builder', lambda _:pytest.fail('invalid contour reached mask builder'))
    rows = parallel._extract_one(task(source, 'PTV'))
    assert {r['extraction_status'] for r in rows} == {'invalid_contour_geometry'}


def test_organizer_uses_unique_complete_geometry_not_largest_series(tmp_path):
    from rtpipeline.organize import select_course_ct_series
    ct, source, rt = fixture(tmp_path)
    other = tmp_path/'other'
    other.mkdir()
    rt2 = _build_real_rtstruct(other, n_slices=6, side=12)
    references = rt.ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence
    references.append(copy.deepcopy(rt2.ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]))
    rt.ds.save_as(source)
    first_uid = str(rt.series_data[0].SeriesInstanceUID)
    second_uid = str(rt2.series_data[0].SeriesInstanceUID)
    a = [SimpleNamespace(path=p, series_uid=first_uid) for p in ct.glob('*.dcm')]
    b = [SimpleNamespace(path=p, series_uid=second_uid) for p in other.glob('*.dcm')]
    selected, status = select_course_ct_series({'P1':{'study':{first_uid:a, second_uid:b}}}, 'P1', source, 'study')
    assert status == 'referenced_geometry_scope'
    assert selected == a
    roi, item = copy.deepcopy(rt2.ds.StructureSetROISequence[0]), copy.deepcopy(rt2.ds.ROIContourSequence[0])
    roi.ROINumber = item.ReferencedROINumber = 2
    roi.ROIName = 'other_scope_target'
    rt.ds.StructureSetROISequence.append(roi)
    rt.ds.ROIContourSequence.append(item)
    rt.ds.save_as(source)
    results = resolve_roi_scopes(rt.ds, rt.series_data+rt2.series_data)
    assert results[1].source_series_uids == (first_uid,)
    assert results[2].source_series_uids == (second_uid,)
    assert all(result.code is None for result in results.values())
    selected, status = select_course_ct_series({'P1':{'study':{first_uid:a, second_uid:b}}}, 'P1', source, 'study')
    assert selected is None
    assert status == 'unresolved_multiseries_scope'


def test_concurrent_real_publication_reuses_one_content_identity(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    ct, source, rt = fixture(tmp_path)
    contract = SimpleNamespace(authoritative_rtstruct_path=source, planning_ct_dir=ct,
        planning_ct={'series_instance_uid':str(rt.series_data[0].SeriesInstanceUID)})
    monkeypatch.setattr(custom, 'load_course_contract', lambda _:contract)
    # Isolate locking from contract-file fixture construction, but require real
    # persisted, self-validating bytes before reuse is accepted.
    def stale(path, *_):
        if not path.exists():
            return True
        require_rtstruct_identity(path, require_derived=True)
        return False
    monkeypatch.setattr(custom, '_is_rs_custom_stale', stale)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _:custom._create_custom_structures_rtstruct(tmp_path, None, source), range(2)))
    assert paths[0] == paths[1]
    assert require_rtstruct_identity(paths[0], require_derived=True) != str(rt.ds.SOPInstanceUID)


def test_valid_two_point_open_planar_is_not_malformed():
    contour = Dataset()
    contour.ContourGeometricType = 'OPEN_PLANAR'
    contour.ContourData = [0.,0.,0.,1.,1.,1.]
    contour.NumberOfContourPoints = 2
    assert contour_geometry(contour) == ('OPEN_PLANAR', True)


def test_serial_nonvolumetric_inventory_uses_explicit_identity(tmp_path, monkeypatch):
    from rtpipeline.radiomics import _rtstruct_masks
    ct, source, rt = nonvolume_fixture(tmp_path)
    rt.ds.StructureSetROISequence = Sequence(rt.ds.StructureSetROISequence[1:])
    rt.ds.ROIContourSequence = Sequence(rt.ds.ROIContourSequence[1:])
    rt.ds.save_as(source)
    outcomes = []
    monkeypatch.setattr(RTStructBuilder, 'create_from', lambda **_:pytest.fail('nonvolume reached builder'))
    assert _rtstruct_masks(ct, source, best_effort=True, failure_outcomes=outcomes) == {}
    assert len(outcomes) == 11
    assert all(row['status'] == 'nonvolumetric_nonmeasurement' for row in outcomes)
    assert all(row['rtstruct_sop_instance_uid'] == str(rt.ds.SOPInstanceUID) for row in outcomes)
