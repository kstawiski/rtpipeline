from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from rtpipeline import radiomics_robustness as rr
from rtpipeline import radiomics_parallel as rp
from rtpipeline import radiomics as rm
from rtpipeline import custom_models
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from rtpipeline.radiomics_robustness_outcomes import (
    GeometricNonmeasurement, extraction_nonmeasurement, nonmeasurement_rows,
)


def course_fixture(tmp_path, monkeypatch, *, parallel=False):
    course=tmp_path/'P'/'C'; course.mkdir(parents=True)
    array=np.zeros((17,17,17),np.uint8); array[0:4,5:12,5:12]=1
    mask=sitk.GetImageFromArray(array)
    image=sitk.GetImageFromArray(np.arange(array.size,dtype=np.float32).reshape(array.shape))
    identity=rr.RobustnessRoiIdentity.from_mapping(dict(
        patient_id='P',course_id='C',series_uid='1.2.3',segmentation_source='Manual',
        mask_identity='source-mask',roi_original_name='ROI',stable_roi_identifier='roi-1',
    ))
    monkeypatch.setattr(rr,'load_course_contract',lambda _:SimpleNamespace(planning_ct_dir=course,planning_ct={'series_instance_uid':'1.2.3'}))
    monkeypatch.setattr(rr,'_load_main_ct_identity_catalog',lambda *a,**k:({('Manual','ROI'):identity},{}))
    monkeypatch.setattr(rm,'_load_series_image',lambda _:image)
    monkeypatch.setattr(rm,'_standard_rtstruct_sources',lambda *a:[('Manual',course/'RS.dcm',None)])
    def _masks_stub(ct_dir, rs_path, *args, **kwargs):
        # Mirror the robustness sink contract: accept (and honor) the
        # failure_outcomes keyword so the stub stays faithful to the real helper.
        return {'ROI': array}
    monkeypatch.setattr(rm,'_rtstruct_masks',_masks_stub)
    monkeypatch.setattr(rm,'_mask_from_array_like',lambda *a:mask)
    monkeypatch.setattr(custom_models,'list_custom_model_outputs',lambda *a:[])
    monkeypatch.setenv('RTPIPELINE_DISABLE_PARALLEL_RADIOMICS','0' if parallel else '1')
    monkeypatch.setenv('RTPIPELINE_RADIOMICS_THREAD_LIMIT','1')
    monkeypatch.setenv('RTPIPELINE_MAX_WORKERS','1')
    cfg=PipelineConfig(tmp_path,tmp_path,tmp_path,max_workers_override=1)
    rob=rr.RobustnessConfig(enabled=True,perturbation=rr.PerturbationConfig(
        small_volume_changes=[0.],max_translation_mm=4.,n_random_contour_realizations=0,
        noise_levels=[0.],apply_to_structures=['ROI'],
    ))
    def features(image,masks,*args,**kwargs):
        rows=[]
        for pid,m in masks.items():
            for arm in CT_EXTRACTION_ARMS:
                rows.append({**identity.as_dict(),'structure':'ROI','modality':'CT',
                    'measurement_type':rr.ROBUSTNESS_MEASUREMENT_TYPE,
                    'perturbed_mask_identity':rr._perturbed_mask_identity(m),
                    'perturbation_id':pid,'extraction_arm':arm,
                    'feature_name':'original_firstorder_Mean','value':1.25})
        return pd.DataFrame(rows)
    monkeypatch.setattr(rr,'extract_features_for_masks',features)
    return course,cfg,rob,features


def test_one_impossible_condition_completes_course_with_enumeration(tmp_path,monkeypatch):
    course,cfg,rob,_=course_fixture(tmp_path,monkeypatch)
    result=rr.robustness_for_course(cfg,rob,course)
    df=pd.read_parquet(result)
    non=df[df.robustness_status=='geometrically_impossible']
    assert len(non)==2
    assert set(non.perturbation_id)=={'ntcv_t0_0_-4_v0'}
    assert set(non.reason_code)=={'translation_outside_image'}
    assert non.value.isna().all() and non.feature_name.isna().all()
    assert set(df.possible_condition_count)=={2}
    assert all(json.loads(x)==['ntcv_t0_0_-4_v0'] for x in df.impossible_condition_ids)
    assert all(len(json.loads(x))==3 for x in df.requested_condition_ids)
    assert np.isfinite(df.loc[df.robustness_status=='measured','value']).all()


def test_unexplained_missing_row_still_fails_course(tmp_path,monkeypatch):
    course,cfg,rob,features=course_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(rr,'extract_features_for_masks',lambda *a,**k:features(*a,**k).iloc[:2])
    with pytest.raises(RuntimeError,match='incomplete radiomics extraction'):
        rr.robustness_for_course(cfg,rob,course)
    assert not (course/'radiomics_robustness_ct.parquet').exists()


class Results:
    def __init__(self,mode): self.mode=mode
    def next(self,timeout):
        if self.mode=='timeout': raise rr.MPTimeoutError('test timeout')
        if self.mode=='crash': raise RuntimeError('worker crashed')
        return None
class Pool:
    def __init__(self,mode): self.mode=mode
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def imap_unordered(self,*a): return Results(self.mode)
    def terminate(self): pass


@pytest.mark.parametrize('mode',['timeout','crash','missing'])
def test_worker_failure_never_becomes_nonmeasurement(tmp_path,monkeypatch,mode):
    course,cfg,rob,_=course_fixture(tmp_path,monkeypatch,parallel=True)
    monkeypatch.setattr(rr,'get_context',lambda _:SimpleNamespace(Pool=lambda _:Pool(mode)))
    monkeypatch.setenv('RTPIPELINE_ROBUSTNESS_PROGRESS_TIMEOUT','-1')
    monkeypatch.setattr(rp,'_prepare_radiomics_task',lambda *a,**k:(tmp_path/'mask',{}))
    with pytest.raises(RuntimeError,match='incomplete robustness extraction'):
        rr.robustness_for_course(cfg,rob,course)
    assert not (course/'radiomics_robustness_ct.parquet').exists()


def test_volume_impossibilities_are_enumerated_not_dropped():
    a=np.zeros((8,8,8),np.uint8); a[3,3,3:5]=1
    masks=rr.generate_perturbed_masks(sitk.GetImageFromArray(a),[-.15,0.,.15],'ROI')
    assert len(masks)==3
    assert sum(isinstance(m,GeometricNonmeasurement) for m in masks.values())==2


def test_unexplained_volume_failure_is_not_geometric(monkeypatch):
    a=np.zeros((8,8,8),np.uint8); a[1:7,1:7,1:7]=1
    monkeypatch.setattr(rr,'volume_adapt_mask',lambda *a:None)
    with pytest.raises(RuntimeError,match='unexplained volume'):
        rr.generate_perturbed_masks(sitk.GetImageFromArray(a),[-.15,0.,.15],'ROI')


def test_resource_failure_cannot_be_typed_as_geometry():
    with pytest.raises(ValueError,match='unrecognized'):
        GeometricNonmeasurement('resource_guard',{'voxels':100})


def test_nonmeasurement_is_not_silently_aggregated(tmp_path,monkeypatch):
    course,cfg,rob,_=course_fixture(tmp_path,monkeypatch)
    df=pd.read_parquet(rr.robustness_for_course(cfg,rob,course))
    with pytest.raises(ValueError,match='explicit comparable-condition'):
        rr.summarize_feature_stability(df,rob)


NATIVE_RADIOMICS_INTERPRETER=Path('/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python')
NATIVE_RADIOMICS_HELPER=Path(__file__).with_name('robustness_native_pyradiomics_helper.py')


def _native_radiomics_payload(case,tmp_path):
    """Run one real-PyRadiomics case in the radiomics interpreter and return its JSON.

    The radiomics environment carries NumPy 1.x, SimpleITK and PyRadiomics but
    no ``pytest``, so the case cannot be re-collected there. The helper module
    performs the identical work with the identical production imports and hands
    back what it measured; every assertion stays here, on the host.
    """
    import os, subprocess
    if not NATIVE_RADIOMICS_INTERPRETER.exists(): pytest.skip('PyRadiomics interpreter unavailable')
    scratch=tmp_path/'native'; scratch.mkdir(parents=True,exist_ok=True)
    # -E -s keep the helper on its own environment: no PYTHON* inheritance and
    # no user site-packages, so the dual-environment separation is real.
    command=[str(NATIVE_RADIOMICS_INTERPRETER),'-B','-E','-s',str(NATIVE_RADIOMICS_HELPER),case,str(scratch)]
    env=dict(os.environ,LD_LIBRARY_PATH=str(NATIVE_RADIOMICS_INTERPRETER.parent.parent/'lib'),
             TMPDIR=str(scratch),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='1')
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_exact_resampled_minimum_is_typed_with_real_pyradiomics(tmp_path):
    try:
        from radiomics import featureextractor
    except ImportError:
        # The main campaign interpreter deliberately has no native PyRadiomics.
        # Measure in the separate radiomics interpreter, not a fake module, and
        # assert here on the reason and evidence the real extractor produced.
        payload=_native_radiomics_payload('resampled_minimum',tmp_path)
        assert payload['reason_code']=='resampled_mask_below_minimum_voxels'
        assert payload['evidence']['resampled_voxels']==8
        assert payload['evidence']['minimumROISize']==64
        return
    a=np.zeros((8,8,8),np.uint8); a[2:4,2:4,2:4]=1
    mask=sitk.GetImageFromArray(a); image=sitk.GetImageFromArray(a.astype(np.float32))
    factory=lambda:featureextractor.RadiomicsFeatureExtractor(minimumROISize=64,minimumROIDimensions=2)
    outcome=extraction_nonmeasurement(image,mask,factory)
    assert outcome.reason_code=='resampled_mask_below_minimum_voxels'
    assert outcome.evidence['resampled_voxels']==8
    assert outcome.evidence['minimumROISize']==64


def test_forged_geometric_reason_is_rejected():
    with pytest.raises(ValueError,match='does not establish'):
        GeometricNonmeasurement('resampled_mask_below_minimum_voxels',
                                {'resampled_voxels':100,'minimumROISize':64})


@pytest.mark.parametrize('error',[TimeoutError('timeout'),MemoryError('memory')])
def test_retry_never_relabels_timeout_or_memory(monkeypatch,error):
    def fail(task): raise error
    monkeypatch.setattr(rp,'_isolated_radiomics_extraction',fail)
    with pytest.raises(type(error)):
        rp._isolated_radiomics_extraction_with_retry(('mask',{}))


def test_real_finite_features_after_identical_condition_retry(tmp_path,monkeypatch):
    try:
        from radiomics import featureextractor
    except ImportError:
        # Same split as above: the real extraction runs in the radiomics
        # interpreter, and the rows it measured are validated and asserted here.
        payload=_native_radiomics_payload('identical_condition_retry',tmp_path)
        assert payload['same_task_object'] is True
        assert payload['robustness_attempts']==2
        frame=pd.DataFrame(payload['rows'])
        rr._validate_extracted_feature_frame(frame,{'ntcv_v0'},'urinary_bladder',
            expected_source_identity=rr.RobustnessRoiIdentity.from_mapping(payload['source_identity']))
        assert set(frame.extraction_arm)==set(CT_EXTRACTION_ARMS)
        assert np.isfinite(frame.value).all()
        assert frame.robustness_retry_errors.str.contains('injected transient read failure').all()
        return
    course=tmp_path/'P'/'C'; course.mkdir(parents=True)
    temp=tmp_path/'inputs'; temp.mkdir()
    params=tmp_path/'params.yaml'
    params.write_text('imageType:\n  Original: {}\nfeatureClass:\n  firstorder: []\n  shape: []\nsetting:\n  minimumROISize: 10\n  minimumROIDimensions: 2\n')
    array=np.zeros((12,12,12),np.uint8); array[2:10,2:10,2:10]=1
    mask=sitk.GetImageFromArray(array)
    image=sitk.GetImageFromArray(np.arange(array.size,dtype=np.float32).reshape(array.shape)%100)
    identity=dict(patient_id='P',course_id='C',series_uid='1.2.3',segmentation_source='AutoRTS_total',
                  roi_original_name='urinary_bladder',mask_identity='source-mask',stable_roi_identifier='roi-1')
    cfg=PipelineConfig(tmp_path,tmp_path,tmp_path)
    cfg.radiomics_params_file=params
    mask_path,task=rp._prepare_radiomics_task(image,mask,cfg,'AutoRTS_total','urinary_bladder',course,temp,False,source_identity=identity)
    task['extra_metadata']={'perturbation_id':'ntcv_v0'}
    original=rp._isolated_radiomics_extraction
    seen=[]
    def transient_once(same_task):
        seen.append(same_task)
        if len(seen)==1: raise OSError('injected transient read failure')
        return original(same_task)
    monkeypatch.setattr(rp,'_isolated_radiomics_extraction',transient_once)
    result=rp._isolated_radiomics_extraction_with_retry((mask_path,task))
    assert seen[0] is seen[1]
    assert result['robustness_attempts']==2
    frame=pd.DataFrame(rr._feature_rows_from_worker_result(result))
    rr._validate_extracted_feature_frame(frame,{'ntcv_v0'},'urinary_bladder',expected_source_identity=rr.RobustnessRoiIdentity.from_mapping(identity))
    assert set(frame.extraction_arm)==set(CT_EXTRACTION_ARMS)
    assert np.isfinite(frame.value).all()
    assert frame.robustness_retry_errors.str.contains('injected transient read failure').all()


# ============================================================================
# Regression: the robustness pass must pass a failure_outcomes sink to
# _rtstruct_masks (standard, RS_custom and custom-model call sites) so that a
# non-volumetric source ROI is recorded as an identity-bound geometric
# disposition, while technical failures stay fail-closed.
# These tests use a REAL synthetic CT series + REAL RTSTRUCT and the REAL
# _rtstruct_masks/_load_series_image/_mask_from_array_like so that geometry is
# faithful; only the course-contract and feature-extraction seams are stubbed.
# ============================================================================
import hashlib as _hashlib
import shutil as _shutil
from pydicom.dataset import Dataset as _PDDataset
from pydicom.sequence import Sequence as _PDSequence
import pydicom as _pydicom
from test_science_batch_c import _build_real_rtstruct as _build_real_rtstruct
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError

_ROBUSTNESS_DISPOSITIONS = 'radiomics_robustness_source_dispositions.json'


def _real_mixed_course(tmp_path, monkeypatch, *, add_bad_roi=False,
                       apply_to_structures=('ROI',), technical_failure_conditions=0):
    """Self-contained synthetic course: REAL CT series + REAL RTSTRUCT holding a
    volumetric ROI named 'ROI' plus a non-volumetric POINT marker 'Marker1'
    (and, optionally, an unparseable contour 'Bad' for the fail-closed case).

    ``apply_to_structures`` selects which source ROIs the robustness grid
    requests, so a course whose only requested structure is non-volumetric can
    be exercised. ``technical_failure_conditions`` makes the extraction stub
    report that many conditions (the last in sorted order) as technical
    failures with null values.
    Returns (course, cfg, rob, rs_path)."""
    course = tmp_path / 'P' / 'C'
    course.mkdir(parents=True)
    ct_dir = course / 'CT'
    ct_dir.mkdir()
    rt = _build_real_rtstruct(ct_dir, side=12, n_slices=4)
    ptv = [r for r in rt.ds.StructureSetROISequence if str(r.ROIName) == 'PTV'][0]
    ptv.ROIName = 'ROI'  # align with catalog key and apply_to_structures
    marker = _PDDataset()
    marker.ROINumber = 2
    marker.ROIName = 'Marker1'
    marker.ReferencedFrameOfReferenceUID = rt.series_data[0].FrameOfReferenceUID
    mitem, mcont = _PDDataset(), _PDDataset()
    mitem.ReferencedROINumber = 2
    mcont.ContourGeometricType = 'POINT'
    mcont.ContourData = [2., 2., 1.]
    mcont.NumberOfContourPoints = 1
    mitem.ContourSequence = _PDSequence([mcont])
    rt.ds.StructureSetROISequence.append(marker)
    rt.ds.ROIContourSequence.append(mitem)
    if add_bad_roi:
        bad = _PDDataset()
        bad.ROINumber = 3
        bad.ROIName = 'Bad'
        bad.ReferencedFrameOfReferenceUID = rt.series_data[0].FrameOfReferenceUID
        bitem, bcont = _PDDataset(), _PDDataset()
        bitem.ReferencedROINumber = 3
        bcont.ContourGeometricType = 'POINT'
        bcont.ContourData = [2., 2., 1.]
        bcont.NumberOfContourPoints = 9  # declared != actual -> unparseable contour
        bitem.ContourSequence = _PDSequence([bcont])
        rt.ds.StructureSetROISequence.append(bad)
        rt.ds.ROIContourSequence.append(bitem)
    rs_path = course / 'RS.dcm'
    rt.ds.save_as(rs_path)
    series_uid = str(rt.series_data[0].SeriesInstanceUID)
    identity = rr.RobustnessRoiIdentity.from_mapping(dict(
        patient_id='P', course_id='C', series_uid=series_uid, segmentation_source='Manual',
        mask_identity='source-mask', roi_original_name='ROI', stable_roi_identifier='roi-1',
    ))
    monkeypatch.setattr(rr, 'load_course_contract',
                        lambda _: SimpleNamespace(planning_ct_dir=ct_dir,
                                                  planning_ct={'series_instance_uid': series_uid}))
    monkeypatch.setattr(rr, '_load_main_ct_identity_catalog',
                        lambda *a, **k: ({('Manual', 'ROI'): identity}, {}))
    monkeypatch.setattr(rm, '_standard_rtstruct_sources', lambda *a: [('Manual', rs_path, None)])
    monkeypatch.setattr(custom_models, 'list_custom_model_outputs', lambda *a: [])
    monkeypatch.setenv('RTPIPELINE_DISABLE_PARALLEL_RADIOMICS', '1')
    monkeypatch.setenv('RTPIPELINE_RADIOMICS_THREAD_LIMIT', '1')
    monkeypatch.setenv('RTPIPELINE_MAX_WORKERS', '1')
    cfg = PipelineConfig(tmp_path, tmp_path, tmp_path, max_workers_override=1)
    rob = rr.RobustnessConfig(enabled=True, perturbation=rr.PerturbationConfig(
        small_volume_changes=[0., .15], max_translation_mm=0.,
        n_random_contour_realizations=0, noise_levels=[0.],
        apply_to_structures=list(apply_to_structures),
    ))
    def features(image, masks, *args, **kwargs):
        rows = []
        failing = set(sorted(masks)[len(masks) - technical_failure_conditions:]
                      if technical_failure_conditions else ())
        for pid, m in masks.items():
            technical = pid in failing
            for arm in CT_EXTRACTION_ARMS:
                row = {**identity.as_dict(), 'structure': 'ROI', 'modality': 'CT',
                       'measurement_type': rr.ROBUSTNESS_MEASUREMENT_TYPE,
                       'perturbed_mask_identity': rr._perturbed_mask_identity(m),
                       'perturbation_id': pid, 'extraction_arm': arm,
                       'feature_name': 'original_firstorder_Mean', 'value': 1.25}
                if technical:
                    # A technical condition keeps null voxel/feature values and
                    # its own failed-stage evidence; it is never a measurement.
                    row.update(robustness_status='technical_failure',
                               feature_name=None, value=None,
                               reason_code='worker_exception',
                               technical_evidence=json.dumps(
                                   {'detail': 'synthetic worker exception'}))
                rows.append(row)
        return pd.DataFrame(rows)
    monkeypatch.setattr(rr, 'extract_features_for_masks', features)
    return course, cfg, rob, rs_path


def test_mixed_source_disposition_published_and_run_bound(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(tmp_path, monkeypatch)
    result = rr.robustness_for_course(cfg, rob, course)
    assert result is not None
    df = pd.read_parquet(result)
    # The volumetric ROI is measured; the point ROI is never measured.
    assert set(df.structure) == {'ROI'}
    assert df.robustness_status.eq('measured').any()
    art = course / 'metadata' / _ROBUSTNESS_DISPOSITIONS
    assert art.exists()
    payload = json.loads(art.read_text())
    assert payload['artifact_kind'] == 'robustness_source_dispositions'
    assert payload['patient_id'] == 'P' and payload['course_id'] == 'C'
    row = next(r for r in payload['rows'] if r['roi_name'] == 'Marker1')
    assert row['status'] == 'nonvolumetric_nonmeasurement'
    assert row['segmentation_source'] == 'Manual'
    assert row['roi_number'] == '2'
    assert row['structural_code'] == 'ROI_NONVOLUMETRIC_POINT'
    assert row['source_path'] == str(rs_path)
    assert row['rtstruct_sop_instance_uid'] == str(_pydicom.dcmread(rs_path).SOPInstanceUID)
    # Round-trips with the recorded run identifier...
    loaded = rr.load_robustness_source_dispositions(
        course, run_identifier=payload['robustness_run_identifier'], rob_config=rob)
    assert any(r['roi_name'] == 'Marker1' for r in loaded)
    # ...and rejects a mismatched (stale) run identifier.
    with pytest.raises(ValueError, match='stale robustness source dispositions'):
        rr.load_robustness_source_dispositions(course, run_identifier='wrong-run-id', rob_config=rob)


def test_stale_disposition_sidecar_replaced_on_rerun(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    art = course / 'metadata' / _ROBUSTNESS_DISPOSITIONS
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({
        'schema_version': 1, 'artifact_kind': 'robustness_source_dispositions',
        'patient_id': 'P', 'course_id': 'C', 'robustness_run_identifier': 'STALE',
        'row_count': 1, 'rows': [{'roi_name': 'old'}],
    }))
    rr.robustness_for_course(cfg, rob, course)
    payload = json.loads(art.read_text())
    assert payload['robustness_run_identifier'] != 'STALE'
    assert not any(r.get('roi_name') == 'old' for r in payload['rows'])
    # The replaced run identifier can never be revived from the new bytes.
    with pytest.raises(ValueError, match='stale robustness source dispositions'):
        rr.load_robustness_source_dispositions(course, run_identifier='STALE', rob_config=rob)


def test_technical_extraction_failure_stays_fail_closed(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch, add_bad_roi=True)
    with pytest.raises(RadiomicsCourseExtractionError):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / 'radiomics_robustness_ct.parquet').exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


def test_custom_model_technical_failure_propagates_not_swallowed(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    model_dir = course / 'Segmentation_CustomModels' / 'ModelA'
    model_dir.mkdir(parents=True)
    # A real RTSTRUCT: the model source binds successfully, so the failure the
    # test asserts on is the mask reader's, not an unreadable-source failure.
    _shutil.copyfile(course / 'RS.dcm', model_dir / 'rtstruct.dcm')
    real = rm._rtstruct_masks

    def raising(ct, rs, *a, **k):
        if 'ModelA' in str(rs):
            raise RuntimeError('synthetic technical model failure')
        return real(ct, rs, *a, **k)
    monkeypatch.setattr(rm, '_rtstruct_masks', raising)
    monkeypatch.setattr(custom_models, 'list_custom_model_outputs',
                        lambda *a: [('ModelA', model_dir)])
    with pytest.raises(RuntimeError, match='synthetic technical model failure'):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / 'radiomics_robustness_ct.parquet').exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


# ============================================================================
# The published sidecar is a contract, not a note: it is bound to the run, the
# source bytes, the deciding code, the effective configuration and the measured
# output, and every one of those bindings is rechecked on load.
# ============================================================================


def _payload(course):
    return json.loads((course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).read_text())


def _rewrite(course, payload):
    """Rewrite the sidecar with a repaired row digest and row count.

    Repairing them first is what keeps the structural checks non-vacuous: the
    artifact is then internally consistent, so only an explicit identity,
    status, geometry, uniqueness or binding check can still reject it.
    """
    payload['row_count'] = len(payload['rows'])
    payload['rows_sha256'] = rr._content_sha256(payload['rows'])
    (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).write_text(json.dumps(payload))


def _digest(path):
    return _hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_completed_dispositions_reload_unchanged(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(tmp_path, monkeypatch)
    output = rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']

    rows = rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob)
    assert rows == payload['rows']
    # Unchanged bytes stay admissible on every read, not just the first.
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob) == rows

    assert payload['measurement_outcome'] == 'measured'
    assert payload['measured_output']['path'] == output.name
    assert payload['measured_output']['sha256'] == _digest(output)
    assert payload['source_only_basis'] is None
    binding, = payload['source_bindings']
    assert binding['source_path'] == str(rs_path)
    assert binding['sha256'] == _digest(rs_path)
    assert binding['rtstruct_sop_instance_uid'] == str(
        _pydicom.dcmread(rs_path).SOPInstanceUID)
    # The sidecar records identity and disposition only: a non-measurement
    # never acquires voxel or feature values by being written down.
    for row in payload['rows']:
        assert not ({'value', 'feature_name', 'voxels', 'volume_mm3',
                     'perturbation_id', 'extraction_arm'} & set(row))
    # The measured table keeps the full requested condition grid.
    frame = pd.read_parquet(output)
    assert set(frame.robustness_status) == {'measured'}
    assert set(frame.structure) == {'ROI'}


@pytest.mark.parametrize('corruption,match', [
    ('row_type', 'not a record'),
    ('missing_identity', 'missing required identity'),
    ('blank_identity', 'missing required identity'),
    ('status', 'not a terminal source disposition'),
    ('failed_status', 'not a terminal source disposition'),
    ('structural_code', 'not a non-volumetric geometry'),
    ('duplicate', 'duplicate robustness source disposition'),
    ('unbound_source', 'cites unbound source'),
])
def test_structural_row_checks_survive_a_repaired_digest(
        tmp_path, monkeypatch, corruption, match):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    # Positive control: the same loader admits the unchanged artifact.
    assert rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)

    if corruption == 'row_type':
        payload['rows'][0] = ['not', 'a', 'record']
    elif corruption == 'missing_identity':
        payload['rows'][0].pop('roi_number')
    elif corruption == 'blank_identity':
        payload['rows'][0]['rtstruct_sop_instance_uid'] = '   '
    elif corruption == 'status':
        payload['rows'][0]['status'] = 'measured'
    elif corruption == 'failed_status':
        # A technical read failure must fail the course, never be published
        # as a completed, terminal source disposition.
        payload['rows'][0]['status'] = 'failed'
    elif corruption == 'structural_code':
        payload['rows'][0]['structural_code'] = 'ROI_EXTRACTION_FAILED'
    elif corruption == 'duplicate':
        payload['rows'].append(dict(payload['rows'][0]))
    else:
        payload['rows'][0]['source_path'] = str(course / 'unbound.dcm')
    _rewrite(course, payload)

    with pytest.raises(ValueError, match=match):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


def test_row_digest_rejects_an_edited_non_identity_field(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    # Structurally valid, but not the bytes that were published.
    payload['rows'][0]['reason'] = 'ROI_NONVOLUMETRIC_OPEN_PLANAR'
    (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='do not match their recorded digest'):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


def test_code_binding_is_content_not_a_revision(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    package_root = Path(rr.__file__).resolve().parent
    recorded = payload['code_identity']['sources']
    assert {entry['path'] for entry in recorded} >= {
        'radiomics_robustness.py', 'radiomics.py'}
    # Every binding is the actual file content, so a dirty worktree at an
    # unchanged revision cannot pass as the code that produced the artifact.
    for entry in recorded:
        assert entry['sha256'] == _digest(package_root / entry['path'])

    payload['code_identity']['sources'][0]['sha256'] = '0' * 64
    payload['code_identity']['sources_sha256'] = rr._content_sha256(
        payload['code_identity']['sources'])
    (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='produced by different code'):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


def test_code_identity_must_agree_with_its_own_per_file_digests(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    payload['code_identity']['sources'][0]['sha256'] = '1' * 64
    (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='own per-file digests'):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


@pytest.mark.parametrize('drift', ['grid', 'threshold'])
def test_loader_rejects_effective_configuration_drift(tmp_path, monkeypatch, drift):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    # Positive control: the configuration that ran is accepted.
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob)

    drifted = rr.RobustnessConfig(enabled=True, perturbation=rr.PerturbationConfig(
        small_volume_changes=[0., .15], max_translation_mm=0.,
        n_random_contour_realizations=0,
        noise_levels=[0., 10.] if drift == 'grid' else [0.],
        apply_to_structures=['ROI'],
    ))
    if drift == 'threshold':
        drifted.thresholds = rr.RobustnessThresholds(icc_robust=0.5)
    with pytest.raises(ValueError, match='different effective configuration'):
        rr.load_robustness_source_dispositions(
            course, run_identifier=run_id, rob_config=drifted)


def test_loader_rejects_a_source_that_is_gone(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    run_id = _payload(course)['robustness_run_identifier']
    assert rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)
    rs_path.unlink()
    with pytest.raises(ValueError, match='no longer readable'):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


def test_all_nonvolumetric_request_is_source_only_without_a_parquet(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(
        tmp_path, monkeypatch, apply_to_structures=('Marker1',))
    assert rr.robustness_for_course(cfg, rob, course) is None
    output = course / 'radiomics_robustness_ct.parquet'
    # A course that measured nothing gets no fabricated measurement table.
    assert not output.exists()

    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    assert payload['measurement_outcome'] == 'source_only_nonvolumetric'
    assert payload['measured_output'] is None
    assert payload['source_only_basis']['selected_structure_count'] == 0
    assert payload['source_only_basis']['unresolved_identity_count'] == 0
    # The outcome rests on a source ROI the selection actually matched.
    assert payload['source_only_basis']['selection_matched_nonvolumetric_count'] == 1
    assert payload['source_only_basis']['requested_selection'] == ['Marker1']
    row = next(r for r in payload['rows'] if r['roi_name'] == 'Marker1')
    assert row['status'] == 'nonvolumetric_nonmeasurement'
    assert row['structural_code'] == 'ROI_NONVOLUMETRIC_POINT'
    assert row['segmentation_source'] == 'Manual'
    assert row['source_path'] == str(rs_path)
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob) == payload['rows']

    # A table appearing later cannot be certified by a source-only record.
    output.write_bytes(b'fabricated')
    with pytest.raises(ValueError, match='source-only'):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=rob)


def test_failed_final_publication_leaves_no_output_and_no_sidecar(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    output = rr.robustness_for_course(cfg, rob, course)
    stale_run = _payload(course)['robustness_run_identifier']
    assert output.exists()

    real_replace = rr.os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith('radiomics_robustness_ct.parquet'):
            raise OSError('synthetic publication failure')
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(rr.os, 'replace', failing_replace)
    with pytest.raises(RuntimeError, match='failed to save robustness results'):
        rr.robustness_for_course(cfg, rob, course)
    monkeypatch.setattr(rr.os, 'replace', real_replace)

    # The previous run's artifacts were invalidated before the rerun, and the
    # failed rerun published neither a table nor a completed sidecar.
    assert not output.exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()
    with pytest.raises(FileNotFoundError):
        rr.load_robustness_source_dispositions(course, run_identifier=stale_run, rob_config=rob)


def test_partial_technical_failure_keeps_evidence_but_publishes_no_sidecar(
        tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(
        tmp_path, monkeypatch, technical_failure_conditions=1)
    with pytest.raises(RuntimeError, match='published partial results'):
        rr.robustness_for_course(cfg, rob, course)

    output = course / 'radiomics_robustness_ct.parquet'
    # The partial technical-condition evidence stays in its failed-stage form.
    frame = pd.read_parquet(output)
    failed = frame[frame.robustness_status == 'technical_failure']
    assert len(failed) == len(CT_EXTRACTION_ARMS)
    assert failed.value.isna().all() and failed.feature_name.isna().all()
    assert set(failed.reason_code) == {'worker_exception'}
    assert (frame.robustness_status == 'measured').any()
    # ...but a failed stage never gets a completed source-disposition sidecar.
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


def test_unreadable_rtstruct_source_fails_closed(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(tmp_path, monkeypatch)
    rs_path.write_bytes(b'not a DICOM instance')
    with pytest.raises(_pydicom.errors.InvalidDicomError):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / 'radiomics_robustness_ct.parquet').exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


# ============================================================================
# Provenance is verified *before* a measured table is published, and a table
# that the final provenance/sidecar step cannot certify is withdrawn again.
# The deliberately partial technical_failure table is a different case: it is
# failed-stage evidence and is asserted to survive above.
# ============================================================================


def test_source_drift_during_extraction_publishes_no_measured_table(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(tmp_path, monkeypatch)
    stub = rr.extract_features_for_masks

    def drifting(*a, **k):
        frame = stub(*a, **k)
        # The bound RTSTRUCT changes while the run is extracting, keeping its
        # SOPInstanceUID. Nothing this run measured is certifiable any more.
        ds = _pydicom.dcmread(rs_path)
        ds.StructureSetLabel = 'DRIFTED'
        ds.save_as(rs_path)
        return frame

    monkeypatch.setattr(rr, 'extract_features_for_masks', drifting)
    with pytest.raises(RuntimeError, match='content changed'):
        rr.robustness_for_course(cfg, rob, course)
    # A measured-looking table must never outlive the provenance that would
    # have certified it.
    assert not (course / 'radiomics_robustness_ct.parquet').exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


def test_sidecar_write_failure_withdraws_the_measured_table(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    real_replace = rr.os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith(_ROBUSTNESS_DISPOSITIONS):
            raise OSError('synthetic sidecar publication failure')
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(rr.os, 'replace', failing_replace)
    with pytest.raises(OSError, match='synthetic sidecar publication failure'):
        rr.robustness_for_course(cfg, rob, course)
    # A nominal completed table with no sidecar would read as a certified
    # measurement that nothing certifies.
    assert not (course / 'radiomics_robustness_ct.parquet').exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


# ============================================================================
# The deciding-code binding is captured before processing and re-verified
# before publication, and the loader compares against the code on disk now.
# Real package files are never edited or swapped; only the file-digest seam is
# redirected, and every other path keeps its true content digest.
# ============================================================================

_DRIFT_TARGET = 'radiomics_robustness_outcomes.py'


def _code_drift_seam(monkeypatch, active, target=_DRIFT_TARGET):
    real_digest = rr._file_sha256

    def digest(path):
        if active() and Path(path).name == target:
            return 'f' * 64
        return real_digest(path)

    monkeypatch.setattr(rr, '_file_sha256', digest)


def test_code_drift_during_the_run_blocks_publication(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    drifted: list = []
    _code_drift_seam(monkeypatch, lambda: bool(drifted))
    stub = rr.extract_features_for_masks

    def drifting(*a, **k):
        frame = stub(*a, **k)
        drifted.append(True)  # a deciding module changed on disk mid-run
        return frame

    monkeypatch.setattr(rr, 'extract_features_for_masks', drifting)
    with pytest.raises(RuntimeError, match='deciding code changed'):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / 'radiomics_robustness_ct.parquet').exists()
    assert not (course / 'metadata' / _ROBUSTNESS_DISPOSITIONS).exists()


def test_loader_rejects_an_artifact_after_the_deciding_code_changes(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    run_id = _payload(course)['robustness_run_identifier']
    # Positive control: unchanged code admits the unchanged artifact.
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob)
    # The same interpreter must not keep admitting it once the deciding code on
    # disk differs from the code the artifact cites.
    _code_drift_seam(monkeypatch, lambda: True)
    with pytest.raises(ValueError, match='produced by different code'):
        rr.load_robustness_source_dispositions(
            course, run_identifier=run_id, rob_config=rob)


def test_captured_code_identity_is_the_current_on_disk_content(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    recorded = _payload(course)['code_identity']
    package_root = Path(rr.__file__).resolve().parent
    for entry in recorded['sources']:
        assert entry['sha256'] == _digest(package_root / entry['path'])
    assert recorded['sources_sha256'] == rr._content_sha256(recorded['sources'])
    # A fresh read of the same unchanged files reproduces the same identity,
    # so the value is not a first-use snapshot cached for the interpreter.
    assert rr._current_robustness_code_identity()['sources_sha256'] == \
        recorded['sources_sha256']


# ============================================================================
# Admission requires the configuration the caller is acting under. Reading the
# artifact for integrity alone is a separate, explicitly named result.
# ============================================================================


def test_admission_requires_the_current_configuration(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    run_id = _payload(course)['robustness_run_identifier']
    with pytest.raises(TypeError):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id)
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob)


def test_integrity_inspection_is_not_validated_admission(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    rr.robustness_for_course(cfg, rob, course)
    payload = _payload(course)
    run_id = payload['robustness_run_identifier']

    report = rr.inspect_robustness_source_dispositions(course, run_identifier=run_id)
    # The integrity-only result cannot be mistaken for an admitted row list.
    assert not isinstance(report, list)
    assert report.configuration_verified is False
    assert report.rows == payload['rows']
    assert report.measurement_outcome == 'measured'

    admitted = rr.inspect_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob)
    assert admitted.configuration_verified is True
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob) == admitted.rows


def test_configuration_snapshot_binds_path_valued_content(tmp_path):
    from dataclasses import dataclass as _dc

    @_dc
    class _PathBoundConfig:
        params_file: object

    params = tmp_path / 'params.yaml'
    params.write_text('setting:\n  binWidth: 25\n')
    first = rr.effective_robustness_configuration(
        _PathBoundConfig(params), output_name='r.parquet')
    assert first['robustness']['params_file']['sha256'] == _digest(params)

    params.write_text('setting:\n  binWidth: 50\n')
    second = rr.effective_robustness_configuration(
        _PathBoundConfig(params), output_name='r.parquet')
    # Same path, different governing bytes: not the same configuration.
    assert rr._content_sha256(first) != rr._content_sha256(second)

    # A bare string with the same text is not the same setting as a bound file.
    as_text = rr.effective_robustness_configuration(
        _PathBoundConfig(str(params)), output_name='r.parquet')
    assert rr._content_sha256(as_text) != rr._content_sha256(second)

    # An absent file is recorded explicitly instead of passing as content.
    missing = rr.effective_robustness_configuration(
        _PathBoundConfig(tmp_path / 'absent.yaml'), output_name='r.parquet')
    assert missing['robustness']['params_file']['content_state'] == 'absent'
    assert missing['robustness']['params_file']['sha256'] is None


# ============================================================================
# A selection that matched nothing is not evidence of non-volumetric anatomy.
# ============================================================================


@pytest.mark.parametrize('selection,expected', [
    (('Nonexistent',), ['Nonexistent']),
    ((), []),
])
def test_unmatched_selection_is_not_a_nonvolumetric_outcome(
        tmp_path, monkeypatch, selection, expected):
    course, cfg, rob, _ = _real_mixed_course(
        tmp_path, monkeypatch, apply_to_structures=selection)
    assert rr.robustness_for_course(cfg, rob, course) is None
    output = course / 'radiomics_robustness_ct.parquet'
    assert not output.exists()

    payload = _payload(course)
    run_id = payload['robustness_run_identifier']
    # An unrelated POINT elsewhere in the source says nothing about a name that
    # was never found, and an unfound name is not a clinical exclusion.
    assert payload['measurement_outcome'] == rr.ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME
    assert payload['measurement_outcome'] != rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME
    assert payload['measured_output'] is None
    basis = payload['source_only_basis']
    assert basis['selected_structure_count'] == 0
    assert basis['selection_matched_nonvolumetric_count'] == 0
    assert basis['requested_selection'] == expected
    # The accounting is still explicit, and still fabricates no measurement.
    assert any(r['roi_name'] == 'Marker1' for r in payload['rows'])
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=run_id, rob_config=rob) == payload['rows']
