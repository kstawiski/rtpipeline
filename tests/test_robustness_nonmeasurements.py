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
    monkeypatch.setattr(rm,'_rtstruct_masks',lambda *a:{'ROI':array})
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


def test_exact_resampled_minimum_is_typed_with_real_pyradiomics(tmp_path):
    try:
        from radiomics import featureextractor
    except ImportError:
        # The main campaign interpreter deliberately has no native PyRadiomics.
        # Exercise this test in the separate radiomics interpreter, not a fake module.
        import subprocess, sys, os
        interpreter=Path('/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python')
        if not interpreter.exists(): pytest.skip('PyRadiomics interpreter unavailable')
        env=dict(os.environ,LD_LIBRARY_PATH=str(interpreter.parent.parent/'lib'))
        command=[str(interpreter),'-B','-m','pytest','-q','-p','no:cacheprovider',
                 f'--basetemp={tmp_path}/native',str(Path(__file__))+'::test_exact_resampled_minimum_is_typed_with_real_pyradiomics']
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
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
        import subprocess,os
        interpreter=Path('/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python')
        if not interpreter.exists(): pytest.skip('PyRadiomics interpreter unavailable')
        result=subprocess.run([str(interpreter),'-B','-m','pytest','-q','-p','no:cacheprovider',
            f'--basetemp={tmp_path}/native',str(Path(__file__))+'::test_real_finite_features_after_identical_condition_retry'],
            env=dict(os.environ,LD_LIBRARY_PATH=str(interpreter.parent.parent/'lib')),
            capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
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
