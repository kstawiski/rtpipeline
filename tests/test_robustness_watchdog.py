import json
import logging
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rtpipeline import robustness_watchdog as wd
from rtpipeline.robustness_mcc import mcc_feature
from rtpipeline import radiomics_robustness as rr
from rtpipeline import radiomics_parallel as rp
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from test_robustness_nonmeasurements import course_fixture


def synthetic_worker(task):
    params=task[1]
    mode=params.get('mode','normal')
    if mode=='hang':
        time.sleep(30)
    if mode=='crash':
        os._exit(23)
    if mode=='error':
        raise RuntimeError('injected extraction error')
    if mode=='progress':
        for index in range(8):
            time.sleep(.05)
            wd.report_progress(f'completed_filter_{index}')
    return {'roi_name':params['roi_name'],'segmentation_source':'Manual',
            'perturbation_id':params['extra_metadata']['perturbation_id'],'value':42}


def tasks(modes):
    return [('mask',dict(roi_name='ROI',segmentation_source='Manual',mode=m,
                         extra_metadata={'perturbation_id':f'p{i}'})) for i,m in enumerate(modes)]


def collect(modes, *, budget=5, stall=.2, workers=1, method='fork'):
    with wd.SupervisedResults(mp.get_context(method),synthetic_worker,tasks(modes),workers,
                              course_timeout=budget,progress_timeout=stall) as runner:
        result=[]
        while len(result)<len(modes):
            try: result.append(runner.next(timeout=.1))
            except mp.TimeoutError: pass
        pids=[s['process'].pid for s in runner.slots]
    assert all(not p.is_alive() for p in mp.active_children() if p.pid in pids)
    return result


def test_slow_real_progress_is_not_a_hang():
    result=collect(['progress'],stall=.15)
    assert result[0]['value']==42
    assert result[0]['robustness_elapsed_s']>.3


@pytest.mark.parametrize('mode,reason',[('hang','watchdog_stall'),('crash','worker_died'),('error','worker_exception')])
def test_failed_worker_is_loud_and_other_tasks_finish(mode,reason,caplog):
    with caplog.at_level(logging.ERROR):
        results=collect(['normal',mode,'normal'])
    assert [r['__task_index__'] for r in results]==[0,1,2]
    assert results[0]['value']==results[2]['value']==42
    assert results[1]['__technical_failure__']['reason_code']==reason
    assert reason in caplog.text
    if mode=='crash':
        assert results[1]['__technical_failure__']['evidence']['exitcode']==23


def test_deadline_enumerates_running_and_queued_conditions():
    results=collect(['normal','progress','normal'],budget=.18,stall=1)
    assert results[0]['value']==42
    failures=[r for r in results if '__technical_failure__' in r]
    assert {r['__task_index__'] for r in failures}=={1,2}
    assert {r['__technical_failure__']['evidence']['task_state'] for r in failures}=={'running','not_started'}


def test_spawn_normal_course_is_unaffected():
    assert collect(['normal','normal'],method='spawn',stall=15,budget=30)[0]['value']==42


def test_observer_uses_completed_work_not_timer_heartbeats(monkeypatch):
    events=[]; monkeypatch.setattr(wd,'_sender',events.append)
    class Extractor:
        settings={'binWidth':25}
        def loadImage(self,*a): return ('image','mask')
        def computeShape(self,*a): return {'shape':2}
        def computeFeatures(self,*a): return {'Mean':3}
    original=Extractor(); extractor=wd.observed_extractor(original)
    assert extractor is original and extractor.settings=={'binWidth':25}
    assert extractor.computeFeatures('image','mask','wavelet-LLL')=={'Mean':3}
    assert events==['computeFeatures:wavelet-LLL']


@pytest.mark.parametrize('n',[1,2,8,30])
@pytest.mark.parametrize('symmetric',[False,True])
def test_mcc_preserves_original_equation_including_epsilon(n,symmetric):
    rng=np.random.default_rng(n)
    p=rng.uniform(size=(2,n,n,3)); p[p<.3]=0
    if symmetric: p=p+p.transpose(0,2,1,3)
    p[:,0,0,:]+=1
    p/=p.sum(axis=(1,2),keepdims=True)
    px=p.sum(axis=2,keepdims=True); py=p.sum(axis=1,keepdims=True)
    # Deliberately material epsilon catches invalid factorization shortcuts.
    eps=.01
    q=p[:,:,None,0,:]*p[:,None,:,0,:]/(px[:,:,None,0,:]*py[:,None,:,0,:]+eps)
    for k in range(1,n):
        q+=p[:,:,None,k,:]*p[:,None,:,k,:]/(px[:,:,None,0,:]*py[:,None,:,k,:]+eps)
    ev=np.linalg.eigvals(q.transpose(0,3,1,2)); ev.sort()
    expected=1 if n<2 else np.nanmean(np.sqrt(ev[:,:,-2]),axis=1).real
    observed=mcc_feature(SimpleNamespace(P_glcm=p,coefficients=dict(px=px,py=py,eps=eps)))
    np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-12)


def course_worker(task):
    params=task[1]; pid=params['extra_metadata']['perturbation_id']
    if '_t0_0_4_' in pid:
        time.sleep(30)
    identity={k:params[k] for k in rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
    rows=[dict(identity,roi_name='ROI',modality='CT',measurement_type=rr.ROBUSTNESS_MEASUREMENT_TYPE,
               perturbed_mask_identity=params['perturbed_mask_identity'],perturbation_id=pid,
               extraction_arm=arm,feature_name='original_firstorder_Mean',value=1.25)
          for arm in CT_EXTRACTION_ARMS]
    return {'__nonmeasurement_rows__':rows,'roi_name':'ROI','segmentation_source':'Manual','perturbation_id':pid}


def test_watchdog_course_publishes_completed_and_names_outstanding(tmp_path,monkeypatch,caplog):
    course,cfg,rob,_=course_fixture(tmp_path,monkeypatch,parallel=True)
    monkeypatch.setattr(rr,'get_context',lambda _:mp.get_context('fork'))
    monkeypatch.setattr(rp,'_isolated_radiomics_extraction_with_retry',course_worker)
    monkeypatch.setenv('RTPIPELINE_ROBUSTNESS_PROGRESS_TIMEOUT','1')
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError,match='published partial results'):
        rr.robustness_for_course(cfg,rob,course)
    path=course/'radiomics_robustness_ct.parquet'; assert path.exists()
    df=pd.read_parquet(path)
    assert set(df.robustness_status)=={'measured','geometrically_impossible','technical_failure'}
    assert set(df.loc[df.robustness_status=='measured','value'])=={1.25}
    bad=df[df.robustness_status=='technical_failure']
    assert set(bad.perturbation_id)=={'ntcv_t0_0_4_v0'}
    assert set(bad.reason_code)=={'watchdog_stall'}
    assert set(bad.extraction_arm)==set(CT_EXTRACTION_ARMS)
    assert bad.value.isna().all() and bad.feature_name.isna().all()
    assert 'watchdog_stall' in caplog.text
    assert not (course/'.radiomics_robustness_done').exists()
    with pytest.raises(RuntimeError,match='require recovery'):
        rr.summarize_feature_stability(df,rob)
    with pytest.raises(RuntimeError,match='require recovery'):
        rr.aggregate_robustness_results([path],tmp_path/'aggregate.xlsx',rob)




def all_hung_worker(task):
    time.sleep(30)


def test_zero_measured_course_still_publishes_exact_failure_inventory(tmp_path,monkeypatch):
    course,cfg,rob,_=course_fixture(tmp_path,monkeypatch,parallel=True)
    monkeypatch.setattr(rr,'get_context',lambda _:mp.get_context('fork'))
    monkeypatch.setattr(rp,'_isolated_radiomics_extraction_with_retry',all_hung_worker)
    monkeypatch.setenv('RTPIPELINE_ROBUSTNESS_PROGRESS_TIMEOUT','1')
    with pytest.raises(RuntimeError,match='published partial results'):
        rr.robustness_for_course(cfg,rob,course)
    df=pd.read_parquet(course/'radiomics_robustness_ct.parquet')
    assert len(df)==6
    assert sum(df.robustness_status=='technical_failure')==4
    assert df.value.isna().all()
    assert {pid for value in df.requested_condition_ids for pid in json.loads(value)}==set(df.perturbation_id)


def test_expired_course_never_starts_a_queued_task():
    with wd.SupervisedResults(mp.get_context('fork'),synthetic_worker,tasks(['normal','normal']),1,
                              course_timeout=.01,progress_timeout=1) as runner:
        time.sleep(.03)
        results=[runner.next(timeout=1),runner.next(timeout=1)]
        assert not runner.slots
    assert all(r['__technical_failure__']['evidence']['task_state']=='not_started' for r in results)




def test_buffered_completion_wins_over_expired_watchdog():
    with wd.SupervisedResults(mp.get_context('fork'),synthetic_worker,tasks(['normal']),1,
                              course_timeout=.05,progress_timeout=1) as runner:
        runner._backfill()
        time.sleep(.15)
        result=runner.next(timeout=1)
    assert result['value']==42


def test_technical_rows_cannot_disguise_numeric_measurements():
    rows=wd.technical_rows({'roi_name':'ROI','perturbation_id':'p','__technical_failure__':{
        'reason_code':'watchdog_stall','evidence':{'budget':300}}},{},'run','mask')
    frame=pd.DataFrame(rows)
    frame.loc[0,'value']=0
    with pytest.raises(RuntimeError,match='numeric'):
        wd.validate_technical_frame(frame,{'p'},'context')
