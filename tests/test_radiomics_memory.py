from __future__ import annotations
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from rtpipeline.radiomics_resource_guard import estimate_resampled_bounding_box, DEFAULT_MAX_RESAMPLED_BBOX_VOXELS
from rtpipeline.radiomics_memory import (preflight_metadata, predicted_peak_bytes, MEMORY_BUDGET_BYTES, STAGE1_CODE, STAGE2_CODE, install_texture_budget, RadiomicsMemoryLimit)


def estimate(mask, spacing=(1,1,1), padding=5):
    return estimate_resampled_bounding_box(mask,native_spacing_xyz=spacing,resampled_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0),pad_distance=padding)


def test_sparse_elongated_admitted_dense_same_bbox_rejected():
    m=np.zeros((300,300,300),np.uint8)
    m[:8,:8,:8]=1; m[-8:,-8:,-8:]=1
    e=estimate(m)
    assert e.estimated_resampled_bbox_voxels>DEFAULT_MAX_RESAMPLED_BBOX_VOXELS
    meta=preflight_metadata(e,m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0))
    assert meta['resource_guard_reason_code']==STAGE2_CODE
    assert meta['resource_guard_predicted_peak_bytes']<MEMORY_BUDGET_BYTES
    m[:]=1
    assert preflight_metadata(estimate(m),m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0)) is None


def test_historical_single_678_gib_allocation_class_rejected():
    # This is the recorded hazard class, not a fabricated replay of lost pixels.
    g=512
    z=math.ceil(6.78*1024**3/(8*g))
    assert predicted_peak_bytes(crop_voxels=60648000,native_image_voxels=512*512*155,
                                mask_voxels=z,gray_levels=g,max_gray_frequency=z)>MEMORY_BUDGET_BYTES


@pytest.mark.parametrize('field',['crop_voxels','native_image_voxels','mask_voxels','gray_levels','max_extent'])
def test_predictor_monotonic(field):
    kw=dict(crop_voxels=100000,native_image_voxels=200000,mask_voxels=1000,gray_levels=64,max_extent=100)
    old=predicted_peak_bytes(**kw);kw[field]*=2
    assert predicted_peak_bytes(**kw)>=old


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_nonfinite_or_negative_dimensions_rejected(bad):
    with pytest.raises(ValueError):
        predicted_peak_bytes(crop_voxels=bad,native_image_voxels=1,mask_voxels=1)


def test_anisotropic_lattice_count_not_volume_ratio():
    m=np.zeros((4,4,4),np.uint8);m[0,0,0]=1;m[-1,-1,-1]=1
    e=replace(estimate(m,spacing=(1.1,1.1,1.1)),estimated_resampled_bbox_shape=(260,260,260),estimated_resampled_bbox_voxels=260**3)
    meta=preflight_metadata(e,m,native_spacing_xyz=(1.1,1.1,1.1),array_axis_to_xyz=(2,1,0),settings={'resampledPixelSpacing':[1,1,1]})
    assert meta['resource_guard_mask_voxels_bound']==16


@pytest.mark.parametrize('empty',[True,False])
def test_empty_and_tiny_preserve_geometry_screen(empty):
    m=np.zeros((4,4,4),np.uint8)
    if not empty:m[1,1,1]=1
    assert preflight_metadata(estimate(m),m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0))['resource_guard_reason_code']==STAGE1_CODE


def test_boundary_is_inclusive_and_stricter_admin_limit_is_not_overridden():
    m=np.ones((2,2,2),np.uint8)
    e=replace(estimate(m),estimated_resampled_bbox_voxels=15000000)
    assert preflight_metadata(e,m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0))['resource_guard_reason_code']==STAGE1_CODE
    assert preflight_metadata(e,m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0),limit=1000) is None


def test_unknown_filter_profile_does_not_override_bbox():
    m=np.ones((2,2,2),np.uint8)
    e=replace(estimate(m),estimated_resampled_bbox_voxels=15000001)
    assert preflight_metadata(e,m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0),image_types={'Exponential':{}}) is None


def test_real_filtered_gray_range_guard_and_cropping():
    sitk=pytest.importorskip('SimpleITK')
    class Extractor:
        settings={}
        def computeFeatures(self,*args,**kwargs):
            return {'finite':1.0}
    m=np.ones((3,3,3),np.uint8)
    a=np.zeros((3,3,3),np.float64);a[0,0,0]=1.e8
    img=sitk.GetImageFromArray(a);mask=sitk.GetImageFromArray(m)
    meta={'resource_guard_native_image_voxels':27,'resource_guard_crop_voxels_bound':27,'resource_guard_predicted_peak_bytes':0}
    e=install_texture_budget(Extractor(),meta)
    assert e.settings['preCrop'] is True
    with pytest.raises(RadiomicsMemoryLimit):e.computeFeatures(img,mask,'Original',binWidth=25)
    a[0,0,0]=25
    assert e.computeFeatures(sitk.GetImageFromArray(a),mask,'Original',binWidth=25)=={'finite':1.0}


def test_per_image_overrides_cannot_evade_texture_dimensions():
    m=np.ones((2,2,2),np.uint8)
    e=replace(estimate(m),estimated_resampled_bbox_voxels=15000001)
    assert preflight_metadata(e,m,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0),
        settings={'distances':[1,2]},image_types={'Original':{'distances':[20]}}) is None


def test_denominators_expose_admission_stage_without_relabeling_success(tmp_path):
    import json
    from rtpipeline.roi_requiredness import DenominatorLedger
    ledger=DenominatorLedger()
    for i,code in enumerate([STAGE1_CODE,STAGE2_CODE]):
        ledger.record_roi('c','p',str(i),reason_code='extracted',disposition='extracted',resource_guard_reason_code=code)
    ledger.write(tmp_path,prefix='radiomics_ct')
    payload=json.loads((tmp_path/'radiomics_ct_denominators.json').read_text())
    assert payload['COURSE_ROI']['0'][STAGE1_CODE]==1
    assert payload['COURSE_ROI']['1'][STAGE2_CODE]==1
    assert payload['COURSE_ROI']['1']['extracted']==1


def test_late_rejection_keeps_both_arms_and_explicit_memory_reason(monkeypatch):
    import json
    from rtpipeline import radiomics_ct_contract as c
    from rtpipeline.radiomics_memory import legacy_rejection
    from test_radiomics_ct_dual_arm import _FakeExtractor
    mask=np.ones((3,3,3),np.uint8)
    legacy=legacy_rejection(estimate(mask),15000000,'m3')
    decision=c.classify_ct_roi('Manual','m3')
    _,raw,primary=c.build_ct_extractors(_FakeExtractor,decision.primary_resegment_range_hu)
    hashes=c._effective_hashes_for_built_extractors(raw,primary,decision)
    expected=c.disposition_rows_for_arms({'roi_name':'m3',**legacy['metadata']},decision=decision,
        disposition='failed',detail=legacy['detail'],failure_kind='resource_limit',
        run_identifier='run',code_revision='revision',native_voxel_count=27,required=False,
        effective_hashes=hashes)
    def reject(*args,**kwargs):raise RadiomicsMemoryLimit('unsafe actual gray range')
    monkeypatch.setattr(c,'_extract_ct_roi_arms_impl',reject)
    actual=c.extract_ct_roi_arms(object(),object(),factory=_FakeExtractor,decision=decision,
        common_metadata={'roi_name':'m3','_resource_guard_legacy':legacy},
        run_identifier='run',code_revision='revision',native_voxel_count=27,required=False)
    assert len(actual) == len(expected) == 2
    assert all(row['roi_structural_code'] == 'ROI_PREDICTED_MEMORY_EXCEEDS_LIMIT' for row in actual)
    assert all(row['extraction_failure_kind'] == 'resource_limit' for row in actual)


def test_admission_metadata_survives_real_arrow_schema(tmp_path):
    import pandas as pd
    from rtpipeline.radiomics_schema import write_radiomics_feature_table_atomic
    frame=pd.DataFrame([{'roi_name':'sparse','resource_guard_reason_code':STAGE2_CODE,
                         'resource_guard_predicted_peak_bytes':1234,'original_firstorder_Mean':1.5}])
    path=write_radiomics_feature_table_atomic(frame,tmp_path/'features.xlsx')
    restored=pd.read_parquet(path)
    assert restored.loc[0,'resource_guard_reason_code']==STAGE2_CODE
    assert restored.loc[0,'original_firstorder_Mean']==1.5


def test_family_scheduling_is_serial_and_configuration_restored():
    sitk=pytest.importorskip('SimpleITK')
    class Extractor:
        settings={}
        enabledFeatures={'firstorder': [], 'glcm': [], 'glszm': []}
        def computeFeatures(self, image, mask, image_type, **kwargs):
            assert len(self.enabledFeatures)==1
            return {next(iter(self.enabledFeatures)): 1.0}
    metadata={'resource_guard_native_image_voxels':64,'resource_guard_crop_voxels_bound':64,'resource_guard_predicted_peak_bytes':0}
    extractor=Extractor(); expected=dict(extractor.enabledFeatures)
    install_texture_budget(extractor,metadata)
    result=extractor.computeFeatures(sitk.GetImageFromArray(np.ones((4,4,4))),sitk.GetImageFromArray(np.ones((4,4,4),np.uint8)),'original',binWidth=25)
    assert set(result)==set(expected) and extractor.enabledFeatures==expected


def test_gray_frequency_and_direction_costs_are_monotonic():
    settings=dict(crop_voxels=1000,native_image_voxels=2000,mask_voxels=1000,gray_levels=100,max_extent=100)
    for field,a,b in [('max_gray_frequency',100,1000),('glcm_directions',13,62),('gldm_neighbors',26,124)]:
        assert predicted_peak_bytes(**settings,**{field:b})>=predicted_peak_bytes(**settings,**{field:a})
