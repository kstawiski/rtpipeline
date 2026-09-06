"""Production PyRadiomics integration. Optional dependency, no surrogate data claims."""
import json
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import pytest
import SimpleITK as sitk
radiomics = pytest.importorskip('radiomics')
from radiomics import featureextractor
from rtpipeline.radiomics_ct_contract import extract_ct_roi_arms, classify_ct_roi
from rtpipeline.radiomics_memory import preflight_metadata, MEMORY_BUDGET_BYTES
from rtpipeline.radiomics_resource_guard import estimate_resampled_bounding_box


def test_sparse_elongated_union_full_configured_features_and_dense_rejection():
    start=time.monotonic()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    mask=np.zeros((300,300,300),np.uint8)
    mask[1:9,1:9,1:9]=1
    mask[-9:-1,-9:-1,-9:-1]=1
    estimate=estimate_resampled_bounding_box(mask,native_spacing_xyz=(1,1,1),resampled_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0),pad_distance=5)
    assert estimate.estimated_resampled_bbox_voxels>15_000_000
    metadata=preflight_metadata(estimate,mask,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0))
    assert metadata is not None
    dense=np.zeros_like(mask);dense[1:-1,1:-1,1:-1]=1
    dense_estimate=estimate_resampled_bounding_box(dense,native_spacing_xyz=(1,1,1),resampled_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0),pad_distance=5)
    assert estimate.estimated_resampled_bbox_shape==dense_estimate.estimated_resampled_bbox_shape
    assert preflight_metadata(dense_estimate,dense,native_spacing_xyz=(1,1,1),array_axis_to_xyz=(2,1,0)) is None
    del dense
    image=np.zeros(mask.shape,np.float32)
    image[mask>0]=np.arange(int(mask.sum()),dtype=np.float32)%100
    image=sitk.GetImageFromArray(image);mask_image=sitk.GetImageFromArray(mask)
    params=Path(__file__).parents[1]/'rtpipeline/radiomics_params.yaml'
    decision=classify_ct_roi('Custom','bowel_bag',custom_provenance={'bowel_bag':{'operation':'union','source_structures':['colon','small_bowel','duodenum'],'margin':0}})
    rows=extract_ct_roi_arms(image,mask_image,factory=lambda:featureextractor.RadiomicsFeatureExtractor(str(params)),decision=decision,
        common_metadata={'roi_name':'synthetic_sparse_union'},run_identifier='integration',code_revision='candidate',native_voxel_count=int(mask.sum()),required=False)
    assert all(r['extraction_status']=='success' for r in rows)
    features=[{k:float(v) for k,v in r.items() if k.startswith(('original_','wavelet-','log-sigma-')) and np.ndim(v)==0 and isinstance(v,(float,int,np.number))} for r in rows]
    assert len(features)==2 and all(len(f)>1000 and all(math.isfinite(v) for v in f.values()) for f in features)
    result={'status':'passed','bbox_voxels':estimate.estimated_resampled_bbox_voxels,'native_foreground_voxels':int(mask.sum()),
       'finite_feature_cells':sum(map(len,features)),'predicted_peak_bytes':rows[0]['resource_guard_predicted_peak_bytes'],
       'budget_bytes':MEMORY_BUDGET_BYTES,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.monotonic()-start}
    assert result['peak_rss_bytes']<=result['predicted_peak_bytes']<=MEMORY_BUDGET_BYTES
    if os.environ.get('RADIOMICS_VALIDATION_RECEIPT'):
        Path(os.environ['RADIOMICS_VALIDATION_RECEIPT']).write_text(json.dumps(result,indent=2))
