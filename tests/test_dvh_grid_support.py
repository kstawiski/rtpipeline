from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from rtpipeline import dvh
from rtpipeline.dvh_support import (mask_grid_coverage, resample_grid_support,
    rtstruct_grid_coverage, nifti_identity, publish_derived_mask, validate_derived_mask)


def annotated(metrics, coverage, **kwargs):
    return dvh.annotate_dvh_metrics(metrics, technique='EBRT', structure_name='BLADDER',
        rtstruct_sop_instance_uid='1.2.3', rtstruct_path=Path('RS.dcm'),
        grid_coverage=coverage, **kwargs)


def boxes(x0, x1, z0=0, z1=2):
    rs=Dataset(); items=[]
    for z in [z0,z1]:
        item=Dataset();item.ContourGeometricType='CLOSED_PLANAR'
        item.ContourData=[x0,2,z,x1,2,z,x1,4,z,x0,4,z];items.append(item)
    roi=Dataset();roi.ReferencedROINumber=1;roi.ContourSequence=Sequence(items)
    rs.ROIContourSequence=Sequence([roi]);dose=Dataset()
    dose.Rows=10;dose.Columns=10;dose.NumberOfFrames=3
    dose.PixelSpacing=[1.,1.];dose.ImagePositionPatient=[0.,0.,0.]
    dose.ImageOrientationPatient=[1.,0.,0.,0.,1.,0.];dose.GridFrameOffsetVector=[0.,1.,2.]
    return rs,dose


@pytest.mark.parametrize('x0,x1,status,fraction',[(2,4,'fully_covered',1),(-2,4,'partial_grid',2/3),(20,22,'outside_grid',0)])
def test_every_positive_rtstruct_has_coverage(x0,x1,status,fraction):
    rs,dose=boxes(x0,x1)
    coverage=rtstruct_grid_coverage(rs,1,dose)
    assert coverage['status']==status
    assert coverage['fraction']==pytest.approx(fraction,abs=1e-12)
    row=annotated({'DmeanGy':4.,'DmaxGy':6.,'D0.03ccGy':5.},coverage)
    assert row['dose_metric_usable_for_dose_response'] == (status=='fully_covered')
    assert row['DmeanGy']==(4. if status=='fully_covered' else None)
    if status!='fully_covered':
        assert row['dose_metric_status']==status
        assert row['D0.03ccGy'] is None


def test_rtstruct_z_partial_has_volume_fraction():
    rs,dose=boxes(2,4,-2,2)
    c=rtstruct_grid_coverage(rs,1,dose)
    assert c['fraction']==pytest.approx(.5)


def test_resampled_support_distinguishes_true_zero_from_padding():
    dose=sitk.Image([2,2,2],sitk.sitkFloat32) # genuine zero on entire grid
    reference=sitk.Image([4,2,2],sitk.sitkFloat32)
    support=resample_grid_support(dose,reference)
    inside=np.zeros((2,2,4),bool);inside[:,:,0]=True
    outside=np.zeros_like(inside);outside[:,:,3]=True
    partial=inside|outside
    for mask,status,fraction in [(inside,'fully_covered',1),(outside,'outside_grid',0),(partial,'partial_grid',.5)]:
        c=mask_grid_coverage(mask,support,.001)
        row=annotated(dvh._compute_metrics_from_arrays(np.zeros(mask.sum()),.001,0.),c)
        assert row['dose_grid_coverage_fraction']==fraction
        assert row['dose_grid_coverage_status']==status
        assert row['DmeanGy']==(0. if status=='fully_covered' else None)
        assert row['DmaxGy']==(0. if status=='fully_covered' else None)


def test_partial_covered_diagnostics_are_separate():
    row=annotated({'DmeanGy':2.,'DmaxGy':4.},{'status':'partial_grid','fraction':.5},covered_metrics={'DmeanGy':4.,'DmaxGy':4.})
    assert row['DmeanGy'] is None
    assert row['diagnostic_legacy_DmeanGy']==2.
    assert row['covered_DmeanGy']==4.


def test_absent_grid_evidence_cannot_claim_whole_roi_measurement():
    row=dvh.annotate_dvh_metrics({'DmeanGy':5.,'DmaxGy':8.},technique='EBRT',structure_name='X',rtstruct_sop_instance_uid='1.2.3',rtstruct_path=Path('RS.dcm'))
    assert row['dose_metric_status']=='coverage_unresolved'
    assert row['DmeanGy'] is None
    assert not row['dose_metric_usable_for_dose_response']


def test_corrupt_nifti_bytes_are_rejected_with_explicit_provenance(tmp_path):
    path=tmp_path/'fake.nii.gz';path.write_bytes(b'not a NIfTI')
    row=dvh.annotate_dvh_metrics({'DmeanGy':5.},technique='EBRT',structure_name='X',rtstruct_sop_instance_uid=None,rtstruct_path=None,structure_provenance_type='NIFTI_MASK',structure_provenance_path=path,grid_coverage={'status':'fully_covered','fraction':1.})
    assert row['structure_provenance_status']=='invalid_mask_provenance'
    assert not row['dose_metric_usable_for_dose_response']


def make_closure(tmp_path):
    image=sitk.Image([4,4,4],sitk.sitkUInt8)+1
    source=tmp_path/'source.nii.gz';sitk.WriteImage(image,str(source))
    ct=tmp_path/'ct.nii.gz';sitk.WriteImage(sitk.Cast(image,sitk.sitkInt16),str(ct))
    config=tmp_path/'custom.yaml';config.write_text('custom_structures: []\n')
    definition=[{'name':'derived','operation':'union','source_structures':['source'],'margin':None}]
    closure=publish_derived_mask(tmp_path/'derived','derived',sitk.GetArrayFromImage(image),image,[source],config,definition,{'derived':{'status':'generated'}},[ct])
    return source,config,closure


def test_yaml_never_passes_nifti_and_complete_derived_closure_does(tmp_path):
    source,config,closure=make_closure(tmp_path)
    with pytest.raises(ValueError):nifti_identity(config)
    row=dvh.annotate_dvh_metrics({'DmeanGy':2.}, technique='EBRT',structure_name='X',rtstruct_path=None,rtstruct_sop_instance_uid=None,structure_provenance_type='NIFTI_MASK',structure_provenance_path=config,grid_coverage={'status':'fully_covered','fraction':1.})
    assert row['structure_provenance_status']=='invalid_mask_provenance'
    assert not row['dose_metric_usable_for_dose_response']
    row=dvh.annotate_dvh_metrics({'DmeanGy':2.}, technique='EBRT',structure_name='X',rtstruct_path=None,rtstruct_sop_instance_uid=None,structure_provenance_type='DERIVED_NIFTI_MASK',structure_provenance_path=closure,grid_coverage={'status':'fully_covered','fraction':1.})
    assert row['structure_provenance_status']=='traceable'
    assert len(row['structure_provenance_sha256'])==64
    assert validate_derived_mask(closure)['sha256']==row['structure_provenance_sha256']


def test_changed_component_rejects_stale_closure_even_with_same_mtime(tmp_path):
    import os
    source,config,closure=make_closure(tmp_path)
    previous=source.stat()
    image=sitk.ReadImage(str(source));image[1,1,1]=0;sitk.WriteImage(image,str(source))
    os.utime(source,ns=(previous.st_atime_ns,previous.st_mtime_ns))
    with pytest.raises(ValueError,match='bytes changed'):validate_derived_mask(closure)


def test_dvh_cache_rejects_changed_component_bytes(tmp_path, monkeypatch):
    import os
    import pandas as pd
    from types import SimpleNamespace
    from rtpipeline.dvh_support import sha256_file
    source, config, closure = make_closure(tmp_path)
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    contract = metadata / 'case_metadata.json'
    contract.write_text('{}')
    row = {k: None for k in [
        'D0.03ccGy', 'D0.03cc_status', 'dose_grid_coverage_fraction',
        'dose_grid_coverage_status', 'dose_response_eligible',
        'dose_metric_usable_for_dose_response', 'dose_metric_status', 'target_like',
        'ROI_Interpreted_Type', 'treatment_technique', 'relative_metric_status',
        'structure_provenance_status', 'zero_dose_status', 'zero_dose_trigger_metric',
        'Dose_Plan_Scope_Status', 'Course_Treatment_Isocenter_Status',
        'Course_Treatment_Isocenter_Count', 'Course_Target_Dose_Coverage_Status']}
    row.update(structure_provenance_type='DERIVED_NIFTI_MASK',
               structure_provenance_path=str(closure),
               structure_provenance_sha256=sha256_file(closure))
    parquet = tmp_path / 'dvh_metrics.parquet'
    pd.DataFrame([row]).to_parquet(parquet)
    qc = {'status': 'ok', 'course_contract_sha256': sha256_file(contract),
          'metric_version': dvh.DVH_METRIC_VERSION, 'row_count': 1,
          'rx_relative_metrics_available': True, 'structure_resolution': {'classification': 'bound'}}
    (metadata / 'dvh_qc.json').write_text(json.dumps(qc))
    workbook = tmp_path / 'dvh_metrics.xlsx'
    workbook.write_bytes(b'cache existence fixture')
    monkeypatch.setattr(dvh, 'load_course_contract', lambda path: SimpleNamespace(
        metadata_path=contract, plan_artifact_path=None, dose_grid_path=None,
        authoritative_rtstruct_path=None))
    monkeypatch.setattr(dvh, 'list_custom_model_outputs', lambda path: [])
    assert dvh._is_dvh_up_to_date(tmp_path, workbook)
    stat = source.stat()
    image = sitk.ReadImage(str(source)); image[1,1,1] = 0
    sitk.WriteImage(image, str(source))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not dvh._is_dvh_up_to_date(tmp_path, workbook)


def test_d003_known_analytic_dvh():
    histogram=SimpleNamespace(bincenters=np.array([0.,5.,10.]),counts=np.array([1.,.5,0.]),mean=5.,min=0.,max=10.)
    metrics=dvh._compute_metrics(histogram,None)
    assert metrics['D0.03ccGy']==pytest.approx(9.7,abs=1e-12)
    assert metrics['D0.03cc_status']=='available'
    assert metrics['D0.1ccGy']<=metrics['D0.03ccGy']<=metrics['DmaxGy']


@pytest.mark.parametrize('volume,code',[(.03,'available'),(np.nextafter(.03,0),'roi_below_0.03cc'),(.02,'roi_below_0.03cc'),(.0005,'roi_below_0.03cc')])
def test_d003_exact_boundary_both_backends(volume,code):
    h=SimpleNamespace(bincenters=np.array([1.,2.]),counts=np.array([volume,0.]),mean=1.,min=1.,max=1.)
    a=dvh._compute_metrics(h,None)
    b=dvh._compute_metrics_from_arrays(np.array([1.]),volume,1.)
    assert a['D0.03cc_status']==b['D0.03cc_status']==code
    assert a['D0.03ccGy']==b['D0.03ccGy']==(1. if code=='available' else None)


def test_d003_nonconstant_backend_agreement_one_cgy_tolerance():
    from dicompylercore.dvh import DVH
    doses=np.arange(1000)*.01+.001
    edges=np.linspace(0,10.02,1003)
    hist,_=np.histogram(doses,bins=edges)
    h=DVH(counts=hist*.001,bins=edges,dvh_type='differential',dose_units='Gy').cumulative
    a=dvh._compute_metrics(h,None);b=dvh._compute_metrics_from_arrays(doses,.001,10.01)
    assert a['D0.03ccGy']==pytest.approx(b['D0.03ccGy'],abs=.01)
    assert b['D0.1ccGy']<=b['D0.03ccGy']<=b['DmaxGy']


def test_all_preexisting_fully_covered_metric_bytes_unchanged():
    import struct
    golden=json.loads((Path(__file__).parent/'fixtures/dvh_prepatch_metrics.json').read_text())
    for case in golden:
        result=dvh._compute_metrics_from_arrays(np.asarray(case['doses']),case['voxel_volume'],case['max_dose'],case['rx'])
        for key,value in case['metrics'].items():
            if value is None:assert result[key] is None
            else:assert struct.pack('!d',result[key])==struct.pack('!d',value),key
        row=annotated(result,{'status':'fully_covered','fraction':1.})
        for key in ['DmeanGy','DmaxGy','DminGy','D95Gy','D0.1ccGy']:
            assert row[key]==result[key]
