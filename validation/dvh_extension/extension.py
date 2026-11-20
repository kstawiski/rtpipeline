"""Predeclared analytic extension; public synthetic data; production full-course DVH.

Run via run_extension.sh. No production function is patched. Case selection and
all reporting rules are read from the immutable DESIGN.json at execution.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from analytic import Geometry, METRICS, metrics as truth
from generate_phantoms import generate, save, ds_numbers
from run_benchmark import PIPELINE_KEYS, REPO, numeric, sha, versions, source_hashes, public_value

HERE = Path(__file__).resolve().parent

def utc():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, data):
    Path(path).write_text(json.dumps(public_value(data), indent=2, allow_nan=False, default=str)+'\n')

def rotation_matrix(rotation):
    axis = np.array(rotation['axis'], dtype=float)
    axis /= np.linalg.norm(axis)
    x,y,z=axis
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    angle=math.radians(rotation['angle_deg'])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)

def field_values(points, spec):
    if spec['field']=='uniform':
        return np.full(points.shape[:-1],20.)
    if spec['field']=='linear':
        return 20.+.1*points[...,2]
    return 30.-.15*np.linalg.norm(points-np.array(spec['center']),axis=-1)

def inside(points, g):
    q=points-np.array(g.center)
    if g.shape=='cube':
        return np.max(np.abs(q),axis=-1)<=g.radius
    if g.shape=='cylinder':
        return (np.sum(q[...,:2]**2,axis=-1)<=g.radius**2)&(np.abs(q[...,2])<=g.radius)
    r2=np.sum(q*q,axis=-1)
    return (r2<=g.radius**2)&((r2>g.inner_radius**2) if g.shape=='shell' else True)

def add_masks_and_rotate(course, spec):
    """CT-grid binary NIfTI inputs plus physically rotated, re-evaluated dose."""
    ct=sitk.ReadImage(str(course/'NIFTI/ct.nii.gz'))
    z,y,x=np.indices(tuple(reversed(ct.GetSize())),dtype=float)
    ijk=np.stack([x,y,z],axis=-1)
    points=np.array(ct.GetOrigin())+(ijk*np.array(ct.GetSpacing()))@np.array(ct.GetDirection()).reshape(3,3).T
    dest=course/'Segmentation_TotalSegmentator';dest.mkdir()
    mask_evidence={}
    for shape in spec['shapes']:
        g=Geometry(shape,spec['volume_cc'],tuple(spec['center']))
        arr=inside(points,g).astype(np.uint8)
        im=sitk.GetImageFromArray(arr);im.CopyInformation(ct)
        p=dest/f'mask_{shape}.nii.gz';sitk.WriteImage(im,str(p))
        reread=sitk.ReadImage(str(p))
        assert np.array_equal(sitk.GetArrayFromImage(reread),arr)
        assert reread.GetSize()==ct.GetSize() and np.allclose(reread.GetOrigin(),ct.GetOrigin(),atol=1e-5)
        mask_evidence[shape]={'voxels':int(arr.sum()),'voxel_volume_cc':float(np.prod(ct.GetSpacing())/1000),'volume_cc':float(arr.sum()*np.prod(ct.GetSpacing())/1000),'sha256':sha(p),'grid':'planning_CT','rule':'binary voxel-centre membership'}
    if spec['rotation']['angle_deg']:
        p=course/'DICOM/RTDOSE/dose.dcm';d=pydicom.dcmread(p)
        R=rotation_matrix(spec['rotation']);oldorigin=np.array(d.ImagePositionPatient,dtype=float)
        origin=R@oldorigin
        nz,ny,nx=int(d.NumberOfFrames),int(d.Rows),int(d.Columns)
        zz,yy,xx=np.indices((nz,ny,nx),dtype=float)
        local=np.stack([xx*float(d.PixelSpacing[1]),yy*float(d.PixelSpacing[0]),np.asarray(d.GridFrameOffsetVector,dtype=float)[:,None,None]+np.zeros_like(xx)],axis=-1)
        physical=origin+local@R.T
        vals=field_values(physical,spec)
        if np.any(vals<0): raise ValueError('Negative analytic field on dose grid')
        d.ImageOrientationPatient=ds_numbers(np.r_[R[:,0],R[:,1]])
        d.ImagePositionPatient=ds_numbers(origin)
        d.PixelData=np.rint(vals/float(d.DoseGridScaling)).astype('<u4').tobytes()
        save(d,p)
    return mask_evidence

def size_class(v):
    return '<1' if v<1 else '1-20' if v<=20 else '20-200' if v<=200 else '>200'

def metric_rows(spec,path,record,reason=''):
    shape=record.pop('_shape')
    g=Geometry(shape,spec['volume_cc'],tuple(spec['center']))
    refs=truth(g,spec['field']);refs['V20Gy_pct']=100*refs['V20Gy_cc']/g.volume_cc
    values={m:numeric(record.get(k)) for m,k in PIPELINE_KEYS.items()}
    v=values['Volume_cc'];v20=values['V20Gy_cc']
    values['V20Gy_pct']=100*v20/v if v20 is not None and v is not None and v>0 else None
    result=[]
    for metric in [*METRICS,'V20Gy_pct']:
        ref,val=refs[metric],values[metric]
        err=val-ref if val is not None and ref is not None else None
        boundary=None
        descriptive=(g.volume_cc<1 or metric in {'Dmin_Gy','Dmax_Gy','D0.03cc_Gy','V20Gy_cc'})
        if not descriptive:
            if metric.startswith('D') and ref is not None: boundary=max(.02*abs(ref),.5)
            elif metric.endswith('_pct'): boundary=2.
            elif metric=='Volume_cc': boundary=.03*g.volume_cc
        unavail=reason
        if val is None and not unavail:
            unavail=str(record.get('dvh_computation_error') or record.get({'D2cc_Gy':'D2cc_status','D0.03cc_Gy':'D0.03cc_status'}.get(metric,'')) or record.get('dose_metric_status') or 'no_published_value')
        ref_reason='analytic_volume_below_request' if ref is None else ''
        status='unavailable' if err is None else 'descriptive' if descriptive else 'beyond_boundary' if abs(err)>boundary else 'within_boundary'
        result.append(dict(case=spec['group']+'_'+path+'_'+shape,group=spec['group'],legacy_case=spec['legacy_group']+'_'+shape,path=path,shape=shape,size_class=size_class(g.volume_cc),volume_cc=g.volume_cc,phase=spec['phase'],center_x_mm=g.center[0],center_y_mm=g.center[1],center_z_mm=g.center[2],ct_inplane_mm=2.,ct_mm=spec['ct_mm'],contour_mm=spec['ct_mm'] if path=='rtstruct' else None,dose_mm=spec['dose_mm'],rotation=spec['rotation']['id'],rotation_deg=spec['rotation']['angle_deg'],rotation_axis=json.dumps(spec['rotation']['axis']),field=spec['field'],metric=metric,reference=ref,value=val,signed_error=err,absolute_error=abs(err) if err is not None else None,relative_error=abs(err/ref) if err is not None and ref!=0 else None,unit='percentage_points' if metric.endswith('_pct') else 'cm3' if metric in {'Volume_cc','V20Gy_cc'} else 'Gy',boundary=boundary,beyond_boundary=bool(abs(err)>boundary) if err is not None and boundary is not None else None,status=status,unavailable_reason=unavail if val is None else '',reference_unavailable_reason=ref_reason,production_source=record.get('Segmentation_Source',''),production_coverage=record.get('dose_grid_coverage_status',''),production_rotation_status=record.get('dose_grid_resampling_status',''),production_metric_status=record.get('dose_metric_status',''),cause=reason))
    return result

def run(args):
    root=args.design.resolve().parent;design=json.loads(args.design.read_text());digest=sha(args.design)
    if not args.out.resolve().is_relative_to(root): raise ValueError('output must be inside the design directory')
    scratch=Path(os.environ.get('TMPDIR', root/'scratch')).resolve()
    if not scratch.is_relative_to(root/'scratch'):raise ValueError('scratch must be inside the design directory scratch folder')
    scratch.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    receipt={'start_utc':utc(),'design_sha256':digest,'design_mtime_ns':args.design.stat().st_mtime_ns,'source_sha256':source_hashes(),'source_module':None,'runtime':sys.executable,'python':sys.version,'versions':versions(),'command':sys.argv,'shard':args.shard,'shards':args.shards,'groups':[],'errors':[],'status':'running'}
    from rtpipeline.dvh import dvh_for_course
    import rtpipeline.dvh as dvh
    receipt['source_module']=dvh.__file__
    receipt['dvh_source_sha256']=sha(dvh.__file__)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError('Require shards >= 1 and 0 <= shard < shards')
    groups=[s for i,s in enumerate(design['groups']) if i%args.shards==args.shard]
    if args.group:groups=[s for s in groups if s['group']==args.group]
    if not groups: raise ValueError('No matching design groups')
    # Case membership and geometry remain frozen; group related CT inputs.
    groups.sort(key=lambda s:(s['volume_cc'],s['phase'],s['ct_mm'],s['field'],s['rotation']['angle_deg'],s['rotation']['id']))
    rows=[]
    cached_rows={};cached_groups={}
    if args.resume_from:
        import csv
        for rp in sorted(args.resume_from.glob('shard-*/receipt.json')):
            prior=json.loads(rp.read_text())
            if prior['design_sha256'] != digest or prior['dvh_source_sha256'] != receipt['dvh_source_sha256']:
                raise ValueError('Resume inputs have a different design or production DVH source')
            finished={r['group']:r for r in prior['groups'] if r['status'] in {'completed','refused'}}
            for row in csv.DictReader((rp.parent/'metrics_long.csv').open()):
                if row['group'] in finished:cached_rows.setdefault(row['group'],[]).append(row)
            for group,record in finished.items():cached_groups[group]=(record,rp.parent/'raw'/group)
        receipt['resume_from']=str(args.resume_from)
    write_json(args.out/'receipt.json',receipt)
    for i,spec in enumerate(groups,1):
        print(f'[{i}/{len(groups)}] {spec["group"]} {utc()}',flush=True)
        if spec['group'] in cached_groups:
            previous,source_raw=cached_groups[spec['group']]
            expected_rows=len(spec['paths'])*len(spec['shapes'])*13
            if len(cached_rows[spec['group']])!=expected_rows:raise ValueError('incomplete cache group')
            rows+=cached_rows[spec['group']]
            raw=args.out/'raw'/spec['group'];raw.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source_raw,raw,dirs_exist_ok=True)
            receipt['groups'].append({**previous,'retained_completed_group':True})
            pd.DataFrame(rows).to_csv(args.out/'metrics_long.csv',index=False,float_format='%.17g')
            write_json(args.out/'receipt.json',receipt)
            continue
        work=Path(tempfile.mkdtemp(prefix='ext-',dir=scratch));raw=args.out/'raw'/spec['group'];raw.mkdir(parents=True,exist_ok=True)
        rec={'group':spec['group'],'started_utc':utc()}
        try:
            course,manifest=generate(work,spec)
            write_json(raw/'phantom.json',manifest)
            masks=add_masks_and_rotate(course,spec);write_json(raw/'masks.json',masks)
            hashes=[{'path':str(p.relative_to(course)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(course.rglob('*')) if p.is_file()]
            write_json(raw/'input_sha256.json',hashes)
            output=dvh_for_course(course,parallel_workers=1,use_cropped=False,rx_dose_gy=25.,max_total_dose_gy=100.)
            for rel,name in [('metadata/dvh_qc.json','dvh_qc.json'),('metadata/case_metadata.json','case_metadata.json'),('dvh_curves.json','dvh_curves.json')]:
                p=course/rel
                if p.exists():shutil.copyfile(p,raw/name)
            table=pd.read_parquet(course/'dvh_metrics.parquet') if (course/'dvh_metrics.parquet').exists() else pd.DataFrame()
            table.to_csv(raw/'rtpipeline_metrics.csv',index=False,float_format='%.16g')
            refused=spec['rotation']['angle_deg']>1
            if refused:
                qc=json.loads((raw/'dvh_qc.json').read_text())
                if 'rotated_dose_rotation_exceeds_1_degree' not in json.dumps(qc):
                    raise RuntimeError('Expected rotation refusal absent from published QC')
            for path in spec['paths']:
                for shape in spec['shapes']:
                    name=shape if path=='rtstruct' else 'mask_'+shape
                    match=table[table.ROI_OriginalName==name] if 'ROI_OriginalName' in table else pd.DataFrame()
                    r=match.iloc[0].to_dict() if len(match)==1 else {}
                    if len(match)>1:raise RuntimeError(f'duplicate published {name}')
                    reason=''
                    if refused:reason='rotated_dose_rotation_exceeds_1_degree; rtpipeline.dvh_rotation.prepare_rotated_dose; NIfTI gate rtpipeline.dvh._compute_dvh_for_course'
                    elif not r and path=='nifti_mask' and masks[shape]['voxels']==0:
                        reason='empty_CT_grid_mask; rtpipeline.dvh._compute_nifti_based_dvh'
                    elif not r: reason='no_published_ROI; see raw/dvh_qc.json; rtpipeline.dvh._compute_dvh_for_course._calculate_roi'
                    if r and not refused:
                        if path=='nifti_mask' and r.get('Segmentation_Source')!='NIfTI_Direct':raise RuntimeError('NIfTI path not exercised')
                        if r.get('dose_grid_coverage_status')!='fully_covered':
                            # A published refusal/nonmeasurement is a case result.
                            # Numeric dose values on an unexpectedly uncovered
                            # synthetic ROI still signal a harness/FOV defect.
                            if any(numeric(r.get(k)) is not None for k in ('DmeanGy','D95Gy','D2Gy')):
                                raise RuntimeError(f'Unexpected numeric dose with incomplete grid coverage {name}')
                            reason=str(r.get('dvh_computation_error') or r.get('dvh_nonmeasurement_class') or r.get('dose_metric_status') or 'production_nonmeasurement')
                            if 'insufficient_contour_planes' in reason:
                                reason+='; rtpipeline.dvh_rotation.RotatedDoseGrid.get_dvh; rtpipeline.dvh._compute_dvh_for_course._calculate_roi'
                            else:
                                reason+='; rtpipeline.dvh._compute_dvh_for_course._calculate_roi'
                        if spec['rotation']['angle_deg'] and r.get('dose_grid_resampling_status')!='rotated_dose_resampled_to_axis_aligned_grid':raise RuntimeError('Rotated sampler not exercised')
                    rows+=metric_rows(spec,path,{**r,'_shape':shape},reason)
            rec.update(status='refused' if refused else 'completed',production_output=bool(output))
        except Exception as e:
            error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'group':spec['group']}
            write_json(raw/'execution_error.json',error);receipt['errors'].append(error)
            rec.update(status='execution_error',reason=str(e))
            # A harness/runtime error is retained, never mislabelled as a production refusal.
            for path in spec['paths']:
                for shape in spec['shapes']:
                    case=spec['group']+'_'+path+'_'+shape
                    if not any(r['case']==case for r in rows):
                        rows+=metric_rows(spec,path,{'_shape':shape},'execution_error: '+str(e))
        finally:
            shutil.rmtree(work)
        rec['ended_utc']=utc();receipt['groups'].append(rec)
        pd.DataFrame(rows).to_csv(args.out/'metrics_long.csv',index=False,float_format='%.17g')
        write_json(args.out/'receipt.json',receipt)
    assert sha(args.design)==digest
    receipt.update(ended_utc=utc(),status='failed' if receipt['errors'] else 'completed',rows=len(rows),cases=len({r['case'] for r in rows}))
    write_json(args.out/'receipt.json',receipt)
    return bool(receipt['errors'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--design',type=Path,default=HERE/'DESIGN.json');p.add_argument('--out',type=Path,default=HERE/'results/shard-0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--group');p.add_argument('--resume-from',type=Path)
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(run(p.parse_args()))
