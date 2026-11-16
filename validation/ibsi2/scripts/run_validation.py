"""IBSI-2 public phantom benchmark; use the radiomics Python -I -B.

Native outputs use RTpipeline's extractor and its unchanged image operations.
Diagnostic adapters are explicitly labelled and never count as native support.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
DATA=ROOT/'data/public_sources_20261106'
OUT=ROOT/'results'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(REPO))
import numpy as np
import SimpleITK as sitk
import pywt
import radiomics
from radiomics import imageoperations as io
from scipy import ndimage
from scipy.signal import fftconvolve
import yaml
from rtpipeline.radiomics import _extractor
from rtpipeline.ibsi_preprocessing import spatial_preprocess
from configurations import all_configurations, PYRAD

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
logging.getLogger('radiomics').setLevel(logging.ERROR)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,
                               default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def table(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def image(array,reference):
    out=sitk.GetImageFromArray(np.asarray(array)); out.CopyInformation(reference); return out
def array(im): return sitk.GetArrayFromImage(im).astype(np.float64)
def read(path): return sitk.ReadImage(str(path),sitk.sitkFloat32)
def describe(im,mask=None):
    d=dict(size_xyz=list(im.GetSize()),spacing_xyz=list(im.GetSpacing()),origin_xyz=list(im.GetOrigin()),direction=list(im.GetDirection()))
    if mask is not None: d['roi_voxels']=int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
    return d

def native_map(im,cfg):
    if cfg['native_operation']=='none': return im
    if cfg['native_operation']=='LoG':
        return next(io.getLoGImage(im,None,sigma=[float(cfg['sigma_mm'])]))[0]
    kwargs=dict(wavelet=cfg['wavelet_family'],start_level=0,level=int(cfg['level']),
                force2D=cfg['dimensions']==2,force2Ddimension=0)
    name=('wavelet-' if int(cfg['level'])==1 else 'wavelet2-')+cfg['subband']
    return next(x for x,n,_ in io.getWaveletImage(im,None,**kwargs) if n==name)

def padded_map(im,cfg,mode,width=40,proper_level=False):
    a=sitk.GetArrayFromImage(im)
    axes=(1,2) if cfg['dimensions']==2 else (0,1,2)
    pads=[(width,width) if k in axes else (0,0) for k in range(3)]
    if proper_level:
        multiple=2**int(cfg['level'])
        pads=[(lo,hi+(-(a.shape[k]+lo+hi)%multiple)) if k in axes else (lo,hi)
              for k,(lo,hi) in enumerate(pads)]
    pa=np.pad(a,pads,mode=mode)
    pi=sitk.GetImageFromArray(pa); pi.SetSpacing(im.GetSpacing())
    if proper_level:
        key=cfg['subband'][::-1].replace('L','a').replace('H','d')
        # PyRadiomics sub-band letters are x,y,z; swtn axes here are z,y,x.
        lev=int(cfg['level'])
        maps=pywt.swtn(pa,cfg['wavelet_family'],level=lev,axes=axes)
        result=maps[0][key]
    else:
        result=array(native_map(pi,cfg))
    cut=tuple(slice(pads[k][0],pads[k][0]+a.shape[k]) for k in range(3))
    return image(result[cut],im)

def rotations(dim):
    if dim==2:
        for n in range(4):
            yield lambda a,n=n:np.rot90(a,n,axes=(1,2)),lambda a,n=n:np.rot90(a,-n,axes=(1,2))
    else:
        for perm in itertools.permutations(range(3)):
            p=np.eye(3)[list(perm)]
            for signs in itertools.product([-1,1],repeat=3):
                if round(np.linalg.det(p)*np.prod(signs))!=1: continue
                flips=tuple(slice(None,None,int(s)) for s in signs)
                inv=np.argsort(perm)
                yield (lambda a,p=perm,f=flips:np.transpose(a,p)[f],
                       lambda a,inv=inv,f=flips:np.transpose(a[f],inv))

def pooled(im,cfg,mode='wrap',proper_level=False):
    a=sitk.GetArrayFromImage(im); out=np.zeros(a.shape,dtype=np.float64); count=0
    for forward,back in rotations(cfg['dimensions']):
        ri=sitk.GetImageFromArray(forward(a).copy());ri.SetSpacing(im.GetSpacing())
        result=padded_map(ri,cfg,mode,proper_level=proper_level)
        out+=back(array(result));count+=1
    assert count in [4,24]
    return image(out/count,im)

def sampled_log(im,sigma,mode='reflect'):
    # IBSI LoG equation (sec:LaplacianOfGaussian): sampled derivatives,
    # finite cutoff 4 sigma.
    # Digital benchmark and phase-2 B have isotropic spacing.
    spacing=np.asarray(im.GetSpacing())
    assert np.allclose(spacing,spacing[0])
    sv=sigma/spacing[0]
    radius=int(np.floor(4*sv+.5))
    x=np.arange(-radius,radius+1,dtype=float)
    g=np.exp(-x*x/(2*sv*sv))/(np.sqrt(2*np.pi)*sv)
    d2=(x*x-sv*sv)/(sv**4)*g
    a=array(im);result=np.zeros_like(a)
    for deriv_axis in range(3):
        temp=a
        for axis in range(3):
            temp=ndimage.convolve1d(temp,d2 if axis==deriv_axis else g,axis=axis,mode=mode,cval=0.)
        result+=temp
    return image(result,im)

def truncated_recursive_log(im,sigma,mode):
    """Truncate the measured native IIR impulse kernel at the IBSI cutoff.

    This separates finite support from the subsequent kernel discretisation
    change. The impulse lattice is fixed, with >=8 sigma to its boundaries.
    """
    spacing=np.asarray(im.GetSpacing());assert np.allclose(spacing,spacing[0])
    sv=sigma/spacing[0];radius=int(np.floor(4*sv+.5))
    half=max(32,int(np.ceil(8*sv)))
    impulse=np.zeros((2*half+1,)*3,dtype=np.float64);impulse[(half,)*3]=1.
    ii=sitk.GetImageFromArray(impulse);ii.SetSpacing(im.GetSpacing())
    kernel=array(next(io.getLoGImage(ii,None,sigma=[sigma]))[0])/(sv**2)
    kernel=kernel[(slice(half-radius,half+radius+1),)*3]
    padded=np.pad(sitk.GetArrayFromImage(im),[(radius,radius)]*3,mode=mode)
    return image(fftconvolve(padded,kernel,mode='valid'),im)

def agreement(cid,cfg,variant,parent,factor,computed,reference):
    x=array(computed);y=array(reference)
    assert x.shape==y.shape,(cid,x.shape,y.shape)
    delta=np.abs(x-y);tol=float(.01*np.ptp(y));within=delta<=tol
    # The public MATLAB tool historically used the combined range. The manual
    # and reference repository specify the reference range; retain both.
    combined=.01*(max(x.max(),y.max())-min(x.min(),y.min()))
    return dict(configuration=cid,expressibility=cfg['expressibility'],variant=variant,parent_variant=parent,
                changed_factor=factor,comparison_scope='native' if variant=='native' else 'diagnostic adapter; not production',
                voxel_count=x.size,reference_min=float(y.min()),reference_max=float(y.max()),absolute_tolerance=tol,
                voxels_within_tolerance=int(within.sum()),fraction_within_tolerance=float(within.mean()),
                maximum_absolute_difference=float(delta.max()),mean_absolute_difference=float(delta.mean()),
                all_voxels_within_tolerance=bool(within.all()),
                historical_combined_range_fraction=float((delta<=combined).mean()),status='within tolerance' if within.all() else 'outside tolerance',reason='')

def phase1(configs):
    rows=[];geometries={}
    for cfg in [c for c in configs if c['phase']==1]:
        cid=cfg['configuration'];refpath=DATA/'theibsi/ibsi_2_reference_data/reference_response_maps'/(cid.replace('.','_')+'-ValidCRM.nii')
        if cfg['expressibility']=='not expressible' or not refpath.exists():
            rows.append(dict(configuration=cid,expressibility=cfg['expressibility'],variant='native',parent_variant='',changed_factor='',
                             comparison_scope='not run',voxel_count='',reference_min='',reference_max='',absolute_tolerance='',
                             voxels_within_tolerance='',fraction_within_tolerance='',maximum_absolute_difference='',mean_absolute_difference='',
                             all_voxels_within_tolerance='',historical_combined_range_fraction='',status='not computed',
                             reason=cfg['parameter_differences'] if cfg['expressibility']=='not expressible' else 'reference file unavailable; branch stopped'))
            continue
        im=read(DATA/f'theibsi/data_sets/ibsi_2_digital_phantom/nifti/{cfg["phantom"]}/image/{cfg["phantom"]}.nii.gz')
        ref=read(refpath)
        geometries[cid]=dict(input=describe(im),reference=describe(ref),comparison='array indices x,y,z; no reference interpolation')
        native=native_map(im,cfg)
        rows.append(agreement(cid,cfg,'native','','none',native,ref))
        mapdir=OUT/'maps';mapdir.mkdir(exist_ok=True)
        sitk.WriteImage(native,str(mapdir/(cid+'_native.nii.gz')),True)
        if cfg['native_operation']=='LoG':
            sigma=float(cfg['sigma_mm'])
            unscaled=image(array(native)/(sigma*sigma),im)
            rows.append(agreement(cid,cfg,'scale_normalisation_off','native','normalisation only',unscaled,ref))
            voxel_units=image(array(unscaled)*im.GetSpacing()[0]**2,im)
            rows.append(agreement(cid,cfg,'voxel_derivative_units','scale_normalisation_off','derivative units: physical mm to IBSI voxel coordinates only',voxel_units,ref))
            mode='constant' if cfg['boundary_condition']=='zero' else 'symmetric'
            boundary=padded_map(im,cfg,mode,width=80)
            rows.append(agreement(cid,cfg,'specified_boundary','native','boundary extension only',boundary,ref))
            both=image(array(boundary)/(sigma*sigma),im)
            rows.append(agreement(cid,cfg,'boundary_and_no_scale','specified_boundary','normalisation only',both,ref))
            both_voxel=image(array(both)*im.GetSpacing()[0]**2,im)
            rows.append(agreement(cid,cfg,'boundary_no_scale_voxel_units','voxel_derivative_units','boundary extension only; scale normalisation off and voxel-coordinate derivatives held fixed',both_voxel,ref))
            truncated=truncated_recursive_log(im,sigma,mode)
            rows.append(agreement(cid,cfg,'recursive_kernel_cutoff4','boundary_no_scale_voxel_units','finite cutoff 4 sigma only; native impulse coefficients retained',truncated,ref))
            finite=sampled_log(im,sigma,mode='constant' if mode=='constant' else 'reflect')
            rows.append(agreement(cid,cfg,'finite_sampled_kernel','recursive_kernel_cutoff4','kernel discretisation only: recursive impulse coefficients to sampled Gaussian derivatives',finite,ref))
        else:
            mode={'zero':'constant','periodic':'wrap','mirror':'symmetric'}[cfg['boundary_condition']]
            base=native
            parent='native'
            if mode!='wrap':
                base=padded_map(im,cfg,mode)
                rows.append(agreement(cid,cfg,'specified_boundary','native','boundary extension only',base,ref));parent='specified_boundary'
            if cfg['pooling']=='average':
                pool=pooled(im,cfg,mode)
                rows.append(agreement(cid,cfg,'orientation_pooling',parent,'right-angle orientation averaging only',pool,ref));parent='orientation_pooling'
            if int(cfg['level'])==2:
                proper=pooled(im,cfg,mode,proper_level=True)
                rows.append(agreement(cid,cfg,'atrous_level2',parent,'level-2 kernel dilation only',proper,ref))
        print('Phase 1 '+cid+' complete',flush=True)
    table(OUT/'response_map_agreement.csv',rows)
    dump(OUT/'response_map_geometry.json',geometries)
    return rows

def settings(cfg):
    return dict(setting=dict(resampledPixelSpacing=[1.,1.,1.] if cfg['configuration'].endswith('B') else None,
                             interpolator='sitkBSpline',normalize=False,voxelArrayShift=0,binWidth=25,
                             minimumROIDimensions=2,minimumROISize=64,distances=[1,2],force2D=cfg['dimensions']==2,
                             force2Ddimension=0,padDistance=5,correctMask=True,additionalInfo=True,label=1,
                             resegmentRange=[-1000.,400.],preCrop=False),
                imageType=({'Original':{}} if cfg['native_operation']=='none' else
                           {'LoG':{'sigma':[float(cfg['sigma_mm'])]}} if cfg['native_operation']=='LoG' else
                           {'Wavelet':{'wavelet':cfg['wavelet_family'],'start_level':0,'level':int(cfg['level'])}}),
                featureClass={'firstorder':[]})

def make_extractor(parameters,name):
    configdir=OUT/'configs';configdir.mkdir(exist_ok=True)
    path=configdir/(name+'.yaml');path.write_text(yaml.safe_dump(parameters,sort_keys=False))
    ext=_extractor(SimpleNamespace(radiomics_params_file=path),'CT')
    assert ext is not None,'RTpipeline extractor construction failed'
    return ext

def feature_mapping():
    path=ROOT.parent/'ibsi/results/feature_mapping.csv'
    return {r['ibsi_tag_stem']:r for r in csv.DictReader(path.open()) if r['ibsi_tag_stem'].startswith('stat_')}

def compare_features(cfg,variant,values,refs,mapping,parent='',factor='none',scope='native'):
    rows=[]
    for ref in refs:
        m=mapping[ref['feature_tag']];name=m['pyradiomics_name'];val=values.get(name)
        reason='';error=''
        if ref['consensus_value'] in ['NA','','NaN']:
            status='not mappable';reason='official reference configuration absent' if ref.get('configuration_absent') else 'official consensus value absent'
        elif cfg['expressibility']=='not expressible':
            status='not computed';reason=cfg['parameter_differences']
        elif ref['feature_tag']=='stat_kurt':
            status='not mappable';reason='PyRadiomics kurtosis differs from IBSI excess kurtosis; raw value retained; no subtraction'
        elif val is None:
            status='not computed';reason='feature absent from enabled PyRadiomics firstorder class'
        else:
            error=abs(float(val)-float(ref['consensus_value']))
            status='within tolerance' if error<=float(ref['tolerance']) else 'outside tolerance'
        rows.append(dict(record_type='official reference',phase=2,configuration=cfg['configuration'],expressibility=cfg['expressibility'],variant=variant,
                         parent_variant=parent,changed_factor=factor,comparison_scope=scope,feature_family='intensity statistics',
                         feature=ref['feature'],ibsi_id=m['ibsi_id'],feature_tag=ref['feature_tag'],pyradiomics_name=name,
                         value=val if val is not None else '',reference=ref['consensus_value'],tolerance=ref['tolerance'],
                         absolute_difference=error,status=status,reason=reason))
    return rows

def simple_features(im,mask,ext):
    bb,corrected=io.checkMask(im,mask,**ext.settings)
    if corrected is not None:mask=corrected
    cropped,cmask=io.cropToTumorMask(im,mask,bb)
    vals=ext.computeFeatures(cropped,cmask,'diagnostic',**ext.settings)
    return {k.split('_',2)[2]:float(v) for k,v in vals.items()}

def prepared(ext,im,mask,pad=None):
    options=dict(ext.settings)
    if pad is not None:options['padDistance']=pad
    pi,pm=ext.loadImage(im,mask,**options)
    return pi,pm

def phase2(configs):
    refs=list(csv.DictReader((DATA/'theibsi/ibsi_2_reference_data/reference_feature_values/reference_values.csv').open(),delimiter=';'))
    assert len(refs)==324
    assert sum(bool(r['consensus_value']) for r in refs)==323
    # The paper assessed 396 pairs. The official CSV has 324 rows (323
    # consensus values); all 72 Riesz pairs are absent. Retain explicit gaps.
    templates=[r for r in refs if r['filter_id']=='1.A']
    for cid in ['10.A','10.B','11.A','11.B']:
        refs += [dict(r,filter_id=cid,consensus_value='',tolerance='',configuration_absent=True) for r in templates]
    assert len(refs)==396
    im=read(DATA/'theibsi/data_sets/ibsi_2_ct_radiomics_phantom/nifti/image/phantom.nii.gz')
    im=image(np.rint(array(im)).astype(np.float32),im)
    mask=read(DATA/'theibsi/data_sets/ibsi_2_ct_radiomics_phantom/nifti/mask/mask.nii.gz')
    mask=image(np.rint(array(mask)).astype(np.uint8),mask)
    mapping=feature_mapping();rows=[];adapters=[];runs={};failures=[]
    for cfg in [c for c in configs if c['phase']==2]:
        cid=cfg['configuration'];cr=[r for r in refs if r['filter_id']==cid]
        if cfg['expressibility']=='not expressible':
            rows+=compare_features(cfg,'native',{},cr,mapping);continue
        native_complete=False
        try:
            ext=make_extractor(settings(cfg),cid)
            result=ext.execute(im,mask)
            dump(OUT/'raw'/(cid+'.json'),{k:float(v) if isinstance(v,(float,np.number)) else v for k,v in result.items()})
            prefix=('original' if cfg['native_operation']=='none' else
                    'log-sigma-1-5-mm-3D' if cfg['native_operation']=='LoG' else
                    ('wavelet-' if int(cfg['level'])==1 else 'wavelet2-')+cfg['subband'])
            values={k[len(prefix+'_firstorder_'):]:float(v) for k,v in result.items() if k.startswith(prefix+'_firstorder_')}
            assert values,(cid,prefix)
            scope='native; filter settings match' if cfg['expressibility']=='expressible' else 'native partial configuration; diagnostic comparison'
            rows+=compare_features(cfg,'native',values,cr,mapping,scope=scope)
            native_complete=True
            mapped_names={mapping[r['feature_tag']]['pyradiomics_name'] for r in cr}
            for name,val in values.items():
                if name in mapped_names:continue
                rows.append(dict(record_type='extra native output',phase=2,configuration=cid,expressibility=cfg['expressibility'],
                                 variant='native',parent_variant='',changed_factor='none',comparison_scope=scope,
                                 feature_family='intensity statistics',feature=name,ibsi_id='',feature_tag='',pyradiomics_name=name,
                                 value=val,reference='',tolerance='',absolute_difference='',status='not mappable',
                                 reason='no IBSI-2 phase-2 reference for this feature definition'))
            pi,pm=prepared(ext,im,mask)
            runs[cid]=dict(input=describe(im,mask),prepared=describe(pi,pm),settings=ext.settings,
                           enabled_image_types=ext.enabledImagetypes,selected_prefix=prefix)
            resampling=dict(performed=cid.endswith('B'),grid='native origin-aligned ROI lattice' if cid.endswith('B') else 'original image',
                            ct_source_image=describe(im),mask_source_image=describe(mask,mask),
                            ct_source_edge_padding_voxels=[0,0,0],ct_source_edge_extension='none',
                            ct_outside_value_hu=0.,ct_interpolator='cubic B-spline' if cid.endswith('B') else 'not applied',
                            mask_interpolator='nearest neighbour' if cid.endswith('B') else 'not applied',
                            mask_outside_value=0.,mask_threshold='not applied',post_interpolation_ct_rounding=False)
            def geometry_record(parent,factor,ci,cm,intensity,response,resampling):
                bb,_=io.checkMask(response,intensity,**ext.settings)
                cropped,cropped_mask=io.cropToTumorMask(response,intensity,bb)
                return dict(parent_variant=parent,changed_factor=factor,
                            image=describe(ci,cm),morphology_mask=describe(cm,cm),
                            intensity_mask=describe(intensity,intensity),response_image=describe(response,intensity),
                            feature_image=describe(cropped,cropped_mask),feature_mask=describe(cropped_mask,cropped_mask),
                            resampling=resampling)
            native_intensity=io.resegmentMask(pi,pm,**ext.settings)
            runs[cid]['variants']={'native':geometry_record('','none',pi,pm,native_intensity,native_map(pi,cfg),resampling)}
            if cid in ['1.B','3.B','6.A','6.B','7.A','7.B']:
                # Common, fixed one-factor sequences; no per-feature settings.
                wide,wm=prepared(ext,im,mask,pad=40)
                bb,_=io.checkMask(wide,wm,**ext.settings)
                wide,wm=io.cropToTumorMask(wide,wm,bb,padDistance=40)
                def emit(label,parent,factor,ci,cm,transform=None,resampling_changes=None):
                    intensity=io.resegmentMask(ci,cm,**ext.settings)
                    response=transform(ci) if transform else native_map(ci,cfg)
                    vals=simple_features(response,intensity,ext)
                    adapters.extend(compare_features(cfg,label,vals,cr,mapping,parent,factor,'diagnostic adapter; not production'))
                    resampling=dict(runs[cid]['variants'][parent]['resampling'])
                    resampling.update(resampling_changes or {})
                    runs[cid]['variants'][label]=geometry_record(parent,factor,ci,cm,intensity,response,resampling)
                padding_factor=('pre-filter ROI crop margin 5 to 40 voxels' if cid.endswith('B') else
                                'pre-filter extent only: full image to ROI bounding box plus 40 voxels, clipped to image bounds')
                emit('padding40','native',padding_factor,wide,wm)
                if cid.endswith('B'):
                    li=sitk.Resample(im,wide,sitk.Transform(),sitk.sitkLinear,0.,sitk.sitkFloat32)
                    emit('linear_CT_interpolation','padding40','CT interpolation only: cubic B-spline to linear',li,wm,
                         resampling_changes=dict(ct_interpolator='linear'))
                    lm=sitk.Resample(sitk.Cast(mask,sitk.sitkFloat32),wide,sitk.Transform(),sitk.sitkLinear,0.,sitk.sitkFloat32)
                    linear=sitk.Cast(lm>=.5,sitk.sitkUInt8)
                    emit('linear_mask','padding40','mask interpolation only: trilinear and >=0.5',wide,linear,
                         resampling_changes=dict(mask_interpolator='trilinear',mask_threshold='>=0.5'))
                    rounded=image(np.rint(array(wide)).astype(np.float32),wide)
                    emit('integer_HU','linear_mask','post-interpolation CT rounding only',rounded,linear,
                         resampling_changes=dict(post_interpolation_ct_rounding=True))
                    padded_source=sitk.ZeroFluxNeumannPad(im,[3]*3,[3]*3)
                    boundary=sitk.Resample(padded_source,wide,sitk.Transform(),sitk.sitkBSpline,-1000.,sitk.sitkFloat32)
                    boundary=image(np.rint(array(boundary)).astype(np.float32),boundary)
                    emit('interpolation_boundary_handling','integer_HU',
                         'interpolation boundary handling only: three replicated source edge voxels and outside value -1000 HU',boundary,linear,
                         resampling_changes=dict(ct_source_image=describe(padded_source),ct_source_edge_padding_voxels=[3,3,3],
                                                 ct_source_edge_extension='replicate edge voxels',ct_outside_value_hu=-1000.))
                    centered,centered_mask=spatial_preprocess(im,mask,ext.settings)
                    bb,_=io.checkMask(centered,centered_mask,**ext.settings)
                    centered,centered_mask=io.cropToTumorMask(centered,centered_mask,bb,padDistance=40)
                    emit('centered_grid','interpolation_boundary_handling',
                         'resampling grid alignment and extent only; same padded source, outside value, B-spline, linear mask and HU rounding',centered,centered_mask,
                         resampling_changes=dict(grid='full-image-centred IBSI lattice; ROI bounding box plus 40 voxels retained'))
                    current,cm=centered,centered_mask;parent='centered_grid'
                else:
                    current,cm=wide,wm;parent='padding40'
                if cfg['native_operation']=='LoG':
                    sigma=float(cfg['sigma_mm'])
                    emit('scale_normalisation_off',parent,'LoG sigma^2 normalisation only',current,cm,
                         lambda x:image(array(native_map(x,cfg))/sigma**2,x))
                    # Native boundary extension has no configurable setter;
                    # a fixed, sufficiently wide mirror adapter isolates it.
                    emit('mirror_extension','scale_normalisation_off','boundary extension only',current,cm,
                         lambda x:image(array(padded_map(x,cfg,'symmetric'))/sigma**2,x))
                    emit('recursive_kernel_cutoff4','mirror_extension','finite cutoff 4 sigma only; native impulse coefficients retained',current,cm,
                         lambda x:truncated_recursive_log(x,sigma,'symmetric'))
                    emit('finite_sampled_kernel','recursive_kernel_cutoff4','kernel discretisation only: recursive impulse coefficients to sampled Gaussian derivatives',current,cm,
                         lambda x:sampled_log(x,sigma))
                elif cfg['native_operation']=='Wavelet':
                    emit('mirror_extension',parent,'boundary extension only',current,cm,
                         lambda x:padded_map(x,cfg,'symmetric'))
                    emit('orientation_pooling','mirror_extension','right-angle orientation averaging only',current,cm,
                         lambda x:pooled(x,cfg,'symmetric'))
                    if int(cfg['level'])==2:
                        emit('atrous_level2','orientation_pooling','level-2 kernel dilation only',current,cm,
                             lambda x:pooled(x,cfg,'symmetric',proper_level=True))
            print('Phase 2 '+cid+' complete',flush=True)
        except Exception as exc:
            failures.append(dict(configuration=cid,utc=now(),error=repr(exc)))
            if not native_complete:
                rows+=compare_features(cfg,'execution_failed',{},cr,mapping)
    # Production configuration is extracted unchanged, with no arbitrary CT
    # resegmentation window imposed on this public object.
    production=yaml.safe_load((REPO/'rtpipeline/radiomics_params.yaml').read_text())
    prodext=make_extractor(production,'production')
    prod=prodext.execute(im,mask)
    dump(OUT/'raw/production.json',{k:float(v) if isinstance(v,(float,np.number)) else v for k,v in prod.items()})
    prodrows=[]
    for key,val in prod.items():
        if not key.startswith(('log-sigma-','wavelet-')):continue
        signal,family,feature=key.split('_',2)
        prodrows.append(dict(signal=signal,feature_family=family,feature=feature,value=float(val),status='not mappable',
                             reason='no exact phase-2 reference configuration for this production signal; textures and histogram statistics not benchmarked by IBSI 2 phase 2'))
    table(OUT/'production_feature_coverage.csv',prodrows)
    table(OUT/'feature_conformance.csv',rows)
    table(OUT/'feature_attribution.csv',adapters)
    dump(OUT/'extraction_runs.json',runs)
    dump(OUT/'execution_failures.json',failures)
    return rows,adapters,prodrows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','phase1','phase2','finalize'],default='all')
    args=parser.parse_args()
    start=now();OUT.mkdir(exist_ok=True)
    configs=all_configurations()
    reference_csv=DATA/'theibsi/ibsi_2_reference_data/reference_feature_values/reference_values.csv'
    reference_rows=list(csv.DictReader(reference_csv.open(),delimiter=';')) if reference_csv.exists() else []
    for cfg in configs:
        cid=cfg['configuration']
        if cfg['phase']==1:
            relative='theibsi/ibsi_2_reference_data/reference_response_maps/'+cid.replace('.','_')+'-ValidCRM.nii'
            available=(DATA/relative).exists()
            cfg['reference_object']=relative
            cfg['reference_object_available']=available
            cfg['established_reference_values']='manual states no Riesz consensus' if cid.startswith(('9.','10.')) else 'one response map' if available else 'not available'
        else:
            cfg['reference_object']='theibsi/ibsi_2_reference_data/reference_feature_values/reference_values.csv#'+cid
            cfg['reference_object_available']=any(r['filter_id']==cid for r in reference_rows)
            cfg['established_reference_values']=sum(r['filter_id']==cid and bool(r['consensus_value']) for r in reference_rows)
    table(OUT/'configuration_map.csv',configs)
    manifest=json.loads((DATA/'manifest.json').read_text())
    for f in manifest['files']:
        assert sha(ROOT/f['path'])==f['sha256'],f['path']
    stagepath=OUT/'stage_runs.json'
    stages=json.loads(stagepath.read_text()) if stagepath.exists() else {}
    if args.stage in ['all','phase1']:
        st=now();phase1(configs);stages['phase1']=dict(started_utc=st,finished_utc=now())
        dump(OUT/'stage_phase1.json',stages['phase1'])
    if args.stage in ['all','phase2']:
        st=now();phase2(configs);stages['phase2']=dict(started_utc=st,finished_utc=now())
        dump(OUT/'stage_phase2.json',stages['phase2'])
    # Separate stage invocations can run independently. Retain each stage's
    # newest completed timestamp when assembling the common receipt.
    candidates=[json.loads(stagepath.read_text())] if stagepath.exists() else []
    for name in ['phase1','phase2']:
        p=OUT/('stage_'+name+'.json')
        if p.exists():candidates.append({name:json.loads(p.read_text())})
    for candidate in candidates:
        for name,value in candidate.items():
            if name not in stages or value['finished_utc']>stages[name]['finished_utc']:
                stages[name]=value
    dump(stagepath,stages)
    if args.stage in ['phase1','phase2']:
        return
    assert 'phase1' in stages and 'phase2' in stages
    inputs=[REPO/'rtpipeline/radiomics.py',REPO/'rtpipeline/radiomics_params.yaml',
            REPO/'rtpipeline/ibsi_preprocessing.py',ROOT.parent/'ibsi/results/feature_mapping.csv']
    inputs += [Path(PYRAD)/p for p in ['imageoperations.py','featureextractor.py','firstorder.py','__init__.py']]
    publisher=json.loads((DATA/'publisher_access_evidence.json').read_text())
    retrieval_failures=list(manifest['failures'])
    if not any(f['url']==publisher['url'] for f in retrieval_failures):
        retrieval_failures.append(dict(url=publisher['url'],utc=publisher['utc'],error='\n'.join(publisher['access_message']),
                                      branch='publisher full text and supplementary material stopped'))
    receipt=dict(started_utc=min(s['started_utc'] for s in stages.values()),finished_utc=now(),stage_runs=stages,python=sys.version,platform=platform.platform(),
                 environment=dict(PYTHONNOUSERSITE=os.environ.get('PYTHONNOUSERSITE'),isolated=sys.flags.isolated,
                                  dont_write_bytecode=sys.dont_write_bytecode,threads=1),
                 versions={p:importlib.metadata.version(p) for p in ['pyradiomics','numpy','PyWavelets','SimpleITK','scipy','PyYAML']},
                 source_repositories=manifest['repositories'],source_files=manifest['files'],retrieval_failures=retrieval_failures,
                 publisher_access=publisher,
                 available_native_image_types=radiomics.getImageTypes(),
                 code_inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],
                 source_files_treated_as_data=True,public_phantoms_only=True,
                 native_construction='rtpipeline.radiomics._extractor; unchanged PyRadiomics image operations',
                 phase1_rule='all absolute voxel differences <= 0.01*(reference maximum-reference minimum); no rounding',
                 phase2_rule='absolute feature error <= official CSV tolerance; no rounding/truncation',
                 scripts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').glob('*.py'))],
                 outputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ['.log'] and p.name!='run_receipt.json'])
    dump(OUT/'run_receipt.json',receipt)
    print('Benchmark complete '+now(),flush=True)

if __name__=='__main__': main()
