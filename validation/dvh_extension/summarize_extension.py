"""Rebuild all tables from the per-case long table, retaining missingness."""
from pathlib import Path
import argparse,json,hashlib
from datetime import datetime,timezone
import pandas as pd


def fmt(x):return '—' if pd.isna(x) else f'{x:.8g}'

def main(root):
    out=root/'results';design=json.loads((root/'DESIGN.json').read_text())
    paths=sorted((out/'cached_execution').glob('shard-*/metrics_long.csv'))+sorted(out.glob('shard-*/metrics_long.csv'))
    if not paths:return
    frames=[pd.read_csv(p,keep_default_na=True,float_precision="round_trip") for p in paths]
    df=pd.concat(frames,ignore_index=True).sort_values(['case','metric'])
    df['processing_path']=df.apply(lambda r:r.path if r.rotation=='axial' else 'rotated_'+r.path,axis=1)
    if df.duplicated(['case','metric']).any():
        for key,g in df[df.duplicated(['case','metric'],keep=False)].groupby(['case','metric']):
            for col in ['reference','value','signed_error','absolute_error','boundary']:
                vals=g[col].dropna()
                if len(vals) and vals.max()-vals.min()>1e-12:raise ValueError('Checkpoint value mismatch '+str(key))
        df=df.drop_duplicates(['case','metric'],keep='last')
    # Causes identify the measured production mechanism, without assigning an
    # unmeasured fraction of error to interpolation versus voxelization.
    def cause(r):
        if pd.notna(r.cause):return r.cause
        if r.status=='beyond_boundary':
            if r.metric=='Volume_cc':
                return ('binary CT voxel-centre mask volume; rtpipeline/dvh.py:3451,3504-3528,2298-2299' if r.path=='nifti_mask' else 'finite contour/slab and dose-lattice rasterization; rtpipeline/dvh.py:3941-3944; dvh_rotation.py:296-306' if r.rotation!='axial' else 'finite contour/slab and dose-lattice rasterization; rtpipeline/dvh.py:3939-3944')
            if r.metric.endswith('_pct'):
                return ('sampled dose distribution and 0.01-Gy cumulative histogram threshold; rtpipeline/dvh.py:2307-2330,2364-2367' if r.path=='nifti_mask' else 'rasterized dose distribution and cumulative histogram threshold; rtpipeline/dvh.py:2158-2163,2209-2213; rotated sampling dvh_rotation.py:296-306' if r.rotation!='axial' else 'rasterized dose distribution and cumulative histogram threshold; rtpipeline/dvh.py:2158-2163,2209-2213')
            return 'sampled/voxelized shape and binned dose endpoint; rtpipeline/dvh.py:2130-2187,2282-2349'
        if r.status=='unavailable':
            if r.reference_unavailable_reason=='analytic_volume_below_request':return 'continuous structure below absolute-volume endpoint; analytic.py:125-127'
            if str(r.unavailable_reason).startswith('roi_below'):return 'measured structure below absolute-volume endpoint; rtpipeline/dvh.py:2079-2091'
        return ''
    native_tables={}
    raw_paths={p.name:p for base in [out/'cached_execution',out] for p in base.glob('shard-*/raw/*') if p.is_dir()}
    import csv
    for idx,r in df[df.value.isna()].iterrows():
        if r.group not in native_tables:
            raw=raw_paths[r.group]
            if not raw.exists():raw=next(out.glob('shard-*/raw/'+r.group))
            with (raw/'rtpipeline_metrics.csv').open() as handle:
                native_tables[r.group]={row['ROI_OriginalName']:row for row in csv.DictReader(handle) if 'ROI_OriginalName' in row}
        name=r['shape'] if r.path=='rtstruct' else 'mask_'+r['shape']
        native=native_tables[r.group].get(name,{})
        if str(r.unavailable_reason).lower() in {'nan','none','', 'no_published_value'}:
            status_column={'D2cc_Gy':'D2cc_status','D0.03cc_Gy':'D0.03cc_status'}.get(r.metric,'')
            reason=native.get('dvh_computation_error') or native.get(status_column) or native.get('dose_metric_status') or 'no_published_value'
            df.at[idx,'unavailable_reason']=reason
    df['cause']=df.apply(cause,axis=1)
    df.to_csv(out/'metrics_long.csv',index=False,float_format='%.17g')
    components=[]
    raw_paths={p.name:p for base in [out/'cached_execution',out] for p in base.glob('shard-*/raw/*') if p.is_dir()}
    artifacts={}
    for r in df[df.metric=='Volume_cc'].itertuples():
        if r.group not in artifacts:
            raw=raw_paths[r.group]
            if not raw.exists():raw=next(out.glob('shard-*/raw/'+r.group))
            artifacts[r.group]=(json.loads((raw/'phantom.json').read_text()),json.loads((raw/'masks.json').read_text()))
        manifest,masks=artifacts[r.group]
        roi=next(item for item in manifest['rois'] if item['geometry']['shape']==r.shape)
        slab=roi['contour_slab_volume_cc']
        mask=masks[r.shape]
        sampling=slab if r.path=='rtstruct' else mask['volume_cc']
        components.append(dict(case=r.case,path=r.path,shape=r.shape,size_class=r.size_class,volume_cc=r.volume_cc,ct_mm=r.ct_mm,dose_mm=r.dose_mm,rotation=r.rotation,field=r.field,continuous_volume_cc=r.reference,input_contour_slab_or_mask_volume_cc=sampling,input_sampling_error_cc=sampling-r.reference,published_volume_cc=r.value,subsequent_volume_difference_cc=r.value-sampling if pd.notna(r.value) else None,total_error_cc=r.signed_error,contour_planes=roi['contour_planes'] if r.path=='rtstruct' else None,mask_voxels=mask['voxels'] if r.path=='nifti_mask' else None))
    pd.DataFrame(components).to_csv(out/'volume_error_components.csv',index=False,float_format='%.17g')
    summaries=[]
    strata=[('processing_path_size',['processing_path','size_class','metric']),('path_size',['path','size_class','metric']),('path_size_spacing_rotation',['path','size_class','ct_mm','dose_mm','rotation','metric']),('shape_size_field_spacing',['path','shape','volume_cc','ct_mm','dose_mm','rotation','field','metric'])]
    for label,cols in strata:
        for key,g in df.groupby(cols,sort=True,dropna=False):
            valid=g[g.absolute_error.notna()];bounded=valid[valid.boundary.notna()]
            summaries.append(dict(stratum=label,**dict(zip(cols,key)),n_total=len(g),n_value=int(g.value.notna().sum()),n_comparable=len(valid),n_missing_value=int(g.value.isna().sum()),n_reference_unavailable=int(g.reference.isna().sum()),n_bounded=len(bounded),n_beyond=int((g.status=='beyond_boundary').sum()),n_descriptive=int((g.status=='descriptive').sum()),median_absolute_error=valid.absolute_error.median(),maximum_absolute_error=valid.absolute_error.max(),median_signed_error=valid.signed_error.median(),maximum_absolute_relative_error=valid.relative_error.max()))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
    failures=df[df.status=='beyond_boundary'];failures.to_csv(out/'boundary_exceedances.csv',index=False,float_format='%.17g')
    unavailable=df[df.value.isna()|df.reference.isna()];unavailable.to_csv(out/'unavailable.csv',index=False,float_format='%.17g')
    counts=df.groupby(['path','size_class','ct_mm','dose_mm','rotation']).agg(cases=('case','nunique'),values=('value','count'),metric_rows=('metric','size')).reset_index()
    counts.to_csv(out/'counts_by_stratum.csv',index=False)
    lines=['# V1a analytic extension: current results','',f'{df.case.nunique()} observed structure–path–dose cases; {len(df)} metric rows. CT in-plane spacing 2 mm; dose spacing 2.5 mm. Missing values and unavailable reference endpoints are counted separately.','', '| Path | Size (cm³) | Metric | Comparisons/declared observed rows | Median absolute error | Maximum absolute error | Beyond/bounded |','|---|---|---|---:|---:|---:|---:|']
    for r in summary[summary.stratum=='processing_path_size'].itertuples():lines.append(f'| {r.processing_path} | {r.size_class} | {r.metric} | {r.n_comparable}/{r.n_total} | {fmt(r.median_absolute_error)} | {fmt(r.maximum_absolute_error)} | {r.n_beyond}/{r.n_bounded} |')
    lines+=['','Errors for dose endpoints are Gy, volume cm³ and volume percentages percentage points. All endpoints below 1 cm³, Dmin, Dmax and D0.03cc are descriptive. V20Gy_cc is descriptive; V20Gy_pct applies the volume-percentage boundary to the same published output. Thresholds are descriptive software boundaries.','',f'{len(failures)} metric comparisons in {failures.case.nunique()} cases exceed a declared boundary; {int(df.value.isna().sum())} metric rows lack a production value. See boundary_exceedances.csv and unavailable.csv for every case and source location.','', 'The processing-path label distinguishes axial from rotated dose grids; the detailed tables retain each rotation. summary.csv also stratifies every metric by spacing and rotation, and by shape, volume, spacing, rotation and field. Medians weight structure–path–dose cases equally; the <1-cm³ class pools 0.1 and 0.5 cm³, while the other classes each contain one specified size.']
    (out/'CURRENT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    expected={s['group']+'_'+p+'_'+shape for s in design['groups'] for p in s['paths'] for shape in s['shapes']}
    obs=set(df.case)
    case_status=[]
    grouped={case:g for case,g in df.groupby('case')}
    for spec in design['groups']:
        for path in spec['paths']:
            for shape in spec['shapes']:
                case=spec['group']+'_'+path+'_'+shape;g=grouped.get(case)
                values=int(g.value.notna().sum()) if g is not None else 0
                reasons=sorted(set(g.unavailable_reason.dropna().astype(str))) if g is not None else []
                state='measured' if values else 'not_executed' if g is None else 'execution_error' if any('execution_error' in q for q in reasons) else 'production_nonmeasurement'
                case_status.append(dict(case=case,group=spec['group'],path=path,shape=shape,volume_cc=spec['volume_cc'],ct_mm=spec['ct_mm'],dose_mm=spec['dose_mm'],rotation=spec['rotation']['id'],field=spec['field'],state=state,numeric_metric_values=values,reason='; '.join(reasons) if g is not None else 'No completed production execution retained yet; this is not a production refusal.'))
    pd.DataFrame(case_status).to_csv(out/'case_execution_status.csv',index=False)
    receipts=[json.loads(p.read_text()) for p in sorted(out.glob('shard-*/receipt.json'))]
    completeness={'updated_utc':datetime.now(timezone.utc).isoformat(),'design_sha256':hashlib.sha256((root/'DESIGN.json').read_bytes()).hexdigest(),'expected_cases':len(expected),'observed_cases':len(obs),'missing_cases':sorted(expected-obs),'unexpected_cases':sorted(obs-expected),'metric_rows':len(df),'expected_metric_rows':len(expected)*13,'execution_errors':sum((r['errors'] for r in receipts),[]),'receipt_statuses':[r['status'] for r in receipts]}
    (out/'completeness.json').write_text(json.dumps(completeness,indent=2)+'\n')
    print(json.dumps({k:v for k,v in completeness.items() if k not in ['missing_cases','execution_errors']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root.resolve())
