"""Independent standard-library audit of the long table and every summary number."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,math,statistics,hashlib,re,sys
from collections import defaultdict,Counter

sys.dont_write_bytecode=True
REPO_DVH=Path(__file__).resolve().parents[2]/'rtpipeline'/'dvh.py'
EXPECTED_SETS={'processing_path_size':['processing_path','size_class','metric'],'path_size':['path','size_class','metric'],'path_size_spacing_rotation':['path','size_class','ct_mm','dose_mm','rotation','metric'],'shape_size_field_spacing':['path','shape','volume_cc','ct_mm','dose_mm','rotation','field','metric']}
RAW_REQUIRED=('phantom.json','masks.json','input_sha256.json','rtpipeline_metrics.csv','dvh_qc.json','case_metadata.json')
NATIVE_KEYS={'Dmean_Gy':'DmeanGy','Dmin_Gy':'DminGy','Dmax_Gy':'DmaxGy','D98_Gy':'D98Gy','D95_Gy':'D95Gy','D50_Gy':'D50Gy','D2_Gy':'D2Gy','D2cc_Gy':'D2ccGy','D0.03cc_Gy':'D0.03ccGy','V95Rx_pct':'V95%Rx (%)','V20Gy_cc':'V20Gy (cm³)','Volume_cc':'Volume (cm³)','V20Gy_pct':'V20Gy (%)'}
HEX64_RE=re.compile(r'[0-9a-f]{64}')

def _sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def _is_sha256_hex(val):
 return isinstance(val,str) and bool(HEX64_RE.fullmatch(val))

def _is_finite_number(val):
 return isinstance(val,(int,float)) and not isinstance(val,bool) and math.isfinite(val)

def _has_terminal_disposition(case_rows,expected_token):
 return bool(case_rows) and all((r.get('unavailable_reason') or '').split(';',1)[0].strip()==expected_token for r in case_rows)

def _expected_input_paths(spec,geometry_cls):
 geoms=[geometry_cls(shape,spec['volume_cc'],tuple(spec['center'])) for shape in spec['shapes']]
 extent=math.ceil((max(g.radius for g in geoms)+6)/15)*15
 ct_count=math.ceil((2*extent+0.01)/spec['ct_mm'])
 paths={f'DICOM/CT/ct_{i:04d}.dcm' for i in range(ct_count)}
 paths|={'DICOM/RTSTRUCT/rs.dcm','DICOM/RTPLAN/plan.dcm','DICOM/RTDOSE/dose.dcm'}
 paths|={f'DICOM_related/RTRECORD/record_{i}.dcm' for i in range(1,6)}
 paths|={'NIFTI/ct.nii.gz','NIFTI/ct.metadata.json','metadata/case_metadata.json','phantom.json'}
 paths|={f'Segmentation_TotalSegmentator/mask_{shape}.nii.gz' for shape in spec['shapes']}
 return paths,extent

def verify(root,partial=False):
 root=Path(root).resolve();out=root/'results';design=json.loads((root/'DESIGN.json').read_text())
 ext_dir=Path(__file__).resolve().parent
 if str(ext_dir) not in sys.path:sys.path.insert(0,str(ext_dir))
 from analytic import Geometry,metrics as truth
 truth_cache={}
 def read(p):return list(csv.DictReader(p.open()))
 def num(s):return float(s) if s else None
 def same(a,b):return a==b if a is None or b is None else math.isclose(a,b,rel_tol=2e-13,abs_tol=2e-12)
 rows=read(out/'metrics_long.csv');expected={s['group']+'_'+p+'_'+shape:s for s in design['groups'] for p in s['paths'] for shape in s['shapes']}
 planned_groups={s['group']:s for s in design['groups']}
 metrics=set(design['metrics']);bycase=defaultdict(list);errors=[];checks=Counter()
 if not rows:errors.append('empty metrics_long.csv')
 for r in rows:
  bycase[r['case']].append(r)
  ref,val,err,ab,bound=[num(r[k]) for k in ['reference','value','signed_error','absolute_error','boundary']]
  predicted=val-ref if ref is not None and val is not None else None
  if not same(err,predicted) or not same(ab,abs(predicted) if predicted is not None else None):errors.append('error arithmetic '+r['case']+'/'+r['metric'])
  if val is None and not r['unavailable_reason']:errors.append('missing reason '+r['case']+'/'+r['metric'])
  spec=expected.get(r['case'])
  if spec:
   if r['case']!=r['group']+'_'+r['path']+'_'+r['shape']:errors.append('case factor identity '+r['case'])
   for column,factor in [('volume_cc','volume_cc'),('ct_mm','ct_mm'),('dose_mm','dose_mm'),('rotation_deg',None)]:
    target=spec[factor] if factor else spec['rotation']['angle_deg']
    if num(r[column])!=target:errors.append('declared factor '+column+' '+r['case'])
   if r['field']!=spec['field'] or r['rotation']!=spec['rotation']['id'] or json.loads(r['rotation_axis'])!=spec['rotation']['axis']:errors.append('field/rotation identity '+r['case'])
   tkey=(r['shape'],spec['volume_cc'],tuple(spec['center']),spec['field'])
   if tkey not in truth_cache:
    truth_cache[tkey]=truth(Geometry(*tkey[:3]),tkey[3]);truth_cache[tkey]['V20Gy_pct']=100*truth_cache[tkey]['V20Gy_cc']/spec['volume_cc']
   if not same(ref,truth_cache[tkey][r['metric']]):errors.append('analytic reference '+r['case']+'/'+r['metric'])
   descriptive=spec['volume_cc']<1 or r['metric'] in ['Dmin_Gy','Dmax_Gy','D0.03cc_Gy','V20Gy_cc']
   expected_bound=None if descriptive else max(.02*abs(ref),.5) if r['metric'].startswith('D') and ref is not None else 2. if r['metric'].endswith('_pct') else .03*spec['volume_cc'] if r['metric']=='Volume_cc' else None
   if not same(bound,expected_bound):errors.append('boundary '+r['case']+'/'+r['metric'])
   status='unavailable' if predicted is None else 'descriptive' if descriptive else 'beyond_boundary' if abs(predicted)>expected_bound else 'within_boundary'
   if r['status']!=status:errors.append('status '+r['case']+'/'+r['metric'])
  rel=num(r['relative_error']);expected_rel=abs(predicted/ref) if predicted is not None and ref not in (None,0.) else None
  if not same(rel,expected_rel):errors.append('relative error '+r['case']+'/'+r['metric'])
  expected_flag='' if predicted is None or bound is None else str(abs(predicted)>bound)
  if r['beyond_boundary'].lower()!=expected_flag.lower():errors.append('boundary flag '+r['case']+'/'+r['metric'])
  if r['processing_path']!=(r['path'] if r['rotation']=='axial' else 'rotated_'+r['path']):errors.append('processing path '+r['case'])
  checks['row_arithmetic']+=1
 for case in (set(bycase) if partial else set(expected)|set(bycase)):
  if case not in expected or case not in bycase:errors.append('case coverage '+case);continue
  if Counter(r['metric'] for r in bycase[case])!=Counter(metrics):errors.append('metric coverage '+case)
  rr={r['metric']:r for r in bycase[case]};v=num(rr['Volume_cc']['value']);v20=num(rr['V20Gy_cc']['value']);pct=num(rr['V20Gy_pct']['value'])
  if v and v20 is not None and not same(pct,100*v20/v):errors.append('V20 conversion '+case)
  checks['case_coverage']+=1
 observed_groups={expected[c]['group'] for c in bycase if c in expected}
 indexes={}
 for name,cols in EXPECTED_SETS.items():
  d=defaultdict(list)
  for r in rows:d[tuple(r[c] for c in cols)].append(r)
  indexes[name]=d
 def canonical(x):
  try:return str(float(x))
  except ValueError:return x
 canonical_indexes={name:{tuple(canonical(k[i]) if EXPECTED_SETS[name][i] in ['ct_mm','dose_mm','volume_cc'] else k[i] for i in range(len(k))):v for k,v in index.items()} for name,index in indexes.items()}
 expected_summary_keys={(name,key) for name,index in canonical_indexes.items() for key in index}
 seen_summary_keys=set()
 summary=read(out/'summary.csv')
 for s in summary:
  stratum=s.get('stratum')
  if stratum not in EXPECTED_SETS:errors.append('unknown summary stratum '+str(stratum));continue
  cols=EXPECTED_SETS[stratum]
  key=tuple(canonical(s[c]) if c in ['ct_mm','dose_mm','volume_cc'] else s[c] for c in cols)
  full_key=(stratum,key)
  if full_key in seen_summary_keys:errors.append('duplicate summary key '+str(full_key))
  seen_summary_keys.add(full_key)
  idx=canonical_indexes[stratum]
  g=idx.get(key)
  if g is None:errors.append('summary key '+str(key));continue
  valid=[r for r in g if r['absolute_error']];bounded=[r for r in valid if r['boundary']]
  counts=dict(n_total=len(g),n_value=sum(bool(r['value']) for r in g),n_comparable=len(valid),n_missing_value=sum(not r['value'] for r in g),n_reference_unavailable=sum(not r['reference'] for r in g),n_bounded=len(bounded),n_beyond=sum(r['status']=='beyond_boundary' for r in g),n_descriptive=sum(r['status']=='descriptive' for r in g))
  for k,v in counts.items():
   if int(s[k])!=v:errors.append('summary count '+k+str(key))
  abses=[num(r['absolute_error']) for r in valid];signed=[num(r['signed_error']) for r in valid];rel=[num(r['relative_error']) for r in valid if r['relative_error']]
  stats=dict(median_absolute_error=statistics.median(abses) if abses else None,maximum_absolute_error=max(abses) if abses else None,median_signed_error=statistics.median(signed) if signed else None,maximum_absolute_relative_error=max(rel) if rel else None)
  for k,v in stats.items():
   if not same(num(s[k]),v):errors.append('summary statistic '+k+str(key))
  checks['summary_rows']+=1
 missing_summary=expected_summary_keys-seen_summary_keys
 if missing_summary:errors.append(f'missing summary keys ({len(missing_summary)}): {sorted(missing_summary)[:5]}')
 receipts=[json.loads(p.read_text()) for p in sorted(out.glob('shard-*/receipt.json'))]
 digest=_sha(root/'DESIGN.json');mtime=(root/'DESIGN.json').stat().st_mtime
 if not receipts:errors.append('missing shard receipts')
 receipt_groups={};dvh_hashes=set();source_hashes_seen=[];shard_counts=set();seen_shards=set()
 expected_repo_dvh=_sha(REPO_DVH) if REPO_DVH.exists() else None
 for rec in receipts:
  start_ts=datetime.fromisoformat(rec['start_utc']).timestamp()
  if rec.get('design_sha256')!=digest or start_ts<=mtime:errors.append('design timing/hash')
  if (not partial and rec.get('status')!='completed') or rec.get('errors'):errors.append('incomplete/failed receipt')
  if not partial:
   if not rec.get('ended_utc') or datetime.fromisoformat(rec['ended_utc']).timestamp()<start_ts:errors.append('receipt ended_utc invalid')
   if 'shards' in rec and 'shard' in rec:shard_counts.add(rec['shards']);seen_shards.add(rec['shard'])
  dvh_hash=rec.get('dvh_source_sha256')
  if not dvh_hash:errors.append('missing receipt dvh_source_sha256')
  else:
   dvh_hashes.add(dvh_hash)
   if expected_repo_dvh and dvh_hash!=expected_repo_dvh:errors.append('receipt dvh_source_sha256 mismatch')
  if 'source_sha256' in rec:
   if not rec['source_sha256']:errors.append('empty receipt source_sha256')
   else:
    source_hashes_seen.append(rec['source_sha256'])
    if isinstance(rec['source_sha256'],dict) and 'rtpipeline/dvh.py' in rec['source_sha256'] and dvh_hash and rec['source_sha256']['rtpipeline/dvh.py']!=dvh_hash:
     errors.append('receipt source_sha256 inconsistent with dvh_source_sha256')
  for group in rec.get('groups',[]):
   gname=group.get('group')
   if gname in receipt_groups:errors.append('duplicate receipt group '+str(gname))
   receipt_groups[gname]=group
   gstart=datetime.fromisoformat(group['started_utc']).timestamp()
   if gstart<=mtime:errors.append('group predates design')
   if gname not in planned_groups:errors.append('unexpected receipt group '+str(gname));continue
   expected_status='refused' if planned_groups[gname]['rotation']['angle_deg']>1 else 'completed'
   if group.get('status')!=expected_status or not group.get('ended_utc') or datetime.fromisoformat(group['ended_utc']).timestamp()<gstart:
    errors.append('receipt group status/timing '+gname)
 if len(dvh_hashes)>1 or (len(source_hashes_seen)>1 and any(s!=source_hashes_seen[0] for s in source_hashes_seen)):
  errors.append('inconsistent source hashes across receipts')
 if not partial and shard_counts and (len(shard_counts)!=1 or seen_shards!=set(range(next(iter(shard_counts))))):
  errors.append('incomplete shard receipt set')
 required_receipt_groups=observed_groups if partial else set(planned_groups)
 missing_receipt_groups=required_receipt_groups-set(receipt_groups)
 if missing_receipt_groups:errors.append('missing receipt groups: '+','.join(sorted(missing_receipt_groups)[:5]))
 for case,g in bycase.items():
  spec=expected.get(case)
  if spec and spec['rotation']['angle_deg']>1:
   if any(r['value'] for r in g) or not _has_terminal_disposition(g,'rotated_dose_rotation_exceeds_1_degree'):errors.append('refusal case '+case)
 checks['design_timing']=len(receipts)
 raw_dirs={}
 for p in sorted(out.glob('shard-*/raw/*')):
  if not p.is_dir():continue
  if p.name not in planned_groups:
   errors.append('unexpected raw group dir '+p.name)
  elif p.name not in raw_dirs:
   raw_dirs[p.name]=p
 for p in sorted((out/'cached_execution').glob('shard-*/raw/*')):
  if p.is_dir() and p.name in planned_groups and p.name not in raw_dirs:raw_dirs[p.name]=p
 required_raw_groups=observed_groups if partial else set(planned_groups)
 raw_file_hashes={}
 unprinted_dispositions=Counter()
 for group in sorted(required_raw_groups):
  rdir=raw_dirs.get(group)
  if rdir is None:errors.append('missing raw group dir '+group);continue
  spec=planned_groups[group]
  refused=spec['rotation']['angle_deg']>1
  req_files=list(RAW_REQUIRED)+([] if refused else ['dvh_curves.json'])
  missing_files=[f for f in req_files if not (rdir/f).exists()]
  if missing_files:errors.append('missing raw files '+group+':'+','.join(missing_files));continue
  checks['raw_groups']+=1
  group_raw_hashes={p.name:_sha(p) for p in sorted(rdir.iterdir()) if p.is_file()}
  raw_file_hashes[group]=group_raw_hashes
  checks['raw_file_digests']+=len(group_raw_hashes)
  group_rec=receipt_groups.get(group,{})
  if 'raw_sha256' in group_rec:
   rec_raw=group_rec['raw_sha256']
   if (not isinstance(rec_raw,dict) or not set(req_files).issubset(rec_raw)
       or any(not _is_sha256_hex(d) or group_raw_hashes.get(fn)!=d for fn,d in rec_raw.items())):
    errors.append('raw file receipt digest mismatch '+group)
  expected_inputs,expected_extent=_expected_input_paths(spec,Geometry)
  try:
   phantom=json.loads((rdir/'phantom.json').read_text())
   if not isinstance(phantom,dict) or not phantom or phantom.get('group')!=group:
    errors.append('invalid phantom '+group)
   else:
    for k,v in spec.items():
     if phantom.get(k)!=v:
      errors.append(f'phantom spec mismatch {group}/{k}')
      break
    if 'extent_mm' in phantom and phantom['extent_mm']!=expected_extent:
     errors.append('phantom extent mismatch '+group)
  except Exception:errors.append('corrupt phantom '+group)
  try:
   case_meta=json.loads((rdir/'case_metadata.json').read_text())
   contract=case_meta.get('course_contract') if isinstance(case_meta,dict) else None
   if (not isinstance(case_meta,dict) or not case_meta
       or case_meta.get('course_id')!=group
       or ('course_contract' in case_meta and (not isinstance(contract,dict) or contract.get('course_id')!=group))):
    errors.append('invalid case_metadata '+group)
  except Exception:errors.append('corrupt case_metadata '+group)
  ih_map=None
  try:
   ih=json.loads((rdir/'input_sha256.json').read_text())
   if (not isinstance(ih,list)
       or any(not isinstance(e,dict) or not isinstance(e.get('path'),str) or not e['path'] or not _is_sha256_hex(e.get('sha256')) for e in ih)
       or len({e['path'] for e in ih})!=len(ih)
       or {e['path'] for e in ih}!=expected_inputs):
    errors.append('invalid input_sha256 manifest '+group)
   else:
    ih_map={e['path']:e['sha256'] for e in ih}
    if ih_map.get('phantom.json')!=group_raw_hashes.get('phantom.json'):
     errors.append('input_sha256 phantom digest mismatch '+group)
    if ih_map.get('metadata/case_metadata.json')!=group_raw_hashes.get('case_metadata.json'):
     errors.append('input_sha256 case_metadata digest mismatch '+group)
  except Exception:errors.append('corrupt input_sha256 '+group)
  try:
   qc=json.loads((rdir/'dvh_qc.json').read_text())
   if not isinstance(qc,dict) or qc.get('status')!='ok' or (refused and qc.get('dose_grid_resampling',{}).get('status')!='refused'):
    errors.append('invalid dvh_qc status '+group)
  except Exception:errors.append('corrupt dvh_qc '+group)
  curve_roi_names=[]
  if not refused:
   try:
    curves=json.loads((rdir/'dvh_curves.json').read_text())
    points=curves.get('points') if isinstance(curves,dict) else None
    if not isinstance(points,list) or not points:
     errors.append('empty dvh_curves '+group)
    else:
     seen_curve_rois=set()
     for idx,curve in enumerate(points):
      data=curve.get('data') if isinstance(curve,dict) else None
      roi_name=curve.get('roi_name') if isinstance(curve,dict) else None
      if (not isinstance(data,list) or len(data)<2
          or any(not isinstance(pt,dict) or not (_is_finite_number(pt.get('x')) and _is_finite_number(pt.get('y'))) for pt in data)):
       errors.append(f'invalid dvh_curves {group}/{idx}')
       break
      if not isinstance(roi_name,str) or not roi_name or roi_name in seen_curve_rois:
       errors.append(f'invalid dvh_curves roi {group}/{idx}')
       break
      seen_curve_rois.add(roi_name)
      curve_roi_names.append(roi_name)
   except Exception:errors.append('corrupt dvh_curves '+group)
  try:
   masks=json.loads((rdir/'masks.json').read_text())
   if not isinstance(masks,dict):
    errors.append('invalid masks '+group);masks={}
  except Exception:
   errors.append('corrupt masks '+group);masks={}
  for shape in spec['shapes']:
   m=masks.get(shape)
   if not isinstance(m,dict) or not _is_sha256_hex(m.get('sha256')):errors.append('missing mask metadata '+group+'/'+shape);continue
   if ih_map is not None and ih_map.get(f'Segmentation_TotalSegmentator/mask_{shape}.nii.gz')!=m['sha256']:
    errors.append(f'input_sha256 mask digest mismatch {group}/{shape}')
   case=group+'_nifti_mask_'+shape
   if case in bycase:
    measured=next((num(r['value']) for r in bycase[case] if r['metric']=='Volume_cc'),None)
    if not refused and measured is not None and not same(measured,m['volume_cc']):errors.append('mask ingestion volume '+case)
   checks['mask_volume_provenance']+=1
  planned_group_cases={group+'_'+p+'_'+shape for p in spec['paths'] for shape in spec['shapes']}
  expected_group_cases={c for c in planned_group_cases if (not partial or c in bycase)}
  matched_cases=set()
  try:
   with (rdir/'rtpipeline_metrics.csv').open() as raw_handle:
    raw_reader=csv.DictReader(raw_handle)
    raw_fields=raw_reader.fieldnames or []
    raw_rows=list(raw_reader)
  except Exception:
   raw_fields=[];raw_rows=[];errors.append('corrupt rtpipeline_metrics.csv '+group)
  raw_field_set=set(raw_fields)
  omitted_v20={'V20Gy (cm³)','V20Gy (%)'} if refused else set()
  required_raw_cols={'ROI_OriginalName','Segmentation_Source'}|(set(NATIVE_KEYS.values())-omitted_v20)
  if spec['rotation']['angle_deg'] not in [0.,2.]:
   required_raw_cols.add('dose_grid_resampling_status')
  missing_raw_cols=sorted(required_raw_cols-raw_field_set)
  if missing_raw_cols:
   errors.append('missing raw columns '+group+':'+','.join(missing_raw_cols))
  published_rois=set()
  if 'ROI_OriginalName' in raw_field_set:
   for rr in raw_rows:
    name=rr.get('ROI_OriginalName','')
    if not name:
     errors.append('blank ROI_OriginalName in rtpipeline_metrics.csv '+group);continue
    path='nifti_mask' if name.startswith('mask_') else 'rtstruct';shape=name[5:] if path=='nifti_mask' else name
    case=group+'_'+path+'_'+shape
    if case not in planned_group_cases:
     errors.append('unexpected raw metric row '+case);continue
    if case in matched_cases:
     errors.append('duplicate raw metric row '+case);continue
    matched_cases.add(case)
    published_rois.add(name)
    if case in bycase:
     long_values={r['metric']:num(r['value']) for r in bycase[case]}
     for metric,column in NATIVE_KEYS.items():
      if column in raw_field_set and metric in long_values and not same(long_values[metric],num(rr.get(column,''))):errors.append('published native value '+case+'/'+metric)
      checks['published_native_values']+=1
    elif not partial:
     errors.append('missing long case for raw metric row '+case)
    expected_producer='NIfTI_Direct' if name.startswith('mask_') else 'Manual'
    if rr.get('Segmentation_Source')!=expected_producer:
     errors.append(('NIfTI' if name.startswith('mask_') else 'RTSTRUCT')+' producer provenance '+group)
    if spec['rotation']['angle_deg'] not in [0.,2.] and rr.get('dose_grid_resampling_status')!='rotated_dose_resampled_to_axis_aligned_grid':errors.append('rotated producer provenance '+group)
  if not refused:
   for roi_name in curve_roi_names:
    if roi_name not in published_rois:
     errors.append(f'unmatched dvh_curves roi {group}/{roi_name}')
  for case in sorted(expected_group_cases-matched_cases):
   case_rows=bycase.get(case,[])
   has_native=any(num(r['value']) is not None for r in case_rows)
   all_unavailable=bool(case_rows) and all(r['status']=='unavailable' for r in case_rows)
   expected_disposition='rotated_dose_rotation_exceeds_1_degree' if refused else 'no_published_ROI'
   if has_native or not all_unavailable or not _has_terminal_disposition(case_rows,expected_disposition):
    errors.append('raw metric case coverage '+case)
   else:
    unprinted_dispositions[expected_disposition]+=1
 checks['unprinted_raw_cases_reconciled']=sum(unprinted_dispositions.values())
 checks['receipt_design_hashes']=len(receipts)
 is_complete=(not errors and not partial and len(bycase)==len(expected) and bool(receipts) and set(receipt_groups)==set(planned_groups) and seen_summary_keys==expected_summary_keys and all(r.get('status')=='completed' for r in receipts))
 report={'checked_utc':datetime.now(timezone.utc).isoformat(),'passed':not errors,'complete':is_complete,'partial_audit':partial,'expected_cases':len(expected),'observed_cases':len(bycase),'rows':len(rows),'checks':dict(checks),'unprinted_raw_case_dispositions':dict(unprinted_dispositions),'errors':errors,'design_sha256':digest,'long_table_sha256':_sha(out/'metrics_long.csv'),'summary_sha256':_sha(out/'summary.csv'),'raw_file_sha256':raw_file_hashes}
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 return report

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parent)
 parser.add_argument('--partial',action='store_true')
 args=parser.parse_args()
 report=verify(args.root,partial=args.partial)
 print(json.dumps(report))
 raise SystemExit(bool(report['errors']))
