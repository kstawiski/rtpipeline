"""Check source integrity, table completeness and independently recompute errors."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import SimpleITK as sitk
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
DATA=ROOT/'data/public_sources_20261106'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return list(csv.DictReader((OUT/name).open()))
def close(a,b):return np.isclose(float(a),float(b),rtol=1e-12,atol=1e-12)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    receipt=json.loads((OUT/'run_receipt.json').read_text());checks=[]
    for f in receipt['source_files']:
        p=ROOT/f['path'];assert p.resolve().is_relative_to(DATA.resolve());assert sha(p)==f['sha256'],str(p)
    checks.append(f'{len(receipt["source_files"])} downloaded files match pinned SHA256 values')
    for f in receipt['code_inputs']:assert sha(Path(f['path']))==f['sha256'],f['path']
    checks.append('RTpipeline, PyRadiomics and IBSI-1 mapping input hashes match the extraction receipt')
    for p in (ROOT/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
    config=read('configuration_map.csv');assert len(config)==58
    assert len({(r['phase'],r['configuration']) for r in config})==58
    for r in config:
        if r['filter_family']=='LoG':
            assert r['reference_log_derivative_coordinates']=='voxel coordinates; unnormalised second derivatives'
            assert r['native_log_derivative_coordinates']=='physical coordinates before native sigma_mm^2 normalisation'
            assert 'spacing_mm^2 on isotropic grids' in r['parameter_differences']
            if r['phase']=='1':assert 'factor 4' in r['parameter_differences']
            if r['phase']=='2' and r['configuration'].endswith('B'):
                assert 'factor 1' in r['parameter_differences']
        else:
            assert r['reference_log_derivative_coordinates']==r['native_log_derivative_coordinates']=='not applicable'
    for cid,index in [('10.A','(0,2)'),('11.A','(0,2)'),('10.B','(0,2,0)'),('11.B','(0,2,0)')]:
        r=next(r for r in config if r['phase']=='2' and r['configuration']==cid)
        assert r['subband']=='B; Riesz '+index and r['level']=='1'
        assert ' or ' not in r['subband']+r['padding_support']
        if cid.startswith('11.'):assert r['padding_support']=='structure alignment sigma=1 mm'
    checks.append('LoG derivative-coordinate metadata and dimension-specific phase-2 Riesz multi-indices are explicit')
    maps=list((DATA/'theibsi/ibsi_2_reference_data/reference_response_maps').glob('*.nii'));assert len(maps)==33
    checks.append('All 36 phase-1 and 22 phase-2 configurations and 33 published response maps are present')
    response=read('response_map_agreement.csv');assert len([r for r in response if r['variant']=='native'])==36
    computed=[r for r in response if r['variant']=='native' and r['fraction_within_tolerance']!=''];assert len(computed)==8
    for r in computed:
        cid=r['configuration'];p=OUT/'maps'/(cid+'_native.nii.gz')
        x=sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float64)
        y=sitk.GetArrayFromImage(sitk.ReadImage(str(DATA/'theibsi/ibsi_2_reference_data/reference_response_maps'/(cid.replace('.','_')+'-ValidCRM.nii')))).astype(np.float64)
        d=np.abs(x-y);tol=.01*np.ptp(y)
        assert x.shape==y.shape==(64,64,64)
        assert close(r['maximum_absolute_difference'],d.max())
        assert close(r['fraction_within_tolerance'],(d<=tol).mean())
        assert int(r['voxels_within_tolerance'])==int((d<=tol).sum())
        assert close(r['absolute_tolerance'],tol)
    checks.append('All 8 native map errors independently recomputed from retained maps and official references')
    feature=read('feature_conformance.csv');official=[r for r in feature if r['record_type']=='official reference']
    assert len(official)==396
    assert len({(r['configuration'],r['feature_tag']) for r in official})==396
    refs=list(csv.DictReader((DATA/'theibsi/ibsi_2_reference_data/reference_feature_values/reference_values.csv').open(),delimiter=';'))
    lookup={(r['filter_id'],r['feature_tag']):r for r in refs}
    assert sum(r['reference']!='' for r in official)==323
    for r in official:
        ref=lookup.get((r['configuration'],r['feature_tag']))
        if ref:
            assert r['reference']==ref['consensus_value'];assert r['tolerance']==ref['tolerance']
        else:
            assert not r['reference'] and not r['tolerance'] and r['status']=='not mappable'
    checks.append('All 323 established references and tolerances exactly match source strings; 73 missing targets stay blank')
    attribution=read('feature_attribution.csv')
    for r in attribution:
        ref=lookup[(r['configuration'],r['feature_tag'])]
        assert r['reference']==ref['consensus_value'] and r['tolerance']==ref['tolerance']
    for r in feature+attribution:
        if r['status'] in ['within tolerance','outside tolerance']:
            delta=abs(float(r['value'])-float(r['reference']))
            assert close(r['absolute_difference'],delta)
            assert (r['status']=='within tolerance')==(delta<=float(r['tolerance']))
        if r['feature_tag']=='stat_kurt':assert r['status'] not in ['within tolerance','outside tolerance']
    checks.append('Every diagnostic reference and tolerance matches source strings; every classified feature error and decision independently recomputed; kurtosis unadjusted')
    response_variants={(r['configuration'],r['variant']):r for r in response}
    feature_variants={(r['configuration'],r['variant']):r for r in feature+attribution}
    for rows,lookup in [(response,response_variants),(attribution,feature_variants)]:
        for r in rows:
            if not r['parent_variant']:continue
            assert (r['configuration'],r['parent_variant']) in lookup
            seen={r['variant']};parent=r['parent_variant']
            while parent:
                assert parent not in seen
                seen.add(parent);parent=lookup[(r['configuration'],parent)]['parent_variant']
    for cid in ['2.a','2.b']:
        r=response_variants[(cid,'boundary_no_scale_voxel_units')]
        assert r['parent_variant']=='voxel_derivative_units'
        assert r['changed_factor']=='boundary extension only; scale normalisation off and voxel-coordinate derivatives held fixed'
        assert response_variants[(cid,'boundary_and_no_scale')]['parent_variant']=='specified_boundary'
    runs=json.loads((OUT/'extraction_runs.json').read_text())
    for (cid,variant),r in feature_variants.items():
        if r['comparison_scope']!='diagnostic adapter; not production':continue
        record=runs[cid]['variants'][variant]
        assert record['parent_variant']==r['parent_variant'] and record['changed_factor']==r['changed_factor']
        for key in ['image','morphology_mask','intensity_mask','response_image','feature_image','feature_mask']:
            geometry=record[key]
            assert len(geometry['size_xyz'])==len(geometry['spacing_xyz'])==len(geometry['origin_xyz'])==3
            assert len(geometry['direction'])==9
            assert 0<geometry['roi_voxels']<=int(np.prod(geometry['size_xyz']))
        assert record['image']==record['morphology_mask']
        assert record['response_image']==record['intensity_mask']
        assert record['feature_image']==record['feature_mask']
    for cid in ['6.A','7.A']:
        record=runs[cid]['variants']['padding40']
        assert record['parent_variant']=='native'
        assert record['changed_factor']=='pre-filter extent only: full image to ROI bounding box plus 40 voxels, clipped to image bounds'
        assert runs[cid]['variants']['native']['image']['size_xyz']==[204,201,60]
        assert record['image']['size_xyz']==[180,179,60]
        assert not record['resampling']['performed']
    for cid in ['1.B','3.B','6.B','7.B']:
        variants=runs[cid]['variants']
        integer=variants['integer_HU'];boundary=variants['interpolation_boundary_handling'];centered=variants['centered_grid']
        assert boundary['parent_variant']=='integer_HU'
        assert centered['parent_variant']=='interpolation_boundary_handling'
        assert '5 to 40 voxels' in variants['padding40']['changed_factor']
        assert integer['image']==boundary['image']
        assert integer['morphology_mask']==boundary['morphology_mask']
        changed={k for k in integer['resampling'] if integer['resampling'][k]!=boundary['resampling'][k]}
        assert changed=={'ct_source_image','ct_source_edge_padding_voxels','ct_source_edge_extension','ct_outside_value_hu'}
        for record in [boundary,centered]:
            settings=record['resampling']
            assert settings['ct_source_edge_padding_voxels']==[3,3,3]
            assert settings['ct_source_edge_extension']=='replicate edge voxels' and settings['ct_outside_value_hu']==-1000.
            assert settings['ct_interpolator']=='cubic B-spline' and settings['mask_interpolator']=='trilinear'
            assert settings['mask_threshold']=='>=0.5' and settings['post_interpolation_ct_rounding']
        assert {k for k in boundary['resampling'] if boundary['resampling'][k]!=centered['resampling'][k]}=={'grid'}
    checks.append('Diagnostic parents resolve without cycles; A-arm extent and separate B-arm boundary/grid controls retain complete geometry and ROI counts')
    production=read('production_feature_coverage.csv')
    assert len(production)==1116;assert len({r['signal'] for r in production})==12
    assert len({(r['signal'],r['feature_family'],r['feature']) for r in production})==1116
    assert all(r['status']=='not mappable' for r in production)
    assert json.loads((OUT/'execution_failures.json').read_text())==[]
    checks.append('12 production signals and 1116 unique filtered outputs retained; no extraction failures')
    family=[r for r in read('family_summary.csv') if r['scope']=='native']
    assert len(family)==22;assert sum(int(r['assessed_pairs']) for r in family)==396
    for r in family:
        assert sum(int(r[k]) for k in ['within_tolerance','outside_tolerance','not_computed','not_mappable'])==int(r['assessed_pairs'])
    for p in DATA.rglob('*'):
        if p.is_file() and p.stat().st_size>20*1024*1024:
            assert '/data/' in (ROOT/'.gitignore').read_text() or str(p.relative_to(ROOT)) in (ROOT/'.gitignore').read_text()
    checks.append('Family denominators reconcile; public download directory is ignored')
    result=dict(checked_utc=now(),passed=True,checks=checks)
    if args.finalize:
        (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        receipt['finalized_utc']=now()
        receipt['scripts']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').glob('*.py'))]
        receipt['documentation']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [ROOT/'README.md',ROOT/'.gitignore']]
        receipt['publisher_access_evidence_sha256']=sha(DATA/'publisher_access_evidence.json')
        receipt['outputs']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
                            for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.log' and p.name!='run_receipt.json']
        (OUT/'run_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for f in receipt['scripts']+receipt.get('documentation',[])+receipt['outputs']:
        assert sha(ROOT/f['path'])==f['sha256'],f['path']
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
