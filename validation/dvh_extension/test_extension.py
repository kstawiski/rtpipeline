"""Independent quadrature for every declared shape/size/field/offset combination."""
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from analytic import Geometry, metrics
from extension import rotation_matrix, field_values, inside

DESIGN=json.loads(Path(os.environ.get('DVH_DESIGN', Path(__file__).with_name('DESIGN.json'))).read_text())
COMBINATIONS=sorted({(shape,s['volume_cc'],tuple(s['center']),s['field']) for s in DESIGN['groups'] for shape in s['shapes']})

@pytest.mark.parametrize('shape,volume,center,field',COMBINATIONS)
def test_continuous_truth_by_numerical_integration(shape,volume,center,field):
    g=Geometry(shape,volume,center);r,a=g.radius,g.inner_radius
    if field=='radial':
        lo,hi=a,r;breaks=[]
        def weight(t):return 4*math.pi*t*t
        def dose(t):return 30-.15*t
    else:
        lo,hi=-r,r;breaks=[-a,a] if a else []
        def weight(t):
            if shape=='cube':return (2*r)**2
            if shape=='cylinder':return math.pi*r*r
            return math.pi*(r*r-t*t-max(0.,a*a-t*t))
        def dose(t):return 20. if field=='uniform' else 20.+.1*(t+center[2])
    def integ(f,left=lo,right=hi):
        return quad(f,left,right,points=[b for b in breaks if left<b<right],epsabs=1e-7,epsrel=2e-12)[0]
    vol=integ(weight);assert vol/1000==pytest.approx(volume,rel=2e-12)
    out=metrics(g,field)
    assert integ(lambda t:weight(t)*dose(t))/vol==pytest.approx(out['Dmean_Gy'],abs=1e-10)
    low,high=min(dose(lo),dose(hi)),max(dose(lo),dose(hi))
    assert out['Dmin_Gy']==pytest.approx(low,abs=1e-12)
    assert out['Dmax_Gy']==pytest.approx(high,abs=1e-12)
    def survival(d):
        if field=='uniform':return float(d<=20.)
        if d<=low:return 1.
        if d>=high:return 0.
        if field=='linear':return integ(weight,(d-20.)/.1-center[2],hi)/vol
        return integ(weight,lo,(30.-d)/.15)/vol
    def quantile(p):
        if field=='uniform':return 20.
        if p==1:return low
        return brentq(lambda d:survival(d)-p,low,high,xtol=1e-12)
    for p in (98,95,50,2):assert out[f'D{p}_Gy']==pytest.approx(quantile(p/100),abs=1e-10)
    for cc in (2.,.03):
        m=f'D{cc:g}cc_Gy'
        if volume<cc:assert out[m] is None
        else:assert out[m]==pytest.approx(quantile(cc/volume),abs=1e-9)
    assert out['V95Rx_pct']==pytest.approx(100*survival(23.75),abs=1e-9)
    assert out['V20Gy_cc']==pytest.approx(volume*survival(20.),abs=1e-9)

@pytest.mark.parametrize('rotation',[s['rotation'] for s in DESIGN['groups'][:1]]+[
 {'axis':[0,0,1],'angle_deg':.5},{'axis':[1,0,0],'angle_deg':.5},{'axis':[0,0,1],'angle_deg':2.}])
def test_rotation_physical_coordinates_and_field(rotation):
    R=rotation_matrix(rotation)
    assert np.allclose(R.T@R,np.eye(3),atol=1e-14)
    assert np.linalg.det(R)==pytest.approx(1.,abs=1e-14)
    origin=np.array([-15.,-15.,-15.]);local=np.array([[0.,0.,0.],[2.5,5.,7.5],[15.,15.,15.]])
    physical=R@origin+local@R.T
    assert np.allclose(physical,(origin+local)@R.T)
    for field in ('uniform','linear','radial'):
        spec={'field':field,'center':[.37,-.61,.43]}
        vals=field_values(physical,spec)
        expected=[20. if field=='uniform' else 20.+.1*p[2] if field=='linear' else 30.-.15*math.sqrt(sum((q-c)**2 for q,c in zip(p,spec['center']))) for p in physical]
        assert np.allclose(vals,expected,atol=1e-13)

@pytest.mark.parametrize('shape',['sphere','shell','cylinder','cube'])
def test_mask_membership_at_known_points(shape):
    g=Geometry(shape,.1,(.37,-.61,.43));c=np.array(g.center)
    assert not inside(np.array([c+np.array([3*g.radius,0,0])]),g)[0]
    expected=shape!='shell'
    assert bool(inside(np.array([c]),g)[0])==expected
    assert inside(np.array([c+np.array([.95*g.radius,0,0])]),g)[0]

def test_shared_ct_cache_preserves_serialized_inputs(tmp_path, monkeypatch):
    """Cache reuse across field/shape subsets must preserve every CT byte."""
    import hashlib
    from generate_phantoms import generate
    spec=dict(group='cache_uniform',volume_cc=.1,phase='aligned',center=[0.,0.,0.],ct_mm=3.,dose_mm=2.5,field='uniform',shapes=['sphere','cylinder','cube','shell'])
    monkeypatch.delenv('DVH_SHARED_CT_CACHE',raising=False)
    first,_=generate(tmp_path/'uncached',spec,with_contract=False)
    monkeypatch.setenv('DVH_SHARED_CT_CACHE',str(tmp_path/'shared'))
    second,_=generate(tmp_path/'cached-first',spec,with_contract=False)
    other={**spec,'group':'cache_radial','field':'radial','shapes':['sphere','shell']}
    third,_=generate(tmp_path/'cached-reuse',other,with_contract=False)
    def hashes(course):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (course/'DICOM/CT').glob('*.dcm')}
    assert hashes(first)==hashes(second)==hashes(third)
    assert len(hashes(first))==11
    for p in (second/'DICOM/CT').glob('*.dcm'):
        assert p.stat().st_ino==(third/'DICOM/CT'/p.name).stat().st_ino


def _write_summary_csv(out, long_rows):
    import csv, statistics
    from collections import defaultdict
    sets = {'processing_path_size': ['processing_path', 'size_class', 'metric'],
            'path_size': ['path', 'size_class', 'metric'],
            'path_size_spacing_rotation': ['path', 'size_class', 'ct_mm', 'dose_mm', 'rotation', 'metric'],
            'shape_size_field_spacing': ['path', 'shape', 'volume_cc', 'ct_mm', 'dose_mm', 'rotation', 'field', 'metric']}
    sum_cols = ['stratum', 'processing_path', 'size_class', 'metric', 'n_total', 'n_value', 'n_comparable',
                'n_missing_value', 'n_reference_unavailable', 'n_bounded', 'n_beyond', 'n_descriptive',
                'median_absolute_error', 'maximum_absolute_error', 'median_signed_error',
                'maximum_absolute_relative_error', 'path', 'ct_mm', 'dose_mm', 'rotation', 'shape', 'volume_cc', 'field']
    sum_rows = []
    for stratum, cols in sets.items():
        grouped = defaultdict(list)
        for r in long_rows:
            grouped[tuple(r[c] for c in cols)].append(r)
        for key, g in grouped.items():
            valid = [r for r in g if r['absolute_error']]
            bounded = [r for r in valid if r['boundary']]
            abses = [float(r['absolute_error']) for r in valid]
            signed = [float(r['signed_error']) for r in valid]
            rels = [float(r['relative_error']) for r in valid if r['relative_error']]
            row = {c: '' for c in sum_cols}
            row['stratum'] = stratum
            for c, val in zip(cols, key):
                row[c] = val
            row.update({
                'n_total': len(g), 'n_value': sum(bool(r['value']) for r in g), 'n_comparable': len(valid),
                'n_missing_value': sum(not r['value'] for r in g),
                'n_reference_unavailable': sum(not r['reference'] for r in g),
                'n_bounded': len(bounded),
                'n_beyond': sum(r['status'] == 'beyond_boundary' for r in g),
                'n_descriptive': sum(r['status'] == 'descriptive' for r in g),
                'median_absolute_error': repr(statistics.median(abses)) if abses else '',
                'maximum_absolute_error': repr(max(abses)) if abses else '',
                'median_signed_error': repr(statistics.median(signed)) if signed else '',
                'maximum_absolute_relative_error': repr(max(rels)) if rels else '',
            })
            sum_rows.append(row)
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sum_cols)
        w.writeheader()
        w.writerows(sum_rows)
    return sum_cols, sum_rows


def _expected_synthetic_inputs(spec, geometry_cls):
    extent = math.ceil((geometry_cls('sphere', spec['volume_cc'], tuple(spec['center'])).radius + 6) / 15) * 15
    ct_count = math.ceil((2 * extent + 0.01) / spec['ct_mm'])
    paths = [f'DICOM/CT/ct_{i:04d}.dcm' for i in range(ct_count)]
    paths += ['DICOM/RTDOSE/dose.dcm', 'DICOM/RTPLAN/plan.dcm', 'DICOM/RTSTRUCT/rs.dcm']
    paths += [f'DICOM_related/RTRECORD/record_{i}.dcm' for i in range(1, 6)]
    paths += ['NIFTI/ct.metadata.json', 'NIFTI/ct.nii.gz', 'metadata/case_metadata.json', 'phantom.json']
    paths += [f'Segmentation_TotalSegmentator/mask_{shape}.nii.gz' for shape in spec['shapes']]
    return sorted(paths), extent


def _build_synthetic_extension_run(tmp_path):
    import csv, hashlib, json, sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    ext_dir = Path(__file__).resolve().parent
    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    from analytic import Geometry, metrics as truth
    root = tmp_path / 'ext_run'
    out = root / 'results'
    out.mkdir(parents=True)
    metrics_list = ['Dmean_Gy', 'Dmin_Gy', 'Dmax_Gy', 'D98_Gy', 'D95_Gy', 'D50_Gy', 'D2_Gy',
                    'D2cc_Gy', 'D0.03cc_Gy', 'V95Rx_pct', 'V20Gy_cc', 'Volume_cc', 'V20Gy_pct']
    groups = [
        {
            'group': 'v2_aligned_ct1_dg2.5_uniform_axial',
            'volume_cc': 2.0,
            'center': [0.0, 0.0, 0.0],
            'ct_mm': 1.0,
            'dose_mm': 2.5,
            'rotation': {'id': 'axial', 'angle_deg': 0.0, 'axis': [0, 0, 1]},
            'field': 'uniform',
            'paths': ['rtstruct', 'nifti_mask'],
            'shapes': ['sphere'],
        },
        {
            'group': 'v0.5_aligned_ct1_dg2.5_uniform_refusal_z2',
            'volume_cc': 0.5,
            'center': [0.0, 0.0, 0.0],
            'ct_mm': 1.0,
            'dose_mm': 2.5,
            'rotation': {'id': 'refusal_z2', 'angle_deg': 2.0, 'axis': [0, 0, 1]},
            'field': 'uniform',
            'paths': ['rtstruct', 'nifti_mask'],
            'shapes': ['sphere'],
        },
    ]
    design = {'schema_version': 1, 'metrics': metrics_list, 'groups': groups}
    (root / 'DESIGN.json').write_text(json.dumps(design, indent=2) + '\n')
    digest = hashlib.sha256((root / 'DESIGN.json').read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    t_start = (now + timedelta(seconds=10)).isoformat()
    t_end = (now + timedelta(seconds=20)).isoformat()
    repo_dvh = ext_dir.parents[1] / 'rtpipeline' / 'dvh.py'
    dvh_hash = hashlib.sha256(repo_dvh.read_bytes()).hexdigest() if repo_dvh.exists() else hashlib.sha256(b'dvh').hexdigest()
    native_keys = {'Dmean_Gy': 'DmeanGy', 'Dmin_Gy': 'DminGy', 'Dmax_Gy': 'DmaxGy', 'D98_Gy': 'D98Gy',
                   'D95_Gy': 'D95Gy', 'D50_Gy': 'D50Gy', 'D2_Gy': 'D2Gy', 'D2cc_Gy': 'D2ccGy',
                   'D0.03cc_Gy': 'D0.03ccGy', 'V95Rx_pct': 'V95%Rx (%)', 'V20Gy_cc': 'V20Gy (cm³)',
                   'Volume_cc': 'Volume (cm³)', 'V20Gy_pct': 'V20Gy (%)'}
    long_rows = []
    rec_groups = []
    for idx, spec in enumerate(groups):
        gname = spec['group']
        refused = spec['rotation']['angle_deg'] > 1
        size_class = '<1' if spec['volume_cc'] < 1 else '1-20'
        raw = out / 'shard-0' / 'raw' / gname
        raw.mkdir(parents=True)
        expected_paths, extent = _expected_synthetic_inputs(spec, Geometry)
        phantom_payload = dict(spec, extent_mm=extent, circle_vertices=256, ct_inplane_mm=2.0,
                               dose_quantization_Gy=0.0001, reference_dose_Gy=25.0)
        (raw / 'phantom.json').write_text(json.dumps(phantom_payload))
        case_meta_payload = {
            'patient_id': 'SYNTH_DVH',
            'course_id': gname,
            'course_contract': {'version': 4, 'course_id': gname, 'course_key': gname},
        }
        (raw / 'case_metadata.json').write_text(json.dumps(case_meta_payload))
        mask_sha = hashlib.sha256(gname.encode()).hexdigest()
        (raw / 'masks.json').write_text(json.dumps({
            'sphere': {'voxels': 10, 'voxel_volume_cc': spec['volume_cc'] / 10.0,
                       'volume_cc': spec['volume_cc'], 'sha256': mask_sha}
        }))
        file_hashes = {
            'phantom.json': hashlib.sha256((raw / 'phantom.json').read_bytes()).hexdigest(),
            'metadata/case_metadata.json': hashlib.sha256((raw / 'case_metadata.json').read_bytes()).hexdigest(),
            'Segmentation_TotalSegmentator/mask_sphere.nii.gz': mask_sha,
        }
        (raw / 'input_sha256.json').write_text(json.dumps([
            {'path': rel, 'bytes': 4,
             'sha256': file_hashes.get(rel, hashlib.sha256(f'{gname}:{rel}'.encode()).hexdigest())}
            for rel in expected_paths
        ]))
        (raw / 'dvh_qc.json').write_text(json.dumps({
            'status': 'ok',
            'dose_grid_resampling': {'status': 'refused'} if refused else {'status': 'not_needed'},
        }))
        if not refused:
            (raw / 'dvh_curves.json').write_text(json.dumps({
                'points': [
                    {'roi_name': 'sphere', 'data': [{'x': 19.995, 'y': 100.0}, {'x': 20.005, 'y': 0.0}]},
                    {'roi_name': 'mask_sphere', 'data': [{'x': 19.995, 'y': 100.0}, {'x': 20.005, 'y': 0.0}]},
                ]
            }))
        refs = truth(Geometry('sphere', spec['volume_cc'], tuple(spec['center'])), spec['field'])
        refs['V20Gy_pct'] = 100.0 * refs['V20Gy_cc'] / spec['volume_cc']
        raw_csv_rows = []
        for path in spec['paths']:
            proc_path = path if spec['rotation']['id'] == 'axial' else 'rotated_' + path
            case = gname + '_' + path + '_sphere'
            raw_row = {
                'ROI_OriginalName': 'mask_sphere' if path == 'nifti_mask' else 'sphere',
                'Segmentation_Source': 'NIfTI_Direct' if path == 'nifti_mask' else 'Manual',
                'dose_grid_resampling_status': 'refused' if refused else 'not_needed',
            }
            for metric in metrics_list:
                ref = refs[metric]
                val = None if refused else ref
                err = None if (val is None or ref is None) else val - ref
                ab = None if err is None else abs(err)
                rel = None if (err is None or ref in (None, 0.0)) else abs(err / ref)
                descriptive = spec['volume_cc'] < 1 or metric in ['Dmin_Gy', 'Dmax_Gy', 'D0.03cc_Gy', 'V20Gy_cc']
                bound = None if (descriptive or ref is None) else (max(0.02 * abs(ref), 0.5) if metric.startswith('D') else 2.0 if metric.endswith('_pct') else 0.03 * spec['volume_cc'])
                status = 'unavailable' if err is None else 'descriptive' if descriptive else 'beyond_boundary' if ab > bound else 'within_boundary'
                flag = '' if (err is None or bound is None) else str(ab > bound)
                long_rows.append({
                    'case': case, 'group': gname, 'path': path, 'processing_path': proc_path,
                    'shape': 'sphere', 'size_class': size_class, 'volume_cc': repr(spec['volume_cc']),
                    'ct_mm': repr(spec['ct_mm']), 'dose_mm': repr(spec['dose_mm']),
                    'rotation': spec['rotation']['id'], 'rotation_deg': repr(spec['rotation']['angle_deg']),
                    'rotation_axis': json.dumps(spec['rotation']['axis']), 'field': spec['field'],
                    'metric': metric, 'reference': '' if ref is None else repr(ref), 'value': '' if val is None else repr(val),
                    'signed_error': '' if err is None else repr(err), 'absolute_error': '' if ab is None else repr(ab),
                    'relative_error': '' if rel is None else repr(rel), 'boundary': '' if bound is None else repr(bound),
                    'beyond_boundary': flag, 'status': status,
                    'unavailable_reason': 'rotated_dose_rotation_exceeds_1_degree' if refused else '',
                })
                raw_row[native_keys[metric]] = '' if val is None else repr(val)
            raw_csv_rows.append(raw_row)
        raw_cols = ['ROI_OriginalName', 'Segmentation_Source', 'dose_grid_resampling_status', *native_keys.values()]
        with (raw / 'rtpipeline_metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=raw_cols)
            w.writeheader()
            w.writerows(raw_csv_rows)
        rec_groups.append({
            'group': gname,
            'started_utc': (now + timedelta(seconds=11 + idx)).isoformat(),
            'ended_utc': (now + timedelta(seconds=12 + idx)).isoformat(),
            'status': 'refused' if refused else 'completed',
        })
    long_cols = list(long_rows[0].keys())
    with (out / 'metrics_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(long_rows)
    sum_cols, sum_rows = _write_summary_csv(out, long_rows)
    (out / 'shard-0' / 'receipt.json').write_text(json.dumps({
        'start_utc': t_start, 'ended_utc': t_end, 'design_sha256': digest,
        'dvh_source_sha256': dvh_hash, 'source_sha256': {'rtpipeline/dvh.py': dvh_hash},
        'shard': 0, 'shards': 1, 'status': 'completed', 'errors': [], 'groups': rec_groups,
    }, indent=2))
    return root, sum_cols, sum_rows


def _run_verify(root, partial=False):
    import json, subprocess, sys
    from pathlib import Path
    script = Path(__file__).resolve().parent / 'verify_extension.py'
    cmd = [sys.executable, '-B', str(script), str(root)] + (['--partial'] if partial else [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, json.loads((root / 'results' / 'verification.json').read_text())


def test_verify_extension_passes_complete_synthetic_run(tmp_path):
    import hashlib
    root, _, _ = _build_synthetic_extension_run(tmp_path)
    proc, report = _run_verify(root)
    assert proc.returncode == 0
    assert report['passed'] is True
    assert report['complete'] is True
    assert report['checks']['raw_groups'] == 2
    assert report['checks']['summary_rows'] > 0
    raw_dir = root / 'results' / 'shard-0' / 'raw' / 'v2_aligned_ct1_dg2.5_uniform_axial'
    assert report['raw_file_sha256']['v2_aligned_ct1_dg2.5_uniform_axial']['phantom.json'] == hashlib.sha256((raw_dir / 'phantom.json').read_bytes()).hexdigest()
    assert 'dvh_curves.json' in report['raw_file_sha256']['v2_aligned_ct1_dg2.5_uniform_axial']


def test_verify_extension_rejects_header_only_summary_and_missing_receipts(tmp_path):
    import csv, shutil
    root, sum_cols, _ = _build_synthetic_extension_run(tmp_path)
    with (root / 'results' / 'summary.csv').open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=sum_cols).writeheader()
    shutil.rmtree(root / 'results' / 'shard-0')
    proc, report = _run_verify(root)
    assert proc.returncode != 0
    assert report['passed'] is False
    assert report['complete'] is False
    err_text = ' | '.join(report['errors'])
    assert 'missing summary keys' in err_text
    assert 'missing shard receipts' in err_text
    assert 'missing raw group dir' in err_text


def test_verify_extension_rejects_duplicate_or_missing_summary_and_incomplete_raw(tmp_path):
    import csv, json
    root, sum_cols, sum_rows = _build_synthetic_extension_run(tmp_path)
    with (root / 'results' / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sum_cols)
        w.writeheader()
        w.writerows(sum_rows[:-1])
        w.writerow(sum_rows[0])
    proc, report = _run_verify(root)
    assert proc.returncode != 0
    err_text = ' | '.join(report['errors'])
    assert 'duplicate summary key' in err_text
    assert 'missing summary keys' in err_text

    root2, _, _ = _build_synthetic_extension_run(tmp_path / 'sub')
    (root2 / 'results' / 'shard-0' / 'raw' / 'v2_aligned_ct1_dg2.5_uniform_axial' / 'dvh_curves.json').unlink()
    proc2, report2 = _run_verify(root2)
    assert proc2.returncode != 0
    assert 'missing raw files' in ' | '.join(report2['errors'])

    root3, _, _ = _build_synthetic_extension_run(tmp_path / 'sub2')
    rpath = root3 / 'results' / 'shard-0' / 'receipt.json'
    rec = json.loads(rpath.read_text())
    rec['groups'][0]['status'] = 'running'
    rpath.write_text(json.dumps(rec))
    proc3, report3 = _run_verify(root3)
    assert proc3.returncode != 0
    assert 'receipt group status/timing' in ' | '.join(report3['errors'])


def test_verify_extension_rejects_vacuous_phantom_curves_and_input_manifest(tmp_path):
    import json
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'

    # 1. Zero-byte phantom.json
    root_zero, _, _ = _build_synthetic_extension_run(tmp_path / 'zero_phantom')
    (root_zero / 'results' / 'shard-0' / 'raw' / axial_group / 'phantom.json').write_text('')
    proc, report = _run_verify(root_zero)
    assert proc.returncode != 0
    assert report['passed'] is False
    assert report['complete'] is False
    assert 'corrupt phantom' in ' | '.join(report['errors'])

    # 2. Mismatched phantom design parameter
    root_mismatch, _, _ = _build_synthetic_extension_run(tmp_path / 'mismatch_phantom')
    ppath = root_mismatch / 'results' / 'shard-0' / 'raw' / axial_group / 'phantom.json'
    pdata = json.loads(ppath.read_text())
    pdata['ct_mm'] = 3.0
    ppath.write_text(json.dumps(pdata))
    proc, report = _run_verify(root_mismatch)
    assert proc.returncode != 0
    assert 'phantom spec mismatch' in ' | '.join(report['errors'])

    # 3. Empty {"points": []} curves and short single-point curve
    root_empty_curves, _, _ = _build_synthetic_extension_run(tmp_path / 'empty_curves')
    cpath = root_empty_curves / 'results' / 'shard-0' / 'raw' / axial_group / 'dvh_curves.json'
    cpath.write_text(json.dumps({'points': []}))
    proc, report = _run_verify(root_empty_curves)
    assert proc.returncode != 0
    assert report['passed'] is False
    assert report['complete'] is False
    assert 'empty dvh_curves' in ' | '.join(report['errors'])

    cpath.write_text(json.dumps({'points': [{'roi_name': 'sphere', 'data': [{'x': 20.0, 'y': 100.0}]}]}))
    proc, report = _run_verify(root_empty_curves)
    assert proc.returncode != 0
    assert 'invalid dvh_curves' in ' | '.join(report['errors'])

    # 4. Single-entry input manifest and malformed digest
    root_single_ih, _, _ = _build_synthetic_extension_run(tmp_path / 'single_ih')
    ih_path = root_single_ih / 'results' / 'shard-0' / 'raw' / axial_group / 'input_sha256.json'
    ih_entries = json.loads(ih_path.read_text())
    ih_path.write_text(json.dumps(ih_entries[:1]))
    proc, report = _run_verify(root_single_ih)
    assert proc.returncode != 0
    assert 'invalid input_sha256 manifest' in ' | '.join(report['errors'])

    bad_hex = json.loads(json.dumps(ih_entries))
    bad_hex[0]['sha256'] = 'NOT-HEX'
    ih_path.write_text(json.dumps(bad_hex))
    proc, report = _run_verify(root_single_ih)
    assert proc.returncode != 0
    assert 'invalid input_sha256 manifest' in ' | '.join(report['errors'])


def test_verify_extension_reconciles_unprinted_raw_cases_and_rejects_unexplained_omissions(tmp_path):
    import csv, json
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'
    refused_group = 'v0.5_aligned_ct1_dg2.5_uniform_refusal_z2'

    # Valid unprinted cases: no_published_ROI on axial rtstruct + planned rotation refusal on refused nifti_mask
    root_ok, _, _ = _build_synthetic_extension_run(tmp_path / 'reconciled')
    out_ok = root_ok / 'results'
    with (out_ok / 'metrics_long.csv').open() as f:
        reader = csv.DictReader(f)
        long_cols = reader.fieldnames
        long_rows = list(reader)
    for r in long_rows:
        if r['case'] == axial_group + '_rtstruct_sphere':
            for k in ('value', 'signed_error', 'absolute_error', 'relative_error', 'beyond_boundary'):
                r[k] = ''
            r['status'] = 'unavailable'
            r['unavailable_reason'] = 'no_published_ROI; see raw/dvh_qc.json; rtpipeline.dvh._compute_dvh_for_course._calculate_roi'
    with (out_ok / 'metrics_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(long_rows)
    _write_summary_csv(out_ok, long_rows)

    for grp, drop_roi in ((axial_group, 'sphere'), (refused_group, 'mask_sphere')):
        csv_path = out_ok / 'shard-0' / 'raw' / grp / 'rtpipeline_metrics.csv'
        with csv_path.open() as f:
            r = csv.DictReader(f)
            cols = r.fieldnames
            kept = [row for row in r if row['ROI_OriginalName'] != drop_roi]
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(kept)
    cpath_ok = out_ok / 'shard-0' / 'raw' / axial_group / 'dvh_curves.json'
    curves_ok = json.loads(cpath_ok.read_text())
    curves_ok['points'] = [pt for pt in curves_ok['points'] if pt['roi_name'] != 'sphere']
    cpath_ok.write_text(json.dumps(curves_ok))

    proc, report = _run_verify(root_ok)
    assert proc.returncode == 0
    assert report['passed'] is True
    assert report['complete'] is True
    assert report['unprinted_raw_case_dispositions'] == {
        'no_published_ROI': 1,
        'rotated_dose_rotation_exceeds_1_degree': 1,
    }

    # Dropping raw row when native values exist or non-qualifying disposition must fail
    root_bad_val, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_unprinted')
    csv_bad = root_bad_val / 'results' / 'shard-0' / 'raw' / axial_group / 'rtpipeline_metrics.csv'
    with csv_bad.open() as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        kept = [row for row in r if row['ROI_OriginalName'] != 'sphere']
    with csv_bad.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(kept)
    proc, report = _run_verify(root_bad_val)
    assert proc.returncode != 0
    assert 'raw metric case coverage ' + axial_group + '_rtstruct_sphere' in report['errors']

    # Duplicate raw row and unexpected raw row must fail
    root_dup, _, _ = _build_synthetic_extension_run(tmp_path / 'dup_raw')
    csv_dup = root_dup / 'results' / 'shard-0' / 'raw' / axial_group / 'rtpipeline_metrics.csv'
    with csv_dup.open() as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows_dup = list(r)
    with csv_dup.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_dup + [rows_dup[0]])
    proc, report = _run_verify(root_dup)
    assert proc.returncode != 0
    assert 'duplicate raw metric row' in ' | '.join(report['errors'])

    unexpected_row = dict(rows_dup[0], ROI_OriginalName='unexpected_shape')
    with csv_dup.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_dup + [unexpected_row])
    proc, report = _run_verify(root_dup)
    assert proc.returncode != 0
    assert 'unexpected raw metric row' in ' | '.join(report['errors'])

    # Receipt raw_sha256 mismatch must fail
    root_rcpt, _, _ = _build_synthetic_extension_run(tmp_path / 'rcpt_raw_hash')
    rpath = root_rcpt / 'results' / 'shard-0' / 'receipt.json'
    rec = json.loads(rpath.read_text())
    rec['groups'][0]['raw_sha256'] = {'phantom.json': '0' * 64}
    rpath.write_text(json.dumps(rec))
    proc, report = _run_verify(root_rcpt)
    assert proc.returncode != 0
    assert 'raw file receipt digest mismatch' in ' | '.join(report['errors'])


def test_verify_extension_partial_mode_stays_explicit_and_incomplete(tmp_path):
    import csv
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'
    root, _, _ = _build_synthetic_extension_run(tmp_path / 'partial')
    out = root / 'results'
    with (out / 'metrics_long.csv').open() as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        kept = [row for row in r if row['group'] == axial_group]
    with (out / 'metrics_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(kept)
    _write_summary_csv(out, kept)
    proc, report = _run_verify(root, partial=True)
    assert proc.returncode == 0
    assert report['passed'] is True
    assert report['partial_audit'] is True
    assert report['complete'] is False


def test_verify_extension_requires_exact_terminal_disposition_tokens(tmp_path):
    import csv
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'
    refused_group = 'v0.5_aligned_ct1_dg2.5_uniform_refusal_z2'

    # 1. Near-match no_published_ROI_* prefix on unprinted non-refused case must fail
    root_bad_no_roi, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_no_roi')
    out_a = root_bad_no_roi / 'results'
    with (out_a / 'metrics_long.csv').open() as f:
        r = csv.DictReader(f)
        long_cols = r.fieldnames
        long_rows = list(r)
    for row in long_rows:
        if row['case'] == axial_group + '_rtstruct_sphere':
            for k in ('value', 'signed_error', 'absolute_error', 'relative_error', 'beyond_boundary'):
                row[k] = ''
            row['status'] = 'unavailable'
            row['unavailable_reason'] = 'no_published_ROI_but_executed_successfully'
    with (out_a / 'metrics_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(long_rows)
    _write_summary_csv(out_a, long_rows)
    csv_a = out_a / 'shard-0' / 'raw' / axial_group / 'rtpipeline_metrics.csv'
    with csv_a.open() as f:
        r = csv.DictReader(f)
        raw_cols = r.fieldnames
        kept = [row for row in r if row['ROI_OriginalName'] != 'sphere']
    with csv_a.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=raw_cols)
        w.writeheader()
        w.writerows(kept)
    proc_a, report_a = _run_verify(root_bad_no_roi)
    assert proc_a.returncode != 0
    assert report_a['passed'] is False
    assert report_a['complete'] is False
    assert 'raw metric case coverage ' + axial_group + '_rtstruct_sphere' in report_a['errors']

    # 2. Substring not_rotated_dose_rotation_exceeds_1_degree on unprinted refused case must fail both checks
    root_bad_ref, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_refusal_unprinted')
    out_b = root_bad_ref / 'results'
    with (out_b / 'metrics_long.csv').open() as f:
        r = csv.DictReader(f)
        long_cols = r.fieldnames
        long_rows = list(r)
    for row in long_rows:
        if row['case'] == refused_group + '_rtstruct_sphere':
            row['unavailable_reason'] = 'not_rotated_dose_rotation_exceeds_1_degree'
    with (out_b / 'metrics_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(long_rows)
    _write_summary_csv(out_b, long_rows)
    csv_b = out_b / 'shard-0' / 'raw' / refused_group / 'rtpipeline_metrics.csv'
    with csv_b.open() as f:
        r = csv.DictReader(f)
        raw_cols = r.fieldnames
        kept = [row for row in r if row['ROI_OriginalName'] != 'sphere']
    with csv_b.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=raw_cols)
        w.writeheader()
        w.writerows(kept)
    proc_b, report_b = _run_verify(root_bad_ref)
    assert proc_b.returncode != 0
    assert report_b['passed'] is False
    assert report_b['complete'] is False
    assert 'refusal case ' + refused_group + '_rtstruct_sphere' in report_b['errors']
    assert 'raw metric case coverage ' + refused_group + '_rtstruct_sphere' in report_b['errors']

    # 3. Substring refusal reason with raw row intact must also fail refusal-case check
    root_bad_ref_raw, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_refusal_raw')
    out_c = root_bad_ref_raw / 'results'
    with (out_c / 'metrics_long.csv').open() as f:
        r = csv.DictReader(f)
        long_cols = r.fieldnames
        long_rows = list(r)
    for row in long_rows:
        if row['case'] == refused_group + '_rtstruct_sphere':
            row['unavailable_reason'] = 'not_rotated_dose_rotation_exceeds_1_degree; detail'
    with (out_c / 'metrics_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(long_rows)
    proc_c, report_c = _run_verify(root_bad_ref_raw)
    assert proc_c.returncode != 0
    assert 'refusal case ' + refused_group + '_rtstruct_sphere' in report_c['errors']


def test_verify_extension_requires_case_metadata_and_consistent_input_digests(tmp_path):
    import json
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'
    refused_group = 'v0.5_aligned_ct1_dg2.5_uniform_refusal_z2'

    # 1. Missing raw case_metadata.json must fail
    root_missing, _, _ = _build_synthetic_extension_run(tmp_path / 'missing_cm')
    (root_missing / 'results' / 'shard-0' / 'raw' / axial_group / 'case_metadata.json').unlink()
    proc, report = _run_verify(root_missing)
    assert proc.returncode != 0
    assert report['passed'] is False
    assert report['complete'] is False
    assert 'missing raw files ' + axial_group + ':case_metadata.json' in report['errors']

    # 2. Zero-byte raw case_metadata.json and mismatched course_id must fail
    root_empty, _, _ = _build_synthetic_extension_run(tmp_path / 'empty_cm')
    cm_path = root_empty / 'results' / 'shard-0' / 'raw' / refused_group / 'case_metadata.json'
    cm_path.write_text('')
    proc, report = _run_verify(root_empty)
    assert proc.returncode != 0
    assert 'corrupt case_metadata ' + refused_group in report['errors']

    cm_path.write_text(json.dumps({'patient_id': 'SYNTH_DVH', 'course_id': axial_group}))
    proc, report = _run_verify(root_empty)
    assert proc.returncode != 0
    assert 'invalid case_metadata ' + refused_group in report['errors']

    # 3. Digest conflicts in input_sha256.json for phantom, case_metadata, and masks must fail
    for target_rel, expected_err in (
        ('phantom.json', 'input_sha256 phantom digest mismatch ' + axial_group),
        ('metadata/case_metadata.json', 'input_sha256 case_metadata digest mismatch ' + axial_group),
        ('Segmentation_TotalSegmentator/mask_sphere.nii.gz', 'input_sha256 mask digest mismatch ' + axial_group + '/sphere'),
    ):
        root_hash, _, _ = _build_synthetic_extension_run(tmp_path / ('hash_' + Path(target_rel).stem))
        ih_path = root_hash / 'results' / 'shard-0' / 'raw' / axial_group / 'input_sha256.json'
        ih_rows = json.loads(ih_path.read_text())
        for row in ih_rows:
            if row['path'] == target_rel:
                row['sha256'] = '0' * 64
        ih_path.write_text(json.dumps(ih_rows))
        proc, report = _run_verify(root_hash)
        assert proc.returncode != 0
        assert report['passed'] is False
        assert report['complete'] is False
        assert expected_err in report['errors']


def test_verify_extension_rejects_unplanned_raw_groups_and_keeps_cache_fallback(tmp_path):
    import shutil
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'

    # 1. Unplanned raw directory in a final shard must fail before per-group checks
    root_unplanned, _, _ = _build_synthetic_extension_run(tmp_path / 'unplanned')
    stray_dir = root_unplanned / 'results' / 'shard-0' / 'raw' / 'unplanned_group'
    stray_dir.mkdir(parents=True)
    (stray_dir / 'rtpipeline_metrics.csv').write_text('ROI_OriginalName,DmeanGy\nunplanned_shape,30\n')
    proc, report = _run_verify(root_unplanned)
    assert proc.returncode != 0
    assert report['passed'] is False
    assert report['complete'] is False
    assert 'unexpected raw group dir unplanned_group' in report['errors']

    # 2. Cache fallback for planned groups under cached_execution stays accepted
    root_cached, _, _ = _build_synthetic_extension_run(tmp_path / 'cached_fallback')
    src_raw = root_cached / 'results' / 'shard-0' / 'raw' / axial_group
    dst_raw = root_cached / 'results' / 'cached_execution' / 'shard-0' / 'raw' / axial_group
    dst_raw.parent.mkdir(parents=True)
    shutil.move(str(src_raw), str(dst_raw))
    proc_ok, report_ok = _run_verify(root_cached)
    assert proc_ok.returncode == 0
    assert report_ok['passed'] is True
    assert report_ok['complete'] is True


def test_verify_extension_checks_curve_labels_producers_and_raw_columns(tmp_path):
    import csv, json
    axial_group = 'v2_aligned_ct1_dg2.5_uniform_axial'
    refused_group = 'v0.5_aligned_ct1_dg2.5_uniform_refusal_z2'

    # 1. Curve roi_name not matching a published ROI (and duplicate curve roi_name) must fail
    root_bad_curve, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_curve_label')
    cpath = root_bad_curve / 'results' / 'shard-0' / 'raw' / axial_group / 'dvh_curves.json'
    cpath.write_text(json.dumps({
        'points': [{'roi_name': 'unplanned_shape', 'data': [{'x': 19.995, 'y': 100.0}, {'x': 20.005, 'y': 0.0}]}]
    }))
    proc, report = _run_verify(root_bad_curve)
    assert proc.returncode != 0
    assert report['passed'] is False
    assert report['complete'] is False
    assert 'unmatched dvh_curves roi ' + axial_group + '/unplanned_shape' in report['errors']

    cpath.write_text(json.dumps({
        'points': [
            {'roi_name': 'sphere', 'data': [{'x': 19.995, 'y': 100.0}, {'x': 20.005, 'y': 0.0}]},
            {'roi_name': 'sphere', 'data': [{'x': 19.995, 'y': 100.0}, {'x': 20.005, 'y': 0.0}]},
        ]
    }))
    proc_dup, report_dup = _run_verify(root_bad_curve)
    assert proc_dup.returncode != 0
    assert f'invalid dvh_curves roi {axial_group}/1' in report_dup['errors']

    # 2. RTSTRUCT row claiming NIfTI_Direct (or wrong RTSTRUCT producer) must fail
    root_bad_prod, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_producer')
    csv_prod = root_bad_prod / 'results' / 'shard-0' / 'raw' / axial_group / 'rtpipeline_metrics.csv'
    with csv_prod.open() as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows = list(r)
    for row in rows:
        if row['ROI_OriginalName'] == 'sphere':
            row['Segmentation_Source'] = 'NIfTI_Direct'
    with csv_prod.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    proc_prod, report_prod = _run_verify(root_bad_prod)
    assert proc_prod.returncode != 0
    assert 'RTSTRUCT producer provenance ' + axial_group in report_prod['errors']

    # 3. Missing required native column (e.g. DmeanGy on refusal group, or V20Gy on axial group) must fail;
    #    omitting only the V20 columns on the rotation refusal must still pass
    root_bad_col, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_col_refusal')
    csv_ref = root_bad_col / 'results' / 'shard-0' / 'raw' / refused_group / 'rtpipeline_metrics.csv'
    with csv_ref.open() as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows = list(r)
    no_dmean_cols = [c for c in cols if c != 'DmeanGy']
    with csv_ref.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=no_dmean_cols)
        w.writeheader()
        w.writerows([{k: row[k] for k in no_dmean_cols} for row in rows])
    proc_col, report_col = _run_verify(root_bad_col)
    assert proc_col.returncode != 0
    assert 'missing raw columns ' + refused_group + ':DmeanGy' in report_col['errors']

    no_v20_cols = [c for c in cols if c not in {'V20Gy (cm³)', 'V20Gy (%)'}]
    with csv_ref.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=no_v20_cols)
        w.writeheader()
        w.writerows([{k: row[k] for k in no_v20_cols} for row in rows])
    proc_v20_ok, report_v20_ok = _run_verify(root_bad_col)
    assert proc_v20_ok.returncode == 0
    assert report_v20_ok['passed'] is True
    assert report_v20_ok['complete'] is True

    # 4. Omitting V20 column on a non-refused axial group must fail
    root_axial_col, _, _ = _build_synthetic_extension_run(tmp_path / 'bad_col_axial')
    csv_ax = root_axial_col / 'results' / 'shard-0' / 'raw' / axial_group / 'rtpipeline_metrics.csv'
    with csv_ax.open() as f:
        r = csv.DictReader(f)
        ax_cols = [c for c in r.fieldnames if c != 'V20Gy (%)']
        ax_rows = list(r)
    with csv_ax.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ax_cols)
        w.writeheader()
        w.writerows([{k: row[k] for k in ax_cols} for row in ax_rows])
    proc_ax_col, report_ax_col = _run_verify(root_axial_col)
    assert proc_ax_col.returncode != 0
    assert 'missing raw columns ' + axial_group + ':V20Gy (%)' in report_ax_col['errors']

    # 5. Curve file covering a unique subset of published ROIs must pass
    root_subset, _, _ = _build_synthetic_extension_run(tmp_path / 'subset_curve_ok')
    cpath_sub = root_subset / 'results' / 'shard-0' / 'raw' / axial_group / 'dvh_curves.json'
    cpath_sub.write_text(json.dumps({
        'points': [{'roi_name': 'mask_sphere', 'data': [{'x': 19.995, 'y': 100.0}, {'x': 20.005, 'y': 0.0}]}]
    }))
    proc_sub, report_sub = _run_verify(root_subset)
    assert proc_sub.returncode == 0
    assert report_sub['passed'] is True
    assert report_sub['complete'] is True







def test_compare_legacy_refuses_duplicate_metric_or_input_keys(tmp_path):
    # PR 57 review: duplicate keys must not overwrite evidence and pass.
    import csv, json, subprocess, sys
    fields = ['case', 'tool', 'metric', 'value', 'analytic', 'abs_err', 'rel_err', 'signed_err', 'status',
              'rel_err_status', 'shape', 'volume_cc', 'phase', 'dose_mm', 'ct_mm', 'field']
    row = dict(case='c1', tool='rtpipeline', metric='DmeanGy', value='20', analytic='20', abs_err='0',
               rel_err='0', signed_err='0', status='ok', rel_err_status='ok', shape='sphere', volume_cc='1',
               phase='', dose_mm='1', ct_mm='1', field='uniform')
    def write(directory, rows, inputs):
        directory.mkdir(parents=True)
        with (directory / 'metrics_long.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        (directory / 'input_sha256.json').write_text(json.dumps(inputs))
    entry = {'group': 'g', 'path': 'a.dcm', 'sha256': '0' * 64, 'bytes': 1}
    script = Path(__file__).resolve().parent / 'compare_legacy.py'
    def run(new_rows, new_inputs):
        ref, res = tmp_path / ('ref%d' % run.n), tmp_path / ('res%d' % run.n)
        run.n += 1
        write(ref, [row], [entry]); write(res / 'legacy', new_rows, new_inputs)
        done = subprocess.run([sys.executable, '-B', str(script), '--reference', str(ref), '--results', str(res)],
                              capture_output=True, text=True)
        return done.returncode, json.loads((res / 'legacy_comparison.json').read_text())
    run.n = 0
    assert run([row], [entry])[1]['passed'] is True
    code, report = run([dict(row, value='999'), row], [entry])
    assert code != 0 and report['passed'] is False and report['unique_metric_keys'] is False
    code, report = run([row], [dict(entry, sha256='1' * 64), entry])
    assert code != 0 and report['passed'] is False and report['unique_input_keys'] is False
