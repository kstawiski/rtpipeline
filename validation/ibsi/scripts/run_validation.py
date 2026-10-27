"""Public IBSI-1 phantom extraction. Run with the radiomics Python -I -B."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public_sources_20261027'
OUT = ROOT / 'results'
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(REPO))
import numpy as np
import SimpleITK as sitk
import yaml
import radiomics
from openpyxl import load_workbook
from rtpipeline.radiomics import _extractor
from rtpipeline.radiomics_ct_contract import build_ct_extractors, _PRIMARY_WINDOWS
from feature_map import load_mapping, stem

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
logging.getLogger('radiomics').setLevel(logging.ERROR)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def references():
    workbook = load_workbook(SOURCE / 'official_ibsi_reference/IBSI-1-submission-table.xlsx', read_only=True, data_only=True)
    out = {}
    for sheet in workbook:
        sheet.reset_dimensions()
        rows = list(sheet.values)
        out['digital' if sheet.title == 'digital phantom' else sheet.title[-1]] = [
            dict(zip(['dataset', 'family', 'feature', 'consensus', 'reference', 'tolerance', 'unused_result', 'unused_difference', 'unused_check', 'tag'], row[:10]))
            for row in rows[1:]
        ]
    return out


def read_phantom(which):
    base = SOURCE / f'theibsi/data_sets/ibsi_1_{which}_phantom/nifti'
    img = sitk.ReadImage(str(base / 'image/phantom.nii.gz'), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(base / 'mask/mask.nii.gz'), sitk.sitkFloat32)
    # The official NIfTI instructions explicitly require rounding both arrays.
    arr = np.rint(sitk.GetArrayFromImage(img)).astype(np.float32)
    img2 = sitk.GetImageFromArray(arr)
    img2.CopyInformation(img)
    binary = sitk.GetImageFromArray(np.rint(sitk.GetArrayFromImage(mask)).astype(np.uint8))
    binary.CopyInformation(mask)
    return img2, binary


def describe(img, mask):
    a = sitk.GetArrayFromImage(img)
    m = sitk.GetArrayFromImage(mask) > 0
    points = np.argwhere(m)
    return {'size_xyz': list(img.GetSize()), 'spacing_xyz_mm': list(img.GetSpacing()),
            'origin_xyz_mm': list(img.GetOrigin()), 'direction': list(img.GetDirection()),
            'voxel_count': int(m.sum()), 'bbox_size_xyz': (np.ptp(points, axis=0) + 1)[::-1].tolist(),
            'roi_mean': float(a[m].mean(dtype=np.float64)), 'roi_min': float(a[m].min()), 'roi_max': float(a[m].max())}


def ibsi_preprocess(image, mask, configuration):
    """One fixed preprocessing rule per IBSI configuration; no fitted parameters.

    Center-aligned grid, ceil extent, linear float mask thresholded at >=0.5,
    nearest-integer HU, and masks kept separate for shape/intensity.
    """
    if configuration == 'A':
        processed, morph = image, mask
    else:
        old_spacing = np.array(image.GetSpacing())
        spacing = np.array([2., 2., old_spacing[2] if configuration == 'B' else 2.])
        old_size = np.array(image.GetSize())
        size = np.ceil(old_size * old_spacing / spacing).astype(int)
        direction = np.array(image.GetDirection()).reshape(3, 3)
        origin = np.array(image.GetOrigin()) + direction @ ((old_size - 1) * old_spacing / 2 - (size - 1) * spacing / 2)
        grid = sitk.Image([int(x) for x in size], sitk.sitkFloat32)
        grid.SetSpacing(spacing.tolist())
        grid.SetOrigin(origin.tolist())
        grid.SetDirection(image.GetDirection())
        method = sitk.sitkBSpline if configuration == 'E' else sitk.sitkLinear
        padded = sitk.ZeroFluxNeumannPad(image, [3, 3, 3], [3, 3, 3])
        processed = sitk.Resample(padded, grid, sitk.Transform(), method, -1000., sitk.sitkFloat32)
        a = np.rint(sitk.GetArrayFromImage(processed)).astype(np.float32)
        rounded = sitk.GetImageFromArray(a)
        rounded.CopyInformation(processed)
        processed = rounded
        fractional = sitk.Resample(sitk.Cast(mask, sitk.sitkFloat32), grid, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32)
        morph = sitk.Cast(fractional >= .5, sitk.sitkUInt8)
    a = sitk.GetArrayFromImage(processed)
    m = sitk.GetArrayFromImage(morph) > 0
    keep = m.copy()
    # IBSI computes sigma from the unresegmented ROI; range and outlier criteria
    # are independent and their intersection is applied in configuration E.
    if configuration in 'ABC':
        lo, hi = (-500., 400.) if configuration in 'AB' else (-1000., 400.)
        keep &= (a >= lo) & (a <= hi)
    elif configuration in 'DE':
        values = a[m].astype(np.float64)
        mean, sd = values.mean(), values.std(ddof=0)
        keep &= (a >= mean - 3 * sd) & (a <= mean + 3 * sd)
        if configuration == 'E':
            keep &= (a >= -1000.) & (a <= 400.)
    intensity = sitk.GetImageFromArray(keep.astype(np.uint8))
    intensity.CopyInformation(morph)
    return processed, morph, intensity


def settings(configuration, native, force2d=False, merged=False):
    s = {'label': 1, 'normalize': False, 'voxelArrayShift': 0, 'minimumROIDimensions': 2,
         'minimumROISize': 64, 'distances': [1], 'force2D': force2d,
         'force2Ddimension': 0, 'padDistance': 5, 'geometryTolerance': 1e-6,
         'correctMask': False, 'weightingNorm': 'no_weighting' if merged else None,
         'additionalInfo': True, 'preCrop': False, 'resegmentShape': False,
         'symmetricalGLCM': True, 'gldm_a': 0}
    if configuration in ['digital', 'A', 'C']:
        s['binWidth'] = 1 if configuration == 'digital' else 25
    else:
        s['binCount'] = 32
    if native and configuration in 'BCD':
        s.update(resampledPixelSpacing=[2., 2., 0. if configuration == 'B' else 2.], interpolator='sitkLinear')
    return s


def extract(image, morph, intensity, configuration, arm, force2d, merged, include_base):
    mode = ('2.5D' if force2d else '3D') + (', merged' if merged else ', direction-merged' if force2d else ', averaged')
    p = {'setting': settings(configuration, arm == 'native', force2d, merged), 'imageType': {'Original': {}},
         'featureClass': {k: [] for k in ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']}}
    path = OUT / 'configs' / f'{arm}_{configuration}_{mode.replace(", ", "_")}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(p, sort_keys=False))
    factory = lambda: _extractor(SimpleNamespace(radiomics_params_file=path), 'CT')
    # These are the production CT extractor-construction helpers. Case-level
    # ingestion and ROI classification are deliberately outside this phantom test.
    window = ((-500., 400.) if configuration in 'AB' else (-1000., 400.) if configuration == 'C' else None) if arm == 'native' else None
    shape_ext, raw_ext, primary_ext = build_ct_extractors(factory, window)
    ext = primary_ext or raw_ext
    if arm == 'native' and configuration == 'D':
        ext.settings.update(resegmentRange=[-3., 3.], resegmentMode='sigma')
    if not include_base:
        ext.enabledFeatures.pop('firstorder', None)
    if merged:
        ext.enabledFeatures = {k: v for k, v in ext.enabledFeatures.items() if k in ['glcm', 'glrlm']}
    result = ext.execute(image, intensity)
    if include_base:
        result.update(shape_ext.execute(image, morph))
    values = {k: float(v) for k, v in result.items() if k.startswith('original_')}
    diagnostics = {k: v for k, v in result.items() if k.startswith('diagnostics_')}
    for k, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite feature {k} in {arm}/{configuration}/{mode}')
    dump(OUT / 'raw' / f'{arm}_{configuration}_{mode.replace(", ", "_")}.json',
         {'values': values, 'diagnostics': diagnostics, 'effective_settings': ext.settings,
          'enabled_features': ext.enabledFeatures, 'shape_settings': shape_ext.settings if include_base else None})
    print(f'{arm} {configuration} {mode}: {len(values)} values', flush=True)
    return {'mode': mode, 'values': values, 'config_path': str(path.relative_to(ROOT)), 'settings': ext.settings}


def workbook_difference(reference, value):
    """Exact published H-column semantics: ABS(ROUNDDOWN(ref-value,digits))."""
    decimals = 0 if reference == 0 else 2 - math.floor(math.log10(abs(reference)))
    quantum = Decimal('1').scaleb(-decimals)
    return float(abs((Decimal(str(reference)) - Decimal(str(value))).quantize(quantum, rounding=ROUND_DOWN)))


def row_result(ref, mapping, runs, arm, configuration):
    family = ref['family']
    base = family.split(' (')[0]
    aggregation = family[len(base) + 2:-1] if family != base else ''
    item = mapping.get(stem(ref['tag']), {'ibsi_id': '', 'pyradiomics_class': '', 'pyradiomics_name': '', 'mapping_basis': 'diagnostic; outside radiomic feature catalogue'})
    key = f"original_{item['pyradiomics_class']}_{item['pyradiomics_name']}" if item['pyradiomics_class'] else ''
    values = None
    chosen = None
    if aggregation in ['2.5D', '3D']:
        expected_mode = '2.5D, direction-merged' if aggregation == '2.5D' else '3D, averaged'
    else:
        expected_mode = aggregation
    for run in runs:
        if not aggregation or run['mode'] == expected_mode:
            if key in run['values']:
                values, chosen = run['values'][key], run
                break
    numeric_ref = isinstance(ref['reference'], (int, float)) and isinstance(ref['tolerance'], (int, float))
    status, reason = 'not computed', ''
    if values is None:
        reason = 'PyRadiomics does not implement this feature or its enabled nondeprecated feature set omits it'
        if aggregation.startswith('2D'):
            reason = 'Slice-wise feature averaging is not implemented by the single-volume PyRadiomics extractor'
        if base.startswith('Diagnostics'):
            reason = 'Preprocessing diagnostic; recorded separately; not a radiomic feature'
    elif not numeric_ref:
        reason = 'No standardized consensus reference/tolerance is available'
    else:
        status = 'within tolerance' if workbook_difference(ref['reference'], values) <= ref['tolerance'] else 'outside tolerance'
        if item['ibsi_id'] == 'IPH6':
            reason = 'PyRadiomics returns kurtosis; IBSI defines excess kurtosis. Raw output is compared without subtracting 3.'
    return {'arm': arm, 'configuration': configuration, 'family': base, 'aggregation': aggregation,
            'ibsi_id': item['ibsi_id'], 'ibsi_tag': ref['tag'], 'ibsi_feature': ref['feature'],
            'pyradiomics_feature': key, 'mapping_basis': item['mapping_basis'],
            'reference_value': ref['reference'], 'published_tolerance': ref['tolerance'], 'consensus': ref['consensus'],
            'computed_value': values, 'absolute_error': abs(values - ref['reference']) if values is not None and numeric_ref else None,
            'workbook_difference': workbook_difference(ref['reference'], values) if values is not None and numeric_ref else None,
            'strict_raw_within_tolerance': abs(values - ref['reference']) <= ref['tolerance'] if values is not None and numeric_ref else None,
            'status': status, 'reason': reason, 'has_consensus_reference': numeric_ref,
            'parameter_file': chosen['config_path'] if chosen else '',
            'extraction_route': 'RTpipeline radiomics._extractor + radiomics_ct_contract.build_ct_extractors + PyRadiomics.execute'}


def main():
    started = now()
    OUT.mkdir(parents=True, exist_ok=True)
    refs = references()
    mapping, published, amendments = load_mapping(SOURCE, refs['digital'])
    write_csv(OUT / 'feature_mapping.csv', [{'ibsi_tag_stem': k, **v} for k, v in mapping.items()])
    all_rows, run_info = [], []
    for configuration in ['digital', 'A', 'B', 'C', 'D', 'E']:
        image, mask = read_phantom('digital' if configuration == 'digital' else 'ct_radiomics')
        arms = ['digital'] if configuration == 'digital' else ['native', 'ibsi_preprocessed'] if configuration != 'E' else ['ibsi_preprocessed']
        for arm in arms:
            entry = {'configuration': configuration, 'arm': arm, 'started_utc': now(), 'input_diagnostics': describe(image, mask)}
            if arm == 'ibsi_preprocessed':
                working, morph, intensity = ibsi_preprocess(image, mask, configuration)
            else:
                working, morph, intensity = image, mask, mask
            entry.update(morphology=describe(working, morph), intensity=describe(working, intensity))
            modes = [False, True] if configuration == 'digital' else [configuration in 'AB']
            runs = []
            for force2d in modes:
                runs.append(extract(working, morph, intensity, configuration, arm, force2d, False, not runs))
                runs.append(extract(working, morph, intensity, configuration, arm, force2d, True, False))
            entry['runs'] = [{k: v for k, v in r.items() if k != 'values'} for r in runs]
            entry['completed_utc'] = now()
            run_info.append(entry)
            mapped_keys = set()
            for ref in refs[configuration]:
                row = row_result(ref, mapping, runs, arm, configuration)
                all_rows.append(row)
                if row['computed_value'] is not None:
                    mapped_keys.add(row['pyradiomics_feature'])
            # Extra PyRadiomics outputs are explicit, without adding false IBSI
            # counterparts. Keep one row per distinct output feature per arm.
            for key, value in runs[0]['values'].items():
                if key in mapped_keys:
                    continue
                row = dict.fromkeys(all_rows[-1])
                extra_family = {'shape': 'Morphology', 'firstorder': 'Statistics', 'glcm': 'Co-occurrence matrix'}[key.split('_')[1]]
                row.update(arm=arm, configuration=configuration, family=extra_family, aggregation='',
                           ibsi_id='', ibsi_tag='', ibsi_feature='', pyradiomics_feature=key,
                           mapping_basis='No counterpart in IBSI 1 reference catalogue', computed_value=value,
                           status='not mappable', reason='PyRadiomics-specific feature; no IBSI 1 reference',
                           has_consensus_reference=False, parameter_file=runs[0]['config_path'],
                           extraction_route=all_rows[-1]['extraction_route'])
                all_rows.append(row)
    for ref in refs['E']:
        row = row_result(ref, mapping, [], 'native', 'E')
        row.update(reason='Native configuration E is not expressible: one PyRadiomics resegmentation mode cannot combine absolute range and sigma filtering.', extraction_route='not executed')
        all_rows.append(row)
    write_csv(OUT / 'feature_conformance.csv', all_rows)
    summaries = []
    groups = sorted({(r['arm'], r['configuration'], r['family']) for r in all_rows if not r['family'].startswith('Diagnostics')})
    for arm, configuration, family in groups:
        rows = [r for r in all_rows if (r['arm'], r['configuration'], r['family']) == (arm, configuration, family)]
        nref = sum(r['has_consensus_reference'] for r in rows)
        within = sum(r['status'] == 'within tolerance' for r in rows)
        outside = sum(r['status'] == 'outside tolerance' for r in rows)
        n = within + outside
        summaries.append({'arm': arm, 'configuration': configuration, 'family': family,
                          'within_tolerance': within, 'outside_tolerance': outside, 'computed_with_reference': n,
                          'not_computed': sum(r['status'] == 'not computed' for r in rows),
                          'not_mappable': sum(r['status'] == 'not mappable' for r in rows),
                          'consensus_reference_rows': nref, 'conformance_proportion_computed': within/n if n else None,
                          'reference_coverage': n/nref if nref else None,
                          'conformance_proportion_all_references': within/nref if nref else None,
                          'strict_raw_within': sum(r['strict_raw_within_tolerance'] is True for r in rows)})
    write_csv(OUT / 'family_summary.csv', summaries)
    dump(OUT / 'extraction_runs.json', run_info)
    production = _extractor(SimpleNamespace(radiomics_params_file=None, logs_root=None), 'CT')
    prod = {'configured_yaml': 'rtpipeline/radiomics_params.yaml',
            'configured_yaml_sha256': sha(REPO / 'rtpipeline/radiomics_params.yaml'),
            'effective_settings': production.settings, 'image_types': production.enabledImagetypes,
            'enabled_features': production.enabledFeatures, 'primary_windows_hu': _PRIMARY_WINDOWS,
            'shape_policy': 'separate unresegmented morphological mask; original image only',
            'sensitivity_policy': 'no intensity resegmentation', 'extraction_performed': False}
    dump(OUT / 'production_configuration.json', prod)
    source_manifests = [json.loads(p.read_text()) for p in SOURCE.glob('manifest*.json')]
    sources = {f['path']: f for m in source_manifests for f in m['files']}
    for f in sources.values():
        assert sha(ROOT / f['path']) == f['sha256'], f['path']
    assert all(str(p.relative_to(ROOT)) in sources for p in SOURCE.rglob('*') if p.is_file() and not p.name.startswith('manifest')), 'Unmanifested download'
    local_sources = sorted({Path(m.__file__).resolve() for n, m in sys.modules.items() if n.startswith('rtpipeline') and getattr(m, '__file__', None)} | {REPO / 'rtpipeline/radiomics_params.yaml', REPO / 'rtpipeline/radiomics_conda.py', REPO / 'rtpipeline/roi_class_map_v1.yaml'})
    receipt = {'started_utc': started, 'completed_utc': now(), 'python': sys.version,
               'python_executable': 'RTPIPELINE_RADIOMICS_PYTHON', 'isolated_mode': sys.flags.isolated,
               'python_bytecode_disabled': sys.dont_write_bytecode,
               'platform': platform.platform(), 'versions': {'rtpipeline': __import__('rtpipeline').__version__, **{k: importlib.metadata.version(k) for k in ['numpy', 'pyradiomics', 'SimpleITK', 'PyWavelets', 'pandas', 'PyYAML', 'pykwalify', 'openpyxl', 'scipy', 'pydicom']}},
               'threads': 1, 'source_files': list(sources.values()),
               'source_repositories': {k: v for m in source_manifests for k, v in m['repositories'].items()},
               'rtpipeline_source_hashes': {str(p.relative_to(REPO)): sha(p) for p in local_sources},
               'pyradiomics_source_hashes': {p.name: sha(p) for p in Path(radiomics.__file__).parent.iterdir() if p.suffix in ['.py', '.so']},
               'scripts_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'scripts').glob('*.py')},
               'reference_rule': 'Official workbook H/I formulas: truncate absolute difference to reference 3-significant-digit precision, then compare with published tolerance. Strict raw comparison also retained.',
               'configuration_E_native': 'not expressible: combined absolute range and sigma resegmentation; native ROI resampling also differs from IBSI',
               'no_patient_directories_accessed': True, 'no_existing_source_modified': True,
               'row_count': len(all_rows), 'mapped_feature_stems': len(mapping),
               'raw_output_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (OUT / 'raw').glob('*.json')},
               'result_hashes': {str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'run_receipt.json' and p.suffix != '.log'}}
    dump(OUT / 'run_receipt.json', receipt)
    print(f'Completed: {len(all_rows)} rows', flush=True)


if __name__ == '__main__':
    main()
