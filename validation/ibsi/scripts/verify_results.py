"""Validate source integrity, workbook arithmetic and complete row accounting."""
import argparse
import csv
import hashlib
import json
import math
import radiomics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / 'results'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_validation import workbook_difference, references


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true', help='Write verification and final artifact hashes after report generation')
    args = parser.parse_args()
    receipt_path = OUT / 'run_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    checks = []
    # Numeric cases are independently specified from the official spreadsheet
    # semantics, including negative decimal places and zero-tolerance targets.
    fixtures = [(1490., 1494.6, 0.), (556., 556.333333333, 0.),
                (9.31, 9.24490489119279, .06), (8.54, 8.478109493212118, .06),
                (-.355, 2.6453795193121667, 3.), (2.15, 2.1486486486486487, 0.),
                (0., .9, 0.), (0., -1.9, 1.)]
    for reference, actual, expected in fixtures:
        assert workbook_difference(reference, actual) == expected
    checks.append(f'{len(fixtures)} independent workbook-arithmetic cases passed')
    for source in receipt['source_files']:
        assert digest(ROOT / source['path']) == source['sha256'], source['path']
    checks.append(f"{len(receipt['source_files'])} downloaded source hashes passed")
    for path, expected in receipt['rtpipeline_source_hashes'].items():
        assert digest(REPO / path) == expected, path
    for path, expected in receipt['pyradiomics_source_hashes'].items():
        assert digest(Path(radiomics.__file__).parent / path) == expected, path
    checks.append('RTpipeline and installed PyRadiomics source/binary hashes passed')
    for name in ['scripts/run_validation.py', 'scripts/feature_map.py']:
        assert digest(ROOT / name) == receipt['scripts_sha256'][name], name
    checks.append('Numerical extraction and mapping scripts match the executed versions')
    for path, expected in receipt['raw_output_hashes'].items():
        assert digest(ROOT / path) == expected, path
    checks.append(f"{len(receipt['raw_output_hashes'])} raw extraction hashes passed")
    rows = list(csv.DictReader((OUT / 'feature_conformance.csv').open()))
    assert len(rows) == receipt['row_count']
    refs = references()
    for arm, config in sorted({(r['arm'], r['configuration']) for r in rows}):
        subset = [r for r in rows if (r['arm'], r['configuration']) == (arm, config)]
        tags = [r['ibsi_tag'] for r in subset if r['ibsi_tag']]
        assert len(tags) == len(set(tags)), (arm, config)
        assert set(tags) == {r['tag'] for r in refs[config]}, (arm, config)
    checks.append(f'{len(rows)} classified rows cover every reference row once per extraction arm')
    compared = 0
    for row in rows:
        assert row['status'] in ['within tolerance', 'outside tolerance', 'not computed', 'not mappable']
        if not row['computed_value']:
            continue
        raw_path = OUT / 'raw' / (Path(row['parameter_file']).stem + '.json')
        raw = json.loads(raw_path.read_text())['values']
        value = float(row['computed_value'])
        assert math.isfinite(value)
        assert raw[row['pyradiomics_feature']] == value
        if row['status'] in ['within tolerance', 'outside tolerance']:
            reference, tolerance = float(row['reference_value']), float(row['published_tolerance'])
            diff = workbook_difference(reference, value)
            assert diff == float(row['workbook_difference'])
            assert (diff <= tolerance) == (row['status'] == 'within tolerance')
            assert (abs(value-reference) <= tolerance) == (row['strict_raw_within_tolerance'] == 'True')
            compared += 1
    checks.append(f'{compared} comparisons independently linked to raw outputs; statuses and strict comparisons passed')
    family_rows = list(csv.DictReader((OUT / 'family_summary.csv').open()))
    for family in family_rows:
        rr = [r for r in rows if (r['arm'], r['configuration'], r['family']) == (family['arm'], family['configuration'], family['family'])]
        for status, field in [('within tolerance', 'within_tolerance'), ('outside tolerance', 'outside_tolerance'), ('not computed', 'not_computed'), ('not mappable', 'not_mappable')]:
            assert sum(r['status'] == status for r in rr) == int(family[field])
        assert sum(r['has_consensus_reference'] == 'True' for r in rr) == int(family['consensus_reference_rows'])
    checks.append('All family counts and denominators independently reconciled')
    digital = json.loads((OUT / 'raw/digital_digital_3D_averaged.json').read_text())['values']
    assert digital['original_firstorder_Energy'] == 567.
    assert digital['original_shape_VoxelVolume'] == 592.
    assert math.isclose(digital['original_firstorder_Mean'], 159/74, rel_tol=1e-14)
    checks.append('Digital phantom analytic checks passed: energy 567, voxel volume 592 mm3, mean 159/74')
    diagnostics = list(csv.DictReader((OUT / 'preprocessing_diagnostics.csv').open()))
    assert len(diagnostics) == 300
    assert all(r['status'] == 'within tolerance' for r in diagnostics)
    checks.append('All 300 IBSI preprocessing diagnostics within published tolerances')
    failures = [r for r in rows if r['configuration'] == 'digital' and r['status'] == 'outside tolerance']
    assert {r['ibsi_tag'] for r in failures} == {'morph_pca_least_axis', 'stat_kurt'}
    assert all(r['status'] == 'outside tolerance' for r in rows if r['ibsi_tag'] == 'stat_kurt' and r['computed_value'])
    checks.append('Digital least-axis and unadjusted kurtosis discrepancies retained')
    if not args.finalize:
        for path, expected in receipt['result_hashes'].items():
            assert digest(ROOT / path) == expected, path
        for path, expected in receipt['scripts_sha256'].items():
            assert digest(ROOT / path) == expected, path
        checks.append('Final artifact and script hashes passed')
    else:
        verification = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'passed': True, 'checks': checks,
                        'row_count': len(rows), 'computed_comparisons': compared}
        (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        receipt['verification'] = verification
        receipt['reproduction_commands'] = [
            '"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/retrieve_pinned.py',
            'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/run_validation.py',
            '"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/check_preprocessing.py',
            '"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/report.py',
            '"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/verify_results.py --finalize',
            '"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/verify_results.py']
        receipt['scripts_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'scripts').glob('*.py')}
        receipt['environment_spec_hashes'] = {str(p.relative_to(REPO)): digest(p) for p in [REPO / 'envs/rtpipeline-radiomics.yaml']}
        receipt['result_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'run_receipt.json' and p.suffix != '.log'}
        receipt['documentation_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / 'README.md', ROOT / '.gitignore']}
        receipt['finalized_utc'] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    for check in checks:
        print(check)
    print('PASS')


if __name__ == '__main__':
    main()
