"""Verify canonical synthetic DVH evidence and report descriptive accuracy limits."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent
OUT = BENCH / 'results'
REPO = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
from analytic import METRICS
from generate_phantoms import designs


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    receipt = json.loads((OUT / 'run_receipt.json').read_text())
    rows = pd.read_csv(OUT / 'metrics_long.csv')
    geometry = pd.read_csv(OUT / 'geometry.csv')
    selected = [s for s in designs() if s['dose_mm'] == 2.5]
    groups = {s['group'] for s in selected}
    cases = {s['group'] + '_' + shape for s in selected for shape in s['shapes']}
    primary = rows[rows.tool == 'rtpipeline'].copy()
    numeric = rows.dropna(subset=['value', 'analytic'])
    checks = {
        'run_completed': receipt['status'] == 'completed' and not receipt['errors'],
        'all_36_dg25_groups': {g['group'] for g in receipt['groups']} == groups
            and len(receipt['groups']) == 36 and all(g['status'] == 'completed' for g in receipt['groups']),
        'all_120_cases': set(rows.case) == cases,
        'all_4440_rows_unique': len(rows) == 4440 and not rows.duplicated(['case', 'tool', 'metric']).any(),
        'tool_row_counts': rows.tool.value_counts().to_dict() == {
            'rtpipeline': 1440, 'dicompyler_native': 1440, 'dicompyler_harmonized': 1440,
            'rtpipeline_curve_supplement': 120},
        'endpoint_sets': all(len(g) == 12 and set(g.metric) == set(METRICS)
            for _, g in rows[rows.tool != 'rtpipeline_curve_supplement'].groupby(['case', 'tool'])),
        'native_missing_only_below_volume_D2cc': bool(
            primary[primary.value.isna()].metric.eq('D2cc_Gy').all()
            and primary[primary.value.isna()].status.eq('roi_below_2cc').all()),
        'absolute_errors_recompute': bool(np.allclose(numeric.abs_err, abs(numeric.value - numeric.analytic),
                                                     atol=2e-8, rtol=1e-9)),
        'pipeline_sources_unchanged_during_run': receipt['pipeline_sources_unchanged'],
        'benchmark_sources_match_receipt': all(sha(BENCH / name) == digest
            for name, digest in receipt.get('public_benchmark_code_sha256', receipt['benchmark_code_sha256']).items()),
        'nice_10': receipt['effective_nice'] == 10,
        'scratch_inside_results': str((OUT / 'tmp').relative_to(REPO)) in receipt['reproduction_command'],
        'helper_diagnostics_labelled': {
            'helper_snap_planes', 'helper_snap_moved_contours', 'helper_snap_max_mm',
            'helper_snap_min_thickness_mm', 'dose_plane_alignment'}.issubset(geometry.columns)
            and not {'snapped_planes', 'moved_contours', 'max_snap_mm',
                     'snapped_min_thickness_mm'}.intersection(geometry.columns),
    }
    alignments = []
    raw_ok = coverage_ok = curves_ok = alignment_ok = d2cc_ok = True
    for spec in selected:
        raw = OUT / 'raw' / spec['group']
        raw_ok &= all((raw / f).is_file() for f in (
            'phantom.json', 'dicompyler.json', 'rtpipeline_metrics.csv', 'dvh_curves.json',
            'dvh_qc.json', 'case_metadata.json'))
        table = pd.read_csv(raw / 'rtpipeline_metrics.csv')
        coverage_ok &= table.dose_grid_coverage_status.eq('fully_covered').all()
        available = table['Volume (cm³)'].ge(2.)
        d2cc_ok &= bool(table.loc[available, 'D2ccGy'].notna().all()
            and table.loc[available, 'D2cc_status'].eq('available').all()
            and table.loc[~available, 'D2ccGy'].isna().all()
            and table.loc[~available, 'D2cc_status'].eq('roi_below_2cc').all())
        for _, row in table.iterrows():
            native = primary[primary.case.eq(spec['group'] + '_' + row.ROI_OriginalName)
                             & primary.metric.eq('D2cc_Gy')].iloc[0]
            d2cc_ok &= bool(native.status == row.D2cc_status and (
                pd.isna(native.value) and pd.isna(row.D2ccGy) or
                np.isclose(native.value, row.D2ccGy, atol=2e-8, rtol=1e-9)))
        qc = json.loads((raw / 'dvh_qc.json').read_text())
        alignment_ok &= qc['dose_plane_alignment_counts'] == table.dose_plane_alignment.value_counts().to_dict()
        for _, row in table.iterrows():
            case = spec['group'] + '_' + row.ROI_OriginalName
            diag = geometry[geometry.case == case]
            alignment_ok &= len(diag) == 1 and diag.iloc[0].dose_plane_alignment == row.dose_plane_alignment
            alignments.append(dict(group=spec['group'], case=case, ROI_Number=int(row.ROI_Number),
                                   ROI_Name=row.ROI_Name, dose_plane_alignment=row.dose_plane_alignment))
        for curve in json.loads((raw / 'dvh_curves.json').read_text())['points']:
            x = np.array([p['x'] for p in curve['data']])
            y = np.array([p['y'] for p in curve['data']])
            curves_ok &= bool(np.isfinite(x).all() and np.isfinite(y).all()
                              and (np.diff(x) > 0).all() and (np.diff(y) <= 1e-10).all()
                              and (y >= -1e-10).all() and (y <= 100 + 1e-10).all())
    nonzero = numeric[numeric.analytic != 0]
    checks['relative_errors_recompute'] = bool(np.allclose(nonzero.rel_err, nonzero.abs_err / abs(nonzero.analytic), atol=2e-9, rtol=1e-9))
    zero = rows[rows.analytic == 0]
    checks['zero_truth_relative_error_undefined'] = bool(zero.rel_err.isna().all() and zero.abs_err.notna().all())
    checks['production_kernel_matches_executed_sources'] = all(sha(REPO / p) == receipt['pipeline_source_sha256_after'][p] for p in ('rtpipeline/utils.py', 'rtpipeline/dvh.py'))
    checks.update(raw_evidence_complete=bool(raw_ok), all_fully_covered=bool(coverage_ok),
                  curves_finite_monotone=bool(curves_ok), production_alignment_consistent=bool(alignment_ok),
                  native_D2cc_matches_published_table=bool(d2cc_ok))
    pd.DataFrame(alignments).to_csv(OUT / 'alignment_by_roi.csv', index=False)

    primary['boundary_type'] = None
    primary['boundary'] = np.nan
    dose = primary.metric.str.endswith('_Gy')
    v95 = primary.metric.eq('V95Rx_pct')
    volume = primary.metric.eq('Volume_cc') & primary.analytic.ge(1.)
    primary.loc[dose, 'boundary_type'] = 'dose_Gy'
    primary.loc[dose, 'boundary'] = np.maximum(.02 * abs(primary.loc[dose, 'analytic']), .5)
    primary.loc[v95, ['boundary_type', 'boundary']] = ['V95Rx_pp', 2.]
    primary.loc[volume, 'boundary_type'] = 'volume_cc'
    primary.loc[volume, 'boundary'] = .03 * primary.loc[volume, 'analytic']
    primary['boundary_pass'] = pd.Series(pd.NA, index=primary.index, dtype='boolean')
    assessed = primary.abs_err.notna() & primary.boundary.notna()
    primary.loc[assessed, 'boundary_pass'] = primary.loc[assessed, 'abs_err'] <= primary.loc[assessed, 'boundary']
    primary['error_boundary_ratio'] = primary.abs_err / primary.boundary
    primary.to_csv(OUT / 'boundary_results.csv', index=False, float_format='%.12g')
    primary[primary.boundary_pass.eq(False)].to_csv(OUT / 'boundary_failures.csv', index=False,
                                                 float_format='%.12g')
    endpoint_summary = []
    for (group, metric), g in primary.groupby(['group', 'metric']):
        endpoint_summary.append(dict(group=group, metric=metric, cases=len(g),
            numeric_comparisons=int(g.abs_err.notna().sum()), missing_values=int(g.value.isna().sum()),
            max_abs_error=g.abs_err.max(), bounded_comparisons=int(g.boundary_pass.notna().sum()),
            boundary_failures=int(g.boundary_pass.eq(False).sum()), max_error_boundary_ratio=g.error_boundary_ratio.max()))
    pd.DataFrame(endpoint_summary).to_csv(OUT / 'boundary_summary_by_group_metric.csv', index=False,
                                        float_format='%.12g')

    def summarize(key, g):
        d = g[g.boundary_type.eq('dose_Gy') & g.boundary_pass.notna()]
        v = g[g.boundary_type.eq('V95Rx_pp')]
        z = g[g.boundary_type.eq('volume_cc')]
        return dict(stratum=key, cases=int(g.case.nunique()), dose_comparisons=len(d),
                    dose_failures=int(d.boundary_pass.eq(False).sum()), max_dose_error_Gy=float(d.abs_err.max()),
                    V95Rx_comparisons=len(v), V95Rx_failures=int(v.boundary_pass.eq(False).sum()),
                    max_V95Rx_error_pp=float(v.abs_err.max()), volume_comparisons=len(z),
                    volume_failures=int(z.boundary_pass.eq(False).sum()), max_volume_error_pct=float(100 * z.rel_err.max()))

    by_ct = [summarize(ct, g) for ct, g in primary.groupby('ct_mm')]
    pd.DataFrame(by_ct).to_csv(OUT / 'boundary_summary_by_ct.csv', index=False, float_format='%.12g')
    by_group = [summarize(group, g) for group, g in primary.groupby('group')]
    pd.DataFrame(by_group).to_csv(OUT / 'boundary_summary_by_group.csv', index=False, float_format='%.12g')
    overall = summarize('overall', primary)
    volume_by_size = [dict(volume_cc=float(size), cases=len(g),
        failures=int(g.boundary_pass.eq(False).sum()), max_relative_error_pct=float(100*g.rel_err.max()))
        for size, g in primary[volume].groupby('volume_cc')]
    volume_components = primary[volume][['case', 'value', 'analytic', 'rel_err']].merge(
        geometry[['case', 'contour_slab_volume_cc']], on='case', validate='one_to_one')
    volume_components['contour_slab_signed_error_cc'] = volume_components.contour_slab_volume_cc - volume_components.analytic
    volume_components['subsequent_rasterization_signed_error_cc'] = volume_components.value - volume_components.contour_slab_volume_cc
    volume_components.to_csv(OUT / 'primary_volume_error_components.csv', index=False, float_format='%.12g')
    native_d2cc = primary[primary.metric.eq('D2cc_Gy')]
    checks['all_native_dose_boundaries_hold'] = (
        overall['dose_comparisons'] == 960 + int(native_d2cc.value.notna().sum())
        and overall['dose_failures'] == 0)
    checks['all_V95Rx_boundaries_hold'] = overall['V95Rx_comparisons'] == 120 and overall['V95Rx_failures'] == 0
    supplements = rows[(rows.tool == 'rtpipeline_curve_supplement') & rows.abs_err.notna()]
    supplement_result = dict(available=len(supplements), unavailable=120-len(supplements),
        failures=int((supplements.abs_err > np.maximum(.02 * abs(supplements.analytic), .5)).sum()),
        max_abs_error_Gy=float(supplements.abs_err.max()))
    paired_d2cc = native_d2cc[['case', 'value']].merge(
        rows[rows.tool.eq('rtpipeline_curve_supplement')][['case', 'value']],
        on='case', suffixes=('_native', '_curve'), validate='one_to_one')
    checks['native_D2cc_matches_full_curve'] = bool(np.allclose(
        paired_d2cc.value_native, paired_d2cc.value_curve, atol=2e-8, rtol=1e-9, equal_nan=True))
    d2cc_result = dict(cases=len(native_d2cc), available=int(native_d2cc.value.notna().sum()),
        unavailable=int(native_d2cc.value.isna().sum()),
        failures=int(native_d2cc.boundary_pass.eq(False).sum()),
        max_abs_error_Gy=float(native_d2cc.abs_err.max()))

    suites = {}
    for name in ('alignment-tests', 'dvh-tests', 'rtstruct-tests', 'analytic-tests'):
        root = ET.parse(OUT / (name + '.xml'))
        counts = dict(tests=0, failures=0, errors=0, skipped=0)
        failures = []
        for suite in root.iter('testsuite'):
            for k in counts:
                counts[k] += int(suite.attrib.get(k, 0))
        for case in root.iter('testcase'):
            for tag in ('failure', 'error'):
                for failure in case.findall(tag):
                    failures.append(dict(classname=case.attrib.get('classname'), name=case.attrib['name'],
                                         type=tag, message=failure.attrib.get('message', '')))
        counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
        suites[name] = dict(**counts, failure_details=failures)

    verification = dict(checks={k:bool(v) for k,v in checks.items()}, all_integrity_and_dose_checks_passed=all(checks.values()),
        boundaries=overall, boundaries_by_ct=by_ct, volume_boundaries_by_size=volume_by_size,
        native_D2cc=d2cc_result, curve_D2cc_supplement=supplement_result, tests=suites,
        python=receipt['python'], effective_nice=receipt['effective_nice'],
        executed_pipeline_source_sha256={p:receipt['pipeline_source_sha256_after'][p] for p in (
            'rtpipeline/utils.py', 'rtpipeline/dvh.py')})
    dump('verification.json', verification)
    lines = ['# Synthetic axial DVH accuracy', '',
        'All 36 dg2.5 groups completed: 120 synthetic structure–dose cases. CT slice spacing was 1 or 3 mm. '
        'The dose fields were uniform, linear in z, and radial. Continuous analytic truth was the reference.', '',
        'Dose-error limits were max(2% of analytic dose, 0.5 Gy). The V95Rx limit was 2 percentage points. '
        'The volume limit was 3% for structures of at least 1 cm³. V20Gy has no specified absolute-volume limit.', '',
        '| CT spacing (mm) | Cases | Dose comparisons | Dose failures | Max dose error (Gy) | V95Rx failures | Max V95Rx error (pp) | Volume failures | Max volume error (%) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in by_ct:
        lines.append(f"| {r['stratum']:g} | {r['cases']} | {r['dose_comparisons']} | {r['dose_failures']} | "
            f"{r['max_dose_error_Gy']:.6g} | {r['V95Rx_failures']}/{r['V95Rx_comparisons']} | "
            f"{r['max_V95Rx_error_pp']:.6g} | {r['volume_failures']}/{r['volume_comparisons']} | {r['max_volume_error_pct']:.6g} |")
    lines += ['', f"All {overall['dose_comparisons']} available production RTpipeline dose comparisons and all 120 V95Rx comparisons met their limits. "
        f"Volume exceeded its limit in {overall['volume_failures']} of 120 cases.", '',
        f"Native D2cc is available in {d2cc_result['available']}/{d2cc_result['cases']} cases; "
        f"{d2cc_result['unavailable']} measured structures are below 2 cm³. "
        f"Its maximum absolute error is {d2cc_result['max_abs_error_Gy']:.6g} Gy, with "
        f"{d2cc_result['failures']} cases exceeding max(2% of analytic dose, 0.5 Gy). "
        'The full-curve D2cc check is reported separately in verification.json. '
        'The isolated dicompyler comparator shares the production rasterizer and is not an independent DVH engine. '
        'The helper_snap_ columns describe the optional snapping helper; dose_plane_alignment records production alignment.', '',
        'These measurements cover synthetic axial geometry and smooth dose fields. They do not assess patient anatomy, '
        'brachytherapy source gradients, delivered dose, or clinical or manuscript readiness.']
    (OUT / 'boundary_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(checks=verification['checks'], boundaries=overall, tests={k:{a:b for a,b in v.items()
        if a != 'failure_details'} for k,v in suites.items()}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.results.resolve()
    if not OUT.is_relative_to(BENCH):
        parser.error('Results must be inside the benchmark directory')
    audit()
    report = json.loads((OUT / 'verification.json').read_text())
    raise SystemExit(0 if report['all_integrity_and_dose_checks_passed'] else 1)
