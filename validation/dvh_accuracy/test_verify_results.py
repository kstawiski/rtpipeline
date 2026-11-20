"""Regression tests for validation/dvh_accuracy boundary verification (C27)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _symlink_results_dir(res_src, res_copy):
    res_copy.mkdir()
    for entry in res_src.iterdir():
        if entry.name == 'history':
            continue
        os.symlink(entry, res_copy / entry.name)
    for out_name in (
        'alignment_by_roi.csv',
        'boundary_failures.csv',
        'boundary_report.md',
        'boundary_results.csv',
        'boundary_summary_by_ct.csv',
        'boundary_summary_by_group.csv',
        'boundary_summary_by_group_metric.csv',
        'primary_volume_error_components.csv',
        'verification.json',
    ):
        target = res_copy / out_name
        if target.is_symlink():
            target.unlink()


def test_verify_results_restricts_boundaries_to_six_dose_endpoints(tmp_path):
    bench = Path(__file__).resolve().parent
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import verify_results

    res_src = bench / 'results'
    res_copy = bench / f'results_tmp_c27_{tmp_path.name}'
    old_out = verify_results.OUT
    try:
        _symlink_results_dir(res_src, res_copy)
        verify_results.OUT = res_copy
        verify_results.audit()

        payload = json.loads((res_copy / 'verification.json').read_text())
        assert payload['all_integrity_and_dose_checks_passed'] is True
        assert payload['checks']['production_kernel_matches_executed_sources'] is True
        assert payload['checks']['benchmark_sources_match_receipt'] is True
        assert payload['checks']['all_native_dose_boundaries_hold'] is True
        assert payload['boundaries']['dose_comparisons'] == 692
        assert payload['boundaries']['dose_failures'] == 0

        df = pd.read_csv(res_copy / 'boundary_results.csv')
        bounded = df[df.metric.isin(['Dmean_Gy', 'D98_Gy', 'D95_Gy', 'D50_Gy', 'D2_Gy', 'D2cc_Gy'])]
        assert (bounded['boundary_type'] == 'dose_Gy').all()
        assert int(bounded['boundary_pass'].notna().sum()) == 692
        assert int(bounded['boundary_pass'].eq(False).sum()) == 0

        descriptive = df[df.metric.isin(['Dmin_Gy', 'Dmax_Gy', 'D0.03cc_Gy'])]
        assert len(descriptive) == 360
        assert descriptive['boundary_type'].isna().all()
        assert descriptive['boundary'].isna().all()
        assert descriptive['boundary_pass'].isna().all()

        report_md = (res_copy / 'boundary_report.md').read_text()
        assert '0/692' in report_md
        assert 'Dmin_Gy, Dmax_Gy and D0.03cc_Gy remain descriptive' in report_md
        assert (res_src / 'history' / '46a9edd' / 'run_receipt.json').exists()
    finally:
        verify_results.OUT = old_out
        shutil.rmtree(res_copy, ignore_errors=True)


def test_verify_results_rejects_empty_or_partial_benchmark_hash_map(tmp_path):
    bench = Path(__file__).resolve().parent
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import verify_results

    res_src = bench / 'results'
    res_copy = bench / f'results_tmp_c27_empty_{tmp_path.name}'
    old_out = verify_results.OUT
    try:
        _symlink_results_dir(res_src, res_copy)
        rcpt_link = res_copy / 'run_receipt.json'
        rcpt_data = json.loads(rcpt_link.read_text())
        rcpt_link.unlink()

        # 1. Both hash maps empty -> must fail (including CLI exit 1)
        empty_rcpt = dict(rcpt_data, benchmark_code_sha256={}, public_benchmark_code_sha256={})
        rcpt_link.write_text(json.dumps(empty_rcpt))
        proc = subprocess.run(
            [sys.executable, '-B', str(bench / 'verify_results.py'), '--results', str(res_copy)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1
        payload = json.loads((res_copy / 'verification.json').read_text())
        assert payload['checks']['benchmark_sources_match_receipt'] is False
        assert payload['all_integrity_and_dose_checks_passed'] is False

        # 2. Partial map (missing run_benchmark.py) -> must fail
        partial_map = {
            k: v for k, v in rcpt_data['public_benchmark_code_sha256'].items()
            if k != 'run_benchmark.py'
        }
        partial_rcpt = dict(rcpt_data, benchmark_code_sha256=partial_map, public_benchmark_code_sha256=partial_map)
        rcpt_link.write_text(json.dumps(partial_rcpt))
        verify_results.OUT = res_copy
        verify_results.audit()
        payload2 = json.loads((res_copy / 'verification.json').read_text())
        assert payload2['checks']['benchmark_sources_match_receipt'] is False
        assert payload2['all_integrity_and_dose_checks_passed'] is False
    finally:
        verify_results.OUT = old_out
        shutil.rmtree(res_copy, ignore_errors=True)
