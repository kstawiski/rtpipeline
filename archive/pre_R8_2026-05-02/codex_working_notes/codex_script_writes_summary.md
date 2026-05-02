# Codex Script Write Summary

## Files Written

### Local staging deliverables

- `/umed-projekty/rtpipeline/manuscript/scripts/36_cov_threshold_deployment.py`
- `/umed-projekty/rtpipeline/manuscript/scripts/run_script36_cov_threshold.sh`
- `/umed-projekty/rtpipeline/manuscript/scripts/37_cov_upgrade_stratified.py`
- `/umed-projekty/rtpipeline/manuscript/scripts/run_script37_stratified_upgrade.sh`
- `/umed-projekty/rtpipeline/manuscript/scripts/40_null_feature_control.py`
- `/umed-projekty/rtpipeline/manuscript/scripts/run_script40_null_feature.sh`
- `/umed-projekty/rtpipeline/manuscript/scripts/codex_script_writes_summary.md`

### Argos smoke-validation outputs written during temporary validation

- `/home/kgs24/rtpipeline_manuscript/analysis/data/cov_threshold_crossings.parquet`
- `/home/kgs24/rtpipeline_manuscript/analysis/tables/cov_threshold_summary.csv`
- `/home/kgs24/rtpipeline_manuscript/analysis/figures/figure_cov_threshold.png`
- `/home/kgs24/rtpipeline_manuscript/analysis/figures/figure_cov_threshold.pdf`
- `/home/kgs24/rtpipeline_manuscript/analysis/logs/36_manifest_codex_smoke36.json`
- `/home/kgs24/rtpipeline_manuscript/analysis/data/feature_class_transitions.parquet`
- `/home/kgs24/rtpipeline_manuscript/analysis/tables/class_transitions_by_quartile.csv`
- `/home/kgs24/rtpipeline_manuscript/analysis/figures/figure_quartile_upgrade.png`
- `/home/kgs24/rtpipeline_manuscript/analysis/figures/figure_quartile_upgrade.pdf`
- `/home/kgs24/rtpipeline_manuscript/analysis/logs/37_manifest_codex_smoke37_v2.json`
- `/home/kgs24/rtpipeline_manuscript/analysis/data/null_feature_control.parquet`
- `/home/kgs24/rtpipeline_manuscript/analysis/tables/null_feature_control_summary.csv`
- `/home/kgs24/rtpipeline_manuscript/analysis/figures/figure_null_feature.png`
- `/home/kgs24/rtpipeline_manuscript/analysis/figures/figure_null_feature.pdf`
- `/home/kgs24/rtpipeline_manuscript/analysis/logs/40_manifest_codex_smoke40.json`

## Script Notes

### `36_cov_threshold_deployment.py`

- Method implemented:
  feature-level median within-patient CoV is computed across subjects, then threshold crossing is defined on that median for Human vs AI contouring. The summary table reports per-threshold × per-family feature percentages above threshold, percentile bootstrap CIs for `delta_pct = pct_human_above - pct_ai_above`, and a McNemar-style exact mid-p test.
- Explicit decisions for orchestrator review:
  the prompt did not specify how to collapse patient-level CoV values into a feature-level threshold crossing. I used the median across subjects because the input parquet is patient-feature level but the requested output is feature-level. A stricter alternative would be thresholding a higher quantile or a maximum across subjects; a looser alternative would be averaging patient-level feature shares.
- Additional implementation note:
  the requested output column name `wilcoxon_p_mid` is kept verbatim, but the stored statistic is a McNemar-style exact mid-p value, not a Wilcoxon p-value.

### `37_cov_upgrade_stratified.py`

- Method implemented:
  the script derives the locked common feature universe from `icc_results.parquet`, aggregates NSCLC interobserver CoV to feature-level medians, bins features into human-CoV quartiles, applies the requested Robust / Acceptable / Poor thresholds, and summarizes transition rates by quartile and family with patient-cluster bootstrap CIs.
- Explicit decisions for orchestrator review:
  `icc_results.parquet` alone is not sufficient for Human-vs-AI class assignment because it does not expose separate `human_icc` / `ai_icc` columns. I therefore made the script prefer `/home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_human_ai_contour_comparison.parquet` for arm-specific ICCs while still using `icc_results.parquet` to derive the locked public-cohort feature set.
- Explicit decisions for orchestrator review:
  the locked 107-feature public intersection shrinks to 106 usable features because `original_firstorder_Minimum` is absent from the arm-specific NSCLC ICC comparison parquet. The script drops that feature with a warning and records it in the manifest.
- Explicit decisions for orchestrator review:
  bootstrap uncertainty is propagated only through the CoV component. ICC values are held fixed across bootstrap replicates because only aggregated arm-specific ICC estimates are available. A better but broader alternative would be to recompute Human-vs-AI ICCs from raw patient-level matrices inside this script.
- Performance note:
  the first smoke version was too slow because it bootstrapped the full deviations table before feature filtering. I patched that so the bootstrap now runs on the filtered feature universe directly.

### `40_null_feature_control.py`

- Method implemented:
  null features are derived from the IBSI phantom table, paired Human-minus-AI CoV deltas are tested per null feature with Wilcoxon signed-rank, per-feature bootstrap CIs are computed on the median paired delta, and a group summary compares null vs non-null feature-level median deltas.
- Explicit decisions for orchestrator review:
  `ibsi_validation.csv` does not expose a direct family column, so first-order membership is derived from the PyRadiomics feature name prefix. The script restricts the null set to `config == "IBSI-compliant"` and `pct_deviation < 0.5`, which yields 16 unique first-order null features in the full dataset.
- Explicit decisions for orchestrator review:
  the summary contrast uses all remaining non-null features as the comparator set. A tighter, more family-matched alternative would be `null first-order` versus `non-null first-order` only. I did not switch to that narrower contrast because the prompt asked for a null vs non-null control at the full-analysis level.

## Dependencies And Data Gaps

- `/home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_within_patient_deviations.parquet` does not contain `course_id`. All three scripts therefore fall back to `subject = patient_id`, log that fact, and record it in the manifest.
- `/home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet` does not contain arm-specific ICC columns, so script 37 depends on `/home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_human_ai_contour_comparison.parquet`. If that file is absent and no equivalent arm-specific columns appear in `icc_results.parquet`, script 37 exits with a diagnostic instead of inventing a fallback.
- `/home/kgs24/rtpipeline_manuscript/analysis/tables/ibsi_validation.csv` does not contain an explicit `family` column, so script 40 infers first-order membership from `pyradiomics_feature`.
- The expected Argos NFS mirror path under `/home/kgs24/umed-projekty-link/rtpipeline/manuscript/scripts` was not present during validation. I therefore treated this as a local-staging delivery and validated by copying the scripts to `/tmp/codex_script_validation_rtmanuscript/` on `argos-worker`.

## Validation

- Local syntax validation passed:
  `python3 -m py_compile` on all 3 Python scripts.
- Local shell validation passed:
  `bash -n` on all 3 SGE wrappers.
- Argos smoke validation completed successfully:
  script 36 with `--smoke-test --bootstrap-iterations 50`
  script 37 with `--smoke-test --bootstrap-iterations 50` after the bootstrap filtering patch
  script 40 with `--smoke-test --bootstrap-iterations 50`

## Smoke-Test Recommendation

Run these first on Argos after placing the staged files into `/home/kgs24/rtpipeline_manuscript/analysis/`:

1. `/home/kgs24/miniforge3/envs/radiomics_sge/bin/python /home/kgs24/rtpipeline_manuscript/analysis/36_cov_threshold_deployment.py --smoke-test --bootstrap-iterations 50 --job-id smoke36`
2. `/home/kgs24/miniforge3/envs/radiomics_sge/bin/python /home/kgs24/rtpipeline_manuscript/analysis/37_cov_upgrade_stratified.py --smoke-test --bootstrap-iterations 50 --job-id smoke37`
3. `/home/kgs24/miniforge3/envs/radiomics_sge/bin/python /home/kgs24/rtpipeline_manuscript/analysis/40_null_feature_control.py --smoke-test --bootstrap-iterations 50 --job-id smoke40`

After those succeed, submit the full jobs with:

1. `qsub /home/kgs24/rtpipeline_manuscript/analysis/run_script36_cov_threshold.sh`
2. `qsub /home/kgs24/rtpipeline_manuscript/analysis/run_script37_stratified_upgrade.sh`
3. `qsub /home/kgs24/rtpipeline_manuscript/analysis/run_script40_null_feature.sh`
