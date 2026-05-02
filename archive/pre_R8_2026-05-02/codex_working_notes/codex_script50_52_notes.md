# Codex Script 50 + 52 Notes

## Files Written

- `/umed-projekty/rtpipeline/manuscript/scripts/50_shared_feature_bridge.py`
- `/umed-projekty/rtpipeline/manuscript/scripts/52_structure_fragility_atlas.py`
- `/umed-projekty/rtpipeline/manuscript/scripts/run_script50_shared_bridge.sh`
- `/umed-projekty/rtpipeline/manuscript/scripts/run_script52_fragility_atlas.sh`
- `/umed-projekty/rtpipeline/manuscript/scripts/codex_script50_52_notes.md`

## Smoke-Validation Outputs Written

- `/umed-projekty/rtpipeline/manuscript/analysis/data/shared_feature_bridge_nsclc_io.parquet`
- `/umed-projekty/rtpipeline/manuscript/analysis/tables/shared_feature_bridge_summary.csv`
- `/umed-projekty/rtpipeline/manuscript/analysis/tables/shared_feature_bridge_vs_full.csv`
- `/umed-projekty/rtpipeline/manuscript/analysis/figures/figure_shared_bridge.png`
- `/umed-projekty/rtpipeline/manuscript/analysis/figures/figure_shared_bridge.pdf`
- `/umed-projekty/rtpipeline/manuscript/analysis/logs/50_manifest_codex_smoke50b.json`
- `/umed-projekty/rtpipeline/manuscript/analysis/data/structure_fragility_atlas.parquet`
- `/umed-projekty/rtpipeline/manuscript/analysis/tables/structure_fragility_summary.csv`
- `/umed-projekty/rtpipeline/manuscript/analysis/tables/worst_structure_by_family.csv`
- `/umed-projekty/rtpipeline/manuscript/analysis/figures/figure_fragility_atlas.png`
- `/umed-projekty/rtpipeline/manuscript/analysis/figures/figure_fragility_atlas.pdf`
- `/umed-projekty/rtpipeline/manuscript/analysis/logs/52_manifest_codex_smoke52c.json`

## Methods Applied

### Script 50

- Loads the section-30 within-patient deviations parquet, constructs `paired_delta_cov_pct = human_cov_pct - ai_cov_pct`, and adds explicit `cohort = NSCLC_Interobserver` / `body_region = Thorax`.
- Loads the 107-feature overlap from step 33 when that parquet is readable; in full mode this is mandatory.
- Recomputes full-set and overlap-set summaries for `overall + each family`: median Human CoV, median AI CoV, median paired delta, percent of pairs favouring AI, Wilcoxon signed-rank p-value.
- Computes hierarchical bootstrap CIs for the median paired delta. Because NSCLC_IO is a single cohort, the cohort→subject bootstrap reduces to a subject-cluster bootstrap in practice.
- Emits the required side-by-side comparison table, including `effect_size_retention_fraction` and `effect_size_retention_pct` for each family and overall.
- Writes a grouped boxplot figure comparing the full section-30 delta distribution against the 107-feature overlap subset.

### Script 52

- Aggregates `perturbation_variance_decomposition.csv` to `body_region x roi_name x feature_family`.
- Computes median and 95th percentile for the Volume / Translation / Contour / Residual variance components per atlas cell.
- Computes structure-level robustness prevalence from `icc_results.parquet` as `robust_feature_fraction = fraction(ICC >= 0.90)` within each `region x structure x family` cell.
- Aggregates `segmentation_qc.csv` to structure-level empty / cropped / multicomponent rates and merges those into the atlas.
- Flags variance fragility via bottom-decile robustness or top-tail volume variance, and flags joint fragility when that overlaps with high QC burden.
- Emits:
  - full atlas parquet
  - top-10 / bottom-10 ranking table per family x region
  - worst structure per family x region
  - region-faceted heatmap of median volume variance

## Decisions Made Where The Prompt Was Underspecified

- Script 50 output parquet:
  I stored the per-subject paired overlap rows, not a 107-row feature-only table. This preserves the actual paired observations used for Wilcoxon and bootstrap, while still restricting the feature universe to the 107-feature bridge set.

- Script 50 effect-retention definition:
  `effect_size_retention_fraction = overlap median paired delta / full median paired delta`, computed per family and overall.

- Script 50 smoke mode:
  if the step-33 parquet is unavailable from the current worker, smoke mode uses a deterministic synthetic overlap subset drawn from the local deviations parquet so the full code path still executes. Full mode still requires the real step-33 parquet.

- Script 52 robust-feature fraction:
  I used `ICC(3,1) >= 0.90` because `icc_results.parquet` does not expose paired CoV columns, so the full `ICC >= 0.90 AND CoV <= 10%` rule cannot be reconstructed from that input alone.

- Script 52 QC/variance flags:
  I used region-specific 75th-percentile cutoffs for QC burden and family-region bottom-decile / top-tail cutoffs for variance burden. This gives a relative fragility flag without hard-coding arbitrary absolute thresholds.

- Script 52 figure:
  the heatmap uses median Volume variance, because the stated clinical story arc is specifically about the risk of overgeneralizing the volume bottleneck.

## Blockers / Review Items

- `/home/kgs24/rtpipeline_manuscript/analysis/data/sigma_calibration_nsclc_io.parquet` and `/home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet` were not readable from this workspace. Smoke validation therefore used synthetic fallbacks only for the missing inputs. Full qsub runs still depend on the real Argos files.

- The section-30 deviations parquet available here has no `course_id`. Script 50 therefore falls back to `subject = patient_id` and records that explicitly in the manifest.

- The main methodological blocker is in script 52:
  `perturbation_variance_decomposition.csv` is already aggregated at `feature x structure x cohort`, so a literal subject-level bootstrap on the Volume variance component is not recoverable from the declared inputs alone. I therefore implemented a cohort-resampled feature-row bootstrap fallback and recorded that in the manifest. If the orchestrator wants a true subject-cluster CI for script 52, the script should be upgraded to ingest a subject-level variance source instead of the already-aggregated decomposition table.

## Validation

- `python3 -m py_compile` passed for:
  - `/umed-projekty/rtpipeline/manuscript/scripts/50_shared_feature_bridge.py`
  - `/umed-projekty/rtpipeline/manuscript/scripts/52_structure_fragility_atlas.py`

- `bash -n` passed for:
  - `/umed-projekty/rtpipeline/manuscript/scripts/run_script50_shared_bridge.sh`
  - `/umed-projekty/rtpipeline/manuscript/scripts/run_script52_fragility_atlas.sh`

- Smoke tests completed successfully with local mirrors plus the explicit smoke-only synthetic fallbacks:
  - `PYTHONPATH=/tmp/codex_pydeps python3 /umed-projekty/rtpipeline/manuscript/scripts/50_shared_feature_bridge.py --smoke-test --bootstrap-iterations 10 --job-id codex_smoke50b`
  - `PYTHONPATH=/tmp/codex_pydeps python3 /umed-projekty/rtpipeline/manuscript/scripts/52_structure_fragility_atlas.py --smoke-test --bootstrap-iterations 10 --job-id codex_smoke52c`
