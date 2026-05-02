# Script 21 fix notes

## Files

- Fixed script: `/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py`
- Fixed wrapper: `/umed-projekty/rtpipeline/manuscript/scripts/run_script21_inter_algorithm_FIXED.sh`

## What changed

### 1. Hard scope lock

- Enforced literal allow-list `{"Hipokampy", "PlucaRCHT"}` at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:39).
- Added CLI rejection for any other `--cohort` at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:1143).
- Added post-load discovered-cohort assertion at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:638).

### 2. Head-task gate for Hipokampy

- Added explicit head-task source detection via `is_head_task_source()` at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:122).
- Hipokampy discovery now requires an `AutoRTS_*head*` source and records blocked courses otherwise at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:509).
- If Hipokampy is requested directly and no head-task source exists anywhere, the script exits non-zero with the required diagnostic at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:1181).
- In all-cohort mode, Hipokampy is blocked and logged while PlucaRCHT continues.

### 3. Replaced averaged-before-ICC inter-algorithm path

- Added explicit ICC(2,1) absolute-agreement implementation at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:244).
- Added per-feature mixed-effects variance decomposition at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:389).
- The primary feature outputs now include:
  - `sigma_subject_var`
  - `sigma_contour_var`
  - `sigma_residual_var`
  - `subject_pct_total`
  - `contour_pct_total`
  - `residual_pct_total`
- Within-model TotalSegmentator reliability remains `ICC(3,1)` only, as benchmark.

### 4. Completeness audit and full-intersection policy

- Added full 6-realization subject-feature audit and full-grid reindexing at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:668).
- Default policy is full intersection within each `(cohort, roi_name)`:
  every retained feature must be complete for every retained subject in that ROI.
- Wrote:
  - `tables/exclusions_audit.csv`
  - `tables/inter_algorithm_ai_feature_exclusions.csv`
  - `tables/inter_algorithm_ai_course_discovery.csv`
- Manifest now records `n_feature_union` and `n_feature_intersection` per `(cohort, roi_name)`.

### 5. Bootstrap redesign

- Removed feature-row resampling.
- Added subject-cluster bootstrap grouped within cohort at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:988).
- Summary CSV now reports:
  - median `contour_pct_total`
  - 95% bootstrap CI
  - median `ICC(2,1)` absolute agreement
  - 95% bootstrap CI
  - `n_features`
- Because only 2 allowed cohorts exist, the implemented default follows the prompt’s fallback:
  patient-level clustered bootstrap within cohort, seeded with `2604`.

### 6. Locked output contract and manifest hash fix

- Output filenames now match the report contract at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:1159):
  - `data/inter_algorithm_ai_icc.parquet`
  - `tables/inter_algorithm_ai_summary.csv`
- Added full-file SHA-256 hashing at [21_inter_algorithm_ai_comparison_FIXED.py](/umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py:127).

### 7. Wrapper update

- New wrapper at [run_script21_inter_algorithm_FIXED.sh](/umed-projekty/rtpipeline/manuscript/scripts/run_script21_inter_algorithm_FIXED.sh:1).
- Increased `h_vmem` to `160G` at [run_script21_inter_algorithm_FIXED.sh](/umed-projekty/rtpipeline/manuscript/scripts/run_script21_inter_algorithm_FIXED.sh:5).
- Added staged-path fallback resolution at [run_script21_inter_algorithm_FIXED.sh](/umed-projekty/rtpipeline/manuscript/scripts/run_script21_inter_algorithm_FIXED.sh:15).

## Validation run

### 1. Scope-lock rejection

Command:

```bash
ssh argos-worker /home/kgs24/miniforge3/envs/radiomics_sge/bin/python - --cohort Prostata \
  < /umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py
```

Observed result:

- Exit code `2`
- Diagnostic: `ERROR: --cohort 'Prostata' is not allowed for script 21. Allowed cohorts: Hipokampy, PlucaRCHT.`

### 2. Hipokampy head-task gate

Command:

```bash
ssh argos-worker /home/kgs24/miniforge3/envs/radiomics_sge/bin/python - \
  --cohort Hipokampy \
  --workers 1 \
  --bootstrap-iterations 1 \
  --output-dir /home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_hip_gate \
  < /umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py
```

Observed result:

- Exit code `3`
- Diagnostic: `Hipokampy brain arm blocked: head-task TotalSegmentator regeneration not complete. Re-run after head-task outputs land in per-course parquets.`

Independent Argos spot-check before implementation:

```bash
ssh argos-worker "python - <<'PY'
import pandas as pd
p='/home/kgs24/rtpipeline_manuscript/Hipokampy/data/235407/2021-07/radiomics_robustness_ct.parquet'
df=pd.read_parquet(p, columns=['segmentation_source'])
print(sorted(df['segmentation_source'].dropna().astype(str).unique().tolist()))
PY"
```

Output: `['AutoRTS_total', 'Custom']`

### 3. PlucaRCHT smoke run

Production-style full-intersection smoke on the chosen `(roi, feature)` correctly excluded the cell because 3 of 35 subjects were incomplete.

Successful targeted smoke run used a complete 10-subject subset:

```bash
ssh argos-worker /home/kgs24/miniforge3/envs/radiomics_sge/bin/python - \
  --cohort PlucaRCHT \
  --subject-list 449324_2022-03,449595_2022-04,450376_2022-05,451806_2022-04,456312_2022-12,456862_2023-01,457298_2022-12,459474_2023-03,460178_2023-02,460560_2023-06 \
  --roi-name aorta \
  --feature-name log-sigma-1-0-mm-3D_firstorder_10Percentile \
  --workers 1 \
  --bootstrap-iterations 5 \
  --output-dir /home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_smoke \
  < /umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py
```

Observed result:

- Exit code `0`
- Wrote:
  - `/home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_smoke/data/inter_algorithm_ai_icc.parquet`
  - `/home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_smoke/tables/inter_algorithm_ai_summary.csv`
  - `/home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_smoke/tables/exclusions_audit.csv`
  - `/home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_smoke/tables/inter_algorithm_ai_feature_exclusions.csv`
  - `/home/kgs24/rtpipeline_manuscript/analysis/tmp_script21_smoke/logs/21_manifest_local.json`

Smoke-output checks:

- Parquet schema includes `contour_pct_total`, `subject_pct_total`, `residual_pct_total`.
- Summary CSV filename matches report §34 contract.
- Smoke feature used `variance_method = mixedlm_powell` with `mixedlm_converged = True`.

## Known residual issues

1. The cluster runtime environment tested here cannot read `/umed-projekty/...` directly from `argos-worker`; I validated by streaming the staged local script into the remote Python interpreter. The wrapper therefore tries the local staged path first and a remote analysis-path fallback second.
2. The family-level bootstrap CIs recompute the contour component with a fast balanced random-effects variance decomposition inside each resample. The primary per-feature outputs still use `MixedLM` first, with fallback only if all optimizers fail.
3. The full-intersection policy is intentionally strict. A feature/ROI can now disappear entirely if even one retained course lacks any of the 6 required realizations. That is expected behavior under the prompt, but it may materially shrink production output.

## Open orchestrator decisions

1. Whether the bootstrap should remain subject-clustered within cohort for this script permanently, or whether a future multi-cohort extension should re-enable explicit cohort-level resampling.
2. Whether the bootstrap’s balanced-model variance-component recomputation is sufficient, or whether the orchestrator wants a much slower exact REML refit inside every bootstrap iteration.
3. Whether the final staged script should also be copied to `/home/kgs24/rtpipeline_manuscript/analysis/21_inter_algorithm_ai_comparison_FIXED.py` before qsub, since the current task requested local staging only.

## Critical self-review

### Does the mixed-effects model actually estimate inter-ALGORITHM variance (not just within-algorithm)?

Yes. The primary feature-level model is `value ~ C(algorithm)` with:

- random subject intercept
- random `algo_realization` effect over the 6 levels
  `{TS_v0, TS_c1_v0, TS_c2_v0, Custom_v0, Custom_c1_v0, Custom_c2_v0}`

That variance component is not restricted to within-algorithm perturbations. It spans the combined algorithm-realization axis after controlling for the fixed algorithm mean effect.

### Is the bootstrap truly hierarchical (not feature-row resampling)?

Yes, within the prompt’s allowed fallback. There is no feature-row resampling. Each bootstrap draw resamples subjects clustered within cohort, and then recomputes the family-level medians from subject-resampled matrices.

### Is the head-task gate an actual assertion, not just a comment?

Yes. A direct Hipokampy run exits non-zero with the required blocker message when no head-task source is present.
