# Script 41 fix notes

## Scope-cut decisions

### P0-1 Odbytnice target ROI

- **Option taken:** **Option B**. Odbytnice was dropped from the production prognostic-retention analysis.
- **Why:** the accessible Odbytnice RTSTRUCT radiomics state was not production-ready at cohort level. I found only partial per-course `radiomics_ct.xlsx` exports and no populated cohort-level merged radiomics table that could support a defensible GTV/rectum analysis. The sacrum surrogate is therefore removed from production scope rather than reused.
- **Production implication:** no Odbytnice DFS or pCR leg is dispatched by `41_prognostic_retention_FIXED.py`. There is no production path that maps Odbytnice to sacrum.

### P0-2 Prostata OS

- **Option taken:** **Option B**. Prostata OS was dropped from the production prognostic-retention analysis.
- **Why:** the accessible audited clinical sources exposed `death_date` but no observed last-follow-up / censoring date field. The prior `RT_end + 365d` fallback was removed entirely instead of being replaced by another synthetic censoring rule.
- **Production implication:** the fixed script runs only the Prostata G2+ lymphopenia toxicity leg.

## Data sources inspected

### Review instructions

- `/umed-projekty/rtpipeline/manuscript/analysis/orchestrator_review_synthesis_2026-04-18.md`
- `/umed-projekty/rtpipeline/manuscript/analysis/codex_review_batch2.md`

### Prostata OS audit

- `/projekty/Prostata/LK/Data/bdo.tsv`
  - Accessible. Contains `patient_id`, `RT_start_date`, `RT_end_date`, `death_date`, `ctcae_2/3/4`, `ctcae_3/4`.
  - No observed last-follow-up / censoring-date field found.
- `/projekty/Prostata/LK/Data/os.tsv`
  - Accessible. Contains `patient_id`, `RT_start_date`, `death_date`.
  - No observed last-follow-up / censoring-date field found.
- `/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/analytic_cohort.parquet`
  - Exact-path checks from this environment hung and could not be completed reliably.
- `/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.csv`
  - Accessible local mirror/fallback of the analytic cohort.
  - Contains `lymphopenia_g2`, `lymphopenia_g3`, `death`, `death_date`, `death_date_parsed`, but no observed last-follow-up / censoring-date field.

### Prostata toxicity audit / fallback radiomics sources

- `/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.csv`
  - Used to confirm the G2+ endpoint columns exist in an analysis-ready clinical file.
- `/projekty/Prostata/Data_raw/DICOM_rtpipeline/Output/_ANALYSIS_READY/radiomics_ct_filtered.xlsx`
  - Accessible aggregated radiomics export with `patient_id`, `course_id`, `roi_name`, and wide radiomics feature columns.
  - Used as a local fallback source path in the fixed script when `/home/kgs24/.../radiomics_ts_all.parquet` is unavailable.

### Odbytnice audit

- `/umed-projekty/ODBYTNICE2026/work/pipeline_output/*/*/radiomics_ct.xlsx`
  - Accessible. Only partial per-course exports were present.
- `/umed-projekty/ODBYTNICE2026/work/pipeline_output/_RESULTS/radiomics_ct.xlsx`
  - Accessible but empty as a cohort-level merged radiomics result.
- `/umed-projekty/ODBYTNICE2026/work/pipeline_logs/radiomics/*.log`
- `/umed-projekty/ODBYTNICE2026/work/pipeline_logs/radiomics_robustness/*.log`
  - Confirm RTSTRUCT/GTV-like structures exist in the source workflow, but not as a completed cohort-level prognostic-analysis input.
- `/home/kgs24/rtpipeline_manuscript/Odbytnice/`
  - Exact-path checks from this environment hung and could not be completed reliably.

### ICC source

- `41_prognostic_retention_FIXED.py` resolves the latest existing exact-path `icc_results.parquet` among the known analysis roots and records the chosen path, mtime, and SHA-256 in the manifest.
- This satisfies the defensive requirement to use the rebuilt file if it is already present when the script runs.

## Code changes implemented

- Removed any production Odbytnice leg from the fixed script.
- Removed any fabricated `default_censor = rt_end + 365d` logic. There is no synthetic censoring path in the fixed script.
- Restricted production analysis to `prostata_toxicity_g2plus`.
- Moved feature prescreening inside each outer CV training fold.
- Capped the fold-level selected feature count by the training-fold events-per-feature budget (`floor(train_events / 10)`).
- Added pooled random-effects log-OR summaries by robustness class and by robustness-class × family.
- Changed behavior to fail nonzero on any production-leg blocker unless `--allow-leg-failure` is explicitly supplied.

## Open orchestrator decisions

- If the real `/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/analytic_cohort.parquet` later proves to contain observed last-follow-up fields, the Prostata OS leg can be reinstated, but that should happen in a separate reviewed revision rather than silently inside this fix.
- If a cohort-complete Odbytnice RTSTRUCT/GTV radiomics export becomes available, the Odbytnice DFS / pCR legs can be re-opened, again in a separate reviewed revision.
- If neither prerequisite becomes available, manuscript text should explicitly state that script 41 production evidence is limited to the Prostata toxicity endpoint and that Odbytnice / Prostata OS remain unavailable because the required clinical-imaging linkage inputs were not complete.

## Odbytnice CTV1 reopened + logistic overflow fix

- Added `aggregate_odbytnice_targets.py` to convert the available per-course Odbytnice `radiomics_ct.xlsx` files into a cohort parquet of harmonized target ROIs (`CTV1`, `CTV2`, `GTVp`, `PTV1`, `PTV2`), plus a per-patient target-availability CSV and a manifest that explicitly counts CTV1 availability.
- Replaced the narrow fixed script with `41_prognostic_retention.py`, which reintroduces Odbytnice as a production target on `CTV1` only. There is no sacrum surrogate code path anywhere in the new script.
- The current rebuilt cohort ICC artifact still contains only the prior Odbytnice pelvic-OAR set (`sacrum`, bladder, bowel, femora, glutei) and no `CTV1` rows. The new script therefore records a concrete `icc_lookup` blocker for Odbytnice legs when the target-ROI robustness classes are still missing, instead of silently falling back to sacrum or to another ROI.
- Odbytnice clinical linkage now uses:
  - `/umed-projekty/ODBYTNICE2026/data_bucket/ClinicalData20260304/final/odbytnice_clinical_2026.csv`
  - `/umed-projekty/ODBYTNICE2026/analysis/patient_id_pesel_mapping.csv`
  - optional audit-only reference to `/umed-projekty/ODBYTNICE2026/analysis/aim3_analysis_dataset.parquet`
- OS and DFS timing are derived from real observed dates with `treatment.rcht_start_date` as the time origin:
  - OS: `death_date` vs `last_followup_date`
  - DFS: earliest observed recurrence/metastasis/regrowth/death date vs `last_followup_date`
  - No synthetic censoring defaults are used.
- All logistic model fits now guard against separation/overflow:
  - first attempt: `statsmodels.Logit`
  - fallback when warnings/instability occur: `sklearn.linear_model.LogisticRegression(penalty="l2")`
  - every fit records `fit_status` (`converged`, `separation_warning`, or `failed`) and a warning/error message
  - production is configured to fail nonzero if more than 50% of logistic univariate features hit `separation_warning`
- Cox fits now use `lifelines.CoxPHFitter(penalizer=0.01)` to reduce divergence risk in low-event settings.
- Feature prescreening for panel models now happens inside the outer CV training folds only, with feature count capped at `floor(train_events / 10)` per fold.
- The new manifest records per-leg exclusion counts, including the number of subjects lost because CTV1 is unavailable or because the PESEL/clinical linkage is missing.
- The smoke path for Prostata now uses per-course `radiomics_ct.parquet` files under `/projekty/Prostata/Data_raw/DICOM_rtpipeline/Output/*/*/` instead of the 335 MB cohort workbook. This keeps smoke validation fast while leaving the standard production candidate order intact.
- Smoke run `smoke_odbytnice_fix4` completed successfully for `prostata_toxicity_g2plus` and produced the standard output parquet/CSV set. Odbytnice legs were skipped cleanly with `stage=icc_lookup` blockers because the current ICC artifact still lacks `CTV1`; the manifest still records CTV1-linked subject counts after clinical linkage and endpoint filtering.
