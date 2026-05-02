# R8 Phase 5 Directory Hygiene Migration Log

Date: 2026-05-02
Branch: `fix/public-rtpipeline-reliability`

Scope: directory hygiene for the IJROBP v3-r8 submission package at
`manuscript/submission/ijrobp-v3-r8/`. Files were moved, not deleted.

## Summary

| Category | Count | Destination | Rationale |
| --- | ---: | --- | --- |
| Root Snakefile backup variants | 3 | `archive/pre_R8_2026-05-02/root_obsolete_files/` | `Snakefile.modified`, `Snakefile.orig`, and `Snakefile.temp` are March backup/working copies; active code uses root `Snakefile`. |
| Pre-R7 QA files | 5 | `archive/pre_R8_2026-05-02/pre_R7_qa/` | Files dated before 2026-04-21 and not cited from the current R8 manuscript package. |
| PLAN history files | 4 | `archive/pre_R8_2026-05-02/manuscript_PLAN_history/` | Older manuscript planning snapshots were removed from the active manuscript directory; `PLAN_v5.md` remains current. |
| Obsolete v3-r8 top-level variants | 3 | `archive/pre_R8_2026-05-02/ijrobp_v3_r8_obsolete_variants/` | R7 baseline manuscript variants and redundant cover-letter copy were present beside current R8 files. |
| Superseded manuscript generator scripts | 41 | `archive/pre_R8_2026-05-02/superseded_scripts/` | Older untracked/script-lane generators and wrappers superseded by the Phase 3 R8 script set under `manuscript/analysis/qa/r8_consensus/scripts/`. |
| Superseded Codex working notes | 2 | `archive/pre_R8_2026-05-02/codex_working_notes/` | Historical script-lane implementation notes were removed from `manuscript/scripts/`; they are not active package-generation inputs. |
| Residual v2 non-baseline artifacts | 172 | `manuscript/submission/archive/20260502_ijrobp_v2_residuals/` | Legacy standalone v2 figure exports and R6 Zenodo-deposit working copy were outside the active v3-r8 bundle and not needed as the R7 baseline package. |

Total archive inventory touched: 230 files. Of these, 229 files were moved
from active locations in this pass; `PLAN_v4.md` was already present under
archive and was registered with the PLAN history set.

## Files Moved

### Root Snakefile Backup Variants

- `Snakefile.modified` -> `archive/pre_R8_2026-05-02/root_obsolete_files/Snakefile.modified`
- `Snakefile.orig` -> `archive/pre_R8_2026-05-02/root_obsolete_files/Snakefile.orig`
- `Snakefile.temp` -> `archive/pre_R8_2026-05-02/root_obsolete_files/Snakefile.temp`

### Pre-R7 QA Files

- `manuscript/analysis/qa/clinical_plausibility_findings.csv` -> `archive/pre_R8_2026-05-02/pre_R7_qa/clinical_plausibility_findings.csv`
- `manuscript/analysis/qa/clinical_plausibility_summary.md` -> `archive/pre_R8_2026-05-02/pre_R7_qa/clinical_plausibility_summary.md`
- `manuscript/analysis/qa/report_consistency_check.csv` -> `archive/pre_R8_2026-05-02/pre_R7_qa/report_consistency_check.csv`
- `manuscript/analysis/qa/traceability_claims.csv` -> `archive/pre_R8_2026-05-02/pre_R7_qa/traceability_claims.csv`
- `manuscript/analysis/qa/traceability_figure_table.csv` -> `archive/pre_R8_2026-05-02/pre_R7_qa/traceability_figure_table.csv`

### PLAN History Files

- `manuscript/PLAN.md` -> `archive/pre_R8_2026-05-02/manuscript_PLAN_history/PLAN.md`
- `manuscript/PLAN_DRAFT_v0.md` -> `archive/pre_R8_2026-05-02/manuscript_PLAN_history/PLAN_DRAFT_v0.md`
- `manuscript/PLAN_v2.0_full_history.md` -> `archive/pre_R8_2026-05-02/manuscript_PLAN_history/PLAN_v2.0_full_history.md`
- `archive/pre_R8_2026-05-02/manuscript_PLAN_history/PLAN_v4.md` was retained in the archive as an older PLAN snapshot.

### Obsolete v3-r8 Top-Level Variants

- `manuscript/submission/ijrobp-v3-r8/03_manuscript_anonymized.docx` -> `archive/pre_R8_2026-05-02/ijrobp_v3_r8_obsolete_variants/03_manuscript_anonymized.docx`
- `manuscript/submission/ijrobp-v3-r8/03_manuscript_anonymized.md` -> `archive/pre_R8_2026-05-02/ijrobp_v3_r8_obsolete_variants/03_manuscript_anonymized.md`
- `manuscript/submission/ijrobp-v3-r8/cover_letter_v3.docx` -> `archive/pre_R8_2026-05-02/ijrobp_v3_r8_obsolete_variants/cover_letter_v3.docx`

### Superseded Manuscript Generator Scripts

Moved 41 files from `manuscript/scripts/` into
`archive/pre_R8_2026-05-02/superseded_scripts/`: legacy numbered
generators `01`, `21`, `26`, `32`-`46`, `50`-`53`, `64`-`67`, their
matching `run_script*.sh` wrappers, and the associated script fix notes.

Active R7/R8 package-generation scripts `68`-`82`, current Figure 1-4
generators, `fig_utils_postcddp.R`, dataset config files, and broad
execution/staging wrappers were left in place because they remain package
provenance or operational inputs.

### Superseded Codex Working Notes

- `manuscript/scripts/codex_script_writes_summary.md` -> `archive/pre_R8_2026-05-02/codex_working_notes/codex_script_writes_summary.md`
- `manuscript/scripts/codex_script50_52_notes.md` -> `archive/pre_R8_2026-05-02/codex_working_notes/codex_script50_52_notes.md`

### Residual v2 Non-Baseline Artifacts

- `manuscript/submission/ijrobp-v2/figures/` -> `manuscript/submission/archive/20260502_ijrobp_v2_residuals/figures/`
- `manuscript/submission/ijrobp-v2/zenodo_deposit_r6/` -> `manuscript/submission/archive/20260502_ijrobp_v2_residuals/zenodo_deposit_r6/`

The R7 baseline `manuscript/submission/ijrobp-v2/submission_package_v3/`
was not moved because it is the stated comparator baseline and is still
referenced by R8 provenance scripts.

## Explicit Non-Moves

- `manuscript/PLAN_v5.md` remains current.
- `output_format.md` and `output_format_quick_ref.md` remain at repository
  root because they are referenced by `mkdocs.yml`, `docs/`, and setup
  scripts.
- No `core/` directory exists. The root `core` entry is an ignored sparse
  crash dump file, not an R7-or-earlier output directory, so it was not
  moved into the committed archive set.
