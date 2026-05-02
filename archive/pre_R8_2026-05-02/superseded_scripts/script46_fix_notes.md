# Script 46 Fix Notes

## Scope

This write-up covers the fixed implementation staged at:

- `/umed-projekty/rtpipeline/manuscript/scripts/46_stratified_shape_icc_FIXED.py`
- `/umed-projekty/rtpipeline/manuscript/scripts/run_script46_shape_strat_FIXED.sh`

The original `46_stratified_shape_icc.py` source was not present in this workspace, so the fixed version was reconstructed from:

- `/umed-projekty/rtpipeline/manuscript/analysis/orchestrator_review_synthesis_2026-04-18.md`
- `/umed-projekty/rtpipeline/manuscript/analysis/codex_review_batch2.md`
- adjacent manuscript scripts (`33`, `36`, `37`, `40`)
- the pipeline ICC contract in `/umed-projekty/rtpipeline/rtpipeline/radiomics_robustness.py`
- the ICC design notes in `/umed-projekty/rtpipeline/docs/features/radiomics_robustness.md`

## Implemented fixes

### P0-1: post-rebuild point estimates

- The fixed script no longer trusts a stale `icc_results.parquet`.
- It blocks on a defensive wait gate before reading point estimates:
  `analysis/data/icc_results.parquet` must have mtime >= `2026-04-18T00:00:00+00:00`.
- The wait gate is framed explicitly as the rebuild artifact for job `3560738`.
- Point estimates are then read from the rebuilt table and shape-family rows are re-annotated as `scale_invariant`, `scale_dependent`, or `unclassified`.

### P0-2: genuine patient bootstrap

- The bootstrap no longer resamples feature rows.
- Raw patient-level rows are read from `analysis/data/robustness_by_cohort/<cohort>.parquet`.
- Patients are the resampling unit.
- When a patient is drawn, all of that patient's course-level subject rows are retained for every relevant `(roi_name, feature_name)` cell.
- ICC(3,1) is recomputed from patient-resampled sufficient statistics, not inherited from the rebuilt point table.
- CoV is recomputed from the same patient-resampled sufficient statistics, so the summary CI path propagates patient composition into both ICC and CoV.
- Aggregation order matches the task brief:
  per `(cohort, roi_name, scale_class)` median across features, then median across cohort-ROI cells to body-region and overall summaries.

### P1-1: honest Wilcoxon labeling

- I did **not** fabricate a pseudo feature-level pairing between scale-invariant and scale-dependent features.
- The fixed script reports an explicitly labeled:
  `cluster_median_paired_wilcoxon`
- Pairing unit:
  `(cohort, roi_name)`
- The output note explains why a one-to-one feature-level pairing is not well-defined for this comparison.

### P2-1: manifest seed

- The manifest now records `args.seed`.
- It also records requested versus effective bootstrap iteration count, which matters when `--smoke-test` reduces the iteration count.

## Additional implementation choices

- Shape-class mapping follows the plan/report language:
  `sphericity`, `compactness1`, `compactness2`, `sphericalDisproportion`, `elongation`, `flatness` -> `scale_invariant`
  `voxelVolume`, `meshVolume`, `surfaceArea`, `major/minor/leastAxisLength`, `maximum2DDiameter*`, `maximum3DDiameter` -> `scale_dependent`
- Any remaining shape features stay visible as `unclassified` rather than being silently dropped.
- The bootstrap path uses patient-level sufficient statistics to keep memory bounded under the requested `64G` SGE allocation.
- The script logs RSS after raw-batch loads and during bootstrap progress so Argos reruns can check memory behavior directly.
- The raw patient-level recomputation is compared back to the rebuilt point table; the script fails if rebuilt coverage is too low or the ICC mismatch is gross (`coverage < 95%` or `max |ΔICC| > 0.05`).

## Validation completed here

- `python3 -m py_compile /umed-projekty/rtpipeline/manuscript/scripts/46_stratified_shape_icc_FIXED.py`
- `bash -n /umed-projekty/rtpipeline/manuscript/scripts/run_script46_shape_strat_FIXED.sh`

## Validation not completed here

- No smoke execution was run in this sandbox.
- Reason 1: the target analysis data live under `/home/kgs24/rtpipeline_manuscript/...`, which was not readable from this session.
- Reason 2: the local `python3` here does not have `pyarrow`, so parquet streaming could not be exercised end-to-end.

## Self-review against the task checklist

- Does the bootstrap resample actual patients?
  Yes. Patient IDs are the resampling unit; all their course-level rows are included after sampling.

- Does the script wait for the post-rebuild input?
  Yes. It blocks on `icc_results.parquet` mtime before reading point estimates.

- Does the raw recomputation use the original ICC formula?
  Yes in spirit and formula. The bootstrap recomputation uses ICC(3,1) on balanced complete cases with subjects defined by `patient_id + course_id`, matching the documented contract in `radiomics_robustness.py` and `docs/features/radiomics_robustness.md`.

- Is the Wilcoxon output still overclaimed?
  No. It is labeled as a cluster-median paired Wilcoxon, with the averaging spelled out in the output.
