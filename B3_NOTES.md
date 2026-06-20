# B3 — implementation notes (functional-MR sampling under anatomic masks)

Implements `B3_IMPLEMENTATION_BRIEF.md` rev5 (plan-gate all-pass: Codex+Claude+Gemini,
`triple_consensus_logs/B3_plan_gate_20260620T160645Z/ledger_allpass.json`). Branch
`feat/phaseB-d1-b1-b3-b2`, after a3274e4 (B1). Implementation pending its own triple-consensus gate.

## What was built

**Part A — classifier regex (surgical).** `modality_classifier.py`: `_MR_FUNCTIONAL_RE` extended with
`\bwo\b|\bwi\b|\bsub\b|\bsubtract` (cross-cohort-safe; comment documents the opposite anchoring
rationale vs the substring `dwi`/`adc`). Reclaims 3 in-cohort DCE-subtraction maps currently
misrouted to `mr_anatomic` (verified against the 2,696-description P0 inventory). Tests in
`test_modality_classifier.py` (+ token-level P0 case-test + sub-prefixed-anatomy rejection).

**Part B — `rtpipeline/mr_functional.py` (new, ~720 lines) + opt-in wiring.**
- `route_functional_subtype` — DEFAULT-DENY 3-bucket token matching (adc/dwi substring;
  perf/dyn/diff/ep2d/subtract `\b`-prefix; wo/wi/sub/twist/grasp + DERIVED_MAP standalone).
  Raw-`perf` defers BEFORE the DWI branch (ep2d_perf≠dwi); MIP two-stage; only
  adc|perfusion|subtraction|dwi sampled, rest → audit QC rows.
- `resample_functional_to_anatomic` — direct / same-FoR linear / rigid-MI (Euler3D, Mattes 32-bin,
  multi-res, deterministic sampling); functional→anatomic-grid LINEAR; acceptance = convergence +
  transform-plausibility bounds + (via `coverage_fraction`) mask coverage; reg-failure → QC.
- `per_structure_stats` — mean/median(+p10/p90 for adc/dwi); `n_native_voxels` (NN mask→F) +
  `low_native_voxels` flag; `deformable_flag` on rigid tier for bladder/rectum/bowel; empty→QC.
- `select_anatomic_for_functional` — geometry-first (shared FoR → same study → tiebreak);
  `ambiguous_anatomic_match` / `no_anatomic_match` QC.
- `load_functional_volume` (dcm2niix; single-scalar-3D gate → 4D/non-scalar QC), units provenance
  (`_units_provenance`: RWVM/RescaleType else subtype convention; rescale baked once by dcm2niix),
  `read_total_mr_label_image` (minimal `total_mr--multilabel.nii.gz` reader, total_mr-scoped),
  `sample_patient_mr_functional` (orchestration; one terminal series-QC + ≥1 row each),
  `write_mr_functional_structures_csv` (atomic; per-structure rows).
- `config.mr_functional_sampling: bool = False` (opt-in); `cli.py` stage-end wiring inside
  `do_segment_all_series` with a warn-loud precondition guard (mr_anatomic must be in the effective
  segment scope, else functional series → `anatomic_out_of_scope`). A B3 failure never flips a
  series to `seg_failed`; idempotent sidecars (skip unless `--force`).

## Tests
`tests/test_mr_functional_b3.py` (52 tests after R1 remediation) + classifier tests. Full Phase-B suite: **364 passed,
1 skipped** (no regression from the shared config/cli edits). Coverage: routing incl. the R3/R4
regressions (ep2d_perf-defers, twist-not-perfusion via `wi`⊄`twist`, vendor-fused adc/dwi, MIP
two-stage); exact-percentile direct tier; **rigid-MI phantom recovers a known translation @ >0.9
coverage**; QC paths; anatomic selection; mask reader; units provenance recorded; orchestrator
ok/excluded/out-of-scope/idempotent paths; series-UID set-equality completeness; opt-in default-off.

## Known deviations from the approved brief (flagged for the implementation gate)

1. **Anatomic-selection affine-similarity ranking → simplified.** rev5 §4.2-step5 specifies ranking
   multi-candidate anatomics by grid/affine similarity to F (+ acq-time). The orchestrator currently
   resolves the COMMON case exactly via shared-FoR match, and for residual ties falls back to the
   documented deterministic `n_slices` tiebreak; full affine-similarity ranking is NOT wired (it
   requires loading every candidate's geometry — a dcm2niix per candidate — which is expensive, and
   `select_anatomic_for_functional` already accepts an `affine_similarity` callable so it can be added
   without an interface change). **Question for reviewers:** acceptable v1 (FoR-match + n_slices +
   `ambiguous_anatomic_match`), or must the orchestrator load candidates and rank by affine similarity?

2. **Per-patient execution is serial, not parallel.** rev5 §4.1 / Claude-r2 #10 asked to parallelize
   per-patient (mirror `_AllSeriesSegmentTask`) because rigid-MI is heavy. The cli wiring currently
   loops `patient_ids` serially (then serial CSV aggregation). Rationale for v1: the heavy rigid-MI
   path fires only for different-FoR functional series; most in-cohort functional series share the
   anatomic FoR (same MR session) → cheap direct/linear resample. **Question for reviewers:** is the
   serial loop acceptable for v1 (correct, simpler), or should it use the parallel task pattern given
   full-cohort scale (~6k patients)?

Both are correctness-preserving (no silent data loss; every series gets a terminal QC + ≥1 CSV row).
No D1/B1/B2/E1/E3 surface was touched beyond the one-line regex and the opt-in stage.

## Implementation-gate Round-1 remediation (all three reviewers HOLD → fixed)

Gate dir `triple_consensus_logs/B3_impl_gate_20260620T190720Z/`; remediation `REMEDIATION_R1.md`.
Full suite after fixes: **364 passed / 1 skipped**. Fixes:
- **#0 [BLOCKER, Claude] wrong manifest key** — orchestrator now reads `"series"` (inventory.py:326),
  not `"rows"`; iteration guards `isinstance(r, dict)`. Test fixtures rebuilt on the production
  `"series"` key (they previously encoded `"rows"` → false green).
- **A [Gemini] register vs anatomic INTENSITY, not the mask** — orchestrator loads the anatomic
  intensity NIfTI (`_load_anatomic_intensity`) as the rigid-MI fixed image; the mask supplies only
  the sampling grid/labels. rigid-MI without intensity → `reg_failed`.
- **B [Codex] ambiguity contract** — `select_anatomic_for_functional` now emits
  `ambiguous_anatomic_match` when top candidates are tied within tolerance (affine-sim + n_slices);
  affine-similarity (physical bbox-overlap vs F) is wired for multi-candidate same-FoR selection;
  unapproved silent-pick fallbacks removed.
- **C [Codex+Gemini] enforce acceptance** — reject (`reg_failed`) when `not converged` (rigid) OR
  UNION-mask coverage < `MIN_COVERAGE_FRAC`; coverage now computed over all nonzero mask voxels and
  stored in sidecar/CSV.
- **D [Codex] no silent CSV drop** — unreadable sidecars are `log.error`-surfaced + counted.
- **E [Codex+Claude] units** — convention-permissive retained (maximize data utilization); the
  `unit_source` enum is **{rwvm | rescale_type | convention | none}** (`convention` disclosed);
  `unknown_units` is a RESERVED defensive status (every sampled subtype has a documented unit
  convention, so it is not auto-triggered in v1).
- **F** tests added: ambiguity, affine-tiebreak, low-coverage→reg_failed, register-vs-intensity,
  native-count, `load_functional_volume` 3D/multivolume/not-materialized, unit_source=convention,
  CSV-unreadable-logged.
- **G** serial per-patient loop retained (both reviewers accepted for v1); documented as a deferred
  scalability item (parallelize before broad cohort use), not a faithfulness claim.
