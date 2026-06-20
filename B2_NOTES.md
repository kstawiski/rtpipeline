# B2 — implementation notes (per-structure PET SUV under `total` masks)

Implements `B2_IMPLEMENTATION_BRIEF.md` (DESIGN_v2 §1 B2, Auriac 2025). Branch
`feat/phaseB-d1-b1-b3-b2` after 0ee8893 (B3). Pending implementation triple-consensus gate.

## What was built
- **`rtpipeline/pet_structures.py`** (new): `pair_petct_ct` (PET↔petct_ct by study+FoR; 0→no_petct_ct,
  1→use, tie→ambiguous_petct_ct, distinguishable→larger n_slices), `read_total_ct_label_image`
  (`total--multilabel.nii.gz`, mirrors B3's reader), `resample_mask_to_suv_grid` (NN, identity —
  shared FoR), `suvpeak_sphere` (PERCIST 1 cm³ sphere ≈6.20 mm on the hottest in-structure voxel in
  PET physical space, NOT clipped to the structure, clamped to image), `per_structure_suv`
  (SUVmax/SUVmean/SUVpeak on the SUV grid + `volume_ml` from the NATIVE CT grid + `min_volume_flag`),
  `sample_patient_pet_suv` (orchestration; one terminal QC + ≥1 row per PET series), and
  `write_pet_suv_structures_csv` (atomic; no-silent-drop logging).
- **Opt-in**: `config.pet_suv_structures: bool = False`; `cli.py` stage inside the
  `do_ingest_pet_suv` block (runs after PET-SUV ingestion + all-series segmentation), warn-loud if
  `petct_ct` not in the effective segment scope; per-patient serial + serial CSV aggregation;
  failure non-fatal. Reads manifest `"series"` key (B3 lesson).
- **MTV/TLG: OUT OF SCOPE** (PI-gated, deferred — not implemented).

## Tests — `tests/test_pet_structures_b2.py` (21). Full suite: **386 passed / 1 skipped**.
Exact SUVmax/SUVmean; SUVpeak uniform-field == value; SUVpeak single-hot-voxel = sphere mean < max
(PERCIST averaging) + not-clipped-to-structure; NN mask→SUV resample; pairing 1/0/tie/distinguishable
+ empty-FoR; empty-mask + min_volume_flag; mask reader; orchestrator ok/no_petct/suv_missing/
masks_missing/ambiguous; completeness (PET series-UID set-equality manifest↔CSV); idempotent;
CSV-unreadable-logged; opt-in default-off. Orchestrator fixtures use the production `"series"` key.

## Decisions flagged for the implementation gate (D-1/D-3/D-4 from the brief)
1. Ambiguous petct_ct pairing → `ambiguous_petct_ct` (never silently pick a tie). Right for clinical correctness?
2. SUVpeak = PERCIST 1 cm³ sphere centered on the hottest in-structure voxel, NOT clipped to the
   structure boundary (documented approximation vs the true peak-maximizing sphere; DESIGN_v2 note).
3. `volume_ml` from the native CT-grid mask (accurate structure volume); SUV stats from the
   SUV-grid-resampled mask; `min_volume_flag` when SUV-grid voxel count < 10.
4. Serial per-patient loop (v1, B3-consistent; documented scalability deferral).

No D1/B1/B3/pet_suv-ingest surface changed beyond the opt-in config flag + the gated stage.

## Implementation-gate Round-1 remediation (Codex+Claude HOLD → fixed; both confirmed code correct)
Gate dir `triple_consensus_logs/B2_impl_gate_20260620T220118Z/` (`REMEDIATION_R1.md`). Both HOLDs
verified the production code correct; the HOLD was test-adequacy + 1 status-filter fix. Applied:
- **SUV-eligibility status filter (Codex MAJOR):** B2 now scopes to PET rows with status ∈
  {`suv_computed`,`suv_skipped_idempotent`} (a SUVbw NIfTI exists). `suv_excluded`/`suv_failed`/
  pending rows are upstream-excluded → NOT B2's denominator (no false `suv_nifti_missing`).
- **Hardened geometric tests:** asymmetric exact SUVpeak (non-cubic+anisotropic+off-center; catches a
  z↔x transposition); hottest-voxel index-ordering; spatially-split-label resample placement;
  end-to-end orchestrator exact-value; real seg-dir mask-discovery (unpatched reader).
- **Status-filter regression + multi-PET completeness** tests added.
- Manifest missing/unreadable/malformed now `log.warning` with patient_id (no silent vanish);
  divergent inline `_safe_token` fallback dropped (let the import surface); oblique-bbox comment added.
- **Denominator/completeness wording**: B2 covers SUV-eligible PET series; completeness is
  best-effort ≥1 row per such series with loud logging (not a runtime-enforced set-equality).
Full suite after fixes: see below.
