# B2 — per-structure PET SUV under `total` masks (implementation brief)

**Status:** PLAN — pending triple-consensus plan gate.
**Branch:** `feat/phaseB-d1-b1-b3-b2` (after 0ee8893 = B3). Next/final per-item imaging build (DESIGN_v3 seq …D1→B1→B3→**B2**).
**Source of truth:** `results/rtpipeline_p5/consensus/phaseBE_design_gate_20260610T220341Z/DESIGN_v2.md` §1 B2 + DESIGN_v3 ("B2-per-structure-SUV (Auriac; SUVpeak in PET physical space; min-volume QC)"). Grounded in **Auriac 2025 (PMID 40993940)** — per-anatomic-structure SUV under TotalSegmentator masks.

**Scope:** implement ONLY B2. Surgical; opt-in; read real code first. Apply B3 lessons (manifest `"series"` key; anchor tests on the real schema; exact-value synthetic tests; complete QC taxonomy; ambiguity-not-silent-pick; opt-in default-off; never break existing callers).

---

## 0. What B2 is
For each **PET series** with a paired **PET-CT CT-component** (`petct_ct`, same `study_uid`+`frame_of_reference_uid`) whose TotalSegmentator `total` masks exist, compute **per-structure SUVmax, SUVpeak, SUVmean, volume_ml** under each `total` structure, from the already-written SUVbw NIfTI. Output per-series `pet_suv_structures.json` sidecar + cohort `Data/pet_suv_structures.csv`. One terminal QC + ≥1 row per PET series. **MTV/TLG is OUT OF SCOPE — PI-gated, default defer (do NOT ship).**

## 1. Foundations (verified via scope map)
- SUVbw NIfTI per PET series: `all_series/NIFTI/SUV/{safe_series}_SUVbw.nii.gz` + `.provenance.json` (`pet_suv.py:1333`); RAS affine in PET physical space (`_affine_from_sorted_datasets` :746, incl. dcm2niix voxel reorder :778) — the affine for SUVpeak sphere math.
- Existing `ingest_pet_suv_for_patient` (`pet_suv.py:1174`) iterates `manifest["series"]`, `image_class=="pt"`, statuses in `PET_SUV_CANDIDATE_STATUSES`; produces the SUVbw NIfTI. **B2 runs AFTER it** (SUV must exist) AND after all-series segmentation (petct_ct `total` masks must exist).
- `petct_ct` ← CT with same `(study_uid, FoR)` as a PT series (`inventory.py:156-184`); `TS_TASK_BY_CLASS["petct_ct"]="total"` (:21) → in default all-series segment scope → `total` masks at `_series_artifact_dirs(petct_ct_dir)[1]/<base_name>/total--multilabel.nii.gz` (+`total--segmentations.json`), the D1 `{model}--multilabel` convention (segmentation.py:789).
- Manifest top-level key is **`"series"`** (inventory.py:326) — the B3 BLOCKER; B2 reads `"series"` + `isinstance(r, dict)`. No existing per-structure SUV code (greenfield).

## 2. Part — `rtpipeline/pet_structures.py` (new) + opt-in wiring

### 2.1 Config + wiring
- `config.pet_suv_structures: bool = False` (opt-in). Wired in the `if getattr(cfg,"do_ingest_pet_suv",False):` block (`cli.py:1402`), AFTER the PET-ingest tasks join: gated `if getattr(cfg,"pet_suv_structures",False):` → per-patient (parallel, mirror existing `_PetSuvTask`) + serial stage-end CSV aggregation. **Precondition guard (warn-loud):** requires `do_ingest_pet_suv` (SUV NIfTI) AND `petct_ct` in the effective all-series segment scope (`total` masks); if unmet → every PET series → `petct_ct_masks_missing`/`suv_nifti_missing` QC. A B2 failure never aborts the run / never flips a series to a failed state.

### 2.2 Per-patient algorithm
1. Enumerate `manifest["series"]` (guard dict): PET rows (`image_class=="pt"`, SUV-eligible status); `petct_ct` rows with `output_dir`.
2. For each PET row: load its SUVbw NIfTI (from `output_dir`/SUV path); missing → `suv_nifti_missing`.
3. **Pair to petct_ct (D-1):** candidates = `petct_ct` rows with same non-empty `(study_uid, FoR)`. 0 → `no_petct_ct`; 1 → use; >1 → if distinguishable (n_slices/geometry) pick deterministically (largest n_slices), else genuine tie → `ambiguous_petct_ct` (audit, excluded). (Mirror B3's never-silently-pick-a-tie.)
4. Read the petct_ct `total` masks (`read_total_ct_label_image`, mirrors B3's `read_total_mr_label_image` but `total--multilabel.nii.gz`); missing → `petct_ct_masks_missing`.
5. **Resample masks → SUV grid (D-2):** PET & petct_ct share FoR → resample the CT-grid label image onto the SUV grid with **nearest-neighbor** (identity transform, same physical space). Sample SUV under each label on the SUV grid.
6. **Per-structure stats (D-3..D-5):** on the SUV array under each label:
   - `SUVmax` = max; `SUVmean` = mean; `n_suv_voxels` = in-structure SUV-grid voxel count.
   - `SUVpeak` = **PERCIST 1 cm³ sphere** mean: sphere radius `r=(3/(4π))^{1/3} cm ≈ 6.20 mm`, centered on the hottest in-structure SUV voxel's physical center; mean of SUV voxels whose physical center lies within the sphere (sphere is the VOI per PERCIST — NOT clipped to the structure; clamp to image bounds). Computed in PET physical space via the SUV affine/spacing.
   - `volume_ml` = **native CT-grid** mask voxel count × CT voxel volume / 1000 (accurate structure volume; not the coarser SUV grid).
   - `min_volume_flag` (bool annotation) when `n_suv_voxels` below threshold (partial-volume guard for tiny ROIs); empty (0) → `empty_mask` row qc.
7. Write per-series sidecar (rows + pairing/QC) + accumulate cohort rows.

### 2.3 Outputs
`Data/pet_suv_structures.csv` (one row per PET-series × structure): `patient_id, pet_series_uid, pet_series_description, petct_ct_series_uid, pairing_basis, structure_name, suvmax, suvpeak, suvmean, volume_ml, n_suv_voxels, min_volume_flag, qc_flag, rtpipeline_version`. Atomic write (unique tmp + replace, per B1/B3). Excluded series emit one qc row (stats null). **Completeness (best-effort):** B2's denominator is the **SUV-eligible** PET series (status ∈ {suv_computed, suv_skipped_idempotent} — a SUVbw NIfTI exists); upstream-excluded/failed/pending PT rows are out of scope (not re-flagged). Each in-scope series gets one terminal qc + ≥1 CSV row; a per-patient failure or unreadable sidecar is surfaced via loud logging (not a runtime-enforced set-equality).

### 2.4 QC taxonomy (complete)
`ok, no_petct_ct, ambiguous_petct_ct, petct_ct_masks_missing, suv_nifti_missing, empty_mask, load_failed`. Annotation column: `min_volume_flag` (bool; never a qc_flag value).

## 3. Tests (rtpipeline env; PYTHONPATH override; anchor on the REAL `"series"` schema)
1. Synthetic SUV volume + same-grid sphere mask → **exact** SUVmax/SUVmean; n_suv_voxels.
2. SUVpeak sphere math: known hot voxel + uniform neighborhood → exact 1 cm³-sphere mean; sphere clamped at image edge; sphere not clipped to structure (PERCIST).
3. masks→SUV-grid NN resample (CT grid ≠ SUV grid, same FoR) → labels land correctly; volume_ml from native CT grid.
4. Pairing: 1 petct_ct → ok; 0 → `no_petct_ct`; 2 tied → `ambiguous_petct_ct`; 2 distinguishable → pick larger n_slices.
5. QC: missing SUV nifti → `suv_nifti_missing`; missing masks → `petct_ct_masks_missing`; empty structure → `empty_mask`; tiny ROI → `min_volume_flag`.
6. Completeness (PET series-UID set-equality manifest↔CSV); opt-in default-off (flag absent → no CSV); idempotent re-run; manifest `"series"` key (production schema, not `"rows"`).

## 4. Decisions flagged for reviewer scrutiny
- **D-1** ambiguous petct_ct pairing → `ambiguous_petct_ct` (vs pick largest). Right call for clinical correctness?
- **D-3** SUVpeak sphere = PERCIST 1 cm³, **not** clipped to the structure boundary (sphere centered on hottest in-structure voxel may include adjacent voxels). DESIGN_v2 documented-approximation note; confirm clinical acceptability.
- **D-4** `volume_ml` from native CT grid (accurate) vs SUV-grid count. SUV stats from SUV grid. Confirm this split is sound.
- masks→SUV NN resample assumes shared FoR (PET-CT). If FoR matches but grids differ, NN is correct for labels. Confirm.

## 5. Out of scope (mention, do not implement)
**MTV/TLG** (PI-gated; DATA_GUIDE §6 amend + PI sign-off required; needs target-VOI + threshold — DEFAULT DEFER). SUVpeak true peak-maximizing sphere (documented approximation used). Cross-vendor SUV harmonization. Any change to D1/B1/B3/pet_suv ingest outputs.
