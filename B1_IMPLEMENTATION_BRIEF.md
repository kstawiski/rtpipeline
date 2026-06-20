# B1 implementation brief — tissue_types/body routing + body-composition writer

**Dispatch ONLY after D1 is triple-consensus-approved and committed.** Worktree: `/umed-projekty/rtpipeline-wt-phaseB`. Env: `/home/konrad/micromamba/envs/rtpipeline/bin/python`.

## What B1 is (approved DESIGN_v2 §1 B1 + DESIGN_v3 carry-forward)
Add TotalSegmentator `tissue_types`/`body` tasks + a body-composition writer, gated by a new config key.

1. **Config key** `body_composition_classes` (in `rtpipeline/config.py`, default `None` = OFF; scale-up YAML will set `[planning_ct, diagnostic_ct, petct_ct]`; CBCT/4DCT excluded). **Wire it in the cli.py YAML→cfg overrides block (`cli.py:~1069-1129`)** — DESIGN_v3 notes the three new keys are currently NOT parsed there, so it would be silently ignored otherwise.
2. **Routing:** for series whose `image_class ∈ body_composition_classes`, run `tissue_types` and `body` as **additional** TS tasks (do not replace `total`). Route via the existing `TS_TASK_BY_CLASS` / segmentation task mechanism (`rtpipeline/inventory.py:18`, `rtpipeline/segmentation.py:901`). The `tissue_types`/`body` weights are already provisioned (`build.sh:172`).
3. **New `rtpipeline/body_composition.py`:** at the L3 mid-vertebral single slice, compute skeletal-muscle cross-sectional area (CSA) + muscle radiodensity (mean HU) + visceral (`torso_fat` ≈ VAT proxy — flag it as a proxy) + subcutaneous fat areas. Compute SMI = muscle_area / height² **only when DICOM `PatientSize` is present**; else set `smi = null` + an explicit `smi_missing_reason` field. Vertebral level selection must be explicit/auditable (e.g. from the `total` vertebrae masks).
4. **Output:** per-series `body_composition.json` + aggregated `Data/body_composition.csv` (provenance: patient_id, series_uid, image_class).
5. **License:** record that `tissue_types` is under the TotalSegmentator **non-commercial** license — add to the release-license boundary note (a comment/doc where other license notes live).

## Constraints
- Default OFF (`body_composition_classes=None`) ⇒ **zero behavior change** when unset. Add a test asserting the pipeline is unchanged when the key is unset.
- Surgical; do not touch D1/B3/B2/E1/E3. Read real code before editing (config.py, cli.py override block, inventory.py TS_TASK_BY_CLASS, segmentation.py task dispatch).
- Tests: routing adds tissue_types/body only for configured classes; body_composition.py math on a synthetic mask (known CSA); SMI null + reason when PatientSize absent; CSV/JSON schema. Keep all baseline tests green.
- Run `python -m pytest tests/ -q`; report count. Do NOT commit. Write `B1_NOTES.md` (file:line changes, deviations, pytest count). Ambiguity → STOP + write the question in B1_NOTES.md.
