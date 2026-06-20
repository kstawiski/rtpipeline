# B3 — verbatim approved-design excerpts (for in-worktree reviewer verification)

Source: `/umed-projekty/KOPERNIK/MIEDNICE/results/rtpipeline_p5/consensus/phaseBE_design_gate_20260610T220341Z/`
(DESIGN.md = v1, DESIGN_v2.md = canonical, DESIGN_v3.md = post-round-2, carries B3 UNCHANGED). Copied here because the design docs live outside this worktree's sandbox; this lets reviewers verify B3 design-fidelity without MIEDNICE access. Quoted exactly.

## Architecture principle 5 (DESIGN.md §0)
> 5. **Methodological forks are flagged for PI, not silently chosen** (§B2 MTV/TLG; §B1 SMI-without-height; §B3 registration). Default = the literature/DATA_GUIDE standard, implemented + gated + flagged.

## B3 — DESIGN.md (v1), §1, lines 33-37 (verbatim)
> ### B3 — functional-MR sampling under anatomic masks  [GAP-3; new stage]
> - **Design (per §0.3).** `mr_functional` is classified (`modality_classifier.py:186`) + routed to `MR_functional/<uid>/DICOM` (`inventory.py:308`) with `ts_task=none` (never segmented). New `rtpipeline/mr_functional.py`: for each mr_functional series, locate the anatomic MR (same patient) whose `total_mr` masks exist; register (shared FoR + same grid → direct; same FoR diff grid → resample NN; else **rigid registration** via SimpleITK, per §0.3 "rigid-reg fallback"); resample masks to the functional grid; compute per-structure **ADC mean/median/p10/p90** (for ADC/DWI) and **perfusion-map summary stats** (mean/median for scanner DCE maps TTP/PEI/WO/WI/SUB). Output `mr_functional_structures.csv`.
> - **Bounded.** Full DCE pharmacokinetic modeling (Ktrans/Ve/Kep + AIF) is **explicitly deferred** (§0.3 future spin-off) — do NOT implement. Only sample the existing scanner-derived maps + ADC.
> - **Flag.** Registration robustness when FoR differs (rigid fallback quality) — default rigid (mutual-information, SimpleITK); flag that non-FoR-matched functional series get a registration-QC field and are excluded from the primary table if registration fails. **Flag for PI** (acceptable for v1?). Note: the §4-matrix functional-MR row already records the WO/WI/SUB regex-extension as B3 work — extend `_MR_FUNCTIONAL_RE` (`modality_classifier.py:63`) to add `wo|wi|sub|subtraction` tokens here.
> - **Tests:** synthetic ADC + same-FoR mask → exact percentiles; rigid-reg path on a shifted phantom; regex extension matches WO/WI/SUB.

## B3 — DESIGN_v2.md (canonical), lines 36-37 (verbatim — supersedes v1 on the points it states)
> ### B3 — functional-MR sampling under anatomic masks
> As v1 (per §0.3: per-structure ADC mean/median/p10/p90 + scanner DCE-map stats under `total_mr` masks; DCE PK modeling deferred), plus: **registration is net-new** (no `ImageRegistrationMethod` exists in the worktree) — rigid MI (SimpleITK) only when FoR differs; **resample the functional map onto the anatomic-mask grid** (NOT masks onto the coarse functional grid — avoids truncating small pelvic structures to empty); empty-mask/registration-failure → QC flag + exclude from the primary table. Regex extension `\b`-anchored (`\bwo\b|\bwi\b|\bsub|subtraction`), case-tested against the P0 MR inventory.

## B3 — DESIGN_v3.md, line 5 (carried UNCHANGED, verbatim)
> B3 (functional-MR sampling; `\b`-anchored WO/WI/SUB regex + P0-inventory case-test; resample functional→anatomic grid; DCE-PK deferred)

---

## Note on v1↔v2 reconciliation (orchestrator)
v1 line says "resample **masks to the functional grid**" with "resample NN"; v2 **reverses** this to "resample the **functional map onto the anatomic-mask grid**" (rationale: avoid truncating small pelvic structures). v3 confirms v2 ("resample functional→anatomic grid"). **The canonical direction is functional-map → anatomic-mask grid.** The v1 "resample NN" referred to the (superseded) mask→functional direction; for a continuous functional map resampled onto the anatomic grid, linear interpolation is the correct *kind* (NN is label-only). The plan's brief follows v2/v3.
