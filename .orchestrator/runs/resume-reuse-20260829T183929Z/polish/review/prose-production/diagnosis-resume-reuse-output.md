# Resume should reuse current content

The clinical processing question is whether a resumed course needs a new segmentation model run. The answer must depend on the current planning CT and the artefacts derived from it, not on whether a stage sentinel happens to exist. This distinction matters because a missing RS_auto can coexist with a complete, valid bank of TotalSegmentator masks.

After the automatic RTSTRUCT geometry fix, 121 DFCI courses and 1 Kopernik course reportedly contained all 117 TotalSegmentator masks but no RS_auto. The reported inventory was 16,424 DFCI mask files and 234 Kopernik mask files. The missing derived structure set did not show that the masks were missing or invalid.

Clearing `.segmentation_done` to rebuild RS_auto made TotalSegmentator run again. Recent courses on s1 produced 55 model-run log lines and took 322 to 494 seconds per course while 117 masks were already present. At roughly seven minutes per course, repeating the model across 230 DFCI courses would take about 27 hours. Repeating it across 122 Kopernik courses would take about 14 hours. These are reported cohort calculations, not a new campaign-time measurement.

The defect was a stage-level completion marker being used as an artefact-level decision. `.segmentation_done` records that a stage published completion. It does not prove that each output exists, is readable, is current, or corresponds to the inputs selected for the present course. The sentinel remains a workflow marker. It is not evidence that an expensive model run is necessary.

The revised segmentation decision starts with the authoritative course contract. The contract identifies the planning CT series and its NIfTI. A reusable mask manifest must then identify that same series, match the current planning-NIfTI hash, and list a complete set of readable masks whose physical geometry matches the current planning NIfTI. If any identity, inventory, readability, geometry, or freshness check is uncertain, the model runs again. This is deliberately fail-closed because reusing a mask from another planning CT is more harmful than recomputing a valid mask.

Legacy mask banks receive only a narrow upgrade path. The sidecar series identity and NIfTI hash must match the current planning CT and NIfTI. When a sidecar contains the NIfTI content-generation timestamp, the mask manifest must be later than that timestamp. Older sidecars do not use their metadata write time as evidence. They require filesystem ordering that shows the mask manifest was written after the current NIfTI. An uncheckable ordering rejects reuse.

A refreshed metadata sidecar therefore does not make valid legacy masks stale merely because its own `generated_at` is newer than the mask manifest. When the NIfTI is reused, its sidecar preserves the existing generation timestamp. A newly converted NIfTI receives a separate `nifti_generated_at` value that describes content generation rather than a later metadata refresh.

The stage now separates model outputs from derived outputs. A complete current mask set is reused without a TotalSegmentator invocation. If the model RTSTRUCT or RS_auto is missing, it is rebuilt from the validated masks. Current and legacy binary-mask naming conventions are normalized in one pass. If both names identify one ROI, the current name wins, so a stale duplicate cannot create a second contour. A genuine duplicate receives a progressing deterministic suffix.

An existing RS_auto that is invalid, stale, or bound to another planning CT loses eligibility before replacement begins. The rejected file is quarantined, and only a successfully published replacement is eligible for downstream radiomics. If rebuilding fails, the audit records failure and the rejected file is not enumerated as a current automatic source. RS_custom follows the same rule. Production calls require a loadable authoritative course contract. Contract uncertainty is not converted into permission to reuse an old custom structure set.

Each course records the decision in `metadata/segmentation_resume.json`. The record names the planning CT identity and records the action, model-run status, artefact, and reason. Model runs are recorded only after the attempt resolves. A QC-ineligible model is `skipped`, and a failed invocation is `failed`, rather than either being reported as rebuilt. A reuse-only run leaves the mask manifest generation time unchanged. Its audit update is separate from mask provenance. Earlier skipped-model information is retained when a manifest is rewritten.

The direct measurement for this revision used the production mask-currentness predicate on Kopernik course 431057, course 2020-07. The course had 117 existing masks occupying 23,201,947 bytes. Five repeated checks had a median wall-clock time of 7.64 seconds. Compared with the measured 322 to 494 seconds for a model rerun, validating this complete mask set saves 314.36 to 486.36 seconds per applicable course. The denominator is one course with a complete, current mask set and missing derived output. The measurement is recorded by `analysis/resume_reuse_evidence.py` in `analysis/resume-reuse-measurement.json`.

This revision applies the content boundary to TotalSegmentator masks and the RS_auto and RS_custom derived structure sets. It does not claim that every stage sentinel is now safe for content-based reuse. The campaign registry still lists `.organized`, `.segmentation_done`, `.custom_models_done`, `.crop_ct_done`, `.dvh_done`, `.radiomics_done`, `.radiomics_robustness_done`, and `.qc_done`. The remaining stages still use their sentinel workflow gates. Extending them requires stage-specific output inventories and dependency identities, so that work is deferred rather than represented as complete. In particular, the NTCV robustness chain remains outside this repair. Its reported scale is 81 perturbations across 5 target ROIs, or 405 perturbed extractions per course, at roughly 0.9 hours per course.

Resume completeness is unchanged. A conversion that never completed remains ineligible for a resume skip. A checkpoint without a required adjudication remains rejected. The existing completeness tests and the new content-reuse tests provide the focused regression boundary. The operational next step is to rerun organize so each course receives its authoritative contract. Then inspect the per-course decisions before expensive stages proceed. Reuse only positively identified current outputs. Rebuild anything missing, stale, mismatched, incomplete, or uncertain.

## Sources

[1] Version-3 repair packet at `.orchestrator/runs/resume-reuse-20260829T183929Z/packets/repair-material-findings.json` and the task packet in the same run directory. It supplies the measured cohort evidence and the updated course-contract context.

[2] `rtpipeline/segmentation.py`. It implements mask completeness, planning-CT correspondence, legacy provenance, selective derived-artefact rebuilding, and resume-decision records.

[3] `rtpipeline/auto_rtstruct.py` and `rtpipeline/custom_structures_rtstruct.py`. They implement RTSTRUCT correspondence, binary-mask normalization, stale-output revocation, and custom-structure currentness.

[4] `analysis/resume_reuse_evidence.py` and `analysis/resume-reuse-measurement.json`. They produce and record the read-only course measurement used for the per-course saving calculation.

[5] `workflow/scripts/campaign_ledger.py`. It is the campaign sentinel registry used to state which stages remain outside this content-based repair.

[6] `tests/test_resume_reuse_content.py` and `tests/test_resume_completeness.py`. They test selective reuse, planning-CT mismatch, incomplete masks, derived-output handling, audit integrity, and existing resume-completeness guarantees.
