# Orchestration closure report

Task: resume-reuse-round2-stale-rtstruct
State: RECHECKING
Specification digest: 685354e5154f879ef929de9311dc6c87c802ca081d68d3be77933f022d963d38
Workspace: /umed-projekty/rtpipeline
Executor family: openai-gpt
Executor family source: route_receipt
Final artifact digest: 38555cebd65d71dbf3f3946952ace1ea13f4b8eb3384f3034c4e92df35f9c7f3
Independent review families: claude, openai-codex, gemini
Review warnings: 0
Material findings recorded: 0
Non-material findings recorded: 7
Disputed findings: none
Unresolved findings: none
Explicit pre-review artifact baselines accepted: 0

## Recoverable interruption

Focused recheck responses were invalid: claude2-opus: Focused recheck included unrequested findings: ['openai-codex::RRC-002']

The phase remains nonterminal. Resume with the same semantic task specification after correcting the route or infrastructure problem.

## Artifact revisions

- executed: `af31d9f8e34857c6d7486cf710ab1d7e5a7f7246ba0873bafb6be76a8a7a54f4` with 110 tracked file(s)
- repaired: `38555cebd65d71dbf3f3946952ace1ea13f4b8eb3384f3034c4e92df35f9c7f3` with 110 tracked file(s)

## Consolidated repair

Family: openai-gpt
Summary: Resolved RRC-002. Invalid, stale, or mismatched RS_auto files are removed before rebuild work begins, the removal and reason are audited immediately, and downstream source selection requires validated correspondence to the contracted planning CT. Added regression coverage for failed rebuilds, audit persistence, stale-artifact removal, source exclusion, and valid reuse. Updated and polished the diagnosis documentation. Focused checks passed 55 tests. Full suite passed 988 tests with 2 skipped.
Tracked artifact changed: True
Added: none
Modified: docs/diagnosis-resume-reuse.md, rtpipeline/auto_rtstruct.py, tests/test_resume_reuse_content.py
Removed: none
Adapter-reported changed paths: docs/diagnosis-resume-reuse.draft.md, docs/diagnosis-resume-reuse.md, rtpipeline/auto_rtstruct.py, tests/test_resume_reuse_content.py

## Objective checks

### pre_review (passed)
- `deliverable_completeness`: pass; issues: 0
- `full-suite`: pass; issues: 0

### post_repair (passed)
- `deliverable_completeness`: pass; issues: 0
- `full-suite`: pass; issues: 0

## Review findings

### Reviewer family `claude` (route `claude2-opus`, decision: changes_required)

- **REV-001** [minor] method_validity at `rtpipeline/radiomics_robustness.py:1831-1836 (and rtpipeline/dvh.py:447-460); documented scope at docs/diagnosis-resume-reuse.md:19-23`. Outcome: non-material; not repaired.
  - Evidence: The presence-is-not-evidence rule was added only to `_standard_rtstruct_sources` (rtpipeline/radiomics.py:121-161, which now calls `_rtstruct_matches_planning_ct` and `_derived_rtstruct_dependencies_are_current`). Two other RS_auto consumers were not changed. `rtpipeline/radiomics_robustness.py:1831` reads `rs_auto = course_dir / "RS_auto.dcm"` and at line 1832 accepts it on `if rs_auto.exists():` alone, then loads its ROIs under the label `AutoRTS_total` with no series, frame, or dependency check. `rtpipeline/dvh.py:449` builds `(rs_auto if rs_auto.exists() else None, "AutoRTS")` and filters only with `_rtstruct_matches_dose_for(candidate, dose_for_uid)` (line 455), a FrameOfReferenceUID comparison against the contracted RTDOSE; the module's own `_select_seg_dir_for_ct` docstring (rtpipeline/auto_rtstruct.py:436-438) states that FrameOfReferenceUID does not separate distinct series that share a frame, such as 4DCT phases and multiple reconstructions. The removal-before-rebuild guarantee reaches these consumers only when `build_auto_rtstruct` runs, and it runs only inside the segmentation stage (rtpipeline/cli.py:173-179); docs/diagnosis-resume-reuse.md:29 records that `.segmentation_done` remains a sentinel workflow gate, and rtpipeline/cli.py:647-661 exposes `--stage dvh` and `--stage radiomics` as independently selectable, with rtpipeline/cli.py:1520 running only the requested stages.
  - Impact: A stale RS_auto bound to a different-but-same-frame planning CT series can still be consumed by the DVH stage and by the radiomics-robustness (NTCV) chain when those stages run without the segmentation stage in the same invocation, which is the exact downstream-consumption hazard the task describes. This is a residual gap outside the named scope of the finding under repair, not a regression: the Snakemake DAG orders `rule segmentation_course` before both consumers (Snakefile:758, 817, 876, 908, 1095), and docs/diagnosis-resume-reuse.md:29 already scopes the NTCV chain out of this repair.
  - Repair: Either extend the correspondence predicate to these consumers (call `_rtstruct_matches_planning_ct(rs_auto, contract.planning_ct['series_instance_uid'])` before loading at rtpipeline/radiomics_robustness.py:1832 and before appending the AutoRTS candidate at rtpipeline/dvh.py:449), or state explicitly in docs/diagnosis-resume-reuse.md that the new evidence requirement is confined to `_standard_rtstruct_sources` and that DVH and the robustness chain still accept RS_auto on presence plus a frame-level check only.
  - Verification: Add a regression test that writes an RS_auto referencing a different SeriesInstanceUID but carrying the planning CT's FrameOfReferenceUID, then asserts that neither `radiomics_robustness` nor the DVH AutoRTS candidate path loads its ROIs; alternatively, confirm the documentation change by reading the scope sentence in docs/diagnosis-resume-reuse.md. Re-run the full suite, which currently reports 988 passed, 1 skipped.

- **REV-002** [minor] reproducibility at `rtpipeline/auto_rtstruct.py:666-679 (`_reject_existing`), read against docs/diagnosis-resume-reuse.md:25`. Outcome: non-material; not repaired.
  - Evidence: `_reject_existing` deletes the rejected file at line 670 and then only assigns the `rejected_artifact` dict to a local closure variable at lines 675-679. The record is persisted solely by `_record_auto_resume_decision`, which is reached from `_failed` (line 658) and from the two success paths (lines 802 and 942). The rebuild path between the rejection and those points contains unguarded calls: `seg_res = _resample_to_reference(seg_img, ct_img)` (line 839), `seg_arr = sitk.GetArrayFromImage(seg_res)` (line 840) and `np.moveaxis` (line 841) are not inside a try block, and neither is `_load_seg_nifti` at line 815. docs/diagnosis-resume-reuse.md:25 states without qualification that 'When RS_auto is rejected, the record includes `rejected_artifact` with the `removed` action, the `RS_auto.dcm` path, and the exact rejection reason' and that 'The same decision records whether rebuilding then succeeded or failed.'
  - Impact: If any of those unguarded calls raises (for example `sitk.Resample` failing on a large multilabel volume, or a MemoryError in `GetArrayFromImage`/`moveaxis`), the exception escapes `build_auto_rtstruct` after the stale RS_auto has already been deleted. The removal is then irreversible and unrecorded: `metadata/segmentation_resume.json` contains no RS_auto entry at all, so the audit record does not state that the stale artifact was removed or why. The fail-closed disk state is still correct; only the audit trail the document promises is lost.
  - Repair: Persist the rejection at the moment of removal: call `_record_auto_resume_decision(course_dir, 'rejected', reason, rejected_artifact=rejected_artifact)` (or an equivalent immediate write) at the end of `_reject_existing`, before returning, and let the later `_failed`/`rebuilt` call overwrite the RS_auto entry with the terminal outcome. `record_segmentation_resume_decision` already merges per-artefact keys (rtpipeline/segmentation.py:1120-1124), so the second write updates rather than duplicates.
  - Verification: Add a test that monkeypatches `auto_rtstruct._resample_to_reference` to raise, starts from an RS_auto bound to a different planning CT series, asserts that `build_auto_rtstruct` propagates the exception, that `RS_auto.dcm` is absent, and that `metadata/segmentation_resume.json` already carries `decisions.RS_auto.rejected_artifact` with `action == 'removed'` and the rejection reason.

- **REV-003** [minor] evidence_traceability at `docs/diagnosis-resume-reuse.md:7`. Outcome: non-material; not repaired.
  - Evidence: The document states: 'At roughly seven minutes per course, repeating the model across 230 DFCI courses would take about 27 hours. Repeating it across 122 Kopernik courses would take about 14 hours.' The cited source, `.orchestrator/runs/resume-reuse-20260829T183929Z/packets/execute-hermes-sol-xhigh.json`, reads: 'At roughly 7 minutes per course and one segmentation worker, that is about 27 hours for the 230-course DFCI cohort and 14 hours for [the Kopernik cohort].' The condition 'and one segmentation worker' is dropped. It is a load-bearing condition, not a stylistic one: rtpipeline/cli.py:1600-1609 sets `seg_worker_limit` from the detected GPU count, so a multi-GPU host divides both wall-clock figures.
  - Impact: The 27-hour and 14-hour figures are the headline cost that motivates the repair. Presented without the single-worker condition they read as unconditional cohort wall-clock times, which overstates the cost of not reusing banked masks on any host with more than one segmentation worker.
  - Repair: Restore the condition, for example: 'At roughly seven minutes per course with one segmentation worker, repeating the model across 230 DFCI courses would take about 27 hours, and across 122 Kopernik courses about 14 hours.'
  - Verification: Read docs/diagnosis-resume-reuse.md:7 and confirm the single-worker condition appears alongside both figures; compare word-for-word against the 'MEASURED EVIDENCE' block of the cited packet.

- **REV-004** [minor] evidence_traceability at `docs/diagnosis-resume-reuse.md:29`. Outcome: non-material; not repaired.
  - Evidence: The document states: 'Its reported scale is 81 perturbations across 5 target ROIs, or 405 perturbed extractions per course, at roughly 0.9 hours per course.' The cited packet reads: '81 perturbations per ROI over 5 target ROIs, about 405 perturbed extractions per course'. Changing 'per ROI' to 'across' changes the denominator of the 81: 81 spread across 5 ROIs is 81 total, which contradicts the 405 stated in the same clause. `analysis/resume-reuse-evidence-ledger.json` claim C6 computes `perturbed_extractions_per_course` as `perturbations_per_roi * target_rois` (analysis/resume_reuse_evidence.py:226), confirming the intended reading is per-ROI.
  - Impact: A reader who takes 'across' at face value cannot reconcile 81 with 405 in the same sentence, and would misstate the per-ROI perturbation count of the NTCV chain by a factor of five when sizing that deferred work.
  - Repair: Replace 'across' with 'per ROI over', matching the source: 'Its reported scale is 81 perturbations per ROI over 5 target ROIs, or 405 perturbed extractions per course, at roughly 0.9 hours per course.' The same wording appears in the C6 statement string at analysis/resume_reuse_evidence.py:224 and should be corrected there for consistency.
  - Verification: Read docs/diagnosis-resume-reuse.md:29 and confirm 81 x 5 = 405 is recoverable from the sentence as written.

- **REV-005** [minor] integrity at `docs/diagnosis-resume-reuse.md:27 (final sentence)`. Outcome: non-material; not repaired.
  - Evidence: The document states: 'The denominator was one course with a complete, current mask set and missing derived output.' The measured course is named in the same paragraph and in analysis/resume-reuse-measurement.json as `/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output/431057/2020-07`. That course does have `RS_auto.dcm` on disk: 6,581,948 bytes, modified 2026-08-29 10:41, which predates the measurement recorded in analysis/resume-reuse-evidence-ledger.json (written 2026-08-29 20:39). The derived output actually absent there is the model RTSTRUCT: the segmentation directory `Segmentation_TotalSegmentator/MIEDNICA_3.0_B31f_3_20200707/` contains 117 `total--*.nii.gz` masks, `manifest.json` and `total--ts_version.json`, but no `MIEDNICA_3.0_B31f_3_20200707--total.dcm`.
  - Impact: The paragraph attributes the per-course saving to 'applicable courses', which the document defines at line 5 as those holding a complete mask set 'but no RS_auto'. The one course actually measured is not a member of that population as described, so the stated denominator misdescribes the evidence base. The timing itself is unaffected, since the predicate cost depends on the mask set rather than on which derived output is missing.
  - Repair: State which derived output was missing on the measured course, for example: 'The denominator was one course with a complete, current mask set whose model RTSTRUCT was absent while RS_auto was present.' If the intent was to measure a course matching the no-RS_auto population, re-run `analysis/resume_reuse_evidence.py --course <course>` against such a course and update the figures.
  - Verification: List the measured course directory and confirm the document's characterisation matches which of `RS_auto.dcm` and `<base>--total.dcm` are present; re-read docs/diagnosis-resume-reuse.md:27 against analysis/resume-reuse-measurement.json.

- **REV-006** [minor] evidence_traceability at `docs/diagnosis-resume-reuse.md:5`. Outcome: non-material; not repaired.
  - Evidence: The document states: 'After the automatic RTSTRUCT geometry fix, 121 DFCI courses and 1 Kopernik course reportedly contained all 117 TotalSegmentator masks but no RS_auto. The reported inventory was 16,424 DFCI mask files and 234 Kopernik mask files.' The cited packet writes the parenthetical without the noun: '(16,424 files on DFCI, 234 on Kopernik)'. Narrowing 'files' to 'mask files' makes the two sentences arithmetically incompatible as written: 1 Kopernik course said to hold 117 masks cannot hold 234 mask files, and 121 DFCI courses at 117 masks each is 14,157, not 16,424. For comparison, the one Kopernik course measured for this revision has 119 entries in its segmentation directory (117 masks plus `manifest.json` and `total--ts_version.json`).
  - Impact: A reader checking the reported cohort evidence finds a contradiction inside a single paragraph and cannot tell whether the counts, the per-course mask total, or the course counts are wrong. The document already hedges these as 'reported', so the defect is the added noun and the unreconciled arithmetic rather than a fabricated number.
  - Repair: Restore the source's noun and flag the discrepancy, for example: 'The packet reported 16,424 files on DFCI and 234 on Kopernik for that state; these totals do not divide evenly by the 117 masks per course and are reproduced as reported rather than reconciled.'
  - Verification: Compare docs/diagnosis-resume-reuse.md:5 word-for-word against the 'MEASURED EVIDENCE' block of `.orchestrator/runs/resume-reuse-20260829T183929Z/packets/execute-hermes-sol-xhigh.json` and confirm no count is narrowed or left silently contradictory.

- **REV-007** [note] reproducibility at `analysis/resume_reuse_evidence.py:43-79, reported at docs/diagnosis-resume-reuse.md:27`. Outcome: non-material; not repaired.
  - Evidence: The document says the measurement 'used the production mask-currentness predicate on Kopernik course 431057, course 2020-07'. The script does call the production predicate `segmentation._series_masks_current` (line 69), but it supplies that predicate's inputs from outside the contract path the document's thesis rests on: `nifti`, `ct_dir` and `seg_root` are read from `metadata/case_metadata.json` top-level keys (lines 45-49) rather than from `load_course_contract`, and `planning_ct_series_uid` comes from `_series_uid(ct_dir)` (line 63), which returns the SeriesInstanceUID of the first readable DICOM found by `sorted(ct_dir.rglob('*.dcm'))` (lines 31-40). The cited packet records that every existing course in both cohorts is rejected by `load_course_contract`, verified against this same course, which is why the contract path could not be used. The recorded output nevertheless carries the predicate's canned reason string 'complete masks match the contracted planning CT' (analysis/resume-reuse-measurement.json).
  - Impact: The document frames the revised decision as one that 'starts with the authoritative course contract' (line 11), and the recorded reason string says 'contracted planning CT', so the measurement reads as exercising the contract-driven path when the planning-CT identity was in fact supplied by a first-file heuristic over a legacy metadata key. The timing figure is unaffected, but a reader cannot tell from the document alone which identity source was used.
  - Repair: Add one sentence at docs/diagnosis-resume-reuse.md:27 recording that the banked course predates the course contract, so the planning-CT series identity was read directly from the CT directory rather than from `load_course_contract`, and that only the mask-currentness predicate itself is the production code path being timed.
  - Verification: Read analysis/resume_reuse_evidence.py:43-79 alongside docs/diagnosis-resume-reuse.md:27 and confirm the document names the identity source actually used.

### Reviewer family `openai-codex` (route `codex-delegate`, decision: pass)

No findings.

### Reviewer family `gemini` (route `agy-gemini`, decision: pass)

No findings.
