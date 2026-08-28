# Diagnosis of course identity and radiotherapy linkage

RTpipeline treated directory shape, study membership, and first-file order as course identity. Those assumptions detached clinically delivered radiotherapy from its planning structures and suppressed supported courses in both cohorts. Explicit DICOM references now define the course and every attached radiotherapy object.

This repair changes the expected rebuild from thousands of CT-derived directories to about 230 DFCI entries and 122 Kopernik entries defined by planning structure sets. Those estimates do not yet equal independent delivered-treatment counts. The real cohorts were inspected read-only and were not rebuilt.

## Evidence and decision boundary

The DFCI export contains 154 patient directories and 154 DICOM PatientIDs. A header census found 1,561 RTPLAN objects, 407 distinct referenced RTSTRUCT objects, and 230 referenced RTSTRUCT objects with GTV, CTV, or PTV names. Every DFCI patient has at least 1 target-bearing planning structure set.

The completed DFCI output contains 3,588 course directories. None contains an RTSTRUCT, RTPLAN, or RTDOSE file. Its organize log records the decisive predicate at line 16. The code classified the source as a patient and series layout, then skipped all RT and registration scans.

The Kopernik source contains 92 patient directories, 375 RTPLAN objects, and 122 distinct target-bearing plan-referenced RTSTRUCT objects. Its current output contains 116 courses. Five patients have fewer output courses than target-bearing referenced structure sets.

A direct comparison of all 116 Kopernik output courses with their copied RTPLAN references found 38 correct RTSTRUCT selections and 78 wrong or missing selections. In 76 of 78 mismatches, the referenced RTSTRUCT contained more target volumes than the selected RTSTRUCT.

The evidence ledger separates source observations, calculations, and mechanistic inferences. It records read-only source paths, transformations, selected-patient checks, and the limits imposed by not rebuilding either cohort.

## Defect 1 selected the wrong RTSTRUCT

The pre-fix `link_rt_sets` function in `rtpipeline/metadata.py` at lines 37–68 indexed RTSTRUCT objects by FrameOfReferenceUID. It retained the first object for a shared frame, then attached that object to any dose and plan on the frame. Shared geometry therefore became false structure-set identity.

The pre-fix course writer added a second failure in `rtpipeline/organize.py` at lines 2305–2311. It selected `struct_candidates[0]`. If no candidate existed, it scanned same-study structures and stopped at the first match.

Both predicates fired on Kopernik. Patient 292929 had 5 plans that referenced 1 RTSTRUCT with 36 ROIs and 11 target volumes. The output copied a different 2-ROI RTSTRUCT containing no target. Patient 333944 had a referenced 15-ROI RTSTRUCT with 6 targets, but course 2018-05 copied no RTSTRUCT.

The 115-ROI auto-contour example confirms that ROI count is not a safe fallback. The referenced SOPInstanceUID is the identity. Structure-set size, target count, study equality, and filesystem order cannot replace that reference.

The repaired `extract_rt` function in `rtpipeline/rt_details.py` at lines 92–169 retains `ReferencedStructureSetSequence`. The repaired `link_rt_sets` function in `rtpipeline/metadata.py` at lines 37–167 resolves RTPLAN to RTSTRUCT by patient and referenced SOPInstanceUID. It logs unresolved and ambiguous references instead of choosing another RTSTRUCT.

## Defect 2 suppressed DFCI RT linkage before study matching

The proposed mechanism was incomplete. Cross-study RT linkage exists in DFCI, but it did not cause the observed zero-RT output first. The patient and series layout branch in pre-fix `organize_and_merge` at `rtpipeline/organize.py` lines 2034–2039 skipped RT extraction completely.

The DFCI log proves that branch fired. The output proves its consequence because 3,588 courses contain zero copied RTSTRUCT, RTPLAN, and RTDOSE files. No study predicate could link objects that were never indexed.

A full header measurement also narrows the cross-study claim. Of 1,561 DFCI RTPLAN objects, 7 lacked a `ReferencedStructureSetSequence`. The remaining 1,554 references resolved to 407 distinct RTSTRUCT objects. Fifteen plan-to-structure links, spanning 13 RTSTRUCT objects, crossed StudyInstanceUID.

The measured cross-study bridge therefore lies mainly between RTPLAN and RTSTRUCT, not between every RT object and CT. In the 4-patient sample, all 61 referenced RTSTRUCT-to-CT series links resolved within the same StudyInstanceUID and shared FrameOfReferenceUID. The sample does not establish the full-cohort RTSTRUCT-to-CT frequency.

The repaired `organize_and_merge` function at lines 2563–2578 scans RT objects for every recognized directory shape. It no longer treats a patient and series layout as proof of a CT-only cohort. Explicit SOP references work whether StudyInstanceUID agrees or differs.

Planning CT selection remains reference-driven. `select_course_ct_series` at lines 2315–2372 follows the RTSTRUCT frame, study, and series sequences across every indexed patient study. The organize caller now requires that reference chain and will not use the largest same-study series when an RTSTRUCT is present.

## Defect 3 merged distinct courses

The pre-fix `group_by_course` function in `rtpipeline/metadata.py` at lines 73–126 grouped by CT StudyInstanceUID or FrameOfReferenceUID. It could then split or merge that group using plan-date proximity. Neither value identifies a treatment course.

That predicate merged plans referencing different planning structure sets. The full Kopernik census found 122 target-bearing referenced RTSTRUCT objects but 116 output courses. The under-generated patients were 351107, 467782, 474021, 481077, and 483634.

Patient 481077 demonstrates the consequence. Its plans referenced 4 target-bearing RTSTRUCT objects that represented 4 supported courses. The pre-fix grouping produced 2 output courses. Reference-driven grouping produces 4 target-bearing groups in the selected-patient metadata check.

The repaired `group_by_course` function at lines 168–202 uses the referenced RTSTRUCT SOPInstanceUID as course identity. Plans sharing 1 RTSTRUCT remain 1 course, including sequential phases. Plans referencing different RTSTRUCT objects remain separate even when study, frame, date, or anatomy agrees.

One Kopernik target-bearing referenced RTSTRUCT in patient 351107 has plans but no RTDOSE that resolves to those plans. The repair retains it as a target-bearing plan-only course and logs the missing dose. This preserves all 122 supported planning structure sets without fabricating an RTDOSE.

The other 121 Kopernik target-bearing course identities have an exact RTDOSE-to-RTPLAN link. A rebuild should therefore yield about 122 courses, with 1 explicit plan-only exception rather than an invented dose.

## Defect 4 summed revisions, replans, phases, and separate courses

The pre-fix `_classify_doses` function in `rtpipeline/organize.py` began at line 679. It used free-text labels to infer boost and replan status, then allowed multiple selected plan doses to be summed. Free text could not distinguish revision history, delivered replacement plans, sequential phases, and unrelated courses.

The repaired active `_classify_doses` function at lines 1229–1571 does not use plan-label text. RTDOSE references first establish membership. The classifier then works within the group already defined by the plan-referenced RTSTRUCT. It does not compare plans across different RTSTRUCT groups.

Within 1 RTSTRUCT group, equivalent prescription and planned-fraction signatures identify revision-equivalent plans. A plan whose prescription equals component prescriptions identifies an alternative course-total representation. The classifier selects 1 representation instead of adding both.

RTRECORD evidence is useful but not sufficient alone. The repair reads each record's referenced plan and treatment date. A plan whose dates are a strict subset of another plan's dates does not add dose. A shorter delivered replacement plan also does not add to an encompassing course prescription.

This record logic avoids counting portal or verification plans solely because they have treatment records. It also avoids reporting a full prescription plus a partially delivered replacement prescription. When records cannot resolve the choice, prescription and fraction evidence governs the deterministic result.

Read-only execution on the selected Kopernik patients resolved the within-group examples. Patient 351107 retained 50 Gy in 25 fractions plus 10 Gy in 5 fractions as 60 Gy. Patient 353398 de-duplicated same-prescription revisions to 69.96 Gy, the DICOM-encoded value for the approximately 70 Gy course.

Patient 482203 does not fit the within-group replan rule. Its plans resolve to 3 target-bearing RTSTRUCT objects. The groups contain 55 Gy in 20 fractions, 55 Gy plus 8.25 Gy in 3 fractions, and 55 Gy plus 46.75 Gy in 17 fractions.

Reference-driven grouping emits 3 entries for patient 482203, each with a selected total of 55 Gy. The classifier cannot test that 46.75 Gy plus 8.25 Gy equals 55 Gy because those partial plans sit in 2 different RTSTRUCT groups.

All 6 available treatment records reference the 55 Gy plan in 1 of those groups. They do not link the 3 RTSTRUCT objects into 1 delivered course. The pipeline therefore prevents a 101.8 Gy sum within any entry but does not de-duplicate this delivered treatment across entries.

This is a deliberate consequence of making each plan-referenced planning RTSTRUCT a course identity. A cohort analysis must not count the 3 entries as independent treatments without adjudication. The rebuild audit should flag cross-RTSTRUCT prescription partitions rather than merge them from dates, labels, or anatomy.

Patient 440657 became 2 structure-defined courses with 55 Gy and 50 Gy prescriptions. The repaired classifier logs every exclusion and whether remaining phases are summed.

A `PLAN_SUM` RTDOSE that references several plans on 1 planning RTSTRUCT remains authoritative. A dose referencing plans on different RTSTRUCT objects is not attached to either course because it cannot be split safely. An unresolved dose reference is excluded rather than guessed onto the only available plan.

## Defect 5 allowed QA and non-planning CT courses

The existing shared classifier already detects the DFCI phantom through manufacturer, model, description, and implausible CT geometry. The defect was reachability. `organize_and_merge` never called `classify_series` before constructing courses.

The same pre-fix CT-only branch at `rtpipeline/organize.py` line 2690 promoted indexed CT series into standalone courses. It did not require an RT reference. The operator's audit therefore found QA phantom, scout, topogram, and orphan CT courses even though the classifier could exclude them elsewhere.

The repair calls the existing `classify_series` function through `_classify_organize_ct_series` at lines 464–521. It does not copy phantom rules. Referenced planning CT must pass the shared classifier before copying or conversion.

The CT-only fallback now rejects every series without an RT planning reference. This removes QA phantom, scout, localizer, topogram, and unrelated diagnostic CT as course identities. No anatomy or series description is used to invent a treatment course.

## Defect 6 lost both CT and RTSTRUCT for patient 333944

Patient 333944 course 2018-05 is not a cross-study plan-to-structure failure. Its RTPLAN objects and referenced 15-ROI RTSTRUCT share the same StudyInstanceUID. All 9 resolved links for this patient and all 48 resolved links across the 7-patient Kopernik probe were same-study.

The pre-fix frame-keyed index assigned no RTSTRUCT to the 25 linked dose-plan sets for the 2018 course. Their recorded frame key was empty. The metadata then used the RTDOSE study as `ct_study_uid`, which differed from the shared RTPLAN and RTSTRUCT study.

The course writer searched for an RTSTRUCT in that RTDOSE study and found none. The referenced RTSTRUCT already identified 2 indexed CT series in the shared RTPLAN and RTSTRUCT study, but the writer could not follow that bridge without the authoritative RTSTRUCT. Both output directories remained empty.

Reference-driven metadata now resolves 3 target-bearing course identities for patient 333944. The 2018 plans resolve to the 15-ROI RTSTRUCT. Its 2 referenced CT series supply the planning CT candidates.

This diagnosis does not claim the rebuilt DICOM directories exist. The real cohort was not rerun. Synthetic cross-study tests and read-only metadata execution verify the repaired decision path before that rebuild.

## QC gate and regression coverage

The new `validate_course_target_qc` gate in `rtpipeline/organize.py` at lines 428–462 fails a course that contains RTPLAN and RTDOSE but has no authoritative target-bearing RTSTRUCT. `organize_and_merge` logs the failure loudly and excludes the invalid course from output.

Target-bearing plan-only courses remain possible when the source has no resolvable RTDOSE. This distinction preserves supported course identity without fabricating delivery evidence. Plan-and-dose courses cannot pass with a setup, BODY-only, auto-contour, or unresolved RTSTRUCT.

Synthetic DICOM regressions cover all required failure shapes. They test authoritative RTSTRUCT selection, rejection of a larger zero-target auto-contour, cross-study and shared-study CT linkage, 4 distinct structure-defined courses, revision de-duplication, sequential boost summation, and year-separated course splitting.

They also test the production-shaped 482203 pattern with 3 target-bearing RTSTRUCT objects and 6 plans. That test asserts 3 structure-defined entries of 55 Gy and prevents the within-group classifier test from being misread as patient-level de-duplication.

Further tests cover shared classifier exclusions, the zero-target QC failure, multiple plan references in a summed dose, a target-bearing plan-only course, and fail-closed unresolved dose references. The tests use no real patient DICOM files.

## Rebuild expectations and unresolved points

A DFCI rebuild should approach 154 patients and 230 target-bearing structure-defined entries. It should not reproduce 3,588 CT-derived directories. A Kopernik rebuild should retain 92 patients and about 122 structure-defined entries, including the 1 verified plan-only exception.

The 122-entry Kopernik estimate counts referenced target-bearing RTSTRUCT objects, not independent delivered treatments. It includes 3 entries for patient 482203 even though the measured prescription pattern supports 1 delivered 55 Gy treatment. The pipeline has no safe cross-RTSTRUCT merge rule for that pattern.

The post-fix counts remain predictions because the task prohibited cohort reprocessing. Dose classification was executed on selected Kopernik source metadata, not on every course. The rebuild must reconcile every target-bearing referenced RTSTRUCT, emitted entry, selected RTSTRUCT, planning CT, plan, dose, target count, and prescription.

The next acceptance action is a read-only rebuild audit after the operator reruns both cohorts. The audit should report structure-defined entries and adjudicated delivered treatments separately. It should also flag cross-RTSTRUCT replacement patterns such as patient 482203 for clinical resolution.

The audit should fail if any target-bearing referenced RTSTRUCT is absent, any plan-and-dose entry has zero targets, any copied RTSTRUCT differs from the RTPLAN reference, or any CT-only orphan entry remains.
