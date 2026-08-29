# One course authority prevents undelivered radiotherapy phases from entering treatment dose grids

The repaired pipeline makes the organizer's course decision binding for every later stage. This matters because an undelivered plan can otherwise inflate dose-volume histograms and corrupt dose-response analyses without changing the organizer's recorded dose.

A read-only reconstruction found legacy organizer and DVH plan membership disagreed in 2 of 4 Kopernik courses with both prescription and DVH data. The fixed resolver reproduced the organizer's selected membership in all 4 courses when replayed against their real DICOM headers.

The dose grid remains a planned dose distribution for the plans with delivery evidence. The delivered exposure remains `delivered_dose_gy`. Dose-response analyses should use that field, while treatment-intent and plan-quality analyses may use the prescribed dose and the explicitly labelled planned grid.

## What failed

The organizer selected a course after linking plans, doses, treatment records, structures, and planning images. It then copied every candidate RT object into the course directory. Later stages scanned those copied candidates and made a second decision.

The copied directory was therefore a transport container, not a valid statement of treatment membership. Directory order, filenames, and the presence of a readable RT object could silently replace the organizer's reference-driven decision.

The legacy `case_metadata.json` exposed this split. `source_plan_uids` and `source_dose_uids` recorded selected source objects, while `rp_path`, `rd_path`, and `rs_path` could remain empty. Downstream code therefore lacked a complete contract even when it attempted to follow metadata.

## Real-course reconstruction

The evidence ledger was computed from the read-only Kopernik output, the corresponding source RTRECORD headers, and header-only replay through the fixed DVH resolver. It includes SHA-256 hashes for every source metadata file and DVH workbook.

| Course | RTPLAN files | Organizer plans | Legacy DVH plans | Target PTV D95 | Prescription | D95 to prescription | Legacy membership | Fixed membership |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 428073/2020-05 | 1 | 1 | 1 | 29.441 Gy | 30.0 Gy | 98.14% | agrees | agrees |
| 431057/2020-07 | 2 | 1 | 2 | 64.494 Gy | 56.0 Gy | 115.17% | disagrees | agrees |
| 441642/2021-05 | 2 | 1 | 2 | 102.357 Gy | 50.0 Gy | 204.71% | disagrees | agrees |
| 475201/2025-03 | 1 | 1 | 1 | 50.713 Gy | 55.0 Gy | 92.20% | agrees | agrees |

These measurements confirm the direction was not constant. The legacy DVH differed from prescription by more than 10% in 2 of 4 courses. One disagreement reflected an undelivered sequential phase. The other reflected an earlier undelivered course.

For 431057/2020-07, plan `etap1` had 10 delivered sessions between 20 July and 31 July 2020. Plan `etap2` had no linked delivery record. The organizer selected `etap1`, while the legacy DVH workbook listed both plan UIDs.

For 441642/2021-05, plan `arc` had no linked delivery record. Plan `arc1+` had 20 delivered sessions between 7 June and 5 July 2021. The organizer selected `arc1+`, while the legacy DVH workbook listed both plan UIDs.

The fixed comparison is a membership result, not a claim that the read-only cohort was rebuilt. Header-only copies retained real SOP UIDs, plan references, and `DoseSummationType`. The fixed resolver consumed a contract made from the organizer's recorded membership and returned identical plan and dose sets in all 4 courses.

No post-fix voxel DVH values are reported. Rebuilding those values would modify or replace cohort outputs, which were read-only evidence for this task.

## The authoritative course contract

The organizer now writes one versioned `course_contract` object inside `metadata/case_metadata.json`. Its authority is explicitly `organize`. The patient and course identifiers must match the directory that contains the contract.

The contract records selected RTPLAN SOP Instance UIDs and course-relative paths. It records selected RTDOSE SOP Instance UIDs, paths, `DoseSummationType`, and referenced plan UIDs. A dose cannot refer to a plan outside selected membership.

The contract records the authoritative RTSTRUCT SOP Instance UID and path. It also records the planning CT Series Instance UID, selected DICOM directory, NIfTI path, reference status, and every referenced CT series UID used during adjudication.

Delivery evidence is retained for every candidate plan, including plans not selected for the dose grid. Each entry carries the plan UID and path, prescribed dose, planned fractions, delivered record count, delivered session count, treatment dates, record paths, zero-record status, and grid-selection status.

The delivery section keeps prescribed dose separate from delivered dose. It also records delivery status, calculation method, unresolved plan references, selected plan UIDs, and the required dose-response field `delivered_dose_gy`.

The plan artifact and dose grid are named separately from their source objects. The grid records its semantics and the exact source plan UIDs, source dose UIDs, and source dose summation types.

The binding dose-quality verdict records `pass` or `fail`, the configured `max_total_dose_gy` threshold, and reasons. A prescribed or delivered value above the threshold cannot coexist with a passing verdict.

Top-level legacy metadata fields remain for compatibility, but downstream selection does not read them as a second authority. The organizer populates `rp_path`, `rd_path`, and `rs_path` when the corresponding contracted artifact exists.

## Planned and delivered dose

A plan with zero linked RTRECORD evidence is excluded when treatment records exist for the patient but none references that plan. It contributes neither delivered dose nor a grid labelled as the delivered treatment plan set.

When no RTRECORD objects exist at all, delivery remains unknown rather than zero. The grid may represent planned dose for a selected plan set with unknown delivery, and the contract labels that uncertainty explicitly.

When both phases of a genuine sequential course have linked delivery evidence, both plans remain selected and their plan-level doses may be summed. Replacement plans are not added merely because their files are present.

`PLAN`, `PLAN_SUM`, and beam-level dose objects have different meanings. The organizer selects plan-level totals when available. It may combine beam-level objects only as components for the same selected plan when no plan-level total exists. The contract rejects a mixture of plan-level and beam-level sources.

The prescribed dose describes treatment intent. The planned grid describes the selected treatment plan set. The delivered dose describes exposure supported by RTRECORD evidence and is the field for dose-response analysis.

## Fail-closed validation

Every downstream entry point calls the same contract loader before it resolves treatment objects. A missing, unreadable, unsupported, path-escaping, incomplete, or internally inconsistent contract raises `CourseContractError`.

The organizer reloads and validates each contract after writing `case_metadata.json`. A producer-side path, UID, delivery, summation-type, or dose-QC inconsistency therefore stops the run before later stages can consume it.

Validation reads the contracted DICOM headers and compares their SOP Instance UIDs, modalities, dose summation types, and dose-to-plan references with the contract. It verifies that planning CT slices contain one declared series and that the contracted NIfTI exists.

Validation also reconciles selected plan membership with per-plan delivery evidence. A fully or partially delivered course cannot select a plan with zero delivery records. A treatment grid cannot exist without selected plan and dose sources.

A stale checkpoint is not hydrated from legacy defaults. Resume hydration uses the validated contract for dose, delivery, planning image, structure, and source membership. A complete contracted checkpoint still hydrates, which avoids forcing an unnecessary full recomputation.

The former directory helper that returned the first DICOM file or a legacy flat path was removed. There is no silent fallback from a missing contract to directory scanning.

## Re-derivation audit

The audit classified every course-level RT or planning-image discovery site found in the package. Reading pixel data or attributes inside an already contracted object remains valid. Choosing a plan, dose, structure set, or planning CT outside the contract does not.

| Site | Previous decision surface | Current authority or reason it remains independent |
|---|---|---|
| `organize._hydrate_existing_course` | Legacy paths and top-level metadata could reconstruct a checkpoint | Validates and hydrates `course_contract`, with no artifact or membership fallback |
| `dvh._resolve_dvh_dose` | Scanned RTPLAN and RTDOSE candidates and classified them again | Uses contracted plan artifact, grid, source membership, semantics, delivered dose, and QC |
| `dvh._resolve_dvh_structures` and `dvh.dvh_for_course` | Searched copied RTSTRUCT candidates and root-level names | Uses the contracted RTSTRUCT, contracted planning CT, and explicitly generated derivative structures |
| `dvh._compute_nifti_based_dvh` | Could use the course CT directory by convention | Uses the contracted planning CT directory |
| `radiomics.radiomics_for_course` | Selected CT and RTSTRUCT from course paths | Uses contracted planning CT and authoritative RTSTRUCT, while generated masks remain separate derivatives |
| `radiomics_parallel.parallel_radiomics_for_course` | Reconstructed parallel-worker inputs from directory contents | Uses the same contracted planning CT and RTSTRUCT as serial radiomics |
| `radiomics_conda.radiomics_for_course` | Used conventional CT and RTSTRUCT paths | Uses contracted planning CT and authoritative RTSTRUCT |
| `radiomics_conda.radiomics_for_course_ct_nifti_fallback` | Could choose a CT NIfTI fallback independently | Uses the contracted planning CT NIfTI and fails when the contract does not provide it |
| `radiomics_robustness.robustness_for_course` | Read planning inputs from conventional paths | Uses contracted planning CT, retains named generated structure inputs, and imports the shared ROI-family definition |
| `segmentation.segment_course` | Used the course CT directory as an implicit planning-series choice | Uses the contracted planning CT directory and NIfTI, while scans of generated outputs and independent MR series remain legitimate |
| `auto_rtstruct.build_auto_rtstruct` | Read the conventional CT directory | Uses the contracted planning CT directory and series UID, with frame UID reading limited to the selected series |
| `structure_merger.StructureMerger` | Used root-level structure paths | Uses the contracted authoritative RTSTRUCT and planning CT, while auto and custom structures remain named generated derivatives |
| `custom_structures_rtstruct._create_custom_structures_rtstruct` | Used conventional CT and structure paths | Uses contracted planning CT and authoritative RTSTRUCT |
| `custom_models.run_custom_models_for_course` | Reconstructed image inputs from course layout | Uses contracted planning CT DICOM and NIfTI |
| `quality_control.DICOMValidator` | Treated conventional paths as selected objects | Uses contracted artifacts, while later CT scans inspect consistency and frame identity within the selected series |
| `visualize.generate_axial_review` | Used conventional structure and CT paths | Uses contracted RTSTRUCT and planning CT |
| `anatomical_cropping` course functions | Chose a CT NIfTI and DICOM series from conventional paths | Use contracted planning CT DICOM and NIfTI |
| `rt_details.extract_rt` | Scans source DICOM and extracts links | Remains upstream organizer input. It inventories source objects and does not select a downstream course artifact |
| `meta.export_metadata` | Scans source DICOM and joins source-level RT tables | Remains an independent source inventory. It does not define organized course membership or feed treatment grids |
| `radiomics` and `segmentation` MR branches | Scan MR series and generated model outputs | Remain independent because the course contract governs planning CT and RT treatment objects, not separate MR series or newly generated outputs |
| `body_composition._first_dicom_patient_size_m` | Reads the first header in its supplied image series | Remains an attribute lookup inside the image series selected by its caller, not a course-selection decision |
| `layout.find_dcm` | Returned the first candidate or a flat-layout fallback | Removed |

## Unified definitions

Dose selection and classification now live in the organizer and are serialized in the contract. DVH no longer contains a second dose classifier or a local `DoseSummationType` selector.

The organizer computes dose QC from `PipelineConfig.max_total_dose_gy`. DVH accepts that configured value only as a consistency check against the contract. The former hard-coded 100.0 Gy decision is gone.

`rt_details.DEFAULT_ROI_FAMILY_NAMES` is the shared ROI-family definition. Target recognition uses its GTV, CTV, and PTV subset. Radiomics robustness imports the same definition rather than maintaining an independent GTV, CTV, PTV, BLADDER, and RECTUM list.

## Verification

Production-shaped synthetic DICOM tests cover a copied plan and dose superset, selected RTSTRUCT pinning, planning CT pinning, missing contracts, stale identities, path escape, UID mismatch, and contract-to-disk reference mismatch.

Dose tests cover `PLAN` and `BEAM` separation, plan-level preference, same-plan beam components, undelivered replacement plans, an undelivered sequential boost, and a genuine delivered sequential boost. A single plan with zero linked records is also excluded.

A TPS `PLAN_SUM` that contains any plan with zero linked delivery records is rejected when no separable delivered-plan dose is available.

Delivery tests distinguish prescribed from delivered dose, retain zero-record plans in evidence, and require `delivered_dose_gy` for dose-response analysis. Resume tests reject incomplete checkpoints and retain a fully contracted plan checkpoint.

Dose-quality tests verify that an implausible total produces a failing verdict. They also reject a contract that records the same implausible value with a passing verdict.

The evidence ledger can be regenerated by running `analysis/course_contract_evidence.py` with the read-only campaign output, source DICOM root, workspace, and ledger paths. It fails unless exactly 4 courses have both a numeric prescription and a DVH workbook.

## Interpretation

The repaired contract removes the mechanism that let copied but unselected plans enter later analyses. The fixed resolver's 4-course header replay showed exact organizer agreement, including the 2 legacy mismatch courses with undelivered plans.

That result establishes source-membership control. It does not establish new voxel-level DVH values for the read-only cohort. The next governed cohort rebuild should regenerate DVH and radiomics outputs, retain the contract provenance columns, and exclude or flag any course whose binding dose-quality verdict fails.
