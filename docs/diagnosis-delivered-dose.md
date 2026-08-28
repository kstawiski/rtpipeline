# Delivered dose requires record-linked evidence

## Clinical question

The pipeline must distinguish prescribed treatment from treatment delivered to the patient. This distinction matters because a plan can be abandoned after several fractions, reused for additional fractions, or present without any treatment record. A full prescription is not evidence that the full dose reached the patient.

The repair keeps `total_prescription_gy` as the prescribed or intended dose. It adds `delivered_dose_gy` as a separate field. Dose-response analyses should use `delivered_dose_gy` when it is known because it is anchored to RTRECORD evidence. Analyses should retain `total_prescription_gy` to describe treatment intent and to compare intent with delivery. They should not replace an unknown delivered dose with zero.

## The failing predicate

The previous delivered-dose path treated every selected plan with a treatment record as if its full prescription had been delivered. It also summed one dose-reference value per RTRECORD object. That logic was unsafe in two directions. It credited abandoned plans with the full course dose and double-counted repeated record objects for one treatment session.

The repaired estimator follows DICOM plan references. It binds a record-level value to the plan's target dose reference through `ReferencedDoseReferenceNumber` and `DoseReferenceNumber`. Values for non-target references are rejected for total delivered dose. The estimator reads only per-session calculated dose-reference sequences. It does not read the cumulative value from a treatment summary record as though it were a per-session value.

For a valid cumulative dose-to-reference value, the estimator selects the latest valid cumulative observation. It does not sum successive cumulative observations. Otherwise, it groups records by the existing treatment-session key and de-duplicates repeated beam or application components within that session. It sums distinct additive components within a session. This prevents duplicate RTRECORD objects from becoming duplicate fractions or duplicate dose contributions.

The explicit per-session estimate is checked against the prescribed dose per fraction. The tolerance is the larger of 0.1 Gy and 5 percent of the prescribed dose per fraction. When the check fails, the estimator logs both estimates and uses the fraction-weighted prescription when the fraction count is available. The fallback is `rx * min(delivered_fraction_count / planned_fraction_count, 1.0)`.

A selected plan with no linked treatment record is not untreated. Its delivered dose is unknown. A course is also unknown when any selected contributing plan cannot be estimated. The metadata distinguishes `fully_delivered`, `partially_delivered`, `delivered_but_records_absent`, and `no_records_at_all`. The last status means that the course had no treatment records in the available patient-level record set. A record that references an absent plan is counted and logged but is never assigned to another plan. The repair deliberately does not fall back to a filename core for that dose because a filename cannot establish plan identity after the DICOM reference fails.

## Worked clinical example

The decisive Kopernik case is patient 419783, course 2020-02. The selected plans prescribed 50 Gy in 25 fractions and 25 Gy in 10 fractions. RTRECORD linkage found six sessions for the first plan and ten sessions for the second plan. The repaired calculation reports 12 Gy plus 25 Gy, or 37 Gy. The preserved `total_prescription_gy` remains 75 Gy. The earlier 75 Gy delivered-dose value therefore overstated delivery by 38 Gy.

Three additional Kopernik checks expose the former record-level error. Patient 482967, course 2024-11, has a 55 Gy plan with 19 counted sessions and 40 RTRECORD objects. The repaired delivered dose is 52.25 Gy, not 55 Gy. Patient 432976, course 2020-04, has a fully recorded 36 Gy plan and a partially recorded 27 Gy plan. The record values disagree with the target prescription-per-fraction check, so the repaired course uses 36 Gy plus 20.25 Gy, or 56.25 Gy. Patient 327471, course 2020-12, has eight recorded sessions for a 5 Gy plan. The record-linked estimate is 40 Gy and the metadata emits a warning that delivered dose exceeds the prescription. The warning is needed because 40 Gy is below the absolute 100 Gy plausibility threshold.

## Cohort scale

The cohort calculations below call the production `rtpipeline.organize._calculate_delivery_summary` function for every course. They do not reimplement its estimator. Reported minus delivered means `total_prescription_gy - delivered_dose_gy`. Negative values remain visible because they may indicate record semantics or plan-selection issues that require clinical adjudication.

The read-only Kopernik rebuild contains 122 courses from 92 patients. Fifteen courses have at least one partially delivered selected plan. Delivered dose is known for 102 courses and unknown for 20. Among the 102 paired courses, reported minus delivered dose has a median of 0.0 Gy, an interquartile range of 0.0–0.0 Gy, a mean of 3.404 Gy, and a range from -35.0 to 46.75 Gy.

The largest Kopernik differences are 46.75 Gy for patient 482203, 46.0 Gy for patient 493649, 42.0 Gy for patient 478079, 38.0 Gy for patient 419783, and 36.0 Gy for patient 431057. These are ranked observations, not evidence that every course has the same error.

The read-only DFCI rebuild contains 230 actual courses from 154 patients and 13,997 RTRECORD rows. Six referenced plan UIDs were absent from the indexed plan export. The 690 associated record rows were counted as unresolved and were not attributed to another plan. Sixteen courses have at least one partially delivered selected plan. Delivered dose is known for 184 courses and unknown for 46. Among the 184 paired courses, reported minus delivered dose has a median of 0.0 Gy, an interquartile range of 0.0–0.0 Gy, a mean of 0.680 Gy, and a range from -25.0 to 48.6 Gy.

The largest DFCI differences are 48.6 Gy for patient 10091271477, 21.0 Gy for patient 10050231785, 13.6125 Gy for patient 10057423435, 12.6 Gy for patient 10146171292, and 11.0 Gy for patient 10150950383. The paired denominator excludes courses with unknown delivered dose. This exclusion prevents a recordless selected plan from contributing zero to a course total.

## Planning CT adjudication

The DFCI output inventory contains 231 second-level directories. It contains 230 actual course directories and one internal `_COURSES/patients` checkpoint directory. Of the actual courses, 228 contain a CT series and two do not. The two actual no-CT courses both contain RTSTRUCT and RTPLAN objects and are explicitly kept from looking complete.

For patient 10058435883, course 2022-05, the RTSTRUCT references a CT series absent from the export. The planning CT status is therefore `unresolved_reference`. For patient 10149603697, course 2023-01, the reference resolves to 154 exported instances described as `Localizer`. The classifier excludes that series, and the planning CT status is `classifier_excluded`. The current evidence does not support a third actual no-CT course. The internal checkpoint directory is not a clinical course.

Resume validation now follows the plan evidence discovered for the course rather than relying on whether one metadata key is populated. A course is plan-bearing when hydration finds a plan file under `DICOM/RTPLAN`, finds the legacy root `RP.dcm`, or reads a nonempty metadata `rp_path`. A fourth layout can therefore be added to discovery without creating a separate resume rule.

Any plan-bearing checkpoint is reprocessed when `delivery_status` is absent, null, blank, `unknown`, or outside the four documented statuses. It is also reprocessed when `planning_ct_status` is absent, null, blank, or `unknown`. Hydration applies no defaults before this gate. Plan-free checkpoints may retain their legacy defaults. A fully adjudicated modern checkpoint still hydrates, so valid resume runs do not become complete recomputations.

Manifest-driven resume is all-or-nothing. If any selected manifest entry is missing, cannot hydrate, or requires reprocessing, the loader rejects the hydrated subset and falls back to organize. This prevents a valid course from hiding an incomplete course in the same manifest.

The read-only Kopernik inventory found nested RTPLAN files in all 122 courses, but neither adjudication field in any checkpoint. The previous predicate would have reprocessed 9 courses and silently skipped 113 of 122 courses, or 92.6 percent, conventionally rounded to 93 percent.

The DFCI inventory found the same defect at scale. All 230 courses had nested RTPLAN files and neither adjudication field. The previous predicate would have reprocessed 44 courses and silently skipped the corresponding 186 of 230 courses, or 80.9 percent.

## Discovery and metadata safeguards

Organize discovery now fails when a non-empty input tree yields zero supported DICOM objects. The error reports the input root, visible file count, and the likely cause. When un-followed symlinked directories are present, it names `RTPIPELINE_FOLLOW_INPUT_SYMLINKS=1` as the explicit opt-in. The organizer no longer writes an empty successful manifest for this failure.

Metadata export applies the same loud-failure predicate to every detected modality. A modality present in the input but absent from its exported rows fails consistently. An RTDOSE with references that resolve to no indexed RTPLAN now produces a warning naming the dose path and each unresolved plan UID. The summary reports the unresolved count. The dose is not assigned by filename fallback.

The unreachable label-text dose classifier and its replan and boost helpers were removed. Plan-label text does not control dose classification. Dose selection uses DICOM references, dose content, and explicit classification rules.

## Plausibility configuration

`max_total_dose_gy` is the single configurable absolute threshold. It defaults to 100.0 Gy. The supported project YAML key and the `--max-total-dose-gy` CLI option both populate the same validated configuration field. The same value is passed to organize metadata and DVH dose plausibility checks. The existing `Implausible total dose` path therefore evaluates the prescribed and delivered fields with one threshold.

The separate prescription-mismatch warning catches a delivered estimate that exceeds its selected plan prescription even when the absolute threshold is not crossed. This warning identified the 40 Gy estimate for the 5 Gy patient 327471 plan.

## Verification and remaining uncertainty

Synthetic DICOM tests cover the 419783 shape, recordless selected plans, fully delivered plans, duplicate records within one session, latest cumulative observations, non-target dose references, absent plan references, empty discovery, modality-specific empty exports, and configurable thresholds.

Resume regressions cover nested RTPLAN files, legacy root plans, and nonempty metadata plan pointers. They cover missing, null, blank, and `unknown` adjudications, all four valid delivery statuses, a mixed manifest, and fully adjudicated modern checkpoints. The complete repository suite remains the software gate.

The cohort artifacts preserve row-level estimates, methods, statuses, warnings, source paths, and calculation notes. The evidence ledger separates implementation evidence, synthetic test evidence, and read-only cohort calculations. Direct DICOM inspection supports the worked examples and the planning-CT adjudication.

The resume calculation is recorded in `analysis/resume_checkpoint_census.py`, `analysis/results/resume-checkpoint-kopernik.json`, and `analysis/results/resume-checkpoint-dfci.json`. These aggregate inventories emit no patient or course identifiers and were generated without modifying either rebuilt cohort.

Delivered dose remains uncertain when records are absent, when a selected plan is not linked to any record, or when record content cannot support either a target-bound explicit estimate or fraction weighting. The estimator is not a substitute for the treatment chart. Courses with negative reported-minus-delivered values, reused plans, unresolved references, or classifier-excluded imaging require clinical review before outcome modeling. The documented results establish a safer analysis field and a fail-closed boundary. They do not establish treatment benefit, toxicity, or causation.
