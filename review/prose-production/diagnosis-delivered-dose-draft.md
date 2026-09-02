# Delivered dose and residual fail-closed defects

## Clinical decision

Radiotherapy dose-response analyses should use `delivered_dose_gy`, not `total_prescription_gy`. The delivered field follows treatment records and therefore reflects what the patient received. The prescription field remains necessary because it records treatment intent.

The previous output could assign a full prescription to a plan stopped after several fractions. That error can make an abandoned course look fully delivered and can reverse dose-response interpretation. The revised output keeps intent and delivery separate and makes unknown delivery explicit.

## Primary dose defect

The failing predicate was any selected plan with at least 1 linked treatment record. The previous calculation added that plan's full prescription, irrespective of the number of delivered fractions.

Patient 419783 in the Kopernik bladder cohort shows the clinical consequence. A 50 Gy plan scheduled for 25 fractions had 6 delivered sessions. A 25 Gy plan scheduled for 10 fractions had 10 delivered sessions.

The previous field reported 75.0 Gy. Direct rereading of the 16 linked treatment records found calculated dose-reference values totaling 12.0 Gy and 25.0 Gy. The delivered dose was therefore 37.0 Gy, which was 38.0 Gy below the reported prescription.

## Delivered-dose method

Each treatment record is linked only through `ReferencedRTPlanSequence`. A record referencing a plan absent from the export is counted and logged. It is never assigned to another plan.

Distinct fractions are counted by a DICOM fraction number and treatment date when available. Treatment date is used when the fraction number is absent. This prevents several beam records from one treatment session being counted as several fractions.

The calculation prefers complete record-level delivered or calculated dose-reference values. When these values are available for every linked record, their plan-level sum becomes the delivered dose.

The fallback uses the plan prescription multiplied by the smaller of 1.0 and delivered fractions divided by planned fractions. The fallback method is recorded as `record_fraction_weighted_prescription`.

A course with no treatment records has `delivered_dose_gy` set to null. The pipeline never converts unknown delivery to 0.0 Gy or to the prescription.

The emitted `delivery_status` distinguishes `fully_delivered`, `partially_delivered`, `delivered_but_records_absent`, and `no_records_at_all`. The third status means treatment records exist for the patient but none resolve to a selected course plan. Dose remains null in that state.

Course metadata also records the method, record count, distinct fraction count, planned fraction count, plan-level details, unresolved plan UIDs, and absent-plan record counts. These fields preserve the calculation and its uncertainty.

## Cohort scale

The cohort analyses were read-only. They compared the prescription already reported in each rebuilt course with a treatment-record estimate for the selected source plans.

| Cohort | Courses | Patients | Courses with at least 1 partially delivered plan | Courses with paired dose estimates | Median difference, Gy | IQR, Gy | Mean difference, Gy | Range, Gy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kopernik bladder | 122 | 92 | 15 | 101 | 0.0 | 0.0 to 0.0 | 3.884 | 0.0 to 46.75 |
| DFCI TMT | 230 | 154 | 16 | 153 | 0.0 | 0.0 to 0.0 | 2.165 | -25.0 to 62.002 |

In Kopernik, the largest reported-minus-delivered differences were 46.75 Gy, 46.0 Gy, 42.0 Gy, 38.0 Gy, and 36.0 Gy. Patient 419783 was the fourth largest difference.

In DFCI, the largest differences were 62.002 Gy, 56.874 Gy, 55.0 Gy, 50.4 Gy, and 45.0 Gy. The audit reread all 13,997 treatment records and 1,561 plans without a read failure.

The DFCI audit found 6 referenced plan UIDs absent from the plan export, covering 690 treatment-record objects. These records were counted but excluded from course attribution.

The DFCI distribution included negative differences down to -25.0 Gy. These values remain visible because normalizing them away would erase discordant evidence. They require clinical adjudication before outcome modeling.

## Planning CT adjudication

The completed DFCI output contains 230 actual course directories. Of these, 228 contain planning CT and 2 do not. The apparent denominator of 231 included the internal `_COURSES/patients` checkpoint directory, which is not a treatment course.

Patient 10058435883 course 2022-05 has RTSTRUCT and RTPLAN but no planning CT. Its RTSTRUCT references series `1.2.840.113619.2.55.3.380389780.814.1387405559.499`. The completed header cache contains 0 instances for this series.

This planning CT is genuinely absent from the export. The revised metadata reports `planning_ct_status` as `unresolved_reference` and preserves the referenced series UID.

Patient 10149603697 course 2023-01 also has RTSTRUCT and RTPLAN but no planning CT. Its RTSTRUCT reference resolves to 154 exported instances in series `1.2.840.113619.2.55.3.464291532.810.1674594526.814`.

The referenced series is described as `Localizer`. The organize log records classifier exclusion with reason `description_localizer`. Excluding it from planning CT is correct because it is not a volumetric planning acquisition.

The revised metadata reports this course as `classifier_excluded` rather than making it look complete. Resume validation also rejects old checkpoints that lack delivery or planning CT adjudication, so affected patients are reprocessed.

## Empty discovery defect

The failing predicate was a nonempty input tree that yielded 0 discoverable DICOM objects. The organizer previously wrote an empty manifest and exited successfully.

The organizer now raises an error before writing an empty successful result. When the input contains symlinked directories and symlink following is disabled, the error names `RTPIPELINE_FOLLOW_INPUT_SYMLINKS=1` as the required opt-in.

A genuinely empty input root retains existing stage-specific behavior. A nonempty input with no readable supported DICOM now fails. A resume run with no remaining scoped patients is not misclassified as an empty discovery failure.

## Unresolved dose-reference defect

The failing predicate was an RTDOSE with at least 1 explicit `ReferencedRTPlanSequence` UID but no UID resolving to an indexed RTPLAN. The previous loop continued without a merged row, fallback, or warning.

The metadata exporter now logs the dose path and every unresolved UID. Its summary reports unresolved reference and dose-object counts.

An explicitly referenced but unresolved dose does not fall back to the legacy filename core key. Such a fallback could attach the dose to a different plan and would violate the reference chain.

## Metadata table consistency

The failing predicate was a modality present in the discovered input with 0 extracted output rows. The previous behavior depended on which table was being produced.

The exporter now applies the same fail-loud rule to RTPLAN, RTDOSE, RTSTRUCT, RTRECORD, and CT. It raises `MetadataExportError` when discovered objects of one of these modalities produce no rows.

Treatment-record metadata now reads plan UIDs specifically from `ReferencedRTPlanSequence`. A recursive search for the first `ReferencedSOPInstanceUID` could otherwise return a non-plan reference.

## Dead text classifier

The unused `_classify_doses_legacy`, `_is_replan_text`, and `_is_boost_text` functions were removed. The active dose classifier does not read plan labels or descriptions into its decision rules.

Dose selection now uses DICOM references, treatment-record evidence, prescription signatures, and geometry. Free-text plan labels cannot silently determine whether plans are replacements or sequential phases.

## Dose plausibility warnings

The plausibility threshold is configurable as `max_total_dose_gy` and defaults to 100.0 Gy. A course receives separate warnings when either prescribed dose or delivered dose exceeds the threshold.

The warning identifies the field and value. This reconciles the former implausible-total-dose path with the separation between treatment intent and delivered treatment.

## Verification

Synthetic DICOM tests cover the 75.0 Gy versus 37.0 Gy example, null delivery without records, full delivery, duplicate beam records within a session, absent referenced plans, empty discovery, unresolved dose references, and configurable plausibility warnings.

The targeted regression set recorded 46 passing tests. The full repository suite recorded 861 passing tests, 1 skipped test, and 0 failures or errors under the supported interpreter.

## Evidence and reproducibility

The evidence ledger is `analysis/evidence_ledger.json`. The Kopernik cohort calculation is recorded in `analysis/output_fraction_census.py`, `analysis/kopernik-output-fraction-dose.json`, and `analysis/kopernik-419783-record-dose.json`.

The DFCI calculation is recorded in `analysis/dfci_table_census.py` and `analysis/dfci-delivered-dose-table.json`. The planning CT inventory and adjudication are recorded in `analysis/dfci-planning-ct-directory-audit.json` and `analysis/dfci-planning-ct-adjudication.json`.

The inputs outside the repository were read-only. The analyses did not alter the rebuilt cohorts or the DFCI source on `s1`.

## Remaining uncertainty and action

Delivered dose is not imputed when records are absent or insufficient. Downstream dose-response work should exclude or separately model null delivered doses rather than replacing them with 0.0 Gy or prescribed dose.

DFCI courses with negative reported-minus-delivered differences need clinical review before outcome analysis. The fields now retain the plan-level evidence needed for that review.

The corrected data contract supports the required clinical distinction. Use `delivered_dose_gy` for dose-response analyses and retain `total_prescription_gy` for treatment-intent analyses and intent-versus-delivery comparisons.
