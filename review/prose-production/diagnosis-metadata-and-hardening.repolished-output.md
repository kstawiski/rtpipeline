# Diagnosis of metadata export and cohort hardening

## Decision and consequence

RTpipeline now identifies CT, RTPLAN, RTSTRUCT, RTDOSE, and RTRECORD objects from the DICOM Modality tag. File names no longer decide which objects enter the cohort metadata tables. This removes the silent failure that hid every Kopernik plan, structure set, and dose.

Plan-to-dose association now follows the RTDOSE reference to the RTPLAN SOPInstanceUID. The old ARIA filename key remains only as a fallback when a dose has no DICOM plan reference. The exporter raises `MetadataExportError` when it discovers RTPLAN objects but cannot extract any plan rows.

DFCI's empty `metadata.xlsx` is a second defect in the plan-to-dose join. It is not caused by missing RT objects in organized course directories. The exporter reads the source DICOM root, where 1,219 dose references resolve to indexed plans, but no plan and dose share the legacy filename core key.

The completed cohort directories were inspected without modification. They were not rebuilt. The reported post-change behavior therefore comes from synthetic DICOM regression tests and direct source censuses, not from new production workbooks.

## Failure mechanism

The previous exporter enumerated files with prefix tests for `RP`, `RS`, `RD`, `RT`, and `CT`. It then required a `.dcm` extension. The plan-to-dose merge also required both filenames to match an ARIA-specific `R[PD].<digits>.<description>.dcm` pattern and to produce the same core key.

These predicates made filenames act as clinical metadata. They also failed silently. A valid `RTPLAN_1.dcm` object was absent from `plans.xlsx` because its name did not start with `RP`. A valid `RTDOSE_1.dcm` could not join to its referenced plan because neither filename supplied the required core key.

The `RT` prefix created a separate misclassification in Kopernik. Its 12,193 matching files comprised 375 RTPLAN, 2,231 RTSTRUCT, 871 RTDOSE, 3,247 RTIMAGE, and 5,469 RTRECORD objects. The completed `fractions.xlsx` contained 12,193 rows because the prefix selected every one of these modalities. The revised exporter selects the 5,469 RTRECORD objects by Modality.

## Cohort evidence

The census read every `.dcm` file under the configured source root. It used `pydicom.dcmread` with `stop_before_pixels=True` and `specific_tags=[Modality]` for classification. Plan, dose, and structure headers were then read for reference and target censuses. Patient identifiers and individual file paths were not written to the aggregate result files.

| Cohort | Source DICOM files | RTPLAN by Modality | RTSTRUCT by Modality | RTDOSE by Modality | Legacy RP | Legacy RS | Legacy RD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kopernik | 340,607 | 375 | 2,231 | 871 | 0 | 0 | 0 |
| DFCI | 415,562 | 1,561 | 3,548 | 1,233 | 1,561 | 3,548 | 1,233 |

The completed Kopernik export has 293,222 CT rows and 12,193 fraction rows. Its plan, structure-set, dosimetry, and merged metadata workbooks are absent. The measured Modality inventory matches the filename failure prediction and provides a positive inventory behind each absence claim.

The completed DFCI export has 352,706 CT rows, 1,561 plan rows, 3,548 structure-set rows, 1,233 dose rows, 13,997 fraction rows, and 0 merged metadata rows. Modality indexing found 352,707 CT objects. One CT object was therefore outside the legacy CT prefix predicate.

## DFCI metadata verdict

The DFCI source contains 1,561 distinct plan SOPInstanceUIDs and 1,233 dose references to plans. Of those references, 1,219 resolve to an indexed source plan. All 1,561 plan files and all 1,233 dose files produce a legacy core key, but the two key sets have no shared value.

This establishes an independent course-level metadata join defect. The evidence does not establish why 14 dose references do not resolve to an indexed plan. The revised join retains only explicit RTDOSE references that resolve to an indexed plan. It does not infer those 14 associations from labels, order, or approximate identifiers.

The separate organizer linkage defect may still leave RT objects absent from emitted course directories. That does not explain the empty merged metadata workbook because `export_metadata` reads `config.dicom_root`, not the emitted course directories. This task did not use course attachment as a substitute for source DICOM references.

## Metadata implementation

The exporter now builds one modality index for the requested source scope. Every candidate file is read without pixel data and with only the Modality tag requested. Filename prefixes remain as hints for common names, but the DICOM tag verifies every hint before classification.

Plans and doses carry internal SOP reference columns during assembly. The merge first joins each dose's `ReferencedRTPlanSequence` to the plan `SOPInstanceUID`. It uses the legacy ARIA core key only for doses with no explicit plan reference. Internal association columns are removed before workbook output.

A cohort with detected RTPLAN objects cannot end with an empty extracted plan table without an exception. Synthetic generic filenames produce populated plan, structure-set, and dosimetry tables. Synthetic ARIA filenames preserve the previous exported row content. A synthetic plan and dose with unrelated filenames associate through the dose's DICOM reference.

## Measured classification cost

Both measurements used 16 workers. The filename baseline and the tag index each performed a full source walk. The filename baseline performed only prefix checks after enumeration. The tag index read the Modality element without pixel data. These are wall-clock measurements on the named hosts and filesystems, not portable throughput guarantees.

| Cohort | Files | Filename scan | Modality-tag index | Added wall time | Ratio | Tag time per 1,000 files |
|---|---:|---:|---:|---:|---:|---:|
| Kopernik | 340,607 | 7.211 s | 325.810 s, or 5.43 min | 318.599 s | 45.18 | 0.957 s |
| DFCI | 415,562 | 5.834 s | 182.264 s, or 3.04 min | 176.429 s | 31.24 | 0.439 s |

The added scan is material but bounded at 3.04–5.43 minutes in these measurements. It prevents a false empty export. The two timings should not be compared as host performance because storage and cache state differed.

## F5 target definition

Production and the cohort probe now call one target-name function. It requires a GTV, CTV, or PTV token to begin at the start of the ROI name or after a non-alphanumeric boundary. This left boundary excludes embedded matches while preserving compact clinical names such as `PTVbt` and `CTVn`.

The function rejects target tokens preceded by the boolean-crop separator ` - `. It rejects names beginning with `marg` and leading-`z` helper names. `Pecherz - PTV`, `marg PTV2`, and `zPtvOpt` therefore do not satisfy the plan-and-dose target gate.

The full census confirmed the expected ceilings. All 230 DFCI and all 122 Kopernik plan-referenced structure sets that had target status under the permissive rule retain target status under the shared rule. No set lost target status in either cohort.

## F6 structure-set path ambiguity

The organizer now groups structure-set source paths by RTSTRUCT SOPInstanceUID before deciding whether the course has one structure set. Two paths carrying the same SOPInstanceUID resolve to one authoritative source identity. The structure set is copied, and CT selection still requires its referenced series.

A course with genuinely distinct RTSTRUCT identities fails the course gate instead of falling through to the largest CT series. The CT selector also returns `missing_reference` when reference-based selection is required but no structure source path is available.

The full census found 0 duplicate RTSTRUCT SOPInstanceUID groups among 3,548 DFCI structure files and 0 among 2,231 Kopernik files. This snapshot does not exercise the duplicate-copy branch in production. The synthetic duplicate-path regression test does.

## F7 CT-only cohorts

CT-only output remains available behind an explicit configuration choice. The default is off so failed RT linkage cannot masquerade as a valid CT-only cohort. A CT-only input now raises `CTOnlyCohortError` unless `organize.allow_ct_only_courses` is true or the CLI receives `--allow-ct-only-courses`.

When enabled, eligible volumetric CT series reach the existing CT-only course writer. Synthetic tests confirm both the default error and successful output under the explicit option. This restores a reachable path for diagnostic CT radiomics without weakening RT cohort handling.

## F8 label-text classifier

The unused legacy dose classifier and its private label-text helpers were deleted. The active reference and delivery-evidence classifier remains. A repository-wide Python source search found no `_classify_doses_legacy`, `_is_replan_text`, `_is_boost_text`, `_bboxes_overlap`, or `_prescription_similarity` symbol after the change.

A regression test asserts that the label-text classifier and helpers are not callable from the organizer module. No dose-selection rule was added that uses a plan label as evidence.

## Verification and limits

The final local suite under `/home/konrad/micromamba/envs/rtpipeline/bin/python` recorded 850 passed, 1 skipped, 0 failures, and 0 errors in 31.64 seconds. The JUnit record is `analysis/results/pytest-full.xml`. Synthetic DICOM tests contain no real patient data.

The source measurements are recorded in `analysis/results/metadata_modality_cost_kopernik.json` and `analysis/results/metadata_modality_cost_dfci.json`. Claim classification, calculations, and inference boundaries are recorded in `analysis/evidence_ledger.json`.

No production cohort was rebuilt, and no production output was modified. A rebuild remains necessary to measure the resulting workbook row counts and to evaluate the 14 unresolved DFCI dose references. The separate organizer linkage repair must also pass its own checks before cohort results are interpreted.
