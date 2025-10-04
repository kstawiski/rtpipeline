# Pipeline Issues To Address

These are the outstanding clinical/technical concerns flagged after reviewing the current Snakemake run (2025-10-04). Track each item to closure.

## 1. DVH Aggregation Includes Duplicated Targets
- `RS_custom.dcm` re-publishes the original manual PTVs, so `Data_Snakemake/_RESULTS/dvh_metrics.xlsx` lists both `Segmentation_Source = "Manual"` and `"Custom"` entries for the same ROI.
- The "Custom" rows drag cohort statistics down (e.g. PTV D95 ≈ 0 Gy).
- **Action:** Filter DVH aggregation to exclude duplicated manual targets, or tag the re-added ROIs with their original source so only one copy survives.

## 2. Fraction Workbook Has Duplicates & Missing Numbers
- `_RESULTS/fractions.xlsx` contains repeated rows per treatment date and no populated `fraction_number` values (all NaN).
- `fractions_count` in `case_metadata.xlsx` reflects the inflated counts.
- **Action:** Deduplicate by SOP/date and recover delivered fraction indices; update metadata totals accordingly.

## 3. Radiomics Timeouts & Large Non-Clinical ROIs
- Twelve ROIs failed pyradiomics extraction (nine timeouts for huge structures like `Bones`, `Couch Exterior`, `m1/m2`; three "ROI too small" aborts).
- These still produce QC warnings and missing features.
- **Action:** Trim the radiomics ROI list (exclude couch/large union masks), or increase timeout/voxel limits for critical structures.

## 4. Systemic Structure Cropping
- Every course QC report shows `overall_status = WARNING` with 20–39 cropped structures.
- Radiomics/DVH tables confirm many abdominal organs are flagged.
- **Action:** Investigate CT FOV vs. auto segmentation; consider excluding/annotating cropped organs so downstream analysis is aware.

## 5. Custom Structure Warnings During DVH
- `Logs_Snakemake/stage_dvh.log` is full of "Source structure ... not found" messages (e.g. `pelvic_bones`, `bowel_bag`).
- Indicates mismatched naming between expected TotalSegmentator outputs and custom structure definitions.
- **Action:** Align custom-structure YAML with actual RTSTRUCT ROI names or adjust the importer to sanitize names before Boolean ops.

## 6. Prescription Dose Missing for Some Courses
- `case_metadata.xlsx` shows `total_prescription_gy = 0` for 480007/2024-08 and 487009/2025-03.
- DVH normalization may be unreliable for these courses.
- **Action:** Extend prescription parser to capture `DoseReference` info for all plans/beam sets.

## 7. Metadata: Reconstruction Algorithm Empty
- `ct_reconstruction_algorithm` column is blank for all courses despite docs requiring it.
- **Action:** Map vendor-specific DICOM tags (e.g. `ReconstructionAlgorithm`, `ConvolutionKernel`, `IterativeReconstructionMethod`) into the metadata exporter.

## 8. Radiomics Source Column Collapsed to "Merged"
- All radiomics rows use `segmentation_source = "Merged"`, losing manual vs. auto provenance.
- **Action:** Bubble the original segmentation source through the conda fallback so analysts can stratify features.

Keep this file updated as issues are fixed or new ones surface.
