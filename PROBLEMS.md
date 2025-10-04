# Pipeline Issues To Address

These are the outstanding clinical/technical concerns flagged after reviewing the current Snakemake run (2025-10-04). Track each item to closure.

## Radiomics Timeouts & Large Non-Clinical ROIs ✅
- **Status:** Fixed 2025-10-04. Radiomics wrapper now skips oversized or non-clinical masks (`Bones`, `CouchExterior`, `m1`, `m2`, etc.) via `radiomics.skip_rois` plus `radiomics.max_voxels`/`radiomics.min_voxels` config knobs. This prevents long-running conda calls from timing out and keeps QC noise down.
- **Follow-up:** Monitor future logs for residual timeouts; adjust the max-voxel ceiling if high-value ROIs are being skipped inadvertently.

## Radiomics Source Column Collapsed to "Merged" ✅
- **Status:** Fixed 2025-10-04. `radiomics_conda.py` infers provenance by comparing RS/RS_auto ROI lists (and custom config where available). Course workbooks and `_RESULTS/radiomics_ct.xlsx` now report `Manual`, `AutoTS`, or `Custom` labels (e.g. AutoTS 270 / Manual 129 / Custom 43 rows current cohort).
- **Follow-up:** When new auto-seg templates are added, confirm their ROI names normalize correctly so provenance stays accurate.

## Systemic Structure Cropping
- Every course QC report shows `overall_status = WARNING` with 20–39 cropped structures.
- Radiomics/DVH tables confirm many abdominal organs are flagged.
- **Action:** Investigate CT FOV vs. auto segmentation; consider excluding/annotating cropped organs so downstream analysis is aware.

## Custom Structure Warnings During DVH
- `Logs_Snakemake/stage_dvh.log` is full of "Source structure ... not found" messages (e.g. `pelvic_bones`, `bowel_bag`).
- Indicates mismatched naming between expected TotalSegmentator outputs and custom structure definitions.
- **Action:** Align custom-structure YAML with actual RTSTRUCT ROI names or adjust the importer to sanitize names before Boolean ops.

## Prescription Dose Missing for Some Courses
- `case_metadata.xlsx` shows `total_prescription_gy = 0` for 480007/2024-08 and 487009/2025-03.
- DVH normalization may be unreliable for these courses.
- **Action:** Extend prescription parser to capture `DoseReference` info for all plans/beam sets.

## Metadata: Reconstruction Algorithm Empty
- `ct_reconstruction_algorithm` column is blank for all courses despite docs requiring it.
- **Action:** Map vendor-specific DICOM tags (e.g. `ReconstructionAlgorithm`, `ConvolutionKernel`, `IterativeReconstructionMethod`) into the metadata exporter.

## In case_metadata.xlsx fractions_count like 75, 90  don't make sense. Is muti-stage treatment being handled correctly in all steps of pipeline?

Keep this file updated as issues are fixed or new ones surface.
