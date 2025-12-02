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

### Pelvic CT Cropping fails when femoral heads are missing
- **Status:** New (2025-11-26). During the Example_data dry-run inside Docker, course `487009/2025-09` logs `Femoral heads not found in TotalSegmentator output` (`Logs/crop_ct/487009_2025-09.log`) and aborts the pelvic cropping stage.
- **Impact:** Those patients quietly fall back to uncropped CTs, so QC/DVH comparisons mix cropped and uncropped cohorts.
- **Action:** Provide a fallback inferior bound (e.g., patient bounding box) and raise a structured QC flag (`ct_cropping.femoral_heads_missing=true`) until segmentation improves.

## Custom Structure Warnings During DVH
- `Logs_Snakemake/stage_dvh.log` is full of "Source structure ... not found" messages (e.g. `pelvic_bones`, `bowel_bag`).
- Indicates mismatched naming between expected TotalSegmentator outputs and custom structure definitions.
- **Action:** Align custom-structure YAML with actual RTSTRUCT ROI names or adjust the importer to sanitize names before Boolean ops.

### dicompylercore missing in Docker image
- **Status:** New (2025-11-26). Radiomics logs report `Failed to import dicompylercore` for every course (`Logs/radiomics/487009_2025-09.log`), so custom Boolean structures are skipped.
- **Impact:** Radiomics never emits pelvic_bones/bowel_bag derived masks, making downstream feature tables incomplete.
- **Action:** Bundle `dicompyler-core` with the runtime environment (Docker + Conda env) and raise a QC flag whenever the dependency is absent.

## Prescription Dose Missing for Some Courses ✅
- **Status:** Fixed 2025-10-04. Prescription inference now falls back to FractionGroup / ReferencedBeam dose totals when `TargetPrescriptionDose` is absent. Current metadata shows 50 Gy (480007/2024-08) and 62 Gy (487009/2025-03).
- **Follow-up:** Double-check atypical totals (e.g., 62 Gy) with clinical team; confirm multi-plan sums remain correct when additional boosts are present.

## Metadata: Reconstruction Algorithm Empty
- `ct_reconstruction_algorithm` column is blank for all courses despite docs requiring it.
- **Action:** Map vendor-specific DICOM tags (e.g. `ReconstructionAlgorithm`, `ConvolutionKernel`, `IterativeReconstructionMethod`) into the metadata exporter.

## Fraction Count Inflation in `case_metadata.xlsx`
- **Status:** Fixed 2025-10-04. Fraction aggregation now deduplicates delivery records by plan/fraction number, so course counts match clinical schedules (25/30 fractions etc.).
- **Follow-up:** Monitor `Logs_Snakemake/stage_organize.log` for the residual `'set' object is not subscriptable'` warning while building the summary—this indicates the pandas-based rollup still hits an edge case. Need to trace and remove the warning path so the organizer can run cleanly without falling back to manual corrections.

## Snakemake Segmentation Parallelism
- Currently segmentation still runs sequentially because the CLI didn't expose a worker limit. Configuration now allows `segmentation.max_workers` in `config.yaml` (legacy `segmentation.workers` still works), but this needs validation with a full pipeline run to confirm multiple courses execute in parallel without overrunning GPU/CPU resources.

Keep this file updated as issues are fixed or new ones surface.
