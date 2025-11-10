# Pipeline Issues To Address

These are the outstanding clinical/technical concerns flagged after reviewing the current Snakemake run (2025-10-04). Track each item to closure.

## Radiomics Timeouts & Large Non-Clinical ROIs ✅
- **Status:** Fixed 2025-10-04. Radiomics wrapper now skips oversized or non-clinical masks (`Bones`, `CouchExterior`, `m1`, `m2`, etc.) via `radiomics.skip_rois` plus `radiomics.max_voxels`/`radiomics.min_voxels` config knobs. This prevents long-running conda calls from timing out and keeps QC noise down.
- **Follow-up:** Monitor future logs for residual timeouts; adjust the max-voxel ceiling if high-value ROIs are being skipped inadvertently.

## Radiomics Source Column Collapsed to "Merged" ✅
- **Status:** Fixed 2025-10-04. `radiomics_conda.py` infers provenance by comparing RS/RS_auto ROI lists (and custom config where available). Course workbooks and `_RESULTS/radiomics_ct.xlsx` now report `Manual`, `AutoTS`, or `Custom` labels (e.g. AutoTS 270 / Manual 129 / Custom 43 rows current cohort).
- **Follow-up:** When new auto-seg templates are added, confirm their ROI names normalize correctly so provenance stays accurate.

## Systemic Structure Cropping ✅
- **Status:** Solution fully implemented 2025-11-09, expanded with multi-region support 2025-11-10.
- **Problem Reframed:** The real issue was not the cropping warnings themselves, but that percentage DVH metrics (V95%, V20Gy) were meaningless because volume denominators varied across patients due to different CT field-of-view extents.
- **Solution:** Systematic CT cropping module (`rtpipeline/anatomical_cropping.py`) that crops all CTs to consistent anatomical boundaries using TotalSegmentator landmarks.
- **Supported Regions:**
  - **Pelvis**: L1 vertebra → femoral heads + margin (default: 10cm inferior)
  - **Thorax**: C7/lung apex → L1/diaphragm (default: 2cm margins)
  - **Abdomen**: T12/L1 → L5 vertebra (default: 2cm margins)
  - **Head & Neck**: Brain/skull apex → C7/clavicles (default: 2cm margins)
  - **Brain**: Brain boundaries (default: 1cm margins)
- **Integration:** Complete CLI, DVH, and radiomics integration. Automatically uses cropped structures when enabled.
- **Configuration:** Added `ct_cropping` section to `config.yaml` (disabled by default, opt-in for multi-patient studies).
- **Benefits:** Makes percentage DVH metrics (V%, D%) meaningful for cross-patient comparison. Enables valid statistical analysis on percentage-based endpoints.
- **Follow-up:** Clinical validation on full cohort needed. Verify cropping boundaries are appropriate for clinical use. See `TESTING_CT_CROPPING.md` for testing procedures.

## Custom Structure Warnings During DVH
- `Logs_Snakemake/stage_dvh.log` is full of "Source structure ... not found" messages (e.g. `pelvic_bones`, `bowel_bag`).
- Indicates mismatched naming between expected TotalSegmentator outputs and custom structure definitions.
- **Action:** Align custom-structure YAML with actual RTSTRUCT ROI names or adjust the importer to sanitize names before Boolean ops.

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
- Currently segmentation still runs sequentially because the CLI didn't expose a worker limit. Configuration now allows `segmentation.workers` in `config.yaml`, but this needs validation with a full pipeline run to confirm multiple courses execute in parallel without overrunning GPU/CPU resources.

## Recent Enhancements (2025-11-09)

### TotalSegmentator DICOM RTSTRUCT Output ✅
- **Status:** Updated 2025-11-09. TotalSegmentator now outputs DICOM RTSTRUCT directly instead of DICOM-SEG.
- **Files Modified:** `rtpipeline/segmentation.py`, `Code/05 TotalSegmentator.py`
- **Impact:** RTSTRUCT files are directly compatible with clinical systems (TPS, contouring software). No conversion needed.
- **Requirement:** Requires `rt_utils` package (already in dependencies).

### Security Fixes ✅
- **Status:** Fixed 2025-11-09. Addressed critical security vulnerabilities.
- **Fixed Issues:**
  - 2 command injection vulnerabilities (shlex.quote() added)
  - 1 unsafe pickle deserialization (RestrictedUnpickler implemented)
  - Path validation helper added to utils.py
  - Custom models disabled by default in config
- **Impact:** Pipeline now significantly more secure for production use.
- **Details:** See `DEEP_DEBUG_REPORT.md` and `IMPROVEMENTS_SUMMARY.md`

### Pre-flight Validation Command ✅
- **Status:** Added 2025-11-09. New `rtpipeline validate` command.
- **Purpose:** Checks environment configuration before running pipeline.
- **Checks:** DICOM root exists, config file valid, external tools available, Python packages installed.
- **Usage:** `rtpipeline validate --config config.yaml --dicom-root Example_data`

Keep this file updated as issues are fixed or new ones surface.
