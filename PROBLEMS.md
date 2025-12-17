# Pipeline Issues To Address

These are the outstanding clinical/technical concerns flagged after reviewing the current Snakemake run (2024-10-04). Track each item to closure.

## Radiomics Timeouts & Large Non-Clinical ROIs ✅
- **Status:** Fixed 2024-10-04. Radiomics wrapper now skips oversized or non-clinical masks (`Bones`, `CouchExterior`, `m1`, `m2`, etc.) via `radiomics.skip_rois` plus `radiomics.max_voxels`/`radiomics.min_voxels` config knobs. This prevents long-running conda calls from timing out and keeps QC noise down.
- **Follow-up:** Monitor future logs for residual timeouts; adjust the max-voxel ceiling if high-value ROIs are being skipped inadvertently.

## Radiomics Source Column Collapsed to "Merged" ✅
- **Status:** Fixed 2024-10-04. `radiomics_conda.py` infers provenance by comparing RS/RS_auto ROI lists (and custom config where available). Course workbooks and `_RESULTS/radiomics_ct.xlsx` now report `Manual`, `AutoTS`, or `Custom` labels (e.g. AutoTS 270 / Manual 129 / Custom 43 rows current cohort).
- **Follow-up:** When new auto-seg templates are added, confirm their ROI names normalize correctly so provenance stays accurate.

## Systemic Structure Cropping
- Every course QC report shows `overall_status = WARNING` with 20–39 cropped structures.
- Radiomics/DVH tables confirm many abdominal organs are flagged.
- **Action:** Investigate CT FOV vs. auto segmentation; consider excluding/annotating cropped organs so downstream analysis is aware.

### Pelvic CT Cropping fails when L1 not detected
- **Status:** New (2025-11-26). The Docker validation run shows repeated `CT cropping failed for ... L1 vertebra not found in TotalSegmentator output` warnings (see `crop_ct/428459_2019-11.log` under `/data/logs` inside the container).
- **Impact:** Cropping bails out entirely for the affected course, so DVH/radiomics are computed on the uncropped CT (inconsistent with other patients) and QC remains noisy.
- **Action:** Implement a fallback landmark (e.g., T12 or patient-bounding-box) when L1 is missing, and surface a structured QC flag (`ct_cropping.l1_missing=true`) so downstream analytics can filter/annotate these cases.
- **Follow-up:** Add regression tests that run `crop_ct_course` against Example_data to ensure the fallback engages and the warning disappears once implemented.

### Pelvic CT Cropping fails when femoral heads are missing
- **Status:** New (2025-11-26). The Example_data cohort still contains cases (e.g., `487009/2025-09`, see `Logs/crop_ct/487009_2025-09.log`) where TotalSegmentator never labels the femoral heads, so the current pelvic-cropping heuristic aborts with `Femoral heads not found... Available structures: []`.
- **Impact:** Courses that trigger this condition revert to the uncropped CT stack, but downstream QC and DVH tables still assume pelvic-cropped geometry, so organ coverage metrics become inconsistent across patients.
- **Action:** Add a fallback inferior bound when femoral heads cannot be resolved (e.g., derive from couch + pelvis bbox, or re-run TotalSegmentator with pelvic ROI-only job) and emit a structured flag (`ct_cropping.femoral_heads_missing=true`) so analytics can filter/annotate impacted courses.

## Custom Structure Warnings During DVH
- `Logs_Snakemake/stage_dvh.log` is full of "Source structure ... not found" messages (e.g. `pelvic_bones`, `bowel_bag`).
- Indicates mismatched naming between expected TotalSegmentator outputs and custom structure definitions.
- **Action:** Align custom-structure YAML with actual RTSTRUCT ROI names or adjust the importer to sanitize names before Boolean ops.

## DVH Catastrophic Underestimation for Thin-Slice Contours ✅
- **Status:** Fixed 2025-12-12. Some cohorts with contour planes at higher resolution than the RTDOSE grid (e.g., CT/RTSTRUCT contours every 1.5 mm with dose planes every 3.0 mm) produced clinically impossible DVHs (tiny volumes, very low Dmax).
- **Root cause:** `snap_rtstruct_to_dose_grid()` was over-aggressive and snapped contour plane `z` coordinates across half-plane distances, collapsing adjacent contour planes onto the same dose plane. dicompyler-core then interpreted the second contour as a hole, catastrophically underestimating area/volume and all DVH metrics.
- **Fix:** Cap snapping tolerance to `< 0.5 × min(dose-plane spacing)` (implemented as `0.49 * min_spacing`) to prevent collapsing real intermediate planes; keep snapping only for small sub-plane deviations.
- **Follow-up:** Consider adding an integration test that computes DVH on an Example_data case with mismatched contour/dose plane spacing (and asserts BODY Dmax is within a plausible range).

### dicompylercore missing in Docker image
- **Status:** New (2025-11-26). During the Dockerized Example_data run, every radiomics job logs `Failed to import dicompylercore...` (see `Logs/radiomics/487009_2025-09.log`).
- **Impact:** When dicompylercore is unavailable the custom-structure pre-processing inside the radiomics stage is skipped, so downstream features never include Boolean composites such as pelvic_bones, and QC silently reports partial coverage.
- **Action:** Bundle `dicompyler-core` inside the runtime environment (both base image and Conda env) or gate the radiomics stage until the dependency is present. Emit an explicit QC flag when custom structure generation is skipped so analysts can filter those rows.

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
