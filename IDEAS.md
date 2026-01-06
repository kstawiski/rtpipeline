# RTpipeline Improvement Ideas

This document logs improvement suggestions and feature ideas identified during the scientific and technical code review. These are **enhancement ideas** (not bugs), logged for future consideration.

## Format

Each idea follows this format:
- **ID**: `IDEA-XXX`
- **Stage**: Pipeline stage where idea was identified
- **Priority**: Low / Medium / High
- **Description**: What could be improved
- **Rationale**: Why this improvement matters

---

## Ideas Log

### IDEA-001: Configurable Dose Plausibility Threshold

- **Stage**: 1. DICOM Organization
- **Priority**: Medium
- **Description**: The `max_total_dose_gy` parameter in `_classify_doses()` is hardcoded to 100 Gy. This should be configurable via config.yaml to support hypofractionated regimens (SBRT, SRS) where total doses can exceed 100 Gy.
- **Rationale**: SBRT lung treatments may prescribe 54 Gy in 3 fractions, brain SRS can reach 120 Gy+ when summing multiple targets. The hardcoded 100 Gy threshold would incorrectly flag these as plausibility warnings.
- **References**: RTOG 0236 (SBRT lung), RTOG 0631 (spine SRS)

### IDEA-002: Use Standard DICOM Frame of Reference Tag for RTDOSE

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: The `rt_details.py` module uses tag (0x3006, 0x0024) for Frame of Reference UID extraction from RTDOSE files. This is the Referenced Frame of Reference UID from the Structure Set Module. The standard tag for Frame of Reference UID is (0x0020, 0x0052).
- **Rationale**: Most TPS export the same UID in both tags, but some may differ. Using the standard tag ensures broader DICOM compliance and prevents potential course grouping errors.
- **References**: DICOM PS3.3 C.7.4.1 (Frame of Reference Module), DICOM PS3.3 C.8.8.3 (RT Dose Module)

### IDEA-003: Adaptive Therapy Workflow Support

- **Stage**: 1. DICOM Organization
- **Priority**: Medium
- **Description**: The current replan detection (Phase 3 of dose classification) uses intention-to-treat (ITT) logic that keeps only the first plan. For adaptive therapy workflows, multiple re-plans may be intentionally delivered and should be summed.
- **Rationale**: Adaptive radiotherapy (ART) protocols like ARTFORCE involve planned re-optimization based on anatomical changes. The current ITT logic would incorrectly discard these legitimate dose contributions.
- **References**: ARTFORCE trial protocol, Elekta MOSAIQ adaptive workflows

### IDEA-004: Z-Extent Calculation Improvement for Dose Grid

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: In `_extract_dose_metadata()`, when `GridFrameOffsetVector` is empty, z-extent assumes 1mm spacing. Should check for SliceThickness attribute in the RTDOSE file as a fallback.
- **Rationale**: Prevents incorrect bounding box calculations that could affect dose grid overlap detection in Phase 2.5 of dose classification.
- **References**: DICOM PS3.3 C.8.8.3.4 (Grid Frame Offset Vector)

### IDEA-005: Course Clustering Algorithm Enhancement

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: The date-based course splitting in `group_by_course()` uses a simple pivot-based approach that may incorrectly split interleaved courses (e.g., concurrent chemo-RT where treatments alternate days).
- **Rationale**: While rare, interleaved treatment schedules exist. A more sophisticated clustering algorithm (e.g., DBSCAN on date differences) could handle these cases better.
- **References**: N/A - edge case enhancement

### IDEA-006: Atomic Cache Writes for Crash Safety

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: The `save_caches()` method in `dicom_copy.py` writes JSON cache files directly without atomic rename pattern. A crash during write could corrupt the cache.
- **Rationale**: Atomic writes (write to temp file, then rename) are the standard pattern for crash-safe file updates. While cache corruption only triggers a rebuild on next run, atomic writes would improve robustness.
- **References**: Python tempfile.NamedTemporaryFile + os.replace() pattern

### IDEA-007: Checksum Verification Failure Handling

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: In `dicom_copy.py`, if checksum verification fails on both initial copy and re-copy, the error is logged but the function returns success. Consider raising an exception or returning a failure status.
- **Rationale**: Silent acceptance of corrupted files could lead to downstream processing issues. Explicit failure handling would allow callers to decide how to proceed.
- **References**: N/A - defensive programming improvement

### IDEA-008: Copy Counter Limit for Name Collisions

- **Stage**: 1. DICOM Organization
- **Priority**: Very Low
- **Description**: The `copy_dicom_into()` method has an unbounded while loop for handling name collisions. Adding a reasonable limit (e.g., 10000) would prevent theoretical infinite loops.
- **Rationale**: While extremely unlikely in practice, a bounded loop is more defensive. Could log an error and raise an exception if limit is reached.
- **References**: N/A - defensive programming improvement

### IDEA-009: SHA-256 Option for Checksum Verification

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: The checksum verification in `dicom_copy.py` uses MD5. While MD5 is fine for integrity checking (not security), offering SHA-256 as an option via `file_sha256()` from utils.py would provide a more modern alternative.
- **Rationale**: Some institutional policies require SHA-256 for data integrity verification. The infrastructure is already in place (`file_sha256()` exists in utils.py).
- **References**: NIST recommendations on hash functions

### IDEA-010: Secondary Criteria for Primary CT Series Selection

- **Stage**: 2. CT Processing
- **Priority**: Low
- **Description**: The `pick_primary_series()` function selects the CT series with the most slices. When multiple series have the same slice count (tie), selection is arbitrary (Python's `max` behavior). Add secondary criteria for tie-breaking.
- **Rationale**: In cases where multiple series have the same slice count (e.g., axial and coronal reconstructions from the same scan), the current behavior is non-deterministic. Preferring axial orientation or thinner slices would improve consistency.
- **References**: N/A - workflow consistency improvement

### IDEA-011: Use ImagePositionPatient for CT Slice Sorting

- **Stage**: 2. CT Processing
- **Priority**: Medium
- **Description**: Replace InstanceNumber-based sorting in `index_ct_series()` with ImagePositionPatient (0020,0032) projected onto the image plane normal vector. This is the DICOM-standard method for spatial slice ordering.
- **Rationale**: InstanceNumber is unreliable across vendors - some set all slices to "1" or omit it entirely. ImagePositionPatient is mandatory for CT images and provides reliable spatial ordering. This would also enable better validation of slice consistency.
- **References**: DICOM PS3.3 C.7.6.2 (Image Plane Module), comp.protocols.dicom best practices

### IDEA-012: Slice Count Validation After NIfTI Conversion

- **Stage**: 2. CT Processing
- **Priority**: Low
- **Description**: Add validation that compares the number of DICOM slices in the source directory with the number of slices in the resulting NIfTI file. Log a warning if they differ.
- **Rationale**: dcm2niix may exclude non-equidistant slices or fail to convert certain enhanced DICOM formats. A post-conversion validation would catch these issues early and alert users to potential data loss.
- **References**: dcm2niix documentation on edge cases

### IDEA-013: Post-Segmentation Geometry Validation

- **Stage**: 3. Segmentation
- **Priority**: Low
- **Description**: Add validation that verifies output segmentation masks have the same geometry (shape, affine transform, voxel spacing) as the input CT NIfTI. Log a warning if they differ.
- **Rationale**: While TotalSegmentator guarantees geometry preservation, rare bugs or version mismatches could cause subtle geometry misalignments. Post-segmentation validation would catch these issues before they propagate to DVH/radiomics extraction, where misaligned masks produce incorrect results.
- **References**: nibabel.spatially_same() or SimpleITK geometry comparison

### IDEA-014: Call TotalSegmentator Environment Validation

- **Stage**: 3. Segmentation
- **Priority**: Low
- **Description**: The `_validate_totalseg_environment()` function (lines 300-335 in segmentation.py) is defined but never called. Consider calling it before the first TotalSegmentator invocation to provide early error detection.
- **Rationale**: Early validation would catch NumPy version incompatibilities, missing TotalSegmentator installation, or environment issues before running potentially long segmentation jobs. Currently, users only discover these issues when segmentation fails.
- **References**: N/A - defensive programming improvement

### IDEA-015: Remove Deprecated segment_extra_models_mr Function

- **Stage**: 3. Segmentation
- **Priority**: Very Low
- **Description**: The `segment_extra_models_mr()` function (lines 1042-1044) is marked deprecated with a no-op body. Schedule removal in the next major version release.
- **Rationale**: Code hygiene - deprecated functions that do nothing should be removed to avoid confusion. The MR segmentation is now handled per-course within `segment_course()`.
- **References**: Semantic versioning (removal in major version)

### IDEA-016: TotalSegmentator Version Pinning for Reproducibility

- **Stage**: 3. Segmentation
- **Priority**: Medium
- **Description**: Log or record the TotalSegmentator version used for each segmentation run in the manifest.json. Consider adding a configurable version requirement to ensure reproducibility.
- **Rationale**: Different TotalSegmentator versions may produce slightly different segmentation results due to model updates. For research reproducibility, knowing which version was used is important. Version pinning would ensure consistent results across runs.
- **References**: Wasserthal et al. 2023, Radiology: AI; TotalSegmentator changelog

### IDEA-017: Add Trainer and Plans Support for nnU-Net v2 Custom Models

- **Stage**: 4. Custom Models
- **Priority**: Medium
- **Description**: The `_run_nnunet_prediction()` function adds `-tr` (trainer) and `-p` (plans) arguments only for nnU-Net v1 models. nnU-Net v2 also supports these arguments but they're currently ignored in the command construction.
- **Rationale**: Custom models trained with non-default trainers (e.g., `nnUNetTrainer_250epochs`) or custom plans will fail to predict correctly because the inference uses default settings instead of the configured ones. This is scientifically important for reproducibility.
- **References**: [nnU-Net v2 documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

### IDEA-018: Validate Label Presence in nnU-Net Output

- **Stage**: 4. Custom Models
- **Priority**: Low
- **Description**: The `_make_binary_masks()` function assumes all configured labels exist in the nnU-Net output. Add validation to warn when expected labels are missing from the segmentation output (i.e., no voxels predicted for a structure).
- **Rationale**: Empty masks are silently created for missing labels, which may indicate a model mismatch or unexpected data. A warning would help debug custom model configuration issues.
- **References**: N/A - diagnostic improvement

### IDEA-019: Post-Inference Geometry Validation for Custom Models

- **Stage**: 4. Custom Models
- **Priority**: Low
- **Description**: Add validation that verifies the output segmentation from nnU-Net has the same geometry (shape, spacing, origin, direction) as the input CT NIfTI before proceeding to mask extraction.
- **Rationale**: Geometry mismatches between input and output could cause subtle errors in DVH and radiomics extraction. While nnU-Net typically preserves geometry, validation provides an early warning for unusual configurations.
- **References**: nibabel.spatially_same() or SimpleITK geometry comparison

### IDEA-020: Support Non-Contiguous Label Indices

- **Stage**: 4. Custom Models
- **Priority**: Low
- **Description**: The `_make_binary_masks()` function assumes labels are 1-indexed and contiguous. Consider supporting a label mapping from the nnU-Net `plans.json` file to handle models with non-contiguous label indices or different background conventions.
- **Rationale**: Some custom models may use non-standard label mappings. The current code would produce incorrect mask assignments if labels are non-contiguous (e.g., using indices 1, 2, 5 instead of 1, 2, 3).
- **References**: nnU-Net plans.json format, dataset.json format

### IDEA-021: Log nnU-Net Model Version for Reproducibility

- **Stage**: 4. Custom Models
- **Priority**: Low
- **Description**: Record the nnU-Net version and checkpoint metadata (e.g., training date, fold information) in the output manifest.json for each custom model inference.
- **Rationale**: Similar to IDEA-016 for TotalSegmentator, knowing which model version was used is important for research reproducibility. Custom models may be updated over time.
- **References**: nnU-Net checkpoint structure (checkpoint_final.pth metadata)

### IDEA-022: Implement True Asymmetric Margin Support

- **Stage**: 5. Custom Structures
- **Priority**: Medium
- **Description**: The `apply_margin()` function in `custom_structures.py` only supports uniform (isotropic) margins. When asymmetric margins are specified (e.g., different anterior/posterior expansion), the code falls back to using the maximum margin value, which over-expands in some directions.
- **Rationale**: In radiotherapy planning, asymmetric margins are clinically important. For example, a PTV might require 5mm anterior, 10mm posterior, 5mm lateral, and 3mm superior/inferior margins. The current implementation would use 10mm in all directions, which is geometrically incorrect and could lead to unnecessarily large irradiation volumes.
- **References**: ICRU Report 62 (margins for PTV), RTOG structure naming conventions for asymmetric margins

### IDEA-023: Add Negative Margin (Erosion) Warning for Small Structures

- **Stage**: 5. Custom Structures
- **Priority**: Low
- **Description**: Add a warning when applying negative margins (erosion) to small structures that could result in empty or very small output masks.
- **Rationale**: When eroding a structure that is smaller than the erosion distance, the result will be empty. Currently this is caught after the fact (lines 368-373), but a pre-operation warning based on structure size vs margin distance would provide better diagnostic feedback.
- **References**: N/A - diagnostic improvement

### IDEA-024: Add Fallback Logic for Non-Pelvic Cropping Regions

- **Stage**: 6. CT Cropping
- **Priority**: Medium
- **Description**: The `determine_pelvic_crop_boundaries()` function has excellent multi-level fallback logic when landmarks are missing (CT extent → best available vertebra → 0.0mm). However, `determine_thorax_crop_boundaries()`, `determine_abdomen_crop_boundaries()`, `determine_head_neck_crop_boundaries()`, and `determine_brain_crop_boundaries()` raise ValueError when required landmarks are missing.
- **Rationale**: For specialized regions (thorax, abdomen, etc.), strict landmark requirements are appropriate. However, adding optional fallback to CT extent (similar to pelvic) would make the cropping more robust for edge cases where segmentation partially fails. This could be controlled by a `strict=True` parameter that defaults to the current behavior.
- **References**: N/A - workflow robustness improvement

### IDEA-025: Document Clamping Percentage for Research Transparency

- **Stage**: 6. CT Cropping
- **Priority**: Low
- **Description**: The cropping metadata documents whether superior/inferior axes were clamped, but doesn't report the percentage of the requested crop region that falls outside the CT extent. Adding this metric would help researchers understand data quality.
- **Rationale**: For cross-patient comparison studies, knowing that 5% of patients had their pelvic cropping clamped superiorly vs 30% provides important context for interpreting results. This is particularly relevant when L1 vertebra falls outside typical prostate CT planning scans.
- **References**: IBSI guidelines on reporting preprocessing parameters

### IDEA-026: Add Geometry Validation for Cropped Images

- **Stage**: 6. CT Cropping
- **Priority**: Low
- **Description**: After cropping CT and masks, validate that all cropped images have consistent geometry (same origin, spacing, size, direction). This would catch rare cases where individual mask files have slightly different geometry than the CT.
- **Rationale**: Geometry mismatches between CT and masks in the cropped set would cause downstream DVH and radiomics extraction errors. Early validation would catch these issues before they propagate.
- **References**: SimpleITK geometry comparison functions

### IDEA-027: Consider Removing Cropped RTSTRUCT Generation

- **Stage**: 6. CT Cropping
- **Priority**: Medium
- **Description**: The `_create_rtstruct_from_cropped_masks()` function attempts to create an RTSTRUCT from cropped masks aligned to the original (uncropped) CT DICOM. This is conceptually problematic because the cropped masks have different z-extent than the original CT. Consider deprecating this feature or documenting its limitations clearly.
- **Rationale**: RTSTRUCT files should be geometrically consistent with their referenced CT series. Creating an RTSTRUCT that references uncropped CT but contains contours only for the cropped region may confuse downstream systems expecting full coverage. The NIfTI cropped masks are sufficient for pipeline-internal use.
- **References**: DICOM PS3.3 C.8.8.5 (ROI Contour Module)

### IDEA-028: Add D100 Metric for TG-263 Compatibility

- **Stage**: 7. DVH Analysis
- **Priority**: Low
- **Description**: The code computes `DminGy` using `abs_dvh.min` but doesn't explicitly output D100 as a named metric. AAPM TG-263 defines D100 as the minimum dose covering 100% of the volume, which is equivalent to Dmin. Add explicit D100Gy metric for naming consistency with TG-263 conventions.
- **Rationale**: While DminGy is mathematically equivalent to D100, clinical reporting tools and multi-institutional data aggregation often expect TG-263 nomenclature (D100, D95, etc.). Explicit naming improves interoperability.
- **References**: AAPM TG-263 Report (Section 9.2), ICRU Report 83

### IDEA-029: Expand Vxx Dose Range for SBRT Support

- **Stage**: 7. DVH Analysis
- **Priority**: Medium
- **Description**: The code computes V1Gy through V60Gy (lines 131-134), which is insufficient for SBRT and SRS prescriptions that can exceed 60 Gy. Expand the range to V100Gy or make it configurable.
- **Rationale**: SBRT lung treatments prescribe 54 Gy in 3 fractions (equivalent dose >100 Gy BED), spine SRS can deliver 18-24 Gy in single fractions, and cranial SRS often exceeds 60 Gy total. The current V60Gy limit truncates clinically relevant dosimetric data for these increasingly common treatment modalities.
- **References**: RTOG 0236 (SBRT lung), RTOG 0631 (spine SRS), ASTRO SBRT guidelines

### IDEA-030: Add Explicit Dose Unit Validation

- **Stage**: 7. DVH Analysis
- **Priority**: Low
- **Description**: The DVH code relies on dicompyler-core to handle dose unit conversion (DoseGridScaling, DoseUnits) but doesn't explicitly validate that the returned values are in Gy. Add validation that checks DoseUnits tag and logs a warning if unexpected units are encountered.
- **Rationale**: While dicompyler-core handles this correctly in most cases, some TPS may export dose in non-standard units (e.g., relative Gy, cGy with incorrect scaling). Explicit validation would catch these edge cases before they propagate to aggregate results.
- **References**: DICOM PS3.3 C.8.8.3.2 (Dose Units), quality_control.py already validates DoseUnits

### IDEA-031: Add Configurable DVH Bin Width

- **Stage**: 7. DVH Analysis
- **Priority**: Low
- **Description**: Dicompyler-core uses 1 cGy bin width by default. Consider making this configurable for scenarios where coarser binning (e.g., 10 cGy) would reduce DVH curve JSON size without clinically relevant precision loss.
- **Rationale**: The DVH curves JSON can become large for high-dose treatments. For web visualization or export purposes, coarser binning reduces file size. The current 1 cGy resolution is appropriate for clinical QA but may be excessive for research aggregation.
- **References**: N/A - performance/usability improvement

### IDEA-032: Use dicompyler-core DVH.statistic() for Dx Metrics

- **Stage**: 7. DVH Analysis
- **Priority**: Low
- **Description**: The code implements custom `_dose_at_fraction()` for Dx metrics instead of using dicompyler-core's built-in `DVH.statistic()` method which supports D100, D98, D95, etc. directly. Consider using the library's method for consistency with dicompyler ecosystem.
- **Rationale**: The custom implementation is mathematically correct (linear interpolation on cumulative DVH), but using the library's method would ensure consistency if dicompyler-core's interpolation algorithm changes. The custom implementation also uses consistent linear interpolation, so this is a minor maintainability improvement rather than a correctness issue.
- **References**: dicompyler-core DVH.statistic() documentation

### IDEA-033: Log PyRadiomics Version for Reproducibility

- **Stage**: 8. Radiomics Extraction
- **Priority**: Medium
- **Description**: Record the PyRadiomics version used for each extraction in the output files (radiomics_ct.xlsx, radiomics_mr.xlsx). Currently, extraction parameters are logged via `additionalInfo: True`, but the library version is not explicitly captured.
- **Rationale**: Different PyRadiomics versions may produce slightly different feature values due to algorithm refinements or bug fixes. For research reproducibility and multi-center study harmonization, knowing the exact library version is important. This is analogous to IDEA-016 for TotalSegmentator.
- **References**: IBSI reporting guidelines, PyRadiomics changelog

### IDEA-034: Add Explicit IBSI Compliance Log Entry

- **Stage**: 8. Radiomics Extraction
- **Priority**: Low
- **Description**: Add a validation step that checks the loaded PyRadiomics parameters against a list of IBSI-required settings and logs the compliance status at the start of extraction. This would produce a log line like: "IBSI compliance: PASS (all required settings validated)"
- **Rationale**: The current `radiomics_params.yaml` files are IBSI-compliant, but users may provide custom parameter files. Explicit validation would catch misconfigurations early and provide audit evidence that IBSI standards were followed.
- **References**: IBSI standardization guidelines (Zwanenburg et al. 2020)

### IDEA-035: Pre-Extraction Mask Voxel Count Logging

- **Stage**: 8. Radiomics Extraction
- **Priority**: Low
- **Description**: Add verbose logging that reports the voxel count for each mask before attempting extraction, not just when skipping due to min_voxels. This would help debug cases where extraction fails silently.
- **Rationale**: The current code logs at DEBUG level when masks are skipped, but doesn't log successful extractions with their voxel counts. This information is valuable for quality control (e.g., detecting unexpectedly small organ volumes) and debugging extraction failures.
- **References**: N/A - diagnostic improvement

### IDEA-036: Feature Name Standardization Documentation

- **Stage**: 8. Radiomics Extraction
- **Priority**: Low
- **Description**: Add documentation clarifying the feature naming convention across the pipeline. PyRadiomics uses names like `original_shape_VoxelVolume` while some downstream analysis may expect IBSI standard names like `Morphology_Volume`.
- **Rationale**: Multi-center studies and radiomics model sharing require consistent feature naming. Documenting the name mappings (PyRadiomics → IBSI standard names) would facilitate interoperability with external radiomics tools and databases.
- **References**: IBSI feature naming standard, PyRadiomics feature documentation

### IDEA-037: Add Radiomics Extraction Timeout Escalation

- **Stage**: 8. Radiomics Extraction
- **Priority**: Low
- **Description**: The parallel radiomics implementation uses a fixed per-ROI timeout (default 600s in radiomics_parallel.py). Consider adding exponential backoff or progressive timeout escalation for retry attempts, or at minimum log which specific ROIs are timing out.
- **Rationale**: Large ROIs like BODY can legitimately take longer to process, especially with wavelet decomposition enabled. The current fixed timeout may cause false positives. More detailed timeout logging would help identify problematic ROIs for configuration tuning.
- **References**: N/A - workflow reliability improvement

<!-- Template for new ideas:

### IDEA-XXX: [Short Title]

- **Stage**: [Stage Name]
- **Priority**: [Low/Medium/High]
- **Description**: [What could be improved]
- **Rationale**: [Why this improvement matters scientifically or technically]
- **References**: [Relevant papers, standards, or documentation]

-->

---

## Summary by Stage

| Stage | Ideas Count |
|-------|-------------|
| 1. DICOM Organization | 9 |
| 2. CT Processing | 3 |
| 3. Segmentation | 4 |
| 4. Custom Models | 5 |
| 5. Custom Structures | 2 |
| 6. CT Cropping | 4 |
| 7. DVH Analysis | 5 |
| 8. Radiomics Extraction | 5 |
| 9. Robustness Analysis | 0 |
| 10. Quality Control | 0 |
| 11. Aggregation | 0 |
| 12. Configuration | 0 |

---

## Priority Summary

- **High Priority**: 0
- **Medium Priority**: 10 (IDEA-001, IDEA-003, IDEA-011, IDEA-016, IDEA-017, IDEA-022, IDEA-024, IDEA-027, IDEA-029, IDEA-033)
- **Low Priority**: 25 (IDEA-002, IDEA-004, IDEA-005, IDEA-006, IDEA-007, IDEA-009, IDEA-010, IDEA-012, IDEA-013, IDEA-014, IDEA-018, IDEA-019, IDEA-020, IDEA-021, IDEA-023, IDEA-025, IDEA-026, IDEA-028, IDEA-030, IDEA-031, IDEA-032, IDEA-034, IDEA-035, IDEA-036, IDEA-037)
- **Very Low Priority**: 2 (IDEA-008, IDEA-015)

---

*Last updated: 2026-01-06*
