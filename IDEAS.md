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
| 6. CT Cropping | 0 |
| 7. DVH Analysis | 0 |
| 8. Radiomics Extraction | 0 |
| 9. Robustness Analysis | 0 |
| 10. Quality Control | 0 |
| 11. Aggregation | 0 |
| 12. Configuration | 0 |

---

## Priority Summary

- **High Priority**: 0
- **Medium Priority**: 6 (IDEA-001, IDEA-003, IDEA-011, IDEA-016, IDEA-017, IDEA-022)
- **Low Priority**: 15 (IDEA-002, IDEA-004, IDEA-005, IDEA-006, IDEA-007, IDEA-009, IDEA-010, IDEA-012, IDEA-013, IDEA-014, IDEA-018, IDEA-019, IDEA-020, IDEA-021, IDEA-023)
- **Very Low Priority**: 2 (IDEA-008, IDEA-015)

---

*Last updated: 2026-01-06*
