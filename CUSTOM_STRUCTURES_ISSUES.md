# Custom Structures Generation Issues Analysis

## Overview
This document analyzes issues with custom structure generation and their usage in DVH and radiomics computations in the rtpipeline.

## Issues Identified

### 1. **TotalSegmentator Mask Lookup Fragility**
**Location**: `rtpipeline/dvh.py:279-335` (`_totalseg_mask` function)

**Problem**:
- The function searches for TotalSegmentator masks using a complex directory traversal through `manifest.json` files
- If the manifest structure is missing, malformed, or structured differently, mask lookup fails silently
- The function returns `None` without clear error messages when masks aren't found

**Impact**:
- Custom structures that depend on TotalSegmentator outputs may fail to build
- Users don't get clear feedback about why structures aren't being created
- Partial structures created without users realizing some components are missing

**Code Example**:
```python
# Current: Silent failure
mask_path = _totalseg_mask(roi_name)  # Returns None if not found
if mask_path is None:
    # Structure skipped without warning
```

### 2. **Structure Name Normalization Mismatch**
**Location**: `rtpipeline/custom_structures.py:183-184`, `rtpipeline/dvh.py:352`

**Problem**:
- Name normalization removes all non-alphanumeric characters: `"iliac_artery_left"` → `"iliacarteryleft"`
- TotalSegmentator uses underscores and specific naming conventions
- Manual structures may have different capitalization or spacing
- Matching logic tries normalized names but may still miss valid structures

**Impact**:
- Custom structures fail to find their source structures even when they exist
- Inconsistent behavior depending on how structures are named in different sources
- `custom_structures_pelvic.yaml` expects specific names like "iliac_artery_left" but these might not match actual TotalSegmentator output

**Example from config**:
```yaml
# Expects these exact names
source_structures:
  - "iliac_artery_left"   # Might be named differently in TotalSegmentator
  - "iliac_vena_left"     # Case sensitivity issues possible
```

### 3. **Dependency Ordering Issues**
**Location**: `rtpipeline/custom_structures.py:274-280`

**Problem**:
- Custom structures can depend on other custom structures (e.g., `iliac_area` depends on `iliac_vess`)
- Processing order follows YAML definition order
- If structure B depends on structure A but A is defined later, B will fail or be partial
- Code adds created structures to `available_masks` iteratively, which helps but doesn't validate dependencies upfront

**Impact**:
- Users must manually order their custom structures definitions
- Circular dependencies are not detected
- Cryptic "missing sources" warnings without explaining the dependency issue

**Example**:
```yaml
# This would fail:
- name: "iliac_area"
  source_structures: ["iliac_vess"]  # Defined later - FAIL!

- name: "iliac_vess"
  source_structures: ["iliac_artery_left"]  # OK
```

### 4. **RS_custom.dcm Stale Cache Problem**
**Location**:
- `rtpipeline/dvh.py:622-624`
- `rtpipeline/radiomics.py:421-429`
- `rtpipeline/radiomics_parallel.py:331-339`

**Problem**:
- Code checks if `RS_custom.dcm` exists before recreating it
- No timestamp checking or hash validation of source files
- If custom structures config changes, old RS_custom.dcm is still used
- If source structures (RS.dcm, RS_auto.dcm) are updated, custom structures aren't regenerated

**Impact**:
- Stale custom structures used for DVH and radiomics
- Users must manually delete RS_custom.dcm to force regeneration
- Difficult to debug why custom structures don't match expected results

**Current Code**:
```python
rs_custom = course_dir / "RS_custom.dcm"
if rs_custom.exists():
    process_struct(rs_custom, "Merged", rx_est)  # Uses old file!
else:
    # Only recreates if missing
    rs_custom_legacy = _create_custom_structures_rtstruct(...)
```

### 5. **Partial Structures Silent Creation**
**Location**: `rtpipeline/custom_structures.py:218-236`, `rtpipeline/dvh.py:403-427`

**Problem**:
- When some source structures are missing, custom structures are still created from available ones
- Only logs a debug message for missing sources
- Marks structure as "__partial" but continues processing
- No option to fail-fast when critical structures are missing

**Impact**:
- Union operations create smaller-than-expected structures
- Subtract operations may not subtract what user intended
- Clinical decisions might be made on incomplete data
- `__partial` suffix may be stripped or not noticed in downstream analysis

**Example**:
```python
# Requests 4 vessels, only 2 found
source_structures: ["iliac_artery_left", "iliac_artery_right",
                    "iliac_vena_left", "iliac_vena_right"]
# Creates "iliac_vess__partial" from only 2 vessels
# User might not notice it's incomplete
```

### 6. **Inconsistent Source Labeling**
**Location**: `rtpipeline/dvh.py:623-640`

**Problem**:
- When RS_custom.dcm exists, all structures are labeled "Merged" regardless of origin
- Loses provenance information about which structures came from which source
- Makes it impossible to distinguish manual vs auto vs custom structures in results
- DVH results don't show which structures are from custom boolean operations

**Impact**:
- Cannot filter or analyze by structure source
- Quality control becomes difficult
- Cannot validate auto-segmentation against manual for custom structures

### 7. **Missing Validation for Boolean Operations**
**Location**: `rtpipeline/custom_structures.py:119-137`

**Problem**:
- Subtract operation requires at least 2 masks but this is only checked at runtime
- No validation of mask dimensions or spacing compatibility
- Intersection of non-overlapping structures creates empty masks (no warning)
- XOR on multiple masks may produce unexpected results

**Impact**:
- Runtime errors instead of config validation errors
- Empty structures created silently
- User doesn't know if operations produced expected results

### 8. **Margin Application Limitations**
**Location**: `rtpipeline/custom_structures.py:139-180`

**Problem**:
- Asymmetric margins are defined but not fully implemented (line 177-178)
- Uses uniform expansion based on maximum margin value
- Comment says "more complex implementation" needed
- Users might define asymmetric margins thinking they work

**Impact**:
- Asymmetric margins don't produce expected results
- PTV margins might be incorrect for clinical use
- No warning that asymmetric margins are approximated

**Code Comment**:
```python
# Handle asymmetric margins if needed (more complex implementation)
# For now, using uniform expansion based on maximum margin
```

### 9. **No Structure Size Validation**
**Location**: `rtpipeline/dvh.py:402-436`

**Problem**:
- Custom structures can be empty or have very few voxels
- No minimum/maximum size checks before adding to RTSTRUCT
- Radiomics and DVH might fail on tiny structures
- No warnings about unrealistic structure sizes

**Impact**:
- Empty structures waste computation time
- Radiomics extraction fails downstream
- DVH computation crashes or produces meaningless results

### 10. **TotalSegmentator Fallback Side Effects**
**Location**: `rtpipeline/dvh.py:345-349`

**Problem**:
- If rt-utils fails to get a mask, code falls back to TotalSegmentator NIfTI files
- This fallback happens silently during structure harvesting
- Mixed coordinate systems possible if resampling fails
- Users don't know when fallback was used

**Impact**:
- Spatial misalignment between DICOM and NIfTI structures
- Custom structures might combine incompatible masks
- Difficult to debug spatial inconsistencies

## Recommendations for Fixes

### High Priority
1. **Add RS_custom.dcm staleness detection** - Check timestamps and config hashes
2. **Improve structure name matching** - Better fuzzy matching and clear error messages
3. **Add dependency validation** - Detect circular dependencies and order requirements
4. **Validate structure sizes** - Warn about empty or suspiciously small structures

### Medium Priority
5. **Enhance logging** - Clear messages when structures not found or fallback used
6. **Add strict mode** - Option to fail instead of creating partial structures
7. **Preserve provenance** - Track source of each structure even in RS_custom.dcm
8. **Implement asymmetric margins** - Or clearly document limitation

### Low Priority
9. **Add config validation** - Check YAML syntax and structure definitions upfront
10. **Add visual QA** - Generate images showing custom structures for manual review

## Testing Recommendations

1. Test with missing source structures
2. Test with renamed TotalSegmentator outputs
3. Test dependency ordering edge cases
4. Test with updated source files and cached RS_custom.dcm
5. Test margin operations with various configurations
6. Test with empty or single-voxel structures
