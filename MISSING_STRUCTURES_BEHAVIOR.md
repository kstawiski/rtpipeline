# Behavior with Missing or Empty Source Structures

## Overview

The custom structures implementation in RTpipeline includes robust error handling for real-world scenarios where source structures may be missing or empty. This document explains exactly what happens in these situations.

## 🔍 What Happens with Missing Structures

### 1. Individual Structure Missing

**Scenario**: A custom structure depends on a source structure that doesn't exist.

```yaml
- name: "both_kidneys"
  operation: "union"
  source_structures: ["kidney_left", "kidney_right"]
  # But kidney_right is missing from TotalSegmentator output
```

**Behavior**:
- ⚠️ **Warning logged**: `Source structure 'kidney_right' not found for custom structure 'both_kidneys'`
- ❌ **Structure skipped**: `both_kidneys` is not created
- ✅ **Pipeline continues**: Processing continues with other structures
- ✅ **No errors**: Pipeline doesn't crash or stop

### 2. All Source Structures Missing

**Scenario**: All dependencies are missing.

```yaml
- name: "missing_organs"
  operation: "union"
  source_structures: ["spleen", "pancreas"]
  # Both structures missing
```

**Behavior**:
- ⚠️ **Multiple warnings**: One for each missing structure
- ❌ **Structure skipped**: `missing_organs` is not created
- ✅ **Graceful handling**: No crashes or pipeline failures

### 3. Cascade Failures

**Scenario**: Structures that depend on other custom structures fail when dependencies fail.

```yaml
- name: "iliac_vess"
  operation: "union"
  source_structures: ["iliac_artery_left", "iliac_artery_right"]
  # Missing arteries → iliac_vess fails

- name: "iliac_area"
  operation: "union"
  source_structures: ["iliac_vess"]
  margin: 7
  # Depends on iliac_vess → also fails
```

**Behavior**:
- ⚠️ **First failure**: `iliac_vess` fails due to missing arteries
- ⚠️ **Cascade failure**: `iliac_area` fails because `iliac_vess` doesn't exist
- ✅ **Contained**: Independent structures still process successfully
- 📊 **Logged**: Clear warnings showing the dependency chain

## 🕳️ What Happens with Empty Structures

### Empty Source Structures

**Scenario**: Source structure exists but has zero volume (empty mask).

```yaml
- name: "union_with_empty"
  operation: "union"
  source_structures: ["normal_structure", "empty_structure"]
```

**Behavior**:
- ✅ **Graceful handling**: Boolean operations work correctly with empty arrays
- ✅ **Structure created**: Result equals the non-empty structure
- ✅ **No warnings**: Empty structures are valid inputs

**Example Result**:
```
normal_structure: 125 voxels
empty_structure: 0 voxels
union_with_empty: 125 voxels  # = normal_structure
```

## 📊 Real-World Impact

### Typical Pelvic RT Scenario

When TotalSegmentator fails to segment some structures:

```
Available: 8/28 TotalSegmentator structures (29%)
- ✅ liver, kidneys, bladder, sacrum, hips, colon
- ❌ iliac vessels, small bowel, vertebrae, muscles

Result: 2/20 custom structures created (10%)
- ✅ kidneys_combined, kidneys_PRV
- ❌ iliac_vess, bowel_bag, pelvic_bones, etc.
```

**Key Points**:
- 🚫 **No pipeline failure**: RTpipeline continues successfully
- ✅ **Partial success**: Available structures are processed
- 📝 **Clear logging**: Warnings identify exactly what's missing
- 🎯 **Targeted**: Only structures with available dependencies are created

## 🔧 Automatic Handling

### In DVH Analysis
```python
# Custom structures are only added if they exist
for name, mask in custom_masks.items():  # Only successful structures
    try:
        rtstruct.add_roi(mask=mask, name=name)
        logger.info("Added custom structure: %s", name)
    except Exception as e:
        logger.warning("Failed to add custom structure %s: %s", name, e)
```

### In Radiomics Analysis
```python
# Only processes structures that were successfully created
rs_custom = course_dir / "RS_custom.dcm"
if rs_custom and rs_custom.exists():
    masks = _rtstruct_masks(course_dir / 'CT_DICOM', rs_custom)
    for roi, mask in masks.items():  # Only existing custom structures
        tasks.append(("Custom", roi, mask))
```

## 📋 Log Output Examples

### Successful Processing
```
INFO: Created custom structure 'kidneys_combined' using union of ['kidney_left', 'kidney_right']
INFO: Created custom structure 'kidneys_PRV' using union of ['kidneys_combined']
```

### Missing Structure Warnings
```
WARNING: Source structure 'iliac_artery_left' not found for custom structure 'iliac_vess'
WARNING: Source structure 'small_bowel' not found for custom structure 'bowel_bag'
WARNING: Source structure 'iliac_vess' not found for custom structure 'iliac_area'
```

### Summary Information
```
INFO: Using default pelvic custom structures template: custom_structures_pelvic.yaml
INFO: Loaded 20 custom structure configurations
INFO: Successfully created 2 out of 20 custom structures
```

## 🎯 Clinical Implications

### Advantages
1. **Robust Pipeline**: Missing structures don't break analysis
2. **Partial Results**: Get value from available structures
3. **Clear Feedback**: Know exactly what's missing
4. **Consistent Behavior**: Same processing regardless of segmentation success

### Strategies for Missing Structures

1. **Review Logs**: Check warnings to identify missing dependencies
2. **Modify Template**: Remove or modify structures with frequently missing dependencies
3. **Improve Segmentation**: Focus TotalSegmentator troubleshooting on critical missing structures
4. **Manual Contours**: Add manual contours for critical missing structures

### Example: Optimized Pelvic Template

For sites with frequent iliac vessel segmentation failures:

```yaml
# Remove problematic structures
# - name: "iliac_vess"  # COMMENTED OUT
# - name: "iliac_area"   # COMMENTED OUT

# Keep robust structures
- name: "kidneys_combined"
  source_structures: ["kidney_left", "kidney_right"]
- name: "pelvic_bones_basic"
  source_structures: ["sacrum", "hip_left", "hip_right"]
  # Removed vertebrae_S1 dependency
```

## 🧪 Testing Missing Structures

Run the test script to see behavior with missing structures:

```bash
python test_missing_structures.py
```

This demonstrates:
- ✅ Graceful handling of missing dependencies
- ✅ Proper warning messages
- ✅ Continued processing of available structures
- ✅ Cascade failure containment

## 🔧 Best Practices

### Template Design
1. **Order Dependencies**: Put independent structures first
2. **Group by Reliability**: Separate robust from fragile structures
3. **Document Requirements**: Note which TotalSegmentator structures are needed

### Error Handling
1. **Monitor Logs**: Regular review of custom structure warnings
2. **Success Metrics**: Track custom structure creation rates
3. **Template Optimization**: Modify based on real-world success rates

### Clinical Workflow
1. **Fallback Plans**: Manual contours for critical missing structures
2. **Quality Checks**: Verify important custom structures were created
3. **Documentation**: Note which custom structures are available in reports

## Summary

The custom structures implementation is designed to be **robust and forgiving**:
- ✅ Missing structures are **gracefully skipped** with clear warnings
- ✅ Empty structures are **handled correctly** in boolean operations
- ✅ Pipeline **never crashes** due to missing dependencies
- ✅ Available structures are **processed successfully**
- ✅ Clear **logging and feedback** helps identify issues

This ensures that RTpipeline continues to work reliably even when TotalSegmentator segmentation is incomplete, while providing maximum value from the structures that are available.