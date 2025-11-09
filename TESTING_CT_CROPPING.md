# Testing CT Cropping Integration

**Date**: 2025-11-09
**Status**: Manual Testing Guide

---

## Overview

This document provides testing procedures for the systematic CT cropping feature.

---

## Prerequisites

1. TotalSegmentator installed and configured
2. Sample patient course with:
   - CT DICOM files
   - Segmentation completed (RS_auto.dcm exists)
   - L1 vertebra and femoral heads segmented

---

## Manual Testing Procedure

### Test 1: Enable Cropping

1. **Edit config.yaml**:
   ```yaml
   ct_cropping:
     enabled: true
     region: "pelvis"
     inferior_margin_cm: 10.0
     use_cropped_for_dvh: true
     use_cropped_for_radiomics: true
     keep_original: true
   ```

2. **Run cropping stage**:
   ```bash
   rtpipeline --stage crop_ct --dicom-root Example_data
   ```

3. **Expected outputs** in each course directory:
   - `cropping_metadata.json` - Contains crop boundaries
   - `RS_auto_cropped.dcm` - Cropped RTSTRUCT file
   - `NIFTI/*_cropped.nii.gz` - Cropped CT and masks

4. **Verify metadata**:
   ```bash
   cat Data_Snakemake/<patient>/<course>/cropping_metadata.json
   ```
   Should show:
   - `region`: "pelvis"
   - `superior_z_mm`: L1 superior boundary
   - `inferior_z_mm`: Femur inferior - 100mm
   - `rs_auto_cropped`: Path to cropped RTSTRUCT

---

### Test 2: DVH with Cropped Structures

1. **Run DVH stage**:
   ```bash
   rtpipeline --stage dvh --dicom-root Example_data
   ```

2. **Check logs** for:
   ```
   INFO: Using systematically cropped RTSTRUCT for DVH (region: pelvis, superior: XXXmm, inferior: YYYmm)
   ```

3. **Verify DVH metrics**:
   - Open `<course>/dvh_metrics.xlsx`
   - Check BODY structure volume is consistent across patients (~12,000 cm³ for pelvis)
   - Percentage metrics (V95%, V20Gy) should now be comparable

---

### Test 3: Radiomics with Cropped Structures

1. **Run radiomics stage**:
   ```bash
   rtpipeline --stage radiomics --dicom-root Example_data
   ```

2. **Check logs** for:
   ```
   INFO: Using systematically cropped RTSTRUCT for radiomics (region: pelvis, superior: XXXmm, inferior: YYYmm)
   ```

3. **Verify radiomics output**:
   - Open `<course>/radiomics_ct.xlsx`
   - Check structure volumes are consistent across patients

---

### Test 4: Insufficient CT Extent

Test what happens when CT doesn't extend far enough for the requested margin.

1. **Find a patient** with short CT FOV (e.g., only 5cm below femurs)

2. **Run cropping** with `inferior_margin_cm: 10.0`

3. **Check logs** for warning:
   ```
   WARNING: Inferior boundary (XXXmm) exceeds CT extent. Clamped to YYYmm (CT boundary). This may affect cross-patient consistency.
   ```

4. **Verify behavior**:
   - Cropping should still work
   - CT is cropped to its actual extent
   - Metadata shows actual (clamped) boundaries

---

### Test 5: Disabled Cropping

1. **Edit config.yaml**:
   ```yaml
   ct_cropping:
     enabled: false
   ```

2. **Run pipeline**:
   ```bash
   rtpipeline --dicom-root Example_data
   ```

3. **Expected behavior**:
   - No cropped files created
   - DVH and radiomics use original RS_auto.dcm
   - Logs show: "CT cropping disabled in config; skipping crop_ct stage"

---

## Validation Checklist

- [ ] Cropping creates all expected output files
- [ ] Cropping metadata is correctly saved
- [ ] RS_auto_cropped.dcm contains all expected ROIs
- [ ] DVH uses cropped RTSTRUCT when enabled
- [ ] Radiomics uses cropped RTSTRUCT when enabled
- [ ] Percentage metrics are consistent across patients
- [ ] Warning logged when CT extent is insufficient
- [ ] Cropping can be disabled via config
- [ ] Original files are preserved when keep_original=true

---

## Expected Results

### Cropping Metadata Example

```json
{
  "region": "pelvis",
  "superior_z_mm": 245.5,
  "inferior_z_mm": -54.8,
  "inferior_margin_cm": 10.0,
  "cropped_files": {
    "ct": "/path/to/ct_cropped.nii.gz",
    "bladder": "/path/to/bladder_cropped.nii.gz",
    ...
  },
  "rs_auto_cropped": "/path/to/RS_auto_cropped.dcm",
  "original_kept": true
}
```

###DVH Consistency (Before vs After)

**Before Cropping**:
```
Patient A - BODY: 18,000 cm³, V20Gy = 2.8%
Patient B - BODY: 15,000 cm³, V20Gy = 3.3%
(Different denominators - NOT comparable!)
```

**After Cropping**:
```
Patient A - BODY: 12,000 cm³, V20Gy = 4.2%
Patient B - BODY: 12,000 cm³, V20Gy = 4.2%
(Same denominators - comparable!)
```

---

## Troubleshooting

### Issue: No cropped files created

**Check**:
1. TotalSegmentator completed successfully?
2. L1 vertebra and femoral heads segmented?
3. Cropping enabled in config?

**Solution**: Run segmentation first, verify landmarks exist

---

### Issue: DVH not using cropped RTSTRUCT

**Check**:
1. RS_auto_cropped.dcm exists?
2. cropping_metadata.json exists?
3. `use_cropped_for_dvh: true` in config?

**Solution**: Verify files exist and config is correct

---

### Issue: Structure volumes still inconsistent

**Check logs for**:
```
WARNING: Inferior boundary exceeds CT extent. Clamped to YYYmm. This may affect cross-patient consistency.
```

**Cause**: CT FOVs vary across patients
**Solution**: Adjust `inferior_margin_cm` or ensure consistent CT acquisition

---

##Performance Considerations

- Cropping adds ~30 seconds per course (landmark extraction + RTSTRUCT creation)
- Parallelizes across courses (uses `effective_workers()`)
- Memory usage: Similar to segmentation stage
- Disk usage: Doubles segmentation output (original + cropped), ~500MB per course

---

## Future Enhancements

### Unit Tests (TODO)

Create automated tests for:
1. Landmark extraction (test with sample NIfTI masks)
2. Image cropping (test coordinate transformations)
3. RTSTRUCT creation (test mask-to-contour conversion)
4. DVH/radiomics integration (test file selection logic)

### GitHub CI (TODO)

Set up GitHub Actions workflow:
1. Install dependencies (rt_utils, SimpleITK, etc.)
2. Run unit tests
3. Test on sample dataset
4. Check for consistency across patients

---

## Clinical Validation

Before using in production:

1. **Visual QC**: Open RS_auto_cropped.dcm in TPS, verify contours
2. **Metric comparison**: Compare DVH metrics before/after cropping
3. **Cohort consistency**: Check that all patients have similar volumes
4. **Boundary validation**: Verify L1 and femur boundaries are clinically appropriate
5. **Publication**: Document cropping methodology in methods section

---

## References

- `SYSTEMATIC_CT_CROPPING.md` - Technical design document
- `IMPROVEMENTS_SUMMARY.md` - Implementation summary
- `rtpipeline/anatomical_cropping.py` - Source code
