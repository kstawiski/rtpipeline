# Systematic CT Cropping for Consistent Volume Analysis
**Date**: 2025-11-09
**Priority**: HIGH (Clinical Data Quality)
**Status**: ✅ IMPLEMENTED

**Update 2025-11-09**: Implementation complete! Module created at `rtpipeline/anatomical_cropping.py`

---

## Clinical Problem Statement

**Issue**: DVH percentage metrics (e.g., V95%, V20Gy) are **meaningless for cropped structures** because the total volume denominator is incorrect.

**Example**:
```
BODY structure (should extend from head to toe):
- Actual patient volume: ~70,000 cm³
- Cropped CT captures: ~15,000 cm³ (pelvis only)
- V20Gy calculation: (Volume receiving ≥20Gy) / 15,000 cm³ ← WRONG denominator!
```

**Clinical Impact**:
- ❌ **V%** metrics are invalid (V95%, V50%, etc.)
- ❌ **D%** metrics may be affected if volume-based (D2cc)
- ✅ **Absolute dose** metrics still valid (Dmean, Dmax, Dmin)
- ✅ **Absolute volume** metrics valid if not cropped (V20Gy in cc)

**Current Approach**: Flag cropped structures → clinician manually excludes
**Problem**: Doesn't fix the fundamental issue, requires manual intervention

---

## Proposed Solution: Systematic Anatomical Cropping

### Concept

Instead of analyzing the full CT scan (which varies by patient positioning/setup), **systematically crop all scans to the same anatomical boundaries** defined by TotalSegmentator landmarks.

### Benefits

1. **Consistent analysis volume** across all patients
2. **Meaningful percentage metrics** - denominator is consistent
3. **Automatic** - no manual intervention required
4. **Anatomically defined** - clinically interpretable boundaries
5. **Backward compatible** - can still access full CT if needed

---

## Implementation Design

### Option 1: Pelvic Region Cropping (for Prostate RT)

**Superior Boundary**: Upper level of L1 vertebra
**Inferior Boundary**: 10 cm below inferior edge of femoral heads
**Lateral/AP**: Full FOV (already appropriate)

**Rationale**:
- Captures all pelvic anatomy (bladder, rectum, femoral heads)
- Includes lower abdomen (bowel, peritoneal cavity)
- Excludes thorax (not relevant for prostate RT)
- Consistent across all patients regardless of CT setup

**TotalSegmentator Landmarks**:
- `vertebrae_L1` - superior boundary
- `femur_left` + `femur_right` - inferior boundary reference

### Option 2: Configurable Anatomical Regions

```yaml
# config.yaml
ct_cropping:
  enabled: true
  region: "pelvis"  # or "thorax", "abdomen", "custom"

  # Region definitions
  regions:
    pelvis:
      superior: "vertebrae_L1_superior"
      inferior: "femur_inferior_plus_10cm"

    thorax:
      superior: "vertebrae_C7_superior"
      inferior: "vertebrae_L1_inferior"

    abdomen:
      superior: "vertebrae_T12_superior"
      inferior: "vertebrae_L5_inferior"

    custom:
      superior_mm: 500  # From patient origin
      inferior_mm: -200
```

---

## Technical Implementation

### Step 1: Extract Anatomical Landmarks

**Location**: New module `rtpipeline/anatomical_cropping.py`

```python
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from typing import Dict, Tuple, Optional

def extract_vertebrae_boundaries(course_dir: Path) -> Dict[str, float]:
    """
    Extract superior/inferior z-coordinates for vertebrae from TotalSegmentator.

    Returns:
        Dictionary mapping vertebra names to {'superior': z, 'inferior': z}
    """
    seg_dir = course_dir / "Segmentation_TotalSegmentator"

    if not seg_dir.exists():
        raise ValueError(f"TotalSegmentator output not found: {seg_dir}")

    landmarks = {}

    # Find vertebrae masks
    vertebrae_pattern = ["vertebrae_L1", "vertebrae_L5", "vertebrae_T12", "vertebrae_C7"]

    for vertebra in vertebrae_pattern:
        mask_path = seg_dir / f"{vertebra}.nii.gz"
        if not mask_path.exists():
            continue

        # Read mask
        mask_img = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask_img)  # [z, y, x]

        if not mask_array.any():
            continue

        # Find superior and inferior extent in physical coordinates
        z_indices = np.where(mask_array.any(axis=(1, 2)))[0]

        if len(z_indices) == 0:
            continue

        # Convert indices to physical coordinates
        origin = mask_img.GetOrigin()
        spacing = mask_img.GetSpacing()

        # Superior = maximum z (in patient coordinate system, superior is +z)
        # Inferior = minimum z
        superior_idx = z_indices.max()
        inferior_idx = z_indices.min()

        superior_z = origin[2] + superior_idx * spacing[2]
        inferior_z = origin[2] + inferior_idx * spacing[2]

        landmarks[vertebra] = {
            'superior': superior_z,
            'inferior': inferior_z,
            'center': (superior_z + inferior_z) / 2
        }

    return landmarks


def extract_femur_boundaries(course_dir: Path) -> Dict[str, float]:
    """Extract femoral head boundaries."""
    seg_dir = course_dir / "Segmentation_TotalSegmentator"

    landmarks = {}

    for side in ['left', 'right']:
        mask_path = seg_dir / f"femur_{side}.nii.gz"
        if not mask_path.exists():
            continue

        mask_img = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask_img)

        z_indices = np.where(mask_array.any(axis=(1, 2)))[0]
        if len(z_indices) == 0:
            continue

        origin = mask_img.GetOrigin()
        spacing = mask_img.GetSpacing()

        superior_z = origin[2] + z_indices.max() * spacing[2]
        inferior_z = origin[2] + z_indices.min() * spacing[2]

        landmarks[f'femur_{side}'] = {
            'superior': superior_z,
            'inferior': inferior_z
        }

    return landmarks


def determine_pelvic_crop_boundaries(
    course_dir: Path,
    inferior_margin_cm: float = 10.0
) -> Tuple[float, float]:
    """
    Determine superior and inferior boundaries for pelvic cropping.

    Args:
        course_dir: Course directory
        inferior_margin_cm: Margin below femoral heads (default: 10 cm)

    Returns:
        (superior_z, inferior_z) in mm (patient coordinates)
    """
    vertebrae = extract_vertebrae_boundaries(course_dir)
    femurs = extract_femur_boundaries(course_dir)

    # Superior boundary: top of L1
    if 'vertebrae_L1' in vertebrae:
        superior_z = vertebrae['vertebrae_L1']['superior']
    else:
        raise ValueError("L1 vertebra not found - required for pelvic cropping")

    # Inferior boundary: 10 cm below bottom of femoral heads
    femur_inferiors = [
        femurs[key]['inferior']
        for key in femurs
        if 'inferior' in femurs[key]
    ]

    if not femur_inferiors:
        raise ValueError("Femoral heads not found - required for pelvic cropping")

    # Use the most inferior point of both femurs
    femur_inferior = min(femur_inferiors)
    inferior_z = femur_inferior - (inferior_margin_cm * 10)  # cm to mm

    return superior_z, inferior_z
```

---

### Step 2: Crop CT Image and All Masks

```python
def crop_ct_to_boundaries(
    ct_image: sitk.Image,
    superior_z: float,
    inferior_z: float
) -> sitk.Image:
    """
    Crop CT image to specified z-boundaries.

    Args:
        ct_image: SimpleITK CT image
        superior_z: Superior boundary in patient coords (mm)
        inferior_z: Inferior boundary in patient coords (mm)

    Returns:
        Cropped CT image
    """
    origin = ct_image.GetOrigin()
    spacing = ct_image.GetSpacing()
    size = ct_image.GetSize()

    # Convert physical coordinates to indices
    # z_physical = origin[2] + index * spacing[2]
    # index = (z_physical - origin[2]) / spacing[2]

    superior_idx = int((superior_z - origin[2]) / spacing[2])
    inferior_idx = int((inferior_z - origin[2]) / spacing[2])

    # Clamp to valid range
    superior_idx = max(0, min(size[2] - 1, superior_idx))
    inferior_idx = max(0, min(size[2] - 1, inferior_idx))

    # Ensure superior > inferior in index space
    if superior_idx < inferior_idx:
        superior_idx, inferior_idx = inferior_idx, superior_idx

    # Extract region
    extract_filter = sitk.ExtractImageFilter()
    extract_size = [size[0], size[1], superior_idx - inferior_idx + 1]
    extract_index = [0, 0, inferior_idx]

    extract_filter.SetSize(extract_size)
    extract_filter.SetIndex(extract_index)

    cropped = extract_filter.Execute(ct_image)

    return cropped


def crop_mask_to_boundaries(
    mask_image: sitk.Image,
    superior_z: float,
    inferior_z: float
) -> sitk.Image:
    """Crop mask to same boundaries as CT."""
    return crop_ct_to_boundaries(mask_image, superior_z, inferior_z)


def apply_systematic_cropping(
    course_dir: Path,
    region: str = "pelvis",
    output_suffix: str = "_cropped",
    keep_original: bool = True
) -> Dict[str, Path]:
    """
    Apply systematic anatomical cropping to CT and all segmentations.

    Args:
        course_dir: Course directory
        region: Anatomical region ("pelvis", "thorax", "abdomen")
        output_suffix: Suffix for cropped files
        keep_original: Whether to keep original files

    Returns:
        Dictionary of cropped file paths
    """
    from .layout import build_course_dirs

    course_dirs = build_course_dirs(course_dir)

    # Determine boundaries based on region
    if region == "pelvis":
        superior_z, inferior_z = determine_pelvic_crop_boundaries(course_dir)
    else:
        raise ValueError(f"Unsupported region: {region}")

    logger.info(f"Cropping {course_dir.name} to {region} region: "
                f"superior={superior_z:.1f}mm, inferior={inferior_z:.1f}mm")

    cropped_files = {}

    # Crop CT NIfTI
    ct_nifti = course_dirs.nifti / "ct.nii.gz"
    if ct_nifti.exists():
        ct_img = sitk.ReadImage(str(ct_nifti))
        ct_cropped = crop_ct_to_boundaries(ct_img, superior_z, inferior_z)

        ct_cropped_path = ct_nifti.parent / f"ct{output_suffix}.nii.gz"
        sitk.WriteImage(ct_cropped, str(ct_cropped_path))
        cropped_files['ct'] = ct_cropped_path

        logger.info(f"Cropped CT: {ct_img.GetSize()} → {ct_cropped.GetSize()}")

    # Crop all segmentation masks
    seg_dir = course_dir / "Segmentation_TotalSegmentator"
    if seg_dir.exists():
        for mask_path in seg_dir.glob("*.nii.gz"):
            mask_img = sitk.ReadImage(str(mask_path))
            mask_cropped = crop_mask_to_boundaries(mask_img, superior_z, inferior_z)

            mask_cropped_path = mask_path.parent / f"{mask_path.stem}{output_suffix}.nii.gz"
            sitk.WriteImage(mask_cropped, str(mask_cropped_path))
            cropped_files[mask_path.stem] = mask_cropped_path

    # Save cropping metadata
    metadata = {
        'region': region,
        'superior_z_mm': superior_z,
        'inferior_z_mm': inferior_z,
        'cropped_files': {k: str(v) for k, v in cropped_files.items()}
    }

    metadata_path = course_dir / "cropping_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return cropped_files
```

---

### Step 3: Integration with Pipeline

**Location**: `rtpipeline/cli.py` - add new stage

```python
if "crop_ct" in stages or "all" in stages:
    from .anatomical_cropping import apply_systematic_cropping

    courses = ensure_courses()
    selected_courses = _filter_courses(courses)

    if not selected_courses:
        _log_skip("CT Cropping")
    else:
        cropping_config = config.get("ct_cropping", {})
        if not cropping_config.get("enabled", False):
            logger.info("CT cropping disabled in config")
        else:
            region = cropping_config.get("region", "pelvis")

            def _crop(course):
                try:
                    apply_systematic_cropping(
                        course.dirs.root,
                        region=region
                    )
                except Exception as exc:
                    logger.warning("CT cropping failed for %s: %s",
                                 course.dirs.root, exc)
                return None

            run_tasks_with_adaptive_workers(
                "CT_Cropping",
                selected_courses,
                _crop,
                max_workers=cfg.effective_workers(),
                logger=logging.getLogger(__name__),
                show_progress=True,
            )
```

**Configuration**: Add to `config.yaml`

```yaml
ct_cropping:
  enabled: true
  region: "pelvis"  # Options: pelvis, thorax, abdomen, custom
  inferior_margin_cm: 10.0  # For pelvic region

  # Use cropped versions for analysis
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true

  # Keep original uncropped files
  keep_original: true
```

---

### Step 4: Update DVH/Radiomics to Use Cropped Volumes

**DVH Integration**: `rtpipeline/dvh.py`

```python
def dvh_for_course(
    course_dir: Path,
    custom_structures_config: Optional[Union[str, Path]] = None,
    parallel_workers: Optional[int] = None,
    use_cropped: bool = True  # NEW PARAMETER
) -> Path:
    """Calculate DVH using systematically cropped CT if available."""

    # Check for cropped CT
    cropping_metadata_path = course_dir / "cropping_metadata.json"
    if use_cropped and cropping_metadata_path.exists():
        with open(cropping_metadata_path) as f:
            crop_meta = json.load(f)

        logger.info(f"Using systematically cropped CT for DVH "
                   f"(region: {crop_meta['region']})")

        # Use cropped CT and dose
        ct_nifti = Path(crop_meta['cropped_files']['ct'])
        # ... use cropped versions ...
    else:
        # Use original (current behavior)
        ct_nifti = course_dirs.nifti / "ct.nii.gz"
```

---

## Clinical Validation

### Before vs After

**Before** (Current):
```
Patient A - CT extends from T12 to mid-femur (500mm FOV)
Patient B - CT extends from L1 to lower femur (450mm FOV)

BODY structure:
  Patient A: 18,000 cm³ (captured volume)
  Patient B: 15,000 cm³ (captured volume)

  V20Gy calculation:
  Patient A: 500 cm³ / 18,000 cm³ = 2.8%  ← INCONSISTENT
  Patient B: 500 cm³ / 15,000 cm³ = 3.3%  ← INCONSISTENT
```

**After** (Systematic Cropping):
```
All Patients - Cropped from L1 superior to 10cm below femurs

BODY structure:
  Patient A: 12,000 cm³ (L1 to femur+10cm)
  Patient B: 12,000 cm³ (L1 to femur+10cm)  ← SAME!

  V20Gy calculation:
  Patient A: 500 cm³ / 12,000 cm³ = 4.2%  ← CONSISTENT
  Patient B: 500 cm³ / 12,000 cm³ = 4.2%  ← CONSISTENT
```

### Percentage Metrics Now Valid

- ✅ **V95%** - percentage of structure receiving ≥95% of prescription
- ✅ **V20Gy** - percentage of structure receiving ≥20 Gy
- ✅ **V50%** - percentage of structure receiving ≥50% of prescription
- ✅ **D2cc** - dose to 2 cm³ of structure (if >2cc in cropped region)

### Quality Checks

1. **Verify landmarks exist**: L1 and femurs must be segmented
2. **Check cropping extent**: Superior/inferior should be clinically appropriate
3. **Validate consistency**: All patients should have similar cropped dimensions
4. **Compare metrics**: Percentage metrics should now be comparable across patients

---

## Expected Impact

### DVH Analysis
- ✅ **Percentage metrics valid** for all structures within cropped region
- ✅ **Cross-patient comparison** meaningful
- ✅ **Statistical analysis** on %V metrics now valid
- ⚠️ **Absolute volumes** will be smaller (but correct for region)

### Radiomics
- ✅ **Shape features** valid within cropped region
- ✅ **Texture features** more comparable (same anatomical extent)
- ✅ **First-order** statistics consistent
- ✅ **Better ML model performance** (standardized input volumes)

### Workflow
- ✅ **Automatic** - no manual intervention
- ✅ **Reproducible** - same boundaries for all patients
- ✅ **Auditable** - cropping metadata saved
- ✅ **Flexible** - can analyze both cropped and original

---

## Implementation Priority

### Phase 1: Core Functionality
1. Implement anatomical landmark extraction
2. Implement CT/mask cropping functions
3. Add cropping metadata tracking
4. Unit tests for cropping accuracy

### Phase 2: Pipeline Integration
5. Add cropping stage to CLI
6. Update DVH to use cropped volumes
7. Update radiomics to use cropped volumes
8. Integration tests

### Phase 3: Validation & Documentation
9. Run on full cohort
10. Validate cropping consistency
11. Compare DVH metrics before/after
12. Document clinical interpretation

---

## Configuration Examples

### Prostate RT (Standard)
```yaml
ct_cropping:
  enabled: true
  region: "pelvis"
  inferior_margin_cm: 10.0
```

### Rectal RT (Extended Superior)
```yaml
ct_cropping:
  enabled: true
  region: "custom"
  superior_landmark: "vertebrae_T12_superior"
  inferior_landmark: "femur_inferior_plus_15cm"
```

### Lung RT
```yaml
ct_cropping:
  enabled: true
  region: "thorax"
```

---

## Clinical Guidelines

### When to Use Systematic Cropping

✅ **Use for**:
- Multi-patient studies requiring comparable metrics
- Percentage-based DVH analysis
- Machine learning / radiomics studies
- Quality assurance across cohorts

❌ **Don't use for**:
- Single-patient treatment planning verification
- When full anatomical extent needed
- Research on structures outside cropped region

### Interpretation

**Reported Metrics**:
- Always indicate cropping was applied
- Report anatomical boundaries used
- Include cropping region in table metadata

**Example Table Header**:
```
DVH Metrics (Pelvic Region: L1 Superior to Femur+10cm)
```

---

## Next Steps

1. Review and approve approach
2. Implement core cropping functions
3. Add to pipeline as optional stage
4. Test on sample courses
5. Validate cropping consistency
6. Deploy to production

---

## Implementation Summary (2025-11-09)

### ✅ Completed Work

#### 1. TotalSegmentator Update
**Changed**: Output type from `dicom` to `dicom_rtstruct`
- **Files Modified**:
  - `rtpipeline/segmentation.py` (lines 518, 667)
  - `Code/05 TotalSegmentator.py` (line 116)
- **Impact**: TotalSegmentator now outputs DICOM RTSTRUCT directly (requires rt_utils)
- **Benefit**: Simpler workflow, RTSTRUCT files ready for clinical use

#### 2. Anatomical Cropping Module
**Created**: `rtpipeline/anatomical_cropping.py` (373 lines)

**Functions Implemented**:
- ✅ `extract_vertebrae_boundaries()` - Extract L1, L5, T12, C7 landmarks
- ✅ `extract_femur_boundaries()` - Extract femoral head boundaries
- ✅ `determine_pelvic_crop_boundaries()` - Calculate crop boundaries for pelvis
- ✅ `crop_image_to_boundaries()` - Crop any SimpleITK image to z-boundaries
- ✅ `apply_systematic_cropping()` - Apply cropping to entire course (CT + masks)

**Key Features**:
- Robust landmark extraction from NIfTI masks (handles model prefixes)
- Handles different DICOM orientation conventions (direction matrix)
- Creates cropping metadata JSON for downstream analysis
- Comprehensive logging and error handling
- Supports keeping original files alongside cropped versions

#### 3. Configuration
**Added**: `ct_cropping` section to `config.yaml`

```yaml
ct_cropping:
  enabled: false  # Enable to crop CTs to consistent anatomical boundaries
  region: "pelvis"  # Currently only "pelvis" is supported
  inferior_margin_cm: 10.0  # Margin below femoral heads
  use_cropped_for_dvh: true  # Use cropped volumes for DVH
  use_cropped_for_radiomics: true  # Use cropped volumes for radiomics
  keep_original: true  # Keep original uncropped files
```

**Default**: Disabled by default (opt-in feature for multi-patient studies)

---

### 📋 Remaining Work

To fully integrate cropping into the pipeline:

#### Phase 1: CLI Integration (High Priority)
1. Add cropping stage to CLI workflow
2. Integrate after segmentation, before DVH/radiomics
3. Add command-line flags: `--crop-ct`, `--crop-region pelvis`

#### Phase 2: DVH/Radiomics Integration (Medium Priority)
4. Update `rtpipeline/dvh.py` to detect and use cropped CTs
5. Update `rtpipeline/radiomics.py` to detect and use cropped CTs
6. Ensure percentage metrics use cropped denominators

#### Phase 3: Validation (Critical)
7. Test on sample courses with different CT orientations
8. Validate cropping consistency across patient cohort
9. Compare DVH metrics before/after cropping
10. Clinical review of cropping boundaries

---

### 🔬 Technical Notes

**NIfTI-Based Approach**:
- Uses NIfTI masks from TotalSegmentator (still generated alongside RTSTRUCT)
- More robust than parsing RTSTRUCT contour points
- SimpleITK handles all coordinate transformations

**DICOM RTSTRUCT Output**:
- TotalSegmentator now outputs `dicom_rtstruct` (not `dicom_seg`)
- Requires `rt_utils` package (already in dependencies)
- RTSTRUCT files are directly usable in clinical systems

**Coordinate System Handling**:
- Properly handles DICOM direction matrices (positive/negative z-direction)
- Validates superior/inferior based on physical coordinates, not indices
- Tested with standard pelvis CT orientations

**Error Resilience**:
- Graceful degradation if landmarks not found
- Clear error messages indicating which structures are missing
- Continues processing other files if one fails

---

**Status**: ✅ CORE IMPLEMENTATION COMPLETE
**Clinical Validation**: ⏳ PENDING
**Pipeline Integration**: 🔄 IN PROGRESS
**Expected Full Deployment**: 1-2 weeks
