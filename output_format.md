# RTPipeline Output Format Guide for AI Agents

This document provides a comprehensive guide for AI agents to understand and work with the output of the RTpipeline radiotherapy data processing system.

## Quick Reference

**Start here:** All aggregated results are in `{output_dir}/_RESULTS/`

**Key files:**
- `dvh_metrics.xlsx` - Dose-volume histogram metrics for all patients
- `radiomics_ct.xlsx` - CT radiomic features for all patients
- `radiomics_mr.xlsx` - MR radiomic features for all patients (if MR data present)
- `case_metadata.xlsx` - Patient and treatment metadata
- `qc_reports.xlsx` - Quality control summary

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Aggregated Results (_RESULTS Directory)](#aggregated-results-_results-directory)
3. [Per-Patient Output](#per-patient-output)
4. [File Formats and Column Descriptions](#file-formats-and-column-descriptions)
5. [Understanding DVH Metrics](#understanding-dvh-metrics)
6. [Understanding Radiomics Features](#understanding-radiomics-features)
7. [Quality Control Interpretation](#quality-control-interpretation)
8. [Common Use Cases](#common-use-cases)
9. [Troubleshooting](#troubleshooting)

---

## Directory Structure

```
{output_dir}/                           # Main output directory (e.g., Data_Snakemake)
│
├── _RESULTS/                           # ★ START HERE - All aggregated data
│   ├── dvh_metrics.xlsx                # All DVH metrics (dose-volume histograms)
│   ├── radiomics_ct.xlsx               # All CT radiomic features
│   ├── radiomics_mr.xlsx               # All MR radiomic features (if applicable)
│   ├── case_metadata.xlsx              # Patient/treatment metadata
│   ├── qc_reports.xlsx                 # Quality control summary
│   └── qc_detailed_*.json              # Detailed QC logs per course
│
└── {PatientID}/                        # Per-patient directory
    └── {CourseID}/                     # Per-treatment course
        ├── DICOM/                      # Original DICOM files
        │   ├── CT/                     # Planning CT slices
        │   ├── MR/                     # MR slices (if present)
        │   ├── RTPLAN.dcm              # Treatment plan
        │   ├── RTDOSE.dcm              # Dose distribution
        │   └── RTSTRUCT.dcm            # Original contours
        │
        ├── NIFTI/                      # CT converted to NIfTI
        │   ├── ct.nii.gz               # CT volume (compressed)
        │   ├── ct_metadata.json        # DICOM metadata
        │   └── ct_spacing.txt          # Voxel spacing (mm)
        │
        ├── Segmentation_TotalSegmentator/  # Auto-segmented organs
        │   ├── lung_left--ct.nii.gz
        │   ├── lung_right--ct.nii.gz
        │   ├── heart--ct.nii.gz
        │   ├── [100+ more organs...]
        │   └── total_mr--*.nii.gz      # MR segmentations (if MR present)
        │
        ├── Segmentation_Original/      # Manual contours converted to NIfTI
        │   └── {structure_name}.nii.gz
        │
        ├── Segmentation_CustomModels/  # Custom model predictions
        │   └── {ModelName}/
        │       ├── prediction.nii.gz
        │       └── rtstruct.dcm
        │
├── RS.dcm                      # Original RTSTRUCT (copy)
├── RS_auto.dcm                 # TotalSegmentator RTSTRUCT (importable)
├── RS_auto_cropped.dcm         # Cropped auto RTSTRUCT (if CT cropping enabled)
├── RS_custom.dcm               # Custom/boolean RTSTRUCT (if applicable)
        │
        ├── radiomics_ct.xlsx           # Per-course CT radiomics
        ├── dvh_metrics.xlsx            # Per-course DVH metrics
        │
        ├── metadata/
        │   ├── case_metadata.xlsx
        │   └── case_metadata.json
        │
        ├── qc_reports/
        │   ├── qc_{course}.json        # Detailed QC checks
        │   └── cropping_metadata.json  # CT cropping details
        │
└── MR/
            └── {SeriesInstanceUID}/    # Per-MR series
                ├── DICOM/              # MR DICOM files
                ├── NIFTI/
                │   ├── mr.nii.gz
                │   └── mr_metadata.json
                ├── Segmentation_TotalSegmentator/
                └── radiomics_mr.xlsx
```

### RTSTRUCT variants

| File | Origin | Notes |
| --- | --- | --- |
| `RS.dcm` | Direct copy of the planning RTSTRUCT detected during **organize**. | Immutable record of the clinical contours; never modified by the pipeline. |
| `RS_auto.dcm` | Generated from TotalSegmentator CT masks via `auto_rtstruct`. | Aligned to the un-cropped CT grid; used whenever the pipeline needs automatic structures. |
| `RS_auto_cropped.dcm` | Created by `anatomical_cropping.py` when `ct_cropping.enabled: true`. | Same labels as `RS_auto.dcm`, but intersected with the systematic cropping box (`cropping_metadata.json` contains clamp metadata). Radiomics/DVH/QC switch to this file when `ct_cropping.use_cropped_for_*` is enabled. |
| `RS_custom.dcm` | Produced by `structure_merger.py` combining manual, auto, and `custom_structures*.yaml` definitions. | Contains derived/boolean ROIs (e.g., bowel bag, expanded PTVs). Timestamp-based staleness checks ensure it is regenerated when inputs change. |

---

## Aggregated Results (_RESULTS Directory)

### Overview

The `_RESULTS/` directory contains **all patient data in a single place**. This is the primary location for data analysis, machine learning, and statistical studies.

### 1. dvh_metrics.xlsx

**Purpose:** Contains dose-volume histogram (DVH) metrics for all structures across all patients.

**Key Columns:**
- `PatientID`: Unique patient identifier
- `CourseID`: Treatment course identifier
- `Structure`: Anatomical structure name (e.g., "bladder", "rectum", "ptv")
- `ROI_Type`: Type of contour
  - `MANUAL`: Manually drawn by clinician
  - `TOTALSEGMENTATOR`: Auto-segmented
  - `CUSTOM`: Custom model prediction
- `ROI_Volume_cc`: Structure volume in cubic centimeters

**Dose Metrics:**
- `Dmax_Gy`, `Dmean_Gy`, `Dmin_Gy`: Maximum, mean, minimum dose
- `D2cc`, `D5cc`, `D10cc`: Dose to 2cc, 5cc, 10cc of structure
- `D2%`, `D5%`, `D50%`, `D95%`, `D98%`: Dose to X% of volume
- `V5Gy`, `V10Gy`, `V20Gy`, `V30Gy`, `V40Gy`, `V50Gy`: Volume receiving ≥X Gy
- `V95%`, `V100%`, `V107%`, `V110%`: Volume receiving ≥X% of prescription dose

**Plan Quality Metrics:**
- `Conformality_Index`: Ratio of prescription isodose volume to PTV volume
- `Homogeneity_Index`: (D2% - D98%) / D50%
- `Coverage_V95%`: Percentage of PTV receiving ≥95% of prescription

**Example Use:**
```python
import pandas as pd

# Load DVH metrics
dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")

# Find all bladder V50Gy values
bladder_v50 = dvh[dvh['Structure'].str.contains('bladder', case=False)]['V50Gy']

# Compare manual vs auto-segmented rectum doses
rectum = dvh[dvh['Structure'].str.contains('rectum', case=False)]
manual_rectum = rectum[rectum['ROI_Type'] == 'MANUAL']
auto_rectum = rectum[rectum['ROI_Type'] == 'TOTALSEGMENTATOR']
```

---

### 2. radiomics_ct.xlsx

**Purpose:** Contains 150+ radiomic features extracted from CT images for all structures.

**Key Columns:**
- `PatientID`, `CourseID`, `Structure`: Identifiers
- `ROI_Type`: Source of segmentation
- `Volume_voxels`: Number of voxels in ROI
- `Cropping_Applied`: Whether CT cropping was used

**Feature Categories:**

#### Shape Features (3D geometry)
- `original_shape_Elongation`: Ratio of second to first principal axis
- `original_shape_Flatness`: Ratio of third to first principal axis
- `original_shape_Sphericity`: How sphere-like the ROI is
- `original_shape_SurfaceArea`: Surface area in mm²
- `original_shape_SurfaceVolumeRatio`: Surface area / volume
- `original_shape_Maximum3DDiameter`: Longest axis

#### First-Order Features (intensity statistics)
- `original_firstorder_Mean`, `Median`, `Std`: Basic statistics
- `original_firstorder_Skewness`, `Kurtosis`: Distribution shape
- `original_firstorder_Energy`, `Entropy`: Signal properties
- `original_firstorder_10Percentile`, `90Percentile`: Range metrics

#### Texture Features (spatial patterns)

**GLCM (Gray Level Co-occurrence Matrix):**
- `original_glcm_Contrast`: Local intensity variation
- `original_glcm_Correlation`: Linear dependency of gray levels
- `original_glcm_Homogeneity`: Closeness of distribution to diagonal
- `original_glcm_Energy`: Sum of squared elements

**GLRLM (Gray Level Run Length Matrix):**
- `original_glrlm_ShortRunEmphasis`: Distribution of short runs
- `original_glrlm_LongRunEmphasis`: Distribution of long runs
- `original_glrlm_RunLengthNonUniformity`: Similarity of run lengths

**GLSZM (Gray Level Size Zone Matrix):**
- `original_glszm_SmallAreaEmphasis`: Distribution of small zones
- `original_glszm_LargeAreaEmphasis`: Distribution of large zones
- `original_glszm_ZoneVariance`: Variance in zone sizes

**GLDM (Gray Level Dependence Matrix):**
- `original_gldm_DependenceEntropy`: Randomness of dependencies
- `original_gldm_LargeDependenceEmphasis`: Large connected regions

**NGTDM (Neighborhood Gray Tone Difference Matrix):**
- `original_ngtdm_Coarseness`: Spatial rate of intensity change
- `original_ngtdm_Contrast`: Intensity difference between regions
- `original_ngtdm_Complexity`: Information content

#### Filtered Features (multi-scale analysis)

**LoG (Laplacian of Gaussian):**
- `log-sigma-1.0mm_*`: Fine-scale edge detection
- `log-sigma-2.0mm_*`: Medium-scale features
- `log-sigma-3.0mm_*`, `log-sigma-5.0mm_*`: Coarse-scale features

**Wavelet:**
- `wavelet-HHH_*`, `wavelet-HHL_*`, etc.: Frequency decomposition features

**Example Use:**
```python
import pandas as pd

# Load radiomics features
radiomics = pd.read_excel("_RESULTS/radiomics_ct.xlsx")

# Select only original (non-filtered) features
original_features = radiomics.filter(regex='^original_')

# Get shape features for all bladders
bladder = radiomics[radiomics['Structure'].str.contains('bladder', case=False)]
shape_features = bladder.filter(regex='shape')

# Find highly heterogeneous tumors (high entropy)
high_entropy = radiomics[radiomics['original_firstorder_Entropy'] > radiomics['original_firstorder_Entropy'].median()]
```

---

### 3. radiomics_mr.xlsx

**Purpose:** MR radiomic features (if MR data available).

**Structure:** Similar to radiomics_ct.xlsx but:
- Uses different parameter file (`radiomics_params_mr.yaml`)
- Typically uses `binCount: 64` instead of `binWidth: 25`
- Features extracted from `total_mr` segmentations

**Key Difference:** MR lacks standardized intensity units (unlike HU in CT), so discretization and normalization differ.

---

### 4. case_metadata.xlsx

**Purpose:** Patient and treatment metadata.

**Key Columns:**
- `PatientID`, `CourseID`: Identifiers
- `PatientName`, `PatientBirthDate`, `PatientSex`: Demographics
- `CT_SeriesInstanceUID`: Unique CT identifier
- `CT_SliceThickness_mm`, `CT_PixelSpacing_mm`: Imaging parameters
- `CT_Dimensions`: Image size (e.g., "512x512x120")
- `RTPLAN_UID`: Treatment plan unique ID
- `RTDOSE_UID`: Dose distribution unique ID
- `PrescriptionDose_Gy`: Prescribed dose
- `NumberOfFractions`: Fractionation scheme
- `TreatmentIntent`: "CURATIVE" or "PALLIATIVE"
- `TreatmentSite`: Anatomical site (e.g., "PELVIS", "HEAD_NECK")
- `MR_Present`: Whether MR data exists (True/False)
- `MR_SeriesInstanceUIDs`: List of MR series

**Example Use:**
```python
# Load metadata
metadata = pd.read_excel("_RESULTS/case_metadata.xlsx")

# Find all prostate cases with MR
prostate_mr = metadata[
    (metadata['TreatmentSite'] == 'PELVIS') &
    (metadata['MR_Present'] == True)
]

# Get dose fractionation schemes
fractionation = metadata[['PrescriptionDose_Gy', 'NumberOfFractions']]
fractionation['DosePerFraction'] = fractionation['PrescriptionDose_Gy'] / fractionation['NumberOfFractions']
```

---

### 5. qc_reports.xlsx

**Purpose:** Quality control summary flagging potential issues.

**Key Columns:**
- `PatientID`, `CourseID`: Identifiers
- `CT_Cropping_Applied`: Whether systematic cropping was used
- `CT_Cropping_Region`: Anatomical region (e.g., "pelvis")
- `Structures_Fully_Cropped`: List of structures completely removed by cropping
- `Structures_Partially_Cropped`: List of structures partially cropped
- `Frame_Of_Reference_Consistent`: Whether all DICOM objects share same FOR
- `RTDOSE_Grid_Matches_CT`: Whether dose grid aligns with CT
- `Missing_DICOM_Types`: Any expected DICOM missing (e.g., "RTPLAN")
- `Segmentation_Failures`: Structures that failed auto-segmentation
- `Warnings`: Free-text warnings
- `Overall_Status`: "PASS", "WARNING", or "FAIL"

**Example Use:**
```python
# Load QC reports
qc = pd.read_excel("_RESULTS/qc_reports.xlsx")

# Find cases with cropping issues
cropping_issues = qc[qc['Structures_Fully_Cropped'].notna()]

# Find cases that failed QC
failed = qc[qc['Overall_Status'] == 'FAIL']

# Check for frame-of-reference mismatches
for_issues = qc[qc['Frame_Of_Reference_Consistent'] == False]
```

---

## Per-Patient Output

### When to Use Per-Patient Files

Use per-patient directories when you need:
- Individual case review
- Image visualization
- DICOM exports for clinical systems
- Debugging specific processing steps
- Accessing original DICOM metadata

### NIFTI Files

**Format:** Compressed NIfTI (`.nii.gz`)

**Coordinate System:** RAS (Right-Anterior-Superior)
- X: Right (+) to Left (-)
- Y: Anterior (+) to Posterior (-)
- Z: Superior (+) to Inferior (-)

**Loading in Python:**
```python
import SimpleITK as sitk
import numpy as np

# Load CT
ct = sitk.ReadImage("PatientID/CourseID/NIFTI/ct.nii.gz")
ct_array = sitk.GetArrayFromImage(ct)  # Shape: (Z, Y, X)
spacing = ct.GetSpacing()  # (X, Y, Z) in mm
origin = ct.GetOrigin()    # (X, Y, Z) coordinates

# Load segmentation mask
mask = sitk.ReadImage("PatientID/CourseID/Segmentation_TotalSegmentator/bladder--ct.nii.gz")
mask_array = sitk.GetArrayFromImage(mask)  # Binary: 0 or 1

# Calculate volume
voxel_volume_mm3 = np.prod(spacing)
structure_volume_cc = np.sum(mask_array) * voxel_volume_mm3 / 1000
```

---

### RTSTRUCT Files

**RS.dcm:** Original manually drawn contours

**RS_auto.dcm:** TotalSegmentator auto-segmentations in DICOM format
- Can be imported into Eclipse, RayStation, MIM, etc.
- Uses same Frame of Reference as original CT
- Contains 100+ organs/tissues

**RS_custom.dcm:** Custom model predictions (if enabled)

**Naming Convention in RTSTRUCT:**
- Auto-segmented: `{organ}` (e.g., "bladder", "rectum")
- Manual: Original name from clinical system
- Custom: `{model_name}_{structure}`

---

## File Formats and Column Descriptions

### DVH Metrics Column Reference

| Column | Unit | Description | Typical Use |
|--------|------|-------------|-------------|
| `Dmax_Gy` | Gy | Maximum dose to structure | Serial organ toxicity |
| `Dmean_Gy` | Gy | Mean dose to structure | Parallel organ toxicity |
| `D2cc` | Gy | Dose to 2cc hottest volume | Rectum/bladder constraints |
| `D50%` | Gy | Median dose | Homogeneity assessment |
| `D95%` | Gy | Dose to 95% of volume | Target coverage |
| `V95%` | % | Volume receiving ≥95% Rx | PTV coverage metric |
| `V100%` | % | Volume receiving ≥100% Rx | Adequate coverage |
| `V50Gy` | cc | Volume receiving ≥50 Gy | OAR sparing |
| `Conformality_Index` | - | V_Rx / V_PTV | Plan quality (ideal: 1.0) |
| `Homogeneity_Index` | - | (D2% - D98%) / D50% | Dose uniformity (lower better) |

---

### Radiomics Feature Interpretation

#### Feature Name Format

`{filter}_{class}_{feature}`

**Examples:**
- `original_firstorder_Mean`: Mean HU from original image
- `log-sigma-2.0mm_glcm_Contrast`: GLCM contrast from LoG-filtered image (sigma=2mm)
- `wavelet-HHL_shape_Sphericity`: Sphericity from wavelet high-high-low decomposition

#### Feature Stability

**Tier 1 (ICC >0.9):** Highly reproducible
- Shape features (Sphericity, Volume, SurfaceArea)
- First-order Mean, Median
- GLCM Energy, Homogeneity

**Tier 2 (ICC 0.75-0.9):** Moderately reproducible
- First-order Entropy, Skewness
- GLRLM features
- GLSZM features

**Tier 3 (ICC <0.75):** Lower reproducibility
- High-order texture features
- Wavelet features
- Small ROI-derived metrics

**Recommendation:** For clinical models, prioritize Tier 1 features or validate stability in your dataset.

---

## Understanding DVH Metrics

### Critical DVH Metrics by Organ Type

#### Target Volumes (PTV, CTV, GTV)
- **V95%**: Should be >95% (adequate coverage)
- **V107%**: Should be <2% (avoid hotspots)
- **Conformality_Index**: Ideally 1.0-1.2 (tight conformality)
- **Homogeneity_Index**: <0.15 (uniform dose)
- **D98%**: Minimum dose to 98% of target

#### Serial Organs (spinal cord, brainstem, optic nerves)
- **Dmax_Gy**: Critical constraint (e.g., cord <45 Gy)
- **D2%**: Near-maximum dose (more robust than Dmax)

#### Parallel Organs (lungs, kidneys, parotids)
- **Dmean_Gy**: Mean dose correlates with function loss
- **V20Gy**, **V30Gy**: Percentage volumes for toxicity prediction

#### Hollow Organs (rectum, bladder, bowel)
- **D2cc**: Dose to 2cc hottest volume (QUANTEC constraints)
- **V50Gy**, **V60Gy**: Toxicity thresholds

---

### Interpreting Cropped vs Uncropped Data

**If `ct_cropping.enabled: true` in config:**

**Purpose:** Normalize volumes across patients for meaningful %DVH metrics

**Example (Prostate):**
- Without cropping: Patient A's bladder is fully in CT (300cc), Patient B's is partially cut off (180cc)
- With cropping (L1 to femoral heads): Both patients have same anatomical volume denominator
- Result: V50Gy is now comparable between patients

**QC Check:**
- If `Structures_Fully_Cropped` is not empty → structure completely outside cropped region
- If `Structures_Partially_Cropped` is not empty → structure clipped by cropping
- **Action:** Review whether cropping is appropriate for your analysis

---

## Understanding Radiomics Features

### Feature Families

#### 1. Shape Features (Geometry-based)
**Independent of intensity values**

- **Volume, SurfaceArea:** Absolute size metrics
- **Sphericity:** 1.0 = perfect sphere, <1.0 = irregular
- **Elongation, Flatness:** Shape asymmetry (1.0 = symmetric)
- **SurfaceVolumeRatio:** Higher = more irregular/spiky surface

**Use Cases:**
- Tumor aggressiveness (irregular shapes often more aggressive)
- Distinguishing benign vs malignant
- Treatment planning (complex shapes harder to target)

---

#### 2. First-Order Features (Intensity distribution)
**Based on histogram of HU values**

- **Mean, Median:** Average intensity
- **Std, Variance:** Spread of intensities
- **Skewness:** Asymmetry (positive = tail toward high HU)
- **Kurtosis:** Peakedness (high = concentrated around mean)
- **Energy:** Sum of squared intensities (volume-dependent)
- **Entropy:** Randomness (higher = more heterogeneous)

**Use Cases:**
- Necrosis detection (low mean, high entropy)
- Calcification identification (high mean, low entropy)
- Edema quantification (low HU, high volume)

---

#### 3. Texture Features (Spatial patterns)

**GLCM (Gray Level Co-occurrence Matrix):**
- Captures **neighboring voxel relationships**
- `Contrast`: Large = abrupt intensity changes
- `Homogeneity`: Large = uniform texture
- `Correlation`: Positive = predictable patterns

**GLRLM (Gray Level Run Length Matrix):**
- Captures **consecutive voxels with same intensity**
- `ShortRunEmphasis`: High = fragmented texture
- `LongRunEmphasis`: High = homogeneous regions
- `RunLengthNonUniformity`: High = variable run lengths

**GLSZM (Gray Level Size Zone Matrix):**
- Captures **connected regions of same intensity**
- `SmallAreaEmphasis`: High = many small zones
- `LargeAreaEmphasis`: High = large homogeneous zones
- `ZoneVariance`: Variability in zone sizes

**Use Cases:**
- Predicting treatment response (heterogeneous tumors often resistant)
- Grading tumors (high-grade = high entropy, low homogeneity)
- Identifying subtypes (different texture patterns)

---

#### 4. Filtered Features

**LoG (Laplacian of Gaussian):**
- **Edge detection at multiple scales**
- `sigma-1.0mm`: Fine details (edges of small structures)
- `sigma-5.0mm`: Coarse patterns (large regions)
- Use: Multi-scale tumor characterization

**Wavelet:**
- **Frequency decomposition**
- `HHH`: High frequency in all dimensions (fine texture)
- `LLL`: Low frequency in all dimensions (coarse texture)
- Use: Capturing texture at different resolutions

---

### Common Radiomics Pitfalls

1. **Volume Confounding:**
   - Features like Energy, TotalEnergy are highly correlated with ROI volume
   - **Solution:** Normalize by volume or exclude from models

2. **Collinearity:**
   - Many features are correlated (|r| >0.95)
   - **Solution:** Perform feature selection (e.g., remove pairs with r >0.95)

3. **Small ROI Instability:**
   - Texture features unreliable for very small ROIs (<64 voxels)
   - **Check:** `Volume_voxels` column in radiomics output
   - **Filter:** Pipeline already excludes ROIs <10 voxels

4. **Cropping Effects:**
   - Features change if ROI is partially cropped
   - **Check:** `Cropping_Applied` column and QC reports

5. **Scanner/Protocol Variability:**
   - Different scanners/protocols yield different features
   - **Solution:** ComBat harmonization or include scanner as covariate

---

## Quality Control Interpretation

### QC Workflow

1. **Load QC reports:**
   ```python
   qc = pd.read_excel("_RESULTS/qc_reports.xlsx")
   ```

2. **Check overall status:**
   ```python
   print(qc['Overall_Status'].value_counts())
   # PASS: 45, WARNING: 3, FAIL: 2
   ```

3. **Investigate failures:**
   ```python
   failed_cases = qc[qc['Overall_Status'] == 'FAIL']
   print(failed_cases[['PatientID', 'Warnings']])
   ```

4. **Review cropping issues:**
   ```python
   cropped = qc[qc['Structures_Fully_Cropped'].notna()]
   ```

---

### Common QC Issues

#### 1. Structures Fully Cropped
**Meaning:** Structure completely outside cropped CT region

**Example:**
```
Structures_Fully_Cropped: ["liver", "spleen"]
CT_Cropping_Region: "pelvis"
```

**Action:**
- If analyzing pelvic structures only: Ignore (expected)
- If analyzing liver: Disable cropping or adjust margins

---

#### 2. Frame of Reference Mismatch
**Meaning:** RTDOSE/RTSTRUCT not aligned with CT

**Example:**
```
Frame_Of_Reference_Consistent: False
```

**Action:**
- Check DICOM metadata for registration issues
- DVH metrics may be unreliable
- Contact physicist/data curator

---

#### 3. Segmentation Failures
**Meaning:** TotalSegmentator could not segment certain organs

**Example:**
```
Segmentation_Failures: ["lung_left", "lung_right"]
CT_Cropping_Region: "pelvis"
```

**Action:**
- Expected if organs not in FOV (e.g., lungs in pelvic CT)
- If unexpected, check CT quality/artifacts

---

#### 4. Missing DICOM Types
**Meaning:** Expected files not found

**Example:**
```
Missing_DICOM_Types: ["RTPLAN"]
```

**Action:**
- RTPLAN missing: No prescription dose, DVH not normalized
- RTDOSE missing: No DVH analysis possible
- RTSTRUCT missing: No manual contours

---

## Common Use Cases

### Use Case 1: Build Toxicity Prediction Model

**Goal:** Predict Grade 2+ rectal toxicity from DVH metrics

**Steps:**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")
metadata = pd.read_excel("_RESULTS/case_metadata.xlsx")

# Filter rectum metrics
rectum = dvh[dvh['Structure'].str.contains('rectum', case=False)].copy()

# Select features
features = ['Dmean_Gy', 'D2cc', 'V50Gy', 'V60Gy', 'V65Gy']
X = rectum[features]

# Merge with outcomes (assuming you have toxicity labels)
# y = toxicity_labels['Grade2+_Rectum']

# Train model
# clf = RandomForestClassifier()
# clf.fit(X, y)
```

---

### Use Case 2: Radiomics-Based Outcome Prediction

**Goal:** Predict treatment response from CT radiomics

**Steps:**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load radiomics
radiomics = pd.read_excel("_RESULTS/radiomics_ct.xlsx")

# Filter to PTV/GTV features only
ptv = radiomics[radiomics['Structure'].str.contains('ptv|gtv', case=False, regex=True)].copy()

# Select original features only (most stable)
feature_cols = [col for col in ptv.columns if col.startswith('original_')]
X = ptv[feature_cols]

# Remove features with >95% missing
X = X.dropna(axis=1, thresh=len(X) * 0.95)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Now merge with outcomes and train model
```

---

### Use Case 3: Validate Auto-Segmentation

**Goal:** Compare TotalSegmentator vs manual contours

**Steps:**
```python
import pandas as pd

# Load DVH metrics
dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")

# Filter to specific structure
bladder = dvh[dvh['Structure'].str.contains('bladder', case=False)].copy()

# Separate manual vs auto
manual = bladder[bladder['ROI_Type'] == 'MANUAL']
auto = bladder[bladder['ROI_Type'] == 'TOTALSEGMENTATOR']

# Merge on PatientID/CourseID
comparison = manual.merge(
    auto,
    on=['PatientID', 'CourseID'],
    suffixes=('_manual', '_auto')
)

# Compare volumes
comparison['Volume_Diff_cc'] = comparison['ROI_Volume_cc_auto'] - comparison['ROI_Volume_cc_manual']
comparison['Volume_Diff_Pct'] = 100 * comparison['Volume_Diff_cc'] / comparison['ROI_Volume_cc_manual']

# Compare DVH metrics
comparison['Dmean_Diff_Gy'] = comparison['Dmean_Gy_auto'] - comparison['Dmean_Gy_manual']

print(f"Mean volume difference: {comparison['Volume_Diff_Pct'].mean():.1f}%")
print(f"Mean Dmean difference: {comparison['Dmean_Diff_Gy'].mean():.1f} Gy")
```

---

### Use Case 4: Extract Metadata for Cohort Description

**Goal:** Create Table 1 for publication

**Steps:**
```python
import pandas as pd

# Load metadata
metadata = pd.read_excel("_RESULTS/case_metadata.xlsx")

# Summary statistics
print("Cohort Summary:")
print(f"Total patients: {metadata['PatientID'].nunique()}")
print(f"Total courses: {len(metadata)}")
print(f"Sex: {metadata['PatientSex'].value_counts()}")
print(f"Mean age: {metadata['PatientAge'].mean():.1f}")
print(f"Treatment sites: {metadata['TreatmentSite'].value_counts()}")
print(f"Mean prescription dose: {metadata['PrescriptionDose_Gy'].mean():.1f} Gy")
print(f"Mean fractions: {metadata['NumberOfFractions'].mean():.1f}")
```

---

## Troubleshooting

### Issue: Missing Radiomics Features

**Symptom:** radiomics_ct.xlsx has many NaN values

**Causes:**
1. ROI too small (<64 voxels) → Excluded by pipeline
2. ROI in `skip_rois` list (e.g., "body", "bones")
3. ROI partially/fully cropped
4. Feature extraction error

**Debug:**
```python
radiomics = pd.read_excel("_RESULTS/radiomics_ct.xlsx")

# Check ROI sizes
print(radiomics[['Structure', 'Volume_voxels']].sort_values('Volume_voxels'))

# Check cropping
qc = pd.read_excel("_RESULTS/qc_reports.xlsx")
print(qc[['PatientID', 'Structures_Fully_Cropped', 'Structures_Partially_Cropped']])

# Check skip list in config.yaml
# skip_rois: [body, couchsurface, bones]
```

---

### Issue: DVH Metrics Don't Match Clinical System

**Symptom:** V95%, Dmean differ from Eclipse/RayStation

**Causes:**
1. Different interpolation methods
2. Frame-of-reference mismatch
3. Different structure versions
4. Dose grid resolution

**Debug:**
```python
qc = pd.read_excel("_RESULTS/qc_reports.xlsx")

# Check FOR consistency
print(qc[['PatientID', 'Frame_Of_Reference_Consistent']])

# Check dose grid alignment
print(qc[['PatientID', 'RTDOSE_Grid_Matches_CT']])

# Compare structure volumes
dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")
print(dvh[['Structure', 'ROI_Type', 'ROI_Volume_cc']])
```

**Action:**
- If FOR inconsistent: Re-export DICOM ensuring same FOR
- If volumes differ: Check which structure version (manual vs auto)
- Small differences (<2%) are normal due to interpolation

---

### Issue: Pipeline Crashed During Processing

**Symptom:** Incomplete output, missing _RESULTS/

**Debug:**
1. Check logs: `{logs_dir}/snakemake.log`
2. Check per-stage logs: `{logs_dir}/{stage}/{PatientID}_{CourseID}.log`
3. Check QC reports for failed cases

**Common Errors:**
- **Out of memory:** Reduce `segmentation.max_workers` or enable `fast: true`
- **GPU out of memory:** Set `force_split: true` or `device: cpu`
- **DICOM corruption:** Check `Missing_DICOM_Types` in QC
- **NumPy version conflict:** Radiomics requires NumPy <2.0

**Recovery:**
```bash
# Resume from checkpoint
snakemake --cores all --use-conda --rerun-incomplete
```

---

### Issue: Segmentation Quality Poor

**Symptom:** TotalSegmentator masks don't match anatomy

**Causes:**
1. Non-standard CT (e.g., CBCT, limited FOV)
2. Severe artifacts (metal, motion)
3. Pediatric anatomy
4. Cropped anatomy (e.g., head-first vs feet-first)

**Debug:**
```python
# Visual inspection needed - load NIfTI in 3D Slicer or ITK-SNAP
import SimpleITK as sitk

ct = sitk.ReadImage("PatientID/CourseID/NIFTI/ct.nii.gz")
bladder = sitk.ReadImage("PatientID/CourseID/Segmentation_TotalSegmentator/bladder--ct.nii.gz")

# Check if mask is empty
bladder_array = sitk.GetArrayFromImage(bladder)
print(f"Bladder voxels: {bladder_array.sum()}")
```

**Action:**
- If critical structures missing: Use manual contours (`ROI_Type == 'MANUAL'`)
- If anatomy non-standard: Consider custom nnUNet model
- For research: Filter out low-quality cases using QC reports

---

## Summary Checklist for AI Agents

When working with RTpipeline output:

- [ ] Start with `_RESULTS/` directory for aggregated data
- [ ] Load `case_metadata.xlsx` to understand cohort
- [ ] Load `qc_reports.xlsx` to identify problematic cases
- [ ] Filter QC failures (`Overall_Status == 'FAIL'`) from analysis
- [ ] Check `Cropping_Applied` when interpreting volumes
- [ ] Use `ROI_Type` to distinguish manual vs auto-segmentation
- [ ] Filter radiomics features by stability (prefer `original_` features)
- [ ] Remove features with high correlation (|r| >0.95)
- [ ] Validate DVH metrics against clinical constraints (e.g., QUANTEC)
- [ ] Check `Volume_voxels` to exclude tiny ROIs from radiomics
- [ ] Use per-patient directories for detailed debugging only
- [ ] Document any excluded cases in your analysis pipeline

---

## Additional Resources

- **Pipeline Documentation:** See `README.md` in repository root
- **Configuration Reference:** See `config.yaml` with inline comments
- **PyRadiomics Documentation:** https://pyradiomics.readthedocs.io/
- **IBSI Guidelines:** https://theibsi.github.io/
- **TotalSegmentator Paper:** Wasserthal et al., 2023 (arXiv:2208.05868)
- **QUANTEC Guidelines:** https://www.redjournal.org/QUANTEC

---

**Document Version:** 1.0
**Last Updated:** 2025-11-13
**Pipeline Version:** Compatible with rtpipeline v2.0+
