# RTpipeline Output Format - Quick Reference

> **Quick reference for the most common tasks.**

---

## 📁 **Where to Start**

**All aggregated results are in:** `{output_dir}/_RESULTS/`

```
_RESULTS/
├── dvh_metrics.xlsx         ← All dose-volume metrics
├── radiomics_ct.xlsx        ← All CT radiomic features
├── radiomics_mr.xlsx        ← All MR radiomic features
├── radiomics_robustness_summary.xlsx ← Cohort stability summary (if enabled)
├── case_metadata.xlsx       ← Patient/treatment info
└── qc_reports.xlsx          ← Quality control summary
```

---

## 🔥 **Most Common Tasks**

### Load DVH Metrics
```python
import pandas as pd

dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")

# Filter by structure
bladder = dvh[dvh['Structure'].str.contains('bladder', case=False)]

# Get specific metrics
v50_values = bladder['V50Gy']  # Volume receiving ≥50 Gy
dmean_values = bladder['Dmean_Gy']  # Mean dose
```

### Load Radiomics Features
```python
radiomics = pd.read_excel("_RESULTS/radiomics_ct.xlsx")

# Get only original (most stable) features
original_features = radiomics.filter(regex='^original_')

# Get shape features
shape_features = radiomics.filter(regex='shape')

# Filter by structure
ptv = radiomics[radiomics['Structure'].str.contains('ptv', case=False)]
```

### Load Robustness Summary
```python
robustness = pd.read_excel(
    "_RESULTS/radiomics_robustness_summary.xlsx",
    sheet_name="robust_features"
)

# Features that passed the default RTpipeline v2.1.0 thresholds
robust_feature_names = robustness["feature_name"].tolist()
```

### Compare Auto vs Manual Segmentation
```python
# Load DVH
dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")

# Separate by source
manual = dvh[dvh['ROI_Type'] == 'MANUAL']
auto = dvh[dvh['ROI_Type'] == 'TOTALSEGMENTATOR']

# Merge and compare
comparison = manual.merge(
    auto,
    on=['PatientID', 'CourseID', 'Structure'],
    suffixes=('_manual', '_auto')
)

# Volume difference
comparison['volume_diff_%'] = 100 * (
    comparison['ROI_Volume_cc_auto'] - comparison['ROI_Volume_cc_manual']
) / comparison['ROI_Volume_cc_manual']
```

---

## 📊 **Key Column Reference**

### DVH Metrics (dvh_metrics.xlsx)

| Column | Description | Example Use |
|--------|-------------|-------------|
| `Dmean_Gy` | Mean dose | Parallel organ toxicity (lung, kidney) |
| `Dmax_Gy` | Maximum dose | Serial organ toxicity (spinal cord) |
| `D2cc` | Dose to 2cc hottest volume | Rectum/bladder constraints |
| `D50%` | Median dose | Homogeneity |
| `D95%` | Dose to 95% of volume | Target coverage |
| `V95%` | Volume receiving ≥95% Rx | PTV coverage |
| `V50Gy` | Volume receiving ≥50 Gy | OAR sparing |
| `ROI_Volume_cc` | Structure volume | Normalization |

### Radiomics Features (radiomics_ct.xlsx)

| Feature Prefix | Type | Description |
|----------------|------|-------------|
| `original_shape_*` | Geometry | Volume, sphericity, elongation |
| `original_firstorder_*` | Intensity | Mean, std, entropy, skewness |
| `original_glcm_*` | Texture | Contrast, homogeneity, correlation |
| `original_glrlm_*` | Texture | Run length patterns |
| `log-sigma-*` | Filtered | Multi-scale edge detection |
| `wavelet-*` | Filtered | Frequency decomposition |

**Recommendation:** Start with `original_*` features (most stable)

---

## ⚠️ **Quality Control Quick Check**

```python
qc = pd.read_excel("_RESULTS/qc_reports.xlsx")

# Check for failures
failures = qc[qc['Overall_Status'] == 'FAIL']
print(f"{len(failures)} cases failed QC")

# Check for cropping issues
cropped = qc[qc['Structures_Fully_Cropped'].notna()]
print(f"{len(cropped)} cases have fully cropped structures")

# Check frame-of-reference issues
for_issues = qc[qc['Frame_Of_Reference_Consistent'] == False]
print(f"{len(for_issues)} cases have FOR mismatches")
```

---

## 🎯 **Feature Selection Tips**

### Remove Correlated Features (r > 0.95)
```python
import numpy as np

def remove_correlated(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)

# Apply
features_clean = remove_correlated(original_features)
```

### Filter by Stability (Use Tier 1 Features)
```python
# Tier 1 features (ICC >0.9, most reproducible)
tier1_features = [
    col for col in radiomics.columns
    if any(x in col for x in [
        'original_shape_',
        'original_firstorder_Mean',
        'original_firstorder_Median',
        'original_glcm_Energy',
        'original_glcm_Homogeneity'
    ])
]

stable_features = radiomics[tier1_features]
```

---

## 📦 **File Locations**

### For a specific patient/course:
```
{output_dir}/{PatientID}/{CourseID}/
├── DICOM/CT/                    # Original DICOM
├── NIFTI/ct.nii.gz              # CT volume
├── Segmentation_TotalSegmentator/  # Auto-segmented organs
│   ├── bladder--ct.nii.gz
│   ├── rectum--ct.nii.gz
│   └── [100+ more structures]
├── RS.dcm / RS_auto(.cropped).dcm / RS_custom.dcm
├── radiomics_ct.xlsx            # Per-course radiomics
├── radiomics_robustness_ct.parquet  # Per-course perturbation-level robustness data
├── dvh_metrics.xlsx             # Per-course DVH
└── qc_reports/                  # Quality control
```

**RTSTRUCT quick guide**
- `RS.dcm`: untouched copy of the clinical RTSTRUCT imported from the TPS.
- `RS_auto.dcm`: TotalSegmentator output on the uncropped CT grid.
- `RS_auto_cropped.dcm`: cropped variant used when `ct_cropping.use_cropped_for_*` is enabled.
- `RS_custom.dcm`: merged manual + auto + `custom_structures*.yaml` definitions (boolean/margin ROIs).

---

## 🔧 **Common Issues & Fixes**

### Issue: Missing radiomics features (NaN values)
**Cause:** ROI too small (<64 voxels) or in skip list
**Fix:**
```python
# Check ROI size
radiomics[['Structure', 'Volume_voxels']].sort_values('Volume_voxels')

# Check QC for cropping
qc[['PatientID', 'Structures_Fully_Cropped']]
```

### Issue: DVH metrics don't match clinical system
**Cause:** Different interpolation or frame-of-reference mismatch
**Fix:**
```python
# Check FOR consistency
qc[['PatientID', 'Frame_Of_Reference_Consistent']]

# Compare volumes
dvh[['Structure', 'ROI_Type', 'ROI_Volume_cc']]
```

---

## 🚀 **Quick Analysis Templates**

### Toxicity Prediction Model
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
dvh = pd.read_excel("_RESULTS/dvh_metrics.xlsx")

# Filter organ
rectum = dvh[dvh['Structure'].str.contains('rectum', case=False)]

# Features
X = rectum[['Dmean_Gy', 'D2cc', 'V50Gy', 'V60Gy']]

# Train (assuming you have outcome labels)
# y = outcomes['Grade2+_Toxicity']
# clf = RandomForestClassifier()
# clf.fit(X, y)
```

### Radiomics Outcome Prediction
```python
# Load radiomics
rad = pd.read_excel("_RESULTS/radiomics_ct.xlsx")

# Filter to target structures
ptv = rad[rad['Structure'].str.contains('ptv', case=False)]

# Original features only (most stable)
features = ptv.filter(regex='^original_')

# Remove high correlation
features_clean = remove_correlated(features, threshold=0.95)

# Standardize and train model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features_clean)
```

### Cohort Description (Table 1)
```python
metadata = pd.read_excel("_RESULTS/case_metadata.xlsx")

print("Cohort Summary:")
print(f"Patients: {metadata['PatientID'].nunique()}")
print(f"Courses: {len(metadata)}")
print(f"\nSex distribution:\n{metadata['PatientSex'].value_counts()}")
print(f"\nMean prescription: {metadata['PrescriptionDose_Gy'].mean():.1f} Gy")
print(f"Mean fractions: {metadata['NumberOfFractions'].mean():.1f}")
```

---

## 📚 **Resources**

- **Setup Script:** `./setup_docker_project.sh`
- **PyRadiomics Docs:** https://pyradiomics.readthedocs.io/
- **IBSI Standards:** https://theibsi.github.io/
- **Pipeline Overview:** [Home](../index.md)

---

## 🆘 **Need Help?**

1. Check QC reports first: `_RESULTS/qc_reports.xlsx`
2. Check logs: `{logs_dir}/`
3. Review technical documentation: [Architecture](../technical/architecture.md)

---

**Document Version:** 1.0
**Compatible with:** rtpipeline v2.1.0+
**Last Updated:** 2025-11-13
