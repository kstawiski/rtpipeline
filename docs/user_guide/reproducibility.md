# Reproducible Research with RTpipeline

**A guide to ensuring methodological rigor and reproducibility in radiotherapy data science**

This document provides guidance for researchers—particularly PhD students and postdocs—on how to leverage RTpipeline's features to produce reproducible, publication-ready analyses.

---

## Why Reproducibility Matters

In radiotherapy research, reproducibility failures often stem from:

1. **Undocumented preprocessing** - DVH interpolation methods, dose grid resolutions, and structure definitions are not specified
2. **Environment drift** - Package versions change between analysis runs
3. **Configuration ambiguity** - Key parameters (e.g., radiomics bin width) are buried in code
4. **Data provenance gaps** - No clear link between raw DICOM and final features

RTpipeline addresses these by encoding all preprocessing decisions in **version-controlled configuration files** and providing complete **audit trails**.

---

## The Reproducibility Checklist

Use this checklist before submitting a manuscript or thesis:

### Essential Elements

- [ ] RTpipeline version recorded (git commit hash or release tag)
- [ ] Complete `config.yaml` saved with results
- [ ] Conda environment exported (`conda env export > environment.yml`)
- [ ] QC reports reviewed and saved
- [ ] Structure mapping dictionary documented
- [ ] Random seeds fixed (where applicable)

### For Radiomics Studies

- [ ] PyRadiomics parameters documented (`radiomics_params.yaml`)
- [ ] IBSI feature definitions referenced
- [ ] Robustness analysis (ICC, CoV) completed
- [ ] Perturbation chain parameters recorded

### For Multi-Center Studies

- [ ] Shared configuration bundle created and versioned
- [ ] All centers using identical RTpipeline version
- [ ] Structure mapping harmonized across sites
- [ ] QC criteria applied consistently

---

## Configuration as Protocol

### The Key Principle

**Your `config.yaml` IS your preprocessing protocol.** It should be:

- Version-controlled (git)
- Published as supplementary material
- Citable (via DOI if possible)

### Example Configuration Header

```yaml
# RTpipeline Configuration for Prostate Toxicity Study
# Protocol Version: 1.2
# Date: 2025-01-15
# DOI: 10.5281/zenodo.XXXXXXX
#
# This configuration defines the complete preprocessing pipeline
# for the PROTECT-01 multi-center prostate radiotherapy study.
#
# Modifications from v1.1:
# - Updated ICC threshold from 0.75 to 0.90 per reviewer feedback
# - Added V75Gy metric for rectal dose-response analysis

version: "1.2"
study_id: "PROTECT-01"
```

### Key Configuration Sections to Document

```yaml
# Structure harmonization
structure_mapping:
  RECTUM:
    patterns: ["Rectum*", "RECT*", "rectum*"]
    required: true
    # Documentation: Maps all local rectum names to canonical RECTUM

# DVH computation parameters
dvh:
  interpolation: "linear"  # Method for DVH curve interpolation
  dose_units: "Gy"         # Absolute dose units
  volume_type: "relative"  # Relative (%) or absolute (cc)
  metrics:
    - Dmean
    - Dmax
    - D2cc
    - V70Gy

# Radiomics extraction
radiomics:
  binWidth: 25              # Fixed bin width in HU (IBSI compliant)
  resampling_mm: [1, 1, 3]  # Isotropic x/y, 3mm z
  interpolator: "sitkBSpline"

# Robustness assessment
radiomics_robustness:
  enabled: true
  thresholds:
    icc:
      robust: 0.90      # Features with ICC >= 0.90 classified as robust
      acceptable: 0.75
    cov:
      robust_pct: 10.0  # Features with CoV <= 10% classified as robust
```

---

## Environment Reproducibility

### Dual Environment Architecture

RTpipeline uses two conda environments to resolve dependency conflicts:

| Environment | Purpose | Key Dependencies |
|------------|---------|------------------|
| `rtpipeline` | Main processing | NumPy 2.x, TotalSegmentator, SimpleITK |
| `rtpipeline-radiomics` | Radiomics extraction | NumPy 1.x, PyRadiomics |

### Exporting Environments

```bash
# Export main environment
conda activate rtpipeline
conda env export --no-builds > rtpipeline-environment.yml

# Export radiomics environment
conda activate rtpipeline-radiomics
conda env export --no-builds > rtpipeline-radiomics-environment.yml
```

### Docker for Complete Reproducibility

For maximum reproducibility, use the Docker image with a specific tag:

```bash
# Use a specific version, not :latest
docker pull kstawiski/rtpipeline:v2.0.0

# Record the image digest
docker inspect --format='{{.RepoDigests}}' kstawiski/rtpipeline:v2.0.0
```

Document in your methods:
> "All preprocessing was performed using RTpipeline Docker image kstawiski/rtpipeline:v2.0.0 (SHA256: abc123...)."

---

## Methods Section Templates

### Template 1: DVH Extraction

```markdown
**Dose-Volume Histogram Analysis**

DVH metrics were extracted using RTpipeline (version X.X, git commit: XXXXXX) [1].
DICOM RTSTRUCT, RTDOSE, and RTPLAN files were exported from [TPS name/version]
and processed using the following standardized protocol:

1. **Structure Harmonization:** Contour names were mapped to canonical nomenclature
   using a predefined dictionary (Supplementary Table S1). Structures named
   [list variations] were mapped to the canonical label [STRUCTURE_NAME].

2. **DVH Computation:** Cumulative DVH curves were computed using [interpolation
   method] interpolation with a dose resolution of [X] Gy. The following metrics
   were derived:
   - Mean dose (D_mean)
   - Maximum dose (D_max)
   - Dose to hottest 2cc (D_2cc)
   - Volume receiving ≥X Gy (V_XGy)

3. **Quality Control:** Cases with [QC criteria] were flagged and reviewed.
   [N] cases were excluded due to [reasons].

The complete preprocessing configuration is available as Supplementary File S2.

[1] RTpipeline. https://github.com/kstawiski/rtpipeline
```

### Template 2: Radiomics with Robustness Assessment

```markdown
**Radiomic Feature Extraction and Stability Assessment**

CT radiomic features were extracted using RTpipeline (version X.X) [1] with
PyRadiomics (version X.X) [2] following Image Biomarker Standardisation
Initiative (IBSI) recommendations [3].

**Preprocessing:**
- Images were resampled to [X × X × X mm] isotropic voxels using
  [interpolation method]
- Intensity discretization used a fixed bin width of [X] HU
- [CT cropping method, if applicable]

**Feature Extraction:**
A total of [N] features were extracted from [structure(s)], comprising:
- Shape features (N = X)
- First-order statistics (N = X)
- Texture features: GLCM (N = X), GLRLM (N = X), GLSZM (N = X), GLDM (N = X)
- [Filtered features if applicable]

**Robustness Assessment:**
Feature stability was assessed using NTCV perturbation chains following
Zwanenburg et al. [4]. For each ROI, [N] perturbations were generated:
- Gaussian noise injection (σ = 0, 10, 20 HU)
- Rigid translations (±3 mm in each axis)
- Contour randomization (N = X realizations)
- Volume adaptation (±15% erosion/dilation)

Intraclass correlation coefficients ICC(3,1) were computed using Pingouin [5].
Features were classified as:
- **Robust:** ICC ≥ 0.90 and CoV ≤ 10%
- **Acceptable:** ICC ≥ 0.75 and CoV ≤ 20%
- **Poor:** ICC < 0.75 or CoV > 20%

[N] features ([X]%) met robustness criteria and were retained for modeling.

**References:**
[1] RTpipeline. https://github.com/kstawiski/rtpipeline
[2] van Griethuysen JJM, et al. Cancer Res. 2017;77(21):e104-e107.
[3] Zwanenburg A, et al. Radiology. 2020;295(2):328-338.
[4] Zwanenburg A, et al. Sci Rep. 2019;9:614.
[5] Vallat R. JOSS. 2018;3(31):1026.
```

### Template 3: Multi-Center Harmonization

```markdown
**Multi-Center Data Harmonization**

Data from [N] centers were harmonized using RTpipeline (version X.X) [1] as a
standardized ETL framework. Each center independently processed local DICOM
exports using an identical configuration bundle (Supplementary File S3).

**Harmonization Protocol:**
1. **Structure Mapping:** A consortium-wide structure dictionary was defined
   (Supplementary Table S1) mapping site-specific nomenclature to [N] canonical
   structure labels.

2. **Preprocessing Standardization:**
   - CT resampling: [X × X × X mm]
   - Dose grid interpolation: [method]
   - Field-of-view cropping: [region]-based using TotalSegmentator landmarks

3. **Quality Control:**
   - All centers applied identical QC criteria (Supplementary Table S2)
   - [N] cases across all sites were excluded due to [reasons]

4. **Feature Extraction:**
   - DVH metrics: [list]
   - Radiomics features: [list or reference to table]

**Data Governance:**
Raw DICOM data remained at each institution. Only [harmonized feature
tables / model weights / aggregated statistics] were shared with the
coordinating center.

The RTpipeline configuration bundle is available at [DOI/URL].

[1] RTpipeline. https://github.com/kstawiski/rtpipeline
```

---

## Provenance Tracking

### RTpipeline Output Structure

Each processed course includes provenance metadata:

```
Output/{PatientID}/{CourseID}/
├── metadata/
│   ├── case_metadata.xlsx       # DICOM tag extraction
│   ├── processing_log.txt       # Complete processing log
│   ├── config_snapshot.yaml     # Config used for this case
│   └── cropping_metadata.json   # CT cropping boundaries
├── qc_reports/
│   └── qc_summary.json          # Quality control flags
└── ...
```

### Linking Outputs to Inputs

```python
import pandas as pd

# Load metadata with DICOM provenance
metadata = pd.read_excel("Output/Patient001/Course001/metadata/case_metadata.xlsx")

# Key provenance fields
print("Source DICOM UIDs:")
print(f"  CT Series: {metadata['ct_series_uid'].iloc[0]}")
print(f"  RTSTRUCT: {metadata['rtstruct_sop_uid'].iloc[0]}")
print(f"  RTDOSE: {metadata['dose_sop_uid'].iloc[0]}")
print(f"  RTPLAN: {metadata['plan_sop_uid'].iloc[0]}")
```

---

## Reproducibility Validation

### Regression Testing

When updating RTpipeline versions or configurations:

```bash
# Run on reference dataset
rtpipeline run --config reference_config.yaml --input test_data/ --output test_output_v2/

# Compare outputs
python -c "
import pandas as pd
import numpy as np

v1 = pd.read_excel('test_output_v1/_RESULTS/dvh_metrics.xlsx')
v2 = pd.read_excel('test_output_v2/_RESULTS/dvh_metrics.xlsx')

# Numerical columns
numeric_cols = v1.select_dtypes(include=[np.number]).columns

# Compare
diff = (v1[numeric_cols] - v2[numeric_cols]).abs()
max_diff = diff.max().max()
print(f'Maximum absolute difference: {max_diff}')

if max_diff > 1e-6:
    print('WARNING: Outputs differ!')
    print(diff[diff > 1e-6].dropna(how='all'))
"
```

### Hash Verification

For critical analyses, compute checksums:

```python
import hashlib
import pandas as pd

def compute_df_hash(filepath):
    """Compute SHA256 hash of a DataFrame."""
    df = pd.read_excel(filepath)
    return hashlib.sha256(
        pd.util.hash_pandas_object(df).values
    ).hexdigest()

# Record hashes
hashes = {
    "dvh_metrics": compute_df_hash("_RESULTS/dvh_metrics.xlsx"),
    "radiomics_ct": compute_df_hash("_RESULTS/radiomics_ct.xlsx")
}

# Save for reproducibility verification
import json
with open("output_hashes.json", "w") as f:
    json.dump(hashes, f, indent=2)
```

---

## Publication Checklist

Before submitting your manuscript:

### Data Availability Statement

```markdown
**Data Availability**

The RTpipeline configuration files used in this study are available at
[DOI/GitHub URL]. Due to [privacy/IRB/GDPR] restrictions, the raw DICOM
data cannot be shared. Aggregated results tables are provided as
Supplementary Data.
```

### Code Availability Statement

```markdown
**Code Availability**

All data preprocessing was performed using RTpipeline (version X.X,
https://github.com/kstawiski/rtpipeline, DOI: XXX). The complete
configuration bundle, including structure mapping dictionaries and
radiomics parameters, is available at [DOI]. Analysis scripts for
statistical modeling are available at [GitHub URL].
```

### Supplementary Materials to Include

1. **config.yaml** - Complete preprocessing configuration
2. **structure_mapping.yaml** - Structure name harmonization dictionary
3. **radiomics_params.yaml** - PyRadiomics extraction settings
4. **environment.yml** - Conda environment specification
5. **qc_criteria.md** - Quality control inclusion/exclusion criteria

---

## Best Practices Summary

| Area | Recommendation |
|------|---------------|
| **Version Control** | Git-track all configuration files |
| **Environment** | Use Docker or export conda environments |
| **Configuration** | Document all parameters in config files, not code |
| **Provenance** | Save config snapshots with each analysis |
| **QC** | Review and document all exclusions |
| **Methods** | Use templates with specific parameter values |
| **Sharing** | Publish config bundles with DOI |

---

## Further Reading

- [IBSI Standards](https://theibsi.github.io/) - Image Biomarker Standardization Initiative
- [TRIPOD](https://www.tripod-statement.org/) - Reporting guidelines for prediction models
- [CLAIM](https://pubs.rsna.org/doi/10.1148/ryai.2020200029) - Checklist for AI in medical imaging
- Zwanenburg A, et al. (2019) "Assessing robustness of radiomic features by image perturbation." *Sci Rep* 9:614
