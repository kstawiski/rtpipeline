# RTpipeline Manuscript - Tables Plan

## Overview

This document provides detailed specifications for each table in the manuscript.

---

## Table 1: Pipeline Capabilities Comparison

### Purpose
Position RTpipeline against existing tools/pipelines in the radiomics ecosystem.

### Columns
| Column | Description |
|--------|-------------|
| Feature | Capability being compared |
| RTpipeline | Our implementation (✓/✗/Partial) |
| PyRadiomics standalone | Reference tool |
| IBSI Reference | Standard compliance |
| LifeX | Commercial alternative |
| CERR | MATLAB-based alternative |

### Rows (Features to Compare)
1. DICOM CT/MR ingestion
2. RTSTRUCT parsing
3. RTDOSE integration
4. Automated structure naming harmonization
5. AI segmentation (TotalSegmentator)
6. Custom nnU-Net model support
7. DVH extraction
8. IBSI-compliant radiomics
9. Robustness analysis (NTCV)
10. Systematic FOV standardization
11. Multi-center deployment (Docker)
12. Web UI for non-programmers
13. Snakemake workflow management
14. Batch processing (100+ patients)

### Data Source
- Manual research of tool documentation
- RTpipeline config.yaml and Snakefile

---

## Table 2: Example Dataset Characteristics

### Purpose
Describe the validation dataset used in the paper.

### Format
Two-part table: (A) Patient demographics, (B) Technical parameters

### Part A: Patient Demographics

| Patient ID | Treatment Site | Diagnosis | Fractions | Prescription Dose |
|------------|---------------|-----------|-----------|-------------------|
| Patient_001 | Pelvis | Prostate Ca | 39 | 78 Gy |
| Patient_002 | Pelvis | Prostate Ca | 39 | 78 Gy |
| Patient_003 | Pelvis | Prostate Ca | 28 | 70 Gy |
| Patient_004 | Pelvis | Rectal Ca | 25 | 50 Gy |
| Patient_005 | Pelvis | Cervical Ca | 25 | 45 Gy |

### Part B: Technical Parameters

| Parameter | Value/Range |
|-----------|-------------|
| Scanner manufacturer | [From metadata] |
| Slice thickness (mm) | [Range from CT headers] |
| Pixel spacing (mm) | [Range] |
| kVp | [Range] |
| CT series per patient | [Mean ± SD] |
| Structures per RTSTRUCT | [Mean ± SD] |
| Total DICOM files | ~2000 |

### Data Source
- `Example_data/` DICOM metadata extraction
- `case_metadata.xlsx` after pipeline run

---

## Table 3: Structure Harmonization Mapping

### Purpose
Show examples of institutional naming variations mapped to canonical labels.

### Columns
| Institutional Label | Canonical Label | Occurrences (n) | Source |
|---------------------|-----------------|-----------------|--------|
| Rectum_full | RECTUM | 3 | Manual |
| RECT | RECTUM | 1 | Manual |
| rectum | RECTUM | 1 | Manual |
| Bladder | BLADDER | 4 | Manual |
| urinary_bladder | BLADDER | 5 | TotalSegmentator |
| PTV_70Gy | PTV_HIGH | 3 | Manual |
| PTV_boost | PTV_HIGH | 2 | Manual |
| femur_left | FEMUR_L | 5 | TotalSegmentator |

### Data Source
- `config.yaml` structure mapping rules
- Pipeline output RS_custom.dcm structure names

---

## Table 4: DVH Metrics Summary

### Purpose
Present extracted dose-volume metrics across all structures and patients.

### Format
Structure-wise summary with statistical aggregation

### Columns
| Structure | N | Dmean (Gy) | D95 (Gy) | D2 (Gy) | V20Gy (%) | V40Gy (%) | Volume (cc) |
|-----------|---|------------|----------|---------|-----------|-----------|-------------|

### Rows
- GTV (if present)
- CTV
- PTV_HIGH
- PTV_LOW (if present)
- RECTUM
- BLADDER
- FEMUR_L
- FEMUR_R
- BOWEL_BAG (custom)
- Body

### Statistics Format
- Mean ± SD for continuous variables
- Median [IQR] for skewed distributions

### Data Source
- `dvh_metrics.xlsx` from pipeline output
- Aggregated `_RESULTS/dvh_metrics.xlsx`

---

## Table 5: Radiomics Feature Classes

### Purpose
Summarize the radiomics features extracted by the pipeline.

### Columns
| Feature Class | N Features | Examples | PyRadiomics Module |
|---------------|------------|----------|-------------------|
| Shape (3D) | 14 | Volume, Surface Area, Sphericity | shape |
| First-Order | 18 | Mean, Median, Energy, Entropy | firstorder |
| GLCM | 24 | Contrast, Correlation, Homogeneity | glcm |
| GLRLM | 16 | SRE, LRE, GLNU, RLNU | glrlm |
| GLSZM | 16 | SAE, LAE, SZNU | glszm |
| GLDM | 14 | DNU, DNN, GLV | gldm |
| NGTDM | 5 | Coarseness, Contrast, Busyness | ngtdm |
| **Total** | **107** | | |

### Notes
- Add row for filtered features if wavelet/LoG applied
- Reference IBSI feature definitions

### Data Source
- PyRadiomics documentation
- `radiomics_ct.xlsx` column headers

---

## Table 6: Robustness Analysis Results (KEY TABLE)

### Purpose
Main results table showing feature robustness across NTCV perturbations.

### Format
Multi-panel table with summary statistics

### Panel A: Overall Robustness Summary

| Metric | All Features | Shape | First-Order | Texture |
|--------|--------------|-------|-------------|---------|
| N features | 107 | 14 | 18 | 75 |
| Mean ICC | X.XX | X.XX | X.XX | X.XX |
| Median ICC | X.XX | X.XX | X.XX | X.XX |
| % Robust (ICC ≥ 0.90) | XX% | XX% | XX% | XX% |
| % Acceptable (ICC ≥ 0.75) | XX% | XX% | XX% | XX% |
| Mean CoV (%) | X.X | X.X | X.X | X.X |

### Panel B: Robustness by Structure

| Structure | N Features | Mean ICC | % Robust | Mean CoV (%) |
|-----------|------------|----------|----------|--------------|
| RECTUM | 107 | X.XX | XX% | X.X |
| BLADDER | 107 | X.XX | XX% | X.X |
| PTV | 107 | X.XX | XX% | X.X |
| FEMUR | 107 | X.XX | XX% | X.X |
| BOWEL_BAG | 107 | X.XX | XX% | X.X |

### Panel C: Top 10 Most Robust Features

| Rank | Feature Name | Class | Mean ICC | CoV (%) |
|------|--------------|-------|----------|---------|
| 1 | original_shape_VoxelVolume | Shape | 0.99 | 2.1 |
| 2 | original_shape_SurfaceArea | Shape | 0.98 | 3.4 |
| ... | ... | ... | ... | ... |

### Panel D: Top 10 Least Robust Features

| Rank | Feature Name | Class | Mean ICC | CoV (%) |
|------|--------------|-------|----------|---------|
| 1 | original_glcm_ClusterShade | GLCM | 0.45 | 45.2 |
| ... | ... | ... | ... | ... |

### Data Source
- `radiomics_robustness_summary.xlsx`
- `perturbation_results.parquet`

---

## Table 7: CT Cropping Impact Analysis

### Purpose
Quantify the effect of systematic FOV standardization on metrics.

### Columns
| Metric | Pre-Crop | Post-Crop | Δ (%) | p-value |
|--------|----------|-----------|-------|---------|

### Rows (Metrics to Compare)
**DVH Metrics:**
- V20Gy_body (%)
- V30Gy_body (%)
- Dmean_body (Gy)

**Radiomics Features:**
- original_firstorder_Energy
- original_firstorder_Entropy
- original_shape_VoxelVolume
- original_glcm_Contrast

### Statistical Test
- Paired Wilcoxon signed-rank test (n=5)

### Data Source
- Pipeline run with `ct_cropping.enabled: false`
- Pipeline run with `ct_cropping.enabled: true`
- Compare output metrics

---

## Table 8: Perturbation Parameter Settings

### Purpose
Document the NTCV perturbation parameters used in robustness analysis.

### Columns
| Perturbation Type | Parameter | Value | Rationale/Reference |
|-------------------|-----------|-------|---------------------|
| **Noise (N)** | | | |
| | Distribution | Gaussian | Scanner noise model |
| | σ (HU) | 10, 20 (0 = baseline) | Typical CT noise level (Zwanenburg 2019) |
| | N samples | 3 | Computational balance |
| **Translation (T)** | | | |
| | Direction | Axial plane (x, y) | Patient positioning uncertainty |
| | Distance (mm) | ±2, ±4 | Setup margins (clinical) |
| | N samples | 4 | |
| **Contour (C)** | | | |
| | Method | Morphological operations | Inter-observer variation |
| | Kernel size (mm) | 1-2 | Typical contouring uncertainty |
| | N samples | 3 | |
| **Volume (V)** | | | |
| | Method | Morphological erosion/dilation | Segmentation uncertainty |
| | Volume change | ±15% | Clinical variability range |
| | N samples | 3 | |
| **Combined (NTCV-full)** | | | |
| | Application | Simultaneous (compound) | Tests feature stability under combined uncertainty |
| | Total perturbations | N×T×C×V | Full factorial design |
| | N samples | 108 | 3×4×3×3 |

**NOTE:** "NTCV-full" applies all perturbation types simultaneously in one pass,
not as separate isolated analyses. This tests feature stability under realistic
compound uncertainty scenarios.

### Data Source
- `config.yaml` radiomics_robustness section
- Zwanenburg et al., 2019 methodology

---

## Table 9: Processing Performance Metrics

### Purpose
Report computational performance for reproducibility.

### Columns
| Stage | Time per Patient | Total Time (n=5) | Hardware |
|-------|------------------|------------------|----------|
| DICOM organization | X min | X min | CPU |
| TotalSegmentator | X min | X min | GPU |
| Custom segmentation | X min | X min | GPU |
| DVH extraction | X min | X min | CPU |
| Radiomics (base) | X min | X min | CPU |
| Radiomics (robustness) | X min | X min | CPU |
| **Total** | **X min** | **X min** | |

### Hardware Specification
- CPU: [Model, cores]
- GPU: [Model, VRAM]
- RAM: [Size]
- Storage: [Type, speed]

### Data Source
- Snakemake benchmark files
- Docker container logs

---

## Supplementary Tables

### Table S1: Full Feature List with Robustness Metrics

All 107 features with ICC, CoV, and robustness classification.

### Table S2: IBSI Phantom Validation Results

| Feature | IBSI Reference | RTpipeline Value | Difference (%) | Status |
|---------|----------------|------------------|----------------|--------|

### Table S3: Structure-wise DVH Metrics (All Patients)

Full patient-by-patient breakdown of DVH metrics.

### Table S4: Custom Structure Definitions

Boolean operations used to create composite structures.

---

## Table Generation Scripts

```
Manuscript/
├── scripts/
│   ├── generate_tables.py       # Main table generation script
│   ├── table_styles.py          # Formatting utilities
│   └── latex_export.py          # LaTeX table export
├── tables/
│   ├── table1_comparison.xlsx
│   ├── table2_demographics.xlsx
│   ├── table3_harmonization.xlsx
│   ├── table4_dvh.xlsx
│   ├── table5_features.xlsx
│   ├── table6_robustness.xlsx
│   ├── table7_cropping.xlsx
│   ├── table8_parameters.xlsx
│   ├── table9_performance.xlsx
│   └── supplementary/
│       ├── tableS1_full_features.xlsx
│       ├── tableS2_ibsi.xlsx
│       └── ...
```

---

## Code Template for Table Generation

```python
import pandas as pd
import numpy as np

def generate_robustness_summary_table(robustness_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 6: Robustness Analysis Results."""

    # Panel A: Overall summary
    summary = {
        'Metric': ['N features', 'Mean ICC', 'Median ICC',
                   '% Robust (ICC ≥ 0.90)', '% Acceptable (ICC ≥ 0.75)',
                   'Mean CoV (%)'],
        'All Features': [
            len(robustness_df),
            robustness_df['icc'].mean(),
            robustness_df['icc'].median(),
            (robustness_df['icc'] >= 0.90).mean() * 100,
            (robustness_df['icc'] >= 0.75).mean() * 100,
            robustness_df['cov_pct'].mean()
        ]
    }

    # Add columns for feature classes
    for feature_class in ['Shape', 'First-Order', 'Texture']:
        class_df = robustness_df[robustness_df['feature_class'] == feature_class]
        summary[feature_class] = [
            len(class_df),
            class_df['icc'].mean(),
            class_df['icc'].median(),
            (class_df['icc'] >= 0.90).mean() * 100,
            (class_df['icc'] >= 0.75).mean() * 100,
            class_df['cov_pct'].mean()
        ]

    return pd.DataFrame(summary)


def generate_top_features_table(robustness_df: pd.DataFrame, n: int = 10,
                                ascending: bool = False) -> pd.DataFrame:
    """Generate top/bottom N features by ICC."""

    sorted_df = robustness_df.sort_values('icc', ascending=ascending)
    top_n = sorted_df.head(n).copy()
    top_n['Rank'] = range(1, n + 1)

    return top_n[['Rank', 'feature_name', 'feature_class', 'icc', 'cov_pct']]
```

---

*Tables plan for RTpipeline CMPB manuscript.*
