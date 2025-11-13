# Radiomics Robustness Module

A built-in feature stability assessment module for rtpipeline that evaluates radiomics features under segmentation perturbations following IBSI guidelines and state-of-the-art methods.

## Overview

The radiomics robustness module helps you identify stable, reproducible radiomics features for modeling by:

1. **Generating perturbed segmentations** via mask erosion/dilation (volume adaptation)
2. **Re-extracting radiomics features** for each perturbation using PyRadiomics
3. **Computing robustness metrics**: ICC (Intraclass Correlation Coefficient), CoV (Coefficient of Variation), QCD (Quartile Coefficient of Dispersion)
4. **Classifying features** as "robust", "acceptable", or "poor" based on configurable thresholds

## Scientific Background

This implementation follows recent literature on radiomics reproducibility:

- **Zwanenburg et al. (2019)** - Sci Rep: Image perturbation chains with ICC thresholds
- **Lo Iacono et al. (2024)** - SpringerLink: Volume adaptation method (±15% for stability)
- **Poirot et al. (2022)** - Sci Rep: Multi-method ICC using Pingouin
- **IBSI (Image Biomarker Standardization Initiative)**: Standardized feature definitions

### Robustness Thresholds

Based on published recommendations:

**ICC (Intraclass Correlation Coefficient):**
- ICC ≥ 0.90: **Robust** / Excellent
- 0.75 ≤ ICC < 0.90: **Acceptable** / Good
- ICC < 0.75: **Poor**

**CoV (Coefficient of Variation):**
- CoV ≤ 10%: **Robust**
- 10% < CoV ≤ 20%: **Acceptable**
- CoV > 20%: **Poor**

A feature is classified as "robust" if it meets **both** ICC and CoV robust thresholds.

## Quick Start

### 1. Enable in config.yaml

```yaml
radiomics_robustness:
  enabled: true
  modes:
    - segmentation_perturbation

  segmentation_perturbation:
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "BLADDER"
      - "RECTUM"
    small_volume_changes: [-0.15, 0.0, 0.15]  # ±15% volume change
```

### 2. Install dependencies

```bash
pip install pingouin  # Required for ICC computation
# or install with radiomics extras:
pip install -e ".[radiomics]"
```

### 3. Run with Snakemake

```bash
snakemake -c 8
```

The robustness analysis runs after standard radiomics extraction and produces:
- Per-course results: `Data_Snakemake/{patient}/{course}/radiomics_robustness_ct.parquet`
- Aggregated summary: `Data_Snakemake/_RESULTS/radiomics_robustness_summary.xlsx`

### 4. Use robust features for modeling

The output Excel file contains multiple sheets:

- **`global_summary`**: Features averaged across all structures/courses
- **`robust_features`**: Only features classified as "robust" (ICC ≥ 0.90, CoV ≤ 10%)
- **`acceptable_features`**: Features meeting "acceptable" thresholds (ICC ≥ 0.75, CoV ≤ 20%)
- **`per_structure`**: Detailed per-structure breakdown

**Example workflow:**
```python
import pandas as pd

# Load results
summary = pd.read_excel("Data_Snakemake/_RESULTS/radiomics_robustness_summary.xlsx",
                        sheet_name="robust_features")

# Filter radiomics data to robust features only
robust_feature_names = summary["feature_name"].tolist()
radiomics = pd.read_excel("Data_Snakemake/_RESULTS/radiomics_ct.xlsx")
radiomics_robust = radiomics[radiomics.columns.intersection(robust_feature_names)]

# Use radiomics_robust for modeling
```

## CLI Commands

### Per-course analysis

```bash
rtpipeline radiomics-robustness \
  --course-dir Data_Snakemake/Patient001/Course001 \
  --config config.yaml \
  --output radiomics_robustness_ct.parquet
```

### Aggregate results

```bash
rtpipeline radiomics-robustness-aggregate \
  --inputs Data_Snakemake/*/*/radiomics_robustness_ct.parquet \
  --output radiomics_robustness_summary.xlsx \
  --config config.yaml
```

## Configuration Reference

### Full config.yaml example

```yaml
radiomics_robustness:
  enabled: true

  modes:
    - segmentation_perturbation  # Currently supported
    # - segmentation_method      # Future: compare manual vs auto segmentation
    # - scan_rescan              # Future: test-retest reliability

  segmentation_perturbation:
    # Structures to analyze (wildcards supported)
    apply_to_structures:
      - "GTV*"      # Matches GTV, GTV_primary, GTV_node, etc.
      - "CTV*"
      - "PTV*"
      - "BLADDER"
      - "RECTUM"
      - "PROSTATE"

    # Volume changes (τ parameter from Lo Iacono 2024)
    small_volume_changes: [-0.15, 0.0, 0.15]   # ±15% volume change
    large_volume_changes: [-0.30, 0.0, 0.30]   # ±30% for stress testing

    # Advanced perturbations (future features)
    n_random_contour_realizations: 0
    max_translation_mm: 0.0

  metrics:
    icc:
      implementation: "pingouin"  # ICC computation library
      icc_type: "ICC2"            # ICC(2,1): two-way mixed, absolute agreement
      ci: true                    # Compute 95% confidence intervals
    cov:
      enabled: true
    qcd:
      enabled: true

  thresholds:
    icc:
      robust: 0.90
      acceptable: 0.75
    cov:
      robust_pct: 10.0
      acceptable_pct: 20.0
```

## Output Format

### Per-course parquet file

Columns:
- `structure`: ROI name (e.g., "BLADDER", "GTV_primary")
- `feature_name`: PyRadiomics feature (e.g., "original_glcm_Correlation")
- `n_perturbations`: Number of perturbations tested
- `icc`: ICC point estimate
- `icc_ci95_low`, `icc_ci95_high`: 95% confidence interval
- `cov_pct`: Coefficient of Variation (%)
- `qcd`: Quartile Coefficient of Dispersion
- `robustness_label`: "robust", "acceptable", or "poor"
- `pass_seg_perturb`: Boolean (True if robust or acceptable)

### Aggregated Excel file

Multiple sheets for easy filtering:
1. **global_summary**: Features averaged across all structures/courses
2. **per_structure**: Detailed per-structure results
3. **robust_features**: Only ICC ≥ 0.90 and CoV ≤ 10%
4. **acceptable_features**: ICC ≥ 0.75 and CoV ≤ 20%

## Best Practices

1. **Feature selection for modeling**: Use only "robust" features (ICC ≥ 0.90, CoV ≤ 10%) for predictive models
2. **Multi-center studies**: Consider "acceptable" features (ICC ≥ 0.75, CoV ≤ 20%) if data scarcity requires it
3. **Structure selection**: Focus on clinically relevant structures (GTV, CTV, organs at risk)
4. **Volume changes**: Start with ±15% (default); use ±30% for stress testing
5. **Validation**: If possible, validate feature stability on independent test-retest data

## Typical Feature Families by Robustness

Based on literature (your summary and Azimi 2025):

**Generally Robust:**
- Shape features (Volume, SurfaceArea, Sphericity)
- First-order statistics (Mean, Median, Energy)
- Some GLDM features

**Moderately Robust:**
- GLCM features (with appropriate normalization)
- GLRLM features

**Often Fragile:**
- High-order texture features without careful preprocessing
- Features from very small ROIs
- Features sensitive to discretization

## Troubleshooting

### "Radiomics robustness is disabled"
Enable it in `config.yaml`: `radiomics_robustness.enabled: true`

### "Pingouin not available"
Install: `pip install pingouin>=0.5.3`

### "No structures matched robustness patterns"
Check `apply_to_structures` patterns in config. Verify structure names in `RS_auto.dcm` match your patterns.

### "Insufficient perturbations for X; skipping"
Some small structures may fail erosion/dilation. This is expected; they'll be skipped automatically.

## References

1. Zwanenburg A, et al. (2019). "Assessing robustness of radiomic features by image perturbation." *Scientific Reports* 9:3418. DOI: 10.1038/s41598-019-41766-y
2. Lo Iacono F, et al. (2024). "A Novel Data Augmentation Method for Radiomics Analysis Using Image Perturbations." *J Imaging Informatics in Medicine*. DOI: 10.1007/s10278-024-01013-0
3. Poirot MG, et al. (2022). "Robustness of radiomics to variations in segmentation methods in multimodal brain MRI." *Scientific Reports* 12:16712.
4. Hosseini SA, et al. (2025). "Robust vs. Non-robust radiomic features." *Cancer Imaging* 25:6.
5. Zwanenburg A, et al. (2020). "The Image Biomarker Standardization Initiative (IBSI)." *Radiology* 295(2):328-338.

## Future Enhancements

Planned for future releases:
- **Segmentation-method robustness**: Compare Manual vs TotalSegmentator vs custom models
- **Scan-rescan ICC**: Test-retest reliability for longitudinal studies
- **Contour randomization**: Boundary noise simulation
- **Translation perturbations**: Small rigid shifts (±3-5 mm)
- **Panel-averaged features**: Zwanenburg-style feature averaging across perturbations
