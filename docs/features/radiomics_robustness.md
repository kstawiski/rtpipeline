# Radiomics Robustness Module

A built-in feature-stability module that evaluates radiomics features with an
RTpipeline-adapted NTCV chain inspired by Zwanenburg et al. (2019). It is not an
exact reimplementation of that study.[^zwanenburg2019]

## Overview

The radiomics robustness module helps you identify stable, reproducible radiomics features for modeling by:

1. **Collecting segmentations** from multiple sources:
   - TotalSegmentator (RS_auto.dcm)
   - Custom structures (RS_custom.dcm)
   - Custom models (Segmentation_{model_name}/rtstruct.dcm)
2. **Generating systematic perturbations** via RTpipeline's NTCV chain (Noise + Translation + Contour + Volume):[^zwanenburg2019]
   - **N**: Image noise injection (Gaussian noise in HU)
   - **T**: Rigid translations (shipped maximum +/-4 mm)
   - **C**: Reproducible random inward/outward physical-space boundary offsets
   - **V**: Distance-ranked adaptation to the closest voxel count representing ±15-30% volume change
3. **Re-extracting radiomics features** for each perturbation using PyRadiomics
4. **Computing robustness metrics**: ICC(3,1) with analytical 95% CIs (via Pingouin[^pingouin]), plus the cohort median of within-subject CoV and QCD.[^koo2016]
5. **Classifying features** as "robust", "acceptable", or "poor" based on configurable thresholds informed by radiomics reproducibility literature.[^koo2016]

## Scientific Background

This implementation is inspired by the perturbation framework from Zwanenburg et al. (2019)[^zwanenburg2019] and related radiomics reproducibility research:

- **Literature-validated perturbation chains:** In the original Zwanenburg et al. (2019) test–retest validation, NTCV and RCV combinations detected ~98–99% of unstable features with <2% false positives in their specific datasets.[^zwanenburg2019] **Important:** These operating characteristics were established on specific test–retest datasets with particular parameter configurations. rtpipeline currently implements an **NTCV-like perturbation chain only** (RCV is not implemented) and has not independently validated these exact figures—users should treat them as **literature benchmarks**, not performance guarantees.
- **Volume adaptation:** A physical-distance ranking selects the exact rounded target voxel count. The achieved change is therefore the closest representable value to the configured target.
- **Reliability statistics:** ICC(3,1) with analytical confidence intervals and complementary within-subject CoV thresholds provide configurable research filters.[^koo2016]
- **Standardization:** The IBSI initiative[^ibsi2020] provides standardized definitions for radiomics features and recommends reporting perturbation details and preprocessing provenance.

### Robustness Thresholds (Configurable Defaults)

The default thresholds in rtpipeline are informed by published recommendations, particularly Koo & Li (2016)[^koo2016] for ICC interpretation and common practices in CT radiomics reproducibility studies:

**ICC (Intraclass Correlation Coefficient):**
Following Koo & Li (2016) qualitative descriptors:
- ICC ≥ 0.90: **Excellent** — used as "robust" threshold in rtpipeline (conservative choice)
- 0.75 ≤ ICC < 0.90: **Good** — used as "acceptable" threshold (standard)
- 0.50 ≤ ICC < 0.75: **Moderate** — not recommended for clinical modeling
- ICC < 0.50: **Poor**

**CoV (Coefficient of Variation):**
CoV is first computed within each patient/course/ROI/source as 100 × (standard deviation / absolute mean) across perturbations. The reported `cov_pct` is the cohort median of those subject-level CoVs; Q1 and Q3 are also reported. This prevents between-patient biological variation from being mistaken for perturbation instability. rtpipeline defaults to CoV ≤10% as "robust" based on thresholds frequently reported in CT radiomics reproducibility studies:
- CoV ≤ 10%: **Robust** (conservative default threshold)
- 10% < CoV ≤ 20%: **Acceptable**
- CoV > 20%: **Poor**

*Note: For features with means close to zero, CoV can become numerically unstable; users should manually review features with extreme CoV values.*

**Important:** These are configurable defaults inspired by published conventions, not consensus standards. Users should adapt thresholds to their specific clinical and statistical context.

*For highly conservative applications (delta-radiomics, adaptive RT), users may configure a stricter CoV ≤5% threshold via `config.yaml`:*
```yaml
radiomics_robustness:
  thresholds:
    cov:
      robust_pct: 5.0  # Override default 10%
```

A feature is classified as "robust" if it meets **both** ICC and CoV thresholds. These are conventions commonly used in the radiomics literature, not universal standards; users should adapt thresholds to their specific clinical context.

### Perturbation Intensity Levels

The module supports three intensity levels to balance computational cost and thoroughness. Typical use cases are annotated with recommended metrics:

- **mild**: 12 perturbations with the shipped defaults (QA spot checks)
- **standard**: 81 perturbations with the shipped defaults (complete standard NTCV chain)
- **aggressive**: 315 perturbations with the shipped defaults (computational stress test)

### Perturbation Parameter Defaults

The following table summarizes the default perturbation parameters used by rtpipeline:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **N (Noise)** | `[0.0, 10.0, 20.0]` HU | Gaussian noise standard deviations |
| **T (Translation)** | `4.0` mm | Maximum rigid shift (up to +/-4 mm) |
| **C (Contour)** | `2` realizations | Boundary randomizations |
| **V (Volume)** | `[-0.15, 0.0, 0.15]` | Volume change ratios (±15% erosion/dilation) |
| **Intensity** | `"standard"` | Controls perturbation count: mild/standard/aggressive |

**Algorithm details:**

- **Volume adaptation**: Voxels are ranked with a spacing-aware signed distance map. Erosion retains the deepest interior voxels and dilation adds the nearest exterior voxels until the exact rounded target voxel count is reached.
- **Contour randomization**: Reproducible random inward or outward offsets between 50% and 100% of `contour_randomization_mm` (or `max_translation_mm / 2` when unset), applied in physical space. Duplicate/original realizations fail closed.
- **Translation**: Rigid shifts applied via SimpleITK `ResampleImageFilter` with nearest-neighbor interpolation

### Reproducibility: Random Seed

For deterministic, reproducible perturbations, rtpipeline uses a fixed random seed:

```python
np.random.Generator(np.random.PCG64(42 + perturbation_count))
```

This ensures that the same configuration produces identical perturbation sequences across runs. The seed is incremented per perturbation to ensure variation while maintaining reproducibility. Currently, this seed is not user-configurable; for different seed values, modify the source code directly.

## Quick Start

### 1. Enable in config.yaml

**Standard configuration (complete NTCV chain):**
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
    max_translation_mm: 4.0
    n_random_contour_realizations: 2
    noise_levels: [0.0, 10.0, 20.0]
    intensity: "standard"  # Options: mild, standard, aggressive
```

**Advanced NTCV configuration (comprehensive testing):**
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
    
    # Volume perturbations (V in NTCV)
    small_volume_changes: [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    large_volume_changes: [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    
    # Translation perturbations (T in NTCV)
    max_translation_mm: 4.0  # Up to +/-4 mm rigid shifts
    
    # Contour randomization (C in NTCV)
    n_random_contour_realizations: 3  # 3 boundary randomization variants
    
    # Image noise injection (N in NTCV)
    noise_levels: [0.0, 10.0, 20.0]  # Gaussian noise std dev in HU
    
    # Perturbation intensity
    intensity: "aggressive"  # Shipped grids: mild 12, standard 81, aggressive 315
```

### 2. Install dependencies

```bash
pip install pingouin  # Required for ICC computation
# RTpipeline intentionally does not expose a combined radiomics extra because
# the main NumPy 2.x and PyRadiomics NumPy 1.26 runtimes must remain separate.
# Use scripts/install_local.sh for the supported two-environment installation.
```

### 3. Run with Snakemake

```bash
snakemake -c 8
```

The robustness analysis runs after standard radiomics extraction and produces:
- Per-course results: `Data_Snakemake/{patient}/{course}/radiomics_robustness_ct.parquet` containing all perturbation-level feature values
- Aggregated summary: `Data_Snakemake/_RESULTS/radiomics_robustness_summary.xlsx`

### 4. Use robust features for modeling

The output Excel file contains multiple sheets:

- **`global_summary`**: Backward-compatible primary summary; each row remains specific to one structure and segmentation source (values are never pooled across heterogeneous ROIs)
- **`robust_features`**: Structure/source/feature rows classified as robust (ICC lower bound or estimate ≥ 0.90 and median within-subject CoV ≤ 10%)
- **`acceptable_features`**: Rows meeting acceptable thresholds (ICC lower bound or estimate ≥ 0.75 and median within-subject CoV ≤ 20%)
- **`per_source_summary`** *(if available)*: Source/structure-specific cohort metrics
- **`per_structure_source`**: Detailed per-structure breakdown (preserving segmentation source)
- **`radiomics_robustness_summary_raw_values.parquet`**: Separate raw perturbation-level values used for the cohort statistics (not an Excel sheet)
- **`robust_features_per_source`** *(if available)*: Robust features for each segmentation source

**Example workflow:**
```python
import pandas as pd

# Load results
summary = pd.read_excel("Data_Snakemake/_RESULTS/radiomics_robustness_summary.xlsx",
                        sheet_name="robust_features")

# Select the exact ROI/source used by the downstream model. Classifications are
# not interchangeable across heterogeneous structures or segmentation sources.
summary = summary[
    (summary["structure"] == "GTV_primary")
    & (summary["segmentation_source"] == "Custom")
]
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

    # V: Volume changes (τ parameter from Lo Iacono 2024)
    small_volume_changes: [-0.15, 0.0, 0.15]   # ±15% volume change (standard)
    large_volume_changes: [-0.30, 0.0, 0.30]   # ±30% for stress testing

    # T: Translation perturbations (mm)
    # Rigid shifts in x, y, z directions to simulate positioning uncertainty
    max_translation_mm: 4.0  # Set to 0.0 to disable

    # C: Contour randomization
    # Simulates inter-observer variability in delineation
    n_random_contour_realizations: 2  # Set to 0 to disable

    # N: Noise injection (HU)
    # Gaussian noise to simulate scanner variability
    noise_levels: [0.0, 10.0, 20.0]  # Standard deviations in HU

    # Perturbation intensity (controls total perturbation count)
    # - "mild": 12 perturbations per ROI with shipped grids
    # - "standard": 81 perturbations per ROI with shipped grids
    # - "aggressive": 315 perturbations per ROI with shipped grids
    intensity: "standard"

  metrics:
    icc:
      implementation: "pingouin"  # ICC computation library
      icc_type: "ICC3"            # ICC(3,1): two-way mixed, consistency for fixed perturbations
      ci: true                    # Compute 95% confidence intervals
    cov:
      enabled: true
    qcd:
      enabled: true    # QCD = (Q3 - Q1) / (Q3 + Q1), a robust dispersion measure

  thresholds:
    icc:
      robust: 0.90       # Conservative clinical threshold (2023-2025 research)
      acceptable: 0.75   # Standard threshold (Zwanenburg 2019)
    cov:
      robust_pct: 10.0      # CoV ≤ 10%: "robust" (standard)
      acceptable_pct: 20.0  # CoV ≤ 20%: "acceptable"
```

### ICC Implementation Details

rtpipeline computes ICC using Pingouin's `intraclass_corr` function[^pingouin] with the following configuration:

- **ICC Type:** ICC(3,1) — two-way mixed effects model, single measurement, consistency agreement
- **Subject encoding:** Each unique combination of `patient_id + course_id + structure + segmentation_source` is treated as one "subject" (so repeated courses for the same patient are distinct subjects)
- **Rater encoding:** Each `perturbation_id` (e.g., `ntcv_n10_t1_0_0_c0_v-0.15`) is treated as a "rater"
- **Confidence intervals:** Pingouin's analytical 95% CIs (not bootstrap-based)

This design treats perturbations as fixed "raters" of the same underlying subject and uses ICC(3,1) (two-way mixed, single, consistency), following the fixed-rater rationale described by Koo & Li (2016).[^koo2016] **This is an adaptation of ICC to perturbation-based robustness analysis rather than a standard inter-rater scenario.** The ICC(3,1) model was chosen because:

1. **Fixed perturbations:** The perturbation set is chosen by the researcher, not randomly sampled from a population of possible raters
2. **Single measurements:** Each perturbation produces one feature value per ROI
3. **Consistency:** We measure relative agreement, not absolute agreement (small systematic offsets are acceptable)

**Methodological caveat:** This interpretation is one reasonable choice for perturbation-based robustness analysis, but it is not universally standardized. Other ICC formulations (e.g., absolute-agreement models or ICC(2,1)) could also be justified depending on study design. Users should verify that ICC(3,1) aligns with their specific reliability framework and may adjust the `icc_type` configuration if needed.

**Conservative thresholding:** When 95% confidence intervals are available, rtpipeline uses the **lower CI bound** for robustness classification rather than the point estimate. This means a feature is classified as "robust" only if `ICC_CI95_lower ≥ 0.90`. This conservative approach reduces false positives in robustness labeling but may exclude borderline features. If the CI is unavailable, the point estimate is used directly.

**Sample size note:** ICC estimates derived from small numbers of perturbations (e.g., <10) or small cohorts (<20 subjects) may be unstable with wide confidence intervals. Users should ensure sufficient perturbations (typically ≥10–15 per ROI) for reliable ICC estimation.

**Example data structure for ICC computation:**
```
| subject                          | rater                    | feature_value |
|----------------------------------|--------------------------|---------------|
| patient001_course1_bladder_auto  | ntcv_n0_t0_0_0_c0_v0.0   | 0.456         |
| patient001_course1_bladder_auto  | ntcv_n10_t0_0_0_c0_v0.0  | 0.461         |
| patient001_course1_bladder_auto  | ntcv_n0_t1_0_0_c0_v-0.15 | 0.448         |
| ...                              | ...                      | ...           |
```

### NTCV Perturbation Chain Explanation

The NTCV chain follows the systematic perturbation methodology from Zwanenburg et al. (2019).[^zwanenburg2019] The order (Noise → Translation → Contour → Volume) ensures proper propagation of uncertainty sources:

1. **N (Noise)**: Adds Gaussian noise to the CT image
   - Simulates scanner variability and acquisition differences
   - Applied to the *image*, not the mask
   - Typical values: 0, 10, 20 HU std dev

2. **T (Translation)**: Applies rigid geometric shifts
   - Simulates positioning uncertainty and registration errors
   - Applied *before* contour randomization
   - The shipped standard grid uses 0 and +/-4 mm superior-inferior shifts;
     aggressive intensity adds +/-4 mm shifts on each axis

3. **C (Contour)**: Randomizes ROI boundaries
   - Simulates inter-observer delineation variability
   - Applied to the already-shifted ROI
   - Uses boundary noise simulation

4. **V (Volume)**: Systematic distance-ranked volume adaptation
   - Final physical-space boundary adjustment to an exact rounded voxel count
   - Tests feature stability across ROI size variations
   - Typical values: ±15% (standard) or ±30% (stress testing)

**Total perturbations** = N_noise × N_translation × N_contour × N_volume

With the shipped grids, standard intensity produces 3 noise states × 3
translation states × 3 contour states (original plus two realizations) × 3
volume states = 81 perturbations. Aggressive intensity uses 7 translation
states and 5 unique volume states, producing 315 perturbations. The module does
not subsample these Cartesian grids.

## Output Format

### Per-course parquet file (raw long-form values)

Columns:
- `patient_id`, `course_id`: Subject/course identifiers
- `structure`: ROI name (e.g., "BLADDER", "GTV_primary")
- `segmentation_source`: Source of segmentation (e.g., "AutoRTS_total", "Custom", "CustomModel:cardiac_STOPSTORM")
- `perturbation_id`: Exact NTCV state identifier
- `feature_name`: PyRadiomics feature (e.g., "original_glcm_Correlation")
- `value`: Scalar feature value

The course command fails and writes no parquet if any configured perturbation
or feature extraction is missing, times out, is non-finite, or has a different
feature set. Cohort statistics are created only by the aggregate command.

### Aggregated Excel file

Multiple sheets for easy filtering:
1. **global_summary**: Backward-compatible primary table with one row per structure/source/feature; no cross-ROI pooling
2. **per_source_summary**: Source/structure/feature rows (present when source metadata exists)
3. **per_structure_source**: Detailed per-structure/source/feature rows
4. **robust_features**: Structure/source/feature rows meeting ICC ≥ 0.90 and median within-subject CoV ≤ 10%
5. **acceptable_features**: Structure/source/feature rows meeting ICC ≥ 0.75 and median within-subject CoV ≤ 20%
6. **robust_features_per_source**: Robust source/structure/feature rows

Summary columns include `n_subjects_complete`, `n_subjects_dropped`,
`n_perturbations`, ICC and its 95% CI, median within-subject `cov_pct` and
`qcd`, their Q1/Q3 columns, and the classification. Raw values are stored next
to the workbook as `radiomics_robustness_summary_raw_values.parquet`; they are
not copied into an Excel sheet.

## Best Practices

### Feature Selection Strategy (Based on 2023-2025 Research)

1. **Conservative research filter**: Consider features meeting ICC ≥ 0.90 and median within-subject CoV ≤ 10%
   - This threshold is a configurable convention, not evidence of clinical validity
   - A substantial proportion of features can meet this threshold, though the exact fraction is highly dataset- and protocol-dependent

2. **Multi-center studies**: Consider "acceptable" features (ICC ≥ 0.75, CoV ≤ 20%) if data scarcity requires it
   - Standard threshold from Zwanenburg 2019
   - Suitable for exploratory analysis or hypothesis generation

3. **Cost-effective alternative**: Perturbation-based stability analysis provides a practical alternative to expensive test-retest imaging
   - In the original Zwanenburg et al. (2019) study, NTCV-like chains detected ~98–99% of unstable features with <2% false positives on their specific test-retest datasets
   - These are **literature benchmarks**; users should not assume identical performance on their data

4. **Perturbation intensity selection**:
   - **mild**: Quick screening, pilot studies
   - **standard**: Complete shipped research grid
   - **aggressive**: Larger computational stress-test grid

5. **Shipped standard NTCV configuration for manuscript analyses**:
   ```yaml
   small_volume_changes: [-0.15, 0.0, 0.15]
   max_translation_mm: 4.0
   n_random_contour_realizations: 2
   noise_levels: [0.0, 10.0, 20.0]
   intensity: "standard"
   ```
   This requires 81 perturbations per ROI. The course fails closed if any state
   cannot be generated or extracted.

### Structure and Modality Considerations

1. **Structure selection**: Focus on clinically relevant structures (GTV, CTV, organs at risk)
2. **Volume thresholds**: Features from very small ROIs are often unstable regardless of type
3. **Multi-source analysis**: Compare robustness across segmentation sources (TotalSegmentator vs Custom structures)
4. **CT-specific**: Current implementation optimized for CT-based radiomics in bladder, prostate, and rectal cancer
5. **Validation**: If possible, validate feature stability on independent test-retest data

### Advanced Considerations

1. **Harmonization**: For multi-scanner studies, consider CovBat harmonization (outperforms traditional ComBat)
   - Note: CovBat implementation not included in this module - apply as preprocessing
   
2. **Feature selection methods**: Consider stability-aware approaches like Graph-FS that maintain performance across institutions
   - Combines stability metrics with predictive performance
   
3. **Discretization**: Ensure consistent bin width (HU) or bin count across all perturbations
   - PyRadiomics settings controlled via radiomics_params.yaml

4. **Preprocessing chains**: Robustness to resampling and discretization variations important for multi-center studies
   - Test with different voxel sizes if data will come from multiple scanners

### Example Methods Paragraph for Publications

When describing rtpipeline's robustness analysis in a manuscript, consider using language similar to:

> **Radiomics Feature Stability Assessment**
>
> Radiomics feature stability was assessed using RTpipeline's adapted NTCV chain, inspired by but not identical to Zwanenburg et al.[1] For each ROI, all [N] configured states were required, combining Gaussian noise (σ = 0, 10, and 20 HU), superior-inferior translations (0 and +/-4 mm), two reproducible random physical-space contour offsets, and distance-ranked adaptation to the closest voxel counts representing -15%, 0%, and +15% volume changes. Features were re-extracted for every state using PyRadiomics [version]. ICC(3,1) was estimated across complete subject grids; CoV was calculated within each patient/course/ROI/source and summarized by its cohort median.
>
> Feature stability was quantified using the intraclass correlation coefficient ICC(3,1) computed via Pingouin[2], with each perturbation treated as a fixed rater measuring the same underlying subject (patient-course-structure combination). This interpretation follows the fixed-rater rationale described by Koo & Li[3] but represents an adaptation of ICC to perturbation-based analysis rather than a standard inter-rater scenario. Features with ICC ≥ 0.90 and coefficient of variation (CoV) ≤ 10% were classified as "robust" following commonly used thresholds in the radiomics literature.[3] Only robust features were retained for subsequent modeling.
>
> [1] Zwanenburg A, et al. Assessing robustness of radiomic features by image perturbation. Sci Rep. 2019;9:614. DOI: 10.1038/s41598-018-36938-4
> [2] Vallat R. Pingouin: statistics in Python. JOSS. 2018;3(31):1026. DOI: 10.21105/joss.01026
> [3] Koo TK, Li MY. A guideline of selecting and reporting ICC for reliability research. J Chiropr Med. 2016;15(2):155-163. DOI: 10.1016/j.jcm.2016.02.012

**Important:** Adjust the specific parameter values (noise levels, translation distances, volume changes, thresholds) to match your actual configuration. If using the CTV1 D95 heuristic for Rx estimation, explicitly note this limitation. If using Fast Mode for segmentation, note that lower-resolution segmentations were used.

## Typical Feature Families by Robustness

Based on radiomics reproducibility literature (feature-type patterns are generally consistent across studies, though specific results vary):

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
Check `apply_to_structures` patterns in config. Verify structure names match your patterns in:
- `RS_auto.dcm` (TotalSegmentator)
- `RS_custom.dcm` (Custom structures)
- `Segmentation_{model_name}/rtstruct.dcm` (Custom models)

### "Insufficient perturbations for X; skipping"
Some small structures may fail erosion/dilation. This is expected; they'll be skipped automatically.

## Key References

1. Zwanenburg A, et al. (2019). "Assessing robustness of radiomic features by image perturbation." *Scientific Reports* 9, 614. DOI: 10.1038/s41598-018-36938-4
2. Zwanenburg A, et al. (2020). "The Image Biomarker Standardization Initiative (IBSI)." *Radiology* 295(2):328-338. DOI: 10.1148/radiol.2020191145
3. Koo TK, Li MY. (2016). "A guideline of selecting and reporting intraclass correlation coefficients for reliability research." *Journal of Chiropractic Medicine* 15(2):155-163. DOI: 10.1016/j.jcm.2016.02.012
4. Vallat R. (2018). "Pingouin: statistics in Python." *Journal of Open Source Software* 3(31):1026. DOI: 10.21105/joss.01026

*Note: Additional references cited in the literature review sections represent reported findings from the radiomics reproducibility literature. Users should verify specific citations for their own publications.*

## What's New (2025)

### NTCV Perturbation Chain Implementation

Based on 2023-2025 radiomics stability research:

✅ **Implemented:**
- NTCV (Noise + Translation + Contour + Volume) perturbation chains
- Image noise injection (Gaussian noise in HU)
- Rigid translation perturbations (configurable; shipped maximum +/-4 mm)
- Contour randomization (boundary noise simulation)
- Configurable perturbation intensity (mild/standard/aggressive)
- Explicit Cartesian grids: 12 (mild), 81 (standard), or 315 (aggressive) with shipped defaults
- Conservative clinical thresholds: ICC >0.90 and CoV <10%

**Key improvements over basic volume-only perturbations:**
- **Comprehensive stability testing** inspired by the Zwanenburg 2019 perturbation framework
- **Literature-reported performance**: Zwanenburg et al. achieved ~98–99% sensitivity with <2% false positives on their specific test-retest datasets—these serve as benchmarks, not guarantees
- **Cost-effective alternative** to expensive test-retest imaging
- **Multi-axis perturbations** capture different sources of variability

### Research Basis

The implementation is informed by radiomics reproducibility literature findings:
1. ICC ≥0.75 and CoV ≤10% are commonly reported thresholds; ICC ≥0.90 is a conservative configurable research filter.[^koo2016]
2. Systematic perturbation chains combining geometric and image-based variations improve robustness assessment
3. Perturbation count should follow the study design and be reported exactly; the shipped standard grid contains 81 states
4. A substantial proportion of features (varies by study and structure) may meet stability thresholds
5. Perturbation-based methods provide a practical alternative when test-retest data is unavailable

## Future Enhancements

The following methods are referenced in the literature sections above but are **not yet implemented** in rtpipeline. They are planned for future releases:

- **CovBat harmonization**: Advanced harmonization method (outperforms traditional ComBat) — must be applied as external preprocessing
- **Graph-FS feature selection**: Stability-aware feature selection maintaining multi-institution performance
- **Segmentation-method robustness**: Compare Manual vs TotalSegmentator vs custom models
- **Scan-rescan ICC**: Test-retest reliability for longitudinal studies
- **Panel-averaged features**: Zwanenburg-style feature averaging across perturbations
- **Preprocessing variations**: Resampling and discretization robustness testing

---

## Footnote References

[^zwanenburg2019]: A. Zwanenburg *et al.*, "Assessing robustness of radiomic features by image perturbation," *Scientific Reports* 9, 614 (2019). DOI: 10.1038/s41598-018-36938-4
[^ibsi2020]: A. Zwanenburg *et al.*, "The Image Biomarker Standardization Initiative," *Radiology* 295(2):328–338 (2020). DOI: 10.1148/radiol.2020191145
[^koo2016]: T. K. Koo and M. Y. Li, "A guideline of selecting and reporting intraclass correlation coefficients for reliability research," *Journal of Chiropractic Medicine* 15(2):155–163 (2016). DOI: 10.1016/j.jcm.2016.02.012
[^pingouin]: R. Vallat, "Pingouin: statistics in Python," *Journal of Open Source Software* 3(31):1026 (2018). DOI: 10.21105/joss.01026

*Note: Some citations in earlier sections reference findings from the broader radiomics reproducibility literature. Users should independently verify specific references for publication purposes.*
