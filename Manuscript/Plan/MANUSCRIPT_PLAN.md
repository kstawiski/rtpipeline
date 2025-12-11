# RTpipeline Manuscript Plan for Computer Methods and Programs in Biomedicine

**Document Version:** 1.0
**Date:** December 2025
**Consensus Source:** GPT-5.1 (8/10 confidence) + Gemini-3-pro (8.5/10 confidence)

---

## Executive Summary

This document outlines the comprehensive manuscript plan for submitting RTpipeline to Computer Methods and Programs in Biomedicine (CMPB). The plan has been developed through multi-model AI consensus and addresses the radiomics reproducibility crisis through integrated robustness analysis and FOV standardization.

---

## 1. Title (Proposed)

**RTpipeline: An Open-Source, Containerized Radiotherapy DICOM Processing Workflow with Integrated Robustness Analysis and Anatomical Field-of-View Standardization**

Alternative shorter title:
*RTpipeline: A Robustness-First Workflow for Reproducible Radiotherapy Radiomics*

---

## 2. Main Framing and Value Proposition

### Primary Angle
**"Addressing the Reproducibility Crisis in Radiomics"** - Frame RTpipeline as more than a processing tool; it's an infrastructure contribution for reliable quantitative RT research.

### One-Line Value Proposition
> "RTpipeline is, to our knowledge, one of the first open-source, containerized RT workflows that transforms raw DICOM exports into robust, IBSI-compliant radiomics and DVH with standardized FOV and canonical structures across institutions."

### Key Novel Contributions (in order of importance)
1. **Integrated NTCV Perturbation Analysis** - Built-in robustness assessment with ICC(3,1)/CoV computation
2. **Systematic CT Cropping** - Anatomical landmark-based FOV standardization (unique feature)
3. **Multi-Source Segmentation Fusion** - Priority-based combination of manual, TotalSegmentator, and nnU-Net outputs
4. **Structure Harmonization** - Maps institutional naming variations to canonical labels
5. **End-to-End Reproducibility** - Snakemake DAG + Docker ensures identical processing across sites
6. **Federated/Local Processing Model** - Institutions run pipeline locally on their data, then share
   only anonymized research-ready tables (dvh_metrics.xlsx, radiomics_ct.xlsx) rather than raw
   imaging data. This enables multi-center studies while:
   - Preserving data privacy (no raw DICOM transfer required)
   - Ensuring methodological harmonization (identical preprocessing at every site)
   - Eliminating bias from centralized re-segmentation

---

## 3. Structured Abstract (350 words max)

### Background and Objective
Radiomics and dose-response analyses in radiotherapy suffer from poor reproducibility due to heterogeneous DICOM exports, inconsistent structure naming, variable scan field-of-view (FOV), and non-robust features. Existing tools address fragments of this problem but lack integrated robustness assessment and standardized preprocessing. We present RTpipeline, an open-source, containerized workflow that transforms raw radiotherapy DICOM data into analysis-ready datasets with built-in robustness qualification.

### Methods
RTpipeline implements a Snakemake-orchestrated, Docker-containerized workflow comprising: (1) automated DICOM organization with structure harmonization to canonical labels; (2) multi-source segmentation fusion combining manual contours, TotalSegmentator (~100 structures), and custom nnU-Net models; (3) systematic anatomical CT cropping to standardize FOV across patients; (4) IBSI-compliant radiomics extraction via PyRadiomics; (5) integrated NTCV (Noise, Translation, Contour, Volume) perturbation analysis following Zwanenburg et al. (2019) with ICC(3,1) and CoV computation; and (6) DVH extraction using dicompyler-core. Technical validation was performed on N patients demonstrating pipeline functionality, IBSI compliance via digital phantom, feature extraction accuracy, and reproducibility across environments.

### Results
RTpipeline successfully processed [N] treatment courses, organizing [X] DICOM files into structured patient-course hierarchies. NTCV robustness analysis revealed that [Y]% of radiomic features met robust thresholds (ICC ≥ 0.90, CoV ≤ 10%). Systematic CT cropping reduced non-biological variance in percentage-based metrics by [Z]%. Feature extraction matched standalone PyRadiomics within numerical tolerance. Output reproducibility was verified across multiple container instances with identical checksums.

### Conclusions
RTpipeline provides the radiotherapy research community with a standardized, reproducible workflow that addresses key sources of radiomics variability through integrated robustness assessment and FOV standardization. The containerized architecture enables consistent multi-center deployment, while the robustness-first design helps researchers identify stable imaging biomarkers suitable for clinical translation.

---

## 4. Manuscript Structure

### 4.1 Introduction

**Content:**
1. Clinical and research context for radiotherapy radiomics
2. The reproducibility crisis (cite multi-center instability studies, IBSI findings)
3. Specific documented problems:
   - Heterogeneous DICOM exports and structure naming
   - Variable scan FOV affecting percentage metrics
   - Non-robust features sensitive to segmentation/acquisition
   - Poorly documented, non-reproducible analysis chains
4. Review of existing tools and their limitations:
   - PyRadiomics: feature calculator, not end-to-end pipeline
   - CERR: MATLAB-based, licensing, GUI-driven
   - 3D Slicer RT: interactive, not batch-processing oriented
5. Aim: Present RTpipeline and demonstrate core functionality

**Word Count Target:** ~800-1000 words

### 4.2 Methods

#### 2.1 Overall Architecture
- Snakemake DAG workflow description
- Docker/Singularity containerization strategy
- Configuration management and version pinning
- Supported inputs/outputs

#### 2.2 DICOM Organization and Structure Harmonization
- Patient/course grouping logic
- Vendor heterogeneity handling
- Structure mapping rules with canonical labels
- TG-263 integration considerations

#### 2.3 Segmentation and Multi-Source Fusion
- TotalSegmentator integration (~100 structures)
- Custom nnU-Net model support
- Manual contour import from RTSTRUCT
- **Fusion algorithm (CRITICAL):**
  - Priority rules (manual > nnU-Net > TotalSegmentator)
  - Conflict resolution logic
  - Missing structure handling
- Extensibility for custom auto-segmenters

#### 2.4 Custom Structure Generation
- Boolean operations (union, intersection, subtraction)
- Configuration syntax (YAML examples)
- Clinical use cases (bowel_bag, PRVs, ring structures)

#### 2.5 CT Cropping / FOV Standardization
- Anatomical landmark detection via TotalSegmentator
- Region-specific cropping rules:
  - Pelvis: L1 superior → 10cm below femoral heads
  - Thorax: C7 → L1
  - Abdomen: T12/L1 → L5
  - Head & Neck: Skull+2cm → C7-2cm
  - Brain: Brain±1cm
- Fallback logic for missing landmarks
- Application to CT, masks, dose grids
- Output files (ct_cropped.nii.gz, RS_auto_cropped.dcm)

**NOTE on CT Cropping and Radiomics:**
With fixed bin width discretization (recommended by IBSI), CT cropping has NO effect on
ROI-based radiomics features because voxel intensities inside the segmentation are unchanged.
The primary value of CT cropping is:
1. **DVH Standardization:** Percentage-based metrics (V20Gy_%) become comparable across
   patients when body volume is standardized
2. **Computational Efficiency:** Reduced memory footprint and faster processing
3. **Storage Optimization:** Smaller file sizes for archival

**Caveat for Filtered Features:** If spatial filters (LoG, Wavelets) are applied, cropping
near boundaries may introduce padding artifacts. Ensure sufficient margins (≥2× kernel size)
beyond the ROI when using filtered features, or apply cropping after filtering.

#### 2.6 Radiomics Extraction (IBSI-Compliant)
- PyRadiomics configuration
- Preprocessing settings:
  - Resampling (spacing, interpolator)
  - Discretization (bin width vs bin count)
- Feature classes (~1000+ features)
- IBSI compliance verification

#### 2.7 NTCV Perturbation and Robustness Metrics
- **NTCV Chain Implementation:**
  - **N (Noise):** Gaussian noise injection (σ = 10, 20 HU; 0 HU = baseline)
  - **T (Translation):** Rigid shifts (±2, ±4 mm in x, y directions)
  - **C (Contour):** Boundary randomization via morphological operations
  - **V (Volume):** Erosion/dilation (±15% volume change)
- Perturbation intensity levels (mild/standard/aggressive)
- **"NTCV-full" Definition:** Perturbations applied simultaneously (not aggregated separately).
  A single NTCV-full perturbation combines noise injection, translation, contour randomization,
  AND volume adaptation in one pass, testing feature stability under compound uncertainty.
- **Robustness Metrics:**
  - ICC(3,1) via Pingouin library - computed **per (structure, feature) tuple**
  - Coefficient of Variation (CoV)
  - Quartile Coefficient of Dispersion (QCD) - descriptive only, not used for classification
- **CRITICAL: ICC Subject Definition:**
  - Targets (subjects) = patient_id ONLY
  - Raters = perturbation_id
  - ICC computed separately for each (structure, feature) combination to avoid
    variance inflation from physiological differences between tissues
- Robustness thresholds:
  - Robust: ICC ≥ 0.90 AND CoV ≤ 10%
  - Acceptable: ICC ≥ 0.75 AND CoV ≤ 20%
- Clinical plausibility of perturbations (rationale per Zwanenburg et al. 2019)

#### 2.8 DVH Extraction
- dicompyler-core integration
- Structure-dose matching logic
- Standard DVH metrics (Dmean, Dmax, D2cc, V20Gy, etc.)
- Interaction with CT cropping

#### 2.9 Web UI and Deployment
- Flask-based interface capabilities
- Non-technical user workflow
- Containerization details
- HPC/cluster integration

#### 2.10 Failure Mode Handling
- Error handling strategy
- Incomplete DICOM series handling
- Missing TotalSegmentator structures
- Logging and QC reports

#### 2.11 Example Data and Evaluation Setup
- Dataset description (5 patients, ~2000 DICOM files)
- Modalities (CT, RTSTRUCT, RTDOSE, RTPLAN)
- Evaluation objectives

**Word Count Target:** ~3000-4000 words

### 4.3 Results

#### 3.1 DICOM Organization and Structure Harmonization
- Number of files processed
- Courses identified
- Harmonization success rate

#### 3.2 Segmentation Outputs
- Number of structures per source
- Fusion coverage
- Dice/HD95 vs manual (if available)

#### 3.3 Radiomics Robustness (NTCV) - **KEY SECTION**
- Per-structure ICC/CoV distributions
- Proportion of robust features by structure and feature class
- Perturbation contribution analysis (which NTCV components matter most)
- Comparison of feature families (shape vs texture vs first-order)

#### 3.4 Effect of CT Cropping on DVH Metrics
- **Primary Focus:** DVH metric changes before/after cropping (V20Gy_%, Dmean_body)
- **Secondary:** Computational performance improvement (time, memory)
- **Confirmation:** Radiomics features unchanged with fixed bin width (sanity check)
- **NOTE:** Radiomics features should be identical pre/post cropping for non-filtered
  features; any differences indicate implementation issues

**Experimental Design:**
- Run segmentation ONCE to generate masks
- Apply cropping transformation to CT/dose/masks
- Compute DVH and radiomics from BOTH cropped and uncropped versions
- This avoids confounds from re-running stochastic segmentation

#### 3.5 IBSI Compliance Verification
- Digital phantom benchmark results
- Feature parity vs standalone PyRadiomics

#### 3.6 Runtime and Resource Profiling
- Time per pipeline stage per patient
- CPU vs GPU requirements
- Memory usage for heavy steps

#### 3.7 Reproducibility Verification
- Cross-run output comparison
- Multi-environment testing (Docker instances)
- Hash/checksum verification

**Word Count Target:** ~1500-2000 words

### 4.4 Discussion

#### Key Points:
1. **Principal Contributions:**
   - End-to-end reproducible RT workflow
   - Integrated robustness analysis as first-class feature
   - FOV standardization addressing known confounder
   - Structure harmonization enabling multi-center aggregation

2. **Comparison with Existing Tools:**
   - PyRadiomics: RTpipeline wraps it with workflow + robustness
   - CERR: Python-native, open-source, container-first
   - 3D Slicer: Complementary (interactive vs batch)

3. **Implications for Multi-Center Studies:**
   - Reduced analytic heterogeneity
   - Prospective standardization
   - Federated learning enablement
   - **Privacy-Preserving Collaboration:** Institutions can run RTpipeline locally on their
     protected health information (PHI), then share only anonymized, research-ready tables
     (DVH metrics, radiomics features). This eliminates the need for raw DICOM transfer,
     addressing regulatory barriers (GDPR, HIPAA) while ensuring identical methodology
   - **Segmentation Bias Elimination:** Local processing with the same AI models ensures
     consistent segmentation without introducing biases from centralized re-contouring

4. **Limitations:**
   - Small technical validation dataset (acknowledge explicitly)
   - Single anatomical site focus
   - Dependency on underlying segmentation model quality
   - No clinical outcome validation

5. **Future Work:**
   - Larger multi-center retrospective validation
   - TCIA public dataset demonstration
   - Additional anatomical sites
   - Integration with clinical data (FHIR)
   - Dose accumulation, deformable registration

**Word Count Target:** ~1500-2000 words

### 4.5 Conclusion
- Succinct summary of RTpipeline's role
- Emphasis on reproducibility, robustness, standardization
- Call for community adoption and extension

**Word Count Target:** ~200-300 words

---

## 5. Figures (Planned)

### Figure 1: High-Level Workflow Diagram (Multi-Panel)
**Panel A:** DICOM chaos → RTpipeline → organized patients/courses
**Panel B:** Segmentation paths (manual, TS, nnU-Net → fusion)
**Panel C:** DVH + radiomics + NTCV robustness
**Panel D:** Outputs (tables, QC plots, web UI)

*Type: Schematic/flowchart*
*Estimated size: Full page*

### Figure 2: Structure Harmonization and DICOM Organization
- Schematic showing institutional labels → canonical labels
- Example mapping configuration snippet
- Course directory structure visualization

*Type: Schematic with code snippet*
*Estimated size: Half page*

### Figure 3: CT Cropping Before/After Illustration
**Panel A:** Full CT with volume of interest highlighted
**Panel B:** Cropped CT with standardized FOV, dose overlaid
**Panel C:** Bar plot showing example metric changes

*Type: Medical imaging visualization*
*Estimated size: Half page*

### Figure 4: Robustness Analysis Summary (KEY FIGURE)
**Panel A:** ICC distribution histogram by perturbation type
**Panel B:** Proportion of robust features (bar chart by structure)
**Panel C:** Feature robustness heatmap (rows=features, cols=perturbations)
**Panel D:** ICC vs CoV scatter plot with robustness quadrants

*Type: Statistical visualization*
*Estimated size: Full page*

### Figure 5: Perturbation Contribution Analysis
- Shows how different NTCV components (N, T, C, V) affect feature stability
- Boxplots or violin plots comparing ICC distributions

*Type: Statistical visualization*
*Estimated size: Half page*

### Supplementary Figure S1: Web UI Screenshots
- Upload interface
- Job monitoring
- Results visualization

*Type: Screenshots*
*Location: Supplementary material*

### Supplementary Figure S2: Snakemake DAG
- Full workflow graph

*Type: Schematic*
*Location: Supplementary material*

---

## 6. Tables (Planned)

### Table 1: Structure Harmonization and Fusion Rules
| Institutional Label(s) | Canonical Label | Source Priority | Boolean Ops |
|------------------------|-----------------|-----------------|-------------|
| Rectum, RECT, rectum* | RECTUM | Manual > TS > nnU-Net | - |
| Bladder, BLAD, Vessie | BLADDER | Manual > TS | - |
| small_bowel, colon, duodenum | bowel_bag | TS | Union |

### Table 2: PyRadiomics Feature Classes and Counts
| Feature Class | Count | Description |
|---------------|-------|-------------|
| Shape | ~14 | 3D morphological features |
| First-order | ~18 | Intensity histogram statistics |
| GLCM | ~24 | Gray-level co-occurrence |
| GLRLM | ~16 | Gray-level run length |
| GLSZM | ~16 | Gray-level size zone |
| GLDM | ~14 | Gray-level dependence |
| NGTDM | ~5 | Neighborhood gray-tone |
| **Total** | **~107×N filters** | |

### Table 3: Example DVH Metrics Output
| Patient | Course | Structure | Segmentation_Source | Dmean_Gy | V20Gy_% | ... |
|---------|--------|-----------|---------------------|----------|---------|-----|
| 001 | C1 | RECTUM | Manual | 45.2 | 65.3 | ... |
| 001 | C1 | BLADDER | AutoRTS_total | 32.1 | 42.1 | ... |

### Table 4: Robustness Summary Statistics (KEY TABLE)
| Structure | Perturbation | n_features | Median ICC | IQR | Median CoV | % Robust |
|-----------|--------------|------------|------------|-----|------------|----------|
| RECTUM | NTCV-full | 1074 | 0.87 | 0.15 | 8.2 | 62% |
| BLADDER | NTCV-full | 1074 | 0.89 | 0.12 | 7.1 | 68% |
| GTV | Volume-only | 1074 | 0.92 | 0.08 | 5.4 | 78% |

### Table 5: CT Cropping Effect on Metrics
| Metric | Pre-Crop Mean | Post-Crop Mean | Mean Diff | % Change |
|--------|---------------|----------------|-----------|----------|
| V20Gy_% (Rectum) | 2.8 | 4.2 | +1.4 | +50% |
| Energy (Rectum) | 1.2e8 | 9.8e7 | -2.2e7 | -18% |

### Table 6: Comparison with Existing Tools
| Feature | RTpipeline | PyRadiomics | CERR | 3D Slicer RT |
|---------|------------|-------------|------|--------------|
| DICOM RT Organization | ++ | - | + | + |
| Structure Harmonization | ++ | - | - | - |
| AI Segmentation | ++ | - | + | + |
| IBSI Radiomics | ++ | ++ | + | + |
| Integrated Robustness | ++ | - | - | - |
| FOV Standardization | ++ | - | - | - |
| Containerized | ++ | - | - | - |
| Open Source | ++ | ++ | + | ++ |

### Table 7: Runtime Benchmarks
| Pipeline Stage | Time per Patient | GPU Required | Peak RAM |
|----------------|------------------|--------------|----------|
| Organization | ~1 min | No | 2 GB |
| Segmentation (TS) | ~3 min | Yes (8 GB) | 12 GB |
| DVH | ~30 sec | No | 4 GB |
| Radiomics | ~5 min | No | 8 GB |
| Robustness (NTCV) | ~30 min | No | 8 GB |
| **Total** | **~40 min** | | |

### Table 8: Reproducibility Verification
| Run | Environment | Output Hash | Match |
|-----|-------------|-------------|-------|
| Run 1 | Docker local | SHA256:abc... | Baseline |
| Run 2 | Docker cloud | SHA256:abc... | 100% |
| Run 3 | Singularity HPC | SHA256:abc... | 100% |

---

## 7. Statistical Analysis Plan

### 7.1 Robustness Metrics
- **ICC(3,1)** computed via Pingouin's `intraclass_corr()` with:
  - Subject = patient_id + course_id + structure + segmentation_source
  - Rater = perturbation_id
  - Analytical 95% CIs
- **CoV** = 100 × (SD / |mean|) across perturbations
- **QCD** = (Q3 - Q1) / (Q3 + Q1) as robust dispersion measure

### 7.2 Robustness Classification Thresholds
Following Koo & Li (2016) ICC guidelines:
- **Robust:** ICC ≥ 0.90 AND CoV ≤ 10%
- **Acceptable:** ICC ≥ 0.75 AND CoV ≤ 20%
- **Poor:** Below acceptable thresholds

### 7.3 CT Cropping Effect Analysis
- Paired comparison of metrics pre- vs post-cropping
- Report: Mean difference, % change, 95% CI
- Visualize with Bland-Altman plots

### 7.4 Segmentation Quality (if manual available)
- Dice Similarity Coefficient (DSC)
- Hausdorff Distance 95th percentile (HD95)
- Surface distance metrics

### 7.5 Perturbation Contribution Analysis
- Compare ICC distributions across perturbation types (N, T, C, V, combined)
- Wilcoxon signed-rank test for paired comparisons
- Identify which perturbation has largest impact on feature stability

### 7.6 Feature Parity Test
- Compare RTpipeline features vs standalone PyRadiomics
- Report: Correlation coefficient, mean absolute difference
- Tolerance: <0.1% relative difference

---

## 8. Validation Experiments

### 8.1 IBSI Digital Phantom Benchmark (MANDATORY)
- **Data Acquisition:** Download IBSI Phase 1 digital phantom from official IBSI GitHub repository
  (https://github.com/theibsi/data_sets)
- **Phantom Configuration:**
  - Use IBSI-recommended preprocessing settings (no resampling, fixed bin width 25 HU)
  - Process all IBSI-defined feature classes
- **Validation Procedure:**
  - Compute all ~107 PyRadiomics IBSI-compliant features
  - Compare against IBSI reference values (published consensus)
  - Calculate absolute and relative differences for each feature
- **Pass/Fail Criteria:**
  - **Pass:** Relative difference ≤ 1% OR absolute difference within IBSI tolerance
  - **Fail:** Deviations documented with root cause analysis
- **Reporting:**
  - Overall match rate (% features within tolerance)
  - Feature-level comparison table (Supplementary Table S2)
  - Any systematic deviations explained (e.g., discretization differences)

### 8.2 Feature Parity vs PyRadiomics
- Same configuration, same input data
- Compare all ~1000+ features
- Verify 1:1 correspondence within numerical tolerance

### 8.3 DVH Correctness vs Clinical TPS
- Compare RTpipeline DVH metrics with clinical TPS output
- Structures: Multiple OARs and targets
- Tolerance: <2% for key metrics

### 8.4 Reproducibility Verification
- Run pipeline twice on same data
- Verify bitwise-identical outputs
- Cross-environment testing (Docker local vs cloud vs Singularity)

### 8.5 Segmentation Quality vs Manual
- If manual contours available:
  - Calculate Dice, HD95 for key structures
  - Report mean ± SD across patients

### 8.6 "Cost of Robustness" Analysis
- Quantify extra compute time for NTCV perturbations
- Compare: Raw extraction vs Robustness-enabled
- Report: Time overhead, resource overhead

---

## 9. Key References (Preliminary List)

### Radiomics Reproducibility
1. Zwanenburg A, et al. (2019). Assessing robustness of radiomic features by image perturbation. *Sci Rep* 9:614.
2. Zwanenburg A, et al. (2020). The Image Biomarker Standardization Initiative (IBSI). *Radiology* 295(2):328-338.
3. Koo TK, Li MY. (2016). A guideline of selecting and reporting ICC for reliability research. *J Chiropr Med* 15(2):155-163.

### Underlying Tools
4. van Griethuysen JJM, et al. (2017). Computational radiomics system to decode the radiographic phenotype. *Cancer Res* 77(21):e104-e107. [PyRadiomics]
5. Wasserthal J, et al. (2023). TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. *Radiol AI* 5(5):e230024.
6. Isensee F, et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nat Methods* 18:203-211.

### Radiotherapy Standards
7. Mayo CS, et al. (2018). AAPM TG-263: Standardizing nomenclatures in radiation oncology.
8. Deasy JO, et al. (2003). CERR: A computational environment for radiotherapy research.

### Workflow/Reproducibility
9. Mölder F, et al. (2021). Sustainable data analysis with Snakemake. *F1000Research* 10:33.

### ICC Computation
10. Vallat R. (2018). Pingouin: statistics in Python. *JOSS* 3(31):1026.

---

## 10. Limitations to Acknowledge

1. **Small Validation Dataset:** This study presents a technical validation on n=5 treatment
   courses. All statistical analyses are descriptive and exploratory; they demonstrate pipeline
   functionality rather than support clinical inference or biomarker discovery. The robustness
   metrics demonstrate the methodology works, not that specific features are universally robust.
2. **Single-Site Data:** May not capture full vendor/protocol heterogeneity
3. **Limited Anatomical Sites:** Focus on pelvic radiotherapy examples
4. **Segmentation Model Dependence:** Robustness depends on underlying AI model quality
5. **No Clinical Outcome Validation:** Features not linked to patient outcomes
6. **Maintenance Burden:** Dependencies (PyRadiomics, TotalSegmentator, etc.) evolve over time

**Framing Guidance:** Position manuscript strictly as "System description and technical
validation" rather than "Clinical robustness study." Use n=5 results to demonstrate
capability to generate metrics, not to draw biological conclusions about feature stability.

---

## 11. Future Work (for Discussion)

1. Multi-center retrospective validation on TCIA public datasets (Lung1, HNSCC)
2. Additional anatomical sites (thorax, head & neck, brain)
3. Integration with clinical data sources (FHIR, HL7)
4. Dose accumulation and deformable registration modules
5. Prospective deployment in clinical trials
6. Community-driven extension of harmonization tables

---

## 12. Checklist for Submission

- [ ] IBSI digital phantom benchmark completed
- [ ] GitHub repository public and well-documented
- [ ] DockerHub image pullable and tested
- [ ] Example data de-identified and licensed
- [ ] "One-click" reproduction scripts for all figures/tables
- [ ] CMPB author guidelines followed (350-word structured abstract, IMRAD format)
- [ ] All references formatted correctly
- [ ] Conflict of interest statement prepared
- [ ] Data availability statement drafted
- [ ] Code availability statement with DOI (Zenodo)

---

## 13. Timeline (Suggested)

| Phase | Activities | Duration |
|-------|------------|----------|
| 1. Pipeline Execution | Run full pipeline on Example_data | 1 day |
| 2. Data Analysis | Generate all metrics, ICC, cropping effects | 2-3 days |
| 3. Figure Generation | Create all planned figures | 2-3 days |
| 4. Manuscript Drafting | Write all sections | 1-2 weeks |
| 5. Internal Review | Co-author feedback | 1 week |
| 6. Revision | Address feedback | 3-5 days |
| 7. Final Preparation | Format, references, supplementary | 2-3 days |
| 8. Submission | Submit to CMPB | - |

---

## Appendix A: CMPB Journal Requirements Summary

- **Abstract:** 350 words max, structured (Background/Objective, Methods, Results, Conclusions)
- **Sections:** Introduction, Methods, Results, Discussion (separate from Results), Acknowledgements
- **References:** Up to 50
- **Figures/Tables:** No specific limits
- **File Format:** Editable source files (.doc/.docx or .tex)
- **Software Papers:** Common in this journal; emphasize methodology and reproducibility
- **Open Access Option:** Available

---

## Appendix B: Consensus Summary

### Models Consulted
- **GPT-5.1:** 8/10 confidence, "for" stance
- **Gemini-3-pro:** 8.5/10 confidence, "neutral" stance

### Key Agreement Points
1. Technical architecture (Snakemake + Docker) is sound
2. NTCV perturbation + FOV standardization are key differentiators
3. "Robustness-first workflow" is the correct framing
4. Segmentation fusion logic needs explicit description
5. IBSI compliance verification is mandatory
6. Public repository essential

### Key Disagreement and Resolution
- **Sample Size:** Gemini recommended 50-100+ patients from TCIA; GPT-5.1 found 5-patient acceptable for technical validation
- **Resolution:** Position as technical demonstration + IBSI phantom; acknowledge limitation and recommend TCIA extension for future validation

### Unified Recommendations Incorporated
1. Failure mode analysis section
2. "Cost of Robustness" compute benchmarks
3. Formalized fusion algorithm description
4. Quantitative impact vs baseline workflow
5. Perturbation contribution analysis
6. Multi-environment reproducibility verification

---

*Document generated via multi-model AI consensus for RTpipeline manuscript planning.*
