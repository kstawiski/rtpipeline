# RTpipeline Manuscript - Figures Plan

## Overview

This document provides detailed specifications for each figure in the manuscript.

---

## Figure 1: High-Level Workflow Diagram

### Purpose
Provide readers with a comprehensive overview of RTpipeline's end-to-end workflow.

### Layout
Multi-panel figure (2x2 or horizontal strip)

### Panel A: DICOM Ingestion and Organization
- **Left:** Chaotic DICOM files icon (scattered files representing vendor exports)
- **Center:** RTpipeline logo/box
- **Right:** Organized hierarchy (Patient → Course → Modality)
- **Labels:** "Raw DICOM Export", "Organize Stage", "Structured Courses"

### Panel B: Segmentation Pathways
- Three parallel input paths:
  1. Manual RTSTRUCT → "Manual Contours"
  2. TotalSegmentator → "~100 Anatomical Structures"
  3. Custom nnU-Net → "Site-Specific Models"
- Converging to: "Multi-Source Fusion" box
- Output: "RS_custom.dcm (Harmonized)"

### Panel C: Feature Extraction and Robustness
- Input: Fused Structures + CT/Dose
- Parallel outputs:
  1. "DVH Metrics" (dicompyler-core)
  2. "Radiomics Features" (PyRadiomics/IBSI)
- Additional processing: "NTCV Perturbation Chain"
- Final output: "Robustness Metrics (ICC, CoV)"

### Panel D: Outputs and Deployment
- Table icons representing:
  - dvh_metrics.xlsx
  - radiomics_ct.xlsx
  - radiomics_robustness_summary.xlsx
  - qc_reports.xlsx
- Additional elements:
  - Web UI screenshot thumbnail
  - Docker/container icon
  - "Multi-Center Deployment" badge

### Technical Specifications
- **Dimensions:** Full page width, ~6 inches wide
- **Format:** Vector graphics (SVG/PDF for submission)
- **Colors:** Consistent color scheme (suggest: blue for processing, green for outputs, orange for robustness)
- **Software:** Matplotlib/Seaborn, BioRender, or draw.io

### Data Required
- N/A (schematic only)

---

## Figure 2: Structure Harmonization and DICOM Organization

### Purpose
Illustrate how institutional naming variations are mapped to canonical labels and how DICOM files are organized.

### Layout
Two-panel figure (side-by-side or stacked)

### Panel A: Structure Harmonization Schematic
- **Left column:** "Institutional Labels"
  - Examples: "Rectum_full", "RECT", "rectum", "Rektum"
  - "Bladder", "BLAD", "Vessie", "urinary_bladder"
  - "PTV_70Gy", "PTV_boost", "PTV70"
- **Center:** Arrows through "Mapping Rules" box
- **Right column:** "Canonical Labels"
  - "RECTUM"
  - "BLADDER"
  - "PTV_HIGH"

### Panel B: Course Directory Structure
```
Data_Snakemake/
├── Patient_001/
│   └── Course_001/
│       ├── DICOM/CT/
│       ├── Segmentation_TotalSegmentator/
│       ├── RS.dcm (original)
│       ├── RS_auto.dcm (TotalSegmentator)
│       ├── RS_custom.dcm (fused)
│       ├── dvh_metrics.xlsx
│       └── radiomics_ct.xlsx
```

### Technical Specifications
- **Dimensions:** Half page width
- **Format:** Vector graphics
- **Font:** Monospace for directory structure

### Data Required
- Example from config.yaml structure mapping
- Actual directory listing from Example_data output

---

## Figure 3: CT Cropping Before/After Illustration

### Purpose
Demonstrate the effect of systematic anatomical cropping on CT field-of-view.

### Layout
Three-panel figure (horizontal)

### Panel A: Full CT Scan
- Coronal or sagittal view of full CT
- Volume of interest (e.g., pelvis) highlighted with box
- Total scan extent marked with arrows
- Label: "Original FOV: L1 → below knees"

### Panel B: Cropped CT
- Same view after cropping
- Standardized extent marked
- Label: "Standardized FOV: L1 → 10cm below femoral heads"
- Overlay: Dose distribution if available

### Panel C: Metric Change Visualization
- Bar plot or lollipop plot showing:
  - V20Gy_% pre-crop vs post-crop
  - Selected radiomics feature changes
  - Delta percentage annotations

### Technical Specifications
- **Dimensions:** Full page width
- **Format:** Mixed (medical imaging PNG + vector plot)
- **Anonymization:** Ensure all patient identifiers removed

### Data Required
- CT NIfTI before/after cropping from Example_data
- DVH metrics before/after cropping
- Radiomics features before/after cropping

### Code to Generate
```python
# Panel A & B: Medical imaging visualization
import nibabel as nib
import matplotlib.pyplot as plt

# Load original and cropped CT
ct_orig = nib.load('ct.nii.gz')
ct_crop = nib.load('ct_cropped.nii.gz')

# Panel C: Metric changes
import pandas as pd
import seaborn as sns

metrics_df = pd.DataFrame({
    'Metric': ['V20Gy_%', 'Dmean', 'Energy', 'Entropy'],
    'Pre-Crop': [...],
    'Post-Crop': [...],
    'Change': [...]
})
```

---

## Figure 4: Robustness Analysis Summary (KEY FIGURE)

### Purpose
Central figure demonstrating the NTCV robustness analysis results - the paper's main methodological contribution.

### Layout
Four-panel figure (2x2 grid)

### Panel A: ICC Distribution Histogram
- X-axis: ICC values (0 to 1)
- Y-axis: Number of features
- Multiple histograms overlaid or faceted by:
  - Perturbation type (N, T, C, V, NTCV-full)
- Vertical dashed lines at:
  - ICC = 0.90 (Robust threshold, green)
  - ICC = 0.75 (Acceptable threshold, orange)
- Legend: Perturbation types

### Panel B: Proportion of Robust Features by Structure
- Bar chart
- X-axis: Structure names (RECTUM, BLADDER, GTV, PTV, etc.)
- Y-axis: Percentage of features
- Stacked or grouped bars for: Robust, Acceptable, Poor
- Color scheme: Green (robust), Orange (acceptable), Red (poor)

### Panel C: Feature Robustness Heatmap
- Rows: Feature names (grouped by class: Shape, First-order, GLCM, etc.)
- Columns: Perturbation IDs or types
- Color: ICC value (diverging colormap, e.g., RdYlGn)
- Annotations: Only for extreme values
- Dendrogram on rows for clustering similar features

### Panel D: ICC vs CoV Scatter Plot ("Robustness Quadrant")
- X-axis: ICC (0.5 to 1.0)
- Y-axis: CoV (%) (0 to 50)
- Each point = one feature (colored by class or structure)
- Quadrant lines:
  - Vertical at ICC = 0.90
  - Horizontal at CoV = 10%
- Quadrant labels:
  - Upper-left: "High ICC, High CoV"
  - Upper-right: "High ICC, Low CoV" (Robust - green background)
  - Lower-left: "Low ICC, High CoV" (Poor)
  - Lower-right: "Low ICC, Low CoV"
- Legend: Feature classes or structures

### Technical Specifications
- **Dimensions:** Full page
- **Format:** Vector graphics
- **Colors:** Consistent with paper theme; colorblind-friendly palette

### Data Required
- ICC values per feature per perturbation
- CoV values per feature
- Structure labels
- Feature class labels

### Code to Generate
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load robustness results
rob_df = pd.read_excel('radiomics_robustness_summary.xlsx', sheet_name='global_summary')

# Panel A: ICC distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A
axes[0, 0].hist(rob_df['icc'].dropna(), bins=30, edgecolor='black')
axes[0, 0].axvline(0.90, color='green', linestyle='--', label='Robust')
axes[0, 0].axvline(0.75, color='orange', linestyle='--', label='Acceptable')

# Panel B
structure_counts = rob_df.groupby(['structure', 'robustness_label']).size().unstack(fill_value=0)
structure_counts.plot(kind='bar', stacked=True, ax=axes[0, 1])

# Panel C
pivot = rob_df.pivot_table(index='feature_name', columns='structure', values='icc')
sns.heatmap(pivot, cmap='RdYlGn', ax=axes[1, 0])

# Panel D
axes[1, 1].scatter(rob_df['icc'], rob_df['cov_pct'], alpha=0.5)
axes[1, 1].axvline(0.90, color='green', linestyle='--')
axes[1, 1].axhline(10, color='green', linestyle='--')
```

---

## Figure 5: Perturbation Contribution Analysis

### Purpose
Show how different NTCV perturbation types affect feature stability differentially.

### Layout
Single panel or two-panel figure

### Main Panel: Violin/Box Plots
- X-axis: Perturbation type (Baseline, Noise, Translation, Contour, Volume, NTCV-Full)
- Y-axis: ICC values
- One violin/box per perturbation type
- Overlay: Individual feature points (jittered)
- Statistical annotations: Wilcoxon p-values between consecutive pairs

### Alternative Panel B: Delta ICC Heatmap (Optional)
- Rows: Feature classes
- Columns: Perturbation types
- Color: Mean ICC decrease from baseline

### Technical Specifications
- **Dimensions:** Half to full page
- **Format:** Vector graphics

### Data Required
- ICC values computed separately for each perturbation type
- Feature-level breakdown

### Code to Generate
```python
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

# Assuming perturbation-specific ICC computed
pert_icc_df = pd.DataFrame({
    'Perturbation': ['Baseline', 'Noise', 'Translation', 'Contour', 'Volume', 'NTCV'],
    'feature_name': [...],
    'icc': [...]
})

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=pert_icc_df, x='Perturbation', y='icc', ax=ax)
ax.axhline(0.90, color='green', linestyle='--')
ax.axhline(0.75, color='orange', linestyle='--')
```

---

## Supplementary Figures

### Figure S1: Web UI Screenshots
- Panel A: Upload interface
- Panel B: Job monitoring dashboard
- Panel C: Results viewer
- Panel D: Configuration panel

### Figure S2: Snakemake DAG
- Full directed acyclic graph of pipeline rules
- Generated via `snakemake --rulegraph | dot -Tpng`

### Figure S3: Segmentation Quality Examples
- Side-by-side comparison: Manual vs TotalSegmentator vs nnU-Net
- Dice/HD95 annotations per structure

### Figure S4: IBSI Phantom Results
- Bar chart showing feature match rates
- Reference values vs RTpipeline values

---

## Figure Generation Scripts Location

All figure generation scripts will be stored in:
```
Manuscript/
├── scripts/
│   ├── fig1_workflow.py
│   ├── fig2_harmonization.py
│   ├── fig3_cropping.py
│   ├── fig4_robustness.py
│   ├── fig5_perturbation.py
│   └── supplementary_figures.py
├── figures/
│   ├── fig1_workflow.pdf
│   ├── fig2_harmonization.pdf
│   ├── fig3_cropping.pdf
│   ├── fig4_robustness.pdf
│   ├── fig5_perturbation.pdf
│   └── supplementary/
│       ├── figS1_webui.png
│       ├── figS2_dag.pdf
│       └── ...
```

---

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Processing steps | Blue | #3498db |
| Outputs | Green | #2ecc71 |
| Robustness elements | Orange | #e67e22 |
| Warning/Poor | Red | #e74c3c |
| Neutral | Gray | #95a5a6 |

---

*Figure plan for RTpipeline CMPB manuscript.*
