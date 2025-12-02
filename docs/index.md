# rtpipeline

**The Big Data Radiotherapy Pipeline**

*From raw clinical exports to research-ready datasets in one command.*

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://kstawiski.github.io/rtpipeline/)
[![Docker](https://img.shields.io/badge/docker-ready-green)](https://hub.docker.com/r/kstawiski/rtpipeline)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🔬 Bridging the Gap: Clinical to Research

Radiotherapy produces some of the most complex data in medicine: 3D imaging (CT/MR), 3D dose distributions, and geometric structures. **But extracting this data for research is painful.**

*   **The Problem:** Clinical Treatment Planning Systems (TPS) export messy, unstructured DICOM files.
    *   Series are scattered across folders.
    *   Structure names are inconsistent (`Heart`, `heart`, `hrt`).
    *   Geometric and dosimetric data is locked in binary files, not dataframes.
    *   Re-segmentation for consistent analysis is manual and slow.

*   **The Solution:** **rtpipeline** acts as a "normalization engine" for radiotherapy big data. It ingests raw DICOM dumps and outputs **tidy, standardized data tables** ready for immediate analysis in Python, R, or JMP.

## 🌟 Key Capabilities

### 1. Automated Data Engineering
*   **Organization:** Automatically groups thousands of DICOM files into patient courses (e.g., `Patient123/2023-01`).
*   **Reconciliation:** Links Plans, Doses, and Images even if they are in different folders.
*   **TotalSegmentator Integration:** Automatically generates ~100 standardized anatomical structures (OARs) for every patient using state-of-the-art AI.

### 2. Standardization for "Big Data"
*   **Consistent Anatomy:** By running TotalSegmentator on every patient, you get consistent structures (`heart`, `lung_left`, `esophagus`) regardless of what the physician drew.
*   **Systematic Cropping:** Automatically crops CTs to a standard anatomical ROI (e.g., L1 to Femur), making DVH volume metrics ($V_{20Gy}$, $V_{95\%}$) comparable across cohorts.
*   **Radiomics Robustness:** Implements "Stress Testing" for radiomics features (perturbing noise, rotation, volume) to ensure you only analyze stable biomarkers.

### 3. Analysis-Ready Outputs
Forget parsing DICOM tags. **rtpipeline** gives you ready-to-use Excel/CSV files:
*   `dvh_metrics.xlsx`: $D_{mean}$, $D_{95\%}$, $V_{20Gy}$ for every structure.
*   `radiomics.xlsx`: 1000+ IBSI-compliant features per structure.
*   `metadata.xlsx`: Extracted clinical tags, reconstruction kernels, and scanner info.

---

## 🚀 Quick Start

### 1. Interactive Docker Setup (Recommended)

The easiest way to start—no git cloning required.

```bash
curl -sSL https://raw.githubusercontent.com/kstawiski/rtpipeline/main/setup_docker_project.sh | bash
```
*Follow the wizard to generate your `docker-compose.yml` and start the Web UI.*

### 2. Manual Start

If you already have the repository:

```bash
# 1. Create folders
mkdir -p Input Output Logs

# 2. Start Pipeline (Web UI + Processing Engine)
docker-compose up -d

# 3. Open Web UI
# Go to http://localhost:8080
```

---

## 📚 Documentation

Visit the full documentation site: **[kstawiski.github.io/rtpipeline](https://kstawiski.github.io/rtpipeline/)**

### Core Guides
*   **[Getting Started](getting_started/index.md):** From zero to your first analyzed patient.
*   **[Web UI Guide](getting_started/webui.md):** How to use the drag-and-drop interface.
*   **[Docker Setup](getting_started/docker_setup.md):** Comprehensive deployment guide.

### Research Features
*   **[Output Format](user_guide/output_format.md):** Detailed schema of the generated data tables.
*   **[Systematic CT Cropping](features/ct_cropping.md):** How standardization improves statistical power.
*   **[Radiomics Robustness](features/radiomics_robustness.md):** Methodology for feature stability.
*   **[Custom Models](features/custom_models.md):** Plug in your own nnU-Net models.

---

## 🛠️ Data Flow Architecture

```mermaid
graph LR
    A[TPS Export\n(Raw DICOM)] --> B(rtpipeline\nOrchestrator);
    B --> C{Organization\nEngine};
    C --> D[AI Segmentation\nTotalSegmentator];
    C --> E[Image Processing\nCropping/NIfTI];
    D --> F[Analysis Engine];
    E --> F;
    F --> G[DVH Calculator];
    F --> H[Radiomics\nExtractor];
    G --> I[Final Tidy Tables\n(.xlsx / .csv)];
    H --> I;
```

---

## 📄 License & Citation

This project is licensed under the MIT License.
If you use rtpipeline for research, please cite the repository and the underlying tools (TotalSegmentator, PyRadiomics, Snakemake).

```