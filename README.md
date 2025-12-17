# RTpipeline

**The Big Data Radiotherapy Pipeline**

*From raw clinical exports to research-ready datasets in one command.*

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://kstawiski.github.io/rtpipeline/)
[![Docker](https://img.shields.io/badge/docker-ready-green)](https://hub.docker.com/r/kstawiski/rtpipeline)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

**RTpipeline** is a comprehensive, research-grade pipeline that transforms raw DICOM radiotherapy exports into analysis-ready data. It bridges the technical gap between clinical Treatment Planning Systems (TPS) and statistical/ML analysis by automating:

- **DICOM Organization** - Groups scattered files into patient courses
- **AI Segmentation** - TotalSegmentator (~100 standardized structures) + custom nnU-Net models
- **DVH Extraction** - Comprehensive dose-volume metrics
- **Radiomics** - IBSI-compliant features with robustness assessment
- **Quality Control** - Automated checks and audit reports

### Who Is This For?

| Audience | Value Proposition |
|----------|-------------------|
| **PhD Students** | Spend your PhD on science, not reinventing DICOM parsing |
| **Clinical Researchers** | Minimal coding—drag & drop in Web UI, get Excel tables |
| **Multi-Center Consortia** | Shared configs ensure identical preprocessing at every site |

---

## Key Features

### 1. Standardized Anatomy via AI

Run TotalSegmentator on every CT to get consistent structure definitions:

```
Input: "Heart", "hrt", "Coeur", "cardiac"  (inconsistent)
Output: "heart"  (standardized for every patient)
```

### 2. Systematic CT Cropping

Normalize field-of-view using anatomical landmarks for comparable metrics:

```
Before: V20Gy = 500cc / 18,000cc = 2.8% (long scan)
After:  V20Gy = 500cc / 12,000cc = 4.2% (standardized FOV)
```

### 3. Robustness-Aware Radiomics

NTCV perturbation chains (Zwanenburg et al., 2019) identify stable features:

- **N**oise injection (scanner variability)
- **T**ranslation (positioning uncertainty)
- **C**ontour randomization (inter-observer variability)
- **V**olume adaptation (segmentation uncertainty)

Features with ICC ≥ 0.90 and CoV ≤ 10% classified as **robust**.

### 4. Analysis-Ready Outputs

```
_RESULTS/
├── dvh_metrics.xlsx      # Dmean, D95%, V20Gy for every structure
├── radiomics_ct.xlsx     # 1000+ IBSI-compliant features
├── case_metadata.xlsx    # Clinical tags, scanner info
└── qc_reports.xlsx       # Quality control summary
```

### 5. High-Performance Computing

Designed for modern hardware with automatic optimization:

- **GPU Acceleration:** CUDA-accelerated deep learning for segmentation
- **Smart Parallelization:** Automatically scales to available CPU cores
- **Resource Management:** Adaptive worker scaling prevents memory overflows
- **Speed:** Process hundreds of patients in hours, not days

---

## Quick Start

### Option 1: Interactive Docker Setup (Recommended)

```bash
curl -sSL https://raw.githubusercontent.com/kstawiski/rtpipeline/main/setup_docker_project.sh | bash
```

### Option 2: Docker Compose

```bash
# Create folders
mkdir -p Input Output Logs

# Start pipeline + Web UI
docker-compose up -d

# Open http://localhost:8080
```

### Option 3: Google Colab

Try it in the cloud with free GPU access:

- [Part 1: GPU Segmentation](rtpipeline_colab_part1_gpu.ipynb)
- [Part 2: CPU Analysis](rtpipeline_colab_part2_cpu.ipynb)

### Option 4: Local Installation

```bash
git clone https://github.com/kstawiski/rtpipeline.git
cd rtpipeline
snakemake --cores all --use-conda
```

---

## Documentation

Full documentation at **[kstawiski.github.io/rtpipeline](https://kstawiski.github.io/rtpipeline/)**

| Section | Description |
|---------|-------------|
| [Getting Started](https://kstawiski.github.io/rtpipeline/getting_started/) | From zero to first analyzed patient |
| [Web UI Guide](https://kstawiski.github.io/rtpipeline/getting_started/webui/) | Drag-and-drop interface |
| [Output Format](https://kstawiski.github.io/rtpipeline/user_guide/output_format/) | Data table schemas |
| [Case Studies](https://kstawiski.github.io/rtpipeline/case_studies/) | Real-world research examples |
| [Reproducibility](https://kstawiski.github.io/rtpipeline/user_guide/reproducibility/) | Methods templates for publications |
| [Radiomics Robustness](https://kstawiski.github.io/rtpipeline/features/radiomics_robustness/) | NTCV perturbation methodology |

---

## Case Studies

### 1. NTCP Modeling for Rectal Toxicity

Build dose-response models from standardized DVH metrics.
[Learn more →](https://kstawiski.github.io/rtpipeline/case_studies/#case-study-1-ntcp-modeling-for-late-rectal-toxicity)

### 2. Radiomics Signature Development

Create robust imaging biomarkers with NTCV perturbation assessment.
[Learn more →](https://kstawiski.github.io/rtpipeline/case_studies/#case-study-2-radiomics-signature-for-treatment-response)

### 3. Multi-Center Data Harmonization

Federated learning with identical preprocessing at every institution.
[Learn more →](https://kstawiski.github.io/rtpipeline/case_studies/#case-study-3-multi-center-data-harmonization)

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────┐     ┌─────────────────┐
│     EXTRACT     │     │           TRANSFORM              │     │      LOAD       │
│                 │     │                                  │     │                 │
│  • DICOM CT     │     │  • Structure harmonization       │     │  • DVH tables   │
│  • RTSTRUCT     │ ──► │  • TotalSegmentator              │ ──► │  • Radiomics    │
│  • RTDOSE       │     │  • Systematic cropping           │     │  • Metadata     │
│  • RTPLAN       │     │  • Robustness analysis           │     │  • QC reports   │
│                 │     │                                  │     │                 │
└─────────────────┘     └──────────────────────────────────┘     └─────────────────┘
```

---

## Citation

If you use RTpipeline in your research, please cite:

```bibtex
@software{rtpipeline,
  title = {RTpipeline: Automated Radiotherapy DICOM Processing Pipeline},
  author = {Stawiski, Konrad},
  url = {https://github.com/kstawiski/rtpipeline},
  year = {2025}
}
```

Also cite the underlying tools:
- **TotalSegmentator:** Wasserthal et al., *Radiology: AI* (2023)
- **PyRadiomics:** van Griethuysen et al., *Cancer Research* (2017)
- **IBSI:** Zwanenburg et al., *Radiology* (2020)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Note:** Model weights for TotalSegmentator are downloaded automatically. Custom nnU-Net models must be provided separately. See [documentation](https://kstawiski.github.io/rtpipeline/) for details.
