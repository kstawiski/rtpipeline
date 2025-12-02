# rtpipeline

**Automated Radiotherapy DICOM Processing Pipeline**

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://kstawiski.github.io/rtpipeline/)
[![Docker](https://img.shields.io/badge/docker-ready-green)](https://hub.docker.com/r/kstawiski/rtpipeline)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**rtpipeline** is a comprehensive, research-grade pipeline that turns raw DICOM radiotherapy exports into analysis-ready data. It automates organization, segmentation (TotalSegmentator & nnU-Net), DVH calculation, radiomics extraction, and quality control.

## 🚀 Key Features

*   **Automated Organization**: Groups DICOM series, RTSTRUCTs, Plans, and Doses by patient course.
*   **AI Segmentation**:
    *   **TotalSegmentator**: Robust whole-body segmentation for CT and MR.
    *   **Custom nnU-Net**: Plug-and-play support for custom models.
*   **Systematic Cropping**: Standardizes anatomical field-of-view for consistent DVH/radiomics analysis.
*   **Analysis**:
    *   **DVH**: Comprehensive dose metrics (V%, D%) relative to prescription.
    *   **Radiomics**: High-throughput feature extraction (PyRadiomics) with robustness testing.
*   **Web UI**: User-friendly drag-and-drop interface.

## 📚 Documentation

Full documentation is available at **[kstawiski.github.io/rtpipeline](https://kstawiski.github.io/rtpipeline/)**.

*   [**Getting Started**](https://kstawiski.github.io/rtpipeline/getting_started/)
*   [**Web UI Guide**](https://kstawiski.github.io/rtpipeline/getting_started/webui/)
*   [**Docker Setup**](https://kstawiski.github.io/rtpipeline/getting_started/docker_setup/)
*   [**Interpretation Guide**](https://kstawiski.github.io/rtpipeline/user_guide/results_interpretation/)

## ⚡ Quick Start

### Option 1: One-Line Docker Installer (Recommended)

```bash
curl -sSL https://raw.githubusercontent.com/kstawiski/rtpipeline/main/setup_docker_project.sh | bash
```

### Option 2: Google Colab

Try it in the cloud with free GPU access:

*   [**Part 1: GPU Segmentation**](rtpipeline_colab_part1_gpu.ipynb)
*   [**Part 2: CPU Analysis**](rtpipeline_colab_part2_cpu.ipynb)

## 🛠️ Manual Installation

If you prefer to run without Docker (requires Conda/Mamba):

```bash
# Clone repository
git clone https://github.com/kstawiski/rtpipeline.git
cd rtpipeline

# Run pipeline
snakemake --cores all --use-conda
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Note:** Model weights for TotalSegmentator and custom nnU-Net models are downloaded automatically or must be provided separately. See documentation for details.