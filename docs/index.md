---
layout: default
title: Home
---

# RTpipeline Documentation

**RTpipeline** is a comprehensive radiotherapy DICOM processing pipeline that automates analysis workflows for research and clinical decision support.

## Features

- **DICOM Organization** - Automatic grouping of series and RT objects per patient course
- **AI-powered Segmentation** - TotalSegmentator (CT/MR) + custom nnU-Net models
- **DVH Analytics** - Comprehensive dose-volume histogram analysis
- **Radiomics Extraction** - PyRadiomics feature extraction with IBSI-aligned settings
- **Robustness Analysis** - Perturbation-based feature stability assessment
- **Systematic CT Cropping** - Anatomical landmark-based FOV standardization
- **Quality Control** - Automated QC reports and validation

## Quick Links

| Guide | Description |
|-------|-------------|
| [Pipeline Architecture](PIPELINE_ARCHITECTURE.md) | Technical design and workflow stages |
| [Parallelization & Resources](PARALLELIZATION.md) | CPU/GPU utilization and worker configuration |
| [Radiomics Robustness](RADIOMICS_ROBUSTNESS.md) | Feature stability assessment methodology |
| [Systematic CT Cropping](SYSTEMATIC_CT_CROPPING.md) | Anatomical cropping for DVH/radiomics |
| [Docker Guide](DOCKER.md) | Container deployment instructions |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |

## Getting Started

The recommended way to run RTpipeline is via Docker:

```bash
# Interactive setup (recommended)
./setup_docker_project.sh

# Or manual setup
docker-compose up -d
# Then access http://localhost:8080
```

## Methodology Notes

### Radiomics Robustness
The pipeline implements perturbation-based feature stability assessment following the NTCV (Noise + Translation + Contour + Volume) methodology described by Zwanenburg et al. (2019). **Important:** Literature-reported performance metrics were validated on specific datasets and should be treated as benchmarks, not guarantees for new cohorts.

- ICC(3,1) with analytical confidence intervals
- Configurable thresholds: ICC ≥ 0.90 (robust), CoV ≤ 10% (robust)
- See [RADIOMICS_ROBUSTNESS.md](RADIOMICS_ROBUSTNESS.md) for full methodology

### CT Cropping
Systematic cropping to anatomical landmarks reduces (but does not eliminate) field-of-view variability in percentage DVH metrics. See [SYSTEMATIC_CT_CROPPING.md](SYSTEMATIC_CT_CROPPING.md).

## Key References

- Zwanenburg A *et al.*, "Assessing robustness of radiomic features by image perturbation," *Sci Rep* 9, 614 (2019). DOI: [10.1038/s41598-018-36758-2](https://doi.org/10.1038/s41598-018-36758-2)
- Zwanenburg A *et al.*, "The Image Biomarker Standardization Initiative," *Radiology* 295(2):328-338 (2020). DOI: [10.1148/radiol.2020191145](https://doi.org/10.1148/radiol.2020191145)
- Koo TK, Li MY, "A guideline of selecting and reporting ICC for reliability research," *J Chiropr Med* 15(2):155-163 (2016). DOI: [10.1016/j.jcm.2016.02.012](https://doi.org/10.1016/j.jcm.2016.02.012)

---

**Repository:** [github.com/kstawiski/rtpipeline](https://github.com/kstawiski/rtpipeline)

**Documentation last updated:** November 2025
