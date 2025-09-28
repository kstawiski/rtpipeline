rtpipeline – DICOM‑RT Processing Pipeline
==========================================

## Overview
rtpipeline is an end-to-end DICOM-RT processing pipeline that turns a folder of raw DICOMs (CT/RTPLAN/RTDOSE/RTSTRUCT/RT treatment records) into clean, per-patient "courses" with organized data, segmentation, quantitative DVH metrics, visual QA, and rich metadata for clinical and research use.

## Quick Start

### Using Snakemake (Recommended)
```bash
# Snakemake automatically manages conda environments
snakemake --use-conda --cores 4
```

### Direct CLI Usage
```bash
pip install -e .
rtpipeline --dicom-root Example_data --outdir Data --logs Logs
```

## Documentation
- [Pipeline overview](docs/pipeline.md) — stage-by-stage breakdown of discovery, grouping, organization, segmentation, DVH, visualization, and radiomics.
- [Quickstart](docs/quickstart.md) — environment setup and first run.
- [CLI reference](docs/cli.md) — exhaustive command-line options with examples.
- [Metadata fields](docs/metadata.md) — global and per-case exports.
- [Segmentation](docs/segmentation.md), [DVH](docs/dvh.md), [Viewers](docs/viewers.md) — deeper dives into optional modules.

At a glance, the pipeline:
- Scans a DICOM root (one or many patients) and extracts global metadata into Excel.
- Groups each patient's data into treatment "courses" by the CT StudyInstanceUID (primary + boost on the same CT are merged; subsequent treatments on a different CT are separated).
- Organizes each course into a canonical structure (CT_DICOM, RP/RD/RS), sums multi-stage doses into a single RD, and synthesizes a "summed" RP with total Rx.
- Runs TotalSegmentator on the course CT (resume‑safe; re-run with `--force-segmentation`) and auto‑generates RTSTRUCT (RS_auto) aligned to the CT, so DVH can include auto‑segmentation even if no manual RS exists.
- Computes DVH metrics for manual and auto structures using dicompyler‑core, including integral dose, hottest small volumes (D1cc, D0.1cc), and coverage at Rx, and writes per‑course and merged workbooks.
- Generates interactive reports: Plotly‑based DVH report (toggle structures) and an Axial viewer to scroll CT slices with semi‑transparent overlays for manual/auto structures.
- Saves comprehensive per‑case metadata (JSON/XLSX) for clinical/research queries (plan approvals, clinicians, prescriptions per ROI, beam geometry and MUs, dose grid, CT acquisition, course dates, etc.) and merges all cases into a single workbook/JSON for cohort analysis.

Typical flow (per patient):
1) Discover → extract global metadata (plans/doses/structures/fractions/CT index) to `outdir/Data/*.xlsx`.
2) Group → courses by CT StudyInstanceUID to avoid merging unrelated treatments.
3) Organize → copy CT, sum RD (with resampling), synthesize RP (total Rx), pick RS; write per‑case metadata.
4) Segment (optional) → run TotalSegmentator (DICOM + NIfTI), resume‑safe; build RS_auto from SEG/RTSTRUCT/NIfTI.
5) DVH → compute metrics for RS and RS_auto; save per‑course and merged across all courses.
6) Visualize → DVH_Report.html (Plotly) and Axial.html (CT viewer + overlays) per course.

Design principles:
- Clinically meaningful course grouping and dose summation.
- Idempotent, resume‑safe segmentation and RTSTRUCT generation.
- DVH driven by RTDOSE/RTPLAN with manual and auto RTSTRUCT.
- Visual QA tools built for rapid review of structure coverage and auto‑seg quality.
- Rich metadata to support downstream clinical and research analyses.
- Parallelized non‑segmentation phases (organize, DVH, visualization, metadata) with `--workers` control.

## Installation

### Dependencies
- Python 3.11 recommended
- Core deps: pydicom (>=3.0,<4), dicompyler-core (DVH), SimpleITK, pydicom-seg, matplotlib, scipy, pandas, nested-lookup, plotly, rt-utils, dicom2nifti, nbformat, ipython.
- TotalSegmentator is pulled from GitHub: `TotalSegmentator @ git+https://github.com/wasserth/TotalSegmentator`
- Optional: pyradiomics for radiomics features extraction
- External: `dcm2niix` via OS package manager or conda (e.g., `conda install -c conda-forge dcm2niix`)

### Installation Steps
```bash
# Clone repository
git clone <repository-url>
cd rtpipeline

# Install package
pip install -e .

# Or with radiomics support
pip install -e ".[radiomics]"

# Verify installation
rtpipeline doctor
```

## Snakemake Workflow

The pipeline includes a Snakemake workflow that automatically manages conda environments for optimal compatibility:
- Main environment with NumPy 2.x for TotalSegmentator
- Radiomics environment with NumPy 1.x for PyRadiomics

### Snakemake Usage
```bash
# Run complete pipeline
snakemake --use-conda --cores 4

# Dry run to see planned actions
snakemake -n

# Run specific rules
snakemake radiomics --use-conda --cores 4

# Clean intermediate files
snakemake clean

# Clean all outputs
snakemake clean_all
```

### Configuration
Edit `config.yaml` to customize:
```yaml
dicom_root: "Example_data"
output_dir: "Data_Snakemake"
logs_dir: "Logs_Snakemake"
workers: 4
```

## CLI Usage
- Minimal run:
  ```bash
  rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs
  ```

- Options:
  - `--merge-criteria {same_ct_study,frame_of_reference}`
  - `--max-days N` to split plans that are far apart in time
  - `--no-segmentation`, `--force-segmentation` to skip or re-run segmentation
  - `--no-dvh`, `--no-visualize`, `--no-metadata` to skip phases
  - `--workers N` for parallel processing control
  - `--totalseg-fast` to add `--fast` (recommended on CPU)
  - `--totalseg-roi-subset roi1,roi2` to restrict ROIs
  - `--resume` to skip completed steps (idempotent resume)
  - `doctor` subcommand: `rtpipeline doctor` prints environment checks

## Outputs
- Per patient: `outdir/<patient_id>/course_<course_key>/`
  - `RP.dcm`, `RD.dcm` (summed if multiple), `RS.dcm` (manual if available)
  - `CT_DICOM/` (CT slices matching the course study)
  - `nifti/` (from dcm2niix), `TotalSegmentator_{DICOM,NIFTI}/` (if segmentation enabled)
  - `RS_auto.dcm` (auto-generated RTSTRUCT from TotalSegmentator for DVH)
  - `dvh_metrics.xlsx` (includes IntegralDose, D1ccGy, V95%Rx, V100%Rx, …)
  - `DVH_Report.html` (interactive Plotly DVH)
  - `Axial.html` (scrollable axial CT QA viewer with manual/auto overlays)
  - `case_metadata.json` and `case_metadata.xlsx` (per-course clinical/research metadata)
  - `radiomics_features_CT.xlsx` (pyradiomics features if enabled)

## Course Merging Logic
- By default, plans/doses are merged only if they refer to the same CT StudyInstanceUID
- Alternative: `--merge-criteria frame_of_reference` groups by FrameOfReferenceUID
- Use `--max-days` to impose a time window for course grouping

## Notes
- TotalSegmentator does not require a license key
- Segmentation is resume-safe (reused if present). Use `--force-segmentation` to re-run
- Radiomics is parallelized across courses and ROIs/labels to speed up extraction
- When using Snakemake, PyRadiomics runs in a separate conda environment with NumPy 1.x for full compatibility

## Troubleshooting
See `TROUBLESHOOTING.md` for common issues and solutions.

## License
MIT