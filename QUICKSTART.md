# RTpipeline Quick Start Guide

## Installation

### Option 1: Using Snakemake (Recommended)

1. Install Snakemake:
```bash
pip install snakemake
```

2. Run the pipeline:
```bash
snakemake --use-conda --cores 4
```

Snakemake will automatically create and manage the required conda environments.

### Option 2: Direct CLI Usage

1. Install the pipeline:
```bash
pip install -e .
```

2. Run the pipeline:
```bash
rtpipeline --dicom-root Example_data --outdir Data --logs Logs
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
dicom_root: "Example_data"       # Input DICOM directory
output_dir: "Data_Snakemake"     # Output directory
logs_dir: "Logs_Snakemake"       # Logs directory
workers: 4                       # Number of parallel workers
```

## Snakemake Commands

```bash
# Run complete pipeline
snakemake --use-conda --cores 4

# Dry run (see what would be executed)
snakemake -n

# Run specific rules
snakemake organize_data --use-conda --cores 2
snakemake radiomics --use-conda --cores 4

# Clean intermediate files
snakemake clean

# Clean all outputs
snakemake clean_all

# Unlock directory (if previous run was interrupted)
snakemake --unlock
```

## Direct CLI Options

```bash
rtpipeline --help  # Show all options

# Common options:
--dicom-root        # Input DICOM directory
--outdir            # Output directory
--logs              # Logs directory
--workers           # Number of parallel workers
--no-segmentation   # Skip TotalSegmentator
--no-radiomics      # Skip radiomics extraction
--no-dvh            # Skip DVH calculation
--custom-structures # Path to custom structures YAML
```

## Outputs

The pipeline generates organized outputs for each patient:

```
Data_Snakemake/
├── <patient_id>/
│   ├── CT_DICOM/              # CT DICOM files
│   ├── metadata.json          # Patient metadata
│   ├── RS_auto.dcm           # Auto-segmentation
│   ├── RS_custom.dcm         # Custom structures
│   ├── Radiomics_CT.xlsx     # Radiomics features
│   ├── dvh_metrics.xlsx      # DVH metrics
│   └── Axial.html            # Visualization
```

## Troubleshooting

1. **Conda environment creation is slow**: This only happens on first run. Subsequent runs will use cached environments.

2. **"Directory cannot be locked" error**: Run `snakemake --unlock` to clear the lock.

3. **Missing dependencies**: Ensure you have conda/mamba installed for Snakemake workflow.

4. **Radiomics fails**: The pipeline uses separate conda environments for NumPy compatibility. Ensure both environments are created.

## Test Run

To test with minimal processing:

```bash
# Direct CLI (no segmentation/radiomics)
rtpipeline --dicom-root Example_data --outdir Test --logs TestLogs \
    --no-segmentation --no-radiomics --workers 1

# With Snakemake (organize only)
snakemake organize_data --use-conda --cores 1
```