# Getting Started with rtpipeline

This guide will help you get started with rtpipeline in just a few minutes.

## Prerequisites

- **Docker** installed (version 19.03 or later)
- **Docker Compose** installed
- For GPU acceleration: **nvidia-docker** or **nvidia-container-toolkit**
- At least **8 GB RAM** available (16 GB+ recommended)
- At least **20 GB free disk space** (more for large datasets)

## Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/kstawiski/rtpipeline.git
cd rtpipeline
```

Or download from Docker Hub (no git clone needed):

```bash
docker pull kstawiski/rtpipeline:latest
```

### 2. Build the Docker Image (if cloning from git)

```bash
./build.sh
```

This will create the Docker image with all dependencies. Build time: 10-30 minutes depending on your internet connection.

## Quick Start: Web UI (Recommended)

The easiest way to use rtpipeline is through the Web UI.

### 1. Create Project Directories

Create the necessary folders on your host machine to store data and logs:

```bash
mkdir -p Input Output Logs Uploads totalseg_weights
```

### 2. Start the Container

**With GPU (recommended for faster processing):**
```bash
docker-compose up -d
```

**CPU-only (no GPU required):**
```bash
docker-compose --profile cpu-only up -d
```

### 3. Access the Web UI

Open your browser and navigate to:
```
http://localhost:8080
```

You should see the rtpipeline upload interface.

### 4. Upload Your DICOM Files

You can upload DICOM files in several ways:

- **Drag and drop** files or folders into the upload area
- **Click "Browse Files"** to select files manually
- Upload **ZIP archives** containing DICOM files
- Upload **DICOMDIR** files

**Supported formats:**
- Individual `.dcm` files
- `.zip` archives
- `.tar`, `.tar.gz`, `.tgz` archives
- Entire directories with subdirectories
- `DICOMDIR` files

### 4. Review Validation Results

After upload, the system will automatically:
- Extract files from archives
- Scan for DICOM files
- Validate the structure
- Show you:
  - Number of DICOM files found
  - Detected modalities (CT, MR, RTPLAN, RTDOSE, RTSTRUCT)
  - Number of patients, studies, and series
  - Warnings and suggestions

### 5. Configure Processing

Choose your processing options:

- ✅ **Fast Mode**: Recommended for CPU-only systems
- ✅ **Radiomics**: Extract radiomic features (recommended)
- ❌ **Custom Models**: Only if you have custom nnUNet models
- ❌ **CT Cropping**: For standardized anatomical regions

### 6. Start Processing

Click **🚀 Start Processing** to begin.

### 7. Monitor Progress

Watch the progress bar as the pipeline:
- Organizes DICOM files (20%)
- Runs segmentation (40%)
- Computes DVH metrics (60%)
- Extracts radiomics (80%)
- Aggregates results (90%)
- Completes (100%)

### 8. Download Results

When processing completes, click **Download Results** to get a ZIP file containing:

- `_RESULTS/dvh_metrics.xlsx` - DVH metrics for all structures
- `_RESULTS/radiomics_ct.xlsx` - Radiomic features
- `_RESULTS/case_metadata.xlsx` - Patient and study metadata
- Per-patient directories with segmentation masks and QC reports

## Alternative: Command Line

If you prefer command-line access:

### 1. Start the Container

```bash
docker-compose up -d
```

### 2. Access the Container Shell

```bash
docker exec -it rtpipeline bash
```

### 3. Place DICOM Files

Your DICOM files should be in the `Input/` directory on your host machine, which is mounted to `/data/input` in the container.

```bash
# On host: copy your DICOM files
cp -r /path/to/your/dicom/files ./Input/
```

### 4. Run the Pipeline

Inside the container:

```bash
snakemake --cores all --use-conda --configfile /app/config.container.yaml
```

### 5. View Results

Results will appear in the `Output/` directory on your host machine.

## Understanding the Results

### Key Output Files

All results are in the `Output/` directory (or `_RESULTS/` subdirectory):

#### 1. **dvh_metrics.xlsx**
Contains dose-volume histogram metrics for all structures:
- **Columns**: PatientID, CourseID, StructureName, ROI_Volume_cc, Segmentation_Source, V95%, V100%, D2cc, etc.
- **Use cases**: Treatment plan evaluation, toxicity analysis, plan comparison

#### 2. **radiomics_ct.xlsx**
Contains radiomic features extracted from CT images:
- **Columns**: PatientID, CourseID, ROI, segmentation_source, plus 150+ radiomic features
- **Use cases**: Radiomics research, outcome prediction, texture analysis

#### 3. **case_metadata.xlsx**
Contains patient and treatment metadata:
- **Columns**: PatientID, CourseID, CT spacing, prescription dose, plan UID, structure names
- **Use cases**: Quality control, cohort description, data verification

#### 4. **qc_reports.xlsx**
Quality control reports flagging potential issues:
- Structure cropping warnings
- Frame-of-reference mismatches
- Missing data

### Directory Structure

```
Output/
├── _RESULTS/                     # Aggregated results (start here!)
│   ├── dvh_metrics.xlsx
│   ├── radiomics_ct.xlsx
│   ├── radiomics_mr.xlsx
│   ├── case_metadata.xlsx
│   └── qc_reports.xlsx
└── <patient_id>/
    └── <course_id>/
        ├── DICOM/                # Original DICOM files
        ├── NIFTI/                # NIfTI images
        ├── Segmentation_TotalSegmentator/  # Auto segmentation
        ├── dvh_metrics.xlsx      # Per-course DVH
        ├── radiomics_ct.xlsx     # Per-course radiomics
        └── qc_reports/           # Per-course QC
```

## Common Use Cases

### Use Case 1: DVH Analysis for Treatment Planning

**Goal**: Extract DVH metrics for treatment plan evaluation

**Steps**:
1. Upload: CT + RTDOSE + RTSTRUCT files
2. Configure: Fast mode ✅, Radiomics ❌ (not needed for DVH only)
3. Process: Wait for completion
4. Download: Open `_RESULTS/dvh_metrics.xlsx`
5. Analyze: Filter by structure names, review V95%, D2cc, etc.

### Use Case 2: Radiomics Research

**Goal**: Extract radiomic features for outcome prediction

**Steps**:
1. Upload: CT files (RTDOSE/RTSTRUCT optional)
2. Configure: Fast mode ✅, Radiomics ✅
3. Process: Wait for completion
4. Download: Open `_RESULTS/radiomics_ct.xlsx`
5. Analyze: 150+ features per ROI ready for machine learning

### Use Case 3: Batch Processing Multiple Patients

**Goal**: Process multiple patients in one job

**Steps**:
1. Organize: Create one ZIP with subdirectories per patient
2. Upload: Upload the entire ZIP
3. Configure: Settings apply to all patients
4. Process: Pipeline processes each patient automatically
5. Download: All results in one download

### Use Case 4: Auto-Segmentation Only

**Goal**: Generate automatic segmentations without DVH

**Steps**:
1. Upload: CT files only (no RT data needed)
2. Configure: Fast mode ✅, Radiomics ❌
3. Process: TotalSegmentator generates 100+ structures
4. Download: Find segmentation masks in `Segmentation_TotalSegmentator/`
5. Use: Import `RS_auto.dcm` into treatment planning system

### RTSTRUCT Variants (RS*.dcm)

Each course under `Data_Snakemake/<patient>/<course>/` contains four RTSTRUCT files. Their roles are:

| File | How it is created | What it contains | Typical use |
| --- | --- | --- | --- |
| `RS.dcm` | Copied verbatim during the **organize** stage. | The original clinical RTSTRUCT (manual contours, fractional beams, physician naming). | Regulatory archive, DVH/radiomics comparisons against physician ROIs. |
| `RS_auto.dcm` | Built from TotalSegmentator CT masks via `auto_rtstruct`. | 100+ automated structures aligned to the full CT volume. | Auto-contour import to TPS, DVH/radiomics when an automated ROI is allowed. |
| `RS_auto_cropped.dcm` | Written by `anatomical_cropping.py` whenever `ct_cropping.enabled: true`. Cropping extents live in `cropping_metadata.json`. | Same labels as `RS_auto.dcm`, but every mask is intersected with the systematic cropping box so voxel counts line up after trimming couch/legs. | Radiomics/DVH/QC when `ct_cropping.use_cropped_for_*` toggles are enabled. Cropped ROIs are flagged as `INFO` in QC reports. |
| `RS_custom.dcm` | Produced by `structure_merger.py` after blending manual, auto, and definitions from `custom_structures*.yaml`. | Clean superset of manual+auto contours plus derived Boolean ROIs (e.g., bowel bag, merged bones). | Default source for DVH, radiomics (Custom segmentations), and robustness perturbations. |

Because the files are separate, you can import whichever flavor matches your downstream workflow (manual QA vs. auto contours vs. cropped analytics) without losing the untouched clinical RTSTRUCT.

## Troubleshooting

### Web UI Won't Load

**Problem**: http://localhost:8080 shows "connection refused"

**Solutions**:
- Check container is running: `docker ps`
- Check logs: `docker logs rtpipeline`
- Restart container: `docker-compose restart`

### Upload Fails

**Problem**: Files won't upload or validation fails

**Solutions**:
- Check file size (max 50 GB per upload)
- Verify files are actually DICOM (use `pydicom` to test)
- Check browser console for errors (F12)

### Processing Stuck

**Problem**: Progress bar stuck at 0% or doesn't move

**Solutions**:
- View logs: Click "View Logs" button
- Check Docker logs: `docker logs rtpipeline`
- Verify Snakemake is running: `docker exec rtpipeline ps aux | grep snakemake`

### Out of Memory

**Problem**: Container crashes or processing fails with memory errors

**Solutions**:
- Increase Docker memory limit (Docker Settings → Resources)
- Use Fast Mode for segmentation
- Process fewer patients at once
- Enable CT cropping to reduce volume size

### No GPU Detected

**Problem**: GPU not being used even though you have one

**Solutions**:
- Check nvidia-docker: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Use GPU-enabled compose: `docker-compose up -d` (not cpu-only profile)
- Check NVIDIA drivers: `nvidia-smi` on host

## Next Steps

### Learn More

- **[webui.md](webui.md)**: Complete Web UI documentation
- **[README.md](README.md)**: Full pipeline documentation
- **[docs/](docs/)**: Technical documentation

### Advanced Features

- **Custom Models**: Add your own nnUNet segmentation models
- **Custom Structures**: Define composite structures via YAML
- **CT Cropping**: Standardize anatomical regions for consistent DVH
- **Radiomics Parameters**: Customize feature extraction

### Get Help

- **GitHub Issues**: https://github.com/kstawiski/rtpipeline/issues
- **Documentation**: See `docs/` directory
- **Examples**: Check `Code/` directory for Jupyter notebooks

## Tips for Success

1. **Start Small**: Test with 1-2 patients first
2. **Use Fast Mode**: Unless you have a powerful GPU
3. **Check Validation**: Review warnings before processing
4. **Monitor Progress**: Watch logs if something seems wrong
5. **Save Results**: Download and backup your results
6. **Read QC Reports**: Check quality control warnings
7. **Organize Input**: Well-organized DICOM folders = better results

## System Requirements

### Minimum

- 4 CPU cores
- 8 GB RAM
- 20 GB disk space
- Docker 19.03+

### Recommended

- 8+ CPU cores
- 16 GB RAM
- 50+ GB disk space
- NVIDIA GPU with 8+ GB VRAM
- Docker with nvidia-container-toolkit

### For Large Datasets

- 16+ CPU cores
- 32+ GB RAM
- 100+ GB disk space
- NVIDIA GPU with 16+ GB VRAM

## FAQ

**Q: Can I use rtpipeline without Docker?**
A: Yes, but Docker is strongly recommended. Native installation requires manual setup of conda environments and dependencies.

**Q: Do I need a GPU?**
A: No, but GPU acceleration makes segmentation 5-10x faster. Use Fast Mode for CPU-only systems.

**Q: What DICOM modalities are supported?**
A: CT, MR, RTPLAN, RTDOSE, RTSTRUCT, REG. PT support is experimental.

**Q: Can I process data from different treatment planning systems?**
A: Yes, rtpipeline is vendor-agnostic and works with DICOM-RT from any TPS.

**Q: How long does processing take?**
A: Depends on dataset size and hardware. Typical single patient: 5-15 minutes (GPU) or 30-60 minutes (CPU).

**Q: Can I run multiple jobs simultaneously?**
A: Yes, the Web UI supports multiple concurrent jobs.

**Q: Where are my results stored?**
A: In the `Output/` directory on your host, mapped from `/data/output` in the container.

**Q: Is my data secure?**
A: Data stays on your machine. The container is isolated. For production use, implement additional security measures (HTTPS, authentication, encryption).

## License

Same as rtpipeline main project.

---

**Ready to get started?** 🚀

```bash
docker-compose up -d
# Then open http://localhost:8080 in your browser
```
