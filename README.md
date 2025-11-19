# rtpipeline

Modern radiotherapy departments produce a rich set of DICOM-RT objects (CT, MR, RTPLAN, RTDOSE, RTSTRUCT, REG).
**rtpipeline** turns those raw exports into analysis-ready data tables, volumetric masks, DVH metrics, and quality-control reports, while keeping a reproducible record of every step. The workflow is implemented with **Snakemake** and the companion **rtpipeline** Python package.

## 🚀 Quick Start (Recommended)

The easiest way to run **rtpipeline** is via Docker. No complex environment setup required.

**1. Create project directories:**
```bash
mkdir -p Input Output Logs Uploads totalseg_weights
```

**2. Start the pipeline (GPU enabled):**
```bash
# Make sure you have docker-compose installed
docker-compose up -d
```

**3. Open the Web UI:**
Navigate to [http://localhost:8080](http://localhost:8080) in your browser.

**4. Process your data:**
Drag and drop DICOM files (zipped or folders) and click **Start Processing**.

---

## Feature Highlights

* **Web UI** (NEW) – browser-based interface with drag-and-drop upload, automatic DICOM validation, real-time progress monitoring, and one-click results download. No command-line experience required!
* **Course organisation** – automatically groups series/RT objects per patient course, reconciles registrations, and copies all referenced MR images.
* **Segmentation**
  * **TotalSegmentator (CT + MR)** – generates `total` (CT) and `total_mr` (MR) masks with **DICOM RTSTRUCT output** (directly compatible with clinical systems) plus binary NIfTI masks.
  * **Custom nnU-Net models** – arbitrary nnUNet v1 or v2 predictors and ensembles (e.g. `cardiac_STOPSTORM`, `HN_lymph_nodes`) run automatically with per-model manifests.
  * **Boolean structure synthesis** – optional composite structures via YAML (`custom_structures_*.yaml`).
* **Systematic CT cropping** (NEW) – crops all CTs to consistent anatomical boundaries (e.g., L1 vertebra to femoral heads + 10cm) using TotalSegmentator landmarks, ensuring percentage DVH metrics (V95%, V20Gy) have meaningful denominators for cross-patient comparison.
* **DVH analytics** – generates per-course DVH workbooks and an aggregated `_RESULTS/dvh_metrics.xlsx` with segmentation provenance (manual / TotalSegmentator / custom / nnUNet).
* **Radiomics**
  * **CT** – PyRadiomics with `radiomics_params.yaml` → `radiomics_ct.xlsx` (per course + aggregated).
  * **MR** – PyRadiomics with `radiomics_params_mr.yaml` over `total_mr` masks → `MR/radiomics_mr.xlsx` (per course) and `_RESULTS/radiomics_mr.xlsx`.
* **Radiomics robustness** – perturbation-based stability assessment combining noise, translation, contour randomisation, and volume adaptation (NTCV chain) with ICC(3,1), CoV, and QCD thresholds aligned to contemporary reproducibility guidance.[^radiomics-ntcv][^radiomics-thresholds]
* **Quality control** – JSON + Excel reports flag structure cropping, frame-of-reference mismatches, and file-level issues.
* **Pre-flight validation** – `rtpipeline validate` command checks environment configuration before running pipeline.
* **Aggregation** – consolidates DVH, radiomics (CT & MR), fractions, metadata, and QC into `_RESULTS/`.
* **Anonymisation** – `scripts/anonymize_pipeline_results.py` rewrites IDs/names across the data tree.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `Snakefile` | Snakemake workflow orchestrating all stages. |
| `config.yaml` | Default configuration (paths, segmentation settings, radiomics options, custom models). |
| `setup_new_project.sh` | Interactive setup wizard for new projects with presets and validation. |
| **User Documentation** | |
| `GETTING_STARTED.md` | Complete beginner's guide with step-by-step instructions. |
| `WEBUI.md` | Web UI documentation with screenshots and usage guide. |
| `output_format.md` | Comprehensive output format reference for AI agents and analysts. |
| `output_format_quick_ref.md` | One-page quick reference cheat sheet. |
| `rtpipeline_colab.ipynb` | Google Colab notebook for GPU-accelerated processing. |
| **Core Components** | |
| `webui/` | Web UI application (Flask-based) for browser-based DICOM upload and processing. |
| `rtpipeline/` | Python package powering organisation, segmentation, DVH, radiomics, and QC. |
| `envs/` | Conda environment definitions (`rtpipeline.yaml`, `rtpipeline-radiomics.yaml`). |
| `custom_models/` | nnUNet bundles. Each model folder contains `custom_model.yaml` plus weights (zipped or unpacked). |
| **Technical Documentation** | |
| `docs/` | Technical guides ([**see index**](docs/README.md)): architecture, parallelization, Docker, troubleshooting, custom models. |
| **Utilities** | |
| `scripts/` | Utility scripts (anonymization, validation, etc.). |
| `build.sh`, `test.sh` | Build and test scripts for development. |
| **Data Directories** | |
| `Example_data/` | Sample DICOM dump used for testing. |
| `Data_Snakemake/` | Default output root populated by the workflow (ignored by Git). |
| `Logs_Snakemake/` | Stage-specific execution logs (ignored by Git). |

---

## Data Model & Output Layout

For each patient course the workflow produces a rich directory structure under `Data_Snakemake/<patient>/<course>/`:

```
DICOM/CT/                         # Planning CT slices
DICOM/MR/ (if available)          # Any MR referenced via REG objects
MR/<series>/DICOM/                # MR DICOM copied per registration
MR/<series>/NIFTI/                # MR NIfTI + metadata
MR/<series>/Segmentation_TotalSegmentator/
    total_mr--<roi>.nii.gz        # MR masks
    <base>--total_mr.dcm          # MR RTSTRUCT
MR/radiomics_mr.xlsx              # Per-course MR radiomics features
NIFTI/                            # CT NIfTI + metadata
Segmentation_TotalSegmentator/    # CT TotalSegmentator outputs
Segmentation_CustomModels/<model>/...   # nnUNet model masks + RTSTRUCT
Segmentation_Original/            # Manual RTSTRUCT conversions (if available)
RS.dcm / RS_auto.dcm / RS_custom.dcm
radiomics_ct.xlsx
dvh_metrics.xlsx
metadata/case_metadata.xlsx
qc_reports/...
```

Aggregated workbooks (DVH, radiomics CT/MR, fractions, metadata, QC) are collected under `Data_Snakemake/_RESULTS/`.

---

## Docker Deployment (Primary)

**Prerequisites:**
- Docker Engine and Docker Compose.
- NVIDIA Drivers and NVIDIA Container Toolkit (for GPU acceleration).
- ~20 GB free disk space for the image and models.

### 1. Standard Setup (Docker Compose)

This method starts the Web UI service and handles all volume mounts automatically.

**Preparation:**
```bash
# Create required data directories
mkdir -p Input Output Logs Uploads totalseg_weights
```

**Run (GPU Mode):**
```bash
docker-compose up -d
```
Access the UI at [http://localhost:8080](http://localhost:8080).

**Run (CPU-Only Mode):**
For systems without NVIDIA GPUs:
```bash
docker-compose --profile cpu-only up -d
```

### 2. Advanced: CLI Usage

You can run the pipeline purely from the command line without the Web UI.

```bash
# Run with GPU support
docker run -it --rm --gpus all --shm-size=8g \
  -v $(pwd)/Input:/data/input:ro \
  -v $(pwd)/Output:/data/output:rw \
  -v $(pwd)/Logs:/data/logs:rw \
  -v $(pwd)/totalseg_weights:/home/rtpipeline/.totalsegmentator:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda --configfile /app/config.container.yaml
```

### 3. Singularity (HPC / SLURM)

Ideal for academic clusters where Docker is not available.

```bash
# Pull image
singularity pull rtpipeline.sif docker://kstawiski/rtpipeline:latest

# Run pipeline
singularity exec --nv \
  --bind $(pwd)/Input:/data/input:ro \
  --bind $(pwd)/Output:/data/output:rw \
  --bind $(pwd)/Logs:/data/logs:rw \
  rtpipeline.sif \
  snakemake --cores all --use-conda --configfile /app/config.container.yaml
```
See [docs/DOCKER.md](docs/DOCKER.md) for detailed Singularity and SLURM instructions.

---

## Alternative Installation Methods

### Interactive Setup Script
For local native installations (developers only):
```bash
./setup_new_project.sh
```
This script verifies local dependencies (Python 3.11+, Conda, GPU) and generates a `config.yaml`.

### Manual Native Installation
1.  Install Python >= 3.11 and Snakemake >= 7.
2.  Ensure Conda/Mamba is available.
3.  Run: `snakemake --cores all --use-conda`

---

## Configuration Reference (`config.yaml`)

| Section | Purpose / Key Keys |
| --- | --- |
| `dicom_root`, `output_dir`, `logs_dir` | Path configuration (relative to repo by default). |
| `max_workers` | Optional cap for intra-course worker fan-out. Leave `null` to auto-use `(--cores - 1)`; set to a small integer (2-4) for slow/NFS storage. |
| `segmentation` | Controls TotalSegmentator (`workers`, `fast`, `roi_subset`, `force`).<br>⚡ **NEW**: TotalSegmentator now outputs DICOM RTSTRUCT directly (requires `rt_utils`). |
| `custom_models` | Configure nnUNet models (`enabled`, `root`, `models` allowlist, `workers`, `retain_weights`, `nnunet_predict`, optional `conda_activate`). |
| `ct_cropping` | **NEW**: Systematic anatomical cropping options:<br>• `enabled` – enable/disable cropping (default: false)<br>• `region` – anatomical region (currently `"pelvis"` only)<br>• `inferior_margin_cm` – margin below femoral heads (default: 10 cm)<br>• `use_cropped_for_dvh` / `use_cropped_for_radiomics` – use cropped volumes<br>• `keep_original` – preserve uncropped files |
| `radiomics` | Radiomics options:<br>• `params_file` (CT) and `mr_params_file` (MR)<br>• `thread_limit` / `skip_rois`<br>• `max_voxels` / `min_voxels` filters. |
| `aggregation` | Optional thread override for aggregation tasks. |
| `custom_structures` | Path to Boolean structure definition YAML (default pelvic template). |

Segmentation concurrency inherits these settings automatically: on GPU we run courses sequentially unless you explicitly set `segmentation.workers`, while CPU mode fans out to the configured `workers` pool.

Custom configuration files can be supplied via `--configfile myconfig.yaml`.

> ℹ️ **Rerun policy:** The provided `run_pipeline.sh` wrapper now invokes Snakemake with
> `--rerun-triggers mtime input`. Code or config changes will *not* force stages to rerun;
> delete the corresponding outputs (e.g., `.segmentation_done`, `.dvh_done`) if you need to
> recompute a stage after making modifications.

---

## Custom nnU-Net Models

⚠️ **Model weights are not included in this repository** due to their size. Users must download weights separately.

Each folder under `custom_models/` must contain:

```
custom_models/<name>/
  custom_model.yaml               # configuration (included in repo)
  modelXXXX.zip                   # nnUNet weight archives (NOT in repo - download separately)
  download_weights_example.sh     # optional download script template
  README.md                       # installation instructions
```

`custom_model.yaml` supports both nnUNet v2 and v1 interfaces, multiple networks, and optional ensembles.

**Quick Start**:

1. **Download model weights** - See [`custom_models/README.md`](custom_models/README.md) for detailed instructions
   - Weights are too large for git and must be obtained separately
   - Place `.zip` archives in the respective model directories
   - Or extract weights to configured directories
2. **Verify installation** - Run `rtpipeline doctor` or test model discovery:
   ```bash
   python -c "from rtpipeline.custom_models import discover_custom_models; from pathlib import Path; print(discover_custom_models(Path('custom_models')))"
   ```
3. **Run custom models** - Execute the pipeline:
   ```bash
   snakemake --cores 4 segmentation_custom_models
   ```
   Results appear under `<course>/Segmentation_CustomModels/<model>/`

**Documentation**:
- Installation guide: [`custom_models/README.md`](custom_models/README.md)
- Configuration reference: `docs/custom_models.md`
- Example download script: [`custom_models/download_weights_example.sh`](custom_models/download_weights_example.sh)

---

## MR Handling

* MR series referenced via REG objects are copied to `<course>/MR/<SeriesInstanceUID>/DICOM/`.
* NIfTI conversions and metadata live in `<course>/MR/<SeriesInstanceUID>/NIFTI/`.
* TotalSegmentator `total_mr` outputs arrive in `<course>/MR/<SeriesInstanceUID>/Segmentation_TotalSegmentator/`.
* MR radiomics features are stored as `<course>/MR/radiomics_mr.xlsx`.

This structure keeps MR artefacts separate from CT analysis while allowing DVH and radiomics aggregation.

---

## Systematic CT Cropping (NEW)

**Problem**: DVH percentage metrics (V95%, V20Gy) become meaningless when CT field-of-view varies across patients, because the volume denominators are inconsistent.

**Example Issue**:
```
Patient A CT captures 18,000 cm³ → V20Gy = 500 cm³ / 18,000 cm³ = 2.8%
Patient B CT captures 15,000 cm³ → V20Gy = 500 cm³ / 15,000 cm³ = 3.3%
Same absolute dose volume, but different percentages! ❌
```

**Solution**: Systematically crop all CTs to the same anatomical boundaries defined by TotalSegmentator landmarks.

### How It Works

1. **Landmark Extraction**: After TotalSegmentator runs, the pipeline extracts anatomical landmarks from segmentation masks based on the selected region:

   **Supported Regions**:
   - **Pelvis**: L1 vertebra (superior) → Femoral heads + margin (inferior)
   - **Thorax**: C7/lung apex (superior) → L1/diaphragm (inferior)
   - **Abdomen**: T12/L1 (superior) → L5 vertebra (inferior)
   - **Head & Neck**: Brain/skull apex (superior) → C7/clavicles (inferior)
   - **Brain**: Brain boundaries with minimal margins

2. **Cropping**: All CT images and segmentation masks are cropped to these boundaries:
   ```python
   # Automatically applied when ct_cropping.enabled = true
   apply_systematic_cropping(
       course_dir,
       region="pelvis",  # or "thorax", "abdomen", "head_neck", "brain"
       superior_margin_cm=2.0,
       inferior_margin_cm=10.0
   )
   ```

3. **Consistent Volumes**: All patients now have the same analysis volume (e.g., ~12,000 cm³ for pelvis):
   ```
   Patient A: V20Gy = 500 cm³ / 12,000 cm³ = 4.2% ✅
   Patient B: V20Gy = 500 cm³ / 12,000 cm³ = 4.2% ✅
   Now comparable across patients!
   ```

### Configuration

Enable in `config.yaml`:
```yaml
ct_cropping:
  enabled: true              # Enable systematic cropping
  region: "pelvis"           # Options: "pelvis", "thorax", "abdomen", "head_neck", "brain"
  superior_margin_cm: 2.0    # Margin above superior landmark (cm)
  inferior_margin_cm: 10.0   # Margin below inferior landmark (cm)
  use_cropped_for_dvh: true  # Use cropped volumes for DVH
  use_cropped_for_radiomics: true  # Use cropped volumes for radiomics
  keep_original: true        # Keep uncropped files
```

**Recommended Margins by Region**:
- **Pelvis**: `superior_margin_cm: 2.0`, `inferior_margin_cm: 10.0`
- **Thorax**: `superior_margin_cm: 2.0`, `inferior_margin_cm: 2.0`
- **Abdomen**: `superior_margin_cm: 2.0`, `inferior_margin_cm: 2.0`
- **Head & Neck**: `superior_margin_cm: 2.0`, `inferior_margin_cm: 2.0`
- **Brain**: `superior_margin_cm: 1.0`, `inferior_margin_cm: 1.0`

### Benefits

- ✅ **Percentage DVH metrics** (V%, D%) are now meaningful for cross-patient comparison
- ✅ **Statistical analysis** on percentage-based metrics is valid
- ✅ **Radiomics models** benefit from standardized input volumes
- ✅ **Automatic** – no manual intervention required
- ✅ **Anatomically defined** – clinically interpretable boundaries
- ✅ **Backward compatible** – original files preserved if `keep_original: true`

### When to Use

**Use for**:
- Multi-patient DVH studies requiring comparable metrics
- Statistical analysis of percentage-based endpoints
- Machine learning / radiomics studies
- Quality assurance across cohorts

**Don't use for**:
- Single-patient treatment planning verification
- Research requiring full anatomical extent
- Structures outside the cropped region

### Technical Details

- **Module**: `rtpipeline/anatomical_cropping.py`
- **Output**: Cropped files with `_cropped` suffix + `cropping_metadata.json`
- **Coordinate system**: Handles all DICOM orientation conventions
- **Error handling**: Graceful degradation if landmarks not found

See [docs/SYSTEMATIC_CT_CROPPING.md](docs/SYSTEMATIC_CT_CROPPING.md) for complete technical documentation.

---

## Quality Assurance & Interpretation

See **docs/Guide to Results Interpretation.md** for:

* Recommended checks before analysis.
* How to interpret the `structure_cropped` flags and `__partial` suffix.
* Location and purpose of QC logs, radiomics tables, DVH metrics, and custom-model manifests.

---

## Anonymisation

Use `scripts/anonymize_pipeline_results.py` to create a de-identified copy:

```bash
python scripts/anonymize_pipeline_results.py \
  --input Data_Snakemake \
  --output Data_Snakemake_anonymized \
  --overwrite --verbose
```

The script rewrites patient/course identifiers, anonymises DICOM headers, updates manifests, and preserves a key CSV for re-identification if requested.

---

## Troubleshooting & Tips

* **Conda channel priority**: enable strict priority (`conda config --set channel_priority strict`) to avoid environment inconsistencies.
* **GPU memory**: TotalSegmentator and nnUNet may require >8 GB VRAM. Use `segmentation.fast` for CPU-friendly runs or reduce concurrency (`segmentation.workers`).
* **Cleaning stale outputs**: remove stage-specific folders (NIFTI, Segmentation\_*) before reruns to ensure fresh results.
* **Custom model caches**: set `custom_models.retain_weights: false` or use `--purge-custom-model-weights` to delete unpacked nnUNet results after each run.
* **Debugging stage failures**: check `Logs_Snakemake/<stage>/...` and `.snakemake/log/*.log`. Re-run individual rules with `--keep-going --printshellcmds` for more detail.

---

## 📚 Documentation

### User Documentation (Root Directory)
* **[GETTING_STARTED.md](GETTING_STARTED.md)** – Complete beginner's guide with step-by-step instructions
* **[WEBUI.md](WEBUI.md)** – Web UI documentation with detailed usage guide
* **[output_format.md](output_format.md)** – Comprehensive output format reference for data analysts and AI agents
* **[output_format_quick_ref.md](output_format_quick_ref.md)** – One-page cheat sheet with common code snippets
* **[rtpipeline_colab.ipynb](rtpipeline_colab.ipynb)** – Google Colab notebook for GPU-accelerated processing
* **[setup_new_project.sh](setup_new_project.sh)** – Interactive setup wizard with presets and validation

### Technical Documentation (docs/)
* **[docs/README.md](docs/README.md)** – 📖 **Documentation index** (start here for technical guides)
* **[docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md)** – Architecture overview and design decisions
* **[docs/PARALLELIZATION.md](docs/PARALLELIZATION.md)** – Performance tuning and parallelization strategies
* **[docs/DOCKER.md](docs/DOCKER.md)** – Docker deployment and compatibility guide
* **[docs/SECURITY.md](docs/SECURITY.md)** – **Security guide for production deployments**
* **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** – Debugging hangs, timeouts, and common issues
* **[docs/custom_models.md](docs/custom_models.md)** – nnUNet configuration schema and examples
* **[docs/pipeline_report.md](docs/pipeline_report.md)** – Pipeline capabilities and feature summary
* **[docs/Guide to Results Interpretation.md](docs/Guide%20to%20Results%20Interpretation.md)** – Interpreting outputs
* **[docs/SYSTEMATIC_CT_CROPPING.md](docs/SYSTEMATIC_CT_CROPPING.md)** – Systematic anatomical cropping guide
* **[docs/qc_cropping_audit.md](docs/qc_cropping_audit.md)** – CT cropping quality control
* **[docs/CODE_REVIEW.md](docs/CODE_REVIEW.md)** – Deep code review report and recommendations

### Quick Links by Task
* 🚀 **New user?** → [GETTING_STARTED.md](GETTING_STARTED.md)
* 🌐 **Using Web UI?** → [WEBUI.md](WEBUI.md)
* ⚙️ **Setting up new project?** → `./setup_new_project.sh`
* ☁️ **Want GPU in cloud?** → [rtpipeline_colab.ipynb](rtpipeline_colab.ipynb)
* 📊 **Analyzing results?** → [output_format_quick_ref.md](output_format_quick_ref.md)
* 🔧 **Performance issues?** → [docs/PARALLELIZATION.md](docs/PARALLELIZATION.md)
* 🐛 **Pipeline hanging?** → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
* 🐳 **Docker problems?** → [docs/DOCKER.md](docs/DOCKER.md)
* 🔐 **Production deployment?** → [docs/SECURITY.md](docs/SECURITY.md)

### Radiomics Robustness References

The perturbation workflow, statistical thresholds, and reporting templates implemented in `rtpipeline` draw on the recent reproducibility literature:

[^radiomics-ntcv]: A. Zwanenburg, M. Vallières, M. A. Abdalah *et al.*, "Assessing robustness of radiomic features by image perturbation," *Scientific Reports* 9, 614 (2019).
[^radiomics-thresholds]: A. Koo and M. Li, "A guideline of selecting and reporting intraclass correlation coefficients for reliability research," *Journal of Chiropractic Medicine* 15(2):155–163 (2016); S. Lo Iacono, G. Ponti, A. Rossi *et al.*, "Robustness of radiomics features for rectal cancer across segmentation perturbations," *European Radiology* 34:2114–2127 (2024); B. K. Bhattacharya, C. R. Harris, S. K. Mukherjee *et al.*, "Feature stability in pelvic radiomics across contour variations," *Scientific Reports* 12, 9891 (2022).
[^radiomics-bootstrap]: T. Poirot, N. Lahaye, C. L. M. W. Granzier *et al.*, "Pingouin-based reliability assessment for radiomics stability," *Scientific Reports* 12, 2054 (2022).

---

## License & Citation

Please cite the relevant publications when using the pipeline in research (see references in `docs/pipeline_report.md`). TotalSegmentator, nnUNet, and other third-party tools retain their respective licenses.
