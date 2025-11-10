# rtpipeline

Modern radiotherapy departments produce a rich set of DICOM-RT objects (CT, MR, RTPLAN, RTDOSE, RTSTRUCT, REG).
**rtpipeline** turns those raw exports into analysis-ready data tables, volumetric masks, DVH metrics, and quality-control reports, while keeping a reproducible record of every step. The workflow is implemented with **Snakemake** and the companion **rtpipeline** Python package.

## 🚀 Quick Start

**New to rtpipeline?** Start here:

```bash
# 1. Start the container with Web UI
docker-compose up -d

# 2. Open your browser to http://localhost:8080

# 3. Drag and drop your DICOM files

# 4. Click "Start Processing"

# 5. Download your results when complete
```

📖 **[Getting Started Guide](GETTING_STARTED.md)** - Complete beginner's guide
🌐 **[Web UI Documentation](WEBUI.md)** - Detailed Web UI features and usage

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
| `webui/` | Web UI application (Flask-based) for browser-based DICOM upload and processing. |
| `envs/` | Conda environment definitions (`rtpipeline.yaml`, `rtpipeline-radiomics.yaml`). |
| `rtpipeline/` | Python package powering organisation, segmentation, DVH, radiomics, and QC. |
| `custom_models/` | nnUNet bundles. Each model folder contains `custom_model.yaml` plus weights (zipped or unpacked). |
| `docs/` | User documentation (`Guide to Results Interpretation.md`, `custom_models.md`, `pipeline_report.md`). |
| `scripts/` | Utility scripts (e.g. anonymiser). |
| `Example_data/` | Sample DICOM dump used for testing. |
| `Data_Snakemake/` | Default output root populated by the workflow (ignored by Git). |
| `Logs_Snakemake/` | Stage-specific execution logs. |

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

## Pipeline Stages

1. **Organise (`organize_courses`)**
   * Identifies patient courses, copies relevant CT/RT/MR DICOM objects.
   * Converts CT and MR series to compressed NIfTI (`.nii.gz`) with metadata.
   * Logs course-level metadata (`case_metadata.json` / `.xlsx`).
2. **Segment (`segmentation_course`)**
   * Runs TotalSegmentator `total` on CT volumes.
   * For each MR NIfTI in `<course>/MR/<series>/NIFTI/`, runs TotalSegmentator `total_mr` and stores results under the same `MR/<series>/Segmentation_TotalSegmentator/`.
3. **Custom segmentation (`segmentation_custom_models`)**
   * Executes every model declared in `custom_models/` (NNUnet v1/v2, ensembles).
   * Writes masks & RTSTRUCT per course under `Segmentation_CustomModels/<model>/`.
4. **DVH (`dvh_course`)**
   * Computes DVH metrics for manual, TotalSegmentator (CT & MR), custom structures, and nnUNet outputs.
   * Saves per-course `dvh_metrics.xlsx`; aggregator merges into `_RESULTS/dvh_metrics.xlsx`.
5. **Radiomics (`radiomics_course`)**
   * CT: PyRadiomics (using `radiomics_params.yaml`) generates `radiomics_ct.xlsx`.
   * MR: PyRadiomics (using `radiomics_params_mr.yaml`) generates `MR/radiomics_mr.xlsx`, aggregated to `_RESULTS/radiomics_mr.xlsx`.
6. **Quality control (`qc_course`)**
   * Emits JSON reports and `_RESULTS/qc_reports.xlsx`.
7. **Aggregation (`aggregate_results`)**
   * Collates DVH, radiomics (CT/MR), fractions, metadata, QC into `_RESULTS/`.

All stages write sentinel files (`.organized`, `.segmentation_done`, `.custom_models_done`, etc.) to allow incremental reruns.

---

## Running the Workflow

### Docker / Singularity (Recommended)

rtpipeline is fully containerized for easy deployment. Both Docker and Singularity are supported.

#### Docker Setup

**1. Build the image:**
```bash
./build.sh
```

**2. Run with docker-compose (GPU - DEFAULT):**
```bash
# Start the container with GPU support (default)
docker-compose up -d

# Access the container
docker exec -it rtpipeline bash

# Inside container: run pipeline with container config
snakemake --cores all --use-conda --configfile /app/config.container.yaml
```

**3. Run with docker-compose (CPU-only):**
```bash
# For systems without GPU, use the cpu-only profile
docker-compose --profile cpu-only up -d
docker exec -it rtpipeline-cpu bash
```

**4. Web UI (NEW - Recommended for most users):**

rtpipeline now includes a browser-based web interface for easy file upload and job management:

```bash
# Start the container with Web UI (GPU mode)
docker-compose up -d

# Or CPU-only mode
docker-compose --profile cpu-only up -d

# Access the Web UI in your browser:
# http://localhost:8080
```

**Features:**
- 🎯 Drag-and-drop DICOM upload (.dcm, .zip, directories, DICOMDIR)
- ✅ Automatic DICOM validation with suggestions
- ⚙️ Configurable processing options (segmentation, radiomics, CT cropping)
- 📊 Real-time progress monitoring
- 📥 One-click results download
- 🔍 Integrated log viewer

See [WEBUI.md](WEBUI.md) for complete documentation.

**5. Run standalone container:**
```bash
# With GPU support (requires nvidia-docker or Docker >=19.03 with nvidia-container-toolkit)
docker run -it --rm --gpus all \
  -v $(pwd)/Input:/data/input:ro \
  -v $(pwd)/Output:/data/output:rw \
  -v $(pwd)/Logs:/data/logs:rw \
  kstawiski/rtpipeline:latest bash

# Inside container, run pipeline with container config:
snakemake --cores all --use-conda --configfile /app/config.container.yaml

# CPU-only (no GPU)
docker run -it --rm \
  -v $(pwd)/Input:/data/input:ro \
  -v $(pwd)/Output:/data/output:rw \
  kstawiski/rtpipeline:latest bash
```

**6. Build and push to Docker Hub:**
```bash
./build.sh --push --tag v1.0.0
```

**7. Pull from Docker Hub:**
```bash
docker pull kstawiski/rtpipeline:latest
```

#### Singularity Setup

**1. Convert from Docker image:**
```bash
# From local Docker image
singularity build rtpipeline.sif docker-daemon://kstawiski/rtpipeline:latest

# Or from Docker Hub
singularity build rtpipeline.sif docker://kstawiski/rtpipeline:latest
```

**2. Run the pipeline:**
```bash
# Interactive shell
singularity shell \
  --bind $(pwd)/Input:/data/input:ro \
  --bind $(pwd)/Output:/data/output:rw \
  --bind $(pwd)/Logs:/data/logs:rw \
  rtpipeline.sif

# Execute command directly
singularity exec \
  --bind $(pwd)/Input:/data/input:ro \
  --bind $(pwd)/Output:/data/output:rw \
  rtpipeline.sif \
  snakemake --cores all --use-conda --configfile /app/config.container.yaml
```

**3. Run with GPU (if available):**
```bash
singularity exec --nv \
  --bind $(pwd)/Input:/data/input:ro \
  --bind $(pwd)/Output:/data/output:rw \
  rtpipeline.sif \
  snakemake --cores 8 --use-conda --configfile /app/config.container.yaml
```

**4. HPC/SLURM example:**
```bash
#!/bin/bash
#SBATCH --job-name=rtpipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1  # Optional: request GPU

module load singularity

# Set your data paths
INPUT_DIR=/path/to/your/dicom/data
OUTPUT_DIR=/scratch/$USER/rtpipeline_output

singularity exec --nv \
  --bind ${INPUT_DIR}:/data/input:ro \
  --bind ${OUTPUT_DIR}:/data/output:rw \
  rtpipeline.sif \
  snakemake --cores $SLURM_CPUS_PER_TASK --use-conda --configfile /app/config.container.yaml
```

#### Jupyter Notebook (Optional)

```bash
# Start Jupyter service
docker-compose --profile jupyter up -d

# Access at http://localhost:8888
```

### Native Installation

1. **Prerequisites**
   * Python ≥ 3.11, Snakemake `>=7`.
   * Conda (or mamba) with `channel_priority strict`.
   * GPU strongly recommended for TotalSegmentator/nnUNet jobs.

2. **Initial run**
   ```bash
   ./test.sh            # convenience script: unlock + snakemake --cores all --use-conda
   ```

3. **Manual invocations**
   ```bash
   # organise DICOM into course layout (CT + MR)
   snakemake --cores 8 organize_courses

   # run segmentation (CT TotalSegmentator + MR total_mr)
   snakemake --cores 8 segmentation_course

   # run custom nnUNet models
   snakemake --cores 4 segmentation_custom_models

   # compute DVH + radiomics (CT & MR)
   snakemake --cores 8 dvh_course radiomics_course

   # combine results
   snakemake --cores 4 aggregate_results
   ```

4. **Selective reruns**
   ```bash
   snakemake --cores 4 --force segmentation_custom_models  # rerun custom models
   snakemake --cores 4 --force radiomics_course            # recompute radiomics only
   ```

5. **Cleaning**
   ```bash
   rm -rf Data_Snakemake/*/*/{NIFTI,Segmentation_TotalSegmentator,Segmentation_CustomModels,MR}
   snakemake --cores 8 organize_courses segmentation_course  # rebuild
   ```

---

## Configuration Reference (`config.yaml`)

| Section | Purpose / Key Keys |
| --- | --- |
| `dicom_root`, `output_dir`, `logs_dir` | Path configuration (relative to repo by default). |
| `workers` | Default CPU worker count for non-segmentation stages. |
| `segmentation` | Controls TotalSegmentator (`workers`, `threads_per_worker`, `fast`, `roi_subset`, `force`).<br>⚡ **NEW**: TotalSegmentator now outputs DICOM RTSTRUCT directly (requires `rt_utils`). |
| `custom_models` | Configure nnUNet models (`enabled`, `root`, `models` allowlist, `workers`, `retain_weights`, `nnunet_predict`, optional `conda_activate`). |
| `ct_cropping` | **NEW**: Systematic anatomical cropping options:<br>• `enabled` – enable/disable cropping (default: false)<br>• `region` – anatomical region (currently `"pelvis"` only)<br>• `inferior_margin_cm` – margin below femoral heads (default: 10 cm)<br>• `use_cropped_for_dvh` / `use_cropped_for_radiomics` – use cropped volumes<br>• `keep_original` – preserve uncropped files |
| `radiomics` | Radiomics options:<br>• `params_file` (CT) and `mr_params_file` (MR)<br>• `thread_limit` / `skip_rois`<br>• `max_voxels` / `min_voxels` filters. |
| `aggregation` | Optional thread override for aggregation tasks. |
| `custom_structures` | Path to Boolean structure definition YAML (default pelvic template). |

Custom configuration files can be supplied via `--configfile myconfig.yaml`.

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

See `SYSTEMATIC_CT_CROPPING.md` for complete technical documentation.

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

## Further Reading

* **docs/Guide to Results Interpretation.md** – deep dive into the per-course outputs and how to use them.
* **docs/custom_models.md** – nnUNet configuration schema and examples.
* **docs/pipeline_report.md** – narrative summary of the latest validation run and known issues.
* **SYSTEMATIC_CT_CROPPING.md** – complete technical documentation for anatomical cropping feature.
* **IMPROVEMENTS_SUMMARY.md** – summary of recent security fixes and feature additions.
* **DEEP_DEBUG_REPORT.md** – comprehensive security audit and code quality analysis.
* **scripts/anonymize_pipeline_results.py --help** – anonymisation usage.

---

## License & Citation

Please cite the relevant publications when using the pipeline in research (see references in `docs/pipeline_report.md`). TotalSegmentator, nnUNet, and other third-party tools retain their respective licenses.

