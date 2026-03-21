# CLAUDE.md - RTpipeline Development Context

This file provides context for AI assistants (Claude, etc.) working on this codebase.

## Project Overview

**RTpipeline** is a comprehensive radiotherapy DICOM processing pipeline that automates:
- DICOM organization and metadata extraction
- AI-powered organ segmentation (TotalSegmentator + custom nnU-Net models)
- DVH (Dose-Volume Histogram) analysis
- Radiomic feature extraction with robustness analysis
- Quality control and systematic CT cropping

### Primary Use Case
Processing radiotherapy treatment data for research and clinical decision support. The pipeline handles CT, MR, RTDOSE, RTSTRUCT, and RTPLAN DICOM files.

---

## Architecture

### Core Components

```
rtpipeline/
├── __init__.py           # Package init, version
├── config.py             # Configuration management (PipelineConfig class)
├── cli.py                # Command-line interface (click-based)
├── organize.py           # DICOM organization and course discovery
├── segment.py            # TotalSegmentator wrapper (CT/MR segmentation)
├── segment_custom_models.py  # Custom nnU-Net model execution
├── custom_structures.py  # Boolean structure combinations (e.g., bowel_bag)
├── dvh.py                # DVH extraction using dicompyler-core
├── radiomics_conda.py    # PyRadiomics feature extraction (conda subprocess)
├── radiomics_robustness.py   # Robustness analysis (ICC, CoV, QCD metrics)
├── crop_ct.py            # Systematic CT cropping by anatomical region
├── aggregate.py          # Results aggregation across patients
├── qc.py                 # Quality control checks
├── helpers.py            # Shared utilities (NIfTI I/O, mask operations)
├── dicom_to_rtstruct.py  # NIfTI mask → DICOM RTSTRUCT conversion
└── webui/                # Flask-based web interface
    ├── app.py            # Main Flask application
    ├── routes.py         # API endpoints
    └── templates/        # Jinja2 HTML templates
```

### Pipeline Stages (Snakemake DAG)

```
organize → segmentation → segmentation_custom_models → custom_structures
    ↓           ↓                    ↓                        ↓
    └───────────┴────────────────────┴────────────────────────┘
                                     ↓
                              crop_ct (optional)
                                     ↓
                                   dvh
                                     ↓
                    ┌────────────────┴────────────────┐
                    ↓                                  ↓
            radiomics_ct                        radiomics_mr
                    ↓                                  ↓
      radiomics_robustness_ct            radiomics_robustness_mr
                    ↓                                  ↓
                    └────────────────┬────────────────┘
                                     ↓
                                    qc
                                     ↓
                              aggregation
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Workflow | Snakemake |
| Container | Docker (kstawiski/rtpipeline:latest), Apptainer/Singularity |
| Segmentation | TotalSegmentator (nnU-Net), custom nnU-Net models |
| Radiomics | PyRadiomics (NumPy 1.x requirement) |
| DVH | dicompyler-core |
| DICOM | pydicom, SimpleITK, dcm2niix |
| Web UI | Flask + Bootstrap |
| GPU | CUDA 11.8+ (optional, CPU fallback available) |

### Dual Environment Architecture

PyRadiomics requires NumPy 1.x, while TotalSegmentator needs NumPy 2.x. Solution:

- **rtpipeline**: Main environment (NumPy 2.x, TotalSegmentator, SimpleITK)
- **rtpipeline-radiomics**: Isolated environment (NumPy 1.x, PyRadiomics)

Radiomics runs via conda subprocess (`radiomics_conda.py`).

---

## Key Configuration Files

### config.yaml (Main Configuration)

```yaml
# Paths
dicom_root: "Input"           # DICOM source directory
output_dir: "Output"          # Results destination
logs_dir: "Logs"              # Log files

# Parallelism
max_workers: null             # null = auto (cores - 1)

# Segmentation
segmentation:
  device: "gpu"               # gpu | cpu | mps
  fast: false                 # 3x faster, lower quality
  force_split: true           # Reduce GPU memory usage

# Radiomics Robustness
radiomics_robustness:
  enabled: true
  segmentation_perturbation:
    intensity: "standard"     # mild | standard | aggressive
    apply_to_structures: ["GTV*", "CTV*", "PTV*", "urinary_bladder"]
    small_volume_changes: [-0.15, 0.0, 0.15]  # ±15% erosion/dilation

# CT Cropping
ct_cropping:
  enabled: true
  region: "pelvis"            # pelvis | thorax | abdomen | head_neck | brain
```

### custom_structures_*.yaml (Boolean Structure Definitions)

```yaml
structures:
  bowel_bag:
    operation: union
    sources: [small_bowel, colon, duodenum]

  pelvic_bones:
    operation: union
    sources: [hip_left, hip_right, sacrum]

  pelvic_bones_3mm:
    operation: margin
    source: pelvic_bones
    margin_mm: 3.0
```

---

## Common Development Tasks

### Running the Pipeline

```bash
# Docker (recommended)
docker-compose up

# Local with Snakemake
snakemake --cores all --use-conda --configfile config.yaml

# Specific target
snakemake --cores all radiomics_ct
```

### Testing Changes

```bash
# Create test config
cp config.yaml config.test.yaml
# Edit: set dicom_root to test data

# Run in Docker
docker run --rm --gpus all \
  -v $(pwd)/Example_data:/data/input:ro \
  -v $(pwd)/Output_Test:/data/output:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores 16 --configfile /app/config.container.yaml
```

### Building Docker Image

```bash
./build.sh              # Build locally
./build.sh --push       # Build and push to Docker Hub
./build.sh --no-cache   # Clean build
```

### Web UI Development

```bash
# Start development server
cd webui && flask run --debug --port 5000

# Or via Docker
docker-compose --profile webui up
```

---

## Code Patterns

### 1. Configuration Access

```python
from rtpipeline.config import PipelineConfig

config = PipelineConfig.from_yaml("config.yaml")
workers = config.effective_workers()  # Handles null → auto
device = config.segmentation.device
```

### 2. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import os

def process_item(item):
    # Set thread limits to prevent oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # ... processing logic

with ProcessPoolExecutor(max_workers=config.effective_workers()) as executor:
    results = list(executor.map(process_item, items))
```

### 3. NIfTI I/O with Caching

```python
from rtpipeline.helpers import load_nifti_cached, save_nifti_like

# Thread-safe cached loading
data, affine, header = load_nifti_cached(nifti_path)

# Save with reference geometry
save_nifti_like(new_data, reference_path, output_path)
```

### 4. Radiomics Conda Fallback

```python
from rtpipeline.radiomics_conda import radiomics_for_course

# Runs in rtpipeline-radiomics conda environment
result = radiomics_for_course(course_dir, config)
```

### 5. Mask Perturbations (Robustness)

```python
from rtpipeline.radiomics_robustness import volume_adapt_mask

# Erode/dilate mask to target volume change
perturbed = volume_adapt_mask(
    mask,
    target_volume_change=-0.15,  # -15% volume
    spacing=(1.0, 1.0, 1.0)
)
```

---

## Output Structure

```
Output/
├── _COURSES/
│   └── manifest.json           # Course discovery manifest
├── _RESULTS/
│   ├── dvh_metrics.xlsx        # Aggregated DVH metrics
│   ├── radiomics_ct.xlsx       # Aggregated CT radiomics
│   ├── radiomics_mr.xlsx       # Aggregated MR radiomics
│   ├── radiomics_robustness.xlsx  # Robustness summary
│   └── qc_summary.xlsx         # Quality control
└── <patient>/
    └── <course>/
        ├── CT/                 # Converted CT NIfTI
        ├── MR/                 # MR series (if present)
        ├── RTDOSE/             # Dose grids
        ├── RTSTRUCT/           # Structure sets
        ├── Segmentation_TotalSegmentator/  # AI contours
        ├── Segmentation_CustomModels/      # Custom model outputs
        ├── dvh_metrics.xlsx
        ├── radiomics_ct.xlsx
        ├── radiomics_robustness/
        │   └── perturbation_results.parquet
        └── metadata/
            ├── case_metadata.xlsx
            └── custom_structure_warnings.json
```

---

## Known Issues & Solutions

### 1. PyRadiomics NumPy Compatibility

**Problem**: PyRadiomics requires NumPy 1.x, TotalSegmentator needs 2.x
**Solution**: Dual conda environments, subprocess execution in `radiomics_conda.py`

### 2. GPU Memory (OOM)

**Problem**: Large CT volumes exhaust GPU memory
**Solution**: Set `segmentation.force_split: true` in config.yaml

### 3. Docker Multiprocessing

**Problem**: Python multiprocessing fails in Docker
**Solution**: Keep `num_proc_preprocessing: 1` and `num_proc_export: 1`

### 4. Robustness ICC = 1.0 (False Positives)

**Problem**: `volume_adapt_mask()` was returning original mask when perturbation failed
**Solution**: Fixed in Nov 2025 - function now properly applies morphological operations

### 5. Slow NFS Storage

**Problem**: High I/O parallelism overwhelms network storage
**Solution**: Set `max_workers: 2-4` for NFS environments

### 6. TotalSegmentator /dev/shm Requirement (CRITICAL)

**Problem**: TotalSegmentator uses nnU-Net v2 which requires `/dev/shm` (POSIX shared memory) for PyTorch multiprocessing data loaders. Default Docker containers have only 64MB `/dev/shm`, causing `RuntimeError: unable to write to file </torch_*>: No space left on device`.

**Workarounds that DO NOT work:**
- `--totalseg-num-proc-pre 0 --totalseg-num-proc-export 0` — flags don't propagate to nnU-Net internals
- `nnUNet_def_n_proc=0` — causes `torch.set_num_threads(0)` which is invalid
- `nnUNet_def_n_proc=1` — nnU-Net still uses shared memory for inter-process communication
- `torch.multiprocessing.set_sharing_strategy('file_system')` — CUDA tensors still use `/dev/shm`
- Apptainer with `--bind /tmp/shm:/dev/shm` inside Docker — can't mount `/proc` (no capabilities)

**Solution**: Ensure `/dev/shm` is at least 4GB. For Docker: `docker run --shm-size=4g` or `--shm-size=64g`. For existing containers: request host admin to resize.

**Running TotalSegmentator with micromamba:**
```bash
~/.local/bin/micromamba run -n rtpipeline \
  TotalSegmentator -i input.nii.gz -o output_dir/ \
  --device gpu --force_split --nr_thr_resamp 1 --nr_thr_saving 1
```

**Environment variables for nnU-Net:**
```bash
export nnUNet_def_n_proc=1          # Minimize multiprocessing workers
export OMP_NUM_THREADS=1            # Prevent OpenMP oversubscription
export MKL_NUM_THREADS=1            # Prevent MKL oversubscription
```

### 7. dicompyler-core + pydicom 3.x Incompatibility

**Problem**: dicompyler-core 0.5.6 imports `pydicom.dicomio.read_file` which was removed in pydicom 3.x. Falls back to `import dicom` (ancient package).
**Solution**: Patch `dicompylercore/dicomparser.py` to use `from pydicom import dcmread as read_file`. Applied in rtpipeline env at:
`~/micromamba/envs/rtpipeline/lib/python3.11/site-packages/dicompylercore/dicomparser.py`

### 8. Organize Flat File Symlinks

**Problem**: DVH module expects `RP.dcm`, `RD.dcm`, `RS.dcm` at course directory root. If organize ran in resume mode, these flat copies may not exist (only files in `DICOM/RTDOSE/`, `DICOM/RTPLAN/`, `DICOM/RTSTRUCT/`).
**Solution**: Create symlinks manually or re-run organize with `--force-redo`.

### 9. Radiomics Conda Subprocess Requires `conda` Command

**Problem**: `radiomics_conda.py` hardcodes `"conda"` for subprocess calls. With micromamba, `conda` is not found.
**Solution**: Created wrapper at `/home/konrad/.local/bin/conda` that delegates to micromamba:
```bash
#!/bin/bash
exec /home/konrad/.local/bin/micromamba "$@"
```

### 10. RTpipeline CLI Syntax

**Correct syntax** (click-based CLI):
```bash
python -m rtpipeline.cli --dicom-root /path/to/dicom --outdir /path/to/output --stage organize
```
- Use `--stage` flag, not subcommands
- Output dir is `--outdir`, NOT `--output-dir`
- Stages: `organize`, `segmentation`, `dvh`, `radiomics`, `robustness`, `qc`, `aggregate`

### 11. Robustness Analysis Requires Conda Fallback for NumPy 2.x

**Problem**: `radiomics_robustness.py` called `_extractor()` which returns `None` with NumPy 2.x,
but had no fallback to the conda subprocess — all perturbation feature extractions were silently skipped.
**Solution**: Fixed `extract_features_for_masks()` to use `extract_radiomics_batch_with_conda()` when
`_extractor()` returns None. Saves perturbed images/masks as temp NRRD files, extracts all perturbations
for a structure in a single conda subprocess call.
**Performance**: ~81 perturbations per structure (standard intensity) × 8 structures = ~648 extractions.
Single-patient robustness takes several hours. Plan cluster execution for full cohort.

---

## Testing Checklist

Before committing changes:

- [ ] `snakemake --lint` passes
- [ ] Pipeline runs on Example_data without errors
- [ ] Docker image builds successfully
- [ ] Web UI starts and responds
- [ ] Radiomics extraction completes (tests conda fallback)
- [ ] Robustness analysis produces valid ICC values (not all 1.0)

---

## Docker Quick Reference

```bash
# Build image
./build.sh

# Run pipeline
docker run --rm --gpus all \
  -v /path/to/dicom:/data/input:ro \
  -v /path/to/output:/data/output:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores all

# Interactive shell
docker run -it --gpus all kstawiski/rtpipeline:latest bash

# Web UI
docker run -p 5000:5000 kstawiski/rtpipeline:latest \
  python -m webui.app
```

---

## Recent Changes (Nov 2025)

1. **Robustness Bug Fix**: Fixed `volume_adapt_mask()` to properly apply perturbations
2. **Performance**: Added Parquet caching, thread-safe NIfTI loading
3. **ICC Computation**: Switched to Pingouin library for accurate ICC(3,1)
4. **MR Radiomics**: Fixed conda environment fallback for MR series

---

## File Locations Reference

| Purpose | Location |
|---------|----------|
| Main workflow | `Snakefile` |
| Configuration | `config.yaml` |
| Python package | `rtpipeline/` |
| Web UI | `webui/` |
| Docker build | `Dockerfile`, `build.sh` |
| Conda environments | `envs/` |
| Documentation | `docs/` |
| Custom structures | `custom_structures_*.yaml` |
| Custom models | `custom_models/` |

---

## Manuscript Analysis Rules (MANDATORY)

These rules apply to ALL manuscript-related work. No exceptions.

1. **No inferior analyses, no skipping, no fallbacks.** Everything must be done in the best possible way. Technical issues are not an excuse — fix them.
2. **Dual conda environments required.** `rtpipeline` (NumPy 2.x, TotalSegmentator) and `rtpipeline-radiomics` (NumPy 1.x, PyRadiomics). This is how the pipeline solves the NumPy incompatibility.
3. **All output goes to `/umed-projekty/DICOMRT-datasets/`.** Never use symlinks for output directories. Input symlinks to `/projekty/` are OK.
4. **Local development + Apptainer for processing.** This machine (`cdp-umed-worker`) is inside a Docker container. Use micromamba (`~/.local/bin/micromamba`) for local envs, Apptainer for containerized pipeline runs.
5. **RTpipeline code modifications happen locally**, then rebuild Docker/Apptainer image as needed.
6. **Full NTCV perturbation chain for all cohorts.** Noise [0, 10, 20 HU] + Translation ±4mm + 2 Contour realizations + ±15% Volume. No partial perturbation sets.
7. **Delegate aggressively to Codex.** Use codex-delegator for implementation tasks. Opus orchestrates, Codex implements.
8. **Manuscript plan is at `manuscript/PLAN.md` v3.0.** This is the authoritative reference. All analysis must follow it exactly.
9. **High CPU load tasks must be delegated to Argos cluster** (`/argos-cluster` skill). Transfer required files temporarily to `/home/kgs24` on `argos-worker` (Tailscale: 100.64.0.5), process there, retrieve results, then **delete temporary files from Argos** to conserve space. Argos has 822 CPUs across 28 nodes — use for full-scale pipeline processing (Phase 4).

## Infrastructure (cdp-umed-worker)

| Component | Status |
|-----------|--------|
| GPU | NVIDIA H100 NVL 96GB (driver 560.35.05). **Safe to parallelize TotalSegmentator** — sufficient VRAM for multiple concurrent inference tasks. |
| Apptainer | 1.4.5 (at /home/linuxbrew/.linuxbrew/bin/apptainer) |
| Micromamba | 2.5.0 (at ~/.local/bin/micromamba) |
| Codex CLI | 0.107.0 |
| Python | 3.14.3 (linuxbrew) |
| RTpipeline | 2.1.0 (installed in system Python) |
| Storage | 2.7 PB NFS (/umed-projekty/), 2% used |
| Docker | NOT available (we are inside a container) |
| Argos HPC | 822 CPUs, 28 nodes. Access via SSH/Tailscale (`argos-worker` 100.64.0.5), user `kgs24`, temp dir `/home/kgs24`. **Clean up after use.** |

---

## Contact & Resources

- **Repository**: https://github.com/kstawiski/rtpipeline
- **Docker Hub**: https://hub.docker.com/r/kstawiski/rtpipeline
- **Documentation**: See `docs/` directory
