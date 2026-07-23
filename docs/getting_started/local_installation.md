# Local installation: macOS, Linux, and WSL2

This tutorial installs and runs RTpipeline directly on a workstation without
Docker. It is intended for research workflows that need local filesystem or GPU
access. RTpipeline is not a medical device and its outputs require independent
quality control before any clinical interpretation.

The installer creates two isolated environments because the two core workloads
have incompatible NumPy requirements:

| Environment | Purpose | NumPy |
|---|---|---|
| `rtpipeline` | DICOM organization, TotalSegmentator, DVH, QC, orchestration | 2.x |
| `rtpipeline-radiomics` | PyRadiomics extraction and robustness statistics | 1.26 |

Do not merge these environments. The pipeline automatically launches radiomics
in the second environment.

## Supported systems

| System | Default compute backend | Important note |
|---|---|---|
| Apple Silicon macOS | MPS | GPU acceleration uses Apple Metal; CUDA is unavailable. |
| Intel macOS | CPU | Use `--device cpu`; large segmentation jobs can be slow. |
| Linux with a visible NVIDIA GPU | CUDA 12.1 | `nvidia-smi` must work before installation. |
| Linux without NVIDIA GPU | CPU | Plan substantial time and memory for segmentation. |
| WSL2 with a supported NVIDIA GPU | CUDA 12.1 | Install the Windows NVIDIA driver; do not install a Linux display driver inside WSL. |

Allow at least 30 GB for environments and model weights, plus space for DICOM
input and results. For routine cohorts, a Linux CUDA workstation or the
versioned container remains the fastest and most reproducible choice.

## 1. Obtain the release

Install Git if it is not already present, then clone the exact release rather
than a moving branch:

```bash
git clone --branch v2.2.3 --depth 1 https://github.com/kstawiski/rtpipeline.git
cd rtpipeline
```

If you downloaded the release source archive instead, extract it and open a
terminal in that directory.

### WSL2 host setup

Run these commands once in an Administrator PowerShell window, then restart
Windows if requested:

```powershell
wsl --install -d Ubuntu
wsl --update
```

For NVIDIA acceleration, install a current CUDA-capable **Windows** NVIDIA
driver before entering WSL. NVIDIA states that this is the only display driver
needed; installing a Linux NVIDIA display driver inside WSL can break the mapped
driver. See the current [Microsoft WSL GPU guide](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl)
and [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

Keep the repository and active data in the WSL Linux filesystem (for example,
`~/rtpipeline` and `~/rt-data`) rather than under `/mnt/c`; this avoids the
large small-file I/O penalty of crossing the Windows filesystem boundary.

## 2. Run the automatic installer

From the repository root:

```bash
bash scripts/install_local.sh
```

The script:

1. detects macOS, Linux, or WSL2 and chooses MPS, CUDA, or CPU;
2. installs missing `curl`, `tar`, and `bzip2` packages on supported Linux
   distributions;
3. downloads micromamba from the official mamba-org endpoint if needed;
4. creates or updates both named environments;
5. installs the platform-appropriate pinned PyTorch 2.3 stack;
6. installs PyRadiomics 3.0.1 in the NumPy 1.26 environment; and
7. verifies NumPy, PyTorch, RTpipeline, PyRadiomics, and Snakemake imports.

Micromamba's official documentation describes the same standalone installation
model and root-prefix behavior: [Micromamba installation](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html).

Useful installer controls:

```bash
# Inspect detection without changing the machine
bash scripts/install_local.sh --plan

# Force a backend
bash scripts/install_local.sh --device cpu
bash scripts/install_local.sh --device gpu   # Linux/WSL2 only
bash scripts/install_local.sh --device mps   # macOS only

# Verify an existing installation without updating it
bash scripts/install_local.sh --verify-only

# Use a non-default micromamba root
bash scripts/install_local.sh --mamba-root /absolute/path/to/micromamba
```

The installer does not need shell activation. Both provided scripts call the
named environments directly. It writes an explicit package receipt to:

```text
~/.local/state/rtpipeline/install-receipt-v2.2.3.txt
```

The receipt includes platform, backend, git commit/dirty state, micromamba's
explicit conda package URLs, `pip freeze`, and `pip inspect` inventories for
both environments, plus the PyRadiomics source hash when a local artifact is
used.

## 3. Create a local configuration

Copy the release configuration and edit the three path fields. Absolute paths
make the run easier to audit:

```bash
cp config.yaml config.local.yaml
```

```yaml
dicom_root: "/absolute/path/to/dicom-input"
output_dir: "/absolute/path/to/rtpipeline-output"
logs_dir: "/absolute/path/to/rtpipeline-logs"
```

For a manuscript radiomics analysis, retain the complete standard NTCV profile:

```yaml
radiomics_robustness:
  enabled: true
  modes:
    - segmentation_perturbation
  segmentation_perturbation:
    intensity: "standard"
    small_volume_changes: [-0.15, 0.0, 0.15]
    max_translation_mm: 4.0
    n_random_contour_realizations: 2
    noise_levels: [0.0, 10.0, 20.0]
```

This represents Gaussian noise at 0, 10, and 20 HU; rigid translations up to
+/-4 mm; two contour realizations; and -15%, 0%, and +15% volume adaptation.
The Cartesian standard grid requires 81 perturbations per ROI. RTpipeline fails
the course if any configured state cannot be generated or extracted. Do not
describe a run as complete NTCV unless all four axes actually ran for the
reported cohort.

Set the segmentation backend to match the installer:

```yaml
segmentation:
  device: "gpu"  # Linux/WSL2 NVIDIA
  # device: "mps"  # Apple Silicon
  # device: "cpu"  # CPU-only systems
```

## 4. Resolve the workflow before processing

First perform a dry run. This reads the input, resolves the complete DAG, and
prints commands without running jobs:

```bash
bash scripts/run_local.sh --config config.local.yaml --cores all --dry-run
```

Correct every missing path or configuration error before removing `--dry-run`.
The runner deliberately does not pass Snakemake's `--use-conda`: the supported
dual environments already exist, and the workflow resolves their Python
executables by name.

## 5. Run the pipeline

```bash
bash scripts/run_local.sh --config config.local.yaml --cores all
```

Limit concurrency on a laptop or network filesystem:

```bash
bash scripts/run_local.sh --config config.local.yaml --cores 4
```

Run a named Snakemake target by adding it at the end:

```bash
bash scripts/run_local.sh --config config.local.yaml --cores 4 radiomics_ct
```

Additional Snakemake flags belong after `--`:

```bash
bash scripts/run_local.sh --config config.local.yaml --cores 4 -- --keep-going
```

Do not close the terminal during processing. Review the configured logs if a
stage fails; an enabled robustness course receives a success sentinel only
after its complete grid is written. Final cohort tables are written under
`<output_dir>/_RESULTS/`.

## 6. Manuscript-ready provenance

Before analysis, freeze and retain:

- the exact RTpipeline tag and commit;
- `config.local.yaml`, structure mappings, and radiomics parameter YAMLs;
- the installer receipt for both environments;
- the input cohort definition and exclusion log; and
- hardware/backend information and the date of processing.

Copy the receipt into the analysis provenance directory without editing it:

```bash
mkdir -p /absolute/path/to/analysis-provenance
cp ~/.local/state/rtpipeline/install-receipt-v2.2.3.txt \
  /absolute/path/to/analysis-provenance/
cp config.local.yaml /absolute/path/to/analysis-provenance/
```

After processing, retain QC summaries, failure logs, the per-course raw
robustness parquet files, and the complete aggregate workbook/raw parquet pair.
Report the actual number of perturbations per ROI. The robustness stage fails
closed rather than silently excluding an incomplete perturbation or subject.

A suitable methods statement is:

> DICOM-RT preprocessing was performed with RTpipeline v2.2.3 (commit
> [COMMIT]) using separate NumPy 2.x processing and NumPy 1.26 PyRadiomics
> environments. Radiomics stability was assessed with the configured standard
> RTpipeline-adapted NTCV chain comprising Gaussian noise (0, 10, and 20 HU),
> superior-inferior rigid translations (0 and +/-4 mm), two reproducible random
> physical-space contour offsets, and distance-ranked adaptation to the closest
> voxel counts representing -15%, 0%, and +15% volume changes. All 81 configured
> states per analyzed ROI were required to complete.

Replace every bracketed field and adjust the text to the analysis that actually
ran. See the [reproducibility guide](../user_guide/reproducibility.md) for the
full submission checklist.

## Troubleshooting

### CUDA was requested but is unavailable

On Linux, confirm that `nvidia-smi` works before rerunning with `--device gpu`.
On WSL2, update WSL and the Windows NVIDIA driver; do not install a Linux display
driver in the distribution. CPU installation remains available with
`--device cpu`.

### MPS verification fails on macOS

Use a current macOS release on Apple Silicon and rerun the installer. If MPS is
still unavailable, install with `--device cpu`. PyTorch documents MPS as its
Metal backend: [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/mps.html).

### PyRadiomics compilation fails on macOS

The installer supplies conda compilers, CMake, Ninja, NumPy, and Cython before
building the pinned official source tag. Re-run once to recover an interrupted
download. If the same error persists, preserve the full build log and open an
issue with the macOS version, architecture, and installer receipt; do not merge
PyRadiomics into the main NumPy 2.x environment.

### TotalSegmentator runs out of memory

Keep `segmentation.force_split: true`, reduce `--cores`, and close other GPU or
memory-heavy applications. Docker's `--shm-size` requirement does not apply to
a native macOS install; on Linux/WSL2, verify that `/dev/shm` has adequate free
space when nnU-Net reports shared-memory errors.

### Processing data on a network filesystem is slow

Use `--cores 2` or `--cores 4`, stage active data on fast local storage when
permitted, and copy only finalized outputs back to network storage. Never make a
second output directory a symlink during a manuscript analysis.
