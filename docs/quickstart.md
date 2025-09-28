# Quickstart

## Install

```bash
pip install -e .
```

This installs pydicom (>=3.0,<4), dicompyler-core, SimpleITK, pydicom-seg, plotly, dicom2nifti, TotalSegmentator (from GitHub via VCS), rt-utils, etc. Ensure your pip supports VCS dependencies (pip >= 21.1).
Note: `dcm2niix` is an external CLI (not a Python package). Install it via your OS package manager or conda (e.g., `conda install -c conda-forge dcm2niix`). If missing, the pipeline still runs DICOM-mode segmentation and skips NIfTI conversion. Alternatively, place platform ZIPs in `rtpipeline/ext/` (e.g., `dcm2niix_lnx.zip`, `dcm2niix_mac.zip`, `dcm2niix_win.zip`) and the tool will auto‑extract and use them when `dcm2niix` is not found.

## Run with Snakemake (recommended)

The repository ships with a Snakemake workflow that creates and pins the two
required conda environments (NumPy 2.x for TotalSegmentator, NumPy 1.x for
PyRadiomics). It also parallelises independent rules automatically.

```bash
cd /path/to/rtpipeline
XDG_CACHE_HOME=$PWD/.cache \
snakemake --use-conda --cores 4 --conda-prefix $HOME/.snakemake_conda_store
```

- Increase `--cores` if you want Snakemake to schedule more rules at once.
- `config.yaml:workers` controls how many worker processes stages such as
  radiomics or metadata aggregation may spawn; keep it ≤ `--cores` for best
  throughput.
- The shared `--conda-prefix` caches environments under your home directory so
  reruns do not rebuild them.
- Segmentation is resource-heavy; the workflow enforces a lock so only one
  TotalSegmentator job runs at a time. Other rules may execute in parallel
  while segmentation is idle.

## Minimal CLI run

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs -v
```

This will:
- Extract metadata (global XLSX under `outdir/Data/`)
- Organize per-patient, per-course folders under `outdir/<PatientID>/course_<key>/`
- (Optional) Run segmentation + auto RTSTRUCT (RS_auto)
- Compute DVH (manual + auto) and generate `DVH_Report.html`
- Generate `Axial.html` (CT + manual/auto overlays)

## Resume and force

- Segmentation is resume-safe: if outputs exist, it’s skipped.
- Force re-run:

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs --force-segmentation
```
## Optional: extra models and MR

- Run extra CT tasks (alongside default CT total):

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs \
  --extra-seg-models lung_vessels,body
```

- Run MR tasks across MR series (tasks ending with `_mr`):

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs \
  --extra-seg-models total_mr,body_mr
```

- Add license and CPU optimizations:

```bash
rtpipeline ... --totalseg-license YOUR_KEY --totalseg-fast --totalseg-roi-subset liver,pancreas
```

- Check environment:

```bash
rtpipeline doctor --logs ./Logs
```

## Optional: radiomics

Install pyradiomics from GitHub:

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics
```

Radiomics outputs:
- CT courses → `radiomics_features_CT.xlsx`
- MR series (with manual RS or `total_mr` seg) → `radiomics_features_MR.xlsx`

Disable with `--no-radiomics`.

When using the CLI directly, `--workers N` controls the non-segmentation
parallelism (organise/metadata/radiomics). With Snakemake, the value in
`config.yaml` is forwarded automatically.
