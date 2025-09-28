# CLI Reference

The `rtpipeline` command wraps the Python package so you can process a DICOM
study without Snakemake. It organises plans/doses/structures into per-course
folders, optionally runs TotalSegmentator, builds RTSTRUCTs, computes DVH,
produces visualisations, extracts radiomics features, and exports metadata.
Resume mode is **enabled by default**: existing outputs are reused unless you
explicitly force regeneration.

```bash
rtpipeline --dicom-root PATH --outdir PATH --logs PATH [options]
rtpipeline doctor [--logs PATH] [--conda-activate CMD] [--dcm2niix NAME] [--totalseg NAME]
```

## Core options
- `--dicom-root PATH` (required) – root containing patient subdirectories with
  DICOM files.
- `--outdir PATH` – destination for organised courses (default `./Data_Organized`).
- `--logs PATH` – log directory (default `./Logs`). A summary log is also written
  to `rtpipeline.log` under this directory.
- `-v/--verbose` – increase logging verbosity (repeatable).

## Course grouping
- `--merge-criteria {same_ct_study,frame_of_reference}` – grouping strategy for
  RT courses (default `same_ct_study`).
- `--max-days N` – split a course when consecutive plans are separated by more
  than *N* days.

## Stage toggles
- `--no-metadata` – skip XLSX/JSON metadata exports.
- `--no-segmentation` – do not invoke TotalSegmentator.
- `--force-segmentation` – re-run TotalSegmentator even when outputs exist.
- `--no-dvh` – skip DVH calculation.
- `--no-visualize` – skip HTML viewer generation (`DVH_Report.html`, `Axial.html`).
- `--no-radiomics` – skip radiomics extraction.
- `--force-redo` – rebuild every stage regardless of existing outputs. Without
  this flag, the pipeline operates in resume mode.

## Segmentation controls
- `--conda-activate CMD` – shell prefix (e.g. `"source ~/miniconda/etc/profile.d/conda.sh && conda activate ts"`).
- `--dcm2niix NAME` – explicit `dcm2niix` command (default `dcm2niix`).
- `--totalseg NAME` – TotalSegmentator command (default `TotalSegmentator`).
- `--totalseg-license KEY` – licence key exported during segmentation runs.
- `--totalseg-weights PATH` – location of pre-trained weights (nnUNet
  pretrained models) for offline setups.
- `--extra-seg-models MODEL[,MODEL...]` – request additional TotalSegmentator
  tasks alongside the default `total`. Repeat the flag or provide a
  comma-separated list. `_mr` suffixed models apply to MR series.
- `--totalseg-fast` – append `--fast` to TotalSegmentator for CPU-friendly runs.
- `--totalseg-roi-subset ROI[,ROI...]` – restrict TotalSegmentator to specific
  ROIs.

## Radiomics and structures
- `--radiomics-params FILE` – YAML settings for PyRadiomics (otherwise the
  packaged `rtpipeline/radiomics_params.yaml` is used).
- `--sequential-radiomics` – disable process-level parallelism for PyRadiomics.
- `--custom-structures FILE` – YAML describing Boolean or margin-derived
  structures. When omitted, the CLI tries the bundled
  `custom_structures_pelvic.yaml`.

## Parallelism
- `--workers N` – cap for non-segmentation worker pools (metadata, DVH,
  visualisation, radiomics). Defaults to an auto value based on CPU count.

## Outputs on disk
Each course is stored under `outdir/<PatientID>/course_<key>/` with:
- `CT_DICOM/` – copied planning CT series.
- `RP.dcm`, `RD.dcm` – summed or copied plan/dose files.
- `RS.dcm` (if a manual structure set was present at input).
- `TotalSegmentator_{DICOM,NIFTI}/`, optional extra-model folders, and `nifti/`
  (dcm2niix output).
- `RS_auto.dcm` – generated from TotalSegmentator outputs.
- `RS_custom.dcm` – merged manual/auto/custom structures plus
  `structure_comparison_report.json` and `structure_mapping.json`.
- Analytic artefacts: `dvh_metrics.xlsx`, `Radiomics_CT.xlsx`, `Axial.html`,
  `DVH_Report.html`, `qc_report.json`, metadata JSON/XLSX.
- Aggregated Excel files (e.g. `DVH_metrics_all.xlsx`) live under
  `outdir/Data/`.

## `doctor` subcommand
`rtpipeline doctor` checks whether core binaries and Python packages are
available. Use it to diagnose environments before running full jobs.

- Verifies Python version and prints package versions (`pydicom`, `SimpleITK`,
  `dicompyler-core`, `pydicom-seg`, `rt-utils`, `TotalSegmentator`).
- Checks `dcm2niix` and `TotalSegmentator` in `PATH` (or notes that a conda
  activation prefix is required).
- Reports CUDA availability via `nvidia-smi` and prints expected locations for
  bundled `dcm2niix` archives.
- Indicates whether a fallback extraction of `dcm2niix` from
  `rtpipeline/ext/*.zip` is possible when the binary is missing.

## Behaviour to remember
- Resume is default: the CLI only recomputes stages whose key outputs are
  missing. Combine `--force-redo` with stage toggles for selective rebuilds.
- Segmentation is sequential by design; DVH, visualisation, metadata, and
  radiomics use adaptive worker pools capped by `--workers`.
- Radiomics falls back to a conda-based execution when PyRadiomics is not
  importable in the main environment.
