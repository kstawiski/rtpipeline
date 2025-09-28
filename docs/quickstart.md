# Quickstart

This project targets Python 3.10+ and assumes access to TotalSegmentator and
`dcm2niix`. The easiest way to run the pipeline is via Snakemake, which
constructs the required conda environments and handles parallel execution.

## 1. Clone and inspect configuration

```bash
git clone <repo-url>
cd KONSTA_DICOMRT_Processing
```

Edit `config.yaml` to point `dicom_root` at your study tree and to adjust
`output_dir`, `logs_dir`, worker counts, segmentation flags, or radiomics
parameters. The defaults write results into `Data_Snakemake` and logs into
`Logs_Snakemake` underneath the project root.

## 2. Run the Snakemake workflow

```bash
XDG_CACHE_HOME=$PWD/.cache \
snakemake --use-conda --cores 4 --conda-prefix $HOME/.snakemake_envs
```

- `--use-conda` tells Snakemake to create the two environments declared under
  `envs/` (`rtpipeline` with NumPy 2.x for TotalSegmentator and
  `rtpipeline-radiomics` with NumPy 1.26 for PyRadiomics).
- `--cores` limits how many rules Snakemake schedules concurrently. Individual
  rules also respect the `workers` value from `config.yaml` and internal thread
  caps defined in the `Snakefile`.
- Outputs land under `output_dir/<PatientID>/`, one directory per patient, and
  aggregated XLSX/JSON files are written into `output_dir/Data/`.
- Logs for each rule are stored in `logs_dir` with filenames such as
  `segment_nifti_<patient>.log`.

Rerunning the command is safe: Snakemake skips rules whose outputs already
exist. Use the `clean` or `clean_all` rules from the `Snakefile` if you need to
purge intermediate data first.

## 3. Optional: use the CLI directly

Install the package in editable mode (handy for development):

```bash
pip install -e .
```

Then invoke the CLI:

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs
```

The CLI organises data into course directories inside `--outdir` and, by
default, skips work for courses that already contain expected artefacts (metadata,
segmentation, DVH, visualisation, radiomics). Pass `--force-redo` to rebuild
everything or `--force-segmentation` to rerun TotalSegmentator while leaving
other stages cached.

## 4. Radiomics dependency notes

The Snakemake workflow launches radiomics extraction through a dedicated conda
environment that pins NumPy 1.26. If you operate the CLI outside Snakemake,
ensure `conda run -n rtpipeline-radiomics` works or install PyRadiomics in a
compatible environment yourself. Radiomics can be disabled for CLI runs via
`--no-radiomics`; with Snakemake you would need to edit the workflow if you want
to skip the rule entirely.

## 5. After the run

Key outputs include:
- Per-patient folders with DICOM copies, `RS_auto.dcm`, merged
  `RS_custom.dcm`, DVH/radiomics/QC spreadsheets, and HTML viewers.
- Aggregated spreadsheets (`metadata_summary.xlsx`, `radiomics_summary.xlsx`,
  `dvh_summary.xlsx`, `qc_summary.xlsx`) plus raw exports in
  `output_dir/Data/`.

Review the [Pipeline Architecture](pipeline_architecture.md) guide for a
stage-by-stage explanation of how these files are generated.
