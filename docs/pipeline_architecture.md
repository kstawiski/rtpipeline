# Current Pipeline Architecture

## Overview
This repository packages a Snakemake workflow that drives the `rtpipeline` Python package to transform raw DICOM-RT studies into curated research outputs. Input data are organised per patient inside the directory declared as `dicom_root` (defaults to `Example_data`). For every patient folder the workflow:
- Organises plans, doses, structures and CT image sets into a reproducible course layout.
- Generates TotalSegmentator segmentations and converts them into DICOM RTSTRUCT files.
- Merges manual, automated and custom structure definitions with audit reports.
- Computes dose–volume histograms, interactive visualisations, radiomics features and quality-control reports.
- Materialises study-wide Excel/JSON summaries in addition to per-patient artefacts.

Outputs live under `output_dir` (defaults to `Data_Snakemake`) with logs written to `logs_dir` (`Logs_Snakemake`). The pipeline is self-contained and avoids relying on manually maintained documentation; behaviour is defined entirely by the code paths referenced below.

## Key Components
- `Snakefile` – declarative orchestration of every processing rule, resource limits and outcome fan-out.
- `config.yaml` – runtime configuration for directories, worker counts, segmentation/radiomics toggles and custom structure templates.
- `rtpipeline/` package – implementation of all heavy lifting (organisation, segmentation, RTSTRUCT synthesis, structure merging, DVH, radiomics, metadata exports, QC, visualisation).
- `envs/rtpipeline.yaml` & `envs/rtpipeline-radiomics.yaml` – conda environments used respectively by most rules and by PyRadiomics execution (NumPy 1.x constraint).

## Execution Flow
Snakemake enumerates `PATIENTS` by scanning subdirectories of `dicom_root`. For each patient the following order is enforced:
1. `organize_data` – invoke the `rtpipeline` CLI once to collect course data and build enriched metadata.
2. `segment_nifti` → `nifti_to_rtstruct` – create TotalSegmentator masks and synthesise `RS_auto.dcm`.
3. `merge_structures` – combine manual, automated and configured custom structures; write `RS_custom.dcm` and a comparison report.
4. `radiomics`, `dvh`, `visualization`, `quality_control` – analytic stages that depend on upstream artefacts.
5. Aggregations: `metadata_exports`, `summarize`, `qc_summary` – produce project-wide Excel/JSON tables.

Two housekeeping rules (`clean`, `clean_all`) are available to purge segmentation intermediates and full outputs when required.

## Patient Output Layout
Each patient ends up with a directory under `output_dir` containing at minimum:
```
<patient>/
  CT_DICOM/                         # organised CT image series
  RP.dcm, RD.dcm                    # summed or copied RTPLAN/RTDOSE
  RS.dcm (if present in source)     # manual structure set
  RS_auto.dcm                       # auto-generated RTSTRUCT from TotalSegmentator
  RS_custom.dcm                     # merged/custom structure set (falls back to RS_auto)
  TotalSegmentator_NIFTI/           # NIfTI masks direct from TotalSegmentator
  metadata.json                     # extended metadata harvested from DICOM headers
  case_metadata.json / .xlsx        # richer per-course metadata table
  Radiomics_CT.xlsx                 # PyRadiomics features per ROI
  dvh_metrics.xlsx                  # DVH metrics derived from dicompyler-core
  Axial.html                        # interactive DVH + axial overlays (with HTML fallbacks)
  qc_report.json                    # QC summary with DICOM consistency checks
  structure_comparison_report.json  # mapping of manual/auto/custom structure priorities
  structure_mapping.json            # emitted by the structure merger for traceability
```
Intermediate log files for each Snakemake rule are stored under `logs_dir` with names such as `organize_<patient>.log` or `radiomics_<patient>.log`.

## Stage Details
### organize_data
- **Inputs**: Patient-specific DICOM folder (`dicom_root/<patient>`).
- **Process**: Calls `rtpipeline` CLI with `--no-segmentation --no-dvh --no-visualize --no-radiomics` to run `rtpipeline.organize.organize_and_merge`. The CLI groups RT objects into “course” directories, optionally sums multi-fraction plans/doses, copies matched CT series, and writes `case_metadata.json`.
- **Post-processing**: The rule copies the first course (`course_*`) into `<output_dir>/<patient>`, preserving `CT_DICOM`, `RP.dcm`, `RD.dcm` and any manual `RS.dcm`. It amends `metadata.json` so downstream rules know where to find CT, dose, manual and automatic structure artefacts.
- **Outputs**: `CT_DICOM/`, `metadata.json`, `RP.dcm`, `RD.dcm`, `case_metadata.json`, `case_metadata.xlsx`.

### segment_nifti
- **Inputs**: Organised `CT_DICOM` and metadata.
- **Process**: Runs `TotalSegmentator` directly against the CT directory. GPU usage is auto-detected via `nvidia-smi`; otherwise CPU mode is enforced. Configurable flags from `config.segmentation` include `fast` and `roi_subset`. A lock file in `Logs_Snakemake/.totalsegmentator.lock` serialises concurrent runs to avoid exhausting VRAM.
- **Outputs**: Fresh `TotalSegmentator_NIFTI/` directory (existing folders are purged before regeneration). Typical contents include `segmentations.nii.gz` plus per-structure mask files.

### nifti_to_rtstruct
- **Inputs**: `TotalSegmentator_NIFTI/` and `CT_DICOM`.
- **Process**: Calls `rtpipeline.auto_rtstruct.build_auto_rtstruct`. The helper resamples TotalSegmentator labels onto the CT grid, converts masks with `rt-utils`, and writes `RS_auto.dcm`. It supports both multi-label volumes and individual ROI masks.
- **Outputs**: `RS_auto.dcm` placed beside the CT. If TotalSegmentator produced a DICOM-SEG file, it is normalised/copied into `RS_auto.dcm` instead of re-rasterising.

### merge_structures
- **Inputs**: `RS_auto.dcm`, optionally manual `RS.dcm`, CT directory.
- **Process**: Uses `rtpipeline.structure_merger.merge_patient_structures`. The merger:
  - Loads manual and automated ROI dictionaries, applies keyword-based priority rules (targets default to manual, most OARs default to auto).
  - Optionally loads `custom_structures` definitions (YAML) to append metadata placeholders for union/intersection/boolean operations. Geometry operations are currently limited; when an advanced merge fails, the Snakefile falls back to `rtpipeline.dvh._create_custom_structures_rtstruct` (legacy implementation) to preserve compatibility.
  - Writes `structure_mapping.json` and `structure_comparison_report.json` for auditing.
- **Outputs**: `RS_custom.dcm` (falls back to `RS_auto.dcm` when merging errors occur) and `structure_comparison_report.json`.

### radiomics
- **Inputs**: `CT_DICOM`, `RS_custom.dcm` (or `RS_auto.dcm` if custom missing).
- **Process**: Calls `rtpipeline.radiomics_conda.radiomics_for_course`. ROIs are rasterised with `rt-utils`, exported as temporary NRRD masks, and processed through `conda run -n rtpipeline-radiomics` so PyRadiomics executes in a NumPy 1.x stack. Tasks run in parallel (ProcessPool) unless `radiomics.sequential: true` is set or the `RTPIPELINE_RADIOMICS_SEQUENTIAL` environment variable is injected. Feature selection is driven by `radiomics.params_file`.
- **Outputs**: `Radiomics_CT.xlsx` containing one row per ROI plus metadata columns (e.g. `roi_name`, PyRadiomics features). Temporary NRRDs are deleted after extraction.

### dvh
- **Inputs**: `RP.dcm`, `RD.dcm`, `CT_DICOM`, `RS_custom.dcm`.
- **Process**: Executes `rtpipeline.dvh.dvh_for_course`, which wraps dicompyler-core with compatibility shims for pydicom 3. It calculates absolute and relative DVHs, standard statistics (Dmean/Dmax/D95), volumetric thresholds (V1–V60 Gy), integral dose and small-volume hot-spot metrics. Missing `RP`/`RD` files trigger creation of an empty Excel shell so downstream steps still succeed.
- **Outputs**: `dvh_metrics.xlsx`.

### visualization
- **Inputs**: `dvh_metrics.xlsx`, `RS_custom.dcm`, `CT_DICOM`.
- **Process**: Attempts `rtpipeline.visualize.visualize_course` to build a Plotly-based DVH dashboard (`Axial.html`). If DVH data are absent or Plotly fails, it falls back to `generate_axial_review`, and finally to a minimal HTML notice. HTML always ends up at `<patient>/Axial.html`.
- **Outputs**: `Axial.html`.

### quality_control
- **Inputs**: `CT_DICOM`, `metadata.json`.
- **Process**: `rtpipeline.quality_control.generate_qc_report` runs structural checks: file existence, CT modality consistency, RTPLAN/RTDOSE SOP class validation, frame-of-reference cross-checks and placeholder segmentation volume checks. Results are flattened with statuses `PASS/WARNING/FAIL` and issue lists per check.
- **Outputs**: `qc_report.json` stored alongside patient data; cleaned summaries feed `qc_summary` later.

### metadata_exports
- **Inputs**: All per-patient `metadata.json` files.
- **Process**: Delegates to `rtpipeline.meta.export_metadata`, which sweeps the original DICOM tree to produce multi-tab Excel exports (`plans.xlsx`, `structure_sets.xlsx`, `dosimetrics.xlsx`, `fractions.xlsx`, `metadata.xlsx`, `CT_images.xlsx`). It also writes consolidated `case_metadata_all.xlsx` and `case_metadata_all.json` into `<output_dir>/Data/`.
- **Outputs**: Excel/JSON files listed above under `<output_dir>/Data/`.

### summarize
- **Inputs**: `metadata.json`, `Radiomics_CT.xlsx`, `dvh_metrics.xlsx` for every patient.
- **Process**: Flattens the JSON metadata, concatenates per-patient radiomics sheets (appending a `patient` column) and merges DVH spreadsheets. Missing inputs yield empty workbooks rather than failures.
- **Outputs**: `metadata_summary.xlsx`, `radiomics_summary.xlsx`, `dvh_summary.xlsx` in the output root.

### qc_summary
- **Inputs**: `qc_report.json` for every patient.
- **Process**: Normalises QC JSON structures into tabular form, capturing statuses and issue descriptions per check.
- **Outputs**: `qc_summary.xlsx`.

### Cleaning rules
- `clean` removes intermediate segmentation folders (`nifti` and `TotalSegmentator_NIFTI`).
- `clean_all` purges the entire output and logs directories.

## Configuration Reference (`config.yaml`)
- `dicom_root`, `output_dir`, `logs_dir` – top-level paths.
- `workers` – caps non-segmentation parallelism; per-rule thread counts are derived from this and system cores (`SEGMENTATION_THREADS`, `RADIOMICS_THREADS`, `IO_THREADS`).
- `segmentation.fast` – adds `--fast` to TotalSegmentator for CPU-friendly acceleration.
- `segmentation.roi_subset` – restricts TotalSegmentator to a named subset (`-rs <roi ...>`).
- `radiomics.sequential` – forces sequential PyRadiomics extraction.
- `radiomics.params_file` – YAML passed to PyRadiomics to control feature selection and image filters.
- `custom_structures` – path to a YAML file (e.g. `custom_structures_pelvic.yaml`) consumed by the structure merger to tag derived ROIs.

Relative paths resolve against the project root. Empty/null entries disable optional behaviours.

## External Dependencies
The workflow expects the conda environments referenced in `envs/` to be available. At runtime the following tools/libraries are required:
- CLI tools: `TotalSegmentator`, `dcm2niix`, optional `nvidia-smi` for GPU detection.
- Python libraries: `rt-utils`, `SimpleITK`, `pydicom` (>= 3), `dicompyler-core`, `pyradiomics`, `numpy`, `pandas`, `plotly`, `matplotlib`, `rtpipeline` itself.
- The radiomics stage uses `conda run -n rtpipeline-radiomics` to execute within a NumPy 1.x environment for PyRadiomics compatibility.

## Concurrency & Fault Tolerance
- Organising, DVH, QC, visualisation and metadata export tasks run through `rtpipeline.utils.run_tasks_with_adaptive_workers`, which honours `workers` and system CPU counts.
- Segmentation enforces a lock to avoid parallel `TotalSegmentator` invocations; thread counts respect available cores. Environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, etc.) are pinned inside each rule to prevent over-subscription.
- Most analytic rules are resume-aware: Snakemake itself will skip steps when target files already exist, and module-level logic creates empty placeholder files when upstream data are missing so the DAG can complete.

## Aggregated Reporting
Beyond per-patient artefacts, the pipeline produces:
- `<output_dir>/metadata_summary.xlsx`, `radiomics_summary.xlsx`, `dvh_summary.xlsx`, `qc_summary.xlsx` – cross-patient tables for quick review.
- `<output_dir>/Data/metadata.xlsx`, `plans.xlsx`, `structure_sets.xlsx`, `dosimetrics.xlsx`, `fractions.xlsx`, `CT_images.xlsx`, `case_metadata_all.xlsx`, `case_metadata_all.json` – research-ready exports.

These files can be regenerated incrementally; rerunning Snakemake updates only the affected pieces.

## Extending the Pipeline
New rules can reuse existing modules from `rtpipeline/`. For example, additional analysis of radiomics outputs can depend on `Radiomics_CT.xlsx`, while custom QC rules can leverage the metadata stored in `metadata.json` and `case_metadata.json`. Ensure any new steps declare appropriate conda environments and respect the existing directory structure so downstream summary rules remain valid.
