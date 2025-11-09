# rtpipeline

Modern radiotherapy departments produce a rich set of DICOM-RT objects (CT, MR, RTPLAN, RTDOSE, RTSTRUCT, REG).  
**rtpipeline** turns those raw exports into analysis-ready data tables, volumetric masks, DVH metrics, and quality-control reports, while keeping a reproducible record of every step. The workflow is implemented with **Snakemake** and the companion **rtpipeline** Python package.

---

## Feature Highlights

* **Course organisation** – automatically groups series/RT objects per patient course, reconciles registrations, and copies all referenced MR images.
* **Segmentation**
  * **TotalSegmentator (CT + MR)** – generates `total` (CT) and `total_mr` (MR) masks, DICOM-SEG/RTSTRUCT files, and binary NIfTI masks.
  * **Custom nnU-Net models** – arbitrary nnUNet v1 or v2 predictors and ensembles (e.g. `cardiac_STOPSTORM`, `HN_lymph_nodes`) run automatically with per-model manifests.
  * **Boolean structure synthesis** – optional composite structures via YAML (`custom_structures_*.yaml`).
* **DVH analytics** – generates per-course DVH workbooks and an aggregated `_RESULTS/dvh_metrics.xlsx` with segmentation provenance (manual / TotalSegmentator / custom / nnUNet).
* **Radiomics**
  * **CT** – PyRadiomics with `radiomics_params.yaml` → `radiomics_ct.xlsx` (per course + aggregated).
  * **MR** – PyRadiomics with `radiomics_params_mr.yaml` over `total_mr` masks → `MR/radiomics_mr.xlsx` (per course) and `_RESULTS/radiomics_mr.xlsx`.
* **Quality control** – JSON + Excel reports flag structure cropping, frame-of-reference mismatches, and file-level issues.
* **Aggregation** – consolidates DVH, radiomics (CT & MR), fractions, metadata, and QC into `_RESULTS/`.
* **Anonymisation** – `scripts/anonymize_pipeline_results.py` rewrites IDs/names across the data tree.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `Snakefile` | Snakemake workflow orchestrating all stages. |
| `config.yaml` | Default configuration (paths, segmentation settings, radiomics options, custom models). |
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
| `segmentation` | Controls TotalSegmentator (`workers`, `threads_per_worker`, `fast`, `roi_subset`, `force`). |
| `custom_models` | Configure nnUNet models (`enabled`, `root`, `models` allowlist, `workers`, `retain_weights`, `nnunet_predict`, optional `conda_activate`). |
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
* **scripts/anonymize_pipeline_results.py --help** – anonymisation usage.

---

## License & Citation

Please cite the relevant publications when using the pipeline in research (see references in `docs/pipeline_report.md`). TotalSegmentator, nnUNet, and other third-party tools retain their respective licenses.

