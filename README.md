# KONSTA_DICOMRT_Processing

## Overview
KONSTA_DICOMRT_Processing provides an end-to-end radiotherapy imaging pipeline that harmonizes DICOM-RT data, performs automated structure segmentation, computes dose–volume histograms (DVHs), extracts radiomics signatures, and aggregates course-level metadata. The workflow is expressed in Snakemake, enabling reproducible execution from discovery workstations through high-performance clusters while retaining full provenance of intermediate artefacts [1]. Clinical imaging operations leverage the `rtpipeline` Python package, which wraps TotalSegmentator-driven contouring and PyRadiomics feature extraction to generate analysis-ready tables and quality-control reports [2,3].

## Workflow Architecture
1. **Organize** – `rtpipeline.cli --stage organize` indexes CT, RTPLAN, RTDOSE, and RTSTRUCT objects, infers course boundaries, and exports harmonized metadata workbooks under `Data_Snakemake/Data`.
2. **Segmentation (TotalSegmentator)** – TotalSegmentator runs both DICOM and NIfTI pathways to construct auto-segmentation masks; bespoke structures are synthesized from Boolean operations defined in `custom_structures_*.yaml`. Auto-contours are written as `RS_auto.dcm` and mirrored into Excel summaries.
3. **Custom nnU-Net models** – The `segmentation_custom_models` stage scans `custom_models/` for nnUNetv2 configurations (see `custom_models.md`), extracts bundled weights, performs inference, and writes per-model masks plus `rtstruct.dcm` into `Data_Snakemake/<patient>/<course>/Segmentation_CustomModels/<model>/`. DVH and radiomics stages now ingest these outputs automatically, tagging metrics with `Segmentation_Source = CustomModel:<name>`.
4. **DVH computation** – Dose–volume metrics are computed per structure, merged with segmentation provenance (manual, TotalSegmentator, merged custom structures, and custom nnU-Net models), and exported as `dvh_metrics.xlsx` per course and in `_RESULTS/dvh_metrics.xlsx`.
5. **Radiomics extraction** – PyRadiomics executes via the dedicated `rtpipeline-radiomics` conda environment when native imports fail, producing per-ROI feature matrices across all segmentation sources and noting voxel-based exclusion criteria.
6. **Quality control** – Structured JSON reports and aggregated Excel sheets capture modality checks, frame-of-reference consistency, and structural cropping audits.
7. **Aggregation** – Course-level Excel workbooks (dose, fractions, metadata, QC) are collated into `_RESULTS`, with supplemental modality manifests surfaced under `Data_Snakemake/Data` for downstream statistical analysis.

Snakemake orchestrates these stages through sentinel files (`.organized`, `.segmentation_done`, `.custom_models_done`, `.dvh_done`, `.radiomics_done`, `.qc_done`) and a terminal aggregation rule specified in `Snakefile`.

## Repository Layout
| Path | Purpose |
| --- | --- |
| `Snakefile` | Canonical Snakemake definition connecting all pipeline stages. |
| `config.yaml` | Default configuration controlling IO paths, concurrency, segmentation, and radiomics options. |
| `envs/rtpipeline*.yaml` | Conda environment specifications for the main and radiomics stages. |
| `rtpipeline/` | Python package implementing orchestration, segmentation interfaces, DVH/radiomics calculators, and QC routines. |
| `custom_models/` | nnUNet model bundles and configuration (`custom_model.yaml`) consumed by the custom segmentation stage. |
| `Data_Snakemake/` | Default output tree populated by the latest test execution (see below). |
| `Logs_Snakemake/` | Stage-specific execution logs suitable for troubleshooting. |
| `Knowledge/` | Background domain notes (radiomics parameter research, modality guidance). |
| `docs/` | Supplemental technical reports (see `docs/pipeline_report.md`). |

## Software Environments
- `envs/rtpipeline.yaml`: Python 3.11 stack with NumPy ≥2.0, dicompyler-core, TotalSegmentator, and allied imaging toolkits.
- `envs/rtpipeline-radiomics.yaml`: Python 3.11 environment pinning NumPy 1.26 for PyRadiomics compatibility.
Conda environments are materialized dynamically by Snakemake under `$HOME/.snakemake_conda_store` during pipeline execution.

## Configuration
Key tunables reside in `config.yaml`:
- `dicom_root`, `output_dir`, `logs_dir`: define input and output roots; defaults target `Example_data` → `Data_Snakemake`.
- `workers`: maximum CPU workers for non-segmentation stages; segmentation has dedicated controls (`segmentation.workers`, `threads_per_worker`, `fast`, `extra_models`, `roi_subset`, `force`).
- `radiomics`: governs sequential vs. parallel execution, parameter file (`radiomics_params.yaml`), thread limits, ROI exclusion lists, and voxel-count thresholds.
- `aggregation.threads`: optional override for aggregation parallelism.
- `custom_structures`: YAML describing Boolean composites merged into DVH and radiomics outputs.
- `custom_models`: enables/disables custom nnU-Net inference, nominates a model root, concurrency limits, nnUNet CLI command, and cache retention. Set `retain_weights: false` to purge extracted `nnUNet_results`, `nnUNet_raw`, and `nnUNet_preprocessed` directories after each run (the default `true` caches weights to avoid re-extraction).

## Running the Pipeline
1. Ensure DICOM datasets reside beneath the configured `dicom_root` and that conda environments can be created in the designated prefix.
2. Execute `./test.sh` (POSIX) to unlock stale Snakemake locks and run the full pipeline with `--cores all`, `--use-conda`, and `--rerun-incomplete` enabled. The script aborts early if another Snakemake instance is active.
3. Alternatively, invoke Snakemake directly, e.g. `snakemake --cores 8 --use-conda aggregate_results` to regenerate only terminal artefacts once course sentinels exist.

## Test Run Summary (5 October 2025)
The bundled `Data_Snakemake/` tree captures the most recent comprehensive run driven by `./test.sh` on 5 October 2025. Organize logs confirm six courses across five patients. Aggregated metrics are summarized below:

| Patient | Course | Rx_Gy | Fractions | Kernel | Slice_mm | FOV_mm | Cropped_ROIs | AutoRTS_ROIs | Custom_ROIs | Manual_ROIs | Radiomics_rows | QC_status |
| ---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 416435 | 2018-09 | 50.4 | 0 | B31f | 5.0 | 500 | 63 | 54 | 57 | 21 | 52 | WARNING |
| 428459 | 2019-11 | 50.0 | 25 | B31f | 3.0 | 500 | 59 | 53 | 55 | 17 | 48 | WARNING |
| 480007 | 2024-08 | 50.0 | 25 | B31f | 3.0 | 500 | 30 | 32 | 31 | 22 | 27 | WARNING |
| 480008 | 2024-09 | 60.0 | 30 | B31f | 3.0 | 500 | 51 | 40 | 48 | 22 | 38 | WARNING |
| 487009 | 2025-03 | 62.0 | 0 | Qr40f | 1.5 | 600 | 44 | 77 | 53 | 26 | 53 | WARNING |
| 487009 | 2025-09 | 27.0 | 3 | Qr40f | 1.5 | 600 | 63 | 86 | 63 | 39 | 61 | WARNING |

Highlights:
- **Segmentation provenance**: 796 DVH rows stratify into 342 AutoRTS, 307 custom, and 147 manual contours, evidencing successful TotalSegmentator execution followed by Boolean structure synthesis.
- **Custom nnU-Net ensembles**: `Segmentation_CustomModels/cardiac_STOPSTORM` contributed 16 cardiac structures (7 non-empty in the demo dataset). DVH and radiomics tables now expose these as `Segmentation_Source = CustomModel:cardiac_STOPSTORM`, enabling downstream comparison with manual or TotalSegmentator contours.
- **Radiomics throughput**: 279 feature rows, covering 97 unique ROIs, were exported despite repeated primary-environment import failures; logs show PyRadiomics falling back to the conda runner while honoring `skip_rois` and voxel thresholds.
- **Quality control**: All six courses report `overall_status = WARNING` due to structure cropping. The most frequent cropped ROIs include spinal cord and autochthonous muscles (six courses each), along with couch-derived structures (five courses). Frame-of-reference and file-integrity checks otherwise pass.
- **Dose consistency**: Prescription inference recovered expected totals (50–62 Gy) across historical cohorts, while fraction counts align with recorded delivery, reflecting recent de-duplication fixes.

## Output Products
- Per-course artefacts (`dvh_metrics.xlsx`, `radiomics_ct.xlsx`, `fractions.xlsx`, `metadata/case_metadata.xlsx`, `qc_reports/*.json`, `RS_auto.dcm`, `RS_custom.dcm`).
- Custom nnU-Net outputs located under `Segmentation_CustomModels/<model>/` within each course directory (masks, `manifest.json`, `rtstruct.dcm`, `rtstruct_manifest.json`).
- Aggregated artefacts in `Data_Snakemake/_RESULTS/` (DVH, radiomics, fractions, metadata, QC reports, modality manifests).
- Supplemental modality-wide workbooks in `Data_Snakemake/Data/` (plans, structure sets, CT inventory, etc.).

## nnU-Net Cache Management
During custom-model inference each nnUNet bundle is unpacked beneath the model directory (e.g. `custom_models/cardiac_STOPSTORM/nnUNet_results`, `nnUNet_raw`, `nnUNet_preprocessed`). These caches let subsequent courses reuse the same weights without re-extracting ~500 MB archives. If disk pressure outweighs the convenience, set `custom_models.retain_weights: false` in `config.yaml` (or pass `--purge-custom-model-weights` to the CLI) and the pipeline will delete those directories after every successful model run. The zipped `model*.zip` files remain untouched so future runs can rehydrate the weights when needed, while course-specific outputs stay under `Segmentation_CustomModels/<model>/` inside each course directory.

## Quality Considerations and Known Issues
- Custom structure definitions previously triggered DVH warnings when TotalSegmentator ROI names drifted. The 5 Oct 2025 update broadens `custom_structures_pelvic.yaml` to include observed label variants and teaches `CustomStructureProcessor` to skip missing/empty masks gracefully, eliminating the recurring “Source structure … not found” noise while retaining visibility into genuinely absent inputs. Any ROI—manual, automatic, or custom—that is truncated or built from incomplete sources now receives a `__partial` suffix in downstream tables/RTSTRUCT output, and custom warnings are logged to `metadata/custom_structure_warnings.json`; treat those contours as unreliable.
- Systematic structure cropping triggers QC warnings (particularly spinal cord, autochthon muscles, and couch components). See `docs/qc_cropping_audit.md` for data and the recommended mitigation (morphological erosion + couch whitelist rather than expanding CT FOV).
- Radiomics execution relies on the auxiliary conda environment when native imports fail; ensure the environment remains synchronized with PyRadiomics releases to avoid feature drift.
- Segmentation parallelism is capped by `segmentation.workers` plus Snakemake’s global resource lock (`seg_workers`) to prevent GPU oversubscription; adjust according to hardware capabilities.

## References
1. Köster J, Rahmann S. Snakemake—a scalable bioinformatics workflow engine. *Bioinformatics.* 2012;28(19):2520–2522. doi:10.1093/bioinformatics/bts480. PMID: 22908215.
2. van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. *Cancer Res.* 2017;77(21):e104–e107. PMID: 29092951.
3. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. *Radiology: Artificial Intelligence.* 2023;5(5):e230024. PMID: 37795137.
