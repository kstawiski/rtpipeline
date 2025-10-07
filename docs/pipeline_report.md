# RTPipeline Snakemake Report (5 October 2025)

## 1. Introduction
This document summarizes the behavior of the KONSTA_DICOMRT_Processing workflow during the latest verification run executed on 5 October 2025 via `./test.sh`. The pipeline, implemented in Snakemake, coordinates the `rtpipeline` Python package to perform course organization, TotalSegmentator-based contouring, DVH analytics, radiomics feature extraction, and automated quality control across prostate radiotherapy cohorts [1–3].

## 2. Materials and Methods
- **Input data**: `Example_data/` (five patients, six treatment courses) copied into the working directory; Snakemake resolves absolute paths through `config.yaml`.
- **Execution command**: `./test.sh`, which unlocks residual Snakemake state and invokes `snakemake --cores all --use-conda --rerun-incomplete --conda-prefix "$HOME/.snakemake_conda_store"`.
- **Software stack**: Conda environments described in `envs/rtpipeline.yaml` (imaging and DVH) and `envs/rtpipeline-radiomics.yaml` (PyRadiomics). TotalSegmentator binaries and PyRadiomics modules are fetched at runtime via Micromamba/conda, while DICOM operations rely on dicompyler-core and SimpleITK.
- **Instrumentation**: Stage-specific logs captured under `Logs_Snakemake/`. Sentinels stored in `Data_Snakemake/{patient}/{course}/.stage_name` expose Snakemake status.

## 3. Results
### 3.1 Data organization
- Six course folders were generated, as recorded in `_COURSES/manifest.json` and `Logs_Snakemake/stage_organize.log` (02:40–02:43 UTC execution window).
- Metadata export produced 81-feature tables per course (`case_metadata.xlsx`), including inferred prescriptions (27–62 Gy) and CT acquisition parameters (slice thickness 1.5–5.0 mm, field of view 500–600 mm).

### 3.2 Segmentation
- TotalSegmentator completed for all courses with sequential scheduling (`segmentation.workers = 4`, but Snakemake resource `seg_workers=1` limited simultaneous courses). Individual courses required 8–12 minutes wall-clock, with combined DICOM and NIfTI passes logged.
- Auto contours were consolidated into `RS_auto.dcm` and merged with manual RTSTRUCTs to yield 796 DVH-ready ROIs. Distribution: 342 AutoRTS, 307 custom Boolean composites, 147 manual ROIs.
- The new `segmentation_custom_models` stage detected `custom_models/cardiac_STOPSTORM`, extracted nnUNet weights into the local cache, and produced 16 cardiac substructures (7 non-empty for patient 487009/2025-03). Outputs are stored under `Data_Snakemake/487009/2025-03/Segmentation_CustomModels/cardiac_STOPSTORM/` with accompanying manifest and RTSTRUCT files.
- `custom_models/HN_lymph_nodes` was added as a nnU-Net v1 ensemble (3D fullres + 2D). Its post-processed output now appears at `Segmentation_CustomModels/HN_lymph_nodes/` inside each course, exposing 20 lymph-node levels for downstream DVH and radiomics processing (`Segmentation_Source = CustomModel:HN_lymph_nodes`).
- Warnings noted previously: absent prerequisites for custom structures (e.g. `pelvic_bones_3mm`, `bowel_bag`). After expanding `custom_structures_pelvic.yaml` with TotalSegmentator label variants and enhancing the processor to tolerate missing/empty masks (5 Oct 2025 update), the DVH stage now completes without “Source structure not found” spam. Any contour that is cropped or assembled from incomplete inputs is now tagged with a `__partial` suffix, and custom-structure warnings (missing sources, cropping) are recorded in `metadata/custom_structure_warnings.json` for each course.

### 3.3 Dose–volume analysis
- Per-course `dvh_metrics.xlsx` files contained 85–188 rows each. Aggregated data retained `Segmentation_Source`, `structure_cropped`, and manual/auto flags for downstream filtering. Custom nnU-Net contours appear with `Segmentation_Source = CustomModel:cardiac_STOPSTORM` and are now part of the default DVH export.
- Cropping burden remained high (30–63 structures per course). Dominant truncated ROIs are summarized in Table 1.

### 3.4 Radiomics extraction
- Native PyRadiomics imports failed in the main environment; the fallback conda executor (`rtpipeline.radiomics_conda`) generated 279 feature rows across 97 unique ROIs.
- `radiomics.skip_rois` (body, couch, bones, etc.) and voxel thresholds (10–1.5×10^9 voxels) prevented oversized non-clinical regions from prolonging runtime.
- Output stored as `radiomics_ct.xlsx` per course and `_RESULTS/radiomics_ct.xlsx` globally. Custom nnU-Net structures are included automatically and labelled with `segmentation_source = CustomModel:cardiac_STOPSTORM`.

### 3.5 Quality control
- All six JSON reports registered `overall_status = WARNING` owing to structure cropping, while file and frame-of-reference checks passed.
- Cropping statistics were re-audited (see `docs/qc_cropping_audit.md`): 150/178 truncated ROIs stem from auto-contours, primarily spinal cord and paraspinal musculature touching the series boundary, with the remainder originating from couch-related manual ROIs. Planned mitigation is to erode/clip auto masks before QC and whitelist couch structures rather than altering CT acquisition range.
- Table 1 lists the most frequently cropped ROIs across courses.

**Table 1. Most frequently cropped ROIs across all courses (n = 6).**

| ROI | Courses with cropping |
| --- | ---: |
| autochthon_left | 6 |
| autochthon_right | 6 |
| spinal_cord | 6 |
| aorta | 5 |
| body | 5 |
| couchinterior | 5 |
| couchsurface | 5 |
| femur_left | 5 |
| femur_right | 5 |
| inferior_vena_cava | 5 |
| rib_right_12 | 5 |
| colon | 4 |

### 3.6 Aggregated deliverables
- `_RESULTS/` hosts ten Excel workbooks (DVH, radiomics, fractions, metadata, QC, plus supplemental imaging manifests).
- `Data_Snakemake/Data/` provides harmonized plan, structure-set, dose, CT, and fraction inventories for statistical reuse.

## 4. Discussion
The pipeline executed deterministically across all stages, validating the integrity of the Snakemake abstraction and the `rtpipeline` complements. Persistent QC warnings trace to field-of-view truncation in abdominal and couch-adjacent regions, suggesting that segmentation eroding or ROI exclusion should precede statistical modelling. Custom structures referencing absent TOTalseg ROIs indicate nomenclature drift between TotalSegmentator releases and local YAML templates. Dose-plane interpolation warnings did not halt DVH generation but may bias absolute metrics for inferior anatomical slices.

## 5. Recommendations
1. Reconcile `custom_structures_pelvic.yaml` with observed TotalSegmentator labels to eliminate missing-source warnings during DVH synthesis.
2. Quantify the dosimetric impact of dose-plane interpolation by spot-checking representative courses in dicompyler or Eclipse.
3. Implement automated FOV audits that either crop ROIs before QC evaluation or flag images for reacquisition to reduce systematic `WARNING` statuses.
4. Package PyRadiomics dependencies inside the primary environment (or pre-warm the radiomics conda env) to avoid repeated import failures when new courses are added.
5. Increase `segmentation.workers` alongside Snakemake resource adjustments if hardware permits parallel segmentation, reducing total runtime for future cohorts.
6. Monitor disk usage under `custom_models/<model>/nnUNet_*`; set `custom_models.retain_weights: false` when one-off runs are preferred over cached nnUNet weights.

## References
1. Köster J, Rahmann S. Snakemake—a scalable bioinformatics workflow engine. *Bioinformatics.* 2012;28(19):2520–2522. doi:10.1093/bioinformatics/bts480. PMID: 22908215.
2. van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. *Cancer Res.* 2017;77(21):e104–e107. PMID: 29092951.
3. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. *Radiology: Artificial Intelligence.* 2023;5(5):e230024. PMID: 37795137.
