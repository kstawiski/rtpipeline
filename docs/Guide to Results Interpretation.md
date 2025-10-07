# Guide to Results Interpretation (KONSTA_DICOMRT_Processing)

## 1. Overview of deliverables
After running `./test.sh` (Snakemake full workflow), the pipeline populates `Data_Snakemake/` with course-level folders and aggregates under `_RESULTS/`. The key artefacts and recommended uses are:

- **Course-specific workbooks**
  - `Data_Snakemake/<patient>/<course>/dvh_metrics.xlsx`: per-ROI dose–volume metrics (absolute and relative). Source provenance (`Segmentation_Source`) and truncation flags (`structure_cropped`, `ROI_Name` suffix) are embedded alongside standard DVH outputs.
  - `Data_Snakemake/<patient>/<course>/radiomics_ct.xlsx`: PyRadiomics feature matrix for CT-derived masks. Columns include ROI identifiers, segmentation source, structure cropping status, and all extracted features.
  - `Data_Snakemake/<patient>/<course>/fractions.xlsx` and `metadata/case_metadata.xlsx`: treatment delivery schedule and DICOM metadata (machine, acquisition parameters, patient demographics).
  - `Data_Snakemake/<patient>/<course>/qc_reports/*.json`: structured QC checks—file integrity, frame-of-reference consistency, and boundary-touch alerts for each ROI.
  - `Data_Snakemake/<patient>/<course>/Segmentation_CustomModels/<model>/`: custom nnUNet masks (`*.nii.gz`), `rtstruct.dcm`, and manifest metadata powering the `CustomModel:<model>` entries in DVH/radiomics tables.

- **Aggregation directory (`Data_Snakemake/_RESULTS/`)**
  - `dvh_metrics.xlsx`, `radiomics_ct.xlsx`, `fractions.xlsx`, `case_metadata.xlsx`, `qc_reports.xlsx`: concatenated versions of the per-course tables with added course/patient identifiers for cohort-level analysis.
  - `plans.xlsx`, `structure_sets.xlsx`, `metadata.xlsx`, etc., copied from `Data_Snakemake/Data/`, providing global listings of plans, structure sets, CT images, and related metadata.

- **Metadata breadcrumbs**
  - `Data_Snakemake/_COURSES/manifest.json`: master index of patient/course directories.
  - `Data_Snakemake/<patient>/<course>/.<stage>_done`: sentinel files for Snakemake bookkeeping.
  - `Data_Snakemake/<patient>/<course>/metadata/custom_structure_warnings.json`: enumerates custom ROIs built from incomplete or cropped sources (see Section 3).

## 2. Applying the `__partial` convention
From the 6 October 2025 run onward, any contour deemed geometrically incomplete is renamed with a `__partial` suffix in all downstream artefacts. Two conditions trigger this label:

1. The input mask touches the CT image boundary (`mask_is_cropped` evaluates to true).
2. A custom Boolean structure lacks one or more requested source ROIs or inherits a cropped mask.

Columns to monitor:
- `ROI_Name` (DVH) / `roi_name` (radiomics): use `str.endswith('__partial')` to filter incomplete ROIs.
- `Segmentation_Source` / `segmentation_source`: values include `Manual`, `AutoRTS`, `Merged`, and `CustomModel:<model>`, enabling head-to-head comparisons between planning, TotalSegmentator, Boolean composites, and custom nnUNet outputs.
- `structure_cropped` (boolean): remains true for all boundary-touching masks; aligns with the suffix for rapid verification.
- `ROI_OriginalName` / `roi_original_name`: original labels prior to suffixing, useful when mapping back to planning nomenclature or QC reports.

**Recommended practice:** exclude `__partial` entries from primary quantitative analyses unless your study explicitly targets truncated anatomies. Retain the original names for traceability and documentation.

## 3. Interpreting custom structure warnings
Custom composites (e.g., `pelvic_bones`, `iliac_vess`) are generated via Boolean operations. If prerequisite masks are missing or cropped, the pipeline retains the available data but:
- Renames the derived contour with `__partial`.
- Writes `metadata/custom_structure_warnings.json` containing structured entries such as:
  ```json
  {
    "structure": "pelvic_bones__partial",
    "original_structure": "pelvic_bones",
    "missing_sources": ["sacrum", "hip_left"],
    "cropped": true
  }
  ```
Researchers should parse this file before using composite structures; if clinically critical components are absent, consider reconstructing the ROI manually or omitting it from analyses.

## 4. Quality control signals
- `qc_reports.xlsx` aggregates each course’s QC JSON into tabular form. Key fields:
  - `overall_status`: `PASS`, `WARNING`, or `FAIL` (demo data typically sit at `WARNING` due to truncated anatomic coverage).
  - `checks.structure_cropping.structures`: JSON-encoded list of cropped ROIs with their segmentation source. Cross-reference with the `__partial` suffix to confirm exclusions.
- `Logs_Snakemake/<stage>`: stage-specific execution logs (organize, segmentation, DVH, radiomics, QC). Inspect these if a stage fails or produces unexpected counts.

## 5. Clinical sanity checks before analysis
1. **Prescription and fractions** (`case_metadata.xlsx` vs. `fractions.xlsx`): ensure total dose and delivered fractions match study design (e.g., 50 Gy / 25 fx). Outliers may indicate multi-plan boosts or incomplete exports.
2. **Contour integrity**: confirm essential targets (`PTV`, `CTV`, `GTV`) and key pelvic OARs (`Bladder`, `Rectum`, `Femur`, `Bowel`) lack the `__partial` suffix. If they appear cropped, revisit DICOM coverage or segmentation logs.
3. **Radiomics cohort balance**: count features per segmentation source and partial flag. High `__partial` ratios can bias statistical models; filter or stratify accordingly.

## 6. Common downstream workflows
- **DVH analytics**: load `_RESULTS/dvh_metrics.xlsx` into Python/R, filter out `__partial`, and compute dose statistics (Dmean, Dmax, Vx). Use `Segmentation_Source` to separate manual, TotalSegmentator, Boolean custom, and nnUNet ensembles (`CustomModel:*`).
- **Radiomics modeling**: subset `_RESULTS/radiomics_ct.xlsx` by organ class, drop `__partial`, normalize features, and integrate metadata (age, machine, acquisition parameters) as covariates.
- **QC reporting**: summarize cropped ROIs per course via `qc_reports.xlsx`, and document exclusion criteria in publications or clinical briefs.

## 7. Troubleshooting tips
- Missing aggregates: rerun `snakemake aggregate_results` or inspect stage logs for failures.
- Excessive `__partial` counts on critical ROIs: check CT coverage or RTSTRUCT integrity; consider re-exporting from the planning system.
- Radiomics failures (`No module named 'radiomics'`): ensure the `rtpipeline-radiomics` conda environment is available (see `Logs_Snakemake/radiomics/*.log`).

## 8. Provenance reminder
All outputs are intended for research analysis. Do **not** re-import derived RTSTRUCTs (`RS_custom.dcm`) into clinical TPS workflows: these files may contain renamed (`__partial`) contours and serve analytical—not treatment—purposes.
