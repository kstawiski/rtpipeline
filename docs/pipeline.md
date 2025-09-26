# Pipeline Details

## 1) Metadata extraction (global)
- Scans all DICOMs and writes XLSX under `outdir/Data/`:
  - `plans.xlsx`, `dosimetrics.xlsx`, `structure_sets.xlsx`, `fractions.xlsx`, `CT_images.xlsx`
  - `metadata.xlsx` merges RP↔RD by filename core key and attaches RS by patient.

## 2) Course grouping
- Courses are grouped by CT `StudyInstanceUID` (default) per patient.
- Alternative: `--merge-criteria frame_of_reference`.
- Optional `--max-days` filter splits by plan dates.

## 3) Organize per course
- Copies course CT to `CT_DICOM/` and picks the largest series in that study.
- Multi-stage plan: sums RD grids (resampling) → single `RD.dcm`; synthesizes `RP.dcm` with total Rx.
- Picks `RS.dcm` from the first matching RS.
- Writes `case_metadata.{json,xlsx}` including clinical/research fields (see docs/metadata.md).

## 4) Segmentation (optional)
- Default CT: runs TotalSegmentator "total" per course (DICOM and NIfTI outputs).
- Default MR: runs `total_mr` for each MR series found under `--dicom-root`.
- Extra models: specify additional tasks with `--extra-seg-models`.
  - CT: tasks without `_mr` suffix; outputs saved in each course directory as `TotalSegmentator_<MODEL>_{DICOM,NIFTI}/`.
  - MR: tasks ending with `_mr`; outputs under `outdir/<PatientID>/MR_<SeriesInstanceUID>/TotalSegmentator_<MODEL>_{DICOM,NIFTI}/`.
- Resume-safe: reuses outputs. Use `--force-segmentation` to redo.
- Builds `RS_auto.dcm` from DICOM-SEG or NIfTI aligned to `CT_DICOM` (used for DVH).

## 5) DVH
- Uses dicompyler-core to compute DVH for both `RS.dcm` (manual) and `RS_auto.dcm`.
- Writes `dvh_metrics.xlsx` per course and `DVH_metrics_all.xlsx` across all courses.

## 6) Viewers
- `DVH_Report.html`: Interactive Plotly DVH.
- `Axial.html`: axial CT viewer with togglable overlays for manual/auto ROIs.
 - MR series are not visualized by default.

## 7) Radiomics (optional)
- If pyradiomics is installed, extracts radiomics features:
  - CT courses: manual RS and AutoRTS_total (RS_auto) → `radiomics_features_CT.xlsx`
  - MR series: manual MR RTSTRUCT (if available) and TotalSegmentator `total_mr` outputs → `radiomics_features_MR.xlsx`
- Disable with `--no-radiomics`.
- Use `--radiomics-params PATH` to pass a custom YAML parameter file.
- MRI normalization toggles automatically based on detected weighting:
  - T2-weighted: normalization disabled
  - T1-weighted: normalization enabled
