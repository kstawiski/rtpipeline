# Guide to Results Interpretation

The pipeline produces a rich hierarchy of artefacts for each patient course. This document explains where to find them, how to interpret key columns, and which sanity checks you should perform before downstream analyses.

---

## 1. Course Directory Overview

Each course lives under `Data_Snakemake/<patient>/<course>/` and contains:

```
DICOM/CT/…                          # planning CT slices
DICOM/MR/…                          # MR slices copied directly under the course
MR/<SeriesInstanceUID>/
    DICOM/…                         # MR source series
    NIFTI/<name>.nii.gz             # compressed NIfTI + metadata JSON
    Segmentation_TotalSegmentator/
        total_mr--<roi>.nii.gz      # MR binary masks
        <name>--total_mr.dcm        # MR RTSTRUCT
Segmentation_TotalSegmentator/…     # CT TotalSegmentator outputs
Segmentation_CustomModels/<model>/… # nnUNet masks (e.g. cardiac_STOPSTORM, HN_lymph_nodes)
Segmentation_Original/…             # manual RTSTRUCT converted to masks (if available)
RS.dcm / RS_auto.dcm / RS_custom.dcm
radiomics_ct.xlsx
MR/radiomics_mr.xlsx
dvh_metrics.xlsx
fractions.xlsx
metadata/case_metadata.xlsx
qc_reports/*.json
```

Aggregated artefacts reside in `Data_Snakemake/_RESULTS/`:

* `dvh_metrics.xlsx`
* `radiomics_ct.xlsx`, `radiomics_mr.xlsx`
* `fractions.xlsx`
* `case_metadata.xlsx`
* `qc_reports.xlsx`

---

## 2. Radiotherapy Metrics (DVH)

**File**: `dvh_metrics.xlsx`

Key columns:

| Column | Meaning |
| --- | --- |
| `ROI_Name` | Clean display name (cropped structures append `__partial`). |
| `ROI_OriginalName` | Raw name from RTSTRUCT / segmentation output. |
| `Segmentation_Source` | Provenance: `Manual`, `AutoRTS_total`, `AutoRTS_total_mr`, `Merged`, `Custom`, `CustomModel:<name>`. |
| `structure_cropped` | `True` when the mask touches the image boundary (quality flag). |
| `DmeanGy`, `DmaxGy`, … | Standard DVH metrics. |

**Recommended checks**

1. Filter out `__partial` structures unless your analysis explicitly handles cropped contours.
2. Cross-check `Segmentation_Source` to ensure you understand which model produced each ROI.
3. Review `_RESULTS/qc_reports.xlsx` for courses marked `WARNING` or `FAIL`.

---

## 3. Radiomics Outputs

### CT (`radiomics_ct.xlsx`)
* Parameters from `rtpipeline/radiomics_params.yaml`.
* Includes manual, TotalSegmentator, custom structures, and nnUNet outputs.
* Important columns: `roi_name`, `segmentation_source`, `structure_cropped`, feature columns (GLCM, GLRLM, etc.).

### MR (`MR/radiomics_mr.xlsx`)
* Parameters from `rtpipeline/radiomics_params_mr.yaml`.
* Rows correspond to masks produced by TotalSegmentator `total_mr`.
* Columns: `roi_name`, `series_uid`, `segmentation_source` (`AutoTS_total_mr`), feature columns.

Aggregated copies are in `_RESULTS/radiomics_ct.xlsx` and `_RESULTS/radiomics_mr.xlsx` with additional `patient_id` / `course_id` context.

**Usage tips**
* Drop or flag rows where `structure_cropped == True`.
* When merging CT and MR features, include `modality` and `segmentation_source` to avoid mixing sources inadvertently.

---

## 4. Segmentation Manifests

* `Segmentation_TotalSegmentator/<base>/manifest.json` – CT segmentation manifest listing models and masks.
* `MR/<series>/Segmentation_TotalSegmentator/manifest.json` – MR counterpart for `total_mr`.
* `Segmentation_CustomModels/<model>/manifest.json` – nnUNet model manifest (structures + originating networks).

These files are useful when auditing which masks exist, the order they were combined in, or updating post-processing pipelines.

---

## 5. MR-Specific Layout

For every referenced MR series:

```
MR/<SeriesInstanceUID>/DICOM/…                     # original slices
MR/<SeriesInstanceUID>/NIFTI/<name>.nii.gz         # compressed volume
MR/<SeriesInstanceUID>/NIFTI/<name>.metadata.json  # modality + linkage
MR/<SeriesInstanceUID>/Segmentation_TotalSegmentator/
    total_mr--<ROI>.nii.gz
    <name>--total_mr.dcm
```

The metadata JSON contains:

* `modality: "MR"`
* `series_instance_uid`
* `nifti_path`/`source_directory`
* Timestamp of conversion

Radiomics results for the whole course are consolidated into `MR/radiomics_mr.xlsx`.

---

## 6. Quality Control

* `qc_reports/*.json` – per-course, structured QC results (`overall_status`, `checks.structure_cropping`, etc.).
* `_RESULTS/qc_reports.xlsx` – flattened version for cohort summaries.
* `metadata/custom_structure_warnings.json` – lists composite structures built from missing sources or cropped inputs (look for `__partial` suffixes).

**Recommended practice**
* Investigate any course with `overall_status != PASS`.
* When `structure_cropping` lists critical OARs or targets, consider excluding them or regenerating masks with manual corrections.

---

## 7. Anonymisation Footprint

If you run `scripts/anonymize_pipeline_results.py`, the anonymised tree mirrors the structure above with remapped identifiers. Keep the generated key file secure; it restores patient/course mappings if needed.

---

## 8. Quick Checklist Before Analysis

1. **Course completeness** – verify expected CT/MR series and check QC status.
2. **Segmentation provenance** – filter by `Segmentation_Source` to separate manual vs automated contours.
3. **Partial structures** – remove or flag `__partial` entries in DVH and radiomics tables.
4. **MR availability** – confirm `MR/<series>/Segmentation_TotalSegmentator/` exists before using MR radiomics.
5. **Custom models** – inspect `Segmentation_CustomModels/<model>/manifest.json` to confirm network and label names.

With these checks you can confidently use the pipeline outputs for statistical modelling, quality monitoring, or downstream AI experiments.

