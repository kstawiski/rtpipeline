# CLI Reference

```bash
rtpipeline --dicom-root PATH --outdir PATH --logs PATH [options]
rtpipeline doctor [--logs PATH] [--conda-activate CMD] [--dcm2niix NAME] [--totalseg NAME]
```

## Required
- `--dicom-root PATH`  Root folder with DICOM files (may contain multiple patients).

## Common
- `--outdir PATH`  Output directory (default `./Data_Organized`).
- `--logs PATH`    Logs directory.
- `-v`/`--verbose` Increase log verbosity (repeat for more).

## Course grouping
- `--merge-criteria {same_ct_study,frame_of_reference}`  Default `same_ct_study`.
- `--max-days N`  Optional time window for grouping.

## Phase control
- `--no-metadata`     Skip global metadata extraction.
- `--no-segmentation` Skip TotalSegmentator.
- `--force-segmentation` Re-run TotalSegmentator even if outputs exist.
- `--no-dvh`          Skip DVH computation.
- `--no-visualize`    Skip HTML reports (DVH + Axial viewer).
- `--no-radiomics`    Skip pyradiomics extraction.
- `--radiomics-params PATH`  Use a custom pyradiomics YAML parameter file instead of the packaged defaults.

## Environment
- `--conda-activate CMD`  Prefix segmentation commands (e.g. activate env).
- `--dcm2niix NAME`  Override dcm2niix command name.
- `--totalseg NAME`  Override TotalSegmentator command name.
- `--totalseg-license KEY`  Optional license key for TotalSegmentator (exported as env var during runs).
- `--totalseg-weights PATH`  Path to pretrained weights (nnUNet_pretrained_models). Use this for offline environments.
- `--extra-seg-models`  Additional TotalSegmentator tasks to run besides the default 'total'. Accepts comma-separated values and can be repeated.
- `--totalseg-fast`  Adds `--fast` to TotalSegmentator calls (recommended on CPU).
- `--totalseg-roi-subset LIST`  Passes `--roi_subset LIST` to TotalSegmentator (comma-separated ROI names).
- `--workers N`  Number of parallel workers for non-segmentation phases (organize, DVH, visualization, metadata). Default: auto.

Note: `dcm2niix` is an external CLI (not installed via pip). If it is not available, NIfTI conversion is skipped and the pipeline continues with DICOM-mode segmentation only.

## Progress & ETA
- The CLI reports progress and estimated time remaining for long-running phases:
  - Organize (per-course)
  - Build RS_auto (after segmentation)
  - DVH (per-course)
  - Visualization (per-course)
  - Segmentation phases print progress as they complete (sequential by design)

## Doctor
- `rtpipeline doctor` prints environment diagnostics:
  - Python and OS info
  - Installed versions of key Python packages
  - Presence of `dcm2niix` and `TotalSegmentator` in PATH (unless `--conda-activate` is provided)
  - Availability of bundled `rtpipeline/ext/dcm2niix_*.zip` inside the package (used as fallback)
  - Whether NIfTI conversion will fallback to bundled dcm2niix or be skipped

## Progress & ETA
- The CLI reports progress and estimated time remaining for long-running phases:
  - Organize (per-course)
  - Build RS_auto (after segmentation)
  - DVH (per-course)
  - Visualization (per-course)
  - Segmentation phases print progress as they complete (sequential by design)

## Radiomics
- The pipeline can compute pyradiomics features for:
  - CT courses: manual RS (RS.dcm) and AutoRTS_total (RS_auto.dcm) → `radiomics_features_CT.xlsx`
  - MR series: manual MR RTSTRUCT (if available) and TotalSegmentator `total_mr` outputs → `radiomics_features_MR.xlsx`
- Install pyradiomics from GitHub:

```
pip install git+https://github.com/AIM-Harvard/pyradiomics
```

- Disable via `--no-radiomics`.
- You can provide a custom parameter file via `--radiomics-params PATH`.

## Outputs (per course)
- `RP.dcm`, `RD.dcm`, `RS.dcm` (if present)
- `RS_auto.dcm` (auto RTSTRUCT from TotalSegmentator)
- `CT_DICOM/`, `TotalSegmentator_{DICOM,NIFTI}/`, `nifti/`
- `dvh_metrics.xlsx`, `DVH_Report.html`, `Axial.html`
- `case_metadata.json` and `case_metadata.xlsx`

## Global outputs
- `outdir/Data/*.xlsx` (plans, doses, structures, fractions, CT_images, merged metadata)
- `outdir/DVH_metrics_all.xlsx` (all courses)
- `outdir/Data/case_metadata_all.{xlsx,json}` (all courses)
