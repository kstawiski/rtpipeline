# CLI Reference

```bash
rtpipeline --dicom-root PATH --outdir PATH --logs PATH [options]
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

## Environment
- `--conda-activate CMD`  Prefix segmentation commands (e.g. activate env).
- `--dcm2niix NAME`  Override dcm2niix command name.
- `--totalseg NAME`  Override TotalSegmentator command name.

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

