# Quickstart

## Install

```bash
pip install -e .
```

This installs pydicom (>=2.4,<3), dicompyler-core, SimpleITK, pydicom-seg, plotly, TotalSegmentator, rt-utils, dcm2niix, etc.

## Minimal run

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs -v
```

This will:
- Extract metadata (global XLSX under `outdir/Data/`)
- Organize per-patient, per-course folders under `outdir/<PatientID>/course_<key>/`
- (Optional) Run segmentation + auto RTSTRUCT (RS_auto)
- Compute DVH (manual + auto) and generate `DVH_Report.html`
- Generate `Axial.html` (CT + manual/auto overlays)

## Resume and force

- Segmentation is resume-safe: if outputs exist, it’s skipped.
- Force re-run:

```bash
rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs --force-segmentation
```

