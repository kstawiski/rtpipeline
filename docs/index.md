# rtpipeline Documentation

## What is rtpipeline?

rtpipeline is a DICOM‑RT processing pipeline that:
- Organizes and merges RT plans into per‑patient, per‑course folders.
- Runs TotalSegmentator and auto‑generates RTSTRUCT (RS_auto) for DVH use.
- Computes DVH metrics (manual and auto structures) and generates interactive DVH reports.
- Provides an Axial QA viewer with manual/auto overlays.
- Extracts both global and per‑case metadata for clinical and research analysis.
 - Optionally runs additional TotalSegmentator tasks for CT courses and MR series.

## Core Concepts

- Course grouping: By default, a course is defined by the CT StudyInstanceUID. Plans/doses on the same CT are merged; different CTs become separate courses.
- Resume‑safe segmentation: Existing outputs are reused. Use `--force-segmentation` to re‑run.
- Auto RTSTRUCT: TotalSegmentator outputs (DICOM‑SEG or NIfTI) are converted into RS_auto.dcm aligned to the course CT, enabling DVH.
- DVH metrics: Computed via dicompyler‑core from RD/RP for both manual RS and RS_auto.
- Viewers: DVH (Plotly) + Axial CT viewer with overlays to visually verify segmentation quality.
- MR handling: MR images are not part of RT “courses” or DVH, but you can run MR‑specific TotalSegmentator models (tasks ending with `_mr`) across all MR series in `--dicom-root`; outputs are organized by patient and SeriesInstanceUID.

Tip: Use `rtpipeline doctor` to check tools and whether bundled `dcm2niix` fallbacks are available.

See the rest of the docs for details.
