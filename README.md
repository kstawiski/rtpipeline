rtpipeline – DICOM‑RT Processing Pipeline
========================================

Overview
- Organizes DICOM‑RT/CT per patient and merges plans within the same treatment course.
- Course definition defaults to “same CT StudyInstanceUID” (primary + boost). This avoids merging subsequent treatments (e.g., SBRT after progression) that are planned on a different CT.
- Optional: run TotalSegmentator on the CT, compute DVH metrics (manual + auto), and generate HTML visualizations.

Install
- From the repo root:
  - `pip install -e .`
  - Installs core deps: pydicom (>=2.4,<3), dicompyler-core (DVH), SimpleITK, pydicom-seg, matplotlib, scipy, pandas, TotalSegmentator, nested-lookup, plotly, rt-utils, dcm2niix.
  - TotalSegmentator does not require a license key.

CLI Usage
- Minimal run:
  - `rtpipeline --dicom-root /path/to/DICOM --outdir ./Data_Organized --logs ./Logs`
- Options:
  - `--merge-criteria {same_ct_study,frame_of_reference}`
  - `--max-days N` to split plans that are far apart in time even if they share the same CT
  - `--no-segmentation`, `--force-segmentation` to skip or re-run segmentation
  - `--no-dvh`, `--no-visualize`, `--no-metadata` to skip phases
  - `--conda-activate "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rt"` to run dcm2niix/TotalSegmentator from a conda env
  - `--dcm2niix`, `--totalseg` to override command names

Outputs
- Per patient: `outdir/<patient_id>/course_<course_key>/`
  - `RP.dcm`, `RD.dcm` (summed if multiple), `RS.dcm` (manual if available)
  - `CT_DICOM/` (CT slices matching the course study)
  - `nifti/` (from dcm2niix), `TotalSegmentator_{DICOM,NIFTI}/` (if segmentation enabled)
  - `RS_auto.dcm` (auto-generated RTSTRUCT from TotalSegmentator for DVH)
  - `dvh_metrics.xlsx` (includes IntegralDose, D1ccGy, V95%Rx, V100%Rx, …) and `DVH_Report.html` (interactive Plotly DVH)
  - `Axial.html` (scrollable axial CT QA viewer with manual/auto overlays)
  - `case_metadata.json` and `case_metadata.xlsx` (per-course clinical/research metadata)

Course Merging Logic
- By default, plans/doses are merged only if they refer to the same CT StudyInstanceUID. This captures primary+boost on the same CT and prevents merging subsequent treatments planned on a new CT (e.g., progression SBRT).
- Alternative policy: `--merge-criteria frame_of_reference` groups by FrameOfReferenceUID instead.
- You can also set `--max-days` to impose a time window for course grouping.

Notes
- The pipeline assumes reasonably consistent DICOMs (Modality = CT/RTPLAN/RTDOSE/RTSTRUCT) and that RTDOSE references the RTPLAN (ReferencedRTPlanSequence).
- TotalSegmentator is installed via pip. Set `--conda-activate` if your environment needs it.
- dcm2niix is optional; if missing, DICOM‑mode segmentation still runs and NIfTI segmentation is skipped.
- DVH uses dicompyler-core and includes both manual RTSTRUCT and auto RTSTRUCT (RS_auto.dcm) when present. Manual prescribed dose is estimated from CTV1 D95 if present, else defaults to 50 Gy.
 - Segmentation is resume-safe (reused if present). Use `--force-segmentation` to re-run.

Documentation
- See docs/ for detailed guides:
  - docs/index.md — Overview & concepts
  - docs/quickstart.md — Install and first run
  - docs/cli.md — CLI options and examples
  - docs/pipeline.md — End-to-end pipeline details
  - docs/metadata.md — Global and per-case metadata fields
  - docs/segmentation.md — Segmentation, resume, RS_auto
  - docs/dvh.md — DVH metrics & formulas
  - docs/viewers.md — DVH (Plotly) and Axial viewer
