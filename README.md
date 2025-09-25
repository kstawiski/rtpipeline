rtpipeline – DICOM‑RT Processing Pipeline
========================================

Overview
rtpipeline is an end‑to‑end DICOM‑RT processing pipeline that turns a folder of raw DICOMs (CT/RTPLAN/RTDOSE/RTSTRUCT/RT treatment records) into clean, per‑patient “courses” with organized data, segmentation, quantitative DVH metrics, visual QA, and rich metadata for clinical and research use.

At a glance, the pipeline:
- Scans a DICOM root (one or many patients) and extracts global metadata into Excel.
- Groups each patient’s data into treatment “courses” by the CT StudyInstanceUID (primary + boost on the same CT are merged; subsequent treatments on a different CT are separated).
- Organizes each course into a canonical structure (CT_DICOM, RP/RD/RS), sums multi‑stage doses into a single RD, and synthesizes a “summed” RP with total Rx.
- Runs TotalSegmentator on the course CT (resume‑safe; re-run with `--force-segmentation`) and auto‑generates RTSTRUCT (RS_auto) aligned to the CT, so DVH can include auto‑segmentation even if no manual RS exists.
- Computes DVH metrics for manual and auto structures using dicompyler‑core, including integral dose, hottest small volumes (D1cc, D0.1cc), and coverage at Rx, and writes per‑course and merged workbooks.
- Generates interactive reports: Plotly‑based DVH report (toggle structures) and an Axial viewer to scroll CT slices with semi‑transparent overlays for manual/auto structures.
- Saves comprehensive per‑case metadata (JSON/XLSX) for clinical/research queries (plan approvals, clinicians, prescriptions per ROI, beam geometry and MUs, dose grid, CT acquisition, course dates, etc.) and merges all cases into a single workbook/JSON for cohort analysis.

Typical flow (per patient):
1) Discover → extract global metadata (plans/doses/structures/fractions/CT index) to `outdir/Data/*.xlsx`.
2) Group → courses by CT StudyInstanceUID to avoid merging unrelated treatments.
3) Organize → copy CT, sum RD (with resampling), synthesize RP (total Rx), pick RS; write per‑case metadata.
4) Segment (optional) → run TotalSegmentator (DICOM + NIfTI), resume‑safe; build RS_auto from SEG/RTSTRUCT/NIfTI.
5) DVH → compute metrics for RS and RS_auto; save per‑course and merged across all courses.
6) Visualize → DVH_Report.html (Plotly) and Axial.html (CT viewer + overlays) per course.

Design principles:
- Clinically meaningful course grouping and dose summation.
- Idempotent, resume‑safe segmentation and RTSTRUCT generation.
- DVH driven by RTDOSE/RTPLAN with manual and auto RTSTRUCT.
- Visual QA tools built for rapid review of structure coverage and auto‑seg quality.
- Rich metadata to support downstream clinical and research analyses.

Install
- From the repo root:
  - `pip install -e .`
  - Installs core deps: pydicom (>=2.4,<3), dicompyler-core (DVH), SimpleITK, pydicom-seg, matplotlib, scipy, pandas, TotalSegmentator, nested-lookup, plotly, rt-utils.
- Note: `dcm2niix` is an external CLI, not a Python package. Install it via your OS package manager or conda (e.g., `conda install -c conda-forge dcm2niix`). The pipeline will skip NIfTI conversion if it is not available.
 - Optional: if you keep platform ZIPs in `rtpipeline/ext/` (e.g., `rtpipeline/ext/dcm2niix_lnx.zip`, `..._mac.zip`, `..._win.zip`), they are included in wheels and the pipeline auto‑extracts and uses the bundled `dcm2niix` when a global install is not found. Extraction happens under `--logs/bin/`.
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
  - `--totalseg-license KEY` to pass a license key when required
  - `--extra-seg-models model1,model2` to run additional TotalSegmentator tasks (besides the default 'total').
    - CT: tasks without `_mr` suffix run for each CT course into `TotalSegmentator_<MODEL>_{DICOM,NIFTI}/`.
    - MR: tasks ending in `_mr` run for each MR series found under `--dicom-root` into `outdir/<PatientID>/MR_<SeriesInstanceUID>/TotalSegmentator_<MODEL>_{DICOM,NIFTI}/`.
  - `--totalseg-fast` to add `--fast` (recommended on CPU), `--totalseg-roi-subset roi1,roi2` to restrict ROIs
  - `doctor` subcommand: `rtpipeline doctor` prints environment checks and bundled dcm2niix fallbacks

Outputs
- Per patient: `outdir/<patient_id>/course_<course_key>/`
  - `RP.dcm`, `RD.dcm` (summed if multiple), `RS.dcm` (manual if available)
  - `CT_DICOM/` (CT slices matching the course study)
  - `nifti/` (from dcm2niix), `TotalSegmentator_{DICOM,NIFTI}/` (if segmentation enabled)
  - `RS_auto.dcm` (auto-generated RTSTRUCT from TotalSegmentator for DVH)
  - `dvh_metrics.xlsx` (includes IntegralDose, D1ccGy, V95%Rx, V100%Rx, …) and `DVH_Report.html` (interactive Plotly DVH)
  - `Axial.html` (scrollable axial CT QA viewer with manual/auto overlays)
  - `case_metadata.json` and `case_metadata.xlsx` (per-course clinical/research metadata)
- MR series (when extra MR models requested): `outdir/<patient_id>/MR_<SeriesInstanceUID>/TotalSegmentator_<MODEL>_{DICOM,NIFTI}/`

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
