# Pipeline Overview

The `rtpipeline` CLI orchestrates a sequence of modular phases that turn a raw
`--dicom-root` into structured, analysis-ready outputs inside `--outdir`. Each
phase can be toggled through CLI flags, and resume logic skips work that has
already completed. This guide explains what each stage does, how the pieces fit
together, and where to find the resulting artifacts.

## Stage Map

| Stage | Purpose | Key CLI Switches | Primary Outputs |
| --- | --- | --- | --- |
| 1. Global discovery | Index every DICOM object and extract metadata | `--no-metadata` (skip) | `outdir/Data/*.xlsx`, `metadata.xlsx`
| 2. Course grouping | Decide which plans/doses belong together | `--merge-criteria`, `--max-days` | In-memory grouping manifest written into per-course folders |
| 3. Course organization | Normalize files per course and curate metadata | `--no-metadata`, `--resume` | `CT_DICOM/`, `RP.dcm`, `RD.dcm`, `RS.dcm`, `case_metadata.{json,xlsx}` |
| 4. Segmentation *(optional)* | Generate auto contours on CT and MR | `--no-segmentation`, `--force-segmentation`, `--extra-seg-models`, `--totalseg-fast` | `TotalSegmentator_{DICOM,NIFTI}/`, `RS_auto.dcm`, MR task directories |
| 5. Dose-volume histograms | Compare manual vs auto structures | `--no-dvh` | `dvh_metrics.xlsx`, `DVH_metrics_all.xlsx` |
| 6. Interactive QA | Produce web viewers for review | `--no-visualize` | `DVH_Report.html`, `Axial.html` |
| 7. Radiomics *(optional)* | Derive feature tables for CT and MR | `--no-radiomics`, `--radiomics-params` | `radiomics_features_{CT,MR}.xlsx`, `outdir/Data/radiomics_all.xlsx` |

The pipeline is parallelized where possible (`--workers N`) and is designed to
be resume-safe; reruns keep previous outputs unless you force-regenerate them.

## 1. Global Discovery & Metadata Extraction

1. Walks the entire DICOM tree supplied via `--dicom-root`.
2. Indexes CT, RTPLAN, RTDOSE, RTSTRUCT, MR, and treatment record series.
3. Writes global summary workbooks under `outdir/Data/`:
   - `plans.xlsx`, `dosimetrics.xlsx`, `fractions.xlsx`, `structure_sets.xlsx`, `CT_images.xlsx`.
   - `metadata.xlsx` cross-references RTPLAN↔RTDOSE by SOPInstanceUID prefixes and decorates
     each plan with patient-level context.

You can skip this stage with `--no-metadata`, though later phases rely on the
in-memory results to stay consistent.

## 2. Course Grouping

Courses represent a single treatment context (primary + boost) on a given CT.

- Default policy: group by `StudyInstanceUID` of the planning CT, which keeps
  primary/boost on the same scan together and isolates re-plans on newer CTs.
- Alternate policy: `--merge-criteria frame_of_reference` groups by
  `FrameOfReferenceUID` when that better reflects your workflow.
- Optional `--max-days N` splits courses if plans are separated by more than N
  calendar days even if they share the same CT.

Grouping decisions are persisted when the per-course folders are created, so
future resumptions keep the same layout.

## 3. Course Organization

For each course produced in Stage 2, the pipeline:

1. Copies the best-matching CT series into `CT_DICOM/` (largest slice count
   within the course study) and exports NIfTI if `dcm2niix` is available.
2. Normalizes dosimetry:
   - Resamples and sums all dose grids to a single `RD.dcm`.
   - Synthesizes or annotates a composite `RP.dcm` carrying the total
     prescription.
3. Selects the RTSTRUCT of record (`RS.dcm`) if a manual structure set exists.
4. Writes `case_metadata.json` and `.xlsx` with both clinical details and
   pipeline provenance fields (see `docs/metadata.md`).

This stage is responsible for the folder layout under `outdir/<PatientID>/` and
is re-used by downstream modules. Resume mode (`--resume`) skips any course that
already contains the expected artifacts.

## 4. Segmentation (Optional Stage)

Segmentation adds automated contours to support DVH comparisons and radiomics.

- Default CT task: TotalSegmentator `total` per course; outputs stored under
  `TotalSegmentator_DICOM/` and `TotalSegmentator_NIFTI/`.
- Default MR task: `total_mr` for every MR series found in the dataset. MR
  results live outside course folders at
  `outdir/<PatientID>/MR_<SeriesInstanceUID>/...`.
- Additional tasks: pass a comma-separated list via `--extra-seg-models`.
  - CT tasks omit the `_mr` suffix and execute inside each course folder.
  - MR tasks end with `_mr` and run per series alongside the default MR task.
- Use `--totalseg-fast` to append `--fast` for CPU-heavy workloads, or
  `--totalseg-roi-subset` to limit labels.
- Segmentation is resume-safe; existing outputs are reused unless
  `--force-segmentation` is supplied.

After TotalSegmentator finishes, the pipeline converts output volumes into
`RS_auto.dcm` (RTSTRUCT), aligned to the course CT so downstream DVH can compare
manual and auto contours.

## 5. Dose-Volume Histogram Calculation

Using dicompyler-core, the DVH stage evaluates both the manual `RS.dcm` (if
present) and `RS_auto.dcm` against the summed dose grid:

- Produces `dvh_metrics.xlsx` per course containing common coverage, hot spot,
  and integral dose metrics (e.g., `V95`, `D1ccGy`, target coverage at
  prescription).
- Aggregates all per-course metrics into `outdir/Data/DVH_metrics_all.xlsx`.
- Honors `--no-dvh` if you want to skip the computation entirely.

## 6. Interactive QA Viewers

Visual QA artifacts help validate the DVH results and inspect auto contours:

- `DVH_Report.html`: Plotly-based viewer with structure filters, manual vs auto
  overlays, and cohort context.
- `Axial.html`: slice-by-slice CT viewer with togglable manual/auto contours.

Both files are regenerated unless `--no-visualize` is set or `--resume` detects
they already exist.

## 7. Radiomics (Optional)

If pyradiomics is installed (either via `pip install -e ".[radiomics]"` or a
manual install), the pipeline extracts tabular features:

- CT courses: features for both the manual RTSTRUCT and `RS_auto` →
  `radiomics_features_CT.xlsx` per course.
- MR series: uses manual MR RTSTRUCT when available and the `total_mr` output →
  `radiomics_features_MR.xlsx` per MR series.
- A cohort merge is written to `outdir/Data/radiomics_all.xlsx`.

Pass `--no-radiomics` to disable the stage. `--radiomics-params PATH` lets you
point to a YAML parameter file. MRI normalization toggles automatically based on
detection of T1 vs T2 weighting.

## Resume Semantics

All stages honor `--resume`. The pipeline inspects the presence of key outputs
(`RS_auto.dcm`, `dvh_metrics.xlsx`, `DVH_Report.html`, etc.) and skips work per
course when they already exist. Combine this with `--force-segmentation` or
other flags to selectively refresh only the pieces you need.

See the other topic guides for deep dives:

- `docs/quickstart.md` – first run and environment.
- `docs/cli.md` – exhaustive CLI reference.
- `docs/metadata.md`, `docs/segmentation.md`, `docs/dvh.md`, `docs/viewers.md`
  – detailed stage-specific notes.
