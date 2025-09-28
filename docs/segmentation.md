# Segmentation Stage

Segmentation is responsible for generating automatic contours via
TotalSegmentator and converting them into an RTSTRUCT (`RS_auto.dcm`). The
Snakemake workflow separates the work into two rules; the CLI follows a similar
pattern internally.

## `segment_nifti` (Snakemake)
- **Inputs**: `CT_DICOM` folder from `organize_data` and the accompanying
  `metadata.json`.
- **Execution**: runs `TotalSegmentator` directly on the CT directory. The rule
  honours the `segmentation` section of `config.yaml`:
  - `fast: true` adds `--fast` to TotalSegmentator (useful on CPU-only hosts).
  - `roi_subset` restricts the run to the listed structures.
- **Resource control**: thread count comes from the `SEGMENTATION_THREADS`
  constant in the `Snakefile`. A lock file under `Logs_Snakemake` serialises
  TotalSegmentator invocations so only one patient is processed at a time.
- **GPU detection**: the rule probes `nvidia-smi`; if available, segmentation is
  executed on GPU (`--device gpu`), otherwise `--device cpu` is used.
- **Outputs**: a fresh `TotalSegmentator_NIFTI/` directory containing the
  generated masks. Existing directories are removed before regeneration so that
  stale files do not leak in.

## `nifti_to_rtstruct`
- Reads the NIfTI masks and CT geometry, then calls
  `rtpipeline.auto_rtstruct.build_auto_rtstruct` to emit `RS_auto.dcm` in the
  patient directory.
- Multi-label NIfTI volumes and per-ROI masks are both supported. When a
  TotalSegmentator DICOM-SEG is present, it is copied/normalised instead of
  rerasterised.

## `merge_structures`
- Invokes `rtpipeline.structure_merger.merge_patient_structures` to reconcile
  manual `RS.dcm`, automated `RS_auto.dcm`, and optional custom definitions from
  `custom_structures` in `config.yaml`.
- Priority rules favour manual targets (PTV/CTV/GTV) while letting automated
  contours win for common OARs. A fallback path uses legacy code when the new
  merger raises an exception.
- Outputs `RS_custom.dcm`, `structure_comparison_report.json`, and
  `structure_mapping.json`. If merging fails, the rule copies `RS_auto.dcm` to
  keep the pipeline flowing and notes the error in the report.

## CLI behaviour
When you run `rtpipeline` directly, the segmentation stage (`segment_course` in
`rtpipeline/segmentation.py`) performs similar steps:
- Optionally converts CT DICOM to NIfTI via `dcm2niix` (with bundled fallbacks
  when available).
- Runs TotalSegmentator in both DICOM and NIfTI modes, respecting
  `--extra-seg-models`, `--totalseg-fast`, `--totalseg-roi-subset`, and
  `--force-segmentation` flags.
- Creates `RS_auto.dcm` from the produced masks.
- Leaves existing outputs untouched unless forced.

Regardless of whether you drive the process via Snakemake or the CLI, `RS_auto`
provides a consistent RTSTRUCT aligned to the organised `CT_DICOM` that enables
DVH, radiomics, and QA visualisation even when no manual structures are present.
