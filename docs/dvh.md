# DVH Metrics

`rtpipeline.dvh.dvh_for_course` computes absolute and relative dose–volume
metrics for every ROI in the available structure sets. The Snakemake rule `dvh`
invokes this function per patient directory after structure merging completes.

## Inputs
- `RP.dcm` (summed or copied by `organize_data`).
- `RD.dcm` (dose grid).
- `RS_custom.dcm` when merging succeeded, otherwise `RS_auto.dcm` or the manual
  `RS.dcm`.
- `CT_DICOM` is required implicitly for dicompyler-core to resolve geometry.

The rule creates an empty Excel workbook if `RP.dcm` or `RD.dcm` is missing so
that downstream steps remain resumable.

## Metric set (per ROI)
- Absolute dose statistics: `DmeanGy`, `DmaxGy`, `DminGy`, `D95Gy`, `D98Gy`,
  `D2Gy`, `D50Gy`, dose spread, `D1ccGy`, `D0.1ccGy` (when the structure volume
  allows computation).
- Relative dose (percentage of prescription): `Dmean%`, `Dmax%`, `D95%`,
  `D98%`, `D2%`, `D50%`, homogeneity index in percent, spread in percent.
- Volume summary: `Volume (cm³)`, integral dose (`IntegralDose_Gycm3`).
- Coverage at prescription: `V95%Rx (cm³/%), V100%Rx (cm³/%)`.
- Binned volumes: `VxGy (cm³)` and `VxGy (%)` for x = 1…60 Gy.

## Prescription handling
When a manual RTSTRUCT is present, the code looks for an ROI whose name contains
`ctv1` (case-insensitive, whitespace ignored) and uses its `D95` value as the
prescription dose. Failing that, it falls back to 50 Gy. The same reference dose
is then applied to manual, auto, and custom structures so relative values are
comparable.

## Outputs
- Per patient: `dvh_metrics.xlsx` in the patient directory with one row per ROI
  and a `Segmentation_Source` column indicating `Manual`, `AutoRTS`, or
  `Custom`.
- Aggregated: the Snakemake `summarize` rule concatenates all per-patient files
  into `dvh_summary.xlsx` at the root of `output_dir`. The CLI additionally
  writes `Data/DVH_metrics_all.xlsx` when multiple courses are processed.

If you extend the metric set, update both this document and
`rtpipeline/dvh.py` so the fields stay aligned.
