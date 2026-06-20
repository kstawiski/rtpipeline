# D1 Notes

## File:line changes

- `rtpipeline/organize.py:1005-1105` refactors `_export_original_segmentation` into `_export_original_segmentation_from_paths(...)`; the existing per-course wrapper still passes the same `CourseOutput` attributes and preserves the legacy manifest schema.
- `rtpipeline/inventory.py:294-435` adds `manual_rtstruct_bindings_from_inventory(...)`, resolving source RTSTRUCT paths from `instances`/`dicom_files` plus `rt_links`.
- `rtpipeline/segmentation.py:826-841` adds per-series `Segmentation_Original` artifact-root selection and the structured `event=no_original_available` log line.
- `rtpipeline/segmentation.py:931-1022` loads manual RTSTRUCT bindings once per patient and exports manual masks after per-series NIfTI conversion, before the idempotent AI segmentation skip.
- `tests/test_d1_all_series_original_segmentation.py:120-500` adds D1 coverage for FoR/manual binding branches, fail-closed ambiguity paths, old inventory no-op, cached manual re-run, per-course legacy manifest shape, all-series referenced-series binding, no-original skip logging, and manual/AI same-name ROI separation.

## Binding mechanism

The all-series path binds manual RTSTRUCTs through the inventory database, anchored to `series_manifest.json` rows:

- Exact `rt_links.relationship='rtstruct_to_series'` plus `target_series_uid == row["series_uid"]` wins.
- `rtstruct_to_for` links bind only when the row already has `rt_link_basis == "rtstruct_to_for_unique"` or the manifest has exactly one row for the `(study_uid, frame_of_reference_uid)` pair.
- Ambiguous exact or study/FoR matches are skipped fail-closed with warnings.
- Rows without a resolved source RTSTRUCT emit a structured `all_series_original_segmentation event=no_original_available ...` debug line and are not exported.

Manual export reuses the same mask extraction core as the per-course path and writes `model="manual"`. AI outputs keep their existing TotalSegmentator model ids such as `total`/`total_mr`, and all-series manual masks live under per-series `Segmentation_Original` roots separate from `Segmentation_TotalSegmentator`.

## Deviations

No temp course tree was added. The brief allowed a small parameter refactor if cleaner; the all-series segmentation loop already has the per-series DICOM directory and NIfTI path, so passing those directly avoids extra filesystem materialization and cleanup while preserving the per-course wrapper behavior.

## Remediation

- Added five focused tests in `tests/test_d1_all_series_original_segmentation.py`: FoR-unique/single-row fallback success, ambiguous FoR fail-closed skip, multiple exact RTSTRUCT fail-closed skip, missing/old inventory no-op with schema warning, and cached manual-manifest re-run behavior.
- Downgraded `event=no_original_available` from INFO to DEBUG while keeping the structured machine-parseable fields.
- Added the stricter-FoR gate comment and clarified the fail-closed inventory schema/`rt_links` warning.
- Full-suite validation after remediation: `296 passed, 1 skipped, 382 warnings in 23.79s`.

## Validation

Command: `/home/konrad/micromamba/envs/rtpipeline/bin/python -m pytest tests/ -q`

Result: `296 passed, 1 skipped, 382 warnings in 23.79s` (289 baseline tests plus 7 D1 tests).
