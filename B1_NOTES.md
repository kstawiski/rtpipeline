# B1 implementation notes

## Scope

Implemented item B1 only: opt-in `body_composition_classes` routing for TotalSegmentator `tissue_types`/`body` tasks plus body-composition JSON/CSV writer. No D1/B3/B2/E1/E3 surfaces were edited.

## File:line changes

- `rtpipeline/config.py:61-63` - added `body_composition_classes: list[str] | None = None`; default OFF keeps legacy behavior.
- `rtpipeline/cli.py:1101-1124` - wired YAML override parsing for `organize.body_composition_classes` and top-level `body_composition_classes`; accepted only `planning_ct`, `diagnostic_ct`, `petct_ct`.
- `rtpipeline/inventory.py:31-61` - added B1 eligible classes and `ts_tasks_for_image_class()` helper. Base `TS_TASK_BY_CLASS` remains unchanged.
- `rtpipeline/inventory.py:226-232` - all-series manifest rows add `ts_tasks` only when body-composition routing adds extra tasks.
- `rtpipeline/segmentation.py:817-830` - added manifest-entry helper for per-model all-series outputs.
- `rtpipeline/segmentation.py:986-1157` - all-series segmentation now loops over effective tasks. Default OFF still runs only the legacy scalar `ts_task`; configured eligible CT classes run `total`, `tissue_types`, and `body`, then write per-series `body_composition.json` and refresh `Data/body_composition.csv`.
- `rtpipeline/body_composition.py:1-243` - new writer module. It selects the middle occupied axial slice of `total--vertebrae_L3`, computes skeletal muscle CSA, mean muscle HU, `torso_fat` VAT-proxy area, subcutaneous fat area, and SMI only when DICOM `PatientSize` is present.
- `README.md:194-196` - added release/license boundary note that TotalSegmentator `tissue_types` is non-commercial upstream.
- `tests/test_body_composition_b1.py:80-270` - added five B1 tests: default-off task list, CLI YAML override, configured routing, synthetic CSA/JSON/CSV schema, and SMI null reason when `PatientSize` is absent.

## Deviations

None. The body-composition metrics use `total` vertebrae and `tissue_types` masks; the `body` task is still routed as required by B1.

## Validation

Command:

```bash
/home/konrad/micromamba/envs/rtpipeline/bin/python -m pytest tests/ -q
```

Result: `301 passed, 1 skipped, 388 warnings in 24.42s`.

This is the requested baseline `296` passing tests plus `5` new B1 tests. No commit was made.

## Round-1 review remediation (2 blocking findings, independently raised + reproduced)

Two reviewers independently flagged (and one reproduced) two blocking issues; both fixed:

1. **HU windowing on L3 metrics.** Muscle CSA/radiodensity and the VAT/SAT areas were computed over the raw TotalSegmentator label with no Hounsfield gate, so partial-volume / mislabelled voxels (e.g. an air voxel at −1000 HU inside the muscle label) biased the named clinical metrics and made them non-comparable to the threshold-based literature. Fix (`body_composition.py`): metrics are now computed over the label intersected with the standard sliceOmatic/Alberta HU windows (Mourtzakis 2008; Martin 2013 JCO; Aubrey 2014) — skeletal muscle −29…150, visceral fat −150…−50, subcutaneous fat −190…−30. The windows are persisted in the per-series JSON (`hu_windows`) and the CSV, and a regression test asserts an out-of-window voxel is excluded from both area and radiodensity.

2. **Cross-process CSV race + mis-attributed failure.** `Data/body_composition.csv` was rewritten per series from inside parallel patient workers via a single shared `body_composition.csv.tmp`, which corrupts/truncates under concurrency (reproduced as `FileNotFoundError` on the shared rename) and is O(N²); a CSV-write error also wrongly marked a fully-segmented series `seg_failed`. Fix: the per-series writer now emits only the collision-free per-series JSON; the global CSV is aggregated **once, serially, at the end of the all-series stage** (`cli.py`); the writer uses a unique per-process temp name; and a body-composition error no longer flips a successful segmentation to failed. A multiprocess regression test asserts concurrent aggregation yields a complete, uncorrupted CSV with no leftover temp files.

Re-validation: `pytest tests/ -q` → `304 passed, 1 skipped` (301 baseline + 3 new regression tests).
