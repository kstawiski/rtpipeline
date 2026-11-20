# Small-volume, NIfTI and rotated-dose benchmark

This benchmark tests RTpipeline's production DVH workflow in 1,208 synthetic structure–path–dose cases, including structures below 1 cm³, binary NIfTI masks and rotated dose grids. In the recorded 2 cm³ axial cases, structure-volume error exceeded 3% in 25/40 NIfTI cases and 21/40 RTSTRUCT cases. Volume accuracy therefore needs inspection alongside dose endpoints, particularly for small structures. The tables retain boundary exceedances and unavailable measurements separately.

At 3 mm contour spacing, RTpipeline published no structure volume or dose–volume histogram for 42 of the 60 RTSTRUCT cases of 0.1 cm³: 14 axial-dose cases had no published ROI (region of interest); among rotated-dose cases, 22 had too few contour planes for a volume and 6 had no published ROI.

## Design

[DESIGN.json](DESIGN.json) declares 181 synthetic course groups and 1,208 cases: 604 RTSTRUCT and 604 NIfTI. A case is one structure processed through one input path in one dose configuration. The design combines sphere, cylinder with height equal to diameter, cube and concentric spherical shell; volumes of 0.1, 0.5, 2, 50 and 1,000 cm³; centres at (0, 0, 0) and (0.37, −0.61, 0.43) mm; CT/contour spacing of 1 or 3 mm; and isotropic dose spacing of 2.5 mm. CT spacing in plane is 2 mm. The shell inner radius is 90% of its outer radius.

There are 400 cases with axial dose, 400 with a positive 0.5° rotation about z, 400 with a positive 0.5° rotation about x, and eight with a 2° rotation about z. RTpipeline refuses dose grids rotated more than 1° from the nearest axis-aligned basis; all eight 2° cases were refused. The <1 cm³ class contains 488 cases, including those eight cases; each other size class contains 240. For NIfTI, CT spacing describes mask sampling; no contour-spacing parameter enters that path.

Fields are uniform 20 Gy, linear 20 + 0.1z Gy in patient LPS coordinates, and radial 30 − 0.15r Gy about the structure centre. Radial fields apply only to sphere and shell. The reference dose is 25 Gy, so V95Rx uses 23.75 Gy. `analytic.py` supplies continuous geometric truth without using contours, voxels or sampled dose. `generate_phantoms.py` writes deterministic synthetic CT, RTSTRUCT, RTDOSE, RTPLAN and delivery records. Circular contours have 256 vertices; stored RTDOSE resolution is 0.0001 Gy.

`extension.py` writes binary voxel-centre masks into the production NIfTI input directory. It rotates dose origin and basis together about the LPS origin and evaluates the field at each physical dose node. It calls `dvh_for_course(course, parallel_workers=1, use_cropped=False, rx_dose_gy=25, max_total_dose_gy=100)` and reads published metrics, curves and QC. No production function is replaced. Source-tree fixtures in `tests/` provide synthetic plans and contracts; use a source checkout. Masks are analytic shape samples and require no segmentation model.

## Run

Run bash commands from the repository root with RTpipeline and its test dependencies installed. The main extension uses the pipeline environment; it does not need the isolated dicompyler environment used by the optional axial replay. The harness installs no packages.

```bash
export DVH_REPO="$PWD"
export DVH_BENCH="$DVH_REPO/validation/dvh_extension"
export RTPIPELINE_SRC="$DVH_REPO"
export RTPIPELINE_PYTHON="${RTPIPELINE_PYTHON:-python}"
export PYTHONPATH="$RTPIPELINE_SRC" PYTHONDONTWRITEBYTECODE=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 RTPIPELINE_CONTRACT_HEADER_THREADS=4
export TMPDIR="$DVH_BENCH/scratch/tests" MPLCONFIGDIR="$DVH_BENCH/scratch/tests/matplotlib"
export XDG_CACHE_HOME="$DVH_BENCH/scratch/tests/cache"
mkdir -p "$TMPDIR" "$DVH_BENCH/results"

# Unit tests only.
nice -n 10 "$RTPIPELINE_PYTHON" -m pytest -q -p no:cacheprovider \
  --basetemp="$TMPDIR/pytest" --junitxml="$DVH_BENCH/results/analytic-tests.xml" \
  "$DVH_BENCH/test_analytic.py" "$DVH_BENCH/test_extension.py"

# One serial shard containing the complete design.
bash "$DVH_BENCH/run_extension.sh" 0 1
"$RTPIPELINE_PYTHON" -B "$DVH_BENCH/summarize_extension.py" "$DVH_BENCH"
"$RTPIPELINE_PYTHON" -B "$DVH_BENCH/verify_extension.py" "$DVH_BENCH"
```

Alternatively, partition the same design into N shards with `run_extension.sh i N` for every integer i from 0 to N−1, then summarize and verify once all shards finish. Use one shard count throughout a run. Each wrapper has separate scratch storage and one numerical-library thread. Outputs go to `results/shard-i/`; the committed `reference/` tables are separate. Do not combine a partial run with a complete run in the same results directory.

For a single-course smoke run, choose an exact `group` from `DESIGN.json`, for example:

```bash
bash "$DVH_BENCH/run_extension.sh" 0 1 \
  --group v0.1_aligned_ct1_dg2.5_uniform_axial
"$RTPIPELINE_PYTHON" -B "$DVH_BENCH/summarize_extension.py" "$DVH_BENCH"
"$RTPIPELINE_PYTHON" -B "$DVH_BENCH/verify_extension.py" "$DVH_BENCH" --partial
```

A partial audit checks observed rows and reports completeness separately. It does not establish complete design coverage. Use a fresh results directory before the full run.

The Python entry point is `python extension.py --design DESIGN.json --out results/shard-0`; defaults select those files relative to this suite, regardless of the current working directory. Output must remain below the selected design's directory and scratch below its `scratch/` directory. `--shard`, `--shards` and `--group` control selection. `--resume-from DIRECTORY` accepts prior `shard-*/receipt.json`, long tables and raw evidence with matching design and production DVH hashes; it copies completed groups and retains their execution records.

| Setting | Default and purpose |
|---|---|
| `RTPIPELINE_SRC` | Repository root, two levels above this suite; supplies source and test fixtures |
| `RTPIPELINE_PYTHON` | `python` on PATH; selects the production interpreter in shell wrappers |
| `DVH_SCRATCH` | `scratch/extension-i` or `scratch/legacy-i` inside this suite; also supplies `TMPDIR` |
| `DICOMPYLER_PYTHON` | `.venv-dicompyler/bin/python` inside this suite; used only for optional axial replay |
| `DVH_SHARED_CT_CACHE` | Unset; optionally set to `scratch/shared-ct` inside this suite to reuse identical closed CT files |
| `DVH_DESIGN` | This suite's `DESIGN.json`; optional design override for unit tests |

The generator removes each temporary synthetic course after extracting evidence, including on errors. Shared CT caching checks byte identity on concurrent publication. Delete scratch and optional cache files after processing. Source, test and summary artifacts contain no patient input.

## Expected outputs and audit

Each shard produces `metrics_long.csv`, `receipt.json` and `raw/<group>/` with synthetic manifests, input hashes, masks' sampling evidence, published production metrics, curves and QC. The receipt records the design hash, source hashes, package versions, commands and execution errors. A production refusal/nonmeasurement is retained as a case result; a harness exception is labelled `execution_error` and makes the runner exit unsuccessfully.

The summarizer creates `summary.csv`, `boundary_exceedances.csv`, `unavailable.csv`, `counts_by_stratum.csv`, `case_execution_status.csv`, `volume_error_components.csv` and `completeness.json`. It also writes a readable result table. The standard-library verifier independently recomputes analytic references, errors, percentage conversion, boundary status, case coverage and every summary statistic; it checks design timing/hash and published path/mask provenance. `verification.json` records failures and completeness separately; errors produce a nonzero exit status.

Dose endpoints are in Gy; Dp is dose to the hottest p% of the structure. D2cc and D0.03cc require the corresponding measured volume. Signed error is production minus continuous truth. V20Gy is in cm³; V20Gy percentage is 100 × published V20Gy / published structure volume. Percentage errors use percentage points. Relative errors remain undefined for zero reference values. Missing production values and unavailable analytic references have separate reasons and denominators.

All endpoints below 1 cm³, Dmin, Dmax, D0.03cc and V20Gy in cm³ are descriptive. Other dose boundaries are absolute error >max(0.02 × absolute analytic endpoint, 0.5 Gy); percentage-volume boundaries are >2 percentage points; structure-volume boundaries are >3%. The summary counts available comparisons, bounded comparisons and exceedances separately.

Recorded results are [summary.csv](reference/summary.csv), [boundary_exceedances.csv](reference/boundary_exceedances.csv), [unavailable.csv](reference/unavailable.csv) and [counts_by_stratum.csv](reference/counts_by_stratum.csv). Their numeric values are preserved from the recorded execution. Recorded reason strings cite line numbers of the tested commit; fresh runs cite production modules and functions. [provenance.json](reference/provenance.json) records summary hashes and source identity. Raw per-case output, source environments and historical receipts are not bundled; these tables do not assert that the complete benchmark has been rerun from this checkout.

## Optional axial replay

`run_benchmark.py` retains the original 120-case, 36-course axial design at 2.5 mm dose spacing. It compares production output with native and harmonized dicompyler-core endpoints on the same serialized DICOM. Since production also uses dicompyler-core, that comparator shares the rasterization engine; continuous analytic truth supplies the independent reference.

Use a separate environment with the direct pins in `requirements-dicompyler.txt` or all pins in `requirements-dicompyler-lock.txt`, and set `DICOMPYLER_PYTHON` if it is outside the default location. Run `bash run_legacy.sh 0 1` from this suite, or all `i N` partitions. `merge_legacy.py SUITE_DIRECTORY` merges disjoint completed batches from `results/legacy-shard-*`; `compare_legacy.py --reference REFERENCE_RESULTS_DIRECTORY` optionally checks exact decimal values and serialized-input hashes against a user-supplied prior run. No external reference table is assumed. For a complete unsharded axial run, invoke `run_benchmark.py --dicompyler-python "$DICOMPYLER_PYTHON" --results "$DVH_BENCH/results/axial"`. `analyze_volume.py`, `finalize_results.py` and `verify_results.py` accept that complete run directory through `--results`. The merged replay table is intended for the exact-value comparison.

## Limitations

Regular analytic shapes, smooth fields and binary masks do not represent clinical anatomy or segmentation uncertainty. The tested positive rotations and one over-limit configuration do not exhaust orientation behavior; tilted contours are outside the design. The <1 cm³ class pools two declared sizes. Threshold-volume results are sensitive to sampling and histogram conventions. Error-component tables describe measured differences without assigning an unmeasured causal contribution. These comparisons do not establish clinical validity.
