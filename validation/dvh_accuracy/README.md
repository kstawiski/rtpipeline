# Synthetic DVH accuracy benchmark

This benchmark measures RTpipeline's production RTSTRUCT DVH workflow against continuous geometric truth. It also runs dicompyler-core in a separate virtual environment on the same serialized DICOM files. All images, identifiers, structures, plans and treatment records are synthetic. No patient data are required or read.

The design has 120 structure–dose cases in 36 synthetic courses:

| Variable | Values |
|---|---|
| Shape | Sphere, cylinder with height equal to diameter, cube, concentric spherical shell |
| Volume | 2, 50, 1,000 cm³ for each shape |
| Shell | Inner radius = 90% of outer radius; thickness = 10% of outer radius |
| Centre (mm) | (0, 0, 0); (0.37, −0.61, 0.43) |
| Dose spacing | Isotropic 2.5 mm |
| CT spacing | 2 × 2 mm in plane; slice thickness and separation 1 or 3 mm |
| Dose fields | Uniform 20 Gy; linear 20 + 0.1z Gy; radial 30 − 0.15r Gy |
| Radial field | Sphere and shell only; field centred on the structure |
| Reference dose | 25 Gy, so V95Rx uses 23.75 Gy |

The shell thickness is relative to its radius, not a fixed 2 mm. Dose and CT grids share an axial frame of reference. Each field covers every structure. Rotated dose grids and tilted contours are outside this design.

Spheres and shells contribute 36 cases each; cylinders and cubes contribute 24 each. Overall medians weight cases equally. Shape-specific summaries are provided because radial fields apply only to spheres and shells.

## Reproduction

Run commands in bash from the repository root. Benchmark data stay within this directory; pytest uses temporary directories under `/tmp`. The host environment supplies RTpipeline; the comparator environment is isolated. Network access is used only by pip to install public packages. No browser or account access is needed.

```bash
export DVH_REPO="$(pwd)"
export DVH_BENCH="$DVH_REPO/validation/dvh_accuracy"
export DVH_PY="${RTPIPELINE_PYTHON:-python}"
mkdir -p "$DVH_BENCH/tmp" "$DVH_BENCH/results"
export TMPDIR="$DVH_BENCH/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export RTPIPELINE_CONTRACT_HEADER_THREADS=4
export MPLCONFIGDIR="$DVH_BENCH/tmp/matplotlib"
nice -n 10 "$DVH_PY" -m venv "$DVH_BENCH/.venv-dicompyler"
nice -n 10 "$DVH_BENCH/.venv-dicompyler/bin/python" -m pip install --disable-pip-version-check --no-cache-dir --no-compile -r "$DVH_BENCH/requirements-dicompyler.txt"
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" -m pytest -q -p no:cacheprovider --basetemp=/tmp/rtpipeline-dvh-accuracy-analytic --junitxml="$DVH_BENCH/results/analytic-tests.xml" "$DVH_BENCH/test_analytic.py"
nice -n 10 "$DVH_PY" -m pytest -q -p no:cacheprovider --basetemp=/tmp/rtpipeline-dvh-accuracy-dvh --junitxml="$DVH_BENCH/results/dvh-tests.xml" tests/test_dvh*.py tests/test_planning_dose_selection.py tests/test_dose_grid_degenerate_contours.py
nice -n 10 "$DVH_PY" -m pytest -q -p no:cacheprovider --basetemp=/tmp/rtpipeline-dvh-accuracy-alignment --junitxml="$DVH_BENCH/results/alignment-tests.xml" tests/test_dvh_contour_plane_alignment.py
nice -n 10 "$DVH_PY" -m pytest -q -p no:cacheprovider --basetemp=/tmp/rtpipeline-dvh-accuracy-rtstruct --junitxml="$DVH_BENCH/results/rtstruct-tests.xml" tests/test_rtstruct_writers_exact_planes.py tests/test_auto_rtstruct_exact_planes.py tests/test_rtstruct_uniform_fastpath.py
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" "$DVH_BENCH/run_benchmark.py" > "$DVH_BENCH/results/benchmark.log" 2>&1
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" "$DVH_BENCH/analyze_volume.py"
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" "$DVH_BENCH/finalize_results.py"
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" "$DVH_BENCH/verify_results.py"
```

`requirements-dicompyler-lock.txt` records all installed comparator dependencies. Use it instead of `requirements-dicompyler.txt` in the pip command to pin transitive dependencies as well. The latter lists the direct dependency pins used for installation. Full primary-environment versions are in the run receipt; the benchmark uses the supplied RTpipeline environment rather than changing it.

The run deletes each synthetic course's temporary DICOM and image files immediately after extracting its evidence, including on failure. It retains input file hashes and production outputs. Remove the remaining temporary test and package cache files after all commands finish:

```bash
"$DVH_PY" -c 'import os, pathlib, shutil; shutil.rmtree(pathlib.Path(os.environ["DVH_BENCH"]) / "tmp")'
```

For one course, use an exact group identifier from `results/raw/`, for example:

```bash
mkdir -p "$DVH_BENCH/tmp"
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" "$DVH_BENCH/run_benchmark.py" --group v2_offset_ct1_dg2.5_linear --results "$DVH_BENCH/results-single"
```

For a smoke test, use `--limit 1 --results "$DVH_BENCH/results-smoke"`. The default runs the 36-group, 2.5-mm design and overwrites summary files. Use a separate `--results` directory for a partial design so that its results do not replace the full table.

To retain inspectable DICOM for one course:

```bash
nice -n 10 env PYTHONPATH="$DVH_REPO" "$DVH_PY" "$DVH_BENCH/generate_phantoms.py" --group v2_offset_ct1_dg2.5_linear --output "$DVH_BENCH/phantoms"
```

## Methods and endpoint definitions

`analytic.py` contains the continuous truth. Sphere and shell cross-sections use the exact spherical-cap polynomial. Cube and cylinder axial distributions are uniform in z. Linear-field quantiles invert the closed-form cumulative distribution by bounded bisection. Radial quantiles and the radial mean have explicit radius-power formulas. Uniform-dose DVHs retain the point mass at 20 Gy, including the inclusive V20 threshold. Independent quadrature checks the volumes, cumulative distributions and mean doses in `test_analytic.py`; additional tests check quantile inversion, symmetry, saturation and the absolute-volume boundary.

Dose endpoints are in Gy. Dp is the dose received by at least p% of the continuous structure. D2cc and D0.03cc use the hottest specified volume; an absolute-volume endpoint is undefined when the structure is smaller than the requested volume. At exactly 2 cm³, analytic D2cc equals Dmin. V95Rx is percentage volume; its absolute error is in percentage points. V20Gy and structure volume are in cm³. Relative error is the absolute error divided by the absolute analytic value; it is undefined when analytic truth is zero. Zero truth is retained in absolute-error summaries.

The DICOM generator evaluates exact shape cross-sections at CT slice centres. Circular contours have 256 equally spaced vertices and an area deficit below 0.0101% before slice discretization. Every shell contour uses `CLOSEDPLANAR_XOR`; the other structures use `CLOSED_PLANAR`. Nested shell contours use even/odd parity. The contour/slab reference volume is the sum of polygon areas multiplied by CT slice spacing; it is reported separately from continuous shape volume. No fitted geometry or sampled dose enters the analytic truth. Integer RTDOSE storage uses a scaling of 0.0001 Gy, with rounding error at most 0.00005 Gy at stored nodes. The tools use their default 0.01 Gy DVH bins.

The production call is:

```python
rtpipeline.dvh.dvh_for_course(
    course, parallel_workers=1, use_cropped=False,
    rx_dose_gy=25.0, max_total_dose_gy=100.0,
)
```

The generator uses the existing synthetic fixture's RTPLAN and course-contract builders, with the production CT provenance serializer and matching synthetic CT NIfTI. Synthetic delivery records allow reference-dose endpoints to follow the existing production contract rules. These records do not represent measured or delivered patient dose. The production function selects dose, aligns dose samples to the original contour planes, rasterizes contours, extracts metrics and publishes results. It retains the original contour positions and intervals. The benchmark reads its published parquet table. It does not replace or patch any function. `geometry.csv` records production `dose_plane_alignment` and separate `helper_snap_*` diagnostics. The latter describe an optional helper on a copied structure; they do not describe production contour handling.

The isolated worker imports dicompyler-core directly and reads the original DICOM files. It does not import RTpipeline. `dicompyler_native` uses the public mean/min/max and dose/volume constraint API. Its nearest-bin endpoint convention is retained, including zero values returned when a requested volume is larger than the measured volume. `dicompyler_harmonized` applies independently written linear cumulative interpolation and dose bounds to the same comparator DVH. This distinguishes metric extraction from workflow geometry effects.

One isolated comparator process serves the sequential cases. It reloads each case's DICOM and retains imports between cases. Identical CT files are hard-linked across dose fields and dose resolutions with the same CT geometry. The temporary CT cache is deleted when the next CT geometry starts; the process and remaining cache are closed at the end of the run.

The run uses one DVH worker, one numerical-library thread and four threads for CT header reads. The supplied and isolated environments have identical `dvh.py` and `dvhcalc.py` hashes. Their `dicomparser.py` files differ only in the pydicom compatibility import block; the exact difference is retained in `results/dicompyler_parser_difference.patch` and both file hashes are recorded in the receipt.

RTpipeline itself uses dicompyler-core for this path. The comparator therefore shares the underlying rasterizer and histogram engine; it is not an independent DVH algorithm. The analytic truth supplies the independent reference. The installed comparator uses dicompyler-core 0.5.6 and pydicom 2.4.4, which work without an RTpipeline compatibility shim. Exact installed versions, including transitive dependencies, are recorded in the run receipt.

The production axial path exports `D2ccGy` and `D2cc_status`. It uses the same `_small_volume_dose` helper as D0.03cc: linear cumulative-DVH interpolation at 2 cm³ divided by the measured structure volume, bounded by Dmin and Dmax. At exactly 2 cm³, D2cc equals Dmin. A smaller measured volume yields an empty dose with status `roi_below_2cc`; incomplete dose-grid coverage propagates its coverage status and leaves the whole-ROI value empty. The `rtpipeline` D2cc rows read the native published table. A separate `rtpipeline_curve_supplement` applies the same helper to the full published curve and checks agreement with the native endpoint. Both are unavailable when the measured volume is below 2 cm³.

## Results and evidence

The canonical results are in `results/`. Across 1,052 available native dose-endpoint
comparisons, the maximum absolute error is 0.309016 Gy. All meet the descriptive
boundary of max(2% of analytic dose, 0.5 Gy). All 120 V95Rx comparisons meet a
2-percentage-point boundary; the maximum error is 0.868749 percentage points.
Structure volume exceeds a 3% relative-error boundary in 29/120 cases.

| CT slice spacing | Cases | Maximum dose error, Gy | Maximum V95Rx error, percentage points | Volume errors >3% | Maximum volume error |
|---|---:|---:|---:|---:|---:|
| 1 mm | 60 | 0.120000 | 0.378788 | 13/60 | 14.6875% |
| 3 mm | 60 | 0.309016 | 0.868749 | 16/60 | 17.1875% |

The largest relative volume errors occur in small structures; 21/40 cases at
2 cm³, 8/40 at 50 cm³ and 0/40 at 1,000 cm³ exceed 3%. Native D2cc is available
in 92/120 cases, with maximum absolute error 0.218 Gy and 0/92 cases exceeding
max(2% of analytic dose, 0.5 Gy). It is unavailable in 28 cases whose measured
volume is below 2 cm³. The full-curve D2cc check agrees with the native endpoint.
[Boundary tables](results/boundary_report.md) retain these limits and denominators.


| File | Contents |
|---|---|
| `results/metrics_long.csv` | Each case, tool and metric; value, continuous truth, absolute/relative error, units, availability and design factors |
| `results/summary.csv` | Counts, median and maximum errors overall, by field, by shape/volume/resolution, and by phase/resolution |
| `results/summary.md` | Clinical endpoint tables, volume error tables, largest-error cases and interpretation limits |
| `results/worst_cases.csv` | Five largest absolute errors per tool and metric |
| `results/geometry.csv` | Polygon/slab volumes and production contour-plane diagnostics |
| `results/volume_error_components.csv` | Contour discretization and subsequent rasterization/plane-handling volume errors |
| `results/metric_status.csv` | Missing endpoints and native below-volume statuses |
| `results/paired_difference.csv` | Production values minus harmonized comparator values on the same DICOM |
| `results/input_sha256.json` | Synthetic input DICOM paths, byte lengths and SHA256 digests |
| `results/raw/<group>/` | Manifest, comparator results, published production metrics/curves/QC and synthetic contract |
| `results/run_receipt.json` | UTC times, interpreter version, package versions, portable commands, executed code hashes, source-integrity checks and errors |
| `results/verification.json` | Completeness, numeric consistency, zero-truth handling, source identity, coverage and curve integrity checks |

The receipt retains the benchmark hashes from execution and the pipeline source hashes measured at launch and completion. Public script hashes are recorded separately; the verifier also checks that the production DVH and contour-handling code match the executed hashes. `finalize_results.py` makes stored paths relative to the repository, removes host identity and records the public test-evidence hash. A completed run requires all production calls to return outputs and all expected structures to have dose metrics and full grid coverage. Analytic tests establish formula consistency only. Error summaries are descriptive for this synthetic design. The reported boundaries are descriptive software comparisons and are not clinical acceptance criteria.

`analyze_volume.py` adds a reproducible volume decomposition after the main run and records its own code hash and command in the receipt. Its original-contour rasterization residual uses the unsnapped comparator. RTpipeline uses the original contour planes with dose interpolated onto them. The residual therefore measures subsequent rasterization on those planes.

Relevant source: [production DVH](../../rtpipeline/dvh.py), [grid support](../../rtpipeline/dvh_support.py), [rotation path](../../rtpipeline/dvh_rotation.py), [contour snapping](../../rtpipeline/utils.py), and [synthetic course fixtures](../../tests/dvh_plan_target_fixtures.py). The rotation module's axial bypass is exercised; the rotated sampler is outside this benchmark.
