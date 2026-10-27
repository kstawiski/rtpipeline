# Surface metrics and IBSI preprocessing

`compare_masks` measures surface agreement with the Nikolov/DeepMind surfel method.
Each 2 × 2 × 2 voxel neighbourhood contributes its marching-cubes surface area in
mm². Distances use physical spacing on the corner lattice. HD95 is the larger of
the two directed, area-weighted 95th percentiles, using the inverse empirical
cumulative distribution. Mean surface distance pools distance × area from both
directions and divides by their total area. Surface Dice divides the area within
the specified inclusive distance tolerance by the total area of both surfaces.
The comparison API, input volumes, grid checks, resampling policy and empty-mask
statuses retain their existing definitions.

The 256-entry normals table is vendored unchanged from Google DeepMind
[`surface-distance`, commit 1f805cea446805e559badc80dd27f34175e4173d](https://github.com/google-deepmind/surface-distance/tree/1f805cea446805e559badc80dd27f34175e4173d).
The surrounding calculation is adapted to RTpipeline's API; the pooled mean is
explicitly defined above. Copyright and attribution are in `eval/geometry.py`;
the [complete Apache 2.0 licence](../licenses/surface-distance-Apache-2.0.txt) is
distributed with this documentation. Comparison with that upstream implementation
on 96 random mask pairs, three spacings and four tolerances gave identical HD95
and absolute differences below 2 × 10⁻¹⁵ for the pooled mean and surface Dice.
[Executed comparison](../validation/f6_surfel_upstream.json).

For two spheres of radius 10 mm with centres 6 mm apart, the continuous
area-weighted distance distribution is uniform from 0 to 6 mm. Thus HD95 is
5.7 mm, mean distance is 3.0 mm and surface Dice at a 3-mm tolerance is 0.500.
The synthetic measurements were:

| Isotropic voxel size | HD95, mm | Mean distance, mm | Surface Dice at 3 mm | Equal-voxel surface Dice |
|---|---:|---:|---:|---:|
| Analytic | 5.700 | 3.000 | 0.500 | — |
| 1 mm | 5.385 | 2.557 | 0.624 | 0.564 |
| 0.5 mm | 5.523 | 2.752 | 0.558 | 0.537 |
| 0.25 mm | 5.573 | 2.859 | 0.523 | 0.513 |

At 0.5-mm spacing the equal-voxel surface Dice is 0.537 against analytic 0.500. Area weighting
does not remove corner-lattice distance error at finite voxel size, and its Dice
bias is larger in this particular example at 0.5 mm. All three surfel metrics
approach the analytic values with finer voxels. For concentric spheres of radii
8 and 12 mm, analytic mean distance and HD95 are both 4 mm. At 1 and 0.5 mm,
the measured HD95 values are 4.123 and 4.031 mm, and the means are 3.519 and
3.732 mm. Surface Dice at 3 mm is 0.145 and 0.000, against analytic 0.000;
at 5 mm it is 1.000 at both resolutions, against analytic 1.000.
[Full measurements](../validation/f6_surfel_measurements.json).

CT radiomics can use explicit IBSI preprocessing by adding this block to the
PyRadiomics parameter YAML supplied as `radiomics_params_file`:

```yaml
radiomics:
  preprocessing: ibsi
setting:
  resampledPixelSpacing: [2.0, 2.0, 2.0]
  interpolator: sitkLinear
  normalize: false
  resegmentShape: false
  resegmentRange: [-3.0, 3.0]
  resegmentMode: sigma
  binCount: 32
```

This example supplies D's preprocessing and discretisation. Exact benchmark
parameter files also specify Original images, distance 1, dimensionality,
aggregation and feature classes; the tested complete files are in
[`validation/f6_ibsi/`](../validation/f6_ibsi/).

The packaged implementation is adapted from
`validation/ibsi/scripts/run_validation.py::ibsi_preprocess`. It rounds input HU
values, uses dimensions `ceil(N × old_spacing / new_spacing)` and aligns the
physical centres of the input and output grids. A zero target spacing preserves
that axis; omitted or null spacing disables resampling. Image interpolation uses
the configured SimpleITK interpolator and three replicated edge voxels. The
binary morphology mask uses linear interpolation with an inclusive ≥0.5
threshold. Interpolated HU values are rounded to the nearest integer with
NumPy's ties-to-even convention. Input CT and mask grids must agree.

The morphology mask remains unresegmented for shape. `resegmentRange` and
`resegmentMode` specify the intensity criterion. Optional
`radiomics.resegment_sigma` intersects a symmetric population-SD criterion with
that range; both criteria use the unresegmented morphology ROI. For E, use
`resegment_sigma: 3`, `resegmentRange: [-1000, 400]`, `resegmentMode: absolute`
and `interpolator: sitkBSpline`. In the production CT dual-arm path, ROI-class
windows continue to define the primary arm. Sigma exclusion applies only to that
primary arm; the sensitivity arm retains the full morphology mask. Shape remains
separate. Normalisation, `removeOutliers` and `resegmentShape: true` are rejected
for this CT option. Preprocessing choices are included in effective settings,
parameter hashes and extraction diagnostics.

| IBSI configuration | Target spacing | Interpolation | Intensity mask | Discretisation |
|---|---|---|---|---|
| A | Native | None | [−500, 400] HU | Width 25 HU |
| B | [2, 2, 0] mm | Linear | [−500, 400] HU | 32 bins |
| C | [2, 2, 2] mm | Linear | [−1000, 400] HU | Width 25 HU |
| D | [2, 2, 2] mm | Linear | Within 3 population SDs | 32 bins |
| E | [2, 2, 2] mm | Cubic B-spline | [−1000, 400] HU and within 3 population SDs | 32 bins |

The option is implemented for the native CT extractor constructed by
`rtpipeline.radiomics._extractor`, including native CT dual-arm extraction.
It requires a NumPy 1.x/PyRadiomics environment. Conda helper and isolated
robustness factories that construct PyRadiomics directly do not support the
additional YAML block. NumPy 2.x delegation explicitly rejects an IBSI request;
direct PyRadiomics YAML parsing also rejects the additional block. MR extraction
does not support this CT option. Whole-image resampling can require more memory
than the native cropped-grid procedure.

On the public IBSI CT phantom, all 735 tested feature outputs across A–E and
the two aggregation modes matched the retained `ibsi_preprocessed` outputs
bit for bit. All 300 preprocessing diagnostics met the official workbook
tolerances. A, B, D and E each matched 140/141 computed reference comparison
pairs; the remaining pair was kurtosis, whose PyRadiomics definition differs
from IBSI excess kurtosis. C matched 109/141; its fixed-bin-width indexing
difference remains. No feature formulas or values are adjusted. The comparison
uses the workbook's truncated-error rule; the CSV also records the strict raw
comparison. Reference coverage remains 141/346 pairs for A/B and 141/210 for
C/D/E. These results cover the available mapped Original-image features and
aggregation methods, rather than the complete IBSI catalogue.
[Feature comparisons](../validation/f6_ibsi/feature_conformance.csv),
[feature equivalence](../validation/f6_ibsi/feature_equivalence.csv),
[preprocessing diagnostics](../validation/f6_ibsi/preprocessing_diagnostics.csv).
Definitions follow the pinned [IBSI reference configurations](https://github.com/theibsi/ibsi-doc/blob/b952e71e9d40569e8a9ff3c740bc389c04dd6c28/docs/05_Reference_data_sets.rst)
and [PyRadiomics preprocessing and feature definitions](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/docs/faq.rst).

Omitting the `radiomics` block retains the original native extraction path and
the packaged production parameter file. A synthetic CT extraction containing
1,223 Original, LoG and wavelet features plus 37 diagnostic entries had the reference
serialized fingerprint:
`4174115f0c5c677b17b56ea525606cbda57639c999035618ada18b69bcc08fd7`.
The [receipt](../validation/f6_ibsi/receipt.json) records versions and source hashes.

To reproduce the public benchmark without writing into `validation/ibsi`, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B \
  tests/validate_f6_ibsi.py --output docs/validation/f6_ibsi
```

The adapter calls the existing validation scripts' settings, feature mapping,
diagnostic functions and workbook comparator. It sends original phantom inputs
through the packaged option and compares outputs with the retained benchmark.
Only public and synthetic data are used. The results measure software agreement on the tested phantoms.
