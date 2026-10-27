# Surface metric and IBSI validation

Synthetic controls test physical grids, area-weighted surface distances,
empty masks, and contour agreement. Comparison with the pinned DeepMind
implementation covers 96 random mask pairs and 384 tolerance-specific
comparisons. HD95 agrees exactly; maximum absolute differences are
1.78 × 10⁻¹⁵ for pooled mean distance and 4.44 × 10⁻¹⁶ for surface Dice.
[Upstream comparison](f6_surfel_upstream.json).
[Synthetic sphere measurements](f6_surfel_measurements.json).

The [IBSI receipt](f6_ibsi/receipt.json) records 735 feature outputs that match
the retained public benchmark bit for bit, and 300 preprocessing diagnostic
comparisons within the published tolerances. A, B, D and E each meet tolerance
for 140/141 computed feature comparisons; C meets tolerance for 109/141.
PyRadiomics kurtosis and fixed-bin-width differences remain in the results.
Reference coverage is 141/346 pairs for A/B and 141/210 for C/D/E.

The synthetic default extraction contains 1,223 radiomic features and
37 diagnostic entries. Its serialized fingerprint is
`4174115f0c5c677b17b56ea525606cbda57639c999035618ada18b69bcc08fd7`.
Tests cover separate morphology and intensity masks, non-unit labels,
quality-control counts, parameter identity and configured feature classes.

Reproduce the package-to-benchmark comparison from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B \
  tests/validate_f6_ibsi.py --output docs/validation/f6_ibsi
```

The radiomics environment needs NumPy 1.x and PyRadiomics. Public inputs are
restored with the pinned retrieval procedure in `validation/ibsi/README.md`.
The receipt verifies 26 public input hashes and 22 retained raw extraction hashes.

These measurements cover public phantoms and synthetic masks. Native CT
extraction supports this preprocessing option; the Conda helper and isolated
robustness factories do not. Finite-resolution sphere bias and feature-formula
differences are quantified in the [feature documentation](../features/surface_metrics_and_ibsi.md).
The clinical ingestion and publication workflow is outside this validation.
