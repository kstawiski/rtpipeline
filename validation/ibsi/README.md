RTpipeline radiomics: public IBSI 1 conformance
=============================================

This benchmark compares RTpipeline with the official IBSI digital phantom, public lung CT phantom and supplied GTV mask. Public inputs remain in the ignored data directory; numerical results and portable provenance receipts are retained in `results/`. No clinical cohort is used.

Read [results/summary.md](results/summary.md) for the results and their limits. [results/feature_conformance.csv](results/feature_conformance.csv) contains every reference row for every extraction arm, plus PyRadiomics outputs without an IBSI counterpart. [results/family_summary.csv](results/family_summary.csv) contains family denominators and conformance proportions. [results/run_receipt.json](results/run_receipt.json) records versions, UTC times, exact source URLs, commits, hashes and commands. [results/verification.json](results/verification.json) records executed integrity and numerical checks.

The principal distinction is native extraction versus explicit IBSI preprocessing. Native A–D use RTpipeline's configurable extractor with PyRadiomics preprocessing. E requires combined range and sigma resegmentation and is not available unchanged in one standard native parameter file. The explicit arm prepares A–E according to the IBSI preprocessing rules, then uses the same RTpipeline extractor construction and PyRadiomics feature formulas. Shape always uses the unresegmented morphology mask. These comparisons do not evaluate the full clinical ingestion and publication workflow.

Reproduce the numeric results
-----------------------------

Use the supplied radiomics environment with Python isolated mode. The scripts derive source and validation paths from their own location; `-B` prevents bytecode writes to existing source directories. Set `RTPIPELINE_RADIOMICS_PYTHON` to the NumPy 1.x/PyRadiomics interpreter, or run from that environment.

```bash
cd validation/ibsi  # from the repository root
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/run_validation.py > results/execution.log 2>&1
OPENBLAS_NUM_THREADS=1 "${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/check_preprocessing.py
"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/report.py
"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/verify_results.py --finalize
"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/verify_results.py
```

To check the retained result without rerunning extraction:

```bash
cd validation/ibsi  # from the repository root
"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/verify_results.py
```

To restore missing downloaded sources, preserve `results/run_receipt.json` and run:

```bash
cd validation/ibsi  # from the repository root
"${RTPIPELINE_RADIOMICS_PYTHON:-python}" -I -B scripts/retrieve_pinned.py
```

This command uses the main Persistent Chrome bridge and exact URLs and SHA256 values from the receipt. The `persistent-chrome` wrapper must be on PATH, or `PERSISTENT_CHROME_BRIDGE` must identify it. It opens and closes its own tab. It neither searches for newer versions nor substitutes an HTTP client. The GitHub sources are commit-pinned. The official reference workbook has no release identifier; its SHA256 pins its retrieved contents. A content change at its URL is an error, not permission to substitute new reference values. Cached valid sources are verified without downloading them. Offline extraction needs only the retained `data/public_sources_20261027/` directory. Retaining the receipt is necessary for exact restoration.

`scripts/retrieve.py` records the initial browser retrieval and supports explicit repository files or URLs. `retrieve_pinned.py` is the exact reproduction path. Downloads are data; none of the downloaded scripts or workbooks is executed.

Interpret the tables
--------------------

`status` is `within tolerance`, `outside tolerance`, `not computed` or `not mappable`. The primary comparison follows the official workbook's three-significant-digit truncation of the error before comparing with tolerance. This matters especially for the digital phantom's zero-tolerance rows. The CSV also retains the unrounded absolute error and strict raw comparison. Blank consensus values are excluded from conformance denominators.

One comparison pair is one feature at one aggregation method. The family table gives both agreement among computed pairs and agreement/coverage against all standardized reference pairs, including unsupported families and aggregations. Diagnostics are excluded from radiomic-feature proportions. All 300 explicit-preprocessing diagnostics are compared separately in [results/preprocessing_diagnostics.csv](results/preprocessing_diagnostics.csv). The `morphology` and `intensity` records in `extraction_runs.json` describe arrays supplied to the extractor; for native resampling, use the retained `diagnostics_*` values in `results/raw/` for the grids and masks actually used internally.

No per-feature settings or value adjustments were used. In particular, the kurtosis value was not reduced by three, and PyRadiomics fixed-bin-width indexing was retained. Missing or deprecated features were not reconstructed from alternative formulas. The five PyRadiomics outputs without an IBSI 1 counterpart are retained as not mappable.

The production default is recorded separately in [results/production_configuration.json](results/production_configuration.json): 25-HU width, 1-mm isotropic B-spline resampling, distances [1,2], Original/LoG/coif1 wavelet signals, no normalization and zero energy shift. Production primary intensity windows depend on ROI class; morphology remains unresegmented. This production configuration has no exact IBSI 1 counterpart. Its filtered features and distance-2 textures were not benchmarked here.

Public sources and attribution
------------------------------

- [IBSI phantom repository](https://github.com/theibsi/data_sets): commit `6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`.
- [IBSI manual](https://github.com/theibsi/ibsi-doc): commit `b952e71e9d40569e8a9ff3c740bc389c04dd6c28`.
- [IBSI reference values](https://theibsi.github.io/ibsi1/), linked [official workbook](https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx): SHA256 `3af20577b580f2161bd546e147338ff226abb1676488e2906f2ce908b7985710`.
- [IBSI consensus-analysis repository](https://github.com/theibsi/ibsi_1_data_analysis): commit `4feb75e02c154b295fcee7def0dc70e47b176e6e`. Historical tables and the PyRadiomics participant submission are retained as context; their submitted outputs are not used as targets.
- [Published PyRadiomics IBSI mapping](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/labs/pyradiomics-dcm/resources/featuresDict_IBSIv7.tsv) and [documented differences](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/docs/faq.rst), release v3.0.1.
- [Official IBSI identifiers](https://github.com/theibsi/ibsi_identifier/blob/d90d867cf4bef7eaf09d2a2990bfdb650aaa201e/ibsi_concepts.csv).

The digital phantom was designed by Alex Zwanenburg and is CC BY 4.0. The CT phantom was contributed by Philippe Lambin, distributed through CancerData (DOI:10.17195/candat.2016.08.1), and converted to NIfTI by Alex Zwanenburg; its license is CC BY-NC 3.0. Both original README and license files are retained with their hashes. The entire data directory is ignored by `.gitignore`; the largest downloaded file is 3,858,588 bytes.

This experiment measures software agreement on two public test objects. It does not establish clinical or manuscript readiness.
