# RTpipeline production convolutional filters: public IBSI 2 benchmark

Read [results/summary.md](results/summary.md) for findings and limitations. This experiment tests software agreement on public IBSI digital phantoms and the public lung CT phantom. It does not use a clinical cohort. RTpipeline source and the IBSI 1 benchmark remain read-only.

## Reproduce

All commands write inside this directory. Use the supplied interpreter in isolated mode; `-B` prevents bytecode writes to existing source. Downloaded files, including MATLAB, R and LaTeX sources, are treated as data and never executed.

```bash
cd /umed-projekty/rtpipeline/validation/ibsi2
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python -I -B scripts/retrieve.py --restore
/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python -I -B scripts/run_validation.py > results/execution.log 2>&1
/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python -I -B scripts/report.py
/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python -I -B scripts/verify_results.py --finalize
/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python -I -B scripts/verify_results.py
```

Keep `results/run_receipt.json` to restore the exact source files. `retrieve.py --restore` uses its URLs, commits and SHA256 values and refuses changed contents. Cached files with matching hashes are reused. It opens and closes a task-owned tab through `/umed-projekty/_t3/scripts/persistent-chrome`; no unauthenticated HTTP client is substituted. Restoring sources does not retry the stopped publisher access branch. For the initial pinned repository acquisition, `scripts/retrieve_sources.py` contains the selected paths and repository commits. Its API tree downloads are also hashed. All source files live in the new `data/public_sources_20261106/` directory, which is ignored by Git; no individual downloaded file exceeds 20 MB.

## Tables and comparison rules

- `configuration_map.csv`: all 36 phase 1 and 22 phase 2 configurations; filter dimension, boundary, support, scale normalisation, wavelet family, decimation, level, sub-band, rotation, pooling and implementation evidence. Expressibility describes the **filter**. A separate column identifies preprocessing differences. Native settings that differ from the reference remain labelled partial.
- `response_map_agreement.csv`: native operations and diagnostic adapters. The official rule is an absolute voxel error no larger than 1% of the reference map's intensity range, at **every** voxel. Fractions are calculated over the entire 64 × 64 × 64 array, without masking, alignment optimisation or interpolation of references. The historical MATLAB combined-range calculation is retained in an additional column; it is not the primary rule.
- `feature_conformance.csv`: 396 assessed feature–configuration pairs, including 72 explicit gaps for configurations absent from the official CSV, plus selected native outputs with no reference counterpart. The official CSV contains 324 rows and 323 reference values. `record_type` separates reference comparisons from extra outputs. An error no greater than the supplied tolerance passes; no error rounding or IBSI 1 truncation rule is applied.
- `feature_attribution.csv`: one fixed sequence of diagnostic changes per configuration. `parent_variant` and `changed_factor` identify each comparison. No setting is selected separately for an individual feature.
- `family_summary.csv`: denominators and statuses by configuration and feature family. `production_feature_coverage.csv` retains every computed production filtered feature and its exact-reference coverage status.
- `run_receipt.json`: UTC times, versions, source URLs, commits, file hashes, source-code hashes, execution environment, output hashes and publisher access evidence. `verification.json` records executed checks.

`within tolerance`, `outside tolerance`, `not computed` and `not mappable` describe individual comparisons. A numerical match from a partial configuration or a diagnostic adapter does not establish native conformance. Missing consensus values are never filled in. PyRadiomics kurtosis is retained as computed and classified as not mappable to IBSI excess kurtosis; three other IBSI statistics lack native implementations. Their values are not reconstructed.

## Filtering and preprocessing

Native CT extraction calls `rtpipeline.radiomics._extractor`, which constructs PyRadiomics 3.0.1 and applies the supplied YAML parameters. Native phase 1 response maps call the same `radiomics.imageoperations.getLoGImage` and `getWaveletImage` generators as that extractor. The production arm uses `rtpipeline/radiomics_params.yaml` unchanged: 1-mm cubic B-spline resampling, 25-HU bin width, LoG sigma 1, 2, 3 and 5 mm, coif1 start level 0 and level 1, all enabled production feature classes. The public phantom has no ROI class assignment; no class-specific CT window is imposed on this arm.

In code-evidence paths, `radiomics/` denotes `/home/konrad/micromamba/envs/rtpipeline-radiomics/lib/python3.10/site-packages/radiomics/`. The installed source hashes are retained alongside the RTpipeline source hashes.

The 2-mm digital phantoms are converted to floating point without resampling. CT and mask NIfTI inputs are rounded before processing as instructed by IBSI. Native phase 2 A uses no interpolation; B uses native PyRadiomics 1-mm cubic B-spline resampling. Both use the official pre-filter range resegmentation of −1000 to 400 HU. Only the requested sub-band is compared with the reference; all extractor outputs are retained in `results/raw/`.

Diagnostic adapters alter boundary extension, normalisation, derivative units, kernel support, kernel discretisation, orientation pooling or level-2 dilation. The LoG support test truncates the measured native impulse kernel at 4 sigma; the next step changes those coefficients to the manual's sampled Gaussian derivatives at the same cutoff. Neither adapter is a production replacement. Digital-phantom derivatives require distinguishing millimetre coordinates from voxel coordinates. Mirror extension includes the edge voxel (`numpy.pad(..., mode='symmetric')`, equivalent to SciPy `reflect`). Wavelet orientation averaging uses four 2D or 24 proper 3D right-angle rotations. The level-2 adapter uses PyWavelets' dilated stationary transform, whereas native PyRadiomics repeats undilated level-1 operations.

For 6.A and 7.A, the native filter input is the full image; the `padding40` diagnostic crops it to the ROI bounding box plus 40 voxels, clipped to image bounds. In the B arm, `padding40` increases the resampling ROI margin from 5 to 40 voxels. B-arm diagnostics test linear CT interpolation separately, change mask interpolation to trilinear at a 0.5 threshold and round interpolated CT values. The `interpolation_boundary_handling` control keeps that native lattice, cubic B-spline CT interpolation, linear mask and rounding while adding three replicated source edge voxels and an outside value of -1000 HU. The following `centered_grid` step changes grid alignment and extent using RTpipeline's existing `spatial_preprocess` with the same boundary settings. Grid alignment and interpolation boundary handling are tested separately. The centred-grid step resamples the full image before retaining a 40-voxel filter margin, larger than the finite kernels used here. `extraction_runs.json` retains geometry and ROI voxel counts for each diagnostic input image, morphology mask, intensity mask, response image and cropped feature image/mask, keyed by configuration and variant under `variants`, with `parent_variant`, `changed_factor` and resampling boundary settings. These adapters are labelled separately from the production arm.

## Sources

- Whybra P, et al. *Radiology*. 2024;310(2). [doi:10.1148/radiol.231319](https://doi.org/10.1148/radiol.231319).
- [Official IBSI 2 project](https://theibsi.github.io/ibsi2/) and [datasets](https://theibsi.github.io/datasets/).
- [Public phantoms](https://github.com/theibsi/data_sets/tree/6da96021bc91faf4c0cb7fd7fa56a4225d2064a8): commit `6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`.
- [Reference maps and feature values](https://github.com/theibsi/ibsi_2_reference_data/tree/5404579fb3e0d17e8db421f0e82d64ce2432ec03): commit `5404579fb3e0d17e8db421f0e82d64ce2432ec03`.
- [Filter manual](https://github.com/theibsi/ibsi2_manual/blob/b6ef5856f97a22af763cea5a424316f667a3de40/IBSI_Filters.tex): commit `b6ef5856f97a22af763cea5a424316f667a3de40`; benchmark filter settings, lung CT processing/filter configuration tables and reference-comparison chapters. Also [arXiv:2006.05470](https://arxiv.org/abs/2006.05470).
- [Official analysis code](https://github.com/theibsi/ibsi_2_data_analysis/tree/fde70ca61819799f673c07dca776deea065d2a7d): commit `fde70ca61819799f673c07dca776deea065d2a7d`.

The digital phantoms were designed by Alex Zwanenburg and are CC BY 4.0. The public CT phantom was contributed by Philippe Lambin, distributed through CancerData ([doi:10.17195/candat.2016.08.1](https://doi.org/10.17195/candat.2016.08.1)), and converted to NIfTI by Alex Zwanenburg; it is CC BY-NC 3.0. Downloaded README and licence files are retained. The publisher supplement could not be retrieved through main or US institutional browsing; its access messages and URLs are retained in the receipt. The official GitHub references support the numerical branches independently.
