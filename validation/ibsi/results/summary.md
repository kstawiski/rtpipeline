RTpipeline CT radiomics: public IBSI 1 phantom conformance

Agreement depended on preprocessing and feature definition. With one aggregation per feature, the digital phantom matched 100/102 features (98.0%). Native CT extraction matched 101/102 in configuration A, 54/102 in B, 27/102 in C and 23/102 in D. These results do not demonstrate full IBSI 1 conformance.

Only the public IBSI digital phantom and the public lung CT phantom with its supplied GTV mask were used. No clinical cohort was accessed. The installed software was RTpipeline 2.3.99, PyRadiomics 3.0.1, NumPy 1.26.4 and SimpleITK 2.5.3.

The subset with one aggregation per feature contains 102 mapped Original-image outputs: GLCM and GLRLM are direction-averaged, with volumetric matrices for C–E and the digital phantom, and pooled axial matrices for A–B. Other texture families use the corresponding 3D or 2.5D matrix. This subset excludes the five PyRadiomics outputs without an IBSI 1 counterpart.

| Extraction | Configuration | One aggregation per feature | All computed comparison pairs | Reference coverage |
|---|---|---:|---:|---:|
| digital | digital | 100/102 (98.0%) | 252/254 (99.2%) | 254/482 (52.7%) |
| native | A | 101/102 (99.0%) | 140/141 (99.3%) | 141/346 (40.8%) |
| native | B | 54/102 (52.9%) | 65/141 (46.1%) | 141/346 (40.8%) |
| native | C | 27/102 (26.5%) | 40/141 (28.4%) | 141/210 (67.1%) |
| native | D | 23/102 (22.5%) | 30/141 (21.3%) | 141/210 (67.1%) |
| native | E | Not run | Not run | 0/210 (0.0%) |
| ibsi_preprocessed | A | 101/102 (99.0%) | 140/141 (99.3%) | 141/346 (40.8%) |
| ibsi_preprocessed | B | 101/102 (99.0%) | 140/141 (99.3%) | 141/346 (40.8%) |
| ibsi_preprocessed | C | 79/102 (77.5%) | 109/141 (77.3%) | 141/210 (67.1%) |
| ibsi_preprocessed | D | 101/102 (99.0%) | 140/141 (99.3%) | 141/210 (67.1%) |
| ibsi_preprocessed | E | 101/102 (99.0%) | 140/141 (99.3%) | 141/210 (67.1%) |

A comparison pair is one feature at one aggregation method. A feature evaluated using both averaged and merged matrices contributes two pairs. Reference coverage includes all standardized reference pairs in the workbook for that configuration; it excludes diagnostics and entries without a consensus value. It therefore also includes unsupported families and slice-wise aggregation methods. Proportions among computed features must be read together with this coverage. No pooled percentage across phantoms or configurations is used.

Native extraction and explicit IBSI preprocessing

Every extraction used RTpipeline’s `radiomics._extractor` and `radiomics_ct_contract.build_ct_extractors`, followed by the configured PyRadiomics `execute` method. Shape used the separate, unresegmented morphology mask. The numerical extraction ran inside the designated radiomics environment, so the conda-launch wrapper added no numerical operation and was not invoked. Clinical case ingestion, RTSTRUCT conversion and publication orchestration were outside this test.

Native A used the original spacing, the HU interval [−500, 400] and 25-HU bins. Native B requested 2 × 2 mm in-plane linear image interpolation, [−500, 400] and 32 bins. Native C requested 2-mm isotropic linear image interpolation, [−1000, 400] and 25-HU bins. Native D requested the same 3D interpolation, 3-SD outlier exclusion and 32 bins. E cannot be represented unchanged by a single standard PyRadiomics parameter file because it combines absolute-range and sigma resegmentation; its native rows are marked not computed. Native B–D are implementation comparisons with IBSI targets, not claims that the complete IBSI preprocessing protocol is expressible by native settings. [IBSI configurations](https://github.com/theibsi/ibsi-doc/blob/b952e71e9d40569e8a9ff3c740bc389c04dd6c28/docs/05_Reference_data_sets.rst), [PyRadiomics preprocessing definitions](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/docs/faq.rst).

The explicit preprocessing arm used the IBSI-centered grid with ceiling-rounded dimensions; bilinear interpolation for B, trilinear interpolation for C–D, and cubic B-spline image interpolation for E; linear floating-point mask interpolation with a ≥0.5 threshold; and nearest-integer HU rounding. Morphology and intensity masks remained separate. A–C applied their specified HU intervals. D excluded values beyond three population SDs of the unresegmented ROI. E intersected that outlier criterion with [−1000, 400]. These were fixed rules per configuration. PyRadiomics feature formulas and fixed-bin-width discretization were retained. No feature value was shifted to obtain agreement. All 300 preprocessing diagnostics matched the official workbook under its comparison rule; values are in `preprocessing_diagnostics.csv`.

The available whole-volume modes were 3D averaged and 3D merged GLCM/GLRLM matrices, and 2.5D direction-merged and fully merged axial matrices. The digital phantom covered both dimensionalities. A–B covered the two 2.5D modes; C–E covered the two 3D modes. In PyRadiomics, `force2D` suppresses through-plane links and pools matrices across slices. It does not perform the separate per-slice feature averaging used by the IBSI 2D reference variants. Those variants are explicitly not computed here.

Family results

native; within tolerance / computed pairs.

| Family | Digital | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| Morphology | 10/11 | 11/11 | 10/11 | 6/11 | 6/11 | — |
| Intensity statistics | 14/15 | 14/15 | 13/15 | 1/15 | 2/15 | — |
| Intensity histogram | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | — |
| GLCM | 92/92 | 46/46 | 15/46 | 14/46 | 5/46 | — |
| GLRLM | 64/64 | 32/32 | 8/32 | 12/32 | 10/32 | — |
| GLSZM | 32/32 | 16/16 | 10/16 | 4/16 | 1/16 | — |
| NGTDM | 10/10 | 5/5 | 3/5 | 1/5 | 2/5 | — |
| NGLDM / PyRadiomics GLDM | 28/28 | 14/14 | 6/14 | 2/14 | 4/14 | — |

ibsi_preprocessed; within tolerance / computed pairs.

| Family | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| Morphology | 11/11 | 11/11 | 11/11 | 11/11 | 11/11 |
| Intensity statistics | 14/15 | 14/15 | 14/15 | 14/15 | 14/15 |
| Intensity histogram | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| GLCM | 46/46 | 46/46 | 40/46 | 46/46 | 46/46 |
| GLRLM | 32/32 | 32/32 | 20/32 | 32/32 | 32/32 |
| GLSZM | 16/16 | 16/16 | 10/16 | 16/16 | 16/16 |
| NGTDM | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 |
| NGLDM / PyRadiomics GLDM | 14/14 | 14/14 | 8/14 | 14/14 | 14/14 |

`family_summary.csv` also reports uncomputed features, unmappable outputs, agreement against all reference pairs, reference coverage and strict unrounded comparisons. The enabled PyRadiomics feature set has no GLDZM, local-intensity or intensity-volume-histogram outputs. It also lacks much of the morphology and intensity-histogram reference set. Deprecated getters omitted by the production feature-class convention remain not computed; no substitute formulas were introduced. The five unmappable outputs are TotalEnergy, GLCM MCC and the three maximum 2D diameters in the shape class.

Feature definitions and remaining discrepancies

Feature identities follow the [published PyRadiomics IBSI dictionary](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/labs/pyradiomics-dcm/resources/featuresDict_IBSIv7.tsv) and the [official IBSI identifier catalogue](https://github.com/theibsi/ibsi_identifier/blob/d90d867cf4bef7eaf09d2a2990bfdb650aaa201e/ibsi_concepts.csv). Class and getter assignments were checked against the installed v3.0.1 implementation. `feature_mapping.csv` records each assignment and its basis. IBSI GLDZM features were not equated to GLSZM features. Kurtosis was mapped to the corresponding IBSI identifier while retaining its documented definition difference.

PyRadiomics reports kurtosis rather than excess kurtosis. In the digital phantom, the raw value was 2.6453795193 against −0.355, with published tolerance 0. It was outside tolerance in every CT configuration as well. Subtracting three was not part of this validation. [PyRadiomics definition](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/docs/faq.rst).

The digital phantom’s least-axis length was 8.4781094932 mm against 8.54 mm, with tolerance 0.05 mm; the workbook difference was 0.06 mm. The installed shape implementation normalizes its covariance by N. Using N−1 for this 74-voxel phantom gives 8.5359812196 mm, consistent with the reference. This normalization explanation is an inference from the implementation and reference value, not an asserted correction to either definition. [PyRadiomics shape source](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/radiomics/shape.py), [IBSI shape references](https://github.com/theibsi/ibsi-doc/blob/b952e71e9d40569e8a9ff3c740bc389c04dd6c28/docs/03_Image_features.rst).

Configuration C retains a fixed-bin-width difference after explicit preprocessing: PyRadiomics resets the lowest occupied bin to 1, whereas IBSI uses the specified −1000-HU lower bound. Here the minimum ROI intensity was −939 HU, so the bin indices differ by two. This affects gray-level-sensitive features: for example, direction-averaged GLCM joint average was 36.9779305122 against 39.0, tolerance 0.2. C had 32 outside-tolerance pairs, including kurtosis. A’s observed minimum of −500 HU coincided with its prescribed lower bound. [Discretization definitions](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/docs/faq.rst).

Native resampling differs from IBSI in grid alignment, mask interpolation and intensity rounding. These differences propagate through shape and texture features. The explicit preprocessing arm isolates the retained PyRadiomics feature and binning conventions; its high agreement in B, D and E does not establish native pipeline agreement with those protocols. [Documented differences](https://github.com/AIM-Harvard/pyradiomics/blob/08bea7067e350303eead533b471aa60697b3b8c3/docs/faq.rst).

Comparison rule

The primary status reproduces the official workbook: `ABS(ROUNDDOWN(reference − result, 2 − floor(log10(abs(reference))))) ≤ tolerance`; for a zero reference the workbook uses zero decimal places. This truncation is material when tolerance is zero or a reference is rounded. `absolute_error` and `strict_raw_within_tolerance` retain the unrounded comparison. Missing consensus values are not assigned a conformance outcome. The unversioned [official reference workbook](https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx), linked from the [IBSI 1 results page](https://theibsi.github.io/ibsi1/), is pinned by SHA256 `3af20577b580f2161bd546e147338ff226abb1676488e2906f2ce908b7985710`. Participant-submission values and older tolerance tables were retained as source context, not used as comparison targets.

Default production CT configuration

| Setting | RTpipeline production default | IBSI benchmark runs |
|---|---|---|
| Discretization | Fixed width, 25 HU | Digital: width 1; A/C: width 25 HU; B/D/E: 32 bins |
| Resampling | 1 × 1 × 1 mm | A/digital: none; B: 2 × 2 mm in plane; C–E: 2-mm isotropic |
| Image interpolation | Cubic B-spline | B: linear in plane; C/D: trilinear; E: cubic B-spline |
| Image signals | Original, LoG σ 1/2/3/5 mm, coif1 wavelets, level 1 | Original only |
| Distances | [1, 2] | [1] |
| Dimensionality | 3D | 2.5D for A/B; 3D for C–E; both for digital |
| Normalization / energy shift | Off / 0 | Off / 0 |
| Minimum ROI | 64 voxels; at least two dimensions | Same extraction admission thresholds |
| Mask handling | Separate unresegmented morphology; nearest-neighbor native resampling | Explicit arm: linear mask interpolation, ≥0.5 threshold |

The production primary arm uses [−1000, 400] HU for targets and hollow pelvic organs, and [−500, 400] HU for solid soft-tissue/neural structures. Its sensitivity arm uses the unresegmented intensity mask. Shape is extracted independently from the unresegmented morphology mask on the Original image. Other ROI classes can be excluded by the production class policy. Full settings, enabled classes and the configured YAML hash are retained in `production_configuration.json`. This default configuration was recorded, not run as an IBSI configuration. The LoG/wavelet outputs and distance-2 production textures were not benchmarked by this IBSI 1 experiment.

Scope and reproducibility

This is a software comparison on two public test objects. It does not measure segmentation reproducibility, scanner robustness, clinical prediction or performance in patient cohorts. Local execution does not establish clinical or manuscript readiness. Exact URLs, commits, file hashes, installed-source hashes, effective settings and UTC times are in `run_receipt.json`; reproduction commands are in `../README.md`.
