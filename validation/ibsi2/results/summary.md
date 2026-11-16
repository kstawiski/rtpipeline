# RTpipeline production filters against IBSI 2

The production coif1 (Coiflet 1) HHL response map met the IBSI 2 voxel rule at all 262,144 voxels (maximum absolute difference 7.62939e-05; tolerance 4.4382). Native LoG failed both overlapping phase-1 scales. None of the 1,116 production filtered feature values (93 features from each of 12 filtered images) has an exact phase-2 reference configuration.

## Native response maps

A response map is the filtered image. LoG denotes the Laplacian of Gaussian, with scale sigma expressed in millimetres. Wavelet sub-band labels specify high-pass (H) or low-pass (L) filtering along the x, y and z image axes; HHL therefore applies high-pass filters in x and y and a low-pass filter in z.

Each comparison includes all 262,144 voxels. Passing requires every voxel to lie within 1% of the reference intensity range. A high voxel fraction can still fail this rule.

| IBSI configuration | Filter | Filter expressibility | Voxels within tolerance | Maximum absolute difference | Absolute tolerance | All voxels pass |
| --- | --- | --- | --- | --- | --- | --- |
| 2.a | LoG | partly expressible | 99.785233% | 8.2029 | 0.0674598 | False |
| 2.b | LoG | partly expressible | 0.000000% | 136.99 | 0.513723 | False |
| 5.a.1 | separable wavelet | partly expressible | 100.000000% | 1.52588e-05 | 2.35446 | True |
| 5.a.2 | separable wavelet | partly expressible | 99.955750% | 139.507 | 0.553904 | False |
| 6.a.1 | separable wavelet | expressible | 100.000000% | 7.62939e-05 | 4.4382 | True |
| 6.a.2 | separable wavelet | partly expressible | 56.460953% | 206.054 | 1.58925 | False |
| 7.a.1 | separable wavelet | partly expressible | 8.389664% | 1530 | 20.4 | False |
| 7.a.2 | separable wavelet | partly expressible | 71.352768% | 127.5 | 2.55 | False |

Only 6.a.1 is fully expressible as an unchanged native filter configuration. Seven configurations share a native family but differ in at least one parameter; their comparisons quantify those differences. The other 28 phase-1 configurations are not expressible. All 36 are retained in configuration_map.csv, and all 33 published response-map files were retrieved.

## Phase 2 feature comparisons

Across 396 assessed feature–configuration pairs, 31 were within tolerance, 67 outside tolerance, 218 not computed and 80 not mappable. Among 98 comparable native values, 31 were within tolerance. These counts combine the unfiltered controls and partial filtered configurations. Production coverage is reported separately below.

A denotes the 2D processing arm; B denotes the resampled 3D processing arm.

| Configuration | Within tolerance | Outside tolerance | Not computed | Not mappable | Established references |
| --- | --- | --- | --- | --- | --- |
| 1.A | 14 | 0 | 3 | 1 | 18 |
| 2.A | 0 | 0 | 18 | 0 | 18 |
| 3.A | 0 | 0 | 18 | 0 | 18 |
| 4.A | 0 | 0 | 18 | 0 | 18 |
| 5.A | 0 | 0 | 18 | 0 | 18 |
| 6.A | 1 | 13 | 3 | 1 | 18 |
| 7.A | 0 | 14 | 3 | 1 | 18 |
| 8.A | 0 | 0 | 18 | 0 | 18 |
| 9.A | 0 | 0 | 18 | 0 | 18 |
| 10.A | 0 | 0 | 0 | 18 | 0 |
| 11.A | 0 | 0 | 0 | 18 | 0 |
| 1.B | 12 | 2 | 3 | 1 | 18 |
| 2.B | 0 | 0 | 18 | 0 | 18 |
| 3.B | 1 | 13 | 3 | 1 | 18 |
| 4.B | 0 | 0 | 18 | 0 | 18 |
| 5.B | 0 | 0 | 18 | 0 | 18 |
| 6.B | 3 | 11 | 3 | 1 | 18 |
| 7.B | 0 | 14 | 3 | 1 | 18 |
| 8.B | 0 | 0 | 17 | 1 | 17 |
| 9.B | 0 | 0 | 18 | 0 | 18 |
| 10.B | 0 | 0 | 0 | 18 | 0 |
| 11.B | 0 | 0 | 0 | 18 | 0 |

The official CSV contains 324 rows with 323 consensus values. It omits 72 Riesz feature–configuration pairs; the quartile coefficient of dispersion for 8.B has no consensus value. Those 73 gaps are explicit and have no substituted reference values. The reference family is intensity statistics only. Texture matrices, histogram entropy/uniformity and 25-HU discretisation have no phase-2 targets. PyRadiomics does not compute median absolute deviation, coefficient of variation or quartile coefficient of dispersion. Its kurtosis differs from IBSI excess kurtosis and is classified as not mappable, with the unadjusted output retained.

## Attribution

The diagnostic sequences change one declared factor at each step and use the same setting for all features. The CSVs retain each parent comparison. Grid alignment and interpolation boundary handling were tested separately. The boundary control retains the native lattice, cubic B-spline CT interpolation, linear mask interpolation at a 0.5 threshold and CT rounding; it adds three replicated source edge voxels and an outside value of -1000 HU. The centred-grid step retains those boundary settings while changing grid alignment and extent.

On the 2-mm digital grid, removing LoG scale normalisation and converting physical derivatives to voxel-coordinate derivatives made 2.b pass at every voxel. Changing its boundary extension reduced the maximum absolute difference from 0.43941 to 0.353651; both were already within tolerance. For 2.a, one voxel remained outside tolerance after the normalisation and unit changes. Truncating the recursive kernel at 4 sigma retained that failure; substituting sampled Gaussian-derivative coefficients at the same cutoff removed it. This identifies kernel approximation as the residual difference on the impulse phantom.

Orientation averaging brought both pooled first-level wavelet configurations, 5.a.2 and 6.a.2, to 100% voxel agreement. For the second-level Haar configurations, boundary extension and orientation averaging were insufficient; adding the prescribed kernel dilation brought 7.a.1 and 7.a.2 to 100%. The unpooled db2 impulse configuration 5.a.1 passed with both native periodic and explicit zero extension: its central impulse does not challenge the boundary-condition difference.

| Configuration | Diagnostic change | Voxels within tolerance | Maximum absolute difference |
| --- | --- | --- | --- |
| 2.a | scale_normalisation_off | 99.824142% | 4.77433 |
| 2.a | voxel_derivative_units | 99.999619% | 0.0921289 |
| 2.a | specified_boundary | 99.785233% | 8.2029 |
| 2.a | boundary_and_no_scale | 99.824142% | 4.77433 |
| 2.a | boundary_no_scale_voxel_units | 99.999619% | 0.0921289 |
| 2.a | recursive_kernel_cutoff4 | 99.999619% | 0.0921286 |
| 2.a | finite_sampled_kernel | 100.000000% | 9.08428e-05 |
| 2.b | scale_normalisation_off | 1.138306% | 19.224 |
| 2.b | voxel_derivative_units | 100.000000% | 0.43941 |
| 2.b | specified_boundary | 0.000000% | 136.996 |
| 2.b | boundary_and_no_scale | 1.138306% | 19.224 |
| 2.b | boundary_no_scale_voxel_units | 100.000000% | 0.353651 |
| 2.b | recursive_kernel_cutoff4 | 100.000000% | 0.294531 |
| 2.b | finite_sampled_kernel | 100.000000% | 0.0168586 |
| 5.a.1 | specified_boundary | 100.000000% | 1.52588e-05 |
| 5.a.2 | specified_boundary | 99.955750% | 139.507 |
| 5.a.2 | orientation_pooling | 100.000000% | 3.8147e-06 |
| 6.a.2 | orientation_pooling | 100.000000% | 1.52588e-05 |
| 7.a.1 | specified_boundary | 9.303665% | 1004.06 |
| 7.a.1 | orientation_pooling | 24.078369% | 589.688 |
| 7.a.1 | atrous_level2 | 100.000000% | 0.000467936 |
| 7.a.2 | specified_boundary | 71.868515% | 127.5 |
| 7.a.2 | orientation_pooling | 72.784424% | 111.563 |
| 7.a.2 | atrous_level2 | 100.000000% | 3.05176e-05 |

CT feature attribution, reported over the same 14 natively mappable statistics per computed configuration:

| Configuration | Variant | Within tolerance | Outside tolerance |
| --- | --- | --- | --- |
| 6.A | padding40 | 1 | 13 |
| 6.A | mirror_extension | 1 | 13 |
| 6.A | orientation_pooling | 14 | 0 |
| 7.A | padding40 | 0 | 14 |
| 7.A | mirror_extension | 0 | 14 |
| 7.A | orientation_pooling | 0 | 14 |
| 7.A | atrous_level2 | 14 | 0 |
| 1.B | padding40 | 12 | 2 |
| 1.B | linear_CT_interpolation | 10 | 4 |
| 1.B | linear_mask | 12 | 2 |
| 1.B | integer_HU | 12 | 2 |
| 1.B | interpolation_boundary_handling | 12 | 2 |
| 1.B | centered_grid | 14 | 0 |
| 3.B | padding40 | 1 | 13 |
| 3.B | linear_CT_interpolation | 0 | 14 |
| 3.B | linear_mask | 1 | 13 |
| 3.B | integer_HU | 1 | 13 |
| 3.B | interpolation_boundary_handling | 1 | 13 |
| 3.B | centered_grid | 1 | 13 |
| 3.B | scale_normalisation_off | 13 | 1 |
| 3.B | mirror_extension | 13 | 1 |
| 3.B | recursive_kernel_cutoff4 | 11 | 3 |
| 3.B | finite_sampled_kernel | 14 | 0 |
| 6.B | padding40 | 3 | 11 |
| 6.B | linear_CT_interpolation | 0 | 14 |
| 6.B | linear_mask | 3 | 11 |
| 6.B | integer_HU | 3 | 11 |
| 6.B | interpolation_boundary_handling | 3 | 11 |
| 6.B | centered_grid | 3 | 11 |
| 6.B | mirror_extension | 3 | 11 |
| 6.B | orientation_pooling | 14 | 0 |
| 7.B | padding40 | 0 | 14 |
| 7.B | linear_CT_interpolation | 0 | 14 |
| 7.B | linear_mask | 0 | 14 |
| 7.B | integer_HU | 0 | 14 |
| 7.B | interpolation_boundary_handling | 0 | 14 |
| 7.B | centered_grid | 0 | 14 |
| 7.B | mirror_extension | 0 | 14 |
| 7.B | orientation_pooling | 0 | 14 |
| 7.B | atrous_level2 | 14 | 0 |

The unfiltered 3D control matched 12 of 14 comparable statistics with native preprocessing and 14 after the prescribed centred grid, linear mask interpolation and CT rounding. For 3.B (LoG sigma 1.5 mm), agreement was 1/14 natively, 13/14 after the preprocessing sequence and removal of scale normalisation, 11/14 after imposing the finite kernel support, and 14/14 with sampled coefficients. Linear CT interpolation did not improve agreement with the prescribed spline configuration.

For first-level db3, orientation averaging produced 14/14 matches in 2D and 14/14 in 3D. At level 2, the corresponding dilated-transform adapters produced 14/14 and 14/14. These results attribute the native wavelet differences to orientation pooling and the level rule under the fixed preprocessing sequence.

## Production coverage

| Production signal | Computed features | Response-map evidence | Feature-value evidence |
| --- | --- | --- | --- |
| log-sigma-1-0-mm-3D | 93 | none at exact scale; partial family coverage | none: no exact phase-2 production configuration |
| log-sigma-2-0-mm-3D | 93 | none at exact scale; partial family coverage | none: no exact phase-2 production configuration |
| log-sigma-3-0-mm-3D | 93 | none demonstrated; partial configuration overlap and native map fails | none: no exact phase-2 production configuration |
| log-sigma-5-0-mm-3D | 93 | none demonstrated; partial configuration overlap and native map fails | none: no exact phase-2 production configuration |
| wavelet-HHH | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |
| wavelet-HHL | 93 | direct response-map support | none: no exact phase-2 production configuration |
| wavelet-HLH | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |
| wavelet-HLL | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |
| wavelet-LHH | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |
| wavelet-LHL | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |
| wavelet-LLH | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |
| wavelet-LLL | 93 | none for this sub-band; partial family coverage | none: no exact phase-2 production configuration |

The principal message for clinicians is that reproducing one standardized filtered image does not validate the feature values from all filtered images. This benchmark provides no basis for a clinical decision threshold or a claim that the full production filtered feature set meets IBSI 2.

## Sources and method

Reference configurations and tolerances follow [Whybra et al., Radiology 2024](https://doi.org/10.1148/radiol.231319), the [official IBSI filter manual](https://github.com/theibsi/ibsi2_manual/blob/b6ef5856f97a22af763cea5a424316f667a3de40/IBSI_Filters.tex), and [official reference objects](https://github.com/theibsi/ibsi_2_reference_data/tree/5404579fb3e0d17e8db421f0e82d64ce2432ec03). The receipt pins every downloaded file by URL, commit and SHA256. RTpipeline constructs the native extractor; phase-1 maps use the same PyRadiomics image generators. Phase 2 A has no interpolation, while B uses native 1-mm B-spline resampling and an official pre-filter −1000 to 400 HU window. The separate production arm uses the production YAML unchanged. No settings were fitted to individual features.

## Limitations

This benchmark uses public synthetic objects and one public CT phantom. It does not establish clinical validity, generalisability or manuscript readiness. The publisher full text and supplement could not be retrieved: main and US browsers required sign-in, and no supplement file link was exposed. URLs, UTC times and verbatim access messages are retained in run_receipt.json; that branch stopped. The official GitHub reference objects and manual were retrieved independently. CT diagnostic filtering retains a 40-voxel margin after full-grid resampling; input precision is preserved for boundary and rotation tests. The official manual states no Riesz consensus, although the repository retains three Riesz response-map files; none is used to assert Riesz conformance.
