# Synthetic DVH accuracy benchmark

120 cases; full production `rtpipeline.dvh.dvh_for_course`, one worker. Dose grids: 2.5 mm. CT slices: 1, 3 mm; CT in-plane spacing: 2 mm. Shapes: sphere, cylinder (height = diameter), cube, and spherical shell (inner radius = 0.9 × outer radius). Volumes: 2, 50, 1,000 cm³. Centres: (0, 0, 0) and (0.37, −0.61, 0.43) mm.

Uniform dose: 20 Gy; axial gradient: 20 + 0.1z Gy; radial gradient: 30 − 0.15r Gy. Radial dose is centred on the sphere/shell. Reference dose is 25 Gy. V95Rx is reported as percentage volume; its absolute error is in percentage points. V20Gy and volume are in cm³. Relative error is |value − truth|/|truth|, left undefined for zero truth.

RTpipeline uses dicompyler-core internally. The isolated comparator uses unmodified dicompyler-core with pydicom 2.4.4. It provides a workflow comparison with shared rasterization, not an independent DVH engine. `dicompyler_native` uses its public metric API; `dicompyler_harmonized` uses the same raw DVH with independently written linear metric interpolation and bounds matching RTpipeline. Native API values, including zero outputs, are retained.

RTpipeline's median absolute volume error was 0.303125 cm³ (median relative error 1.25%). The largest relative volume error was 17.1875% in `v2_aligned_ct3_dg2.5_uniform_cube` (2.34375 versus 2 cm³). Median absolute D95 error was 0.01 Gy; the maximum was 0.233964 Gy.

## Absolute and relative errors

Each row pools all fields, sizes, phases and resolutions for that metric. N counts numeric comparisons; missing endpoints are excluded from errors and counted explicitly. Medians describe the tested cases, not population estimates. Spheres and shells contribute 36 cases each; cylinders and cubes contribute 24 each because radial fields apply only to spheres and shells.

| Tool | Metric | N/total | Median absolute | Maximum absolute | Median relative (%) | Maximum relative (%) |
|---|---|---:|---:|---:|---:|---:|
| dicompyler_harmonized | D0.03cc_Gy | 120/120 | 0.01 | 0.24964 | 0.05 | 1.05312 |
| dicompyler_harmonized | D2_Gy | 120/120 | 0.01 | 0.20251 | 0.05 | 0.896081 |
| dicompyler_harmonized | D2cc_Gy | 92/120 | 0.01 | 0.218 | 0.05 | 0.871198 |
| dicompyler_harmonized | D50_Gy | 120/120 | 0.01 | 0.262385 | 0.05 | 1.30911 |
| dicompyler_harmonized | D95_Gy | 120/120 | 0.01 | 0.233964 | 0.05 | 1.38313 |
| dicompyler_harmonized | D98_Gy | 120/120 | 0.01 | 0.271762 | 0.05 | 1.39808 |
| dicompyler_harmonized | Dmax_Gy | 120/120 | 0.01 | 0.276219 | 0.05 | 1.12237 |
| dicompyler_harmonized | Dmean_Gy | 120/120 | 0.005 | 0.112 | 0.025 | 0.558799 |
| dicompyler_harmonized | Dmin_Gy | 120/120 | 0.01 | 0.309016 | 0.05 | 2.84449 |
| dicompyler_harmonized | V20Gy_cc | 120/120 | 0.334621 | 27.6063 | 1.78148 | 40.625 |
| dicompyler_harmonized | V95Rx_pct | 120/120 | 0 | 0.868749 | 0.0188327 | 6.25513 |
| dicompyler_harmonized | Volume_cc | 120/120 | 0.303125 | 27.4437 | 1.25 | 17.1875 |
| dicompyler_native | D0.03cc_Gy | 120/120 | 0.398294 | 20 | 1.78945 | 100 |
| dicompyler_native | D2_Gy | 120/120 | 0.259149 | 20 | 1.14412 | 100 |
| dicompyler_native | D2cc_Gy | 120/120 | 19.3387 | 28.8276 | 100 | 100 |
| dicompyler_native | D50_Gy | 120/120 | 0.09 | 20 | 0.45 | 100 |
| dicompyler_native | D95_Gy | 120/120 | 0.17624 | 28.1965 | 1.15553 | 100 |
| dicompyler_native | D98_Gy | 120/120 | 19.3687 | 28.8355 | 100 | 100 |
| dicompyler_native | Dmax_Gy | 120/120 | 0.01 | 0.276219 | 0.05 | 1.12237 |
| dicompyler_native | Dmean_Gy | 120/120 | 0.005 | 0.112 | 0.025 | 0.558799 |
| dicompyler_native | Dmin_Gy | 120/120 | 0.01 | 0.309016 | 0.05 | 2.84449 |
| dicompyler_native | V20Gy_cc | 120/120 | 0.334621 | 27.6063 | 1.78148 | 40.625 |
| dicompyler_native | V95Rx_pct | 120/120 | 0 | 0.868749 | 0.00188729 | 6.25513 |
| dicompyler_native | Volume_cc | 120/120 | 0.303125 | 27.4437 | 1.25 | 17.1875 |
| rtpipeline | D0.03cc_Gy | 120/120 | 0.01 | 0.24964 | 0.05 | 1.05312 |
| rtpipeline | D2_Gy | 120/120 | 0.01 | 0.20251 | 0.05 | 0.896081 |
| rtpipeline | D2cc_Gy | 0/120 | — | — | — | — |
| rtpipeline | D50_Gy | 120/120 | 0.01 | 0.262385 | 0.05 | 1.30911 |
| rtpipeline | D95_Gy | 120/120 | 0.01 | 0.233964 | 0.05 | 1.38313 |
| rtpipeline | D98_Gy | 120/120 | 0.01 | 0.271762 | 0.05 | 1.39808 |
| rtpipeline | Dmax_Gy | 120/120 | 0.01 | 0.276219 | 0.05 | 1.12237 |
| rtpipeline | Dmean_Gy | 120/120 | 0.005 | 0.112 | 0.025 | 0.558799 |
| rtpipeline | Dmin_Gy | 120/120 | 0.01 | 0.309016 | 0.05 | 2.84449 |
| rtpipeline | V20Gy_cc | 120/120 | 0.334621 | 27.6063 | 1.78148 | 40.625 |
| rtpipeline | V95Rx_pct | 120/120 | 0 | 0.868749 | 0.0188327 | 6.25513 |
| rtpipeline | Volume_cc | 120/120 | 0.303125 | 27.4437 | 1.25 | 17.1875 |
| rtpipeline_curve_supplement | D2cc_Gy | 92/120 | 0.01 | 0.218 | 0.05 | 0.871198 |

## Contour rasterization volume error

These errors include finite contour spacing, polygon boundaries, rasterization on dose nodes, and each workflow's handling of contour planes. CT slice thickness changes the contour spacing; these are RTSTRUCT-based measurements, not CT-mask resampling. `geometry.csv` also reports the pre-rasterization polygon/slab volume, production dose-plane alignment and optional snapping-helper diagnostics. Columns prefixed `helper_snap_` describe the helper only; production retains the original contours.

| Tool | Dose grid (mm) | CT slice (mm) | Median absolute volume error (cm³) | Maximum absolute (cm³) | Median relative (%) | Maximum relative (%) |
|---|---:|---:|---:|---:|---:|---:|
| dicompyler_harmonized | 2.5 | 1 | 0.25 | 14.75 | 0.759063 | 14.6875 |
| dicompyler_harmonized | 2.5 | 3 | 0.3625 | 27.4437 | 1.3625 | 17.1875 |
| dicompyler_native | 2.5 | 1 | 0.25 | 14.75 | 0.759063 | 14.6875 |
| dicompyler_native | 2.5 | 3 | 0.3625 | 27.4437 | 1.3625 | 17.1875 |
| rtpipeline | 2.5 | 1 | 0.25 | 14.75 | 0.759063 | 14.6875 |
| rtpipeline | 2.5 | 3 | 0.3625 | 27.4437 | 1.3625 | 17.1875 |

## Largest absolute error per metric and workflow

| Tool | Metric | Case | Value | Analytic | Absolute error |
|---|---|---|---:|---:|---:|
| dicompyler_harmonized | D0.03cc_Gy | v1000_aligned_ct3_dg2.5_linear_shell | 29.305 | 29.5546 | 0.24964 |
| dicompyler_harmonized | D2_Gy | v1000_aligned_ct3_dg2.5_linear_cylinder | 25.405 | 25.2025 | 0.20251 |
| dicompyler_harmonized | D2cc_Gy | v1000_offset_ct3_dg2.5_linear_cube | 24.805 | 25.023 | 0.218 |
| dicompyler_harmonized | D50_Gy | v2_offset_ct3_dg2.5_linear_shell | 20.3054 | 20.043 | 0.262385 |
| dicompyler_harmonized | D95_Gy | v2_offset_ct3_dg2.5_linear_cube | 19.71 | 19.476 | 0.233964 |
| dicompyler_harmonized | D98_Gy | v2_offset_ct3_dg2.5_linear_cube | 19.71 | 19.4382 | 0.271762 |
| dicompyler_harmonized | Dmax_Gy | v1000_aligned_ct3_dg2.5_linear_shell | 29.31 | 29.5862 | 0.276219 |
| dicompyler_harmonized | Dmean_Gy | v2_offset_ct3_dg2.5_linear_cube | 20.155 | 20.043 | 0.112 |
| dicompyler_harmonized | Dmin_Gy | v50_offset_ct3_dg2.5_linear_cube | 18.51 | 18.201 | 0.309016 |
| dicompyler_harmonized | V20Gy_cc | v1000_aligned_ct3_dg2.5_linear_cylinder | 527.606 | 500 | 27.6063 |
| dicompyler_harmonized | V95Rx_pct | v1000_offset_ct3_dg2.5_linear_cylinder | 16.6667 | 15.7979 | 0.868749 |
| dicompyler_harmonized | Volume_cc | v1000_aligned_ct3_dg2.5_uniform_cylinder | 1027.44 | 1000 | 27.4437 |
| dicompyler_native | D0.03cc_Gy | v2_aligned_ct1_dg2.5_uniform_sphere | 0 | 20 | 20 |
| dicompyler_native | D2_Gy | v2_aligned_ct1_dg2.5_uniform_sphere | 0 | 20 | 20 |
| dicompyler_native | D2cc_Gy | v2_aligned_ct1_dg2.5_radial_sphere | 0 | 28.8276 | 28.8276 |
| dicompyler_native | D50_Gy | v2_aligned_ct1_dg2.5_uniform_sphere | 0 | 20 | 20 |
| dicompyler_native | D95_Gy | v2_aligned_ct3_dg2.5_radial_shell | 0 | 28.1965 | 28.1965 |
| dicompyler_native | D98_Gy | v2_aligned_ct1_dg2.5_radial_sphere | 0 | 28.8355 | 28.8355 |
| dicompyler_native | Dmax_Gy | v1000_aligned_ct3_dg2.5_linear_shell | 29.31 | 29.5862 | 0.276219 |
| dicompyler_native | Dmean_Gy | v2_offset_ct3_dg2.5_linear_cube | 20.155 | 20.043 | 0.112 |
| dicompyler_native | Dmin_Gy | v50_offset_ct3_dg2.5_linear_cube | 18.51 | 18.201 | 0.309016 |
| dicompyler_native | V20Gy_cc | v1000_aligned_ct3_dg2.5_linear_cylinder | 527.606 | 500 | 27.6063 |
| dicompyler_native | V95Rx_pct | v1000_offset_ct3_dg2.5_linear_cylinder | 16.6667 | 15.7979 | 0.868749 |
| dicompyler_native | Volume_cc | v1000_aligned_ct3_dg2.5_uniform_cylinder | 1027.44 | 1000 | 27.4437 |
| rtpipeline | D0.03cc_Gy | v1000_aligned_ct3_dg2.5_linear_shell | 29.305 | 29.5546 | 0.24964 |
| rtpipeline | D2_Gy | v1000_aligned_ct3_dg2.5_linear_cylinder | 25.405 | 25.2025 | 0.20251 |
| rtpipeline | D50_Gy | v2_offset_ct3_dg2.5_linear_shell | 20.3054 | 20.043 | 0.262385 |
| rtpipeline | D95_Gy | v2_offset_ct3_dg2.5_linear_cube | 19.71 | 19.476 | 0.233964 |
| rtpipeline | D98_Gy | v2_offset_ct3_dg2.5_linear_cube | 19.71 | 19.4382 | 0.271762 |
| rtpipeline | Dmax_Gy | v1000_aligned_ct3_dg2.5_linear_shell | 29.31 | 29.5862 | 0.276219 |
| rtpipeline | Dmean_Gy | v2_offset_ct3_dg2.5_linear_cube | 20.155 | 20.043 | 0.112 |
| rtpipeline | Dmin_Gy | v50_offset_ct3_dg2.5_linear_cube | 18.51 | 18.201 | 0.309016 |
| rtpipeline | V20Gy_cc | v1000_aligned_ct3_dg2.5_linear_cylinder | 527.606 | 500 | 27.6063 |
| rtpipeline | V95Rx_pct | v1000_offset_ct3_dg2.5_linear_cylinder | 16.6667 | 15.7979 | 0.868749 |
| rtpipeline | Volume_cc | v1000_aligned_ct3_dg2.5_uniform_cylinder | 1027.44 | 1000 | 27.4437 |
| rtpipeline_curve_supplement | D2cc_Gy | v1000_offset_ct3_dg2.5_linear_cube | 24.805 | 25.023 | 0.218 |

## Endpoint availability and interpretation

The production axial path does not publish D2cc. Its native D2cc rows are unavailable. `rtpipeline_curve_supplement` reports only D2cc, extracted from the production curve with the production small-volume helper; it is a benchmark supplement, not a pipeline-exported endpoint. D2cc is undefined for a measured ROI volume below 2 cm³. The isolated native API instead returns zero in that situation; these outputs have an explicit status in the long table. Continuous truth includes D2cc at exactly 2 cm³, where it equals Dmin.

Circular contours have 256 vertices (area deficit <0.0101% before slice discretization). Dose storage quantization is 0.0001 Gy; DVHs use the tools' default 0.01 Gy bins. Dmin and Dmax compare continuous extrema with histogram extrema and therefore include sampling and binning effects. The continuous formulas do not use the generated DICOM or either tool's voxels.

This benchmark covers synthetic axial geometry and smooth dose fields. It does not assess oblique or rotated dose grids, patient anatomy, registration, heterogeneity corrections, delivered dose, or clinical or manuscript readiness. No acceptance threshold was prespecified, and no pass/fail accuracy claim is made.

See `summary.csv` for errors by shape, volume, dose resolution and CT thickness; `worst_cases.csv` for five largest errors per metric; `paired_difference.csv` for production versus harmonized comparator; `run_receipt.json` for versions, source hashes, execution status and exact commands.

## Volume error components

The polygon/slab reference integrates the finite contours with the original CT slice spacing. Its difference from continuous volume measures contour discretization. Measured volume minus this reference measures the original-contour rasterization residual for dicompyler. For RTpipeline it uses the original contour planes with dose interpolated onto them. Decimal contour rounding is negligible at the shown precision. The signed components sum to total volume error. Residual relative errors use polygon/slab volume as the denominator; contour and total errors use continuous volume. Native and harmonized dicompyler metrics share the same volume, so the comparator volume is shown once.

| Workflow | Dose (mm) | CT (mm) | N | Contour error median/max (%) | Subsequent error median/max (%) | Total error median/max (%) |
|---|---:|---:|---:|---:|---:|---:|
| dicompyler_native | 2.5 | 1 | 60 | 0.0554602 / 4.81108 | 0.52124 / 10.3756 | 0.759062 / 14.6875 |
| dicompyler_native | 2.5 | 3 | 60 | 1.03978 / 19.0551 | 0.529889 / 10.3756 | 1.3625 / 17.1875 |
| rtpipeline | 2.5 | 1 | 60 | 0.0554602 / 4.81108 | 0.52124 / 10.3756 | 0.759062 / 14.6875 |
| rtpipeline | 2.5 | 3 | 60 | 1.03978 / 19.0551 | 0.529889 / 10.3756 | 1.3625 / 17.1875 |

Case-level components and geometry diagnostics are in `volume_error_components.csv`.

## Uniform-dose endpoint behavior

For the uniform 20 Gy field, dicompyler's native D95 API returned zero in 48/48 cases. RTpipeline D95 ranged from 20.01 to 20.01 Gy; matched comparator extraction ranged from 20.01 to 20.01 Gy. RTpipeline Dmean ranged from 20.005 to 20.005 Gy. The native constraint API selects the first nearest-volume bin on a flat cumulative DVH. The 0.01 Gy bins also impose a distinct endpoint convention on the interpolated outputs. These results distinguish endpoint extraction from the accuracy of the shared histogram engine.
