# Synthetic axial DVH accuracy

All 36 dg2.5 groups completed: 120 synthetic structure–dose cases. CT slice spacing was 1 or 3 mm. The dose fields were uniform, linear in z, and radial. Continuous analytic truth was the reference.

Dose-error limits were max(2% of analytic dose, 0.5 Gy). The V95Rx limit was 2 percentage points. The volume limit was 3% for structures of at least 1 cm³. V20Gy has no specified absolute-volume limit.

| CT spacing (mm) | Cases | Dose comparisons | Dose failures | Max dose error (Gy) | V95Rx failures | Max V95Rx error (pp) | Volume failures | Max volume error (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 525 | 0 | 0.12 | 0/60 | 0.378788 | 13/60 | 14.6875 |
| 3 | 60 | 527 | 0 | 0.309016 | 0/60 | 0.868749 | 16/60 | 17.1875 |

All 1052 available production RTpipeline dose comparisons and all 120 V95Rx comparisons met their limits. Volume exceeded its limit in 29 of 120 cases.

Native D2cc is available in 92/120 cases; 28 measured structures are below 2 cm³. Its maximum absolute error is 0.218 Gy, with 0 cases exceeding max(2% of analytic dose, 0.5 Gy). The full-curve D2cc check is reported separately in verification.json. The isolated dicompyler comparator shares the production rasterizer and is not an independent DVH engine. The helper_snap_ columns describe the optional snapping helper; dose_plane_alignment records production alignment.

These measurements cover synthetic axial geometry and smooth dose fields. They do not assess patient anatomy, brachytherapy source gradients, delivered dose, or clinical or manuscript readiness.
