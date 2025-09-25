# DVH Metrics

## Source
- dicompyler-core is used to compute absolute DVHs for each ROI in `RS.dcm` (manual) and `RS_auto.dcm` (auto).
- Dose from organized `RD.dcm`; plan context from `RP.dcm`.

## Metrics (per ROI)
- Absolute (Gy): DmeanGy, DmaxGy, DminGy, D95Gy, D98Gy, D2Gy, D50Gy, SpreadGy (max-min)
- Relative (% of Rx): Dmean%, Dmax%, Dmin%, D95%, D98%, D2%, D50%, HI%
- Homogeneity Index: HI = (D2Gy - D98Gy)/D50Gy
- Volume: `Volume (cm³)`
- Integral dose: `IntegralDose_Gycm3 = DmeanGy × Volume (cm³)`
- Small hottest volumes: D1ccGy, D0.1ccGy (if structure is large enough)
- Coverage at Rx: V95%Rx (cm³ and %), V100%Rx (cm³ and %)
- VxGy (cm³ and %) for x = 1..60 Gy

## Rx estimation
- For target-relative percentages, Rx is estimated from manual CTV1 D95 when present; fallback is 50 Gy.
- You can adjust this logic in code if you have a site‑specific policy.

## Merged output
- `outdir/DVH_metrics_all.xlsx`: concatenation of all per‑course `dvh_metrics.xlsx` with identifiers.

