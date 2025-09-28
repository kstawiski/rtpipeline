# Viewers

The Snakemake `visualization` rule generates HTML dashboards to support QA and
review. It first attempts the full DVH report and then falls back to simpler
artefacts if inputs are missing.

## DVH Report (`DVH_Report.html`)
- Plotly-based interactive chart with one trace per ROI.
- Legend toggles allow you to hide/show structures; hover text displays dose and
  relative volume for the selected point.
- Generated only when `dvh_metrics.xlsx` exists and contains data. If the DVH
  stage produced an empty workbook, the report gracefully omits missing curves.

## Axial Viewer (`Axial.html`)
- Scrollable axial CT viewer with opacity control for overlays.
- Separate checklists for manual (`RS.dcm`) and automatic (`RS_auto.dcm`) ROIs.
  Common targets/OARs (PTV, CTV, prostate, bladder, rectum) are pre-selected for
  convenience.
- Supports simultaneous display of multiple ROIs; overlays are colour-coded and
  rendered as semi-transparent RGBA masks.
- Slice slider and opacity slider update the view instantly without reloading.
- Requires `rt-utils` to rasterise structures onto the CT grid. If `RS.dcm` or
  `RS_auto.dcm` is missing, the corresponding panel remains empty.

## Fallbacks
If Plotly generation fails or DVH data are unavailable, the pipeline continues
with Axial-only output or, as a last resort, writes a minimal HTML stub noting
that no visualisation data could be produced. Errors are logged to the
patient-specific log file under `logs_dir`.
