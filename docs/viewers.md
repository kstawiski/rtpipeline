# Viewers

## DVH Report (DVH_Report.html)
- Interactive Plotly chart with one line per ROI (Manual and Auto if both present).
- Toggle structures via the legend. Hover shows Dose and Volume%.

## Axial Viewer (Axial.html)
- Scrollable axial CT viewer with opacity control for overlays.
- Two ROI panels:
  - Manual (RS): structures from `RS.dcm`.
  - Auto (RS_auto): structures from `RS_auto.dcm`.
- Check any combination of structures to overlay; overlays are semi‑transparent and color‑coded.
- Preselects common targets/OARs (PTV/CTV/prostate/rectum/bladder) for quick QA.

Notes
- Masks are rasterized via rt-utils from the RTSTRUCT onto the `CT_DICOM` geometry.
- If structures don’t appear: ensure `RS.dcm` or `RS_auto.dcm` exists and rt-utils is installed in your environment.
- MR: There is no MR viewer by default. If you need one, consider exporting NIfTI overlays or open an issue to request an MR viewer similar to Axial.html.
