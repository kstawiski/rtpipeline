# Segmentation

## Modes
- DICOM: `TotalSegmentator_DICOM/segmentations.dcm` (DICOM‑SEG when supported, sometimes RTSTRUCT depending on version).
- NIfTI: `TotalSegmentator_NIFTI/` with label maps.

## Resume and force
- Default: If DICOM or NIfTI outputs exist, segmentation is skipped.
- Force re-run: `--force-segmentation` reruns TotalSegmentator regardless of existing outputs.

## Auto RTSTRUCT (RS_auto.dcm)
- Built per course after segmentation.
- If DICOM‑SEG present → convert to RTSTRUCT via rt-utils.
- If TotalSegmentator wrote RTSTRUCT → normalize/copy to `RS_auto.dcm`.
- Else if NIfTI present → resample to CT geometry and convert via rt-utils.

## Why RS_auto?
DVH requires RTSTRUCT for ROI definitions. RS_auto mirrors TotalSegmentator segments as RTSTRUCT aligned to `CT_DICOM`, enabling DVH for auto segmentation even if no manual RS is available.

