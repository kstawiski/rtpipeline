# QC Cropping Audit (5 October 2025)

## 1. Objective
Quantify structure cropping warnings emitted by the Snakemake/rtpipeline quality-control stage for the latest verification run (executed 5 October 2025) and determine whether mitigation should focus on morphological processing of segmentations or on acquisition protocol changes.

## 2. Data Sources
- `_RESULTS/qc_reports.xlsx` and per-course `qc_reports/*.json` captured during the 5 October 2025 run.
- `_RESULTS/case_metadata.xlsx` for CT acquisition parameters (field-of-view, slice thickness).
- Custom scripts (see notebooks/logs for this audit) aggregating counts by segmentation source and ROI.

## 3. Findings
### 3.1 Cropping prevalence by course
| Course | Cropped ROIs | Auto-derived | Manual | CT FOV (mm) | Slice Thickness (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 416435/2018-09 | 35 | 33 | 2 | 500 | 5.0 |
| 428459/2019-11 | 31 | 27 | 4 | 500 | 3.0 |
| 480007/2024-08 | 20 | 16 | 4 | 500 | 3.0 |
| 480008/2024-09 | 25 | 21 | 4 | 500 | 3.0 |
| 487009/2025-03 | 28 | 23 | 5 | 600 | 1.5 |
| 487009/2025-09 | 39 | 30 | 9 | 600 | 1.5 |

Out of 178 total cropped structures, 150 (84%) stem from TotalSegmentator-derived ROIs, while 28 (16%) originate from manual contours (primarily couch components).

### 3.2 Frequently cropped ROIs
The most recurrently truncated structures are spinal cord and paraspinal musculature (autochthon left/right), followed by great vessels (aorta, inferior vena cava) and bilateral femora. Manual outliers are predominantly `CouchSurface`/`CouchInterior`, reflecting immobilisation hardware captured only partially in the pelvic CT stack.

### 3.3 Acquisition context
All scans are pelvis-focused with limited cranio-caudal coverage (500–600 mm FOV, slice thickness 1.5–5.0 mm). The QC flagging therefore reflects deliberate acquisition truncation rather than inconsistent patient setup. Extending the CT range to encompass thoracic and upper-abdominal anatomy would imply longer scans, increased dose, and additional contouring workload without clinical benefit for prostate-targeted therapy.

## 4. Interpretation
- Cropping warnings predominantly trace to auto-segmented structures whose anatomical extents exceed the imaged volume (e.g., spinal cord, inferior vena cava). The current QC heuristic flags any voxel touching the series boundary, regardless of whether the missing portion is clinically relevant.
- Manual cropping cases concentrate in support structures (couch) that are routinely disregarded during plan evaluation; their truncation does not compromise downstream DVH or radiomics analysis once excluded from model inputs.

## 5. Recommendation
Implement computational mitigation rather than modify acquisition protocols.

1. **Clip or erode auto-segmented masks prior to QC:** Apply a 2–3 voxel morphological erosion followed by bounding-box clipping to the imaged volume for TotalSegmentator ROIs before running `mask_is_cropped`. This preserves parenchymal regions used for DVH/radiomics while preventing boundary-touch artefacts. As of 5 Oct 2025, any cropped contour is automatically tagged with a `__partial` suffix in downstream tables, but geometric corrections remain preferable to post-hoc filtering.
2. **Whitelist non-clinical structures:** Filter couch-related manual ROIs (`CouchSurface`, `CouchInterior`, `Couch Exterior`) from cropping checks to avoid false alarms.
3. **Retain acquisition protocol:** Pelvic CT coverage is appropriate for the treated anatomy; extending FOV would add unnecessary dose and is unlikely to resolve the dominant auto-derived warnings.

These adjustments will allow downstream modelling to proceed using the existing imaging protocol while maintaining QC sensitivity to genuinely truncated targets or OARs.
