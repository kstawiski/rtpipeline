# Structure boundary quality control

RTpipeline checks whether a segmentation mask reaches an image boundary. A
boundary touch can indicate that part of a structure lies outside the acquired
CT volume, but it is a screening signal rather than proof that a clinically
important contour is incomplete.

## What the check records

For each non-empty mask, the quality-control stage records:

- the image sides touched by the mask;
- whether any touch is in-plane rather than limited to the axial scan extent;
- whether systematic CT-cropping metadata is available; and
- the configured crop region and clamped axes, when present.

Axial-only touches are reported but do not by themselves make the overall QC
status fail. In-plane boundary touches are treated as warnings unless the
structure is explicitly excluded from status evaluation. This distinction
helps separate limited scan length or planned axial cropping from possible
left, right, anterior, or posterior truncation.

## Interpreting a warning

Review the source CT and contour before excluding a structure. Common benign
causes include support devices, the external body contour, or anatomy that
normally extends beyond the planned scan range. Target volumes and organs at
risk require closer review because incomplete coverage can invalidate volume,
DVH, and radiomic measurements.

Downstream radiomics tables retain a `structure_cropped` flag. Cropped names may
also carry the `__partial` suffix so that incomplete structures remain visible
instead of being silently treated as complete.

## Configuration

The cropping check can be skipped for faster QC when boundary assessment has
already been performed elsewhere. When skipped, the report records a `SKIP`
status and reason. For research workflows, keep the check enabled and define
the handling of boundary-touching structures before statistical analysis.

Systematic anatomical cropping is configured separately. See
[CT Cropping](ct_cropping.md) for crop-region and landmark settings.
