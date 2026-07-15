# Release notes

## 2.1.1

RTpipeline 2.1.1 is a maintenance release focused on correctness and safe
recovery behavior.

- DICOM Segmentation Storage can be decoded with pydicom 3.x when the optional
  dependencies are installed with `pip install "rtpipeline[dcmseg]"`. The
  compatibility layer returns a real multilabel SimpleITK image and fails
  explicitly for unsupported overlapping or fractional representations.
- ICC confidence intervals work with both the current `CI95` and legacy `CI95%`
  Pingouin result columns.
- A transient no-CT selection no longer deletes populated per-course CT,
  NIfTI, or TotalSegmentator outputs. Administrators can deliberately restore
  the previous destructive cleanup behavior with
  `RTPIPELINE_ALLOW_DESTRUCTIVE_CT_CLEAR=1` when scrubbing a confirmed wrong-CT
  course.

The release includes regression tests for each behavior. Existing v2.1.x
configuration files remain compatible.
