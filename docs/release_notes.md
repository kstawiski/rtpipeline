# Release notes

## 2.1.2

RTpipeline 2.1.2 fixes two correctness defects in direct PyRadiomics robustness
extraction.

- Scalar NumPy arrays returned by PyRadiomics are now retained as numeric
  feature values. Previously, the direct NumPy 1.x path silently omitted these
  values while accepting only Python and NumPy scalar classes.
- A `featureClass` subset in the radiomics parameter file is now honored
  exactly. Omitted feature classes are no longer implicitly enabled.
- Diagnostic keys and non-scalar arrays remain excluded from tidy robustness
  output.

The release includes focused regression tests for scalar-value coercion,
diagnostic exclusion, exact feature-class activation, and invalid feature-class
names. Existing v2.1.x configuration files remain compatible.

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
