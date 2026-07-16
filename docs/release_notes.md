# Release notes

## 2.2.0

RTpipeline 2.2.0 adds a fail-closed contract for distributed aggregate
radiomics reliability analysis. It is designed for identical local execution
across cohorts and deterministic central reconstruction without transferring
raw images or patient-level feature rows.

- Sites can export deterministic cohort-level reliability packets and
  coordinators can validate and combine them with `rtpipeline federation`.
- The contract binds schema, thresholds, semantic rules, permitted files, and
  content-audit behavior; packet identity is normalized and collisions are
  rejected.
- Extra files, symlinks, forged metadata, unexpected identifiers, nonfinite or
  inconsistent metrics, and node-declared threshold downgrades fail closed.
- Documentation now distinguishes distributed measurement from federated model
  training, secure aggregation, differential privacy, legal anonymity, and
  institution-specific governance decisions.

Existing analysis APIs and configuration files require no migration from 2.1.4.

## 2.1.4

RTpipeline 2.1.4 fixes container installation after the three runtime
environments have been built.

- Editable installation now reuses the dependencies already solved in the
  container environments instead of resolving and downloading a second
  Torch/CUDA stack.
- Direct Python runtime dependencies are declared in the primary conda
  environment so the no-dependency package install remains complete.
- Runtime pip caching is disabled to keep peak build-disk use bounded.
Existing analysis APIs and configuration files require no migration from 2.1.3.

## 2.1.3

RTpipeline 2.1.3 is the container-delivery follow-up to 2.1.2. It preserves the
radiomics correctness fixes from 2.1.2 and makes clean Docker builds reliable
on storage-constrained CI runners.

- Conda and pip caches are cleared between creation of the three isolated
  runtime environments, reducing peak build-disk use without changing their
  dependency contracts.
- GitHub Actions now frees unused preinstalled SDKs, loads the test image
  directly, and uses the native BuildKit GitHub cache instead of keeping a
  second full local image archive.
- Package, container, documentation, and citation metadata now report the same
  release version.

No analysis API or configuration migration is required from 2.1.2.

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
