# Release notes

## 2.2.3

RTpipeline 2.2.3 is a maintenance and release-integrity update. It preserves
the complete NTCV scientific contract introduced in 2.2.2.

- The top-level Snakemake workflow now delegates organization, course-stage,
  and aggregation bodies to focused scripts under `workflow/scripts/`, making
  the DAG easier to lint and maintain without changing its stage contract.
- Ruff is configured for Snakemake's runtime-injected `snakemake` global while
  retaining the existing undefined-name checks elsewhere.
- The public-repository boundary check fails when manuscript, submission,
  source-data, or private workflow artifacts are present in the checkout,
  including when `.gitignore` would conceal them.
- Release CI now runs the complete pytest suite inside the production
  `rtpipeline` environment before the container smoke test and Snakemake
  dry-run.
- Container entrypoints now default to the complete `rtpipeline` environment,
  container-mode routing is explicit, and the required PyArrow Parquet engine
  is a declared dependency rather than an incidental transitive install.
- DVH imports now preserve dicompyler-core 0.5.6 compatibility with pydicom
  3.0.2, including the current pixel utility namespace.
- Robustness ICC extraction accepts both the historical Pingouin ICC labels and
  the equivalent labels introduced in Pingouin 0.6.
- Packaged radiomics parameter defaults no longer create an untracked copy in
  the caller's current directory when no log directory is configured.
- `rtpipeline.def` is now a thin, version-pinned Apptainer conversion of the
  canonical Docker image, preventing the Docker and Apptainer recipes from
  drifting into different software stacks.
- The reproducibility lock was refreshed to compatible dependency releases
  without known vulnerabilities in the current advisory audit.
- The source distribution now includes the advertised MIT license and the
  source/configuration files required by its shipped tests, while package,
  container, citation, installer, and documentation versions are synchronized.
- The pinned dcm2niix release archive is now verified by SHA-256 before
  extraction, and the locked test dependency group runs the suite on the
  minimum supported Python 3.10 interpreter.
- Reproducibility examples use the Docker tags actually emitted by release CI
  (for example, `kstawiski/rtpipeline:2.2.3`).

## 2.2.2

RTpipeline 2.2.2 corrects the scientific implementation contract for NTCV
robustness and distributed reliability packets.

- NTCV now constructs a strict Cartesian product. Each seeded contour
  realization is determined only by translation and contour level and is
  reused across noise and volume levels; noise draws are likewise reused
  across all geometric states.
- Physical translations now follow the documented object-displacement
  direction and fail closed on boundary clipping, topology changes, or a
  displacement that is not realized within grid resolution.
- CoV and QCD are evaluated only for features prespecified as suitable for
  relative dispersion and for finite, strictly positive observations. Signed
  or zero-crossing features retain their ICC but are reported as
  `not_evaluable` instead of being classified from silently reduced subject
  denominators.
- Federation packet schema 2 carries explicit CoV and QCD subject denominators,
  supports validated missing relative-dispersion values, and prevents an
  incomplete CoV denominator from receiving a robustness classification. Its
  contract digest also binds the ICC lower-confidence-bound and CoV thresholds,
  and every packet label is recomputed and validated at export and receipt.
- Regression coverage includes an unmocked 81-state run, factor independence,
  physical translation and clipping, signed metrics, and schema-2 packet
  interoperability.

## 2.2.1

RTpipeline 2.2.1 is a release-hardening update for source, wheel, container,
and radiomics-robustness execution.

- Built wheels now include every `rtpipeline` subpackage, including the
  `segmenters` and `eval` runtime modules.
- Related-series organization now records metadata and performs cleanup through
  the same validated helper used by primary series.
- Downloaded custom-model archives are validated against path traversal and
  link-based extraction before any member is written.
- Subprocess launchers use argument arrays or an explicit shell boundary,
  eliminating ambiguous nested shell execution while retaining documented
  environment-activation support.
- The shipped robustness profile now runs the complete standard NTCV chain:
  Gaussian noise at 0, 10, and 20 HU; translations up to +/-4 mm; two contour
  realizations; and -15%, 0%, and +15% volume adaptation. All 81 combinations
  per ROI are required; generation or extraction gaps fail the course.
- CoV is computed within each patient/course/ROI/source and summarized across
  subjects, so between-patient biology is not mislabeled as perturbation
  instability. Structure and segmentation sources are not pooled for feature
  classification. Aggregation also requires the same feature inventory for
  every contributing subject within an ROI/source comparison.
- Contour and volume perturbations use physical-space signed distance maps;
  volume targets reach the exact rounded voxel count and duplicate contour
  realizations fail closed.
- Runtime dependency floors include patched `pydicom` and `filelock` releases;
  release CI verifies static analysis and wheel contents before container build.
- A supported automatic local installer, runner, and self-contained tutorial
  cover macOS, Linux, and WSL2 while preserving the required dual environments
  and manuscript provenance. The NumPy 1.26 helper imports the checked-out
  source without installing incompatible NumPy 2.x main-runtime metadata.
- Incompatible NumPy 1.x tooling (`PyRadiomics` and `pydicom-seg`) is not
  exposed as a merged package extra. PyRadiomics uses the supported isolated
  environment; optional DICOM-SEG conversion requires its own pinned NumPy 1.x
  environment.

Existing v2.2.0 configurations remain loadable. Runs that relied on the old
volume-only implicit robustness defaults should set the three NTC axes to zero
explicitly if that historical behavior is required.

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

- In the v2.1.1 source checkout, DICOM Segmentation Storage could be decoded
  with pydicom 3.x when the then-available optional `dcmseg` extra was used. The
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
