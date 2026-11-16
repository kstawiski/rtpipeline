# Release notes

## Unreleased

No changes since 3.0.2.

## 3.0.2 — 2026-11-16

Cite this release by version and by the DOI archived for the `v3.0.2` tag (see
[citing](citing.md)). Pipeline behaviour is identical to 3.0.1, including the
placement of clinical contour masks by physical voxel coordinates (see 3.0.1).
Run `organize` again on projects processed with 3.0.0 or earlier.

- `uv.lock` records the package version, so `uv sync --locked` succeeds on a
  release checkout.
- The local-installer check runs on a macOS runner image that GitHub supports, so
  the release workflow publishes the container image
  `kstawiski/rtpipeline:3.0.2`. No container image was published for 3.0.1.
- The container label, the Apptainer definition, the setup scripts, the installer
  receipt and the documentation examples name 3.0.2.

## 3.0.1 — 2026-11-16

Run `organize` again on projects processed with 3.0.0 or earlier: clinical
contour masks exported by those versions are refused and re-exported
automatically.

### Clinical contour masks are placed by physical voxel coordinates

- In 3.0.0 and earlier, the NIfTI masks exported from a course's clinical
  RTSTRUCT into `Segmentation_Original/` could be mirrored front to back, along
  the image row axis. The exporter wrote the RTSTRUCT mask in DICOM row order
  under the header of the dcm2niix planning-CT NIfTI, whose rows dcm2niix stores
  in the opposite order. The header described the correct grid while the voxels
  were mirrored, so a mirrored mask lies partly outside the anatomy it outlines.
- 3.0.1 maps every mask onto the planning-CT NIfTI by physical voxel
  coordinates. Only an exact permutation or flip of axes is accepted, and a
  grid that is not equivalent is refused rather than resampled. Each export
  records the schema `rtpipeline-original-segmentation-v2` and the geometry
  status `exact_physical_mapping`.
- Every consumer refuses an export without that status, and `organize`
  re-exports it, so masks written by earlier versions cannot be used.
- Only the exported `Segmentation_Original` masks, and the custom-model anchor
  and Dice-reference masks taken from them, were affected. DVH, radiomics and
  robustness built their masks from the RTSTRUCT against the CT read from the
  DICOM series and were not affected. Repeat any analysis outside RTpipeline
  that read `Segmentation_Original` masks produced by 3.0.0 or earlier.

### Other changes

- RTSTRUCT rasterization uses one pinned OpenCV provider (4.8.1), and outputs
  record the rasterizer identity they were built with.
- A durable progress journal and verified result cache let DVH, radiomics and
  robustness resume after an interruption without repeating completed work or
  reusing a result whose inputs changed.
- D2cc is exported on every DVH path, sub-centigray doses are preserved, and
  DVH contour dispositions and outside-grid structures are published
  explicitly. The published DVH table schema is pinned across pandas versions.
- Treatment-scope rules 1.1.0 are available, with governed robustness target
  selection, structure exclusions and gap accounting.
- A multi-plan composite course dose can be published for descriptive DVH.
- Planning-CT conversion, conversion provenance and cache reuse are validated
  more strictly, and radiomics rows are published in ROI enumeration order on
  every backend.
- Tests use synthetic course keys only, and a guard refuses real ones.

## 3.0.0 — 2026-10-25

Cite this release by version and by the DOI archived for the `v3.0.0` tag (see
[citing](citing.md)). Projects upgrading from 2.x re-run the stages whose
completion evidence was produced by earlier code; outputs without
configuration-bound completion receipts are invalidated automatically.

### RTsegmentator as a segmentation source

- The `segmentation_custom` stage can send each course's contracted planning CT to
  an [RTsegmentator](features/rtsegmentator.md) portal, configured under the new
  `rtsegmentator` section, and import the returned RTSTRUCTs into
  `Segmentation_CustomModels/rtseg_<model>/`. DVH, radiomics and robustness use
  them through the custom-model contract with the segmentation source
  `CustomModel:rtseg_<model>`.
- `rtpipeline rtsegmentator-import` imports result packages downloaded from a
  portal into an organized course and removes that course's `segmentation_custom`
  receipt, so the next workflow run repeats the stage and re-runs DVH, radiomics and
  robustness on the imported structures.
- Imports this portal produced for models no longer configured are removed; offline
  imports and imports from another portal are kept. DVH output is regenerated when a
  custom-model source it was built from no longer exists.
- A package is imported only when every file matches its manifest checksum, the
  manifest and each RTSTRUCT reference the course's planning CT series, and the ROI
  names agree with the manifest. The custom-model `manifest.json` records the
  portal URL, job ID, RTsegmentator version, model licence and limitations, and
  `metadata/rtsegmentator_imports.json` records every model's outcome, including
  models that found no structure.
- The RTsegmentator client (`rtpipeline/segmenters/rtsegmentator_client.py`) is
  vendored unchanged from RTsegmentator 1.0.0 and uses only the standard library. It
  streams results to disk, compares the packaged manifest with the portal's copy,
  never replays a job submission after an ambiguous failure and refuses redirects to
  another origin, so proxy credentials stay with the portal.

### Course organization and dose classification

- The organized-course manifest is parsed by `rtpipeline.course_manifest`, and it
  is the authoritative list of validated courses. Consumers no longer rediscover
  courses by scanning the output tree, so a directory no manifest entry names is
  not treated as a validated course.
- Delivered-plan selection requires complete record coverage and strict plan
  equivalence from the exported source files before it deduplicates revisions.
  Prescription, fraction count, and shared treatment records are not sufficient
  on their own.
- When independent delivered plans cannot be reconciled into one course, the
  course keeps its plan membership, and course dose and fraction totals are
  withheld as null rather than summed. `course_fraction_totals_basis` and
  `course_fraction_totals_reason` state which case applies. Per-plan counts stay
  in `delivery_plan_details`. See
  [dose classification](features/dose_classification.md).
- Delivered dose stays null when treatment records cannot support an estimate.
  Unknown delivery is never written as 0 Gy or replaced by the prescription.
- Source-plan dispositions distinguish clinical exclusion from technical hold, so
  a plan that failed materialization or validation is not reported as an included
  course member.

### Radiomics robustness

- A per-course robustness step writes a completion receipt bound to the course,
  the run, the outcome, and the artifact bytes it certifies, replacing a sentinel
  file whose contents were the literal text `ok`. Cohort aggregation revalidates
  every manifest course's receipt instead of scanning for sentinel files, and a
  run that reports success without a receipt fails the rule.
- Robustness failures propagate. The workflow no longer converts a failed
  robustness stage into a successful one.
- The standard perturbation grid is 81 states: three noise levels, identity and
  translated geometry, two contour realizations, and three volume levels. Both
  the resegmented primary arm and the raw sensitivity arm are reported, and shape
  features are computed on full morphological masks.

### MR and multimodality

- MR series and MR course acceptance, rejection, and typed helper failures are
  accounted for explicitly, so a rejected MR series is recorded as rejected
  rather than silently absent from the ledger.

### Custom structures

- A manually authored custom structure keeps authority over an auto-harvested one
  of the same name, and generator epochs are recorded so a stale derived
  structure is regenerated rather than reused.

### Packaging and testing

- The package requires Python 3.10 or newer. The locked `test` dependency group
  runs the suite on the minimum supported Python, so the version floor is
  exercised rather than only declared.
- `xlsxwriter` is a declared dependency. Every conda environment already carried
  it, and pandas prefers it for `.xlsx` output, so pip and conda installations now
  write the same workbook bytes.
- Tests that need the separate PyRadiomics environment or the Snakemake executable
  carry `requires_radiomics_env` and `requires_snakemake` markers. They run in the
  container job and report an explicit skip reason in an environment without those
  runtimes. Tests that compare against earlier revisions read full git history.
- Citation metadata for archiving: `CITATION.cff`, `codemeta.json` and
  `.zenodo.json`.

### Distribution scope

Shared configs and versioned artifacts make consistent local processing
operationally possible. Hash-bound cohort-level packets support
distributed aggregate radiomics reliability analysis. Current packets do not themselves
establish federated model training, secure aggregation, differential privacy,
privacy guarantees, or outcome federation. Federated learning requires a
separately implemented downstream system.
