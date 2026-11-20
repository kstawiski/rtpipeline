# Release notes

## Unreleased

- DVH reuse now checks the bytes of the authoritative RTPLAN, RTDOSE and
  RTSTRUCT, including course-root copies and derived grids, and binds the
  treatment records used to establish course membership.
- D2cc uses the minimum dose to the hottest qualifying 2 cm³. At an exact
  cumulative-volume plateau it selects the hotter end. Other Dx endpoints
  use the same rule.
- Axial RTSTRUCT DVHs weight each contour plane by its own slab thickness.
  Slab boundaries lie midway between consecutive contours. The first and
  last slabs extend by half their adjacent interval.
- Each ROI must explicitly reference the dose frame of reference. A missing
  or different ROI frame produces a row with no dose measurement.
- Cohort DVH aggregation requires a shared RTSTRUCT SOP identity and ROI
  geometry binding before collapsing duplicate measurements.
- NIfTI DVHs interpolate at the frame positions declared in RTDOSE, including
  nonuniform frame spacing.
- Covered contours with no DVH volume get explicit rows. The row distinguishes
  a structure with fewer than two contour planes from a structure that misses
  every sampling point. ROI number 0 is retained.
- Vx measures the sampled volume receiving at least the threshold. It is zero
  when every sampled dose is below the threshold.
- Dose selection refuses equally ranked grids with `ambiguous_dose_grid`
  when the course evidence does not identify one.
- BEAM dose accumulation requires complete, disjoint fraction-group and beam
  coverage. Duplicate beam exports are refused.
- A composite dose requires support from all component grids at the
  interpolation nodes. A gap between component grids has no dose measurement.
- The authoritative RTDOSE must declare physical dose in Gy when read.
- Original-mask re-export into a contracted course requires that course's
  authoritative RTSTRUCT and planning CT. Consumers check the same identity.
- A DVH table containing only refused rows has degraded QC and a
  `not_measured` receipt. Its ROI rows remain available with their dispositions.
- Prescription-relative DVH metrics use the nominal total reported in the
  row. An unresolved target or dose scope leaves these metrics unavailable.
- Manual DVH rows take precedence over matching merged and custom rows
  for the same patient and course.
- NIfTI DVHs interpolate stored integer dose samples in floating point before
  applying dose scaling.
- When a histogram collapses into the first centigray bin, unbinned
  samples determine all reported dose endpoints and homogeneity. Otherwise
  the RTSTRUCT histogram uses 0.01 Gy bins with mean doses at bin centres,
  minimum and maximum at populated upper bin edges, and interpolated
  Dx values between bin centres. When only a histogram is available, Vx includes
  bins whose lower edge meets the threshold.
- A partially covered RTSTRUCT ROI reports its contour-derived volume
  between the first and last contour planes. Dose metrics for its covered
  samples and their volume appear separately.
- Name-based ROI matching uses the original-mask collision rules for case,
  whitespace and sanitized names. Custom structures with ambiguous
  dependencies record a refusal. Independent custom structures can still
  be produced.

- Robustness keeps conditions rejected by worker identity checks in the requested
  grid. It writes technical-failure evidence and withholds measured completion.
- Robustness records the requested and realized translations in millimetres.
  Shifts that round to zero or exceed the tolerance receive a geometric disposition.
- Clipped contour offsets and unavailable contour realizations receive geometric
  dispositions. The grid supports zero translation and keeps distinct identifiers
  for fractional perturbations.
- Robustness binds the planning CT bytes and upstream CT radiomics publication
  to the run. Resume and aggregation require those bindings to match the current inputs.
- CT radiomics refuses HU-dependent features and HU noise when the DICOM rescale
  is non-HU or cannot be verified. Eligible shape features remain available.
- CT image caching uses the requested series UID and the ordered series content.
  A missing requested UID causes the read to fail.
- Optional IBSI preprocessing accepts preserved spacing axes and counts the prepared
  mask at its binary label. NumPy 2 entry points report an explicit compatibility error.
- The exploratory package summary marks groups with geometric nonmeasurements
  and features with undefined ICC as `not_evaluable`. Study analysis O7 follows
  its own specification.
- `--stage radiomics` uses the settings from the `radiomics` configuration
  block. The conda backend uses the packaged CT parameters when no file is configured.
- The Snakemake robustness rule loads libraries from the selected radiomics
  environment and requires an interpreter there. ICC computation errors propagate
  to the caller.
- RTpipeline admits native ROIs at `radiomics.min_voxels` or above. The default
  is 120 when the setting is omitted, and the shipped `config.yaml` sets 64.
  PyRadiomics separately requires more than `minimumROISize: 64` voxels on the
  resampled mask. Sequential robustness resume binds the extraction parameters,
  ROI map and effective perturbation settings.
- Set `radiomics_robustness.segmentation_perturbation.course_specific_seeds` to `true` to
  derive noise and contour seeds per course. The default preserves the shared seed sequence.

## 3.0.4 — 2026-11-17

Cite this release by version and by the DOI archived for the `v3.0.4` tag (see
[citing](citing.md)). This release fixes the duplicate-ROI-name issue listed
under 3.0.3: contours are selected by ROI number wherever ROI identity matters.
Structure sets in which every ROI name is distinct give the same exported masks
and the same DVH, radiomics and robustness values as in 3.0.3.

- When two ROIs in one structure set share a name, `Segmentation_Original`
  exports each from its own contours, and each DVH row takes its cropping flag
  from its own ROI. 3.0.3 and earlier exported the first ROI's mask for both.
  Names that differ only in case or surrounding whitespace, or that become the
  same file name once sanitized, also count as colliding.
- Collision-free `Segmentation_Original` exports keep their released file names
  and stay current after upgrade. Only ROIs in a colliding group get file names
  of the form `<stem>__roi-<number>.nii.gz`, and the manifest records their ROI
  numbers. Custom-model anchors, Dice references, QC and `re-export-original`
  find masks through the manifest.
- Radiomics and robustness are keyed by ROI name, so they refuse a course in
  which any source holds an ambiguous ROI name and write no measurement rows for
  that course, including its unambiguous ROIs. Serial CT radiomics in 3.0.3
  measured only the first ROI of a duplicated name. The CT ROI ledger and the
  robustness failure disposition record the refusal as
  `RTSTRUCT_AMBIGUOUS_ROI_NAME`, with the colliding ROI numbers and source, and
  count it as an identity refusal, not a technical failure. Once the colliding
  ROIs are renamed in the source structure set, the next run measures the course.
- A custom structure is no longer built from the first of several source ROIs
  whose names match once case and punctuation are ignored. The structure is
  withheld and the refusal is recorded as `CUSTOM_SOURCE_AMBIGUOUS_NAME`. An
  exact name match still takes precedence.
- The new command `rtpipeline re-export-original` rebuilds
  `Segmentation_Original` from explicit RTSTRUCT, CT series and reference NIfTI
  paths, for legacy trees that source discovery cannot rebuild (see
  [original-structure re-export](original_segmentation_reexport.md)).
- After upgrade, the workflow recomputes QC, RS_custom, radiomics and robustness
  outputs made by earlier 3.0 releases, because the code that produces them has
  changed. Collision-free original exports are reused.

## 3.0.3 — 2026-11-16

Cite this release by version and by the DOI archived for the `v3.0.3` tag (see
[citing](citing.md)). DVH, radiomics and robustness are computed by the same code
as in 3.0.1 and 3.0.2. This is the first 3.0 release with a published container
image, `kstawiski/rtpipeline:3.0.3`: the tests that run inside the image failed
for 3.0.1 and 3.0.2, so neither published one. Run `organize` again on projects
processed with 3.0.0 or earlier (see 3.0.1).

- The main environment pins rt-utils 1.2.7, the RTSTRUCT rasterizer used for
  every validated measurement. Without the pin, the container build, and any
  environment created from `envs/rtpipeline.yaml` or installed with pip, could
  resolve rt-utils 2.5.0, which rasterizes contours differently: a contour
  written and read back gained boundary voxels. The image build checks the
  installed version, and installs from `pyproject.toml` pin it as well. In an
  existing environment, `python -c "from importlib.metadata import version;
  print(version('rt-utils'))"` should print 1.2.7; if it does not, reinstall and
  rerun the affected courses.
- The workflow validates organize receipts with the main-environment interpreter,
  through the existing workflow delegate, instead of inside the Snakemake
  interpreter, so a Snakemake installation without the scientific packages can
  resume a project. Course expansion also waits when a course that the organize
  ledger lists as validated no longer passes its contract, progress or receipt
  check; organize then runs again for that course.
- An invalid `declared_on` date in a concurrent-dose declaration reports
  `declared_on must be YYYY-MM-DD` on every supported Python. Python 3.10 used to
  stop at its own date error first. Dose selection is unchanged.
- Release testing runs the full suite inside the image, with the repository
  history mounted read-only and the Snakemake and radiomics runtimes declared. The
  Python 3.10 suite runs on the locked dependencies; tests that need conda,
  Snakemake or the radiomics environment report an explicit skip there.

Known issue, fixed in 3.0.4: when an RTSTRUCT holds two ROIs with the
same name, the `Segmentation_Original` export writes the first ROI's mask for
both, serial CT radiomics measures only the first, and a custom structure whose
source name matches several ROIs once case and punctuation are ignored uses the
first of them. DVH dose values are computed per ROI number and are not affected.
Structure sets in which every ROI name is distinct are not affected.

## 3.0.2 — 2026-11-16

Cite this release by version and by the DOI archived for the `v3.0.2` tag (see
[citing](citing.md)). Pipeline behaviour is identical to 3.0.1, including the
placement of clinical contour masks by physical voxel coordinates (see 3.0.1).
Run `organize` again on projects processed with 3.0.0 or earlier.

- `uv.lock` records the package version, so `uv sync --locked` succeeds on a
  release checkout.
- The local-installer check runs on a macOS runner image that GitHub supports.
  The container image still did not publish, because its in-image tests failed
  (fixed in 3.0.3). No container image was published for 3.0.1 either.
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
  grid that is not equivalent is refused rather than resampled. Each
  successful export records the schema `rtpipeline-original-segmentation-v2`
  and the geometry status `exact_physical_mapping`. A refused grid leaves a
  record with the status `MASK_GRID_NOT_EQUIVALENT` and no masks.
- Every consumer refuses a mask without the `exact_physical_mapping` status,
  and `organize` re-exports legacy exports, so masks written by earlier
  versions cannot be used.
- The clinical DVH, radiomics and robustness paths read contours from the
  RTSTRUCT itself and never read the exported masks, so their results for
  clinical structures were not affected. Custom models were affected where
  they used an exported clinical mask as the anchor that selects which
  predicted component is kept, as the box prompt for MedSAM, or as a Dice
  reference. Their structures, and the DVH, radiomics and robustness results
  computed from them, can differ. After re-exporting the clinical masks,
  regenerate those custom-model outputs and the analyses that depend on them,
  and repeat any analysis outside RTpipeline that read `Segmentation_Original`
  masks produced by 3.0.0 or earlier.

### Other changes

- RTSTRUCT rasterization for radiomics uses one pinned OpenCV provider
  (headless 4.8.1) in the radiomics environment; the main environment keeps its
  own OpenCV pin. Outputs record the rasterizer identity they were built with.
- A durable progress journal and verified result cache let DVH, radiomics and
  robustness resume after an interruption without repeating completed work or
  reusing a result whose inputs changed.
- D2cc is exported on every DVH path. A collapsed DVH keeps its unbinned mean
  and maximum dose and integral dose; dose quantiles keep histogram resolution.
  DVH contour dispositions and outside-grid structures are published
  explicitly. The published DVH table schema is pinned across pandas versions.
- Equal prescriptions without treatment records no longer establish that plans
  are revisions of one another. Such plans keep their plan membership without
  a course dose grid unless revision evidence or strict plan equivalence links
  them. An explicit concurrent-component declaration can authorize a physical
  sum of dose grids, while course fraction totals and summed reference-point
  doses stay withheld.
- Treatment-scope rules 1.1.0 are available, with governed robustness target
  selection, structure exclusions and gap accounting.
- A multi-plan composite course dose can be published for descriptive DVH.
- Planning-CT conversion, conversion provenance and cache reuse are validated
  more strictly, and the parallel CT radiomics backends publish rows in ROI
  enumeration order, including rows reused from a checkpoint.
- Tests use synthetic course keys, and a CI check flags likely patient course
  keys in tracked text.

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
