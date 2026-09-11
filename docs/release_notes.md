# Release notes

## Unreleased

Version 2.3.99 is an unreleased candidate. There is no published tag or archived
DOI, so cite an immutable git commit or container digest.

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

### Distribution scope

Shared configs and versioned artifacts make consistent local processing
operationally possible. Hash-bound cohort-level packets support
distributed aggregate radiomics reliability analysis. Current packets do not themselves
establish federated model training, secure aggregation, differential privacy,
privacy guarantees, or outcome federation. Federated learning requires a
separately implemented downstream system.
