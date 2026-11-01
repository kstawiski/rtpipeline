# Campaign accounting

With `campaign_mode: true`, each robustness course closes with a recorded
outcome. The workflow revalidates the CLI receipt before recording the outcome
in `_campaign_ledger/records`. Both local and container rules use this closure.
A missing or invalid receipt stops the job.

A robustness extraction failure records readable source bindings and digest-bound
source errors, including malformed contracts or RTSTRUCTs. The campaign closes
that course as failed and proceeds to later courses. Aggregation retains the
failed course in the denominator.

| Robustness outcome | Campaign status | Measurement contribution |
| --- | --- | --- |
| `measured`, without selection gaps | `ok` | Published measurement rows |
| `measured`, with `selection_gap_targets` | `technically_missing` | Published measurement rows |
| `source_only_nonvolumetric` | `ok` | Zero |
| `not_applicable_no_target_structure` | `not_applicable` | Zero |
| `selection_gap_targets_unselected` | `technically_missing` | Zero |
| `not_applicable_no_planning_ct` | `not_applicable` | Zero |
| `failed_extraction` | `failed` | Zero |

`radiomics_robustness.target_selection: patterns` is the default and preserves
case-insensitive whole-name fnmatch selection. `targets` adds every detected
target to the configured patterns. Unknown modes are rejected.

The governed ROI class map and feature policy decide first. Mapped targets
qualify. Mapped OARs, OAR subvolumes and planning helpers do not qualify, even
when their names or interpreted types contain target labels. For example,
`CTVN_Prostate`, `GTVm - lopatka`, `GTVm_SUMA` and `GTVn4p` are targets;
`Jelita w PTV`, `Jelita + PTV1`, `2cm od PTV1`, `z_PTV1` and
`PTV1-poprzednio` are not.

For unmapped names, the detector accepts a `GTV`, `CTV`, `PTV` or `ITV` token
at the start or after a space, hyphen, underscore, dot or slash. A single
`p`, `n` or `m` qualifier may precede the end, digits or a separator.
`GTVp`, `CTVn`, `PTVn1` and `HR-CTV` qualify; `CTVx` and `GTVnode` do not.
The interpreted types `GTV`, `CTV`, `PTV` and `ITV` also qualify unmapped names.
`TREATED_VOLUME` alone does not qualify.

A target must carry volumetric contour data or nonzero voxels. Template slots,
POINT markers and zero masks remain in the source inventory but are not missed
targets. Inventory-only helpers cannot enter the measurement base. In `targets`
mode, `apply_to_structures: []` requests detected targets alone.

Classification uses the source label. Canonical AutoRTS names use the
TotalSegmentator class map; Custom Boolean structures use recorded configuration
provenance. A same-named Custom ROI is copied support only when a standard source
declares that name with contours. A contourless template slot cannot hide a
contoured Custom target.

When selection matches nothing, the detector checks every evaluated source
structure set. No detected target gives `not_applicable_no_target_structure`.
A detected but unselected target gives `selection_gap_targets_unselected`.
Both close the stage without aborting the workflow. A selection gap remains
technically missing in the ledger, rollup, cohort admission and aggregation.
It never counts as complete or not applicable; measured rows remain available.

No-match receipts record the mode, requested patterns and missed target names
with their deciding rules. Default `patterns` mode with no exclusions preserves
historical configuration hashes and measured columns. Non-default selection
settings enter the effective configuration and measured parameter hashes, with
`extraction_configured_parameter_hash` retaining the extractor hash. The separate
target inventory records the evaluated source evidence on every CT path.

Rollup accepts robustness success only from a validated JSON receipt. Plain-text
`disabled` and `failed` remain terminal states; other text is missing evidence.
A newer validated receipt supplies the current status and detail. Metadata from
a superseded ledger record, including its log and duration, is cleared.
Rollup reports `technically_missing_unit_count` and the units with their reasons.

The aggregate workbook retains each admitted course in `course_outcomes`, with
`accounting_status`, `technically_missing`, `target_selection` and
`nonmeasurement_reason`. `course_counts` separates measured, technically missing,
not-applicable and source-only courses. No-match courses contribute no raw
measurement rows or ICC, CoV or QCD estimates. Evaluated source dispositions
remain in `source_dispositions`.

Receipts carrying `not_applicable_selection_unmatched` are stale and require
recomputation; startup moves only these receipts into
`_ROBUSTNESS_RECEIPT_QUARANTINE`, retaining their bytes and an identity record.
Migration is cached per resolved output directory and dependency content. Shared
configuration mtimes and valid course receipts remain unchanged.

Campaign aggregation applies `campaign_require_all_courses` and
`campaign_min_completion_fraction` (default `0.5`). Upstream-failed courses stay
in the manifest denominator. Tightening these gates reruns aggregation without
rerunning per-course extraction. The Python API applies the same gates as the CLI.

Changes to the mode, patterns,
source bytes, code or receipt bindings require a new run. Stage completion
alone does not establish clinical or manuscript readiness.
