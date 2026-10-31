# Campaign accounting

With `campaign_mode: true`, each robustness course closes with a recorded
outcome. The workflow revalidates the CLI receipt before recording the outcome
in `_campaign_ledger/records`. Both local and container rules use this closure.
A missing or invalid receipt stops the job.

| Robustness outcome | Campaign status | Measurement contribution |
| --- | --- | --- |
| `measured` | `ok` | Published measurement rows |
| `source_only_nonvolumetric` | `ok` | Zero |
| `not_applicable_no_target_structure` | `not_applicable` | Zero |
| `selection_gap_targets_unselected` | `technically_missing` | Zero |
| `not_applicable_no_planning_ct` | `not_applicable` | Zero |
| `failed_extraction` | `failed` | Zero |

`radiomics_robustness.target_selection: patterns` is the default and preserves
case-insensitive whole-name fnmatch selection. `targets` adds every detected
target to the configured patterns. Unknown modes are rejected.

The governed detector accepts a `GTV`, `CTV`, `PTV` or `ITV` name token at the
start or after a space, hyphen, underscore, dot or slash, ignoring case. The
token cannot be followed by a letter. `HR-CTV`, `IR-CTV`, `LR-CTV`, `CTV_45`,
`PTV1` and `ptv boost` qualify; `CTVx`, `ODBYTNICA` and `PECHERZ` do not.
An `RTROIInterpretedType` of `GTV`, `CTV` or `PTV` also qualifies.
`TREATED_VOLUME` alone does not. Helpers such as `z_PTV_opt` and `PTV-ring`
are detected; detection does not establish clinical target status. Existing
identity, publication and measurability requirements still apply.

When selection matches nothing, the detector checks every evaluated source
structure set. No detected target gives `not_applicable_no_target_structure`.
A detected but unselected target gives `selection_gap_targets_unselected`.
Both close the stage without aborting the workflow. A selection gap remains
technically missing in the ledger, rollup, cohort admission and aggregation.
It never counts as not applicable or measured.

The receipt records the mode, requested patterns and missed target names with
the matching detector rules. The mode enters the effective configuration hash
and measured `configured_parameter_hash`. The original extractor hash remains in
`extraction_configured_parameter_hash`. The ledger `detail` preserves the named outcome, including when
rollup reconstructs accounting from the receipt alone. Rollup also reports
`technically_missing_unit_count` and the units with their reasons.

The aggregate workbook retains each admitted course in `course_outcomes`, with
`accounting_status`, `technically_missing`, `target_selection` and
`nonmeasurement_reason`. `course_counts` separates measured, technically missing,
not-applicable and source-only courses. No-match courses contribute no raw
measurement rows or ICC, CoV or QCD estimates. Evaluated source dispositions
remain in `source_dispositions`.

Receipts carrying `not_applicable_selection_unmatched` are stale and require
recomputation; they are never reinterpreted. Changes to the mode, patterns,
source bytes, code or receipt bindings require a new run. Stage completion
alone does not establish clinical or manuscript readiness.
