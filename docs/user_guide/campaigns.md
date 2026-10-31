# Campaign accounting

With `campaign_mode: true`, each robustness course closes with a recorded
outcome. A successful CLI call publishes a receipt; the workflow revalidates it
before recording the outcome in `_campaign_ledger/records`. Both local and
container rules use this closure. A missing or invalid receipt stops the job.

| Robustness outcome | Campaign status | Measurement contribution |
| --- | --- | --- |
| `measured` | `ok` | Published measurement rows |
| `source_only_nonvolumetric` | `ok` | Zero |
| `not_applicable_selection_unmatched` | `not_applicable` | Zero |
| `not_applicable_no_planning_ct` | `not_applicable` | Zero |
| `failed_extraction` | `failed` | Zero |

`not_applicable_selection_unmatched` means that no source structure matched the
requested robustness patterns. It describes selection or configuration and
does not establish anatomical absence. The stage returns success so other
courses can continue. Its receipt records the patterns and binds the evaluated
source dispositions. The ledger `detail` preserves the named outcome, including
when a rollup reconstructs accounting from the receipt alone.

The aggregate workbook retains each admitted course in `course_outcomes`, with
the reason in `nonmeasurement_reason`. Unmatched courses contribute no raw
measurement rows and no ICC, CoV or QCD estimates. Evaluated source dispositions
remain available in `source_dispositions`. Changing the selection requires a
new run; source, code, configuration and receipt checks still apply.

Outside campaign mode, the same unmatched-selection receipt closes the
individual stage. Technical extraction failures retain their separate failure
contract. Stage completion alone does not establish clinical or manuscript
readiness.
