# D1 remediation — close the triple-consensus HOLD (test coverage + minor polish)

Worktree: `/umed-projekty/rtpipeline-wt-phaseB`. Env: `/home/konrad/micromamba/envs/rtpipeline/bin/python`. The D1 feature is already implemented (working tree, uncommitted). One reviewer raised an evidence-backed HOLD on **test coverage of the FoR-unique binding branch** plus two minor polish items. Fix ALL of the following. Do NOT change the D1 binding LOGIC (it was reviewed correct); this is additive tests + a log-level tweak + a clarifying comment.

## 1. Tests (the HOLD driver) — add to `tests/test_d1_all_series_original_segmentation.py`
The current tests only exercise the exact `rtstruct_to_series` path. Add focused unit tests for `manual_rtstruct_bindings_from_inventory(...)` (`rtpipeline/inventory.py:294-435`) covering the untested branches:
- **FoR-unique success**: a synthetic inventory where the RTSTRUCT binds via `rtstruct_to_for` with `rt_link_basis == "rtstruct_to_for_unique"` (and/or the single-row `(study_uid, frame_of_reference_uid)` fallback) → the series resolves to the correct source RTSTRUCT.
- **FoR ambiguity skip (fail-closed)**: >1 RTSTRUCT for the same `(study_uid, for_uid)` → binding is skipped (no wrong-series export) with the warning path taken.
- **Multiple-RTSTRUCT-per-series skip (fail-closed)**: >1 exact `rtstruct_to_series` match → skipped fail-closed.
- **Missing/old DB**: `inventory_db_path=None` or an inventory lacking `rt_links` → returns `{}` (no-op), does not raise.
- **Idempotent re-run**: exporting twice returns the cached manual manifest without re-deriving (organize.py:1021-1025 path).
Construct synthetic inventory rows directly (mirror the existing test fixtures' construction of `series_manifest`/inventory); do not require real DICOM where a fixture suffices.

## 2. Minor polish (from the same review)
- **Log volume**: `no_original_available` is emitted at INFO for every unbound segmentable series on every run (`segmentation.py:1021-1022`). At 6,125-patient scale (every CBCT/MR/PT/diagnostic-CT series) this floods logs. Downgrade to `logger.debug(...)`, OR aggregate to a single per-patient summary line (e.g. "N series had no referenced original"). Keep it machine-parseable.
- **Stricter-FoR comment**: add a one-line comment at `inventory.py:~377` noting the intentional stricter gate (D1 builds `by_study_for` only from `rtstruct_to_for` links; this is deliberately stricter than C1's `_rtstruct_targets_for_patient` and can only cause a fail-closed miss, never a wrong-series export).
- **Schema-drift visibility**: where the broad `except Exception → return {}` lives (`inventory.py:~434`), ensure the warning clearly states D1 original-export is disabled due to inventory schema/`rt_links` absence (so a silent no-op is diagnosable). Do not change the fail-closed behavior.

## Constraints
- No change to binding LOGIC or the per-course byte-identical path. Surgical.
- Run `python -m pytest tests/ -q`; ALL must pass (was 291 passed / 1 skipped; expect +~4-5 new tests).
- Do NOT commit. Update `D1_NOTES.md` with a "## Remediation" section: new tests added, the log change, pytest count.
