# D1 implementation brief — original + AI structures in the all-series path

**Worktree (work HERE only):** `/umed-projekty/rtpipeline-wt-phaseB` (branch `feat/phaseB-d1-b1-b3-b2`, integrated Phase-B base, 289 tests green).
**Scope:** implement ONLY item D1 from the triple-consensus-approved design. Do NOT implement B1/B3/B2/E1/E3/MTV-TLG. Surgical changes only.

## What D1 is (from approved DESIGN_v3 §D1 / DESIGN_v2 §D1, R6)

The standing project directive is "every image gets BOTH the original clinical RTSTRUCT structures AND the AI (TotalSegmentator) structures." Today, original-structure export (`_export_original_segmentation`, `rtpipeline/organize.py:1005`) runs only on the **per-course** path (it is hardwired to `CourseOutput` attrs: `course.rs_path`, `course.primary_nifti`, `course.dirs.dicom_ct`, `course.dirs.segmentation_original`). The **all-series** path (`do_segment_all_series`, `organize.py:~2498`) emits AI structures for every eligible series but never exports the original clinician contours for those series.

D1 wires original-structure export into the all-series path, **manifest-anchored**:

1. **Bind** the original clinical RTSTRUCT to each all-series series by the **manifest referenced-series-UID / study / FrameOfReferenceUID identity** (reuse C1's `referenced_ct_series_uids` / source-series identity already in `series_manifest.json`), **NOT** directory heuristics. Only the series that the RTSTRUCT actually references (in practice: the RT-course planning CT) gets an original export.
2. **Export with a distinct `source` provenance field** so clinician + AI ROI names can never collide/mix: original export manifest must carry `source="manual"` (or `model="manual"` as the existing code already does at `organize.py:1044`), AI exports keep their `total`/`total_mr` model id. Verify the consuming/merge code never unions a `manual` ROI with an AI ROI of the same name.
3. **Series with no referenced original → log a structured "no original available" line and skip** (reality: only the planning CT carries clinician contours; this is expected, not an error).
4. **Reuse the existing `_export_original_segmentation` core** (`organize.py:1005`) — refactor it so the all-series caller can pass series-level inputs (rtstruct path, primary nifti, dicom-CT dir, output seg-original dir) instead of a whole `CourseOutput`. Keep the per-course caller byte-identical in behavior. The design notes D1 reuses "the same real-temp-course-tree mechanism as §B4" — see `run_radiomics_all_series` in `rtpipeline/radiomics.py` for the established pattern of materializing a real temp course-shaped tree (`DICOM/CT/` + masks + RTSTRUCT) under `work/.all_series_*/<pid>/<series_uid>/`, symlink where possible, **deleted on completion**. Prefer a small parameter refactor of `_export_original_segmentation` over a temp tree if it is cleaner and equally correct; if a temp tree is needed, mirror §B4 exactly (cleanup + minimal `DICOM/CT/` + `RS_auto.dcm` layout).

## Constraints (hard)
- **Course path byte-identical**: no behavior change to the existing per-course `_export_original_segmentation` callers. Add a regression test asserting this.
- Read the real artifacts first: `organize.py:1005-1090` (`_export_original_segmentation`), `organize.py:2490-2515` (all-series block), `rtpipeline/inventory.py` (`series_manifest.json` schema + `TS_TASK_BY_CLASS` + `referenced_ct_series_uids`), `rtpipeline/radiomics.py` `run_radiomics_all_series` (the temp-tree pattern + manifest reading), and C1's referenced-series identity logic (`rtpipeline/auto_rtstruct.py` / `percourse` selection).
- **Tests:** add focused unit tests (synthetic manifest + a referenced series with an RTSTRUCT → original exported with `source/model="manual"`; a series NOT referenced → skipped + logged; collision guard: a clinician ROI named like an AI label stays separate). Keep the existing 289 baseline tests passing.
- Run `python -m pytest tests/ -q` (env: `/home/konrad/micromamba/envs/rtpipeline/bin/python`) and report the count.
- Do NOT commit. Leave changes staged/unstaged in the worktree; I will review the diff, run triple-consensus, then commit.
- If anything in the spec is ambiguous against the real code, STOP and write your question into `D1_NOTES.md` rather than guessing.

## Deliverable
- Code changes implementing D1 + new tests, all tests green.
- A short `D1_NOTES.md`: what you changed (file:line), the binding mechanism you used, any deviations from this brief with justification, and the final pytest count.
