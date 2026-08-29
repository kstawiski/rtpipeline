# Radiomics ROI requiredness diagnosis

Source-based region-of-interest requiredness restored radiomics output for 1 of 4 staged Kopernik pelvic courses without accepting failed clinical contours. The restored course retained 4 failed of 77 automatic organ regions and completed as degraded, while 3 courses remained fail-closed on clinical regions.

This result fixes the observed automatic-organ failure but does not make the sampled cohort radiomics-ready. Clinical contour defects remain the next blocking problem and require a separate governed repair.

## Clinical and computational problem

The affected material is the rebuilt Kopernik pelvic cohort. Its radiomics stage combines clinical radiotherapy structure set contours, 5 configured custom structures, and TotalSegmentator organ masks. The endpoint is a course-level feature table that identifies every attempted, extracted, and failed region by source.

The pre-change completion predicate made every region task course-critical. Any task exception or `None` result set `fatal_error` to `RadiomicsCourseExtractionError` and stopped the course in `rtpipeline/radiomics_parallel.py`.

Course 475201/2025-03 demonstrated the targeted defect. Its log records failure of `AutoRTS_total/vertebrae_T8` because the automatic structure lacked a readable `ContourSequence`. Recorded progress reached 91 of 103 tasks before that single automatic-organ failure prevented a course result.

The name identifies a thoracic vertebra in a pelvic scan. This mismatch supports an out-of-field mechanism but does not prove that every failed automatic mask is geometrically degenerate. TotalSegmentator emits the total-task organ inventory independently of the scanned region.

The sampled automatic structure inventory was as follows.

| Course | Automatic organ regions | Thoracic-region organs |
| --- | ---: | ---: |
| 428073/2020-05 | 84 | 34 |
| 431057/2020-07 | 56 | 12 |
| 475201/2025-03 | 77 | 29 |
| 482967/2024-11 | 60 | 14 |

The 4 courses contained 277 automatic regions, including 89 thoracic-region organs. These are sums of the packet-supplied course counts.

The packet reported failure of all 4 sampled courses before repair. Read-only log inspection identified named course-level region failures in 3. The fourth log stopped at 6 of 97 tasks with no course result, no named region failure, and no radiomics workbook.

## Requiredness rule adopted

Requiredness now follows the region source and operator configuration. It does not follow anatomy names, substring lists, or a wider `skip_rois` list.

The configured structures `iliac_vess`, `iliac_area`, `pelvic_bones`, `pelvic_bones_3mm`, and `bowel_bag` are required. Every region from the clinical structure set is also required, including targets and other investigator contours.

A read, mask, timeout, or feature-extraction failure in a required region fails the course and invalidates its workbook and sidecar. A persisted required failure or missing required outcome receives the same fail-closed treatment during resume.

Regions from `AutoRTS_total` are best effort. An individual failure is retained but does not fail the course. Unclassified sources remain required by default, which keeps the policy conservative.

## Degenerate masks and the voxel threshold

The repair distinguishes a present but undersized mask from an unreadable or empty mask. A region with fewer than the configured 64 voxels receives `below_minimum_voxels` and `degenerate_mask` status.

An undersized region is not newly fatal when its source is required. This preserves the prior minimum-voxel behavior. The status row remains visible and counts as an attempted but failed extraction, so the course is degraded rather than silently complete.

An empty or unreadable required mask remains fatal because no eligible observation exists. The same condition in an automatic organ is recorded as a best-effort failure.

The `vertebrae_T8` log records a missing `ContourSequence`, not a measured voxel count. The worker could not obtain a usable mask, so the 64-voxel gate could not classify it before extraction. The repair records this as an extraction error rather than inferring a threshold failure.

The clinical `znacznikAg` and `m5` failures are a distinct unresolved class. Their structure entries have no readable `ContourSequence`. They are not evidence that the minimum-voxel rule should become fatal, and they are outside this source-policy change.

## Failure surfacing

Each failed best-effort region is retained in the course table with its source, original name, extraction status, failure kind, and reason. The row appears beside successful feature rows rather than disappearing from the denominator.

`RadiomicsCourseOutcome` exposes attempted, extracted, and failed counts for each source. Any retained failure changes the course status to `EXTRACTED_WITH_FAILURES`, which is also exposed through the `DEGRADED` alias.

The course and cohort workbooks carry the course status, total counts, and a stable source-count mapping. A reader can distinguish complete automatic-organ extraction from a course in which some or all automatic regions failed.

A course with successful required regions and failure of every automatic organ is written as degraded. A course with any failed required region produces no valid feature table.

## Measured before and after results

A separately staged rerun used the 4 campaign course inputs through read-only links. All outputs were written under the orchestrator workspace. The rerun used the configured 64-voxel minimum, the pelvic custom-structure file, 4 workers per course, and the repaired working tree based on commit `efce0145e7d771f4440f62fc2734c2865ab0b61f`.

| Course | Pre-change evidence | Post-change staged result | Disposition |
| --- | --- | --- | --- |
| 428073/2020-05 | `Manual/znacznikAg` failed because `ContourSequence` was absent | `Manual/znacznikAg` remained fatal | Required clinical failure remains unresolved |
| 431057/2020-07 | `Manual/znacznikAg` failed because `ContourSequence` was absent | `Manual/znacznikAg` remained fatal | Required clinical failure remains unresolved |
| 475201/2025-03 | `AutoRTS_total/vertebrae_T8` failed because `ContourSequence` was absent | Completed as `EXTRACTED_WITH_FAILURES` | Targeted automatic-organ failure resolved at course level |
| 482967/2024-11 | No course-level failing region or source was recorded before the log stopped at 6 of 97 tasks | `Manual/m5` failed because `ContourSequence` was absent | Staged rerun identified a required clinical failure |

Before repair, 0 of 4 sampled courses had a completed radiomics workbook in the available campaign evidence. Three logs named a course-level failure, while 1 course was interrupted without a recorded course outcome.

After repair, 1 of 4 staged courses completed as degraded and 3 of 4 failed on required clinical regions. The repair therefore restored 1 of the 3 courses with an observable pre-change region failure.

The restored 475201/2025-03 course attempted 103 regions. All 21 clinical regions and all 5 custom regions produced features. The automatic source produced features for 73 of 77 regions and retained 4 failures.

The 4 automatic failures were `kidney_cyst_right` with 51 voxels, `lung_upper_lobe_right` with 22 voxels, `skull` with 34 voxels, and unreadable `vertebrae_T8`. The first 3 received `below_minimum_voxels`. The fourth received an extraction-error status.

The staged workbook contains 103 rows and reports `extracted_with_failures`. Its SHA-256 is `2cd683265087f4dc096bed1f24d4205bca59b53481bb4e8b6129b83ead01d3cd`. No workbook or parquet sidecar remained for the 3 failed courses.

## Verification and traceability

Focused checks passed 121 tests across `tests/test_radiomics_fail_closed.py`, `tests/test_radiomics_conda_resilience.py`, and `tests/test_reliability_batch_b.py`. They covered fresh required and best-effort failures, all-auto failure, source counts, minimum-voxel behavior, and resume reconstruction.

The full suite passed 934 tests and skipped 1 test on the supported Python 3.11 environment. The required Manual and Custom resume cases invalidated persisted outputs, while the automatic resume case retained its failure reason and source counts.

Pre-change failures are recorded at lines 38 and 53 for 428073/2020-05, lines 37 and 52 for 431057/2020-07, and lines 1260 and 1275 for 475201/2025-03. The 482967/2024-11 log ends at lines 202 and 203 without a course result.

The staged course receipts, configuration, implementation hashes, workbook identity, and output locations are recorded in `.orchestrator/runs/roi-requiredness-20260829T111012Z/recheck/sampled-course-summary.json`. Fact, calculation, and inference records are separated in `analysis/roi_requiredness_evidence.json`.

The source-policy repair prevents out-of-field automatic organs from erasing an otherwise valid course. Cohort radiomics should not proceed until malformed clinical contour entries are repaired or assigned an explicit governed eligibility status. Clinical regions should not be reclassified as best effort.
