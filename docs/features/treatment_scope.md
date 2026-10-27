# Treatment-region classification

`rtpipeline scope` applies two versioned research classifiers to radiotherapy targets: a names-only rule and four geometric rules, C1–C4. Each course receives a class from each geometric rule. Consensus is the common class when all four agree; otherwise it is U. No criterion is selected automatically.

| Class | Names-only interpretation | Geometric interpretation |
|---|---|---|
| B | Explicit bladder target anatomy, with optional regional pelvic targets and no other named pelvic primary or nonpelvic target | All targets pelvic, with a qualifying bladder overlap and no qualifying second pelvic gross target |
| P | Pelvic target anatomy without explicit bladder target anatomy | All targets pelvic, without qualifying bladder overlap |
| MP | Bladder and another named pelvic primary target | Distinct bladder-dominant and other pelvic organ-dominant GTVs |
| X | Nonpelvic target anatomy only | All targets outside the pelvic slab, or sufficient superior nonpelvic landmark evidence |
| MX | Pelvic and nonpelvic target anatomy | Pelvic and nonpelvic targets, or a target crossing the pelvic boundary |
| U | Insufficient, contradictory or unreadable target evidence | Unresolved geometry, failed input checks or disagreement between C1–C4 |

These are descriptors of target anatomy and overlap. They do not establish diagnosis, histology, treatment delivery, appropriateness, clinical eligibility. In particular, geometric MP is not proof of two primary malignancies. Agreement between thresholds measures stability, not anatomical validity. A names-only U can coexist with a resolved geometric class.

## Rules and target selection

The default resource is `rtpipeline/treatment_scope_rules_v1_1.yaml`, schema `rtpipeline-treatment-scope-v1.1`, version `1.1.0`. It contains the names vocabulary, helper exclusions, target selection predicates, C1–C4 thresholds and geometric tolerances. Version `1.0.0` is available through `load_rules("1.0.0")` or the explicit `treatment_scope_rules_v1.yaml` path. Both versions have embedded JSON resources for distributions without YAML package data. The target-name predicate is implemented locally so that changes to `rt_details` cannot change this classifier.

A nonblank interpreted ROI type takes precedence over the name: GTV, CTV or PTV qualifies. For geometry, an ROI whose sole interpreted type is TREATED_VOLUME also qualifies as CTV when its name is exactly HR-CTV or IR-CTV, case-insensitive, with one hyphen, underscore or space separator. Other TREATED_VOLUME names and mixed interpreted types do not receive this allowance. Without an interpreted type, the packaged target-name predicate identifies candidates. Helper and optimization structures are excluded. A selected target needs at least one closed planar or XOR contour with at least three points. Contradictory target and nontarget interpreted types produce U. Repeated ROI numbers are rejected.

For names-only classification, names undergo lowercase conversion, Polish ł transliteration and accent removal. Only selected target names contribute anatomy. Generic target names, unlocalized nodes and ambiguous anatomical terms remain unresolved unless another selected target supplies explicit anatomy. Organ-at-risk vocabulary and prescription schedules do not assign anatomy. The names rule retains its prescription compatibility parameters as contextual information; the public classifier does not use a prescription to resolve U. Names-only and geometric helper selection retain their respective frozen behavior.

## Geometry and overlap

Fresh measurement uses the authoritative original RTSTRUCT, planning CT and automatic organ masks from an organized course contract. Plan membership comes from the contract's selected plans. When that membership is empty, geometry uses original RTPLANs in the course's `DICOM/RTPLAN` collection that reference the authoritative structure, regardless of ApprovalStatus. A nonempty selected membership is never replaced by this fallback. All admitted plans must reference the authoritative structure and pass the identity checks. The output records `plan_selection_basis` and each plan's `approval_status` and `selection_basis` in `selected_plan_evidence`; absent approval status is recorded as `ABSENT`. The fallback establishes planned-geometry membership only. Prescription, dose and delivery authority remain separate. CT slices, frame, series, provenance, grid and sampled CT pixel equivalence are checked. The segmentation manifest and completion receipt bind named masks to the CT and to their bytes. Source file hashes and collection membership are checked before and after measurement. The input directory is read only.

Contours are filled at voxel centers on the planning CT grid with an even-odd rule. There is no interpolation or supersampling. Actual DICOM slice planes support equivalent signed or permuted storage grids. Volume is voxel count multiplied by voxel volume. Millilitres and cubic centimetres are numerically identical.

The pelvic slab extends from the lower extent of the bilateral hip masks to the upper extent of L5, with 10 mm padding at each end. At least 95% of a target must lie within the slab to be pelvic; at most 5% makes it outside. Intermediate fractions, or separate pelvic and outside targets, produce MX. Without complete pelvic landmarks, X requires every target to be at least 10 mm above the lowest visible inferior extent among C1–L4 vertebral masks. Otherwise the course is U. Automatic organ occupancy must span the target superior–inferior extent within 3 mm. An empty bladder mask makes a wholly pelvic course U; it does not prevent X or MX.

Each selected target is tested separately:

| Criterion | Required bladder intersection |
|---|---|
| C1 | At least 5 cm³ **and** at least 10% of the target |
| C2 | At least 5 cm³ |
| C3 | At least 50% of the bladder mask |
| C4 | At least 90% of the bladder mask |

Thresholds are inclusive. Raw bladder-hit flags are returned for every target even when geometry prevents a course-level B or MP assignment. A raw hit alone is not a course class.

MP requires a bladder-hit GTV whose bladder target fraction equals or exceeds its fraction in each available other pelvic primary organ. Another GTV must overlap prostate, uterus or rectum by at least 30% of its own volume and by at least twice its bladder fraction. Pairwise overlap divided by the smaller GTV volume must be at most 50%, and centroid separation at least 20 mm. CTVs and PTVs cannot establish the second gross target. Absent organ labels are not inferred from other masks.

C1 and C2 are not nested at the course-class level. A synthetic example has a CTV with 200 cm³ bladder intersection occupying 20% of the target; a GTV with 8 cm³ bladder intersection occupying 8% of the target and 2% prostate overlap; and a second GTV with 1 cm³ bladder intersection occupying 1% of the target and 40% prostate overlap. With zero pairwise containment and 30 mm GTV centroid separation, C1 is B and C2 is MP. C2 admits the first GTV as the bladder-dominant gross target. The public tests include this example.

The representative target is the actual selected target with the largest bladder intersection; the lowest ROI number breaks ties. Its reported values are target volume, bladder intersection volume, percentage of target, percentage of bladder and bladder-mask volume. These values are not a union or a sum. All individual-target values remain available. Missing measurements are blank with an explicit reason, rather than zero.

## U reasons

The API returns criterion-specific reasons, a consensus `u_reason`, and a separate reason when overlap values are unavailable. For measurement failures, `u_reason` retains the saved technical hold code; `reasons` identifies the classification failure.

| Reason | Meaning |
|---|---|
| `measurement_or_identity_failure` | Measurement failed, input identity could not be established, or required geometry is incomplete |
| `target_outside_CT_or_unsupported_contour` | Target contour points extend outside the CT or an unsupported contour is present |
| `contradictory_target_identification` | Selected ROI observations contain contradictory interpreted types |
| `automatic_organs_do_not_span_target_z` | Organ occupancy does not cover the target superior–inferior extent |
| `pelvic_landmarks_absent_and_no_complete_nonpelvic_proof` | The available landmarks cannot resolve pelvic extent |
| `unresolved_pelvic_extent` | Pelvic extent remains unresolved |
| `empty_automatic_bladder_mask_prevents_pelvic_subclassification` | A wholly pelvic course cannot be separated into B, P or MP |
| `cross_criterion_disagreement` | C1–C4 give different classes |
| `insufficient_explicit_target_anatomy` | Names-only evidence does not localize target anatomy |
| `contradictory_interpreted_types` | Names-only target identification is contradictory |
| `source_resolution_error` | Names-only source evidence cannot be resolved |

Overlap can additionally be unavailable because of `empty_automatic_bladder_mask`, `bladder_overlap_evidence_absent` or `selected_target_evidence_absent`. These codes concern the numerical summary and do not replace the class reason.

## Python interface

```python
from rtpipeline.treatment_scope import names_only_class, geometric_classes, load_rules

rules = load_rules()
name_class, name_evidence = names_only_class("/project/course/RTSTRUCT.dcm", rules)
result = geometric_classes("/project/output/person/course", rules)

result["classes"]                  # C1, C2, C3, C4
result["consensus"]
result["raw_bladder_hits"]          # one row per selected target
result["representative_overlap"]
result["selected_target_overlaps"]  # volumes, percentages and meets_C1–meets_C4
result["u_reason"]
result["plan_selection_basis"]
result["selected_plan_evidence"]    # approval status and membership basis
```

`names_only_class` accepts a pydicom Dataset, a DICOM path or saved names evidence containing ROI names, interpreted types and valid closed-contour counts. Saved names evidence is retokenized and reselected; cached class, token and target flags are not used. Source-resolution failures remain U.

`geometric_classes` accepts an organized course directory, a saved measurement mapping or a saved record with an `evidence` member. A saved measurement must retain targets, organ statistics, target exclusions, per-target overlaps, z histograms and pairwise containment from the measurement engine. Replaying evidence does not reopen or authenticate its original source paths. A course directory triggers fresh measurement and requires its original metadata, DICOM CT/RTSTRUCT/RTPLAN collections, CT NIfTI, segmentation manifest, masks and completion receipt. There is no cohort root or campaign path in the package.

Either API accepts a packaged version string, an explicit JSON or YAML path or the relevant rule mapping. A full bundle is recommended when supplying rules. Changes to rules describe a different analysis and require their own recorded provenance; replay equivalence depends on the selected version and retained evidence. Saved contour-count evidence supports replay of the frozen selection rule, not reconstruction of the original contour points under a different selection threshold.

## Command line

```bash
# Names only from an RTSTRUCT
rtpipeline scope --rtstruct /project/course/RTSTRUCT.dcm --output /project/results/names.json

# Fresh geometry from one organized course
rtpipeline scope --course /project/output/person/course --output /project/results/geometry.json

# Replay saved evidence, optionally with saved names evidence
rtpipeline scope --evidence /project/evidence/geometry.json \
  --names-evidence /project/evidence/names.json \
  --output /project/results/scope.json

# Explicit rule bundle
rtpipeline scope --evidence /project/evidence/geometry.json \
  --rules /project/rules/treatment_scope_rules_v1.yaml
```

The equivalent module entry point is `python -m rtpipeline.treatment_scope`. Stdout contains a summary without source identifiers or ROI names. `--output` creates a new detailed JSON file with owner-only permissions and refuses to overwrite an existing file. Names evidence retains original ROI names and numbers and must be handled as clinical data. Exit status 0 means classification completed, including a U result; status 2 means invalid CLI input or an output failure.

## Verification

Public synthetic controls cover the geometry-only plan fallback, exact-name TREATED_VOLUME admission and its exclusions, version selection, class assignment, inclusive thresholds, interpreted-type precedence, helper exclusions, ambiguous names, the B-to-MP example, representative-target selection, signed and permuted grids, contour holes, fresh DICOM measurement, source-byte checks and CLI dispatch:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q -c /dev/null -p no:cacheprovider tests/test_treatment_scope.py
```

Saved-evidence regression checks replay equivalence for each selected rule version. It does not independently validate anatomical accuracy.
