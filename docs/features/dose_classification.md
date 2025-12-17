# Intelligent Dose Classification

## Overview

Radiotherapy patients often have multiple dose files associated with a single treatment course. These may represent:
- **Primary + Boost treatments** - Sequential treatment phases targeting the same region at different dose levels
- **Replanning** - Same treatment re-planned due to anatomical changes (weight loss, tumor shrinkage)
- **Different anatomical regions** - Separate RT courses to different sites (e.g., prostate + brain met)
- **TPS-provided summation** - Treatment Planning System already exported a summed dose

The pipeline uses a **6-phase classification algorithm** to intelligently handle these scenarios and avoid incorrect dose summation (which could produce clinically impossible values like 180 Gy).

---

## Classification Algorithm

### Phase 1: TPS PLAN_SUM Detection

**Goal:** Detect if the Treatment Planning System already exported a summed dose.

**Logic:**
- Scan all dose files for `DoseSummationType == "PLAN_SUM"`
- If found, use this pre-computed sum directly
- Individual component doses are excluded to avoid double-counting

**Result:** `PLAN_SUM_used` - No further summation needed

### Phase 2: FrameOfReference Separation

**Goal:** Identify doses in different coordinate systems (cannot be summed).

**Logic:**
- Extract `FrameOfReferenceUID` from each dose file
- If multiple unique UIDs exist, doses are on different imaging grids
- These represent separate RT courses that should not be combined

**Result:** `separate_courses_no_sum` - Use first dose only

### Phase 2.5: Geometric Overlap Detection

**Goal:** Identify doses targeting different anatomical regions even within the same coordinate system.

**Logic:**
- Extract 3D bounding box from each dose grid:
  - `ImagePositionPatient` (origin)
  - `PixelSpacing`, `Rows`, `Columns` (XY extent)
  - `GridFrameOffsetVector`, `NumberOfFrames` (Z extent)
- Calculate pairwise intersection volume
- Require at least 30% overlap of the smaller volume to consider doses as overlapping

**Result:** `separate_regions_no_sum` - Use first dose only (different anatomical sites)

### Phase 3: Replan Detection (Intention-to-Treat)

**Goal:** Identify replanning scenarios and apply ITT principle (use first plan only).

**Logic:**
- Link doses to their referenced RT plans via `ReferencedRTPlanSequence`
- Sort by plan date (earliest first)
- Detect replans by:
  - **Text patterns:** "replan", "re-plan", "adaptive", "v2", "copy", "resim", etc.
  - **Prescription similarity:** >90% similar Rx suggests same treatment intent
- Exclude later plans that appear to be replans of the first

**Result:** `replan_itt_first` - Use first plan only (ITT approach)

### Phase 4: Primary + Boost Identification

**Goal:** Identify genuine multi-stage treatments that should be summed.

**Logic:**
- Detect boost by text patterns: "boost", "cone", "conedown", "phase 2", "sib", etc.
- Detect by prescription pattern: significantly different Rx values (>10% difference)
- If primary and boost are identified, sum them

**Result:** `primary_boost_summed` - Sum the doses

### Phase 5: Plausibility Safeguards

**Goal:** Flag clinically implausible total doses.

**Logic:**
- After classification, calculate expected total dose from prescriptions
- If total > 100 Gy, add warning (possible classification error)
- Proceed with summation but log warning for review

**Result:** Warning added to classification result

### Fallback: Ambiguous Cases

**Goal:** Conservative handling when classification is uncertain.

**Logic:**
- If none of the above phases definitively classify the doses
- Use only the first dose (conservative approach)
- Log warning for manual review

**Result:** `ambiguous_no_sum` - Use first dose only

---

## Classification Results

The algorithm returns a `DoseClassification` object containing:

| Field | Type | Description |
|-------|------|-------------|
| `classification` | str | Classification label (see below) |
| `selected_doses` | List[Path] | Doses to use for DVH/analysis |
| `selected_plans` | List[Path] | Plans associated with selected doses |
| `excluded_doses` | List[Path] | Doses excluded from analysis |
| `should_sum` | bool | Whether selected doses should be summed |
| `warnings` | List[str] | Any warnings generated |
| `reason` | str | Human-readable explanation |

### Classification Labels

| Label | Meaning | Action |
|-------|---------|--------|
| `PLAN_SUM_used` | TPS-provided sum detected | Use PLAN_SUM directly |
| `separate_courses_no_sum` | Different FrameOfReference | Use first dose |
| `separate_regions_no_sum` | Non-overlapping dose grids | Use first dose |
| `replan_itt_first` | Replan detected | Use first plan (ITT) |
| `primary_boost_summed` | Primary + boost identified | Sum doses |
| `ambiguous_no_sum` | Cannot classify | Use first dose |
| `single_dose` | Only one dose file | Use it directly |
| `no_doses` | No dose files found | No DVH analysis |

---

## Technical Implementation

### Replan Detection Keywords

```python
replan_keywords = [
    "replan", "re-plan", "adapt", "adaptive", "revision", "rev",
    "v2", "v3", "v4", "v5", "copy", "fx change", "new ct", "resim",
    "replanning", "modified", "adjusted", "corrected",
]
```

### Boost Detection Keywords

```python
boost_keywords = [
    "boost", "cone", "conedown", "cone down", "cd", "phase 2",
    "phase2", "ph2", "reduced", "sib", "sequential",
]
```

### Bounding Box Overlap Calculation

```python
def _bboxes_overlap(bbox1, bbox2, min_overlap_fraction=0.3):
    """
    Check if two 3D dose grids have significant spatial overlap.

    Requires at least 30% of the smaller volume to overlap.
    Returns True if geometry is unavailable (conservative).
    """
```

---

## Configuration

The maximum plausible dose threshold can be configured:

```python
max_total_dose_gy: float = 100.0  # Default threshold for plausibility warning
```

---

## Logging

The classification process logs detailed information:

```
Phase 1: Found TPS PLAN_SUM covering 6 plans, excluding 6 individual doses
Phase 2: Multiple FrameOfReference detected - using first dose only (conservative)
Phase 2.5: Non-overlapping dose grids detected: RD_pelvis.dcm vs RD_brain.dcm
Phase 3: Detected replan by text: 'v2_adaptive_ct2'
Phase 4: Detected primary+boost by Rx pattern: 50.0 Gy (primary) + 16.0 Gy (boost)
Phase 5: Implausible total dose 180.6 Gy > 100.0 Gy threshold - flagging but proceeding
```

---

## Example Scenarios

### Scenario 1: Prostate with Boost
- **Input:** RD_primary.dcm (50 Gy), RD_boost.dcm (16 Gy)
- **Classification:** `primary_boost_summed`
- **Output:** Summed dose (66 Gy)

### Scenario 2: Adaptive Replanning
- **Input:** RD_plan1.dcm (60 Gy), RD_plan1_v2.dcm (60 Gy)
- **Classification:** `replan_itt_first`
- **Output:** First plan only (60 Gy)

### Scenario 3: TPS Export with PLAN_SUM
- **Input:** RD_sum.dcm (PLAN_SUM, 66 Gy), RD_primary.dcm, RD_boost.dcm
- **Classification:** `PLAN_SUM_used`
- **Output:** RD_sum.dcm only

### Scenario 4: Brain Met + Prostate (different regions)
- **Input:** RD_prostate.dcm (78 Gy), RD_brain.dcm (30 Gy, non-overlapping)
- **Classification:** `separate_regions_no_sum`
- **Output:** First dose only (requires manual review for multi-site analysis)

---

## Limitations

1. **Text pattern matching:** Relies on common naming conventions. Unusual naming may miss patterns.
2. **Prescription extraction:** Requires DoseReferenceSequence in RT plan. May be missing in some TPS exports.
3. **Geometric overlap:** Uses bounding box approximation. Actual dose overlap may differ.
4. **Multi-site treatment:** Currently uses first dose only. Future enhancement may support separate analysis per site.

---

## Related Documentation

- [Output Format](../user_guide/output_format.md) - How DVH metrics are exported
- [QC Reports](qc_cropping.md) - Quality control for dose and structure alignment
- [Architecture](../technical/architecture.md) - Pipeline processing flow

---

**Document Version:** 1.0
**Compatible with:** rtpipeline v2.1+
**Last Updated:** 2025-12-17
