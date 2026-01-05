# RTpipeline Improvement Ideas

This document logs improvement suggestions and feature ideas identified during the scientific and technical code review. These are **enhancement ideas** (not bugs), logged for future consideration.

## Format

Each idea follows this format:
- **ID**: `IDEA-XXX`
- **Stage**: Pipeline stage where idea was identified
- **Priority**: Low / Medium / High
- **Description**: What could be improved
- **Rationale**: Why this improvement matters

---

## Ideas Log

### IDEA-001: Configurable Dose Plausibility Threshold

- **Stage**: 1. DICOM Organization
- **Priority**: Medium
- **Description**: The `max_total_dose_gy` parameter in `_classify_doses()` is hardcoded to 100 Gy. This should be configurable via config.yaml to support hypofractionated regimens (SBRT, SRS) where total doses can exceed 100 Gy.
- **Rationale**: SBRT lung treatments may prescribe 54 Gy in 3 fractions, brain SRS can reach 120 Gy+ when summing multiple targets. The hardcoded 100 Gy threshold would incorrectly flag these as plausibility warnings.
- **References**: RTOG 0236 (SBRT lung), RTOG 0631 (spine SRS)

### IDEA-002: Use Standard DICOM Frame of Reference Tag for RTDOSE

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: The `rt_details.py` module uses tag (0x3006, 0x0024) for Frame of Reference UID extraction from RTDOSE files. This is the Referenced Frame of Reference UID from the Structure Set Module. The standard tag for Frame of Reference UID is (0x0020, 0x0052).
- **Rationale**: Most TPS export the same UID in both tags, but some may differ. Using the standard tag ensures broader DICOM compliance and prevents potential course grouping errors.
- **References**: DICOM PS3.3 C.7.4.1 (Frame of Reference Module), DICOM PS3.3 C.8.8.3 (RT Dose Module)

### IDEA-003: Adaptive Therapy Workflow Support

- **Stage**: 1. DICOM Organization
- **Priority**: Medium
- **Description**: The current replan detection (Phase 3 of dose classification) uses intention-to-treat (ITT) logic that keeps only the first plan. For adaptive therapy workflows, multiple re-plans may be intentionally delivered and should be summed.
- **Rationale**: Adaptive radiotherapy (ART) protocols like ARTFORCE involve planned re-optimization based on anatomical changes. The current ITT logic would incorrectly discard these legitimate dose contributions.
- **References**: ARTFORCE trial protocol, Elekta MOSAIQ adaptive workflows

### IDEA-004: Z-Extent Calculation Improvement for Dose Grid

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: In `_extract_dose_metadata()`, when `GridFrameOffsetVector` is empty, z-extent assumes 1mm spacing. Should check for SliceThickness attribute in the RTDOSE file as a fallback.
- **Rationale**: Prevents incorrect bounding box calculations that could affect dose grid overlap detection in Phase 2.5 of dose classification.
- **References**: DICOM PS3.3 C.8.8.3.4 (Grid Frame Offset Vector)

### IDEA-005: Course Clustering Algorithm Enhancement

- **Stage**: 1. DICOM Organization
- **Priority**: Low
- **Description**: The date-based course splitting in `group_by_course()` uses a simple pivot-based approach that may incorrectly split interleaved courses (e.g., concurrent chemo-RT where treatments alternate days).
- **Rationale**: While rare, interleaved treatment schedules exist. A more sophisticated clustering algorithm (e.g., DBSCAN on date differences) could handle these cases better.
- **References**: N/A - edge case enhancement

<!-- Template for new ideas:

### IDEA-XXX: [Short Title]

- **Stage**: [Stage Name]
- **Priority**: [Low/Medium/High]
- **Description**: [What could be improved]
- **Rationale**: [Why this improvement matters scientifically or technically]
- **References**: [Relevant papers, standards, or documentation]

-->

---

## Summary by Stage

| Stage | Ideas Count |
|-------|-------------|
| 1. DICOM Organization | 5 |
| 2. CT Processing | 0 |
| 3. Segmentation | 0 |
| 4. Custom Models | 0 |
| 5. Custom Structures | 0 |
| 6. CT Cropping | 0 |
| 7. DVH Analysis | 0 |
| 8. Radiomics Extraction | 0 |
| 9. Robustness Analysis | 0 |
| 10. Quality Control | 0 |
| 11. Aggregation | 0 |
| 12. Configuration | 0 |

---

## Priority Summary

- **High Priority**: 0
- **Medium Priority**: 2 (IDEA-001, IDEA-003)
- **Low Priority**: 3 (IDEA-002, IDEA-004, IDEA-005)

---

*Last updated: 2026-01-06*
