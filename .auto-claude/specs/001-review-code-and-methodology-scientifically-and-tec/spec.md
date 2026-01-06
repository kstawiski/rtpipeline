# Specification: Scientific and Technical Code Review of RTpipeline

## Overview

This task involves a comprehensive scientific and technical review of the RTpipeline radiotherapy DICOM processing pipeline. The review will systematically evaluate each pipeline stage for correctness in implementation, adherence to scientific standards (IBSI, AAPM/RTOG), and research methodology soundness. The review process requires dual validation with consensus from both the primary reviewer and an external validator (Pal/gpt-5.2) before any component is deemed correct. All improvement suggestions will be logged to IDEAS.md, issues will be fixed iteratively, and the review cycle will continue until both reviewers confirm full correctness.

## Workflow Type

**Type**: feature (investigation + implementation)

**Rationale**: This task combines investigation (reviewing each pipeline component for scientific correctness) with implementation (fixing identified issues). It requires iterative cycles of review → fix → re-review until validation is complete. The "feature" workflow type accommodates the multi-phase nature: discovery, validation, implementation, and verification.

## Task Scope

### Services Involved
- **main** (primary) - RTpipeline Python package containing all pipeline modules
- **webui** (reference only) - Flask web interface (not primary focus but may be reviewed for integration correctness)

### This Task Will:
- [ ] Review each pipeline stage module for scientific/technical correctness
- [ ] Validate methodology against IBSI standards (radiomics) and AAPM/RTOG guidelines (DVH)
- [ ] Create IDEAS.md to log improvement suggestions and new feature ideas
- [ ] Fix identified issues in implementation
- [ ] Re-review fixed code with Pal validation
- [ ] Continue iterative review until consensus on correctness is achieved

### Out of Scope:
- Performance optimization (unless scientifically relevant)
- UI/UX improvements to webui
- Adding new pipeline stages
- Infrastructure changes (Docker, CI/CD)
- Documentation updates (except IDEAS.md)

## Service Context

### RTpipeline (Main Service)

**Tech Stack:**
- Language: Python
- Framework: Snakemake (workflow orchestration)
- Testing: pytest
- Package Manager: pip/conda

**Key Directories:**
- `rtpipeline/` - Core Python modules
- `envs/` - Conda environment definitions
- `webui/` - Flask web interface (reference)

**Entry Point:** `rtpipeline/cli.py`

**How to Run:**
```bash
# Docker (recommended)
docker-compose up

# Local with Snakemake
snakemake --cores all --use-conda --configfile config.yaml
```

**Port:** 5000 (webui)

## Pipeline Stages for Review

The following stages must be reviewed in order:

| Stage | Module(s) | Scientific Focus |
|-------|-----------|------------------|
| 1. DICOM Organization | `organize.py`, `dicom_copy.py` | DICOM standard compliance, course/patient identification |
| 2. CT Processing | `ct.py` | HU normalization, NIfTI conversion, geometry preservation |
| 3. Segmentation | `segmentation.py` | TotalSegmentator integration, anatomical correctness |
| 4. Custom Models | `custom_models.py` | nnU-Net model execution, output validation |
| 5. Custom Structures | `custom_structures.py`, `custom_structures_rtstruct.py` | Boolean operations, margin calculations |
| 6. CT Cropping | `anatomical_cropping.py` | Region-based cropping, geometry preservation |
| 7. DVH Analysis | `dvh.py` | dicompyler-core usage, AAPM/RTOG metric standards |
| 8. Radiomics Extraction | `radiomics.py`, `radiomics_conda.py`, `radiomics_parallel.py` | IBSI compliance, feature extraction correctness |
| 9. Robustness Analysis | `radiomics_robustness.py` | ICC computation, mask perturbation methodology |
| 10. Quality Control | `quality_control.py` | Validation checks, body region detection |
| 11. Aggregation | `scripts/run_aggregate_robustness.py` | Results merging, deduplication logic |
| 12. Configuration | `config.py` | Parameter validation, defaults correctness |

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `rtpipeline/dvh.py` | main | Review and fix DVH metric calculations per AAPM/RTOG |
| `rtpipeline/radiomics.py` | main | Validate IBSI compliance in feature extraction |
| `rtpipeline/radiomics_robustness.py` | main | Review ICC calculation methodology (pingouin usage) |
| `rtpipeline/custom_structures.py` | main | Review boolean operations and margin calculations |
| `rtpipeline/segmentation.py` | main | Review TotalSegmentator integration correctness (see TotalSegmentator pattern below) |
| `rtpipeline/organize.py` | main | Review DICOM parsing and course identification |
| `rtpipeline/ct.py` | main | Review CT preprocessing and NIfTI conversion |
| `rtpipeline/quality_control.py` | main | Review QC check implementations |
| `IDEAS.md` | main | Create to log improvement suggestions |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `CLAUDE.md` | Project architecture documentation, known issues |
| `Snakefile` | Pipeline stage orchestration, dependency graph |
| `config.yaml` | Configuration structure and defaults |
| `rtpipeline/utils.py` | Utility functions and shared patterns |
| `envs/rtpipeline.yaml` | NumPy 2.x environment (TotalSegmentator) |
| `envs/rtpipeline-radiomics.yaml` | NumPy 1.x environment (PyRadiomics) |

## Patterns to Follow

### 1. TotalSegmentator Integration

From research context (TotalSegmentator documentation):

```python
from totalsegmentator.python_api import totalsegmentator

# Basic usage
totalsegmentator(input_path, output_path, ml=False, nr_thr_resamp=1, nr_thr_saving=6)

# Can also return nibabel image object directly
result = totalsegmentator(input_img)
```

**Key Points:**
- GPU recommended (16GB VRAM) but CPU fallback available
- Use `force_split=True` for large volumes to reduce GPU memory
- Models auto-download to `~/.totalsegmentator/nnunet/results`
- Different tasks: `total` (CT, 117 classes) vs `total_mr` (MR, 50 classes)
- Cite: Wasserthal et al. 2023, Radiology: AI

### 2. PyRadiomics IBSI Compliance

From research context (PyRadiomics documentation):

```python
from radiomics.featureextractor import RadiomicsFeatureExtractor

# IBSI-compliant settings are required for reproducibility
extractor = RadiomicsFeatureExtractor(paramPath)
# Config should enable: binWidth, resegmentRange, interpolator settings
features = extractor.execute(imagePath, maskPath)
```

**Key Points:**
- Must use IBSI-compliant parameter settings
- Reproducibility requires reporting all preprocessing/extraction parameters
- Only 32-47% of features are reproducible in inter-observer studies
- Shape features have best ICC (>0.9)

### 3. DVH Metric Extraction

From dicompyler-core documentation:

```python
from dicompylercore import dvhcalc

dvh = dvhcalc.get_dvh(structure, dose, roi, calculate_full_volume=True)
# Standard metrics
metrics = {
    'D100': dvh.statistic('D100'),  # Min dose
    'D98': dvh.statistic('D98'),
    'D95': dvh.statistic('D95'),
    'D50': dvh.statistic('D50'),    # Median dose
    'D2': dvh.statistic('D2'),      # Near-max dose
}
```

**Key Points:**
- Watch dose unit conversion (Gy vs cGy)
- QA tolerance should be <2% deviation from planning system
- Verify against AAPM/RTOG format standards

### 4. ICC Computation with Pingouin

From pingouin documentation:

```python
import pingouin as pg

# ICC(3,1) for fixed raters, single measurement
icc = pg.intraclass_corr(
    data=df,
    targets='subject',
    raters='rater',
    ratings='score'
)
# Select ICC3: ICC(3,1)
icc_value = icc[icc['Type'] == 'ICC3']['ICC'].values[0]
```

**Key Points:**
- Requires complete-case data (no missing values)
- Fully balanced design required
- Interpretation: >0.9 excellent, 0.75-0.9 good, 0.5-0.75 moderate, <0.5 poor
- Uses ANOVA (not mixed effects like R's psych package)

### 5. Mask Perturbation for Robustness

From `radiomics_robustness.py`:

```python
from rtpipeline.radiomics_robustness import volume_adapt_mask

# Erode/dilate mask to target volume change
perturbed = volume_adapt_mask(
    mask,
    target_volume_change=-0.15,  # -15% volume
    spacing=(1.0, 1.0, 1.0)
)
```

**Key Points:**
- Previous bug: function was returning original mask when perturbation failed (fixed Nov 2025)
- Must verify morphological operations are actually applied
- Volume change targets typically: [-0.15, 0.0, 0.15]

## Requirements

### Functional Requirements

1. **IBSI Compliance Verification**
   - Description: Verify PyRadiomics configuration adheres to IBSI standard
   - Acceptance: All radiomics parameter files use IBSI-compliant settings; documentation exists for any deviations

2. **DVH Metric Accuracy**
   - Description: Validate DVH calculations against AAPM/RTOG guidelines
   - Acceptance: DVH metrics (D100, D98, D95, D50, D2, Dcc, Vxx) calculated correctly; unit handling verified

3. **ICC Methodology Correctness**
   - Description: Verify ICC(3,1) implementation using pingouin
   - Acceptance: Complete-case filtering applied; balanced design ensured; no artificial ICC=1.0 values

4. **Segmentation Integration**
   - Description: Validate TotalSegmentator wrapper implementation
   - Acceptance: GPU/CPU fallback works; force_split option properly reduces memory; output geometry matches input

5. **Boolean Structure Operations**
   - Description: Verify union, intersection, margin operations on masks
   - Acceptance: Operations produce geometrically correct results; margin calculations use proper voxel spacing

6. **Dual Validation Process**
   - Description: Each reviewed component requires consensus with Pal
   - Acceptance: Both reviewers confirm correctness before moving to next component

### Edge Cases

1. **Empty Masks** - Radiomics/DVH must handle ROIs with zero voxels gracefully
2. **Single Rater ICC** - ICC computation must handle edge case of single measurement
3. **Dose Unit Mismatch** - DVH must detect and convert Gy/cGy appropriately
4. **Missing DICOM Tags** - Organization must handle incomplete DICOM files
5. **Out-of-Memory Segmentation** - Must verify force_split actually prevents OOM
6. **Parquet/Excel Staleness** - Aggregation must prefer newer files correctly
7. **ROI Number Mapping** - DVH must correctly map ROI numbers from RTSTRUCT to structures
8. **Structure Volumes Outside Dose Grid** - DVH must handle structures that extend beyond dose grid boundaries (use `calculate_full_volume=True`)

## Implementation Notes

### DO
- Review each pipeline stage in order (organize → segment → ... → aggregate)
- Consult Pal for scientific methodology validation at each stage
- Document issues found before fixing them
- Log improvement ideas to IDEAS.md (separate from bug fixes)
- Verify fixes don't introduce regressions
- Follow existing code patterns from `helpers.py` and `utils.py`
- Use the dual conda environment architecture (NumPy 1.x for radiomics, 2.x for segmentation)

### DON'T
- Skip stages or review out of order
- Fix issues without Pal consensus on the problem
- Merge improvement ideas into bug fixes (keep IDEAS.md separate)
- Modify infrastructure (Docker, CI/CD) unless scientifically necessary
- Add new dependencies without verifying NumPy compatibility
- Change the dual-environment architecture

## Development Environment

### Start Services

```bash
# Run pipeline with Snakemake
snakemake --cores all --use-conda --configfile config.yaml

# Docker execution
docker run --rm --gpus all \
  -v $(pwd)/Example_data:/data/input:ro \
  -v $(pwd)/Output_Test:/data/output:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores 16 --configfile /app/config.container.yaml

# Start Web UI (for testing)
cd webui && flask run --debug --port 5000
```

### Service URLs
- Web UI: http://localhost:5000
- Health Check: http://localhost:5000/health

### Required Environment Variables
- `PYTHONPATH`: Project root for module imports
- `RTPIPELINE_CONFIGFILE`: Path to config.yaml
- `OMP_NUM_THREADS`: Thread control (set to 1 in parallel contexts)
- `MKL_NUM_THREADS`: BLAS thread control
- `TOTALSEG_HOME_DIR`: TotalSegmentator model storage location (optional, defaults to `~/.totalsegmentator`)

## Success Criteria

The task is complete when:

1. [ ] All 12 pipeline stages have been reviewed for scientific correctness
2. [ ] Pal (gpt-5.2) has confirmed correctness for each reviewed stage
3. [ ] IDEAS.md has been created with improvement suggestions
4. [ ] All identified issues have been fixed
5. [ ] Fixed code has been re-reviewed and approved by both reviewers
6. [ ] No console errors during pipeline execution
7. [ ] Existing tests still pass (`pytest` runs clean)
8. [ ] Pipeline produces valid outputs on Example_data

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| DVH Calculation | `tests/test_dvh.py` | Metric accuracy within <2% of reference |
| Radiomics Extraction | `tests/test_radiomics.py` | IBSI compliance, feature reproducibility |
| ICC Computation | `tests/test_robustness.py` | No false ICC=1.0, proper complete-case handling |
| Boolean Operations | `tests/test_custom_structures.py` | Union/intersection/margin correctness |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| Full Pipeline | organize → aggregate | End-to-end execution completes |
| Radiomics Conda | main → rtpipeline-radiomics | Subprocess invocation works correctly |
| Segmentation GPU/CPU | segmentation module | Device fallback functions |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Example Data Run | 1. Load Example_data 2. Run snakemake 3. Check outputs | All output files generated in _RESULTS/ |
| Docker Execution | 1. docker-compose up 2. Wait for completion | Pipeline completes without errors |

### Database Verification (if applicable)
| Check | Query/Command | Expected |
|-------|---------------|----------|
| Output manifest exists | `cat Output/_COURSES/manifest.json` | Valid JSON with course entries |
| DVH metrics file | `head Output/_RESULTS/dvh_metrics.xlsx` | Non-empty Excel with expected columns |
| Radiomics output | `head Output/_RESULTS/radiomics_ct.xlsx` | Non-empty Excel with feature columns |

### QA Sign-off Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All E2E tests pass
- [ ] IDEAS.md created with logged suggestions
- [ ] No regressions in existing functionality
- [ ] Code follows established patterns
- [ ] No security vulnerabilities introduced
- [ ] Dual reviewer consensus achieved for all stages

## Review Process Protocol

### Stage Review Template

For each pipeline stage:

1. **Read and Understand**
   - Read the source code completely
   - Identify scientific methodology being implemented
   - Note any external standards that apply (IBSI, AAPM, RTOG)

2. **Validate Implementation**
   - Check parameter usage against documentation
   - Verify mathematical/statistical operations
   - Review error handling for edge cases

3. **Consult Pal**
   - Present findings to Pal (gpt-5.2)
   - Discuss any concerns about methodology
   - Reach consensus on correctness or identify issues

4. **Document & Fix**
   - Log issues for fixing
   - Log improvement ideas to IDEAS.md
   - Implement fixes

5. **Re-review**
   - Review fixed code
   - Confirm with Pal that issues are resolved
   - Mark stage as complete only with dual consensus

### Consensus States

- **APPROVED**: Both reviewers confirm correctness
- **ISSUES_IDENTIFIED**: Problems found, need fixing
- **UNDER_REVIEW**: Currently being evaluated
- **PENDING**: Not yet started
