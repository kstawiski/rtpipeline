# Custom Models (nnUNet) Issues Analysis

## Overview
Analysis of custom model functionality in rtpipeline, examining configuration, execution, and integration with DVH and radiomics.

## Current Status

### Models Configured
1. **HN_lymph_nodes** - Head & neck lymph node segmentation (nnUNet v1, ensemble)
2. **cardiac_STOPSTORM** - Cardiac substructures (nnUNet v2, multi-network)

### Key Finding: **Model Weights Are Missing**

**Location**: `custom_models/HN_lymph_nodes/` and `custom_models/cardiac_STOPSTORM/`

**Problem**: Configuration files exist but model weights are absent:
- `custom_models/cardiac_STOPSTORM/model603.zip` - **MISSING**
- `custom_models/cardiac_STOPSTORM/model605.zip` - **MISSING**
- `custom_models/HN_lymph_nodes/` - No weight archives or source directories present

## Issues Identified

### 1. **Missing Model Weights** ❌ CRITICAL
**Status**: Models will fail at runtime with FileNotFoundError

**Evidence**:
```bash
$ ls custom_models/HN_lymph_nodes/
custom_model.yaml  # Config exists

$ ls custom_models/cardiac_STOPSTORM/
custom_model.yaml  script.py  # Weights missing
```

**Expected**:
```bash
$ ls custom_models/cardiac_STOPSTORM/
custom_model.yaml  model603.zip  model605.zip  script.py
```

**Impact**:
- Pipeline crashes when trying to run custom models
- Error: `FileNotFoundError: Unable to locate weights for network...`
- Location: `rtpipeline/custom_models.py:392-395`

**Error Message**:
```python
raise FileNotFoundError(
    f"Unable to locate weights for network {network.network_id}: "
    f"archive={archive_path}, source_dir={network.source_dir}"
)
```

### 2. **Error Handling Propagates to Course Level**
**Location**: `rtpipeline/cli.py:543-545`

**Problem**:
```python
def _custom(course):
    try:
        run_custom_models_for_course(cfg, course, available_models, force=force_custom)
    except Exception as exc:
        logger.warning("Custom segmentation failed for %s: %s", course.dirs.root, exc)
    return None  # Error swallowed, continues to next course
```

**Impact**:
- Errors are logged as warnings but not re-raised
- Pipeline continues silently without custom model outputs
- Users may not notice custom models didn't run
- DVH and radiomics will be missing custom model structures

**Good**: Prevents one failed model from stopping the entire pipeline
**Bad**: No clear indication in final outputs that custom models were skipped

### 3. **No Pre-Execution Validation**
**Location**: Model discovery happens at `rtpipeline/cli.py:520-537`

**Problem**:
- Models are discovered by checking `custom_model.yaml` exists
- Weight files are NOT validated until execution time
- Users only discover missing weights when pipeline runs

**Current Flow**:
```
1. discover_custom_models() -> Parses YAML ✓
2. run_custom_models_for_course() -> Tries to run
3. _prepare_network_weights() -> FAILS: FileNotFoundError ✗
```

**Better Flow**:
```
1. discover_custom_models() -> Parse YAML + validate weights
2. Log clear warnings about unavailable models
3. Only run models with valid weights
```

### 4. **Integration with DVH Works Correctly** ✓
**Location**: `rtpipeline/dvh.py:696-700`, `rtpipeline/custom_models.py:785-797`

**Analysis**:
```python
# DVH automatically includes custom model outputs
for model_name, model_course_dir in list_custom_model_outputs(course_dir):
    rs_model = model_course_dir / "rtstruct.dcm"
    if not rs_model.exists():
        continue
    source_label = f"CustomModel:{model_name}"
    process_struct(rs_model, source_label, rx_est)
```

**Status**: ✓ **Works as designed**
- DVH looks for `Segmentation_CustomModels/<model_name>/rtstruct.dcm`
- Includes structures labeled as `CustomModel:<model_name>`
- Fails gracefully if outputs don't exist

### 5. **Integration with Radiomics Works Correctly** ✓
**Location**: `rtpipeline/radiomics.py:442-444`

**Analysis**:
```python
# Radiomics automatically includes custom model outputs
for model_name, model_course_dir in list_custom_model_outputs(course_dir):
    rs_path = model_course_dir / "rtstruct.dcm"
    if not rs_path.exists():
        continue
    masks = _rtstruct_masks(course_dirs.dicom_ct, rs_path)
    for roi, mask in masks.items():
        tasks.append((f"CustomModel:{model_name}", roi, mask, mask_is_cropped(mask)))
```

**Status**: ✓ **Works as designed**
- Radiomics checks for custom model RTSTRUCT files
- Extracts features with source labeled as `CustomModel:<model_name>`
- Fails gracefully if outputs don't exist

### 6. **Sentinel File Caching Issue**
**Location**: `rtpipeline/custom_models.py:325-337`

**Problem**:
```python
sentinel_path = course.dirs.root / ".custom_models_done"
models_to_run = []
for model in models:
    if force or not _model_outputs_ready(course.dirs.root, model):
        models_to_run.append(model)

if not models_to_run and sentinel_path.exists():
    logger.info("Custom segmentation outputs already present for %s/%s; skipping", ...)
    return
```

**Issue**:
- If ANY model succeeds, sentinel file is created
- If one model fails but another succeeds, sentinel prevents retry
- User must manually delete `.custom_models_done` to retry failed models

**Impact**:
- Failed models won't be retried on subsequent runs
- Mixed success/failure scenarios leave incomplete outputs
- No per-model sentinels, only global flag

### 7. **Ensemble Model Complexity**
**Location**: HN_lymph_nodes model configuration

**Observation**:
The HN_lymph_nodes model uses:
- 2 networks (3d_fullres + 2d)
- Ensemble step combining both
- nnUNet v1 interface
- Postprocessing JSON file reference

**Potential Issues**:
- Postprocessing JSON path: `HNLNL_autosegmentation_trained_models/ensemble/postprocessing.json` - not validated to exist
- Ensemble command `nnUNet_ensemble` must be on PATH
- Both networks must complete before ensemble can run
- More complex failure modes than single-network models

### 8. **Model Weight Extraction and Cleanup**
**Location**: `rtpipeline/custom_models.py:370-408`, `rtpipeline/custom_models.py:750-782`

**How It Works**:
```python
# Step 1: Extract weights from ZIP to nnUNet_results
if archive_path and archive_path.exists() and archive_path.is_file():
    logger.info("Extracting weights for model %s network %s", model.name, network.network_id)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(results_root)

# Step 2: Run model prediction
...

# Step 3: Cleanup (if configured)
if not cfg.custom_models_retain_weights:
    _cleanup_model_cache(model)  # Removes nnUNet_results, nnUNet_raw, nnUNet_preprocessed
```

**Good**:
- Weights are extracted on-demand
- Optional cleanup saves disk space
- Cleanup only removes directories within model directory

**Potential Issue**:
- If extraction fails partway, corrupted state might persist
- No verification that extracted weights are complete/valid

### 9. **No Logging of Expected vs Actual Structures**
**Location**: `rtpipeline/custom_models.py:718-726`

**Problem**:
```python
manifest_entries, structure_paths = _write_structure_masks(
    combined_structures,
    structure_source,
    output_root,
)

if not manifest_entries:
    raise RuntimeError(f"No structures were produced for model {model.name}")
```

**Missing**:
- No comparison between expected structures (from config) and actually generated structures
- Model might produce fewer structures than expected without warning
- Example: Cardiac model expects 15 structures, produces 12 → no alert

**Recommendation**:
```python
expected = model.expected_structures()
produced = list(combined_structures.keys())
missing = set(expected) - set(produced)
if missing:
    logger.warning("Model %s missing expected structures: %s", model.name, ", ".join(missing))
```

### 10. **Command Availability Not Checked Upfront**
**Location**: `rtpipeline/custom_models.py:486-492`

**Problem**:
```python
executable = shlex.split(base_cmd)[0]
need_path_check = not (cfg.custom_models_conda_activate or cfg.conda_activate)
if need_path_check and _shutil.which(executable) is None:
    raise RuntimeError(
        f"nnU-Net command '{executable}' is not available on PATH. "
        "Ensure the custom models environment provides this executable..."
    )
```

**Issue**:
- Check only happens at prediction time, not during model discovery
- Each course will fail with same error if nnUNet not installed
- Users only find out after pipeline starts running

## Summary Table

| Issue | Severity | Impact | Current Behavior |
|-------|----------|--------|-----------------|
| Missing model weights | CRITICAL | Models don't run | FileNotFoundError at runtime |
| Error handling | MEDIUM | Silent failures | Logs warning, continues |
| No pre-validation | MEDIUM | Late failure detection | Discover at execution time |
| Sentinel file caching | LOW | Incomplete retries | Manual intervention needed |
| No structure verification | LOW | Incomplete outputs undetected | No warning |
| Command availability | LOW | Repeated failures | Check per execution |

## Integration Status

| Component | Status | Notes |
|-----------|--------|-------|
| DVH computation | ✓ Working | Correctly finds and processes custom model outputs |
| Radiomics extraction | ✓ Working | Correctly includes custom model structures |
| Output file structure | ✓ Working | Generates proper RTSTRUCT and NIfTI files |
| Manifest generation | ✓ Working | Creates detailed JSON manifest |

## Recommendations

### Immediate (Critical)
1. **Obtain model weights**
   - Download or create model603.zip and model605.zip for cardiac model
   - Provide weights for HN_lymph_nodes model
   - Document where to obtain these files

### High Priority
2. **Add weight validation during discovery**
   - Check archive files exist during `discover_custom_models()`
   - Log clear warnings about unavailable models
   - Skip models with missing weights instead of failing later

3. **Improve error reporting**
   - Create summary of which models succeeded/failed
   - Include in final pipeline output report
   - Don't silently swallow custom model errors

### Medium Priority
4. **Per-model sentinel files**
   - Use `.custom_model_<model_name>_done` instead of global sentinel
   - Allow selective retry of failed models
   - Track success/failure per model

5. **Structure verification**
   - Compare produced structures against expected list
   - Log warnings for missing structures
   - Include in manifest.json

### Low Priority
6. **Command validation upfront**
   - Check nnUNet commands available during initialization
   - Fail fast with clear error message
   - Don't start pipeline if prerequisites missing

## How to Fix Missing Weights

### Option 1: Provide Weight Files
```bash
# For cardiac_STOPSTORM model
# Place model weights in custom_models/cardiac_STOPSTORM/
cp /path/to/model603.zip custom_models/cardiac_STOPSTORM/
cp /path/to/model605.zip custom_models/cardiac_STOPSTORM/
```

### Option 2: Disable Models
```yaml
# In config.yaml, explicitly disable models with missing weights
custom_models:
  enabled: true
  models: []  # Empty list = don't run any models
  # Or selectively: models: ["cardiac_STOPSTORM"]  # Only this one
```

### Option 3: Remove Model Directories
```bash
# If you don't need these models
rm -rf custom_models/HN_lymph_nodes
rm -rf custom_models/cardiac_STOPSTORM
```

## Testing Recommendations

1. **Test with valid weights**: Obtain real model weights and verify execution
2. **Test with missing weights**: Confirm error messages are clear
3. **Test partial success**: Have one model succeed and another fail
4. **Test DVH integration**: Verify custom model structures appear in DVH results
5. **Test radiomics integration**: Verify features extracted from custom model structures
6. **Test sentinel behavior**: Run twice, ensure second run skips completed models
