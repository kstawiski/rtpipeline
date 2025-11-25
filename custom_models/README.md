# Custom Models (nnUNet) for rtpipeline

This directory contains configurations for custom nnUNet segmentation models. Model weights are **not included in the repository** due to their size and must be downloaded separately.

## Available Models

### 1. cardiac_STOPSTORM
**Description**: STOPSTORM nnUNetv2 ensemble for automated cardiac substructure segmentation

**Structures**: 15 cardiac structures including:
- Heart valves (Aortic, Mitral, Pulmonic, Tricuspid)
- Coronary arteries (LAD, LCX, LM, RCA)
- Cardiac chambers (L/R Atrium, L/R Ventricle)
- Great vessels (Aorta, Pulmonary artery, Inferior/Superior vena cava)

**Required Files**:
- `model603.zip` - Small structures network (~XX MB)
- `model605.zip` - Large structures network (~XX MB)

**Status**: ⚠️ **Weights not included** - Download required

### 2. HN_lymph_nodes
**Description**: nnUNet v1 ensemble for head & neck lymph node level segmentation

**Structures**: 19 lymph node levels (Level_1a, Level_1b_left/right, Level_2-8 bilateral)

**Required Files**:
- Weight archives or extracted directories for 3d_fullres and 2d networks
- Ensemble postprocessing JSON

**Status**: ⚠️ **Weights not included** - Download required

## Installing Model Weights

### Step 1: Download Weights

Model weights must be obtained separately. Contact your model provider or check project documentation for download links.

For **cardiac_STOPSTORM**:
```bash
# Example - replace with actual download location
wget https://example.com/models/model603.zip -O custom_models/cardiac_STOPSTORM/model603.zip
wget https://example.com/models/model605.zip -O custom_models/cardiac_STOPSTORM/model605.zip
```

For **HN_lymph_nodes**:
```bash
# Example - replace with actual download/extraction location
# Option 1: Extract weights to the model directory
tar -xzf hnln_weights.tar.gz -C custom_models/HN_lymph_nodes/

# Option 2: Place weight archives in the model directory
cp *.zip custom_models/HN_lymph_nodes/
```

### Step 2: Verify Installation

After downloading weights, verify they're properly placed:

```bash
# For cardiac_STOPSTORM
ls -lh custom_models/cardiac_STOPSTORM/
# Should show: custom_model.yaml, model603.zip, model605.zip, script.py

# For HN_lymph_nodes
ls -lh custom_models/HN_lymph_nodes/
# Should show: custom_model.yaml, and weight files/directories
```

### Step 3: Test Discovery

Run the pipeline in doctor mode to verify models are discovered:

```bash
rtpipeline doctor
```

Or run a test to check model discovery:

```bash
python -c "
from rtpipeline.custom_models import discover_custom_models
from pathlib import Path
models = discover_custom_models(Path('custom_models'))
print(f'Discovered {len(models)} model(s):')
for m in models:
    print(f'  - {m.name}: {len(m.networks)} network(s)')
"
```

**Expected output with weights**:
```
INFO: Discovered 2 custom model(s): cardiac_STOPSTORM, HN_lymph_nodes
```

**Expected output without weights**:
```
WARNING: Custom model 'cardiac_STOPSTORM' is configured but weights are missing:
603: model603.zip, 605: model605.zip. Model will be skipped.
WARNING: Custom model 'HN_lymph_nodes' is configured but weights are missing: ...
INFO: Discovered 0 custom model(s):
```

## Disabling Models

If you don't want to use custom models, you have several options:

### Option 1: Disable in Configuration
```yaml
# In config.yaml
custom_models:
  enabled: false
```

### Option 2: Skip Custom Models Stage
```bash
# When running pipeline, omit segmentation_custom stage
rtpipeline --stages organize,segmentation,dvh,radiomics --dicom-root /path/to/data
```

### Option 3: Remove Model Directories
```bash
# Remove unused model directories
rm -rf custom_models/HN_lymph_nodes
rm -rf custom_models/cardiac_STOPSTORM
```

## Adding New Models

To add a new custom model:

1. Create a new directory under `custom_models/`:
   ```bash
   mkdir custom_models/my_new_model
   ```

2. Create `custom_model.yaml` configuration (see examples in existing models)

3. Place model weights in the directory:
   - As ZIP archives (will be auto-extracted)
   - As pre-extracted directories
   - Or configure `source_directory` to point to weights location

4. Test discovery:
   ```bash
   python -c "from rtpipeline.custom_models import discover_custom_models; from pathlib import Path; print(discover_custom_models(Path('custom_models')))"
   ```

## Model Configuration Format

See `custom_model.yaml` files in existing models for examples. Key fields:

```yaml
name: model_name
description: Model description
nnunet:
  interface: nnunetv2  # or nnunetv1
  command: nnUNetv2_predict  # or nnUNet_predict for v1
  model: 3d_fullres
  folds: all  # or [0, 1, 2, 3, 4]

  networks:
    - id: "dataset_id"
      alias: network_name
      archive: weights.zip  # Path to weight archive
      dataset_directory: Dataset_name  # Where weights will be extracted
      label_order:  # List of structure names in order
        - Structure_1
        - Structure_2

  combine:
    ordered_networks: ["network_name"]  # Precedence order for overlapping structures
```

## Integration with Pipeline

Custom model outputs are automatically integrated:

### DVH Computation
Structures from custom models are included in DVH analysis with source labeled as `CustomModel:<model_name>`.

### Radiomics Extraction
Features are extracted from custom model structures with source labeled as `CustomModel:<model_name>`.

### Output Location
Custom model results are saved to:
```
Data_Snakemake/<patient>/<course>/Segmentation_CustomModels/<model_name>/
├── rtstruct.dcm           # DICOM RT Structure Set
├── manifest.json          # Execution metadata
└── <structure_name>.nii.gz  # Individual structure masks
```

## Troubleshooting

### "Model will be skipped" Warning
**Cause**: Weight files are missing
**Solution**: Download and place weight files as described above

### "No custom models discovered"
**Cause**: No valid models found
**Check**:
- `custom_models/` directory exists
- Each model has `custom_model.yaml`
- Weight files are present
- Configuration is valid YAML

### "FileNotFoundError: Unable to locate weights"
**Cause**: This error should no longer occur due to pre-validation
**Solution**: If you see this, it means validation was bypassed. Check weight files exist and are readable.

### Model runs but produces fewer structures than expected
**Check**: Look for warnings in logs like:
```
WARNING: Custom model 'X' did not produce N expected structure(s): ...
```

**Review**: `Segmentation_CustomModels/<model_name>/manifest.json` for:
- `expected_structures` - What the config says should be produced
- `produced_structures` - What was actually produced
- `missing_structures` - Structures that weren't generated

## Performance Notes

- **GPU Recommended**: nnUNet models run much faster on GPU
- **CPU Mode**: Set `CUDA_VISIBLE_DEVICES=-1` to force CPU (slower)
- **Parallelization**: Custom models run sequentially by default (GPU-bound)
  - Configure `custom_models.max_workers` in `config.yaml` if you have multiple GPUs
- **Weight Caching**:
  - Weights are extracted on first use
  - Set `custom_models.retain_weights: false` to auto-cleanup after execution
  - Set `custom_models.retain_weights: true` to keep extracted weights for faster reruns

## References

- [nnU-Net v1 Documentation](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)
- [nnU-Net v2 Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [STOPSTORM Cardiac Segmentation](https://github.com/nathandecaux/STOPSTORM)
- rtpipeline custom models implementation: `rtpipeline/custom_models.py`
