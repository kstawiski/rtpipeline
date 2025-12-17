# Custom Segmentation Models

The pipeline can execute additional nnU-Net v2 models that live under `custom_models/`.
Each model resides in its own directory and is configured by a `custom_model.yaml`
file. During the Snakemake stage `segmentation_custom_models` the pipeline scans
this directory, extracts any bundled weights, runs the predictors, and writes the
outputs to the patient folder.

## Directory Layout

```
custom_models/
  <model_name>/
    custom_model.yaml
    modelXYZ.zip        # one or more nnUNet weight archives
    ...                 # optional helper scripts or notes
```

All paths in the YAML file are resolved relative to the model directory. Weight
archives are unpacked on first use into the configured `nnUNet_results` folder.

## YAML Schema

Minimal example:

```yaml
name: my_model
description: Optional free-form text.
# Body region requirements for QC gating (optional)
required_body_regions:
  - THORAX                            # Valid regions: HEAD_NECK, THORAX, ABDOMEN, PELVIS
min_region_confidence: 0.6            # optional, defaults to 0.5
nnunet:
  command: nnUNetv2_predict         # optional, defaults to CLI --nnunet-predict
  model: 3d_fullres                 # optional, defaults to 3d_fullres
  folds: all                        # optional, defaults to all
  interface: nnunetv1|nnunetv2      # optional, defaults to nnunetv2
  env:                              # optional environment overrides
    nnUNet_results: ./nnUNet_results
    nnUNet_raw: ./nnUNet_raw
    nnUNet_preprocessed: ./nnUNet_preprocessed
  networks:
    - id: "603"                     # nnUNet dataset/task identifier
      alias: 603                    # optional shorthand used elsewhere in the config
      archive: model603.zip         # zip containing the nnUNet weights
      source_directory: null        # optional pre-unpacked weight location (relative to model)
      dataset_directory: Dataset603_STST_small_full   # folder created or copied into nnUNet_results
      architecture: 3d_fullres      # optional override for CLI (v1) / model selection (v2)
      trainer: nnUNetTrainerV2      # optional trainer name (v1)
      plans: nnUNetPlansv2.1        # optional plan identifier (v1)
      label_order:                  # list of class labels in order of nnUNet output IDs
        - Structure_A
        - Structure_B
    - id: "605"
      archive: model605.zip
      dataset_directory: Dataset605_STSTplusLung_large_full
      label_order:
        - Structure_C
        - Structure_D
  combine:
    ordered_networks: ["603", "605"]   # precedence when merging structures
```

**Key fields**

- `name`: Human-readable identifier. Also used for the output directory name.
- `description`: Optional documentation string shown in manifests.
- `required_body_regions`: Optional list of body regions that must be present in the
  CT scan for this model to run. Valid values: `HEAD_NECK`, `THORAX`, `ABDOMEN`, `PELVIS`.
  If not specified, the model runs on any CT scan. See [Body Region QC](#body-region-qc) below.
- `min_region_confidence`: Minimum confidence score (0.0–1.0) required for each region.
  Defaults to 0.5. Higher values require stronger evidence of region presence.
- `nnunet.command`: Command used to launch inference. Defaults to the pipeline
  flag `--nnunet-predict` (default `nnUNetv2_predict`).
- `nnunet.interface`: Choose between `nnunetv2` (default) and `nnunetv1`. v1 enables
  legacy commands such as `nnUNet_predict`/`nnUNet_ensemble`.
- `nnunet.model` / `nnunet.folds`: Passed directly to `nnUNetv2_predict`.
- `nnunet.env`: Environment variables applied before running nnU-Net. Relative
  paths are resolved inside the model directory. `nnUNet_results`,
  `nnUNet_raw`, and `nnUNet_preprocessed` default to subdirectories in the
  model folder if not supplied.
- `nnunet.networks`: One or more nnUNet datasets to run. Each entry must provide
  an `id`, the desired `alias`, the weights location (`archive` for ZIPs or
  `source_directory` for pre-unpacked models), the `dataset_directory` to create
  under `nnUNet_results`, and a `label_order` list describing the structures in
  the order produced by nnU-Net (label index `1` corresponds to the first name).
- `nnunet.combine.ordered_networks`: Optional list that specifies the precedence
  when multiple networks produce the same structure name. Later entries take
  priority. Defaults to the order defined under `networks`.
- `nnunet.ensemble`: Optional block describing an `nnUNet_ensemble` call. Specify
  a command, the list of input network aliases, an optional
  `postprocessing_json`, and (when it differs from individual networks) a
  `label_order` that enumerates the expected structures in the final combined
  output.

## Generated Outputs

For each patient/course pair the pipeline produces:

```
<output_root>/<patient>/<course>/Segmentation_CustomModels/<model_name>/
  <structure>.nii.gz
  manifest.json
  rtstruct.dcm
  rtstruct_manifest.json
```

Structure filenames are sanitized versions of the labels provided in the YAML.
The manifest records which nnUNet network produced each structure, the command
that was run, and the path to the source CT NIfTI. No duplicate copies are kept
at the patient root—each course owns its own custom-model directory alongside
`Segmentation_TotalSegmentator/`.

To add another model, replicate the directory layout, adjust the YAML to point
to the proper nnUNet archives and label names, and re-run the Snakemake rule
`segmentation_custom_models` (e.g. `snakemake segmentation_custom_models`). If
you change a model configuration, re-run with `--force-custom-models` or delete
the per-course `.custom_models_done` sentinel to regenerate the outputs.

## Pipeline Integration
- The Snakemake stage `segmentation_custom_models` runs after TotalSegmentator
  segmentation and before DVH/radiomics. Downstream rules (`dvh_course`,
  `radiomics_course`, and `aggregate_results`) now require the
  `.custom_models_done` sentinel so custom-model RTSTRUCTs are always consumed.
- DVH rows store `Segmentation_Source = CustomModel:<model_name>`; radiomics
  rows expose the same label in `segmentation_source`. This allows direct
  comparisons with manual (`Manual`), TotalSegmentator (`AutoRTS`), or Boolean
  merged (`Merged`) contours in the aggregated workbooks.
- Quality-control JSON and Excel reports include any `__partial` suffixes or
  cropping flags associated with custom-model structures, mirroring the
  behaviour of other segmentation sources.

## Weight Caching and Cleanup
Inference requires unpacking the nnUNet archives into
`custom_models/<model>/nnUNet_results` (and accompanying `nnUNet_raw` and
`nnUNet_preprocessed` folders). By default these caches are retained so later
courses reuse the weights without re-extracting ~500 MB archives. Set
`custom_models.retain_weights: false` in `config.yaml` (or launch the CLI with
`--purge-custom-model-weights`) to delete these directories after each
successful run. The original `model*.zip` files remain in place so the next
invocation can restore the caches if needed; course outputs under
`Segmentation_CustomModels/<model>/` are unaffected by this cleanup option.

## Body Region QC

The pipeline performs automatic body region detection after TotalSegmentator's
"total" task completes. This QC step uses anatomical anchor structures (vertebrae,
organs) to determine which body regions are present in the CT scan:

- **HEAD_NECK**: Cervical spine (C1–C7), brain, skull
- **THORAX**: Thoracic spine (T1–T12), lungs, heart
- **ABDOMEN**: Lumbar spine (L1–L5), liver, spleen, kidneys
- **PELVIS**: Sacrum, hip bones, femurs

Results are saved to `<course>/qc_reports/body_regions.json` with:
- Boolean flags: `CONTAINS_HEAD_NECK`, `CONTAINS_THORAX`, `CONTAINS_ABDOMEN`, `CONTAINS_PELVIS`
- Confidence scores (0.0–1.0) per region based on vertebrae count and organ volumes
- Model eligibility status for each configured model
- **Contrast phase** detection (native, arterial_early, arterial_late, portal_venous)
- **Image modality** detection (CT or MR)

### Phase and Modality Detection

The pipeline uses TotalSegmentator's auxiliary tools to detect:

1. **Contrast Phase** (`totalseg_get_phase`): Classifies CT images by contrast phase:
   - `native`: Non-contrast CT
   - `arterial_early`: Early arterial phase
   - `arterial_late`: Late arterial phase
   - `portal_venous`: Portal venous phase

2. **Image Modality** (`totalseg_get_modality`): Detects whether the image is CT or MR

These values are included in:
- The `body_regions.json` QC report
- Aggregated DVH results (`contrast_phase`, `image_modality`, `body_regions` columns)
- Aggregated radiomics results (same columns)
- Aggregated QC reports (with detailed body region confidence scores)

**Requirements**: These features require `xgboost` to be installed (included in the
`rtpipeline.yaml` conda environment).

### Model Gating

When `required_body_regions` is specified in a custom model's YAML, the pipeline
checks eligibility before running inference. If the required region is missing:

- **Default behavior** (`body_region_qc_block_missing: true`): The model is skipped
  with a warning logged. The sentinel file records which models were skipped.
- **Warning-only** (`body_region_qc_block_missing: false`): A warning is logged
  but inference proceeds anyway.

This prevents running cardiac segmentation models on head CTs, head/neck models on
pelvis CTs, etc., avoiding wasted GPU time and invalid outputs.

### Configuration

Model region requirements can also be set globally in `config.yaml`:

```yaml
model_region_requirements:
  cardiac_STOPSTORM:
    required_regions: [THORAX]
    min_confidence: 0.6
  heartchambers_highres:
    required_regions: [THORAX]
    min_confidence: 0.5
  head_neck_oar:
    required_regions: [HEAD_NECK]
    min_confidence: 0.5

body_region_qc_block_missing: true  # true = block, false = warn only
```

Requirements in `custom_model.yaml` take precedence over config.yaml for that model.
Models without any requirements (either in YAML or config) run on all CT scans.
