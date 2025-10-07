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
nnunet:
  command: nnUNetv2_predict         # optional, defaults to CLI --nnunet-predict
  model: 3d_fullres                 # optional, defaults to 3d_fullres
  folds: all                        # optional, defaults to all
  env:                              # optional environment overrides
    nnUNet_results: ./nnUNet_results
    nnUNet_raw: ./nnUNet_raw
    nnUNet_preprocessed: ./nnUNet_preprocessed
  networks:
    - id: "603"                     # nnUNet dataset / task identifier
      archive: model603.zip         # zip containing the nnUNet weights
      dataset_directory: Dataset603_STST_small_full   # folder created by the archive
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
- `nnunet.command`: Command used to launch inference. Defaults to the pipeline
  flag `--nnunet-predict` (default `nnUNetv2_predict`).
- `nnunet.model` / `nnunet.folds`: Passed directly to `nnUNetv2_predict`.
- `nnunet.env`: Environment variables applied before running nnU-Net. Relative
  paths are resolved inside the model directory. `nnUNet_results`,
  `nnUNet_raw`, and `nnUNet_preprocessed` default to subdirectories in the
  model folder if not supplied.
- `nnunet.networks`: One or more nnUNet datasets to run. Each entry must provide
  an `id`, the `archive` that contains the weights, the `dataset_directory`
  created by that archive, and a `label_order` list describing the structures in
  the order produced by nnU-Net (label index `1` corresponds to the first name).
- `nnunet.combine.ordered_networks`: Optional list that specifies the precedence
  when multiple networks produce the same structure name. Later entries take
  priority. Defaults to the order defined under `networks`.

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
