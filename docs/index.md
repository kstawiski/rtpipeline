# rtpipeline Documentation

The repository bundles a Snakemake workflow and the `rtpipeline` Python package
for transforming raw DICOM-RT studies into analysable research datasets. Source
code is the ultimate reference for behaviour; this documentation summarizes the
current implementation so you can navigate the pipeline without reverse
engineering every module.

## Start Here
- [Quickstart](quickstart.md) – environment preparation and first run using
  Snakemake or the CLI.
- [Pipeline Architecture](pipeline_architecture.md) – detailed description of
  the Snakemake rules, produced artefacts, configuration knobs, and execution
  order.
- [CLI Reference](cli.md) – options exposed by the `rtpipeline` command-line
  interface and how they map to internal modules.

## Stage Guides
- [Segmentation](segmentation.md) – how TotalSegmentator is invoked, resume
  behaviour, and RS_auto creation.
- [DVH Metrics](dvh.md) – the metrics computed per ROI and where summaries are
  written.
- [Metadata Outputs](metadata.md) – structure of the JSON/XLSX exports created
  during organisation.
- [Viewers](viewers.md) – features of the generated HTML dashboards.

## Repository Layout
- `Snakefile` orchestrates the workflow described in the architecture guide.
- `config.yaml` supplies runtime paths, degree of parallelism, segmentation and
  radiomics settings, and the custom structure template.
- `rtpipeline/` contains the implementation modules used by both Snakemake and
  the CLI. Consult this package when you need the authoritative behaviour of a
  given stage.

All documentation in this folder is backed by the current code. If you add new
features or change the pipeline flow, update the relevant guide or remove the
section to keep the repo tidy.
