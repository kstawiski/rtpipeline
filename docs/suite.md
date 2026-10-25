# RT research suite

RTpipeline and [RTsegmentator](https://github.com/kstawiski/rtsegmentator) cover the
path from treatment-planning exports to analysis tables.

| Tool | Role |
| --- | --- |
| **RTpipeline** | Organizes raw DICOM RT exports into validated patient courses and computes dose-volume, radiomics, robustness, multimodal and quality-control tables for whole cohorts. |
| **RTsegmentator** | Runs segmentation models on single DICOM series and returns RTSTRUCTs checked against the source geometry, through a web interface or a versioned API. Documentation: [kstawiski.github.io/rtsegmentator](https://kstawiski.github.io/rtsegmentator/). |

```text
 TPS export ──► RTpipeline organize ──► planning CT per course
                                             │
                                             ▼  upload
                                       RTsegmentator ──► RTSTRUCT + manifest
                                             │
                     verify checksums, series and ROI inventory
                                             ▼
               Segmentation_CustomModels/rtseg_<model>/rtstruct.dcm
                                             │
                     DVH · radiomics · robustness tables
```

Each tool runs on its own. RTpipeline runs TotalSegmentator and local nnU-Net models
without the portal; the portal serves clinicians and other software without the
pipeline. Together, a study can use any model in the portal's catalog for cohort-wide
dose and radiomics analysis without installing that model's runtime next to the
pipeline, and one GPU host can serve several projects.

## The interface between them

The tools exchange data through one documented contract:

- RTsegmentator writes `rtsegmentator_manifest.json` into every result ZIP: the source
  series, each model's outcome and exact ROI names, its licence and limitations, and
  the size and SHA-256 of every file. The format is a published JSON Schema
  (`rtsegmentator.result-manifest`, version 1.0).
- RTpipeline ships RTsegmentator's standard-library client unchanged
  (`rtpipeline/segmenters/rtsegmentator_client.py`) and imports a package only after
  verifying it and checking that every RTSTRUCT references the course's planning CT.

Configuration, checks and output layout are described in
[RTsegmentator models](features/rtsegmentator.md).

## Running both

`deploy/suite/` contains a Docker Compose file that starts the RTsegmentator portal next
to the RTpipeline Web UI on one host, with the pipeline configured to reach the portal by
its service name. The portal needs an NVIDIA GPU and licensed model assets; see its
deployment guide.
