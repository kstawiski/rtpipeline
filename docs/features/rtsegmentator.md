# RTsegmentator models

[RTsegmentator](https://github.com/kstawiski/rtsegmentator) is a DICOM RT segmentation
portal that runs a large catalog of open models (TotalSegmentator tasks, nnU-Net and
MONAI bundles, nodal, prostate, brain and PET models) behind one API. RTpipeline can
use it as a segmentation source: each course's planning CT goes to the portal, and the
returned RTSTRUCTs are analysed like any other structure set.

Use it when a study needs structures that TotalSegmentator and the locally installed
custom models do not provide, or when one GPU host should serve several projects.

## Configure

```yaml
rtsegmentator:
  enabled: true
  url: "https://rtseg.example.org"
  models: ["total", "lung_nodules"]     # portal task names
  headers_env: "RTSEG_CLIENT_HEADERS"   # environment variable with proxy headers
  ca_bundle: null                       # CA bundle for a private certificate authority
  poll_seconds: 15
  job_timeout_seconds: 21600            # 0 waits without limit
```

List the task names a portal offers with
`python rtseg_client.py --url URL models --enabled` (the client is
`rtpipeline/segmenters/rtsegmentator_client.py`; it runs on its own with the
standard library) or in the portal's model catalog. RTsegmentator has no built-in
authentication; when it sits behind an authenticating proxy, export the headers before
running the pipeline, one `Name: value` per line:

```bash
export RTSEG_CLIENT_HEADERS="Authorization: Bearer $(cat ~/.rtseg-token)"
```

Never put credentials in the configuration file. Unknown keys and non-boolean
`enabled` values are configuration errors.

## What runs

The `segmentation_custom` stage runs after TotalSegmentator. For each course it:

1. reads the contracted planning CT and its SeriesInstanceUID from the course contract;
2. uploads that series through a resumable upload session and submits one job with the
   configured models;
3. waits for the job, downloads the result ZIP and verifies it against the packaged
   `rtsegmentator_manifest.json`;
4. writes each model's RTSTRUCT to `Segmentation_CustomModels/rtseg_<model>/`.

Courses are submitted one at a time because the portal serializes GPU work. Results are
reused when the course's planning CT (the set of SOP Instance UIDs), the portal URL and
the model are unchanged; `custom_models.force: true` resubmits. Any failed request,
failed job or rejected package fails the course's stage, as for every other stage.

Locally installed custom models still run when `custom_models.enabled` is true; with
`custom_models.enabled: false` and RTsegmentator enabled, only the portal models run.

## Import downloaded results

Without network access between the pipeline and the portal, download result ZIPs from
the portal and import them into an organized course:

```bash
rtpipeline rtsegmentator-import --course-dir Output/PATIENT/COURSE results/JOB_ID.zip
rtpipeline rtsegmentator-import --course-dir Output/PATIENT/COURSE --model total results/JOB_ID.zip
```

The command prints one JSON line per imported model and applies the checks below.

## Checks

Nothing is written for a package unless all of these hold:

- every file matches the manifest's size and SHA-256, and no file is missing or unlisted;
- the manifest's source series is the course's planning CT series;
- each RTSTRUCT references that series, and its ROI names are present, unique and equal
  to the names in the manifest;
- every requested model is in the package.

A model that ran and found nothing (`no_structures_found`) removes an earlier import of
that model, so stale structures are not analysed.

## Outputs

```text
Segmentation_CustomModels/rtseg_total/
├── rtstruct.dcm                   # the portal's RTSTRUCT, byte for byte
├── manifest.json                  # custom-model manifest with RTsegmentator provenance
└── rtsegmentator_manifest.json    # the package manifest as received
metadata/rtsegmentator_imports.json  # per-model outcome, job ID, portal URL and checksums
```

`manifest.json` follows the custom-model contract: `expected_structures` and
`produced_structures` are the ROI names the portal wrote, so radiomics extraction
requires exactly those ROIs. `catalog_structures` lists the task's full label set and
`absent_catalog_structures` the labels the model did not find in this scan. The
`rtsegmentator` block records the portal URL, job ID, RTsegmentator version, model
licence, limitations and reference, and the SHA-256 of the package and the RTSTRUCT.

In `dvh_metrics.xlsx`, `radiomics_ct.xlsx` and the robustness tables these structures
carry the segmentation source `CustomModel:rtseg_<model>`, separate from manual
contours (`Manual`), TotalSegmentator (`AutoTS_*`) and local custom models.

## Licences

Model licences differ. Some portal models allow non-commercial research only. The
`license`, `noncommercial` and `limitations` fields in each `manifest.json` come from
the portal's catalog; check them before using results beyond research, and cite each
model's publication alongside RTpipeline and RTsegmentator.
