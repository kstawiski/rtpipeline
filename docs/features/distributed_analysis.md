# Distributed aggregate analysis

RTpipeline can export and combine strict cohort-level radiomics reliability
summaries. Each site runs its imaging pipeline locally and shares one
allow-listed aggregate table plus a validated manifest. The coordinator applies
the schema-v3 exact-match compatibility policy before combining packets.

This is distributed aggregate radiomics reliability analysis. It is not
federated learning, secure aggregation, differential privacy, an anonymity
mechanism, or a privacy guarantee. Local governance review may still be required
before a packet is shared.

## Schema v3 contract

Schema v3 is backward-incompatible with v2. A v2 packet cannot be validated or
aggregated as v3. The packet table contains one row per body region,
segmentation source, ROI, and radiomic feature, with these exact columns and
order:

```text
feature_name, roi_name, body_region, segmentation_source, n_subjects, n_raters,
n_subjects_cov, n_subjects_qcd,
icc, icc_ci_low, icc_ci_high, cov_percent, qcd,
classification, feature_family, image_type
```

The v3 contract additionally binds all of the following:

- the SHA-256 of a processing configuration normalized as canonical JSON;
- an exact source artifact kind, either `source_tree` or `container_image`, and
  its lowercase SHA-256;
- the actual RTpipeline package version used to create the packet;
- the expected feature/ROI inventory SHA-256, including `body_region`,
  `segmentation_source`, `roi_name`, `feature_name`, `feature_family`, and
  `image_type`;
- the explicit compatibility policy, which requires exact matches for the
  normalized configuration digest, artifact kind and digest, RTpipeline
  version, and feature/ROI inventory digest.

The exporter reads the actual software version from RTpipeline and refuses to
publish when it differs from the contract version. It recomputes the observed
inventory digest from the packet rows and requires it to equal the expected
inventory digest. Validation independently recomputes the schema, contract,
metrics, inventory, summaries, and content audit. The coordinator supplies the
expected values and rejects a mismatch by default.

The artifact digest is a provenance assertion supplied by the operator. For a
container, use the immutable image or exported-container digest, not a mutable
tag. For a source checkout, use the digest of a reproducibly archived source
tree. RTpipeline binds and compares that identity but does not inspect a remote
registry or prove how the supplied digest was obtained.

## ICC confidence intervals and classification

`icc` must always be finite. The two confidence interval fields must either both
be finite or both be empty. One missing limit is invalid. When both limits are
finite, classification uses `icc_ci_low`. When both are empty, and only then,
classification uses the point estimate in `icc`.

`Robust` requires the selected ICC statistic to be at least 0.90 and CoV to be
at most 10%. `Acceptable` requires ICC to be at least 0.75 and CoV to be at most
20%. Other complete rows are `Poor`.

`n_subjects_cov` and `n_subjects_qcd` make relative-dispersion eligibility
explicit. If no subject is eligible, the corresponding metric is empty and its
denominator is zero. If CoV is unavailable for any contributing subject, the
row must be `Not Evaluable`. It cannot be silently promoted.

## Freeze configuration and inventory

Create a JSON object containing the normalized processing choices that must be
identical at every site. JSON object key order and insignificant whitespace do
not affect its digest. Nonfinite JSON numbers are rejected.

```bash
rtpipeline federation config-digest --input processing-config.json \
  > processing-config-digest.json
PROCESSING_CONFIG_SHA256=$(jq -r .processing_config_sha256 processing-config-digest.json)
```

Create an expected inventory CSV or Parquet with exactly these columns and no
others:

```text
body_region,segmentation_source,roi_name,feature_name,feature_family,image_type
```

Every expected row must be present in the native robustness summary, and no
unexpected segmentation-source/structure/feature row may be present. A source
such as `Manual` and a source such as `AutoTS` remain distinct compatibility
identities even when the ROI and feature names match.

```bash
rtpipeline federation inventory-digest --input expected-inventory.csv \
  > inventory-digest.json
EXPECTED_INVENTORY_SHA256=$(jq -r .expected_feature_roi_inventory_sha256 inventory-digest.json)
```

Set the immutable software and artifact identity, then create the shared
contract once:

```bash
CONTRACT_ID=example-ntcv-icc-v3
MINIMUM_SUBJECTS=5
RTPIPELINE_VERSION=$(python -c 'import rtpipeline; print(rtpipeline.__version__)')
SOURCE_ARTIFACT_KIND=container_image
SOURCE_ARTIFACT_SHA256=<64-lowercase-hex-digest>

rtpipeline federation contract \
  --contract-id "$CONTRACT_ID" \
  --minimum-subjects "$MINIMUM_SUBJECTS" \
  --processing-config-sha256 "$PROCESSING_CONFIG_SHA256" \
  --source-artifact-kind "$SOURCE_ARTIFACT_KIND" \
  --source-artifact-sha256 "$SOURCE_ARTIFACT_SHA256" \
  --rtpipeline-version "$RTPIPELINE_VERSION" \
  --expected-inventory-sha256 "$EXPECTED_INVENTORY_SHA256" \
  > contract.json

CONTRACT_SHA256=$(jq -r .contract_sha256 contract.json)
```

Distribute the unchanged contract and the exact inventory file to every site.
Changing any bound value produces a different contract digest.

## Strict native-output adapter and site export

The recommended command consumes native cohort-summary columns emitted by
`radiomics_robustness`, including:

```text
structure, segmentation_source, feature_name, n_subjects, n_perturbations,
n_subjects_cov, n_subjects_qcd, icc, icc_ci95_low, icc_ci95_high,
cov_pct, qcd, robustness_label
```

`robustness_label` must use the exact native lowercase values `robust`,
`acceptable`, `poor`, or `not_evaluable`. The adapter does not guess aliases.
For example, `n_raters` is not accepted as a substitute for
`n_perturbations`, and `roi_name` is not accepted as a substitute for
`structure`. Body region, feature family, and image type come only from the
explicit expected inventory. They are not parsed from feature names.

Run the strict adapter and packet exporter together:

```bash
rtpipeline federation export-native \
  --input radiomics_robustness_summary.parquet \
  --inventory expected-inventory.csv \
  --output packet-node-a13f \
  --node-id node-a13f \
  --contract-id "$CONTRACT_ID" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --minimum-subjects "$MINIMUM_SUBJECTS" \
  --processing-config-sha256 "$PROCESSING_CONFIG_SHA256" \
  --source-artifact-kind "$SOURCE_ARTIFACT_KIND" \
  --source-artifact-sha256 "$SOURCE_ARTIFACT_SHA256" \
  --rtpipeline-version "$RTPIPELINE_VERSION" \
  --expected-inventory-sha256 "$EXPECTED_INVENTORY_SHA256"
```

For a table already shaped exactly as the packet schema, use `federation
export` with the same provenance and compatibility arguments.

The packet directory contains exactly:

```text
packet-node-a13f/
├── manifest.json
└── metrics.csv.gz
```

The exporter rejects extra packet columns, missing native columns, guessed
aliases, duplicate identities, inventory drift, nonfinite or semantically
invalid metrics, inconsistent denominators, small cells, identifier-like
content, paths, URIs, dates, DICOM UIDs, and unsupported classifications.
Deterministic gzip and 17-significant-digit formatting preserve float64
roundtrips.

Packet publication is staged in a sibling directory. Only after both files are
complete is the staged directory moved to the final path with `os.replace`.
An interrupted write therefore cannot expose a valid-looking partial packet.
With `--force`, staging failure leaves the old published packet unchanged.

## Validate and aggregate centrally

The coordinator repeats every contract-bound value rather than trusting values
from a site manifest:

```bash
rtpipeline federation aggregate \
  --packet packet-node-a13f \
  --packet packet-node-b72c \
  --output aggregate \
  --contract-id "$CONTRACT_ID" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --minimum-subjects "$MINIMUM_SUBJECTS" \
  --processing-config-sha256 "$PROCESSING_CONFIG_SHA256" \
  --source-artifact-kind "$SOURCE_ARTIFACT_KIND" \
  --source-artifact-sha256 "$SOURCE_ARTIFACT_SHA256" \
  --rtpipeline-version "$RTPIPELINE_VERSION" \
  --expected-inventory-sha256 "$EXPECTED_INVENTORY_SHA256"
```

Central aggregation performs node-tagged concatenation of validated cohort-level
rows. It computes no pooled cross-site reliability estimate, estimator, or
meta-analysis. Aggregation rejects version, configuration, artifact, or inventory
mismatch by default. It also rejects unexpected files, directories, symlinks, manifest or
audit keys, hashes, summary values, contract values, or metrics. The aggregate
manifest records the compatibility policy and binds both the manifest and
metrics SHA-256 for every accepted packet. Aggregate publication uses the same
staging and final-replace process as packet publication.

## What remains local

The packet contract has no field for raw DICOM objects, patient or course IDs,
patient-level feature values, local paths, dates, hostnames, or clinical
outcomes. This is a narrow data-minimization property of the implemented
contract. It does not establish anonymity, confidentiality, or a privacy
assurance, and it does not remove the need for an institution-specific
disclosure assessment.
