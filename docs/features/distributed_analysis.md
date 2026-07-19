# Distributed aggregate analysis

RTpipeline can export and combine a strict cohort-level radiomics reliability
packet. Each site runs its imaging pipeline locally and shares only one
allow-listed aggregate table plus a machine-validated manifest. The coordinator
rejects packets that do not match the same hash-bound contract.

This feature supports distributed measurement and validation. It is not
federated model training, secure aggregation, differential privacy, or a claim
that aggregate outputs are anonymous. Local governance review may still be
required before a packet is shared.

## Packet contract

The current contract contains one row per body-region, ROI, and radiomic
feature. Its exact columns are:

```text
feature_name, roi_name, body_region, n_subjects, n_raters,
n_subjects_cov, n_subjects_qcd,
icc, icc_ci_low, icc_ci_high, cov_percent, qcd,
classification, feature_family, image_type
```

`n_subjects_cov` and `n_subjects_qcd` make relative-dispersion eligibility
explicit. CoV and QCD are computed only for finite, strictly positive values
when the feature has been prespecified as suitable for relative dispersion; the
software cannot infer measurement-scale semantics from numeric values alone.
If no subject is eligible, the corresponding metric is empty and its
denominator is zero. If CoV is unavailable for any contributing subject, the
row must be classified `Not Evaluable`; it cannot be silently promoted to
`Robust`, `Acceptable`, or `Poor`. This schema is packet version 2 and is not
interchangeable with the older denominator-free packet schema.

The contract also binds and centrally validates the classification rule. It
uses the ICC 95% confidence-interval lower bound (falling back to the point ICC
only when a confidence interval is unavailable): `Robust` requires ICC ≥ 0.90
and CoV ≤ 10%; `Acceptable` requires ICC ≥ 0.75 and CoV ≤ 20%; all other
complete rows are `Poor`.

The contract digest binds the column schema, exact manifest and audit fields,
permitted packet filenames, minimum subject and rater counts, allowed robustness
classes, numerical constraints, and serialization format.

Create the shared contract once:

```bash
CONTRACT_ID=example-ntcv-icc-v1
MINIMUM_SUBJECTS=5

rtpipeline federation contract \
  --contract-id "$CONTRACT_ID" \
  --minimum-subjects "$MINIMUM_SUBJECTS" \
  > contract.json

CONTRACT_SHA256=$(jq -r .contract_sha256 contract.json)
```

Distribute the unchanged contract ID and digest to every site; the digest binds
the thresholds and lower-bound rule.

## Export at each site

Prepare a CSV or Parquet table with exactly the contracted columns, then run:

```bash
rtpipeline federation export \
  --input cohort_icc.parquet \
  --output packet-node-a13f \
  --node-id node-a13f \
  --contract-id "$CONTRACT_ID" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --minimum-subjects "$MINIMUM_SUBJECTS"
```

The packet directory contains exactly:

```text
packet-node-a13f/
├── manifest.json
└── metrics.csv.gz
```

The exporter uses deterministic gzip and 17-significant-digit float formatting,
so IEEE-754 float64 values round-trip without parser drift. It fails on extra
columns, duplicate feature identities, nonfinite or semantically invalid
metrics, inconsistent metric denominators, small cells, identifier-like
content, paths, URIs, dates, DICOM UIDs, or unsupported robustness classes.

## Validate and aggregate centrally

The coordinator supplies its own frozen contract values rather than trusting
thresholds declared by a node:

```bash
rtpipeline federation aggregate \
  --packet packet-node-a13f \
  --packet packet-node-b72c \
  --output aggregate \
  --contract-id "$CONTRACT_ID" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --minimum-subjects "$MINIMUM_SUBJECTS"
```

Validation rejects any unexpected file, directory, symlink, manifest key,
audit key, hash, summary value, contract value, or metric. The aggregate
manifest binds both the manifest and metrics SHA-256 for every accepted packet.

## What remains local

The packet contract has no field for raw DICOM objects, patient/course IDs,
patient-level feature values, local paths, dates, hostnames, or clinical
outcomes. This is a narrow data-minimization property of the implemented
contract. It does not establish legal anonymity or remove the need for an
institution-specific disclosure assessment.
