# Re-export original clinical structures

Use explicit plaintext sources when an organized legacy tree cannot be rebuilt by
source discovery:

```sh
rtpipeline re-export-original \
  --rtstruct /sources/RS.dcm \
  --ct-series /sources/CT \
  --reference-nifti /sources/ct.nii.gz \
  --output-root /exports/rebuilt
```

The command verifies that the RTSTRUCT references the supplied CT series and
that the reference NIfTI has the same physical grid. It writes
`Segmentation_Original/<reference-stem>/metadata.json`, the masks, and current
provenance under the output root. Each mask filename begins with its ROI number;
the manifest preserves both the stored name and number. Duplicate names can be
exported separately because contours are selected by number.

An older input NIfTI sidecar is left unchanged. Reference provenance derived from
the explicit CT series and NIfTI is stored under `metadata/original_reexport/` in
the output root. Sealed `.age` inputs require separately supplied readable
plaintext paths; this command performs no decryption.

Current owned exports can be reused; `--overwrite` rebuilds them. Stale exports
identified by an RTpipeline schema or its exact historical 2.3.99 manifest
signature are retained in `Segmentation_Quarantine`. Other producers' directories
and loose files are left in place. An unfamiliar destination with the requested
reference stem causes an explicit refusal instead of replacement.

DVH and original export use ROI numbers. Name-keyed radiomics, robustness and
custom-source consumers refuse ambiguous identities rather than choosing a
same-name contour. An exact custom-source name retains precedence over normalized
aliases; an alias matching several sources records `CUSTOM_SOURCE_AMBIGUOUS_NAME`.

Existing label-based anchor and Dice-reference globs are resolved through the
manifest, so they continue to select the same masks after filename changes. A
folder containing undeclared files is preserved as a mixed destination and must
be resolved explicitly before it can be replaced.
