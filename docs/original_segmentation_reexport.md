# Re-export original clinical structures

`rtpipeline re-export-original` rebuilds the `Segmentation_Original` export of
one structure set from explicit plaintext sources. Use it when an organized
legacy tree cannot be rebuilt by source discovery:

```sh
rtpipeline re-export-original \
  --rtstruct /sources/RS.dcm \
  --ct-series /sources/CT \
  --reference-nifti /sources/ct.nii.gz \
  --output-root /exports/rebuilt
```

The command checks that the RTSTRUCT references the supplied CT series and that
the reference NIfTI has the same physical grid. It then writes the masks,
`metadata.json` and `provenance.json` to `Segmentation_Original/<reference-stem>/`
under the output root. Reference provenance derived from the CT series and the
NIfTI goes to `metadata/original_reexport/` under the output root; an older
sidecar next to the input NIfTI is left unchanged. The command does not decrypt:
for sealed `.age` inputs, supply readable plaintext copies.

## File names

Each ROI is exported from its own contours, selected by ROI number. When every
ROI name in the structure set is distinct, the masks keep their released file
names and the manifest keeps its name-only `v2` entries. Only ROIs in a colliding
group get file names of the form `<stem>__roi-<number>.nii.gz`, with `v3`
manifest entries that record the stored ROI name and number. Names collide when
they are identical, differ only in case or surrounding whitespace, or become the
same file name once sanitized.

## Reuse, overwrite and quarantine

A current export is reused. `--overwrite` rebuilds it and quarantines the export
it replaces. Anything in `Segmentation_Original` that the producer cannot verify
is moved to `Segmentation_Quarantine`: stale or unmanifested exports, other
producers' directories, loose files, an unfamiliar directory with the requested
reference stem, and directories that mix in undeclared files. Each moved entry
keeps its bytes and gets a `record.json` receipt with its original path, the
reason and the time.

An output root holds one reference stem. Running the command with a different
`--reference-nifti` stem in the same output root supersedes the earlier stem and
quarantines its export, even when that export was current. To keep exports for
several reference stems, use one output root for each.

## How other stages use the export

DVH and the original export identify ROIs by number. Radiomics and robustness
are keyed by ROI name, so they refuse the whole course when any source holds an
ambiguous ROI name, including the course's unambiguous ROIs, and record the
colliding ROI numbers and source as `RTSTRUCT_AMBIGUOUS_ROI_NAME`. For custom
structures, an exact source-name match takes precedence over normalized aliases;
an alias that matches several sources is refused as
`CUSTOM_SOURCE_AMBIGUOUS_NAME`.

Label-based anchor and Dice-reference globs in the configuration are resolved
through the manifest. A wildcard label selects every matching numbered mask; a
literal label that matches several ROIs is refused. A file-name reference that
also names a different stored ROI label is refused as ambiguous. The scoped
reader keeps stored names exactly, including leading whitespace, and consumers
that strip whitespace from their inventories check for ambiguity before they
normalize names.
