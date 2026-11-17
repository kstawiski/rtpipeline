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
that the reference NIfTI has the same physical grid. It writes the masks,
`metadata.json` and `provenance.json` in
`Segmentation_Original/<reference-stem>/` under the output root.

Collision-free ROI names keep their released filenames and name-only `v2`
manifest entries. Only colliding groups receive filenames of the form
`<stem>__roi-<number>.nii.gz`; their `v3` manifest entries retain the stored ROI
name and number. This includes names that collide after filename sanitization,
case folding or removal of surrounding whitespace. Contours are selected by ROI
number, so duplicate names can be exported separately.

An older input NIfTI sidecar is left unchanged. Reference provenance derived from
the explicit CT series and NIfTI is stored under `metadata/original_reexport/` in
the output root. Sealed `.age` inputs require separately supplied readable
plaintext paths; this command performs no decryption.

Current exports can be reused; `--overwrite` rebuilds them and quarantines the
export it supersedes. At the producer boundary, unverifiable entries are moved
from `Segmentation_Original` into
`Segmentation_Quarantine`. This includes stale or unmanifested exports, foreign
producers' directories, loose files, an unfamiliar destination with the requested
reference stem, and mixed directories containing undeclared files. Each move
keeps the entry's bytes and writes a `record.json` receipt with its original path,
reason and time.

One output root retains one reference stem. Running the command with a different
`--reference-nifti` stem in the same output root supersedes the earlier stem:
the earlier export is quarantined, even if it was current. Use separate output
roots to retain exports for separate reference stems in active directories.

DVH and original export use ROI numbers. Name-keyed radiomics and robustness
refuse the whole course when a source contains an ambiguous ROI name, including
otherwise unambiguous ROIs in that course. The refusal records the colliding ROI
numbers and source as `RTSTRUCT_AMBIGUOUS_ROI_NAME`. An exact custom-source name
retains precedence over normalized aliases; an alias matching several sources
records `CUSTOM_SOURCE_AMBIGUOUS_NAME`.

Configured label-based anchor and Dice-reference globs resolve through the
manifest. A wildcard label selects the complete matching set of numbered masks;
a literal label matching several declarations is refused. A physical filename
reference that also denotes a different stored ROI label is refused as ambiguous.
The scoped reader preserves exact stored names, including leading whitespace.
Consumers whose inventories strip whitespace check ambiguity before normalizing.
