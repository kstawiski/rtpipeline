# Scientific output inventories and durable state

`rtpipeline.output_inventory.is_operational_path` defines the shared exclusion
rule for scientific output inventories. The `_CACHE` namespace and reserved progress directories hold
journals, item caches, configuration records and rejected transactions. The
`.organize-progress.json` record, files ending in `.commit.json`, and hidden
directories ending in `.commit.json-generations` also hold operational state.
These files remain available to resume validators and are excluded from
scientific output sets and historical byte comparisons.

Scientific manifests, NIfTI sidecars, masks, tables and completion receipts
remain scientific outputs. A hidden name alone does not make a file operational.
Excluding operational state never normalizes or changes a scientific byte.

Organize artifact discovery, stage output discovery, DVH input manifests,
radiomics course bindings and aggregate source snapshots use this rule.
Historical output inventories use the same rule. Tests verify the reserved
paths, preserve scientific manifests and receipts, and check that changing
bookkeeping leaves scientific inventories unchanged while a changed scientific
artifact changes its digest.
