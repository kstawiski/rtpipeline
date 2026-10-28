"""Compare measurement payloads across revisions that add runtime provenance."""
from __future__ import annotations

import json

RASTERIZER_PROVENANCE_FIELDS = frozenset({
    "rasterizer_providers", "rasterizer_probe_error",
    "rasterizer", "rasterizer_backend", "rasterizer_opencv_version",
    "rasterizer_opencv_path", "rasterizer_rt_utils_version", "environment_fingerprint",
})


def measurement_table_and_schema(path):
    """Remove only runtime provenance; retain actual Arrow types and metadata."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = pd.read_parquet(path).drop(columns=list(RASTERIZER_PROVENANCE_FIELDS), errors="ignore")
    original = pq.read_schema(path)
    metadata = dict(original.metadata or {})
    if b"pandas" in metadata:
        pandas_metadata = json.loads(metadata[b"pandas"])
        pandas_metadata["columns"] = [
            column for column in pandas_metadata["columns"]
            if column["field_name"] not in RASTERIZER_PROVENANCE_FIELDS
        ]
        metadata[b"pandas"] = json.dumps(pandas_metadata, sort_keys=True).encode()
    schema = pa.schema([field for field in original if field.name not in RASTERIZER_PROVENANCE_FIELDS],
                       metadata=metadata)
    return frame, schema
