"""Lightweight provenance contract for physically placed clinical masks."""
import json
from pathlib import Path

ORIGINAL_SEGMENTATION_SCHEMA = "rtpipeline-original-segmentation-v2"
ORIGINAL_PROVENANCE_SCHEMA = "rtpipeline-original-segmentation-provenance-v2"


def require_current_original_mask(path: Path) -> None:
    """Reject legacy clinical exports before a consumer uses their coordinates."""
    if "Segmentation_Original" not in Path(path).parts:
        return
    try:
        metadata = json.loads((Path(path).parent / "metadata.json").read_text())
        provenance = json.loads((Path(path).parent / "provenance.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Original segmentation is stale: missing physical-space provenance") from exc
    if (not isinstance(metadata, dict) or not isinstance(provenance, dict)
            or metadata.get("schema") != ORIGINAL_SEGMENTATION_SCHEMA
            or metadata.get("geometry_status") != "exact_physical_mapping"
            or provenance.get("schema") != ORIGINAL_PROVENANCE_SCHEMA):
        raise ValueError("Original segmentation is stale: physical-space export required")

