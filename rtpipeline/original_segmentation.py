"""Lightweight provenance contract for physically placed clinical masks."""
import hashlib
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
    # A v2 label alone cannot keep an export current after CT repair. Read only
    # the small course contract here so workflow resume needs no imaging stack.
    original = next(parent for parent in Path(path).parents if parent.name == "Segmentation_Original")
    contract_path = original.parent / "metadata/case_metadata.json"
    if not contract_path.exists():
        return  # Standalone exports have no course-level input binding.
    try:
        contract = json.loads(contract_path.read_text())["course_contract"]
        current = contract["planning_ct"]["nifti_provenance"]
        expected = {
            "planning_ct_series_instance_uid": current["series_instance_uid"],
            "source_ct_sop_hash": current["sop_hash"],
            "source_nifti_geometry": current["nifti_geometry"],
            "source_nifti_sha256": current["nifti_sha256"],
        }
        rtstruct = original.parent / contract["authoritative_rtstruct"]["path"]
        expected["source_rtstruct_sha256"] = hashlib.sha256(rtstruct.read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Original segmentation is stale: current input provenance unavailable") from exc
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Original segmentation is stale: source inputs changed")

