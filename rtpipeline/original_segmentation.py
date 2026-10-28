"""Content and physical-space contract for clinical mask exports."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

ORIGINAL_SEGMENTATION_SCHEMA = "rtpipeline-original-segmentation-v2"
ORIGINAL_PROVENANCE_SCHEMA = "rtpipeline-original-segmentation-provenance-v2"


def require_current_original_mask(path: Path) -> None:
    """Reject unlisted masks, changed bytes and exports with changed inputs."""
    path = Path(path)
    if "Segmentation_Original" not in path.parts:
        return
    try:
        metadata = json.loads((path.parent / "metadata.json").read_text())
        if not isinstance(metadata, dict):
            raise ValueError("invalid manifest")
        if metadata.get("geometry_status") != "exact_physical_mapping":
            raise ValueError(str(metadata.get("geometry_status") or "physical-space export required"))
        if metadata.get("schema") != ORIGINAL_SEGMENTATION_SCHEMA:
            raise ValueError("physical-space export required")
        original = next(p for p in path.parents if p.name == "Segmentation_Original")
        relative = path.relative_to(path.parent.parent).as_posix()
        entries = [item for item in metadata.get("structures", []) if item.get("mask") == relative]
        if len(entries) != 1:
            raise ValueError("mask is not uniquely listed in manifest")
        entry = entries[0]
        if entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest() or entry.get("size_bytes") != path.stat().st_size:
            raise ValueError("mask content changed")
        provenance = json.loads((path.parent / "provenance.json").read_text())
        if provenance.get("schema") != ORIGINAL_PROVENANCE_SCHEMA:
            raise ValueError("missing physical-space provenance")
        source_nifti = Path(metadata["source_nifti"])
        source_rs = Path(metadata["source_rtstruct"])
        # Prefer course-relative copies during atomic localizer repair.
        contract_path = original.parent / "metadata/case_metadata.json"
        if contract_path.exists():
            contract = json.loads(contract_path.read_text())["course_contract"]
            current = contract["planning_ct"]["nifti_provenance"]
            source_rs = original.parent / contract["authoritative_rtstruct"]["path"]
            nifti_path = contract["planning_ct"].get("nifti_path") or contract["planning_ct"].get("nifti")
            if nifti_path:
                source_nifti = original.parent / nifti_path
        else:
            stem = source_nifti.name.removesuffix('.gz').removesuffix('.nii')
            current = json.loads(source_nifti.with_name(stem + '.metadata.json').read_text())
        expected = {
            "planning_ct_series_instance_uid": current["series_instance_uid"],
            "source_ct_sop_hash": current["sop_hash"],
            "source_nifti_geometry": current["nifti_geometry"],
            "source_nifti_sha256": current["nifti_sha256"],
            "source_rtstruct_sha256": hashlib.sha256(source_rs.read_bytes()).hexdigest(),
        }
        if not all(expected.values()) or any(provenance.get(k) != v for k, v in expected.items()):
            raise ValueError("source inputs changed")
        if hashlib.sha256(source_nifti.read_bytes()).hexdigest() != expected["source_nifti_sha256"]:
            raise ValueError("source NIfTI content changed")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Original segmentation is stale: {exc}") from exc


def quarantine_original_export(directory: Path, reason: str) -> Path:
    """Move an export outside consumer globs, retaining its bytes and a receipt."""
    directory = Path(directory)
    root = next((p for p in (directory, *directory.parents) if p.name == 'Segmentation_Original'), directory.parent)
    destination = root.parent / 'Segmentation_Quarantine' / uuid.uuid4().hex
    destination.mkdir(parents=True)
    record = {'path': directory.relative_to(root.parent).as_posix(),
              'reason': reason, 'time': datetime.now(timezone.utc).isoformat()}
    (destination / 'record.json').write_text(json.dumps(record, indent=2))
    directory.rename(destination / 'export')
    return destination


def quarantine_stale_original_exports(root: Path) -> None:
    """Producer boundary: no stale export remains in the active namespace."""
    root = Path(root)
    for directory in sorted({p.parent for p in root.rglob('*') if p.is_file()}):
        if not directory.exists():
            continue
        masks = list(directory.glob('*.nii*'))
        try:
            if not masks:
                metadata_path = directory / 'metadata.json'
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text())
                    if isinstance(metadata, dict) and metadata.get('geometry_status') == 'MASK_GRID_NOT_EQUIVALENT':
                        raise ValueError(f"MASK_GRID_NOT_EQUIVALENT: {metadata.get('detail', '')}")
                raise ValueError('export contains no masks')
            for mask in masks:
                require_current_original_mask(mask)
        except ValueError as exc:
            quarantine_original_export(directory, str(exc))
