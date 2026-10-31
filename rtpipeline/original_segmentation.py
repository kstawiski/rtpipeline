"""Content and physical-space contract for clinical mask exports."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

ORIGINAL_SEGMENTATION_SCHEMA = "rtpipeline-original-segmentation-v2"
ORIGINAL_PROVENANCE_SCHEMA = "rtpipeline-original-segmentation-provenance-v2"


def _load_manifest(directory: Path) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("invalid manifest")
    return metadata


def _load_provenance(directory: Path) -> dict:
    provenance = json.loads((directory / "provenance.json").read_text())
    if not isinstance(provenance, dict) or provenance.get("schema") != ORIGINAL_PROVENANCE_SCHEMA:
        raise ValueError("missing physical-space provenance")
    return provenance


def _validate_mask_content(path: Path, entry: dict) -> None:
    if not isinstance(entry, dict):
        raise ValueError("invalid manifest")
    content = path.read_bytes()
    if (
        entry.get("sha256") != hashlib.sha256(content).hexdigest()
        or entry.get("size_bytes") != len(content)
    ):
        raise ValueError("mask content changed")


def _declared_export_masks(directory: Path, metadata: dict) -> dict[Path, dict]:
    structures = metadata.get("structures")
    if not isinstance(structures, list) or not structures:
        raise ValueError("export declares no masks")
    declared: dict[Path, dict] = {}
    for item in structures:
        if not isinstance(item, dict):
            raise ValueError("invalid manifest")
        relative = item.get("mask")
        if not isinstance(relative, str) or not relative:
            raise ValueError("invalid manifest mask path")
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError("manifest mask escapes export directory")
        mask_path = directory.parent / rel_path
        if (
            mask_path.parent != directory
            or mask_path.relative_to(directory.parent).as_posix() != relative
        ):
            raise ValueError("manifest mask does not belong to export directory")
        if mask_path in declared:
            raise ValueError("mask is not uniquely listed in manifest")
        declared[mask_path] = item
    return declared


def require_current_original_mask(path: Path) -> None:
    """Reject unlisted masks, changed bytes and exports with changed inputs."""
    path = Path(path)
    if "Segmentation_Original" not in path.parts:
        return
    try:
        metadata = _load_manifest(path.parent)
        if metadata.get("geometry_status") != "exact_physical_mapping":
            raise ValueError(str(metadata.get("geometry_status") or "physical-space export required"))
        if metadata.get("schema") != ORIGINAL_SEGMENTATION_SCHEMA:
            raise ValueError("physical-space export required")
        relative = path.relative_to(path.parent.parent).as_posix()
        structures = metadata.get("structures")
        if not isinstance(structures, list):
            raise ValueError("invalid manifest")
        entries = [item for item in structures if isinstance(item, dict) and item.get("mask") == relative]
        if len(entries) != 1:
            raise ValueError("mask is not uniquely listed in manifest")
        _validate_mask_content(path, entries[0])
        provenance = _load_provenance(path.parent)
        _require_current_sources(path.parent, metadata, provenance)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Original segmentation is stale: {exc}") from exc


def _require_current_sources(directory: Path, metadata: dict, provenance: dict) -> None:
    original = next((p for p in directory.parents if p.name == "Segmentation_Original"), None)
    if original is None:
        raise ValueError("export is outside Segmentation_Original")
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
    stem = source_nifti.name.removesuffix('.gz').removesuffix('.nii')
    publication = json.loads(source_nifti.with_name(stem + '.metadata.json').read_text())
    for source_key, export_key in (
        ("nifti_generated_at", "source_nifti_generated_at"),
        ("nifti_conversion", "source_nifti_conversion"),
    ):
        if provenance.get(export_key) != publication.get(source_key):
            raise ValueError("source CT publication or selection changed")
    if hashlib.sha256(source_nifti.read_bytes()).hexdigest() != expected["source_nifti_sha256"]:
        raise ValueError("source NIfTI content changed")


def quarantine_original_export(directory: Path, reason: str) -> Path:
    """Move an export outside consumer globs, retaining its bytes and a receipt."""
    directory = Path(directory)
    root = next((p for p in (directory, *directory.parents) if p.name == 'Segmentation_Original'), directory.parent)
    if directory == root:
        raise ValueError('Cannot quarantine the Segmentation_Original root')
    destination = root.parent / 'Segmentation_Quarantine' / uuid.uuid4().hex
    destination.mkdir(parents=True)
    record = {'path': directory.relative_to(root.parent).as_posix(),
              'reason': reason, 'time': datetime.now(timezone.utc).isoformat()}
    (destination / 'record.json').write_text(json.dumps(record, indent=2))
    directory.rename(destination / 'export')
    return destination


def require_current_original_export(directory: Path) -> None:
    """Accept current masks or a content-bound, maskless grid rejection."""
    directory = Path(directory)
    try:
        metadata = _load_manifest(directory)
        payload_entries = {
            entry for entry in directory.iterdir()
            if entry.name not in {"metadata.json", "provenance.json"}
        }
        geometry_status = metadata.get("geometry_status")
        if geometry_status == "MASK_GRID_NOT_EQUIVALENT" and not payload_entries:
            provenance = _load_provenance(directory)
            if (
                metadata.get("schema") != ORIGINAL_SEGMENTATION_SCHEMA
                or metadata.get("structures") != []
            ):
                raise ValueError("export contains no masks or current rejection record")
            _require_current_sources(directory, metadata, provenance)
            return
        if geometry_status != "exact_physical_mapping":
            raise ValueError(
                str(geometry_status or "export contains no masks or current rejection record")
            )
        if metadata.get("schema") != ORIGINAL_SEGMENTATION_SCHEMA:
            raise ValueError("physical-space export required")
        declared = _declared_export_masks(directory, metadata)
        missing = sorted(
            path.name for path in declared if path not in payload_entries or not path.is_file()
        )
        if missing:
            raise ValueError(f"missing declared mask(s): {', '.join(missing)}")
        extra = sorted(path.name for path in payload_entries if path not in declared)
        if extra:
            raise ValueError(f"unlisted file(s) in export: {', '.join(extra)}")
        provenance = _load_provenance(directory)
        _require_current_sources(directory, metadata, provenance)
        for mask_path, entry in declared.items():
            _validate_mask_content(mask_path, entry)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Original segmentation is stale: {exc}") from exc


def original_export_entries(root: Path) -> list[Path]:
    """Enumerate loose files explicitly, never the active root itself."""
    if not root.is_dir():
        return []
    directories = {p.parent for p in root.rglob('*') if p.is_file() and p.parent != root}
    loose_files = {p for p in root.iterdir() if p.is_file()}
    return sorted(directories | loose_files)


def quarantine_stale_original_exports(root: Path) -> None:
    """Producer boundary: retain current exports and current rejection records."""
    for entry in original_export_entries(Path(root)):
        if not entry.exists():
            continue
        try:
            if not entry.is_dir():
                raise ValueError('loose file outside an export directory')
            require_current_original_export(entry)
        except ValueError as exc:
            quarantine_original_export(entry, str(exc))
