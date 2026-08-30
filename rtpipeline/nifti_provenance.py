"""One definition of NIfTI provenance, shared by every writer.

The course contract requires ``series_instance_uid``, ``sop_hash``, ``geometry``,
``nifti_geometry`` and ``nifti_sha256`` on a planning-CT sidecar. Two separate
conversion paths used to build that sidecar independently and only one recorded
the NIfTI-derived fields, so a course converted by the other path failed
contract validation with an incomplete sidecar. Both paths now call this.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

# Keys course_contract requires on a planning-CT NIfTI sidecar.
REQUIRED_SIDECAR_KEYS = (
    "series_instance_uid",
    "sop_hash",
    "geometry",
    "nifti_geometry",
    "nifti_sha256",
)


def nifti_geometry(nifti_path: Path) -> dict[str, Any]:
    """Return the NIfTI's geometry, or an empty mapping when it cannot be read."""
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(nifti_path))
        return {
            "size": [int(value) for value in image.GetSize()],
            "spacing": [float(value) for value in image.GetSpacing()],
            "origin": [float(value) for value in image.GetOrigin()],
            "direction": [float(value) for value in image.GetDirection()],
        }
    except Exception as exc:
        logger.warning("Could not record NIfTI geometry for %s: %s", nifti_path, exc)
        return {}


def annotate(
    metadata: MutableMapping[str, Any],
    nifti_path: Path,
    source_directory: Path,
    *,
    regenerated: bool,
    existing_sidecar: Optional[Mapping[str, Any]] = None,
    default_modality: str = "CT",
) -> MutableMapping[str, Any]:
    """Add the NIfTI-derived provenance a course contract requires.

    ``regenerated`` says whether the NIfTI content was just written. When it was
    not, the caller's previous ``generated_at``/``nifti_generated_at`` are carried
    forward, because those timestamps describe content and are load-bearing for
    segmentation mask reuse.
    """
    metadata.update(
        {
            "nifti_path": str(nifti_path),
            "source_directory": str(source_directory),
            "modality": metadata.get("modality") or default_modality,
            "nifti_sha256": hashlib.sha256(nifti_path.read_bytes()).hexdigest(),
        }
    )
    if regenerated:
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata["generated_at"] = stamp
        metadata["nifti_generated_at"] = stamp
    else:
        for key in ("generated_at", "nifti_generated_at"):
            if existing_sidecar and key in existing_sidecar:
                metadata[key] = existing_sidecar[key]
    metadata["nifti_geometry"] = nifti_geometry(nifti_path)
    return metadata


def sidecar_is_complete(metadata: Mapping[str, Any]) -> bool:
    """True when every contract-required provenance key is present."""
    return all(key in metadata for key in REQUIRED_SIDECAR_KEYS)
