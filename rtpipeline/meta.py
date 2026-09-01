from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
"""Compatibility shim to let dicompyler-core import against pydicom>=3.

dicompyler-core 0.5.x expects a legacy 'dicom' package with
- dicom.read_file
- dicom.dataset.Dataset / FileDataset
- dicom.tag.Tag

This shim provides those bindings to pydicom equivalents.
"""
try:
    import sys as _sys, types as _types, pydicom as _pyd
    # Base 'dicom' package
    _m = _sys.modules.get('dicom') or _types.ModuleType('dicom')
    _m.read_file = getattr(_pyd, 'dcmread', None)
    _sys.modules['dicom'] = _m
    # dicom.dataset
    _mds = _types.ModuleType('dicom.dataset')
    _mds.Dataset = _pyd.dataset.Dataset
    _mds.FileDataset = _pyd.dataset.FileDataset
    _sys.modules['dicom.dataset'] = _mds
    # dicom.tag
    _mtag = _types.ModuleType('dicom.tag')
    _mtag.Tag = _pyd.tag.Tag
    _sys.modules['dicom.tag'] = _mtag
    # dicom.uid (optional)
    _muid = _types.ModuleType('dicom.uid')
    _muid.UID = _pyd.uid.UID
    _sys.modules['dicom.uid'] = _muid
except Exception:
    pass
import pydicom
from pydicom.dataset import FileDataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import Tag

from .config import PipelineConfig
from .utils import DEFAULT_INDEX_WORKERS, _scoped_walk, parallel_map_files

logger = logging.getLogger(__name__)

_METADATA_CACHE_SCHEMA = "rtpipeline-metadata-export-cache-v1"
_METADATA_EXTRACTOR_SCHEMA = "rtpipeline-metadata-export-v2"
_METADATA_OUTPUT_NAMES = {
    "plans": "plans.xlsx",
    "structures": "structure_sets.xlsx",
    "doses": "dosimetrics.xlsx",
    "fractions": "fractions.xlsx",
    "metadata": "metadata.xlsx",
    "ct_images": "CT_images.xlsx",
}


def _format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, MultiValue) or isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "NA" if not parts else "\\".join(parts)
    text = str(value).strip()
    return text if text else "NA"


def _nested_get(ds: pydicom.dataset.FileDataset, tag: str) -> str:
    """Fetch a DICOM tag by walking the dataset, including sequences."""
    try:
        target = Tag(int(tag[:4], 16), int(tag[4:], 16))
    except Exception:
        return "NA"

    for element in ds.iterall():
        if element.tag != target:
            continue
        value = element.value
        if isinstance(value, pydicom.dataset.Dataset):
            continue
        if isinstance(value, Sequence):
            continue
        return _format_value(value)
    return "NA"


@dataclass
class ExportPaths:
    root: Path
    plans_xlsx: Path
    structures_xlsx: Path
    doses_xlsx: Path
    fractions_xlsx: Path
    metadata_xlsx: Path
    ct_images_xlsx: Path


def _export_dir(base: Path) -> ExportPaths:
    data = base / "Data"
    data.mkdir(parents=True, exist_ok=True)
    return ExportPaths(
        root=data,
        plans_xlsx=data / "plans.xlsx",
        structures_xlsx=data / "structure_sets.xlsx",
        doses_xlsx=data / "dosimetrics.xlsx",
        fractions_xlsx=data / "fractions.xlsx",
        metadata_xlsx=data / "metadata.xlsx",
        ct_images_xlsx=data / "CT_images.xlsx",
    )


class MetadataExportError(RuntimeError):
    """Raised when discovered DICOM objects cannot be represented in an export."""


@dataclass(frozen=True)
class MetadataSourceIdentity:
    """Cryptographic identity of the exact source-file namespace being exported."""

    digest: str
    file_count: int
    scope_digest: str


@dataclass(frozen=True)
class MetadataReadResult:
    """One source candidate classified and extracted from a single header read."""

    path: Path
    modality: str | None
    row: dict | None
    extraction_error: str | None = None


@dataclass(frozen=True)
class MetadataSourceFile:
    """One file's stable inventory metadata and metadata-export row."""

    path: Path
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    result: MetadataReadResult
    dataset: FileDataset | None


_MODALITY_TAG = Tag(0x0008, 0x0060)
_LEGACY_PREFIX_MODALITIES = {
    "RP": "RTPLAN",
    "RD": "RTDOSE",
    "RS": "RTSTRUCT",
    "RT": "RTRECORD",
    "CT": "CT",
}
_FILENAME_MODALITY_HINTS = (
    ("RTPLAN", "RTPLAN"),
    ("RTSTRUCT", "RTSTRUCT"),
    ("RTDOSE", "RTDOSE"),
    ("RTRECORD", "RTRECORD"),
    ("RP.", "RTPLAN"),
    ("RS.", "RTSTRUCT"),
    ("RD.", "RTDOSE"),
    ("RT.", "RTRECORD"),
    ("CT", "CT"),
)
_INTERNAL_METADATA_COLUMNS = {"_sop_instance_uid", "_referenced_plan_sop_uids"}


def _filename_modality_hint(path: Path) -> str | None:
    """Return a filename hint for diagnostics; callers must verify the DICOM tag."""
    name = path.name.upper()
    for prefix, modality in _FILENAME_MODALITY_HINTS:
        if name.startswith(prefix):
            return modality
    return None


def _read_verified_modality(path: Path) -> tuple[Path, str] | None:
    """Read only Modality and report filename hints that contradict the tag."""
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            specific_tags=[_MODALITY_TAG],
            force=True,
        )
    except Exception as exc:
        logger.warning("Could not read DICOM Modality from %s: %s", path, exc)
        return None
    modality = str(getattr(ds, "Modality", "") or "").strip().upper()
    if not modality:
        logger.warning("DICOM file has no Modality tag: %s", path)
        return None
    hint = _filename_modality_hint(path)
    if hint and hint != modality:
        logger.warning(
            "Filename suggested %s but DICOM Modality is %s; using the tag for %s",
            hint,
            modality,
            path,
        )
    return path, modality


def _index_dicom_files_by_modality(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]] = None,
    *,
    max_workers: int | None = None,
) -> Dict[str, List[Path]]:
    """Index every source file once from its verified DICOM Modality tag."""
    candidates: List[Path] = []
    for base, _, files in _scoped_walk(dicom_root, patient_ids):
        candidates.extend(Path(base) / name for name in files)
    candidates.sort(key=str)
    if not candidates:
        return {}
    workers = max_workers if max_workers is not None else DEFAULT_INDEX_WORKERS
    rows = parallel_map_files(candidates, _read_verified_modality, workers)
    indexed: Dict[str, List[Path]] = defaultdict(list)
    for row in rows:
        if row is None:
            continue
        path, modality = row
        indexed[modality].append(path)
    return {modality: paths for modality, paths in sorted(indexed.items())}


def _list_files(
    dicom_root: Path,
    pattern_prefix: str,
    patient_ids: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Compatibility wrapper that now interprets legacy prefixes as modalities."""
    requested = _LEGACY_PREFIX_MODALITIES.get(
        str(pattern_prefix).strip().upper(),
        str(pattern_prefix).strip().upper(),
    )
    return _index_dicom_files_by_modality(dicom_root, patient_ids).get(requested, [])


def _core_key_from_filename(fp: str) -> str | None:
    """Extract the shared RP/RD core key (numeric ID + description) from a
    plan/dose DICOM filename, or None if it doesn't match the expected
    ``R[PD].<id>.<description>.dcm`` pattern."""
    m = re.search(r"R[PD]\.(\d+)\.(.*?)\.dcm", os.path.basename(fp))
    return f"{m.group(1)}.{m.group(2)}" if m else None


def _reference_uids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, MultiValue)):
        values = value
    else:
        values = (value,)
    result: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text != "NA" and text not in result:
            result.append(text)
    return tuple(result)


def _merged_output_columns(plans_df: pd.DataFrame, doses_df: pd.DataFrame) -> List[str]:
    plan_columns = [
        column
        for column in plans_df.columns
        if column not in _INTERNAL_METADATA_COLUMNS and column != "core_key"
    ]
    dose_columns = [
        column
        for column in doses_df.columns
        if column not in _INTERNAL_METADATA_COLUMNS and column != "core_key"
    ]
    overlapping = set(plan_columns).intersection(dose_columns)
    columns = [
        f"{column}_plans" if column in overlapping else column
        for column in plan_columns
    ]
    columns.append("core_key")
    columns.extend(
        f"{column}_dosimetrics" if column in overlapping else column
        for column in dose_columns
    )
    return columns


def _merge_plans_doses(plans_df: pd.DataFrame, doses_df: pd.DataFrame) -> pd.DataFrame:
    """Associate RTPLAN and RTDOSE rows by DICOM reference, then ARIA fallback.

    An explicit RTDOSE reference is authoritative. The legacy filename core key
    remains a fallback only when a dose has no ReferencedRTPlanSequence, which
    preserves historical ARIA exports without guessing over contradictory DICOM.
    """
    plans = plans_df.copy().reset_index(drop=True)
    doses = doses_df.copy().reset_index(drop=True)
    plans["core_key"] = plans["file_path"].map(_core_key_from_filename)
    doses["core_key"] = doses["file_path"].map(_core_key_from_filename)
    output_columns = _merged_output_columns(plans, doses)

    plan_uid_rows: Dict[str, List[int]] = defaultdict(list)
    if "_sop_instance_uid" in plans.columns:
        for plan_index, value in plans["_sop_instance_uid"].items():
            uid = str(value or "").strip()
            if uid and uid != "NA":
                plan_uid_rows[uid].append(int(plan_index))

    pairs: List[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    unresolved_reference_count = 0
    unresolved_dose_count = 0
    for dose_index, dose in doses.iterrows():
        references = _reference_uids(
            dose.get("_referenced_plan_sop_uids")
            if "_referenced_plan_sop_uids" in doses.columns
            else None
        )
        if references:
            resolved_uids = {uid for uid in references if plan_uid_rows.get(uid)}
            unresolved_uids = [uid for uid in references if uid not in resolved_uids]
            if unresolved_uids:
                unresolved_reference_count += len(unresolved_uids)
                if not resolved_uids:
                    unresolved_dose_count += 1
                for uid in unresolved_uids:
                    logger.warning(
                        "RTDOSE %s references RTPLAN UID %s, but that plan is absent from the indexed plan table; "
                        "filename core-key fallback is refused",
                        dose.get("file_path", "<unknown dose>"),
                        uid,
                    )
            for uid in references:
                for plan_index in plan_uid_rows.get(uid, []):
                    pair = (plan_index, int(dose_index))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        pairs.append(pair)
            continue
        core_key = dose.get("core_key")
        if core_key is None or pd.isna(core_key):
            continue
        for plan_index in plans.index[plans["core_key"] == core_key].tolist():
            pair = (int(plan_index), int(dose_index))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)
    if unresolved_reference_count:
        logger.warning(
            "RTDOSE reference audit: %d unresolved RTPLAN UID reference(s) across %d dose object(s); "
            "explicit references remained authoritative and no filename fallback was used",
            unresolved_reference_count,
            unresolved_dose_count,
        )

    if not pairs:
        return pd.DataFrame(columns=output_columns)

    plan_columns = [
        column
        for column in plans.columns
        if column not in _INTERNAL_METADATA_COLUMNS and column != "core_key"
    ]
    dose_columns = [
        column
        for column in doses.columns
        if column not in _INTERNAL_METADATA_COLUMNS and column != "core_key"
    ]
    overlapping = set(plan_columns).intersection(dose_columns)
    records: List[dict] = []
    for plan_index, dose_index in pairs:
        plan = plans.iloc[plan_index]
        dose = doses.iloc[dose_index]
        record: dict = {}
        for column in plan_columns:
            output_name = f"{column}_plans" if column in overlapping else column
            record[output_name] = plan[column]
        plan_core = plan.get("core_key")
        dose_core = dose.get("core_key")
        record["core_key"] = (
            plan_core
            if plan_core is not None
            and not pd.isna(plan_core)
            and plan_core == dose_core
            else None
        )
        for column in dose_columns:
            output_name = f"{column}_dosimetrics" if column in overlapping else column
            record[output_name] = dose[column]
        records.append(record)
    return pd.DataFrame.from_records(records, columns=output_columns)


def _public_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[column for column in _INTERNAL_METADATA_COLUMNS if column in frame.columns],
        errors="ignore",
    )


def _referenced_sop_uids(ds: pydicom.dataset.FileDataset, sequence_name: str) -> tuple[str, ...]:
    values: List[str] = []
    for item in getattr(ds, sequence_name, []) or []:
        uid = str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
        if uid and uid not in values:
            values.append(uid)
    return tuple(values)


def _metadata_row(path: Path, modality: str, ds: pydicom.dataset.FileDataset) -> dict | None:
    """Build one export row from an already-read DICOM header."""
    if modality == "RTPLAN":
        return {
            "file_path": str(path),
            "_sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
            "plan_name": _nested_get(ds, "300A0002"),
            "plan_date": _nested_get(ds, "300A0006"),
            "reference_dose_name": _nested_get(ds, "300A0016"),
            "approval": _nested_get(ds, "300E0002"),
            "CT_series": _nested_get(ds, "0020000E"),
            "CT_study": _nested_get(ds, "0020000D"),
            "patient_id": _nested_get(ds, "00100020"),
            "patient_dob": _nested_get(ds, "00100030"),
            "patient_gender": _nested_get(ds, "00100040"),
            "patient_pesel": _nested_get(ds, "00101000"),
        }
    if modality == "RTDOSE":
        return {
            "file_path": str(path),
            "_referenced_plan_sop_uids": _referenced_sop_uids(
                ds, "ReferencedRTPlanSequence"
            ),
            "CT_series": _nested_get(ds, "0020000E"),
            "CT_study": _nested_get(ds, "0020000D"),
            "plan_id": _nested_get(ds, "00081155"),
            "patient_id": _nested_get(ds, "00100020"),
        }
    if modality == "RTSTRUCT":
        structures = [
            str(getattr(roi, "ROIName", ""))
            for roi in getattr(ds, "StructureSetROISequence", []) or []
            if getattr(roi, "ROIName", None)
        ]
        return {
            "file_path": str(path),
            "CT_series": _nested_get(ds, "0020000E"),
            "CT_study": _nested_get(ds, "0020000D"),
            "approval": _nested_get(ds, "300E0002"),
            "patient_id": _nested_get(ds, "00100020"),
            "available_structures": ", ".join(structures) if structures else "",
        }
    if modality == "RTRECORD":
        plan_ids = [
            str(getattr(ref, "ReferencedSOPInstanceUID", "") or "")
            for ref in getattr(ds, "ReferencedRTPlanSequence", []) or []
            if str(getattr(ref, "ReferencedSOPInstanceUID", "") or "")
        ]
        fraction_number = (
            getattr(ds, "CurrentFractionNumber", None)
            or getattr(ds, "ReferencedFractionNumber", None)
            or _nested_get(ds, "30080022")
        )
        return {
            "file_path": str(path),
            "fraction_id": str(
                getattr(ds, "SOPInstanceUID", "") or _nested_get(ds, "00080018")
            ),
            "date": getattr(ds, "TreatmentDate", None) or _nested_get(ds, "30080024"),
            "time": getattr(ds, "TreatmentTime", None) or _nested_get(ds, "30080025"),
            "fraction_number": fraction_number,
            "verification_status": _nested_get(ds, "3008002C"),
            "termination_status": _nested_get(ds, "3008002A"),
            "delivery_time": _nested_get(ds, "3008003B"),
            "fluence_mode": _nested_get(ds, "30020052"),
            "plan_id": plan_ids[0] if len(plan_ids) == 1 else None,
            "referenced_plan_ids": ";".join(plan_ids),
            "machine": _nested_get(ds, "300A00B2"),
            "patient_id": str(
                getattr(ds, "PatientID", "") or _nested_get(ds, "00100020")
            ),
        }
    if modality == "CT":
        return {
            "original_path": str(path),
            "PatientID": _nested_get(ds, "00100020"),
            "CT_study": _nested_get(ds, "0020000D"),
            "CT_series": _nested_get(ds, "0020000E"),
            "SeriesNumber": _nested_get(ds, "00200011"),
            "InstanceNumber": _nested_get(ds, "00200013"),
        }
    return None


def _metadata_result_from_dataset(
    path: Path,
    ds: FileDataset,
) -> MetadataReadResult:
    """Build a metadata result from a header already read by another phase."""
    modality = str(getattr(ds, "Modality", "") or "").strip().upper()
    if not modality:
        logger.warning("DICOM file has no Modality tag: %s", path)
        return MetadataReadResult(path=path, modality=None, row=None)
    hint = _filename_modality_hint(path)
    if hint and hint != modality:
        logger.warning(
            "Filename suggested %s but DICOM Modality is %s; using the tag for %s",
            hint,
            modality,
            path,
        )
    try:
        row = _metadata_row(path, modality, ds)
    except Exception as exc:
        return MetadataReadResult(path, modality, None, str(exc))
    return MetadataReadResult(path=path, modality=modality, row=row)


def _metadata_source_file(path: Path) -> MetadataSourceFile:
    """Read one source header and bind its row to a stable inventory tuple."""
    before = _inventory_stat(path)
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:
        verified = _read_verified_modality(path)
        dataset = None
        result = MetadataReadResult(
            path=path,
            modality=verified[1] if verified is not None else None,
            row=None,
            extraction_error=str(exc),
        )
    else:
        result = _metadata_result_from_dataset(path, dataset)
    after = _inventory_stat(path)
    if before != after:
        raise MetadataExportError(
            f"Metadata source changed while it was read: {path}"
        )
    _, size, mtime_ns, ctime_ns, device, inode = after
    return MetadataSourceFile(
        path=path,
        size=size,
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        device=device,
        inode=inode,
        result=result,
        dataset=dataset,
    )


def _read_metadata_file(path: Path) -> MetadataReadResult:
    """Classify and extract one candidate using one normal header read."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception as exc:
        # Preserve the old two-stage fail-closed behavior. If the detailed read
        # fails but a minimal forced read identifies a supported modality, the
        # caller must count the object and reject a wholly empty modality table.
        verified = _read_verified_modality(path)
        return MetadataReadResult(
            path=path,
            modality=verified[1] if verified is not None else None,
            row=None,
            extraction_error=str(exc),
        )
    return _metadata_result_from_dataset(path, ds)


def _source_candidate_paths(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]],
) -> List[Path]:
    def _raise_walk_error(error: OSError) -> None:
        raise error

    paths: List[Path] = []
    try:
        for base, _, files in _scoped_walk(
            dicom_root,
            patient_ids,
            onerror=_raise_walk_error,
        ):
            paths.extend(Path(base) / name for name in files)
    except OSError as exc:
        raise MetadataExportError(
            f"Could not enumerate the complete metadata source inventory under {dicom_root}: {exc}"
        ) from exc
    paths.sort(key=str)
    return paths


def _inventory_stat(path: Path) -> tuple[Path, int, int, int, int, int]:
    """Return a strong file-inventory record or raise on an unreadable candidate."""
    stat = path.stat()
    if not path.is_file():
        raise OSError(f"metadata source candidate is not a regular file: {path}")
    return (
        path,
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_dev),
        int(stat.st_ino),
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_inventory_identity_from_paths(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]],
    paths: List[Path],
    *,
    max_workers: int,
) -> tuple[MetadataSourceIdentity, List[Path]]:
    """Hash the exact path/size/time/inode inventory used by the exporter.

    This is deliberately not an mtime-only cache key. It binds the source root,
    requested cohort scope, lexical relative path, size, nanosecond mtime and
    ctime, device, and inode for every candidate. Any inability to enumerate or
    stat the full scope aborts metadata export rather than permitting reuse.
    """
    root = Path(dicom_root)
    try:
        stats = list(
            parallel_map_files(paths, _inventory_stat, max(1, max_workers))
        )
    except Exception as exc:
        raise MetadataExportError(
            f"Could not establish the complete metadata source inventory under {root}: {exc}"
        ) from exc

    scope = sorted({str(item) for item in (patient_ids or [])})
    scope_payload = {
        "patient_ids": scope,
        "follow_input_symlinks": os.environ.get(
            "RTPIPELINE_FOLLOW_INPUT_SYMLINKS", ""
        ).strip().lower(),
    }
    records = []
    for path, size, mtime_ns, ctime_ns, device, inode in stats:
        try:
            relative_path = os.path.relpath(path, root)
        except Exception as exc:
            raise MetadataExportError(
                f"Could not normalize metadata source path {path} against {root}: {exc}"
            ) from exc
        records.append(
            [relative_path, size, mtime_ns, ctime_ns, device, inode]
        )
    scope_digest = _canonical_sha256(scope_payload)
    digest = _canonical_sha256(
        {
            "schema": "rtpipeline-source-inventory-v1",
            "root": str(root.resolve(strict=False)),
            "scope_sha256": scope_digest,
            "files": records,
        }
    )
    return MetadataSourceIdentity(digest, len(records), scope_digest), paths


def _source_inventory_identity_from_files(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]],
    files: List[MetadataSourceFile],
) -> MetadataSourceIdentity:
    root = Path(dicom_root)
    scope_payload = {
        "patient_ids": sorted({str(item) for item in (patient_ids or [])}),
        "follow_input_symlinks": os.environ.get(
            "RTPIPELINE_FOLLOW_INPUT_SYMLINKS", ""
        ).strip().lower(),
    }
    records = [
        [
            os.path.relpath(item.path, root),
            item.size,
            item.mtime_ns,
            item.ctime_ns,
            item.device,
            item.inode,
        ]
        for item in files
    ]
    scope_digest = _canonical_sha256(scope_payload)
    digest = _canonical_sha256(
        {
            "schema": "rtpipeline-source-inventory-v1",
            "root": str(root.resolve(strict=False)),
            "scope_sha256": scope_digest,
            "files": records,
        }
    )
    return MetadataSourceIdentity(digest, len(records), scope_digest)


def _source_inventory_identity(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]],
    *,
    max_workers: int,
) -> tuple[MetadataSourceIdentity, List[Path]]:
    paths = _source_candidate_paths(dicom_root, patient_ids)
    return _source_inventory_identity_from_paths(
        dicom_root,
        patient_ids,
        paths,
        max_workers=max_workers,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_manifest_path(output_root: Path) -> Path:
    return Path(output_root) / "_CACHE" / "metadata_export.json"


def _cache_key(identity: MetadataSourceIdentity) -> str:
    return _canonical_sha256(
        {
            "extractor_schema": _METADATA_EXTRACTOR_SCHEMA,
            "source_inventory_sha256": identity.digest,
            "scope_sha256": identity.scope_digest,
        }
    )


def _expected_output_paths(paths: ExportPaths) -> Dict[str, Path]:
    return {
        "plans": paths.plans_xlsx,
        "structures": paths.structures_xlsx,
        "doses": paths.doses_xlsx,
        "fractions": paths.fractions_xlsx,
        "metadata": paths.metadata_xlsx,
        "ct_images": paths.ct_images_xlsx,
    }


def _load_cached_outputs(
    config: PipelineConfig,
    paths: ExportPaths,
    identity: MetadataSourceIdentity,
) -> Dict[str, Path] | None:
    manifest_path = _cache_manifest_path(config.output_root)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    manifest_digest = payload.pop("manifest_sha256", None)
    if not isinstance(manifest_digest, str) or manifest_digest != _canonical_sha256(payload):
        return None
    if payload.get("schema") != _METADATA_CACHE_SCHEMA:
        return None
    if payload.get("cache_key_sha256") != _cache_key(identity):
        return None
    source = payload.get("source")
    if not isinstance(source, dict) or source != {
        "inventory_sha256": identity.digest,
        "file_count": identity.file_count,
        "scope_sha256": identity.scope_digest,
    }:
        return None
    output_records = payload.get("outputs")
    if not isinstance(output_records, dict):
        return None
    expected = _expected_output_paths(paths)
    if set(output_records) != set(expected):
        return None
    for name, path in expected.items():
        record = output_records.get(name)
        if not isinstance(record, dict):
            return None
        state = record.get("state")
        if state == "absent":
            if path.exists():
                return None
            continue
        if state != "present" or not path.is_file():
            return None
        try:
            if _sha256_file(path) != record.get("sha256"):
                return None
        except OSError:
            return None
    logger.info(
        "Metadata export cache hit for %d source files; verified existing workbooks",
        identity.file_count,
    )
    return expected


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _publish_metadata_frames(
    paths: ExportPaths,
    frames: Dict[str, pd.DataFrame | None],
    config: PipelineConfig,
    identity: MetadataSourceIdentity,
    identity_paths: List[Path],
    modality_counts: Dict[str, dict],
) -> Dict[str, Path]:
    outputs = _expected_output_paths(paths)
    with tempfile.TemporaryDirectory(
        dir=str(paths.root), prefix=".metadata-export-"
    ) as temp_dir:
        staged: Dict[str, Path] = {}
        for name, frame in frames.items():
            if frame is None:
                continue
            staged_path = Path(temp_dir) / _METADATA_OUTPUT_NAMES[name]
            frame.to_excel(staged_path, index=False)
            # A parseable workbook is the minimum publication check.
            pd.read_excel(staged_path, engine="openpyxl")
            staged[name] = staged_path

        # The cache key must still describe the source after all reads complete.
        current_paths = _source_candidate_paths(
            config.dicom_root,
            list(getattr(config, "discover_patient_ids", []) or []) or None,
        )
        if current_paths != identity_paths:
            raise MetadataExportError(
                "Metadata source inventory changed during export; refusing to publish mixed-generation outputs"
            )
        current_identity, _ = _source_inventory_identity_from_paths(
            config.dicom_root,
            list(getattr(config, "discover_patient_ids", []) or []) or None,
            current_paths,
            max_workers=min(
                max(1, config.effective_workers()), DEFAULT_INDEX_WORKERS
            ),
        )
        if current_identity != identity:
            raise MetadataExportError(
                "Metadata source inventory changed during export; refusing to publish mixed-generation outputs"
            )

        for name, destination in outputs.items():
            staged_path = staged.get(name)
            if staged_path is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(staged_path, destination)

    output_records: Dict[str, dict] = {}
    for name, path in outputs.items():
        if path.is_file():
            output_records[name] = {
                "state": "present",
                "sha256": _sha256_file(path),
                "rows": int(len(frames[name])) if frames[name] is not None else 0,
            }
        else:
            output_records[name] = {"state": "absent"}
    manifest = {
        "schema": _METADATA_CACHE_SCHEMA,
        "extractor_schema": _METADATA_EXTRACTOR_SCHEMA,
        "cache_key_sha256": _cache_key(identity),
        "source": {
            "inventory_sha256": identity.digest,
            "file_count": identity.file_count,
            "scope_sha256": identity.scope_digest,
        },
        "modalities": modality_counts,
        "outputs": output_records,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_json_atomic(_cache_manifest_path(config.output_root), manifest)
    return outputs


def export_metadata(
    config: PipelineConfig,
    *,
    source_snapshot: dict | None = None,
) -> Dict[str, Path]:
    """Extract metadata in one header pass and reuse verified unchanged exports."""
    paths = _export_dir(config.output_root)
    scope_ids = list(getattr(config, "discover_patient_ids", []) or []) or None
    workers = min(max(1, config.effective_workers()), DEFAULT_INDEX_WORKERS)

    identity, candidates = _source_inventory_identity(
        config.dicom_root,
        scope_ids,
        max_workers=workers,
    )
    cached = _load_cached_outputs(config, paths, identity)
    if cached is not None:
        return cached

    snapshot_results = None
    if source_snapshot:
        if (
            source_snapshot.get("identity") == identity
            and source_snapshot.get("candidates") == candidates
            and isinstance(source_snapshot.get("results"), list)
        ):
            snapshot_results = source_snapshot["results"]
    if snapshot_results is not None:
        logger.info(
            "Metadata export cache miss; reusing %d headers from organize discovery",
            len(snapshot_results),
        )
        results = snapshot_results
    else:
        logger.info(
            "Metadata export cache miss; reading %d candidate header(s) with %d worker(s)",
            len(candidates),
            workers,
        )
        results = list(
            parallel_map_files(candidates, _read_metadata_file, workers)
        )
    supported = ("RTPLAN", "RTDOSE", "RTSTRUCT", "RTRECORD", "CT")
    discovered: Dict[str, int] = {modality: 0 for modality in supported}
    rows: Dict[str, List[dict]] = {modality: [] for modality in supported}
    extraction_failures: Dict[str, List[MetadataReadResult]] = {
        modality: [] for modality in supported
    }
    for result in results:
        modality = result.modality
        if modality not in discovered:
            continue
        discovered[modality] += 1
        if result.row is not None:
            rows[modality].append(result.row)
        elif result.extraction_error:
            extraction_failures[modality].append(result)

    failed = {
        modality: failures
        for modality, failures in extraction_failures.items()
        if failures
    }
    if failed:
        details = ", ".join(
            f"{modality}={len(failures)}"
            for modality, failures in failed.items()
        )
        examples = "; ".join(
            f"{modality}: {failures[0].path} ({failures[0].extraction_error})"
            for modality, failures in failed.items()
        )
        raise MetadataExportError(
            "Failed to extract metadata from supported DICOM object(s); refusing "
            f"to publish incomplete tables ({details}). Examples: {examples}"
        )

    logger.info(
        "Metadata header pass found %s",
        ", ".join(
            f"{modality}={discovered[modality]}"
            for modality in supported
            if discovered[modality]
        ) or "no supported DICOM objects",
    )
    for modality in supported:
        if discovered[modality] and not rows[modality]:
            label = {
                "RTPLAN": "plan",
                "RTDOSE": "dose",
                "RTSTRUCT": "structure",
                "RTRECORD": "fraction",
                "CT": "CT",
            }[modality]
            output = _METADATA_OUTPUT_NAMES[
                {
                    "RTPLAN": "plans",
                    "RTDOSE": "doses",
                    "RTSTRUCT": "structures",
                    "RTRECORD": "fractions",
                    "CT": "ct_images",
                }[modality]
            ]
            raise MetadataExportError(
                f"Discovered {discovered[modality]} {modality} object(s) by DICOM Modality "
                f"but extracted zero {label} rows; refusing to omit {output}"
            )

    plans_df = pd.DataFrame(rows["RTPLAN"])
    doses_df = pd.DataFrame(rows["RTDOSE"])
    structs_df = pd.DataFrame(rows["RTSTRUCT"])
    fractions_df = pd.DataFrame(rows["RTRECORD"])
    ct_df = pd.DataFrame(rows["CT"])

    meta_df = pd.DataFrame()
    if not plans_df.empty and not doses_df.empty:
        meta_df = _merge_plans_doses(plans_df, doses_df)
        if not structs_df.empty:
            meta_df = meta_df.merge(
                structs_df,
                left_on="patient_id_plans",
                right_on="patient_id",
                how="left",
                suffixes=("", "_structures"),
            )

    frames: Dict[str, pd.DataFrame | None] = {
        "plans": _public_metadata_frame(plans_df) if not plans_df.empty else None,
        "doses": _public_metadata_frame(doses_df) if not doses_df.empty else None,
        "structures": structs_df if not structs_df.empty else None,
        "fractions": fractions_df if not fractions_df.empty else None,
        "ct_images": ct_df if not ct_df.empty else None,
        "metadata": meta_df if not meta_df.empty else None,
    }
    modality_counts = {
        modality: {
            "discovered": discovered[modality],
            "rows": len(rows[modality]),
        }
        for modality in supported
    }
    exported = _publish_metadata_frames(
        paths,
        frames,
        config,
        identity,
        candidates,
        modality_counts,
    )
    logger.info("Exported metadata to %s", paths.root)
    return exported
