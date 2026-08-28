from __future__ import annotations

import logging
import os
import re
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
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import Tag

from .config import PipelineConfig
from .utils import run_tasks_with_adaptive_workers, _scoped_walk

logger = logging.getLogger(__name__)


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
    rows = run_tasks_with_adaptive_workers(
        "Metadata modality index",
        candidates,
        _read_verified_modality,
        max_workers=max_workers or 1,
        logger=logger,
    )
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
    for dose_index, dose in doses.iterrows():
        references = _reference_uids(
            dose.get("_referenced_plan_sop_uids")
            if "_referenced_plan_sop_uids" in doses.columns
            else None
        )
        if references:
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


def export_metadata(config: PipelineConfig) -> Dict[str, Path]:
    """Extract metadata for plans, doses, structures, fractions and write XLSX files."""
    paths = _export_dir(config.output_root)
    dicom_root = config.dicom_root
    # Scope discovery to the requested cohort when set (matches organize stage).
    scope_ids = list(getattr(config, "discover_patient_ids", []) or []) or None
    modality_files = _index_dicom_files_by_modality(
        dicom_root,
        scope_ids,
        max_workers=config.effective_workers(),
    )
    logger.info(
        "Metadata modality index found %s",
        ", ".join(
            f"{modality}={len(files)}"
            for modality, files in modality_files.items()
        ) or "no readable DICOM objects",
    )

    # Collect RTPLAN.
    rp_files = modality_files.get("RTPLAN", [])
    def _rp_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return None
        return {
            'file_path': str(p),
            '_sop_instance_uid': str(getattr(ds, 'SOPInstanceUID', '') or ''),
            'plan_name': _nested_get(ds, '300A0002'),
            'plan_date': _nested_get(ds, '300A0006'),
            'reference_dose_name': _nested_get(ds, '300A0016'),
            'approval': _nested_get(ds, '300E0002'),
            'CT_series': _nested_get(ds, '0020000E'),
            'CT_study': _nested_get(ds, '0020000D'),
            'patient_id': _nested_get(ds, '00100020'),
            'patient_dob': _nested_get(ds, '00100030'),
            'patient_gender': _nested_get(ds, '00100040'),
            'patient_pesel': _nested_get(ds, '00101000'),
        }
    rp_rows = [
        r
        for r in run_tasks_with_adaptive_workers(
            "Metadata (RP)",
            rp_files,
            _rp_row,
            max_workers=config.effective_workers(),
            logger=logger,
        )
        if r
    ]
    plans_df = pd.DataFrame(rp_rows)
    if rp_files and plans_df.empty:
        raise MetadataExportError(
            f"Discovered {len(rp_files)} RTPLAN object(s) by DICOM Modality but extracted zero plan rows; "
            "refusing to omit plans.xlsx"
        )
    if not plans_df.empty:
        _public_metadata_frame(plans_df).to_excel(paths.plans_xlsx, index=False)

    # Collect RTDOSE.
    rd_files = modality_files.get("RTDOSE", [])
    def _rd_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return None
        return {
            'file_path': str(p),
            '_referenced_plan_sop_uids': _referenced_sop_uids(ds, 'ReferencedRTPlanSequence'),
            'CT_series': _nested_get(ds, '0020000E'),
            'CT_study': _nested_get(ds, '0020000D'),
            'plan_id': _nested_get(ds, '00081155'),
            'patient_id': _nested_get(ds, '00100020'),
        }
    rd_rows = [
        r
        for r in run_tasks_with_adaptive_workers(
            "Metadata (RD)",
            rd_files,
            _rd_row,
            max_workers=config.effective_workers(),
            logger=logger,
        )
        if r
    ]
    doses_df = pd.DataFrame(rd_rows)
    if not doses_df.empty:
        _public_metadata_frame(doses_df).to_excel(paths.doses_xlsx, index=False)

    # Collect RTSTRUCT.
    rs_files = modality_files.get("RTSTRUCT", [])
    def _rs_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            structs = []
            if hasattr(ds, 'StructureSetROISequence'):
                for roi in ds.StructureSetROISequence:
                    nm = getattr(roi, 'ROIName', None)
                    if nm:
                        structs.append(str(nm))
        except Exception:
            return None
        return {
            'file_path': str(p),
            'CT_series': _nested_get(ds, '0020000E'),
            'CT_study': _nested_get(ds, '0020000D'),
            'approval': _nested_get(ds, '300E0002'),
            'patient_id': _nested_get(ds, '00100020'),
            'available_structures': ', '.join(structs) if structs else ''
        }
    rs_rows = [
        r
        for r in run_tasks_with_adaptive_workers(
            "Metadata (RS)",
            rs_files,
            _rs_row,
            max_workers=config.effective_workers(),
            logger=logger,
        )
        if r
    ]
    structs_df = pd.DataFrame(rs_rows)
    if not structs_df.empty:
        structs_df.to_excel(paths.structures_xlsx, index=False)

    # Collect RT treatment records for fraction delivery metadata.
    rt_files = modality_files.get("RTRECORD", [])
    def _rt_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return None
        return {
            'file_path': str(p),
            'fraction_id': _nested_get(ds, '00080018'),
            'date': _nested_get(ds, '30080024'),
            'time': _nested_get(ds, '30080025'),
            'fraction_number': _nested_get(ds, '30080022'),
            'verification_status': _nested_get(ds, '3008002C'),
            'termination_status': _nested_get(ds, '3008002A'),
            'delivery_time': _nested_get(ds, '3008003B'),
            'fluence_mode': _nested_get(ds, '30020052'),
            'plan_id': _nested_get(ds, '00081155'),
            'machine': _nested_get(ds, '300A00B2'),
            'patient_id': _nested_get(ds, '00100020'),
        }
    rt_rows = [
        r
        for r in run_tasks_with_adaptive_workers(
            "Metadata (RT)",
            rt_files,
            _rt_row,
            max_workers=config.effective_workers(),
            logger=logger,
        )
        if r
    ]
    fractions_df = pd.DataFrame(rt_rows)
    if not fractions_df.empty:
        fractions_df.to_excel(paths.fractions_xlsx, index=False)

    # CT images index (PatientID, Study, Series, Instance).
    ct_files = modality_files.get("CT", [])
    def _ct_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return None
        return {
            'original_path': str(p),
            'PatientID': _nested_get(ds, '00100020'),
            'CT_study': _nested_get(ds, '0020000D'),
            'CT_series': _nested_get(ds, '0020000E'),
            'SeriesNumber': _nested_get(ds, '00200011'),
            'InstanceNumber': _nested_get(ds, '00200013'),
        }
    ct_rows = [
        r
        for r in run_tasks_with_adaptive_workers(
            "Metadata (CT)",
            ct_files,
            _ct_row,
            max_workers=config.effective_workers(),
            logger=logger,
        )
        if r
    ]
    ct_df = pd.DataFrame(ct_rows)
    if not ct_df.empty:
        ct_df.to_excel(paths.ct_images_xlsx, index=False)

    # Merge metadata: RP<->RD by core key from filename; RS by patient
    meta_df = pd.DataFrame()
    if not plans_df.empty and not doses_df.empty:
        meta_df = _merge_plans_doses(plans_df, doses_df)
        if not structs_df.empty:
            meta_df = meta_df.merge(structs_df, left_on='patient_id_plans', right_on='patient_id', how='left', suffixes=("", "_structures"))
        meta_df.to_excel(paths.metadata_xlsx, index=False)

    logger.info("Exported metadata to %s", paths.root)
    return {
        'plans': paths.plans_xlsx,
        'doses': paths.doses_xlsx,
        'structures': paths.structures_xlsx,
        'fractions': paths.fractions_xlsx,
        'ct_images': paths.ct_images_xlsx,
        'metadata': paths.metadata_xlsx,
    }
