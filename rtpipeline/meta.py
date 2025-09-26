from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
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
from dicompylercore import dicomparser
from nested_lookup import nested_lookup
import pydicom

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _nested_get(ds: pydicom.dataset.FileDataset, tag: str) -> str:
    """Fetch a DICOM tag using dicompylercore JSON dict to mimic notebook behavior."""
    try:
        dd = dicomparser.DicomParser(ds).ds.to_json_dict()
        v = nested_lookup(tag, dd)
        if not v:
            return "NA"
        v0 = v[0]
        # Standard JSON schema uses {'vr':'XX','Value':[...]} per tag
        if isinstance(v0, dict) and 'Value' in v0 and v0['Value']:
            return str(v0['Value'][0])
        return str(v0)
    except Exception:
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


def _list_files(dicom_root: Path, pattern_prefix: str) -> List[Path]:
    out: List[Path] = []
    for base, _, files in os.walk(dicom_root):
        for fn in files:
            if fn.startswith(pattern_prefix) and fn.lower().endswith('.dcm'):
                out.append(Path(base) / fn)
    return out


def export_metadata(config: PipelineConfig) -> Dict[str, Path]:
    """Extract metadata for plans, doses, structures, fractions and write XLSX files."""
    paths = _export_dir(config.output_root)
    dicom_root = config.dicom_root

    # Collect RP
    rp_files = _list_files(dicom_root, 'RP')
    def _rp_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return None
        return {
            'file_path': str(p),
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
    with ThreadPoolExecutor(max_workers=config.effective_workers()) as ex:
        rp_rows = [r for r in ex.map(_rp_row, rp_files) if r]
    plans_df = pd.DataFrame(rp_rows)
    if not plans_df.empty:
        plans_df.to_excel(paths.plans_xlsx, index=False)

    # Collect RD
    rd_files = _list_files(dicom_root, 'RD')
    def _rd_row(p: Path) -> dict | None:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return None
        return {
            'file_path': str(p),
            'CT_series': _nested_get(ds, '0020000E'),
            'CT_study': _nested_get(ds, '0020000D'),
            'plan_id': _nested_get(ds, '00081155'),
            'patient_id': _nested_get(ds, '00100020'),
        }
    with ThreadPoolExecutor(max_workers=config.effective_workers()) as ex:
        rd_rows = [r for r in ex.map(_rd_row, rd_files) if r]
    doses_df = pd.DataFrame(rd_rows)
    if not doses_df.empty:
        doses_df.to_excel(paths.doses_xlsx, index=False)

    # Collect RS
    rs_files = _list_files(dicom_root, 'RS')
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
    with ThreadPoolExecutor(max_workers=config.effective_workers()) as ex:
        rs_rows = [r for r in ex.map(_rs_row, rs_files) if r]
    structs_df = pd.DataFrame(rs_rows)
    if not structs_df.empty:
        structs_df.to_excel(paths.structures_xlsx, index=False)

    # Collect RT* (fractions/treatment records)
    rt_files = _list_files(dicom_root, 'RT')
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
    with ThreadPoolExecutor(max_workers=config.effective_workers()) as ex:
        rt_rows = [r for r in ex.map(_rt_row, rt_files) if r]
    fractions_df = pd.DataFrame(rt_rows)
    if not fractions_df.empty:
        fractions_df.to_excel(paths.fractions_xlsx, index=False)

    # CT images index (PatientID, Study, Series, Instance)
    ct_files = _list_files(dicom_root, 'CT')
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
    with ThreadPoolExecutor(max_workers=config.effective_workers()) as ex:
        ct_rows = [r for r in ex.map(_ct_row, ct_files) if r]
    ct_df = pd.DataFrame(ct_rows)
    if not ct_df.empty:
        ct_df.to_excel(paths.ct_images_xlsx, index=False)

    # Merge metadata: RP<->RD by core key from filename; RS by patient
    meta_df = pd.DataFrame()
    if not plans_df.empty and not doses_df.empty:
        def core_key(fp: str) -> str | None:
            m = re.search(r"R[PD]\.(\d+)\.(.*?)\.dcm", os.path.basename(fp))
            return f"{m.group(1)}.{m.group(2)}" if m else None
        plans_df = plans_df.copy()
        doses_df = doses_df.copy()
        plans_df['core_key'] = plans_df['file_path'].map(core_key)
        doses_df['core_key'] = doses_df['file_path'].map(core_key)
        merged = plans_df.merge(doses_df, on='core_key', suffixes=("_plans", "_dosimetrics"), how='inner')
        meta_df = merged
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
