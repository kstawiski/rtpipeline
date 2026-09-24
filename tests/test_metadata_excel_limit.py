"""Metadata tables that exceed an Excel sheet are published as Parquet.

All inputs are synthetic DICOM written under pytest's tmp_path. The Excel limit
is monkeypatched to a few rows so no large workbook is written.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydicom.uid import RTBeamsTreatmentRecordStorage, RTPlanStorage, generate_uid
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline import meta
from test_aggregate_course_contract import _aggregate_functions
from test_metadata_modality_hardening import (
    _config,
    _file_dataset,
    _write,
    _write_ct,
    _write_export,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "923233e"
EXPORT_NAMES = {
    "plan": "RTPLAN_1.dcm",
    "dose": "RTDOSE_1.dcm",
    "struct": "RTSTRUCT_1.dcm",
    "record": "RTRECORD_1.dcm",
    "ct": "CT_1.dcm",
}


def _write_numbered_record(path: Path, plan_uid: str, fraction: int | None) -> Path:
    ds = _file_dataset(path, RTBeamsTreatmentRecordStorage, "RTRECORD", generate_uid())
    ds.TreatmentDate = "20240102"
    if fraction is not None:
        ds.CurrentFractionNumber = fraction
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTPlanStorage
    ref.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref])
    return _write(ds, path)


def _manifest(cfg) -> dict:
    return json.loads(meta._cache_manifest_path(cfg.output_root).read_text(encoding="utf-8"))


def _data(cfg) -> Path:
    return cfg.output_root / "Data"


def test_excel_limit_constants_match_excel_sheet_limits():
    from pandas.io.formats.excel import ExcelFormatter

    assert meta._EXCEL_MAX_ROWS == 1_048_576 == ExcelFormatter.max_rows
    assert meta._EXCEL_MAX_COLUMNS == 16_384 == ExcelFormatter.max_cols


def test_real_limit_routes_every_table_pandas_rejects_to_parquet():
    """pandas rejects more than 1,048,576 data rows; the header needs one more row."""
    import io

    rejected = pd.DataFrame({"a": [0] * 1_048_577})
    with pytest.raises(ValueError, match="This sheet is too large"):
        rejected.to_excel(io.BytesIO(), index=False)
    assert not meta._fits_excel_sheet(rejected)
    assert not meta._fits_excel_sheet(pd.DataFrame({"a": [0] * 1_048_576}))
    assert meta._fits_excel_sheet(pd.DataFrame({"a": [0] * 1_048_575}))


def test_fits_excel_sheet_counts_the_header_row(monkeypatch):
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 4)
    monkeypatch.setattr(meta, "_EXCEL_MAX_COLUMNS", 2)

    assert meta._fits_excel_sheet(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}))
    assert not meta._fits_excel_sheet(pd.DataFrame({"a": [1, 2, 3, 4]}))
    assert not meta._fits_excel_sheet(pd.DataFrame({"a": [1], "b": [1], "c": [1]}))


def test_oversized_table_is_published_as_parquet_without_workbook(
    tmp_path, monkeypatch, caplog
):
    cfg = _config(tmp_path)
    _write_export(cfg.dicom_root, EXPORT_NAMES)
    for index in range(2, 4):
        _write_ct(cfg.dicom_root / f"CT_{index}.dcm")
    # Three CT rows plus a header exceed three sheet rows; one-row tables fit.
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 3)

    with caplog.at_level(logging.WARNING, logger=meta.logger.name):
        exported = meta.export_metadata(cfg)

    data = _data(cfg)
    assert exported["ct_images"] == data / "CT_images.parquet"
    assert not (data / "CT_images.xlsx").exists()
    frame = pd.read_parquet(exported["ct_images"])
    assert frame.shape == (3, 6)
    assert sorted(frame["PatientID"]) == ["P1", "P1", "P1"]
    for name in ("plans", "structures", "doses", "fractions", "metadata"):
        assert exported[name].suffix == ".xlsx"
        assert exported[name].is_file()
        assert not exported[name].with_suffix(".parquet").exists()
    assert not list(data.glob(".metadata-export-*"))

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "Excel sheet limit" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "ct_images" in warnings[0]
    assert "3 rows and 6 columns" in warnings[0]
    assert "CT_images.parquet instead of CT_images.xlsx" in warnings[0]

    record = _manifest(cfg)["outputs"]["ct_images"]
    assert record["state"] == "present"
    assert record["format"] == "parquet"
    assert record["rows"] == 3
    assert record["sha256"] == meta._sha256_file(exported["ct_images"])
    assert "format" not in _manifest(cfg)["outputs"]["plans"]


def test_oversized_parquet_stores_mixed_header_values_as_text(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    plan_uid = generate_uid()
    # One record carries an integer fraction number, the other falls back to "NA".
    _write_numbered_record(cfg.dicom_root / "RTRECORD_1.dcm", plan_uid, 1)
    _write_numbered_record(cfg.dicom_root / "RTRECORD_2.dcm", plan_uid, None)
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 2)

    exported = meta.export_metadata(cfg)

    assert exported["fractions"].name == "fractions.parquet"
    frame = pd.read_parquet(exported["fractions"])
    assert sorted(frame["fraction_number"].astype(str)) == ["1", "NA"]


def test_too_many_columns_is_published_as_parquet(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    monkeypatch.setattr(meta, "_EXCEL_MAX_COLUMNS", 5)

    exported = meta.export_metadata(cfg)

    assert exported["ct_images"].name == "CT_images.parquet"
    assert pd.read_parquet(exported["ct_images"]).shape == (1, 6)
    assert not (_data(cfg) / "CT_images.xlsx").exists()


def test_parquet_read_back_mismatch_refuses_publication(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    _write_ct(cfg.dicom_root / "CT_2.dcm")
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 2)
    original = meta._parquet_shape

    def truncated(path):
        rows, columns = original(path)
        return rows - 1, columns

    monkeypatch.setattr(meta, "_parquet_shape", truncated)

    with pytest.raises(meta.MetadataExportError, match="reads back as 1 rows"):
        meta.export_metadata(cfg)
    assert not (_data(cfg) / "CT_images.parquet").exists()
    assert not meta._cache_manifest_path(cfg.output_root).exists()


def test_format_switch_removes_stale_sibling_in_both_directions(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    _write_ct(cfg.dicom_root / "CT_2.dcm")
    data = _data(cfg)

    first = meta.export_metadata(cfg)
    assert first["ct_images"] == data / "CT_images.xlsx"
    assert not (data / "CT_images.parquet").exists()

    # A new source file invalidates the cache; the table now exceeds the limit.
    _write_ct(cfg.dicom_root / "CT_3.dcm")
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 3)
    second = meta.export_metadata(cfg)
    assert second["ct_images"] == data / "CT_images.parquet"
    assert not (data / "CT_images.xlsx").exists()
    assert len(pd.read_parquet(second["ct_images"])) == 3

    # The table fits again: the workbook returns and the Parquet file goes.
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 1_048_576)
    _write_ct(cfg.dicom_root / "CT_4.dcm")
    third = meta.export_metadata(cfg)
    assert third["ct_images"] == data / "CT_images.xlsx"
    assert not (data / "CT_images.parquet").exists()
    assert len(pd.read_excel(third["ct_images"])) == 4
    assert "format" not in _manifest(cfg)["outputs"]["ct_images"]


def test_absent_table_removes_both_formats(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    _write_ct(cfg.dicom_root / "CT_2.dcm")
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 2)
    meta.export_metadata(cfg)
    data = _data(cfg)
    assert (data / "CT_images.parquet").is_file()

    for path in cfg.dicom_root.glob("CT_*.dcm"):
        path.unlink()
    plan_uid = generate_uid()
    _write_numbered_record(cfg.dicom_root / "RTRECORD_1.dcm", plan_uid, 1)
    exported = meta.export_metadata(cfg)

    assert not (data / "CT_images.parquet").exists()
    assert not (data / "CT_images.xlsx").exists()
    assert exported["ct_images"] == data / "CT_images.xlsx"
    assert _manifest(cfg)["outputs"]["ct_images"] == {"state": "absent"}


def test_cached_parquet_output_is_reused_without_reads(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_export(cfg.dicom_root, EXPORT_NAMES)
    _write_ct(cfg.dicom_root / "CT_2.dcm")
    _write_ct(cfg.dicom_root / "CT_3.dcm")
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 3)
    first = meta.export_metadata(cfg)
    before = {name: path.stat().st_mtime_ns for name, path in first.items() if path.exists()}
    manifest_before = meta._cache_manifest_path(cfg.output_root).read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("cache hit must not read DICOM headers")

    monkeypatch.setattr(meta.pydicom, "dcmread", forbidden)
    second = meta.export_metadata(cfg)

    assert second == first
    assert second["ct_images"].suffix == ".parquet"
    assert before == {
        name: path.stat().st_mtime_ns for name, path in second.items() if path.exists()
    }
    assert meta._cache_manifest_path(cfg.output_root).read_bytes() == manifest_before


@pytest.mark.parametrize("damage", ["stale_workbook", "modified_parquet", "missing_parquet"])
def test_cached_parquet_output_is_rejected_when_inconsistent(
    tmp_path, monkeypatch, damage
):
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    _write_ct(cfg.dicom_root / "CT_2.dcm")
    monkeypatch.setattr(meta, "_EXCEL_MAX_ROWS", 2)
    exported = meta.export_metadata(cfg)
    parquet_path = exported["ct_images"]
    workbook_path = parquet_path.with_suffix(".xlsx")
    if damage == "stale_workbook":
        workbook_path.write_bytes(b"stale workbook from an earlier run")
    elif damage == "modified_parquet":
        parquet_path.write_bytes(b"not parquet")
    else:
        parquet_path.unlink()

    reads = 0
    original_read = meta.pydicom.dcmread

    def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", counted)
    repaired = meta.export_metadata(cfg)

    assert reads > 0
    assert repaired["ct_images"] == parquet_path
    assert len(pd.read_parquet(parquet_path)) == 2
    assert not workbook_path.exists()


def test_cache_with_unrecorded_parquet_sibling_is_rejected(tmp_path, monkeypatch):
    """A crash between moving a new Parquet file and removing the workbook is not reused."""
    cfg = _config(tmp_path)
    _write_ct(cfg.dicom_root / "CT_1.dcm")
    exported = meta.export_metadata(cfg)
    exported["ct_images"].with_suffix(".parquet").write_bytes(b"residue")

    reads = 0
    original_read = meta.pydicom.dcmread

    def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(meta.pydicom, "dcmread", counted)
    repaired = meta.export_metadata(cfg)

    assert reads > 0
    assert repaired["ct_images"].suffix == ".xlsx"
    assert not repaired["ct_images"].with_suffix(".parquet").exists()


def test_cache_written_before_format_records_remains_valid(tmp_path, monkeypatch):
    """Records without a format key describe workbooks, as before this change."""
    cfg = _config(tmp_path)
    _write_export(cfg.dicom_root, EXPORT_NAMES)
    first = meta.export_metadata(cfg)
    manifest = _manifest(cfg)
    assert all("format" not in record for record in manifest["outputs"].values())

    monkeypatch.setattr(meta.pydicom, "dcmread", lambda *a, **k: pytest.fail("cache miss"))
    assert meta.export_metadata(cfg) == first


# --- byte identity against the baseline revision --------------------------------


class _FrozenDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 12, 0, 0, tzinfo=tz)


def _freeze_workbook_clock(monkeypatch) -> None:
    """Workbook writers stamp document properties with the current time."""
    import openpyxl.packaging.core as openpyxl_core
    import openpyxl.writer.excel as openpyxl_writer

    frozen = types.SimpleNamespace(datetime=_FrozenDatetime, timezone=datetime.timezone)
    monkeypatch.setattr(openpyxl_core, "datetime", frozen)
    monkeypatch.setattr(openpyxl_writer, "datetime", frozen)
    try:
        import xlsxwriter.core as xlsxwriter_core
        import xlsxwriter.workbook as xlsxwriter_workbook
    except ImportError:
        return
    monkeypatch.setattr(xlsxwriter_core, "datetime", _FrozenDatetime)
    monkeypatch.setattr(xlsxwriter_workbook, "datetime", _FrozenDatetime)


def _baseline_meta(monkeypatch):
    try:
        source = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{BASELINE_REVISION}:rtpipeline/meta.py"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"baseline revision {BASELINE_REVISION} is not available")
    name = "rtpipeline._meta_baseline"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "rtpipeline"
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, f"{BASELINE_REVISION}:rtpipeline/meta.py", "exec"), module.__dict__)
    return module


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_normal_size_export_is_byte_identical_to_baseline(tmp_path, monkeypatch):
    baseline = _baseline_meta(monkeypatch)
    dicom_root = tmp_path / "dicom"
    _write_export(dicom_root, EXPORT_NAMES)
    plan_uid = generate_uid()
    _write_numbered_record(dicom_root / "RTRECORD_2.dcm", plan_uid, 2)
    for index in range(2, 5):
        _write_ct(dicom_root / f"CT_{index}.dcm")
    _freeze_workbook_clock(monkeypatch)

    outputs = {}
    for label, module in (("baseline", baseline), ("current", meta)):
        cfg = meta.PipelineConfig(
            dicom_root=dicom_root,
            output_root=tmp_path / label,
            logs_root=tmp_path / f"logs-{label}",
            max_workers_override=1,
        )
        exported = module.export_metadata(cfg)
        outputs[label] = (
            {name: path.relative_to(cfg.output_root) for name, path in exported.items()},
            _tree_bytes(cfg.output_root),
        )

    baseline_paths, baseline_files = outputs["baseline"]
    current_paths, current_files = outputs["current"]
    assert current_paths == baseline_paths
    assert sorted(current_files) == sorted(baseline_files)
    assert "_CACHE/metadata_export.json" in current_files
    assert len([name for name in current_files if name.startswith("Data/")]) == 6
    for name in baseline_files:
        assert current_files[name] == baseline_files[name], name


# --- aggregation copy into _RESULTS ---------------------------------------------


def _copy_namespace(tmp_path: Path) -> tuple[dict, Path, Path]:
    namespace = _aggregate_functions()
    output_dir = tmp_path / "Output"
    results_dir = output_dir / "_RESULTS"
    (output_dir / "Data").mkdir(parents=True)
    results_dir.mkdir()
    namespace["OUTPUT_DIR"] = output_dir
    namespace["RESULTS_DIR"] = results_dir
    namespace["snakemake"] = SimpleNamespace(
        output=SimpleNamespace(fractions=str(results_dir / "fractions.xlsx"))
    )
    return namespace, output_dir / "Data", results_dir


def test_supplemental_copy_takes_parquet_and_removes_stale_workbook(tmp_path):
    namespace, data, results = _copy_namespace(tmp_path)
    pd.DataFrame({"PatientID": ["P1", "P1"]}).to_parquet(data / "CT_images.parquet", index=False)
    pd.DataFrame({"plan_name": ["a"]}).to_excel(data / "plans.xlsx", index=False)
    (results / "CT_images.xlsx").write_bytes(b"stale copy from an earlier run")
    (results / "plans.parquet").write_bytes(b"stale copy from an earlier run")

    namespace["_copy_supplemental_sources"]()

    assert (results / "CT_images.parquet").read_bytes() == (data / "CT_images.parquet").read_bytes()
    assert not (results / "CT_images.xlsx").exists()
    assert (results / "plans.xlsx").read_bytes() == (data / "plans.xlsx").read_bytes()
    assert not (results / "plans.parquet").exists()


def test_supplemental_copy_keeps_aggregated_fraction_workbook(tmp_path):
    namespace, data, results = _copy_namespace(tmp_path)
    pd.DataFrame({"fraction_id": ["1"]}).to_parquet(data / "fractions.parquet", index=False)
    aggregated = results / "fractions.xlsx"
    pd.DataFrame({"patient_id": ["P1"], "course_id": ["C1"]}).to_excel(aggregated, index=False)
    aggregated_bytes = aggregated.read_bytes()

    namespace["_copy_supplemental_sources"]()

    assert (results / "fractions.parquet").is_file()
    assert aggregated.read_bytes() == aggregated_bytes


def test_supplemental_copy_skips_ambiguous_generation(tmp_path, capsys):
    namespace, data, results = _copy_namespace(tmp_path)
    pd.DataFrame({"PatientID": ["P1"]}).to_excel(data / "CT_images.xlsx", index=False)
    pd.DataFrame({"PatientID": ["P1"]}).to_parquet(data / "CT_images.parquet", index=False)
    (results / "CT_images.xlsx").write_bytes(b"stale copy from an earlier run")

    namespace["_copy_supplemental_sources"]()

    assert not (results / "CT_images.xlsx").exists()
    assert not (results / "CT_images.parquet").exists()
    assert "neither is copied" in capsys.readouterr().out


def test_supplemental_copy_leaves_results_alone_when_source_is_absent(tmp_path):
    namespace, _data_dir, results = _copy_namespace(tmp_path)
    existing = results / "CT_images.xlsx"
    existing.write_bytes(b"earlier copy")

    namespace["_copy_supplemental_sources"]()

    assert existing.read_bytes() == b"earlier copy"
