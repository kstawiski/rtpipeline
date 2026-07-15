from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .layout import CourseDirs, build_course_dirs
from .modality_classifier import CBCT_MANUFACTURER_MODELS, FOURDCT_MODELS, classify_series

logger = logging.getLogger(__name__)


TS_TASK_BY_CLASS = {
    "planning_ct": "total",
    "diagnostic_ct": "total",
    "petct_ct": "total",
    "cbct": "total",
    "fourdct_ave": "total",
    "fourdct_phase": "total",
    "mr_anatomic": "total_mr",
    "mr_functional": "none",
    "pt": "none",
    "exclude": "none",
}


@dataclass(slots=True)
class InventoryInstance:
    sop_instance_uid: str
    path: Path
    instance_number: int | None
    image_type: str
    slice_thickness: float | None
    file_id: int


@dataclass(slots=True)
class InventorySeries:
    patient_id: str
    study_uid: str
    study_description: str
    series_uid: str
    modality: str
    series_description: str
    manufacturer: str
    manufacturer_model: str
    frame_of_reference_uid: str
    n_instances: int
    instances: list[InventoryInstance]
    has_pt_same_study_for: bool = False
    rt_linked: bool = False
    rt_series_linked: bool = False
    rt_for_linked: bool = False
    is_planning_ct: bool = False
    rt_link_basis: str = "none"

    @property
    def image_types(self) -> list[str]:
        return sorted({inst.image_type for inst in self.instances if inst.image_type})

    @property
    def n_slices(self) -> int:
        return len(self.instances) or int(self.n_instances or 0)

    def classifier_meta(self, *, config: Any | None = None) -> dict[str, Any]:
        meta = {
            "patient_id": self.patient_id,
            "study_uid": self.study_uid,
            "study_description": self.study_description,
            "series_uid": self.series_uid,
            "modality": self.modality,
            "series_description": self.series_description,
            "manufacturer": self.manufacturer,
            "manufacturer_model": self.manufacturer_model,
            "frame_of_reference_uid": self.frame_of_reference_uid,
            "n_instances": self.n_slices,
            "image_types": self.image_types,
            "has_pt_same_study_for": self.has_pt_same_study_for,
            "rt_linked": self.rt_linked,
            "rt_series_linked": self.rt_series_linked,
            "is_planning_ct": self.is_planning_ct,
            "rt_link_basis": self.rt_link_basis,
        }
        if config is not None:
            meta["cbct_manufacturer_models"] = getattr(config, "cbct_manufacturer_models", None)
            meta["fourdct_models"] = getattr(config, "fourdct_models", None)
        return meta


def enumerate_patient_series(
    db_path: Path | str,
    patient_id: str,
) -> list[InventorySeries]:
    """Return all inventory series for one patient using DB file paths only."""
    db_path = Path(db_path)
    with _connect_readonly(db_path) as conn:
        conn.row_factory = sqlite3.Row
        series_rows = conn.execute(
            """
            SELECT
                st.patient_id,
                s.study_uid,
                COALESCE(st.study_description, '') AS study_description,
                s.series_uid,
                s.modality,
                COALESCE(s.series_description, '') AS series_description,
                COALESCE(s.manufacturer, '') AS manufacturer,
                COALESCE(s.manufacturer_model, '') AS manufacturer_model,
                COALESCE(s.frame_of_reference_uid, '') AS frame_of_reference_uid,
                s.n_instances
            FROM series s
            JOIN studies st ON st.study_uid = s.study_uid
            WHERE st.patient_id = ?
            ORDER BY s.study_uid, s.series_uid
            """,
            (str(patient_id),),
        ).fetchall()

        pt_study_for = {
            (str(row["study_uid"]), str(row["frame_of_reference_uid"]))
            for row in series_rows
            if str(row["modality"]).upper() == "PT" and str(row["frame_of_reference_uid"])
        }
        rt_series, rt_study_for = _rtstruct_targets_for_patient(conn, str(patient_id))

        result: list[InventorySeries] = []
        for row in series_rows:
            study_uid = str(row["study_uid"])
            series_uid = str(row["series_uid"])
            for_uid = str(row["frame_of_reference_uid"] or "")
            instances = _instances_for_series(conn, series_uid)
            rt_series_linked = series_uid in rt_series
            rt_for_linked = (study_uid, for_uid) in rt_study_for if for_uid else False
            result.append(
                InventorySeries(
                    patient_id=str(row["patient_id"]),
                    study_uid=study_uid,
                    study_description=str(row["study_description"] or ""),
                    series_uid=series_uid,
                    modality=str(row["modality"] or ""),
                    series_description=str(row["series_description"] or ""),
                    manufacturer=str(row["manufacturer"] or ""),
                    manufacturer_model=str(row["manufacturer_model"] or ""),
                    frame_of_reference_uid=for_uid,
                    n_instances=int(row["n_instances"] or 0),
                    instances=instances,
                    has_pt_same_study_for=(study_uid, for_uid) in pt_study_for if for_uid else False,
                    rt_linked=rt_series_linked or rt_for_linked,
                    rt_series_linked=rt_series_linked,
                    rt_for_linked=rt_for_linked,
                )
            )
        _assign_planning_flags(result)
        return result


def build_patient_series_manifest_rows(
    db_path: Path | str,
    patient_id: str,
    *,
    course_dirs: CourseDirs,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series in enumerate_patient_series(db_path, patient_id):
        image_class, reason = classify_series(series.classifier_meta(config=config))
        output_dir = ""
        if image_class != "exclude":
            output_dir = str(output_dir_for_image_class(course_dirs, image_class, series.series_uid))
        rows.append(
            {
                "patient_id": series.patient_id,
                "study_uid": series.study_uid,
                "series_uid": series.series_uid,
                "modality": series.modality,
                "series_description": series.series_description,
                "manufacturer": series.manufacturer,
                "image_class": image_class,
                "manufacturer_model": series.manufacturer_model,
                "frame_of_reference_uid": series.frame_of_reference_uid,
                "image_types": series.image_types,
                "is_planning_ct": series.is_planning_ct,
                "rt_link_basis": series.rt_link_basis,
                "n_slices": series.n_slices,
                "ts_task": TS_TASK_BY_CLASS.get(image_class, "none"),
                "output_dir": output_dir,
                "status": "excluded" if image_class == "exclude" else "classified",
                "exclusion_reason": reason or "",
            }
        )
    return rows


def materialize_patient_series_from_inventory(
    config: Any,
    patient_id: str,
    *,
    patient_series_root: Path | None = None,
) -> Path:
    """Classify, copy eligible series to per-SeriesUID dirs, and write a manifest."""
    db_path = getattr(config, "inventory_db_path", None)
    if not db_path:
        raise ValueError("do_segment_all_series requires config.inventory_db_path")

    root = patient_series_root or (Path(config.output_root) / str(patient_id) / "all_series")
    course_dirs = build_course_dirs(root)
    course_dirs.ensure_all_series()

    series_by_uid = {
        series.series_uid: series
        for series in enumerate_patient_series(Path(db_path), str(patient_id))
    }
    rows = build_patient_series_manifest_rows(
        Path(db_path),
        str(patient_id),
        course_dirs=course_dirs,
        config=config,
    )

    # Materialization allow-list. None => materialize every non-excluded series (legacy). An explicit
    # list (including []) is an allow-list: only listed image_classes are byte-copied. This matches the
    # sibling all_series_segment_classes contract ([] => none, None => all; see tests/test_all_series_scope.py).
    materialize_classes = getattr(config, "all_series_materialize_classes", None)
    allow_classes = (
        {str(c).strip() for c in materialize_classes if str(c).strip()}
        if materialize_classes is not None
        else None
    )
    if allow_classes is not None:
        # Fail closed: never skip-materialize a class the all-series segmentation stage would segment.
        # With an explicit segment allow-list, union it; with the legacy segment-everything scope
        # (all_series_segment_classes is None), union every class that carries a TotalSegmentator task.
        seg_classes = getattr(config, "all_series_segment_classes", None)
        if seg_classes is not None:
            # Explicit segmentation scope. Mirrors segmentation's `if allowed is not None` semantics:
            # [] => segment nothing => union nothing; a non-empty list => union exactly those classes.
            allow_classes |= {str(c).strip() for c in seg_classes if str(c).strip()}
        elif getattr(config, "do_segment_all_series", False):
            # Legacy segment-everything scope (None): keep every TotalSegmentator-eligible class.
            allow_classes |= {cls for cls, task in TS_TASK_BY_CLASS.items() if task != "none"}

    for row in rows:
        if row["image_class"] == "exclude":
            continue
        if allow_classes is not None and row["image_class"] not in allow_classes:
            row["status"] = "materialize_skipped_out_of_scope"
            row["materialized_n_slices"] = 0
            continue
        series = series_by_uid.get(str(row["series_uid"]))
        if series is None:
            row["status"] = "missing_inventory_series"
            continue
        output_dir = Path(str(row["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        missing = _copy_instances(
            series.instances,
            output_dir,
            series.modality,
            use_hardlinks=bool(getattr(config, "dicom_copy_use_hardlinks", False)),
        )
        row["materialized_n_slices"] = len(list(output_dir.glob("*.dcm")))
        row["status"] = "missing_source_file" if missing else "materialized"

    manifest_path = course_dirs.metadata / "series_manifest.json"
    write_patient_series_manifest(
        manifest_path,
        patient_id=str(patient_id),
        rows=rows,
        db_path=Path(db_path),
        scan_run_id=getattr(config, "inventory_scan_run_id", None),
    )
    return manifest_path


def write_patient_series_manifest(
    manifest_path: Path,
    *,
    patient_id: str,
    rows: list[dict[str, Any]],
    db_path: Path,
    scan_run_id: int | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "patient_id": patient_id,
        "inventory_db": str(db_path),
        "scan_run_id": scan_run_id,
        "generated_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "series": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def output_dir_for_image_class(course_dirs: CourseDirs, image_class: str, series_uid: str) -> Path:
    safe_uid = _safe_uid(series_uid)
    if image_class == "planning_ct":
        return course_dirs.dicom_ct / safe_uid
    if image_class == "diagnostic_ct":
        return course_dirs.dicom_ct_diagnostic / safe_uid
    if image_class == "petct_ct":
        return course_dirs.dicom_petct / safe_uid
    if image_class == "cbct":
        return course_dirs.dicom_cbct / safe_uid
    if image_class in {"fourdct_ave", "fourdct_phase"}:
        return course_dirs.dicom_4dct / safe_uid
    if image_class == "mr_anatomic":
        return course_dirs.dicom_mr / safe_uid / "DICOM"
    if image_class == "mr_functional":
        return course_dirs.dicom_mr_functional / safe_uid / "DICOM"
    if image_class == "pt":
        return course_dirs.dicom_pt / safe_uid
    raise ValueError(f"Unsupported image_class for output directory: {image_class}")


def _assign_planning_flags(series_list: list[InventorySeries]) -> None:
    # Planning candidates are calibrated-CT-eligible series; CBCT and model-routed
    # 4DCT reconstructions are classified through their dedicated paths.
    cbct = {m.lower() for m in CBCT_MANUFACTURER_MODELS}
    fourd = {m.lower() for m in FOURDCT_MODELS}

    def candidate(s: InventorySeries) -> bool:
        return (
            (s.modality or "").upper() == "CT"
            and s.manufacturer_model.strip().lower() not in cbct
            and s.manufacturer_model.strip().lower() not in fourd
            and (s.rt_series_linked or s.rt_for_linked)
        )

    groups: dict[tuple[str, str], list[InventorySeries]] = {}
    for s in series_list:
        if candidate(s):
            groups.setdefault((s.study_uid, s.frame_of_reference_uid), []).append(s)

    for s in series_list:
        if not candidate(s):
            continue
        grp = groups[(s.study_uid, s.frame_of_reference_uid)]
        if s.rt_series_linked:
            s.is_planning_ct, s.rt_link_basis = True, "rtstruct_to_series"
        elif s.rt_for_linked:
            if any(g.rt_series_linked for g in grp):
                s.is_planning_ct, s.rt_link_basis = False, "for_superseded_by_series"
            elif sum(1 for g in grp if g.rt_for_linked) == 1:
                s.is_planning_ct, s.rt_link_basis = True, "rtstruct_to_for_unique"
            else:
                s.is_planning_ct, s.rt_link_basis = False, "rtstruct_to_for_ambiguous"


def list_inventory_patient_ids(db_path: Path | str) -> list[str]:
    with _connect_readonly(Path(db_path)) as conn:
        rows = conn.execute("SELECT patient_id FROM patients ORDER BY patient_id").fetchall()
    return [str(row[0]) for row in rows]


def load_scan_run_metadata(db_path: Path | str, run_id: int) -> dict[str, Any]:
    with _connect_readonly(Path(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM scan_runs WHERE run_id = ?",
            (int(run_id),),
        ).fetchone()
    return dict(row) if row is not None else {}


def _instances_for_series(conn: sqlite3.Connection, series_uid: str) -> list[InventoryInstance]:
    rows = conn.execute(
        """
        SELECT
            i.sop_instance_uid,
            i.instance_number,
            COALESCE(i.image_type, '') AS image_type,
            i.slice_thickness,
            i.primary_file_id,
            f.file_path
        FROM instances i
        JOIN dicom_files f ON f.file_id = i.primary_file_id
        WHERE i.series_uid = ?
        ORDER BY
            CASE WHEN i.instance_number IS NULL THEN 1 ELSE 0 END,
            i.instance_number,
            i.sop_instance_uid
        """,
        (series_uid,),
    ).fetchall()

    by_sop: dict[str, InventoryInstance] = {}
    for row in rows:
        sop_uid = str(row["sop_instance_uid"] or "")
        if not sop_uid or sop_uid in by_sop:
            continue
        by_sop[sop_uid] = InventoryInstance(
            sop_instance_uid=sop_uid,
            path=Path(str(row["file_path"])),
            instance_number=int(row["instance_number"]) if row["instance_number"] is not None else None,
            image_type=str(row["image_type"] or ""),
            slice_thickness=float(row["slice_thickness"]) if row["slice_thickness"] is not None else None,
            file_id=int(row["primary_file_id"]),
        )
    return list(by_sop.values())


def _rtstruct_targets_for_patient(conn: sqlite3.Connection, patient_id: str) -> tuple[set[str], set[tuple[str, str]]]:
    source_rows = conn.execute(
        """
        SELECT
            i.sop_instance_uid,
            i.series_uid
        FROM instances i
        JOIN series s ON s.series_uid = i.series_uid
        JOIN studies st ON st.study_uid = s.study_uid
        WHERE st.patient_id = ?
          AND i.modality = 'RTSTRUCT'
        """,
        (patient_id,),
    ).fetchall()
    if not source_rows:
        return set(), set()

    source_sops = [str(row["sop_instance_uid"]) for row in source_rows if row["sop_instance_uid"]]
    series_uids: set[str] = set()
    study_for: set[tuple[str, str]] = set()
    for chunk_start in range(0, len(source_sops), 500):
        chunk = source_sops[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT
                target_series_uid,
                target_for_uid,
                target_study_uid
            FROM rt_links
            WHERE relationship IN ('rtstruct_to_series', 'rtstruct_to_for')
              AND source_sop_uid IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        for row in rows:
            target_series = str(row["target_series_uid"] or "")
            target_for = str(row["target_for_uid"] or "")
            target_study = str(row["target_study_uid"] or "")
            if target_series:
                series_uids.add(target_series)
            if target_study and target_for:
                study_for.add((target_study, target_for))
    return series_uids, study_for


def _image_type_has_token(image_type: str, token: str) -> bool:
    return token.upper() in {part.strip().upper() for part in str(image_type or "").split("\\")}


def _filter_localizer_instances(instances: list[InventoryInstance]) -> list[InventoryInstance]:
    has_localizer = any(_image_type_has_token(inst.image_type, "LOCALIZER") for inst in instances)
    has_axial_or_helical = any(
        _image_type_has_token(inst.image_type, "AXIAL") or _image_type_has_token(inst.image_type, "HELICAL")
        for inst in instances
    )
    if not (has_localizer and has_axial_or_helical):
        return instances

    filtered = [
        inst
        for inst in instances
        if not _image_type_has_token(inst.image_type, "LOCALIZER")
    ]
    if len(filtered) < 10:
        logger.warning(
            "Keeping LOCALIZER instances because filtering would leave only %d instances",
            len(filtered),
        )
        return instances
    logger.info(
        "Dropped %d LOCALIZER instance(s) from mixed axial/helical series before materialization",
        len(instances) - len(filtered),
    )
    return filtered


def _copy_instances(
    instances: Iterable[InventoryInstance],
    output_dir: Path,
    modality: str,
    use_hardlinks: bool = False,
) -> bool:
    used: set[int] = set()
    missing = False
    prefix = (modality or "DICOM").upper()
    filtered_instances = _filter_localizer_instances(list(instances))
    destinations: list[tuple[InventoryInstance, Path]] = []
    for idx, instance in enumerate(filtered_instances, start=1):
        file_idx = instance.instance_number if instance.instance_number is not None else idx
        while file_idx in used:
            file_idx += 1
        used.add(file_idx)
        destinations.append((instance, output_dir / f"{prefix}_{file_idx:05d}.dcm"))

    intended = {destination for _, destination in destinations}
    for existing in output_dir.glob("*.dcm"):
        if existing not in intended:
            existing.unlink()

    for instance, destination in destinations:
        source = instance.path
        if not source.exists():
            missing = True
            continue
        if destination.exists():
            continue
        # Prefer hardlinks when enabled and source/output share a filesystem. This avoids
        # byte-copying large materialized DICOM collections. Materialized DICOMs are only
        # read downstream (dcm2niix), never modified in place, so a shared inode is safe.
        # Fall back to copy2 on any OSError (e.g. EXDEV cross-device, EMLINK).
        if use_hardlinks:
            try:
                os.link(source, destination)
                continue
            except OSError:
                pass
        shutil.copy2(source, destination)
    return missing


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _safe_uid(series_uid: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in str(series_uid))
    return safe[:128] or "series"


__all__ = [
    "InventoryInstance",
    "InventorySeries",
    "TS_TASK_BY_CLASS",
    "build_patient_series_manifest_rows",
    "enumerate_patient_series",
    "list_inventory_patient_ids",
    "load_scan_run_metadata",
    "materialize_patient_series_from_inventory",
    "output_dir_for_image_class",
    "write_patient_series_manifest",
]
