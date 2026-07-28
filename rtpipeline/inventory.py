from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
import sqlite3
import uuid
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

BODY_COMPOSITION_ELIGIBLE_CLASSES = frozenset({"planning_ct", "diagnostic_ct", "petct_ct"})
BODY_COMPOSITION_TS_TASKS = ("tissue_types", "body")


def ts_tasks_for_image_class(
    image_class: str,
    body_composition_classes: Iterable[str] | None = None,
) -> list[str]:
    """Return TotalSegmentator tasks for an image_class.

    The base ``ts_task`` stays unchanged for backward compatibility. Body-composition
    tasks are opt-in, CT-only add-ons and never replace the base ``total`` task.
    """
    base_task = TS_TASK_BY_CLASS.get(str(image_class), "none")
    if base_task == "none":
        return []

    tasks = [base_task]
    if isinstance(body_composition_classes, str):
        configured = {
            item.strip()
            for item in body_composition_classes.replace(";", ",").split(",")
            if item.strip()
        }
    elif body_composition_classes is not None:
        configured = {str(item).strip() for item in body_composition_classes if str(item).strip()}
    else:
        configured = set()
    if str(image_class) in BODY_COMPOSITION_ELIGIBLE_CLASSES and str(image_class) in configured:
        tasks.extend(BODY_COMPOSITION_TS_TASKS)
    return tasks


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
        row = {
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
        tasks = ts_tasks_for_image_class(
            image_class,
            getattr(config, "body_composition_classes", None) if config is not None else None,
        )
        if len(tasks) > 1:
            row["ts_tasks"] = tasks
        rows.append(row)
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


def manual_rtstruct_bindings_from_inventory(
    db_path: Path | str | None,
    patient_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    """Resolve all-series manifest rows to source manual RTSTRUCTs via inventory RT links.

    Exact RTSTRUCT->SeriesInstanceUID links win. RTSTRUCT->(StudyInstanceUID,
    FrameOfReferenceUID) links are used only when the manifest row was already
    marked unique by inventory classification, or when the manifest itself has a
    single row for that study/frame pair.
    """
    if not db_path:
        return {}
    db_path = Path(db_path)
    if not db_path.exists():
        return {}

    rows_by_study_for: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        study_uid = str(row.get("study_uid") or "").strip()
        for_uid = str(row.get("frame_of_reference_uid") or "").strip()
        if study_uid and for_uid:
            rows_by_study_for.setdefault((study_uid, for_uid), []).append(row)

    try:
        with _connect_readonly(db_path) as conn:
            conn.row_factory = sqlite3.Row
            source_rows = conn.execute(
                """
                SELECT
                    i.sop_instance_uid,
                    f.file_path
                FROM instances i
                JOIN series s ON s.series_uid = i.series_uid
                JOIN studies st ON st.study_uid = s.study_uid
                JOIN dicom_files f ON f.file_id = i.primary_file_id
                WHERE st.patient_id = ?
                  AND i.modality = 'RTSTRUCT'
                """,
                (str(patient_id),),
            ).fetchall()

            source_paths = {
                str(row["sop_instance_uid"]): Path(str(row["file_path"]))
                for row in source_rows
                if row["sop_instance_uid"] and row["file_path"]
            }
            if not source_paths:
                return {}

            exact_by_series: dict[str, list[Path]] = {}
            by_study_for: dict[tuple[str, str], list[Path]] = {}
            source_sops = list(source_paths)
            for chunk_start in range(0, len(source_sops), 500):
                chunk = source_sops[chunk_start : chunk_start + 500]
                placeholders = ",".join("?" for _ in chunk)
                # ``placeholders`` contains only a generated comma-separated list of
                # SQLite bind markers; every SOP UID remains a bound parameter.
                link_rows = conn.execute(
                    f"""
                    SELECT
                        source_sop_uid,
                        relationship,
                        target_series_uid,
                        target_for_uid,
                        target_study_uid
                    FROM rt_links
                    WHERE relationship IN ('rtstruct_to_series', 'rtstruct_to_for')
                      AND source_sop_uid IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()
                for link in link_rows:
                    source_path = source_paths.get(str(link["source_sop_uid"] or ""))
                    if source_path is None:
                        continue
                    relationship = str(link["relationship"] or "")
                    target_series = str(link["target_series_uid"] or "").strip()
                    target_for = str(link["target_for_uid"] or "").strip()
                    target_study = str(link["target_study_uid"] or "").strip()
                    if relationship == "rtstruct_to_series" and target_series:
                        exact_by_series.setdefault(target_series, []).append(source_path)
                    elif relationship == "rtstruct_to_for" and target_study and target_for:
                        # D1 is stricter than C1's target inference: only explicit FoR links can match, so misses fail closed without wrong-series export.
                        by_study_for.setdefault((target_study, target_for), []).append(source_path)
    except Exception as exc:
        logger.warning(
            "D1 original-export disabled for patient %s: unable to resolve manual RTSTRUCT bindings "
            "from inventory %s (inventory schema/rt_links absent or unreadable): %s",
            patient_id,
            db_path,
            exc,
        )
        return {}

    def _unique_paths(paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key not in seen:
                unique.append(path)
                seen.add(key)
        return unique

    bindings: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        series_uid = str(row.get("series_uid") or "").strip()
        if not series_uid:
            continue

        exact = _unique_paths(exact_by_series.get(series_uid, []))
        if len(exact) == 1:
            bindings[series_uid] = exact[0]
            continue
        if len(exact) > 1:
            logger.warning(
                "Multiple manual RTSTRUCTs reference patient %s series %s; skipping all-series original export",
                patient_id,
                series_uid,
            )
            continue

        study_uid = str(row.get("study_uid") or "").strip()
        for_uid = str(row.get("frame_of_reference_uid") or "").strip()
        if not study_uid or not for_uid:
            continue
        for_key = (study_uid, for_uid)
        for_matches = _unique_paths(by_study_for.get(for_key, []))
        if len(for_matches) != 1:
            if len(for_matches) > 1:
                logger.warning(
                    "Multiple manual RTSTRUCTs reference patient %s study %s FrameOfReferenceUID %s; "
                    "skipping all-series original export for series %s",
                    patient_id,
                    study_uid,
                    for_uid,
                    series_uid,
                )
            continue
        basis = str(row.get("rt_link_basis") or "")
        if basis == "rtstruct_to_for_unique" or (
            not basis and len(rows_by_study_for.get(for_key, [])) == 1
        ):
            bindings[series_uid] = for_matches[0]
    return bindings


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
            destination.unlink(missing_ok=True)
            continue
        # Prefer hardlinks when enabled and source/output share a filesystem. This avoids
        # byte-copying large materialized DICOM collections. Materialized DICOMs are only
        # read downstream (dcm2niix), never modified in place, so a shared inode is safe.
        # Fall back to copy2 on any OSError (e.g. EXDEV cross-device, EMLINK).
        if use_hardlinks:
            if destination.exists():
                try:
                    if os.path.samefile(source, destination):
                        continue
                except OSError:
                    pass
                destination.unlink()
            try:
                os.link(source, destination)
                continue
            except OSError:
                pass
        elif destination.exists():
            try:
                source_stat = source.stat()
                destination_stat = destination.stat()
                if (
                    source_stat.st_size == destination_stat.st_size
                    and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
                ):
                    continue
            except OSError:
                pass

        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
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
    "BODY_COMPOSITION_ELIGIBLE_CLASSES",
    "BODY_COMPOSITION_TS_TASKS",
    "TS_TASK_BY_CLASS",
    "build_patient_series_manifest_rows",
    "enumerate_patient_series",
    "list_inventory_patient_ids",
    "load_scan_run_metadata",
    "manual_rtstruct_bindings_from_inventory",
    "materialize_patient_series_from_inventory",
    "output_dir_for_image_class",
    "ts_tasks_for_image_class",
    "write_patient_series_manifest",
]
