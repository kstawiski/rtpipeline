"""Aggregate per-course RTpipeline artifacts into cohort workbooks."""

import hashlib
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore

from rtpipeline.dvh_aggregate import build_dvh_aggregate, write_dvh_aggregate
from rtpipeline.radiomics_cohort import (
    attach_radiomics_cohort_provenance,
    build_radiomics_cohort_provenance,
    is_valid_radiomics_cohort_table,
)


OUTPUT_DIR = Path(snakemake.params.output_dir)  # type: ignore[name-defined]
RESULTS_DIR = Path(snakemake.params.results_dir)  # type: ignore[name-defined]
RADIOMICS_ENABLED = bool(snakemake.params.radiomics_enabled)  # type: ignore[name-defined]
CAMPAIGN_MODE = bool(getattr(snakemake.params, "campaign_mode", False))  # type: ignore[name-defined]
CAMPAIGN_MIN_COMPLETION_FRACTION = float(
    getattr(snakemake.params, "campaign_min_completion_fraction", 0.5)  # type: ignore[name-defined]
)
CAMPAIGN_REQUIRE_ALL_COURSES = bool(
    getattr(snakemake.params, "campaign_require_all_courses", False)  # type: ignore[name-defined]
)
WORKER_BUDGET = max(1, int(snakemake.params.worker_budget))  # type: ignore[name-defined]
AUTO_WORKER_BUDGET = max(1, int(snakemake.params.auto_worker_budget))  # type: ignore[name-defined]
aggregation_threads_value = int(snakemake.params.aggregation_threads)  # type: ignore[name-defined]
AGGREGATION_THREADS = (
    None if aggregation_threads_value < 1 else aggregation_threads_value
)


def _manifest_courses():
    manifest_path = Path(snakemake.input.manifest)  # type: ignore[name-defined]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Course manifest is unreadable: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("courses"), list):
        raise RuntimeError(
            f"Course manifest is malformed: {manifest_path} must contain a courses list"
        )

    courses = []
    seen = set()
    for index, entry in enumerate(payload["courses"], start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Course manifest is malformed: entry {index} is not a mapping"
            )
        patient_id = entry.get("patient")
        course_id = entry.get("course")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise RuntimeError(
                f"Course manifest is malformed: entry {index} has no patient identifier"
            )
        if not isinstance(course_id, str) or not course_id.strip():
            raise RuntimeError(
                f"Course manifest is malformed: entry {index} has no course identifier"
            )
        key = (patient_id, course_id)
        if key in seen:
            raise RuntimeError(
                f"Course manifest is malformed: duplicate course {patient_id}/{course_id}"
            )
        seen.add(key)
        courses.append((patient_id, course_id, OUTPUT_DIR / patient_id / course_id))
    if payload.get("schema") == "rtpipeline-organized-course-manifest-v2":
        quarantine_entries = payload.get("technical_quarantines")
        if not isinstance(quarantine_entries, list):
            raise RuntimeError(
                "Course manifest is malformed: technical_quarantines must be a list"
            )
        try:
            attempted = int(payload["attempted_course_count"])
            intended = int(payload["intended_course_count"])
            validated = int(payload["validated_course_count"])
            quarantined = int(payload["technical_quarantine_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Course manifest is malformed: organize denominator fields are invalid"
            ) from exc
        if intended != attempted:
            raise RuntimeError(
                "Course manifest is malformed: intended and attempted course counts disagree"
            )
        if validated != len(courses):
            raise RuntimeError(
                "Course manifest is malformed: validated count does not match courses"
            )
        if quarantined != len(quarantine_entries):
            raise RuntimeError(
                "Course manifest is malformed: technical quarantine count does not match records"
            )
        if attempted != validated + quarantined:
            raise RuntimeError(
                "Course manifest is malformed: attempted count does not reconcile with "
                "validated and technically quarantined courses"
            )
        quarantine_ids = set()
        for index, entry in enumerate(quarantine_entries, start=1):
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Course manifest is malformed: technical quarantine {index} is not a mapping"
                )
            patient_id = str(entry.get("patient") or "").strip()
            course_id = str(entry.get("course") or "").strip()
            reason = str(entry.get("reason") or "").strip()
            if not patient_id or not course_id or not reason:
                raise RuntimeError(
                    f"Course manifest is malformed: technical quarantine {index} lacks "
                    "patient, course, or exact reason"
                )
            if (
                entry.get("disposition_type") != "technical_quarantine"
                or entry.get("clinical_exclusion") is not False
            ):
                raise RuntimeError(
                    f"Course manifest is malformed: technical quarantine {index} "
                    "is not explicitly separated from clinical exclusion"
                )
            key = (patient_id, course_id)
            if key in seen or key in quarantine_ids:
                raise RuntimeError(
                    f"Course manifest is malformed: duplicate disposition for "
                    f"{patient_id}/{course_id}"
                )
            quarantine_ids.add(key)
        cohort = {
            "intended_course_count": intended,
            "attempted_course_count": attempted,
            "validated_course_count": validated,
            "technical_quarantine_count": quarantined,
            "technical_quarantines": quarantine_entries,
        }
    else:
        # Legacy manifests did not carry an organize denominator. The current
        # writer always emits v2, but retain deterministic compatibility for
        # historical unit artifacts by treating their explicit list as intended.
        cohort = {
            "intended_course_count": len(courses),
            "attempted_course_count": len(courses),
            "validated_course_count": len(courses),
            "technical_quarantine_count": 0,
            "technical_quarantines": [],
        }
    return courses, cohort


def _read_prefer_parquet(xlsx_path: Path) -> pd.DataFrame | None:
    """Read a current Parquet sidecar when possible, otherwise its workbook."""
    parquet_path = xlsx_path.with_suffix(".parquet")
    use_parquet = parquet_path.exists()
    if use_parquet and xlsx_path.exists():
        try:
            if parquet_path.stat().st_mtime < xlsx_path.stat().st_mtime:
                use_parquet = False
        except OSError:
            pass
    parquet_error = None
    if use_parquet:
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            parquet_error = exc
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    if parquet_error is not None:
        raise RuntimeError(f"{parquet_path.name} is unreadable: {parquet_error}")
    return None


def _read_sentinel(
    path: Path, patient_id: str, course_id: str, allowed_statuses: set[str]
) -> str | None:
    if not path.is_file():
        return f"{patient_id}/{course_id}: required sentinel is missing: {path.name}"
    try:
        text = path.read_text(encoding="utf-8").strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        status = (
            str(decoded.get("status", "")).strip().lower()
            if isinstance(decoded, dict)
            else text.lower()
        )
    except Exception as exc:
        return (
            f"{patient_id}/{course_id}: required sentinel is unreadable: "
            f"{path.name}: {exc}"
        )
    if status not in allowed_statuses:
        allowed = ", ".join(sorted(allowed_statuses))
        return (
            f"{patient_id}/{course_id}: required sentinel is failed or malformed: "
            f"{path.name} has status {status!r}; expected {allowed}"
        )
    return None


def _validate_required_frame(
    frame: pd.DataFrame | None,
    path: Path,
    patient_id: str,
    course_id: str,
    identity_columns: set[str],
) -> str | None:
    if frame is None:
        return f"{patient_id}/{course_id}: required output {path.name} is missing"
    if frame.empty:
        return f"{patient_id}/{course_id}: required output {path.name} is malformed (no rows)"
    if not identity_columns.intersection(frame.columns):
        expected = " or ".join(sorted(identity_columns))
        return (
            f"{patient_id}/{course_id}: required output {path.name} is malformed "
            f"(missing {expected})"
        )
    return None


def _validate_expected_dvh_skip(
    course_dir: Path,
    patient_id: str,
    course_id: str,
    decision: dict,
) -> str | None:
    """Validate the machine-readable no-metrics record for a contracted skip."""
    output_path = course_dir / "dvh_metrics.xlsx"
    if output_path.exists() or output_path.with_suffix(".parquet").exists():
        return (
            f"{patient_id}/{course_id}: contract declares DVH metrics not computed "
            "but a stale DVH metrics output is present"
        )
    qc_path = course_dir / "metadata" / "dvh_qc.json"
    try:
        payload = json.loads(qc_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (
            f"{patient_id}/{course_id}: expected DVH not-computed record is missing: "
            f"{qc_path}"
        )
    except (OSError, json.JSONDecodeError) as exc:
        return (
            f"{patient_id}/{course_id}: expected DVH not-computed record is unreadable: "
            f"{exc}"
        )
    recorded = (payload.get("dose_resolution") or {}).get("dvh")
    if payload.get("status") != "skipped" or recorded != decision:
        return (
            f"{patient_id}/{course_id}: DVH not-computed record disagrees with the "
            "authoritative course contract"
        )
    return None


def _validate_required_inputs(courses):
    errors: list[str] = []
    incomplete: dict[tuple[str, str], list[str]] = {}
    expected_noncomputed: dict[tuple[str, str], str] = {}
    required_frames: dict[tuple[Path, str], pd.DataFrame] = {}
    sentinel_contract = [
        (".dvh_done", {"ok"}),
        (".qc_done", {"ok"}),
        (".custom_models_done", {"disabled", "ok"}),
    ]
    if RADIOMICS_ENABLED:
        sentinel_contract.append((".radiomics_done", {"ok"}))

    for patient_id, course_id, course_dir in courses:
        course_errors: list[str] = []
        if not course_dir.is_dir():
            message = (
                f"{patient_id}/{course_id}: required course directory is missing: {course_dir}"
            )
            errors.append(message)
            incomplete[(patient_id, course_id)] = [message]
            continue
        try:
            from rtpipeline.course_contract import load_course_contract

            course_contract = load_course_contract(course_dir)
        except Exception as exc:
            message = (
                f"{patient_id}/{course_id}: authoritative course contract is missing, "
                f"malformed, or stale: {exc}"
            )
            errors.append(message)
            incomplete[(patient_id, course_id)] = [message]
            continue
        if course_contract.dose_qc.get("pass") is not True:
            message = (
                f"{patient_id}/{course_id}: excluded because authoritative dose QC failed: "
                + "; ".join(str(value) for value in course_contract.dose_qc.get("reasons") or [])
            )
            errors.append(message)
            incomplete[(patient_id, course_id)] = [message]
            continue
        dvh_decision = dict(course_contract.data["dvh"])
        dvh_not_computed = (
            dvh_decision.get("metrics_status") == "not_computed"
            and dvh_decision.get("output") is None
        )
        for sentinel_name, allowed_statuses in sentinel_contract:
            error = _read_sentinel(
                course_dir / sentinel_name,
                patient_id,
                course_id,
                allowed_statuses,
            )
            if error:
                errors.append(error)
                course_errors.append(error)

        required_outputs = []
        if dvh_not_computed:
            error = _validate_expected_dvh_skip(
                course_dir, patient_id, course_id, dvh_decision
            )
            if error:
                errors.append(error)
                course_errors.append(error)
            else:
                required_frames[(course_dir, "dvh")] = pd.DataFrame(
                    {"ROI_Name": pd.Series(dtype="object")}
                )
                expected_noncomputed[(patient_id, course_id)] = str(
                    dvh_decision["reason_code"]
                )
        else:
            required_outputs.append(
                ("dvh", course_dir / "dvh_metrics.xlsx", {"ROI_Name"})
            )
        if RADIOMICS_ENABLED:
            required_outputs.append(
                (
                    "radiomics",
                    course_dir / "radiomics_ct.parquet",
                    {"extraction_arm"},
                )
            )
        for key, output_path, identity_columns in required_outputs:
            try:
                if key == "radiomics":
                    from rtpipeline.radiomics_ct_contract import (
                        read_authoritative_ct_publication,
                        validate_completion_sentinel,
                    )

                    validate_completion_sentinel(
                        course_dir, course_dir / ".radiomics_done"
                    )
                    frame = read_authoritative_ct_publication(output_path)
                else:
                    frame = _read_prefer_parquet(output_path)
            except Exception as exc:
                message = (
                    f"{patient_id}/{course_id}: required output {output_path.name} "
                    f"is unreadable: {exc}"
                )
                errors.append(message)
                course_errors.append(message)
                continue
            error = _validate_required_frame(
                frame,
                output_path,
                patient_id,
                course_id,
                identity_columns,
            )
            if error:
                errors.append(error)
                course_errors.append(error)
            elif frame is not None:
                required_frames[(course_dir, key)] = frame
        if course_errors:
            incomplete[(patient_id, course_id)] = course_errors
    return required_frames, errors, incomplete, expected_noncomputed


def _write_radiomics_denominator_aggregate(courses) -> None:
    """Combine per-course ledgers without replacing course counts by row counts."""
    course_rows = []
    course_roi_rows = []
    for patient_id, course_id, course_dir in courses:
        path = course_dir / "metadata" / "radiomics_roi_ledger.json"
        if not path.exists():
            publication = course_dir / "radiomics_ct.parquet"
            if not publication.exists():
                publication = course_dir / "radiomics_ct.xlsx"
            if not publication.exists():
                raise RuntimeError(f"Radiomics denominator ledger and publication are missing for {patient_id}/{course_id}")
            try:
                frame = pd.read_parquet(publication) if publication.suffix == ".parquet" else pd.read_excel(publication)
                legacy_rows = frame.to_dict("records")
            except Exception as exc:
                raise RuntimeError(f"Radiomics denominator ledger is missing and publication is unreadable for {patient_id}/{course_id}: {exc}") from exc
            course_rows.append({"entity": "COURSE", "course_id": str(course_id), "patient_id": str(patient_id), "screened": 1, "in_scope": 1, "out_of_scope": 0, "adequate_coverage": int(bool(legacy_rows)), "insufficient_coverage": int(not bool(legacy_rows)), "valid_derivation": 0, "technical_exclusion": int(any(str(row.get("extraction_status", "success")) != "success" for row in legacy_rows)), "indeterminate": 0, "extracted": int(bool(legacy_rows)), "reason_code": "extracted" if legacy_rows else "failed_radiomics_extraction"})
            for row in legacy_rows:
                roi = str(row.get("roi_original_name", row.get("roi_name", "")))
                if roi:
                    course_roi_rows.append({"entity": "COURSE_ROI", "course_id": str(course_id), "patient_id": str(patient_id), "roi_name": roi, "disposition": "extracted" if str(row.get("extraction_status", "success")) == "success" else "excluded", "reason_code": "extracted" if str(row.get("extraction_status", "success")) == "success" else "failed_radiomics_extraction"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Radiomics denominator ledger is unreadable for {patient_id}/{course_id}: {exc}") from exc
        rows = payload.get("course", [])
        roi_rows = payload.get("course_roi", [])
        if not any(str(row.get("course_id")) == str(course_id) and str(row.get("patient_id")) == str(patient_id) for row in rows):
            raise RuntimeError(f"Radiomics denominator ledger has no COURSE row for {patient_id}/{course_id}")
        if any(str(row.get("course_id")) == str(course_id) for row in roi_rows):
            course_rows.extend(rows)
            course_roi_rows.extend(roi_rows)
        else:
            raise RuntimeError(f"Radiomics denominator ledger has no COURSE_ROI rows for {patient_id}/{course_id}")
    patient_map: dict[str, dict[str, Any]] = {}
    state_names = ("screened", "in_scope", "out_of_scope", "adequate_coverage", "insufficient_coverage", "valid_derivation", "technical_exclusion", "indeterminate", "extracted")
    for row in course_rows:
        patient = patient_map.setdefault(str(row["patient_id"]), {"entity": "PATIENT", "patient_id": str(row["patient_id"]), "course_count": 0})
        patient["course_count"] += 1
        for state in state_names:
            patient[state] = int(bool(patient.get(state)) or bool(row.get(state)))
    output = {
        "course": course_rows,
        "course_roi": course_roi_rows,
        "patient": sorted(patient_map.values(), key=lambda row: row["patient_id"]),
    }
    destination = RESULTS_DIR / "radiomics_denominator_ledger.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_campaign_attrition(
    courses, incomplete, expected_noncomputed, technical_quarantines=()
) -> None:
    """Record why each course was excluded, so the denominator is defensible."""
    rows = []
    for patient_id, course_id, _ in courses:
        key = (patient_id, course_id)
        failures = incomplete.get(key, [])
        reason = expected_noncomputed.get(key)
        if failures:
            status = "excluded"
            reasons = failures
        elif reason:
            status = "not_computed"
            reasons = [reason]
        else:
            status = "aggregated"
            reasons = []
        rows.append(
            {
                "patient_id": patient_id,
                "course_id": course_id,
                "status": status,
                "reason_count": len(reasons),
                "reasons": " | ".join(reasons),
            }
        )
    for entry in technical_quarantines:
        rows.append(
            {
                "patient_id": entry["patient"],
                "course_id": entry["course"],
                "status": "technical_quarantine",
                "reason_count": 1,
                "reasons": entry["reason"],
                "disposition_type": "technical_quarantine",
                "clinical_exclusion": False,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "campaign_attrition.csv", index=False)


def _write_organization_gate(cohort: dict, *, blocked: bool, reason: str) -> Path:
    payload = dict(cohort)
    payload.update(
        {
            "gate": "pre_scientific_aggregation",
            "status": "blocked" if blocked else "passed",
            "campaign_require_all_courses": CAMPAIGN_REQUIRE_ALL_COURSES,
            "reason": reason,
        }
    )
    path = RESULTS_DIR / "organization_gate.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _campaign_failure_stage_aliases(error: str) -> set[str]:
    mappings = (
        ((".dvh_done", "dvh_metrics"), {"dvh"}),
        ((".qc_done",), {"qc"}),
        ((".custom_models_done",), {"custom_models", "segmentation_custom"}),
        ((".radiomics_done", "radiomics_ct.parquet"), {"radiomics"}),
    )
    for tokens, aliases in mappings:
        if any(token in error for token in tokens):
            return aliases
    return set()


def _valid_failed_campaign_record(
    record: dict[str, Any], *, denominator_mtime_ns: int
) -> bool:
    try:
        returncode = int(record["returncode"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        str(record.get("status") or "").strip() == "failed"
        and bool(str(record.get("detail") or "").strip())
        and returncode != 0
        and int(record.get("_source_mtime_ns") or 0) >= denominator_mtime_ns
    )


def _recorded_campaign_exclusions(incomplete) -> list[dict[str, Any]]:
    """Require explicit failed campaign records for every downstream omission."""

    if not incomplete:
        return []
    records_dir = OUTPUT_DIR / "_campaign_ledger" / "records"
    denominator_paths = [
        path
        for path in (
            OUTPUT_DIR / "_COURSES" / "organize_ledger.json",
            Path(snakemake.input.manifest),  # type: ignore[name-defined]
        )
        if path.exists()
    ]
    denominator_mtime_ns = max(
        (path.stat().st_mtime_ns for path in denominator_paths), default=0
    )
    records: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    if records_dir.is_dir():
        for path in sorted(records_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Campaign ledger record is unreadable: {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"Campaign ledger record is malformed: {path}")
            patient = str(payload.get("patient") or "").strip()
            course = str(payload.get("course") or "").strip()
            stage = str(payload.get("stage") or "").strip()
            if not patient or not course or not stage:
                raise RuntimeError(
                    f"Campaign ledger record lacks patient, course, or stage: {path}"
                )
            payload["_source_mtime_ns"] = path.stat().st_mtime_ns
            payload["_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            records[(patient, course)][stage] = payload

    exclusions: list[dict[str, Any]] = []
    unrecorded: list[str] = []
    for (patient, course), errors in sorted(incomplete.items()):
        required_alias_groups = []
        for error in errors:
            aliases = _campaign_failure_stage_aliases(error)
            if not aliases:
                unrecorded.append(f"{patient}/{course}: {error}")
            else:
                required_alias_groups.append(aliases)

        matched: dict[str, dict[str, Any]] = {}
        course_records = records.get((patient, course), {})
        for aliases in required_alias_groups:
            candidates = [
                course_records[stage]
                for stage in sorted(aliases)
                if stage in course_records
            ]
            valid = [
                record
                for record in candidates
                if _valid_failed_campaign_record(
                    record, denominator_mtime_ns=denominator_mtime_ns
                )
            ]
            if not valid:
                unrecorded.append(
                    f"{patient}/{course}: no failed campaign record with a reason "
                    f"for stage {sorted(aliases)}"
                )
                continue
            record = valid[0]
            matched[str(record["stage"])] = record

        if matched and not any(item.startswith(f"{patient}/{course}:") for item in unrecorded):
            reasons = [
                f"{stage}: {str(record['detail']).strip()}"
                for stage, record in sorted(matched.items())
            ]
            source_hashes = sorted(
                {str(record["_source_sha256"]) for record in matched.values()}
            )
            source_record_sha256 = source_hashes[0]
            if len(source_hashes) > 1:
                source_record_sha256 = hashlib.sha256(
                    "\n".join(source_hashes).encode("ascii")
                ).hexdigest()
            exclusions.append(
                {
                    "patient_id": patient,
                    "course_id": course,
                    "stages": sorted(matched),
                    "reason": " | ".join(reasons),
                    "source_record_sha256": source_record_sha256,
                }
            )

    if unrecorded:
        raise RuntimeError(
            "Campaign aggregation has unrecorded course failure(s); every omission "
            "requires a matching failed campaign-ledger record with a nonempty reason:\n"
            + "".join(f" - {reason}\n" for reason in unrecorded)
        )
    if len(exclusions) != len(incomplete):
        raise RuntimeError(
            "Campaign exclusion ledger does not reconcile with incomplete courses"
        )
    return exclusions


def _declared_output_paths() -> list[Path]:
    names = ["dvh", "dvh_parquet", "fractions", "metadata", "qc"]
    if RADIOMICS_ENABLED:
        names.extend(["radiomics", "radiomics_mr"])
    return [
        Path(str(getattr(snakemake.output, name)))  # type: ignore[name-defined]
        for name in names
    ]


def _invalidate_declared_outputs() -> None:
    paths = _declared_output_paths()
    canonical_radiomics = OUTPUT_DIR / "Data" / "radiomics_all.xlsx"
    protected_radiomics: set[Path] = set()
    if RADIOMICS_ENABLED:
        declared_radiomics = Path(str(snakemake.output.radiomics))  # type: ignore[name-defined]
        for workbook in (declared_radiomics, canonical_radiomics):
            if is_valid_radiomics_cohort_table(workbook):
                protected_radiomics.update({workbook, workbook.with_suffix(".parquet")})
            else:
                paths.extend([workbook, workbook.with_suffix(".parquet")])

    for path in dict.fromkeys(paths):
        if path in protected_radiomics:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to invalidate stale aggregate output {path}: {exc}") from exc


def _add_course_ids(frame: pd.DataFrame, patient_id: str, course_id: str) -> None:
    if "patient_id" in frame.columns:
        frame["patient_id"] = frame["patient_id"].fillna(patient_id)
    else:
        frame.insert(0, "patient_id", patient_id)
    if "course_id" in frame.columns:
        frame["course_id"] = frame["course_id"].fillna(course_id)
    else:
        frame.insert(1, "course_id", course_id)


def _load_body_region_data(course_dir: Path, patient_id: str, course_id: str):
    body_region_path = course_dir / "qc_reports" / "body_regions.json"
    if not body_region_path.exists():
        return {}, None
    try:
        return json.loads(body_region_path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Body region QC read error {patient_id}/{course_id}: {exc}"


def _load_course(course, required_frames):
    patient_id, course_id, course_dir = course
    course_results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    body_region_data, body_region_error = _load_body_region_data(
        course_dir, patient_id, course_id
    )
    if body_region_error:
        errors.append(body_region_error)
    contrast_phase = body_region_data.get("contrast_phase", {}).get(
        "phase", "unknown"
    )
    image_modality = body_region_data.get("image_modality", {}).get(
        "modality", "unknown"
    )
    body_regions_present = [
        region
        for region in ["HEAD_NECK", "THORAX", "ABDOMEN", "PELVIS"]
        if body_region_data.get("body_regions", {}).get(
            f"CONTAINS_{region}", False
        )
    ]
    body_regions = ",".join(body_regions_present) or "unknown"

    frame = required_frames[(course_dir, "dvh")].copy()
    _add_course_ids(frame, patient_id, course_id)
    try:
        from rtpipeline.course_contract import load_course_contract

        technique = str(
            load_course_contract(course_dir).treatment_technique.get("classification")
            or "UNKNOWN"
        ).upper()
    except Exception:
        technique = "UNKNOWN"
    if "treatment_technique" in frame.columns:
        frame["treatment_technique"] = frame["treatment_technique"].fillna(technique)
    else:
        frame["treatment_technique"] = technique
    frame["treatment_technique_source"] = "DICOM_RTPLAN"
    if "structure_cropped" not in frame.columns:
        frame["structure_cropped"] = False
    frame["contrast_phase"] = contrast_phase
    frame["image_modality"] = image_modality
    frame["body_regions"] = body_regions
    course_results["dvh"] = frame

    if RADIOMICS_ENABLED:
        from rtpipeline.radiomics_ct_contract import analysis_eligible_feature_rows

        frame = analysis_eligible_feature_rows(
            required_frames[(course_dir, "radiomics")]
        )
        _add_course_ids(frame, patient_id, course_id)
        if "structure_cropped" not in frame.columns:
            frame["structure_cropped"] = False
        frame["contrast_phase"] = contrast_phase
        frame["image_modality"] = image_modality
        frame["body_regions"] = body_regions
        course_results["radiomics"] = frame

        try:
            frame = _read_prefer_parquet(
                course_dir / "MR" / "radiomics_mr.xlsx"
            )
            if frame is not None:
                _add_course_ids(frame, patient_id, course_id)
                frame["contrast_phase"] = "unknown"
                frame["image_modality"] = "MR"
                frame["body_regions"] = "unknown"
                course_results["radiomics_mr"] = frame
        except Exception as exc:
            errors.append(f"Radiomics MR error {patient_id}/{course_id}: {exc}")

    fractions_path = course_dir / "fractions.xlsx"
    if fractions_path.exists():
        try:
            frame = pd.read_excel(fractions_path)
            _add_course_ids(frame, patient_id, course_id)
            course_results["fractions"] = frame
        except Exception as exc:
            errors.append(f"Fractions error {patient_id}/{course_id}: {exc}")

    metadata_path = course_dir / "metadata" / "case_metadata.xlsx"
    if metadata_path.exists():
        try:
            frame = pd.read_excel(metadata_path)
            _add_course_ids(frame, patient_id, course_id)
            course_results["metadata"] = frame
        except Exception as exc:
            errors.append(f"Metadata error {patient_id}/{course_id}: {exc}")
    return course_results, errors


def _worker_count(courses) -> int:
    if not courses:
        return 1
    effective_cap = min(len(courses), WORKER_BUDGET)
    if AGGREGATION_THREADS is not None:
        return min(effective_cap, AGGREGATION_THREADS)
    return min(effective_cap, AUTO_WORKER_BUDGET)


def _collect_all_frames(courses, required_frames):
    results: dict[str, list[pd.DataFrame]] = defaultdict(list)
    errors: list[str] = []
    if not courses:
        return results, errors

    def load(course):
        return _load_course(course, required_frames)

    with ThreadPoolExecutor(max_workers=_worker_count(courses)) as pool:
        for course_results, course_errors in pool.map(load, courses):
            errors.extend(course_errors)
            for key, frame in course_results.items():
                if frame is not None and not frame.empty:
                    results[key].append(frame)
    return results, errors


def _report_aggregation_errors(
    errors: list[str], heading: str = "warnings"
) -> None:
    if not errors:
        return
    print(f"Aggregation {heading} ({len(errors)}):")
    for error in errors[:20]:
        print(f" - {error}")
    if len(errors) > 20:
        print(f" ... and {len(errors) - 20} more.")
    try:
        error_log_path = RESULTS_DIR / "aggregation_errors.log"
        error_log_path.write_text("".join(f"{error}\n" for error in errors))
        print(f"Full error log written to {error_log_path}")
    except Exception:
        pass


def _write_dvh(
    frames: list[pd.DataFrame],
    courses,
    incomplete=None,
    expected_noncomputed=None,
) -> None:
    """Write a typed cohort DVH table, retaining one failure row per course."""
    combined = build_dvh_aggregate(
        frames,
        courses,
        incomplete=incomplete,
        expected_noncomputed=expected_noncomputed,
    )
    if "Segmentation_Source" not in combined.columns:
        combined["Segmentation_Source"] = "Unknown"
    if "ROI_Name" in combined.columns:
        roi_series = combined["ROI_Name"].astype("string")
    else:
        roi_series = pd.Series([pd.NA] * len(combined), index=combined.index, dtype="string")
        combined.insert(len(combined.columns), "ROI_Name", roi_series)
    combined["_roi_key"] = roi_series.fillna("").str.strip().str.lower()
    computed = combined["row_status"].astype("string") == "computed"
    manual_keys = set(
        combined.loc[
            computed
            & combined["Segmentation_Source"].astype("string").str.lower().eq("manual"),
            "_roi_key",
        ].dropna()
    )
    drop_mask = (
        computed
        & combined["Segmentation_Source"]
        .astype("string")
        .str.lower()
        .isin({"custom", "merged"})
        & combined["_roi_key"].isin(manual_keys)
    )
    if drop_mask.any():
        combined = combined.loc[~drop_mask].copy()
    combined.drop(columns=["_roi_key"], errors="ignore", inplace=True)
    write_dvh_aggregate(combined, Path(str(snakemake.output.dvh)))  # type: ignore[name-defined]


def _write_tabular_outputs(
    all_frames,
    courses,
    incomplete=None,
    expected_noncomputed=None,
    radiomics_cohort_provenance=None,
) -> None:
    _write_dvh(
        all_frames.get("dvh", []),
        courses,
        incomplete=incomplete,
        expected_noncomputed=expected_noncomputed,
    )
    if RADIOMICS_ENABLED:
        radiomics_frames = all_frames.get("radiomics", [])
        if radiomics_frames:
            from rtpipeline.radiomics_ct_contract import (
                publication_key,
                write_ct_publication_atomic,
            )

            combined_radiomics = pd.concat(radiomics_frames, ignore_index=True)
            expected_courses = {
                (str(patient_id), str(course_id))
                for patient_id, course_id, _course_dir in courses
                if (patient_id, course_id) not in (incomplete or {})
            }
            if not {"patient_id", "course_id"}.issubset(combined_radiomics.columns):
                raise RuntimeError(
                    "Radiomics cohort rows lack patient_id or course_id"
                )
            observed_courses = {
                (str(patient), str(course))
                for patient, course in zip(
                    combined_radiomics["patient_id"],
                    combined_radiomics["course_id"],
                )
            }
            if observed_courses != expected_courses:
                missing = sorted(expected_courses - observed_courses)
                unexpected = sorted(observed_courses - expected_courses)
                raise RuntimeError(
                    "Radiomics cohort row identities do not reconcile with extracted "
                    f"courses: missing={missing}, unexpected={unexpected}"
                )
            if radiomics_cohort_provenance is None:
                raise RuntimeError(
                    "Radiomics cohort publication lacks denominator provenance"
                )
            combined_radiomics = attach_radiomics_cohort_provenance(
                combined_radiomics, radiomics_cohort_provenance
            )
            expected_keys = {
                publication_key(record)
                for record in combined_radiomics.to_dict("records")
            }
            write_ct_publication_atomic(
                combined_radiomics,
                Path(snakemake.output.radiomics),  # type: ignore[name-defined]
                expected_keys=expected_keys,
            )
            write_ct_publication_atomic(
                combined_radiomics,
                OUTPUT_DIR / "Data" / "radiomics_all.xlsx",
                expected_keys=expected_keys,
            )
        else:
            pd.DataFrame(
                columns=[
                    "patient_id",
                    "course_id",
                    "roi_name",
                    "structure_cropped",
                ]
            ).to_excel(snakemake.output.radiomics, index=False)  # type: ignore[name-defined]
        radiomics_mr_frames = all_frames.get("radiomics_mr", [])
        if radiomics_mr_frames:
            pd.concat(radiomics_mr_frames, ignore_index=True).to_excel(
                snakemake.output.radiomics_mr, index=False  # type: ignore[name-defined]
            )
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "roi_name"]).to_excel(
                snakemake.output.radiomics_mr, index=False  # type: ignore[name-defined]
            )

    fraction_frames = all_frames.get("fractions", [])
    if fraction_frames:
        pd.concat(fraction_frames, ignore_index=True).to_excel(
            snakemake.output.fractions, index=False  # type: ignore[name-defined]
        )
    else:
        pd.DataFrame(
            columns=["patient_id", "course_id", "treatment_date", "source_path"]
        ).to_excel(snakemake.output.fractions, index=False)  # type: ignore[name-defined]

    metadata_frames = all_frames.get("metadata", [])
    if metadata_frames:
        pd.concat(metadata_frames, ignore_index=True).to_excel(
            snakemake.output.metadata, index=False  # type: ignore[name-defined]
        )
    else:
        pd.DataFrame(columns=["patient_id", "course_id"]).to_excel(
            snakemake.output.metadata, index=False  # type: ignore[name-defined]
        )


def _copy_supplemental_sources() -> None:
    supplemental_sources = {
        "plans.xlsx": OUTPUT_DIR / "Data" / "plans.xlsx",
        "structure_sets.xlsx": OUTPUT_DIR / "Data" / "structure_sets.xlsx",
        "dosimetrics.xlsx": OUTPUT_DIR / "Data" / "dosimetrics.xlsx",
        "fractions.xlsx": OUTPUT_DIR / "Data" / "fractions.xlsx",
        "metadata.xlsx": OUTPUT_DIR / "Data" / "metadata.xlsx",
        "CT_images.xlsx": OUTPUT_DIR / "Data" / "CT_images.xlsx",
    }
    for filename, source_path in supplemental_sources.items():
        if not source_path.exists():
            continue
        destination_path = RESULTS_DIR / filename
        try:
            shutil.copy2(source_path, destination_path)
        except Exception as exc:
            print(
                "[aggregate_results] Warning: failed to copy "
                f"{source_path} -> {destination_path}: {exc}"
            )


def _write_qc(courses) -> None:
    qc_rows = []
    for patient_id, course_id, course_dir in courses:
        qc_dir = course_dir / "qc_reports"
        if not qc_dir.exists():
            continue
        for report_path in qc_dir.glob("*.json"):
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            row = {
                "patient_id": patient_id,
                "course_id": course_id,
                "report_name": report_path.name,
                "overall_status": data.get("overall_status") or data.get("status"),
                "structure_cropping": json.dumps(
                    data.get("checks", {}).get("structure_cropping", {})
                ),
                "checks": json.dumps(data.get("checks", {})),
            }
            if report_path.name == "body_regions.json":
                row["contrast_phase"] = data.get("contrast_phase", {}).get(
                    "phase", "unknown"
                )
                row["image_modality"] = data.get("image_modality", {}).get(
                    "modality", "unknown"
                )
                body_regions = data.get("body_regions", {})
                row["contains_head_neck"] = body_regions.get(
                    "CONTAINS_HEAD_NECK", False
                )
                row["contains_thorax"] = body_regions.get("CONTAINS_THORAX", False)
                row["contains_abdomen"] = body_regions.get(
                    "CONTAINS_ABDOMEN", False
                )
                row["contains_pelvis"] = body_regions.get("CONTAINS_PELVIS", False)
                confidence = data.get("confidence", {})
                row["head_neck_confidence"] = confidence.get("HEAD_NECK", 0.0)
                row["thorax_confidence"] = confidence.get("THORAX", 0.0)
                row["abdomen_confidence"] = confidence.get("ABDOMEN", 0.0)
                row["pelvis_confidence"] = confidence.get("PELVIS", 0.0)
            qc_rows.append(row)
    if qc_rows:
        pd.DataFrame(qc_rows).to_excel(snakemake.output.qc, index=False)  # type: ignore[name-defined]
    else:
        pd.DataFrame(
            columns=["patient_id", "course_id", "report_name", "overall_status"]
        ).to_excel(snakemake.output.qc, index=False)  # type: ignore[name-defined]


log_path = Path(snakemake.log[0])  # type: ignore[name-defined]
_invalidate_declared_outputs()
log_path.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "aggregation_errors.log").unlink(missing_ok=True)
(RESULTS_DIR / "campaign_attrition.csv").unlink(missing_ok=True)
(RESULTS_DIR / "organization_gate.json").unlink(missing_ok=True)

try:
    courses, organize_cohort = _manifest_courses()
except RuntimeError as exc:
    log_path.write_text(f"Aggregation blocked: {exc}\n", encoding="utf-8")
    raise

technical_quarantines = organize_cohort["technical_quarantines"]
organization_gate_blocks = bool(technical_quarantines) and (
    not CAMPAIGN_MODE or CAMPAIGN_REQUIRE_ALL_COURSES
)
if organization_gate_blocks:
    _write_campaign_attrition(courses, {}, {}, technical_quarantines)
    reason = (
        "all intended courses are required, but organize reported "
        f"{len(technical_quarantines)} technical quarantine(s)"
    )
    gate_path = _write_organization_gate(
        organize_cohort, blocked=True, reason=reason
    )
    message = (
        f"Campaign aggregation blocked before scientific aggregation: {reason}. "
        f"Attempted {organize_cohort['attempted_course_count']}, producer-validated "
        f"{organize_cohort['validated_course_count']}. See {gate_path}."
    )
    log_path.write_text(message + "\n", encoding="utf-8")
    raise RuntimeError(message)
_write_organization_gate(
    organize_cohort,
    blocked=False,
    reason=(
        "partial campaign aggregation is explicitly permitted"
        if technical_quarantines
        else "all attempted courses passed organize contract validation"
    ),
)

(
    required_frames,
    required_errors,
    incomplete_courses,
    expected_noncomputed_courses,
) = _validate_required_inputs(courses)
downstream_exclusions: list[dict[str, Any]] = []

if not CAMPAIGN_MODE:
    if required_errors:
        _report_aggregation_errors(required_errors, "required input failures")
        message = "Required aggregation inputs are incomplete:\n" + "".join(
            f" - {error}\n" for error in required_errors
        )
        log_path.write_text(message, encoding="utf-8")
        raise RuntimeError(message.rstrip())
    aggregated_courses = courses
    summary = f"Aggregated {len(courses)} course(s).\n"
elif CAMPAIGN_REQUIRE_ALL_COURSES:
    _write_campaign_attrition(
        courses,
        incomplete_courses,
        expected_noncomputed_courses,
        technical_quarantines,
    )
    if required_errors:
        _report_aggregation_errors(required_errors, "required input failures")
        gate_reason = (
            "all intended courses are required and one or more producer-validated "
            "courses have incomplete required inputs"
        )
        gate_path = _write_organization_gate(
            organize_cohort, blocked=True, reason=gate_reason
        )
        message = (
            "Campaign aggregation blocked before scientific aggregation because "
            f"{gate_reason}. See {gate_path}:\n"
            + "".join(f" - {error}\n" for error in required_errors)
        )
        log_path.write_text(message, encoding="utf-8")
        raise RuntimeError(message.rstrip())
    aggregated_courses = courses
    summary = (
        f"Aggregated all {organize_cohort['attempted_course_count']} intended course(s).\n"
    )
else:
    # Campaign mode aggregates the courses that completed and reports the rest as
    # declared attrition. Every exclusion keeps its reason in campaign_attrition.csv,
    # so the denominator is auditable rather than silently smaller.
    downstream_exclusions = _recorded_campaign_exclusions(incomplete_courses)
    for exclusion in downstream_exclusions:
        key = (exclusion["patient_id"], exclusion["course_id"])
        incomplete_courses[key] = [exclusion["reason"]]
    _write_campaign_attrition(
        courses,
        incomplete_courses,
        expected_noncomputed_courses,
        technical_quarantines,
    )
    aggregated_courses = [
        course for course in courses if (course[0], course[1]) not in incomplete_courses
    ]
    total = organize_cohort["attempted_course_count"]
    completed = len(aggregated_courses)
    fraction = (completed / total) if total else 0.0

    if required_errors:
        _report_aggregation_errors(required_errors, "excluded course failures")

    # A campaign that lost most of its courses is a broken campaign, not a small
    # cohort, so aggregation still fails closed below the declared floor.
    if completed == 0:
        message = (
            f"Campaign aggregation blocked: no course completed out of {total}. "
            f"See {RESULTS_DIR / 'campaign_attrition.csv'}."
        )
        log_path.write_text(message + "\n", encoding="utf-8")
        raise RuntimeError(message)
    if fraction < CAMPAIGN_MIN_COMPLETION_FRACTION:
        message = (
            f"Campaign aggregation blocked: only {completed} of {total} courses "
            f"completed ({fraction:.1%}), below the declared floor of "
            f"{CAMPAIGN_MIN_COMPLETION_FRACTION:.1%}. "
            f"See {RESULTS_DIR / 'campaign_attrition.csv'}."
        )
        log_path.write_text(message + "\n", encoding="utf-8")
        raise RuntimeError(message)

    summary = (
        f"Aggregated {completed} of {total} intended course(s) "
        f"({fraction:.1%}); {total - completed} excluded with recorded reasons "
        f"in campaign_attrition.csv.\n"
    )

radiomics_cohort_provenance = None
if RADIOMICS_ENABLED:
    denominator_source_path = OUTPUT_DIR / "_COURSES" / "organize_ledger.json"
    if not denominator_source_path.exists():
        denominator_source_path = Path(snakemake.input.manifest)  # type: ignore[name-defined]
    denominator_source_sha256 = hashlib.sha256(
        denominator_source_path.read_bytes()
    ).hexdigest()
    provenance_technical_quarantines = [
        {**entry, "source_record_sha256": denominator_source_sha256}
        for entry in technical_quarantines
    ]
    radiomics_cohort_provenance = build_radiomics_cohort_provenance(
        intended_count=int(organize_cohort["intended_course_count"]),
        validated_count=int(organize_cohort["validated_course_count"]),
        extracted_courses=[
            (patient_id, course_id)
            for patient_id, course_id, _course_dir in aggregated_courses
        ],
        technical_quarantines=provenance_technical_quarantines,
        downstream_exclusions=downstream_exclusions,
        denominator_source_sha256=denominator_source_sha256,
    )

all_frames, aggregation_errors = _collect_all_frames(aggregated_courses, required_frames)
_report_aggregation_errors(aggregation_errors)
if RADIOMICS_ENABLED:
    _write_radiomics_denominator_aggregate(aggregated_courses)
_write_tabular_outputs(
    all_frames,
    courses,
    incomplete=incomplete_courses,
    expected_noncomputed=expected_noncomputed_courses,
    radiomics_cohort_provenance=radiomics_cohort_provenance,
)
_copy_supplemental_sources()
_write_qc(aggregated_courses)
log_path.write_text(summary, encoding="utf-8")
