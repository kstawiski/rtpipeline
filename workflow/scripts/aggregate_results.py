"""Aggregate per-course RTpipeline artifacts into cohort workbooks."""

import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd  # type: ignore


OUTPUT_DIR = Path(snakemake.params.output_dir)  # type: ignore[name-defined]
RESULTS_DIR = Path(snakemake.params.results_dir)  # type: ignore[name-defined]
RADIOMICS_ENABLED = bool(snakemake.params.radiomics_enabled)  # type: ignore[name-defined]
CAMPAIGN_MODE = bool(getattr(snakemake.params, "campaign_mode", False))  # type: ignore[name-defined]
CAMPAIGN_MIN_COMPLETION_FRACTION = float(
    getattr(snakemake.params, "campaign_min_completion_fraction", 0.5)  # type: ignore[name-defined]
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
    return courses


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
        status = path.read_text(encoding="utf-8").strip().lower()
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


def _validate_required_inputs(courses):
    errors: list[str] = []
    incomplete: dict[tuple[str, str], list[str]] = {}
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

        required_outputs = [
            ("dvh", course_dir / "dvh_metrics.xlsx", {"ROI_Name"}),
        ]
        if RADIOMICS_ENABLED:
            required_outputs.append(
                (
                    "radiomics",
                    course_dir / "radiomics_ct.xlsx",
                    {"roi_name", "roi_original_name"},
                )
            )
        for key, output_path, identity_columns in required_outputs:
            try:
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
    return required_frames, errors, incomplete


def _write_campaign_attrition(courses, incomplete) -> None:
    """Record why each course was excluded, so the denominator is defensible."""
    rows = [
        {
            "patient_id": patient_id,
            "course_id": course_id,
            "status": "excluded" if (patient_id, course_id) in incomplete else "aggregated",
            "reason_count": len(incomplete.get((patient_id, course_id), [])),
            "reasons": " | ".join(incomplete.get((patient_id, course_id), [])),
        }
        for patient_id, course_id, _ in courses
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "campaign_attrition.csv", index=False)


def _declared_output_paths() -> list[Path]:
    names = ["dvh", "fractions", "metadata", "qc"]
    if RADIOMICS_ENABLED:
        names.extend(["radiomics", "radiomics_mr"])
    return [
        Path(str(getattr(snakemake.output, name)))  # type: ignore[name-defined]
        for name in names
    ]


def _invalidate_declared_outputs() -> None:
    for path in _declared_output_paths():
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
    if "structure_cropped" not in frame.columns:
        frame["structure_cropped"] = False
    frame["contrast_phase"] = contrast_phase
    frame["image_modality"] = image_modality
    frame["body_regions"] = body_regions
    course_results["dvh"] = frame

    if RADIOMICS_ENABLED:
        frame = required_frames[(course_dir, "radiomics")].copy()
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


def _write_dvh(frames: list[pd.DataFrame]) -> None:
    if not frames:
        pd.DataFrame(
            columns=["patient_id", "course_id", "ROI_Name", "structure_cropped"]
        ).to_excel(snakemake.output.dvh, index=False)  # type: ignore[name-defined]
        return
    combined = pd.concat(frames, ignore_index=True)
    if "Segmentation_Source" not in combined.columns:
        combined["Segmentation_Source"] = "Unknown"
    if "ROI_Name" in combined.columns:
        roi_series = combined["ROI_Name"].astype(str)
    else:
        roi_series = pd.Series([""] * len(combined), index=combined.index)
        combined.insert(len(combined.columns), "ROI_Name", roi_series)
    combined["_roi_key"] = roi_series.str.strip().str.lower()
    manual_keys = set(
        combined.loc[
            combined["Segmentation_Source"].astype(str).str.lower() == "manual",
            "_roi_key",
        ].dropna()
    )
    drop_mask = (
        combined["Segmentation_Source"]
        .astype(str)
        .str.lower()
        .isin({"custom", "merged"})
        & combined["_roi_key"].isin(manual_keys)
    )
    if drop_mask.any():
        combined = combined.loc[~drop_mask].copy()
    combined.drop(columns=["_roi_key"], errors="ignore", inplace=True)
    combined.to_excel(snakemake.output.dvh, index=False)  # type: ignore[name-defined]


def _write_tabular_outputs(all_frames) -> None:
    _write_dvh(all_frames.get("dvh", []))
    if RADIOMICS_ENABLED:
        radiomics_frames = all_frames.get("radiomics", [])
        if radiomics_frames:
            pd.concat(radiomics_frames, ignore_index=True).to_excel(
                snakemake.output.radiomics, index=False  # type: ignore[name-defined]
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

try:
    courses = _manifest_courses()
except RuntimeError as exc:
    log_path.write_text(f"Aggregation blocked: {exc}\n", encoding="utf-8")
    raise

required_frames, required_errors, incomplete_courses = _validate_required_inputs(courses)

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
else:
    # Campaign mode aggregates the courses that completed and reports the rest as
    # declared attrition. Every exclusion keeps its reason in campaign_attrition.csv,
    # so the denominator is auditable rather than silently smaller.
    _write_campaign_attrition(courses, incomplete_courses)
    aggregated_courses = [
        course for course in courses if (course[0], course[1]) not in incomplete_courses
    ]
    total = len(courses)
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
        f"Aggregated {completed} of {total} course(s) "
        f"({fraction:.1%}); {total - completed} excluded with recorded reasons "
        f"in campaign_attrition.csv.\n"
    )

all_frames, aggregation_errors = _collect_all_frames(aggregated_courses, required_frames)
_report_aggregation_errors(aggregation_errors)
_write_tabular_outputs(all_frames)
_copy_supplemental_sources()
_write_qc(aggregated_courses)
log_path.write_text(summary, encoding="utf-8")
