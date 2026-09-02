from __future__ import annotations

"""Publication provenance for cohort-level radiomics tables."""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RADIOMICS_COHORT_PROVENANCE_SCHEMA = "rtpipeline-radiomics-cohort-v1"
PROVENANCE_SCHEMA_COLUMN = "radiomics_cohort_provenance_schema"
DENOMINATOR_SOURCE_SHA256_COLUMN = "radiomics_denominator_source_sha256"
INTENDED_COUNT_COLUMN = "radiomics_cohort_intended_n"
VALIDATED_COUNT_COLUMN = "radiomics_cohort_validated_n"
EXTRACTED_COUNT_COLUMN = "radiomics_cohort_extracted_n"
EXCLUDED_COUNT_COLUMN = "radiomics_cohort_excluded_n"
TECHNICAL_QUARANTINE_COUNT_COLUMN = (
    "radiomics_cohort_technical_quarantine_n"
)
DOWNSTREAM_EXCLUSION_COUNT_COLUMN = "radiomics_cohort_downstream_exclusion_n"
EXCLUSIONS_COLUMN = "radiomics_cohort_exclusions_json"

PROVENANCE_COLUMNS = (
    PROVENANCE_SCHEMA_COLUMN,
    DENOMINATOR_SOURCE_SHA256_COLUMN,
    INTENDED_COUNT_COLUMN,
    VALIDATED_COUNT_COLUMN,
    EXTRACTED_COUNT_COLUMN,
    EXCLUDED_COUNT_COLUMN,
    TECHNICAL_QUARANTINE_COUNT_COLUMN,
    DOWNSTREAM_EXCLUSION_COUNT_COLUMN,
    EXCLUSIONS_COLUMN,
)


def _identity(entry: Mapping[str, Any]) -> tuple[str, str]:
    patient = str(entry.get("patient_id") or entry.get("patient") or "").strip()
    course = str(
        entry.get("course_id") or entry.get("course_key") or entry.get("course") or ""
    ).strip()
    if not patient or not course:
        raise ValueError("radiomics cohort disposition requires patient and course")
    return patient, course


def _normalize_exclusion(
    entry: Mapping[str, Any], *, source: str, disposition_type: str
) -> dict[str, Any]:
    patient, course = _identity(entry)
    reason = str(entry.get("reason") or entry.get("detail") or "").strip()
    source_sha256 = str(entry.get("source_record_sha256") or "").strip().lower()
    if not reason:
        raise ValueError(f"radiomics cohort exclusion {patient}/{course} has no reason")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError(
            f"radiomics cohort exclusion {patient}/{course} lacks a source hash"
        )
    normalized: dict[str, Any] = {
        "patient_id": patient,
        "course_id": course,
        "disposition_type": disposition_type,
        "source": source,
        "reason": reason,
        "source_record_sha256": source_sha256,
    }
    stages = entry.get("stages")
    if stages:
        normalized["stages"] = sorted({str(stage).strip() for stage in stages if str(stage).strip()})
    return normalized


def build_radiomics_cohort_provenance(
    *,
    intended_count: int,
    validated_count: int,
    extracted_courses: Iterable[tuple[str, str]],
    technical_quarantines: Sequence[Mapping[str, Any]] = (),
    downstream_exclusions: Sequence[Mapping[str, Any]] = (),
    denominator_source_sha256: str,
) -> dict[str, Any]:
    """Build and reconcile the denominator carried by every aggregate row."""

    extracted = sorted(
        {(str(patient).strip(), str(course).strip()) for patient, course in extracted_courses}
    )
    if any(not patient or not course for patient, course in extracted):
        raise ValueError("extracted radiomics courses require patient and course")

    exclusions = [
        _normalize_exclusion(
            entry,
            source="organize_ledger",
            disposition_type="technical_quarantine",
        )
        for entry in technical_quarantines
    ]
    exclusions.extend(
        _normalize_exclusion(
            entry,
            source="campaign_ledger",
            disposition_type="downstream_technical_exclusion",
        )
        for entry in downstream_exclusions
    )
    exclusions.sort(
        key=lambda entry: (
            entry["patient_id"],
            entry["course_id"],
            entry["source"],
        )
    )

    identities = {
        (entry["patient_id"], entry["course_id"]) for entry in exclusions
    }
    if len(identities) != len(exclusions):
        raise ValueError("radiomics cohort exclusions contain duplicate course dispositions")
    overlap = identities.intersection(extracted)
    if overlap:
        formatted = ", ".join(f"{patient}/{course}" for patient, course in sorted(overlap))
        raise ValueError(f"radiomics courses are both extracted and excluded: {formatted}")

    technical_count = len(technical_quarantines)
    downstream_count = len(downstream_exclusions)
    excluded_count = len(exclusions)
    extracted_count = len(extracted)
    if validated_count != extracted_count + downstream_count:
        raise ValueError(
            "radiomics validated denominator does not reconcile with extracted and "
            "downstream-excluded courses"
        )
    if intended_count != validated_count + technical_count:
        raise ValueError(
            "radiomics intended denominator does not reconcile with validated and "
            "organize-quarantined courses"
        )
    if intended_count != extracted_count + excluded_count:
        raise ValueError(
            "radiomics intended denominator does not reconcile with extracted and "
            "excluded courses"
        )

    ledger_sha256 = str(denominator_source_sha256 or "").strip().lower()
    if len(ledger_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in ledger_sha256
    ):
        raise ValueError("radiomics cohort provenance lacks an organize-ledger hash")

    return {
        "schema": RADIOMICS_COHORT_PROVENANCE_SCHEMA,
        "denominator_source_sha256": ledger_sha256,
        "intended_course_count": int(intended_count),
        "validated_course_count": int(validated_count),
        "extracted_course_count": extracted_count,
        "excluded_course_count": excluded_count,
        "technical_quarantine_count": technical_count,
        "downstream_exclusion_count": downstream_count,
        "exclusions": exclusions,
    }


def attach_radiomics_cohort_provenance(dataframe: Any, provenance: Mapping[str, Any]) -> Any:
    """Return a copy whose every row carries one reconciled cohort denominator."""

    validate_radiomics_cohort_provenance(provenance)
    output = dataframe.copy()
    exclusions_json = json.dumps(
        provenance["exclusions"], sort_keys=True, separators=(",", ":")
    )
    values = {
        PROVENANCE_SCHEMA_COLUMN: provenance["schema"],
        DENOMINATOR_SOURCE_SHA256_COLUMN: provenance["denominator_source_sha256"],
        INTENDED_COUNT_COLUMN: provenance["intended_course_count"],
        VALIDATED_COUNT_COLUMN: provenance["validated_course_count"],
        EXTRACTED_COUNT_COLUMN: provenance["extracted_course_count"],
        EXCLUDED_COUNT_COLUMN: provenance["excluded_course_count"],
        TECHNICAL_QUARANTINE_COUNT_COLUMN: provenance[
            "technical_quarantine_count"
        ],
        DOWNSTREAM_EXCLUSION_COUNT_COLUMN: provenance[
            "downstream_exclusion_count"
        ],
        EXCLUSIONS_COLUMN: exclusions_json,
    }
    for column, value in values.items():
        output[column] = value
    return output


def validate_radiomics_cohort_provenance(provenance: Mapping[str, Any]) -> None:
    if provenance.get("schema") != RADIOMICS_COHORT_PROVENANCE_SCHEMA:
        raise ValueError("unsupported radiomics cohort provenance schema")
    ledger_sha256 = str(provenance.get("denominator_source_sha256") or "").lower()
    if len(ledger_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in ledger_sha256
    ):
        raise ValueError("radiomics cohort provenance has an invalid ledger hash")
    try:
        intended = int(provenance["intended_course_count"])
        validated = int(provenance["validated_course_count"])
        extracted = int(provenance["extracted_course_count"])
        excluded = int(provenance["excluded_course_count"])
        technical = int(provenance["technical_quarantine_count"])
        downstream = int(provenance["downstream_exclusion_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("radiomics cohort provenance counts are invalid") from exc
    exclusions = provenance.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("radiomics cohort provenance exclusions must be a list")
    if any(count < 0 for count in (intended, validated, extracted, excluded, technical, downstream)):
        raise ValueError("radiomics cohort provenance counts cannot be negative")
    if intended != validated + technical:
        raise ValueError("radiomics intended and organize disposition counts disagree")
    if validated != extracted + downstream:
        raise ValueError("radiomics validated and downstream disposition counts disagree")
    if intended != extracted + excluded:
        raise ValueError("radiomics intended and final disposition counts disagree")
    if excluded != technical + downstream or excluded != len(exclusions):
        raise ValueError("radiomics exclusion counts and records disagree")
    seen: set[tuple[str, str]] = set()
    for entry in exclusions:
        if not isinstance(entry, Mapping):
            raise ValueError("radiomics cohort exclusion is not an object")
        identity = _identity(entry)
        reason = str(entry.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"radiomics cohort exclusion {identity[0]}/{identity[1]} has no reason")
        source_sha256 = str(entry.get("source_record_sha256") or "").lower()
        if len(source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_sha256
        ):
            raise ValueError(
                f"radiomics cohort exclusion {identity[0]}/{identity[1]} has an invalid source hash"
            )
        if identity in seen:
            raise ValueError("radiomics cohort provenance repeats an excluded course")
        seen.add(identity)


def provenance_from_frame(dataframe: Any) -> dict[str, Any]:
    """Read and validate the single provenance carried by an aggregate table."""

    if dataframe is None or dataframe.empty:
        raise ValueError("radiomics cohort table is empty")
    missing = [column for column in PROVENANCE_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "radiomics cohort table lacks denominator provenance columns: "
            + ", ".join(missing)
        )

    unique: dict[str, Any] = {}
    for column in PROVENANCE_COLUMNS:
        values = dataframe[column].dropna().unique().tolist()
        if len(values) != 1:
            raise ValueError(
                f"radiomics cohort provenance column {column} is not constant"
            )
        unique[column] = values[0]
    try:
        exclusions = json.loads(str(unique[EXCLUSIONS_COLUMN]))
    except json.JSONDecodeError as exc:
        raise ValueError("radiomics cohort exclusions JSON is invalid") from exc
    provenance = {
        "schema": str(unique[PROVENANCE_SCHEMA_COLUMN]),
        "denominator_source_sha256": str(unique[DENOMINATOR_SOURCE_SHA256_COLUMN]),
        "intended_course_count": int(unique[INTENDED_COUNT_COLUMN]),
        "validated_course_count": int(unique[VALIDATED_COUNT_COLUMN]),
        "extracted_course_count": int(unique[EXTRACTED_COUNT_COLUMN]),
        "excluded_course_count": int(unique[EXCLUDED_COUNT_COLUMN]),
        "technical_quarantine_count": int(
            unique[TECHNICAL_QUARANTINE_COUNT_COLUMN]
        ),
        "downstream_exclusion_count": int(
            unique[DOWNSTREAM_EXCLUSION_COUNT_COLUMN]
        ),
        "exclusions": exclusions,
    }
    validate_radiomics_cohort_provenance(provenance)

    course_column = next(
        (
            column
            for column in ("course_key", "course_id")
            if column in dataframe.columns and dataframe[column].notna().any()
        ),
        None,
    )
    if "patient_id" not in dataframe.columns or course_column is None:
        raise ValueError("radiomics cohort table lacks row-level course identity")
    identity_rows = dataframe.loc[
        dataframe["patient_id"].notna() & dataframe[course_column].notna(),
        ["patient_id", course_column],
    ]
    observed = {
        (str(patient).strip(), str(course).strip())
        for patient, course in identity_rows.itertuples(index=False, name=None)
        if str(patient).strip() and str(course).strip()
    }
    if len(observed) != provenance["extracted_course_count"]:
        raise ValueError(
            "radiomics cohort row identities do not match the extracted denominator"
        )
    excluded = {
        (entry["patient_id"], entry["course_id"])
        for entry in provenance["exclusions"]
    }
    if observed.intersection(excluded):
        raise ValueError("radiomics cohort rows include an excluded course")
    return provenance


def is_valid_radiomics_cohort_table(path: Path) -> bool:
    """Return whether an existing workbook or Parquet has valid denominator data."""

    import pandas as pd

    candidate = Path(path)
    parquet = candidate if candidate.suffix == ".parquet" else candidate.with_suffix(".parquet")
    workbook = candidate.with_suffix(".xlsx")
    try:
        if parquet.exists():
            provenance_from_frame(pd.read_parquet(parquet, engine="pyarrow"))
            return True
        if workbook.exists():
            provenance_from_frame(pd.read_excel(workbook, engine="openpyxl"))
            return True
    except Exception:
        return False
    return False
