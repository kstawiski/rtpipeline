"""Auditable prescription evidence from the Kopernik treatment register.

Supported grammar
-----------------
The parser accepts one or more named treatment sites. Each site must carry one
of these dose expressions::

    [do dawki] TOTAL Gy/p.ref po PER_FRACTION Gy
    [do dawki] TOTAL Gy w N frakcjach po PER_FRACTION Gy
    [do dawki] TOTAL Gy/p.ref po PER_1 Gy (N_1 frakcji)
        i PER_2 Gy (N_2 frakcji) [...]

A site normally follows ``na obszar`` or ``obszar``. The first site may instead
follow a named technique such as IMRT, VMAT, or SBRT. Additional sites may be
introduced by ``oraz`` or ``i``. The exact source text and exact site text are
retained in the evidence object.

Refusal cases
-------------
The parser refuses empty descriptions, missing named sites, missing totals,
missing per-fraction doses, nonpositive values, nonintegral implied fraction
counts, stated fraction-count mismatches, incomplete multiphase counts,
unsupported extra dose expressions, overlapping site clauses, and any other
partially parsed dose-bearing text. Record matching refuses missing dates,
invalid windows, absent exact temporal evidence, and tied matches. Distinct
multi-site totals remain named per-site evidence and never become one course
scalar.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


CLINICAL_EVIDENCE_SCHEMA = "rtpipeline-clinical-prescription-evidence-v1"
CLINICAL_PARSER_VERSION = "kopernik-opis-leczenia-v1"
CLINICAL_SOURCE_FORMAT = "kopernik-rt-treatments-xlsx-v1"
CLINICAL_RESOLVED_SCOPE = "COURSE_TOTAL_CLINICAL_RECORD"
CLINICAL_EVIDENCE_REGENERATION_SCHEMA = (
    "rtpipeline-clinical-prescription-evidence-regeneration-v1"
)


def clinical_evidence_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded evidence content without a recursively nested receipt."""

    payload = copy.deepcopy(dict(evidence))
    payload.pop("regeneration_provenance", None)
    return payload


def clinical_evidence_content_sha256(evidence: Mapping[str, Any]) -> str:
    """Hash the bounded clinical-evidence payload in canonical JSON form."""

    encoded = json.dumps(
        clinical_evidence_payload(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_clinical_evidence_regeneration(
    evidence: dict[str, Any],
    previous_evidence: object,
) -> dict[str, Any]:
    """Link regenerated evidence to the exact prior clinical-evidence payload."""

    if not isinstance(previous_evidence, Mapping):
        return evidence
    previous_payload = clinical_evidence_payload(previous_evidence)
    regenerated = dict(evidence)
    regenerated["regeneration_provenance"] = {
        "schema": CLINICAL_EVIDENCE_REGENERATION_SCHEMA,
        "authority": "organize",
        "reason": "organize_resume_republication",
        "previous_evidence_payload_sha256": clinical_evidence_content_sha256(
            previous_payload
        ),
        "previous_evidence_payload": previous_payload,
        "current_source": dict(evidence.get("source") or {}),
        "current_dicom_snapshot": dict(evidence.get("dicom") or {}),
    }
    return regenerated

_REQUIRED_COLUMNS = (
    "ID",
    "Data rozp lecz",
    "Data zak lecz",
    "Rozpoznanie wg ICD 10",
    "Rodzaj Leczenia",
    "Rozpoznanie",
    "Zalecenia",
    "Opis leczenia",
)

_NUMBER = r"\d+(?:[,.]\d+)?"
_DOSE_TOKEN_RE = re.compile(rf"(?P<value>{_NUMBER})\s*gy\b", re.IGNORECASE)
_PER_FRACTION_RE = re.compile(
    rf"\b(?P<connector>po|i)\s*(?P<dose>{_NUMBER})\s*gy\b"
    rf"(?:\s*\(\s*(?P<count>\d+)\s*frakcj\w*\s*\))?",
    re.IGNORECASE,
)
_STATED_BEFORE_RE = re.compile(r"\bw\s*(?P<count>\d+)\s*frakcj\w*\s*$", re.IGNORECASE)
_SITE_MARKER_RE = re.compile(r"\b(?:na\s+)?obszar\s+", re.IGNORECASE)
_TECHNIQUE_RE = re.compile(
    r"\b(?:imrt|vmat|sbrt|srt|3dcrt|3d|2d|teleradioterapi\w*)\b",
    re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"\b(?:oraz|i)\b", re.IGNORECASE)


class ClinicalPrescriptionSourceError(RuntimeError):
    """The configured clinical source cannot be audited safely."""


@dataclass(frozen=True)
class ClinicalRecord:
    patient_id: str
    start_date: date | None
    end_date: date | None
    diagnosis_icd10: str
    treatment_type: str
    diagnosis: str
    recommendations: str
    description: str
    workbook_path: str
    workbook_sha256: str
    sheet_name: str
    excel_row: int
    record_id: str

    def audit_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "workbook_path": self.workbook_path,
            "workbook_sha256": self.workbook_sha256,
            "sheet_name": self.sheet_name,
            "excel_row": self.excel_row,
            "patient_id": self.patient_id,
            "treatment_start_date": (
                self.start_date.isoformat() if self.start_date is not None else None
            ),
            "treatment_end_date": (
                self.end_date.isoformat() if self.end_date is not None else None
            ),
            "diagnosis_icd10": self.diagnosis_icd10,
            "treatment_type": self.treatment_type,
            "diagnosis": self.diagnosis,
            "recommendations": self.recommendations,
            "parsed_field": "Opis leczenia",
            "source_text": self.description,
        }


@dataclass(frozen=True)
class ClinicalRecordIndex:
    source_path: Path
    workbook_sha256: str
    sheet_name: str
    row_count: int
    records_by_patient: Mapping[str, tuple[ClinicalRecord, ...]]

    def records_for(self, patient_id: object) -> tuple[ClinicalRecord, ...]:
        return self.records_by_patient.get(_patient_text(patient_id), ())

    def source_dict(self) -> dict[str, Any]:
        return {
            "format": CLINICAL_SOURCE_FORMAT,
            "workbook_path": str(self.source_path),
            "workbook_sha256": self.workbook_sha256,
            "sheet_name": self.sheet_name,
            "row_count": self.row_count,
        }


def _ascii_lower(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(math.isnan(value)):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text.casefold() in {"nan", "nat", "none"}:
        return ""
    return text


def _patient_text(value: object) -> str:
    text = _cell_text(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _cell_text(value).strip()
    if not text:
        return None
    try:
        import pandas as pd

        parsed = pd.to_datetime(text, errors="raise")
        return parsed.date()
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ClinicalPrescriptionSourceError(
            f"cannot hash configured clinical prescription source {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def load_kopernik_treatment_records(
    path: Path | str,
    *,
    sheet_name: str | int = 0,
) -> ClinicalRecordIndex:
    """Load the configured workbook once and retain row-level provenance."""

    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise ClinicalPrescriptionSourceError(
            f"configured clinical prescription source is missing: {source}"
        )
    workbook_sha256 = _file_sha256(source)
    try:
        import pandas as pd

        excel = pd.ExcelFile(source)
        selected_sheet = (
            excel.sheet_names[int(sheet_name)]
            if isinstance(sheet_name, int)
            else str(sheet_name)
        )
        frame = pd.read_excel(excel, sheet_name=selected_sheet, dtype=object)
    except Exception as exc:
        raise ClinicalPrescriptionSourceError(
            f"cannot read configured clinical prescription source {source}: {exc}"
        ) from exc
    missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ClinicalPrescriptionSourceError(
            "configured clinical prescription source is missing required columns: "
            + ", ".join(missing)
        )

    by_patient: dict[str, list[ClinicalRecord]] = {}
    for row_position, (_, row) in enumerate(frame.iterrows(), start=2):
        patient_id = _patient_text(row["ID"])
        if not patient_id:
            continue
        excel_row = row_position
        record_id = hashlib.sha256(
            f"{workbook_sha256}\0{selected_sheet}\0{excel_row}".encode("utf-8")
        ).hexdigest()
        record = ClinicalRecord(
            patient_id=patient_id,
            start_date=_date_value(row["Data rozp lecz"]),
            end_date=_date_value(row["Data zak lecz"]),
            diagnosis_icd10=_cell_text(row["Rozpoznanie wg ICD 10"]),
            treatment_type=_cell_text(row["Rodzaj Leczenia"]),
            diagnosis=_cell_text(row["Rozpoznanie"]),
            recommendations=_cell_text(row["Zalecenia"]),
            description=_cell_text(row["Opis leczenia"]),
            workbook_path=str(source),
            workbook_sha256=workbook_sha256,
            sheet_name=selected_sheet,
            excel_row=excel_row,
            record_id=record_id,
        )
        by_patient.setdefault(patient_id, []).append(record)
    frozen = {
        patient: tuple(
            sorted(
                records,
                key=lambda item: (
                    item.start_date or date.min,
                    item.end_date or date.min,
                    item.excel_row,
                ),
            )
        )
        for patient, records in by_patient.items()
    }
    return ClinicalRecordIndex(
        source_path=source,
        workbook_sha256=workbook_sha256,
        sheet_name=selected_sheet,
        row_count=len(frame),
        records_by_patient=frozen,
    )


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", "."))


def _dose_total_candidates(text: str, normalized: str) -> list[re.Match[str]]:
    candidates: list[re.Match[str]] = []
    for match in _DOSE_TOKEN_RE.finditer(normalized):
        before = normalized[max(0, match.start() - 28) : match.start()]
        after = normalized[match.end() : match.end() + 40]
        if (
            re.search(r"do\s+dawki\s*$", before)
            or re.match(r"\s*/\s*p\.?\s*ref\.?", after)
            or re.match(r"\s*w\s*\d+\s*frakcj", after)
        ):
            candidates.append(match)
    return candidates


def _site_name(prefix: str) -> str | None:
    normalized = _ascii_lower(prefix)
    markers = list(_SITE_MARKER_RE.finditer(normalized))
    if markers:
        start = markers[-1].end()
    else:
        techniques = list(_TECHNIQUE_RE.finditer(normalized))
        if not techniques:
            return None
        start = techniques[-1].end()
    value = prefix[start:]
    value = re.sub(r"\bdo\s+dawki\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,.;:-")
    if not value or not any(ch.isalpha() for ch in value):
        return None
    return value


def _fractionation_for_total(
    *,
    text: str,
    normalized: str,
    total_match: re.Match[str],
    clause_end: int,
) -> tuple[list[dict[str, Any]], str | None]:
    tail_norm = normalized[total_match.end() : clause_end]
    tail_text = text[total_match.end() : clause_end]
    total = _decimal(total_match.group("value"))
    if total <= 0:
        return [], "NONPOSITIVE_TOTAL_DOSE"

    phase_matches = list(_PER_FRACTION_RE.finditer(tail_norm))
    if not phase_matches:
        return [], "MISSING_PER_FRACTION_DOSE"

    phases: list[dict[str, Any]] = []
    stated_before: int | None = None
    before_first = tail_norm[: phase_matches[0].start()]
    before_match = _STATED_BEFORE_RE.search(before_first)
    if before_match:
        stated_before = int(before_match.group("count"))

    for index, match in enumerate(phase_matches):
        dose = _decimal(match.group("dose"))
        if dose <= 0:
            return [], "NONPOSITIVE_PER_FRACTION_DOSE"
        count_text = match.group("count")
        count = int(count_text) if count_text is not None else None
        if index == 0 and stated_before is not None:
            if count is not None and count != stated_before:
                return [], "CONFLICTING_STATED_FRACTION_COUNTS"
            count = stated_before
        phases.append(
            {
                "fraction_count": count,
                "dose_per_fraction_gy": float(dose),
                "phase_total_gy": float(dose * count) if count is not None else None,
                "source_text": tail_text[match.start() : match.end()].strip(),
            }
        )

    if len(phases) > 1 or any(item["fraction_count"] is not None for item in phases):
        if any(item["fraction_count"] is None for item in phases):
            return [], "INCOMPLETE_MULTIPHASE_FRACTION_COUNTS"
        calculated = sum(
            _decimal(str(item["dose_per_fraction_gy"]))
            * int(item["fraction_count"])
            for item in phases
        )
        if calculated != total:
            return [], "STATED_FRACTIONATION_TOTAL_MISMATCH"
    else:
        per_fraction = _decimal(str(phases[0]["dose_per_fraction_gy"]))
        implied = total / per_fraction
        if implied != implied.to_integral_value():
            return [], "NONINTEGRAL_IMPLIED_FRACTION_COUNT"
        count = int(implied)
        if count <= 0:
            return [], "NONPOSITIVE_FRACTION_COUNT"
        phases[0]["fraction_count"] = count
        phases[0]["phase_total_gy"] = float(per_fraction * count)

    parsed_spans = [(match.start(), match.end()) for match in phase_matches]
    for token in _DOSE_TOKEN_RE.finditer(tail_norm):
        if not any(start <= token.start() < end for start, end in parsed_spans):
            return [], "UNSUPPORTED_EXTRA_DOSE_EXPRESSION"
    return phases, None


def parse_kopernik_treatment_description(source_text: object) -> dict[str, Any]:
    """Parse one ``Opis leczenia`` value without completing partial evidence."""

    text = _cell_text(source_text)
    if not text.strip():
        return {
            "parser": CLINICAL_PARSER_VERSION,
            "status": "REFUSED",
            "reason": "EMPTY_DESCRIPTION",
            "sites": [],
        }
    normalized = _ascii_lower(text)
    totals = _dose_total_candidates(text, normalized)
    if not totals:
        return {
            "parser": CLINICAL_PARSER_VERSION,
            "status": "REFUSED",
            "reason": "MISSING_TOTAL_DOSE",
            "sites": [],
        }

    boundaries: list[int] = [0]
    for previous, current in zip(totals, totals[1:]):
        gap = normalized[previous.end() : current.start()]
        connectors = list(_CONNECTOR_RE.finditer(gap))
        if not connectors:
            return {
                "parser": CLINICAL_PARSER_VERSION,
                "status": "REFUSED",
                "reason": "OVERLAPPING_OR_UNSEPARATED_SITE_CLAUSES",
                "sites": [],
            }
        boundaries.append(previous.end() + connectors[-1].end())
    clause_ends: list[int] = []
    for current, following in zip(totals, totals[1:]):
        gap = normalized[current.end() : following.start()]
        connectors = list(_CONNECTOR_RE.finditer(gap))
        if not connectors:
            return {
                "parser": CLINICAL_PARSER_VERSION,
                "status": "REFUSED",
                "reason": "OVERLAPPING_OR_UNSEPARATED_SITE_CLAUSES",
                "sites": [],
            }
        clause_ends.append(current.end() + connectors[-1].start())
    clause_ends.append(len(text))

    sites: list[dict[str, Any]] = []
    for index, total_match in enumerate(totals):
        prefix = text[boundaries[index] : total_match.start()]
        site = _site_name(prefix)
        if site is None:
            return {
                "parser": CLINICAL_PARSER_VERSION,
                "status": "REFUSED",
                "reason": "MISSING_NAMED_SITE",
                "sites": [],
            }
        try:
            total = _decimal(total_match.group("value"))
        except InvalidOperation:
            return {
                "parser": CLINICAL_PARSER_VERSION,
                "status": "REFUSED",
                "reason": "INVALID_TOTAL_DOSE",
                "sites": [],
            }
        phases, refusal = _fractionation_for_total(
            text=text,
            normalized=normalized,
            total_match=total_match,
            clause_end=clause_ends[index],
        )
        if refusal is not None:
            return {
                "parser": CLINICAL_PARSER_VERSION,
                "status": "REFUSED",
                "reason": refusal,
                "sites": [],
            }
        fraction_count = sum(int(item["fraction_count"]) for item in phases)
        sites.append(
            {
                "site": site,
                "source_text": text[boundaries[index] : clause_ends[index]].strip(),
                "total_dose_gy": float(total),
                "fraction_count": fraction_count,
                "phases": phases,
                "self_check": {
                    "calculated_total_gy": float(
                        sum(
                            _decimal(str(item["dose_per_fraction_gy"]))
                            * int(item["fraction_count"])
                            for item in phases
                        )
                    ),
                    "matches_stated_total": True,
                },
            }
        )

    if len({item["site"].casefold() for item in sites}) != len(sites):
        return {
            "parser": CLINICAL_PARSER_VERSION,
            "status": "REFUSED",
            "reason": "DUPLICATE_SITE_NAME",
            "sites": [],
        }
    return {
        "parser": CLINICAL_PARSER_VERSION,
        "status": "PARSED",
        "reason": None,
        "sites": sites,
    }


def _parse_iso_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def match_clinical_record(
    records: Sequence[ClinicalRecord],
    *,
    course_start_date: object = None,
    course_end_date: object = None,
    plan_dates: Iterable[object] = (),
    treatment_dates: Iterable[object] = (),
    required_icd10_prefix: str = "C67",
) -> dict[str, Any]:
    """Match by explicit interval evidence and refuse tied candidates."""

    start = _parse_iso_date(course_start_date)
    end = _parse_iso_date(course_end_date)
    plan_evidence = sorted(
        {value for item in plan_dates if (value := _parse_iso_date(item)) is not None}
    )
    treatment_evidence = sorted(
        {value for item in treatment_dates if (value := _parse_iso_date(item)) is not None}
    )
    candidates: list[tuple[int, ClinicalRecord, dict[str, Any]]] = []
    eligible_record_count = 0
    for record in records:
        if required_icd10_prefix and not record.diagnosis_icd10.strip().upper().startswith(
            required_icd10_prefix.upper()
        ):
            continue
        eligible_record_count += 1
        if (
            record.start_date is None
            or record.end_date is None
            or record.start_date > record.end_date
        ):
            continue
        within_treatment = [
            value
            for value in treatment_evidence
            if record.start_date <= value <= record.end_date
        ]
        within_plans = [
            value for value in plan_evidence if record.start_date <= value <= record.end_date
        ]
        course_overlap = bool(
            start is not None
            and end is not None
            and start <= record.end_date
            and record.start_date <= end
        )
        if treatment_evidence and len(within_treatment) == len(treatment_evidence):
            rank = 4
            basis = "FULL_TREATMENT_WINDOW_CONTAINMENT"
        elif within_treatment:
            rank = 3
            basis = "TREATMENT_DATE_OVERLAP"
        elif course_overlap:
            rank = 2
            basis = "COURSE_WINDOW_OVERLAP"
        elif within_plans:
            rank = 1
            basis = "RTPLAN_DATE_WITHIN_RECORD_WINDOW"
        else:
            continue
        candidates.append(
            (
                rank,
                record,
                {
                    "basis": basis,
                    "record_window": [
                        record.start_date.isoformat(),
                        record.end_date.isoformat(),
                    ],
                    "course_window": [
                        start.isoformat() if start else None,
                        end.isoformat() if end else None,
                    ],
                    "treatment_dates_within_record": [
                        value.isoformat() for value in within_treatment
                    ],
                    "plan_dates_within_record": [
                        value.isoformat() for value in within_plans
                    ],
                },
            )
        )

    if not candidates:
        reason = "NO_ELIGIBLE_PATIENT_RECORD" if eligible_record_count == 0 else "NO_TEMPORAL_MATCH"
        return {
            "status": "REFUSED",
            "reason": reason,
            "matched_record": None,
            "match_evidence": {
                "eligible_record_count": eligible_record_count,
                "course_window": [
                    start.isoformat() if start else None,
                    end.isoformat() if end else None,
                ],
                "treatment_dates": [value.isoformat() for value in treatment_evidence],
                "plan_dates": [value.isoformat() for value in plan_evidence],
            },
            "candidate_record_ids": [],
        }
    best_rank = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_rank]
    if len(best) != 1:
        return {
            "status": "REFUSED",
            "reason": "AMBIGUOUS_RECORD_MATCH",
            "matched_record": None,
            "match_evidence": {
                "rank": best_rank,
                "basis": best[0][2]["basis"],
                "candidate_count": len(best),
            },
            "candidate_record_ids": [item[1].record_id for item in best],
        }
    _, record, evidence = best[0]
    return {
        "status": "MATCHED",
        "reason": None,
        "matched_record": record,
        "match_evidence": evidence,
        "candidate_record_ids": [record.record_id],
    }


def _close(left: float | None, right: float | None, tolerance_gy: float = 0.05) -> bool:
    return left is not None and right is not None and abs(left - right) <= tolerance_gy


def confirm_two_phase_fractionation(
    sites: Sequence[Mapping[str, Any]],
    per_plan_delivery: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind delivered clinical phases to DICOM plans without using plan order.

    The register describes delivered fractions. A superseded DICOM plan can
    retain a larger planned count, so matching requires the delivered record
    count and the plan prescription per planned fraction. Every phase and plan
    must bind exactly once and the delivered phase totals must equal the
    self-checked clinical total.
    """
    if len(sites) != 1:
        return None
    phases = list(sites[0].get("phases") or [])
    if len(phases) < 2:
        return None
    observed: list[dict[str, Any]] = []
    for item in per_plan_delivery:
        delivered = int(item.get("delivered_fraction_count") or 0)
        planned = int(item.get("planned_fraction_count") or 0)
        total_value = item.get("prescribed_dose_gy")
        try:
            total = float(total_value) if total_value not in (None, "") else None
        except (TypeError, ValueError):
            total = None
        if delivered <= 0 or planned <= 0 or total is None:
            continue
        observed.append(
            {
                "plan_sop_uid": str(item.get("plan_sop_uid") or ""),
                "delivered_fraction_count": delivered,
                "planned_fraction_count": planned,
                "dose_per_fraction_gy": total / planned,
            }
        )
    if len(observed) != len(phases):
        return None
    unmatched = list(observed)
    bindings: list[dict[str, Any]] = []
    for phase in phases:
        fraction_count = int(phase.get("fraction_count") or 0)
        per_fraction = float(phase.get("dose_per_fraction_gy") or 0.0)
        matches = [
            item
            for item in unmatched
            if item["delivered_fraction_count"] == fraction_count
            and _close(float(item["dose_per_fraction_gy"]), per_fraction, 0.001)
        ]
        if len(matches) != 1:
            return None
        selected = matches[0]
        unmatched.remove(selected)
        bindings.append(
            {
                "clinical_phase": dict(phase),
                "dicom_delivery": selected,
            }
        )
    clinical_total = float(sites[0].get("total_dose_gy") or 0.0)
    dicom_delivered_total = float(
        sum(
            int(item["delivered_fraction_count"])
            * float(item["dose_per_fraction_gy"])
            for item in observed
        )
    )
    if not _close(clinical_total, dicom_delivered_total, 0.001):
        return None
    return {
        "classification": "TWO_FRACTIONATION_PHASES",
        "basis": "Each clinical phase matches one delivered RTPLAN by fraction count and dose per fraction",
        "clinical_total_gy": clinical_total,
        "dicom_delivered_total_gy": dicom_delivered_total,
        "phase_plan_bindings": [
            {
                "clinical_fraction_count": int(
                    item["clinical_phase"]["fraction_count"]
                ),
                "clinical_dose_per_fraction_gy": float(
                    item["clinical_phase"]["dose_per_fraction_gy"]
                ),
                "plan_sop_uid": item["dicom_delivery"]["plan_sop_uid"],
                "delivered_fraction_count": int(
                    item["dicom_delivery"]["delivered_fraction_count"]
                ),
                "dose_per_fraction_gy": float(
                    item["dicom_delivery"]["dose_per_fraction_gy"]
                ),
            }
            for item in bindings
        ],
        "bindings": bindings,
    }


def adjudicate_clinical_prescription(
    index: ClinicalRecordIndex,
    *,
    patient_id: object,
    course_id: object,
    course_start_date: object,
    course_end_date: object,
    plan_dates: Iterable[object],
    treatment_dates: Iterable[object],
    dicom_resolved_total_gy: float | None,
    dicom_prescribed_dose_scope: str,
    dicom_classification: str | None,
    per_plan_delivery: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve, corroborate, or flag without replacing DICOM evidence."""

    records = index.records_for(patient_id)
    match = match_clinical_record(
        records,
        course_start_date=course_start_date,
        course_end_date=course_end_date,
        plan_dates=plan_dates,
        treatment_dates=treatment_dates,
    )
    base: dict[str, Any] = {
        "schema": CLINICAL_EVIDENCE_SCHEMA,
        "parser": CLINICAL_PARSER_VERSION,
        "patient_id": _patient_text(patient_id),
        "course_id": str(course_id),
        "source": index.source_dict(),
        "dicom": {
            "resolved_prescribed_dose_total_gy": dicom_resolved_total_gy,
            "prescribed_dose_scope": dicom_prescribed_dose_scope,
            "dose_classification": dicom_classification,
        },
        "match": {
            key: value for key, value in match.items() if key != "matched_record"
        },
        "record": None,
        "parse": None,
        "outcome": "UNRESOLVED",
        "reason": match.get("reason"),
        "effective_prescription_source": (
            "DICOM" if dicom_resolved_total_gy is not None else None
        ),
        "clinical_resolved_total_gy": None,
        "effective_resolved_total_gy": dicom_resolved_total_gy,
        "fractionation_classification": None,
    }
    record = match.get("matched_record")
    if not isinstance(record, ClinicalRecord):
        return base
    base["record"] = record.audit_dict()
    parsed = parse_kopernik_treatment_description(record.description)
    base["parse"] = parsed
    if parsed["status"] != "PARSED":
        base["reason"] = parsed["reason"]
        return base

    sites = list(parsed["sites"])
    totals = sorted({float(item["total_dose_gy"]) for item in sites})
    phase_confirmation = confirm_two_phase_fractionation(sites, per_plan_delivery)
    base["fractionation_classification"] = phase_confirmation

    if dicom_resolved_total_gy is not None:
        matching_sites = [
            item for item in sites if _close(float(item["total_dose_gy"]), dicom_resolved_total_gy)
        ]
        if matching_sites:
            base["outcome"] = "CORROBORATED_DICOM"
            base["reason"] = "CLINICAL_SITE_TOTAL_MATCHES_DICOM"
            base["corroborating_sites"] = [item["site"] for item in matching_sites]
        else:
            base["outcome"] = "DISAGREES_WITH_DICOM"
            base["reason"] = "NO_CLINICAL_SITE_TOTAL_MATCHES_DICOM"
            base["disagreement"] = {
                "dicom_total_gy": dicom_resolved_total_gy,
                "clinical_site_totals": [
                    {"site": item["site"], "total_dose_gy": item["total_dose_gy"]}
                    for item in sites
                ],
            }
        return base

    if len(totals) != 1:
        base["reason"] = "MULTISITE_DISTINCT_TOTALS"
        base["per_site_only"] = True
        return base
    match_basis = str(
        (match.get("match_evidence") or {}).get("basis") or ""
    )
    if match_basis not in {
        "FULL_TREATMENT_WINDOW_CONTAINMENT",
        "RTPLAN_DATE_WITHIN_RECORD_WINDOW",
    }:
        base["reason"] = "INSUFFICIENT_TEMPORAL_EVIDENCE_FOR_RESOLUTION"
        return base
    clinical_total = totals[0]
    base["outcome"] = "RESOLVED_FROM_CLINICAL_RECORD"
    base["reason"] = "UNIQUE_SELF_CHECKED_CLINICAL_TOTAL"
    base["effective_prescription_source"] = "CLINICAL_RECORD"
    base["clinical_resolved_total_gy"] = clinical_total
    base["effective_resolved_total_gy"] = clinical_total
    return base


def clinical_evidence_matches_source(
    contract_data: Mapping[str, Any], index: ClinicalRecordIndex | None
) -> bool:
    """Reject resume reuse when optional-source identity changed or disappeared."""

    evidence = contract_data.get("clinical_prescription_evidence")
    if index is None:
        return evidence is None
    if not isinstance(evidence, Mapping):
        return False
    source = evidence.get("source")
    return isinstance(source, Mapping) and source.get("workbook_sha256") == index.workbook_sha256
