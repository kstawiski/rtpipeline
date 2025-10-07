#!/usr/bin/env python3
"""
Anonymize pipeline outputs generated under Data_Snakemake/.

The script rewrites directory names, file contents, and metadata so that
patient-identifying information is replaced with synthetic identifiers. A key
file is produced to recover the original identifiers if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pydicom
from pydicom.dataset import Dataset


@dataclass
class CourseInfo:
    patient_id: str
    course_dir_name: str
    course_id: str
    course_key: Optional[str]
    path: Path
    anon_course_id: str = ""
    anon_course_key: str = ""

    def identifier_replacements(self) -> List[Tuple[str, str]]:
        repl: List[Tuple[str, str]] = []
        if self.course_key:
            repl.append((self.course_key, self.anon_course_key))
        if self.course_id:
            repl.append((self.course_id, self.anon_course_id))
        repl.append((self.course_dir_name, self.anon_course_id))
        return repl


@dataclass
class PatientInfo:
    patient_id: str
    path: Path
    courses: List[CourseInfo] = field(default_factory=list)
    patient_name: Optional[str] = None
    anon_patient_id: str = ""
    anon_patient_name: str = ""

    def identifier_replacements(self) -> List[Tuple[str, str]]:
        repl: List[Tuple[str, str]] = [(self.patient_id, self.anon_patient_id)]
        if self.patient_name:
            repl.append((self.patient_name, self.anon_patient_name))
            repl.append((self.patient_name.replace("^", " "), self.anon_patient_name))
        return repl


@dataclass
class GlobalContext:
    input_root: Path
    output_root: Path
    patients: List[PatientInfo]
    patient_id_map: Dict[str, PatientInfo]
    course_id_map: Dict[str, CourseInfo]
    course_key_map: Dict[str, CourseInfo]
    patient_name_map: Dict[str, PatientInfo]
    string_replacements: List[Tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymize pipeline results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data_Snakemake"),
        help="Input directory containing pipeline outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for anonymized data.",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="CSV file to store the anonymization key (defaults inside output).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def collect_patients(input_root: Path) -> List[PatientInfo]:
    patients: List[PatientInfo] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        patient = PatientInfo(patient_id=child.name, path=child)
        patient.courses = collect_courses(patient)
        patient.patient_name = determine_patient_name(patient)
        patients.append(patient)
    if not patients:
        raise ValueError(f"No patient directories found under {input_root}")
    return patients


def collect_courses(patient: PatientInfo) -> List[CourseInfo]:
    courses: List[CourseInfo] = []
    for course_dir in sorted(patient.path.iterdir()):
        if not course_dir.is_dir():
            continue
        meta_path = course_dir / "metadata" / "case_metadata.json"
        course_id = course_dir.name
        course_key: Optional[str] = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            course_id = data.get("course_id", course_id)
            course_key = data.get("course_key")
        courses.append(
            CourseInfo(
                patient_id=patient.patient_id,
                course_dir_name=course_dir.name,
                course_id=course_id,
                course_key=course_key,
                path=course_dir,
            )
        )
    return courses


def determine_patient_name(patient: PatientInfo) -> Optional[str]:
    for course in patient.courses:
        meta_path = course.path / "metadata" / "case_metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("patient_name")
            if name:
                return name
    return None


def assign_identifiers(patients: List[PatientInfo]) -> None:
    course_counter = 1
    for idx, patient in enumerate(patients, start=1):
        patient.anon_patient_id = f"PAT{idx:04d}"
        patient.anon_patient_name = f"Anon{idx:04d}"
        for course in patient.courses:
            course.anon_course_id = f"CRS{course_counter:04d}"
            course.anon_course_key = f"{patient.anon_patient_id}_{course.anon_course_id}"
            course_counter += 1


def build_global_context(input_root: Path, output_root: Path, patients: List[PatientInfo]) -> GlobalContext:
    patient_id_map = {patient.patient_id: patient for patient in patients}
    patient_name_map = {
        patient.patient_name: patient
        for patient in patients
        if patient.patient_name
    }
    course_id_map: Dict[str, CourseInfo] = {}
    course_key_map: Dict[str, CourseInfo] = {}
    for patient in patients:
        for course in patient.courses:
            if course.course_id:
                if course.course_id in course_id_map:
                    logging.debug(
                        "Duplicate course_id detected (%s); mapping will be keyed via course_key only.",
                        course.course_id,
                    )
                else:
                    course_id_map[course.course_id] = course
            if course.course_key:
                course_key_map[course.course_key] = course
    replacements: List[Tuple[str, str]] = []
    replacements.append((str(input_root.resolve()), str(output_root.resolve())))
    for patient in patients:
        replacements.extend(patient.identifier_replacements())
        for course in patient.courses:
            combo_forward = f"{patient.patient_id}/{course.course_dir_name}"
            replacements.append((combo_forward, f"{patient.anon_patient_id}/{course.anon_course_id}"))
            combo_backward = f"{patient.patient_id}\\{course.course_dir_name}"
            replacements.append((combo_backward, f"{patient.anon_patient_id}\\{course.anon_course_id}"))
            replacements.extend(course.identifier_replacements())
    # longer patterns first to avoid partial collisions
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)
    return GlobalContext(
        input_root=input_root,
        output_root=output_root,
        patients=patients,
        patient_id_map=patient_id_map,
        course_id_map=course_id_map,
        course_key_map=course_key_map,
        patient_name_map=patient_name_map,
        string_replacements=replacements,
    )


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory {path} already exists. Use --overwrite to replace it.")
        if any(path.iterdir()):
            logging.info("Clearing existing output directory: %s", path)
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def replace_identifiers(value: str, replacements: Sequence[Tuple[str, str]]) -> str:
    result = value
    for old, new in replacements:
        if not old:
            continue
        result = result.replace(old, new)
    return result


def dedupe_replacements(replacements: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    ordered: List[Tuple[str, str]] = []
    for pair in sorted(replacements, key=lambda p: len(p[0]), reverse=True):
        if not pair[0]:
            continue
        if pair not in seen:
            seen.add(pair)
            ordered.append(pair)
    return ordered


def anonymize_patient_directories(context: GlobalContext) -> None:
    for patient in context.patients:
        src = patient.path
        dest = context.output_root / patient.anon_patient_id
        logging.info("Anonymizing patient %s -> %s", patient.patient_id, patient.anon_patient_id)
        dest.mkdir(parents=True, exist_ok=True)
        anonymize_patient_dir(src, dest, patient, context)


def anonymize_patient_dir(src: Path, dest: Path, patient: PatientInfo, context: GlobalContext) -> None:
    for item in sorted(src.iterdir()):
        if item.is_dir():
            course = next((c for c in patient.courses if c.course_dir_name == item.name), None)
            if course:
                target_dir = dest / course.anon_course_id
                target_dir.mkdir(parents=True, exist_ok=True)
                anonymize_course_dir(item, target_dir, patient, course, context)
            elif item.name.startswith("Segmentation_"):
                target_dir = dest / replace_identifiers(item.name, patient.identifier_replacements())
                anonymize_segmentation_dir(item, target_dir, patient, context)
            else:
                target_dir = dest / replace_identifiers(item.name, patient.identifier_replacements())
                target_dir.mkdir(parents=True, exist_ok=True)
                copy_directory_generic(item, target_dir, patient.identifier_replacements())
        else:
            rel_name = replace_identifiers(item.name, patient.identifier_replacements())
            copy_file_generic(item, dest / rel_name, patient.identifier_replacements())


def anonymize_course_dir(
    src: Path,
    dest: Path,
    patient: PatientInfo,
    course: CourseInfo,
    context: GlobalContext,
) -> None:
    replacements = dedupe_replacements(list(patient.identifier_replacements()) + list(course.identifier_replacements()))
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        dest_rel = Path(*(replace_identifiers(part, replacements) for part in rel.parts))
        target = dest / dest_rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".dcm":
            anonymize_dicom(path, target, patient, course)
        elif path.suffix.lower() in {".json"}:
            anonymize_json_file(path, target, replacements, patient, course, context)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            anonymize_excel_file(path, target, replacements, context)
        elif path.suffix.lower() == ".csv":
            anonymize_csv_file(path, target, replacements, context)
        else:
            copy_file_generic(path, target, replacements)


def anonymize_segmentation_dir(
    src: Path,
    dest: Path,
    patient: PatientInfo,
    context: GlobalContext,
) -> None:
    combined_repl = dedupe_replacements(
        list(patient.identifier_replacements())
        + [pair for course in patient.courses for pair in course.identifier_replacements()]
    )
    dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        if item.is_dir():
            course = next((c for c in patient.courses if c.course_dir_name == item.name), None)
            if course:
                sub_dest = dest / course.anon_course_id
                sub_dest.mkdir(parents=True, exist_ok=True)
                anonymize_course_dir(item, sub_dest, patient, course, context)
            else:
                sub_dest = dest / replace_identifiers(item.name, combined_repl)
                sub_dest.mkdir(parents=True, exist_ok=True)
                copy_directory_generic(item, sub_dest, combined_repl)
        else:
            target = dest / replace_identifiers(item.name, combined_repl)
            target.parent.mkdir(parents=True, exist_ok=True)
            suffix = item.suffix.lower()
            fallback_course = patient.courses[0] if patient.courses else None
            if suffix == ".dcm":
                if fallback_course is not None:
                    anonymize_dicom(item, target, patient, fallback_course)
                else:
                    copy_file_generic(item, target, combined_repl)
            elif suffix == ".json":
                anonymize_json_file(item, target, combined_repl, patient, None, context)
            elif suffix in {".xlsx", ".xls"}:
                anonymize_excel_file(item, target, combined_repl, context)
            elif suffix == ".csv":
                anonymize_csv_file(item, target, combined_repl, context)
            else:
                copy_file_generic(item, target, combined_repl)


def copy_directory_generic(src: Path, dest: Path, replacements: Sequence[Tuple[str, str]]) -> None:
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        dest_rel = Path(*(replace_identifiers(part, replacements) for part in rel.parts))
        target = dest / dest_rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file_generic(path, target, replacements)


def copy_file_generic(src: Path, dest: Path, replacements: Sequence[Tuple[str, str]]) -> None:
    if src.is_symlink():
        resolved = src.resolve()
        new_target = replace_identifiers(str(resolved), replacements)
        dest.symlink_to(new_target)
        return
    data = src.read_bytes()
    dest.write_bytes(data)


def anonymize_dicom(src: Path, dest: Path, patient: PatientInfo, course: CourseInfo) -> None:
    ds: Dataset = pydicom.dcmread(src, force=True)
    ds.remove_private_tags()
    ds.PatientID = patient.anon_patient_id
    ds.PatientName = patient.anon_patient_name
    for tag in [
        "OtherPatientIDs",
        "OtherPatientNames",
        "PatientBirthDate",
        "PatientBirthTime",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "EthnicGroup",
    ]:
        if tag in ds:
            ds[tag].value = ""
    for tag in [
        "PhysiciansOfRecord",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
    ]:
        if tag in ds:
            ds[tag].value = "ANONYMIZED"
    for tag in ["ReviewDate", "StudyDate"]:
        if tag in ds:
            ds[tag].value = ""
    for tag in ["ReviewTime", "StudyTime"]:
        if tag in ds:
            ds[tag].value = ""
    if "StudyID" in ds:
        ds.StudyID = replace_identifiers(str(ds.StudyID), course.identifier_replacements())
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(dest)


SENSITIVE_NAME_KEYS = {
    "patient_name",
    "reviewer_name",
    "operators_name",
    "physicians_of_record",
    "referring_physician",
    "performing_physician",
    "primary_physician",
}
SENSITIVE_ID_KEYS = {
    "patient_id",
    "patient_id_plans",
    "patient_id_dosimetrics",
}
SENSITIVE_DATE_KEYS = {
    "patient_birth_date",
    "patient_dob",
}


def anonymize_json_file(
    src: Path,
    dest: Path,
    replacements: Sequence[Tuple[str, str]],
    patient: PatientInfo,
    course: Optional[CourseInfo],
    context: GlobalContext,
) -> None:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sanitized = sanitize_structure(
        data,
        replacements,
        patient,
        course,
        context,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)


def sanitize_structure(
    value: Any,
    replacements: Sequence[Tuple[str, str]],
    patient: PatientInfo,
    course: Optional[CourseInfo],
    context: GlobalContext,
    key_hint: Optional[str] = None,
) -> Any:
    if isinstance(value, dict):
        return {
            k: sanitize_structure(
                v,
                replacements,
                patient,
                course,
                context,
                key_hint=k,
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_structure(item, replacements, patient, course, context, key_hint=key_hint)
            for item in value
        ]
    if isinstance(value, str):
        lower_key = (key_hint or "").lower()
        if lower_key in SENSITIVE_NAME_KEYS:
            return "ANONYMIZED"
        if lower_key in SENSITIVE_ID_KEYS:
            return patient.anon_patient_id
        if lower_key == "course_id":
            if course:
                return course.anon_course_id
            mapped = context.course_id_map.get(value)
            if mapped:
                return mapped.anon_course_id
        if lower_key == "course_key":
            if course and course.course_key:
                return course.anon_course_key
            mapped = context.course_key_map.get(value)
            if mapped:
                return mapped.anon_course_key
        if lower_key in SENSITIVE_DATE_KEYS:
            return ""
        if lower_key == "patient_sex":
            return value
        replaced = replace_identifiers(value, replacements)
        replaced = replace_identifiers(replaced, context.string_replacements)
        return replaced
    if isinstance(value, (int, float)):
        text_value = str(int(value)) if isinstance(value, float) else str(value)
        if text_value == patient.patient_id:
            return patient.anon_patient_id
        return value
    return value


NAME_COLUMNS = {
    "patient_name",
    "reviewer_name",
    "operators_name",
    "physicians_of_record",
    "referring_physician",
    "performing_physician",
    "primary_physician",
}
DATE_COLUMNS = {
    "patient_birth_date",
    "patient_dob",
}
ID_COLUMNS = {
    "patient_id",
    "patient_id_plans",
    "patient_id_dosimetrics",
    "patient_id_structures",
}
COURSE_ID_COLUMNS = {
    "course_id",
}
COURSE_KEY_COLUMNS = {
    "course_key",
}


def anonymize_excel_file(
    src: Path,
    dest: Path,
    replacements: Sequence[Tuple[str, str]],
    context: GlobalContext,
) -> None:
    xls = pd.ExcelFile(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, dtype=object)
            df = anonymize_dataframe(df, replacements, context)
            df.to_excel(writer, sheet_name=sheet, index=False)


def anonymize_csv_file(
    src: Path,
    dest: Path,
    replacements: Sequence[Tuple[str, str]],
    context: GlobalContext,
) -> None:
    df = pd.read_csv(src, dtype=object)
    df = anonymize_dataframe(df, replacements, context)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)


def anonymize_dataframe(
    df: pd.DataFrame,
    replacements: Sequence[Tuple[str, str]],
    context: GlobalContext,
) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in NAME_COLUMNS:
            df[column] = df[column].apply(lambda _: "ANONYMIZED")
            continue
        if lowered in DATE_COLUMNS:
            df[column] = ""
            continue
        if lowered in ID_COLUMNS:
            df[column] = df[column].apply(lambda v: map_patient_id(v, context))
            continue
        if lowered in COURSE_KEY_COLUMNS:
            df[column] = df[column].apply(lambda v: map_course_key(v, context))
            continue
        if lowered in COURSE_ID_COLUMNS:
            df[column] = df[column].apply(lambda v: map_course_id(v, context))
            continue
        df[column] = df[column].apply(
            lambda v: replace_strings_generic(v, replacements, context)
        )
    return df


def map_patient_id(value: Any, context: GlobalContext) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(int(value)) if isinstance(value, float) else str(value)
    patient = context.patient_id_map.get(text)
    if patient:
        return patient.anon_patient_id
    return replace_strings_generic(value, [], context)


def map_course_id(value: Any, context: GlobalContext) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value)
    course = context.course_id_map.get(text)
    if course:
        return course.anon_course_id
    return replace_strings_generic(value, [], context)


def map_course_key(value: Any, context: GlobalContext) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value)
    course = context.course_key_map.get(text)
    if course:
        return course.anon_course_key
    return replace_strings_generic(value, [], context)


def replace_strings_generic(
    value: Any,
    local_replacements: Sequence[Tuple[str, str]],
    context: GlobalContext,
) -> Any:
    if not isinstance(value, str):
        return value
    replaced = replace_identifiers(value, local_replacements)
    replaced = replace_identifiers(replaced, context.string_replacements)
    return replaced


def anonymize_global_tables(context: GlobalContext) -> None:
    for relative in ["Data", "_RESULTS"]:
        src_dir = context.input_root / relative
        if not src_dir.exists():
            continue
        dest_dir = context.output_root / relative
        dest_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Anonymizing tables in %s", relative)
        for path in sorted(src_dir.glob("*.xlsx")):
            dest = dest_dir / path.name
            anonymize_excel_file(path, dest, context.string_replacements, context)


def write_key_csv(context: GlobalContext, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anon_patient_id",
        "original_patient_id",
        "original_patient_name",
        "anon_course_id",
        "original_course_id",
        "original_course_dir",
        "anon_course_key",
        "original_course_key",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient in context.patients:
            if patient.courses:
                for course in patient.courses:
                    writer.writerow(
                        {
                            "anon_patient_id": patient.anon_patient_id,
                            "original_patient_id": patient.patient_id,
                            "original_patient_name": patient.patient_name or "",
                            "anon_course_id": course.anon_course_id,
                            "original_course_id": course.course_id,
                            "original_course_dir": course.course_dir_name,
                            "anon_course_key": course.anon_course_key,
                            "original_course_key": course.course_key or "",
                        }
                    )
            else:
                writer.writerow(
                    {
                        "anon_patient_id": patient.anon_patient_id,
                        "original_patient_id": patient.patient_id,
                        "original_patient_name": patient.patient_name or "",
                        "anon_course_id": "",
                        "original_course_id": "",
                        "original_course_dir": "",
                        "anon_course_key": "",
                        "original_course_key": "",
                    }
                )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    input_root = args.input.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory {input_root} does not exist.")
    output_root = (
        args.output.expanduser().resolve()
        if args.output
        else (input_root.parent / f"{input_root.name}_anonymized").resolve()
    )
    ensure_output_dir(output_root, args.overwrite)
    patients = collect_patients(input_root)
    assign_identifiers(patients)
    context = build_global_context(input_root, output_root, patients)
    anonymize_patient_directories(context)
    anonymize_global_tables(context)
    key_file = args.key_file or (output_root / "anonymization_key.csv")
    write_key_csv(context, key_file.resolve())
    logging.info("Anonymization complete.")
    logging.info("Output directory: %s", output_root)
    logging.info("Key file: %s", key_file.resolve())


if __name__ == "__main__":
    main()
