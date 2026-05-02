#!/usr/bin/env python3
"""
Step 42: stage stratification of the robustness hierarchy.

Purpose
-------
Recompute feature-level ICC(3,1) / CoV from the corrected aggregated
`robustness_by_cohort/*.parquet` artifacts after stratifying patients by the
best available stage variables for the two clinically joined Polish cohorts:

  - Prostata: direct `patient_id` join to the ASTRO2026 analytic cohort.
  - Odbytnice: `patient_id -> PESEL -> clinical CSV` join via the audited
    mapping table.

Primary outputs are written to a caller-supplied output directory:

  - `stage42_join_summary.csv`
  - `stage42_family_metrics.csv`
  - `stage42_rank_stability.csv`
  - `stage42_feature_metrics.parquet`
  - `stage42_manifest.json`

Design notes
------------
1. Uses only the corrected cohort parquets as the robustness source.
2. Recomputes ICC / CoV from raw patient-course perturbation rows.
3. Ranks feature families by median ICC within each stage stratum.
4. Reports robust-fraction ranks as a sensitivity metric.
5. Uses `subject = patient_id + "_" + course_id` for repeated-measure handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sp_stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "gldm", "glszm", "ngtdm"]
CLASS_ORDER = ["Robust", "Acceptable", "Poor"]
NON_OVERALL_FAMILY_ORDER = FAMILY_ORDER.copy()
OVERALL_LABEL = "Overall"

ROBUSTNESS_PATHS = {
    "Prostata": Path(
        "/home/kgs24/rtpipeline_manuscript/analysis/data/robustness_by_cohort/Prostata.parquet"
    ),
    "Odbytnice": Path(
        "/home/kgs24/rtpipeline_manuscript/analysis/data/robustness_by_cohort/Odbytnice.parquet"
    ),
}

PROSTATE_CLINICAL_CANDIDATES = [
    Path("/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/analytic_cohort.parquet"),
    Path("/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.parquet"),
    Path("/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.csv"),
    Path("/projekty/Prostata/LK/Data/bdo.tsv"),
]

ODBYTNICE_CLINICAL_CANDIDATES = [
    Path("/projekty/Odbytnice/ClinicalData20260304/final/odbytnice_clinical_2026_values.csv"),
    Path("/projekty/Odbytnice/ClinicalData20260304/final/odbytnice_clinical_2026.csv"),
    Path("/umed-projekty/ODBYTNICE2026/data_bucket/ClinicalData20260304/final/odbytnice_clinical_2026.csv"),
]

ODBYTNICE_MAP_CANDIDATES = [
    Path("/projekty/Odbytnice/analysis/patient_id_pesel_mapping.csv"),
    Path("/umed-projekty/ODBYTNICE2026/analysis/patient_id_pesel_mapping.csv"),
]


@dataclass(frozen=True)
class StageDefinition:
    cohort: str
    name: str
    label: str
    min_subjects: int


STAGE_DEFINITIONS = [
    StageDefinition("Prostata", "t_group", "T group", 20),
    StageDefinition("Prostata", "isup_band", "ISUP band", 20),
    StageDefinition("Odbytnice", "clinical_t_group", "Clinical T group", 20),
    StageDefinition("Odbytnice", "clinical_stage_group", "Clinical AJCC group", 20),
]


class ContextFilter(logging.Filter):
    def __init__(self, host: str, job_id: str):
        super().__init__()
        self.host = host
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.host = self.host
        record.job_id = self.job_id
        return True


def configure_logging(job_id: str) -> logging.Logger:
    logger = logging.getLogger("stage42")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [host=%(host)s job=%(job_id)s] %(message)s"
        )
    )
    handler.addFilter(ContextFilter(socket.gethostname(), job_id))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", default=os.environ.get("JOB_ID", "manual"))
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--odbytnice-map", type=Path, default=None)
    parser.add_argument("--min-subjects-per-stratum", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def choose_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def load_table(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=dtype)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=dtype)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_yes_no(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[text.isin({"yes", "true", "1"})] = 1.0
    out.loc[text.isin({"no", "false", "0"})] = 0.0
    return out


def infer_feature_family(feature_name: object) -> str:
    text = str(feature_name).lower()
    for family in FAMILY_ORDER:
        if f"_{family}_" in text or text.startswith(family + "_"):
            return family
    return "other"


def classify_feature(icc_value: float, cov_value: float) -> str:
    if pd.notna(icc_value) and pd.notna(cov_value):
        if icc_value >= 0.90 and cov_value <= 10.0:
            return "Robust"
        if 0.75 <= icc_value < 0.90 and cov_value <= 20.0:
            return "Acceptable"
    return "Poor"


def compute_icc31_from_sufficient_stats(
    *,
    n_subjects: float,
    total_sum: float,
    total_sum_sq: float,
    row_sum_sq: float,
    col_sums: np.ndarray,
) -> float:
    k_raters = int(col_sums.shape[0])
    if n_subjects < 2 or k_raters < 2:
        return float("nan")

    n = float(n_subjects)
    k = float(k_raters)
    correction = (total_sum * total_sum) / (n * k)
    ss_total = total_sum_sq - correction
    ss_rows = (row_sum_sq / k) - correction
    ss_cols = (np.square(col_sums).sum() / n) - correction
    ss_error = ss_total - ss_rows - ss_cols
    if ss_error < 0 and abs(ss_error) < 1e-9:
        ss_error = 0.0

    ms_rows = ss_rows / (n - 1.0)
    denom_error = (n - 1.0) * (k - 1.0)
    if denom_error <= 0:
        return float("nan")
    ms_error = ss_error / denom_error
    denom = ms_rows + (k - 1.0) * ms_error
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


def compute_cov_from_sufficient_stats(
    *,
    total_sum: float,
    total_sum_sq: float,
    n_values: float,
) -> float:
    if n_values < 2:
        return float("nan")
    mean_value = total_sum / n_values
    if abs(mean_value) < 1e-10:
        return float("nan")
    numerator = total_sum_sq - ((total_sum * total_sum) / n_values)
    if numerator < 0 and abs(numerator) < 1e-9:
        numerator = 0.0
    variance = numerator / (n_values - 1.0)
    if variance < 0:
        return float("nan")
    return float((math.sqrt(variance) / abs(mean_value)) * 100.0)


def normalize_prostate_t(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"t\s*([0-4x])", text, flags=re.IGNORECASE)
    if not match:
        return None
    level = match.group(1).upper()
    if level == "X":
        return "Tx"
    return f"T{level}"


def normalize_isup_band(value: object) -> str | None:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    match = re.search(r"([1-5])", text)
    if not match:
        return None
    grade = int(match.group(1))
    if grade <= 2:
        return "GG1-2"
    if grade == 3:
        return "GG3"
    return "GG4-5"


def normalize_odbytnice_clinical_t(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"t\s*([0-4x])", text, flags=re.IGNORECASE)
    if not match:
        return None
    level = match.group(1).upper()
    if level == "X":
        return "cTx"
    return f"cT{level}"


def normalize_stage_group(value: object) -> str | None:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    text = text.replace("STAGE", "").replace("AJCC", "").strip()
    text = text.split()[0]
    text = text.replace("0", "O")
    mapping = {
        "I": "I",
        "II": "II",
        "III": "III",
        "IV": "IV",
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
    }
    return mapping.get(text)


def resolve_prostate_clinical(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    source_path = choose_existing_path(PROSTATE_CLINICAL_CANDIDATES)
    clinical = load_table(source_path)
    clinical.columns = [str(column) for column in clinical.columns]
    if "patient_id" not in clinical.columns:
        raise RuntimeError(f"{source_path}: missing patient_id column.")
    out = clinical.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    t_source = "T_main" if "T_main" in out.columns else "T"
    out["t_group"] = out[t_source].map(normalize_prostate_t)
    out["isup_band"] = out["ISUP"].map(normalize_isup_band) if "ISUP" in out.columns else None
    keep = ["patient_id", "t_group", "isup_band"]
    meta = {
        "clinical_source_path": str(source_path),
        "clinical_rows": int(len(out)),
        "t_source_column": t_source,
        "t_group_counts_raw": {
            str(key): int(value)
            for key, value in out["t_group"].value_counts(dropna=False).sort_index().items()
        },
        "isup_band_counts_raw": {
            str(key): int(value)
            for key, value in out["isup_band"].value_counts(dropna=False).sort_index().items()
        },
    }
    logger.info("Resolved Prostata clinical source: %s rows=%d", source_path, len(out))
    return out[keep].drop_duplicates(subset=["patient_id"]), meta


def resolve_odbytnice_clinical(
    logger: logging.Logger, mapping_override: Path | None
) -> tuple[pd.DataFrame, dict[str, object]]:
    clinical_path = choose_existing_path(ODBYTNICE_CLINICAL_CANDIDATES)
    map_candidates = [mapping_override] if mapping_override is not None else []
    map_candidates.extend(ODBYTNICE_MAP_CANDIDATES)
    mapping_path = choose_existing_path([path for path in map_candidates if path is not None])

    clinical = load_table(clinical_path, dtype={"PESEL": str})
    mapping = load_table(mapping_path, dtype={"PESEL": str, "patient_id": str})
    clinical.columns = [str(column) for column in clinical.columns]
    mapping.columns = [str(column) for column in mapping.columns]
    clinical["PESEL"] = clinical["PESEL"].astype(str).str.strip()
    mapping["PESEL"] = mapping["PESEL"].astype(str).str.strip()
    mapping["patient_id"] = mapping["patient_id"].astype(str).str.strip()

    merged = clinical.merge(mapping, on="PESEL", how="left", indicator=True)
    merged["clinical_t_group"] = merged["tumor_pre.clinical_t"].map(normalize_odbytnice_clinical_t)
    merged["clinical_stage_group"] = merged["tumor_pre.clinical_stage_group"].map(normalize_stage_group)

    keep = ["patient_id", "PESEL", "clinical_t_group", "clinical_stage_group"]
    out = merged[keep].dropna(subset=["patient_id"]).copy()
    out["patient_id"] = out["patient_id"].astype(str)

    meta = {
        "clinical_source_path": str(clinical_path),
        "clinical_rows": int(len(clinical)),
        "mapping_source_path": str(mapping_path),
        "mapping_rows": int(len(mapping)),
        "mapped_rows": int(merged["patient_id"].notna().sum()),
        "clinical_t_group_counts_raw": {
            str(key): int(value)
            for key, value in out["clinical_t_group"].value_counts(dropna=False).sort_index().items()
        },
        "clinical_stage_group_counts_raw": {
            str(key): int(value)
            for key, value in out["clinical_stage_group"].value_counts(dropna=False).sort_index().items()
        },
    }
    logger.info(
        "Resolved Odbytnice clinical source=%s mapping=%s mapped=%d",
        clinical_path,
        mapping_path,
        merged["patient_id"].notna().sum(),
    )
    return out.drop_duplicates(subset=["patient_id"]), meta


def prepare_stage_tables(
    *,
    cohort: str,
    clinical_df: pd.DataFrame,
    min_subjects_per_stratum: int,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for definition in STAGE_DEFINITIONS:
        if definition.cohort != cohort:
            continue
        threshold = max(definition.min_subjects, min_subjects_per_stratum)
        stage_df = clinical_df[["patient_id", definition.name]].copy()
        stage_df = stage_df.rename(columns={definition.name: "stage_stratum"})
        stage_df = stage_df.dropna(subset=["stage_stratum"]).copy()
        counts = stage_df["stage_stratum"].value_counts()
        keep = counts[counts >= threshold].index.tolist()
        stage_df = stage_df.loc[stage_df["stage_stratum"].isin(keep)].copy()
        stage_df["stage_definition"] = definition.name
        stage_df["stage_label"] = definition.label
        out[definition.name] = stage_df
    return out


def summarize_join(
    *,
    cohort: str,
    robustness_subjects: pd.DataFrame,
    clinical_df: pd.DataFrame,
    stage_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    subject_df = robustness_subjects.copy()
    join_rows: list[dict[str, object]] = []
    cohort_meta: dict[str, object] = {
        "robustness_subjects_total": int(len(subject_df)),
        "robustness_patients_total": int(subject_df["patient_id"].nunique()),
        "clinical_patients_total": int(clinical_df["patient_id"].nunique()),
    }

    patient_df = subject_df[["patient_id"]].drop_duplicates()
    patient_join = patient_df.merge(clinical_df, on="patient_id", how="left", indicator=True)
    cohort_meta["clinical_patients_matched_to_robustness"] = int(patient_join["_merge"].eq("both").sum())

    for definition in STAGE_DEFINITIONS:
        if definition.cohort != cohort:
            continue
        stage_df = stage_tables.get(definition.name)
        if stage_df is None:
            continue
        joined = subject_df.merge(
            stage_df[["patient_id", "stage_stratum"]],
            on="patient_id",
            how="left",
        )
        matched = joined["stage_stratum"].notna()
        join_rows.append(
            {
                "cohort": cohort,
                "stage_definition": definition.name,
                "stage_label": definition.label,
                "robustness_subjects_total": int(len(subject_df)),
                "robustness_patients_total": int(subject_df["patient_id"].nunique()),
                "clinical_patients_total": int(clinical_df["patient_id"].nunique()),
                "matched_subjects": int(matched.sum()),
                "matched_patients": int(joined.loc[matched, "patient_id"].nunique()),
                "subjects_missing_stage": int((~matched).sum()),
                "patients_missing_stage": int(
                    joined.loc[~matched, "patient_id"].nunique()
                ),
            }
        )
        for stage_stratum, block in joined.loc[matched].groupby("stage_stratum", sort=True):
            join_rows.append(
                {
                    "cohort": cohort,
                    "stage_definition": definition.name,
                    "stage_label": definition.label,
                    "stage_stratum": str(stage_stratum),
                    "matched_subjects": int(len(block)),
                    "matched_patients": int(block["patient_id"].nunique()),
                }
            )
    return pd.DataFrame(join_rows), cohort_meta


def iter_parquet_batches(parquet_path: Path, batch_size: int):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["patient_id", "course_id", "roi_name", "perturbation_id", "feature_name", "value"],
    ):
        yield batch.to_pandas()


def aggregate_stage_batches(
    *,
    parquet_path: Path,
    cohort: str,
    stage_tables: dict[str, pd.DataFrame],
    batch_size: int,
    logger: logging.Logger,
) -> tuple[dict[str, list[pd.DataFrame]], pd.DataFrame]:
    accum: dict[str, dict[str, list[pd.DataFrame]]] = {
        definition_name: {"subject": [], "col": []} for definition_name in stage_tables
    }
    subject_inventory_frames: list[pd.DataFrame] = []

    patient_maps = {
        definition_name: stage_df.set_index("patient_id")["stage_stratum"]
        for definition_name, stage_df in stage_tables.items()
    }

    start = time.time()
    loaded_rows = 0
    for batch_idx, batch_df in enumerate(iter_parquet_batches(parquet_path, batch_size), start=1):
        loaded_rows += int(len(batch_df))
        batch_df = batch_df.dropna(
            subset=["patient_id", "course_id", "roi_name", "perturbation_id", "feature_name", "value"]
        ).copy()
        if batch_df.empty:
            continue
        batch_df["patient_id"] = batch_df["patient_id"].astype(str).str.strip()
        batch_df["course_id"] = batch_df["course_id"].astype(str).str.strip()
        batch_df["roi_name"] = batch_df["roi_name"].astype(str)
        batch_df["perturbation_id"] = batch_df["perturbation_id"].astype(str)
        batch_df["feature_name"] = batch_df["feature_name"].astype(str)
        batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
        batch_df = batch_df.dropna(subset=["value"])
        if batch_df.empty:
            continue

        subject_inventory_frames.append(batch_df[["patient_id", "course_id"]].drop_duplicates())

        batch_df["value_sq"] = np.square(batch_df["value"].to_numpy(dtype=float))

        for definition_name, patient_map in patient_maps.items():
            sub = batch_df.copy()
            sub["stage_stratum"] = sub["patient_id"].map(patient_map)
            sub = sub.dropna(subset=["stage_stratum"]).copy()
            if sub.empty:
                continue
            sub = sub.groupby(
                [
                    "stage_stratum",
                    "roi_name",
                    "feature_name",
                    "patient_id",
                    "course_id",
                    "perturbation_id",
                ],
                as_index=False,
                sort=False,
            ).agg(
                value=("value", "mean"),
                value_sq=("value_sq", "mean"),
            )
            subject_batch = sub.groupby(
                ["stage_stratum", "roi_name", "feature_name", "patient_id", "course_id"],
                as_index=False,
                sort=False,
            ).agg(
                subject_sum=("value", "sum"),
                subject_sum_sq=("value_sq", "sum"),
                n_values=("value", "size"),
            )
            col_batch = sub.groupby(
                ["stage_stratum", "roi_name", "feature_name", "perturbation_id"],
                as_index=False,
                sort=False,
            ).agg(col_sum=("value", "sum"))
            accum[definition_name]["subject"].append(subject_batch)
            accum[definition_name]["col"].append(col_batch)

        if batch_idx % 10 == 0:
            logger.info(
                "%s raw aggregation: batch=%d loaded_rows=%d elapsed=%.1fs",
                cohort,
                batch_idx,
                loaded_rows,
                time.time() - start,
            )

    if subject_inventory_frames:
        subject_inventory = pd.concat(subject_inventory_frames, ignore_index=True).drop_duplicates()
    else:
        subject_inventory = pd.DataFrame(columns=["patient_id", "course_id"])
    logger.info(
        "%s aggregation complete: loaded_rows=%d subjects=%d elapsed=%.1fs",
        cohort,
        loaded_rows,
        len(subject_inventory),
        time.time() - start,
    )
    return {key: value["subject"] + value["col"] for key, value in accum.items()}, subject_inventory


def finalize_definition_metrics(
    *,
    cohort: str,
    definition_name: str,
    stage_label: str,
    subject_frames: list[pd.DataFrame],
    col_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    if not subject_frames or not col_frames:
        empty_feature = pd.DataFrame(
            columns=[
                "cohort",
                "stage_definition",
                "stage_label",
                "stage_stratum",
                "roi_name",
                "feature_name",
                "feature_family",
                "icc",
                "cov_pct",
                "robustness_class",
                "n_subjects",
                "k_raters",
                "complete_subject_fraction",
            ]
        )
        empty_family = pd.DataFrame(
            columns=[
                "cohort",
                "stage_definition",
                "stage_label",
                "stage_stratum",
                "feature_family",
                "n_feature_cells",
                "median_icc",
                "median_cov_pct",
                "robust_fraction",
                "acceptable_fraction",
                "poor_fraction",
                "rank_icc",
                "rank_robust_fraction",
            ]
        )
        return empty_feature, empty_family, []

    subject_df = pd.concat(subject_frames, ignore_index=True)
    subject_df = (
        subject_df.groupby(
            ["stage_stratum", "roi_name", "feature_name", "patient_id", "course_id"],
            as_index=False,
            sort=False,
        )
        .agg(
            subject_sum=("subject_sum", "sum"),
            subject_sum_sq=("subject_sum_sq", "sum"),
            n_values=("n_values", "sum"),
        )
        .copy()
    )

    col_df = pd.concat(col_frames, ignore_index=True)
    col_df = (
        col_df.groupby(
            ["stage_stratum", "roi_name", "feature_name", "perturbation_id"],
            as_index=False,
            sort=False,
        )
        .agg(col_sum=("col_sum", "sum"))
        .copy()
    )

    k_lookup = (
        col_df.groupby(["stage_stratum", "roi_name", "feature_name"], as_index=False)
        .agg(k_raters=("perturbation_id", "nunique"))
        .copy()
    )
    subject_df = subject_df.merge(
        k_lookup,
        on=["stage_stratum", "roi_name", "feature_name"],
        how="left",
    )
    subject_df["is_complete"] = subject_df["n_values"] == subject_df["k_raters"]

    completeness_rows = []
    for stage_stratum, block in subject_df.groupby("stage_stratum", sort=True):
        completeness_rows.append(
            {
                "cohort": cohort,
                "stage_definition": definition_name,
                "stage_label": stage_label,
                "stage_stratum": str(stage_stratum),
                "subject_rows": int(len(block)),
                "complete_subject_rows": int(block["is_complete"].sum()),
                "complete_subject_fraction": float(block["is_complete"].mean()),
            }
        )

    feature_rows: list[dict[str, object]] = []
    # Per-stratum metrics.
    for stage_stratum, subject_stage in subject_df.groupby("stage_stratum", sort=True):
        col_stage = col_df.loc[col_df["stage_stratum"] == stage_stratum].copy()
        feature_rows.extend(
            compute_feature_rows(
                cohort=cohort,
                definition_name=definition_name,
                stage_label=stage_label,
                stage_stratum=str(stage_stratum),
                subject_df=subject_stage,
                col_df=col_stage,
            )
        )

    # Overall metrics among stage-matched subjects for this definition.
    feature_rows.extend(
        compute_feature_rows(
            cohort=cohort,
            definition_name=definition_name,
            stage_label=stage_label,
            stage_stratum=OVERALL_LABEL,
            subject_df=subject_df,
            col_df=col_df,
        )
    )

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        family_df = pd.DataFrame()
        return feature_df, family_df, completeness_rows

    family_df = summarize_family_metrics(feature_df)
    return feature_df, family_df, completeness_rows


def compute_feature_rows(
    *,
    cohort: str,
    definition_name: str,
    stage_label: str,
    stage_stratum: str,
    subject_df: pd.DataFrame,
    col_df: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subject_block = subject_df.copy()
    col_block = col_df.copy()

    group_keys = ["roi_name", "feature_name"]
    k_lookup = (
        col_block.groupby(group_keys, as_index=False)
        .agg(k_raters=("perturbation_id", "nunique"))
        .copy()
    )
    subject_block = subject_block.merge(k_lookup, on=group_keys, how="left", suffixes=("", "_dup"))
    if "k_raters_dup" in subject_block.columns:
        subject_block["k_raters"] = subject_block["k_raters_dup"].fillna(subject_block["k_raters"])
        subject_block = subject_block.drop(columns=["k_raters_dup"])
    subject_block["is_complete"] = subject_block["n_values"] == subject_block["k_raters"]
    completeness_lookup = (
        subject_block.groupby(group_keys, as_index=False)
        .agg(complete_subject_fraction=("is_complete", "mean"))
        .copy()
    )
    subject_complete = subject_block.loc[subject_block["is_complete"]].copy()
    if subject_complete.empty:
        return rows

    group_agg = (
        subject_complete.groupby(group_keys, as_index=False)
        .agg(
            n_subjects=("patient_id", "size"),
            total_sum=("subject_sum", "sum"),
            total_sum_sq=("subject_sum_sq", "sum"),
            row_sum_sq=("subject_sum", lambda values: float(np.square(values.to_numpy(dtype=float)).sum())),
        )
        .copy()
    )
    group_agg = group_agg.merge(completeness_lookup, on=group_keys, how="left")

    col_block = col_block.sort_values(group_keys + ["perturbation_id"]).copy()
    col_lookup = {}
    for (roi_name, feature_name), block in col_block.groupby(group_keys, sort=False):
        col_lookup[(str(roi_name), str(feature_name))] = block["col_sum"].to_numpy(dtype=float)

    for row in group_agg.itertuples(index=False):
        col_sums = col_lookup.get((str(row.roi_name), str(row.feature_name)))
        if col_sums is None:
            continue
        icc = compute_icc31_from_sufficient_stats(
            n_subjects=float(row.n_subjects),
            total_sum=float(row.total_sum),
            total_sum_sq=float(row.total_sum_sq),
            row_sum_sq=float(row.row_sum_sq),
            col_sums=col_sums,
        )
        cov_pct = compute_cov_from_sufficient_stats(
            total_sum=float(row.total_sum),
            total_sum_sq=float(row.total_sum_sq),
            n_values=float(row.n_subjects) * float(len(col_sums)),
        )
        rows.append(
            {
                "cohort": cohort,
                "stage_definition": definition_name,
                "stage_label": stage_label,
                "stage_stratum": stage_stratum,
                "roi_name": str(row.roi_name),
                "feature_name": str(row.feature_name),
                "feature_family": infer_feature_family(row.feature_name),
                "icc": icc,
                "cov_pct": cov_pct,
                "robustness_class": classify_feature(icc, cov_pct),
                "n_subjects": int(row.n_subjects),
                "k_raters": int(len(col_sums)),
                "complete_subject_fraction": float(row.complete_subject_fraction),
            }
        )
    return rows


def summarize_family_metrics(feature_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.loc[feature_df["feature_family"].isin(FAMILY_ORDER)].copy()
    if out.empty:
        return pd.DataFrame()
    out["is_robust"] = out["robustness_class"].eq("Robust").astype(float)
    out["is_acceptable"] = out["robustness_class"].eq("Acceptable").astype(float)
    out["is_poor"] = out["robustness_class"].eq("Poor").astype(float)

    family_df = (
        out.groupby(
            ["cohort", "stage_definition", "stage_label", "stage_stratum", "feature_family"],
            as_index=False,
        )
        .agg(
            n_feature_cells=("feature_name", "size"),
            median_icc=("icc", "median"),
            median_cov_pct=("cov_pct", "median"),
            robust_fraction=("is_robust", "mean"),
            acceptable_fraction=("is_acceptable", "mean"),
            poor_fraction=("is_poor", "mean"),
        )
        .copy()
    )

    family_df["rank_icc"] = family_df.groupby(
        ["cohort", "stage_definition", "stage_stratum"]
    )["median_icc"].rank(method="average", ascending=False)
    family_df["rank_robust_fraction"] = family_df.groupby(
        ["cohort", "stage_definition", "stage_stratum"]
    )["robust_fraction"].rank(method="average", ascending=False)
    return family_df.sort_values(
        ["cohort", "stage_definition", "stage_stratum", "rank_icc", "feature_family"]
    ).reset_index(drop=True)


def compute_rank_stability(family_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if family_df.empty:
        return pd.DataFrame(rows)

    for (cohort, definition_name), block in family_df.groupby(["cohort", "stage_definition"], sort=True):
        strata = sorted(block["stage_stratum"].astype(str).unique().tolist())
        non_overall = [value for value in strata if value != OVERALL_LABEL]
        comparisons: list[tuple[str, str, str]] = []
        for stage_stratum in non_overall:
            if OVERALL_LABEL in strata:
                comparisons.append((stage_stratum, OVERALL_LABEL, "vs_overall"))
        for idx, left in enumerate(non_overall):
            for right in non_overall[idx + 1 :]:
                comparisons.append((left, right, "pairwise"))

        for left, right, comparison_type in comparisons:
            left_df = block.loc[block["stage_stratum"] == left].copy()
            right_df = block.loc[block["stage_stratum"] == right].copy()
            merged = left_df.merge(
                right_df,
                on="feature_family",
                how="inner",
                suffixes=("_left", "_right"),
            )
            if merged.empty:
                continue
            for metric in ["rank_icc", "rank_robust_fraction"]:
                left_metric = merged[f"{metric}_left"].to_numpy(dtype=float)
                right_metric = merged[f"{metric}_right"].to_numpy(dtype=float)
                rho, p_value = sp_stats.spearmanr(left_metric, right_metric)
                footrule = float(np.abs(left_metric - right_metric).sum())
                order_left = " > ".join(
                    merged.sort_values(f"{metric}_left")["feature_family"].astype(str).tolist()
                )
                order_right = " > ".join(
                    merged.sort_values(f"{metric}_right")["feature_family"].astype(str).tolist()
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "stage_definition": definition_name,
                        "comparison_type": comparison_type,
                        "left_stratum": left,
                        "right_stratum": right,
                        "metric": metric,
                        "n_families": int(len(merged)),
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "spearman_p": float(p_value) if np.isfinite(p_value) else np.nan,
                        "footrule_distance": footrule,
                        "left_order": order_left,
                        "right_order": order_right,
                    }
                )
    return pd.DataFrame(rows)


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.job_id)
    t0 = time.time()

    manifest: dict[str, object] = {
        "job_id": args.job_id,
        "hostname": socket.gethostname(),
        "git_sha": resolve_git_sha(),
        "started_unix": t0,
        "batch_size": args.batch_size,
        "min_subjects_per_stratum": args.min_subjects_per_stratum,
        "cohorts": {},
    }

    all_join_rows: list[pd.DataFrame] = []
    all_feature_rows: list[pd.DataFrame] = []
    all_family_rows: list[pd.DataFrame] = []
    all_completeness_rows: list[dict[str, object]] = []

    clinical_loaders = {
        "Prostata": lambda: resolve_prostate_clinical(logger),
        "Odbytnice": lambda: resolve_odbytnice_clinical(logger, args.odbytnice_map),
    }

    for cohort in ["Prostata", "Odbytnice"]:
        logger.info("Starting cohort %s", cohort)
        parquet_path = ROBUSTNESS_PATHS[cohort]
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing corrected robustness parquet: {parquet_path}")

        clinical_df, clinical_meta = clinical_loaders[cohort]()
        stage_tables = prepare_stage_tables(
            cohort=cohort,
            clinical_df=clinical_df,
            min_subjects_per_stratum=args.min_subjects_per_stratum,
        )
        if not stage_tables:
            raise RuntimeError(f"No usable stage tables for cohort {cohort}.")

        # Subject inventory is built from the raw parquet, not from the clinical join.
        subject_inventory_frames: list[pd.DataFrame] = []
        accum_subject: dict[str, list[pd.DataFrame]] = {name: [] for name in stage_tables}
        accum_col: dict[str, list[pd.DataFrame]] = {name: [] for name in stage_tables}
        patient_maps = {
            definition_name: stage_df.set_index("patient_id")["stage_stratum"]
            for definition_name, stage_df in stage_tables.items()
        }

        start = time.time()
        loaded_rows = 0
        for batch_idx, batch_df in enumerate(iter_parquet_batches(parquet_path, args.batch_size), start=1):
            loaded_rows += int(len(batch_df))
            batch_df = batch_df.dropna(
                subset=["patient_id", "course_id", "roi_name", "perturbation_id", "feature_name", "value"]
            ).copy()
            if batch_df.empty:
                continue
            batch_df["patient_id"] = batch_df["patient_id"].astype(str).str.strip()
            batch_df["course_id"] = batch_df["course_id"].astype(str).str.strip()
            batch_df["roi_name"] = batch_df["roi_name"].astype(str)
            batch_df["perturbation_id"] = batch_df["perturbation_id"].astype(str)
            batch_df["feature_name"] = batch_df["feature_name"].astype(str)
            batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
            batch_df = batch_df.dropna(subset=["value"])
            if batch_df.empty:
                continue

            subject_inventory_frames.append(batch_df[["patient_id", "course_id"]].drop_duplicates())
            batch_df["value_sq"] = np.square(batch_df["value"].to_numpy(dtype=float))

            for definition_name, patient_map in patient_maps.items():
                sub = batch_df.copy()
                sub["stage_stratum"] = sub["patient_id"].map(patient_map)
                sub = sub.dropna(subset=["stage_stratum"]).copy()
                if sub.empty:
                    continue
                sub = sub.groupby(
                    [
                        "stage_stratum",
                        "roi_name",
                        "feature_name",
                        "patient_id",
                        "course_id",
                        "perturbation_id",
                    ],
                    as_index=False,
                    sort=False,
                ).agg(
                    value=("value", "mean"),
                    value_sq=("value_sq", "mean"),
                )
                subject_batch = sub.groupby(
                    ["stage_stratum", "roi_name", "feature_name", "patient_id", "course_id"],
                    as_index=False,
                    sort=False,
                ).agg(
                    subject_sum=("value", "sum"),
                    subject_sum_sq=("value_sq", "sum"),
                    n_values=("value", "size"),
                )
                col_batch = sub.groupby(
                    ["stage_stratum", "roi_name", "feature_name", "perturbation_id"],
                    as_index=False,
                    sort=False,
                ).agg(col_sum=("value", "sum"))
                accum_subject[definition_name].append(subject_batch)
                accum_col[definition_name].append(col_batch)

            if batch_idx % 10 == 0:
                logger.info(
                    "%s batch=%d loaded_rows=%d elapsed=%.1fs",
                    cohort,
                    batch_idx,
                    loaded_rows,
                    time.time() - start,
                )

        if subject_inventory_frames:
            subject_inventory = (
                pd.concat(subject_inventory_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
            )
        else:
            subject_inventory = pd.DataFrame(columns=["patient_id", "course_id"])

        join_df, join_meta = summarize_join(
            cohort=cohort,
            robustness_subjects=subject_inventory,
            clinical_df=clinical_df,
            stage_tables=stage_tables,
        )
        all_join_rows.append(join_df)

        cohort_manifest = {
            "robustness_path": str(parquet_path),
            "robustness_sha256": sha256_file(parquet_path),
            "loaded_rows": loaded_rows,
            "elapsed_seconds": time.time() - start,
            "clinical_meta": clinical_meta,
            "join_meta": join_meta,
            "stage_tables": {},
        }

        for definition in STAGE_DEFINITIONS:
            if definition.cohort != cohort:
                continue
            stage_df = stage_tables.get(definition.name)
            subject_frames = accum_subject.get(definition.name, [])
            col_frames = accum_col.get(definition.name, [])
            feature_df, family_df, completeness_rows = finalize_definition_metrics(
                cohort=cohort,
                definition_name=definition.name,
                stage_label=definition.label,
                subject_frames=subject_frames,
                col_frames=col_frames,
            )
            all_completeness_rows.extend(completeness_rows)
            if not feature_df.empty:
                all_feature_rows.append(feature_df)
            if not family_df.empty:
                all_family_rows.append(family_df)

            stage_counts = {}
            if stage_df is not None and not stage_df.empty:
                stage_counts = {
                    str(key): int(value)
                    for key, value in stage_df["stage_stratum"].value_counts().sort_index().items()
                }
            cohort_manifest["stage_tables"][definition.name] = {
                "stage_label": definition.label,
                "eligible_patient_counts": stage_counts,
                "feature_rows": int(len(feature_df)),
                "family_rows": int(len(family_df)),
            }

        manifest["cohorts"][cohort] = cohort_manifest

    join_out = pd.concat(all_join_rows, ignore_index=True) if all_join_rows else pd.DataFrame()
    feature_out = (
        pd.concat(all_feature_rows, ignore_index=True)
        if all_feature_rows
        else pd.DataFrame()
    )
    family_out = (
        pd.concat(all_family_rows, ignore_index=True)
        if all_family_rows
        else pd.DataFrame()
    )
    completeness_out = pd.DataFrame(all_completeness_rows)
    rank_out = compute_rank_stability(family_out)

    join_path = args.output_dir / "stage42_join_summary.csv"
    family_path = args.output_dir / "stage42_family_metrics.csv"
    rank_path = args.output_dir / "stage42_rank_stability.csv"
    feature_path = args.output_dir / "stage42_feature_metrics.parquet"
    completeness_path = args.output_dir / "stage42_completeness.csv"
    manifest_path = args.output_dir / "stage42_manifest.json"

    join_out.to_csv(join_path, index=False)
    family_out.to_csv(family_path, index=False)
    rank_out.to_csv(rank_path, index=False)
    completeness_out.to_csv(completeness_path, index=False)
    feature_out.to_parquet(feature_path, index=False)

    manifest["outputs"] = {
        "join_summary_csv": str(join_path),
        "family_metrics_csv": str(family_path),
        "rank_stability_csv": str(rank_path),
        "completeness_csv": str(completeness_path),
        "feature_metrics_parquet": str(feature_path),
    }
    manifest["completed_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["completed_unix"] - t0
    write_manifest(manifest_path, manifest)

    logger.info("Wrote %s", join_path)
    logger.info("Wrote %s", family_path)
    logger.info("Wrote %s", rank_path)
    logger.info("Wrote %s", completeness_path)
    logger.info("Wrote %s", feature_path)
    logger.info("Wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
