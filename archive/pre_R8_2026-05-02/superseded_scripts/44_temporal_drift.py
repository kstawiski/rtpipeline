#!/usr/bin/env python3
"""Gap #44: treatment-era / scanner-generation drift on feature stability.

This analysis recomputes ICC(3,1) within treatment-era bins from the raw
`robustness_by_cohort` stores so the trend test uses the same statistic as the
main manuscript pipeline.

Primary summaries per cohort-bin:
  - median feature ICC(3,1)
  - robust-feature share (ICC >= 0.90 and CoV <= 10%)
  - median CoV

Primary trend test:
  - Mann-Kendall monotonic trend on ordered treatment-era bins, implemented via
    Kendall's tau on bin order.

Date sources:
  - Odbytnice: `clinical_clean.parquet:treatment.rcht_start_date` joined via
    `patient_id_pesel_mapping.csv`, with `metadata_all.parquet:course_id`
    year-month fallback when the exact clinical date is unavailable.
  - Prostata: `metadata_summary.xlsx:course_start_date` joined by `patient_id`,
    with `metadata_all.parquet:course_id` year-month fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy import stats as sp_stats

ICC_ROBUST = 0.90
COV_ROBUST = 10.0
ICC_ACCEPTABLE = 0.75
COV_ACCEPTABLE = 20.0
MIN_SUBJECTS_FOR_ICC = 5

DEFAULT_REQUESTED_BINS = 4
DEFAULT_MIN_SUBJECTS_PER_BIN = 40

ANALYSIS_ROOT = Path("/home/kgs24/rtpipeline_manuscript/analysis")
ROBUSTNESS_ROOT = ANALYSIS_ROOT / "data" / "robustness_by_cohort"
METADATA_ALL_PATH = ANALYSIS_ROOT / "data" / "metadata_all.parquet"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manuscript" / "analysis" / "tables"

ODBYTNICE_CLINICAL_PATH = Path("/umed-projekty/ODBYTNICE2026/analysis/clinical_clean.parquet")
ODBYTNICE_MAPPING_PATH = Path("/umed-projekty/ODBYTNICE2026/analysis/patient_id_pesel_mapping.csv")

PROSTATA_METADATA_XLSX = Path("/projekty/Prostata/Data_Snakemake/metadata_summary.xlsx")
RAW_COURSE_ROOT = Path("/umed-projekty/DICOMRT-datasets")

FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
COURSE_ID_YM_RE = re.compile(r"(?P<year>20\d{2})-(?P<month>\d{2})")
COHORT_BODY_REGION = {
    "Odbytnice": "Pelvis",
    "Prostata": "Pelvis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["Odbytnice", "Prostata"],
        help="Cohorts to analyze. Default: Odbytnice Prostata",
    )
    parser.add_argument(
        "--requested-bins",
        type=int,
        default=DEFAULT_REQUESTED_BINS,
        help="Target number of treatment-era bins per cohort before min-size reduction.",
    )
    parser.add_argument(
        "--min-subjects-per-bin",
        type=int,
        default=DEFAULT_MIN_SUBJECTS_PER_BIN,
        help="Minimum unique subjects per bin.",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=ANALYSIS_ROOT,
        help=f"Analysis root containing metadata and robustness_by_cohort. Default: {ANALYSIS_ROOT}",
    )
    parser.add_argument(
        "--metadata-all-path",
        type=Path,
        default=None,
        help="Override metadata_all parquet path. Default: <analysis-root>/data/metadata_all.parquet",
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=None,
        help="Override robustness_by_cohort directory. Default: <analysis-root>/data/robustness_by_cohort",
    )
    parser.add_argument(
        "--raw-course-root",
        type=Path,
        default=RAW_COURSE_ROOT,
        help=(
            "Root of local per-course radiomics_robustness_ct.parquet trees. "
            f"Default: {RAW_COURSE_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for #44 CSV/JSON outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def parse_course_id_fallback(course_id: object) -> pd.Timestamp | pd.NaT:
    match = COURSE_ID_YM_RE.search(str(course_id))
    if not match:
        return pd.NaT
    year = int(match.group("year"))
    month = int(match.group("month"))
    return pd.Timestamp(year=year, month=month, day=1)


def standardize_patient_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def standardize_pesel(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype(str)


def icc31_from_matrix(values: np.ndarray) -> tuple[float, float, float] | None:
    n_subjects, n_raters = values.shape
    if n_subjects < MIN_SUBJECTS_FOR_ICC or n_raters < 3:
        return None
    grand_mean = values.mean()
    subject_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)
    ssb = n_raters * ((subject_means - grand_mean) ** 2).sum()
    ssr = n_subjects * ((rater_means - grand_mean) ** 2).sum()
    sst = ((values - grand_mean) ** 2).sum()
    sse = sst - ssb - ssr
    df_subjects = n_subjects - 1
    df_raters = n_raters - 1
    df_error = df_subjects * df_raters
    if df_subjects <= 0 or df_error <= 0:
        return None
    bms = ssb / df_subjects
    ems = sse / df_error
    denom = bms + (n_raters - 1) * ems
    if denom == 0:
        return None
    icc = (bms - ems) / denom
    if ems == 0:
        return float(icc), float(icc), float(icc)
    f_value = bms / ems
    f_low = f_value / sp_stats.f.ppf(0.975, df_subjects, df_error)
    ci_low = (f_low - 1) / (f_low + n_raters - 1)
    f_high = f_value * sp_stats.f.ppf(0.975, df_error, df_subjects)
    ci_high = (f_high - 1) / (f_high + n_raters - 1)
    return float(icc), float(ci_low), float(ci_high)


def get_feature_family(feature_name: str) -> str:
    name = str(feature_name).lower()
    for family in FAMILY_ORDER:
        if f"_{family}_" in name:
            return family
    return "unknown"


def load_metadata_all(metadata_all_path: Path) -> pd.DataFrame:
    meta = pd.read_parquet(metadata_all_path)
    meta["patient_id"] = standardize_patient_id(meta["patient_id"])
    meta["course_id"] = meta["course_id"].astype(str).str.strip()
    return meta


def load_local_course_metadata(
    raw_course_root: Path,
    cohorts: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for cohort in cohorts:
        root = raw_course_root / cohort / "output"
        for case_metadata_path in root.glob("*/*/metadata/case_metadata.json"):
            with case_metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records.append(
                {
                    "cohort": cohort,
                    "patient_id": payload.get("patient_id"),
                    "course_id": payload.get("course_id"),
                    "ct_model": payload.get("ct_model"),
                    "ct_convolution_kernel": payload.get("ct_convolution_kernel"),
                    "ct_slice_thickness": payload.get("ct_slice_thickness"),
                }
            )

    meta = pd.DataFrame.from_records(records)
    meta["patient_id"] = standardize_patient_id(meta["patient_id"])
    meta["course_id"] = meta["course_id"].astype(str).str.strip()
    return meta


def build_odbytnice_date_map(meta_all: pd.DataFrame) -> pd.DataFrame:
    meta = meta_all.loc[meta_all["cohort"] == "Odbytnice", [
        "patient_id",
        "course_id",
        "ct_model",
        "ct_convolution_kernel",
        "ct_slice_thickness",
    ]].drop_duplicates()
    mapping = pd.read_csv(ODBYTNICE_MAPPING_PATH)
    mapping["patient_id"] = standardize_patient_id(mapping["patient_id"])
    mapping["PESEL"] = standardize_pesel(mapping["PESEL"])
    clinical = pd.read_parquet(
        ODBYTNICE_CLINICAL_PATH,
        columns=["PESEL", "treatment.rcht_start_date"],
    )
    clinical["PESEL"] = standardize_pesel(clinical["PESEL"])
    clinical["exact_date"] = pd.to_datetime(
        clinical["treatment.rcht_start_date"], errors="coerce"
    )
    clinical = clinical[["PESEL", "exact_date"]].drop_duplicates()

    merged = meta.merge(mapping, on="patient_id", how="left").merge(
        clinical, on="PESEL", how="left"
    )
    merged["fallback_date"] = merged["course_id"].map(parse_course_id_fallback)
    merged["analysis_date"] = merged["exact_date"].fillna(merged["fallback_date"])
    merged["date_source"] = np.where(
        merged["exact_date"].notna(),
        "clinical_clean:treatment.rcht_start_date",
        np.where(
            merged["fallback_date"].notna(),
            "metadata_all:course_id_yyyy_mm_fallback",
            "missing",
        ),
    )
    return merged


def build_prostata_date_map(meta_all: pd.DataFrame) -> pd.DataFrame:
    meta = meta_all.loc[meta_all["cohort"] == "Prostata", [
        "patient_id",
        "course_id",
        "ct_model",
        "ct_convolution_kernel",
        "ct_slice_thickness",
    ]].drop_duplicates()
    summary = pd.read_excel(
        PROSTATA_METADATA_XLSX,
        usecols=["patient_id", "course_start_date", "course_end_date"],
    )
    summary["patient_id"] = standardize_patient_id(summary["patient_id"])
    summary["exact_date"] = pd.to_datetime(summary["course_start_date"], errors="coerce")
    summary = summary[["patient_id", "exact_date"]].drop_duplicates()

    merged = meta.merge(summary, on="patient_id", how="left")
    merged["fallback_date"] = merged["course_id"].map(parse_course_id_fallback)
    merged["analysis_date"] = merged["exact_date"].fillna(merged["fallback_date"])
    merged["date_source"] = np.where(
        merged["exact_date"].notna(),
        "metadata_summary:course_start_date",
        np.where(
            merged["fallback_date"].notna(),
            "metadata_all:course_id_yyyy_mm_fallback",
            "missing",
        ),
    )
    return merged


def assign_treatment_bins(
    date_map: pd.DataFrame, requested_bins: int, min_subjects_per_bin: int
) -> pd.DataFrame:
    work = date_map.copy()
    work = work.loc[work["analysis_date"].notna()].copy()
    if work.empty:
        raise RuntimeError("No analyzable dates were resolved.")

    n_subjects = len(work)
    bin_count = requested_bins
    while bin_count > 2 and (n_subjects / bin_count) < min_subjects_per_bin:
        bin_count -= 1
    ranks = work["analysis_date"].rank(method="first")
    work["bin_index"] = pd.qcut(ranks, q=bin_count, labels=False, duplicates="drop")
    work["bin_index"] = work["bin_index"].astype(int)

    labels: dict[int, str] = {}
    for bin_index, frame in work.groupby("bin_index", sort=True):
        start = frame["analysis_date"].min().date().isoformat()
        end = frame["analysis_date"].max().date().isoformat()
        labels[int(bin_index)] = f"B{int(bin_index) + 1}: {start} to {end}"
    work["bin_label"] = work["bin_index"].map(labels)
    return work


def attach_raw_course_paths(
    cohort: str,
    date_map: pd.DataFrame,
    raw_course_root: Path,
) -> pd.DataFrame:
    work = date_map.copy()
    root = raw_course_root / cohort / "output"
    work["raw_parquet_path"] = work.apply(
        lambda row: root
        / str(row["patient_id"]).strip()
        / str(row["course_id"]).strip()
        / "radiomics_robustness_ct.parquet",
        axis=1,
    )
    work["raw_parquet_exists"] = work["raw_parquet_path"].map(lambda path: path.exists())
    return work


def load_bin_rows(
    cohort: str,
    cohort_path: Path,
    subject_map: pd.DataFrame,
    raw_course_root: Path | None = None,
) -> pd.DataFrame:
    if raw_course_root is not None:
        return load_bin_rows_from_raw_courses(cohort, subject_map)

    patient_ids = subject_map["patient_id"].astype(str).unique().tolist()
    dataset = ds.dataset(cohort_path, format="parquet")
    table = dataset.to_table(
        columns=[
            "patient_id",
            "course_id",
            "cohort",
            "body_region",
            "roi_name",
            "feature_name",
            "perturbation_id",
            "value",
        ],
        filter=ds.field("patient_id").isin(patient_ids),
    )
    frame = table.to_pandas()
    frame["patient_id"] = standardize_patient_id(frame["patient_id"])
    frame["course_id"] = frame["course_id"].astype(str).str.strip()
    merged = frame.merge(
        subject_map[
            [
                "patient_id",
                "course_id",
                "analysis_date",
                "date_source",
                "bin_index",
                "bin_label",
                "ct_model",
                "ct_convolution_kernel",
                "ct_slice_thickness",
            ]
        ],
        on=["patient_id", "course_id"],
        how="inner",
    )
    merged["subject"] = (
        merged["patient_id"].astype(str).str.strip()
        + "_"
        + merged["course_id"].astype(str).str.strip()
    )
    return merged


def load_bin_rows_from_raw_courses(
    cohort: str,
    subject_map: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required_columns = [
        "patient_id",
        "course_id",
        "roi_name",
        "feature_name",
        "perturbation_id",
        "value",
    ]
    course_rows = subject_map[
        [
            "patient_id",
            "course_id",
            "analysis_date",
            "date_source",
            "bin_index",
            "bin_label",
            "ct_model",
            "ct_convolution_kernel",
            "ct_slice_thickness",
            "raw_parquet_path",
            "raw_parquet_exists",
        ]
    ].drop_duplicates()
    course_rows = course_rows.loc[course_rows["raw_parquet_exists"]].copy()
    if course_rows.empty:
        return pd.DataFrame()

    for row in course_rows.itertuples(index=False):
        frame = pd.read_parquet(row.raw_parquet_path, columns=required_columns)
        frame["patient_id"] = standardize_patient_id(frame["patient_id"])
        frame["course_id"] = frame["course_id"].astype(str).str.strip()
        frame["cohort"] = cohort
        frame["body_region"] = COHORT_BODY_REGION[cohort]
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)
    merged = frame.merge(
        course_rows.drop(columns=["raw_parquet_exists"]),
        on=["patient_id", "course_id"],
        how="inner",
    )
    merged["subject"] = (
        merged["patient_id"].astype(str).str.strip()
        + "_"
        + merged["course_id"].astype(str).str.strip()
    )
    return merged


def compute_bin_feature_icc(bin_rows: pd.DataFrame) -> pd.DataFrame:
    cohort = str(bin_rows["cohort"].iloc[0])
    body_region = str(bin_rows["body_region"].iloc[0])
    records: list[dict[str, object]] = []

    for roi_name, roi_frame in bin_rows.groupby("roi_name", sort=True):
        raters = sorted(roi_frame["perturbation_id"].dropna().astype(str).unique().tolist())
        n_raters = len(raters)
        if n_raters < 3:
            continue
        subject_counts = (
            roi_frame.groupby(["feature_name", "subject"], observed=True)["perturbation_id"]
            .nunique()
            .rename("n_raters")
            .reset_index()
        )
        complete_subjects = subject_counts.loc[
            subject_counts["n_raters"] == n_raters, ["feature_name", "subject"]
        ]
        if complete_subjects.empty:
            continue
        complete = roi_frame.merge(
            complete_subjects, on=["feature_name", "subject"], how="inner"
        )
        n_subjects = (
            complete.groupby("feature_name", observed=True)["subject"]
            .nunique()
            .rename("n_subjects")
        )
        keep_features = n_subjects[n_subjects >= MIN_SUBJECTS_FOR_ICC]
        if keep_features.empty:
            continue
        complete = complete.loc[complete["feature_name"].isin(keep_features.index)].copy()
        n_subjects = keep_features.astype(int)

        grand_mean = (
            complete.groupby("feature_name", observed=True)["value"].mean().rename("grand_mean")
        )
        work = complete.merge(grand_mean, on="feature_name", how="left")
        sst = (
            ((work["value"] - work["grand_mean"]) ** 2)
            .groupby(work["feature_name"], observed=True)
            .sum()
            .rename("sst")
        )

        subject_stats = (
            complete.groupby(["feature_name", "subject"], observed=True)["value"]
            .agg(subject_mean="mean", subject_sd=lambda values: float(np.std(values, ddof=1)))
            .reset_index()
        )
        subject_stats = subject_stats.merge(grand_mean, on="feature_name", how="left")
        ssb = (
            (subject_stats["subject_mean"] - subject_stats["grand_mean"]) ** 2
        ).groupby(subject_stats["feature_name"], observed=True).sum() * n_raters
        ssb = ssb.rename("ssb")

        rater_means = (
            complete.groupby(["feature_name", "perturbation_id"], observed=True)["value"]
            .mean()
            .rename("rater_mean")
            .reset_index()
        )
        rater_means = rater_means.merge(grand_mean, on="feature_name", how="left")
        rater_means = rater_means.merge(
            n_subjects.rename("n_subjects"), on="feature_name", how="left"
        )
        ssr = (
            rater_means["n_subjects"] * (rater_means["rater_mean"] - rater_means["grand_mean"]) ** 2
        ).groupby(rater_means["feature_name"], observed=True).sum()
        ssr = ssr.rename("ssr")

        cov_nonzero = subject_stats.loc[np.abs(subject_stats["subject_mean"]) > 1e-10].copy()
        cov_nonzero["cov"] = (
            cov_nonzero["subject_sd"] / np.abs(cov_nonzero["subject_mean"]) * 100.0
        )
        cov_percent = (
            cov_nonzero.groupby("feature_name", observed=True)["cov"].median().rename("cov_percent")
        )

        stats_df = pd.concat([n_subjects, grand_mean, sst, ssb, ssr, cov_percent], axis=1)
        stats_df["sse"] = stats_df["sst"] - stats_df["ssb"] - stats_df["ssr"]
        stats_df["df_subjects"] = stats_df["n_subjects"] - 1
        stats_df["df_error"] = stats_df["df_subjects"] * (n_raters - 1)
        stats_df["bms"] = stats_df["ssb"] / stats_df["df_subjects"]
        stats_df["ems"] = stats_df["sse"] / stats_df["df_error"]
        stats_df["icc"] = (
            (stats_df["bms"] - stats_df["ems"])
            / (stats_df["bms"] + (n_raters - 1) * stats_df["ems"])
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            f_value = stats_df["bms"] / stats_df["ems"]
            f_low = f_value / sp_stats.f.ppf(0.975, stats_df["df_subjects"], stats_df["df_error"])
            f_high = f_value * sp_stats.f.ppf(0.975, stats_df["df_error"], stats_df["df_subjects"])
            stats_df["icc_ci_low"] = (f_low - 1.0) / (f_low + n_raters - 1.0)
            stats_df["icc_ci_high"] = (f_high - 1.0) / (f_high + n_raters - 1.0)

        zero_ems = stats_df["ems"] == 0
        stats_df.loc[zero_ems, "icc_ci_low"] = stats_df.loc[zero_ems, "icc"]
        stats_df.loc[zero_ems, "icc_ci_high"] = stats_df.loc[zero_ems, "icc"]
        stats_df["classification"] = np.where(
            (stats_df["icc"] >= ICC_ROBUST) & (stats_df["cov_percent"] <= COV_ROBUST),
            "Robust",
            np.where(
                (stats_df["icc"] >= ICC_ACCEPTABLE)
                & (stats_df["cov_percent"] <= COV_ACCEPTABLE),
                "Acceptable",
                "Poor",
            ),
        )

        for feature_name, row in stats_df.reset_index(names="feature_name").iterrows():
            records.append(
                {
                    "cohort": cohort,
                    "body_region": body_region,
                    "roi_name": roi_name,
                    "feature_name": row["feature_name"],
                    "feature_family": get_feature_family(row["feature_name"]),
                    "n_subjects": int(row["n_subjects"]),
                    "n_raters": int(n_raters),
                    "icc": float(row["icc"]),
                    "icc_ci_low": float(row["icc_ci_low"]),
                    "icc_ci_high": float(row["icc_ci_high"]),
                    "cov_percent": float(row["cov_percent"]),
                    "qcd": float("nan"),
                    "classification": str(row["classification"]),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_scanners(subject_map: pd.DataFrame) -> list[dict[str, object]]:
    scanner = (
        subject_map.groupby(
            ["ct_model", "ct_convolution_kernel", "ct_slice_thickness"], dropna=False
        )
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["n_subjects", "ct_model"], ascending=[False, True])
    )
    records = []
    total = int(scanner["n_subjects"].sum())
    for row in scanner.itertuples(index=False):
        records.append(
            {
                "ct_model": None if pd.isna(row.ct_model) else str(row.ct_model),
                "ct_convolution_kernel": None
                if pd.isna(row.ct_convolution_kernel)
                else str(row.ct_convolution_kernel),
                "ct_slice_thickness": None
                if pd.isna(row.ct_slice_thickness)
                else float(row.ct_slice_thickness),
                "n_subjects": int(row.n_subjects),
                "share": float(row.n_subjects / total) if total else float("nan"),
            }
        )
    return records


def summarize_scanner_generation_feasibility(
    subject_dates: pd.DataFrame,
    min_subjects_per_group: int,
) -> dict[str, object]:
    counts = (
        subject_dates.assign(
            ct_model=subject_dates["ct_model"].fillna("missing").astype(str).str.strip()
        )
        .groupby("ct_model", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    eligible = counts[counts >= min_subjects_per_group]
    dominant_share = float(counts.iloc[0] / counts.sum()) if not counts.empty else float("nan")

    if counts.empty:
        feasible = False
        reason = "no_scanner_metadata"
    elif len(counts) == 1:
        feasible = False
        reason = "single_scanner_model"
    elif len(eligible) < 2:
        feasible = False
        reason = "insufficient_subjects_in_non_dominant_scanner_model"
    else:
        feasible = True
        reason = "at_least_two_scanner_models_with_sufficient_subjects"

    return {
        "feasible": feasible,
        "reason": reason,
        "min_subjects_per_group": int(min_subjects_per_group),
        "n_scanner_models": int(len(counts)),
        "n_eligible_scanner_models": int(len(eligible)),
        "dominant_scanner_model": None if counts.empty else str(counts.index[0]),
        "dominant_scanner_share": dominant_share,
        "scanner_model_counts": {str(k): int(v) for k, v in counts.items()},
    }


def mann_kendall(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 3:
        return {"tau": None, "p_value": None, "theil_sen_slope_per_bin": None}
    x = np.arange(valid.sum(), dtype=float)
    y = arr[valid]
    tau, p_value = sp_stats.kendalltau(x, y)
    slope = sp_stats.theilslopes(y, x).slope if len(y) >= 2 else np.nan
    return {
        "tau": None if not np.isfinite(tau) else float(tau),
        "p_value": None if not np.isfinite(p_value) else float(p_value),
        "theil_sen_slope_per_bin": None if not np.isfinite(slope) else float(slope),
    }


def make_cohort_summary(
    cohort: str,
    subject_dates: pd.DataFrame,
    feature_results_by_bin: list[pd.DataFrame],
    min_subjects_per_bin: int,
) -> dict[str, object]:
    bin_summaries: list[dict[str, object]] = []
    median_icc_series: list[float] = []
    robust_share_series: list[float] = []
    median_cov_series: list[float] = []

    for feature_results in feature_results_by_bin:
        if feature_results.empty:
            continue
        bin_index = int(feature_results["bin_index"].iloc[0])
        bin_label = str(feature_results["bin_label"].iloc[0])
        bin_subjects = subject_dates.loc[subject_dates["bin_index"] == bin_index].copy()
        median_icc = float(feature_results["icc"].median())
        robust_share = float((feature_results["classification"] == "Robust").mean())
        median_cov = float(feature_results["cov_percent"].median())
        median_icc_series.append(median_icc)
        robust_share_series.append(robust_share)
        median_cov_series.append(median_cov)
        bin_summaries.append(
            {
                "bin_index": bin_index,
                "bin_label": bin_label,
                "date_start": bin_subjects["analysis_date"].min().date().isoformat(),
                "date_end": bin_subjects["analysis_date"].max().date().isoformat(),
                "n_subjects": int(len(bin_subjects)),
                "n_unique_patients": int(bin_subjects["patient_id"].nunique()),
                "n_feature_roi_pairs": int(len(feature_results)),
                "median_icc": median_icc,
                "median_cov_percent": median_cov,
                "robust_share": robust_share,
                "acceptable_share": float(
                    (feature_results["classification"] == "Acceptable").mean()
                ),
                "poor_share": float(
                    (feature_results["classification"] == "Poor").mean()
                ),
                "scanner_mix": summarize_scanners(bin_subjects),
            }
        )

    drift_call = "null"
    trend_icc = mann_kendall(median_icc_series)
    trend_robust = mann_kendall(robust_share_series)
    trend_cov = mann_kendall(median_cov_series)
    scanner_generation = summarize_scanner_generation_feasibility(
        subject_dates=subject_dates,
        min_subjects_per_group=min_subjects_per_bin,
    )
    p_values = [
        value
        for value in [
            trend_icc["p_value"],
            trend_robust["p_value"],
            trend_cov["p_value"],
        ]
        if value is not None
    ]
    if p_values and min(p_values) < 0.05:
        drift_call = "present"

    source_counts = (
        subject_dates["date_source"].value_counts(dropna=False).sort_index().to_dict()
    )
    return {
        "cohort": cohort,
        "n_subjects_analyzed": int(len(subject_dates)),
        "n_unique_patients": int(subject_dates["patient_id"].nunique()),
        "treatment_date_coverage": {
            "min_date": subject_dates["analysis_date"].min().date().isoformat(),
            "max_date": subject_dates["analysis_date"].max().date().isoformat(),
            "date_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        },
        "binning_strategy": {
            "type": "treatment_start_quantile_bins",
            "n_bins": int(len(bin_summaries)),
            "min_subjects_per_bin_target": int(min_subjects_per_bin),
        },
        "scanner_generation_feasibility": scanner_generation,
        "trend_tests": {
            "median_icc_mann_kendall": trend_icc,
            "robust_share_mann_kendall": trend_robust,
            "median_cov_mann_kendall": trend_cov,
        },
        "delta_first_to_last_bin": {
            "median_icc": None
            if len(bin_summaries) < 2
            else float(bin_summaries[-1]["median_icc"] - bin_summaries[0]["median_icc"]),
            "robust_share": None
            if len(bin_summaries) < 2
            else float(
                bin_summaries[-1]["robust_share"] - bin_summaries[0]["robust_share"]
            ),
            "median_cov_percent": None
            if len(bin_summaries) < 2
            else float(
                bin_summaries[-1]["median_cov_percent"]
                - bin_summaries[0]["median_cov_percent"]
            ),
        },
        "drift_call": drift_call,
        "bins": bin_summaries,
    }


def flatten_outputs(
    payload: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    scanner_rows: list[dict[str, object]] = []

    for cohort_summary in payload["cohorts"]:
        cohort = str(cohort_summary["cohort"])
        treatment_cov = cohort_summary["treatment_date_coverage"]
        trend_tests = cohort_summary["trend_tests"]
        scanner_generation = cohort_summary["scanner_generation_feasibility"]

        cohort_rows.append(
            {
                "cohort": cohort,
                "n_subjects_analyzed": cohort_summary["n_subjects_analyzed"],
                "n_unique_patients": cohort_summary["n_unique_patients"],
                "min_date": treatment_cov["min_date"],
                "max_date": treatment_cov["max_date"],
                "n_bins": cohort_summary["binning_strategy"]["n_bins"],
                "drift_call": cohort_summary["drift_call"],
                "delta_median_icc_first_to_last_bin": cohort_summary["delta_first_to_last_bin"][
                    "median_icc"
                ],
                "delta_robust_share_first_to_last_bin": cohort_summary["delta_first_to_last_bin"][
                    "robust_share"
                ],
                "delta_median_cov_first_to_last_bin": cohort_summary["delta_first_to_last_bin"][
                    "median_cov_percent"
                ],
                "tau_median_icc": trend_tests["median_icc_mann_kendall"]["tau"],
                "p_value_median_icc": trend_tests["median_icc_mann_kendall"]["p_value"],
                "theil_sen_slope_median_icc": trend_tests["median_icc_mann_kendall"][
                    "theil_sen_slope_per_bin"
                ],
                "tau_robust_share": trend_tests["robust_share_mann_kendall"]["tau"],
                "p_value_robust_share": trend_tests["robust_share_mann_kendall"]["p_value"],
                "theil_sen_slope_robust_share": trend_tests["robust_share_mann_kendall"][
                    "theil_sen_slope_per_bin"
                ],
                "tau_median_cov_percent": trend_tests["median_cov_mann_kendall"]["tau"],
                "p_value_median_cov_percent": trend_tests["median_cov_mann_kendall"]["p_value"],
                "theil_sen_slope_median_cov_percent": trend_tests["median_cov_mann_kendall"][
                    "theil_sen_slope_per_bin"
                ],
                "date_source_counts_json": json.dumps(
                    treatment_cov["date_source_counts"], sort_keys=True
                ),
                "scanner_generation_feasible": scanner_generation["feasible"],
                "scanner_generation_reason": scanner_generation["reason"],
                "n_scanner_models": scanner_generation["n_scanner_models"],
                "n_eligible_scanner_models": scanner_generation["n_eligible_scanner_models"],
                "dominant_scanner_model": scanner_generation["dominant_scanner_model"],
                "dominant_scanner_share": scanner_generation["dominant_scanner_share"],
                "scanner_model_counts_json": json.dumps(
                    scanner_generation["scanner_model_counts"], sort_keys=True
                ),
            }
        )

        for bin_summary in cohort_summary["bins"]:
            bin_rows.append(
                {
                    "cohort": cohort,
                    "bin_index": bin_summary["bin_index"],
                    "bin_label": bin_summary["bin_label"],
                    "date_start": bin_summary["date_start"],
                    "date_end": bin_summary["date_end"],
                    "n_subjects": bin_summary["n_subjects"],
                    "n_unique_patients": bin_summary["n_unique_patients"],
                    "n_feature_roi_pairs": bin_summary["n_feature_roi_pairs"],
                    "median_icc": bin_summary["median_icc"],
                    "median_cov_percent": bin_summary["median_cov_percent"],
                    "robust_share": bin_summary["robust_share"],
                    "acceptable_share": bin_summary["acceptable_share"],
                    "poor_share": bin_summary["poor_share"],
                }
            )
            for scanner in bin_summary["scanner_mix"]:
                scanner_rows.append(
                    {
                        "cohort": cohort,
                        "bin_index": bin_summary["bin_index"],
                        "bin_label": bin_summary["bin_label"],
                        "date_start": bin_summary["date_start"],
                        "date_end": bin_summary["date_end"],
                        "ct_model": scanner["ct_model"],
                        "ct_convolution_kernel": scanner["ct_convolution_kernel"],
                        "ct_slice_thickness": scanner["ct_slice_thickness"],
                        "n_subjects": scanner["n_subjects"],
                        "share": scanner["share"],
                    }
                )

    return (
        pd.DataFrame.from_records(cohort_rows),
        pd.DataFrame.from_records(bin_rows),
        pd.DataFrame.from_records(scanner_rows),
    )


def write_outputs(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_df, bin_df, scanner_df = flatten_outputs(payload)
    cohort_df.to_csv(output_dir / "44_temporal_drift_cohort_summary.csv", index=False)
    bin_df.to_csv(output_dir / "44_temporal_drift_bin_summary.csv", index=False)
    scanner_df.to_csv(output_dir / "44_temporal_drift_scanner_mix.csv", index=False)
    with (output_dir / "44_temporal_drift_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def analyze_cohort(
    cohort: str,
    requested_bins: int,
    min_subjects_per_bin: int,
    meta_all: pd.DataFrame,
    robustness_root: Path,
    raw_course_root: Path | None,
) -> dict[str, object]:
    if cohort == "Odbytnice":
        date_map = build_odbytnice_date_map(meta_all)
    elif cohort == "Prostata":
        date_map = build_prostata_date_map(meta_all)
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")

    if raw_course_root is not None:
        date_map = attach_raw_course_paths(cohort, date_map, raw_course_root)
        date_map = date_map.loc[date_map["raw_parquet_exists"]].copy()

    date_map = assign_treatment_bins(date_map, requested_bins, min_subjects_per_bin)
    cohort_path = robustness_root / f"{cohort}.parquet"
    feature_results_by_bin: list[pd.DataFrame] = []

    for bin_index in sorted(date_map["bin_index"].unique()):
        subject_map = date_map.loc[date_map["bin_index"] == bin_index].copy()
        t0 = time.time()
        bin_rows = load_bin_rows(
            cohort=cohort,
            cohort_path=cohort_path,
            subject_map=subject_map,
            raw_course_root=raw_course_root,
        )
        feature_results = compute_bin_feature_icc(bin_rows)
        feature_results["bin_index"] = int(bin_index)
        feature_results["bin_label"] = str(subject_map["bin_label"].iloc[0])
        feature_results_by_bin.append(feature_results)
        elapsed = time.time() - t0
        print(
            f"[{cohort}] {subject_map['bin_label'].iloc[0]}: "
            f"{len(subject_map)} subjects, {len(feature_results)} feature/ROI ICCs, "
            f"{elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return make_cohort_summary(
        cohort=cohort,
        subject_dates=date_map,
        feature_results_by_bin=feature_results_by_bin,
        min_subjects_per_bin=min_subjects_per_bin,
    )


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root
    metadata_all_path = args.metadata_all_path or (analysis_root / "data" / "metadata_all.parquet")
    robustness_root = args.robustness_root or (analysis_root / "data" / "robustness_by_cohort")
    raw_course_root = args.raw_course_root
    if raw_course_root is not None and not raw_course_root.exists():
        raw_course_root = None
    if raw_course_root is not None:
        meta_all = load_local_course_metadata(raw_course_root, args.cohorts)
        metadata_all_path_str = None
    else:
        meta_all = load_metadata_all(metadata_all_path)
        metadata_all_path_str = str(metadata_all_path)
    summaries = []
    for cohort in args.cohorts:
        summaries.append(
            analyze_cohort(
                cohort=cohort,
                requested_bins=args.requested_bins,
                min_subjects_per_bin=args.min_subjects_per_bin,
                meta_all=meta_all,
                robustness_root=robustness_root,
                raw_course_root=raw_course_root,
            )
        )
    payload = {
        "script": str(Path(__file__).resolve()),
        "analysis_root": str(analysis_root),
        "robustness_root": str(robustness_root),
        "metadata_all_path": metadata_all_path_str,
        "raw_course_root": None if raw_course_root is None else str(raw_course_root),
        "requested_bins": int(args.requested_bins),
        "min_subjects_per_bin": int(args.min_subjects_per_bin),
        "cohorts": summaries,
    }
    write_outputs(payload, args.output_dir)
    if args.pretty_json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(json.dumps(payload, sort_keys=False))


if __name__ == "__main__":
    main()
