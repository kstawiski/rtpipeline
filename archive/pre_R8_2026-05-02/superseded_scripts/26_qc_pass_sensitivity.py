#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 26: QC-pass sensitivity analysis on the rebuilt multicohort ICC table.

PURPOSE
-------
Rerun the manuscript-facing multicohort robustness summaries after excluding
patient-course structure instances flagged by segmentation QC as:

  - boundary-cropped
  - per-structure volume outlier (>3 SD from cohort+structure mean volume)
  - bilateral asymmetry (>50% left-right volume difference)

The sensitivity is executed at the raw `robustness_by_cohort/*.parquet` level
so ICC(3,1) and CoV are recomputed on the QC-pass subset instead of filtering
the already-aggregated `icc_results.parquet`.

INPUTS
------
  /home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/tables/segmentation_qc.csv
  /home/kgs24/rtpipeline_manuscript/analysis/data/robustness_by_cohort/*.parquet

OUTPUTS
-------
  data/icc_results_qc_pass.parquet
      Feature-level QC-pass ICC table aligned to the rebuilt all-data keys.

  tables/qc_pass_flag_summary.csv
      Cohort-level QC flag counts and pass rates.

  tables/qc_pass_sensitivity_cohort_summary.csv
      Cohort x family summaries for all-data vs QC-pass.

  tables/qc_pass_sensitivity_comparison.csv
      Region x family comparison table with delta ICC / delta robust-rate.

  tables/qc_pass_rank_stability.csv
      Spearman rank stability of family ordering under QC filtering.

  figures/figure_qc_pass_sensitivity.{png,pdf}
      Three-panel dumbbell summary of median ICC by family before vs after QC.

  logs/26_manifest_<JOBID>.json
      Provenance, QC filter counts, and output checksums.
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
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ANALYSIS_DIR = Path("/home/kgs24/rtpipeline_manuscript/analysis")
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"

ICC_RESULTS_PARQUET = DATA_DIR / "icc_results.parquet"
ROBUSTNESS_DIR = DATA_DIR / "robustness_by_cohort"
SEG_QC_CSV = TABLES_DIR / "segmentation_qc.csv"

OUT_QC_PARQUET = DATA_DIR / "icc_results_qc_pass.parquet"
OUT_FLAG_SUMMARY = TABLES_DIR / "qc_pass_flag_summary.csv"
OUT_COHORT_SUMMARY = TABLES_DIR / "qc_pass_sensitivity_cohort_summary.csv"
OUT_COMPARISON = TABLES_DIR / "qc_pass_sensitivity_comparison.csv"
OUT_RANK = TABLES_DIR / "qc_pass_rank_stability.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_qc_pass_sensitivity.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_qc_pass_sensitivity.pdf"

BODY_REGION_BY_COHORT = {
    "Prostata": "Pelvis",
    "Odbytnice": "Pelvis",
    "Immunodozymetria": "Thorax",
    "PlucaRCHT": "Thorax",
    "Hipokampy": "Brain",
    "GBM": "Brain",
    "LCTSC": "Thorax",
    "NSCLC_Interobserver": "Thorax",
    "NSCLC_Radiomics": "Thorax",
    "RIDER": "Thorax",
}
DEFAULT_COHORTS = list(BODY_REGION_BY_COHORT.keys())
FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
REGION_ORDER = ["Brain", "Pelvis", "Thorax"]
SUMMARY_LEVELS = ["cohort", "body_region", "overall"]


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
    logger = logging.getLogger("script26_qc_pass")
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


def parse_cohorts(raw_value: str) -> list[str]:
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def normalize_qc_structure(value: object) -> str:
    token = str(value).strip()
    if token.startswith("total--"):
        token = token.split("total--", 1)[1]
    while token.endswith("_cropped"):
        token = token[: -len("_cropped")]
    return token


def parse_left_right(value: str) -> tuple[str | None, str | None]:
    if value.endswith("_left"):
        return value[: -len("_left")], "left"
    if value.endswith("_right"):
        return value[: -len("_right")], "right"
    return None, None


def classify_icc_cov(icc: float, cov_pct: float) -> str:
    if not np.isfinite(icc) or not np.isfinite(cov_pct):
        return "Dropped"
    if icc >= 0.90 and cov_pct <= 10.0:
        return "Robust"
    if icc >= 0.75 and cov_pct <= 20.0:
        return "Acceptable"
    return "Poor"


def segmentation_source_priority(value: object) -> int:
    token = str(value).strip().lower()
    if "custom" in token:
        return 9
    if token.startswith("autorts") or "totalsegmentator" in token:
        return 0
    if token in {"", "nan", "none"}:
        return 5
    return 3


def resolve_duplicate_measurements(
    df: pd.DataFrame, cohort: str, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, int]]:
    key_cols = ["patient_id", "course_id", "roi_name", "feature_name", "perturbation_id"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return df, {"duplicate_rows_seen": 0, "duplicate_rows_dropped": 0}

    if "segmentation_source" not in df.columns:
        raise RuntimeError(
            f"{cohort}: duplicate raw rows detected but no segmentation_source column is available for resolution."
        )

    dup_rows_seen = int(dup_mask.sum())
    work = df.copy()
    work["segmentation_priority"] = work["segmentation_source"].map(segmentation_source_priority)
    work = work.sort_values(
        key_cols + ["segmentation_priority", "segmentation_source"],
        kind="stable",
    )
    before = len(work)
    work = work.drop_duplicates(subset=key_cols, keep="first").copy()
    work = work.drop(columns=["segmentation_priority"])
    if work.duplicated(subset=key_cols, keep=False).any():
        raise RuntimeError(
            f"{cohort}: duplicate raw rows remained after segmentation-source resolution."
        )

    dup_rows_dropped = before - len(work)
    logger.info(
        "%s: resolved %d duplicate raw rows by prioritizing AI segmentation sources over Custom.",
        cohort,
        dup_rows_dropped,
    )
    return work, {
        "duplicate_rows_seen": dup_rows_seen,
        "duplicate_rows_dropped": dup_rows_dropped,
    }


def build_patient_stats_dict(group_df: pd.DataFrame) -> dict[str, object]:
    raters = sorted(group_df["perturbation_id"].astype(str).unique().tolist())
    if len(raters) < 2:
        return {"k_raters": len(raters), "patient_stats": {}}

    dup_mask = group_df.duplicated(
        subset=["patient_id", "course_id", "perturbation_id"], keep=False
    )
    if dup_mask.any():
        raise RuntimeError(
            "Duplicate patient/course/perturbation rows detected in raw robustness data. "
            "This would silently corrupt ICC recomputation."
        )

    pivot = group_df.pivot(
        index=["patient_id", "course_id"],
        columns="perturbation_id",
        values="value",
    )
    pivot = pivot.reindex(columns=raters)
    pivot = pivot.dropna(axis=0, how="any")

    patient_stats: dict[str, dict[str, object]] = {}
    for patient_id, patient_block in pivot.groupby(level=0, sort=False):
        matrix = patient_block.to_numpy(dtype=float)
        if matrix.size == 0:
            continue
        row_sums = matrix.sum(axis=1)
        patient_stats[str(patient_id)] = {
            "n_subjects": int(matrix.shape[0]),
            "total_sum": float(matrix.sum()),
            "total_sum_sq": float(np.square(matrix).sum()),
            "row_sum_sq": float(np.square(row_sums).sum()),
            "col_sums": matrix.sum(axis=0).astype(float),
        }
    return {"k_raters": len(raters), "patient_stats": patient_stats}


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


def load_baseline_points(
    requested_cohorts: list[str], logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not ICC_RESULTS_PARQUET.exists():
        raise FileNotFoundError(f"Missing baseline ICC parquet: {ICC_RESULTS_PARQUET}")

    df = pd.read_parquet(ICC_RESULTS_PARQUET)
    required = {"feature_name", "roi_name", "cohort", "icc", "classification"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Baseline ICC parquet missing required columns: {sorted(missing)}")

    cov_col = "cov_percent" if "cov_percent" in df.columns else "cov_pct"
    family_col = "feature_family" if "feature_family" in df.columns else "family"
    image_type_col = "image_type" if "image_type" in df.columns else None

    out = df.copy()
    out["cohort"] = out["cohort"].astype(str)
    out = out.loc[out["cohort"].isin(requested_cohorts)].copy()
    out["roi_name"] = out["roi_name"].astype(str)
    out["feature_name"] = out["feature_name"].astype(str)
    out["body_region"] = out.get("body_region", out["cohort"].map(BODY_REGION_BY_COHORT))
    out["icc"] = pd.to_numeric(out["icc"], errors="coerce")
    out["cov_percent"] = pd.to_numeric(out[cov_col], errors="coerce")
    out["classification"] = out["classification"].astype(str)
    out["feature_family"] = out[family_col].astype(str).str.lower()
    out["image_type"] = (
        out[image_type_col].astype(str) if image_type_col is not None else "unknown"
    )
    out["n_subjects"] = pd.to_numeric(out.get("n_subjects"), errors="coerce")
    out["n_raters"] = pd.to_numeric(out.get("n_raters"), errors="coerce")
    out = out.loc[out["feature_family"].isin(FAMILY_ORDER)].copy()

    meta = {
        "n_rows": int(len(out)),
        "n_features": int(out["feature_name"].nunique()),
        "n_rois": int(out["roi_name"].nunique()),
        "n_cohorts": int(out["cohort"].nunique()),
    }
    logger.info(
        "Loaded baseline ICC table: %d rows, %d features, %d ROIs, %d cohorts.",
        meta["n_rows"],
        meta["n_features"],
        meta["n_rois"],
        meta["n_cohorts"],
    )
    keep_cols = [
        "cohort",
        "body_region",
        "roi_name",
        "feature_name",
        "feature_family",
        "image_type",
        "icc",
        "cov_percent",
        "classification",
        "n_subjects",
        "n_raters",
    ]
    return out[keep_cols].copy(), meta


def load_segmentation_qc(
    allowed_cohorts: list[str],
    allowed_rois: set[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not SEG_QC_CSV.exists():
        raise FileNotFoundError(f"Missing segmentation QC CSV: {SEG_QC_CSV}")

    df = pd.read_csv(SEG_QC_CSV)
    required = {
        "patient_id",
        "course_id",
        "cohort",
        "body_region",
        "structure",
        "volume_cc",
        "is_empty",
        "is_cropped",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Segmentation QC CSV missing required columns: {sorted(missing)}")

    work = df.copy()
    work["cohort"] = work["cohort"].astype(str)
    work = work.loc[work["cohort"].isin(allowed_cohorts)].copy()
    work["patient_id"] = work["patient_id"].astype(str)
    work["course_id"] = work["course_id"].astype(str)
    work["roi_name"] = work["structure"].map(normalize_qc_structure)
    work = work.loc[work["roi_name"].isin(allowed_rois)].copy()
    work["volume_cc"] = pd.to_numeric(work["volume_cc"], errors="coerce")
    work["is_empty"] = work["is_empty"].astype(bool)
    work["is_cropped"] = work["is_cropped"].astype(bool)

    dedup = (
        work.groupby(
            ["cohort", "body_region", "patient_id", "course_id", "roi_name"],
            dropna=False,
            as_index=False,
        )
        .agg(
            volume_cc=("volume_cc", "max"),
            is_empty=("is_empty", "all"),
            is_cropped=("is_cropped", "any"),
        )
        .copy()
    )

    non_empty = dedup.loc[~dedup["is_empty"]].copy()
    vol_stats = (
        non_empty.groupby(["cohort", "roi_name"], as_index=False)["volume_cc"]
        .agg(volume_mean="mean", volume_sd="std")
        .copy()
    )
    dedup = dedup.merge(vol_stats, on=["cohort", "roi_name"], how="left")
    dedup["volume_outlier_flag"] = False
    valid_vol = (
        (~dedup["is_empty"])
        & dedup["volume_sd"].notna()
        & (dedup["volume_sd"] > 0)
        & dedup["volume_cc"].notna()
    )
    dedup.loc[valid_vol, "volume_outlier_flag"] = (
        (dedup.loc[valid_vol, "volume_cc"] - dedup.loc[valid_vol, "volume_mean"]).abs()
        > (3.0 * dedup.loc[valid_vol, "volume_sd"])
    )

    dedup["pair_base"], dedup["pair_side"] = zip(
        *dedup["roi_name"].map(parse_left_right), strict=False
    )
    pairs = dedup.loc[
        dedup["pair_base"].notna() & (~dedup["is_empty"]) & dedup["volume_cc"].notna(),
        ["cohort", "patient_id", "course_id", "pair_base", "pair_side", "volume_cc"],
    ].copy()
    pair_pivot = (
        pairs.pivot_table(
            index=["cohort", "patient_id", "course_id", "pair_base"],
            columns="pair_side",
            values="volume_cc",
            aggfunc="max",
        )
        .reset_index()
        .copy()
    )
    pair_pivot["bilateral_asymmetry_flag"] = False
    both_sides = pair_pivot["left"].notna() & pair_pivot["right"].notna()
    left_right_max = pair_pivot.loc[both_sides, ["left", "right"]].max(axis=1)
    pair_pivot.loc[both_sides, "bilateral_asymmetry_flag"] = (
        (pair_pivot.loc[both_sides, "left"] - pair_pivot.loc[both_sides, "right"]).abs()
        / left_right_max.replace(0, np.nan)
        > 0.50
    )

    bilateral_rows: list[pd.DataFrame] = []
    flagged_pairs = pair_pivot.loc[pair_pivot["bilateral_asymmetry_flag"]].copy()
    for side in ["left", "right"]:
        tmp = flagged_pairs[
            ["cohort", "patient_id", "course_id", "pair_base", "bilateral_asymmetry_flag"]
        ].copy()
        tmp["roi_name"] = tmp["pair_base"] + f"_{side}"
        bilateral_rows.append(tmp.drop(columns=["pair_base"]))
    bilateral_flags = (
        pd.concat(bilateral_rows, ignore_index=True)
        if bilateral_rows
        else pd.DataFrame(
            columns=[
                "cohort",
                "patient_id",
                "course_id",
                "roi_name",
                "bilateral_asymmetry_flag",
            ]
        )
    )

    dedup = dedup.merge(
        bilateral_flags,
        on=["cohort", "patient_id", "course_id", "roi_name"],
        how="left",
    )
    dedup["bilateral_asymmetry_flag"] = dedup["bilateral_asymmetry_flag"].fillna(False)
    dedup["qc_fail"] = (
        dedup["is_cropped"] | dedup["volume_outlier_flag"] | dedup["bilateral_asymmetry_flag"]
    )
    dedup["qc_pass"] = ~dedup["qc_fail"]

    summary = (
        dedup.groupby(["cohort", "body_region"], as_index=False)
        .agg(
            n_structure_instances=("roi_name", "size"),
            n_nonempty=("is_empty", lambda x: int((~x).sum())),
            n_cropped=("is_cropped", "sum"),
            n_volume_outlier=("volume_outlier_flag", "sum"),
            n_bilateral_asymmetry=("bilateral_asymmetry_flag", "sum"),
            n_qc_fail=("qc_fail", "sum"),
            n_qc_pass=("qc_pass", "sum"),
        )
        .copy()
    )
    summary["qc_pass_rate_pct"] = (
        100.0 * summary["n_qc_pass"] / summary["n_structure_instances"].replace(0, np.nan)
    )

    overall = pd.DataFrame(
        [
            {
                "cohort": "Overall",
                "body_region": "Overall",
                "n_structure_instances": int(summary["n_structure_instances"].sum()),
                "n_nonempty": int(summary["n_nonempty"].sum()),
                "n_cropped": int(summary["n_cropped"].sum()),
                "n_volume_outlier": int(summary["n_volume_outlier"].sum()),
                "n_bilateral_asymmetry": int(summary["n_bilateral_asymmetry"].sum()),
                "n_qc_fail": int(summary["n_qc_fail"].sum()),
                "n_qc_pass": int(summary["n_qc_pass"].sum()),
                "qc_pass_rate_pct": float(
                    100.0
                    * summary["n_qc_pass"].sum()
                    / max(summary["n_structure_instances"].sum(), 1)
                ),
            }
        ]
    )
    summary = pd.concat([summary, overall], ignore_index=True)

    meta = {
        "n_rows_raw": int(len(df)),
        "n_rows_filtered": int(len(work)),
        "n_rows_dedup": int(len(dedup)),
        "n_allowed_rois": int(len(allowed_rois)),
        "volume_outlier_count": int(dedup["volume_outlier_flag"].sum()),
        "bilateral_asymmetry_count": int(dedup["bilateral_asymmetry_flag"].sum()),
        "cropped_count": int(dedup["is_cropped"].sum()),
        "qc_fail_count": int(dedup["qc_fail"].sum()),
        "qc_pass_count": int(dedup["qc_pass"].sum()),
    }
    logger.info(
        "Loaded segmentation QC subset: %d deduplicated structure instances, %d cropped, %d volume outliers, %d bilateral asymmetry flags.",
        meta["n_rows_dedup"],
        meta["cropped_count"],
        meta["volume_outlier_count"],
        meta["bilateral_asymmetry_count"],
    )
    return dedup, {"summary": summary, **meta}


def load_raw_batch(
    cohort: str,
    feature_names: list[str],
    roi_names: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    cohort_path = ROBUSTNESS_DIR / f"{cohort}.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort parquet: {cohort_path}")

    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to stream robustness parquet files.") from exc

    columns = [
        "patient_id",
        "course_id",
        "roi_name",
        "perturbation_id",
        "feature_name",
        "value",
        "segmentation_source",
    ]
    dataset = ds.dataset(str(cohort_path), format="parquet")
    filter_expr = ds.field("feature_name").isin(feature_names)
    if roi_names:
        filter_expr = filter_expr & ds.field("roi_name").isin(roi_names)
    table = dataset.scanner(columns=columns, filter=filter_expr, use_threads=True).to_table()
    df = table.to_pandas()
    logger.info(
        "Loaded %s batch (%d features, %d rois): %d rows.",
        cohort,
        len(feature_names),
        len(roi_names),
        len(df),
    )
    return df


def recompute_qc_pass_points(
    baseline_df: pd.DataFrame,
    qc_lookup: pd.DataFrame,
    requested_cohorts: list[str],
    batch_size: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    qc_keep = qc_lookup[
        [
            "cohort",
            "patient_id",
            "course_id",
            "roi_name",
            "qc_pass",
        ]
    ].copy()
    qc_meta_rows: list[dict[str, object]] = []
    out_rows: list[dict[str, object]] = []

    for cohort in requested_cohorts:
        cohort_base = baseline_df.loc[baseline_df["cohort"] == cohort].copy()
        if cohort_base.empty:
            continue
        cohort_qc = qc_keep.loc[qc_keep["cohort"] == cohort].copy()
        feature_names = sorted(cohort_base["feature_name"].unique().tolist())
        roi_names = sorted(cohort_base["roi_name"].unique().tolist())
        duplicate_rows_seen = 0
        duplicate_rows_dropped = 0
        missing_qc_rows = 0
        loaded_rows = 0
        kept_rows = 0

        for start in range(0, len(feature_names), batch_size):
            feature_chunk = feature_names[start : start + batch_size]
            batch_df = load_raw_batch(cohort, feature_chunk, roi_names, logger)
            if batch_df.empty:
                continue
            loaded_rows += int(len(batch_df))
            batch_df = batch_df.dropna(
                subset=["patient_id", "course_id", "roi_name", "perturbation_id", "feature_name", "value"]
            ).copy()
            batch_df["patient_id"] = batch_df["patient_id"].astype(str)
            batch_df["course_id"] = batch_df["course_id"].astype(str)
            batch_df["cohort"] = cohort
            batch_df["roi_name"] = batch_df["roi_name"].astype(str)
            batch_df["perturbation_id"] = batch_df["perturbation_id"].astype(str)
            batch_df["feature_name"] = batch_df["feature_name"].astype(str)
            batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
            batch_df = batch_df.dropna(subset=["value"])
            batch_df, dup_meta = resolve_duplicate_measurements(batch_df, cohort=cohort, logger=logger)
            duplicate_rows_seen += int(dup_meta["duplicate_rows_seen"])
            duplicate_rows_dropped += int(dup_meta["duplicate_rows_dropped"])

            merged = batch_df.merge(
                cohort_qc,
                on=["cohort", "patient_id", "course_id", "roi_name"],
                how="left",
            )
            missing_qc_rows += int(merged["qc_pass"].isna().sum())
            merged = merged.loc[merged["qc_pass"].fillna(False)].copy()
            kept_rows += int(len(merged))
            if merged.empty:
                continue

            for (roi_name, feature_name), sub in merged.groupby(["roi_name", "feature_name"], sort=False):
                stats_info = build_patient_stats_dict(sub)
                k_raters = int(stats_info["k_raters"])
                patient_stats = stats_info["patient_stats"]
                if k_raters < 2 or not patient_stats:
                    continue
                n_subjects = float(sum(item["n_subjects"] for item in patient_stats.values()))
                total_sum = float(sum(item["total_sum"] for item in patient_stats.values()))
                total_sum_sq = float(sum(item["total_sum_sq"] for item in patient_stats.values()))
                row_sum_sq = float(sum(item["row_sum_sq"] for item in patient_stats.values()))
                col_sums = np.sum(
                    [np.asarray(item["col_sums"], dtype=float) for item in patient_stats.values()],
                    axis=0,
                )
                icc = compute_icc31_from_sufficient_stats(
                    n_subjects=n_subjects,
                    total_sum=total_sum,
                    total_sum_sq=total_sum_sq,
                    row_sum_sq=row_sum_sq,
                    col_sums=np.asarray(col_sums, dtype=float),
                )
                cov_pct = compute_cov_from_sufficient_stats(
                    total_sum=total_sum,
                    total_sum_sq=total_sum_sq,
                    n_values=n_subjects * float(k_raters),
                )
                out_rows.append(
                    {
                        "cohort": cohort,
                        "roi_name": roi_name,
                        "feature_name": feature_name,
                        "icc": icc,
                        "cov_percent": cov_pct,
                        "classification": classify_icc_cov(icc, cov_pct),
                        "n_subjects": n_subjects,
                        "n_raters": k_raters,
                    }
                )

        qc_meta_rows.append(
            {
                "cohort": cohort,
                "loaded_rows": int(loaded_rows),
                "kept_rows": int(kept_rows),
                "duplicate_rows_seen": int(duplicate_rows_seen),
                "duplicate_rows_dropped": int(duplicate_rows_dropped),
                "missing_qc_rows": int(missing_qc_rows),
            }
        )
        logger.info(
            "Recomputed QC-pass points for %s: loaded=%d kept=%d missing_qc=%d duplicate_dropped=%d.",
            cohort,
            loaded_rows,
            kept_rows,
            missing_qc_rows,
            duplicate_rows_dropped,
        )

    qc_points = pd.DataFrame(out_rows)
    qc_points["cohort"] = qc_points["cohort"].astype(str)
    qc_points["roi_name"] = qc_points["roi_name"].astype(str)
    qc_points["feature_name"] = qc_points["feature_name"].astype(str)

    merged = baseline_df[
        [
            "cohort",
            "body_region",
            "roi_name",
            "feature_name",
            "feature_family",
            "image_type",
        ]
    ].merge(
        qc_points,
        on=["cohort", "roi_name", "feature_name"],
        how="left",
    )
    merged["body_region"] = merged["body_region"].fillna(merged["cohort"].map(BODY_REGION_BY_COHORT))
    merged["classification"] = merged["classification"].fillna("Dropped")
    return merged, qc_meta_rows


def summarise_points(points: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    df = points.copy()
    df["analysis_set"] = analysis_set
    df["is_robust"] = df["classification"].eq("Robust")
    df["is_acceptable"] = df["classification"].eq("Acceptable")
    df["is_poor"] = df["classification"].eq("Poor")
    df["is_dropped"] = df["classification"].eq("Dropped")
    df["has_metric"] = df["icc"].notna() & df["cov_percent"].notna()

    rows: list[dict[str, object]] = []
    for (cohort, body_region, feature_family), sub in df.groupby(
        ["cohort", "body_region", "feature_family"], sort=False
    ):
        rows.append(
            {
                "summary_level": "cohort",
                "analysis_set": analysis_set,
                "cohort": cohort,
                "body_region": body_region,
                "feature_family": feature_family,
                "row_count_total": int(len(sub)),
                "row_count_nonmissing": int(sub["has_metric"].sum()),
                "roi_count": int(sub["roi_name"].nunique()),
                "feature_count": int(sub["feature_name"].nunique()),
                "median_icc": float(sub["icc"].median(skipna=True)),
                "median_cov_percent": float(sub["cov_percent"].median(skipna=True)),
                "robust_rate_pct": float(100.0 * sub["is_robust"].mean()),
                "acceptable_rate_pct": float(100.0 * sub["is_acceptable"].mean()),
                "poor_rate_pct": float(100.0 * sub["is_poor"].mean()),
                "dropped_rate_pct": float(100.0 * sub["is_dropped"].mean()),
            }
        )
    cohort_summary = pd.DataFrame(rows)

    agg_rows: list[dict[str, object]] = []
    for (body_region, feature_family), sub in df.groupby(["body_region", "feature_family"], sort=False):
        agg_rows.append(
            {
                "summary_level": "body_region",
                "analysis_set": analysis_set,
                "cohort": "",
                "body_region": body_region,
                "feature_family": feature_family,
                "row_count_total": int(len(sub)),
                "row_count_nonmissing": int(sub["has_metric"].sum()),
                "roi_count": int(sub["roi_name"].nunique()),
                "feature_count": int(sub["feature_name"].nunique()),
                "median_icc": float(sub["icc"].median(skipna=True)),
                "median_cov_percent": float(sub["cov_percent"].median(skipna=True)),
                "robust_rate_pct": float(100.0 * sub["is_robust"].mean()),
                "acceptable_rate_pct": float(100.0 * sub["is_acceptable"].mean()),
                "poor_rate_pct": float(100.0 * sub["is_poor"].mean()),
                "dropped_rate_pct": float(100.0 * sub["is_dropped"].mean()),
            }
        )
    for feature_family, sub in df.groupby("feature_family", sort=False):
        agg_rows.append(
            {
                "summary_level": "overall",
                "analysis_set": analysis_set,
                "cohort": "",
                "body_region": "Overall",
                "feature_family": feature_family,
                "row_count_total": int(len(sub)),
                "row_count_nonmissing": int(sub["has_metric"].sum()),
                "roi_count": int(sub["roi_name"].nunique()),
                "feature_count": int(sub["feature_name"].nunique()),
                "median_icc": float(sub["icc"].median(skipna=True)),
                "median_cov_percent": float(sub["cov_percent"].median(skipna=True)),
                "robust_rate_pct": float(100.0 * sub["is_robust"].mean()),
                "acceptable_rate_pct": float(100.0 * sub["is_acceptable"].mean()),
                "poor_rate_pct": float(100.0 * sub["is_poor"].mean()),
                "dropped_rate_pct": float(100.0 * sub["is_dropped"].mean()),
            }
        )
    return pd.concat([cohort_summary, pd.DataFrame(agg_rows)], ignore_index=True)


def build_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    keep = summary_df.loc[summary_df["summary_level"].isin(["body_region", "overall"])].copy()
    all_df = keep.loc[keep["analysis_set"] == "all_data"].copy()
    qc_df = keep.loc[keep["analysis_set"] == "qc_pass"].copy()
    join_cols = ["summary_level", "body_region", "feature_family"]
    merged = all_df.merge(qc_df, on=join_cols, how="outer", suffixes=("_all_data", "_qc_pass"))
    for metric in [
        "median_icc",
        "median_cov_percent",
        "robust_rate_pct",
        "acceptable_rate_pct",
        "poor_rate_pct",
        "dropped_rate_pct",
        "row_count_nonmissing",
        "row_count_total",
    ]:
        merged[f"delta_{metric}"] = merged[f"{metric}_qc_pass"] - merged[f"{metric}_all_data"]
    merged["row_retention_pct"] = (
        100.0
        * merged["row_count_nonmissing_qc_pass"]
        / merged["row_count_nonmissing_all_data"].replace(0, np.nan)
    )
    return merged


def build_rank_stability(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keep = summary_df.loc[summary_df["summary_level"] == "body_region"].copy()
    for body_region in REGION_ORDER:
        base = keep.loc[
            (keep["body_region"] == body_region) & (keep["analysis_set"] == "all_data"),
            ["feature_family", "median_icc", "robust_rate_pct"],
        ].copy()
        qc = keep.loc[
            (keep["body_region"] == body_region) & (keep["analysis_set"] == "qc_pass"),
            ["feature_family", "median_icc", "robust_rate_pct"],
        ].copy()
        merged = base.merge(qc, on="feature_family", suffixes=("_all_data", "_qc_pass"))
        if len(merged) >= 2:
            rho_icc = float(
                sp_stats.spearmanr(merged["median_icc_all_data"], merged["median_icc_qc_pass"]).statistic
            )
            rho_robust = float(
                sp_stats.spearmanr(
                    merged["robust_rate_pct_all_data"],
                    merged["robust_rate_pct_qc_pass"],
                ).statistic
            )
        else:
            rho_icc = float("nan")
            rho_robust = float("nan")
        rows.append(
            {
                "body_region": body_region,
                "spearman_rho_median_icc": rho_icc,
                "spearman_rho_robust_rate_pct": rho_robust,
                "n_families_compared": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def render_figure(summary_df: pd.DataFrame) -> None:
    keep = summary_df.loc[summary_df["summary_level"] == "body_region"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.0), sharey=True)
    family_positions = {fam: idx for idx, fam in enumerate(FAMILY_ORDER)}

    for ax, body_region in zip(axes, REGION_ORDER, strict=False):
        base = keep.loc[
            (keep["body_region"] == body_region) & (keep["analysis_set"] == "all_data")
        ].copy()
        qc = keep.loc[
            (keep["body_region"] == body_region) & (keep["analysis_set"] == "qc_pass")
        ].copy()
        merged = base.merge(qc, on="feature_family", suffixes=("_all", "_qc"))
        merged["y"] = merged["feature_family"].map(family_positions)
        merged = merged.sort_values("y")
        for _, row in merged.iterrows():
            ax.plot(
                [row["median_icc_all"], row["median_icc_qc"]],
                [row["y"], row["y"]],
                color="#b0b0b0",
                linewidth=1.2,
                zorder=1,
            )
        ax.scatter(
            merged["median_icc_all"],
            merged["y"],
            color="#3b82f6",
            label="All data",
            s=45,
            zorder=2,
        )
        ax.scatter(
            merged["median_icc_qc"],
            merged["y"],
            color="#d97706",
            label="QC-pass",
            s=45,
            zorder=3,
        )
        ax.axvline(0.75, color="#9ca3af", linestyle="--", linewidth=1.0)
        ax.axvline(0.90, color="#9ca3af", linestyle=":", linewidth=1.0)
        ax.set_title(body_region)
        ax.set_xlabel("Median ICC")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)

    axes[0].set_yticks(list(family_positions.values()))
    axes[0].set_yticklabels(FAMILY_ORDER)
    axes[0].set_ylabel("Feature family")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("QC-pass sensitivity: family median ICC before vs after QC filtering", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 26 QC-pass sensitivity analysis")
    parser.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--job-id", default=os.environ.get("JOB_ID", "manual"))
    args = parser.parse_args()

    for directory in [DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.job_id)
    requested_cohorts = parse_cohorts(args.cohorts)
    logger.info("Starting step 26 QC-pass sensitivity for cohorts: %s", requested_cohorts)

    try:
        baseline_df, baseline_meta = load_baseline_points(requested_cohorts, logger)
        allowed_rois = set(baseline_df["roi_name"].unique().tolist())
        qc_lookup, qc_meta = load_segmentation_qc(requested_cohorts, allowed_rois, logger)

        qc_flag_summary = qc_meta["summary"].copy()
        qc_flag_summary.to_csv(OUT_FLAG_SUMMARY, index=False)

        qc_pass_df, raw_meta_rows = recompute_qc_pass_points(
            baseline_df=baseline_df,
            qc_lookup=qc_lookup,
            requested_cohorts=requested_cohorts,
            batch_size=args.batch_size,
            logger=logger,
        )

        all_points = baseline_df.copy()
        all_points["classification"] = all_points["classification"].fillna("Dropped")
        qc_pass_df["classification"] = qc_pass_df["classification"].fillna("Dropped")

        all_summary = summarise_points(all_points, analysis_set="all_data")
        qc_summary = summarise_points(qc_pass_df, analysis_set="qc_pass")
        cohort_summary = pd.concat([all_summary, qc_summary], ignore_index=True)
        comparison = build_comparison_table(cohort_summary)
        rank_stability = build_rank_stability(cohort_summary)

        qc_pass_df.to_parquet(OUT_QC_PARQUET, index=False)
        cohort_summary.to_csv(OUT_COHORT_SUMMARY, index=False)
        comparison.to_csv(OUT_COMPARISON, index=False)
        rank_stability.to_csv(OUT_RANK, index=False)
        render_figure(cohort_summary)

        manifest = {
            "completed": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": str(args.job_id),
            "hostname": socket.gethostname(),
            "git_sha": resolve_git_sha(),
            "cohorts": requested_cohorts,
            "batch_size": int(args.batch_size),
            "baseline_meta": baseline_meta,
            "qc_meta": {k: v for k, v in qc_meta.items() if k != "summary"},
            "raw_recompute_meta": raw_meta_rows,
            "notes": [
                "QC-pass excludes cropped OR volume-outlier OR bilateral-asymmetric structure instances.",
                "Volume outlier is defined within cohort+roi_name on non-empty segmentation_qc rows using >3 SD from the mean.",
                "Bilateral asymmetry is defined on left/right non-empty pairs as abs(left-right) / max(left,right) > 0.50.",
                "All-data baseline is the rebuilt icc_results.parquet; QC-pass metrics are recomputed from robustness_by_cohort raw values.",
            ],
            "outputs": {
                "qc_pass_parquet": str(OUT_QC_PARQUET),
                "qc_flag_summary_csv": str(OUT_FLAG_SUMMARY),
                "cohort_summary_csv": str(OUT_COHORT_SUMMARY),
                "comparison_csv": str(OUT_COMPARISON),
                "rank_stability_csv": str(OUT_RANK),
                "figure_png": str(OUT_FIG_PNG),
                "figure_pdf": str(OUT_FIG_PDF),
            },
            "output_checksums": {
                str(path): sha256_file(path)
                for path in [
                    OUT_QC_PARQUET,
                    OUT_FLAG_SUMMARY,
                    OUT_COHORT_SUMMARY,
                    OUT_COMPARISON,
                    OUT_RANK,
                    OUT_FIG_PNG,
                    OUT_FIG_PDF,
                ]
            },
        }
        manifest_path = LOGS_DIR / f"26_manifest_{args.job_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "Step 26 complete: wrote %s, %s, %s, %s.",
            OUT_QC_PARQUET,
            OUT_FLAG_SUMMARY,
            OUT_COMPARISON,
            manifest_path,
        )
        return 0

    except Exception as exc:
        error_manifest = {
            "completed": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": str(args.job_id),
            "hostname": socket.gethostname(),
            "git_sha": resolve_git_sha(),
            "cohorts": requested_cohorts,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        manifest_path = LOGS_DIR / f"26_manifest_{args.job_id}.json"
        manifest_path.write_text(json.dumps(error_manifest, indent=2) + "\n", encoding="utf-8")
        logger.exception("Step 26 failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
