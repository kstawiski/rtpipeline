#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 41 FIXED: prognostic retention with fail-closed scope cuts.

This replacement removes the two scientific blockers identified in the
2026-04-18 review:

1. No sacrum-surrogate Odbytnice production analysis.
2. No fabricated 365-day default censoring for Prostata OS.

Production scope in this fixed script is intentionally narrow:
  - Prostata toxicity (G2+ lymphopenia) only.

Dropped legs are recorded in the manifest:
  - Prostata OS: dropped because no observed last-follow-up field was available
    in the audited accessible sources at implementation time.
  - Odbytnice DFS / pCR: dropped because cohort-complete RTSTRUCT/GTV
    radiomics were not available in an analysis-ready merged export.

The statistical layer is also repaired:
  - feature prescreening happens inside each outer CV training fold only;
  - the selected feature count is capped by an events-per-feature rule using the
    training-fold event count;
  - pooled random-effects log-OR summaries are emitted by robustness class and
    by robustness-class x family;
  - any production-leg blocker triggers a nonzero exit unless
    --allow-leg-failure is explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
CLASS_ORDER = ["Robust", "Acceptable", "Poor"]
PRODUCTION_LEGS = ["prostata_toxicity_g2plus"]
PROSTATE_TARGET_ROI = "prostate"
MAX_LOG_EXP = math.log(sys.float_info.max)

ICC_RESULTS_CANDIDATES = [
    Path("/home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet"),
    Path("/umed-projekty/rtpipeline/manuscript/analysis/data/icc_results.parquet"),
]

PROSTATE_CLINICAL_CANDIDATES = [
    Path("/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/analytic_cohort.parquet"),
    Path("/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.parquet"),
    Path("/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.csv"),
    Path("/projekty/Prostata/LK/Data/bdo.tsv"),
]

PROSTATE_RADIOMICS_CANDIDATES = [
    Path("/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/radiomics_ts_all.parquet"),
    Path("/projekty/Prostata/Data_raw/DICOM_rtpipeline/Output/_ANALYSIS_READY/radiomics_ct_filtered.xlsx"),
]

FOLLOW_UP_COLUMN_CANDIDATES = [
    "last_fu_date",
    "last_follow_up_date",
    "last_followup_date",
    "follow_up_end_date",
    "fu_end_date",
    "censoring_date",
    "last_contact_date",
    "last_seen_date",
    "os_days",
    "time_to_event_days",
    "time_to_death_or_last_fu_days",
]

META_COLUMNS = {
    "patient_id",
    "course_id",
    "subject",
    "roi_name",
    "roi",
    "structure",
    "structure_name",
    "__roi_name__",
    "modality",
    "image_type",
    "segmentation_source",
    "output_dir",
    "course_dir",
    "structure_cropped",
    "roi_original_name",
}

RADIOMICS_FAMILY_TOKENS = tuple(f"_{family}_" for family in FAMILY_ORDER)


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
    logger = logging.getLogger("script41_fixed")
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


def resolve_analysis_dir() -> Path:
    candidates = [
        Path("/home/kgs24/rtpipeline_manuscript/analysis"),
        Path("/umed-projekty/rtpipeline/manuscript/analysis"),
        Path(__file__).resolve().parents[1] / "analysis",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


ANALYSIS_DIR = resolve_analysis_dir()
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"

OUT_FEATURES = DATA_DIR / "prognostic_retention_feature_effects_fixed.parquet"
OUT_CLASS_SUMMARY = TABLES_DIR / "prognostic_retention_class_summary_fixed.csv"
OUT_CLASS_FAMILY_SUMMARY = (
    TABLES_DIR / "prognostic_retention_class_family_summary_fixed.csv"
)
OUT_KS = TABLES_DIR / "prognostic_retention_ks_fixed.csv"
OUT_PANEL = TABLES_DIR / "prognostic_retention_panel_cv_fixed.csv"
OUT_PANEL_FOLDS = TABLES_DIR / "prognostic_retention_panel_cv_folds_fixed.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_prognostic_retention_fixed.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_prognostic_retention_fixed.pdf"


def ensure_output_dirs() -> None:
    for path in (DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


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


def safe_exp(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if value >= MAX_LOG_EXP:
        return float("inf")
    if value <= -MAX_LOG_EXP:
        return 0.0
    return float(math.exp(value))


def normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def normalize_class_label(value: object) -> str | None:
    token = normalize_token(value)
    mapping = {
        "robust": "Robust",
        "acceptable": "Acceptable",
        "poor": "Poor",
    }
    return mapping.get(token)


def infer_feature_family(feature_name: object) -> str:
    text = str(feature_name).lower()
    for family in FAMILY_ORDER:
        if f"_{family}_" in text:
            return family
        if text.startswith(family + "_"):
            return family
    return "other"


def choose_existing_path(candidates: list[Path], logger: logging.Logger) -> Path:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")
    if len(existing) == 1:
        return existing[0]
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    logger.info("Resolved latest existing path: %s", newest)
    return newest


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table suffix: {path.suffix}")


def resolve_latest_icc(logger: logging.Logger) -> tuple[Path, pd.DataFrame, dict[str, object]]:
    existing = [path for path in ICC_RESULTS_CANDIDATES if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "Missing icc_results.parquet in all candidate analysis roots."
        )
    chosen = max(existing, key=lambda path: path.stat().st_mtime)
    logger.info("Using ICC table: %s", chosen)
    df = pd.read_parquet(chosen)
    meta = {
        "icc_source_path": str(chosen),
        "icc_source_mtime_epoch": chosen.stat().st_mtime,
        "icc_source_sha256": sha256_file(chosen),
        "icc_candidate_paths": [str(path) for path in existing],
    }
    return chosen, df, meta


def standardize_icc_table(
    icc_df: pd.DataFrame, cohort: str, roi: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = icc_df.copy()
    out.columns = [str(col) for col in out.columns]

    if "feature_name" not in out.columns:
        raise RuntimeError("ICC table is missing feature_name.")
    if "cohort" not in out.columns:
        raise RuntimeError("ICC table is missing cohort.")

    roi_col = None
    for candidate in ("roi_name", "roi", "structure", "structure_name", "__roi_name__"):
        if candidate in out.columns:
            roi_col = candidate
            break
    if roi_col is None:
        raise RuntimeError("ICC table is missing an ROI identifier column.")

    if "feature_family" in out.columns:
        out["feature_family"] = out["feature_family"].astype(str).str.lower()
    elif "family" in out.columns:
        out["feature_family"] = out["family"].astype(str).str.lower()
    else:
        out["feature_family"] = out["feature_name"].map(infer_feature_family)

    class_col = None
    for candidate in (
        "icc_class",
        "robustness_label",
        "robustness_class",
        "classification_icc31",
        "classification",
    ):
        if candidate in out.columns:
            class_col = candidate
            break

    icc_col = None
    for candidate in ("icc", "icc31", "median_icc31", "mean_icc31", "icc_value"):
        if candidate in out.columns:
            icc_col = candidate
            break

    cov_col = None
    for candidate in ("cov_pct", "cov_percent", "median_cov_pct", "cov", "cv_pct"):
        if candidate in out.columns:
            cov_col = candidate
            break

    if class_col is None and (icc_col is None or cov_col is None):
        raise RuntimeError(
            "ICC table has neither an explicit class column nor usable ICC/COV columns."
        )

    out = out.loc[out["cohort"].astype(str) == cohort].copy()
    target_norm = normalize_token(roi)
    roi_norm = out[roi_col].map(normalize_token)
    out = out.loc[roi_norm == target_norm].copy()
    if out.empty:
        raise RuntimeError(f"No ICC rows found for cohort={cohort} roi={roi}.")

    if "image_type" in out.columns:
        ct_mask = out["image_type"].astype(str).str.upper().eq("CT")
        if ct_mask.any():
            out = out.loc[ct_mask].copy()

    if class_col is None:
        out["robustness_class"] = [
            classify_feature(icc_value, cov_value)
            for icc_value, cov_value in zip(
                pd.to_numeric(out[icc_col], errors="coerce"),
                pd.to_numeric(out[cov_col], errors="coerce"),
            )
        ]
    else:
        out["robustness_class"] = out[class_col].map(normalize_class_label)

    out["robustness_class"] = out["robustness_class"].fillna("Poor")
    out["feature_name"] = out["feature_name"].astype(str)
    out["feature_family"] = out["feature_family"].astype(str).str.lower()

    agg_rows: list[dict[str, object]] = []
    duplicate_features = 0
    for feature_name, group in out.groupby("feature_name", sort=False):
        duplicate_features += int(len(group) > 1)
        class_counts = Counter(group["robustness_class"].tolist())
        chosen_class = class_counts.most_common(1)[0][0]
        family_counts = Counter(group["feature_family"].tolist())
        chosen_family = family_counts.most_common(1)[0][0]
        record = {
            "feature_name": feature_name,
            "feature_family": chosen_family,
            "robustness_class": chosen_class,
        }
        if icc_col is not None:
            record["icc_value"] = float(
                pd.to_numeric(group[icc_col], errors="coerce").median()
            )
        if cov_col is not None:
            record["cov_pct"] = float(
                pd.to_numeric(group[cov_col], errors="coerce").median()
            )
        agg_rows.append(record)
    feature_df = pd.DataFrame(agg_rows)
    feature_df = feature_df.loc[feature_df["feature_family"].isin(FAMILY_ORDER)].copy()

    meta = {
        "icc_rows_after_filter": int(len(out)),
        "icc_feature_count": int(feature_df["feature_name"].nunique()),
        "icc_duplicate_feature_rows_collapsed": int(duplicate_features),
    }
    return feature_df, meta


def classify_feature(icc_value: float, cov_value: float) -> str:
    if pd.notna(icc_value) and pd.notna(cov_value):
        if icc_value >= 0.90 and cov_value <= 10.0:
            return "Robust"
        if 0.75 <= icc_value < 0.90 and cov_value <= 20.0:
            return "Acceptable"
    return "Poor"


def resolve_prostate_clinical(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    errors: list[str] = []
    for path in PROSTATE_CLINICAL_CANDIDATES:
        if not path.exists():
            continue
        try:
            df = load_table(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        if "patient_id" not in df.columns:
            errors.append(f"{path}: missing patient_id")
            continue
        meta = {
            "clinical_source_path": str(path),
            "clinical_source_sha256": sha256_file(path),
            "clinical_rows": int(len(df)),
            "clinical_unique_patients": int(df["patient_id"].nunique()),
        }
        logger.info("Using Prostata clinical source: %s", path)
        return df, meta
    raise RuntimeError(f"Failed to load Prostata clinical data: {errors}")


def resolve_follow_up_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FOLLOW_UP_COLUMN_CANDIDATES if col in df.columns]


def pick_toxicity_column(df: pd.DataFrame) -> str:
    priority = [
        "lymphopenia_g2",
        "lymphopenia_g2_derived",
        "lymphopenia_g2_42d",
        "lymphopenia_g2_90d",
        "ctcae_2/3/4",
    ]
    for column in priority:
        if column in df.columns:
            return column
    raise RuntimeError("No usable G2+ toxicity endpoint column found.")


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(f"Missing {label}; tried {list(candidates)}")


def filter_to_automatic_source(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if "segmentation_source" not in df.columns:
        return df, {"segmentation_source_filter": "not_present"}
    source_series = df["segmentation_source"].astype(str)
    auto_mask = source_series.str.contains(
        "autorts|totalsegment|total_segment|ts", case=False, regex=True, na=False
    )
    if auto_mask.any():
        filtered = df.loc[auto_mask].copy()
        return filtered, {
            "segmentation_source_filter": "automatic_only",
            "segmentation_source_values": sorted(
                filtered["segmentation_source"].astype(str).unique().tolist()
            ),
        }
    return df, {
        "segmentation_source_filter": "no_automatic_match_retained_all",
        "segmentation_source_values": sorted(source_series.unique().tolist()),
    }


def collect_wide_feature_columns(columns: list[str], allowed_features: set[str]) -> list[str]:
    feature_columns: list[str] = []
    for column in columns:
        lower = column.lower()
        if column in META_COLUMNS:
            continue
        if lower.startswith("diagnostics_"):
            continue
        if allowed_features and column in allowed_features:
            feature_columns.append(column)
            continue
        if any(token in lower for token in RADIOMICS_FAMILY_TOKENS):
            feature_columns.append(column)
    return feature_columns


def load_prostate_radiomics(
    logger: logging.Logger, allowed_features: set[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    errors: list[str] = []
    for path in PROSTATE_RADIOMICS_CANDIDATES:
        if not path.exists():
            continue
        try:
            raw = load_table(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue

        raw.columns = [str(col) for col in raw.columns]
        roi_col = None
        for candidate in ("roi_name", "roi", "structure", "structure_name", "__roi_name__"):
            if candidate in raw.columns:
                roi_col = candidate
                break
        if roi_col is None:
            errors.append(f"{path}: missing ROI column")
            continue
        patient_col = pick_column(raw, ("patient_id",), "patient_id")
        course_col = "course_id" if "course_id" in raw.columns else None

        if "modality" in raw.columns:
            ct_mask = raw["modality"].astype(str).str.upper().eq("CT")
            if ct_mask.any():
                raw = raw.loc[ct_mask].copy()
        elif "image_type" in raw.columns:
            ct_mask = raw["image_type"].astype(str).str.upper().eq("CT")
            if ct_mask.any():
                raw = raw.loc[ct_mask].copy()

        raw, source_meta = filter_to_automatic_source(raw)

        roi_mask = raw[roi_col].map(normalize_token) == normalize_token(PROSTATE_TARGET_ROI)
        raw = raw.loc[roi_mask].copy()
        if raw.empty:
            errors.append(f"{path}: no rows for ROI={PROSTATE_TARGET_ROI}")
            continue

        if {"feature_name", "value"}.issubset(raw.columns):
            long = raw[["patient_id", "feature_name", "value"]].copy()
            if course_col is not None:
                long["course_id"] = raw[course_col].astype(str)
            long["feature_name"] = long["feature_name"].astype(str)
            if allowed_features:
                long = long.loc[long["feature_name"].isin(allowed_features)].copy()
            if long.empty:
                errors.append(f"{path}: no allowed features after long-format filter")
                continue
            index_cols = ["patient_id"] + (["course_id"] if course_col is not None else [])
            wide = (
                long.pivot_table(
                    index=index_cols,
                    columns="feature_name",
                    values="value",
                    aggfunc="median",
                )
                .reset_index()
            )
            feature_columns = [col for col in wide.columns if col not in index_cols]
        else:
            feature_columns = collect_wide_feature_columns(list(raw.columns), allowed_features)
            if not feature_columns:
                errors.append(f"{path}: no radiomics feature columns detected")
                continue
            keep_cols = [patient_col] + feature_columns
            if course_col is not None:
                keep_cols.insert(1, course_col)
            wide = raw[keep_cols].copy()

        wide.rename(columns={patient_col: "patient_id"}, inplace=True)
        if course_col is not None and course_col != "course_id":
            wide.rename(columns={course_col: "course_id"}, inplace=True)

        wide["patient_id"] = wide["patient_id"].astype(str)
        if "course_id" in wide.columns:
            wide["course_id"] = wide["course_id"].astype(str)

        meta = {
            "radiomics_source_path": str(path),
            "radiomics_source_sha256": sha256_file(path),
            "radiomics_rows_after_roi_filter": int(len(wide)),
            "radiomics_feature_columns": int(len(feature_columns)),
            "radiomics_unique_patients": int(wide["patient_id"].nunique()),
        }
        meta.update(source_meta)
        logger.info("Using Prostata radiomics source: %s", path)
        return wide, meta
    raise RuntimeError(f"Failed to load Prostata radiomics: {errors}")


def collapse_duplicate_patients(
    radiomics_df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_columns = [col for col in radiomics_df.columns if col not in {"patient_id", "course_id"}]
    counts = radiomics_df.groupby("patient_id").size()
    dup_patients = counts[counts > 1]
    if dup_patients.empty:
        return radiomics_df.copy(), {
            "duplicate_patient_rows_collapsed": 0,
            "duplicate_patient_ids": [],
        }
    logger.warning(
        "Collapsing duplicate radiomics rows for %d patient_id values by median across rows.",
        len(dup_patients),
    )
    collapsed = (
        radiomics_df.groupby("patient_id", as_index=False)[feature_columns]
        .median(numeric_only=True)
        .copy()
    )
    return collapsed, {
        "duplicate_patient_rows_collapsed": int(len(dup_patients)),
        "duplicate_patient_ids": dup_patients.index.astype(str).tolist(),
    }


def build_prostata_toxicity_frame(
    clinical_df: pd.DataFrame,
    radiomics_df: pd.DataFrame,
    icc_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    toxicity_col = pick_toxicity_column(clinical_df)
    clinical = clinical_df.copy()
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    clinical["endpoint"] = pd.to_numeric(clinical[toxicity_col], errors="coerce")
    clinical = clinical.loc[clinical["endpoint"].isin([0, 1])].copy()

    if "course_id" not in clinical.columns and "course_id" in radiomics_df.columns:
        radiomics_wide, collapse_meta = collapse_duplicate_patients(radiomics_df, logger)
    else:
        radiomics_wide = radiomics_df.copy()
        collapse_meta = {
            "duplicate_patient_rows_collapsed": 0,
            "duplicate_patient_ids": [],
        }

    merge_keys = ["patient_id"]
    if "course_id" in clinical.columns and "course_id" in radiomics_wide.columns:
        clinical["course_id"] = clinical["course_id"].astype(str)
        merge_keys = ["patient_id", "course_id"]

    merged = clinical.merge(radiomics_wide, on=merge_keys, how="inner")
    if merged.empty:
        raise RuntimeError("Clinical/radiomics merge yielded zero rows for Prostata toxicity.")

    allowed_features = set(icc_df["feature_name"].astype(str))
    feature_columns = [col for col in merged.columns if col in allowed_features]
    if not feature_columns:
        raise RuntimeError("Merged Prostata frame has no ICC-mapped radiomics features.")

    merged[feature_columns] = merged[feature_columns].apply(pd.to_numeric, errors="coerce")
    nonempty = [col for col in feature_columns if merged[col].notna().sum() >= 20]
    if not nonempty:
        raise RuntimeError("No Prostata radiomics features have at least 20 non-missing values.")

    out = merged[merge_keys + ["endpoint"] + nonempty].copy()
    meta = {
        "endpoint_column": toxicity_col,
        "n_rows_merged": int(len(out)),
        "n_patients_merged": int(out["patient_id"].nunique()),
        "n_events": int(out["endpoint"].sum()),
        "n_features_after_missingness_filter": int(len(nonempty)),
        "merge_keys": merge_keys,
    }
    meta.update(collapse_meta)
    return out, meta


def fit_univariate_logistic(
    df: pd.DataFrame, outcome_col: str, feature_name: str
) -> dict[str, object] | None:
    sub = df[[outcome_col, feature_name]].dropna().copy()
    if len(sub) < 20:
        return None
    y = sub[outcome_col].astype(int)
    if y.nunique() < 2:
        return None
    x_raw = pd.to_numeric(sub[feature_name], errors="coerce")
    if x_raw.isna().any():
        sub = sub.loc[x_raw.notna()].copy()
        x_raw = x_raw.loc[x_raw.notna()]
        y = sub[outcome_col].astype(int)
    if len(sub) < 20:
        return None
    sd = float(x_raw.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return None
    x = (x_raw - float(x_raw.mean())) / sd
    X = sm.add_constant(x, has_constant="add")
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False, maxiter=200)
    except Exception:
        return None
    coef = float(result.params.iloc[1])
    se = float(result.bse.iloc[1])
    p_value = float(result.pvalues.iloc[1])
    z_value = sp_stats.norm.ppf(0.975)
    ci_lo = coef - z_value * se
    ci_hi = coef + z_value * se
    if not all(np.isfinite(value) for value in (coef, se, p_value, ci_lo, ci_hi)):
        return None
    return {
        "feature_name": feature_name,
        "n_obs": int(len(sub)),
        "n_events": int(y.sum()),
        "log_or": coef,
        "log_or_se": se,
        "or": safe_exp(coef),
        "or_ci_lo": safe_exp(ci_lo),
        "or_ci_hi": safe_exp(ci_hi),
        "wald_z": float(coef / se) if se > 0 else np.nan,
        "p_value": p_value,
        "neg_log10_p": float(-np.log10(max(p_value, 1e-300))),
    }


def run_univariate_logistic(
    frame_df: pd.DataFrame, outcome_col: str, icc_df: pd.DataFrame
) -> pd.DataFrame:
    class_map = icc_df.set_index("feature_name")["robustness_class"].to_dict()
    family_map = icc_df.set_index("feature_name")["feature_family"].to_dict()
    rows: list[dict[str, object]] = []
    feature_columns = [col for col in frame_df.columns if col in class_map]
    for feature_name in feature_columns:
        result = fit_univariate_logistic(frame_df, outcome_col, feature_name)
        if result is None:
            continue
        result["robustness_class"] = class_map[feature_name]
        result["feature_family"] = family_map.get(feature_name, infer_feature_family(feature_name))
        result["leg"] = "prostata_toxicity_g2plus"
        result["effect_model"] = "logistic"
        rows.append(result)
    return pd.DataFrame(rows)


def dersimonian_laird(log_effects: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    weights = 1.0 / variances
    fixed_mean = float(np.sum(weights * log_effects) / np.sum(weights))
    q_stat = float(np.sum(weights * (log_effects - fixed_mean) ** 2))
    df_q = max(len(log_effects) - 1, 0)
    c_term = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau_sq = 0.0
    if df_q > 0 and c_term > 0:
        tau_sq = max(0.0, (q_stat - df_q) / c_term)
    weights_re = 1.0 / (variances + tau_sq)
    pooled = float(np.sum(weights_re * log_effects) / np.sum(weights_re))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights_re)))
    z_value = sp_stats.norm.ppf(0.975)
    ci_lo = pooled - z_value * pooled_se
    ci_hi = pooled + z_value * pooled_se
    i_sq = 0.0
    if q_stat > 0 and df_q > 0:
        i_sq = max(0.0, (q_stat - df_q) / q_stat) * 100.0
    return {
        "pooled_log_effect": pooled,
        "pooled_log_effect_ci_lo": ci_lo,
        "pooled_log_effect_ci_hi": ci_hi,
        "pooled_effect": safe_exp(pooled),
        "pooled_effect_ci_lo": safe_exp(ci_lo),
        "pooled_effect_ci_hi": safe_exp(ci_hi),
        "tau_sq": tau_sq,
        "i2_pct": i_sq,
        "q_stat": q_stat,
    }


def pooled_meta_summary(univar_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = univar_df.loc[
        univar_df["log_or"].notna()
        & univar_df["log_or_se"].notna()
        & (univar_df["log_or_se"] > 0)
    ].copy()
    usable["variance"] = usable["log_or_se"] ** 2

    class_rows: list[dict[str, object]] = []
    class_family_rows: list[dict[str, object]] = []

    for class_name in CLASS_ORDER:
        block = usable.loc[usable["robustness_class"] == class_name].copy()
        if len(block) < 2:
            continue
        meta = dersimonian_laird(
            block["log_or"].to_numpy(dtype=float),
            block["variance"].to_numpy(dtype=float),
        )
        meta.update(
            {
                "leg": "prostata_toxicity_g2plus",
                "effect_model": "logistic",
                "robustness_class": class_name,
                "n_features": int(len(block)),
                "median_neg_log10_p": float(block["neg_log10_p"].median()),
            }
        )
        class_rows.append(meta)

        for family_name in FAMILY_ORDER:
            sub = block.loc[block["feature_family"] == family_name].copy()
            if len(sub) < 2:
                continue
            fam_meta = dersimonian_laird(
                sub["log_or"].to_numpy(dtype=float),
                sub["variance"].to_numpy(dtype=float),
            )
            fam_meta.update(
                {
                    "leg": "prostata_toxicity_g2plus",
                    "effect_model": "logistic",
                    "robustness_class": class_name,
                    "feature_family": family_name,
                    "n_features": int(len(sub)),
                }
            )
            class_family_rows.append(fam_meta)

    return pd.DataFrame(class_rows), pd.DataFrame(class_family_rows)


def ks_class_summary(univar_df: pd.DataFrame) -> pd.DataFrame:
    values = univar_df[["robustness_class", "neg_log10_p"]].dropna().copy()
    rows: list[dict[str, object]] = []
    for class_name in CLASS_ORDER:
        block = values.loc[values["robustness_class"] == class_name, "neg_log10_p"].to_numpy(
            dtype=float
        )
        if block.size == 0:
            continue
        rows.append(
            {
                "comparison_type": "descriptive",
                "class_a": class_name,
                "class_b": None,
                "n_a": int(block.size),
                "n_b": None,
                "median_neg_log10_p_a": float(np.median(block)),
                "median_neg_log10_p_b": None,
                "ks_statistic": None,
                "p_value": None,
            }
        )
    for first, second in (("Robust", "Poor"), ("Robust", "Acceptable"), ("Acceptable", "Poor")):
        a = values.loc[values["robustness_class"] == first, "neg_log10_p"].to_numpy(dtype=float)
        b = values.loc[values["robustness_class"] == second, "neg_log10_p"].to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            continue
        ks_stat, p_value = sp_stats.ks_2samp(a, b, alternative="two-sided", mode="auto")
        rows.append(
            {
                "comparison_type": "ks",
                "class_a": first,
                "class_b": second,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "median_neg_log10_p_a": float(np.median(a)),
                "median_neg_log10_p_b": float(np.median(b)),
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def build_outer_cv(
    y: np.ndarray, outer_folds: int, seed: int
) -> StratifiedKFold:
    min_class = int(np.min(np.bincount(y.astype(int))))
    if min_class < 2:
        raise RuntimeError("Outcome lacks enough members for stratified CV.")
    n_splits = min(outer_folds, min_class)
    if n_splits < 2:
        raise RuntimeError("Need at least 2 stratified outer folds.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def nested_panel_compare(
    frame_df: pd.DataFrame,
    outcome_col: str,
    icc_df: pd.DataFrame,
    outer_folds: int,
    prescreen_k: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_features = {
        class_name: sorted(
            icc_df.loc[icc_df["robustness_class"] == class_name, "feature_name"].astype(str)
        )
        for class_name in CLASS_ORDER
    }
    available_features = set(frame_df.columns).intersection(
        icc_df["feature_name"].astype(str).tolist()
    )
    y = frame_df[outcome_col].to_numpy(dtype=int)
    splitter = build_outer_cv(y, outer_folds=outer_folds, seed=seed)

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for class_name in CLASS_ORDER:
        features = [feature for feature in class_features[class_name] if feature in available_features]
        if not features:
            continue
        class_fold_rows: list[dict[str, object]] = []

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(frame_df, y), start=1):
            train_df = frame_df.iloc[train_idx].copy()
            test_df = frame_df.iloc[test_idx].copy()
            train_events = int(train_df[outcome_col].sum())
            event_budget = max(1, train_events // 10)
            prescreen_rows: list[dict[str, object]] = []
            for feature_name in features:
                result = fit_univariate_logistic(train_df, outcome_col, feature_name)
                if result is not None:
                    prescreen_rows.append(result)
            if not prescreen_rows:
                continue
            prescreen_df = pd.DataFrame(prescreen_rows).sort_values("p_value", kind="stable")
            selected_k = min(prescreen_k, event_budget, len(prescreen_df))
            selected_features = prescreen_df["feature_name"].head(selected_k).tolist()
            if not selected_features:
                continue

            estimator = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            solver="liblinear",
                            max_iter=1000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            X_train = train_df[selected_features]
            X_test = test_df[selected_features]
            y_train = train_df[outcome_col].to_numpy(dtype=int)
            y_test = test_df[outcome_col].to_numpy(dtype=int)
            estimator.fit(X_train, y_train)
            prob = estimator.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
            brier = brier_score_loss(y_test, prob)
            row = {
                "leg": "prostata_toxicity_g2plus",
                "robustness_class": class_name,
                "fold_id": fold_id,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "train_events": train_events,
                "selected_k": int(len(selected_features)),
                "event_budget_train": int(event_budget),
                "auc": float(auc),
                "brier": float(brier),
                "selected_features": "|".join(selected_features),
            }
            fold_rows.append(row)
            class_fold_rows.append(row)

        if not class_fold_rows:
            continue
        block = pd.DataFrame(class_fold_rows)
        feature_counter = Counter()
        for feature_blob in block["selected_features"].tolist():
            feature_counter.update(token for token in feature_blob.split("|") if token)
        top_features = "|".join([feature for feature, _ in feature_counter.most_common(10)])
        summary_rows.append(
            {
                "leg": "prostata_toxicity_g2plus",
                "robustness_class": class_name,
                "n_folds": int(len(block)),
                "mean_auc": float(block["auc"].mean()),
                "median_auc": float(block["auc"].median()),
                "sd_auc": float(block["auc"].std(ddof=0)),
                "mean_brier": float(block["brier"].mean()),
                "mean_selected_k": float(block["selected_k"].mean()),
                "top_selected_features": top_features,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def build_figure(
    univar_df: pd.DataFrame, pooled_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    plot_df = univar_df.copy()
    positions = np.arange(len(CLASS_ORDER))
    box_data = []
    labels = []
    for class_name in CLASS_ORDER:
        values = plot_df.loc[
            plot_df["robustness_class"] == class_name, "neg_log10_p"
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        box_data.append(values)
        labels.append(class_name)
    if box_data:
        axes[0].boxplot(box_data, labels=labels, patch_artist=True)
    axes[0].set_title("-log10(p) by robustness class")
    axes[0].set_ylabel("-log10(p)")

    if not pooled_df.empty:
        pooled_df = pooled_df.copy()
        pooled_df["class_order"] = pooled_df["robustness_class"].map(
            {label: index for index, label in enumerate(CLASS_ORDER)}
        )
        pooled_df = pooled_df.sort_values("class_order")
        pooled_df["pooled_effect_plot"] = pooled_df["pooled_log_effect"].apply(
            lambda value: safe_exp(float(np.clip(value, -MAX_LOG_EXP, MAX_LOG_EXP)))
        )
        pooled_df["pooled_effect_ci_lo_plot"] = pooled_df["pooled_log_effect_ci_lo"].apply(
            lambda value: safe_exp(float(np.clip(value, -MAX_LOG_EXP, MAX_LOG_EXP)))
        )
        pooled_df["pooled_effect_ci_hi_plot"] = pooled_df["pooled_log_effect_ci_hi"].apply(
            lambda value: safe_exp(float(np.clip(value, -MAX_LOG_EXP, MAX_LOG_EXP)))
        )
        axes[1].errorbar(
            pooled_df["pooled_effect_plot"],
            np.arange(len(pooled_df)),
            xerr=[
                pooled_df["pooled_effect_plot"] - pooled_df["pooled_effect_ci_lo_plot"],
                pooled_df["pooled_effect_ci_hi_plot"] - pooled_df["pooled_effect_plot"],
            ],
            fmt="o",
            color="#1f77b4",
            capsize=4,
        )
        axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_yticks(np.arange(len(pooled_df)))
        axes[1].set_yticklabels(pooled_df["robustness_class"].tolist())
        axes[1].set_xscale("log")
    axes[1].set_title("Pooled OR by robustness class")
    axes[1].set_xlabel("Pooled OR (log scale)")

    if not panel_df.empty:
        panel_df = panel_df.copy()
        panel_df["class_order"] = panel_df["robustness_class"].map(
            {label: index for index, label in enumerate(CLASS_ORDER)}
        )
        panel_df = panel_df.sort_values("class_order")
        axes[2].bar(panel_df["robustness_class"], panel_df["mean_auc"], color="#2a9d8f")
        axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Nested CV panel AUC")
    axes[2].set_ylabel("Mean AUC")

    fig.suptitle("Step 41 fixed: Prostata toxicity retention")
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300)
    fig.savefig(OUT_FIG_PDF)
    plt.close(fig)


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest_base(job_id: str, args: argparse.Namespace) -> dict[str, object]:
    return {
        "job_id": job_id,
        "hostname": socket.gethostname(),
        "started_at_epoch": time.time(),
        "analysis_dir": str(ANALYSIS_DIR),
        "git_sha": resolve_git_sha(),
        "script_path": str(Path(__file__).resolve()),
        "production_legs": PRODUCTION_LEGS,
        "dropped_legs": {
            "prostata_os": "Dropped in fixed production scope because no observed last-follow-up date was available in the audited accessible sources; fabricated 365d censoring removed entirely.",
            "odbytnice_dfs": "Dropped in fixed production scope because cohort-complete RTSTRUCT/GTV radiomics were not available in an analysis-ready merged export; sacrum surrogate is prohibited.",
            "odbytnice_pcr": "Dropped in fixed production scope because cohort-complete RTSTRUCT/GTV radiomics were not available in an analysis-ready merged export; sacrum surrogate is prohibited.",
        },
        "options_taken": {
            "p0_1_odbytnice": "Option B - drop Odbytnice from production analysis",
            "p0_2_prostata_os": "Option B - drop Prostata OS from production analysis",
        },
        "arguments": {
            "allow_leg_failure": bool(args.allow_leg_failure),
            "outer_folds": int(args.outer_folds),
            "prescreen_k": int(args.prescreen_k),
            "seed": int(args.seed),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=os.environ.get("JOB_ID", "manual"))
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--prescreen-k", type=int, default=25)
    parser.add_argument(
        "--allow-leg-failure",
        action="store_true",
        help="Diagnostic mode only. Production should not use this flag.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_output_dirs()
    args = parse_args()
    logger = configure_logging(args.job_id)
    manifest_path = LOGS_DIR / f"41_manifest_{args.job_id}.json"
    manifest = build_manifest_base(args.job_id, args)
    blockers_all: list[dict[str, object]] = []

    try:
        icc_path, icc_raw, icc_meta = resolve_latest_icc(logger)
        icc_df, icc_filter_meta = standardize_icc_table(
            icc_raw, cohort="Prostata", roi=PROSTATE_TARGET_ROI
        )
        manifest.update(icc_meta)
        manifest.update(icc_filter_meta)
        manifest["icc_table_selected_path"] = str(icc_path)

        clinical_df, clinical_meta = resolve_prostate_clinical(logger)
        manifest.update(clinical_meta)
        manifest["prostata_os_follow_up_columns_found"] = resolve_follow_up_columns(clinical_df)
        manifest["prostata_os_leg_active"] = False

        radiomics_df, radiomics_meta = load_prostate_radiomics(
            logger, allowed_features=set(icc_df["feature_name"].astype(str))
        )
        manifest.update(radiomics_meta)

        frame_df, frame_meta = build_prostata_toxicity_frame(
            clinical_df=clinical_df,
            radiomics_df=radiomics_df,
            icc_df=icc_df,
            logger=logger,
        )
        manifest.update(frame_meta)

        if frame_meta["n_events"] <= 0:
            raise RuntimeError("Prostata toxicity frame has zero observed events.")

        univar_df = run_univariate_logistic(frame_df, outcome_col="endpoint", icc_df=icc_df)
        if univar_df.empty:
            raise RuntimeError("No valid univariate logistic fits were produced.")

        pooled_df, pooled_family_df = pooled_meta_summary(univar_df)
        ks_df = ks_class_summary(univar_df)
        panel_df, panel_fold_df = nested_panel_compare(
            frame_df=frame_df,
            outcome_col="endpoint",
            icc_df=icc_df,
            outer_folds=args.outer_folds,
            prescreen_k=args.prescreen_k,
            seed=args.seed,
        )

        if panel_df.empty:
            raise RuntimeError("Nested CV panel comparison produced no valid folds.")

        univar_df.to_parquet(OUT_FEATURES, index=False)
        pooled_df.to_csv(OUT_CLASS_SUMMARY, index=False)
        pooled_family_df.to_csv(OUT_CLASS_FAMILY_SUMMARY, index=False)
        ks_df.to_csv(OUT_KS, index=False)
        panel_df.to_csv(OUT_PANEL, index=False)
        panel_fold_df.to_csv(OUT_PANEL_FOLDS, index=False)
        build_figure(univar_df=univar_df, pooled_df=pooled_df, panel_df=panel_df)

        manifest["outputs"] = {
            "feature_effects_parquet": str(OUT_FEATURES),
            "class_summary_csv": str(OUT_CLASS_SUMMARY),
            "class_family_summary_csv": str(OUT_CLASS_FAMILY_SUMMARY),
            "ks_csv": str(OUT_KS),
            "panel_summary_csv": str(OUT_PANEL),
            "panel_folds_csv": str(OUT_PANEL_FOLDS),
            "figure_png": str(OUT_FIG_PNG),
            "figure_pdf": str(OUT_FIG_PDF),
        }
        manifest["result_counts"] = {
            "univariate_features": int(len(univar_df)),
            "pooled_class_rows": int(len(pooled_df)),
            "pooled_class_family_rows": int(len(pooled_family_df)),
            "ks_rows": int(len(ks_df)),
            "panel_summary_rows": int(len(panel_df)),
            "panel_fold_rows": int(len(panel_fold_df)),
        }
        manifest["blockers"] = blockers_all
        manifest["completed"] = True
        manifest["ended_at_epoch"] = time.time()
        write_manifest(manifest_path, manifest)
        logger.info("Step 41 fixed complete. Manifest: %s", manifest_path)

    except Exception as exc:
        blockers_all.append(
            {
                "leg": "prostata_toxicity_g2plus",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        manifest["blockers"] = blockers_all
        manifest["completed"] = False
        manifest["ended_at_epoch"] = time.time()
        write_manifest(manifest_path, manifest)
        logger.error("Step 41 fixed failed: %s", exc)
        if args.allow_leg_failure:
            return
        raise SystemExit(2) from exc


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
