#!/usr/bin/python3
"""Step 41: prognostic retention with Odbytnice CTV1 reopened.

Production contract implemented in this revision
-----------------------------------------------
1. Odbytnice uses CTV1 only. There is no sacrum surrogate path.
2. Prostata OS remains dropped because no real observed censoring field is
   available in the audited accessible sources.
3. Odbytnice OS uses observed follow-up dates only; no synthetic censoring.
4. Logistic fits no longer rely on unstable unpenalized exponentiation paths:
   when statsmodels warns or fails, the script falls back to L2-penalized
   sklearn logistic regression and records `fit_status=separation_warning`.
5. Cox fits use `lifelines.CoxPHFitter(penalizer=0.01)`.
6. Feature prescreening is performed inside each outer CV training fold only,
   with the selected feature count capped by `floor(train_events / 10)`.

Outputs
-------
Under the selected analysis root:
    data/prognostic_retention_feature_effects.parquet
    tables/prognostic_retention_class_summary.csv
    tables/prognostic_retention_class_family_summary.csv
    tables/prognostic_retention_cross_leg_summary.csv
    tables/prognostic_retention_ks.csv
    tables/prognostic_retention_panel_cv.csv
    tables/prognostic_retention_panel_cv_folds.csv
    logs/41_manifest_<job_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
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
TARGET_ROI = {
    "Prostata": "prostate",
    "Odbytnice": "CTV1",
}
PRODUCTION_LEGS = [
    "prostata_toxicity_g2plus",
    "odbytnice_os",
    "odbytnice_dfs",
    "odbytnice_pcr",
]
LOGISTIC_DEFAULT_C = 0.25
COX_PENALIZER = 0.01
MIN_OBS_UNIVARIATE = 20
MIN_EVENTS_SURVIVAL = 5
MIN_TEST_CLASS_MEMBERS = 1
MIN_TEST_EVENTS_SURVIVAL = 1
ARGOS_ICC = "argos-worker:/home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet"

LOCAL_ANALYSIS_DIR = Path("/umed-projekty/rtpipeline/manuscript/analysis")
DEFAULT_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"

PROSTATE_CLINICAL_CANDIDATES = [
    Path("/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.parquet"),
    Path("/projekty/Prostata/ASCO2026_radiomics/results/analytic_cohort.csv"),
    Path("/projekty/Prostata/LK/Data/bdo.tsv"),
    Path("/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/analytic_cohort.parquet"),
]
PROSTATE_RADIOMICS_CANDIDATES = [
    Path("/projekty/Prostata/Data_raw/DICOM_rtpipeline/Output/_ANALYSIS_READY/radiomics_ct_filtered.xlsx"),
    Path("/home/kgs24/Prostate_Radiomics_Kopernik_ASTRO2026/results/radiomics_ts_all.parquet"),
]
PROSTATE_PER_COURSE_ROOT = Path("/projekty/Prostata/Data_raw/DICOM_rtpipeline/Output")

ODBYTNICE_CLINICAL_CSV = Path(
    "/umed-projekty/ODBYTNICE2026/data_bucket/ClinicalData20260304/final/odbytnice_clinical_2026.csv"
)
ODBYTNICE_PESEL_MAP = Path("/umed-projekty/ODBYTNICE2026/analysis/patient_id_pesel_mapping.csv")
ODBYTNICE_AIM3 = Path("/umed-projekty/ODBYTNICE2026/analysis/aim3_analysis_dataset.parquet")

ROI_COLUMN_CANDIDATES = ("roi_name", "roi", "structure", "structure_name", "__roi_name__")
META_COLUMNS = {
    "patient_id",
    "course_id",
    "subject",
    "roi_name",
    "source_roi_name",
    "PESEL",
    "modality",
    "image_type",
    "segmentation_source",
    "output_dir",
    "course_dir",
    "structure_cropped",
    "roi_original_name",
}
RADIOMICS_FAMILY_TOKENS = tuple(f"_{family}_" for family in FAMILY_ORDER)
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
]

LEG_SPECS = [
    {
        "leg": "prostata_toxicity_g2plus",
        "cohort": "Prostata",
        "effect_model": "logistic",
        "roi_name": TARGET_ROI["Prostata"],
        "endpoint_kind": "binary",
        "outcome_col": "endpoint",
    },
    {
        "leg": "odbytnice_os",
        "cohort": "Odbytnice",
        "effect_model": "cox",
        "roi_name": TARGET_ROI["Odbytnice"],
        "time_col": "os_days",
        "event_col": "os_event",
    },
    {
        "leg": "odbytnice_dfs",
        "cohort": "Odbytnice",
        "effect_model": "cox",
        "roi_name": TARGET_ROI["Odbytnice"],
        "time_col": "dfs_days",
        "event_col": "dfs_event",
    },
    {
        "leg": "odbytnice_pcr",
        "cohort": "Odbytnice",
        "effect_model": "logistic",
        "roi_name": TARGET_ROI["Odbytnice"],
        "endpoint_kind": "binary",
        "outcome_col": "pcr",
    },
]


class ContextFilter(logging.Filter):
    def __init__(self, host: str, job_id: str) -> None:
        super().__init__()
        self.host = host
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.host = self.host
        record.job_id = self.job_id
        return True


def configure_logging(job_id: str) -> logging.Logger:
    logger = logging.getLogger("script41")
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
    parser.add_argument("--job-id", default=os.environ.get("JOB_ID", "manual"))
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--prescreen-k", type=int, default=25)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-features", type=int, default=80)
    parser.add_argument("--logistic-c", type=float, default=LOGISTIC_DEFAULT_C)
    parser.add_argument("--allow-leg-failure", action="store_true")
    parser.add_argument(
        "--fetch-icc-from-argos",
        action="store_true",
        default=True,
        help="If local icc_results.parquet is absent, copy the rebuilt file from argos-worker.",
    )
    return parser.parse_args()


def resolve_analysis_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def normalize_class_label(value: object) -> str | None:
    mapping = {
        "robust": "Robust",
        "acceptable": "Acceptable",
        "poor": "Poor",
    }
    return mapping.get(normalize_token(value))


def safe_exp(value: float) -> float:
    return float(np.exp(np.clip(value, -700.0, 700.0)))


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


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_table(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=dtype)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=dtype)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def safe_is_readable(path: Path, timeout_seconds: float = 2.0) -> bool:
    path_str = str(path)
    if not path_str.startswith("/home/"):
        return path.exists()
    try:
        result = subprocess.run(
            ["bash", "-lc", f"test -r {shlex.quote(path_str)}"],
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def fetch_remote_file(remote: str, local_path: Path, logger: logging.Logger) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying missing source from %s to %s", remote, local_path)
    subprocess.run(["scp", remote, str(local_path)], check=True)
    return local_path


def resolve_icc_path(analysis_dir: Path, fetch_remote: bool, logger: logging.Logger) -> Path:
    local_candidates = [
        analysis_dir / "data" / "icc_results.parquet",
        LOCAL_ANALYSIS_DIR / "data" / "icc_results.parquet",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate
    if not fetch_remote:
        raise FileNotFoundError("Local icc_results.parquet not found and remote fetch disabled.")
    target = analysis_dir / "data" / "icc_results.parquet"
    return fetch_remote_file(ARGOS_ICC, target, logger)


def standardize_icc_table(
    icc_df: pd.DataFrame, cohort: str, roi: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = icc_df.copy()
    out.columns = [str(column) for column in out.columns]
    if "feature_name" not in out.columns or "cohort" not in out.columns:
        raise RuntimeError("ICC table is missing feature_name or cohort.")

    roi_column = next((c for c in ROI_COLUMN_CANDIDATES if c in out.columns), None)
    if roi_column is None:
        raise RuntimeError("ICC table is missing an ROI identifier column.")

    class_column = next(
        (
            c
            for c in (
                "icc_class",
                "robustness_label",
                "robustness_class",
                "classification_icc31",
                "classification",
            )
            if c in out.columns
        ),
        None,
    )
    icc_column = next(
        (c for c in ("icc", "icc31", "median_icc31", "mean_icc31", "icc_value") if c in out.columns),
        None,
    )
    cov_column = next(
        (c for c in ("cov_pct", "median_cov_pct", "cov", "cv_pct", "cov_percent") if c in out.columns),
        None,
    )

    out = out.loc[out["cohort"].astype(str) == cohort].copy()
    roi_norm = out[roi_column].astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    out = out.loc[roi_norm == roi.replace(" ", "").lower()].copy()
    if out.empty:
        raise RuntimeError(f"No ICC rows found for cohort={cohort} roi={roi}.")

    if "image_type" in out.columns:
        ct_mask = out["image_type"].astype(str).str.upper().eq("CT")
        if ct_mask.any():
            out = out.loc[ct_mask].copy()

    if "feature_family" in out.columns:
        out["feature_family"] = out["feature_family"].astype(str).str.lower()
    elif "family" in out.columns:
        out["feature_family"] = out["family"].astype(str).str.lower()
    else:
        out["feature_family"] = out["feature_name"].map(infer_feature_family)

    if class_column is None:
        if icc_column is None or cov_column is None:
            raise RuntimeError("ICC table has neither explicit class labels nor ICC/COV values.")
        out["robustness_class"] = [
            classify_feature(icc_value, cov_value)
            for icc_value, cov_value in zip(
                pd.to_numeric(out[icc_column], errors="coerce"),
                pd.to_numeric(out[cov_column], errors="coerce"),
            )
        ]
    else:
        out["robustness_class"] = out[class_column].map(normalize_class_label).fillna("Poor")

    aggregated_rows: list[dict[str, object]] = []
    duplicate_features = 0
    for feature_name, group in out.groupby("feature_name", sort=False):
        duplicate_features += int(len(group) > 1)
        class_counts = Counter(group["robustness_class"].astype(str).tolist())
        family_counts = Counter(group["feature_family"].astype(str).tolist())
        row = {
            "feature_name": str(feature_name),
            "feature_family": family_counts.most_common(1)[0][0],
            "robustness_class": class_counts.most_common(1)[0][0],
        }
        if icc_column is not None:
            row["icc_value"] = float(pd.to_numeric(group[icc_column], errors="coerce").median())
        if cov_column is not None:
            row["cov_pct"] = float(pd.to_numeric(group[cov_column], errors="coerce").median())
        aggregated_rows.append(row)

    feature_df = pd.DataFrame(aggregated_rows)
    feature_df = feature_df.loc[feature_df["feature_family"].isin(FAMILY_ORDER)].copy()
    meta = {
        "cohort": cohort,
        "roi_name": roi,
        "icc_rows_after_filter": int(len(out)),
        "icc_feature_count": int(feature_df["feature_name"].nunique()),
        "icc_duplicate_feature_rows_collapsed": int(duplicate_features),
    }
    return feature_df, meta


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(f"Missing {label}; tried {list(candidates)}")


def collect_feature_columns(columns: list[str], allowed_features: set[str]) -> list[str]:
    feature_columns: list[str] = []
    for column in columns:
        lower = column.lower()
        if column in META_COLUMNS or lower.startswith("diagnostics_"):
            continue
        if column in allowed_features:
            feature_columns.append(column)
        elif any(token in lower for token in RADIOMICS_FAMILY_TOKENS):
            feature_columns.append(column)
    return feature_columns


def filter_to_automatic_source(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if "segmentation_source" not in df.columns:
        return df, {"segmentation_source_filter": "not_present"}
    series = df["segmentation_source"].astype(str)
    auto_mask = series.str.contains("autorts|totalsegment|total_segment|ts", case=False, regex=True, na=False)
    if auto_mask.any():
        filtered = df.loc[auto_mask].copy()
        return filtered, {
            "segmentation_source_filter": "automatic_only",
            "segmentation_source_values": sorted(filtered["segmentation_source"].astype(str).unique().tolist()),
        }
    return df, {
        "segmentation_source_filter": "no_automatic_match_retained_all",
        "segmentation_source_values": sorted(series.unique().tolist()),
    }


def resolve_prostate_clinical(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    errors: list[str] = []
    for path in PROSTATE_CLINICAL_CANDIDATES:
        if not safe_is_readable(path):
            continue
        try:
            df = load_table(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        if "patient_id" not in df.columns:
            errors.append(f"{path}: missing patient_id")
            continue
        df["patient_id"] = df["patient_id"].astype(str)
        meta = {
            "clinical_source_path": str(path),
            "clinical_source_sha256": sha256_file(path),
            "clinical_rows": int(len(df)),
            "clinical_unique_patients": int(df["patient_id"].nunique()),
            "prostata_os_follow_up_columns_found": [
                column for column in FOLLOW_UP_COLUMN_CANDIDATES if column in df.columns
            ],
        }
        logger.info("Using Prostata clinical source: %s", path)
        return df, meta
    raise RuntimeError(f"Failed to load Prostata clinical data: {errors}")


def pick_toxicity_column(df: pd.DataFrame) -> str:
    for column in (
        "lymphopenia_g2",
        "lymphopenia_g2_derived",
        "lymphopenia_g2_42d",
        "lymphopenia_g2_90d",
        "ctcae_2/3/4",
    ):
        if column in df.columns:
            return column
    raise RuntimeError("No usable G2+ toxicity endpoint column found.")


def pick_prostate_smoke_patient_ids(clinical_df: pd.DataFrame) -> tuple[set[str], dict[str, object]]:
    endpoint_col = pick_toxicity_column(clinical_df)
    endpoint = pd.to_numeric(clinical_df[endpoint_col], errors="coerce")
    eligible = clinical_df.loc[endpoint.isin([0, 1]), ["patient_id"]].copy()
    patient_ids = set(eligible["patient_id"].astype(str))
    meta = {
        "smoke_endpoint_column": endpoint_col,
        "smoke_patient_pool": int(len(patient_ids)),
        "smoke_positive_patients": int((endpoint == 1).sum()),
        "smoke_negative_patients": int((endpoint == 0).sum()),
    }
    return patient_ids, meta


def extract_prostate_radiomics_wide(
    raw: pd.DataFrame,
    *,
    source_path: Path,
    allowed_features: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw.columns = [str(column) for column in raw.columns]
    roi_column = next((c for c in ROI_COLUMN_CANDIDATES if c in raw.columns), None)
    if roi_column is None:
        raise RuntimeError(f"{source_path}: missing ROI column")
    patient_column = pick_column(raw, ("patient_id",), "patient_id")
    course_column = "course_id" if "course_id" in raw.columns else None

    if "modality" in raw.columns:
        ct_mask = raw["modality"].astype(str).str.upper().eq("CT")
        if ct_mask.any():
            raw = raw.loc[ct_mask].copy()
    elif "image_type" in raw.columns:
        ct_mask = raw["image_type"].astype(str).str.upper().eq("CT")
        if ct_mask.any():
            raw = raw.loc[ct_mask].copy()

    raw, source_meta = filter_to_automatic_source(raw)
    roi_mask = raw[roi_column].astype(str).map(normalize_token) == normalize_token(TARGET_ROI["Prostata"])
    raw = raw.loc[roi_mask].copy()
    if raw.empty:
        raise RuntimeError(f"{source_path}: no rows for ROI={TARGET_ROI['Prostata']}")

    feature_columns = collect_feature_columns(raw.columns.tolist(), allowed_features)
    if not feature_columns:
        raise RuntimeError(f"{source_path}: no radiomics feature columns detected")

    keep_columns = [patient_column] + feature_columns
    if course_column is not None:
        keep_columns.insert(1, course_column)
    wide = raw[keep_columns].copy()
    wide.rename(columns={patient_column: "patient_id"}, inplace=True)
    if course_column is not None and course_column != "course_id":
        wide.rename(columns={course_column: "course_id"}, inplace=True)
    wide["patient_id"] = wide["patient_id"].astype(str)
    if "course_id" in wide.columns:
        wide["course_id"] = wide["course_id"].astype(str)
    else:
        wide["course_id"] = "course0"
    wide["subject"] = wide["patient_id"] + "_" + wide["course_id"]

    meta = {
        "radiomics_source_path": str(source_path),
        "radiomics_source_sha256": sha256_file(source_path),
        "radiomics_rows_after_roi_filter": int(len(wide)),
        "radiomics_unique_patients": int(wide["patient_id"].nunique()),
        "radiomics_unique_subjects": int(wide["subject"].nunique()),
        "radiomics_feature_columns": int(len(feature_columns)),
    }
    meta.update(source_meta)
    return wide, meta


def load_prostate_radiomics_smoke(
    logger: logging.Logger,
    allowed_features: set[str],
    smoke_patient_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    parquet_paths: list[Path] = []
    for patient_id in sorted(smoke_patient_ids):
        parquet_paths.extend(sorted(PROSTATE_PER_COURSE_ROOT.glob(f"{patient_id}/*/radiomics_ct.parquet")))
    if not parquet_paths:
        raise RuntimeError("No per-course Prostata parquet files matched the smoke patient pool.")

    frames: list[pd.DataFrame] = []
    load_errors: list[str] = []
    loaded_sources: list[str] = []
    subject_count = 0
    for path in parquet_paths:
        try:
            raw = pd.read_parquet(path)
            wide, _ = extract_prostate_radiomics_wide(raw, source_path=path, allowed_features=allowed_features)
        except Exception as exc:
            load_errors.append(f"{path}: {exc}")
            continue
        frames.append(wide)
        loaded_sources.append(str(path))
        subject_count += int(wide["subject"].nunique())
    if not frames:
        raise RuntimeError(f"Smoke parquet loading failed: {load_errors[:10]}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["subject"], keep="first").reset_index(drop=True)
    feature_columns = collect_feature_columns(combined.columns.tolist(), allowed_features)
    meta = {
        "radiomics_source_mode": "smoke_per_course_parquet",
        "radiomics_source_path": str(PROSTATE_PER_COURSE_ROOT),
        "radiomics_rows_after_roi_filter": int(len(combined)),
        "radiomics_unique_patients": int(combined["patient_id"].nunique()),
        "radiomics_unique_subjects": int(combined["subject"].nunique()),
        "radiomics_feature_columns": int(len(feature_columns)),
        "radiomics_course_files_requested": int(len(parquet_paths)),
        "radiomics_course_files_loaded": int(len(loaded_sources)),
        "radiomics_subjects_loaded_pre_dedup": int(subject_count),
        "radiomics_loaded_source_paths_sample": loaded_sources[:20],
    }
    if load_errors:
        meta["radiomics_load_errors_sample"] = load_errors[:20]
    logger.info(
        "Using Prostata smoke radiomics source: %s (%d course parquet files, %d subjects)",
        PROSTATE_PER_COURSE_ROOT,
        len(loaded_sources),
        combined["subject"].nunique(),
    )
    return combined, meta


def load_prostate_radiomics(
    logger: logging.Logger,
    allowed_features: set[str],
    smoke_patient_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    errors: list[str] = []
    if smoke_patient_ids:
        try:
            return load_prostate_radiomics_smoke(logger, allowed_features, smoke_patient_ids)
        except Exception as exc:
            errors.append(f"smoke_per_course_parquet: {exc}")
    for path in PROSTATE_RADIOMICS_CANDIDATES:
        if not safe_is_readable(path):
            continue
        try:
            if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"} and allowed_features:
                required = {
                    "patient_id",
                    "course_id",
                    "roi_name",
                    "roi",
                    "structure",
                    "structure_name",
                    "__roi_name__",
                    "segmentation_source",
                    "modality",
                    "image_type",
                } | set(allowed_features)
                raw = pd.read_excel(path, usecols=lambda column: str(column) in required)
            else:
                raw = load_table(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        try:
            wide, meta = extract_prostate_radiomics_wide(raw, source_path=path, allowed_features=allowed_features)
        except Exception as exc:
            errors.append(str(exc))
            continue
        logger.info("Using Prostata radiomics source: %s", path)
        return wide, meta
    raise RuntimeError(f"Failed to load Prostata radiomics: {errors}")


def parse_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def normalize_yes_no(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[text.isin({"yes", "true", "1"})] = 1.0
    out.loc[text.isin({"no", "false", "0"})] = 0.0
    return out


def derive_days(origin: pd.Series, target: pd.Series) -> pd.Series:
    delta = (target - origin).dt.days.astype("float64")
    delta.loc[delta < 0] = np.nan
    return delta


def load_odbytnice_clinical(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    clinical = load_table(ODBYTNICE_CLINICAL_CSV, dtype={"PESEL": str})
    mapping = load_table(ODBYTNICE_PESEL_MAP, dtype={"PESEL": str, "patient_id": str})
    clinical["PESEL"] = clinical["PESEL"].astype(str).str.strip()
    mapping["PESEL"] = mapping["PESEL"].astype(str).str.strip()
    mapping["patient_id"] = mapping["patient_id"].astype(str)
    merged = clinical.merge(mapping, on="PESEL", how="left", indicator=True)

    start_date = parse_date_column(merged["treatment.rcht_start_date"])
    last_follow = parse_date_column(merged["outcomes.last_followup_date"])
    death_date = parse_date_column(merged["outcomes.death_date"])
    local_recur = parse_date_column(merged["outcomes.local_recurrence_date"])
    distant_meta = parse_date_column(merged["outcomes.distant_metastasis_date"])
    ww_regrowth = parse_date_column(merged["outcomes.ww_regrowth_date"])

    os_event = death_date.notna().astype(int)
    os_exit = death_date.fillna(last_follow)
    os_days = derive_days(start_date, os_exit)

    dfs_event_flag = normalize_yes_no(merged["outcomes.dfs_event"])
    dfs_event_date = pd.concat([local_recur, distant_meta, ww_regrowth, death_date], axis=1).min(axis=1)
    dfs_exit = dfs_event_date.where(dfs_event_flag.eq(1.0), last_follow)
    dfs_days = derive_days(start_date, dfs_exit)

    pcr = normalize_yes_no(merged["outcomes.complete_pathological_response"])

    derived = merged[["patient_id", "PESEL"]].copy()
    derived["os_event"] = os_event
    derived["os_days"] = os_days
    derived["dfs_event"] = dfs_event_flag
    derived["dfs_days"] = dfs_days
    derived["pcr"] = pcr
    derived["clinical_has_mapping"] = merged["_merge"].eq("both")

    meta = {
        "odbytnice_clinical_source_path": str(ODBYTNICE_CLINICAL_CSV),
        "odbytnice_clinical_rows": int(len(clinical)),
        "odbytnice_mapping_source_path": str(ODBYTNICE_PESEL_MAP),
        "odbytnice_mapping_rows": int(len(mapping)),
        "odbytnice_mapped_patients": int(derived["patient_id"].notna().sum()),
        "odbytnice_raw_os_events": int(os_event.sum()),
        "odbytnice_raw_os_nonmissing_days": int(os_days.notna().sum()),
        "odbytnice_raw_dfs_events": int(dfs_event_flag.fillna(0).sum()),
        "odbytnice_raw_dfs_nonmissing_days": int(dfs_days.notna().sum()),
        "odbytnice_raw_pcr_positive": int(pcr.fillna(0).sum()),
        "odbytnice_raw_pcr_nonmissing": int(pcr.notna().sum()),
    }
    if ODBYTNICE_AIM3.exists():
        aim3 = pd.read_parquet(ODBYTNICE_AIM3)
        meta["odbytnice_aim3_rows"] = int(len(aim3))
        for column in ("os_event", "dfs_event", "pcr"):
            if column in aim3.columns:
                meta[f"odbytnice_aim3_{column}_nonnull"] = int(aim3[column].notna().sum())
    logger.info(
        "Loaded Odbytnice clinical rows=%d mapped=%d.",
        len(clinical),
        derived["patient_id"].notna().sum(),
    )
    return derived, meta


def load_odbytnice_radiomics(
    analysis_dir: Path, logger: logging.Logger, allowed_features: set[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = [
        analysis_dir / "data" / "odbytnice_target_radiomics.parquet",
        LOCAL_ANALYSIS_DIR / "data" / "odbytnice_target_radiomics.parquet",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Missing aggregated Odbytnice target radiomics parquet. Run "
            "aggregate_odbytnice_targets.py first."
        )
    raw = pd.read_parquet(source_path)
    if "roi_name" not in raw.columns:
        raise RuntimeError("Odbytnice target parquet is missing roi_name.")
    raw = raw.loc[raw["roi_name"].astype(str) == TARGET_ROI["Odbytnice"]].copy()
    if raw.empty:
        raise RuntimeError("Aggregated Odbytnice target parquet has no CTV1 rows.")
    raw["patient_id"] = raw["patient_id"].astype(str)
    raw["course_id"] = raw["course_id"].astype(str)
    raw["subject"] = raw["patient_id"] + "_" + raw["course_id"]
    feature_columns = collect_feature_columns(raw.columns.tolist(), allowed_features)
    if not feature_columns:
        raise RuntimeError("No radiomics feature columns detected in Odbytnice target parquet.")
    keep = raw[["patient_id", "course_id", "subject", "source_roi_name", "roi_name"] + feature_columns].copy()
    meta = {
        "odbytnice_radiomics_source_path": str(source_path),
        "odbytnice_radiomics_rows_ctv1": int(len(keep)),
        "odbytnice_radiomics_unique_patients_ctv1": int(keep["patient_id"].nunique()),
        "odbytnice_radiomics_unique_subjects_ctv1": int(keep["subject"].nunique()),
        "odbytnice_radiomics_feature_columns": int(len(feature_columns)),
    }
    logger.info("Loaded Odbytnice target radiomics from %s", source_path)
    return keep, meta


def prepare_feature_columns(df: pd.DataFrame, allowed_features: set[str]) -> list[str]:
    candidates = [column for column in df.columns if column in allowed_features]
    numeric_candidates: list[str] = []
    for column in candidates:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() >= MIN_OBS_UNIVARIATE:
            numeric_candidates.append(column)
    return sorted(numeric_candidates)


def build_prostata_toxicity_frame(
    clinical_df: pd.DataFrame,
    radiomics_df: pd.DataFrame,
    allowed_features: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    toxicity_col = pick_toxicity_column(clinical_df)
    clinical = clinical_df.copy()
    clinical["endpoint"] = pd.to_numeric(clinical[toxicity_col], errors="coerce")
    clinical = clinical.loc[clinical["endpoint"].isin([0, 1])].copy()
    merged = clinical.merge(radiomics_df, on="patient_id", how="inner")
    if merged.empty:
        raise RuntimeError("Clinical/radiomics merge yielded zero rows for Prostata toxicity.")
    feature_columns = prepare_feature_columns(merged, allowed_features)
    if not feature_columns:
        raise RuntimeError("Merged Prostata toxicity frame has no eligible radiomics features.")
    keep = merged[["patient_id", "course_id", "subject", "endpoint"] + feature_columns].copy()
    keep[feature_columns] = keep[feature_columns].apply(pd.to_numeric, errors="coerce")
    meta = {
        "leg": "prostata_toxicity_g2plus",
        "endpoint_column": toxicity_col,
        "n_rows_merged": int(len(keep)),
        "n_patients_merged": int(keep["patient_id"].nunique()),
        "n_subjects_merged": int(keep["subject"].nunique()),
        "n_events": int(keep["endpoint"].sum()),
        "n_features_after_missingness_filter": int(len(feature_columns)),
    }
    return keep, meta


def build_odbytnice_leg_frame(
    radiomics_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    allowed_features: set[str],
    leg_spec: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = radiomics_df.merge(clinical_df, on="patient_id", how="left")
    meta: dict[str, object] = {
        "leg": leg_spec["leg"],
        "raw_radiomics_subjects": int(radiomics_df["subject"].nunique()),
        "raw_radiomics_patients": int(radiomics_df["patient_id"].nunique()),
        "clinical_rows_with_mapping": int(clinical_df["patient_id"].notna().sum()),
        "subjects_missing_clinical_mapping": int(merged["PESEL"].isna().sum()),
    }

    if leg_spec["effect_model"] == "logistic":
        outcome_col = str(leg_spec["outcome_col"])
        merged = merged.loc[merged[outcome_col].isin([0.0, 1.0])].copy()
        merged[outcome_col] = merged[outcome_col].astype(int)
        event_count = int(merged[outcome_col].sum())
        meta["n_events"] = event_count
        meta["n_rows_after_endpoint_filter"] = int(len(merged))
    else:
        time_col = str(leg_spec["time_col"])
        event_col = str(leg_spec["event_col"])
        merged = merged.loc[merged[time_col].notna() & merged[event_col].isin([0.0, 1.0])].copy()
        merged[event_col] = merged[event_col].astype(int)
        event_count = int(merged[event_col].sum())
        meta["n_events"] = event_count
        meta["n_rows_after_endpoint_filter"] = int(len(merged))

    feature_columns = prepare_feature_columns(merged, allowed_features)
    meta["n_features_after_missingness_filter"] = int(len(feature_columns))
    meta["n_patients_after_endpoint_filter"] = int(merged["patient_id"].nunique())
    meta["n_subjects_after_endpoint_filter"] = int(merged["subject"].nunique())
    meta["excluded_subjects_missing_ctv1"] = int(
        clinical_df["patient_id"].notna().sum() - merged["patient_id"].nunique()
    )

    if merged.empty:
        meta["frame_ready"] = False
        meta["frame_error"] = f"No merged rows remain for {leg_spec['leg']}."
        return pd.DataFrame(), meta
    if not feature_columns:
        meta["frame_ready"] = False
        meta["frame_error"] = f"No eligible radiomics features remain for {leg_spec['leg']}."
        return pd.DataFrame(), meta

    if leg_spec["effect_model"] == "logistic":
        keep = merged[["patient_id", "course_id", "subject", str(leg_spec["outcome_col"])] + feature_columns].copy()
        keep.rename(columns={str(leg_spec["outcome_col"]): "endpoint"}, inplace=True)
    else:
        keep = merged[
            ["patient_id", "course_id", "subject", str(leg_spec["time_col"]), str(leg_spec["event_col"])]
            + feature_columns
        ].copy()
        keep.rename(
            columns={
                str(leg_spec["time_col"]): "time",
                str(leg_spec["event_col"]): "event",
            },
            inplace=True,
        )
    keep[feature_columns] = keep[feature_columns].apply(pd.to_numeric, errors="coerce")
    meta["frame_ready"] = True
    return keep, meta


def standardize_series(series: pd.Series) -> tuple[pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError("Feature has zero or non-finite variance.")
    return (numeric - float(numeric.mean())) / sd, sd


def build_failed_fit_row(
    *,
    feature_name: str,
    effect_model: str,
    n_obs: int,
    n_events: int,
    fit_status: str,
    warning_message: str,
) -> dict[str, object]:
    return {
        "feature_name": feature_name,
        "n_obs": int(n_obs),
        "n_events": int(n_events),
        "fit_status": fit_status,
        "warning_message": warning_message,
        "effect_model": effect_model,
        "selection_score": np.nan,
    }


def fit_l2_logistic_fallback(
    x: np.ndarray, y: np.ndarray, feature_name: str, logistic_c: float, reason: str
) -> dict[str, object]:
    model = LogisticRegression(
        penalty="l2",
        C=logistic_c,
        solver="liblinear",
        max_iter=2000,
        random_state=2604,
    )
    model.fit(x.reshape(-1, 1), y)
    coef = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    p_hat = model.predict_proba(x.reshape(-1, 1))[:, 1]
    weights = np.clip(p_hat * (1.0 - p_hat), 1e-8, None)
    design = np.column_stack([np.ones(len(x)), x])
    info = design.T @ (weights[:, None] * design)
    info[1, 1] += 1.0 / logistic_c
    covariance = np.linalg.pinv(info)
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    z_value = float(coef / se) if se > 0 else np.nan
    p_value = float(2.0 * sp_stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    ci_q = float(sp_stats.norm.ppf(0.975))
    ci_lo = coef - ci_q * se if np.isfinite(se) else np.nan
    ci_hi = coef + ci_q * se if np.isfinite(se) else np.nan
    return {
        "feature_name": feature_name,
        "n_obs": int(len(x)),
        "n_events": int(y.sum()),
        "log_effect": coef,
        "log_effect_se": se,
        "effect": safe_exp(coef),
        "effect_ci_lo": safe_exp(ci_lo) if np.isfinite(ci_lo) else np.nan,
        "effect_ci_hi": safe_exp(ci_hi) if np.isfinite(ci_hi) else np.nan,
        "wald_z": z_value,
        "p_value": p_value,
        "neg_log10_p": float(-np.log10(max(p_value, 1e-300))) if np.isfinite(p_value) else np.nan,
        "fit_status": "separation_warning",
        "warning_message": reason,
        "effect_model": "logistic",
        "selection_score": abs(z_value) if np.isfinite(z_value) else abs(coef),
        "log_or": coef,
        "log_or_se": se,
        "or": safe_exp(coef),
        "or_ci_lo": safe_exp(ci_lo) if np.isfinite(ci_lo) else np.nan,
        "or_ci_hi": safe_exp(ci_hi) if np.isfinite(ci_hi) else np.nan,
        "intercept": intercept,
    }


def fit_univariate_logistic(
    df: pd.DataFrame, outcome_col: str, feature_name: str, logistic_c: float
) -> dict[str, object] | None:
    sub = df[[outcome_col, feature_name]].dropna().copy()
    if len(sub) < MIN_OBS_UNIVARIATE:
        return None
    y = sub[outcome_col].astype(int).to_numpy()
    if np.unique(y).size < 2:
        return None
    try:
        x_series, _ = standardize_series(sub[feature_name])
    except Exception as exc:
        return build_failed_fit_row(
            feature_name=feature_name,
            effect_model="logistic",
            n_obs=len(sub),
            n_events=int(y.sum()),
            fit_status="failed",
            warning_message=str(exc),
        )
    x = x_series.to_numpy(dtype=float)
    design = sm.add_constant(x, has_constant="add")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.Logit(y, design)
            result = model.fit(disp=False, maxiter=200)
            coef = float(result.params[1])
            se = float(result.bse[1])
            p_value = float(result.pvalues[1])
            warning_message = "; ".join(str(item.message) for item in caught if item.message)
            if (
                warning_message
                or not np.isfinite(coef)
                or not np.isfinite(se)
                or abs(coef) > 25.0
            ):
                raise RuntimeError(warning_message or "unstable logistic coefficient")
            z_value = float(coef / se) if se > 0 else np.nan
            ci_q = float(sp_stats.norm.ppf(0.975))
            ci_lo = coef - ci_q * se
            ci_hi = coef + ci_q * se
            return {
                "feature_name": feature_name,
                "n_obs": int(len(sub)),
                "n_events": int(y.sum()),
                "log_effect": coef,
                "log_effect_se": se,
                "effect": safe_exp(coef),
                "effect_ci_lo": safe_exp(ci_lo),
                "effect_ci_hi": safe_exp(ci_hi),
                "wald_z": z_value,
                "p_value": p_value,
                "neg_log10_p": float(-np.log10(max(p_value, 1e-300))),
                "fit_status": "converged",
                "warning_message": "",
                "effect_model": "logistic",
                "selection_score": abs(z_value) if np.isfinite(z_value) else abs(coef),
                "log_or": coef,
                "log_or_se": se,
                "or": safe_exp(coef),
                "or_ci_lo": safe_exp(ci_lo),
                "or_ci_hi": safe_exp(ci_hi),
            }
        except Exception as exc:
            try:
                return fit_l2_logistic_fallback(x, y, feature_name, logistic_c, str(exc))
            except Exception as fallback_exc:
                return build_failed_fit_row(
                    feature_name=feature_name,
                    effect_model="logistic",
                    n_obs=len(sub),
                    n_events=int(y.sum()),
                    fit_status="failed",
                    warning_message=f"{exc}; fallback failed: {fallback_exc}",
                )


def fit_univariate_cox(
    df: pd.DataFrame, time_col: str, event_col: str, feature_name: str
) -> dict[str, object] | None:
    sub = df[[time_col, event_col, feature_name]].dropna().copy()
    if len(sub) < MIN_OBS_UNIVARIATE:
        return None
    sub[event_col] = sub[event_col].astype(int)
    if sub[event_col].sum() < MIN_EVENTS_SURVIVAL or sub[event_col].nunique() < 2:
        return None
    try:
        x_series, _ = standardize_series(sub[feature_name])
    except Exception as exc:
        return build_failed_fit_row(
            feature_name=feature_name,
            effect_model="cox",
            n_obs=len(sub),
            n_events=int(sub[event_col].sum()),
            fit_status="failed",
            warning_message=str(exc),
        )
    fit_df = pd.DataFrame(
        {
            "time": pd.to_numeric(sub[time_col], errors="coerce"),
            "event": sub[event_col].astype(int),
            feature_name: x_series.to_numpy(dtype=float),
        }
    ).dropna()
    if len(fit_df) < MIN_OBS_UNIVARIATE or fit_df["event"].sum() < MIN_EVENTS_SURVIVAL:
        return None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cph = CoxPHFitter(penalizer=COX_PENALIZER)
            cph.fit(fit_df, duration_col="time", event_col="event", show_progress=False)
            summary = cph.summary.loc[feature_name]
            coef = float(summary["coef"])
            se = float(summary["se(coef)"])
            p_value = float(summary["p"])
            warning_message = "; ".join(str(item.message) for item in caught if item.message)
            fit_status = "separation_warning" if warning_message else "converged"
            z_value = float(summary["z"]) if "z" in summary.index else (coef / se if se > 0 else np.nan)
            ci_lo = float(summary.get("exp(coef) lower 95%", np.nan))
            ci_hi = float(summary.get("exp(coef) upper 95%", np.nan))
            return {
                "feature_name": feature_name,
                "n_obs": int(len(fit_df)),
                "n_events": int(fit_df["event"].sum()),
                "log_effect": coef,
                "log_effect_se": se,
                "effect": safe_exp(coef),
                "effect_ci_lo": ci_lo,
                "effect_ci_hi": ci_hi,
                "wald_z": z_value,
                "p_value": p_value,
                "neg_log10_p": float(-np.log10(max(p_value, 1e-300))),
                "fit_status": fit_status,
                "warning_message": warning_message,
                "effect_model": "cox",
                "selection_score": abs(z_value) if np.isfinite(z_value) else abs(coef),
                "log_hr": coef,
                "log_hr_se": se,
                "hr": safe_exp(coef),
                "hr_ci_lo": ci_lo,
                "hr_ci_hi": ci_hi,
            }
        except Exception as exc:
            return build_failed_fit_row(
                feature_name=feature_name,
                effect_model="cox",
                n_obs=len(fit_df),
                n_events=int(fit_df["event"].sum()),
                fit_status="failed",
                warning_message=str(exc),
            )


def run_univariate_for_leg(
    frame_df: pd.DataFrame,
    leg_spec: dict[str, object],
    icc_df: pd.DataFrame,
    shared_feature_universe: set[str],
    logistic_c: float,
) -> pd.DataFrame:
    class_map = icc_df.set_index("feature_name")["robustness_class"].to_dict()
    family_map = icc_df.set_index("feature_name")["feature_family"].to_dict()
    feature_columns = [column for column in frame_df.columns if column in shared_feature_universe]
    rows: list[dict[str, object]] = []

    for feature_name in feature_columns:
        if leg_spec["effect_model"] == "logistic":
            result = fit_univariate_logistic(frame_df, "endpoint", feature_name, logistic_c)
        else:
            result = fit_univariate_cox(frame_df, "time", "event", feature_name)
        if result is None:
            continue
        result["leg"] = leg_spec["leg"]
        result["cohort"] = leg_spec["cohort"]
        result["roi_name"] = leg_spec["roi_name"]
        result["robustness_class"] = class_map.get(feature_name, "Poor")
        result["feature_family"] = family_map.get(feature_name, infer_feature_family(feature_name))
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
    z_value = float(sp_stats.norm.ppf(0.975))
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


def pooled_meta_summary(univar_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = univar_df.loc[
        univar_df["log_effect"].notna()
        & univar_df["log_effect_se"].notna()
        & (univar_df["log_effect_se"] > 0)
    ].copy()
    usable["variance"] = usable["log_effect_se"] ** 2

    class_rows: list[dict[str, object]] = []
    class_family_rows: list[dict[str, object]] = []
    cross_leg_rows: list[dict[str, object]] = []

    for leg_name, leg_block in usable.groupby("leg", sort=False):
        for class_name in CLASS_ORDER:
            block = leg_block.loc[leg_block["robustness_class"] == class_name].copy()
            if len(block) < 2:
                continue
            meta = dersimonian_laird(
                block["log_effect"].to_numpy(dtype=float),
                block["variance"].to_numpy(dtype=float),
            )
            meta.update(
                {
                    "leg": leg_name,
                    "effect_model": block["effect_model"].iloc[0],
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
                    sub["log_effect"].to_numpy(dtype=float),
                    sub["variance"].to_numpy(dtype=float),
                )
                fam_meta.update(
                    {
                        "leg": leg_name,
                        "effect_model": block["effect_model"].iloc[0],
                        "robustness_class": class_name,
                        "feature_family": family_name,
                        "n_features": int(len(sub)),
                    }
                )
                class_family_rows.append(fam_meta)

    for effect_model, effect_block in usable.groupby("effect_model", sort=False):
        for class_name in CLASS_ORDER:
            block = effect_block.loc[effect_block["robustness_class"] == class_name].copy()
            if len(block) < 2:
                continue
            meta = dersimonian_laird(
                block["log_effect"].to_numpy(dtype=float),
                block["variance"].to_numpy(dtype=float),
            )
            meta.update(
                {
                    "effect_model": effect_model,
                    "robustness_class": class_name,
                    "n_features": int(len(block)),
                    "n_legs": int(block["leg"].nunique()),
                    "legs": "|".join(sorted(block["leg"].astype(str).unique().tolist())),
                }
            )
            cross_leg_rows.append(meta)

    return (
        pd.DataFrame(class_rows),
        pd.DataFrame(class_family_rows),
        pd.DataFrame(cross_leg_rows),
    )


def ks_class_summary(univar_df: pd.DataFrame) -> pd.DataFrame:
    values = univar_df[["leg", "robustness_class", "neg_log10_p"]].dropna().copy()
    rows: list[dict[str, object]] = []
    for leg_name, leg_block in values.groupby("leg", sort=False):
        for class_name in CLASS_ORDER:
            block = leg_block.loc[leg_block["robustness_class"] == class_name, "neg_log10_p"].to_numpy(dtype=float)
            if block.size == 0:
                continue
            rows.append(
                {
                    "leg": leg_name,
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
            a = leg_block.loc[leg_block["robustness_class"] == first, "neg_log10_p"].to_numpy(dtype=float)
            b = leg_block.loc[leg_block["robustness_class"] == second, "neg_log10_p"].to_numpy(dtype=float)
            if len(a) == 0 or len(b) == 0:
                continue
            ks_stat, p_value = sp_stats.ks_2samp(a, b, alternative="two-sided", mode="auto")
            rows.append(
                {
                    "leg": leg_name,
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


def build_outer_cv(event_values: np.ndarray, outer_folds: int, seed: int) -> StratifiedKFold:
    counts = np.bincount(event_values.astype(int))
    min_class = int(np.min(counts)) if len(counts) > 1 else 0
    if min_class < 2:
        raise RuntimeError("Outcome lacks enough members for stratified outer CV.")
    n_splits = min(outer_folds, min_class)
    if n_splits < 2:
        raise RuntimeError("Need at least two stratified outer folds.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fit_panel_logistic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    logistic_c: float,
    seed: int,
) -> tuple[dict[str, object], str]:
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=logistic_c,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    y_train = train_df["endpoint"].to_numpy(dtype=int)
    y_test = test_df["endpoint"].to_numpy(dtype=int)
    estimator.fit(train_df[selected_features], y_train)
    prob = estimator.predict_proba(test_df[selected_features])[:, 1]
    if len(np.unique(y_test)) < 2:
        raise RuntimeError("Test fold lacks both outcome classes for AUC.")
    return {
        "metric_name": "auc",
        "score": float(roc_auc_score(y_test, prob)),
        "brier": float(brier_score_loss(y_test, prob)),
    }, "converged"


def fit_panel_cox(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[dict[str, object], str]:
    train_x = train_df[selected_features].copy()
    test_x = test_df[selected_features].copy()
    medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    means = train_x.mean()
    sds = train_x.std(ddof=0).replace(0.0, 1.0)
    train_x = (train_x - means) / sds
    test_x = (test_x - means) / sds

    fit_train = train_x.copy()
    fit_train["time"] = train_df["time"].to_numpy(dtype=float)
    fit_train["event"] = train_df["event"].to_numpy(dtype=int)
    fit_test = test_x.copy()
    fit_test["time"] = test_df["time"].to_numpy(dtype=float)
    fit_test["event"] = test_df["event"].to_numpy(dtype=int)

    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(fit_train, duration_col="time", event_col="event", show_progress=False)
    risk = cph.predict_partial_hazard(fit_test[selected_features]).to_numpy().reshape(-1)
    if fit_test["event"].sum() < MIN_TEST_EVENTS_SURVIVAL:
        raise RuntimeError("Test fold lacks events for C-index.")
    c_index = concordance_index(
        fit_test["time"].to_numpy(dtype=float),
        -risk,
        fit_test["event"].to_numpy(dtype=int),
    )
    return {"metric_name": "c_index", "score": float(c_index), "brier": np.nan}, "converged"


def nested_panel_compare(
    frame_df: pd.DataFrame,
    leg_spec: dict[str, object],
    icc_df: pd.DataFrame,
    shared_feature_universe: set[str],
    outer_folds: int,
    prescreen_k: int,
    seed: int,
    logistic_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_features = {
        class_name: sorted(
            feature
            for feature in icc_df.loc[icc_df["robustness_class"] == class_name, "feature_name"].astype(str)
            if feature in shared_feature_universe
        )
        for class_name in CLASS_ORDER
    }
    event_values = (
        frame_df["endpoint"].to_numpy(dtype=int)
        if leg_spec["effect_model"] == "logistic"
        else frame_df["event"].to_numpy(dtype=int)
    )
    splitter = build_outer_cv(event_values, outer_folds=outer_folds, seed=seed)

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for class_name in CLASS_ORDER:
        features = class_features[class_name]
        if not features:
            continue
        class_fold_rows: list[dict[str, object]] = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(frame_df, event_values), start=1):
            train_df = frame_df.iloc[train_idx].copy()
            test_df = frame_df.iloc[test_idx].copy()
            train_events = int(train_df["endpoint"].sum()) if leg_spec["effect_model"] == "logistic" else int(train_df["event"].sum())
            event_budget = max(1, train_events // 10)
            prescreen_rows: list[dict[str, object]] = []
            for feature_name in features:
                if leg_spec["effect_model"] == "logistic":
                    result = fit_univariate_logistic(train_df, "endpoint", feature_name, logistic_c)
                else:
                    result = fit_univariate_cox(train_df, "time", "event", feature_name)
                if result is not None and np.isfinite(result.get("selection_score", np.nan)):
                    prescreen_rows.append(result)
            if not prescreen_rows:
                continue
            prescreen_df = pd.DataFrame(prescreen_rows).sort_values("selection_score", ascending=False, kind="stable")
            selected_k = min(prescreen_k, event_budget, len(prescreen_df))
            selected_features = prescreen_df["feature_name"].head(selected_k).tolist()
            if not selected_features:
                continue
            try:
                if leg_spec["effect_model"] == "logistic":
                    metrics, fit_status = fit_panel_logistic(train_df, test_df, selected_features, logistic_c, seed)
                else:
                    metrics, fit_status = fit_panel_cox(train_df, test_df, selected_features)
            except Exception as exc:
                fold_rows.append(
                    {
                        "leg": leg_spec["leg"],
                        "robustness_class": class_name,
                        "fold_id": fold_id,
                        "train_events": train_events,
                        "selected_k": int(len(selected_features)),
                        "event_budget_train": int(event_budget),
                        "metric_name": "auc" if leg_spec["effect_model"] == "logistic" else "c_index",
                        "score": np.nan,
                        "brier": np.nan,
                        "fit_status": "failed",
                        "warning_message": str(exc),
                        "selected_features": "|".join(selected_features),
                    }
                )
                continue

            row = {
                "leg": leg_spec["leg"],
                "robustness_class": class_name,
                "fold_id": fold_id,
                "train_events": train_events,
                "selected_k": int(len(selected_features)),
                "event_budget_train": int(event_budget),
                "metric_name": metrics["metric_name"],
                "score": metrics["score"],
                "brier": metrics["brier"],
                "fit_status": fit_status,
                "warning_message": "",
                "selected_features": "|".join(selected_features),
            }
            fold_rows.append(row)
            class_fold_rows.append(row)

        if not class_fold_rows:
            continue
        block = pd.DataFrame(class_fold_rows)
        feature_counter = Counter()
        for blob in block["selected_features"].astype(str).tolist():
            feature_counter.update(token for token in blob.split("|") if token)
        summary_rows.append(
            {
                "leg": leg_spec["leg"],
                "robustness_class": class_name,
                "metric_name": block["metric_name"].iloc[0],
                "n_folds": int(len(block)),
                "mean_score": float(block["score"].mean()),
                "median_score": float(block["score"].median()),
                "sd_score": float(block["score"].std(ddof=0)),
                "mean_brier": float(block["brier"].dropna().mean()) if block["brier"].notna().any() else np.nan,
                "mean_selected_k": float(block["selected_k"].mean()),
                "top_selected_features": "|".join(feature for feature, _ in feature_counter.most_common(10)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def build_shared_feature_universe(
    frames_by_leg: dict[str, pd.DataFrame],
    specs_by_leg: dict[str, dict[str, object]],
    icc_by_leg: dict[str, pd.DataFrame],
    smoke_test: bool,
    smoke_max_features: int,
) -> dict[str, set[str]]:
    feature_sets_by_model: defaultdict[str, list[set[str]]] = defaultdict(list)
    for leg_name, frame_df in frames_by_leg.items():
        feature_columns = {
            column
            for column in frame_df.columns
            if column not in {"patient_id", "course_id", "subject", "endpoint", "time", "event"}
        }
        icc_features = set(icc_by_leg[leg_name]["feature_name"].astype(str))
        feature_sets_by_model[str(specs_by_leg[leg_name]["effect_model"])].append(feature_columns & icc_features)

    shared: dict[str, set[str]] = {}
    for effect_model, feature_sets in feature_sets_by_model.items():
        shared_features = set.intersection(*feature_sets) if feature_sets else set()
        if smoke_test and len(shared_features) > smoke_max_features:
            shared_features = set(sorted(shared_features)[:smoke_max_features])
        shared[effect_model] = shared_features
    return shared


def build_manifest_base(job_id: str, args: argparse.Namespace, analysis_dir: Path) -> dict[str, object]:
    return {
        "job_id": job_id,
        "hostname": socket.gethostname(),
        "started_at_epoch": time.time(),
        "analysis_dir": str(analysis_dir),
        "git_sha": resolve_git_sha(),
        "script_path": str(Path(__file__).resolve()),
        "production_legs": PRODUCTION_LEGS,
        "target_roi": TARGET_ROI,
        "arguments": {
            "seed": int(args.seed),
            "outer_folds": int(args.outer_folds),
            "prescreen_k": int(args.prescreen_k),
            "smoke_test": bool(args.smoke_test),
            "smoke_max_features": int(args.smoke_max_features),
            "logistic_c": float(args.logistic_c),
            "allow_leg_failure": bool(args.allow_leg_failure),
            "fetch_icc_from_argos": bool(args.fetch_icc_from_argos),
        },
        "dropped_legs": {
            "prostata_os": "Dropped because no observed last-follow-up/censoring field is available in the audited accessible Prostata sources; synthetic defaults remain prohibited."
        },
        "notes": {
            "odbytnice_time_origin": "Derived from treatment.rcht_start_date to observed death/follow-up or DFS event dates, matching the available clinical date structure.",
            "odbytnice_target_policy": "CTV1 only; no CTV2/GTVp/PTV fallback is permitted.",
        },
    }


def main() -> None:
    args = parse_args()
    analysis_dir = resolve_analysis_dir(args.analysis_dir)
    data_dir = analysis_dir / "data"
    tables_dir = analysis_dir / "tables"
    logs_dir = analysis_dir / "logs"
    for path in (data_dir, tables_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    out_features = data_dir / "prognostic_retention_feature_effects.parquet"
    out_class_summary = tables_dir / "prognostic_retention_class_summary.csv"
    out_class_family = tables_dir / "prognostic_retention_class_family_summary.csv"
    out_cross_leg = tables_dir / "prognostic_retention_cross_leg_summary.csv"
    out_ks = tables_dir / "prognostic_retention_ks.csv"
    out_panel = tables_dir / "prognostic_retention_panel_cv.csv"
    out_panel_folds = tables_dir / "prognostic_retention_panel_cv_folds.csv"
    manifest_path = logs_dir / f"41_manifest_{args.job_id}.json"

    logger = configure_logging(args.job_id)
    manifest = build_manifest_base(args.job_id, args, analysis_dir)
    blockers_all: list[dict[str, object]] = []

    try:
        icc_path = resolve_icc_path(analysis_dir, args.fetch_icc_from_argos, logger)
        icc_raw = pd.read_parquet(icc_path)
        manifest["icc_source_path"] = str(icc_path)
        manifest["icc_source_sha256"] = sha256_file(icc_path)
        manifest["icc_source_mtime_epoch"] = icc_path.stat().st_mtime

        icc_by_leg: dict[str, pd.DataFrame] = {}
        missing_icc_legs: dict[str, str] = {}
        for leg_spec in LEG_SPECS:
            leg_name = str(leg_spec["leg"])
            try:
                icc_df, icc_meta = standardize_icc_table(
                    icc_raw,
                    cohort=str(leg_spec["cohort"]),
                    roi=str(leg_spec["roi_name"]),
                )
                icc_by_leg[leg_name] = icc_df
                manifest.setdefault("icc_filters", {})[leg_name] = icc_meta
            except Exception as exc:
                missing_icc_legs[leg_name] = str(exc)
                manifest.setdefault("legs", {}).setdefault(leg_name, {})
                manifest["legs"][leg_name]["blocker"] = {
                    "stage": "icc_lookup",
                    "error": str(exc),
                }

        if "prostata_toxicity_g2plus" not in icc_by_leg:
            raise RuntimeError(
                "Prostata ICC rows are unavailable, which blocks the mandatory smoke/prod leg."
            )
        if missing_icc_legs and not args.allow_leg_failure:
            joined = "; ".join(f"{leg}: {reason}" for leg, reason in missing_icc_legs.items())
            raise RuntimeError(f"Required leg ICC lookup failed: {joined}")

        prostata_clinical, prostata_clinical_meta = resolve_prostate_clinical(logger)
        manifest["prostata_clinical"] = prostata_clinical_meta
        prostata_smoke_patient_ids: set[str] | None = None
        if args.smoke_test:
            prostata_smoke_patient_ids, smoke_meta = pick_prostate_smoke_patient_ids(prostata_clinical)
            manifest["prostata_smoke_sampling"] = smoke_meta
        prostata_allowed_features = sorted(set(icc_by_leg["prostata_toxicity_g2plus"]["feature_name"].astype(str)))
        if args.smoke_test:
            preload_cap = max(args.smoke_max_features * 3, 30)
            prostata_allowed_features = prostata_allowed_features[:preload_cap]
        prostata_radiomics, prostata_radiomics_meta = load_prostate_radiomics(
            logger,
            allowed_features=set(prostata_allowed_features),
            smoke_patient_ids=prostata_smoke_patient_ids,
        )
        manifest["prostata_radiomics"] = prostata_radiomics_meta

        odbytnice_clinical, odbytnice_clinical_meta = load_odbytnice_clinical(logger)
        manifest["odbytnice_clinical"] = odbytnice_clinical_meta
        odbytnice_feature_pool: set[str] = set()
        for leg_name in ("odbytnice_os", "odbytnice_dfs", "odbytnice_pcr"):
            if leg_name in icc_by_leg:
                odbytnice_feature_pool.update(icc_by_leg[leg_name]["feature_name"].astype(str))
        odbytnice_radiomics, odbytnice_radiomics_meta = load_odbytnice_radiomics(
            analysis_dir=analysis_dir,
            logger=logger,
            allowed_features=odbytnice_feature_pool,
        )
        if not odbytnice_feature_pool:
            odbytnice_radiomics_meta["odbytnice_icc_feature_pool_missing"] = True
            odbytnice_radiomics_meta["odbytnice_icc_feature_pool_reason"] = (
                "No Odbytnice ICC rows were available for CTV1 in the current artifact."
            )
        manifest["odbytnice_radiomics"] = odbytnice_radiomics_meta
        odbytnice_fallback_features = set(
            collect_feature_columns(odbytnice_radiomics.columns.tolist(), allowed_features=set())
        )

        frames_by_leg: dict[str, pd.DataFrame] = {}
        specs_by_leg = {str(spec["leg"]): spec for spec in LEG_SPECS}

        prostata_frame, prostata_meta = build_prostata_toxicity_frame(
            clinical_df=prostata_clinical,
            radiomics_df=prostata_radiomics,
            allowed_features=set(icc_by_leg["prostata_toxicity_g2plus"]["feature_name"].astype(str)),
        )
        frames_by_leg["prostata_toxicity_g2plus"] = prostata_frame
        manifest.setdefault("legs", {})["prostata_toxicity_g2plus"] = prostata_meta

        for leg_name in ("odbytnice_os", "odbytnice_dfs", "odbytnice_pcr"):
            allowed_features = (
                set(icc_by_leg[leg_name]["feature_name"].astype(str))
                if leg_name in icc_by_leg
                else odbytnice_fallback_features
            )
            leg_frame, leg_meta = build_odbytnice_leg_frame(
                radiomics_df=odbytnice_radiomics,
                clinical_df=odbytnice_clinical,
                allowed_features=allowed_features,
                leg_spec=specs_by_leg[leg_name],
            )
            manifest.setdefault("legs", {})[leg_name] = leg_meta
            if leg_name not in icc_by_leg:
                blockers_all.append(
                    {
                        "leg": leg_name,
                        "error": f"ICC lookup failed: {missing_icc_legs[leg_name]}",
                        "traceback": "",
                    }
                )
                manifest["legs"][leg_name]["blocker"] = {
                    "stage": "icc_lookup",
                    "error": missing_icc_legs[leg_name],
                }
                continue
            if leg_frame.empty:
                blocker_error = str(leg_meta.get("frame_error", "No model-ready rows remain."))
                blocker = {
                    "leg": leg_name,
                    "error": blocker_error,
                    "traceback": "",
                }
                blockers_all.append(blocker)
                manifest["legs"][leg_name]["blocker"] = {
                    "stage": "frame_build",
                    "error": blocker_error,
                }
                if not args.allow_leg_failure:
                    raise RuntimeError(blocker_error)
                continue
            frames_by_leg[leg_name] = leg_frame

        shared_features = build_shared_feature_universe(
            frames_by_leg=frames_by_leg,
            specs_by_leg=specs_by_leg,
            icc_by_leg=icc_by_leg,
            smoke_test=args.smoke_test,
            smoke_max_features=args.smoke_max_features,
        )
        manifest["shared_feature_universe"] = {
            effect_model: {
                "feature_count": int(len(features)),
                "feature_names_sample": sorted(features)[:20],
            }
            for effect_model, features in shared_features.items()
        }

        all_univar: list[pd.DataFrame] = []
        all_panel: list[pd.DataFrame] = []
        all_panel_folds: list[pd.DataFrame] = []

        for leg_name in PRODUCTION_LEGS:
            leg_spec = specs_by_leg[leg_name]
            if leg_name not in frames_by_leg or leg_name not in icc_by_leg:
                logger.warning("Skipping leg %s because required frame/ICC inputs are unavailable.", leg_name)
                continue
            try:
                univar_df = run_univariate_for_leg(
                    frame_df=frames_by_leg[leg_name],
                    leg_spec=leg_spec,
                    icc_df=icc_by_leg[leg_name],
                    shared_feature_universe=shared_features[str(leg_spec["effect_model"])],
                    logistic_c=args.logistic_c,
                )
                if univar_df.empty:
                    raise RuntimeError("No valid univariate fits were produced.")

                if leg_spec["effect_model"] == "logistic":
                    attempted = univar_df["fit_status"].isin(["converged", "separation_warning", "failed"]).sum()
                    separation = univar_df["fit_status"].eq("separation_warning").sum()
                    if attempted and separation / attempted > 0.5:
                        raise RuntimeError(
                            f"More than 50% of logistic features hit separation warnings ({separation}/{attempted})."
                        )

                panel_df, panel_folds_df = nested_panel_compare(
                    frame_df=frames_by_leg[leg_name],
                    leg_spec=leg_spec,
                    icc_df=icc_by_leg[leg_name],
                    shared_feature_universe=shared_features[str(leg_spec["effect_model"])],
                    outer_folds=min(args.outer_folds, 3) if args.smoke_test else args.outer_folds,
                    prescreen_k=min(args.prescreen_k, 8) if args.smoke_test else args.prescreen_k,
                    seed=args.seed,
                    logistic_c=args.logistic_c,
                )
                if panel_df.empty:
                    raise RuntimeError("Nested CV panel comparison produced no valid folds.")

                manifest["legs"][leg_name]["univariate_rows"] = int(len(univar_df))
                manifest["legs"][leg_name]["panel_rows"] = int(len(panel_df))
                manifest["legs"][leg_name]["panel_fold_rows"] = int(len(panel_folds_df))
                manifest["legs"][leg_name]["fit_status_counts"] = {
                    str(key): int(value)
                    for key, value in Counter(univar_df["fit_status"].astype(str).tolist()).items()
                }
                all_univar.append(univar_df)
                all_panel.append(panel_df)
                all_panel_folds.append(panel_folds_df)
            except Exception as exc:
                blocker = {
                    "leg": leg_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                blockers_all.append(blocker)
                manifest["legs"].setdefault(leg_name, {})
                manifest["legs"][leg_name]["blocker"] = blocker
                logger.error("Leg %s failed: %s", leg_name, exc)
                if not args.allow_leg_failure:
                    raise

        if not all_univar:
            raise RuntimeError("No production leg completed successfully.")

        univar_df = pd.concat(all_univar, ignore_index=True)
        panel_df = pd.concat(all_panel, ignore_index=True) if all_panel else pd.DataFrame()
        panel_fold_df = pd.concat(all_panel_folds, ignore_index=True) if all_panel_folds else pd.DataFrame()

        class_summary_df, class_family_df, cross_leg_df = pooled_meta_summary(univar_df)
        ks_df = ks_class_summary(univar_df)

        univar_df.to_parquet(out_features, index=False)
        class_summary_df.to_csv(out_class_summary, index=False)
        class_family_df.to_csv(out_class_family, index=False)
        cross_leg_df.to_csv(out_cross_leg, index=False)
        ks_df.to_csv(out_ks, index=False)
        panel_df.to_csv(out_panel, index=False)
        panel_fold_df.to_csv(out_panel_folds, index=False)

        manifest["outputs"] = {
            "feature_effects_parquet": str(out_features),
            "class_summary_csv": str(out_class_summary),
            "class_family_summary_csv": str(out_class_family),
            "cross_leg_summary_csv": str(out_cross_leg),
            "ks_csv": str(out_ks),
            "panel_summary_csv": str(out_panel),
            "panel_folds_csv": str(out_panel_folds),
        }
        manifest["result_counts"] = {
            "feature_effect_rows": int(len(univar_df)),
            "class_summary_rows": int(len(class_summary_df)),
            "class_family_summary_rows": int(len(class_family_df)),
            "cross_leg_summary_rows": int(len(cross_leg_df)),
            "ks_rows": int(len(ks_df)),
            "panel_summary_rows": int(len(panel_df)),
            "panel_fold_rows": int(len(panel_fold_df)),
        }
        manifest["blockers"] = blockers_all
        manifest["completed"] = len(blockers_all) == 0 or args.allow_leg_failure
        manifest["ended_at_epoch"] = time.time()
        write_manifest(manifest_path, manifest)
        logger.info("Step 41 complete. Manifest: %s", manifest_path)
    except Exception as exc:
        blockers_all.append(
            {
                "leg": "script",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        manifest["blockers"] = blockers_all
        manifest["completed"] = False
        manifest["ended_at_epoch"] = time.time()
        write_manifest(manifest_path, manifest)
        logger.error("Step 41 failed: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
