#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 21 fixed: inter-algorithm AI comparison with hard scope lock, head-task
gate, complete 6-realization auditing, mixed-model contour variance outputs,
and locked report filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sp_stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

MANUSCRIPT_ROOT = Path("/home/kgs24/rtpipeline_manuscript")
ANALYSIS_DIR = MANUSCRIPT_ROOT / "analysis"

ALLOWED_COHORTS = {"Hipokampy", "PlucaRCHT"}
COHORT_ORDER = ("Hipokampy", "PlucaRCHT")
COHORT_REGION = {
    "Hipokampy": "Brain",
    "PlucaRCHT": "Thorax",
}

SEG_SRC_CUSTOM = "Custom"
SEG_SRC_THORAX = "AutoRTS_total"
PERTURBATION_MAP = {
    "ntcv_v0": "v0",
    "ntcv_c1_v0": "c1_v0",
    "ntcv_c2_v0": "c2_v0",
}
MODEL_COLUMNS = (
    "TS_v0",
    "TS_c1_v0",
    "TS_c2_v0",
    "Custom_v0",
    "Custom_c1_v0",
    "Custom_c2_v0",
)
TS_COLUMNS = MODEL_COLUMNS[:3]
CUSTOM_COLUMNS = MODEL_COLUMNS[3:]
ALGORITHM_BY_COLUMN = np.array(["TS", "TS", "TS", "Custom", "Custom", "Custom"])
REALIZATION_BY_COLUMN = np.array(["v0", "c1_v0", "c2_v0", "v0", "c1_v0", "c2_v0"])

FAMILIES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")

MIN_SUBJECTS_FOR_ICC = 5
ZERO_VAR_EPS = 1e-12
FISHER_Z_CLIP = 0.999999
BOOTSTRAP_SEED = 2604


@dataclass(frozen=True)
class CourseRecord:
    cohort: str
    body_region: str
    patient_id: str
    course_id: str
    parquet: str
    size_bytes: int
    ts_source: str
    segmentation_sources: tuple[str, ...]


@dataclass(frozen=True)
class FeaturePayload:
    cohort: str
    body_region: str
    roi_name: str
    feature_name: str
    feature_family: str
    image_type: str
    subjects: tuple[str, ...]
    matrix: np.ndarray


def get_feature_family(feature_name: str) -> str:
    lname = str(feature_name).lower()
    for family in FAMILIES:
        if f"_{family}_" in lname:
            return family
    return "unknown"


def get_image_type(feature_name: str) -> str:
    feature_name = str(feature_name)
    if feature_name.startswith("original_"):
        return "original"
    if feature_name.startswith("log-sigma-"):
        return "log-sigma"
    if feature_name.startswith("wavelet-"):
        return feature_name.split("_", 1)[0]
    return "other"


def fisher_z(value: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.arctanh(np.clip(arr, -FISHER_Z_CLIP, FISHER_Z_CLIP))


def is_head_task_source(source: str) -> bool:
    lowered = str(source).strip().lower()
    return lowered.startswith("autorts") and "head" in lowered


def file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed step 21 inter-algorithm AI comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cohort",
        default=None,
        help="Single cohort to process. Allowed: Hipokampy or PlucaRCHT.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument("--roi-name", type=str, default=None)
    parser.add_argument("--feature-name", type=str, default=None)
    parser.add_argument(
        "--subject-list",
        type=str,
        default=None,
        help="Optional comma-separated subject list for targeted smoke testing.",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional smoke-test limiter after filtering and sorting ROI names.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional smoke-test limiter after filtering and sorting feature names.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional smoke-test limiter after filtering and sorting subject IDs.",
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def icc31_from_matrix(matrix: np.ndarray) -> tuple[float, float, float] | None:
    if matrix.ndim != 2:
        return None
    n_subjects, k_raters = matrix.shape
    if n_subjects < MIN_SUBJECTS_FOR_ICC or k_raters < 2:
        return None
    if not np.all(np.isfinite(matrix)):
        return None

    grand_mean = matrix.mean()
    subject_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)

    ss_subject = k_raters * ((subject_means - grand_mean) ** 2).sum()
    ss_rater = n_subjects * ((rater_means - grand_mean) ** 2).sum()
    ss_total = ((matrix - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_subject - ss_rater

    df_subject = n_subjects - 1
    df_error = (n_subjects - 1) * (k_raters - 1)
    if df_subject <= 0 or df_error <= 0:
        return None

    bms = ss_subject / df_subject
    ems = ss_error / df_error
    denom = bms + (k_raters - 1) * ems
    if denom == 0 or not np.isfinite(denom):
        return None

    icc = (bms - ems) / denom
    ci_low = np.nan
    ci_high = np.nan

    if np.isfinite(ems) and ems > 0:
        f_stat = bms / ems
        if np.isfinite(f_stat) and f_stat > 0:
            f_low = f_stat / sp_stats.f.ppf(0.975, df_subject, df_error)
            f_high = f_stat * sp_stats.f.ppf(0.975, df_error, df_subject)
            ci_low = (f_low - 1.0) / (f_low + k_raters - 1.0)
            ci_high = (f_high - 1.0) / (f_high + k_raters - 1.0)

    return (float(icc), float(ci_low), float(ci_high))


def icc21_absolute_agreement_from_matrix(
    matrix: np.ndarray,
) -> tuple[float, float, float] | None:
    if matrix.ndim != 2:
        return None
    n_subjects, k_raters = matrix.shape
    if n_subjects < MIN_SUBJECTS_FOR_ICC or k_raters < 2:
        return None
    if not np.all(np.isfinite(matrix)):
        return None

    grand_mean = matrix.mean()
    subject_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)

    ss_subject = k_raters * ((subject_means - grand_mean) ** 2).sum()
    ss_rater = n_subjects * ((rater_means - grand_mean) ** 2).sum()
    ss_total = ((matrix - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_subject - ss_rater

    df_subject = n_subjects - 1
    df_rater = k_raters - 1
    df_error = df_subject * df_rater
    if df_subject <= 0 or df_rater <= 0 or df_error <= 0:
        return None

    bms = ss_subject / df_subject
    jms = ss_rater / df_rater
    ems = ss_error / df_error
    denom = bms + (k_raters - 1.0) * ems + k_raters * (jms - ems) / n_subjects
    if denom == 0 or not np.isfinite(denom):
        return None

    icc = (bms - ems) / denom
    ci_low = np.nan
    ci_high = np.nan

    if np.isfinite(ems) and ems > 0:
        try:
            fj = jms / ems
            vn = df_error * (
                k_raters * icc * fj
                + n_subjects * (1.0 + (k_raters - 1.0) * icc)
                - k_raters * icc
            ) ** 2
            vd = (
                df_subject * (k_raters**2) * (icc**2) * (fj**2)
                + (
                    n_subjects * (1.0 + (k_raters - 1.0) * icc)
                    - k_raters * icc
                )
                ** 2
            )
            if vd > 0 and np.isfinite(vd):
                v = vn / vd
                f_upper = sp_stats.f.ppf(0.975, df_subject, v)
                f_lower = sp_stats.f.ppf(0.975, v, df_subject)
                denom_low = (
                    f_upper
                    * (k_raters * jms + (k_raters * n_subjects - k_raters - n_subjects) * ems)
                    + n_subjects * bms
                )
                denom_high = (
                    k_raters * jms
                    + (k_raters * n_subjects - k_raters - n_subjects) * ems
                    + n_subjects * f_lower * bms
                )
                if denom_low != 0 and denom_high != 0:
                    ci_low = n_subjects * (bms - f_upper * ems) / denom_low
                    ci_high = n_subjects * (f_lower * bms - ems) / denom_high
        except Exception:
            ci_low = np.nan
            ci_high = np.nan

    return (
        float(icc),
        float(np.clip(ci_low, -1.0, 1.0) if np.isfinite(ci_low) else np.nan),
        float(np.clip(ci_high, -1.0, 1.0) if np.isfinite(ci_high) else np.nan),
    )


def algorithm_mean_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.mean(matrix[:, :3], axis=1),
            np.mean(matrix[:, 3:], axis=1),
        ]
    )


def residualize_algorithm_main_effect(matrix: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(matrix, dtype=float).copy()
    grand_mean = adjusted.mean()
    for cols in (slice(0, 3), slice(3, 6)):
        alg_mean = adjusted[:, cols].mean()
        adjusted[:, cols] = adjusted[:, cols] - alg_mean + grand_mean
    return adjusted


def fast_variance_components_from_matrix(matrix: np.ndarray) -> dict[str, float] | None:
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        return None
    if matrix.shape[0] < MIN_SUBJECTS_FOR_ICC or not np.all(np.isfinite(matrix)):
        return None

    adjusted = residualize_algorithm_main_effect(matrix)
    n_subjects, k_levels = adjusted.shape
    grand_mean = adjusted.mean()
    subject_means = adjusted.mean(axis=1)
    level_means = adjusted.mean(axis=0)

    ss_subject = k_levels * ((subject_means - grand_mean) ** 2).sum()
    ss_level = n_subjects * ((level_means - grand_mean) ** 2).sum()
    ss_total = ((adjusted - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_subject - ss_level

    df_subject = n_subjects - 1
    df_level = k_levels - 1
    df_error = df_subject * df_level
    if df_subject <= 0 or df_level <= 0 or df_error <= 0:
        return None

    ms_subject = ss_subject / df_subject
    ms_level = ss_level / df_level
    ms_error = ss_error / df_error

    sigma_subject = max((ms_subject - ms_error) / k_levels, 0.0)
    sigma_contour = max((ms_level - ms_error) / n_subjects, 0.0)
    sigma_residual = max(ms_error, 0.0)
    total = sigma_subject + sigma_contour + sigma_residual
    if total <= 0:
        return None

    return {
        "sigma_subject_var": float(sigma_subject),
        "sigma_contour_var": float(sigma_contour),
        "sigma_residual_var": float(sigma_residual),
        "subject_pct_total": float(sigma_subject / total),
        "contour_pct_total": float(sigma_contour / total),
        "residual_pct_total": float(sigma_residual / total),
        "variance_method": "anova_mom_fallback",
        "mixedlm_converged": False,
    }


def fit_mixedlm_variance_components(
    matrix: np.ndarray,
    subjects: tuple[str, ...],
) -> tuple[dict[str, float], str | None]:
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        return {}, "matrix_not_6_columns"
    if matrix.shape[0] < MIN_SUBJECTS_FOR_ICC:
        return {}, "insufficient_subjects"
    if not np.all(np.isfinite(matrix)):
        return {}, "nonfinite_matrix"

    long_df = pd.DataFrame(
        {
            "subject": np.repeat(np.asarray(subjects, dtype=object), len(MODEL_COLUMNS)),
            "algorithm": np.tile(ALGORITHM_BY_COLUMN, len(subjects)),
            "algo_realization": np.tile(np.asarray(MODEL_COLUMNS, dtype=object), len(subjects)),
            "value": matrix.reshape(len(subjects) * len(MODEL_COLUMNS)),
        }
    )

    last_error = "mixedlm_not_run"
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    "value ~ C(algorithm)",
                    data=long_df,
                    groups=long_df["subject"],
                    re_formula="1",
                    vc_formula={"algo_real": "0 + C(algo_realization)"},
                )
                fit = model.fit(reml=True, method=method, disp=False)
            sigma_subject = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else 0.0
            sigma_contour = float(np.asarray(fit.vcomp)[0]) if len(np.asarray(fit.vcomp)) else 0.0
            sigma_residual = float(fit.scale)
            sigma_subject = max(sigma_subject, 0.0)
            sigma_contour = max(sigma_contour, 0.0)
            sigma_residual = max(sigma_residual, 0.0)
            total = sigma_subject + sigma_contour + sigma_residual
            if total <= 0:
                last_error = f"mixedlm_{method}_nonpositive_total_variance"
                continue
            if getattr(fit, "converged", False):
                return (
                    {
                        "sigma_subject_var": float(sigma_subject),
                        "sigma_contour_var": float(sigma_contour),
                        "sigma_residual_var": float(sigma_residual),
                        "subject_pct_total": float(sigma_subject / total),
                        "contour_pct_total": float(sigma_contour / total),
                        "residual_pct_total": float(sigma_residual / total),
                        "variance_method": f"mixedlm_{method}",
                        "mixedlm_converged": True,
                    },
                    None,
                )
            last_error = f"mixedlm_{method}_not_converged"
        except Exception as exc:
            last_error = f"mixedlm_{method}_error:{type(exc).__name__}:{exc}"
    return {}, last_error


def discover_paired_courses(
    manuscript_root: Path,
    cohorts: list[str],
) -> tuple[list[CourseRecord], pd.DataFrame, dict[str, Any]]:
    records: list[CourseRecord] = []
    discovery_rows: list[dict[str, Any]] = []
    cohort_stats: dict[str, dict[str, int]] = {}
    hipokampy_head_found = 0

    for cohort in cohorts:
        data_dir = manuscript_root / cohort / "data"
        candidate_files = sorted(data_dir.glob("*/*/radiomics_robustness_ct.parquet")) if data_dir.is_dir() else []
        stats_row = {
            "files_seen": len(candidate_files),
            "paired_retained": 0,
            "missing_custom": 0,
            "missing_ts": 0,
            "hipokampy_head_missing": 0,
            "read_error": 0,
        }
        for parquet_path in candidate_files:
            patient_id = parquet_path.parent.parent.name
            course_id = parquet_path.parent.name
            discovery = {
                "cohort": cohort,
                "body_region": COHORT_REGION[cohort],
                "patient_id": patient_id,
                "course_id": course_id,
                "parquet": str(parquet_path),
                "status": "excluded",
                "reason": "",
                "ts_source": "",
                "segmentation_sources": "",
            }
            try:
                source_values = pq.read_table(parquet_path, columns=["segmentation_source"]).to_pandas()
                sources = sorted(
                    {
                        str(value)
                        for value in source_values["segmentation_source"].dropna().tolist()
                    }
                )
            except Exception as exc:
                discovery["reason"] = f"read_error:{type(exc).__name__}"
                stats_row["read_error"] += 1
                discovery_rows.append(discovery)
                continue

            discovery["segmentation_sources"] = "|".join(sources)
            sources_set = set(sources)
            if SEG_SRC_CUSTOM not in sources_set:
                discovery["reason"] = "missing_custom"
                stats_row["missing_custom"] += 1
                discovery_rows.append(discovery)
                continue

            if cohort == "Hipokampy":
                head_sources = [source for source in sources if is_head_task_source(source)]
                if not head_sources:
                    discovery["reason"] = "hipokampy_head_task_missing"
                    stats_row["hipokampy_head_missing"] += 1
                    discovery_rows.append(discovery)
                    continue
                ts_source = head_sources[0]
                hipokampy_head_found += 1
            else:
                if SEG_SRC_THORAX not in sources_set:
                    discovery["reason"] = "missing_autorts_total"
                    stats_row["missing_ts"] += 1
                    discovery_rows.append(discovery)
                    continue
                ts_source = SEG_SRC_THORAX

            records.append(
                CourseRecord(
                    cohort=cohort,
                    body_region=COHORT_REGION[cohort],
                    patient_id=patient_id,
                    course_id=course_id,
                    parquet=str(parquet_path),
                    size_bytes=parquet_path.stat().st_size,
                    ts_source=ts_source,
                    segmentation_sources=tuple(sources),
                )
            )
            discovery["status"] = "retained"
            discovery["reason"] = "paired_complete_sources"
            discovery["ts_source"] = ts_source
            stats_row["paired_retained"] += 1
            discovery_rows.append(discovery)

        cohort_stats[cohort] = stats_row

    discovery_df = pd.DataFrame(discovery_rows).sort_values(
        ["cohort", "patient_id", "course_id"],
        kind="stable",
    )
    diagnostics = {
        "cohort_stats": cohort_stats,
        "hipokampy_head_task_courses_found": hipokampy_head_found,
        "hipokampy_head_task_blocked": hipokampy_head_found == 0 and "Hipokampy" in cohorts,
    }
    return records, discovery_df, diagnostics


def load_course_paired(course: CourseRecord) -> pd.DataFrame | None:
    try:
        df = pq.read_table(
            course.parquet,
            columns=[
                "segmentation_source",
                "perturbation_id",
                "roi_name",
                "feature_name",
                "value",
            ],
        ).to_pandas()
    except Exception as exc:
        print(f"[load] {course.parquet}: read error: {exc}", file=sys.stderr, flush=True)
        return None

    df = df[
        df["segmentation_source"].isin([course.ts_source, SEG_SRC_CUSTOM])
        & df["perturbation_id"].isin(PERTURBATION_MAP)
    ].copy()
    if df.empty:
        return None

    df["algorithm"] = np.where(df["segmentation_source"] == course.ts_source, "TS", "Custom")
    df["realization"] = df["perturbation_id"].map(PERTURBATION_MAP)
    df["algorithm_realization"] = df["algorithm"] + "_" + df["realization"]
    df["patient_id"] = course.patient_id
    df["course_id"] = course.course_id
    df["subject"] = f"{course.patient_id}_{course.course_id}"
    df["cohort"] = course.cohort
    df["body_region"] = course.body_region
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[
        [
            "cohort",
            "body_region",
            "patient_id",
            "course_id",
            "subject",
            "roi_name",
            "feature_name",
            "algorithm_realization",
            "value",
        ]
    ]


def load_all_courses(courses: list[CourseRecord], workers: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not courses:
        return pd.DataFrame(
            columns=[
                "cohort",
                "body_region",
                "patient_id",
                "course_id",
                "subject",
                "roi_name",
                "feature_name",
                "algorithm_realization",
                "value",
            ]
        )
    if workers <= 1 or len(courses) < 4:
        for course in courses:
            frame = load_course_paired(course)
            if frame is not None and not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(load_course_paired, course): course for course in courses}
        for future in as_completed(futures):
            frame = future.result()
            if frame is not None and not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def enforce_allowed_cohorts(long_df: pd.DataFrame) -> None:
    discovered = set(long_df["cohort"].dropna().astype(str).unique().tolist())
    if not discovered <= ALLOWED_COHORTS:
        raise RuntimeError(
            f"Discovered cohorts outside allow-list: {sorted(discovered - ALLOWED_COHORTS)}"
        )


def apply_smoke_filters(long_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = long_df.copy()
    if args.subject_list:
        allowed_subjects = [item.strip() for item in args.subject_list.split(",") if item.strip()]
        filtered = filtered[filtered["subject"].isin(allowed_subjects)].copy()
    if args.max_subjects is not None:
        allowed_subjects = sorted(filtered["subject"].dropna().astype(str).unique().tolist())[: args.max_subjects]
        filtered = filtered[filtered["subject"].isin(allowed_subjects)].copy()
    if args.roi_name:
        filtered = filtered[filtered["roi_name"] == args.roi_name].copy()
    if args.feature_name:
        filtered = filtered[filtered["feature_name"] == args.feature_name].copy()

    if args.max_rois is not None:
        allowed_rois = sorted(filtered["roi_name"].dropna().astype(str).unique().tolist())[: args.max_rois]
        filtered = filtered[filtered["roi_name"].isin(allowed_rois)].copy()
    if args.max_features is not None:
        allowed_features = sorted(filtered["feature_name"].dropna().astype(str).unique().tolist())[
            : args.max_features
        ]
        filtered = filtered[filtered["feature_name"].isin(allowed_features)].copy()
    return filtered


def build_feature_payloads(
    long_df: pd.DataFrame,
) -> tuple[list[FeaturePayload], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if long_df.empty:
        return [], pd.DataFrame(), pd.DataFrame(), []

    duplicate_counts = (
        long_df.groupby(
            [
                "cohort",
                "body_region",
                "patient_id",
                "course_id",
                "subject",
                "roi_name",
                "feature_name",
                "algorithm_realization",
            ]
        )
        .size()
        .reset_index(name="n_rows")
    )
    problematic = duplicate_counts[duplicate_counts["n_rows"] > 1]
    if not problematic.empty:
        raise RuntimeError(
            "Duplicate measurements detected for subject/ROI/feature/algorithm_realization."
        )

    payloads: list[FeaturePayload] = []
    row_audits: list[pd.DataFrame] = []
    feature_exclusions: list[dict[str, Any]] = []
    intersection_log: list[dict[str, Any]] = []

    group_cols = ["cohort", "body_region", "roi_name"]
    for (cohort, body_region, roi_name), sub in long_df.groupby(group_cols, sort=True):
        subjects_df = (
            sub[["patient_id", "course_id", "subject"]]
            .drop_duplicates()
            .sort_values(["patient_id", "course_id"], kind="stable")
            .reset_index(drop=True)
        )
        feature_names = sorted(sub["feature_name"].dropna().astype(str).unique().tolist())
        if subjects_df.empty or not feature_names:
            continue

        wide_existing = (
            sub.pivot_table(
                index=["subject", "patient_id", "course_id", "feature_name"],
                columns="algorithm_realization",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for column in MODEL_COLUMNS:
            if column not in wide_existing.columns:
                wide_existing[column] = np.nan

        subjects_grid = subjects_df.assign(_merge_key=1)
        features_grid = pd.DataFrame({"feature_name": feature_names, "_merge_key": 1})
        full_grid = (
            subjects_grid.merge(features_grid, on="_merge_key", how="inner")
            .drop(columns="_merge_key")
            .sort_values(["patient_id", "course_id", "feature_name"], kind="stable")
            .reset_index(drop=True)
        )

        wide = full_grid.merge(
            wide_existing[
                [
                    "subject",
                    "patient_id",
                    "course_id",
                    "feature_name",
                    *MODEL_COLUMNS,
                ]
            ],
            on=["subject", "patient_id", "course_id", "feature_name"],
            how="left",
        )
        wide["cohort"] = cohort
        wide["body_region"] = body_region
        wide["roi_name"] = roi_name
        wide = wide[
            [
                "cohort",
                "body_region",
                "roi_name",
                "patient_id",
                "course_id",
                "subject",
                "feature_name",
                *MODEL_COLUMNS,
            ]
        ]

        matrix = wide.loc[:, MODEL_COLUMNS].to_numpy(dtype=float)
        finite_mask = np.isfinite(matrix)
        n_missing = (~finite_mask).sum(axis=1)
        row_complete = finite_mask.all(axis=1)
        row_zero_var = np.nanvar(matrix, axis=1) <= ZERO_VAR_EPS
        row_status = np.where(row_complete, "included", "excluded")
        row_reason = np.where(row_complete, "complete", "missing_realization_or_nonfinite")

        audit_df = wide[
            ["cohort", "body_region", "roi_name", "patient_id", "course_id", "subject", "feature_name"]
        ].copy()
        audit_df["row_status"] = row_status
        audit_df["row_reason"] = row_reason
        audit_df["n_missing_realizations"] = n_missing
        audit_df["zero_variance_all_6"] = row_zero_var
        row_audits.append(audit_df)

        total_subjects = len(subjects_df)
        feature_summary = (
            audit_df.groupby("feature_name", sort=True)
            .agg(
                n_subject_rows=("subject", "size"),
                n_subject_rows_complete=("row_status", lambda values: int((values == "included").sum())),
                n_subject_rows_missing=("n_missing_realizations", lambda values: int((values > 0).sum())),
                n_zero_variance_rows=("zero_variance_all_6", "sum"),
            )
            .reset_index()
        )
        feature_summary["n_subjects_expected"] = total_subjects
        feature_summary["is_full_intersection"] = (
            feature_summary["n_subject_rows_complete"] == total_subjects
        )

        n_intersection = int(feature_summary["is_full_intersection"].sum())
        intersection_log.append(
            {
                "cohort": cohort,
                "body_region": body_region,
                "roi_name": roi_name,
                "n_subjects_expected": int(total_subjects),
                "n_feature_union": int(len(feature_names)),
                "n_feature_intersection": int(n_intersection),
            }
        )

        for feature_row in feature_summary.itertuples(index=False):
            feature_name = str(feature_row.feature_name)
            if not bool(feature_row.is_full_intersection):
                feature_exclusions.append(
                    {
                        "cohort": cohort,
                        "body_region": body_region,
                        "roi_name": roi_name,
                        "feature_name": feature_name,
                        "feature_family": get_feature_family(feature_name),
                        "image_type": get_image_type(feature_name),
                        "reason": "incomplete_feature_intersection",
                        "n_subjects_expected": int(feature_row.n_subjects_expected),
                        "n_subject_rows_complete": int(feature_row.n_subject_rows_complete),
                        "n_subject_rows_missing": int(feature_row.n_subject_rows_missing),
                    }
                )
                continue

            feature_block = wide.loc[
                (wide["feature_name"] == feature_name) & row_complete,
                ["subject", *MODEL_COLUMNS],
            ].copy()
            feature_block = feature_block.sort_values("subject", kind="stable").reset_index(drop=True)
            feature_matrix = feature_block.loc[:, MODEL_COLUMNS].to_numpy(dtype=float)
            if feature_matrix.shape[0] < MIN_SUBJECTS_FOR_ICC:
                feature_exclusions.append(
                    {
                        "cohort": cohort,
                        "body_region": body_region,
                        "roi_name": roi_name,
                        "feature_name": feature_name,
                        "feature_family": get_feature_family(feature_name),
                        "image_type": get_image_type(feature_name),
                        "reason": "insufficient_subjects",
                        "n_subjects_expected": int(feature_row.n_subjects_expected),
                        "n_subject_rows_complete": int(feature_row.n_subject_rows_complete),
                        "n_subject_rows_missing": int(feature_row.n_subject_rows_missing),
                    }
                )
                continue

            payloads.append(
                FeaturePayload(
                    cohort=cohort,
                    body_region=body_region,
                    roi_name=roi_name,
                    feature_name=feature_name,
                    feature_family=get_feature_family(feature_name),
                    image_type=get_image_type(feature_name),
                    subjects=tuple(feature_block["subject"].tolist()),
                    matrix=feature_matrix,
                )
            )

    audit_out = pd.concat(row_audits, ignore_index=True) if row_audits else pd.DataFrame()
    feature_exclusions_out = (
        pd.DataFrame(feature_exclusions).sort_values(
            ["cohort", "roi_name", "feature_name"],
            kind="stable",
        )
        if feature_exclusions
        else pd.DataFrame(
            columns=[
                "cohort",
                "body_region",
                "roi_name",
                "feature_name",
                "feature_family",
                "image_type",
                "reason",
                "n_subjects_expected",
                "n_subject_rows_complete",
                "n_subject_rows_missing",
            ]
        )
    )
    return payloads, audit_out, feature_exclusions_out, intersection_log


def compute_feature_metrics(payload: FeaturePayload) -> dict[str, Any]:
    matrix = np.asarray(payload.matrix, dtype=float)
    base = {
        "cohort": payload.cohort,
        "body_region": payload.body_region,
        "roi_name": payload.roi_name,
        "feature_name": payload.feature_name,
        "feature_family": payload.feature_family,
        "image_type": payload.image_type,
        "n_subjects": int(matrix.shape[0]),
    }

    if not np.all(np.isfinite(matrix)):
        base["status"] = "excluded"
        base["reason"] = "nonfinite_matrix"
        return base
    if np.var(matrix) <= ZERO_VAR_EPS:
        base["status"] = "excluded"
        base["reason"] = "zero_total_variance"
        return base

    icc31 = icc31_from_matrix(matrix[:, :3])
    if icc31 is None:
        base["status"] = "excluded"
        base["reason"] = "zero_variance_ts_within_or_icc31_failed"
        return base

    icc21 = icc21_absolute_agreement_from_matrix(algorithm_mean_matrix(matrix))
    if icc21 is None:
        base["status"] = "excluded"
        base["reason"] = "zero_variance_inter_algorithm_or_icc21_failed"
        return base

    variance_payload, variance_error = fit_mixedlm_variance_components(matrix, payload.subjects)
    if variance_error is not None or not variance_payload:
        fallback_payload = fast_variance_components_from_matrix(matrix)
        if fallback_payload is None:
            base["status"] = "excluded"
            base["reason"] = variance_error or "variance_decomposition_failed"
            return base
        variance_payload = fallback_payload
        variance_payload["mixedlm_error"] = variance_error
    else:
        variance_payload["mixedlm_error"] = ""

    base.update(
        {
            "status": "ok",
            "reason": "",
            "icc31_ts_within": float(icc31[0]),
            "icc31_ts_within_ci_lo": float(icc31[1]),
            "icc31_ts_within_ci_hi": float(icc31[2]),
            "icc21_absolute_agreement": float(icc21[0]),
            "icc21_absolute_agreement_ci_lo": float(icc21[1]),
            "icc21_absolute_agreement_ci_hi": float(icc21[2]),
            "fisher_z_ts_within": float(fisher_z(icc31[0])),
            "fisher_z_inter_algorithm": float(fisher_z(icc21[0])),
            "fisher_z_delta": float(fisher_z(icc21[0]) - fisher_z(icc31[0])),
        }
    )
    base.update(variance_payload)
    return base


def compute_all_feature_metrics(
    payloads: list[FeaturePayload],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not payloads:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    if workers <= 1 or len(payloads) < 8:
        for payload in payloads:
            rows.append(compute_feature_metrics(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(compute_feature_metrics, payload): payload for payload in payloads}
            for future in as_completed(futures):
                rows.append(future.result())

    all_rows = pd.DataFrame(rows)
    results_df = (
        all_rows[all_rows["status"] == "ok"]
        .drop(columns=["status", "reason"])
        .sort_values(["cohort", "roi_name", "feature_name"], kind="stable")
        .reset_index(drop=True)
    )
    exclusions_df = (
        all_rows[all_rows["status"] != "ok"]
        .drop(columns=["status"])
        .rename(columns={"reason": "reason"})
        .sort_values(["cohort", "roi_name", "feature_name"], kind="stable")
        .reset_index(drop=True)
    )
    return results_df, exclusions_df


def bootstrap_summary_group(
    body_region: str,
    feature_family: str,
    payloads: list[FeaturePayload],
    n_iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cohort_subjects: dict[str, np.ndarray] = {}
    subject_lookups: dict[tuple[str, str, str], dict[str, int]] = {}

    for payload in payloads:
        cohort_subjects.setdefault(payload.cohort, np.array(sorted(set(payload.subjects)), dtype=object))
        subject_lookups[(payload.cohort, payload.roi_name, payload.feature_name)] = {
            subject: idx for idx, subject in enumerate(payload.subjects)
        }

    contour_medians: list[float] = []
    icc_medians: list[float] = []

    for _ in range(n_iterations):
        sampled_by_cohort = {
            cohort: rng.choice(subjects, size=len(subjects), replace=True)
            for cohort, subjects in cohort_subjects.items()
        }
        contour_values: list[float] = []
        icc_values: list[float] = []
        for payload in payloads:
            lookup = subject_lookups[(payload.cohort, payload.roi_name, payload.feature_name)]
            sampled_subjects = sampled_by_cohort[payload.cohort]
            indices = [lookup[subject] for subject in sampled_subjects if subject in lookup]
            if len(indices) < MIN_SUBJECTS_FOR_ICC:
                continue
            boot_matrix = payload.matrix[np.asarray(indices), :]
            variance_payload = fast_variance_components_from_matrix(boot_matrix)
            icc21 = icc21_absolute_agreement_from_matrix(algorithm_mean_matrix(boot_matrix))
            if variance_payload is None or icc21 is None:
                continue
            contour_values.append(float(variance_payload["contour_pct_total"]))
            icc_values.append(float(icc21[0]))
        if contour_values:
            contour_medians.append(float(np.median(contour_values)))
        if icc_values:
            icc_medians.append(float(np.median(icc_values)))

    return {
        "body_region": body_region,
        "feature_family": feature_family,
        "n_features": len(payloads),
        "bootstrap_strategy": "subject_cluster_within_cohort",
        "bootstrap_iterations_requested": int(n_iterations),
        "bootstrap_iterations_completed_contour": int(len(contour_medians)),
        "bootstrap_iterations_completed_icc21": int(len(icc_medians)),
        "contour_pct_total_ci_lo": float(np.percentile(contour_medians, 2.5)) if contour_medians else np.nan,
        "contour_pct_total_ci_hi": float(np.percentile(contour_medians, 97.5)) if contour_medians else np.nan,
        "icc21_absolute_agreement_ci_lo": float(np.percentile(icc_medians, 2.5)) if icc_medians else np.nan,
        "icc21_absolute_agreement_ci_hi": float(np.percentile(icc_medians, 97.5)) if icc_medians else np.nan,
    }


def build_bootstrap_summary(
    results_df: pd.DataFrame,
    payloads: list[FeaturePayload],
    n_iterations: int,
    workers: int,
) -> pd.DataFrame:
    if n_iterations <= 0 or results_df.empty or not payloads:
        return pd.DataFrame(
            columns=[
                "body_region",
                "feature_family",
                "n_features",
                "bootstrap_strategy",
                "bootstrap_iterations_requested",
                "bootstrap_iterations_completed_contour",
                "bootstrap_iterations_completed_icc21",
                "contour_pct_total_ci_lo",
                "contour_pct_total_ci_hi",
                "icc21_absolute_agreement_ci_lo",
                "icc21_absolute_agreement_ci_hi",
            ]
        )

    payload_lookup = {
        (payload.cohort, payload.roi_name, payload.feature_name): payload for payload in payloads
    }
    grouped_payloads: dict[tuple[str, str], list[FeaturePayload]] = defaultdict(list)
    for row in results_df.itertuples(index=False):
        key = (row.cohort, row.roi_name, row.feature_name)
        grouped_payloads[(row.body_region, row.feature_family)].append(payload_lookup[key])

    tasks = []
    for idx, ((body_region, feature_family), group_payloads) in enumerate(sorted(grouped_payloads.items())):
        tasks.append(
            (
                body_region,
                feature_family,
                group_payloads,
                n_iterations,
                BOOTSTRAP_SEED + idx,
            )
        )

    rows: list[dict[str, Any]] = []
    if workers <= 1 or len(tasks) < 2:
        for task in tasks:
            rows.append(bootstrap_summary_group(*task))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = {pool.submit(bootstrap_summary_group, *task): task for task in tasks}
            for future in as_completed(futures):
                rows.append(future.result())

    return pd.DataFrame(rows).sort_values(["body_region", "feature_family"], kind="stable")


def build_summary(results_df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    boot_lookup = {
        (row.body_region, row.feature_family): row for row in bootstrap_df.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for (body_region, feature_family), sub in results_df.groupby(["body_region", "feature_family"], sort=True):
        boot_row = boot_lookup.get((body_region, feature_family))
        rows.append(
            {
                "body_region": body_region,
                "feature_family": feature_family,
                "n_features": int(len(sub)),
                "median_contour_pct_total": float(sub["contour_pct_total"].median()),
                "contour_pct_total_ci_lo": float(getattr(boot_row, "contour_pct_total_ci_lo", np.nan)),
                "contour_pct_total_ci_hi": float(getattr(boot_row, "contour_pct_total_ci_hi", np.nan)),
                "median_icc21_absolute_agreement": float(sub["icc21_absolute_agreement"].median()),
                "icc21_absolute_agreement_ci_lo": float(
                    getattr(boot_row, "icc21_absolute_agreement_ci_lo", np.nan)
                ),
                "icc21_absolute_agreement_ci_hi": float(
                    getattr(boot_row, "icc21_absolute_agreement_ci_hi", np.nan)
                ),
                "bootstrap_strategy": getattr(boot_row, "bootstrap_strategy", ""),
                "bootstrap_iterations_requested": int(
                    getattr(boot_row, "bootstrap_iterations_requested", 0)
                ),
                "bootstrap_iterations_completed_contour": int(
                    getattr(boot_row, "bootstrap_iterations_completed_contour", 0)
                ),
                "bootstrap_iterations_completed_icc21": int(
                    getattr(boot_row, "bootstrap_iterations_completed_icc21", 0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["body_region", "feature_family"], kind="stable")


def main() -> int:
    args = parse_args()
    if args.cohort is not None and args.cohort not in ALLOWED_COHORTS:
        allowed = ", ".join(COHORT_ORDER)
        print(
            f"ERROR: --cohort {args.cohort!r} is not allowed for script 21. Allowed cohorts: {allowed}.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    output_root = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
    out_data = output_root / "data"
    out_tables = output_root / "tables"
    out_logs = output_root / "logs"
    for path in (out_data, out_tables, out_logs):
        path.mkdir(parents=True, exist_ok=True)

    out_icc = out_data / "inter_algorithm_ai_icc.parquet"
    out_summary = out_tables / "inter_algorithm_ai_summary.csv"
    out_bootstrap = out_tables / "inter_algorithm_ai_bootstrap_summary.csv"
    out_audit = out_tables / "exclusions_audit.csv"
    out_feature_exclusions = out_tables / "inter_algorithm_ai_feature_exclusions.csv"
    out_discovery = out_tables / "inter_algorithm_ai_course_discovery.csv"
    job_id = args.job_id or os.environ.get("JOB_ID", "local")
    manifest_path = out_logs / f"21_manifest_{job_id}.json"

    cohorts = [args.cohort] if args.cohort else list(COHORT_ORDER)
    started = time.time()
    print(f"[main] cohorts={cohorts}", flush=True)
    print(f"[main] output_root={output_root}", flush=True)
    print(
        "[main] bootstrap strategy: patient-level clustered within cohort "
        "(n_cohorts=2 too small for stable cohort-level resampling)",
        flush=True,
    )

    courses, discovery_df, discovery_diag = discover_paired_courses(MANUSCRIPT_ROOT, cohorts)
    atomic_write_csv(discovery_df, out_discovery)

    hipokampy_blocked = discovery_diag["hipokampy_head_task_blocked"]
    if args.cohort == "Hipokampy" and hipokampy_blocked:
        message = (
            "Hipokampy brain arm blocked: head-task TotalSegmentator regeneration "
            "not complete. Re-run after head-task outputs land in per-course parquets."
        )
        atomic_write_json(
            {
                "status": "blocked",
                "reason": message,
                "project_scope_lock": sorted(ALLOWED_COHORTS),
                "output_root": str(output_root),
                "project_path": str(Path("/umed-projekty/rtpipeline/manuscript")),
            },
            manifest_path,
        )
        print(message, file=sys.stderr, flush=True)
        return 3

    if hipokampy_blocked and args.cohort is None:
        print(
            "Hipokampy brain arm blocked: head-task TotalSegmentator regeneration "
            "not complete. Continuing with PlucaRCHT only.",
            flush=True,
        )

    if not courses:
        atomic_write_json(
            {
                "status": "blocked",
                "reason": "no paired courses after scope lock and head-task gate",
                "project_scope_lock": sorted(ALLOWED_COHORTS),
                "output_root": str(output_root),
                "project_path": str(Path("/umed-projekty/rtpipeline/manuscript")),
            },
            manifest_path,
        )
        print("No paired courses discovered after gating.", file=sys.stderr, flush=True)
        return 4

    long_df = load_all_courses(courses, workers=max(args.workers, 1))
    if long_df.empty:
        print("No paired rows remained after loading.", file=sys.stderr, flush=True)
        return 5

    long_df = apply_smoke_filters(long_df, args)
    if long_df.empty:
        print("All rows were removed by ROI/feature smoke filters.", file=sys.stderr, flush=True)
        return 6

    enforce_allowed_cohorts(long_df)

    payloads, row_audit_df, intersection_exclusions_df, intersection_log = build_feature_payloads(long_df)
    atomic_write_csv(row_audit_df, out_audit)

    results_df, metric_exclusions_df = compute_all_feature_metrics(payloads, workers=max(args.workers, 1))
    feature_exclusions_df = pd.concat(
        [intersection_exclusions_df, metric_exclusions_df],
        ignore_index=True,
        sort=False,
    )
    if not feature_exclusions_df.empty:
        feature_exclusions_df = feature_exclusions_df.sort_values(
            ["cohort", "roi_name", "feature_name"], kind="stable"
        ).reset_index(drop=True)
    atomic_write_csv(feature_exclusions_df, out_feature_exclusions)

    if results_df.empty:
        print("No feature-level results survived completeness and metric checks.", file=sys.stderr, flush=True)
        return 7

    included_keys = {
        (row.cohort, row.roi_name, row.feature_name)
        for row in results_df[["cohort", "roi_name", "feature_name"]].itertuples(index=False)
    }
    included_payloads = [
        payload
        for payload in payloads
        if (payload.cohort, payload.roi_name, payload.feature_name) in included_keys
    ]

    bootstrap_df = (
        build_bootstrap_summary(
            results_df,
            included_payloads,
            0 if args.skip_bootstrap else args.bootstrap_iterations,
            workers=max(args.workers, 1),
        )
        if not args.skip_bootstrap
        else pd.DataFrame()
    )
    summary_df = build_summary(results_df, bootstrap_df)

    atomic_write_parquet(results_df, out_icc)
    atomic_write_csv(summary_df, out_summary)
    atomic_write_csv(bootstrap_df, out_bootstrap)

    manifest = {
        "status": "ok",
        "job_id": job_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_path": "/umed-projekty/rtpipeline/manuscript",
        "project_scope_lock": sorted(ALLOWED_COHORTS),
        "selected_cohort": args.cohort,
        "workers": int(args.workers),
        "bootstrap_iterations_requested": 0 if args.skip_bootstrap else int(args.bootstrap_iterations),
        "bootstrap_strategy": "subject_cluster_within_cohort",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "hipokampy_head_task_blocked": bool(hipokampy_blocked),
        "discovery": discovery_diag,
        "n_courses_retained": int(len(courses)),
        "n_subjects_retained": int(long_df["subject"].nunique()),
        "n_rois_retained": int(long_df["roi_name"].nunique()),
        "n_feature_union_retained": int(long_df["feature_name"].nunique()),
        "n_feature_payloads": int(len(payloads)),
        "n_feature_results": int(len(results_df)),
        "n_feature_exclusions": int(len(feature_exclusions_df)),
        "feature_intersection_by_cohort_roi": intersection_log,
        "outputs": {
            "inter_algorithm_ai_icc": str(out_icc),
            "inter_algorithm_ai_summary": str(out_summary),
            "inter_algorithm_ai_bootstrap_summary": str(out_bootstrap),
            "exclusions_audit": str(out_audit),
            "feature_exclusions": str(out_feature_exclusions),
            "course_discovery": str(out_discovery),
        },
        "inputs": [
            {
                **asdict(course),
                "sha256": file_sha256(Path(course.parquet)),
            }
            for course in courses
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    atomic_write_json(manifest, manifest_path)

    print(f"[main] wrote {out_icc}", flush=True)
    print(f"[main] wrote {out_summary}", flush=True)
    print(f"[main] wrote {out_bootstrap}", flush=True)
    print(f"[main] wrote {out_audit}", flush=True)
    print(f"[main] wrote {out_feature_exclusions}", flush=True)
    print(f"[main] wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
