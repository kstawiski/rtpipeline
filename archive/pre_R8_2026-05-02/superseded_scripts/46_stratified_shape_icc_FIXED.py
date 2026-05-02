#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 46 FIXED: Stratified shape ICC after the 2026-04-18 aggregator rebuild.

PURPOSE
-------
Quantify the isotropy caveat recorded in report §26 by splitting the shape
family into:

  - scale_invariant
      sphericity, compactness1, compactness2, sphericalDisproportion,
      elongation, flatness

  - scale_dependent
      voxelVolume, meshVolume, surfaceArea, major/minor/leastAxisLength,
      maximum2DDiameter*, maximum3DDiameter

The point estimates are read from the rebuilt cohort-wide `icc_results.parquet`
only after a defensive wait gate confirms that the post-2026-04-18 rebuild
artifact is present. Bootstrap uncertainty is then recomputed from the
patient-level `robustness_by_cohort/*.parquet` tables so the CI path genuinely
resamples patients rather than feature rows.

ANALYSIS DESIGN
---------------
1. Wait for the rebuilt `icc_results.parquet` artifact (job 3560738) by
   requiring a file mtime at or after the configured 2026-04-18 threshold.
2. Read the rebuilt point-estimate table and retain shape-family rows.
3. Classify each shape feature as scale_invariant, scale_dependent, or
   unclassified.
4. For each cohort parquet, load only the requested shape features and compute
   patient-level sufficient statistics for each `(roi_name, feature_name)` cell.
   ICC subjects remain `patient_id + course_id`, but the bootstrap resamples
   patient clusters so multi-course patients contribute all of their courses
   whenever that patient is drawn.
5. Bootstrap (seed 2604 by default):
   - Level 1: resample patients with replacement within each cohort.
   - Level 2: include all course / ROI / perturbation rows for each sampled
     patient.
   - Recompute ICC(3,1) and CoV for every `(cohort, roi_name, feature_name)`.
   - Aggregate the median within each `(cohort, roi_name, scale_class)`.
   - Aggregate those cohort-ROI medians to body-region and overall summaries.
6. Pairwise inference is reported honestly as a cluster-median paired Wilcoxon
   on `(cohort, roi_name)` summaries because there is no defensible 1:1
   feature-level pairing between scale-invariant and scale-dependent shape
   features.

INPUTS
------
  /home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/data/robustness_by_cohort/*.parquet

Expected raw columns:
  patient_id, course_id, roi_name, perturbation_id, feature_name, value

OUTPUTS
-------
  analysis/data/shape_icc_stratified.parquet
      Feature-level rebuilt point estimates with scale-class annotations.

  analysis/tables/shape_icc_stratified_summary.csv
      Cohort-ROI medians plus body-region / overall summaries with bootstrap CIs.

  analysis/tables/shape_icc_stratified_pairwise.csv
      Honest cluster-median paired Wilcoxon outputs.

  analysis/figures/figure_shape_icc_stratified.{png,pdf}
      Two-panel summary figure (ICC and CoV).

  analysis/logs/46_manifest_<JOBID>.json
      Provenance, wait-gate metadata, validation stats, and output checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import resource
import socket
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
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

OUT_PARQUET = DATA_DIR / "shape_icc_stratified.parquet"
OUT_SUMMARY = TABLES_DIR / "shape_icc_stratified_summary.csv"
OUT_PAIRWISE = TABLES_DIR / "shape_icc_stratified_pairwise.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_shape_icc_stratified.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_shape_icc_stratified.pdf"

SEED = 2604
EXPECTED_REBUILD_JOB = "3560738"
DEFAULT_MIN_REBUILD_MTIME = "2026-04-18T00:00:00+00:00"
DEFAULT_POLL_SECONDS = 300
DEFAULT_MAX_WAIT_SECONDS = 21600
N_BOOT_DEFAULT = 1000

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
SUMMARY_SCALE_ORDER = ["scale_invariant", "scale_dependent", "unclassified"]
REGION_ORDER = ["Brain", "Pelvis", "Thorax", "Overall"]

SCALE_INVARIANT_BASES = {
    "compactness1",
    "compactness2",
    "elongation",
    "flatness",
    "sphericaldisproportion",
    "sphericity",
}
SCALE_DEPENDENT_BASES = {
    "leastaxislength",
    "majoraxislength",
    "maximum3ddiameter",
    "meshvolume",
    "minoraxislength",
    "surfacearea",
    "voxelvolume",
}


@dataclass
class GroupBootstrapData:
    roi_name: str
    feature_name: str
    scale_class: str
    k_raters: int
    n_subjects_by_patient: np.ndarray
    total_sum_by_patient: np.ndarray
    total_sum_sq_by_patient: np.ndarray
    row_sum_sq_by_patient: np.ndarray
    col_sums_by_patient: np.ndarray


@dataclass
class CohortBootstrapData:
    cohort: str
    body_region: str
    patient_ids: np.ndarray
    groups: list[GroupBootstrapData]
    cell_to_group_indices: dict[tuple[str, str], list[int]]


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
    logger = logging.getLogger("script46_fixed")
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


def rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0**3)
    return float(usage) / (1024.0**2)


def parse_iso_utc(raw_value: str) -> float:
    parsed = datetime.fromisoformat(raw_value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def parse_cohorts(raw_value: str) -> list[str]:
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def first_present(columns: list[str], candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise RuntimeError(f"Could not resolve {label}; looked for {candidates}")


def feature_base_name(feature_name: str) -> str:
    lower = str(feature_name).lower()
    marker = "_shape_"
    if marker in lower:
        return lower.split(marker, 1)[1]
    return lower.rsplit("_", 1)[-1]


def classify_shape_scale(feature_name: str) -> str:
    base = feature_base_name(feature_name)
    if base in SCALE_INVARIANT_BASES:
        return "scale_invariant"
    if base in SCALE_DEPENDENT_BASES:
        return "scale_dependent"
    if base.startswith("maximum2ddiameter"):
        return "scale_dependent"
    return "unclassified"


def wait_for_rebuilt_icc(
    *,
    min_mtime_ts: float,
    poll_seconds: int,
    max_wait_seconds: int,
    logger: logging.Logger,
) -> dict[str, object]:
    start = time.time()
    attempts = 0
    while True:
        attempts += 1
        if ICC_RESULTS_PARQUET.exists():
            stat = ICC_RESULTS_PARQUET.stat()
            if stat.st_mtime >= min_mtime_ts:
                logger.info(
                    "Rebuild gate satisfied for %s (mtime=%s, size=%.3f GB).",
                    ICC_RESULTS_PARQUET,
                    datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    stat.st_size / 1e9,
                )
                return {
                    "artifact_path": str(ICC_RESULTS_PARQUET),
                    "artifact_mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "artifact_size_bytes": int(stat.st_size),
                    "wait_seconds": float(time.time() - start),
                    "attempts": attempts,
                    "expected_job_id": EXPECTED_REBUILD_JOB,
                }
        if time.time() - start > max_wait_seconds:
            raise TimeoutError(
                "Timed out waiting for rebuilt icc_results.parquet "
                f"(expected job {EXPECTED_REBUILD_JOB}, minimum mtime "
                f"{datetime.fromtimestamp(min_mtime_ts, tz=timezone.utc).isoformat()})."
            )
        logger.info(
            "Waiting for rebuilt %s (job %s, minimum mtime %s). Sleeping %ds.",
            ICC_RESULTS_PARQUET,
            EXPECTED_REBUILD_JOB,
            datetime.fromtimestamp(min_mtime_ts, tz=timezone.utc).isoformat(),
            poll_seconds,
        )
        time.sleep(poll_seconds)


def load_rebuilt_shape_points(
    requested_cohorts: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not ICC_RESULTS_PARQUET.exists():
        raise FileNotFoundError(f"Missing rebuilt ICC parquet: {ICC_RESULTS_PARQUET}")

    df = pd.read_parquet(ICC_RESULTS_PARQUET)
    columns = df.columns.tolist()
    cohort_col = first_present(columns, ["cohort"], "cohort column")
    roi_col = first_present(columns, ["roi_name", "structure", "roi"], "ROI column")
    feature_col = first_present(columns, ["feature_name"], "feature column")
    icc_col = first_present(columns, ["icc", "icc_value"], "ICC column")
    cov_col = first_present(columns, ["cov_pct", "cov", "cov_percent"], "CoV column")
    family_col = next((c for c in ["feature_family", "family"] if c in columns), None)
    body_region_col = next((c for c in ["body_region", "region"] if c in columns), None)
    qcd_col = next((c for c in ["qcd"] if c in columns), None)
    n_subjects_col = next((c for c in ["n_subjects"] if c in columns), None)
    n_raters_col = next((c for c in ["n_perturbations", "n_raters"] if c in columns), None)

    out = df.copy()
    out["cohort"] = out[cohort_col].astype(str)
    out["roi_name"] = out[roi_col].astype(str)
    out["feature_name"] = out[feature_col].astype(str)
    out["icc"] = pd.to_numeric(out[icc_col], errors="coerce")
    out["cov_pct"] = pd.to_numeric(out[cov_col], errors="coerce")
    out["body_region"] = (
        out[body_region_col].astype(str)
        if body_region_col is not None
        else out["cohort"].map(BODY_REGION_BY_COHORT)
    )
    out["feature_family"] = (
        out[family_col].astype(str)
        if family_col is not None
        else np.where(out["feature_name"].str.contains("_shape_", case=False, na=False), "shape", "other")
    )
    out["feature_base_name"] = out["feature_name"].map(feature_base_name)
    out["scale_class"] = out["feature_name"].map(classify_shape_scale)

    if qcd_col is not None:
        out["qcd"] = pd.to_numeric(out[qcd_col], errors="coerce")
    else:
        out["qcd"] = np.nan
    if n_subjects_col is not None:
        out["n_subjects"] = pd.to_numeric(out[n_subjects_col], errors="coerce")
    else:
        out["n_subjects"] = np.nan
    if n_raters_col is not None:
        out["n_perturbations"] = pd.to_numeric(out[n_raters_col], errors="coerce")
    else:
        out["n_perturbations"] = np.nan

    out = out.loc[out["cohort"].isin(requested_cohorts)].copy()
    out = out.loc[out["feature_family"].str.lower() == "shape"].copy()

    if out.empty:
        raise RuntimeError(
            "No shape-family rows were found in rebuilt icc_results.parquet "
            f"for cohorts {requested_cohorts}."
        )

    dup_mask = out.duplicated(["cohort", "roi_name", "feature_name"], keep=False)
    if dup_mask.any():
        duplicate_count = int(dup_mask.sum())
        dup_cols = ["cohort", "roi_name", "feature_name"]
        if "segmentation_source" in out.columns:
            raise RuntimeError(
                "Rebuilt ICC table has duplicate (cohort, roi_name, feature_name) rows, "
                "likely from multiple segmentation sources. Script 46 fixed version will "
                "not silently double-count these rows."
            )
        raise RuntimeError(
            f"Rebuilt ICC table has {duplicate_count} duplicated rows on {dup_cols}."
        )

    meta = {
        "n_rows": int(len(out)),
        "n_features": int(out["feature_name"].nunique()),
        "n_rois": int(out["roi_name"].nunique()),
        "n_cohorts": int(out["cohort"].nunique()),
        "scale_class_counts": out["scale_class"].value_counts(dropna=False).to_dict(),
        "shape_feature_names": sorted(out["feature_name"].unique().tolist()),
    }
    logger.info(
        "Loaded rebuilt point-estimate shape table: %d rows, %d features, %d cohorts.",
        meta["n_rows"],
        meta["n_features"],
        meta["n_cohorts"],
    )
    return out[
        [
            "cohort",
            "body_region",
            "roi_name",
            "feature_name",
            "feature_base_name",
            "feature_family",
            "scale_class",
            "icc",
            "cov_pct",
            "qcd",
            "n_subjects",
            "n_perturbations",
        ]
    ].copy(), meta


def load_raw_shape_batch(
    cohort: str,
    feature_names: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    cohort_path = ROBUSTNESS_DIR / f"{cohort}.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort parquet: {cohort_path}")
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - target env dependency
        raise RuntimeError(
            "pyarrow is required to stream robustness_by_cohort parquet files."
        ) from exc

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
    scanner = dataset.scanner(
        columns=columns,
        filter=ds.field("feature_name").isin(feature_names),
        use_threads=True,
    )
    table = scanner.to_table()
    df = table.to_pandas()
    logger.info(
        "Loaded %s shape batch (%d features): %d rows, RSS %.2f GB.",
        cohort,
        len(feature_names),
        len(df),
        rss_gb(),
    )
    return df


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
    remaining_dup = work.duplicated(subset=key_cols, keep=False)
    if remaining_dup.any():
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


def prepare_bootstrap_data_for_cohort(
    cohort: str,
    feature_names: list[str],
    batch_size: int,
    logger: logging.Logger,
) -> tuple[CohortBootstrapData, dict[str, object]]:
    body_region = BODY_REGION_BY_COHORT.get(cohort)
    if body_region is None:
        raise RuntimeError(f"No body-region mapping registered for cohort {cohort}")

    feature_names = sorted(feature_names)
    prepared_groups: dict[tuple[str, str], dict[str, object]] = {}
    patient_union: set[str] = set()
    total_rows_loaded = 0
    duplicate_rows_seen_total = 0
    duplicate_rows_dropped_total = 0

    for start in range(0, len(feature_names), batch_size):
        feature_chunk = feature_names[start : start + batch_size]
        batch_df = load_raw_shape_batch(cohort, feature_chunk, logger)
        if batch_df.empty:
            continue
        total_rows_loaded += int(len(batch_df))
        required = {
            "patient_id",
            "course_id",
            "roi_name",
            "perturbation_id",
            "feature_name",
            "value",
        }
        missing = required.difference(batch_df.columns)
        if missing:
            raise RuntimeError(f"{cohort} raw batch missing columns: {sorted(missing)}")

        batch_df = batch_df.dropna(
            subset=["patient_id", "course_id", "roi_name", "perturbation_id", "feature_name", "value"]
        ).copy()
        batch_df["patient_id"] = batch_df["patient_id"].astype(str)
        batch_df["course_id"] = batch_df["course_id"].astype(str)
        batch_df["roi_name"] = batch_df["roi_name"].astype(str)
        batch_df["perturbation_id"] = batch_df["perturbation_id"].astype(str)
        batch_df["feature_name"] = batch_df["feature_name"].astype(str)
        batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
        batch_df = batch_df.dropna(subset=["value"])
        batch_df, dup_meta = resolve_duplicate_measurements(batch_df, cohort=cohort, logger=logger)
        duplicate_rows_seen_total += int(dup_meta["duplicate_rows_seen"])
        duplicate_rows_dropped_total += int(dup_meta["duplicate_rows_dropped"])

        for (roi_name, feature_name), sub in batch_df.groupby(
            ["roi_name", "feature_name"], sort=False
        ):
            stats_info = build_patient_stats_dict(sub)
            patient_stats = stats_info["patient_stats"]
            patient_union.update(patient_stats.keys())
            prepared_groups[(roi_name, feature_name)] = {
                "roi_name": roi_name,
                "feature_name": feature_name,
                "scale_class": classify_shape_scale(feature_name),
                "k_raters": int(stats_info["k_raters"]),
                "patient_stats": patient_stats,
            }
        logger.info(
            "Prepared %s batch %d-%d/%d; %d group cells accumulated; RSS %.2f GB.",
            cohort,
            start + 1,
            min(start + batch_size, len(feature_names)),
            len(feature_names),
            len(prepared_groups),
            rss_gb(),
        )

    patient_ids = np.array(sorted(patient_union), dtype=object)
    patient_index = {pid: idx for idx, pid in enumerate(patient_ids.tolist())}
    groups: list[GroupBootstrapData] = []
    cell_to_group_indices: dict[tuple[str, str], list[int]] = defaultdict(list)

    for key in sorted(prepared_groups.keys()):
        info = prepared_groups[key]
        k_raters = int(info["k_raters"])
        p = len(patient_ids)
        n_subjects = np.zeros(p, dtype=float)
        total_sum = np.zeros(p, dtype=float)
        total_sum_sq = np.zeros(p, dtype=float)
        row_sum_sq = np.zeros(p, dtype=float)
        col_sums = np.zeros((p, max(k_raters, 1)), dtype=float)

        for patient_id, stats in info["patient_stats"].items():
            idx = patient_index[patient_id]
            n_subjects[idx] = float(stats["n_subjects"])
            total_sum[idx] = float(stats["total_sum"])
            total_sum_sq[idx] = float(stats["total_sum_sq"])
            row_sum_sq[idx] = float(stats["row_sum_sq"])
            if k_raters > 0:
                col_sums[idx, :k_raters] = np.asarray(stats["col_sums"], dtype=float)

        group = GroupBootstrapData(
            roi_name=info["roi_name"],
            feature_name=info["feature_name"],
            scale_class=info["scale_class"],
            k_raters=k_raters,
            n_subjects_by_patient=n_subjects,
            total_sum_by_patient=total_sum,
            total_sum_sq_by_patient=total_sum_sq,
            row_sum_sq_by_patient=row_sum_sq,
            col_sums_by_patient=col_sums[:, :k_raters] if k_raters > 0 else col_sums[:, :0],
        )
        groups.append(group)
        cell_to_group_indices[(group.roi_name, group.scale_class)].append(len(groups) - 1)

    logger.info(
        "Prepared bootstrap data for %s: %d patients, %d feature cells, %d loaded rows.",
        cohort,
        len(patient_ids),
        len(groups),
        total_rows_loaded,
    )
    meta = {
        "cohort": cohort,
        "body_region": body_region,
        "n_patients": int(len(patient_ids)),
        "n_group_cells": int(len(groups)),
        "rows_loaded": int(total_rows_loaded),
        "duplicate_rows_seen": int(duplicate_rows_seen_total),
        "duplicate_rows_dropped": int(duplicate_rows_dropped_total),
    }
    return (
        CohortBootstrapData(
            cohort=cohort,
            body_region=body_region,
            patient_ids=patient_ids,
            groups=groups,
            cell_to_group_indices=dict(cell_to_group_indices),
        ),
        meta,
    )


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


def raw_point_metrics_from_bootstrap_data(cohort_data: CohortBootstrapData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weights = np.ones(len(cohort_data.patient_ids), dtype=float)
    for group in cohort_data.groups:
        n_subjects = float(np.dot(group.n_subjects_by_patient, weights))
        total_sum = float(np.dot(group.total_sum_by_patient, weights))
        total_sum_sq = float(np.dot(group.total_sum_sq_by_patient, weights))
        row_sum_sq = float(np.dot(group.row_sum_sq_by_patient, weights))
        col_sums = group.col_sums_by_patient.T @ weights if group.k_raters > 0 else np.array([])
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
            n_values=n_subjects * float(group.k_raters),
        )
        rows.append(
            {
                "cohort": cohort_data.cohort,
                "body_region": cohort_data.body_region,
                "roi_name": group.roi_name,
                "feature_name": group.feature_name,
                "scale_class": group.scale_class,
                "raw_icc_recomputed": icc,
                "raw_cov_pct_recomputed": cov_pct,
                "n_patients_nonzero": int(np.count_nonzero(group.n_subjects_by_patient > 0)),
                "n_subjects_raw": n_subjects,
                "n_perturbations_raw": int(group.k_raters),
            }
        )
    return pd.DataFrame(rows)


def validate_raw_vs_rebuilt(
    rebuilt_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    logger: logging.Logger,
) -> dict[str, object]:
    merged = rebuilt_df.merge(
        raw_df,
        on=["cohort", "body_region", "roi_name", "feature_name", "scale_class"],
        how="left",
    )
    matched = merged["raw_icc_recomputed"].notna().sum()
    coverage = float(matched / len(merged)) if len(merged) else float("nan")
    merged["abs_diff_icc"] = (merged["icc"] - merged["raw_icc_recomputed"]).abs()
    merged["abs_diff_cov"] = (merged["cov_pct"] - merged["raw_cov_pct_recomputed"]).abs()
    max_diff_icc = float(merged["abs_diff_icc"].max(skipna=True)) if matched else float("nan")
    max_diff_cov = float(merged["abs_diff_cov"].max(skipna=True)) if matched else float("nan")
    median_diff_icc = float(merged["abs_diff_icc"].median(skipna=True)) if matched else float("nan")
    median_diff_cov = float(merged["abs_diff_cov"].median(skipna=True)) if matched else float("nan")

    if matched and (coverage < 0.95 or (np.isfinite(max_diff_icc) and max_diff_icc > 0.05)):
        raise RuntimeError(
            "Raw-vs-rebuilt validation failed for script 46 fixed: "
            f"coverage={coverage:.3f}, max ICC diff={max_diff_icc:.4f}."
        )

    logger.info(
        "Raw-vs-rebuilt validation: matched=%d/%d (%.1f%%), median/max |ΔICC|=%.5f/%.5f, "
        "median/max |ΔCoV|=%.5f/%.5f.",
        matched,
        len(merged),
        coverage * 100.0 if np.isfinite(coverage) else float("nan"),
        median_diff_icc,
        max_diff_icc,
        median_diff_cov,
        max_diff_cov,
    )
    return {
        "matched_rows": int(matched),
        "total_rows": int(len(merged)),
        "coverage": coverage,
        "median_abs_diff_icc": median_diff_icc,
        "max_abs_diff_icc": max_diff_icc,
        "median_abs_diff_cov_pct": median_diff_cov,
        "max_abs_diff_cov_pct": max_diff_cov,
    }


def build_cohort_roi_point_summary(point_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cohort, body_region, roi_name, scale_class), sub in point_df.groupby(
        ["cohort", "body_region", "roi_name", "scale_class"], sort=False
    ):
        rows.append(
            {
                "summary_level": "cohort_roi",
                "cohort": cohort,
                "body_region": body_region,
                "roi_name": roi_name,
                "scale_class": scale_class,
                "median_icc": float(sub["icc"].median()),
                "median_cov_pct": float(sub["cov_pct"].median()),
                "n_features": int(sub["feature_name"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarise_cell_medians(cells_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (body_region, scale_class), sub in cells_df.groupby(
        ["body_region", "scale_class"], sort=False
    ):
        rows.append(
            {
                "summary_level": "body_region",
                "cohort": "",
                "body_region": body_region,
                "roi_name": "",
                "scale_class": scale_class,
                "median_icc": float(sub["median_icc"].median()),
                "median_cov_pct": float(sub["median_cov_pct"].median()),
                "n_cohort_roi_groups": int(len(sub)),
            }
        )
    for scale_class, sub in cells_df.groupby("scale_class", sort=False):
        rows.append(
            {
                "summary_level": "overall",
                "cohort": "",
                "body_region": "Overall",
                "roi_name": "",
                "scale_class": scale_class,
                "median_icc": float(sub["median_icc"].median()),
                "median_cov_pct": float(sub["median_cov_pct"].median()),
                "n_cohort_roi_groups": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_patient_cluster_cis(
    cohort_bootstrap_data: list[CohortBootstrapData],
    n_boot: int,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[pd.DataFrame] = []
    for iteration in range(n_boot):
        cell_rows: list[dict[str, object]] = []
        for cohort_data in cohort_bootstrap_data:
            p = len(cohort_data.patient_ids)
            if p == 0:
                continue
            weights = rng.multinomial(p, np.full(p, 1.0 / p))
            group_icc = np.full(len(cohort_data.groups), np.nan, dtype=float)
            group_cov = np.full(len(cohort_data.groups), np.nan, dtype=float)

            for idx, group in enumerate(cohort_data.groups):
                n_subjects = float(np.dot(group.n_subjects_by_patient, weights))
                if n_subjects < 2 or group.k_raters < 2:
                    continue
                total_sum = float(np.dot(group.total_sum_by_patient, weights))
                total_sum_sq = float(np.dot(group.total_sum_sq_by_patient, weights))
                row_sum_sq = float(np.dot(group.row_sum_sq_by_patient, weights))
                col_sums = group.col_sums_by_patient.T @ weights
                group_icc[idx] = compute_icc31_from_sufficient_stats(
                    n_subjects=n_subjects,
                    total_sum=total_sum,
                    total_sum_sq=total_sum_sq,
                    row_sum_sq=row_sum_sq,
                    col_sums=np.asarray(col_sums, dtype=float),
                )
                group_cov[idx] = compute_cov_from_sufficient_stats(
                    total_sum=total_sum,
                    total_sum_sq=total_sum_sq,
                    n_values=n_subjects * float(group.k_raters),
                )

            for (roi_name, scale_class), group_indices in cohort_data.cell_to_group_indices.items():
                icc_vals = group_icc[group_indices]
                cov_vals = group_cov[group_indices]
                finite_icc = icc_vals[np.isfinite(icc_vals)]
                finite_cov = cov_vals[np.isfinite(cov_vals)]
                cell_rows.append(
                    {
                        "cohort": cohort_data.cohort,
                        "body_region": cohort_data.body_region,
                        "roi_name": roi_name,
                        "scale_class": scale_class,
                        "median_icc": float(np.median(finite_icc)) if finite_icc.size else np.nan,
                        "median_cov_pct": float(np.median(finite_cov)) if finite_cov.size else np.nan,
                    }
                )

        iteration_cells = pd.DataFrame(cell_rows)
        if not iteration_cells.empty:
            summary_rows = summarise_cell_medians(iteration_cells)
            summary_rows["bootstrap_iteration"] = iteration
            records.append(summary_rows)

        if (iteration + 1) % 100 == 0 or iteration == n_boot - 1:
            logger.info("Bootstrap %d/%d complete; RSS %.2f GB.", iteration + 1, n_boot, rss_gb())

    boot_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if boot_df.empty:
        return pd.DataFrame(
            columns=[
                "summary_level",
                "body_region",
                "scale_class",
                "ci_low_icc",
                "ci_high_icc",
                "ci_low_cov_pct",
                "ci_high_cov_pct",
            ]
        )

    rows: list[dict[str, object]] = []
    for (summary_level, body_region, scale_class), sub in boot_df.groupby(
        ["summary_level", "body_region", "scale_class"], sort=False
    ):
        rows.append(
            {
                "summary_level": summary_level,
                "body_region": body_region,
                "scale_class": scale_class,
                "ci_low_icc": float(np.nanpercentile(sub["median_icc"], 2.5)),
                "ci_high_icc": float(np.nanpercentile(sub["median_icc"], 97.5)),
                "ci_low_cov_pct": float(np.nanpercentile(sub["median_cov_pct"], 2.5)),
                "ci_high_cov_pct": float(np.nanpercentile(sub["median_cov_pct"], 97.5)),
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_cluster_median_tests(cohort_roi_df: pd.DataFrame) -> pd.DataFrame:
    base = cohort_roi_df.loc[
        cohort_roi_df["scale_class"].isin(["scale_invariant", "scale_dependent"])
    ].copy()
    rows: list[dict[str, object]] = []
    for metric in ["median_icc", "median_cov_pct"]:
        wide = (
            base.pivot_table(
                index=["body_region", "cohort", "roi_name"],
                columns="scale_class",
                values=metric,
                aggfunc="first",
            )
            .reset_index()
        )
        for scope in ["Brain", "Pelvis", "Thorax", "Overall"]:
            if scope == "Overall":
                sub = wide.copy()
            else:
                sub = wide.loc[wide["body_region"] == scope].copy()
            sub = sub.dropna(subset=["scale_invariant", "scale_dependent"])
            n_pairs = int(len(sub))
            stat = float("nan")
            p_value = float("nan")
            if n_pairs >= 2:
                diffs = sub["scale_invariant"] - sub["scale_dependent"]
                if np.allclose(diffs.to_numpy(dtype=float), 0.0, atol=1e-12, rtol=0.0):
                    stat = 0.0
                    p_value = 1.0
                else:
                    result = sp_stats.wilcoxon(
                        sub["scale_invariant"].to_numpy(dtype=float),
                        sub["scale_dependent"].to_numpy(dtype=float),
                        zero_method="wilcox",
                        alternative="two-sided",
                        mode="auto",
                    )
                    stat = float(result.statistic)
                    p_value = float(result.pvalue)
            rows.append(
                {
                    "comparison_scope": scope,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "pair_unit": "(cohort, roi_name)",
                    "test_label": "cluster_median_paired_wilcoxon",
                    "median_scale_invariant": float(sub["scale_invariant"].median()) if n_pairs else np.nan,
                    "median_scale_dependent": float(sub["scale_dependent"].median()) if n_pairs else np.nan,
                    "median_paired_delta": float(
                        (sub["scale_invariant"] - sub["scale_dependent"]).median()
                    )
                    if n_pairs
                    else np.nan,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p_value,
                    "note": (
                        "Paired Wilcoxon is performed on (cohort, roi_name)-level class medians. "
                        "A one-to-one feature-level pairing between scale classes is not well-defined."
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    metrics = [
        ("median_icc", "Median ICC(3,1)", "ICC"),
        ("median_cov_pct", "Median CoV (%)", "CoV"),
    ]
    colors = {
        "scale_invariant": "#1b9e77",
        "scale_dependent": "#d95f02",
    }
    plot_df = summary_df.loc[
        (summary_df["summary_level"].isin(["body_region", "overall"]))
        & (summary_df["scale_class"].isin(["scale_invariant", "scale_dependent"]))
    ].copy()
    x_labels = [label for label in REGION_ORDER if label in set(plot_df["body_region"])]
    x_positions = np.arange(len(x_labels))
    offsets = {"scale_invariant": -0.17, "scale_dependent": 0.17}

    for axis, (metric, ylabel, title) in zip(axes, metrics):
        for scale_class in ["scale_invariant", "scale_dependent"]:
            sub = plot_df.loc[plot_df["scale_class"] == scale_class].set_index("body_region")
            y = np.array([sub.loc[label, metric] if label in sub.index else np.nan for label in x_labels], dtype=float)
            ci_low_col = "ci_low_icc" if metric == "median_icc" else "ci_low_cov_pct"
            ci_high_col = "ci_high_icc" if metric == "median_icc" else "ci_high_cov_pct"
            yerr_low = []
            yerr_high = []
            for label, val in zip(x_labels, y):
                if label in sub.index and np.isfinite(val):
                    yerr_low.append(max(val - float(sub.loc[label, ci_low_col]), 0.0))
                    yerr_high.append(max(float(sub.loc[label, ci_high_col]) - val, 0.0))
                else:
                    yerr_low.append(np.nan)
                    yerr_high.append(np.nan)
            axis.errorbar(
                x_positions + offsets[scale_class],
                y,
                yerr=np.vstack([yerr_low, yerr_high]),
                fmt="o",
                color=colors[scale_class],
                capsize=4,
                linewidth=1.5,
                label=scale_class.replace("_", " "),
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(x_labels, rotation=25, ha="right")
        axis.grid(axis="y", linestyle=":", alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Scale-invariant vs scale-dependent shape robustness", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    *,
    args: argparse.Namespace,
    argv: list[str],
    runtime_seconds: float,
    effective_bootstrap_iterations: int,
    wait_meta: dict[str, object],
    rebuilt_meta: dict[str, object],
    cohort_prep_meta: list[dict[str, object]],
    validation_meta: dict[str, object],
) -> Path:
    manifest_path = LOGS_DIR / f"46_manifest_{args.job_id}.json"
    outputs = [OUT_PARQUET, OUT_SUMMARY, OUT_PAIRWISE, OUT_FIG_PNG, OUT_FIG_PDF]
    manifest = {
        "script_path": str(Path(__file__).resolve()),
        "argv": argv,
        "host": socket.gethostname(),
        "job_id": args.job_id,
        "project_path": str(Path(__file__).resolve().parents[1]),
        "bootstrap_seed": args.seed,
        "bootstrap_iterations_requested": args.bootstrap_iterations,
        "bootstrap_iterations_effective": effective_bootstrap_iterations,
        "smoke_test": bool(args.smoke_test),
        "requested_cohorts": parse_cohorts(args.cohorts),
        "wait_gate": wait_meta,
        "input_paths": {
            "icc_results_parquet": str(ICC_RESULTS_PARQUET),
            "robustness_dir": str(ROBUSTNESS_DIR),
        },
        "input_file_checksums": {
            str(ICC_RESULTS_PARQUET): sha256_file(ICC_RESULTS_PARQUET),
        },
        "output_file_paths": [str(path) for path in outputs],
        "output_file_checksums": {str(path): sha256_file(path) for path in outputs},
        "runtime_seconds": runtime_seconds,
        "git_sha": resolve_git_sha(),
        "rebuilt_point_meta": rebuilt_meta,
        "cohort_prep_meta": cohort_prep_meta,
        "raw_vs_rebuilt_validation": validation_meta,
        "method_notes": {
            "point_estimates": (
                "Read from rebuilt icc_results.parquet only after the post-2026-04-18 wait gate "
                "passes; bootstrap recomputes ICC and CoV from patient-level robustness parquets."
            ),
            "bootstrap": (
                "Patients are resampled with replacement within cohort. Subject identities for ICC "
                "remain patient_id + course_id, so sampled patients contribute all of their courses."
            ),
            "pairwise": (
                "Pairwise inference is honestly labeled as a cluster-median paired Wilcoxon on "
                "(cohort, roi_name) units. No one-to-one feature-level pairing was imposed."
            ),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    parser.add_argument("--job-id", default=os.environ.get("JOB_ID", "manual"))
    parser.add_argument("--min-icc-mtime", default=DEFAULT_MIN_REBUILD_MTIME)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-wait-seconds", type=int, default=DEFAULT_MAX_WAIT_SECONDS)
    parser.add_argument(
        "--raw-feature-batch-size",
        type=int,
        default=4,
        help="Number of shape features to read from each cohort parquet at once.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Restrict to 4 shape features and 50 bootstrap iterations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for directory in (DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.job_id)
    requested_cohorts = parse_cohorts(args.cohorts)
    min_mtime_ts = parse_iso_utc(args.min_icc_mtime)
    n_boot = args.bootstrap_iterations

    if args.smoke_test and n_boot > 50:
        logger.info(
            "Smoke test requested: reducing bootstrap iterations from %d to 50.",
            n_boot,
        )
        n_boot = 50

    t0 = time.time()
    wait_meta = wait_for_rebuilt_icc(
        min_mtime_ts=min_mtime_ts,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        logger=logger,
    )
    point_df, rebuilt_meta = load_rebuilt_shape_points(requested_cohorts, logger)
    if args.smoke_test:
        keep_features = sorted(point_df["feature_name"].unique().tolist())[:4]
        point_df = point_df.loc[point_df["feature_name"].isin(keep_features)].copy()
        rebuilt_meta["shape_feature_names"] = keep_features
        rebuilt_meta["smoke_test_features"] = keep_features
        logger.info(
            "Smoke test enabled: restricted rebuilt point table to %d features.",
            len(keep_features),
        )

    cohort_bootstrap_data: list[CohortBootstrapData] = []
    cohort_prep_meta: list[dict[str, object]] = []
    raw_point_frames: list[pd.DataFrame] = []
    for cohort in requested_cohorts:
        cohort_features = sorted(
            point_df.loc[point_df["cohort"] == cohort, "feature_name"].unique().tolist()
        )
        if not cohort_features:
            logger.warning("Skipping bootstrap prep for %s: no shape features.", cohort)
            continue
        cohort_data, prep_meta = prepare_bootstrap_data_for_cohort(
            cohort=cohort,
            feature_names=cohort_features,
            batch_size=args.raw_feature_batch_size,
            logger=logger,
        )
        cohort_bootstrap_data.append(cohort_data)
        cohort_prep_meta.append(prep_meta)
        raw_point_frames.append(raw_point_metrics_from_bootstrap_data(cohort_data))

    raw_point_df = pd.concat(raw_point_frames, ignore_index=True) if raw_point_frames else pd.DataFrame()
    validation_meta = validate_raw_vs_rebuilt(point_df, raw_point_df, logger)

    point_df.to_parquet(OUT_PARQUET, index=False)

    cohort_roi_df = build_cohort_roi_point_summary(point_df)
    summary_df = summarise_cell_medians(cohort_roi_df)
    ci_df = bootstrap_patient_cluster_cis(
        cohort_bootstrap_data=cohort_bootstrap_data,
        n_boot=n_boot,
        seed=args.seed,
        logger=logger,
    )
    summary_df = summary_df.merge(
        ci_df,
        on=["summary_level", "body_region", "scale_class"],
        how="left",
    )
    summary_export = pd.concat([cohort_roi_df, summary_df], ignore_index=True, sort=False)
    summary_export.to_csv(OUT_SUMMARY, index=False)

    pairwise_df = compute_pairwise_cluster_median_tests(cohort_roi_df)
    pairwise_df.to_csv(OUT_PAIRWISE, index=False)

    plot_summary(summary_df)

    manifest_path = write_manifest(
        args=args,
        argv=sys.argv,
        runtime_seconds=time.time() - t0,
        effective_bootstrap_iterations=n_boot,
        wait_meta=wait_meta,
        rebuilt_meta=rebuilt_meta,
        cohort_prep_meta=cohort_prep_meta,
        validation_meta=validation_meta,
    )

    logger.info("Wrote %s", OUT_PARQUET)
    logger.info("Wrote %s", OUT_SUMMARY)
    logger.info("Wrote %s", OUT_PAIRWISE)
    logger.info("Wrote %s and %s", OUT_FIG_PNG, OUT_FIG_PDF)
    logger.info("Wrote %s", manifest_path)
    logger.info("Step 46 fixed completed successfully; RSS %.2f GB.", rss_gb())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive exit path
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(2)
