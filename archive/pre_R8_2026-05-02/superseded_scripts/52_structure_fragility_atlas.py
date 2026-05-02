#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 52: Structure-level fragility atlas.

PURPOSE
-------
Disaggregate the region x family NTCV variance story to the
structure x family x region level, then cross-reference that atlas against
structure-level robustness prevalence and segmentation QC burden.

INPUTS
------
Authoritative inputs:
  /home/kgs24/rtpipeline_manuscript/analysis/tables/perturbation_variance_decomposition.csv
  /home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/tables/segmentation_qc.csv

Local mirrors used when present:
  <project>/analysis/tables/perturbation_variance_decomposition.csv
  <project>/analysis/tables/segmentation_qc.csv

OUTPUTS
-------
  analysis/data/structure_fragility_atlas.parquet
  analysis/tables/structure_fragility_summary.csv
  analysis/tables/worst_structure_by_family.csv
  analysis/figures/figure_fragility_atlas.png
  analysis/figures/figure_fragility_atlas.pdf
  analysis/logs/52_manifest_<JOBID>.json

NOTES
-----
- The variance-decomposition CSV available in this workspace is already
  aggregated at feature x structure x cohort level. When no subject-level
  identifiers are available, the script falls back to cohort-resampled
  feature-row bootstrap for the volume-component CI and records that
  limitation explicitly in the manifest.
- Smoke-test mode synthesizes a deterministic ICC table when the ICC parquet is
  unavailable from the current worker.
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SEED_DEFAULT = 2604
BOOTSTRAP_DEFAULT = 1000
FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
ROBUST_ICC_THRESHOLD = 0.90


def resolve_analysis_dir() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "analysis":
        return here.parent
    candidate = here.parents[1] / "analysis"
    if candidate.exists():
        return candidate
    return here.parent


ANALYSIS_DIR = resolve_analysis_dir()
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"

LOCAL_VAR_DECOMP = TABLES_DIR / "perturbation_variance_decomposition.csv"
LOCAL_SEG_QC = TABLES_DIR / "segmentation_qc.csv"
LOCAL_ICC = DATA_DIR / "icc_results.parquet"
ARGOS_VAR_DECOMP = Path(
    "/home/kgs24/rtpipeline_manuscript/analysis/tables/perturbation_variance_decomposition.csv"
)
ARGOS_SEG_QC = Path(
    "/home/kgs24/rtpipeline_manuscript/analysis/tables/segmentation_qc.csv"
)
ARGOS_ICC = Path(
    "/home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet"
)

OUT_PARQUET = DATA_DIR / "structure_fragility_atlas.parquet"
OUT_SUMMARY = TABLES_DIR / "structure_fragility_summary.csv"
OUT_WORST = TABLES_DIR / "worst_structure_by_family.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_fragility_atlas.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_fragility_atlas.pdf"


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
    logger = logging.getLogger("script52")
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


def safe_is_readable(path: Path, timeout_seconds: float = 1.0) -> bool:
    if not str(path).startswith("/home/"):
        return path.exists()
    try:
        result = subprocess.run(
            ["bash", "-lc", f"test -r {shlex.quote(str(path))}"],
            timeout=timeout_seconds,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_first_readable(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if safe_is_readable(candidate):
            return candidate
    return None


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


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    low = float(np.nanpercentile(values, 100.0 * alpha / 2.0))
    high = float(np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def safe_nanpercentile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def q95(values: pd.Series) -> float:
    return safe_nanpercentile(values, 95.0)


def resolve_column(columns: list[str], options: list[str], label: str) -> str:
    for option in options:
        if option in columns:
            return option
    raise RuntimeError(f"Could not resolve {label} column from candidates: {options}")


def load_variance_decomposition(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = resolve_first_readable([LOCAL_VAR_DECOMP, ARGOS_VAR_DECOMP])
    if input_path is None:
        raise FileNotFoundError(
            "Could not resolve perturbation_variance_decomposition.csv from "
            f"{LOCAL_VAR_DECOMP} or {ARGOS_VAR_DECOMP}."
        )
    df = pd.read_csv(input_path)
    required = {
        "cohort",
        "body_region",
        "roi_name",
        "feature_name",
        "pct_T",
        "pct_C",
        "pct_V",
        "pct_residual",
        "feature_family",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Variance decomposition CSV missing columns: {sorted(missing)}")
    df = df.loc[df["feature_family"].isin(FAMILY_ORDER)].copy()
    meta = {
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_cohorts": int(df["cohort"].nunique()),
        "n_structures": int(df["roi_name"].nunique()),
        "n_features": int(df["feature_name"].nunique()),
    }
    logger.info(
        "Loaded variance decomposition rows=%d, cohorts=%d, structures=%d.",
        len(df),
        df["cohort"].nunique(),
        df["roi_name"].nunique(),
    )
    return df, meta


def load_segmentation_qc(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = resolve_first_readable([LOCAL_SEG_QC, ARGOS_SEG_QC])
    if input_path is None:
        raise FileNotFoundError(
            "Could not resolve segmentation_qc.csv from "
            f"{LOCAL_SEG_QC} or {ARGOS_SEG_QC}."
        )
    qc_df = pd.read_csv(input_path)
    required = {"body_region", "structure", "is_empty", "is_cropped", "n_components"}
    missing = required.difference(qc_df.columns)
    if missing:
        raise RuntimeError(f"Segmentation QC CSV missing columns: {sorted(missing)}")

    qc_df["is_empty"] = qc_df["is_empty"].astype(bool)
    qc_df["is_cropped"] = qc_df["is_cropped"].astype(bool)
    qc_df["is_multicomponent"] = qc_df["n_components"].fillna(0).astype(float) > 1.0
    summary = (
        qc_df.groupby(["body_region", "structure"], sort=False)
        .agg(
            qc_cases=("structure", "size"),
            empty_rate_pct=("is_empty", lambda x: 100.0 * float(np.mean(x))),
            cropped_rate_pct=("is_cropped", lambda x: 100.0 * float(np.mean(x))),
            multicomponent_rate_pct=("is_multicomponent", lambda x: 100.0 * float(np.mean(x))),
        )
        .reset_index()
        .rename(columns={"structure": "roi_name"})
    )
    meta = {
        "input_path": str(input_path),
        "n_rows": int(len(qc_df)),
        "n_structures": int(summary["roi_name"].nunique()),
    }
    logger.info(
        "Loaded segmentation QC rows=%d, structures=%d.",
        len(qc_df),
        summary["roi_name"].nunique(),
    )
    return summary, meta


def synthesise_smoke_icc(var_df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        var_df[["cohort", "body_region", "roi_name", "feature_name", "feature_family", "pct_V", "pct_T", "pct_C"]]
        .drop_duplicates()
        .copy()
    )
    approx = 1.0 - (keep["pct_V"] / 120.0) - (keep["pct_T"] / 200.0) - (keep["pct_C"] / 200.0)
    keep["icc_31"] = approx.clip(lower=0.05, upper=0.995)
    return keep[
        ["cohort", "body_region", "roi_name", "feature_name", "feature_family", "icc_31"]
    ].copy()


def load_icc_results(
    var_df: pd.DataFrame,
    smoke_test: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    icc_candidates = [LOCAL_ICC] if smoke_test else [LOCAL_ICC, ARGOS_ICC]
    input_path = resolve_first_readable(icc_candidates)
    if input_path is None:
        if not smoke_test:
            raise FileNotFoundError(
                f"Could not resolve icc_results.parquet from {LOCAL_ICC} or {ARGOS_ICC}."
            )
        icc_df = synthesise_smoke_icc(var_df)
        meta = {
            "input_path": None,
            "source_type": "synthetic_smoke_icc",
            "row_count": int(len(icc_df)),
            "mtime_utc": None,
            "post_rebuild_job_3560738_verified": False,
            "version_note": "Synthetic ICC table generated from variance rows for smoke-test only.",
        }
        logger.info(
            "Smoke test: ICC parquet unavailable; synthesised %d ICC rows.",
            len(icc_df),
        )
        return icc_df, meta

    icc_df = pd.read_parquet(input_path)
    columns = list(icc_df.columns)
    cohort_col = resolve_column(columns, ["cohort"], "cohort")
    region_col = resolve_column(columns, ["body_region", "region"], "body region")
    structure_col = resolve_column(columns, ["roi_name", "structure"], "structure")
    family_col = resolve_column(columns, ["feature_family", "family"], "feature family")
    icc_col = resolve_column(columns, ["icc_31", "icc31", "icc", "icc_value"], "ICC value")

    out = icc_df[
        [cohort_col, region_col, structure_col, "feature_name", family_col, icc_col]
    ].copy()
    out.columns = ["cohort", "body_region", "roi_name", "feature_name", "feature_family", "icc_31"]
    out = out.loc[out["feature_family"].isin(FAMILY_ORDER)].copy()
    stat_result = input_path.stat()
    meta = {
        "input_path": str(input_path),
        "source_type": "icc_results_parquet",
        "row_count": int(len(out)),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat_result.st_mtime)),
        "post_rebuild_job_3560738_verified": False,
        "version_note": (
            "File chosen by first-readable-path resolution; job-state verification for "
            "3560738 was not available from this worker, so the source mtime is recorded "
            "instead of asserting post-rebuild provenance."
        ),
    }
    logger.info("Loaded ICC rows=%d from %s.", len(out), input_path)
    return out, meta


def aggregate_icc_summary(icc_df: pd.DataFrame) -> pd.DataFrame:
    icc_df = icc_df.copy()
    icc_df["robust_flag"] = icc_df["icc_31"].astype(float) >= ROBUST_ICC_THRESHOLD
    return (
        icc_df.groupby(["body_region", "roi_name", "feature_family"], sort=False)
        .agg(
            robust_feature_fraction=("robust_flag", "mean"),
            median_icc_31=("icc_31", "median"),
            n_icc_rows=("icc_31", "size"),
            n_icc_features=("feature_name", "nunique"),
            n_icc_cohorts=("cohort", "nunique"),
        )
        .reset_index()
    )


def feature_row_bootstrap_ci(
    group_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    cohorts = group_df["cohort"].drop_duplicates().to_numpy()
    for _ in range(n_boot):
        sampled_cohorts = rng.choice(cohorts, size=len(cohorts), replace=True)
        sampled_parts: list[pd.DataFrame] = []
        for cohort_id in sampled_cohorts:
            cohort_df = group_df.loc[group_df["cohort"] == cohort_id]
            take = rng.integers(0, len(cohort_df), size=len(cohort_df))
            sampled_parts.append(cohort_df.iloc[take])
        boot_df = pd.concat(sampled_parts, ignore_index=True)
        values.append(float(np.nanmedian(boot_df["pct_V"].to_numpy(dtype=float))))
    return percentile_ci(np.asarray(values, dtype=float))


def build_atlas(
    var_df: pd.DataFrame,
    icc_summary: pd.DataFrame,
    qc_summary: pd.DataFrame,
    n_boot: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    bootstrap_mode = "feature_row_fallback_no_subject_ids"
    grouped = var_df.groupby(["body_region", "roi_name", "feature_family"], sort=False)
    total_groups = grouped.ngroups
    for idx, ((region, structure, family), group_df) in enumerate(grouped, start=1):
        ci_low, ci_high = feature_row_bootstrap_ci(
            group_df=group_df,
            n_boot=n_boot,
            seed=seed + idx,
        )
        rows.append(
            {
                "body_region": region,
                "roi_name": structure,
                "feature_family": family,
                "n_rows": int(len(group_df)),
                "n_features": int(group_df["feature_name"].nunique()),
                "n_cohorts": int(group_df["cohort"].nunique()),
                "median_volume_variance_pct": float(group_df["pct_V"].median()),
                "median_translation_variance_pct": float(group_df["pct_T"].median()),
                "median_contour_variance_pct": float(group_df["pct_C"].median()),
                "median_residual_variance_pct": float(group_df["pct_residual"].median()),
                "p95_volume_variance_pct": q95(group_df["pct_V"]),
                "p95_translation_variance_pct": q95(group_df["pct_T"]),
                "p95_contour_variance_pct": q95(group_df["pct_C"]),
                "p95_residual_variance_pct": q95(group_df["pct_residual"]),
                "volume_ci_low": ci_low,
                "volume_ci_high": ci_high,
            }
        )
        if idx % 25 == 0 or idx == total_groups:
            logger.info("Atlas aggregation %d/%d groups complete.", idx, total_groups)

    atlas = pd.DataFrame(rows)
    atlas = atlas.merge(
        icc_summary,
        on=["body_region", "roi_name", "feature_family"],
        how="left",
    )
    atlas = atlas.merge(
        qc_summary,
        on=["body_region", "roi_name"],
        how="left",
    )
    atlas["robust_feature_fraction"] = atlas["robust_feature_fraction"].fillna(np.nan)

    atlas["bottom_decile_cutoff"] = atlas.groupby(
        ["body_region", "feature_family"], sort=False
    )["robust_feature_fraction"].transform(lambda x: safe_nanpercentile(x, 10.0))
    atlas["top_decile_cutoff"] = atlas.groupby(
        ["body_region", "feature_family"], sort=False
    )["robust_feature_fraction"].transform(lambda x: safe_nanpercentile(x, 90.0))
    atlas["volume_tail_cutoff"] = atlas.groupby(
        ["body_region", "feature_family"], sort=False
    )["median_volume_variance_pct"].transform(lambda x: safe_nanpercentile(x, 90.0))
    atlas["volume_p95_tail_cutoff"] = atlas.groupby(
        ["body_region", "feature_family"], sort=False
    )["p95_volume_variance_pct"].transform(lambda x: safe_nanpercentile(x, 90.0))

    atlas["empty_rate_cutoff"] = atlas.groupby("body_region", sort=False)["empty_rate_pct"].transform(
        lambda x: safe_nanpercentile(x, 75.0)
    )
    atlas["cropped_rate_cutoff"] = atlas.groupby("body_region", sort=False)["cropped_rate_pct"].transform(
        lambda x: safe_nanpercentile(x, 75.0)
    )
    atlas["multicomponent_rate_cutoff"] = atlas.groupby(
        "body_region", sort=False
    )["multicomponent_rate_pct"].transform(lambda x: safe_nanpercentile(x, 75.0))

    atlas["bottom_decile_fragility_flag"] = (
        atlas["robust_feature_fraction"] <= atlas["bottom_decile_cutoff"]
    )
    atlas["qc_problem_flag"] = (
        (atlas["empty_rate_pct"] >= atlas["empty_rate_cutoff"])
        | (atlas["cropped_rate_pct"] >= atlas["cropped_rate_cutoff"])
        | (atlas["multicomponent_rate_pct"] >= atlas["multicomponent_rate_cutoff"])
    )
    atlas["variance_problem_flag"] = (
        atlas["bottom_decile_fragility_flag"]
        | (atlas["median_volume_variance_pct"] >= atlas["volume_tail_cutoff"])
        | (atlas["p95_volume_variance_pct"] >= atlas["volume_p95_tail_cutoff"])
    )
    atlas["joint_qc_variance_flag"] = atlas["qc_problem_flag"] & atlas["variance_problem_flag"]
    atlas["robust_feature_fraction_pct"] = 100.0 * atlas["robust_feature_fraction"]

    atlas["family_order"] = atlas["feature_family"].map(
        {name: idx for idx, name in enumerate(FAMILY_ORDER)}
    )
    atlas = atlas.sort_values(
        ["body_region", "family_order", "robust_feature_fraction", "median_volume_variance_pct", "roi_name"],
        ascending=[True, True, True, False, True],
    ).drop(columns="family_order")
    return atlas, {"bootstrap_mode": bootstrap_mode}


def build_rank_tables(atlas_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[pd.DataFrame] = []
    worst_rows: list[dict[str, object]] = []
    for (region, family), sub in atlas_df.groupby(["body_region", "feature_family"], sort=False):
        ranked = sub.sort_values(
            ["robust_feature_fraction", "median_volume_variance_pct", "roi_name"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        n_structures = len(ranked)
        bottom_count = max(1, int(math.ceil(0.10 * n_structures)))

        bottom = ranked.head(min(10, n_structures)).copy()
        bottom["rank_direction"] = "bottom"
        bottom["rank"] = np.arange(1, len(bottom) + 1)
        top = ranked.sort_values(
            ["robust_feature_fraction", "median_volume_variance_pct", "roi_name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        top = top.head(min(10, n_structures)).copy()
        top["rank_direction"] = "top"
        top["rank"] = np.arange(1, len(top) + 1)
        summary_rows.extend([bottom, top])

        worst = ranked.iloc[0].to_dict()
        worst["bottom_decile_member"] = bool(ranked.index[0] < bottom_count)
        worst["bottom_decile_group_size"] = int(bottom_count)
        worst["n_structures_in_family_region"] = int(n_structures)
        worst_rows.append(worst)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    worst_df = pd.DataFrame(worst_rows)
    return summary_df, worst_df


def plot_fragility_atlas(atlas_df: pd.DataFrame) -> None:
    regions = ["Brain", "Thorax", "Pelvis"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), constrained_layout=True)
    image = None
    for axis, region in zip(axes, regions):
        sub = atlas_df.loc[atlas_df["body_region"] == region].copy()
        if sub.empty:
            axis.axis("off")
            continue
        structure_order = (
            sub.groupby("roi_name")["median_volume_variance_pct"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        pivot = (
            sub.pivot_table(
                index="roi_name",
                columns="feature_family",
                values="median_volume_variance_pct",
                aggfunc="first",
            )
            .reindex(index=structure_order, columns=FAMILY_ORDER)
        )
        data = pivot.to_numpy(dtype=float)
        image = axis.imshow(data, aspect="auto", cmap="viridis")
        axis.set_title(region)
        axis.set_xticks(np.arange(len(FAMILY_ORDER)))
        axis.set_xticklabels(FAMILY_ORDER, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels(pivot.index, fontsize=8)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Median volume variance component (%)")
    fig.suptitle("Structure-level fragility atlas: median volume variance by family", y=1.02)
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    *,
    job_id: str,
    argv: list[str],
    runtime_seconds: float,
    bootstrap_iterations: int,
    seed: int,
    variance_meta: dict[str, object],
    qc_meta: dict[str, object],
    icc_meta: dict[str, object],
    atlas_meta: dict[str, object],
    atlas_df: pd.DataFrame,
) -> Path:
    manifest_path = LOGS_DIR / f"52_manifest_{job_id}.json"
    input_checksums: dict[str, str] = {}
    for meta in (variance_meta, qc_meta, icc_meta):
        path_str = meta.get("input_path")
        if path_str:
            path = Path(path_str)
            if path.exists():
                input_checksums[str(path)] = sha256_file(path)

    overall_rows = atlas_df.sort_values(
        ["robust_feature_fraction", "median_volume_variance_pct"], ascending=[True, False]
    ).head(5)
    manifest = {
        "script_path": str(Path(__file__).resolve()),
        "argv": argv,
        "host": socket.gethostname(),
        "job_id": job_id,
        "project_path": str(Path(__file__).resolve().parents[1]),
        "input_file_sha256": input_checksums,
        "output_paths": [
            str(OUT_PARQUET),
            str(OUT_SUMMARY),
            str(OUT_WORST),
            str(OUT_FIG_PNG),
            str(OUT_FIG_PDF),
        ],
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "git_sha": resolve_git_sha(),
        "variance_meta": variance_meta,
        "segmentation_qc_meta": qc_meta,
        "icc_meta": icc_meta,
        "atlas_meta": atlas_meta,
        "method_notes": {
            "robust_feature_fraction": "fraction of ICC rows with ICC(3,1) >= 0.90 within structure x family x region",
            "qc_problem_flag": "empty/cropped/multicomponent rate above the region-specific 75th percentile among atlas structures",
            "variance_problem_flag": "bottom robust-feature decile or top volume-variance tail within region x family",
        },
        "lowest_robustness_examples": json.loads(
            overall_rows[
                [
                    "body_region",
                    "roi_name",
                    "feature_family",
                    "robust_feature_fraction",
                    "median_volume_variance_pct",
                    "joint_qc_variance_flag",
                ]
            ].to_json(orient="records")
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Restrict the atlas to a small subset of structures and synthesise ICC rows if needed; cap bootstrap iterations at 50.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_DEFAULT,
        help="Number of bootstrap iterations used for the volume-component CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Random seed used for the bootstrap.",
    )
    parser.add_argument(
        "--job-id",
        default=os.environ.get("JOB_ID", "local"),
        help="Job identifier used in logs and manifest output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for directory in (DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.job_id)
    n_boot = args.bootstrap_iterations
    if args.smoke_test and n_boot > 50:
        logger.info(
            "Smoke test requested: reducing bootstrap iterations from %d to 50.",
            n_boot,
        )
        n_boot = 50

    t0 = time.time()
    logger.info("Starting step 52 structure fragility atlas.")
    var_df, variance_meta = load_variance_decomposition(logger)
    qc_summary, qc_meta = load_segmentation_qc(logger)

    if args.smoke_test:
        keep_regions = ["Brain", "Thorax", "Pelvis"]
        keep_structures = []
        for region in keep_regions:
            keep_structures.extend(
                sorted(var_df.loc[var_df["body_region"] == region, "roi_name"].unique())[:2]
            )
        var_df = var_df.loc[var_df["roi_name"].isin(set(keep_structures))].copy()
        logger.info(
            "Smoke test enabled: restricted variance atlas to %d structures.",
            len(set(keep_structures)),
        )

    icc_df, icc_meta = load_icc_results(var_df=var_df, smoke_test=args.smoke_test, logger=logger)
    icc_summary = aggregate_icc_summary(icc_df)
    atlas_df, atlas_meta = build_atlas(
        var_df=var_df,
        icc_summary=icc_summary,
        qc_summary=qc_summary,
        n_boot=n_boot,
        seed=args.seed,
        logger=logger,
    )
    summary_df, worst_df = build_rank_tables(atlas_df)

    atlas_df.to_parquet(OUT_PARQUET, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    worst_df.to_csv(OUT_WORST, index=False)
    plot_fragility_atlas(atlas_df)

    manifest_path = write_manifest(
        job_id=args.job_id,
        argv=sys.argv,
        runtime_seconds=time.time() - t0,
        bootstrap_iterations=n_boot,
        seed=args.seed,
        variance_meta=variance_meta,
        qc_meta=qc_meta,
        icc_meta=icc_meta,
        atlas_meta=atlas_meta,
        atlas_df=atlas_df,
    )

    logger.info("Wrote %s", OUT_PARQUET)
    logger.info("Wrote %s", OUT_SUMMARY)
    logger.info("Wrote %s", OUT_WORST)
    logger.info("Wrote %s and %s", OUT_FIG_PNG, OUT_FIG_PDF)
    logger.info("Wrote %s", manifest_path)
    logger.info("Step 52 complete in %.1fs.", time.time() - t0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(2)
