#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 50: 107-feature bridge sensitivity for the NSCLC_IO within-patient CoV analysis.

PURPOSE
-------
Restrict the section 30 human-vs-AI within-patient CoV comparison to the
107-feature overlap used by step 33, then compare that overlap subset against
the original full-feature result to test whether the AI advantage survives the
feature-universe bridge.

INPUTS
------
Authoritative inputs:
  /home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_within_patient_deviations.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/data/sigma_calibration_nsclc_io.parquet

Local mirror used when present:
  <project>/analysis/data/nsclc_io_within_patient_deviations.parquet

OUTPUTS
-------
  analysis/data/shared_feature_bridge_nsclc_io.parquet
  analysis/tables/shared_feature_bridge_summary.csv
  analysis/tables/shared_feature_bridge_vs_full.csv
  analysis/figures/figure_shared_bridge.png
  analysis/figures/figure_shared_bridge.pdf
  analysis/logs/50_manifest_<JOBID>.json

NOTES
-----
- Positive paired deltas mean Human CoV > AI CoV, which favours AI.
- The input deviations parquet currently lacks `course_id`; when absent, the
  script falls back to `subject = patient_id` and records that choice.
- Smoke-test mode uses a deterministic synthetic overlap subset if the step-33
  parquet is not readable from the current worker.
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
from scipy import stats as sp_stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SEED_DEFAULT = 2604
BOOTSTRAP_DEFAULT = 1000
FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
COHORT_NAME = "NSCLC_Interobserver"
BODY_REGION = "Thorax"


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

LOCAL_DEVIATIONS = DATA_DIR / "nsclc_io_within_patient_deviations.parquet"
LOCAL_SIGMA33 = DATA_DIR / "sigma_calibration_nsclc_io.parquet"
ARGOS_DEVIATIONS = Path(
    "/home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_within_patient_deviations.parquet"
)
ARGOS_SIGMA33 = Path(
    "/home/kgs24/rtpipeline_manuscript/analysis/data/sigma_calibration_nsclc_io.parquet"
)

OUT_PARQUET = DATA_DIR / "shared_feature_bridge_nsclc_io.parquet"
OUT_SUMMARY = TABLES_DIR / "shared_feature_bridge_summary.csv"
OUT_COMPARISON = TABLES_DIR / "shared_feature_bridge_vs_full.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_shared_bridge.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_shared_bridge.pdf"


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
    logger = logging.getLogger("script50")
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


def build_subject_column(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if "subject" in out.columns:
        return out, "subject"
    if {"patient_id", "course_id"}.issubset(out.columns):
        out["subject"] = out["patient_id"].astype(str) + "_" + out["course_id"].astype(str)
        return out, "patient_id_plus_course_id"
    if "patient_id" in out.columns:
        logger.info(
            "Input parquet has no course_id column; using patient_id as subject identifier."
        )
        out["subject"] = out["patient_id"].astype(str)
        return out, "patient_id_only_fallback"
    raise RuntimeError("Input parquet is missing both subject and patient_id columns.")


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    low = float(np.nanpercentile(values, 100.0 * alpha / 2.0))
    high = float(np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def wilcoxon_with_guard(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return (float("nan"), float("nan"))
    if np.allclose(clean, 0.0):
        return (0.0, 1.0)
    stat, p_value = sp_stats.wilcoxon(clean, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_value)


def load_deviations(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = resolve_first_readable([LOCAL_DEVIATIONS, ARGOS_DEVIATIONS])
    if input_path is None:
        raise FileNotFoundError(
            "Could not resolve nsclc_io_within_patient_deviations.parquet from "
            f"{LOCAL_DEVIATIONS} or {ARGOS_DEVIATIONS}."
        )
    df = pd.read_parquet(input_path)
    required = {"patient_id", "feature_name", "family", "human_cov_pct", "ai_cov_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Deviation parquet missing columns: {sorted(missing)}")

    df, subject_mode = build_subject_column(df, logger)
    df = df.loc[df["family"].isin(FAMILY_ORDER)].copy()
    df["feature_name"] = df["feature_name"].astype(str)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    df["cohort"] = COHORT_NAME
    df["body_region"] = BODY_REGION
    df["paired_delta_cov_pct"] = df["human_cov_pct"] - df["ai_cov_pct"]
    meta = {
        "input_path": str(input_path),
        "subject_mode": subject_mode,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(df["feature_name"].nunique()),
    }
    return df, meta


def load_overlap_feature_set(
    deviations_df: pd.DataFrame,
    smoke_test: bool,
    logger: logging.Logger,
) -> tuple[set[str], dict[str, object]]:
    sigma_candidates = [LOCAL_SIGMA33] if smoke_test else [LOCAL_SIGMA33, ARGOS_SIGMA33]
    sigma_path = resolve_first_readable(sigma_candidates)
    if sigma_path is None:
        if not smoke_test:
            raise FileNotFoundError(
                "Could not resolve sigma_calibration_nsclc_io.parquet from "
                f"{LOCAL_SIGMA33} or {ARGOS_SIGMA33}."
            )
        synthetic_features = set(
            sorted(deviations_df["feature_name"].astype(str).unique())[:12]
        )
        logger.info(
            "Smoke test: sigma calibration parquet unavailable; using a deterministic "
            "synthetic overlap subset of %d features.",
            len(synthetic_features),
        )
        return synthetic_features, {
            "source_path": None,
            "source_type": "synthetic_smoke_subset",
            "feature_count": int(len(synthetic_features)),
        }

    sigma_df = pd.read_parquet(sigma_path, columns=["feature_name"])
    overlap = set(sigma_df["feature_name"].astype(str).dropna().unique())
    logger.info(
        "Loaded %d overlap features from %s.",
        len(overlap),
        sigma_path,
    )
    return overlap, {
        "source_path": str(sigma_path),
        "source_type": "step33_sigma_calibration",
        "feature_count": int(len(overlap)),
    }


def resample_hierarchical_subjects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sampled_blocks: list[pd.DataFrame] = []
    if "body_region" in df.columns:
        stratum_groups = df.groupby("body_region", sort=False)
    else:
        stratum_groups = [("__all__", df)]

    for _, stratum_df in stratum_groups:
        cohort_ids = stratum_df["cohort"].drop_duplicates().to_list()
        sampled_cohorts = rng.choice(cohort_ids, size=len(cohort_ids), replace=True)
        for cohort_draw, cohort_id in enumerate(sampled_cohorts):
            cohort_df = stratum_df.loc[stratum_df["cohort"] == cohort_id]
            subject_ids = cohort_df["subject"].drop_duplicates().to_list()
            sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
            for subject_draw, subject_id in enumerate(sampled_subjects):
                subject_df = cohort_df.loc[cohort_df["subject"] == subject_id].copy()
                subject_df["subject"] = f"{subject_id}__boot{cohort_draw}_{subject_draw}"
                sampled_blocks.append(subject_df)
    return pd.concat(sampled_blocks, ignore_index=True)


def summarise_set(df: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("overall", df)]
    for family in FAMILY_ORDER:
        family_df = df.loc[df["family"] == family].copy()
        if not family_df.empty:
            groups.append((family, family_df))

    for family_label, sub in groups:
        deltas = sub["paired_delta_cov_pct"].to_numpy(dtype=float)
        stat, p_value = wilcoxon_with_guard(deltas)
        rows.append(
            {
                "analysis_set": analysis_set,
                "family": family_label,
                "n_pairs": int(len(sub)),
                "n_subjects": int(sub["subject"].nunique()),
                "n_features": int(sub["feature_name"].nunique()),
                "human_median_cov_pct": float(np.nanmedian(sub["human_cov_pct"].to_numpy(dtype=float))),
                "ai_median_cov_pct": float(np.nanmedian(sub["ai_cov_pct"].to_numpy(dtype=float))),
                "median_paired_delta_pct": float(np.nanmedian(deltas)),
                "pct_pairs_ai_lower": float(100.0 * np.mean(deltas > 0.0)),
                "wilcoxon_stat": stat,
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_cis(
    df: pd.DataFrame,
    analysis_set: str,
    n_boot: int,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    families = ["overall"] + FAMILY_ORDER
    for iteration in range(n_boot):
        boot_df = resample_hierarchical_subjects(df, rng)
        for family in families:
            sub = boot_df if family == "overall" else boot_df.loc[boot_df["family"] == family]
            if sub.empty:
                continue
            records.append(
                {
                    "analysis_set": analysis_set,
                    "family": family,
                    "bootstrap_iteration": iteration,
                    "median_paired_delta_pct": float(
                        np.nanmedian(sub["paired_delta_cov_pct"].to_numpy(dtype=float))
                    ),
                }
            )
        if (iteration + 1) % 100 == 0 or iteration == n_boot - 1:
            logger.info(
                "Bootstrap %s %d/%d complete.",
                analysis_set,
                iteration + 1,
                n_boot,
            )
    records_df = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    for (set_name, family), sub in records_df.groupby(["analysis_set", "family"], sort=False):
        ci_low, ci_high = percentile_ci(sub["median_paired_delta_pct"].to_numpy(dtype=float))
        rows.append(
            {
                "analysis_set": set_name,
                "family": family,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_comparison(full_summary: pd.DataFrame, overlap_summary: pd.DataFrame) -> pd.DataFrame:
    merged = full_summary.merge(
        overlap_summary,
        on="family",
        how="inner",
        suffixes=("_full", "_overlap"),
    )
    full_delta = merged["median_paired_delta_pct_full"].to_numpy(dtype=float)
    overlap_delta = merged["median_paired_delta_pct_overlap"].to_numpy(dtype=float)
    merged["effect_size_retention_fraction"] = np.where(
        np.isclose(full_delta, 0.0),
        np.nan,
        overlap_delta / full_delta,
    )
    merged["effect_size_retention_pct"] = 100.0 * merged["effect_size_retention_fraction"]
    merged["delta_change_pct_points"] = overlap_delta - full_delta
    merged["direction_consistent"] = np.sign(full_delta) == np.sign(overlap_delta)
    merged["ai_advantage_persists_overlap"] = (
        (merged["median_paired_delta_pct_overlap"] > 0.0)
        & (merged["wilcoxon_p_overlap"] < 0.05)
    )
    ordered = ["overall"] + FAMILY_ORDER
    merged["family_order"] = merged["family"].map({name: idx for idx, name in enumerate(ordered)})
    merged = merged.sort_values("family_order").drop(columns="family_order")
    return merged


def make_bridge_output(
    overlap_df: pd.DataFrame,
    overlap_feature_meta: dict[str, object],
) -> pd.DataFrame:
    out = overlap_df.copy()
    out["overlap_feature_source"] = overlap_feature_meta["source_type"]
    return out[
        [
            "cohort",
            "body_region",
            "subject",
            "patient_id",
            "feature_name",
            "family",
            "human_cov_pct",
            "ai_cov_pct",
            "paired_delta_cov_pct",
            "human_med_abs_dev",
            "ai_med_abs_dev",
            "human_median",
            "ai_median",
            "human_ai_diff_pct",
            "overlap_feature_source",
        ]
    ].copy()


def plot_bridge_comparison(full_df: pd.DataFrame, overlap_df: pd.DataFrame) -> None:
    ordered_families = ["overall"] + FAMILY_ORDER
    plot_rows: list[pd.DataFrame] = []
    for analysis_set, frame in (("full", full_df), ("overlap_107", overlap_df)):
        tmp = frame[["family", "paired_delta_cov_pct"]].copy()
        tmp["analysis_set"] = analysis_set
        plot_rows.append(tmp)
        overall = frame[["paired_delta_cov_pct"]].copy()
        overall["family"] = "overall"
        overall["analysis_set"] = analysis_set
        plot_rows.append(overall)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    colors = {"full": "#bab0ac", "overlap_107": "#4c78a8"}
    fig, ax = plt.subplots(figsize=(14, 6))
    base_positions = np.arange(len(ordered_families))
    width = 0.32

    for idx, analysis_set in enumerate(["full", "overlap_107"]):
        positions = base_positions + (-width / 2 if idx == 0 else width / 2)
        box_data = [
            plot_df.loc[
                (plot_df["family"] == family) & (plot_df["analysis_set"] == analysis_set),
                "paired_delta_cov_pct",
            ].to_numpy(dtype=float)
            for family in ordered_families
        ]
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "black", "linewidth": 0.9},
            capprops={"color": "black", "linewidth": 0.9},
            boxprops={"linewidth": 0.9, "edgecolor": "black"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[analysis_set])
            patch.set_alpha(0.85)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(["overall", "shape", "first", "glcm", "glrlm", "glszm", "gldm", "ngtdm"])
    ax.set_ylabel("Paired delta CoV% (Human - AI)")
    ax.set_title("Shared-feature bridge: full vs 107-feature overlap")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors["full"], alpha=0.85, label="Full section-30 feature set"),
            plt.Rectangle((0, 0), 1, 1, color=colors["overlap_107"], alpha=0.85, label="107-feature overlap subset"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
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
    deviations_meta: dict[str, object],
    overlap_meta: dict[str, object],
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> Path:
    manifest_path = LOGS_DIR / f"50_manifest_{job_id}.json"
    input_checksums: dict[str, str] = {}
    input_paths = [deviations_meta.get("input_path"), overlap_meta.get("source_path")]
    for path_str in input_paths:
        if path_str:
            path = Path(path_str)
            if path.exists():
                input_checksums[str(path)] = sha256_file(path)

    overall_row = comparison_df.loc[comparison_df["family"] == "overall"].iloc[0].to_dict()
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
            str(OUT_COMPARISON),
            str(OUT_FIG_PNG),
            str(OUT_FIG_PDF),
        ],
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "git_sha": resolve_git_sha(),
        "deviations_meta": deviations_meta,
        "overlap_feature_meta": overlap_meta,
        "overall_bridge_summary": overall_row,
        "method_notes": {
            "delta_definition": "paired_delta_cov_pct = human_cov_pct - ai_cov_pct",
            "comparison_sets": "full section-30 feature universe vs the 107-feature overlap used by step 33",
            "bootstrap_design": "hierarchical cohort -> subject bootstrap; in NSCLC_IO this collapses to a single-cohort subject bootstrap",
        },
        "summary_rows": json.loads(summary_df.to_json(orient="records")),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a smaller deterministic overlap subset when step-33 inputs are unavailable and cap bootstrap iterations at 50.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_DEFAULT,
        help="Number of hierarchical bootstrap iterations.",
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
    logger.info("Starting step 50 shared-feature bridge analysis.")
    deviations_df, deviations_meta = load_deviations(logger)
    overlap_features, overlap_meta = load_overlap_feature_set(
        deviations_df=deviations_df,
        smoke_test=args.smoke_test,
        logger=logger,
    )

    full_df = deviations_df.copy()
    overlap_df = deviations_df.loc[deviations_df["feature_name"].isin(overlap_features)].copy()
    if overlap_df.empty:
        raise RuntimeError("Overlap feature set produced zero paired rows.")
    logger.info(
        "Full rows=%d (%d features); overlap rows=%d (%d features).",
        len(full_df),
        full_df["feature_name"].nunique(),
        len(overlap_df),
        overlap_df["feature_name"].nunique(),
    )

    bridge_df = make_bridge_output(overlap_df, overlap_meta)
    full_summary = summarise_set(full_df, "full")
    overlap_summary = summarise_set(overlap_df, "overlap_107")
    full_ci = bootstrap_cis(full_df, "full", n_boot=n_boot, seed=args.seed, logger=logger)
    overlap_ci = bootstrap_cis(
        overlap_df,
        "overlap_107",
        n_boot=n_boot,
        seed=args.seed + 1,
        logger=logger,
    )
    summary_df = pd.concat([full_summary, overlap_summary], ignore_index=True).merge(
        pd.concat([full_ci, overlap_ci], ignore_index=True),
        on=["analysis_set", "family"],
        how="left",
    )
    comparison_df = build_comparison(
        full_summary=summary_df.loc[summary_df["analysis_set"] == "full"].copy(),
        overlap_summary=summary_df.loc[summary_df["analysis_set"] == "overlap_107"].copy(),
    )

    bridge_df.to_parquet(OUT_PARQUET, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    comparison_df.to_csv(OUT_COMPARISON, index=False)
    plot_bridge_comparison(full_df, overlap_df)

    manifest_path = write_manifest(
        job_id=args.job_id,
        argv=sys.argv,
        runtime_seconds=time.time() - t0,
        bootstrap_iterations=n_boot,
        seed=args.seed,
        deviations_meta=deviations_meta,
        overlap_meta=overlap_meta,
        summary_df=summary_df,
        comparison_df=comparison_df,
    )

    logger.info("Wrote %s", OUT_PARQUET)
    logger.info("Wrote %s", OUT_SUMMARY)
    logger.info("Wrote %s", OUT_COMPARISON)
    logger.info("Wrote %s and %s", OUT_FIG_PNG, OUT_FIG_PDF)
    logger.info("Wrote %s", manifest_path)
    logger.info("Step 50 complete in %.1fs.", time.time() - t0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(2)
