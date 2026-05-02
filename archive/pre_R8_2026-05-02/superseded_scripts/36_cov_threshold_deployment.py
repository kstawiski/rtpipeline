#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 36: Pre-registered CoV threshold summary for deployment-oriented stability.

PURPOSE
-------
Translate the within-patient CoV reduction observed in the NSCLC interobserver
dataset into a thresholded summary:

  "What fraction of radiomic features remain above a pre-registered CoV
   threshold under specialist human contouring, and what fraction remain above
   the same threshold under AI contouring?"

Three pre-registered thresholds are carried through the same pipeline:
  * aspirational: 15%  (Traverso 2018; PMID 30016803)
  * strict:       20%  (Balagurunathan 2014; PMID 24935552)
  * moderate:     25%  (Zwanenburg 2020 / IBSI guidance; PMID 32276405)

INPUT
-----
  /home/kgs24/rtpipeline_manuscript/analysis/data/
      nsclc_io_within_patient_deviations.parquet

Required columns:
  patient_id, feature_name, family, human_cov_pct, ai_cov_pct

NOTES ON IDENTIFIERS
--------------------
The task-level convention is `subject = patient_id + "_" + course_id`.
The current NSCLC interobserver deviations parquet does not expose `course_id`,
only `patient_id`.  This script therefore uses:

  subject = patient_id                      if course_id is absent
  subject = patient_id + "_" + course_id   otherwise

This fallback is recorded in the manifest.

ANALYSIS DESIGN
---------------
1. Compute the feature-level median within-patient CoV across subjects for both
   contouring arms.
2. For each threshold and feature family, classify each feature as:
      human_above = median human CoV > threshold
      ai_above    = median AI CoV    > threshold
3. Summarise the percentage of features above threshold under each arm.
4. Estimate 95% percentile confidence intervals for the paired delta
   (human% above - AI% above) using a patient-clustered bootstrap.

Bootstrap convention (matching the manuscript tasking)
------------------------------------------------------
Level 1: resample patient clusters with replacement.
Level 2: within each sampled patient, resample rows within `feature_name`.

Because this parquet contains exactly one CoV summary per patient-feature cell,
the level-2 resample is formally present but degenerate: each patient-feature
group has size 1, so the uncertainty is driven by the cluster resampling step.

OUTPUTS
-------
  data/cov_threshold_crossings.parquet
      Feature-level threshold-crossing table.

  tables/cov_threshold_summary.csv
      Aggregated threshold × family summary.

  figures/figure_cov_threshold.{png,pdf}
      Three-panel grouped bar chart (one panel per threshold).

  logs/36_manifest_<JOBID>.json
      Run provenance, checksums, parameters, and output paths.

USAGE
-----
  qsub /home/kgs24/rtpipeline_manuscript/analysis/run_script36_cov_threshold.sh

  /home/kgs24/miniforge3/envs/radiomics_sge/bin/python \
      /home/kgs24/rtpipeline_manuscript/analysis/36_cov_threshold_deployment.py \
      --bootstrap-iterations 50 \
      --smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
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

INPUT_PARQUET = DATA_DIR / "nsclc_io_within_patient_deviations.parquet"

OUT_PARQUET = DATA_DIR / "cov_threshold_crossings.parquet"
OUT_SUMMARY = TABLES_DIR / "cov_threshold_summary.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_cov_threshold.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_cov_threshold.pdf"

SEED = 2604
N_BOOT_DEFAULT = 1000
FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
THRESHOLD_ORDER = ["aspirational", "strict", "moderate"]


@dataclass(frozen=True)
class ThresholdSpec:
    label: str
    threshold_pct: float
    reference: str
    pmid: str


THRESHOLDS = (
    ThresholdSpec(
        label="aspirational",
        threshold_pct=15.0,
        reference="Traverso 2018 systematic review",
        pmid="30016803",
    ),
    ThresholdSpec(
        label="strict",
        threshold_pct=20.0,
        reference="Balagurunathan 2014 test-retest",
        pmid="24935552",
    ),
    ThresholdSpec(
        label="moderate",
        threshold_pct=25.0,
        reference="Zwanenburg 2020 / IBSI framing",
        pmid="32276405",
    ),
)


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
    logger = logging.getLogger("script36")
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


def load_deviations(smoke_test: bool, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    required = {"feature_name", "family", "human_cov_pct", "ai_cov_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Input parquet missing required columns: {sorted(missing)}")

    df, subject_mode = build_subject_column(df, logger)
    df = df.loc[df["family"].isin(FAMILY_ORDER)].copy()
    df["feature_name"] = df["feature_name"].astype(str)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)

    selected_features: list[str] | None = None
    if smoke_test:
        selected_features = sorted(df["feature_name"].unique())[:10]
        df = df.loc[df["feature_name"].isin(selected_features)].copy()
        logger.info(
            "Smoke test enabled: restricted analysis to %d features.",
            len(selected_features),
        )

    meta = {
        "subject_mode": subject_mode,
        "selected_features": selected_features,
        "n_subjects": int(df["subject"].nunique()),
        "n_rows": int(len(df)),
        "n_features": int(df["feature_name"].nunique()),
    }
    return df, meta


def compute_feature_level_cov(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["feature_name", "family"], observed=True, sort=False)
        .agg(
            n_subjects=("subject", "nunique"),
            human_median_cov_pct=("human_cov_pct", "median"),
            ai_median_cov_pct=("ai_cov_pct", "median"),
        )
        .reset_index()
    )
    grouped = grouped.dropna(
        subset=["human_median_cov_pct", "ai_median_cov_pct"]
    ).copy()
    grouped["human_minus_ai_cov_pct"] = (
        grouped["human_median_cov_pct"] - grouped["ai_median_cov_pct"]
    )
    return grouped


def build_threshold_crossings(feature_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for threshold in THRESHOLDS:
        block = feature_df.copy()
        block["threshold"] = threshold.label
        block["threshold_pct"] = threshold.threshold_pct
        block["reference"] = threshold.reference
        block["pmid"] = threshold.pmid
        block["human_above"] = block["human_median_cov_pct"] > threshold.threshold_pct
        block["ai_above"] = block["ai_median_cov_pct"] > threshold.threshold_pct
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    out["threshold"] = pd.Categorical(
        out["threshold"], categories=THRESHOLD_ORDER, ordered=True
    )
    return out


def mcnemar_mid_p(human_above: pd.Series, ai_above: pd.Series) -> float:
    human_bool = human_above.astype(bool).to_numpy()
    ai_bool = ai_above.astype(bool).to_numpy()
    b = int(np.sum(human_bool & ~ai_bool))
    c = int(np.sum(~human_bool & ai_bool))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_exact = 2.0 * float(sp_stats.binom.cdf(k, n, 0.5))
    p_mid = p_exact - float(sp_stats.binom.pmf(k, n, 0.5))
    return float(min(1.0, max(0.0, p_mid)))


def summarise_crossings(crossings_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        thr_block = crossings_df.loc[crossings_df["threshold"] == threshold.label]
        for family in FAMILY_ORDER:
            sub = thr_block.loc[thr_block["family"] == family]
            if sub.empty:
                continue
            pct_human = 100.0 * float(sub["human_above"].mean())
            pct_ai = 100.0 * float(sub["ai_above"].mean())
            rows.append(
                {
                    "threshold": threshold.label,
                    "threshold_pct": threshold.threshold_pct,
                    "reference": threshold.reference,
                    "pmid": threshold.pmid,
                    "family": family,
                    "n_features": int(sub["feature_name"].nunique()),
                    "pct_human_above": pct_human,
                    "pct_ai_above": pct_ai,
                    "delta_pct": pct_human - pct_ai,
                    "wilcoxon_p_mid": mcnemar_mid_p(
                        sub["human_above"], sub["ai_above"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def resample_subject_clusters(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    subject_ids = df["subject"].drop_duplicates().to_numpy()
    sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
    subject_blocks: list[pd.DataFrame] = []
    for draw_index, subject_id in enumerate(sampled_subjects):
        subject_block = df.loc[df["subject"] == subject_id].copy()
        sampled_blocks = []
        for _, group in subject_block.groupby("feature_name", sort=False):
            take = rng.integers(0, len(group), size=len(group))
            sampled_blocks.append(group.iloc[take].copy())
        sampled_subject = pd.concat(sampled_blocks, ignore_index=True)
        sampled_subject["subject"] = f"{subject_id}__boot{draw_index}"
        subject_blocks.append(sampled_subject)
    return pd.concat(subject_blocks, ignore_index=True)


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    low = float(np.nanpercentile(values, 100.0 * alpha / 2.0))
    high = float(np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def bootstrap_summary(
    raw_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[pd.DataFrame] = []
    for iteration in range(n_boot):
        boot_df = resample_subject_clusters(raw_df, rng)
        feature_df = compute_feature_level_cov(boot_df)
        crossings = build_threshold_crossings(feature_df)
        summary = summarise_crossings(crossings)
        summary["bootstrap_iteration"] = iteration
        records.append(summary[["threshold", "family", "delta_pct", "bootstrap_iteration"]])
        if (iteration + 1) % 100 == 0 or iteration == n_boot - 1:
            logger.info("Bootstrap %d/%d complete.", iteration + 1, n_boot)
    boot_df = pd.concat(records, ignore_index=True)
    ci_rows: list[dict[str, object]] = []
    for (threshold, family), sub in boot_df.groupby(["threshold", "family"], sort=False):
        ci_low, ci_high = percentile_ci(sub["delta_pct"].to_numpy(dtype=float))
        ci_rows.append(
            {
                "threshold": threshold,
                "family": family,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(ci_rows)


def plot_threshold_summary(summary_df: pd.DataFrame) -> None:
    colors = {"Human": "#8a6f3a", "AI": "#1f78b4"}
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)
    family_positions = np.arange(len(FAMILY_ORDER))
    width = 0.38

    for axis, threshold in zip(axes, THRESHOLDS):
        sub = (
            summary_df.loc[summary_df["threshold"] == threshold.label]
            .set_index("family")
            .reindex(FAMILY_ORDER)
        )
        axis.bar(
            family_positions - width / 2,
            sub["pct_human_above"].to_numpy(dtype=float),
            width=width,
            color=colors["Human"],
            label="Human",
        )
        axis.bar(
            family_positions + width / 2,
            sub["pct_ai_above"].to_numpy(dtype=float),
            width=width,
            color=colors["AI"],
            label="AI",
        )
        axis.set_title(f"{threshold.label.capitalize()} threshold ({threshold.threshold_pct:.0f}%)")
        axis.set_xticks(family_positions)
        axis.set_xticklabels(FAMILY_ORDER, rotation=35, ha="right")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", linestyle=":", alpha=0.35)
        axis.axhline(0, color="black", linewidth=0.8)

    axes[0].set_ylabel("Features above threshold (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Feature-level median CoV threshold crossings by contouring arm", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    *,
    job_id: str,
    argv: list[str],
    cohorts: list[str],
    runtime_seconds: float,
    bootstrap_iterations: int,
    input_meta: dict[str, object],
) -> Path:
    manifest_path = LOGS_DIR / f"36_manifest_{job_id}.json"
    manifest = {
        "script_path": str(Path(__file__).resolve()),
        "argv": argv,
        "host": socket.gethostname(),
        "job_id": job_id,
        "project_path": str(Path(__file__).resolve().parents[1]),
        "bootstrap_seed": SEED,
        "bootstrap_iterations": bootstrap_iterations,
        "cohorts_argument": cohorts,
        "input_file_checksums": {
            str(INPUT_PARQUET): sha256_file(INPUT_PARQUET),
        },
        "output_file_paths": [
            str(OUT_PARQUET),
            str(OUT_SUMMARY),
            str(OUT_FIG_PNG),
            str(OUT_FIG_PDF),
        ],
        "output_file_checksums": {
            str(OUT_PARQUET): sha256_file(OUT_PARQUET),
            str(OUT_SUMMARY): sha256_file(OUT_SUMMARY),
            str(OUT_FIG_PNG): sha256_file(OUT_FIG_PNG),
            str(OUT_FIG_PDF): sha256_file(OUT_FIG_PDF),
        },
        "runtime_seconds": runtime_seconds,
        "git_sha": resolve_git_sha(),
        "input_meta": input_meta,
        "thresholds": [threshold.__dict__ for threshold in THRESHOLDS],
        "method_notes": {
            "feature_level_summary": "Threshold crossing is defined on median within-patient CoV per feature.",
            "delta_definition": "delta_pct = pct_human_above - pct_ai_above",
            "mcnemar_column_name": "The output column wilcoxon_p_mid stores a McNemar-style exact mid-p value to match task naming.",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Restrict to 10 features and cap bootstrap iterations at 50.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=N_BOOT_DEFAULT,
        help="Number of patient-clustered bootstrap iterations.",
    )
    parser.add_argument(
        "--cohorts",
        default="NSCLC_Interobserver",
        help="Accepted for interface compatibility; recorded in the manifest.",
    )
    parser.add_argument(
        "--job-id",
        default=os.environ.get("JOB_ID", "local"),
        help="Job identifier used in logs and manifest output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cohorts = parse_cohorts(args.cohorts)
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
    logger.info("Starting step 36 threshold summary.")
    logger.info("Input parquet: %s", INPUT_PARQUET)
    logger.info("Cohorts argument recorded as: %s", ",".join(cohorts))

    raw_df, input_meta = load_deviations(args.smoke_test, logger)
    logger.info(
        "Loaded %d rows across %d subjects and %d features.",
        input_meta["n_rows"],
        input_meta["n_subjects"],
        input_meta["n_features"],
    )

    feature_df = compute_feature_level_cov(raw_df)
    crossings_df = build_threshold_crossings(feature_df)
    summary_df = summarise_crossings(crossings_df)
    ci_df = bootstrap_summary(raw_df, n_boot=n_boot, seed=SEED, logger=logger)
    summary_df = summary_df.merge(ci_df, on=["threshold", "family"], how="left")

    crossings_df.to_parquet(OUT_PARQUET, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    plot_threshold_summary(summary_df)

    manifest_path = write_manifest(
        job_id=args.job_id,
        argv=sys.argv,
        cohorts=cohorts,
        runtime_seconds=time.time() - t0,
        bootstrap_iterations=n_boot,
        input_meta=input_meta,
    )

    logger.info("Wrote %s", OUT_PARQUET)
    logger.info("Wrote %s", OUT_SUMMARY)
    logger.info("Wrote %s and %s", OUT_FIG_PNG, OUT_FIG_PDF)
    logger.info("Wrote %s", manifest_path)
    logger.info("Step 36 completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive exit path
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(2)
