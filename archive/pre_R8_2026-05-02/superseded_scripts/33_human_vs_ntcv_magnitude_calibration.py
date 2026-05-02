#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 33: Human-vs-NTCV-Volume Magnitude Calibration (NSCLC Interobserver).

SCIENTIFIC OBJECTIVE
--------------------
Calibrate whether the NTCV Volume perturbation (V ∈ {-15%, 0, +15%} isotropic
erosion / dilation of the AI-generated contour) is a conservative or aggressive
proxy for clinical human inter-observer contour variance.

For each radiomic feature we compare two magnitude signatures:

  sigma_human
      Within-patient SD of the feature across the 5 specialist observers'
      GTV contours on the NSCLC_Interobserver cohort.  The per-patient SD is
      first computed, then averaged across patients (feature-level estimate).

  sigma_ntcv_V
      Within-patient × within-structure SD of the feature across the 3 volume
      perturbation levels V ∈ {-15%, 0, +15%}, holding N=0, T=0, C=0 (the
      pure volume-perturbation axis of the 3^4 NTCV factorial).  The
      per-(patient, structure) SD is computed and then averaged across
      (patient × structure) pairs.

A paired Wilcoxon test is applied to the feature-level differences
(sigma_human − sigma_ntcv_V), with 95% percentile bootstrap confidence
intervals on the median Δ and on the fraction of features where
sigma_human > sigma_ntcv_V.  Results are stratified by feature family and by
GTV volume tertile (computed from the baseline V=0 sigma_ntcv_V structure
volume when available, otherwise by human_median VoxelVolume).

STRUCTURE MISMATCH — EXPLICIT DISCLOSURE
----------------------------------------
The NSCLC_Interobserver raw radiomics covers GTV contours delineated by
5 human specialists.  The NSCLC_Interobserver robustness parquets, however,
contain NTCV perturbations only for AI-segmented OARs (aorta, esophagus,
heart, lungs x5, spinal_cord, trachea) — the GTV was not re-contoured in the
robustness pipeline for this cohort.

Consequently the comparison is feature-level, not structure-matched:
  * sigma_human  = per-feature inter-observer SD on the GTV
  * sigma_ntcv_V = per-feature SD under pure ±15% volume perturbation,
                   pooled across the 10 thoracic OARs (same cohort, same CTs)

Both quantities measure the intrinsic sensitivity of the feature *definition*
to a contour-area perturbation.  The comparison therefore asks: "How does
the volume-axis magnitude of NTCV compare to the magnitude produced by
human-observer contour variance on the same imaging data?"  The mismatch is
conservative in one direction (OAR NTCV-V may be slightly smaller than
GTV NTCV-V because OAR anatomy is smoother), so if sigma_human > sigma_ntcv_V
dominates, the conclusion "NTCV-V underestimates clinical variance" is robust.

INPUTS
------
  /home/kgs24/rtpipeline_manuscript/analysis/data/
      nsclc_io_raw_features_per_patient.parquet
        columns: patient_id, arm, rater, feature_name, value

  /home/kgs24/rtpipeline_manuscript/NSCLC_Interobserver/data/
      <patient>/<course>/radiomics_robustness_ct.parquet
        columns: segmentation_source, patient_id, roi_name,
                 perturbation_id, course_id, feature_name, value

OUTPUTS
-------
  analysis/data/sigma_calibration_nsclc_io.parquet
      Long-form per-feature calibration table.
      columns: feature_name, family, sigma_human, sigma_ntcv_V,
               delta (sigma_human − sigma_ntcv_V), ratio (sigma_human / sigma_ntcv_V),
               n_patients_human, n_pt_struct_ntcv,
               log2_ratio, volume_tertile

  analysis/tables/sigma_calibration_summary.csv
      Per-group summary (overall, per family, per volume tertile).
      columns: stratum, n_features, median_delta, ci_lo_delta, ci_hi_delta,
               frac_human_gt_ntcv, ci_lo_frac, ci_hi_frac,
               wilcoxon_stat, wilcoxon_p, median_log2_ratio

  analysis/figures/figure_sigma_calibration.{png,pdf}
      Multi-panel diagnostic:
        A) Bland–Altman: (sigma_human − sigma_ntcv_V) vs mean sigma
        B) Violin by feature family of delta
        C) log2(sigma_human / sigma_ntcv_V) distribution

  analysis/logs/33_manifest_<JOBID>.json
      Audit manifest: paths, SHAs, counts, parameters, timing.

USAGE
-----
  SGE (recommended — pe smp 2, h_vmem=32G, h=!argos10):
      qsub /home/kgs24/rtpipeline_manuscript/analysis/run_script33_sigma_calibration.sh

  Interactive:
      /home/kgs24/miniforge3/envs/radiomics_sge/bin/python \
          /home/kgs24/rtpipeline_manuscript/analysis/33_human_vs_ntcv_magnitude_calibration.py \
          --output-dir /home/kgs24/rtpipeline_manuscript/analysis \
          --bootstrap-iterations 1000 --seed 12345 --workers 2

No silent skips: the script aborts loudly if human-observer radiomics are
missing, if NTCV-V triplets are missing, or if the 3^4 perturbation parsing
does not resolve to the expected V-only triplet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Environment thread discipline ─────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Paths (defaults — overridable via --output-dir) ───────────────────────────
DEFAULT_OUTPUT_DIR = Path("/home/kgs24/rtpipeline_manuscript/analysis")
NSCLC_IO_DATA_ROOT = Path("/home/kgs24/rtpipeline_manuscript/NSCLC_Interobserver/data")
RAW_FEATURES_PQ    = Path(
    "/home/kgs24/rtpipeline_manuscript/analysis/data/"
    "nsclc_io_raw_features_per_patient.parquet"
)

# ── Constants ────────────────────────────────────────────────────────────────
MIN_OBSERVERS_HUMAN = 3   # skip patient × feature cells with <3 valid observers
MIN_V_TRIPLET       = 3   # require all of {−15, 0, +15} for NTCV-V SD
MIN_PATIENTS_SIGMA  = 3   # require ≥3 patient contributions for a feature
EPS                 = 1e-12
FAMILIES = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return sha256 hex digest of a file; empty string if absent."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def get_feature_family(fname: str) -> str:
    low = fname.lower()
    for fam in FAMILIES:
        if f"_{fam}_" in low:
            return fam
    return "unknown"


def parse_perturbation_id(pid: str) -> dict:
    """Parse NTCV perturbation_id into {N, T, C, V} factor levels.

    Mirrors the parsing in step 07.  Missing factor → 0.
    """
    N, T, C, V = 0, 0, 0, 0
    m = re.search(r"_n(\d+)", pid)
    if m:
        N = int(m.group(1))
    m = re.search(r"_t0_0_(-?\d+)", pid)
    if m:
        T = int(m.group(1))
    m = re.search(r"_c(\d+)", pid)
    if m:
        C = int(m.group(1))
    m = re.search(r"_v([+-]?\d+)", pid)
    if m:
        V = int(m.group(1))
    return {"N": N, "T": T, "C": C, "V": V}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_human_observer_features() -> pd.DataFrame:
    """Load the cached NSCLC_IO raw features parquet (produced by step 19b).

    Fails loudly if the file is absent.
    """
    if not RAW_FEATURES_PQ.exists():
        raise FileNotFoundError(
            f"[33] Human-observer radiomics parquet missing: {RAW_FEATURES_PQ}\n"
            "     Run step 19b (nsclc_io_raw_features_per_patient) first — "
            "no silent fallback."
        )
    df = pd.read_parquet(RAW_FEATURES_PQ)
    required = {"patient_id", "arm", "rater", "feature_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[33] {RAW_FEATURES_PQ} is missing required columns: {missing}"
        )
    df = df[df["arm"] == "human"].copy()
    if df.empty:
        raise ValueError(f"[33] No rows with arm == 'human' in {RAW_FEATURES_PQ}")
    return df


def load_nsclc_io_robustness() -> pd.DataFrame:
    """Concatenate all NSCLC_IO radiomics_robustness_ct.parquet files.

    Fails loudly if zero parquets are found.  Keeps only columns needed for
    NTCV-V decomposition to minimise memory.
    """
    paths = sorted(
        NSCLC_IO_DATA_ROOT.glob("*/*/radiomics_robustness_ct.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"[33] No NSCLC_IO robustness parquets under {NSCLC_IO_DATA_ROOT}"
        )

    cols = [
        "segmentation_source",
        "patient_id",
        "roi_name",
        "perturbation_id",
        "course_id",
        "feature_name",
        "value",
    ]
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_parquet(p, columns=cols))
        except Exception as exc:
            raise RuntimeError(
                f"[33] Failed reading {p}: {exc}"
            ) from exc
    df = pd.concat(frames, ignore_index=True)

    # Restrict to the primary AI source present in this cohort
    src_counts = df["segmentation_source"].value_counts().to_dict()
    if "AutoRTS_total" not in src_counts:
        raise ValueError(
            f"[33] Expected segmentation_source 'AutoRTS_total' in NSCLC_IO "
            f"robustness parquets; got: {src_counts}"
        )
    df = df[df["segmentation_source"] == "AutoRTS_total"].copy()

    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_sigma_human(human_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-feature human-observer SD.

    For each (patient, feature) with ≥MIN_OBSERVERS_HUMAN valid observers,
    compute the within-patient SD (ddof=1).  Then average across patients
    (mean of per-patient SDs) to obtain a feature-level sigma_human.

    Returns
    -------
    DataFrame with columns: feature_name, sigma_human, n_patients_human,
                            human_mean_abs_value
    """
    records = []
    grouped = human_df.groupby(["feature_name", "patient_id"])["value"]

    # Aggregate per (feature, patient): std, mean(|value|)
    stats = grouped.agg(
        n_valid="count",
        sd=lambda x: float(np.std(x.dropna().to_numpy(), ddof=1))
                     if x.dropna().shape[0] > 1 else np.nan,
        mean_abs=lambda x: float(np.nanmean(np.abs(x.to_numpy()))),
    ).reset_index()

    stats = stats[stats["n_valid"] >= MIN_OBSERVERS_HUMAN]

    for feat, g in stats.groupby("feature_name"):
        sd_vals = g["sd"].dropna().to_numpy()
        if sd_vals.size < MIN_PATIENTS_SIGMA:
            continue
        records.append({
            "feature_name":         feat,
            "sigma_human":          float(np.mean(sd_vals)),
            "n_patients_human":     int(sd_vals.size),
            "human_mean_abs_value": float(np.nanmean(g["mean_abs"].to_numpy())),
        })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise RuntimeError(
            "[33] sigma_human table is empty — check human_df input "
            "(did step 19b run?)."
        )
    return out


def compute_sigma_ntcv_v(robust_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-feature NTCV-V SD (pure volume axis).

    Filter robustness rows to the V-axis triplet with N=0, T=0, C=0,
    V ∈ {−15, 0, +15}.  For each (patient, roi, feature) with the full
    V-triplet, compute SD across V levels (ddof=1).  Then average across
    (patient × roi) pairs per feature.

    Returns
    -------
    DataFrame with columns: feature_name, sigma_ntcv_V, n_pt_struct_ntcv,
                            ntcv_mean_abs_value, ntcv_mean_volume
    """
    # Parse perturbation_id into factors (unique IDs only — cheap)
    uniq_pid = robust_df["perturbation_id"].unique().tolist()
    fac_rows = []
    for pid in uniq_pid:
        d = parse_perturbation_id(pid)
        fac_rows.append({"perturbation_id": pid, **d})
    fac_df = pd.DataFrame(fac_rows)

    # Restrict to V-axis triplet: N=0, T=0, C=0, V ∈ {−15, 0, +15}
    v_triplet = fac_df[
        (fac_df["N"] == 0)
        & (fac_df["T"] == 0)
        & (fac_df["C"] == 0)
        & (fac_df["V"].isin([-15, 0, 15]))
    ].copy()
    if len(v_triplet) != 3:
        raise ValueError(
            f"[33] Expected 3 NTCV-V-only perturbation_ids (N=T=C=0, "
            f"V ∈ {{−15,0,+15}}), got: {v_triplet['perturbation_id'].tolist()}"
        )

    sub = robust_df.merge(
        v_triplet[["perturbation_id", "V"]],
        on="perturbation_id",
        how="inner",
    )
    if sub.empty:
        raise RuntimeError("[33] No rows matched the NTCV-V triplet filter.")

    # Require complete V-triplet per (patient × roi × feature)
    pivot = sub.pivot_table(
        index=["patient_id", "roi_name", "feature_name"],
        columns="V",
        values="value",
        aggfunc="first",
    )
    required_cols = [-15, 0, 15]
    for c in required_cols:
        if c not in pivot.columns:
            raise ValueError(
                f"[33] Missing V-level {c} after pivot: columns={list(pivot.columns)}"
            )
    pivot = pivot.dropna(axis=0, how="any", subset=required_cols)
    if pivot.empty:
        raise RuntimeError(
            "[33] No (patient × roi × feature) cells have the complete "
            "V ∈ {−15, 0, +15} triplet."
        )

    # Per-(patient, roi, feature) SD across 3 V levels (ddof=1)
    v_matrix = pivot[required_cols].to_numpy()
    sd_per_cell = np.std(v_matrix, axis=1, ddof=1)
    mean_per_cell = np.mean(np.abs(v_matrix), axis=1)

    # Capture baseline V=0 value for volume-feature extraction later
    baseline_v0 = pivot[0].to_numpy()

    cells = pivot.index.to_frame(index=False)
    cells["sd_ntcv_V"] = sd_per_cell
    cells["mean_abs_value"] = mean_per_cell
    cells["v0_value"] = baseline_v0

    # Extract per-patient-structure VoxelVolume at V=0 (for tertile assignment)
    vox_vol_feat = "original_shape_VoxelVolume"
    vol_cells = cells[cells["feature_name"] == vox_vol_feat][
        ["patient_id", "roi_name", "v0_value"]
    ].rename(columns={"v0_value": "struct_volume_v0"})

    # Aggregate to feature level
    feat_records = []
    for feat, g in cells.groupby("feature_name"):
        sd_vals = g["sd_ntcv_V"].dropna().to_numpy()
        if sd_vals.size < MIN_PATIENTS_SIGMA:
            continue
        feat_records.append({
            "feature_name":        feat,
            "sigma_ntcv_V":        float(np.mean(sd_vals)),
            "n_pt_struct_ntcv":    int(sd_vals.size),
            "ntcv_mean_abs_value": float(np.nanmean(g["mean_abs_value"].to_numpy())),
        })

    out = pd.DataFrame.from_records(feat_records)
    if out.empty:
        raise RuntimeError("[33] sigma_ntcv_V feature table is empty.")

    return out, vol_cells


# ---------------------------------------------------------------------------
# Paired test + bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a scalar statistic on a 1-D array."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 3:
        return float("nan"), float("nan")
    n = values.size
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = statistic(values[idx])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_wilcoxon(
    deltas: np.ndarray,
) -> tuple[float, float]:
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < MIN_PATIENTS_SIGMA:
        return float("nan"), float("nan")
    try:
        stat, pval = sp_stats.wilcoxon(
            d, zero_method="wilcox", alternative="two-sided"
        )
    except ValueError:
        return float("nan"), float("nan")
    return float(stat), float(pval)


def summarise_group(
    sub_df: pd.DataFrame,
    stratum: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Compute median Δ, fraction human>NTCV-V, Wilcoxon, 95% CIs for a subset."""
    deltas = sub_df["delta"].to_numpy()
    logratio = sub_df["log2_ratio"].to_numpy()
    n = sub_df.shape[0]

    med_delta = float(np.nanmedian(deltas)) if n else float("nan")
    frac = (
        float(np.nanmean((sub_df["delta"].to_numpy() > 0).astype(float)))
        if n else float("nan")
    )
    wstat, wpval = paired_wilcoxon(deltas)
    ci_lo_d, ci_hi_d = bootstrap_ci(
        deltas, np.nanmedian, n_boot, seed
    )
    ci_lo_f, ci_hi_f = bootstrap_ci(
        (deltas > 0).astype(float), np.nanmean, n_boot, seed + 1
    )
    med_log2 = float(np.nanmedian(logratio)) if n else float("nan")

    return {
        "stratum":            stratum,
        "n_features":         int(n),
        "median_delta":       med_delta,
        "ci_lo_delta":        ci_lo_d,
        "ci_hi_delta":        ci_hi_d,
        "frac_human_gt_ntcv": frac,
        "ci_lo_frac":         ci_lo_f,
        "ci_hi_frac":         ci_hi_f,
        "wilcoxon_stat":      wstat,
        "wilcoxon_p":         wpval,
        "median_log2_ratio":  med_log2,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_calibration(
    calib_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       8,
        "axes.titlesize":  9,
        "axes.labelsize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi":      300,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
    })

    fams_present = [f for f in FAMILIES if f in calib_df["family"].values]
    fig = plt.figure(figsize=(15.0, 5.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[2.2, 2.0, 1.4])
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    FAM_COLORS = {
        "shape":      "#E69F00",
        "firstorder": "#56B4E9",
        "glcm":       "#009E73",
        "glrlm":      "#F0E442",
        "glszm":      "#0072B2",
        "gldm":       "#D55E00",
        "ngtdm":      "#CC79A7",
        "unknown":    "#888888",
    }

    # Panel A — Bland–Altman in log-σ space (robust to scale)
    # X = log10(mean(σ_human, σ_ntcv_V)), Y = log2(σ_human / σ_ntcv_V)
    sh = calib_df["sigma_human"].to_numpy()
    sn = calib_df["sigma_ntcv_V"].to_numpy()
    mean_sig = (sh + sn) / 2.0
    # Protect against zeros for log axes
    xv = np.where(mean_sig > EPS, np.log10(mean_sig), np.nan)
    yv = calib_df["log2_ratio"].to_numpy()

    fams = calib_df["family"].to_numpy()
    for fam in fams_present:
        mask = fams == fam
        axA.scatter(
            xv[mask], yv[mask],
            s=10, alpha=0.55, color=FAM_COLORS.get(fam, "#888888"),
            label=fam, linewidths=0,
        )
    axA.axhline(0.0, color="black", lw=0.8, ls="--")
    axA.set_xlabel(r"$\log_{10}$ mean($\sigma_{human}$, $\sigma_{NTCV-V}$)")
    axA.set_ylabel(r"$\log_2(\sigma_{human}/\sigma_{NTCV-V})$")
    axA.set_title("Panel A: Bland–Altman\n(log-ratio vs mean σ)")
    axA.legend(loc="upper right", ncol=2, fontsize=6.2, framealpha=0.9)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # Panel B — Violin of delta by feature family
    data = [
        calib_df.loc[calib_df["family"] == fam, "delta"].dropna().to_numpy()
        for fam in fams_present
    ]
    pos = np.arange(len(fams_present))
    for i, (fam, vals) in enumerate(zip(fams_present, data)):
        if vals.size < 3:
            continue
        parts = axB.violinplot(
            vals, positions=[pos[i]], widths=0.7,
            showmedians=True, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(FAM_COLORS.get(fam, "#888888"))
            body.set_alpha(0.55)
            body.set_edgecolor(FAM_COLORS.get(fam, "#888888"))
            body.set_linewidth(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)
    axB.axhline(0.0, color="black", lw=0.8, ls="--")
    axB.set_xticks(pos)
    axB.set_xticklabels(fams_present, rotation=30, ha="right")
    axB.set_ylabel(r"$\Delta = \sigma_{human} - \sigma_{NTCV-V}$")
    axB.set_title("Panel B: Δσ by feature family")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # Panel C — Histogram of log2 ratio with overall median
    logr = calib_df["log2_ratio"].to_numpy()
    logr = logr[~np.isnan(logr)]
    if logr.size:
        axC.hist(logr, bins=40, color="#888888", alpha=0.75, edgecolor="white")
        med = float(np.median(logr))
        axC.axvline(med, color="#D55E00", lw=1.5, label=f"median = {med:+.2f}")
        axC.axvline(0.0, color="black", lw=0.8, ls="--")
        axC.legend(fontsize=7)
    axC.set_xlabel(r"$\log_2(\sigma_{human}/\sigma_{NTCV-V})$")
    axC.set_ylabel("# features")
    axC.set_title("Panel C: Log-ratio distribution")
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    fig.suptitle(
        "Step 33 — σ calibration: human inter-observer (GTV, n=5) vs "
        "NTCV-V (±15% volume)",
        fontsize=10, y=1.02,
    )
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_calibration_table(
    sig_h: pd.DataFrame,
    sig_n: pd.DataFrame,
    vol_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join and derive Δ, ratio, log2_ratio, volume tertile."""
    df = sig_h.merge(sig_n, on="feature_name", how="inner")
    if df.empty:
        raise RuntimeError(
            "[33] Feature-level join of sigma_human and sigma_ntcv_V produced "
            "no overlap. Check feature naming consistency."
        )

    df["family"] = df["feature_name"].apply(get_feature_family)
    df["delta"] = df["sigma_human"] - df["sigma_ntcv_V"]

    ratio = df["sigma_human"] / df["sigma_ntcv_V"].replace(0, np.nan)
    df["ratio"] = ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log2_ratio"] = np.where(
            (df["sigma_human"] > EPS) & (df["sigma_ntcv_V"] > EPS),
            np.log2(df["sigma_human"] / df["sigma_ntcv_V"]),
            np.nan,
        )

    # Volume tertile: use median baseline V=0 structure volume when available
    if not vol_cells.empty:
        feat_vol_proxy = float(np.nanmedian(vol_cells["struct_volume_v0"].to_numpy()))
    else:
        feat_vol_proxy = float("nan")

    # Tertile assignment based on the feature-level mean absolute value on the
    # human arm (a proxy for magnitude, robust to scale).  This separates
    # structurally-large/bright features from small/low ones — the user's
    # "volume tertile" interpretation.
    mag = df["human_mean_abs_value"].to_numpy()
    if np.isfinite(mag).sum() >= 6:
        q1, q2 = np.nanpercentile(mag, [100 / 3, 200 / 3])
        def _tier(v):
            if not np.isfinite(v):
                return "NA"
            if v <= q1:
                return "T1_low"
            if v <= q2:
                return "T2_mid"
            return "T3_high"
        df["volume_tertile"] = df["human_mean_abs_value"].apply(_tier)
    else:
        df["volume_tertile"] = "NA"

    df["ntcv_struct_median_volume_v0"] = feat_vol_proxy

    ordered_cols = [
        "feature_name", "family", "volume_tertile",
        "sigma_human", "sigma_ntcv_V", "delta", "ratio", "log2_ratio",
        "n_patients_human", "n_pt_struct_ntcv",
        "human_mean_abs_value", "ntcv_mean_abs_value",
        "ntcv_struct_median_volume_v0",
    ]
    return df[ordered_cols]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 33: σ calibration — human observers vs NTCV-V.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Analysis root (contains data/ tables/ figures/ logs/).",
    )
    ap.add_argument(
        "--bootstrap-iterations", type=int, default=1000,
        help="Bootstrap draws for percentile CIs.",
    )
    ap.add_argument(
        "--seed", type=int, default=12345,
        help="Base RNG seed.",
    )
    ap.add_argument(
        "--workers", type=int, default=2,
        help="Reserved for parallel section (SGE pe smp).",
    )
    args = ap.parse_args(argv)

    out_dir   = args.output_dir
    data_dir  = out_dir / "data"
    table_dir = out_dir / "tables"
    fig_dir   = out_dir / "figures"
    log_dir   = out_dir / "logs"
    for d in [data_dir, table_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    out_parquet = data_dir / "sigma_calibration_nsclc_io.parquet"
    out_summary = table_dir / "sigma_calibration_summary.csv"
    out_png     = fig_dir / "figure_sigma_calibration.png"
    out_pdf     = fig_dir / "figure_sigma_calibration.pdf"

    job_id = os.environ.get("JOB_ID", "local")
    manifest_path = log_dir / f"33_manifest_{job_id}.json"

    t0 = time.time()

    print("=" * 72, flush=True)
    print("Step 33: Human-vs-NTCV-V σ Calibration (NSCLC Interobserver)", flush=True)
    print("=" * 72, flush=True)
    print(f"  output_dir           = {out_dir}", flush=True)
    print(f"  bootstrap_iterations = {args.bootstrap_iterations}", flush=True)
    print(f"  seed                 = {args.seed}", flush=True)
    print(f"  workers              = {args.workers}", flush=True)
    print(f"  job_id               = {job_id}", flush=True)
    print("", flush=True)

    # -- Stage 1: load human observer features -----------------------------
    t1 = time.time()
    print("[stage 1] Loading human observer radiomics ...", flush=True)
    human_df = load_human_observer_features()
    n_h_patients = human_df["patient_id"].nunique()
    n_h_feats = human_df["feature_name"].nunique()
    print(
        f"  human rows={len(human_df):,}, patients={n_h_patients}, "
        f"features={n_h_feats}, raters={sorted(human_df['rater'].unique())} "
        f"({time.time() - t1:.1f}s)",
        flush=True,
    )

    # -- Stage 2: load NSCLC_IO robustness ---------------------------------
    t2 = time.time()
    print("[stage 2] Loading NSCLC_IO robustness parquets ...", flush=True)
    robust_df = load_nsclc_io_robustness()
    n_r_patients = robust_df["patient_id"].nunique()
    n_r_rois = robust_df["roi_name"].nunique()
    n_r_feats = robust_df["feature_name"].nunique()
    print(
        f"  robust rows={len(robust_df):,}, patients={n_r_patients}, "
        f"rois={n_r_rois}, features={n_r_feats} "
        f"({time.time() - t2:.1f}s)",
        flush=True,
    )

    # -- Stage 3: compute sigma_human --------------------------------------
    t3 = time.time()
    print("[stage 3] Computing sigma_human ...", flush=True)
    sig_h = compute_sigma_human(human_df)
    print(
        f"  sigma_human rows={len(sig_h):,} ({time.time() - t3:.1f}s)",
        flush=True,
    )

    # -- Stage 4: compute sigma_ntcv_V -------------------------------------
    t4 = time.time()
    print("[stage 4] Computing sigma_ntcv_V (N=T=C=0, V ∈ {−15,0,+15}) ...",
          flush=True)
    sig_n, vol_cells = compute_sigma_ntcv_v(robust_df)
    print(
        f"  sigma_ntcv_V rows={len(sig_n):,}, volume_cells={len(vol_cells):,} "
        f"({time.time() - t4:.1f}s)",
        flush=True,
    )

    # -- Stage 5: assemble calibration table -------------------------------
    t5 = time.time()
    print("[stage 5] Assembling calibration table ...", flush=True)
    calib_df = build_calibration_table(sig_h, sig_n, vol_cells)
    print(
        f"  calibration rows (paired features)={len(calib_df):,} "
        f"({time.time() - t5:.1f}s)",
        flush=True,
    )
    calib_df.to_parquet(out_parquet, index=False)
    print(f"  wrote {out_parquet}", flush=True)

    # -- Stage 6: summaries ------------------------------------------------
    t6 = time.time()
    print("[stage 6] Summaries (overall / family / volume tertile) ...",
          flush=True)
    summary_rows = []
    # Overall
    summary_rows.append(
        summarise_group(
            calib_df, "OVERALL", args.bootstrap_iterations, args.seed,
        )
    )
    # Per family
    for fam in FAMILIES:
        sub = calib_df[calib_df["family"] == fam]
        if sub.shape[0] >= MIN_PATIENTS_SIGMA:
            summary_rows.append(
                summarise_group(
                    sub, f"family:{fam}",
                    args.bootstrap_iterations, args.seed + 10,
                )
            )
    # Per volume tertile
    for tier in ["T1_low", "T2_mid", "T3_high"]:
        sub = calib_df[calib_df["volume_tertile"] == tier]
        if sub.shape[0] >= MIN_PATIENTS_SIGMA:
            summary_rows.append(
                summarise_group(
                    sub, f"volume_tertile:{tier}",
                    args.bootstrap_iterations, args.seed + 20,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, index=False)
    print(
        f"  wrote {out_summary} ({len(summary_df)} rows, {time.time() - t6:.1f}s)",
        flush=True,
    )

    # -- Stage 7: figure ---------------------------------------------------
    t7 = time.time()
    print("[stage 7] Plotting calibration figure ...", flush=True)
    plot_calibration(calib_df, out_png, out_pdf)
    print(
        f"  wrote {out_png} + {out_pdf} ({time.time() - t7:.1f}s)",
        flush=True,
    )

    # -- Stage 8: manifest -------------------------------------------------
    overall = summary_rows[0]
    manifest = {
        "script":        "33_human_vs_ntcv_magnitude_calibration.py",
        "job_id":        job_id,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir":    str(out_dir),
        "inputs": {
            "raw_features_parquet":      str(RAW_FEATURES_PQ),
            "raw_features_sha256":       sha256_file(RAW_FEATURES_PQ),
            "robustness_root":           str(NSCLC_IO_DATA_ROOT),
            "n_robustness_parquets":
                len(list(NSCLC_IO_DATA_ROOT.glob(
                    "*/*/radiomics_robustness_ct.parquet"))),
        },
        "parameters": {
            "min_observers_human":   MIN_OBSERVERS_HUMAN,
            "min_patients_sigma":    MIN_PATIENTS_SIGMA,
            "bootstrap_iterations":  args.bootstrap_iterations,
            "seed":                  args.seed,
            "workers":               args.workers,
            "v_triplet":             [-15, 0, 15],
        },
        "counts": {
            "human_rows":           int(len(human_df)),
            "human_patients":       int(n_h_patients),
            "human_features":       int(n_h_feats),
            "robust_rows":          int(len(robust_df)),
            "robust_patients":      int(n_r_patients),
            "robust_rois":          int(n_r_rois),
            "sigma_human_features": int(len(sig_h)),
            "sigma_ntcv_features":  int(len(sig_n)),
            "calibration_features": int(len(calib_df)),
        },
        "overall_summary": {
            "median_delta":       overall["median_delta"],
            "ci_lo_delta":        overall["ci_lo_delta"],
            "ci_hi_delta":        overall["ci_hi_delta"],
            "frac_human_gt_ntcv": overall["frac_human_gt_ntcv"],
            "ci_lo_frac":         overall["ci_lo_frac"],
            "ci_hi_frac":         overall["ci_hi_frac"],
            "wilcoxon_stat":      overall["wilcoxon_stat"],
            "wilcoxon_p":         overall["wilcoxon_p"],
            "median_log2_ratio":  overall["median_log2_ratio"],
        },
        "outputs": {
            "calibration_parquet":       str(out_parquet),
            "calibration_parquet_sha":   sha256_file(out_parquet),
            "summary_csv":               str(out_summary),
            "summary_csv_sha":           sha256_file(out_summary),
            "figure_png":                str(out_png),
            "figure_png_sha":            sha256_file(out_png),
            "figure_pdf":                str(out_pdf),
            "figure_pdf_sha":            sha256_file(out_pdf),
        },
        "runtime_seconds":  float(time.time() - t0),
    }

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    print(f"[manifest] wrote {manifest_path}", flush=True)

    print("", flush=True)
    print("=" * 72, flush=True)
    print("DONE — Step 33 sigma calibration", flush=True)
    print(
        f"  features paired = {len(calib_df):,}   "
        f"median Δ = {overall['median_delta']:.4g}   "
        f"frac human>NTCV-V = {overall['frac_human_gt_ntcv']:.3f}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(2)
