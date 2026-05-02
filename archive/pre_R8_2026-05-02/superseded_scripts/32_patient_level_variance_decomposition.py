#!/usr/bin/env python3
"""
Step 32: Patient-level variance decomposition (crossed random effects).

The existing NTCV ANOVA (07_perturbation_decomposition.py) attributes
~12-19% of variance to Volume, ~0.6-2.3% to Translation, <0.1% to
Noise/Contour, and leaves ~83% as "residual". The manuscript calls that
residual "true biological inter-patient variability" — but that is
asserted, never decomposed.

This script decomposes that residual using a crossed random-effects model
fit per (cohort x feature):

    y_ijk = beta0 + beta_N * N_k + beta_T * T_k + beta_C * C_k + beta_V * V_k
            + u_patient_i + u_structure_j + u_patient:structure_ij
            + epsilon_ijk

Variance components (from lme4/statsmodels-style REML fit):
  sigma^2_patient           -> between-patient biology (canonical "biology")
  sigma^2_structure         -> between-structure (ROI identity)
  sigma^2_patient:structure -> patient-specific delineation / anatomy
                               interaction — reinforces the bottleneck
                               claim if large.
  sigma^2_fixed  (from NTCV) -> explained by perturbation factors
  sigma^2_residual          -> unmodeled noise (measurement / replicate)

Approach: model backend = statsmodels.MixedLM with vc_formula for
crossed variance components (pymer4 / lme4 attempted first, then
statsmodels if the R backend is unavailable).

For each feature within a cohort: n_patients x n_structures x 81
observations. We pool all structures so that between-structure and
patient:structure variance are identifiable.

Output (long-format):
  /home/kgs24/rtpipeline_manuscript/analysis/data/patient_level_variance.parquet
Summary:
  /home/kgs24/rtpipeline_manuscript/analysis/tables/patient_level_variance_summary.csv
Figure:
  /home/kgs24/rtpipeline_manuscript/analysis/figures/figure_patient_variance.{png,pdf}
Manifest:
  /home/kgs24/rtpipeline_manuscript/analysis/logs/32_manifest_<JOBID>.json

Memory strategy: load one aggregated cohort parquet at a time from
analysis/data/robustness_by_cohort/, then stream feature-by-feature and
write incremental per-cohort parquets. Any patient / structure / feature
subsampling is strictly opt-in via CLI, except for --smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/kgs24/rtpipeline_manuscript")
ANALYSIS_DIR = Path("/home/kgs24/rtpipeline_manuscript/analysis")
AGGREGATE_DIR = ANALYSIS_DIR / "data" / "robustness_by_cohort"
OUT_DATA = ANALYSIS_DIR / "data" / "patient_level_variance.parquet"
OUT_SUMMARY = ANALYSIS_DIR / "tables" / "patient_level_variance_summary.csv"
OUT_FIG_PNG = ANALYSIS_DIR / "figures" / "figure_patient_variance.png"
OUT_FIG_PDF = ANALYSIS_DIR / "figures" / "figure_patient_variance.pdf"
LOG_DIR = ANALYSIS_DIR / "logs"
INTERMEDIATE_DIR = ANALYSIS_DIR / "data" / "_patient_variance_intermediate"

COHORTS = {
    "Prostata":         {"region": "Pelvis",  "site": "Lodz"},
    "Odbytnice":        {"region": "Pelvis",  "site": "Lodz"},
    "Immunodozymetria": {"region": "Thorax",  "site": "Gdansk+Lodz"},
    "PlucaRCHT":        {"region": "Thorax",  "site": "Lodz"},
    "Hipokampy":        {"region": "Brain",   "site": "Lodz"},
}

# Smoke-test defaults only. Production runs should leave these uncapped.
SMOKE_MAX_PATIENTS = 40
SMOKE_MAX_STRUCTURES = 6
SMOKE_MAX_FEATURES_PER_FAMILY = 40
RNG_SEED = 42
FEATURE_FAMILIES = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]

# Model selection: try lme4 via pymer4 first, fall back to statsmodels.
BACKEND = os.environ.get("VARDEC_BACKEND", "auto").lower()  # auto|lme4|statsmodels


# ---------------------------------------------------------------------------
# Perturbation ID parsing (reuse logic from script 07)
# ---------------------------------------------------------------------------

def parse_perturbation_id(pid: str) -> dict:
    """Return integer factor levels for noise/translation/contour/volume.

    Keys are prefixed (fx_*) to avoid Patsy name collisions with its
    categorical wrapper ``C()``.
    """
    N, T, Cf, V = 0, 0, 0, 0
    m = re.search(r"_n(\d+)", pid)
    if m:
        N = int(m.group(1))
    m = re.search(r"_t0_0_(-?\d+)", pid)
    if m:
        T = int(m.group(1))
    m = re.search(r"_c(\d+)", pid)
    if m:
        Cf = int(m.group(1))
    m = re.search(r"_v([+-]?\d+)", pid)
    if m:
        V = int(m.group(1))
    return {"fx_N": N, "fx_T": T, "fx_C": Cf, "fx_V": V}


def get_feature_family(fname: str) -> str:
    low = fname.lower()
    for fam in FEATURE_FAMILIES:
        if f"_{fam}_" in low:
            return fam
    return "unknown"


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------

def probe_backend(preferred: str = "auto") -> str:
    """Try to load pymer4 (lme4). Fall back to statsmodels if broken.

    Returns the string 'lme4' or 'statsmodels'.
    """
    if preferred == "statsmodels":
        import statsmodels.api as _  # noqa: F401
        return "statsmodels"
    if preferred in ("auto", "lme4"):
        try:
            # Silence stderr noise from rpy2 while probing.
            import pymer4  # noqa: F401
            from pymer4.models import Lmer  # noqa: F401
            # Try a tiny sanity fit to catch R/rpy2 ABI breakage at probe time.
            rng = np.random.default_rng(0)
            df = pd.DataFrame({
                "y": rng.standard_normal(60),
                "g1": np.repeat(range(6), 10),
                "g2": np.tile(range(5), 12),
            })
            m = Lmer("y ~ 1 + (1|g1) + (1|g2)", data=df)
            m.fit(summarize=False)
            return "lme4"
        except Exception as exc:  # noqa: BLE001
            if preferred == "lme4":
                raise RuntimeError(
                    f"lme4/pymer4 backend requested but unavailable: {exc}"
                ) from exc
            print(
                f"[backend] pymer4/lme4 unavailable ({type(exc).__name__}: {exc}); "
                "falling back to statsmodels.MixedLM (vc_formula crossed RE).",
                flush=True,
            )
    import statsmodels.api as _  # noqa: F401
    return "statsmodels"


# ---------------------------------------------------------------------------
# Variance decomposition — lme4 backend
# ---------------------------------------------------------------------------

def fit_vardec_lme4(df: pd.DataFrame) -> Optional[dict]:
    """Fit crossed-RE model with lme4 via pymer4. Return variance components.

    df columns: y, patient, structure, pxs (patient:structure), N, T, C, V
    """
    from pymer4.models import Lmer

    formula = (
        "y ~ C(fx_N) + C(fx_T) + C(fx_C) + C(fx_V) "
        "+ (1|patient) + (1|structure) + (1|pxs)"
    )
    try:
        m = Lmer(formula, data=df)
        m.fit(REML=True, summarize=False)
    except Exception as exc:  # noqa: BLE001
        return {"status": "fit_failed", "error": repr(exc)}

    # pymer4 exposes ranef_var (DataFrame: Name, Var, Std, ...)
    try:
        rv = m.ranef_var
        # Index names are typically 'patient', 'structure', 'pxs', 'Residual'.
        var_patient = float(rv.loc["patient", "Var"])
        var_structure = float(rv.loc["structure", "Var"])
        var_pxs = float(rv.loc["pxs", "Var"])
        var_residual = float(rv.loc["Residual", "Var"])
    except Exception as exc:  # noqa: BLE001
        return {"status": "extract_failed", "error": repr(exc)}

    # Perturbation (fixed) variance: var of fitted linear-predictor minus the
    # grand mean contribution, i.e. variance of X * beta.
    try:
        # pymer4 returns fits as m.fits (fitted values on the response).
        fitted = np.asarray(m.fits)
        # Subtract the random-effects contribution to isolate fixed portion.
        # pymer4 does not always expose BLUPs cleanly; compute residuals from
        # observed and fitted, plus ranef_var, plus residual_var. The
        # perturbation-explained variance is approximated by var(X*beta).
        #
        # Better: compute var(X * beta_hat) directly using the design matrix.
        coefs = m.coefs["Estimate"].values
        # Build design matrix matching the formula (without intercept randoms).
        import patsy
        X = patsy.dmatrix("C(fx_N) + C(fx_T) + C(fx_C) + C(fx_V)", data=df, return_type="dataframe")
        if X.shape[1] != len(coefs):
            # Order-agnostic: fall back to sample variance of fitted minus RE.
            var_fixed = float(np.var(fitted, ddof=1))
        else:
            Xb = X.values @ coefs
            var_fixed = float(np.var(Xb, ddof=1))
    except Exception:
        fitted = np.asarray(getattr(m, "fits", df["y"].values))
        var_fixed = float(np.var(fitted, ddof=1))

    return {
        "status": "ok",
        "backend": "lme4",
        "var_patient": var_patient,
        "var_structure": var_structure,
        "var_patient_x_structure": var_pxs,
        "var_perturbation": var_fixed,
        "var_residual": var_residual,
        "converged": bool(getattr(m, "warnings", None) in (None, "", [])),
    }


# ---------------------------------------------------------------------------
# Variance decomposition — statsmodels backend (crossed RE via vc_formula)
# ---------------------------------------------------------------------------

def fit_vardec_statsmodels(df: pd.DataFrame) -> dict:
    """Fit crossed-RE model via statsmodels.MixedLM.

    MixedLM's API requires a single `groups` argument, but supports
    variance-components formulas that implement crossed random effects by
    declaring each as a separate VC within a pseudo-group. We force a
    single artificial group (all rows = 0) and declare patient, structure
    and patient:structure as independent variance components.

    This implementation is validated against lme4 in the statsmodels docs
    (see 'Variance Components Analysis'). For cohorts with only a small
    number of structures (e.g. n_structures==2) we drop the structure
    random effect and treat structure as a fixed effect — the RE term is
    unidentifiable with <3 levels and fits land on the variance boundary.
    """
    import statsmodels.formula.api as smf

    df = df.copy()
    df["_grp"] = 0  # single pseudo-group: all observations
    df["patient"] = df["patient"].astype("category")
    df["structure"] = df["structure"].astype("category")
    df["pxs"] = df["pxs"].astype("category")
    for col in ("fx_N", "fx_T", "fx_C", "fx_V"):
        df[col] = df[col].astype("category")

    n_struct = int(df["structure"].nunique())
    drop_structure_re = n_struct < 3

    # Crossed variance components. Structure added as RE only when n>=3.
    vc: dict[str, str] = {
        "patient":   "0 + C(patient)",
        "pxs":       "0 + C(pxs)",
    }
    if not drop_structure_re:
        vc["structure"] = "0 + C(structure)"

    if drop_structure_re:
        # Treat structure as fixed effect.
        formula = "y ~ C(structure) + C(fx_N) + C(fx_T) + C(fx_C) + C(fx_V)"
    else:
        formula = "y ~ C(fx_N) + C(fx_T) + C(fx_C) + C(fx_V)"
    try:
        md = smf.mixedlm(formula, data=df, groups="_grp", vc_formula=vc)
        try:
            mdf = md.fit(reml=True, method="lbfgs", maxiter=200)
        except Exception:
            mdf = md.fit(reml=True, method="powell", maxiter=200)
    except Exception as exc:  # noqa: BLE001
        return {"status": "fit_failed", "error": repr(exc)}

    # vcomp is an ndarray of variance components in the order of vc keys.
    try:
        vcomp = np.asarray(mdf.vcomp, dtype=float)
        vc_keys = list(vc.keys())
        var_map = dict(zip(vc_keys, vcomp.tolist()))
        var_patient = float(var_map["patient"])
        var_pxs = float(var_map["pxs"])
        if "structure" in var_map:
            var_structure = float(var_map["structure"])
        else:
            # Structure pulled into fixed effects: measure its variance
            # contribution from the design matrix contribution of C(structure).
            var_structure = float("nan")
        var_residual = float(mdf.scale)
    except Exception as exc:  # noqa: BLE001
        return {"status": "extract_failed", "error": repr(exc)}

    # Fixed-effect variance (perturbation + [structure if fixed]):
    # decompose into perturbation vs structure parts when structure is
    # a fixed effect.
    try:
        import patsy
        pert_des = "C(fx_N) + C(fx_T) + C(fx_C) + C(fx_V)"
        X_pert = patsy.dmatrix(pert_des, data=df, return_type="dataframe")
        beta_series = mdf.fe_params  # pandas Series
        common = [c for c in X_pert.columns if c in beta_series.index]
        if common:
            Xv = X_pert[common].values
            bv = beta_series.loc[common].values
            var_fixed_pert = float(np.var(Xv @ bv, ddof=1))
        else:
            var_fixed_pert = 0.0

        if drop_structure_re:
            X_struct = patsy.dmatrix(
                "0 + C(structure)", data=df, return_type="dataframe"
            )
            common_s = [c for c in X_struct.columns if c in beta_series.index]
            if common_s:
                Xs = X_struct[common_s].values
                bs = beta_series.loc[common_s].values
                var_structure = float(np.var(Xs @ bs, ddof=1))
            else:
                var_structure = 0.0
    except Exception:
        total_y_var = float(np.var(df["y"].values, ddof=1))
        var_structure = var_structure if np.isfinite(var_structure) else 0.0
        var_fixed_pert = max(
            total_y_var
            - (var_patient + var_structure + var_pxs + var_residual),
            0.0,
        )

    if not np.isfinite(var_structure):
        var_structure = 0.0

    converged = bool(getattr(mdf, "converged", True))
    return {
        "status": "ok",
        "backend": "statsmodels",
        "structure_re_used": not drop_structure_re,
        "var_patient": var_patient,
        "var_structure": var_structure,
        "var_patient_x_structure": var_pxs,
        "var_perturbation": var_fixed_pert,
        "var_residual": var_residual,
        "converged": converged,
    }


def fit_vardec(df: pd.DataFrame, backend: str) -> dict:
    if backend == "lme4":
        return fit_vardec_lme4(df)
    return fit_vardec_statsmodels(df)


# ---------------------------------------------------------------------------
# Worker (picklable) for parallel per-feature fits.
# ---------------------------------------------------------------------------

def _feature_worker(payload: dict) -> dict:
    """Fit a single feature's variance decomposition in a subprocess.

    payload keys: feature, cohort, region, backend, frame (dict of np arrays
    ready to become a DataFrame).
    """
    # Limit BLAS threads inside the worker so N workers x M threads doesn't
    # oversubscribe the SGE allocation.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        frame = payload["frame"]
        mf = pd.DataFrame(frame)
        if payload["backend"] == "lme4":
            r = fit_vardec_lme4(mf)
        else:
            r = fit_vardec_statsmodels(mf)
        r["feature"] = payload["feature"]
        r["cohort"] = payload["cohort"]
        r["region"] = payload["region"]
        r["n_patients"] = int(pd.Series(frame["patient"]).nunique())
        r["n_structures"] = int(pd.Series(frame["structure"]).nunique())
        r["n_obs"] = int(len(mf))
        return r
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "worker_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "feature": payload.get("feature"),
            "cohort": payload.get("cohort"),
            "region": payload.get("region"),
        }


# ---------------------------------------------------------------------------
# Data loading / wrangling
# ---------------------------------------------------------------------------

def load_cohort_long(cohort: str) -> pd.DataFrame:
    """Load one cohort as a long DataFrame.

    Prefer the aggregated per-cohort parquet produced by step 01. That file
    was regenerated after the historical patient_id path-index bug was fixed
    on 2026-04-18, so patient_id/course_id are now trustworthy there.
    Fall back to the raw per-course files only if the aggregated parquet is
    missing.
    """
    aggregate_path = AGGREGATE_DIR / f"{cohort}.parquet"
    columns = [
        "patient_id",
        "course_id",
        "roi_name",
        "perturbation_id",
        "feature_name",
        "value",
    ]
    if aggregate_path.exists():
        return pd.read_parquet(aggregate_path, columns=columns)

    cohort_dir = DATA_ROOT / cohort / "data"
    parquets = sorted(cohort_dir.rglob("radiomics_robustness_ct.parquet"))
    if not parquets:
        return pd.DataFrame()
    dfs = []
    for p in parquets:
        try:
            df = pd.read_parquet(p, columns=columns)
            dfs.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARN: {p}: {exc}", file=sys.stderr, flush=True)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_cohort_long_subset(
    cohort: str,
    rng: np.random.Generator,
    max_patients: Optional[int],
    max_structures: Optional[int],
    max_features_per_family: Optional[int],
    max_features: Optional[int],
) -> tuple[pd.DataFrame, dict]:
    """Stream a cohort parquet and materialize only the requested subset.

    This is the memory-safe path for SGE runs: first scan lightweight columns
    to choose patients / structures / features, then stream again and keep
    only the selected rows.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    aggregate_path = AGGREGATE_DIR / f"{cohort}.parquet"
    if not aggregate_path.exists():
        return load_cohort_long(cohort), {
            "selected_patients": None,
            "selected_structures": None,
            "selected_features": None,
            "all_feature_count": None,
        }

    structure_patients: dict[str, set[str]] = defaultdict(set)
    all_patients: set[str] = set()
    feature_buckets: dict[str, set[str]] = defaultdict(set)

    pf = pq.ParquetFile(aggregate_path)
    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=["patient_id", "roi_name", "feature_name"],
    ):
        meta = pa.Table.from_batches([batch]).to_pandas()
        all_patients.update(meta["patient_id"].dropna().astype(str).unique().tolist())

        dedup = meta[["roi_name", "patient_id"]].drop_duplicates()
        for roi_name, grp in dedup.groupby("roi_name", sort=False):
            structure_patients[str(roi_name)].update(
                grp["patient_id"].dropna().astype(str).tolist()
            )

        for feat in meta["feature_name"].dropna().astype(str).unique().tolist():
            feature_buckets[get_feature_family(feat)].add(feat)

    selected_patients = None
    if max_patients is not None and len(all_patients) > max_patients:
        selected_patients = sorted(
            rng.choice(sorted(all_patients), size=max_patients, replace=False).tolist()
        )

    if max_structures is not None:
        selected_structures = sorted(
            (
                pd.Series({k: len(v) for k, v in structure_patients.items()})
                .sort_values(ascending=False)
                .head(max_structures)
                .index.tolist()
            )
        )
    else:
        selected_structures = sorted(structure_patients)

    all_features = sorted({feat for feats in feature_buckets.values() for feat in feats})
    if max_features_per_family is None:
        selected_features = all_features
    else:
        chosen: list[str] = []
        for fam in FEATURE_FAMILIES + ["unknown"]:
            flist = sorted(feature_buckets.get(fam, set()))
            if not flist:
                continue
            k = min(len(flist), max_features_per_family)
            idx = rng.choice(len(flist), size=k, replace=False)
            chosen.extend([flist[i] for i in sorted(idx.tolist())])
        selected_features = sorted(set(chosen))
    if max_features is not None:
        selected_features = selected_features[:max_features]

    patient_vals = pa.array(selected_patients) if selected_patients is not None else None
    structure_vals = pa.array(selected_structures) if selected_structures else None
    feature_vals = pa.array(selected_features) if selected_features else None

    keep_columns = [
        "patient_id",
        "course_id",
        "roi_name",
        "perturbation_id",
        "feature_name",
        "value",
    ]
    parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=250_000, columns=keep_columns):
        mask = None
        if patient_vals is not None:
            cur = pc.is_in(batch.column("patient_id"), value_set=patient_vals)
            mask = cur if mask is None else pc.and_(mask, cur)
        if structure_vals is not None:
            cur = pc.is_in(batch.column("roi_name"), value_set=structure_vals)
            mask = cur if mask is None else pc.and_(mask, cur)
        if feature_vals is not None:
            cur = pc.is_in(batch.column("feature_name"), value_set=feature_vals)
            mask = cur if mask is None else pc.and_(mask, cur)
        filtered = batch if mask is None else batch.filter(mask)
        if filtered.num_rows:
            parts.append(pa.Table.from_batches([filtered]).to_pandas())

    if not parts:
        return pd.DataFrame(), {
            "selected_patients": selected_patients,
            "selected_structures": selected_structures,
            "selected_features": selected_features,
            "all_feature_count": len(all_features),
        }

    return pd.concat(parts, ignore_index=True), {
        "selected_patients": selected_patients,
        "selected_structures": selected_structures,
        "selected_features": selected_features,
        "all_feature_count": len(all_features),
    }


def attach_ntcv(df: pd.DataFrame) -> pd.DataFrame:
    """Attach N, T, C, V integer columns derived from perturbation_id."""
    uniq = df["perturbation_id"].drop_duplicates()
    parsed = {pid: parse_perturbation_id(pid) for pid in uniq}
    map_df = pd.DataFrame.from_dict(parsed, orient="index").reset_index()
    map_df.columns = ["perturbation_id", "fx_N", "fx_T", "fx_C", "fx_V"]
    return df.merge(map_df, on="perturbation_id", how="left")


def select_structures(df: pd.DataFrame, max_k: int) -> list[str]:
    """Pick the top-k structures ranked by number of distinct patients."""
    counts = (
        df.groupby("roi_name")["patient_id"].nunique().sort_values(ascending=False)
    )
    return counts.head(max_k).index.tolist()


# ---------------------------------------------------------------------------
# Per-cohort analysis
# ---------------------------------------------------------------------------

def process_cohort(
    cohort: str,
    meta: dict,
    rng: np.random.Generator,
    backend: str,
    max_features: Optional[int],
    max_patients: Optional[int],
    max_structures: Optional[int],
    max_features_per_family: Optional[int],
    intermediate_dir: Path,
    n_workers: int = 1,
) -> pd.DataFrame:
    region = meta["region"]
    print(f"\n{'='*66}", flush=True)
    print(f"Cohort: {cohort} ({region})", flush=True)
    print(f"{'='*66}", flush=True)

    t0 = time.time()
    use_subset_loader = any(
        x is not None
        for x in (max_patients, max_structures, max_features_per_family, max_features)
    )
    selection_meta = None
    if use_subset_loader:
        raw, selection_meta = load_cohort_long_subset(
            cohort=cohort,
            rng=rng,
            max_patients=max_patients,
            max_structures=max_structures,
            max_features_per_family=max_features_per_family,
            max_features=max_features,
        )
    else:
        raw = load_cohort_long(cohort)
    if raw.empty:
        print(f"  {cohort}: NO DATA", flush=True)
        return pd.DataFrame()

    print(f"  loaded {len(raw):,} rows in {time.time() - t0:.1f}s", flush=True)

    raw = attach_ntcv(raw)

    # Subsample patients if too many.
    patients = sorted(raw["patient_id"].astype(str).unique().tolist())
    if selection_meta and selection_meta.get("selected_patients") is not None:
        picked = selection_meta["selected_patients"]
        print(f"  subsampled to {len(picked)} / {len(patients)} patients", flush=True)
    elif max_patients is not None and len(patients) > max_patients:
        picked = sorted(
            rng.choice(patients, size=max_patients, replace=False).tolist()
        )
        raw = raw[raw["patient_id"].astype(str).isin(picked)]
        print(f"  subsampled to {len(picked)} / {len(patients)} patients", flush=True)
    else:
        picked = patients

    # Pick top structures when requested, otherwise keep all.
    if selection_meta and selection_meta.get("selected_structures") is not None:
        structures = selection_meta["selected_structures"]
    elif max_structures is not None:
        structures = select_structures(raw, max_structures)
        raw = raw[raw["roi_name"].isin(structures)]
    else:
        structures = sorted(raw["roi_name"].dropna().unique().tolist())
    print(f"  structures ({len(structures)}): {structures}", flush=True)

    all_features = sorted(raw["feature_name"].unique().tolist())
    total_feature_count = (
        selection_meta["all_feature_count"]
        if selection_meta and selection_meta.get("all_feature_count") is not None
        else len(all_features)
    )
    if selection_meta and selection_meta.get("selected_features") is not None:
        features = selection_meta["selected_features"]
    elif max_features_per_family is None:
        features = all_features
    else:
        # Balanced per-family subsampling: keeps the per-family figure honest.
        fam_buckets: dict[str, list[str]] = {fam: [] for fam in FEATURE_FAMILIES}
        fam_buckets["unknown"] = []
        for f in all_features:
            fam_buckets.setdefault(get_feature_family(f), []).append(f)
        features = []
        for fam, flist in fam_buckets.items():
            if not flist:
                continue
            k = min(len(flist), max_features_per_family)
            idx = rng.choice(len(flist), size=k, replace=False)
            features.extend([flist[i] for i in sorted(idx.tolist())])
        features = sorted(set(features))
    if max_features is not None:
        features = features[:max_features]
    raw = raw[raw["feature_name"].isin(features)]
    n_feat = len(features)
    print(
        f"  features to process: {n_feat} "
        f"(of {total_feature_count} total; "
        f"cap {max_features_per_family if max_features_per_family is not None else 'ALL'}/family)",
        flush=True,
    )

    # Index by feature for fast slicing.
    raw = raw.sort_values(["feature_name", "patient_id", "roi_name", "perturbation_id"])
    feature_groups = raw.groupby("feature_name", sort=False)

    # Build payloads (model frames) before entering the fit loop, so that
    # workers do not have to re-read the global DataFrame.
    payloads: list[dict] = []
    n_skipped_prep = 0
    for feat in features:
        try:
            sub = feature_groups.get_group(feat)
        except KeyError:
            n_skipped_prep += 1
            continue
        if sub["patient_id"].nunique() < 3 or sub["roi_name"].nunique() < 2:
            n_skipped_prep += 1
            continue
        yvals = sub["value"].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(yvals)
        if finite.sum() < 100:
            n_skipped_prep += 1
            continue
        yvals = yvals[finite]
        y_std = float(np.std(yvals, ddof=1))
        if y_std == 0 or not np.isfinite(y_std):
            n_skipped_prep += 1
            continue
        y_z = (yvals - float(np.mean(yvals))) / y_std
        pat_vals = sub.loc[finite, "patient_id"].astype(str).to_numpy(dtype=object)
        roi_vals = sub.loc[finite, "roi_name"].astype(str).to_numpy(dtype=object)
        pxs_vals = np.array(
            [f"{p}__{s}" for p, s in zip(pat_vals, roi_vals)], dtype=object
        )
        payloads.append({
            "feature": feat,
            "cohort": cohort,
            "region": region,
            "backend": backend,
            "frame": {
                "y": y_z,
                "patient": pat_vals,
                "structure": roi_vals,
                "pxs": pxs_vals,
                "fx_N": sub.loc[finite, "fx_N"].to_numpy(dtype=int),
                "fx_T": sub.loc[finite, "fx_T"].to_numpy(dtype=int),
                "fx_C": sub.loc[finite, "fx_C"].to_numpy(dtype=int),
                "fx_V": sub.loc[finite, "fx_V"].to_numpy(dtype=int),
            },
        })

    print(
        f"  payloads prepared: {len(payloads)} "
        f"(skipped {n_skipped_prep} features)",
        flush=True,
    )

    # Free the large raw DataFrame — each payload is self-contained.
    del raw, feature_groups

    results: list[dict] = []
    t_start = time.time()
    n_ok = 0
    n_fail = 0
    last_print = t_start
    n_done = 0

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = intermediate_dir / f"{cohort}.parquet"

    def _record(r: dict) -> None:
        nonlocal n_ok, n_fail
        if r.get("status") != "ok":
            n_fail += 1
            return
        comp = np.array([
            r.get("var_patient", 0.0),
            r.get("var_structure", 0.0),
            r.get("var_patient_x_structure", 0.0),
            r.get("var_perturbation", 0.0),
            r.get("var_residual", 0.0),
        ], dtype=float)
        comp = np.where(np.isfinite(comp) & (comp > 0), comp, 0.0)
        total = comp.sum()
        if total <= 0:
            n_fail += 1
            return
        frac = comp / total
        results.append({
            "cohort": r["cohort"],
            "region": r["region"],
            "feature": r["feature"],
            "feature_family": get_feature_family(r["feature"]),
            "n_patients": r.get("n_patients"),
            "n_structures": r.get("n_structures"),
            "n_obs": r.get("n_obs"),
            "backend": r.get("backend"),
            "converged": bool(r.get("converged", True)),
            "structure_re_used": bool(r.get("structure_re_used", True)),
            "var_patient": float(comp[0]),
            "var_structure": float(comp[1]),
            "var_patient_x_structure": float(comp[2]),
            "var_perturbation": float(comp[3]),
            "var_residual": float(comp[4]),
            "frac_patient": float(frac[0]),
            "frac_structure": float(frac[1]),
            "frac_patient_x_structure": float(frac[2]),
            "frac_perturbation": float(frac[3]),
            "frac_residual": float(frac[4]),
        })
        n_ok += 1

    def _progress_pulse(force: bool = False) -> None:
        nonlocal last_print
        now = time.time()
        if not force and (now - last_print) < 20:
            return
        elapsed = now - t_start
        rate = n_done / elapsed if elapsed > 0 else 0.0
        remaining = max(len(payloads) - n_done, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"  [{cohort}] {n_done}/{len(payloads)} feat  ok={n_ok} "
            f"fail={n_fail}  rate={rate:.2f}/s  eta={eta/60:.1f}min",
            flush=True,
        )
        last_print = now
        if results:
            pd.DataFrame(results).to_parquet(intermediate_path, index=False)

    if n_workers <= 1:
        for payload in payloads:
            r = _feature_worker(payload)
            _record(r)
            n_done += 1
            _progress_pulse()
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_feature_worker, p) for p in payloads]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {"status": "future_exception", "error": repr(exc)}
                _record(r)
                n_done += 1
                _progress_pulse()

    _progress_pulse(force=True)

    final_df = pd.DataFrame(results)
    if not final_df.empty:
        final_df.to_parquet(intermediate_path, index=False)
    print(
        f"  [{cohort}] done in {(time.time()-t_start)/60:.1f}min  "
        f"ok={n_ok} fail={n_fail}",
        flush=True,
    )
    return final_df


# ---------------------------------------------------------------------------
# Summary + figure
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    frac_cols = [
        "frac_patient",
        "frac_structure",
        "frac_patient_x_structure",
        "frac_perturbation",
        "frac_residual",
    ]
    agg = (
        df.groupby(["region", "feature_family"])
        .agg(
            n=("feature", "count"),
            n_features=("feature", "nunique"),
            **{f"{c}_median": (c, "median") for c in frac_cols},
            **{f"{c}_q25": (c, lambda x: float(np.quantile(x, 0.25))) for c in frac_cols},
            **{f"{c}_q75": (c, lambda x: float(np.quantile(x, 0.75))) for c in frac_cols},
        )
        .reset_index()
    )
    return agg


def make_figure(summary: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = ["Brain", "Pelvis", "Thorax"]
    regions = [r for r in regions if r in summary["region"].unique()]
    fams = [f for f in FEATURE_FAMILIES if f in summary["feature_family"].unique()]

    if not regions or not fams:
        print("  figure: no data to plot", flush=True)
        return

    comps = [
        ("frac_patient_median",           "Patient (biology)",              "#1f77b4"),
        ("frac_structure_median",         "Structure",                       "#ff7f0e"),
        ("frac_patient_x_structure_median", "Patient x Structure",          "#2ca02c"),
        ("frac_perturbation_median",      "Perturbation (fixed)",           "#d62728"),
        ("frac_residual_median",          "Residual (unmodeled)",           "#7f7f7f"),
    ]

    fig, axes = plt.subplots(
        1, len(regions), figsize=(3.2 * len(regions) + 1.5, 4.2), sharey=True
    )
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        sub = summary[summary["region"] == region].set_index("feature_family")
        sub = sub.reindex([f for f in fams if f in sub.index])
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for col, label, color in comps:
            vals = sub[col].values
            ax.bar(x, vals, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_title(region)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction of total variance")

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        "Patient-level variance decomposition\n"
        "(crossed random effects: patient, structure, interaction, perturbation, residual)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out_png}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 100-feature smoke test on a single cohort (Hipokampy).",
    )
    ap.add_argument(
        "--cohorts",
        nargs="*",
        default=None,
        help="Restrict processing to the given cohort names.",
    )
    ap.add_argument(
        "--backend",
        default=BACKEND,
        choices=["auto", "lme4", "statsmodels"],
        help="Model backend. 'auto' tries lme4 then statsmodels.",
    )
    ap.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Cap features per cohort (for smoke tests).",
    )
    ap.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Randomly subsample patients per cohort. Default: no patient subsampling.",
    )
    ap.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Keep only the top-k structures by patient coverage. Default: keep all structures.",
    )
    ap.add_argument(
        "--max-features-per-family",
        type=int,
        default=None,
        help="Balanced random feature subsampling per family. Default: keep all features.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NSLOTS", os.environ.get("VARDEC_WORKERS", "1"))),
        help="Parallel worker processes per cohort (default: NSLOTS).",
    )
    args = ap.parse_args()

    t_all = time.time()
    job_id = os.environ.get("JOB_ID", "interactive")
    host = socket.gethostname()
    started = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("Script 32 — Patient-level variance decomposition", flush=True)
    print(f"  host={host}  job_id={job_id}  started={started}", flush=True)
    print(f"  backend_pref={args.backend}  smoke={args.smoke}", flush=True)
    print("=" * 70, flush=True)

    backend = probe_backend(args.backend)
    print(f"[backend] resolved -> {backend}", flush=True)

    max_patients = args.max_patients
    max_structures = args.max_structures
    max_features_per_family = args.max_features_per_family
    if args.smoke:
        target_cohorts = {"Hipokampy": COHORTS["Hipokampy"]}
        max_feat = args.max_features or 100
        if max_patients is None:
            max_patients = SMOKE_MAX_PATIENTS
        if max_structures is None:
            max_structures = SMOKE_MAX_STRUCTURES
        if max_features_per_family is None:
            max_features_per_family = SMOKE_MAX_FEATURES_PER_FAMILY
    else:
        if args.cohorts:
            target_cohorts = {c: COHORTS[c] for c in args.cohorts if c in COHORTS}
        else:
            target_cohorts = COHORTS
        max_feat = args.max_features

    print(
        f"[run] cohorts={list(target_cohorts)}  "
        f"max_patients={max_patients if max_patients is not None else 'ALL'}  "
        f"max_structures={max_structures if max_structures is not None else 'ALL'}  "
        f"max_features_per_family={max_features_per_family if max_features_per_family is not None else 'ALL'}  "
        f"max_features={max_feat if max_feat is not None else 'ALL'}  "
        f"workers={args.workers}",
        flush=True,
    )

    rng = np.random.default_rng(RNG_SEED)
    all_parts: list[pd.DataFrame] = []
    per_cohort_stats: dict[str, dict] = {}

    for cohort, meta in target_cohorts.items():
        t_c = time.time()
        try:
            part = process_cohort(
                cohort=cohort,
                meta=meta,
                rng=rng,
                backend=backend,
                max_features=max_feat,
                max_patients=max_patients,
                max_structures=max_structures,
                max_features_per_family=max_features_per_family,
                intermediate_dir=INTERMEDIATE_DIR,
                n_workers=args.workers,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {cohort}: FATAL {exc}", flush=True)
            traceback.print_exc()
            per_cohort_stats[cohort] = {
                "status": "error",
                "error": repr(exc),
                "elapsed_s": time.time() - t_c,
            }
            continue
        per_cohort_stats[cohort] = {
            "status": "ok",
            "rows": int(len(part)),
            "elapsed_s": time.time() - t_c,
        }
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        print("ERROR: no results produced", file=sys.stderr, flush=True)
        return 1

    full = pd.concat(all_parts, ignore_index=True)
    full.to_parquet(OUT_DATA, index=False)
    print(f"\nsaved long-format variance: {OUT_DATA}  rows={len(full):,}", flush=True)

    summary = build_summary(full)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"saved summary: {OUT_SUMMARY}  rows={len(summary)}", flush=True)

    # Validation: variance fractions sum to ~1.
    frac_cols = [
        "frac_patient",
        "frac_structure",
        "frac_patient_x_structure",
        "frac_perturbation",
        "frac_residual",
    ]
    sums = full[frac_cols].sum(axis=1)
    ok_sum = float(((sums > 0.995) & (sums < 1.005)).mean())
    print(f"\nvalidation: fraction of rows with |sum-1|<0.005 = {ok_sum*100:.2f}%", flush=True)

    # Quick textual summary per region.
    print("\n" + "=" * 70, flush=True)
    print("Median variance fractions per region", flush=True)
    print("=" * 70, flush=True)
    for region in ["Brain", "Pelvis", "Thorax"]:
        sub = full[full["region"] == region]
        if sub.empty:
            continue
        print(
            f"  {region:7s}  pat={sub['frac_patient'].median():.3f}  "
            f"struct={sub['frac_structure'].median():.3f}  "
            f"pxs={sub['frac_patient_x_structure'].median():.3f}  "
            f"pert={sub['frac_perturbation'].median():.3f}  "
            f"resid={sub['frac_residual'].median():.3f}",
            flush=True,
        )

    try:
        make_figure(summary, OUT_FIG_PNG, OUT_FIG_PDF)
    except Exception as exc:  # noqa: BLE001
        print(f"  figure: FAILED ({exc})", flush=True)
        traceback.print_exc()

    manifest = {
        "job_id": job_id,
        "host": host,
        "started_utc": started,
        "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "elapsed_s": time.time() - t_all,
        "backend": backend,
        "backend_requested": args.backend,
        "smoke": args.smoke,
        "workers": args.workers,
        "cohorts_requested": list(target_cohorts),
        "max_patients": max_patients,
        "max_structures": max_structures,
        "max_features_per_family": max_features_per_family,
        "max_features": max_feat,
        "cohorts_processed": per_cohort_stats,
        "output_rows": int(len(full)),
        "validation_fraction_sum_near_one": ok_sum,
        "outputs": {
            "long_parquet": str(OUT_DATA),
            "summary_csv": str(OUT_SUMMARY),
            "figure_png": str(OUT_FIG_PNG),
            "figure_pdf": str(OUT_FIG_PDF),
        },
    }
    manifest_path = LOG_DIR / f"32_manifest_{job_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"manifest: {manifest_path}", flush=True)
    print(f"\ntotal elapsed: {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
