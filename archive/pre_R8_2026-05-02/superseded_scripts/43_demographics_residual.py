#!/usr/bin/env python3
"""Demographics residual extension of #32 for unlocked Polish cohorts.

This script reruns the #32 crossed random-effects variance decomposition on
Prostata and Odbytnice, then refits the same model with available clinical
demographics as fixed effects. Baseline and adjusted models are always fit on
the same complete-case patient subset for a fair before/after comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

DATA_ROOT = Path("/home/kgs24/rtpipeline_manuscript")
AGGREGATE_DIR = DATA_ROOT / "analysis" / "data" / "robustness_by_cohort"
PROSTATA_META = Path("/projekty/Prostata/Data_Snakemake/metadata_summary.xlsx")
ODBYTNICE_CLIN = Path("/umed-projekty/ODBYTNICE2026/analysis/clinical_clean.parquet")
ODBYTNICE_MAP = Path("/umed-projekty/ODBYTNICE2026/analysis/patient_id_pesel_mapping.csv")

FEATURE_FAMILIES = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
COHORT_REGION = {"Prostata": "Pelvis", "Odbytnice": "Pelvis"}
RNG_SEED = 42
MIN_NONMISSING_COVARIATE = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohorts",
        nargs="*",
        default=["Prostata", "Odbytnice"],
        choices=["Prostata", "Odbytnice"],
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument(
        "--max-features-per-family",
        type=int,
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def normalize_patient_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_pesel(value: Any) -> str:
    text = normalize_patient_id(value)
    if not text:
        return ""
    return text.zfill(11)


def make_subject_id(patient_id: Any, course_id: Any) -> str:
    patient = normalize_patient_id(patient_id)
    course = normalize_patient_id(course_id)
    if patient and course:
        return f"{patient}_{course}"
    return patient or course


def clean_sex(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"m", "male", "man"}:
        return "M"
    if text in {"f", "female", "woman"}:
        return "F"
    return None


def clean_categorical_number(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if math.isnan(number):
        return None
    return str(int(number)) if number.is_integer() else str(number)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    mean = float(numeric.mean())
    return (numeric - mean) / std


def parse_perturbation_id(pid: str) -> dict[str, int]:
    noise, translation, contour, volume = 0, 0, 0, 0
    match = re.search(r"_n(\d+)", pid)
    if match:
        noise = int(match.group(1))
    match = re.search(r"_t0_0_(-?\d+)", pid)
    if match:
        translation = int(match.group(1))
    match = re.search(r"_c(\d+)", pid)
    if match:
        contour = int(match.group(1))
    match = re.search(r"_v([+-]?\d+)", pid)
    if match:
        volume = int(match.group(1))
    return {
        "fx_N": noise,
        "fx_T": translation,
        "fx_C": contour,
        "fx_V": volume,
    }


def get_feature_family(feature_name: str) -> str:
    low = feature_name.lower()
    for family in FEATURE_FAMILIES:
        if f"_{family}_" in low:
            return family
    return "unknown"


def list_cohort_patient_ids(cohort: str) -> list[str]:
    cohort_dir = DATA_ROOT / cohort / "data"
    return sorted(path.name for path in cohort_dir.iterdir() if path.is_dir())


def load_cohort_long(cohort: str, patient_ids: list[str]) -> pd.DataFrame:
    aggregate_path = AGGREGATE_DIR / f"{cohort}.parquet"
    if aggregate_path.exists():
        try:
            df = pd.read_parquet(
                aggregate_path,
                columns=[
                    "patient_id",
                    "course_id",
                    "roi_name",
                    "perturbation_id",
                    "feature_name",
                    "value",
                ],
                filters=[("patient_id", "in", patient_ids)],
            )
            df["patient_id"] = df["patient_id"].map(normalize_patient_id)
            df["course_id"] = df["course_id"].map(normalize_patient_id)
            return df[df["patient_id"].isin(patient_ids)].copy()
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [{cohort}] aggregate parquet read failed ({type(exc).__name__}: {exc}); "
                "falling back to per-patient scan.",
                flush=True,
            )

    frames: list[pd.DataFrame] = []
    cohort_dir = DATA_ROOT / cohort / "data"
    for patient_id in patient_ids:
        patient_dir = cohort_dir / patient_id
        for path in sorted(patient_dir.rglob("radiomics_robustness_ct.parquet")):
            df = pd.read_parquet(
                path,
                columns=[
                    "patient_id",
                    "course_id",
                    "roi_name",
                    "perturbation_id",
                    "feature_name",
                    "value",
                ],
            )
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["patient_id"] = out["patient_id"].map(normalize_patient_id)
    out["course_id"] = out["course_id"].map(normalize_patient_id)
    return out


def attach_ntcv(df: pd.DataFrame) -> pd.DataFrame:
    unique_ids = df["perturbation_id"].drop_duplicates()
    parsed = {pid: parse_perturbation_id(pid) for pid in unique_ids}
    map_df = pd.DataFrame.from_dict(parsed, orient="index").reset_index()
    map_df.columns = ["perturbation_id", "fx_N", "fx_T", "fx_C", "fx_V"]
    return df.merge(map_df, on="perturbation_id", how="left")


def load_prostata_covariates() -> pd.DataFrame:
    meta = pd.read_excel(PROSTATA_META)
    meta["patient_id"] = meta["patient_id"].map(normalize_patient_id)
    out = pd.DataFrame(
        {
            "patient_id": meta["patient_id"],
            "age": pd.to_numeric(meta.get("patient_age_at_plan"), errors="coerce"),
            "sex": meta.get("patient_sex").map(clean_sex),
            "bmi": pd.to_numeric(meta.get("patient_bmi"), errors="coerce"),
        }
    )
    return out.drop_duplicates("patient_id")


def load_odbytnice_covariates() -> pd.DataFrame:
    clin = pd.read_parquet(ODBYTNICE_CLIN)
    mapping = pd.read_csv(ODBYTNICE_MAP, dtype={"patient_id": str, "PESEL": str})
    mapping["patient_id"] = mapping["patient_id"].map(normalize_patient_id)
    mapping["PESEL"] = mapping["PESEL"].map(normalize_pesel)
    clin = clin.copy()
    clin["PESEL"] = clin["PESEL"].map(normalize_pesel)
    merged = mapping.merge(clin, on="PESEL", how="left")
    out = pd.DataFrame(
        {
            "patient_id": merged["patient_id"].map(normalize_patient_id),
            "age": pd.to_numeric(
                merged["demographics.age_at_treatment"], errors="coerce"
            ),
            "sex": merged["demographics.sex"].map(clean_sex),
            "bmi": pd.to_numeric(merged["demographics.bmi"], errors="coerce"),
            "ecog": merged["demographics.ecog_ps"].map(clean_categorical_number),
            "asa": merged["demographics.asa_score"].map(clean_categorical_number),
        }
    )
    return out.drop_duplicates("patient_id")


def load_covariates(cohort: str) -> pd.DataFrame:
    if cohort == "Prostata":
        return load_prostata_covariates()
    if cohort == "Odbytnice":
        return load_odbytnice_covariates()
    raise ValueError(f"Unsupported cohort: {cohort}")


def choose_covariates(
    patient_cov: pd.DataFrame,
    requested: list[str],
    min_nonmissing: int = MIN_NONMISSING_COVARIATE,
) -> tuple[list[str], dict[str, str]]:
    selected: list[str] = []
    dropped: dict[str, str] = {}
    for covariate in requested:
        if covariate not in patient_cov.columns:
            dropped[covariate] = "missing_column"
            continue
        series = patient_cov[covariate]
        nonmissing = int(series.notna().sum())
        unique_values = series.dropna().astype(str).nunique()
        if nonmissing < min_nonmissing:
            dropped[covariate] = f"too_sparse_n={nonmissing}"
            continue
        if unique_values < 2:
            dropped[covariate] = "single_level"
            continue
        selected.append(covariate)
    return selected, dropped


def make_patient_covariates(
    cohort: str,
    requested: list[str],
    sample_patients: list[str],
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    full_cov = load_covariates(cohort)
    available = full_cov[full_cov["patient_id"].isin(sample_patients)].copy()
    selected, dropped = choose_covariates(available, requested)
    if not selected:
        return available[["patient_id"]].drop_duplicates(), [], [], dropped

    complete = available.dropna(subset=selected).copy()
    if complete.empty:
        return available[["patient_id"]].drop_duplicates(), [], [], {
            **dropped,
            **{name: "no_complete_cases" for name in selected},
        }

    out = complete[["patient_id"]].copy()
    formula_terms: list[str] = []
    used_labels: list[str] = []
    for covariate in selected:
        if covariate in {"age", "bmi"}:
            col = f"{covariate}_z"
            out[col] = zscore(complete[covariate])
            formula_terms.append(col)
            used_labels.append(covariate)
        else:
            out[covariate] = complete[covariate].astype(str)
            formula_terms.append(f"C({covariate})")
            used_labels.append(covariate)
    return out, formula_terms, used_labels, dropped


def select_structures(df: pd.DataFrame, max_structures: int | None) -> list[str]:
    counts = (
        df.groupby("roi_name")["patient_id"].nunique().sort_values(ascending=False)
    )
    if max_structures is None or max_structures <= 0:
        return counts.index.tolist()
    return counts.head(max_structures).index.tolist()


def select_features(
    df: pd.DataFrame,
    rng: np.random.Generator,
    max_features_per_family: int | None,
) -> list[str]:
    all_features = sorted(df["feature_name"].unique().tolist())
    buckets: dict[str, list[str]] = {family: [] for family in FEATURE_FAMILIES}
    buckets["unknown"] = []
    for feature in all_features:
        buckets.setdefault(get_feature_family(feature), []).append(feature)
    chosen: list[str] = []
    for family, features in buckets.items():
        if not features:
            continue
        if max_features_per_family is None or max_features_per_family <= 0:
            chosen.extend(features)
            continue
        n_take = min(len(features), max_features_per_family)
        indices = rng.choice(len(features), size=n_take, replace=False)
        chosen.extend(features[i] for i in sorted(indices.tolist()))
    return sorted(set(chosen))


def fit_vardec_statsmodels(df: pd.DataFrame, extra_terms: list[str]) -> dict[str, Any]:
    model_df = df.copy()
    model_df["_grp"] = 0
    model_df["subject"] = model_df["subject"].astype("category")
    model_df["structure"] = model_df["structure"].astype("category")
    model_df["sxs"] = model_df["sxs"].astype("category")
    for column in ("fx_N", "fx_T", "fx_C", "fx_V"):
        model_df[column] = model_df[column].astype("category")
    for column in ("sex", "ecog", "asa"):
        if column in model_df.columns:
            model_df[column] = model_df[column].astype("category")

    n_structures = int(model_df["structure"].nunique())
    drop_structure_re = n_structures < 3
    fixed_terms = [*extra_terms, "C(fx_N)", "C(fx_T)", "C(fx_C)", "C(fx_V)"]
    if drop_structure_re:
        fixed_terms = ["C(structure)", *fixed_terms]
    formula = "y ~ " + " + ".join(fixed_terms)

    vc_formula: dict[str, str] = {
        "subject": "0 + C(subject)",
        "sxs": "0 + C(sxs)",
    }
    if not drop_structure_re:
        vc_formula["structure"] = "0 + C(structure)"

    try:
        model = smf.mixedlm(formula, data=model_df, groups="_grp", vc_formula=vc_formula)
        try:
            fit = model.fit(reml=True, method="lbfgs", maxiter=200)
        except Exception:
            fit = model.fit(reml=True, method="powell", maxiter=200)
    except Exception as exc:  # noqa: BLE001
        return {"status": "fit_failed", "error": repr(exc), "formula": formula}

    try:
        components = dict(zip(vc_formula.keys(), np.asarray(fit.vcomp, dtype=float).tolist()))
        var_subject = float(components["subject"])
        var_sxs = float(components["sxs"])
        var_structure = float(components.get("structure", 0.0))
        var_residual = float(fit.scale)
    except Exception as exc:  # noqa: BLE001
        return {"status": "extract_failed", "error": repr(exc), "formula": formula}

    try:
        design = patsy.dmatrix(" + ".join(fixed_terms), data=model_df, return_type="dataframe")
        common = [column for column in design.columns if column in fit.fe_params.index]
        if common:
            fixed_lp = design[common].to_numpy() @ fit.fe_params.loc[common].to_numpy()
            var_fixed = float(np.var(fixed_lp, ddof=1))
        else:
            var_fixed = 0.0
    except Exception:
        total_y_var = float(np.var(model_df["y"].to_numpy(), ddof=1))
        var_fixed = max(
            total_y_var - (var_subject + var_structure + var_sxs + var_residual),
            0.0,
        )

    values = np.array(
        [var_subject, var_structure, var_sxs, var_fixed, var_residual],
        dtype=float,
    )
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    total = float(values.sum())
    if total <= 0:
        return {"status": "zero_total", "formula": formula}
    fractions = values / total
    return {
        "status": "ok",
        "formula": formula,
        "converged": bool(getattr(fit, "converged", True)),
        "structure_re_used": not drop_structure_re,
        "var_subject": float(values[0]),
        "var_structure": float(values[1]),
        "var_subject_x_structure": float(values[2]),
        "var_fixed": float(values[3]),
        "var_residual": float(values[4]),
        "frac_subject": float(fractions[0]),
        "frac_structure": float(fractions[1]),
        "frac_subject_x_structure": float(fractions[2]),
        "frac_fixed": float(fractions[3]),
        "frac_residual": float(fractions[4]),
    }


def _feature_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frame = payload["frame"]
    df = pd.DataFrame(frame)
    baseline = fit_vardec_statsmodels(df, [])
    adjusted = fit_vardec_statsmodels(df, payload["adjust_terms"])
    rows: list[dict[str, Any]] = []
    for label, result in [
        (payload["baseline_label"], baseline),
        (payload["adjusted_label"], adjusted),
    ]:
        if result.get("status") != "ok":
            rows.append(
                {
                    "cohort": payload["cohort"],
                    "analysis_id": payload["analysis_id"],
                    "model_label": label,
                    "feature": payload["feature"],
                    "feature_family": payload["feature_family"],
                    "status": result.get("status"),
                    "error": result.get("error", ""),
                    "formula": result.get("formula", ""),
                    "covariates_used": payload["covariates_used"],
                    "n_patients": payload["n_patients"],
                    "n_structures": payload["n_structures"],
                    "n_obs": payload["n_obs"],
                }
            )
            continue
        row = {
            "cohort": payload["cohort"],
            "region": payload["region"],
            "analysis_id": payload["analysis_id"],
            "model_label": label,
            "feature": payload["feature"],
            "feature_family": payload["feature_family"],
            "status": "ok",
            "covariates_used": payload["covariates_used"],
            "formula": result["formula"],
            "n_patients": payload["n_patients"],
            "n_structures": payload["n_structures"],
            "n_obs": payload["n_obs"],
            "converged": result["converged"],
            "structure_re_used": result["structure_re_used"],
        }
        for name in [
            "var_subject",
            "var_structure",
            "var_subject_x_structure",
            "var_fixed",
            "var_residual",
            "frac_subject",
            "frac_structure",
            "frac_subject_x_structure",
            "frac_fixed",
            "frac_residual",
        ]:
            row[name] = result[name]
        rows.append(row)
    return rows


def prepare_payloads(
    cohort: str,
    analysis_id: str,
    raw: pd.DataFrame,
    adjust_terms: list[str],
    covariates_used: list[str],
    max_structures: int,
    max_features_per_family: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    structures = select_structures(raw, max_structures)
    trimmed = raw[raw["roi_name"].isin(structures)].copy()
    features = select_features(trimmed, rng, max_features_per_family)
    trimmed = trimmed[trimmed["feature_name"].isin(features)].copy()
    trimmed = trimmed.sort_values(
        ["feature_name", "patient_id", "course_id", "roi_name", "perturbation_id"]
    )

    payloads: list[dict[str, Any]] = []
    skipped = 0
    for feature, sub in trimmed.groupby("feature_name", sort=False):
        if sub["patient_id"].nunique() < 3 or sub["roi_name"].nunique() < 2:
            skipped += 1
            continue
        y = sub["value"].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(y)
        if finite.sum() < 100:
            skipped += 1
            continue
        y = y[finite]
        y_std = float(np.std(y, ddof=1))
        if not np.isfinite(y_std) or y_std == 0:
            skipped += 1
            continue
        patient = sub.loc[finite, "patient_id"].astype(str).to_numpy(dtype=object)
        course = sub.loc[finite, "course_id"].astype(str).to_numpy(dtype=object)
        subject = np.array(
            [make_subject_id(patient_id, course_id) for patient_id, course_id in zip(patient, course)],
            dtype=object,
        )
        structure = sub.loc[finite, "roi_name"].astype(str).to_numpy(dtype=object)
        frame: dict[str, Any] = {
            "y": (y - float(np.mean(y))) / y_std,
            "subject": subject,
            "structure": structure,
            "sxs": np.array([f"{u}__{s}" for u, s in zip(subject, structure)], dtype=object),
            "fx_N": sub.loc[finite, "fx_N"].to_numpy(dtype=int),
            "fx_T": sub.loc[finite, "fx_T"].to_numpy(dtype=int),
            "fx_C": sub.loc[finite, "fx_C"].to_numpy(dtype=int),
            "fx_V": sub.loc[finite, "fx_V"].to_numpy(dtype=int),
        }
        for covariate in ["age_z", "bmi_z"]:
            if covariate in sub.columns:
                frame[covariate] = sub.loc[finite, covariate].to_numpy(dtype=float)
        for covariate in ["sex", "ecog", "asa"]:
            if covariate in sub.columns:
                frame[covariate] = sub.loc[finite, covariate].to_numpy(dtype=object)
        payloads.append(
            {
                "cohort": cohort,
                "region": COHORT_REGION[cohort],
                "analysis_id": analysis_id,
                "baseline_label": f"baseline_{analysis_id}",
                "adjusted_label": f"adjusted_{analysis_id}",
                "feature": feature,
                "feature_family": get_feature_family(feature),
                "covariates_used": ",".join(covariates_used),
                "adjust_terms": adjust_terms,
                "n_patients": int(pd.Series(patient).nunique()),
                "n_subjects": int(pd.Series(subject).nunique()),
                "n_structures": int(pd.Series(structure).nunique()),
                "n_obs": int(len(y)),
                "frame": frame,
            }
        )

    meta = {
        "structures": structures,
        "selected_features": features,
        "n_features_selected": len(features),
        "n_payloads": len(payloads),
        "n_skipped_features": skipped,
    }
    return payloads, meta


def summarize_pair(results: pd.DataFrame, baseline_label: str, adjusted_label: str) -> dict[str, Any]:
    ok = results[results["status"] == "ok"].copy()
    base = ok[ok["model_label"] == baseline_label]
    adj = ok[ok["model_label"] == adjusted_label]
    merged = base.merge(
        adj,
        on=["cohort", "analysis_id", "feature", "feature_family"],
        suffixes=("_base", "_adj"),
    )
    summary: dict[str, Any] = {
        "features_compared": int(len(merged)),
    }
    for component in [
        "frac_subject",
        "frac_structure",
        "frac_subject_x_structure",
        "frac_fixed",
        "frac_residual",
    ]:
        base_median = float(base[component].median()) if not base.empty else float("nan")
        adj_median = float(adj[component].median()) if not adj.empty else float("nan")
        delta_median = float((merged[f"{component}_adj"] - merged[f"{component}_base"]).median()) if not merged.empty else float("nan")
        summary[f"{component}_baseline_median"] = base_median
        summary[f"{component}_adjusted_median"] = adj_median
        summary[f"{component}_delta_median"] = delta_median
    return summary


def run_analysis(
    cohort: str,
    analysis_id: str,
    requested_covariates: list[str],
    available_patient_ids: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    covars, adjust_terms, covariates_used, dropped = make_patient_covariates(
        cohort=cohort,
        requested=requested_covariates,
        sample_patients=available_patient_ids,
    )
    if adjust_terms:
        eligible_patients = sorted(covars["patient_id"].dropna().unique().tolist())
    else:
        eligible_patients = available_patient_ids

    rng = np.random.default_rng(seed)
    if args.max_patients and len(eligible_patients) > args.max_patients:
        sampled_patients = sorted(
            rng.choice(eligible_patients, size=args.max_patients, replace=False).tolist()
        )
    else:
        sampled_patients = eligible_patients

    print(
        f"  [{cohort}/{analysis_id}] covariates_used={covariates_used or ['(none)']} "
        f"eligible={len(eligible_patients)} sampled={len(sampled_patients)} "
        f"max_patients={args.max_patients or 'ALL'}",
        flush=True,
    )

    covars = covars[covars["patient_id"].isin(sampled_patients)].copy()
    print(f"  [{cohort}/{analysis_id}] loading sampled patient parquets ...", flush=True)
    subset = load_cohort_long(cohort, sampled_patients)
    subset = attach_ntcv(subset)
    print(
        f"  [{cohort}/{analysis_id}] loaded rows={len(subset):,} "
        f"structures={subset['roi_name'].nunique()} features={subset['feature_name'].nunique()}",
        flush=True,
    )
    subset = subset.merge(covars, on="patient_id", how="inner" if adjust_terms else "left")

    payloads, prep_meta = prepare_payloads(
        cohort=cohort,
        analysis_id=analysis_id,
        raw=subset,
        adjust_terms=adjust_terms,
        covariates_used=covariates_used,
        max_structures=args.max_structures,
        max_features_per_family=args.max_features_per_family,
        seed=seed,
    )
    print(
        f"  [{cohort}/{analysis_id}] payloads={len(payloads)} "
        f"selected_features={prep_meta['n_features_selected']} "
        f"skipped={prep_meta['n_skipped_features']}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    if payloads:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_feature_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                try:
                    rows.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "cohort": cohort,
                            "analysis_id": analysis_id,
                            "model_label": f"worker_failure_{analysis_id}",
                            "feature": "",
                            "feature_family": "",
                            "status": "future_exception",
                            "error": repr(exc),
                            "formula": "",
                            "covariates_used": ",".join(covariates_used),
                            "n_patients": len(sampled_patients),
                            "n_subjects": 0,
                            "n_structures": len(prep_meta["structures"]),
                            "n_obs": 0,
                        }
                    )

    results = pd.DataFrame(rows)
    baseline_label = f"baseline_{analysis_id}"
    adjusted_label = f"adjusted_{analysis_id}"
    comparison = summarize_pair(results, baseline_label, adjusted_label) if not results.empty else {}
    meta = {
        "cohort": cohort,
        "analysis_id": analysis_id,
        "requested_covariates": requested_covariates,
        "covariates_used": covariates_used,
        "covariates_dropped": dropped,
        "total_radiomics_patients": len(available_patient_ids),
        "eligible_patients_after_covariates": len(eligible_patients),
        "sampled_patients": len(sampled_patients),
        "sampled_patient_ids": sampled_patients,
        **prep_meta,
        **comparison,
        "baseline_label": baseline_label,
        "adjusted_label": adjusted_label,
    }
    return results, meta


def print_meta(meta: dict[str, Any]) -> None:
    print(f"\n[{meta['cohort']} / {meta['analysis_id']}]")
    print(f"  requested covariates: {', '.join(meta['requested_covariates'])}")
    print(f"  used covariates: {', '.join(meta['covariates_used']) or '(none)'}")
    if meta["covariates_dropped"]:
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(meta["covariates_dropped"].items()))
        print(f"  dropped covariates: {dropped}")
    print(
        "  counts: "
        f"radiomics={meta['total_radiomics_patients']} "
        f"eligible={meta['eligible_patients_after_covariates']} "
        f"sampled={meta['sampled_patients']} "
        f"features={meta.get('features_compared', 0)} "
        f"selected={meta['n_features_selected']}"
    )
    print(f"  structures: {', '.join(meta['structures'])}")
    for component in [
        "frac_subject",
        "frac_structure",
        "frac_subject_x_structure",
        "frac_fixed",
        "frac_residual",
    ]:
        base = meta.get(f"{component}_baseline_median")
        adj = meta.get(f"{component}_adjusted_median")
        delta = meta.get(f"{component}_delta_median")
        if base is None:
            continue
        print(f"  {component}: baseline={base:.3f} adjusted={adj:.3f} delta={delta:+.3f}")


def main() -> int:
    args = parse_args()
    t0 = time.time()

    all_results: list[pd.DataFrame] = []
    run_meta: list[dict[str, Any]] = []
    for cohort in args.cohorts:
        available_patient_ids = list_cohort_patient_ids(cohort)
        print(
            f"\nPlanning cohort {cohort} ... available patients={len(available_patient_ids)}",
            flush=True,
        )

        primary_results, primary_meta = run_analysis(
            cohort=cohort,
            analysis_id="primary",
            requested_covariates=["age", "sex", "bmi"],
            available_patient_ids=available_patient_ids,
            args=args,
            seed=args.seed + (0 if cohort == "Prostata" else 100),
        )
        all_results.append(primary_results)
        run_meta.append(primary_meta)
        print_meta(primary_meta)

        if cohort == "Odbytnice":
            sensitivity_results, sensitivity_meta = run_analysis(
                cohort=cohort,
                analysis_id="ecog_sensitivity",
                requested_covariates=["age", "sex", "bmi", "ecog", "asa"],
                available_patient_ids=available_patient_ids,
                args=args,
                seed=args.seed + 200,
            )
            all_results.append(sensitivity_results)
            run_meta.append(sensitivity_meta)
            print_meta(sensitivity_meta)

    combined = pd.concat([df for df in all_results if not df.empty], ignore_index=True)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_csv, index=False)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 2),
        "args": vars(args),
        "runs": run_meta,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nCompleted in {time.time() - t0:.1f}s", flush=True)
    if args.out_json:
        print(f"Summary JSON: {args.out_json}", flush=True)
    if args.out_csv:
        print(f"Row CSV: {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
