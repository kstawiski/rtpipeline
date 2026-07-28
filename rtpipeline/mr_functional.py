"""B3 — functional-MR sampling under anatomic masks.

Implements the triple-consensus-APPROVED design (B3_IMPLEMENTATION_BRIEF.md rev5;
plan-gate ledger triple_consensus_logs/B3_plan_gate_20260620T160645Z/ledger_allpass.json).

For each single 3-D scalar functional-MR map (ADC / scanner perfusion map / DCE
subtraction) of a patient, bring it into the patient's anatomic-MR ``total_mr`` mask
geometry (direct / same-FoR linear resample / rigid-MI fallback), then compute
per-structure summary statistics. DEFAULT-DENY: only explicitly-recognized quantitative
map subtypes are sampled; every other classified ``mr_functional`` series emits an
auditable QC row and is excluded from the primary table. DCE-PK modelling is DEFERRED.

Opt-in via ``config.mr_functional_sampling`` (default False); runs post-segmentation in
the all-series path (masks must already exist). A B3 failure never flips a series to
``seg_failed`` (it only reads masks).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:  # SimpleITK is a hard dep of the imaging pipeline; guard only for import-time safety.
    import SimpleITK as sitk
except Exception:  # pragma: no cover - environment guard
    sitk = None  # type: ignore

# --- token taxonomy (mirrors modality_classifier._MR_FUNCTIONAL_RE per-token, rev5) ---
# Standalone-token (\b…\b / set-membership) — NEVER substring (wi ⊂ twist!).
_DERIVED_MAP_TOKENS = frozenset(
    {"ttp", "pei", "relcbf", "relcbv", "relccbv", "relmtt", "pbp", "wo", "wi"}
)
_RAW_DYNAMIC_TOKENS = frozenset({"twist", "grasp"})  # standalone; "dyn" handled as \b-prefix
_MIP_TOKENS = frozenset({"mip", "maxip", "minip", "mipt"})
# \b-prefix, open-suffix tokens (token.startswith) — match _MR_FUNCTIONAL_RE.
_PREFIX_TOKENS_PERF = ("perf",)
_PREFIX_TOKENS_DYNAMIC = ("dyn",)
_PREFIX_TOKENS_DWI = ("diff", "ep2d")
_PREFIX_TOKENS_SUBTRACT = ("subtract",)
# substring (vendor-fused, no separator: IsoADC/dADC/IsoDWI/cDWI/DWIBS) — match :77-79.
_SUBSTR_ADC = "adc"
_SUBSTR_DWI = "dwi"

# Deformable pelvic organs: rigid registration cannot model filling/peristalsis between
# sequences; sampled values get a ``deformable_flag`` annotation on the rigid_mi tier.
_DEFORMABLE_SUBSTRINGS = ("bladder", "rectum", "colon", "bowel", "small_bowel", "duodenum")

# Registration acceptance (scale-invariant; NOT a raw-MI threshold) + audit thresholds.
MAX_TRANSLATION_MM = 80.0   # same-patient same-region rigid: implausible beyond this
MAX_ROTATION_DEG = 25.0
MIN_COVERAGE_FRAC = 0.70    # fraction of mask voxels that must receive an in-FOV F sample
LOW_NATIVE_VOXELS = 10      # below this, percentiles are interpolation-dominated → flag
GEOM_TOL = 1e-3             # origin/spacing/direction match tolerance

# QC terminal statuses (one per discovered candidate series).
QC_OK = "ok"
QC_NO_ANATOMIC = "no_anatomic_match"
QC_AMBIGUOUS_ANATOMIC = "ambiguous_anatomic_match"
QC_ANATOMIC_OUT_OF_SCOPE = "anatomic_out_of_scope"
QC_MULTIVOLUME = "multivolume_4d_pk_deferred"
QC_NON_SCALAR = "unsupported_non_scalar_functional"
QC_UNSUPPORTED = "unsupported_functional"
QC_RAW_SOURCE = "raw_source_pk_deferred"
QC_MIP = "mip_excluded"
QC_REG_FAILED = "reg_failed"
QC_EMPTY_MASK = "empty_mask"
QC_UNKNOWN_UNITS = "unknown_units"
QC_LOAD_FAILED = "load_failed"
QC_NOT_MATERIALIZED = "not_materialized"


@dataclass(slots=True)
class SubtypeRoute:
    subtype: str           # adc | perfusion | subtraction | dwi | <raw/unsupported>
    sampled: bool
    stats_kind: str | None  # "percentiles" | "mean_median" | None
    qc_reason: str | None   # set when not sampled


def _norm_tokens(description: str) -> tuple[str, set[str]]:
    """Normalize exactly like modality_classifier._classify_mr (:192-194)."""
    toks = re.findall(r"[a-z0-9]+", (description or "").lower())
    return " ".join(toks), set(toks)


def _has_prefix_token(tokset: set[str], prefixes: Sequence[str]) -> bool:
    return any(tok.startswith(p) for tok in tokset for p in prefixes)


def _image_type_is_mip(image_types: Sequence[str] | None) -> bool:
    if not image_types:
        return False
    joined = "\\".join(str(t) for t in image_types).upper()
    return any(m in joined for m in ("MIP", "MAXIP", "MINIP"))


def route_functional_subtype(
    description: str, image_types: Sequence[str] | None = None
) -> SubtypeRoute:
    """DEFAULT-DENY subtype routing (rev5 §4.2-step4), 3-bucket token matching.

    Priority: MIP-exclude → DERIVED_MAP(perfusion) → adc → subtraction → raw-perf(defer)
    → dwi → raw-dynamic(defer) → unsupported. Raw-`perf` defers BEFORE the DWI branch
    because ep2d_perf shares the `ep2d` token with ep2d_diff DWI (Claude-r3).
    """
    norm, tokset = _norm_tokens(description)

    # MIP: projection, not a 3-D quantitative map.
    if (tokset & _MIP_TOKENS) or _image_type_is_mip(image_types):
        return SubtypeRoute("mip", False, None, QC_MIP)

    # (a) derived perfusion/wash MAP — standalone token membership.
    if tokset & _DERIVED_MAP_TOKENS:
        return SubtypeRoute("perfusion", True, "mean_median", None)

    # (b) ADC — substring (vendor-fused IsoADC/dADC) OR spelled-out "apparent diffusion".
    if _SUBSTR_ADC in norm or "apparent diffusion" in norm:
        return SubtypeRoute("adc", True, "percentiles", None)

    # (c) subtraction — standalone `sub` token or `subtract`-prefix.
    if "sub" in tokset or _has_prefix_token(tokset, _PREFIX_TOKENS_SUBTRACT):
        return SubtypeRoute("subtraction", True, "mean_median", None)

    # (d) raw perfusion acquisition (perf-prefix, no derived-map token) — DEFER before DWI.
    if _has_prefix_token(tokset, _PREFIX_TOKENS_PERF):
        return SubtypeRoute("raw_perf", False, None, QC_RAW_SOURCE)

    # (e) DWI — substring (IsoDWI/cDWI/DWIBS) or diff/ep2d prefix.
    if _SUBSTR_DWI in norm or _has_prefix_token(tokset, _PREFIX_TOKENS_DWI):
        return SubtypeRoute("dwi", True, "percentiles", None)

    # (f) other raw dynamics (dyn-prefix / twist / grasp) — defer.
    if _has_prefix_token(tokset, _PREFIX_TOKENS_DYNAMIC) or (tokset & _RAW_DYNAMIC_TOKENS):
        return SubtypeRoute("raw_dynamic", False, None, QC_RAW_SOURCE)

    return SubtypeRoute("unsupported", False, None, QC_UNSUPPORTED)


# --- geometry / registration --------------------------------------------------------

def _geometry_matches(a: "sitk.Image", b: "sitk.Image") -> bool:
    if a.GetSize() != b.GetSize():
        return False
    for va, vb in zip(a.GetOrigin(), b.GetOrigin()):
        if abs(va - vb) > GEOM_TOL:
            return False
    for va, vb in zip(a.GetSpacing(), b.GetSpacing()):
        if abs(va - vb) > GEOM_TOL:
            return False
    for va, vb in zip(a.GetDirection(), b.GetDirection()):
        if abs(va - vb) > GEOM_TOL:
            return False
    return True


def _euler_translation_rotation(tx: "sitk.Transform") -> tuple[float, float]:
    """Return (max |translation| mm, max |rotation| deg) for an Euler3D transform."""
    try:
        params = list(tx.GetParameters())  # (rx, ry, rz, tx, ty, tz)
        rot = max(abs(math.degrees(p)) for p in params[0:3]) if len(params) >= 3 else 0.0
        trans = max(abs(p) for p in params[3:6]) if len(params) >= 6 else 0.0
        return trans, rot
    except Exception:
        return 0.0, 0.0


@dataclass(slots=True)
class ResampleResult:
    image: "sitk.Image | None"           # functional map resampled onto the anatomic grid
    tier: str                            # direct | resample_samefor | rigid_mi
    reg_converged: bool
    coverage_frac: float | None          # set/checked after masks known; here None
    transform: "sitk.Transform | None"
    qc_reason: str | None                # QC_REG_FAILED if rejected


def resample_functional_to_anatomic(
    functional: "sitk.Image",
    anatomic_geom: "sitk.Image",
    *,
    same_for: bool,
    default_value: float = float("nan"),
) -> ResampleResult:
    """Resample the functional MAP onto the anatomic-mask grid (DESIGN direction).

    same FoR + same grid → direct; same FoR + diff grid → linear resample; diff FoR →
    rigid-MI (Euler3D, Mattes 32-bin, multi-res) then linear resample. NaN-pad so coverage
    can be measured. Continuous map → LINEAR interpolation (NN is label-only).
    """
    if sitk is None:  # pragma: no cover
        raise RuntimeError("SimpleITK unavailable")
    functional = sitk.Cast(functional, sitk.sitkFloat32)

    if same_for and _geometry_matches(functional, anatomic_geom):
        return ResampleResult(functional, "direct", True, None, sitk.Transform(3, sitk.sitkIdentity), None)

    if same_for:
        out = sitk.Resample(
            functional, anatomic_geom, sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkLinear, default_value, sitk.sitkFloat32,
        )
        return ResampleResult(out, "resample_samefor", True, None, sitk.Transform(3, sitk.sitkIdentity), None)

    # Different FoR → rigid mutual-information registration. fixed=anatomic, moving=functional.
    fixed = sitk.Cast(anatomic_geom, sitk.sitkFloat32)
    moving = functional
    try:
        initial = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.20, seed=20260620)  # deterministic
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=200,
            convergenceMinimumValue=1e-6, convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(initial, inPlace=False)
        final_tx = reg.Execute(fixed, moving)
        stop = reg.GetOptimizerStopConditionDescription()
        converged = "exceeded" not in stop.lower() and reg.GetMetricValue() is not None
    except Exception as exc:
        logger.warning("B3 rigid registration raised: %s", exc)
        return ResampleResult(None, "rigid_mi", False, None, None, QC_REG_FAILED)

    trans, rot = _euler_translation_rotation(final_tx)
    if trans > MAX_TRANSLATION_MM or rot > MAX_ROTATION_DEG:
        logger.info("B3 registration implausible (|t|=%.1fmm |r|=%.1fdeg) → reg_failed", trans, rot)
        return ResampleResult(None, "rigid_mi", converged, None, final_tx, QC_REG_FAILED)

    out = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, default_value, sitk.sitkFloat32)
    return ResampleResult(out, "rigid_mi", converged, None, final_tx, None)


def coverage_fraction(resampled: "sitk.Image", mask_arr: np.ndarray, label: int) -> float:
    """Fraction of structure-mask voxels receiving an in-FOV (non-NaN) functional sample."""
    arr = sitk.GetArrayFromImage(resampled)
    sel = mask_arr == label
    n = int(sel.sum())
    if n == 0:
        return 0.0
    valid = np.isfinite(arr[sel])
    return float(valid.sum()) / float(n)


# --- per-structure statistics -------------------------------------------------------

def _structure_is_deformable(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in _DEFORMABLE_SUBSTRINGS)


def per_structure_stats(
    resampled: "sitk.Image",
    mask_arr: np.ndarray,
    label_map: Mapping[int, str],
    *,
    stats_kind: str,
    tier: str,
    native_count: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-structure stats on the resampled functional map under each label.

    ``mask_arr`` is on the resampled (anatomic) grid. ``native_count`` maps label→count of
    native-functional voxels (mask NN-resampled to F grid) for the low_native_voxels audit.
    """
    arr = sitk.GetArrayFromImage(resampled)
    rows: list[dict[str, Any]] = []
    for label, name in sorted(label_map.items()):
        sel = (mask_arr == label) & np.isfinite(arr)
        n = int(sel.sum())
        n_native = int(native_count.get(label, 0)) if native_count else None
        row: dict[str, Any] = {
            "structure_name": name,
            "n_voxels": n,
            "n_native_voxels": n_native,
            "low_native_voxels": (n_native is not None and n_native < LOW_NATIVE_VOXELS),
            "deformable_flag": (tier == "rigid_mi" and _structure_is_deformable(name)),
            "mean": None, "median": None, "p10": None, "p90": None,
            "qc_flag": QC_OK,
        }
        if n == 0:
            row["qc_flag"] = QC_EMPTY_MASK
            rows.append(row)
            continue
        vals = arr[sel].astype(np.float64)
        row["mean"] = float(np.mean(vals))
        row["median"] = float(np.median(vals))
        if stats_kind == "percentiles":
            row["p10"] = float(np.percentile(vals, 10))
            row["p90"] = float(np.percentile(vals, 90))
        rows.append(row)
    return rows


# --- anatomic-MR selection (geometry-first, deterministic) --------------------------

def select_anatomic_for_functional(
    functional_for_uid: str | None,
    functional_study_uid: str | None,
    anatomic_candidates: Sequence[Mapping[str, Any]],
    *,
    affine_similarity: Any = None,
) -> tuple[Mapping[str, Any] | None, str, str | None]:
    """Pick the anatomic-MR whose total_mr masks B3 samples under (rev5 §4.2-step5).

    Priority: (a) shared FoR; (b) same study; tie-break by affine-similarity then acq-time
    then n_slices. Genuinely ambiguous (≥2 equally-good, no discriminator) → QC ambiguous.
    Returns (chosen_row|None, basis, qc_reason|None). ``affine_similarity`` optional callable
    (row)->float (higher=closer); when absent, falls back to n_slices tiebreak.
    """
    cands = list(anatomic_candidates)
    if not cands:
        return None, "none", QC_NO_ANATOMIC

    def _score(r: Mapping[str, Any]) -> tuple[float | None, int]:
        sim = affine_similarity(r) if affine_similarity else None
        return (sim, int(r.get("n_slices") or 0))

    def _tied(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        sa, sb = _score(a), _score(b)
        sim_tied = (sa[0] is None and sb[0] is None) or (
            sa[0] is not None and sb[0] is not None and abs(sa[0] - sb[0]) < 1e-3
        )
        return sim_tied and sa[1] == sb[1]

    def _pick_or_ambiguous(pool: list[Mapping[str, Any]], basis: str):
        if len(pool) == 1:
            return pool[0], basis, None
        ranked = sorted(
            pool,
            key=lambda r: (-((_score(r)[0]) or 0.0), -_score(r)[1], str(r.get("series_uid") or "")),
        )
        # rev5 §4.2-step5(c): if the top two remain tied within tolerance on every available
        # discriminator (affine similarity + n_slices), emit ambiguous rather than silently pick.
        if _tied(ranked[0], ranked[1]):
            return None, basis + "_ambiguous", QC_AMBIGUOUS_ANATOMIC
        return ranked[0], basis + "_ranked", None

    if functional_for_uid:
        for_match = [r for r in cands if r.get("frame_of_reference_uid") == functional_for_uid]
        if for_match:
            return _pick_or_ambiguous(for_match, "shared_for")

    if functional_study_uid:
        study_match = [r for r in cands if r.get("study_uid") == functional_study_uid]
        if study_match:
            return _pick_or_ambiguous(study_match, "same_study")

    return _pick_or_ambiguous(cands, "fallback")


# --- I/O: load functional volume + read total_mr masks ------------------------------

def load_functional_volume(dicom_dir: Path) -> tuple["sitk.Image | None", int, str | None]:
    """Load an mr_functional series via dcm2niix (shared ITK-LPS geometry; rescale once).

    Returns (image|None, n_volumes, qc_reason|None). >1 volume (4-D dynamic) →
    QC_MULTIVOLUME; non-scalar / load failure → QC_NON_SCALAR / QC_LOAD_FAILED.
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists() or not any(dicom_dir.iterdir()):
        return None, 0, QC_NOT_MATERIALIZED
    if sitk is None:  # pragma: no cover
        return None, 0, QC_LOAD_FAILED
    with tempfile.TemporaryDirectory(prefix="b3_dcm2niix_") as tmp:
        try:
            # Static executable and argument vector; no shell interpolation.
            subprocess.run(  # nosec B603 B607
                ["dcm2niix", "-z", "y", "-f", "func_%s", "-o", tmp, str(dicom_dir)],
                check=True, capture_output=True, timeout=600,
            )
        except Exception as exc:
            logger.warning("B3 dcm2niix failed for %s: %s", dicom_dir, exc)
            return None, 0, QC_LOAD_FAILED
        niftis = sorted(Path(tmp).glob("*.nii*"))
        if not niftis:
            return None, 0, QC_LOAD_FAILED
        # dcm2niix splits multi-volume/multi-echo into separate files → 4-D / non-single.
        if len(niftis) > 1:
            return None, len(niftis), QC_MULTIVOLUME
        try:
            img = sitk.ReadImage(str(niftis[0]))
        except Exception as exc:
            logger.warning("B3 sitk.ReadImage failed for %s: %s", niftis[0], exc)
            return None, 0, QC_LOAD_FAILED
        if img.GetDimension() == 4 or img.GetNumberOfComponentsPerPixel() > 1:
            return None, max(2, img.GetSize()[3] if img.GetDimension() == 4 else 2), QC_MULTIVOLUME
        if img.GetDimension() != 3:
            return None, 1, QC_NON_SCALAR
        return img, 1, None


def read_total_mr_label_image(
    mask_dir: Path, base_name: str | None = None
) -> tuple["sitk.Image | None", dict[int, str]]:
    """Minimal total_mr multilabel reader (all-series convention `total_mr--multilabel.nii.gz`
    + `total_mr--segmentations.json`). Scoped to total_mr only; E1 will consolidate readers."""
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return None, {}
    label_img = mask_dir / "total_mr--multilabel.nii.gz"
    seg_json = mask_dir / "total_mr--segmentations.json"
    if not label_img.exists():
        cands = sorted(mask_dir.glob("total_mr--multilabel.nii*"))
        if not cands:
            return None, {}
        label_img = cands[0]
    if sitk is None:  # pragma: no cover
        return None, {}
    try:
        img = sitk.ReadImage(str(label_img))
    except Exception as exc:
        logger.warning("B3 mask read failed %s: %s", label_img, exc)
        return None, {}
    label_map: dict[int, str] = {}
    if seg_json.exists():
        try:
            payload = json.loads(seg_json.read_text(encoding="utf-8"))
            # segmentations.json: {"labels": {"1": "name", ...}} or {"name": idx} variants.
            labels = payload.get("labels", payload) if isinstance(payload, dict) else {}
            for k, v in labels.items():
                try:
                    if isinstance(v, str):
                        label_map[int(k)] = v
                    elif isinstance(v, int):
                        label_map[int(v)] = str(k)
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass
    if not label_map:  # fall back to present non-zero labels
        arr = sitk.GetArrayViewFromImage(img)
        for lab in sorted(int(x) for x in np.unique(arr) if int(x) != 0):
            label_map[lab] = f"label_{lab}"
    return img, label_map


# --- units provenance ---------------------------------------------------------------

_SUBTYPE_UNIT_CONVENTION = {
    "adc": ("unknown", "none"),
    "dwi": ("arbitrary", "convention"),
    "perfusion": ("map-specific (convention)", "convention"),
    "subtraction": ("arbitrary", "convention"),
}


def _units_provenance(subtype: str, dicom_dir: Path | None) -> tuple[str, str, bool]:
    """Return (raw_unit, unit_source, rescale_applied).

    Prefer DICOM RealWorldValueMapping / RescaleType when present. Missing ADC unit metadata
    is not sufficient to infer scale from the series description, so ADC remains explicitly
    unknown and is excluded from calibrated interpretation. dcm2niix bakes a documented
    RescaleSlope/Intercept into the NIfTI, so rescale_applied is True only for documented units.
    """
    raw_unit, unit_source = _SUBTYPE_UNIT_CONVENTION.get(subtype, ("raw", "none"))
    if dicom_dir is not None:
        try:
            import pydicom  # local import; optional at unit-test time

            files = sorted(Path(dicom_dir).glob("*"))
            for fp in files[:1]:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                rwvm = getattr(ds, "RealWorldValueMappingSequence", None)
                if rwvm:
                    units_seq = getattr(rwvm[0], "MeasurementUnitsCodeSequence", None)
                    if units_seq:
                        cm = getattr(units_seq[0], "CodeMeaning", None)
                        if cm:
                            return str(cm), "rwvm", True
                rtype = getattr(ds, "RescaleType", None) or getattr(ds, "Units", None)
                if rtype:
                    return str(rtype), "rescale_type", True
        except Exception:
            pass
    return raw_unit, unit_source, unit_source != "none"


def _native_voxel_counts(
    mask_img: "sitk.Image", functional_native: "sitk.Image", transform: "sitk.Transform | None"
) -> dict[int, int]:
    """Count native-functional voxels under each label (mask NN-resampled to F's grid).

    Flags small structures under-resolved on the coarse functional grid (low_native_voxels)
    without interpolation bias. Uses the inverse of the F→anatomic transform (identity for
    direct/same-FoR resample).
    """
    if sitk is None:  # pragma: no cover
        return {}
    try:
        inv = transform.GetInverse() if transform is not None else sitk.Transform(3, sitk.sitkIdentity)
    except Exception:
        inv = sitk.Transform(3, sitk.sitkIdentity)
    resampled_mask = sitk.Resample(
        mask_img, functional_native, inv, sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID()
    )
    arr = sitk.GetArrayViewFromImage(resampled_mask)
    labels, counts = np.unique(arr, return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts) if int(l) != 0}


# --- per-patient orchestration ------------------------------------------------------

def _load_anatomic_intensity(anat_dir: Path, intensity_fn=None) -> "sitk.Image | None":
    """Load the anatomic-MR INTENSITY volume (the NIfTI total_mr was segmented from) — the
    correct registration fixed image (NOT the discrete mask; Mattes-MI on a label mask
    misregisters). Mask geometry == this intensity geometry, so resampling onto either grid
    is equivalent. Returns None if unavailable (rigid-MI then impossible)."""
    if intensity_fn is not None:
        return intensity_fn(anat_dir)
    if sitk is None:  # pragma: no cover
        return None
    from .segmentation import _series_artifact_dirs
    nifti_dir = _series_artifact_dirs(Path(anat_dir))[0]
    for p in (sorted(nifti_dir.glob("*.nii*")) if nifti_dir.exists() else []):
        try:
            return sitk.ReadImage(str(p))
        except Exception:
            continue
    return None


def _bbox_overlap_frac(f_img: "sitk.Image", a_img: "sitk.Image") -> float:
    """Physical bounding-box overlap (intersection / F-bbox) — a same-FoR geometry-similarity
    proxy for ranking multiple anatomic candidates (meaningful only when they share F's FoR)."""
    def _bounds(img):
        sz = img.GetSize()
        pts = [img.TransformIndexToPhysicalPoint((i, j, k))
               for i in (0, sz[0] - 1) for j in (0, sz[1] - 1) for k in (0, sz[2] - 1)]
        xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    fb, ab = _bounds(f_img), _bounds(a_img)
    inter, fvol = 1.0, 1.0
    for d in range(3):
        inter *= max(0.0, min(fb[d + 3], ab[d + 3]) - max(fb[d], ab[d]))
        fvol *= max(1e-6, fb[d + 3] - fb[d])
    return inter / fvol if fvol > 0 else 0.0


def _find_manifest(output_root: Path, patient_id: str) -> Path | None:
    root = Path(output_root)
    direct = root / patient_id / "all_series" / "metadata" / "series_manifest.json"
    if direct.exists():
        return direct
    cands = sorted(root.glob(f"{patient_id}/**/series_manifest.json"))
    return cands[0] if cands else None


def _excluded_row(patient_id: str, frow: Mapping[str, Any], qc: str, *, subtype: str = "",
                  anatomic_uid: str = "", basis: str = "", tier: str = "") -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "functional_series_uid": frow.get("series_uid"),
        "functional_series_description": frow.get("series_description"),
        "functional_subtype": subtype,
        "anatomic_series_uid": anatomic_uid,
        "anatomic_basis": basis,
        "registration_tier": tier,
        "reg_converged": None, "reg_coverage_frac": None,
        "structure_name": "", "n_voxels": None, "n_native_voxels": None,
        "low_native_voxels": None, "mean": None, "median": None, "p10": None, "p90": None,
        "raw_unit": None, "unit_source": None, "rescale_applied": None, "normalized_unit": None,
        "qc_flag": qc, "deformable_flag": None,
    }


def sample_patient_mr_functional(
    output_root: Path,
    patient_id: str,
    *,
    rtpipeline_version: str = "",
    anatomic_in_scope: bool = True,
    force: bool = False,
    _load_fn=None,
    _mask_fn=None,
    _intensity_fn=None,
) -> dict[str, Any]:
    """B3 per-patient orchestration. Writes one sidecar per functional series (one terminal
    series-level QC + >=1 row each). Returns a summary. Never raises for a single-series
    failure. ``anatomic_in_scope`` False => every functional series -> anatomic_out_of_scope.
    """
    load_fn = _load_fn or load_functional_volume
    mask_fn = _mask_fn or read_total_mr_label_image
    summary = {"patient_id": patient_id, "n_functional": 0, "n_sampled": 0, "sidecars": []}

    manifest_path = _find_manifest(output_root, patient_id)
    if manifest_path is None:
        return summary
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return summary
    # The canonical on-disk manifest stores the series list under "series" (inventory.py:326);
    # every other consumer reads that key. (rows fallback kept only for hand-built inputs.)
    rows = manifest.get("series", manifest.get("rows", [])) if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list):
        return summary
    func_rows = [r for r in rows if isinstance(r, dict) and r.get("image_class") == "mr_functional"]
    anat_rows = [r for r in rows if isinstance(r, dict)
                 and r.get("image_class") == "mr_anatomic" and r.get("output_dir")]
    summary["n_functional"] = len(func_rows)

    from .segmentation import _series_artifact_dirs  # local import to avoid import cycle

    def _abs_dir(row: Mapping[str, Any]) -> Path:
        d = Path(row.get("output_dir") or "")
        return d if d.is_absolute() else (Path(output_root) / d)

    for frow in func_rows:
        sidecar_dir = Path(frow.get("output_dir") or (output_root / patient_id))
        sidecar_dir = sidecar_dir if sidecar_dir.is_absolute() else (Path(output_root) / sidecar_dir)
        sidecar = sidecar_dir.parent / "mr_functional.json" if sidecar_dir.name == "DICOM" else sidecar_dir / "mr_functional.json"
        if sidecar.exists() and not force:
            summary["sidecars"].append(str(sidecar))
            continue

        def _emit(row_list: list[dict[str, Any]], series_qc: str, reg: dict | None = None):
            for r in row_list:
                r.setdefault("rtpipeline_version", rtpipeline_version)
            payload = {
                "patient_id": patient_id, "functional_series_uid": frow.get("series_uid"),
                "series_qc": series_qc, "registration": reg or {}, "rows": row_list,
            }
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp = sidecar.parent / f".mr_functional.{os.getpid()}.{uuid.uuid4().hex}.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(sidecar)
            summary["sidecars"].append(str(sidecar))

        if not anatomic_in_scope:
            _emit([_excluded_row(patient_id, frow, QC_ANATOMIC_OUT_OF_SCOPE)], QC_ANATOMIC_OUT_OF_SCOPE)
            continue

        route = route_functional_subtype(frow.get("series_description", ""), frow.get("image_types"))
        if not route.sampled:
            _emit([_excluded_row(patient_id, frow, route.qc_reason, subtype=route.subtype)], route.qc_reason)
            continue

        f_img, n_vol, load_qc = load_fn(sidecar_dir)
        if load_qc is not None:
            _emit([_excluded_row(patient_id, frow, load_qc, subtype=route.subtype)], load_qc)
            continue

        def _affine_sim(cand_row, _f=f_img):
            ci = _load_anatomic_intensity(_abs_dir(cand_row), _intensity_fn)
            if ci is None:
                return None
            try:
                return _bbox_overlap_frac(_f, ci)
            except Exception:
                return None

        chosen, basis, sel_qc = select_anatomic_for_functional(
            frow.get("frame_of_reference_uid"), frow.get("study_uid"), anat_rows,
            affine_similarity=_affine_sim,
        )
        if sel_qc is not None or chosen is None:
            _emit([_excluded_row(patient_id, frow, sel_qc or QC_NO_ANATOMIC, subtype=route.subtype, basis=basis)],
                  sel_qc or QC_NO_ANATOMIC)
            continue

        anat_dir = _abs_dir(chosen)
        anat_uid = str(chosen.get("series_uid") or "")
        seg_dir = _series_artifact_dirs(anat_dir)[1]
        mask_img, label_map = (None, {})
        for cand_dir in ([seg_dir, *sorted(seg_dir.glob("*"))] if seg_dir.exists() else []):
            mask_img, label_map = mask_fn(cand_dir)
            if mask_img is not None:
                break
        if mask_img is None:
            _emit([_excluded_row(patient_id, frow, QC_ANATOMIC_OUT_OF_SCOPE, subtype=route.subtype,
                                 anatomic_uid=anat_uid, basis=basis)], QC_ANATOMIC_OUT_OF_SCOPE)
            continue

        same_for = bool(frow.get("frame_of_reference_uid")) and \
            frow.get("frame_of_reference_uid") == chosen.get("frame_of_reference_uid")
        # A: register against the anatomic INTENSITY volume, NOT the discrete mask (Mattes-MI on
        # a label mask misregisters). The mask supplies only the sampling grid/labels. rigid-MI
        # needs intensity; without it a different-FoR series cannot be registered -> reg_failed.
        anat_int = _load_anatomic_intensity(anat_dir, _intensity_fn)
        if anat_int is None and not same_for:
            _emit([_excluded_row(patient_id, frow, QC_REG_FAILED, subtype=route.subtype,
                                 anatomic_uid=anat_uid, basis=basis, tier="rigid_mi")], QC_REG_FAILED)
            continue
        reg_ref = anat_int if anat_int is not None else mask_img
        res = resample_functional_to_anatomic(f_img, reg_ref, same_for=same_for)
        if res.qc_reason is not None or res.image is None:
            _emit([_excluded_row(patient_id, frow, res.qc_reason or QC_REG_FAILED, subtype=route.subtype,
                                 anatomic_uid=anat_uid, basis=basis, tier=res.tier)],
                  res.qc_reason or QC_REG_FAILED)
            continue

        # Mask must be on the resampled grid for sampling (mask geom == intensity geom normally).
        if not _geometry_matches(mask_img, res.image):
            mask_img = sitk.Resample(mask_img, res.image, sitk.Transform(3, sitk.sitkIdentity),
                                     sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID())
        mask_arr = sitk.GetArrayFromImage(mask_img)

        # C: enforce acceptance — UNION-mask coverage (all nonzero labels) + rigid convergence.
        res_arr = sitk.GetArrayFromImage(res.image)
        nonzero = mask_arr != 0
        n_nonzero = int(nonzero.sum())
        union_cov = (float(np.isfinite(res_arr[nonzero]).sum()) / n_nonzero) if n_nonzero else 0.0
        if (res.tier == "rigid_mi" and not res.reg_converged) or union_cov < MIN_COVERAGE_FRAC:
            ex = _excluded_row(patient_id, frow, QC_REG_FAILED, subtype=route.subtype,
                               anatomic_uid=anat_uid, basis=basis, tier=res.tier)
            ex["reg_converged"] = res.reg_converged
            ex["reg_coverage_frac"] = union_cov
            _emit([ex], QC_REG_FAILED, reg={"tier": res.tier, "converged": res.reg_converged,
                                            "coverage_frac": union_cov, "anatomic_series_uid": anat_uid})
            continue

        native = _native_voxel_counts(mask_img, f_img, res.transform)
        raw_unit, unit_source, rescale_applied = _units_provenance(route.subtype, sidecar_dir)
        stats_rows = per_structure_stats(
            res.image, mask_arr, label_map, stats_kind=route.stats_kind or "mean_median",
            tier=res.tier, native_count=native,
        )
        out_rows = [{
            "patient_id": patient_id,
            "functional_series_uid": frow.get("series_uid"),
            "functional_series_description": frow.get("series_description"),
            "functional_subtype": route.subtype,
            "anatomic_series_uid": chosen.get("series_uid"),
            "anatomic_basis": basis,
            "registration_tier": res.tier,
            "reg_converged": res.reg_converged,
            "reg_coverage_frac": union_cov,
            "raw_unit": raw_unit, "unit_source": unit_source,
            "rescale_applied": rescale_applied, "normalized_unit": None,
            **sr,
        } for sr in stats_rows]
        unit_qc = QC_UNKNOWN_UNITS if unit_source == "none" else QC_OK
        if unit_qc != QC_OK:
            for row in out_rows:
                row["qc_flag"] = unit_qc
        _emit(out_rows, unit_qc, reg={"tier": res.tier, "converged": res.reg_converged,
                                    "coverage_frac": union_cov, "anatomic_series_uid": chosen.get("series_uid"),
                                    "basis": basis})
        summary["n_sampled"] += 1
    return summary


# --- CSV aggregation ----------------------------------------------------------------

_CSV_FIELDS = [
    "patient_id", "functional_series_uid", "functional_series_description",
    "functional_subtype", "anatomic_series_uid", "anatomic_basis", "registration_tier",
    "reg_converged", "reg_coverage_frac", "structure_name", "n_voxels", "n_native_voxels",
    "low_native_voxels", "mean", "median", "p10", "p90", "raw_unit", "unit_source",
    "rescale_applied", "normalized_unit", "qc_flag", "deformable_flag", "rtpipeline_version",
]


def write_mr_functional_structures_csv(output_root: Path) -> Path | None:
    """Aggregate per-series mr_functional.json sidecars into Data/mr_functional_structures.csv.

    Atomic (unique tmp + replace), serial, post-join — mirrors B1's aggregator.
    """
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    n_unreadable = 0
    for json_path in sorted(output_root.rglob("mr_functional.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            # No silent drop: surface the loss loudly so a corrupt/truncated sidecar cannot
            # silently shrink the cohort CSV (violating series-UID set-equality).
            n_unreadable += 1
            logger.error("B3 CSV aggregation: UNREADABLE sidecar dropped from cohort CSV: %s (%s)",
                         json_path, exc)
            continue
        for row in payload.get("rows", []) if isinstance(payload, dict) else []:
            rows.append({k: row.get(k) for k in _CSV_FIELDS})
    if n_unreadable:
        logger.error("B3 CSV aggregation: %d sidecar(s) were unreadable and excluded — "
                     "cohort completeness is NOT guaranteed; investigate before use.", n_unreadable)
    if not rows:
        return None
    data_dir = output_root / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "mr_functional_structures.csv"
    rows.sort(key=lambda r: (str(r.get("patient_id") or ""), str(r.get("functional_series_uid") or ""), str(r.get("structure_name") or "")))
    tmp_path = data_dir / f".mr_functional_structures.{os.getpid()}.{uuid.uuid4().hex}.csv.tmp"
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path


__all__ = [
    "SubtypeRoute", "route_functional_subtype", "select_anatomic_for_functional",
    "resample_functional_to_anatomic", "coverage_fraction", "per_structure_stats",
    "load_functional_volume", "read_total_mr_label_image", "write_mr_functional_structures_csv",
    "sample_patient_mr_functional",
]
