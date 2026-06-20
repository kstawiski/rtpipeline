"""B2 — per-structure PET SUV under TotalSegmentator `total` masks.

Implements the DESIGN_v2 §1 B2 spec (per Auriac 2025, PMID 40993940): for each PET series
with a paired PET-CT CT-component (`petct_ct`, same study_uid+FoR) whose `total` masks exist,
compute per-structure SUVmax, SUVpeak (PERCIST 1 cm^3 sphere on the hottest in-structure
voxel, PET physical space), SUVmean, volume_ml. Output a per-series sidecar + cohort
Data/pet_suv_structures.csv. DEFAULT-DENY-style QC: one terminal QC + >=1 row per PET series.
MTV/TLG is OUT OF SCOPE (PI-gated, deferred).

Opt-in via ``config.pet_suv_structures`` (default False); runs after PET-SUV ingestion (SUVbw
NIfTI must exist) and all-series segmentation (petct_ct `total` masks must exist). A B2 failure
never aborts the run.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None  # type: ignore

# PERCIST SUVpeak: a 1 cm^3 spherical VOI; radius r = (3/(4*pi))^(1/3) cm.
SPHERE_VOLUME_CM3 = 1.0
SPHERE_RADIUS_MM = ((3.0 * SPHERE_VOLUME_CM3) / (4.0 * math.pi)) ** (1.0 / 3.0) * 10.0  # ~6.2038 mm
MIN_VOLUME_VOXELS = 10  # partial-volume guard for tiny ROIs (SUV-grid voxel count)
# B2 denominator: PET series whose post-ingestion status means a SUVbw NIfTI was produced
# (pet_suv.py assigns suv_computed:1445 / suv_skipped_idempotent:1337 with a NIfTI; suv_excluded:1146
# / suv_failed:1465 have none; pre-ingestion `materialized` also has none → out of B2 scope, never
# mislabeled suv_nifti_missing).
_SUV_OUTPUT_STATUSES = frozenset({"suv_computed", "suv_skipped_idempotent"})

QC_OK = "ok"
QC_NO_PETCT = "no_petct_ct"
QC_AMBIGUOUS_PETCT = "ambiguous_petct_ct"
QC_MASKS_MISSING = "petct_ct_masks_missing"
QC_SUV_MISSING = "suv_nifti_missing"
QC_EMPTY_MASK = "empty_mask"
QC_LOAD_FAILED = "load_failed"


def read_total_ct_label_image(
    mask_dir: Path, base_name: str | None = None
) -> tuple["sitk.Image | None", dict[int, str]]:
    """Minimal `total` multilabel reader (all-series convention `total--multilabel.nii.gz`
    + `total--segmentations.json`). Mirrors B3's `read_total_mr_label_image`, scoped to `total`."""
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return None, {}
    label_img = mask_dir / "total--multilabel.nii.gz"
    seg_json = mask_dir / "total--segmentations.json"
    if not label_img.exists():
        cands = sorted(mask_dir.glob("total--multilabel.nii*"))
        if not cands:
            return None, {}
        label_img = cands[0]
    if sitk is None:  # pragma: no cover
        return None, {}
    try:
        img = sitk.ReadImage(str(label_img))
    except Exception as exc:
        logger.warning("B2 mask read failed %s: %s", label_img, exc)
        return None, {}
    label_map: dict[int, str] = {}
    if seg_json.exists():
        try:
            payload = json.loads(seg_json.read_text(encoding="utf-8"))
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
    if not label_map:
        arr = sitk.GetArrayViewFromImage(img)
        for lab in sorted(int(x) for x in np.unique(arr) if int(x) != 0):
            label_map[lab] = f"label_{lab}"
    return img, label_map


def pair_petct_ct(
    pt_row: Mapping[str, Any], petct_rows: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, str, str | None]:
    """Pair a PET series to its PET-CT CT-component by (study_uid, FoR). Returns
    (chosen|None, basis, qc_reason|None). 0 → no_petct_ct; 1 → use; ≥2 distinguishable →
    pick larger n_slices; ≥2 tied → ambiguous_petct_ct (never silently pick — B3 lesson)."""
    study = str(pt_row.get("study_uid") or "").strip()
    for_uid = str(pt_row.get("frame_of_reference_uid") or "").strip()
    if not study or not for_uid:
        return None, "no_study_or_for", QC_NO_PETCT
    cands = [r for r in petct_rows
             if str(r.get("study_uid") or "").strip() == study
             and str(r.get("frame_of_reference_uid") or "").strip() == for_uid]
    if not cands:
        return None, "none", QC_NO_PETCT
    if len(cands) == 1:
        return cands[0], "study_for", None
    ranked = sorted(cands, key=lambda r: (-(int(r.get("n_slices") or 0)), str(r.get("series_uid") or "")))
    if int(ranked[0].get("n_slices") or 0) == int(ranked[1].get("n_slices") or 0):
        return None, "tied", QC_AMBIGUOUS_PETCT
    return ranked[0], "study_for_ranked", None


def resample_mask_to_suv_grid(mask_img: "sitk.Image", suv_img: "sitk.Image") -> "sitk.Image":
    """Resample the CT-grid label image onto the SUV grid (NN; identity transform — PET & the
    PET-CT CT-component share the Frame of Reference, so they are in the same physical space)."""
    return sitk.Resample(mask_img, suv_img, sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID())


def suvpeak_sphere(suv_arr: np.ndarray, suv_img: "sitk.Image", center_index_xyz: tuple[int, int, int],
                   radius_mm: float = SPHERE_RADIUS_MM) -> float:
    """PERCIST SUVpeak: mean SUV in a 1 cm^3 sphere centered on the hottest in-structure voxel,
    in PET physical space. The sphere is the VOI (NOT clipped to the structure); clamped to the
    image. ``suv_arr`` is the numpy (z,y,x) array; ``center_index_xyz`` is a sitk (x,y,z) index."""
    sx, sy, sz = suv_img.GetSpacing()
    size_x, size_y, size_z = suv_img.GetSize()
    cx, cy, cz = center_index_xyz
    center_phys = np.array(suv_img.TransformIndexToPhysicalPoint((int(cx), int(cy), int(cz))), dtype=float)
    # bounding box of the sphere in voxels (per-axis), clamped to image. Per-axis spacing is a
    # safe BB for axis-aligned grids (PET in RAS is axis-aligned); the membership test below uses
    # true physical distance, so the BB only needs to be a superset of the sphere.
    rx = int(math.ceil(radius_mm / sx)); ry = int(math.ceil(radius_mm / sy)); rz = int(math.ceil(radius_mm / sz))
    vals: list[float] = []
    for iz in range(max(0, cz - rz), min(size_z, cz + rz + 1)):
        for iy in range(max(0, cy - ry), min(size_y, cy + ry + 1)):
            for ix in range(max(0, cx - rx), min(size_x, cx + rx + 1)):
                p = np.array(suv_img.TransformIndexToPhysicalPoint((ix, iy, iz)), dtype=float)
                if np.linalg.norm(p - center_phys) <= radius_mm:
                    vals.append(float(suv_arr[iz, iy, ix]))  # numpy is (z,y,x)
    return float(np.mean(vals)) if vals else float(suv_arr[cz, cy, cx])


def per_structure_suv(
    suv_img: "sitk.Image",
    suv_mask_arr: np.ndarray,
    ct_mask_arr: np.ndarray,
    label_map: Mapping[int, str],
    ct_voxel_vol_ml: float,
) -> list[dict[str, Any]]:
    """Per-structure SUVmax/SUVpeak/SUVmean (SUV grid) + volume_ml (native CT grid)."""
    suv_arr = sitk.GetArrayFromImage(suv_img)  # (z,y,x)
    rows: list[dict[str, Any]] = []
    for label, name in sorted(label_map.items()):
        n_native = int((ct_mask_arr == label).sum())
        volume_ml = n_native * ct_voxel_vol_ml
        sel = suv_mask_arr == label
        n_suv = int(sel.sum())
        row: dict[str, Any] = {
            "structure_name": name, "n_suv_voxels": n_suv, "volume_ml": volume_ml,
            "suvmax": None, "suvpeak": None, "suvmean": None,
            "min_volume_flag": (0 < n_suv < MIN_VOLUME_VOXELS), "qc_flag": QC_OK,
        }
        if n_suv == 0:
            row["qc_flag"] = QC_EMPTY_MASK
            rows.append(row)
            continue
        vals = suv_arr[sel].astype(np.float64)
        row["suvmax"] = float(np.max(vals))
        row["suvmean"] = float(np.mean(vals))
        # hottest in-structure voxel (numpy z,y,x) -> sitk (x,y,z) index for the sphere center
        zyx = np.argwhere(sel)
        hottest = zyx[int(np.argmax(suv_arr[sel]))]
        center_xyz = (int(hottest[2]), int(hottest[1]), int(hottest[0]))
        row["suvpeak"] = suvpeak_sphere(suv_arr, suv_img, center_xyz)
        rows.append(row)
    return rows


# --- orchestration ------------------------------------------------------------------

def _find_manifest(output_root: Path, patient_id: str) -> Path | None:
    root = Path(output_root)
    direct = root / patient_id / "all_series" / "metadata" / "series_manifest.json"
    if direct.exists():
        return direct
    cands = sorted(root.glob(f"{patient_id}/**/series_manifest.json"))
    return cands[0] if cands else None


_CSV_FIELDS = [
    "patient_id", "pet_series_uid", "pet_series_description", "petct_ct_series_uid",
    "pairing_basis", "structure_name", "suvmax", "suvpeak", "suvmean", "volume_ml",
    "n_suv_voxels", "min_volume_flag", "qc_flag", "rtpipeline_version",
]


def _excluded_row(patient_id: str, ptrow: Mapping[str, Any], qc: str, *,
                  petct_uid: str = "", basis: str = "") -> dict[str, Any]:
    return {
        "patient_id": patient_id, "pet_series_uid": ptrow.get("series_uid"),
        "pet_series_description": ptrow.get("series_description"),
        "petct_ct_series_uid": petct_uid, "pairing_basis": basis, "structure_name": "",
        "suvmax": None, "suvpeak": None, "suvmean": None, "volume_ml": None,
        "n_suv_voxels": None, "min_volume_flag": None, "qc_flag": qc,
    }


def sample_patient_pet_suv(
    output_root: Path, patient_id: str, *, rtpipeline_version: str = "", force: bool = False,
    _suv_fn=None, _mask_fn=None,
) -> dict[str, Any]:
    """B2 per-patient: per PET series → pair petct_ct → sample SUV under `total` masks → sidecar
    (one terminal QC + ≥1 row each). Never raises for a single-series failure."""
    mask_fn = _mask_fn or read_total_ct_label_image
    summary = {"patient_id": patient_id, "n_pet": 0, "n_sampled": 0, "sidecars": []}
    manifest_path = _find_manifest(output_root, patient_id)
    if manifest_path is None:
        logger.warning("B2: no series_manifest.json for patient %s — skipped (no B2 output).", patient_id)
        return summary
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("B2: unreadable manifest for patient %s (%s) — skipped.", patient_id, exc)
        return summary
    rows = manifest.get("series", manifest.get("rows", [])) if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list):
        logger.warning("B2: malformed manifest (no series list) for patient %s — skipped.", patient_id)
        return summary
    all_series_root = manifest_path.parent.parent  # <patient>/all_series
    # B2 scope = PET series for which SUV was successfully computed (a SUVbw NIfTI exists).
    # Upstream-excluded/failed/pending PT rows have no NIfTI and are NOT B2's denominator —
    # including them would mislabel them suv_nifti_missing (Codex/Claude impl-gate r1).
    pet_rows = [r for r in rows if isinstance(r, dict) and r.get("image_class") == "pt"
                and str(r.get("status") or "") in _SUV_OUTPUT_STATUSES]
    petct_rows = [r for r in rows if isinstance(r, dict) and r.get("image_class") == "petct_ct" and r.get("output_dir")]
    summary["n_pet"] = len(pet_rows)

    from .segmentation import _series_artifact_dirs  # local import to avoid cycle
    from .pet_suv import _safe_token  # canonical SUV NIfTI naming (single source of truth)

    for ptrow in pet_rows:
        safe = _safe_token(ptrow.get("series_uid") or "", "pt_series")
        suv_dir = all_series_root / "NIFTI" / "SUV" / safe
        suv_path = suv_dir / f"{safe}_SUVbw.nii.gz"
        sidecar = suv_dir / "pet_suv_structures.json"
        if sidecar.exists() and not force:
            summary["sidecars"].append(str(sidecar)); continue

        def _emit(row_list: list[dict[str, Any]], series_qc: str, meta: dict | None = None):
            for r in row_list:
                r.setdefault("rtpipeline_version", rtpipeline_version)
            payload = {"patient_id": patient_id, "pet_series_uid": ptrow.get("series_uid"),
                       "series_qc": series_qc, "pairing": meta or {}, "rows": row_list}
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp = sidecar.parent / f".pet_suv_structures.{os.getpid()}.{uuid.uuid4().hex}.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(sidecar)
            summary["sidecars"].append(str(sidecar))

        suv_img = None
        if _suv_fn is not None:
            suv_img = _suv_fn(suv_path)
        elif suv_path.exists() and sitk is not None:
            try:
                suv_img = sitk.ReadImage(str(suv_path))
            except Exception as exc:
                logger.warning("B2 SUV read failed %s: %s", suv_path, exc)
        if suv_img is None:
            _emit([_excluded_row(patient_id, ptrow, QC_SUV_MISSING)], QC_SUV_MISSING); continue

        chosen, basis, qc = pair_petct_ct(ptrow, petct_rows)
        if qc is not None or chosen is None:
            _emit([_excluded_row(patient_id, ptrow, qc or QC_NO_PETCT, basis=basis)], qc or QC_NO_PETCT); continue
        petct_uid = str(chosen.get("series_uid") or "")

        anat_dir = Path(chosen.get("output_dir") or "")
        anat_dir = anat_dir if anat_dir.is_absolute() else (Path(output_root) / anat_dir)
        seg_dir = _series_artifact_dirs(anat_dir)[1]
        ct_mask_img, label_map = (None, {})
        for cand in ([seg_dir, *sorted(seg_dir.glob("*"))] if seg_dir.exists() else []):
            ct_mask_img, label_map = mask_fn(cand)
            if ct_mask_img is not None:
                break
        if ct_mask_img is None:
            _emit([_excluded_row(patient_id, ptrow, QC_MASKS_MISSING, petct_uid=petct_uid, basis=basis)],
                  QC_MASKS_MISSING); continue

        try:
            suv_mask_img = resample_mask_to_suv_grid(ct_mask_img, suv_img)
            ct_vox_ml = float(np.prod(ct_mask_img.GetSpacing())) / 1000.0
            stats = per_structure_suv(suv_img, sitk.GetArrayFromImage(suv_mask_img),
                                      sitk.GetArrayFromImage(ct_mask_img), label_map, ct_vox_ml)
        except Exception as exc:
            logger.warning("B2 sampling failed for %s: %s", ptrow.get("series_uid"), exc)
            _emit([_excluded_row(patient_id, ptrow, QC_LOAD_FAILED, petct_uid=petct_uid, basis=basis)],
                  QC_LOAD_FAILED); continue

        out_rows = [{
            "patient_id": patient_id, "pet_series_uid": ptrow.get("series_uid"),
            "pet_series_description": ptrow.get("series_description"),
            "petct_ct_series_uid": petct_uid, "pairing_basis": basis, **s,
        } for s in stats]
        _emit(out_rows, QC_OK, meta={"petct_ct_series_uid": petct_uid, "basis": basis})
        summary["n_sampled"] += 1
    return summary


def write_pet_suv_structures_csv(output_root: Path) -> Path | None:
    """Aggregate per-series pet_suv_structures.json sidecars into Data/pet_suv_structures.csv.
    Atomic; surfaces unreadable sidecars loudly (no silent drop — B3 lesson)."""
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    n_unreadable = 0
    for jp in sorted(output_root.rglob("pet_suv_structures.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:
            n_unreadable += 1
            logger.error("B2 CSV: UNREADABLE sidecar dropped: %s (%s)", jp, exc)
            continue
        for row in payload.get("rows", []) if isinstance(payload, dict) else []:
            rows.append({k: row.get(k) for k in _CSV_FIELDS})
    if n_unreadable:
        logger.error("B2 CSV: %d unreadable sidecar(s) excluded — completeness NOT guaranteed.", n_unreadable)
    if not rows:
        return None
    data_dir = output_root / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "pet_suv_structures.csv"
    rows.sort(key=lambda r: (str(r.get("patient_id") or ""), str(r.get("pet_series_uid") or ""), str(r.get("structure_name") or "")))
    tmp = data_dir / f".pet_suv_structures.{os.getpid()}.{uuid.uuid4().hex}.csv.tmp"
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader(); w.writerows(rows)
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


__all__ = [
    "read_total_ct_label_image", "pair_petct_ct", "resample_mask_to_suv_grid", "suvpeak_sphere",
    "per_structure_suv", "sample_patient_pet_suv", "write_pet_suv_structures_csv",
    "SPHERE_RADIUS_MM", "MIN_VOLUME_VOXELS",
]
