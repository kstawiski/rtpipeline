"""Frozen CT-grid measurement and C1–C4 classification primitives.

The public interface is :mod:`rtpipeline.treatment_scope`. No cohort roots,
network access, output writes, or process-wide thread settings are used here.
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon

from .treatment_scope_rules import load_rules


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


VERSION = "rtpipeline-treatment-scope-evidence-v1"
_SELECTION = load_rules()["target_selection"]


def _pipeline_helper(name, selection=None):
    selection = _SELECTION if selection is None else selection
    return any(
        re.search(selection[key], name, re.I)
        for key in ("leading_margin_helper_regex", "leading_z_helper_regex")
    )


def _is_target_name(name, selection=None):
    selection = _SELECTION if selection is None else selection
    text = str(name or "").strip()
    if not text or _pipeline_helper(text, selection):
        return False
    return any(
        not re.search(selection["boolean_crop_separator_regex"], text[: match.start()])
        for match in re.finditer(selection["name_token_regex"], text, re.I)
    )


def norm(x):
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(x).lower().replace("ł", "l"))
        if not unicodedata.combining(c)
    )


def geometry(p):
    r = sitk.ImageFileReader()
    r.SetFileName(str(p))
    r.ReadImageInformation()
    return {
        "size": list(r.GetSize()),
        "spacing": list(r.GetSpacing()),
        "origin": list(r.GetOrigin()),
        "direction": list(r.GetDirection()),
    }


def affine(g):
    return np.asarray(g["direction"]).reshape(3, 3) @ np.diag(g["spacing"])


def physical(ijk, g):
    return np.asarray(ijk) @ affine(g).T + np.asarray(g["origin"])


def read_dcm(p, tags):
    return pydicom.dcmread(p, stop_before_pixels=True, specific_tags=tags)


def fingerprint(p, hash_bytes=True):
    s = Path(p).stat()
    x = {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if hash_bytes:
        x["sha256"] = sha(p)
    return x


def select_targets(ds, rule):
    selection = rule.get("target_selection_settings", _SELECTION)
    obs = collections.defaultdict(set)
    for x in getattr(ds, "RTROIObservationsSequence", []):
        t = str(getattr(x, "RTROIInterpretedType", "")).strip().upper()
        if t:
            obs[int(x.ReferencedROINumber)].add(t)
    contours = {
        int(x.ReferencedROINumber): list(getattr(x, "ContourSequence", []))
        for x in getattr(ds, "ROIContourSequence", [])
    }
    selected = []
    exclusions = []
    seen = set()
    for roi in ds.StructureSetROISequence:
        n = int(roi.ROINumber)
        if n in seen:
            raise ValueError("duplicate_ROI_number")
        seen.add(n)
        name = str(roi.ROIName)
        types = obs[n]
        route = "interpreted_type" if types else "pipeline_name_rule"
        treated_ctv = (
            types == {"TREATED_VOLUME"}
            and bool(rule.get("treated_volume_ctv_name_regex"))
            and re.fullmatch(rule["treated_volume_ctv_name_regex"], name, re.I)
            is not None
        )
        qualifies = treated_ctv or (
            bool(types & set(selection["target_types"]))
            if types
            else _is_target_name(name, selection)
        )
        if not qualifies:
            continue
        reason = None
        if types - set(selection["target_types"]) and not treated_ctv:
            reason = "contradictory_interpreted_types"
        elif _pipeline_helper(name, selection):
            reason = "pipeline_helper_exclusion"
        elif any(re.search(p, norm(name)) for p in rule["helper_regex"]):
            reason = "frozen_helper_exclusion"
        elif not any(
            str(getattr(x, "ContourGeometricType", ""))
            in selection["closed_contour_types"]
            and int(getattr(x, "NumberOfContourPoints", 0))
            >= selection["minimum_contour_points"]
            for x in contours.get(n, [])
        ):
            reason = "no_valid_closed_contour"
        rec = {
            "roi_number": n,
            "type": "CTV"
            if treated_ctv
            else next(
                (t for t in selection["type_precedence"] if t in types),
                next(
                    (
                        t
                        for t in selection["type_precedence"]
                        if re.search(t, name, re.I)
                    ),
                    "TARGET",
                ),
            ),
            "route": "exact_name_treated_volume_ctv" if treated_ctv else route,
            "interpreted_types": sorted(types),
        }
        # No source free text is persisted. Names are used only for target/helper selection.
        if reason:
            exclusions.append(dict(rec, reason=reason))
            continue
        rec["contours"] = contours.get(n, [])
        selected.append(rec)
    return selected, exclusions


def rasterize_axial(rec, g, settings=None):
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    selection = settings.get("target_selection", _SELECTION)
    inv = np.linalg.inv(affine(g))
    size = np.asarray(g["size"])
    slices = {}
    points = []
    outside = 0
    bad = 0
    missingrefs = set()
    for c in rec.pop("contours"):
        typ = str(getattr(c, "ContourGeometricType", ""))
        if typ not in selection["closed_contour_types"]:
            bad += 1
            continue
        xyz = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
        if len(xyz) < selection["minimum_contour_points"] or not np.isfinite(xyz).all():
            bad += 1
            continue
        ijk = (xyz - np.asarray(g["origin"])) @ inv.T
        points.append(xyz)
        outside += int(
            np.count_nonzero(
                np.any(
                    (
                        ijk
                        < -settings["contour_bound_half_voxel"]
                        - settings["contour_bound_extra_voxel"]
                    )
                    | (
                        ijk
                        > size
                        - settings["contour_bound_half_voxel"]
                        + settings["contour_bound_extra_voxel"]
                    ),
                    axis=1,
                )
            )
        )
        k = int(round(float(ijk[:, 2].mean())))
        if (
            np.ptp(ijk[:, 2]) > settings["contour_plane_spread_voxel"]
            or abs(ijk[:, 2].mean() - k) > settings["contour_plane_offset_voxel"]
        ):
            raise ValueError("non_coplanar_or_off_grid_target")
        if k < 0 or k >= size[2]:
            continue
        rr, cc = polygon(ijk[:, 1], ijk[:, 0], shape=(int(size[1]), int(size[0])))
        # Even-odd fill handles nested holes, disjoint loops and DICOM XOR contours.
        a = slices.setdefault(k, np.zeros((size[1], size[0]), dtype=bool))
        a[rr, cc] ^= True
    if not points:
        raise ValueError("target_has_no_valid_closed_contour")
    xyz = np.concatenate(points)
    ids = []
    for k, a in sorted(slices.items()):
        ids.append(np.flatnonzero(a) + k * size[0] * size[1])
    ids = np.concatenate(ids) if ids else np.array([], dtype=np.int64)
    rec.update(
        contour_bbox_lps_mm=[xyz.min(0).tolist(), xyz.max(0).tolist()],
        contour_points=len(xyz),
        outside_ct_contour_points=outside,
        unsupported_contours=bad,
    )
    if not len(ids):
        raise ValueError("target_rasterization_empty")
    k, j, i = np.unravel_index(ids, tuple(size[::-1]))
    coords = physical(np.array([i, j, k]).T, g)
    rec.update(
        n_voxels=len(ids),
        volume_ml=float(len(ids) * np.prod(g["spacing"]) / 1000),
        centroid_lps_mm=coords.mean(0).tolist(),
        bbox_lps_mm=[coords.min(0).tolist(), coords.max(0).tolist()],
        z_histogram_mm_counts=[
            [float(z), int(n)]
            for z, n in zip(*np.unique(coords[:, 2], return_counts=True))
        ],
        organ_overlaps={},
        organ_centroid_distances_mm={},
    )
    return rec, ids


def is_axial_grid(g, settings=None):
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    return np.allclose(
        np.abs(np.asarray(g["direction"]).reshape(3, 3)),
        np.eye(3),
        atol=settings["axial_grid_atol"],
        rtol=settings["allclose_default_rtol"],
    )


def slice_plane(d, g, settings=None):
    """Bind one actual DICOM plane to a signed permutation of CT voxel axes.

    Origin tolerance (0.01 voxel), spacing tolerance (0.001 mm), and IOP
    tolerance (1e-5) retain the frozen checks. No resampling is admitted.
    """
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    ipp = np.asarray(d.ImagePositionPatient, float)
    iop = np.asarray(d.ImageOrientationPatient, float).reshape(2, 3)
    ps = np.asarray(d.PixelSpacing, float)
    a = affine(g)
    inv = np.linalg.inv(a)
    if not (
        np.isfinite(ipp).all() and np.isfinite(iop).all() and np.isfinite(ps).all()
    ) or np.any(ps <= 0):
        raise ValueError("DICOM_CT_invalid_plane")
    if not np.allclose(
        iop @ iop.T, np.eye(2), atol=settings["orientation_atol"], rtol=0
    ):
        raise ValueError("DICOM_CT_invalid_orientation")
    q = (ipp - np.asarray(g["origin"])) @ inv.T
    if not np.allclose(
        q,
        np.round(q),
        atol=settings["origin_voxel_atol"],
        rtol=settings["allclose_default_rtol"],
    ):
        raise ValueError("DICOM_NIFTI_grid_origin_mismatch")
    basis = (iop * ps[::-1, None]) @ inv.T
    steps = np.rint(basis).astype(int)
    if not np.all(np.sum(np.abs(steps), axis=1) == 1) or np.any(
        np.sum(np.abs(steps), axis=0) > 1
    ):
        raise ValueError("DICOM_NIFTI_plane_axis_mismatch")
    axes = np.argmax(np.abs(steps), axis=1)
    axis = int(next(x for x in range(3) if x not in axes))
    sizes = np.array([int(d.Columns), int(d.Rows)])
    size = np.asarray(g["size"])
    if not np.array_equal(sizes, size[axes]) or not np.allclose(
        ps[::-1],
        np.asarray(g["spacing"])[axes],
        atol=settings["spacing_mm_atol"],
        rtol=settings["allclose_default_rtol"],
    ):
        raise ValueError("DICOM_NIFTI_grid_spacing_mismatch")
    direction = np.asarray(g["direction"]).reshape(3, 3)
    expected = np.array([direction[:, ax] * steps[n, ax] for n, ax in enumerate(axes)])
    if not np.allclose(iop, expected, atol=settings["orientation_atol"], rtol=0):
        raise ValueError("DICOM_NIFTI_plane_orientation_mismatch")
    origin = np.rint(q).astype(int)
    for n, ax in enumerate(axes):
        if origin[ax] != (0 if steps[n, ax] > 0 else size[ax] - 1):
            raise ValueError("DICOM_NIFTI_plane_extent_mismatch")
    # Check all four corners: rounding alone must not conceal accumulated drift.
    for i, j in [
        (0, 0),
        (sizes[0] - 1, 0),
        (0, sizes[1] - 1),
        (sizes[0] - 1, sizes[1] - 1),
    ]:
        actual = q + i * basis[0] + j * basis[1]
        wanted = origin + i * steps[0] + j * steps[1]
        if not np.allclose(actual, wanted, atol=settings["origin_voxel_atol"], rtol=0):
            raise ValueError("DICOM_NIFTI_plane_corner_mismatch")
    normal = np.cross(iop[0], iop[1])
    normal /= np.linalg.norm(normal)
    spacing = abs(float(a[:, axis] @ normal))
    if spacing <= 0:
        raise ValueError("DICOM_CT_invalid_plane_spacing")
    return {
        "ipp": ipp,
        "iop": iop,
        "ps": ps,
        "normal": normal,
        "spacing": spacing,
        "origin": origin,
        "steps": steps,
        "axis": axis,
        "index": int(origin[axis]),
        "rows": int(sizes[1]),
        "columns": int(sizes[0]),
    }


def validate_slice_planes(planes, g):
    if not planes or len({p["axis"] for p in planes}) != 1:
        raise ValueError("DICOM_CT_inconsistent_slice_axes")
    axis = planes[0]["axis"]
    if sorted(p["index"] for p in planes) != list(range(g["size"][axis])):
        raise ValueError("DICOM_CT_incomplete_slice_grid")
    return sorted(planes, key=lambda p: p["index"])


def rasterize_planes(rec, g, planes, settings=None):
    """Even-odd fill at voxel centers in each acquired DICOM slice plane.

    Plane spread <=0.1 slice and mean distance <=0.35 slice are exactly the
    frozen contour tolerances. Filled integer pixels map to CT storage without
    interpolation, supersampling, or changing geometric thresholds.
    """
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    planes = validate_slice_planes(planes, g)
    selection = settings.get("target_selection", _SELECTION)
    inv = np.linalg.inv(affine(g))
    size = np.asarray(g["size"])
    slices = {}
    points = []
    outside = 0
    bad = 0
    for c in rec.pop("contours"):
        typ = str(getattr(c, "ContourGeometricType", ""))
        if typ not in selection["closed_contour_types"]:
            bad += 1
            continue
        xyz = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
        if len(xyz) < selection["minimum_contour_points"] or not np.isfinite(xyz).all():
            bad += 1
            continue
        ijk = (xyz - np.asarray(g["origin"])) @ inv.T
        points.append(xyz)
        outside += int(
            np.count_nonzero(
                np.any(
                    (
                        ijk
                        < -settings["contour_bound_half_voxel"]
                        - settings["contour_bound_extra_voxel"]
                    )
                    | (
                        ijk
                        > size
                        - settings["contour_bound_half_voxel"]
                        + settings["contour_bound_extra_voxel"]
                    ),
                    axis=1,
                )
            )
        )
        # Every actual slice origin/orientation participates in plane matching.
        candidates = []
        for p in planes:
            distance = (xyz - p["ipp"]) @ p["normal"] / p["spacing"]
            if (
                np.ptp(distance) <= settings["contour_plane_spread_voxel"]
                and abs(float(distance.mean()))
                <= settings["contour_plane_offset_voxel"]
            ):
                candidates.append(p)
        if len(candidates) != 1:
            raise ValueError("non_coplanar_or_off_grid_target")
        p = candidates[0]
        # Orthogonal projection removes only the plane-normal component.
        local = (xyz - p["ipp"]) @ p["iop"].T / p["ps"][::-1]
        # Floating arithmetic in a rigid rotation can move an exactly integral
        # vertex by a few ulps and exclude boundary voxel centers. Normalize only
        # that operation-scale roundoff, not a geometric distance tolerance.
        scale = (
            np.abs(xyz) @ np.abs(p["iop"]).T + np.abs(p["ipp"]) @ np.abs(p["iop"]).T
        ) / p["ps"][::-1]
        roundoff = (
            settings["roundoff_epsilon_multiplier"]
            * np.finfo(float).eps
            * np.maximum(1.0, scale)
        )
        nearest = np.rint(local)
        local = np.where(np.abs(local - nearest) <= roundoff, nearest, local)
        rr, cc = polygon(local[:, 1], local[:, 0], shape=(p["rows"], p["columns"]))
        key = p["index"]
        entry = slices.setdefault(
            key, (p, np.zeros((p["rows"], p["columns"]), dtype=bool))
        )
        entry[1][rr, cc] ^= True
    if not points:
        raise ValueError("target_has_no_valid_closed_contour")
    xyz = np.concatenate(points)
    ids = []
    for p, mask in slices.values():
        rr, cc = np.nonzero(mask)
        ijk = p["origin"] + cc[:, None] * p["steps"][0] + rr[:, None] * p["steps"][1]
        ids.append(np.ravel_multi_index(ijk[:, ::-1].T, tuple(size[::-1])))
    ids = np.sort(np.concatenate(ids)) if ids else np.array([], dtype=np.int64)
    rec.update(
        contour_bbox_lps_mm=[xyz.min(0).tolist(), xyz.max(0).tolist()],
        contour_points=len(xyz),
        outside_ct_contour_points=outside,
        unsupported_contours=bad,
    )
    if not len(ids):
        raise ValueError("target_rasterization_empty")
    k, j, i = np.unravel_index(ids, tuple(size[::-1]))
    coords = physical(np.array([i, j, k]).T, g)
    rec.update(
        n_voxels=len(ids),
        volume_ml=float(len(ids) * np.prod(g["spacing"]) / 1000),
        centroid_lps_mm=coords.mean(0).tolist(),
        bbox_lps_mm=[coords.min(0).tolist(), coords.max(0).tolist()],
        z_histogram_mm_counts=[
            [float(z), int(n)]
            for z, n in zip(*np.unique(coords[:, 2], return_counts=True))
        ],
        organ_overlaps={},
        organ_centroid_distances_mm={},
    )
    return rec, ids


def rasterize(rec, g, planes=None, settings=None):
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    if planes is None:
        if not is_axial_grid(g, settings=settings):
            raise ValueError("nonaxial_rasterization_requires_slice_planes")
        return rasterize_axial(rec, g, settings=settings)
    return rasterize_planes(rec, g, planes, settings=settings)


def organ_stats(a, g, exact_bounds=False, settings=None):
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    occupied = np.flatnonzero(np.any(a, axis=(1, 2)))
    if not len(occupied):
        return None
    first, last = int(occupied[0]), int(occupied[-1])
    k, j, i = np.nonzero(a[first : last + 1])
    k = k + first
    n = len(k)
    if not n:
        return None
    mean = np.array([i.mean(), j.mean(), k.mean()])
    low = np.array([i.min(), j.min(), k.min()])
    high = np.array([i.max(), j.max(), k.max()])
    corners = np.array(
        [
            [x, y, z]
            for x in [low[0], high[0]]
            for y in [low[1], high[1]]
            for z in [low[2], high[2]]
        ]
    )
    pts = physical(corners, g)
    if exact_bounds or not is_axial_grid(g, settings=settings):
        pts = physical(np.array([i, j, k]).T, g)
    return {
        "n_voxels": n,
        "volume_ml": float(n * np.prod(g["spacing"]) / 1000),
        "centroid_lps_mm": physical(mean, g).tolist(),
        "bbox_lps_mm": [pts.min(0).tolist(), pts.max(0).tolist()],
    }


def measure(course, rule):
    settings = dict(
        rule["measurement"],
        target_selection=rule.get("target_selection_settings", _SELECTION),
    )
    start = time.monotonic()
    root = Path(course)
    key = (root.parent.name, root.name)
    p, c = key
    ev = {
        "schema": VERSION,
        "patient_id": p,
        "course_id": c,
        "status": "in_progress",
        "targets": [],
        "organs": {},
        "errors": [],
        "input_fingerprints": [],
        "started_utc": now(),
    }
    used = []

    def use(f):
        x = fingerprint(f)
        used.append(x)
        return x

    try:
        mp = root / "metadata/case_metadata.json"
        use(mp)
        m = json.loads(mp.read_text())
        co = m["course_contract"]
        ct = co["planning_ct"]
        ev["structure_identity_sha256"] = hashlib.sha256(
            str(co["authoritative_rtstruct"].get("sop_instance_uid", "")).encode()
        ).hexdigest()
        ev["pipeline_dose_classification"] = {
            k: co.get("dose_classification", {}).get(k)
            for k in ["classification", "should_sum"]
        }
        if not ct.get("nifti_path"):
            raise ValueError("no_planning_CT_NIfTI_in_contract")
        if (str(m["patient_id"]), str(m["course_id"])) != key:
            raise ValueError("course_identity_mismatch")
        ev["prescription"] = {
            k: m.get(k)
            for k in [
                "resolved_prescribed_dose_total_gy",
                "planned_fraction_count",
                "prescribed_dose_scope",
            ]
        }
        uid = co["authoritative_rtstruct"]["sop_instance_uid"]
        matches = []
        for f in sorted((root / "DICOM/RTSTRUCT").glob("*")):
            if f.is_file():
                d = read_dcm(f, ["SOPInstanceUID"])
                if str(d.SOPInstanceUID) == uid:
                    matches.append(f)
        if len(matches) != 1:
            raise ValueError("authoritative_original_RTSTRUCT_not_unique")
        f = matches[0]
        use(f)
        ds = read_dcm(
            f,
            [
                "SOPInstanceUID",
                "StructureSetROISequence",
                "RTROIObservationsSequence",
                "ROIContourSequence",
                "ReferencedFrameOfReferenceSequence",
            ],
        )
        ev["rtstruct_sha256"] = used[-1]["sha256"]
        ev["selected_plan_evidence"] = []
        ev["plan_selection_basis"] = "selected_plan_membership"
        ev["linked_plan_prescriptions"] = []
        linked_uids = set()
        linked_originals = []
        for pf in sorted((root / "DICOM/RTPLAN").glob("*")):
            if not pf.is_file():
                continue
            pd = read_dcm(pf, ["SOPInstanceUID", "ReferencedStructureSetSequence"])
            if uid in {
                str(x.ReferencedSOPInstanceUID)
                for x in getattr(pd, "ReferencedStructureSetSequence", [])
            }:
                linked_uids.add(str(pd.SOPInstanceUID))
                linked_originals.append(
                    {
                        "path": str(pf.relative_to(root)),
                        "sop_instance_uid": str(pd.SOPInstanceUID),
                    }
                )
        for pl in m.get("delivery_plan_details", []):
            if str(pl.get("plan_sop_uid", "")) in linked_uids:
                ev["linked_plan_prescriptions"].append(
                    {
                        k: pl.get(k)
                        for k in [
                            "resolved_prescribed_dose_total_gy",
                            "planned_fraction_count",
                            "prescription_resolution_status",
                        ]
                    }
                )
        ev["n_original_plans_referencing_structure"] = len(linked_uids)
        plans = co["selected_plans"]
        if (
            not plans
            and rule.get("plan_selection", {}).get("empty_selected_plans_fallback")
            == "linked_original_rtplan_geometry_only"
        ):
            plans = linked_originals
            ev["plan_selection_basis"] = "linked_original_rtplan_geometry_only"
        if not plans:
            ev["plan_selection_basis"] = "no_selected_plan"
            raise ValueError("no_selected_plan")
        for plan in plans:
            pf = root / plan["path"]
            use(pf)
            pd = read_dcm(
                pf,
                [
                    "SOPInstanceUID",
                    "ReferencedStructureSetSequence",
                    "DoseReferenceSequence",
                    "FractionGroupSequence",
                    "ApprovalStatus",
                ],
            )
            if str(pd.SOPInstanceUID) != plan["sop_instance_uid"] or uid not in {
                str(x.ReferencedSOPInstanceUID)
                for x in pd.ReferencedStructureSetSequence
            }:
                raise ValueError("plan_structure_identity_mismatch")
            ev["selected_plan_evidence"].append(
                {
                    "sha256": used[-1]["sha256"],
                    "approval_status": str(
                        getattr(pd, "ApprovalStatus", "") or ""
                    ).strip()
                    or "ABSENT",
                    "selection_basis": ev["plan_selection_basis"],
                    "prescription_gy": [
                        float(x.TargetPrescriptionDose)
                        for x in getattr(pd, "DoseReferenceSequence", [])
                        if hasattr(x, "TargetPrescriptionDose")
                        and x.TargetPrescriptionDose is not None
                    ],
                    "fractions": [
                        int(x.NumberOfFractionsPlanned)
                        for x in getattr(pd, "FractionGroupSequence", [])
                        if hasattr(x, "NumberOfFractionsPlanned")
                        and x.NumberOfFractionsPlanned is not None
                    ],
                    "resolved_total_gy": plan.get("resolved_prescribed_dose_total_gy"),
                    "planned_fraction_count": plan.get("planned_fraction_count"),
                    "resolved_per_fraction_gy": plan.get(
                        "resolved_prescribed_dose_per_fraction_gy"
                    ),
                    "resolution_status": plan.get("prescription_resolution_status"),
                    "scope": plan.get("prescribed_dose_scope"),
                }
            )
        nf = root / ct["nifti_path"]
        use(nf)
        g = geometry(nf)
        ev["grid"] = g
        axial = is_axial_grid(g, settings=settings)
        planes = []
        series = ct["series_instance_uid"]
        pro = ct.get("nifti_provenance", {})
        if pro.get("nifti_sha256") and pro["nifti_sha256"] != used[-1]["sha256"]:
            raise ValueError("CT_nifti_hash_mismatch")
        ctfiles = sorted(f for f in (root / ct["dicom_dir"]).rglob("*") if f.is_file())
        sops = []
        slice_indices = []
        frames = set()
        # Acquisition planes can be sagittal/coronal even when NIfTI storage has
        # been permuted into an axial-looking grid. Keep the legacy path only
        # when both the grid and the acquired in-plane axes are axial.
        if axial and ctfiles:
            first = read_dcm(ctfiles[0], ["ImageOrientationPatient"])
            axial = np.allclose(
                np.abs(np.asarray(first.ImageOrientationPatient, float).reshape(2, 3)),
                np.array([[1, 0, 0], [0, 1, 0]]),
                atol=settings["orientation_atol"],
                rtol=settings["allclose_default_rtol"],
            )
        for f in ctfiles:
            d = read_dcm(
                f,
                [
                    "SOPInstanceUID",
                    "SeriesInstanceUID",
                    "FrameOfReferenceUID",
                    "ImagePositionPatient",
                    "ImageOrientationPatient",
                    "PixelSpacing",
                    "Rows",
                    "Columns",
                ],
            )
            if str(d.SeriesInstanceUID) != series:
                raise ValueError("DICOM_CT_series_mismatch")
            sops.append(str(d.SOPInstanceUID))
            frames.add(str(d.FrameOfReferenceUID))
            ipp = np.asarray(d.ImagePositionPatient, float)
            iop = np.asarray(d.ImageOrientationPatient, float)
            ps = np.asarray(d.PixelSpacing, float)
            q = (ipp - np.asarray(g["origin"])) @ np.linalg.inv(affine(g)).T
            if not np.allclose(
                q,
                np.round(q),
                atol=settings["origin_voxel_atol"],
                rtol=settings["allclose_default_rtol"],
            ):
                raise ValueError("DICOM_NIFTI_grid_origin_mismatch")
            if axial:
                if not np.allclose(
                    np.abs(iop.reshape(2, 3)),
                    np.array([[1, 0, 0], [0, 1, 0]]),
                    atol=settings["orientation_atol"],
                    rtol=settings["allclose_default_rtol"],
                ):
                    raise ValueError("DICOM_CT_nonaxial")
                if [int(d.Columns), int(d.Rows)] != g["size"][:2] or not np.allclose(
                    ps[::-1],
                    g["spacing"][:2],
                    atol=settings["spacing_mm_atol"],
                    rtol=settings["allclose_default_rtol"],
                ):
                    raise ValueError("DICOM_NIFTI_grid_spacing_mismatch")
                slice_indices.append(int(round(q[2])))
            else:
                planes.append(slice_plane(d, g, settings=settings))
        if axial:
            if sorted(slice_indices) != list(range(g["size"][2])):
                raise ValueError("DICOM_CT_incomplete_slice_grid")
        else:
            planes = validate_slice_planes(planes, g)
        if len(set(sops)) != len(sops) or len(frames) != 1:
            raise ValueError("DICOM_CT_duplicate_or_frame_mismatch")
        for roi in ds.StructureSetROISequence:
            if str(roi.ReferencedFrameOfReferenceUID) not in frames:
                raise ValueError("RTSTRUCT_CT_frame_mismatch")
        ev["ct_identity"] = {
            "series_identity_sha256": hashlib.sha256(series.encode()).hexdigest(),
            "n_dicom_slices": len(sops),
            "sop_set_sha256": hashlib.sha256(
                "".join(sorted(sops)).encode()
            ).hexdigest(),
            "equivalent_signed_axis_grid_verified": True,
        }
        if not axial:
            ev["rasterization_extension"] = "dicom_slice_planes_v1"
        ev["CT_pixel_equivalence"] = pixel_equivalence(
            ctfiles, nf, g, settings=settings
        )
        targets, exclusions = select_targets(ds, rule)
        ev["target_exclusions"] = exclusions
        ev["n_targets_selected"] = len(targets)
        ev["target_selection_routes"] = dict(
            collections.Counter(x["route"] for x in targets)
        )
        if not targets:
            raise ValueError("no_targets_after_helper_exclusion")
        indices = []
        for rec in targets:
            rec, ids = rasterize(rec, g, None if axial else planes, settings=settings)
            ev["targets"].append(rec)
            indices.append(ids)
        ev["target_pairwise_containment"] = []
        for a in range(len(indices)):
            for b in range(a):
                n = len(np.intersect1d(indices[a], indices[b], assume_unique=True))
                ev["target_pairwise_containment"].append(
                    {
                        "a": targets[a]["roi_number"],
                        "b": targets[b]["roi_number"],
                        "overlap_over_smaller": n
                        / min(len(indices[a]), len(indices[b])),
                    }
                )
        sf = root / ".segmentation_done"
        use(sf)
        senttext = sf.read_text()
        sent = (
            json.loads(senttext)
            if senttext.strip().startswith("{")
            else {"status": senttext.strip()}
        )
        ev["segmentation_sentinel_status"] = sent.get("status")
        if sent.get("status") != "ok":
            raise ValueError("segmentation_sentinel_not_ok")
        sp = root / "metadata/segmentation_status.json"
        use(sp)
        status = json.loads(sp.read_text())
        if status["status"] != "ok":
            raise ValueError("segmentation_status_not_ok")
        sd = Path(status["evidence"]["segmentation_directory"])
        if not sd.is_relative_to(root / "Segmentation_TotalSegmentator"):
            raise ValueError("segmentation_directory_outside_course")
        mf = sd / "manifest.json"
        use(mf)
        manifest = json.loads(mf.read_text())
        if manifest.get("source_series_instance_uid") != series or manifest.get(
            "source_nifti_sha256"
        ) != sha(nf):
            raise ValueError("segmentation_source_identity_mismatch")
        ev["segmentation_identity"] = {
            "manifest_sha256": used[-1]["sha256"],
            "source_series_matches": True,
            "source_CT_bytes_match": True,
        }
        declared = [n for model in manifest["models"] for n in model.get("masks", [])]
        actual = {f.name for f in sd.glob("*.nii.gz")}
        if set(declared) != actual:
            raise ValueError("segmentation_manifest_mask_set_mismatch")
        bindings = {x["path"]: x for x in sent.get("outputs", [])}
        empty = []
        empty_hashes = set()
        for name in sorted(declared):
            f = sd / name
            fp = use(f)
            rel = str(f.relative_to(root))
            bound = bindings.get(rel)
            if not bound or bound.get("sha256") != fp["sha256"]:
                raise ValueError("segmentation_mask_sentinel_hash_mismatch")
            organ = name.removesuffix(".nii.gz").split("--", 1)[-1]
            if fp["sha256"] in empty_hashes:
                empty.append(organ)
                continue
            og = geometry(f)
            if og["size"] != g["size"] or not all(
                np.allclose(
                    og[x],
                    g[x],
                    atol=settings["mask_grid_atol"],
                    rtol=settings["allclose_default_rtol"],
                )
                for x in ["origin", "direction", "spacing"]
            ):
                raise ValueError("segmentation_grid_mismatch")
            a = read_mask_fast(f, g)
            st = organ_stats(a, g, exact_bounds=not axial, settings=settings)
            if st is None:
                empty.append(organ)
                empty_hashes.add(fp["sha256"])
                continue
            ev["organs"][organ] = st
            flat = a.reshape(-1)
            for rec, ids in zip(ev["targets"], indices):
                n = int(np.count_nonzero(flat[ids]))
                frac = n / rec["n_voxels"]
                rec["organ_overlaps"][organ] = {
                    "n_voxels": n,
                    "target_fraction": frac,
                    "organ_fraction": n / st["n_voxels"],
                    "volume_ml": float(n * np.prod(g["spacing"]) / 1000),
                }
                rec["organ_centroid_distances_mm"][organ] = float(
                    np.linalg.norm(
                        np.asarray(rec["centroid_lps_mm"]) - st["centroid_lps_mm"]
                    )
                )
        ev["empty_organ_masks"] = empty
        ev["organ_panel"] = sorted(ev["organs"]) + empty
        if not ev["organs"]:
            raise ValueError("automatic_segmentation_all_empty")
        ev["status"] = "measured"
    except Exception as ex:
        ev["errors"].append(
            type(ex).__name__
            + ": "
            + (
                str(ex)
                if isinstance(ex, ValueError)
                and re.fullmatch(r"[A-Za-z0-9_]+", str(ex))
                else "technical_failure_details_not_exported"
            )
        )
        ev["status"] = "measurement_error"
    ev["input_fingerprints"] = used
    ev["input_changed_during_measurement"] = [
        x["path"]
        for x in used
        if not Path(x["path"]).exists()
        or Path(x["path"]).stat().st_size != x["size"]
        or Path(x["path"]).stat().st_mtime_ns != x["mtime_ns"]
    ]
    if ev["input_changed_during_measurement"]:
        ev["errors"].append("inputs_changed_during_measurement")
        ev["status"] = "measurement_error"
    ev["elapsed_seconds"] = round(time.monotonic() - start, 3)
    ev["finished_utc"] = now()
    return ev


def fraction_in(t, lo, hi):
    return (
        sum(n for z, n in t["z_histogram_mm_counts"] if lo <= z <= hi) / t["n_voxels"]
    )


def classify(ev, rule, variant="C1"):
    q = rule["variants"][variant]
    ts = ev["targets"]
    org = ev["organs"]
    details = []
    if ev["errors"] or ev["status"] != "measured":
        return "U", {
            "reason": "measurement_or_identity_failure",
            "errors": ev["errors"],
        }
    if any(t["outside_ct_contour_points"] or t["unsupported_contours"] for t in ts):
        return "U", {
            "reason": "target_outside_CT_or_unsupported_contour",
            "outside_points": sum(t["outside_ct_contour_points"] for t in ts),
        }
    if any(
        x["reason"] == "contradictory_interpreted_types"
        for x in ev["target_exclusions"]
    ):
        return "U", {"reason": "contradictory_target_identification"}
    # Organ occupancy must span targets. Blank field of view alone is not anatomical coverage.
    zlo = min(o["bbox_lps_mm"][0][2] for o in org.values())
    zhi = max(o["bbox_lps_mm"][1][2] for o in org.values())
    if any(
        t["bbox_lps_mm"][0][2] < zlo - q["coverage_tolerance_mm"]
        or t["bbox_lps_mm"][1][2] > zhi + q["coverage_tolerance_mm"]
        for t in ts
    ):
        return "U", {
            "reason": "automatic_organs_do_not_span_target_z",
            "organ_z_span": [zlo, zhi],
        }
    hips = [org[x] for x in ["hip_left", "hip_right"] if x in org]
    upper = org.get("vertebrae_L5")
    pelvis = None
    if len(hips) == 2 and upper:
        pelvis = [
            min(h["bbox_lps_mm"][0][2] for h in hips) - q["pelvis_padding_mm"],
            upper["bbox_lps_mm"][1][2] + q["pelvis_padding_mm"],
        ]
        fractions = [fraction_in(t, *pelvis) for t in ts]
    else:
        # Sufficient nonpelvic proof without extrapolating an unseen pelvis.
        anchors = [
            o
            for n, o in org.items()
            if re.fullmatch(r"vertebrae_(?:C[1-7]|T(?:[1-9]|1[0-2])|L[1-4])", n)
        ]
        cutoff = min([o["bbox_lps_mm"][0][2] for o in anchors], default=None)
        if cutoff is not None:
            cutoff += q["pelvis_padding_mm"]
        if cutoff is not None and all(t["bbox_lps_mm"][0][2] >= cutoff for t in ts):
            return "X", {
                "reason": "all_targets_superior_to_observed_nonpelvic_vertebra_lower_extent",
                "nonpelvic_cutoff_z_mm": cutoff,
                "missing_pelvic_landmarks": [
                    n for n in ["hip_left", "hip_right", "vertebrae_L5"] if n not in org
                ],
                "target_pelvic_fractions": [0.0] * len(ts),
                "pelvis": None,
            }
        return "U", {
            "reason": "pelvic_landmarks_absent_and_no_complete_nonpelvic_proof",
            "missing_landmarks": [
                n for n in ["hip_left", "hip_right", "vertebrae_L5"] if n not in org
            ],
        }
    pelvic = [f >= q["pelvic_target_fraction"] for f in fractions]
    outside = [f <= 1 - q["pelvic_target_fraction"] for f in fractions]
    bladder = [t["roi_number"] for t in ts if bladder_hit(t, variant, rule)]
    evd = {
        "pelvis": pelvis,
        "target_pelvic_fractions": fractions,
        "bladder_target_roi_numbers": bladder,
    }
    if all(outside):
        return "X", dict(evd, reason="all_targets_outside_pelvic_slab")
    # A crossing target with material components on both sides is a mixed field.
    if any(not p and not x for p, x in zip(pelvic, outside)) or (
        any(pelvic) and any(outside)
    ):
        return "MX", dict(evd, reason="pelvic_and_nonpelvic_target_volume")
    if not all(pelvic):
        return "U", dict(evd, reason="unresolved_pelvic_extent")
    if "urinary_bladder" not in org:
        return "U", dict(
            evd, reason="empty_automatic_bladder_mask_prevents_pelvic_subclassification"
        )
    if not bladder:
        return "P", dict(
            evd, reason="all_targets_pelvic_without_substantial_bladder_overlap"
        )
    other = []

    # Gross targets may abut or overlap while having distinct organ-dominant centers.
    # Never mistake a shared encompassing CTV/PTV for a second primary target.
    def frac(t, n):
        return t["organ_overlaps"].get(n, {}).get("target_fraction", 0)

    bladder_gtvs = [
        t
        for t in ts
        if t["roi_number"] in bladder
        and t["type"] == "GTV"
        and frac(t, "urinary_bladder")
        >= max([frac(t, n) for n in rule["other_pelvic_primary_organs"]], default=0)
    ]
    for t in ts:
        if t["type"] != "GTV" or t in bladder_gtvs:
            continue
        hits = [
            n
            for n in rule["other_pelvic_primary_organs"]
            if frac(t, n) >= q["other_primary_target_fraction"]
            and frac(t, n)
            >= q["other_primary_dominance_ratio"] * frac(t, "urinary_bladder")
        ]
        if not hits:
            continue
        for b in bladder_gtvs:
            containment = next(
                x["overlap_over_smaller"]
                for x in ev["target_pairwise_containment"]
                if {x["a"], x["b"]} == {t["roi_number"], b["roi_number"]}
            )
            distance = float(
                np.linalg.norm(np.asarray(t["centroid_lps_mm"]) - b["centroid_lps_mm"])
            )
            if (
                containment <= q["distinct_target_max_containment"]
                and distance >= q["distinct_target_min_centroid_distance_mm"]
            ):
                other.append(
                    {
                        "roi_number": t["roi_number"],
                        "organs": hits,
                        "bladder_GTV_roi_number": b["roi_number"],
                        "containment_with_bladder_GTV": containment,
                        "centroid_separation_mm": distance,
                    }
                )
    if other:
        return "MP", dict(
            evd,
            reason="distinct_bladder_and_other_pelvic_gross_targets",
            other_primary_geometry=other,
        )
    return "B", dict(
        evd, reason="substantial_bladder_target_overlap_no_nonpelvic_target"
    )


def bladder_hit(t, criterion, rule):
    x = t.get("organ_overlaps", {}).get("urinary_bladder", {})
    q = rule["criteria"][criterion]
    return (
        x.get("target_fraction", 0) >= q.get("target_fraction_min", 0)
        and x.get("volume_ml", 0) >= q.get("intersection_ml_min", 0)
        and x.get("organ_fraction", 0) >= q.get("organ_fraction_min", 0)
    )


def pixel_equivalence(ctfiles, nf, g, settings=None):
    settings = load_rules()["geometry"]["measurement"] if settings is None else settings
    im = sitk.ReadImage(str(nf))
    a = sitk.GetArrayViewFromImage(im)
    maximum = 0.0
    pixels = 0
    for index in sorted(set([0, len(ctfiles) // 2, len(ctfiles) - 1])):
        d = pydicom.dcmread(ctfiles[index])
        v = d.pixel_array.astype(float) * float(getattr(d, "RescaleSlope", 1)) + float(
            getattr(d, "RescaleIntercept", 0)
        )
        ipp = np.asarray(d.ImagePositionPatient, float)
        iop = np.asarray(d.ImageOrientationPatient, float)
        ps = np.asarray(d.PixelSpacing, float)
        jj, ii = np.indices(v.shape)
        xyz = ipp + ii[..., None] * iop[:3] * ps[1] + jj[..., None] * iop[3:] * ps[0]
        ijk = np.rint(
            (xyz - np.asarray(g["origin"])) @ np.linalg.inv(affine(g)).T
        ).astype(int)
        diff = float(np.max(np.abs(v - a[ijk[..., 2], ijk[..., 1], ijk[..., 0]])))
        maximum = max(maximum, diff)
        pixels += v.size
    if maximum > settings["pixel_equivalence_hu_atol"]:
        raise ValueError("DICOM_NIfTI_pixel_equivalence_failed")
    return {
        "sampled_slices": len(set([0, len(ctfiles) // 2, len(ctfiles) - 1])),
        "pixels_compared": pixels,
        "maximum_absolute_HU_difference": maximum,
    }


def read_mask_fast(path, g):
    """Decode scalar NIfTI-1 masks without image copies; grid is separately checked."""
    import gzip, struct

    data = gzip.decompress(Path(path).read_bytes())
    endian = "<" if struct.unpack("<i", data[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", data[:4])[0] != 348 or data[344:348] != b"n+1\0":
        raise ValueError("unsupported_mask_NIfTI_header")
    dim = struct.unpack(endian + "8h", data[40:56])
    dtype_code = struct.unpack(endian + "h", data[70:72])[0]
    dtype = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        256: "i1",
        512: "u2",
        768: "u4",
    }.get(dtype_code)
    if (
        dtype is None
        or list(dim[1:4]) != g["size"]
        or any(v > 1 for v in dim[4 : dim[0] + 1])
    ):
        raise ValueError("unsupported_mask_NIfTI_dimensions_or_type")
    offset = int(struct.unpack(endian + "f", data[108:112])[0])
    slope, intercept = struct.unpack(endian + "2f", data[112:120])
    if slope not in [0.0, 1.0] or intercept not in [0.0] and slope != 0.0:
        raise ValueError("scaled_mask_NIfTI_requires_validated_decoder")
    count = int(np.prod(g["size"]))
    a = np.frombuffer(
        data, dtype=np.dtype(endian + dtype), count=count, offset=offset
    ).reshape(tuple(g["size"][::-1]))
    return a


def signature(p):
    p = Path(p)
    s = p.stat()
    h = sha(p)
    t = p.stat()
    if (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) != (
        t.st_dev,
        t.st_ino,
        t.st_size,
        t.st_mtime_ns,
        t.st_ctime_ns,
    ):
        raise ValueError("input_changed_during_hash")
    return {"sha256": h, "size": t.st_size}


def contained(p, root):
    p = Path(p)
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError("input_path_outside_course")
    return p


def source_state(root):
    """Exact byte closure, including DICOM CT and collection membership omitted by old cache."""
    mp = root / "metadata/case_metadata.json"
    m = _load(mp)
    co = m["course_contract"]
    ct = co["planning_ct"]
    files = {
        mp,
        root / ".segmentation_done",
        root / "metadata/segmentation_status.json",
    }
    for kind in ("RTSTRUCT", "RTPLAN"):
        files.update(p for p in (root / "DICOM" / kind).rglob("*") if p.is_file())
    if ct.get("nifti_path"):
        files.add(contained(root / ct["nifti_path"], root))
    if ct.get("dicom_dir"):
        files.update(
            p for p in contained(root / ct["dicom_dir"], root).rglob("*") if p.is_file()
        )
    status = _load(root / "metadata/segmentation_status.json")
    sd = contained(
        status["evidence"]["segmentation_directory"],
        root / "Segmentation_TotalSegmentator",
    )
    mf = sd / "manifest.json"
    files.add(mf)
    manifest = _load(mf)
    names = [n for model in manifest["models"] for n in model.get("masks", [])]
    actual = {p.name for p in sd.glob("*.nii.gz")}
    if len(names) != len(set(names)) or set(names) != actual:
        raise ValueError("segmentation_manifest_mask_set_mismatch")
    labels = [n.removesuffix(".nii.gz").split("--", 1)[-1] for n in names]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate_mask_label")
    files.update(contained(sd / n, sd) for n in names)
    fp = {str(p.relative_to(root)): signature(p) for p in sorted(files)}
    sent = _load(root / ".segmentation_done")
    if sent.get("status") != "ok" or status.get("status") != "ok":
        raise ValueError("segmentation_not_complete")
    if (
        str(sent.get("patient_id")) != root.parent.name
        or str(sent.get("course_id")) != root.name
    ):
        raise ValueError("segmentation_sentinel_identity_mismatch")
    outputs = {x["path"]: x for x in sent.get("outputs", [])}
    if len(outputs) != len(sent.get("outputs", [])):
        raise ValueError("duplicate_sentinel_output")
    masks = {}
    for n, label in zip(names, labels):
        rel = str((sd / n).relative_to(root))
        v = fp[rel]
        bound = outputs.get(rel, {})
        if bound.get("sha256") != v["sha256"] or (
            "size" in bound and bound["size"] != v["size"]
        ):
            raise ValueError("segmentation_mask_sentinel_hash_mismatch")
        masks[label] = v
    if (
        ct.get("nifti_path")
        and manifest.get("source_nifti_sha256") != fp[ct["nifti_path"]]["sha256"]
    ):
        raise ValueError("segmentation_source_hash_mismatch")
    if manifest.get("source_series_instance_uid") != ct.get("series_instance_uid"):
        raise ValueError("segmentation_source_identity_mismatch")
    return {
        "files": fp,
        "masks": masks,
        "metadata": m,
        "manifest": manifest,
        "segmentation_code_identity": sent.get("code_identity"),
        "segmentation_code_revisions": sent.get("code_revisions"),
    }


def _load(path):
    return json.loads(Path(path).read_text())


def measure_course(course, rule):
    """Measure an organized course with the frozen byte-closure checks."""
    root = Path(course).resolve()
    try:
        before = source_state(root)
        metadata = before["metadata"]
        if (str(metadata.get("patient_id")), str(metadata.get("course_id"))) != (
            root.parent.name,
            root.name,
        ):
            raise ValueError("course_metadata_identity_mismatch")
        evidence = measure(root, rule)
        if source_state(root)["files"] != before["files"]:
            raise ValueError("input_changed_during_measurement")
        return evidence
    except Exception as exc:
        reason = (
            str(exc)
            if isinstance(exc, ValueError) and re.fullmatch(r"[A-Za-z0-9_]+", str(exc))
            else "input_validation_failed"
        )
        return {
            "status": "measurement_error",
            "errors": [reason],
            "targets": [],
            "organs": {},
        }
