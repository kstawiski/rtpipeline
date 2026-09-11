"""Synthetic regression: one D16 (partially-unparseable) ROI in the
integration RTSTRUCT must not suppress the whole custom-structure harvest.

When RS_auto carries a new ROI_CONTOUR_PARTIALLY_UNPARSEABLE ROI on a course
whose other contours are clean, the integration branches of RS_custom
production went through ``_rtstruct_builder_source``, which is
all-or-nothing: a single unresolved ROI code raised and the entire
manual/auto integration was skipped with only a warning. The base harvest
already used the per-ROI-lenient ``create_scoped_rtstruct``, so every clean
dependency contour in the same integration RTSTRUCT was lost and the
configured structures degraded to ``source_unavailable`` with
``available_sources=[]``.

Behaviour under test:
  1. The integration branches use ``create_scoped_rtstruct`` (per-ROI
     lenient): clean contours in the same RTSTRUCT are still harvested.
  2. No ROI is ever taken from a segmentation mask. A D16 contour-quality
     disposition leaves that ROI unavailable and the structure honestly
     partial. An earlier revision recovered a D16 ROI from a same-name
     TotalSegmentator mask whose manifest named the contracted planning CT
     series and whose geometry matched. Inspecting the producers showed that
     this is not source identity (see the block comment above the
     source-identity tests below), so the recovery was withdrawn.
  3. Scope-resolution failures (``ROI_UNRESOLVED_SOURCE_SCOPE``) and
     ``ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE`` are never substituted, even
     when a same-name NIfTI exists, and the base-scope
     ``CUSTOM_SOURCE_GEOMETRY_UNRESOLVED`` pre-check is unchanged.
  4. A same name is not identity: an unprovenanced mask manifest, a manifest
     bound to a different planning CT series, two masks that map to one ROI
     name, a mask that does not share the CT's physical space, and a fully
     provenanced same-name mask alike leave the structure honestly partial.
  5. A failed or empty rasterisation of a manual ROI, and a technical reader
     error, stay failures instead of becoming a same-name automatic mask.
"""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from rt_utils import RTStructBuilder

from test_science_batch_c import _build_real_rtstruct
from rtpipeline import custom_structures_rtstruct as custom
from rtpipeline.rtstruct_geometry import create_scoped_rtstruct


def _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch):
    contract = SimpleNamespace(
        authoritative_rtstruct_path=rs_manual,
        planning_ct_dir=ct,
        planning_ct={"series_instance_uid": series_uid},
        planning_ct_nifti=None,
    )
    monkeypatch.setattr(custom, "load_course_contract", lambda _: contract)


def _write_totalseg_mask(
    course_dir,
    mask_file,
    arr_zyx,
    series_uid,
    *,
    run="ts_run",
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
):
    """Write a TotalSegmentator run dir (manifest.json + NIfTI) the way
    ``segmentation.py`` does: the manifest records the planning CT series the
    masks were produced against, which is the provenance the recovery requires.
    By default the NIfTI shares the fixture CT's physical space exactly
    (origin 0, spacing 1, identity orientation), so the resample onto the CT is
    an identity mapping. ``series_uid=None`` writes the legacy unprovenanced
    manifest."""
    seg_dir = course_dir / "Segmentation_TotalSegmentator" / run
    seg_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = seg_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"models": [{"model": "total", "masks": []}]}
    )
    if series_uid is not None:
        manifest["source_series_instance_uid"] = str(series_uid)
        manifest["planning_ct_series_instance_uid"] = str(series_uid)
    manifest["models"][0]["masks"].append(mask_file)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx).astype(np.uint8))
    img.SetOrigin(tuple(float(v) for v in origin))
    img.SetSpacing(tuple(float(v) for v in spacing))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img, str(seg_dir / mask_file))
    return seg_dir / mask_file


def _roi_number(rt, roi_name):
    for roi in rt.ds.StructureSetROISequence:
        if str(roi.ROIName) == roi_name:
            return int(roi.ROINumber)
    raise AssertionError(f"ROI {roi_name!r} not found in RTSTRUCT")


def _contour_item(rt, roi_name):
    number = _roi_number(rt, roi_name)
    for item in rt.ds.ROIContourSequence:
        if int(item.ReferencedROINumber) == number:
            return item
    raise AssertionError(f"no ROIContourSequence item for {roi_name!r}")


def _make_partially_unparseable(rt, roi_name):
    """Append a 3-point collinear contour to an ROI that already has valid
    closed-planar contours: exactly one contour is invalid geometry, so
    roi_geometry_code -> ROI_CONTOUR_PARTIALLY_UNPARSEABLE (the D16 code)."""
    item = _contour_item(rt, roi_name)
    assert item.ContourSequence, "ROI has no contours to corrupt"
    good = item.ContourSequence[0]
    x0, y0, z = float(good.ContourData[0]), float(good.ContourData[1]), float(good.ContourData[2])
    bad = copy.deepcopy(good)
    bad.ContourData = [x0, y0, z, x0 + 2.0, y0, z, x0 + 4.0, y0, z]
    bad.NumberOfContourPoints = 3
    bad.ContourImageSequence = Sequence([])
    item.ContourSequence.append(bad)


def _make_unresolved_scope(rt, roi_name, dz=30.0):
    """Drop image references and shift all contours dz off the CT planes:
    geometry stays valid but no image plane matches, so
    resolve_roi_scopes -> ROI_UNRESOLVED_SOURCE_SCOPE for the whole ROI."""
    item = _contour_item(rt, roi_name)
    for contour in item.ContourSequence:
        del contour.ContourImageSequence
        points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        points[:, 2] += dz
        contour.ContourData = [float(v) for v in points.ravel()]


def _add_declared_empty_roi(rt, roi_name):
    """Append an ROI whose ContourSequence is empty: an explicit clinical
    empty -> ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE."""
    number = max(_roi_number(rt, str(r.ROIName)) for r in rt.ds.StructureSetROISequence) + 1
    frame_uid = str(rt.ds.StructureSetROISequence[0].ReferencedFrameOfReferenceUID)
    roi = Dataset()
    roi.ROINumber = number
    roi.ROIName = roi_name
    roi.ReferencedFrameOfReferenceUID = frame_uid
    item = Dataset()
    item.ReferencedROINumber = number
    item.ContourSequence = Sequence([])
    rt.ds.StructureSetROISequence.append(roi)
    rt.ds.ROIContourSequence.append(item)


def _fixture(tmp_path):
    """CT (12x12x4, 1mm) + clean RS_manual (PTV) returned for reuse."""
    ct = tmp_path / "ct"
    ct.mkdir()
    rt_manual = _build_real_rtstruct(ct, n_slices=4, side=12)
    rs_manual = tmp_path / "RS_manual.dcm"
    rt_manual.save(str(rs_manual))
    return ct, rs_manual, str(rt_manual.series_data[0].SeriesInstanceUID)


def test_auto_integration_survives_d16_roi_without_substituting_a_mask(tmp_path, monkeypatch):
    """Production shape: base=RS_manual, RS_auto carries the clean union
    dependencies plus one D16 ROI. Pre-D20 the all-or-nothing
    _rtstruct_builder_source raised on the D16 code and the whole auto
    harvest was skipped -> source_unavailable, available_sources=[]. The clean
    dependencies must survive; the D16 ROI must not be filled in from the
    same-name mask, so the union is partial over the two contours."""
    ct, rs_manual, series_uid = _fixture(tmp_path)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)

    rt_auto = RTStructBuilder.create_new(dicom_series_path=str(ct))
    mask_a = np.zeros((12, 12, 4), dtype=bool); mask_a[2:5, 2:5, 1:3] = True
    mask_b = np.zeros((12, 12, 4), dtype=bool); mask_b[2:5, 7:10, 1:3] = True
    # Every fixture mask stays off the CT boundary: mask_is_cropped would
    # otherwise mark the union partial for a reason unrelated to this repair.
    mask_c = np.zeros((12, 12, 4), dtype=bool); mask_c[7:10, 2:10, 1:3] = True
    rt_auto.add_roi(mask=mask_a, name="iliac_artery_L")
    rt_auto.add_roi(mask=mask_b, name="iliac_artery_R")
    rt_auto.add_roi(mask=mask_c, name="iliac_venous_L")
    _make_partially_unparseable(rt_auto, "iliac_venous_L")
    rs_auto = tmp_path / "RS_auto.dcm"
    rt_auto.ds.save_as(str(rs_auto))

    # Fixture sanity: the clean ROIs are parseable, the third is exactly D16.
    codes = {r.roi_name: r.code for r in create_scoped_rtstruct(ct, rs_auto).scopes.values()}
    assert codes["iliac_artery_L"] is None
    assert codes["iliac_artery_R"] is None
    assert codes["iliac_venous_L"] == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"

    nifti_zyx = np.zeros((4, 12, 12), dtype=np.uint8)
    nifti_zyx[1:3, 7:10, 2:10] = 1
    _write_totalseg_mask(tmp_path, "total--iliac_venous_L.nii.gz", nifti_zyx, series_uid)

    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: iliac_vess\n"
        "    source_structures: [iliac_artery_L, iliac_artery_R, iliac_venous_L]\n"
        "    operation: union\n",
        encoding="utf-8",
    )

    out = custom._create_custom_structures_rtstruct_unlocked(tmp_path, config, rs_manual, rs_auto)
    assert out is not None and out.exists()

    meta = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    outcome = meta["custom_structure_outcomes"]["iliac_vess"]
    # D20: the two clean dependencies in the same RTSTRUCT are still harvested.
    assert sorted(outcome["available_sources"]) == ["iliac_artery_L", "iliac_artery_R"]
    # D16: the ROI whose contour cannot be read stays unavailable.
    assert outcome["status"] == "generated_partial"
    assert outcome["unavailable_sources"] == ["iliac_venous_L"]
    assert _recorded_fallbacks(meta) == {}
    assert meta["unread_source_rois"]["iliac_venous_L"]["code"] == (
        "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"
    )

    # The published structure is the union of the two clean contour masks only.
    expected = mask_a | mask_b
    published = pydicom.dcmread(str(out))
    names = [str(r.ROIName) for r in published.StructureSetROISequence]
    assert "iliac_vess__partial" in names
    assert {"iliac_artery_L", "iliac_artery_R"} <= set(names)
    got = RTStructBuilder.create_from(str(ct), str(out)).get_roi_mask_by_name(
        "iliac_vess__partial"
    )
    assert np.array_equal(got, expected)


def test_unresolved_scope_and_declared_empty_are_never_nifti_substituted(tmp_path, monkeypatch):
    """Guardrail: in the auto integration, an out-of-scope ROI and a
    declared-empty ROI must NOT be rescued by a same-name NIfTI, even though
    one exists. D16 contour-quality failures are also not substituted."""
    ct, rs_manual, series_uid = _fixture(tmp_path)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)

    rt_auto = RTStructBuilder.create_new(dicom_series_path=str(ct))
    mask_d = np.zeros((12, 12, 4), dtype=bool); mask_d[2:5, 2:5, 1:3] = True
    mask_e = np.zeros((12, 12, 4), dtype=bool); mask_e[7:10, 7:10, 1:3] = True
    rt_auto.add_roi(mask=mask_d, name="colon")
    rt_auto.add_roi(mask=mask_e, name="small_bowel")
    _make_unresolved_scope(rt_auto, "small_bowel")
    _add_declared_empty_roi(rt_auto, "rectum")
    rs_auto = tmp_path / "RS_auto.dcm"
    rt_auto.ds.save_as(str(rs_auto))

    codes = {r.roi_name: r.code for r in create_scoped_rtstruct(ct, rs_auto).scopes.values()}
    assert codes["colon"] is None
    assert codes["small_bowel"] == "ROI_UNRESOLVED_SOURCE_SCOPE"
    assert codes["rectum"] == "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE"

    # Fully provenanced same-name NIfTI masks exist for both bad ROIs: their
    # dispositions are identity/clinical statements, so they must not win.
    for roi_name in ("small_bowel", "rectum"):
        arr = np.zeros((4, 12, 12), dtype=np.uint8)
        arr[1:2, 1:2, 1:2] = 1
        _write_totalseg_mask(tmp_path, f"total--{roi_name}.nii.gz", arr, series_uid)

    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: bowel_bag\n"
        "    source_structures: [colon, small_bowel, rectum]\n"
        "    operation: union\n",
        encoding="utf-8",
    )

    out = custom._create_custom_structures_rtstruct_unlocked(tmp_path, config, rs_manual, rs_auto)
    assert out is not None and out.exists()
    meta = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    outcome = meta["custom_structure_outcomes"]["bowel_bag"]
    assert outcome["status"] == "generated_partial"
    assert outcome["available_sources"] == ["colon"]
    assert outcome["unavailable_sources"] == ["small_bowel", "rectum"]
    assert outcome["realized_name"] == "bowel_bag__partial"
    # No rescue happened, so no fallback provenance is recorded.
    assert _recorded_fallbacks(meta) == {}

    got = RTStructBuilder.create_from(str(ct), str(out)).get_roi_mask_by_name("bowel_bag__partial")
    assert np.array_equal(got, mask_d)


def test_base_unresolved_source_still_raises_precheck(tmp_path, monkeypatch):
    """Guardrail: a D16 ROI in the base (authoritative manual) RTSTRUCT that a
    configured structure references must still hard-fail with
    CUSTOM_SOURCE_GEOMETRY_UNRESOLVED — even when a same-name NIfTI exists.
    The fix must not weaken the required-source geometry check."""
    ct, rs_manual, series_uid = _fixture(tmp_path)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)

    # Corrupt the base PTV contours in place and re-publish the manual RS.
    from test_science_batch_c import _build_real_rtstruct as _unused  # noqa: F401  (clarity)
    rt_manual = _build_real_rtstruct(ct, n_slices=4, side=12)
    _make_partially_unparseable(rt_manual, "PTV")
    rt_manual.ds.save_as(str(rs_manual))
    codes = {r.roi_name: r.code for r in create_scoped_rtstruct(ct, rs_manual).scopes.values()}
    assert codes["PTV"] == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"

    # A fully provenanced same-name NIfTI for PTV exists; it must not stand in
    # for a base (authoritative) D16 source.
    arr = np.zeros((4, 12, 12), dtype=np.uint8)
    arr[1:3, 2:10, 2:10] = 1
    _write_totalseg_mask(tmp_path, "total--PTV.nii.gz", arr, series_uid)

    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: ptv_union\n"
        "    source_structures: [PTV]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    with pytest.raises(custom.CustomStructureRTStructError,
                       match="CUSTOM_SOURCE_GEOMETRY_UNRESOLVED"):
        custom._create_custom_structures_rtstruct_unlocked(tmp_path, config, rs_manual)


def _d16_auto_course(tmp_path, monkeypatch):
    """base = clean RS_manual; RS_auto = one clean dependency + one D16 one.

    ``iliac_vess`` is configured over both, so the union is ``generated`` only
    when the D16 ROI is recovered and ``generated_partial`` when it is refused.
    """
    ct, rs_manual, series_uid = _fixture(tmp_path)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)

    rt_auto = RTStructBuilder.create_new(dicom_series_path=str(ct))
    clean = np.zeros((12, 12, 4), dtype=bool); clean[2:5, 2:5, 1:3] = True
    broken = np.zeros((12, 12, 4), dtype=bool); broken[7:10, 2:10, 1:3] = True
    rt_auto.add_roi(mask=clean, name="iliac_artery_L")
    rt_auto.add_roi(mask=broken, name="iliac_venous_L")
    _make_partially_unparseable(rt_auto, "iliac_venous_L")
    rs_auto = tmp_path / "RS_auto.dcm"
    rt_auto.ds.save_as(str(rs_auto))

    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: iliac_vess\n"
        "    source_structures: [iliac_artery_L, iliac_venous_L]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        ct=ct, rs_manual=rs_manual, rs_auto=rs_auto, series_uid=series_uid,
        config=config, clean=clean,
    )


def _recovery_nifti():
    arr = np.zeros((4, 12, 12), dtype=np.uint8)
    arr[1:3, 7:10, 2:10] = 1
    return arr


def _assert_recovery_refused(course, tmp_path):
    """The D16 ROI stays unavailable and the union is honestly partial."""
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, course.config, course.rs_manual, course.rs_auto
    )
    assert out is not None and out.exists()
    meta = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    outcome = meta["custom_structure_outcomes"]["iliac_vess"]
    assert outcome["status"] == "generated_partial"
    assert outcome["available_sources"] == ["iliac_artery_L"]
    assert outcome["unavailable_sources"] == ["iliac_venous_L"]
    assert _recorded_fallbacks(meta) == {}
    assert meta["unread_source_rois"]["iliac_venous_L"]["code"] == (
        "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"
    )
    published = pydicom.dcmread(str(out))
    assert "iliac_venous_L" not in [str(r.ROIName) for r in published.StructureSetROISequence]
    got = RTStructBuilder.create_from(str(course.ct), str(out)).get_roi_mask_by_name(
        "iliac_vess__partial"
    )
    assert np.array_equal(got, course.clean)


def test_the_best_provenanced_same_name_mask_is_still_refused(tmp_path, monkeypatch):
    """This case previously succeeded: the manifest names the contracted
    planning CT series, exactly one mask maps to the ROI name, the mask shares
    the CT's physical space, and its content agrees with the ROI. It is now
    refused, because none of that identifies the mask the ROI was published
    from -- the producers record no such link, so an equally well-matched mask
    from a different run or a rewritten mask file is indistinguishable from
    this one (see the source-identity block below). The structure stays
    honestly partial and the ROI is reported unread."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    _write_totalseg_mask(
        tmp_path, "total--iliac_venous_L.nii.gz", _recovery_nifti(), course.series_uid
    )
    _assert_recovery_refused(course, tmp_path)


def test_unprovenanced_mask_manifest_never_recovers(tmp_path, monkeypatch):
    """A legacy manifest that records no planning CT series is not provenance:
    the mask may belong to any CT, so the same name proves nothing."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    _write_totalseg_mask(
        tmp_path, "total--iliac_venous_L.nii.gz", _recovery_nifti(), None
    )
    _assert_recovery_refused(course, tmp_path)


def test_mask_manifest_from_another_series_never_recovers(tmp_path, monkeypatch):
    """Same ROI name, same physical space, different planning CT series: the
    mask set was produced for another image and must not stand in."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    _write_totalseg_mask(
        tmp_path,
        "total--iliac_venous_L.nii.gz",
        _recovery_nifti(),
        generate_uid(),
    )
    _assert_recovery_refused(course, tmp_path)


def test_two_masks_for_one_roi_name_are_ambiguous_not_recovered(tmp_path, monkeypatch):
    """Current and legacy naming can both resolve to one ROI name. Two candidate
    sources means the ROI's source is unknown, so neither may be used."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    _write_totalseg_mask(
        tmp_path, "total--iliac_venous_L.nii.gz", _recovery_nifti(), course.series_uid
    )
    other = np.zeros((4, 12, 12), dtype=np.uint8)
    other[1:3, 2:5, 7:10] = 1
    _write_totalseg_mask(
        tmp_path,
        "synth--total--iliac_venous_L.nii.gz",
        other,
        course.series_uid,
        run="ts_run_legacy",
    )
    _assert_recovery_refused(course, tmp_path)


def test_mask_outside_the_planning_ct_physical_space_never_recovers(tmp_path, monkeypatch):
    """Provenance is not geometry. This mask is 4x coarser in-plane, so it covers
    a 44mm field where the planning CT covers 11mm. Resampling it onto the CT
    grid still yields a NON-EMPTY mask -- a fabricated structure at the wrong
    scale -- so an emptiness check alone would not catch it."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    coarse = np.zeros((4, 12, 12), dtype=np.uint8)
    coarse[1:3, 0:2, 0:2] = 1
    _write_totalseg_mask(
        tmp_path,
        "total--iliac_venous_L.nii.gz",
        coarse,
        course.series_uid,
        spacing=(4.0, 4.0, 1.0),
    )
    _assert_recovery_refused(course, tmp_path)


def test_auto_rtstruct_bound_to_another_series_never_recovers(tmp_path, monkeypatch):
    """The recovery is only defensible for an automatic RTSTRUCT this pipeline
    published against the contracted planning CT. An RS_auto that references a
    different planning image series has unverified provenance for its ROIs."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    _write_totalseg_mask(
        tmp_path, "total--iliac_venous_L.nii.gz", _recovery_nifti(), course.series_uid
    )
    dataset = pydicom.dcmread(str(course.rs_auto))
    referenced = (
        dataset.ReferencedFrameOfReferenceSequence[0]
        .RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0]
    )
    referenced.SeriesInstanceUID = generate_uid()
    dataset.save_as(str(course.rs_auto))
    _assert_recovery_refused(course, tmp_path)


def test_a_base_contour_is_never_displaced_by_a_mask(tmp_path, monkeypatch):
    """The authoritative RTSTRUCT already carries ``iliac_venous_L``. RS_auto's
    copy of that ROI is D16 and a provenanced mask exists, but the harvested
    clinical contour is the authority: the union must use it, and no recovery
    may be claimed in the provenance record."""
    ct = tmp_path / "ct"
    ct.mkdir()
    rt_manual = _build_real_rtstruct(ct, n_slices=4, side=12)
    manual_venous = np.zeros((12, 12, 4), dtype=bool); manual_venous[6:9, 3:6, 1:3] = True
    rt_manual.add_roi(mask=manual_venous, name="iliac_venous_L")
    rs_manual = tmp_path / "RS_manual.dcm"
    rt_manual.save(str(rs_manual))
    series_uid = str(rt_manual.series_data[0].SeriesInstanceUID)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)

    rt_auto = RTStructBuilder.create_new(dicom_series_path=str(ct))
    clean = np.zeros((12, 12, 4), dtype=bool); clean[2:5, 2:5, 1:3] = True
    auto_venous = np.zeros((12, 12, 4), dtype=bool); auto_venous[7:10, 2:10, 1:3] = True
    rt_auto.add_roi(mask=clean, name="iliac_artery_L")
    rt_auto.add_roi(mask=auto_venous, name="iliac_venous_L")
    _make_partially_unparseable(rt_auto, "iliac_venous_L")
    rs_auto = tmp_path / "RS_auto.dcm"
    rt_auto.ds.save_as(str(rs_auto))
    _write_totalseg_mask(
        tmp_path, "total--iliac_venous_L.nii.gz", _recovery_nifti(), series_uid
    )

    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: iliac_vess\n"
        "    source_structures: [iliac_artery_L, iliac_venous_L]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, config, rs_manual, rs_auto
    )
    meta = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    assert meta["custom_structure_outcomes"]["iliac_vess"]["status"] == "generated"
    assert _recorded_fallbacks(meta) == {}
    got = RTStructBuilder.create_from(str(ct), str(out)).get_roi_mask_by_name("iliac_vess")
    assert np.array_equal(got, clean | manual_venous)


# ===========================================================================
# Source-identity reproductions
#
# The recovery above matches a mask by (manifest planning-CT series, ROI name
# derived with ``_pretty_roi_name``, shared physical space). Inspection of the
# two producers shows that this is not source identity:
#
#   * ``segmentation.py`` writes ``manifest.json`` with ``source_nifti``,
#     ``source_series_instance_uid``/``planning_ct_series_instance_uid``,
#     ``source_nifti_sha256``, ``source_ct_sop_hash``, ``generated_at`` and
#     ``models[].masks`` -- a list of *file names*. No per-mask content hash is
#     recorded anywhere, so no manifest can certify which bytes a named mask
#     file held when RS_auto was published.
#   * ``auto_rtstruct.build_auto_rtstruct`` publishes RS_auto's ROIs from the
#     multilabel labelmap (``_load_seg_dicom`` on ``*--total.dcm`` or
#     ``_load_seg_nifti`` on ``*_total_multilabel.nii.gz``) with names taken from
#     ``*_total_segmentations.json``. The per-ROI ``total--<roi>.nii.gz`` files
#     the index above walks are only the *fallback* source, used when the
#     labelmap contributed no ROI. Names additionally pass through
#     ``_unique_roi_name``, which renames a collision to ``<name>_dup``.
#   * ``_record_auto_resume_decision`` records one line for RS_auto as a whole:
#     action, reason, artefact and the planning CT series UID. Nothing binds an
#     ROI to a mask file, a run directory, or a byte range.
#
# So a hit proves only that *some* mask currently on disk carries that name on
# the same planning CT. The tests below reproduce the resulting failures.
# ===========================================================================


def _recorded_fallbacks(meta):
    """Mask substitutions the publication claims, tolerating the key's removal."""
    return meta.get("totalseg_fallback_sources") or {}


def _break_reader(monkeypatch, roi_name, *, exception=None):
    """Make the scoped reader fail for one ROI the way a rasteriser can.

    ``exception=None`` models a rasterisation that returns an all-false mask (a
    valid, fully bound contour that produced no voxels); an exception models a
    technical reader failure. Neither is a ``ROIContourDisposition``, so neither
    is a statement about the ROI's geometry -- both are failures to read it.
    """
    from rtpipeline import rtstruct_geometry

    original = rtstruct_geometry.ScopedRTStruct.get_roi_mask_by_name

    def patched(self, name):
        if name != roi_name:
            return original(self, name)
        if exception is not None:
            raise exception
        return np.zeros_like(original(self, name))

    monkeypatch.setattr(
        rtstruct_geometry.ScopedRTStruct, "get_roi_mask_by_name", patched
    )


def _manual_only_course(tmp_path, monkeypatch):
    """base = RS_manual carrying PTV; ``ptv_union`` is configured over PTV alone.

    A fully provenanced ``total--PTV.nii.gz`` covering a different volume sits in
    the segmentation directory, so any substitution is visible in the published
    mask as well as in the outcome.
    """
    ct, rs_manual, series_uid = _fixture(tmp_path)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)
    automatic = np.zeros((4, 12, 12), dtype=np.uint8)
    automatic[1:3, 1:4, 1:4] = 1
    _write_totalseg_mask(tmp_path, "total--PTV.nii.gz", automatic, series_uid)
    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: ptv_union\n"
        "    source_structures: [PTV]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        ct=ct, rs_manual=rs_manual, series_uid=series_uid, config=config,
        automatic=np.moveaxis(automatic.astype(bool), 0, -1),
    )


def _assert_manual_ptv_not_substituted(course, tmp_path):
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, course.config, course.rs_manual
    )
    assert out is not None and out.exists()
    meta = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    outcome = meta["custom_structure_outcomes"]["ptv_union"]
    assert outcome["status"] == "source_unavailable", (
        "a manual ROI that could not be read was filled in from an automatic "
        f"segmentation mask: {outcome}"
    )
    assert _recorded_fallbacks(meta) == {}
    published = pydicom.dcmread(str(out))
    names = [str(r.ROIName) for r in published.StructureSetROISequence]
    assert "ptv_union" not in names and "ptv_union__partial" not in names


def test_a_failed_manual_rasterisation_is_never_filled_from_a_mask(tmp_path, monkeypatch):
    """The base harvest is the *authoritative* RTSTRUCT. When its rasterisation
    yields nothing, the ROI is unread -- not empty and not automatic. Replacing
    it with a same-name TotalSegmentator mask silently republishes an automatic
    volume under a clinician's ROI name."""
    course = _manual_only_course(tmp_path, monkeypatch)
    _break_reader(monkeypatch, "PTV")
    _assert_manual_ptv_not_substituted(course, tmp_path)


def test_a_reader_exception_never_becomes_a_successful_automatic_mask(tmp_path, monkeypatch):
    """A technical read failure must stay a failure. Converting it into a
    same-name automatic mask turns an error into a measurement."""
    course = _manual_only_course(tmp_path, monkeypatch)
    _break_reader(
        monkeypatch, "PTV", exception=RuntimeError("rasteriser failed on PTV")
    )
    _assert_manual_ptv_not_substituted(course, tmp_path)


def test_a_same_name_mask_that_contradicts_the_roi_is_not_its_source(tmp_path, monkeypatch):
    """Same planning CT series, same derived name, same physical space -- and a
    completely different structure. The D16 ROI's own surviving contours cover
    rows 7-10; this mask covers rows 1-4 and is disjoint from them. Nothing in
    the manifest or in RS_auto distinguishes it from the mask the ROI was
    actually published from, because neither producer records that link."""
    course = _d16_auto_course(tmp_path, monkeypatch)
    contradicting = np.zeros((4, 12, 12), dtype=np.uint8)
    contradicting[1:3, 1:4, 1:4] = 1
    assert not (
        np.moveaxis(contradicting.astype(bool), 0, -1)
        & np.moveaxis(_recovery_nifti().astype(bool), 0, -1)
    ).any()
    _write_totalseg_mask(
        tmp_path, "total--iliac_venous_L.nii.gz", contradicting, course.series_uid
    )
    _assert_recovery_refused(course, tmp_path)


def _publish_clean_course(tmp_path, monkeypatch):
    """A course with no disposition at all, published once."""
    ct, rs_manual, series_uid = _fixture(tmp_path)
    _patch_contract(tmp_path, ct, rs_manual, series_uid, monkeypatch)
    rt_auto = RTStructBuilder.create_new(dicom_series_path=str(ct))
    clean = np.zeros((12, 12, 4), dtype=bool); clean[2:5, 2:5, 1:3] = True
    rt_auto.add_roi(mask=clean, name="iliac_artery_L")
    rs_auto = tmp_path / "RS_auto.dcm"
    rt_auto.ds.save_as(str(rs_auto))
    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: iliac_vess\n"
        "    source_structures: [iliac_artery_L]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, config, rs_manual, rs_auto
    )
    assert out is not None and out.exists()
    return SimpleNamespace(
        ct=ct, rs_manual=rs_manual, rs_auto=rs_auto, config=config,
        out=out, clean=clean, series_uid=series_uid,
    )


def test_clean_auto_integration_is_fully_harvested(tmp_path, monkeypatch):
    """Positive control: with no disposition anywhere, the auto integration is
    complete and the configured union is generated from the contour itself."""
    course = _publish_clean_course(tmp_path, monkeypatch)
    meta = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    outcome = meta["custom_structure_outcomes"]["iliac_vess"]
    assert outcome["status"] == "generated"
    assert outcome["available_sources"] == ["iliac_artery_L"]
    assert _recorded_fallbacks(meta) == {}
    got = RTStructBuilder.create_from(str(course.ct), str(course.out)).get_roi_mask_by_name(
        "iliac_vess"
    )
    assert np.array_equal(got, course.clean)


def test_rs_custom_published_by_mask_substitution_is_never_reused(tmp_path, monkeypatch):
    """A publication whose own metadata says an ROI came from a segmentation
    mask instead of a contour cannot be reused: that provenance was never
    verifiable, and the recorded fields are not re-checked by any reuse path.
    The mtimes are restored so only the recorded claim can reject it."""
    course = _publish_clean_course(tmp_path, monkeypatch)
    assert custom._is_rs_custom_stale(
        course.out, course.config, course.rs_manual, course.rs_auto
    ) is False, "positive control: the clean publication must be reusable"

    meta_path = tmp_path / "metadata" / "rs_custom_meta.json"
    before = meta_path.stat()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["totalseg_fallback_sources"] = {
        "iliac_venous_L": {
            "label": "auto",
            "code": "ROI_CONTOUR_PARTIALLY_UNPARSEABLE",
            "mask": "total--iliac_venous_L.nii.gz",
            "segmentation_run": "ts_run",
            "mask_sha256": "0" * 64,
            "source_series_instance_uid": course.series_uid,
        }
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.utime(meta_path, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert custom._is_rs_custom_stale(
        course.out, course.config, course.rs_manual, course.rs_auto
    ) is True
