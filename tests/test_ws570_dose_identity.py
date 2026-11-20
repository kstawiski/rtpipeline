"""RTP305 dose-chain and ROI-identity regressions on synthetic inputs."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import generate_uid

import dvh_plan_target_fixtures as fx
from course_contract_test_utils import write_minimal_course_contract
from rtpipeline import dvh, dvh_generation, organize
from rtpipeline.course_contract import CourseContractError, load_course_contract
from rtpipeline.dvh_aggregate import build_dvh_aggregate
from rtpipeline.reexport_original import reexport_original
from test_aggregate_course_contract import _aggregate_functions
from test_composite_course_dose import edit_contract
from test_roi_identity import source


ROI = ("Organ", "ORGAN", fx.DOSED_LEFT_X)
RP = "DICOM/RTPLAN/plan.dcm"
RD = "DICOM/RTDOSE/dose.dcm"
RS = "DICOM/RTSTRUCT/rs.dcm"


def _course(tmp_path, **kwargs):
    return fx.build_course(tmp_path, rois=[ROI], **kwargs)


def _table(course):
    assert dvh.dvh_for_course(course, parallel_workers=1)
    return pd.read_parquet(course / "dvh_metrics.parquet")


def _dose_values(dose, values, scaling=None):
    if scaling is not None:
        dose.DoseGridScaling = scaling
    stored = np.broadcast_to(values, dose.pixel_array.shape) / float(dose.DoseGridScaling)
    dose.PixelData = np.rint(stored).astype("<u4").tobytes()


def _root_artifacts(course):
    for source_path, name in ((RP, "RP.dcm"), (RD, "RD.dcm"), (RS, "RS.dcm")):
        shutil.copyfile(course / source_path, course / name)
    def use_root(contract):
        for key, name in (("plan_artifact", "RP.dcm"), ("dose_grid", "RD.dcm"),
                          ("authoritative_rtstruct", "RS.dcm")):
            contract[key]["path"] = name
    edit_contract(course, use_root)


@pytest.mark.parametrize("artifact", ["RP.dcm", "RD.dcm", "RS.dcm", "DICOM_related/RTRECORD/record_1.dcm"])
def test_c1_root_artifacts_and_record_bytes_bind_resume(tmp_path, artifact):
    course = _course(tmp_path)
    _root_artifacts(course)
    original = _table(course).iloc[0].DmeanGy
    before = dvh_generation.input_manifest(course)
    path = course / artifact
    if artifact == "DICOM_related/RTRECORD/record_1.dcm":
        path.unlink()
    else:
        dataset = pydicom.dcmread(path)
        if artifact == "RD.dcm":
            _dose_values(dataset, dataset.pixel_array * float(dataset.DoseGridScaling) / 2)
        else:
            dataset.SeriesDescription = "Synthetic bytes changed"
        dataset.save_as(path, enforce_file_format=True)
    assert dvh_generation.input_manifest(course) != before
    if artifact == "RD.dcm":
        old = dvh_generation.current(course)["generation_id"]
        assert _table(course).iloc[0].DmeanGy == pytest.approx((original - .005) / 2 + .005)
        assert dvh_generation.current(course)["generation_id"] != old
    elif artifact.startswith("DICOM_related"):
        with pytest.raises(CourseContractError):
            dvh.dvh_for_course(course, parallel_workers=1)


def test_c2_exact_hot_volume_uses_hot_end_of_plateau():
    samples = np.r_[np.full(50, 10.), np.full(50, 50.)]
    metrics = dvh._compute_metrics_from_arrays(samples, .04, 50.)
    assert metrics["D2ccGy"] == 50.
    assert metrics["D50Gy"] == 50.


def _set_box(rs, index, x, y, zs):
    items = []
    for z in zs:
        contour = Dataset()
        contour.ContourGeometricType = "CLOSED_PLANAR"
        contour.NumberOfContourPoints = 4
        contour.ContourData = [x[0], y[0], z, x[1], y[0], z,
                               x[1], y[1], z, x[0], y[1], z]
        items.append(contour)
    rs.ROIContourSequence[index].ContourSequence = Sequence(items)


def test_c3_axial_contour_planes_have_local_slab_widths(tmp_path):
    course = _course(tmp_path, delivered=False)
    rs = pydicom.dcmread(course / RS)
    _set_box(rs, 0, (1.1, 21.1), (1.1, 21.1), (2, 4, 10))
    rs.save_as(course / RS, enforce_file_format=True)
    dose = pydicom.dcmread(course / RD)
    dose.NumberOfFrames = 6
    dose.GridFrameOffsetVector = list(range(0, 12, 2))
    levels = np.full((6, int(dose.Rows), int(dose.Columns)), 10.)
    levels[2] = 20.
    levels[5] = 30.
    dose.PixelData = np.rint(levels / float(dose.DoseGridScaling)).astype("<u4").tobytes()
    dose.save_as(course / RD, enforce_file_format=True)
    row = _table(course).iloc[0]
    assert row["Volume (cm³)"] == pytest.approx(4.8)
    assert row.DmeanGy == pytest.approx((2*10 + 4*20 + 6*30) / 12 + .005)


@pytest.mark.parametrize("missing", [False, True])
def test_c4_roi_frame_matches_dose_individually(tmp_path, missing):
    course = fx.build_course(tmp_path, rois=[ROI, ("Other", "ORGAN", fx.DOSED_LEFT_X)])
    rs = pydicom.dcmread(course / RS)
    other_frame = Dataset()
    other_frame.FrameOfReferenceUID = generate_uid()
    rs.ReferencedFrameOfReferenceSequence.append(other_frame)
    other_roi = rs.StructureSetROISequence[1]
    if missing:
        del other_roi.ReferencedFrameOfReferenceUID
    else:
        other_roi.ReferencedFrameOfReferenceUID = other_frame.FrameOfReferenceUID
    rs.save_as(course / RS, enforce_file_format=True)
    rows = _table(course).set_index("ROI_Name")
    assert np.isfinite(rows.loc["Organ", "DmeanGy"])
    assert pd.isna(rows.loc["Other", "DmeanGy"])
    assert rows.loc["Other", "row_status"] == "not_computed"
    assert rows.loc["Other", "dose_metric_status"] == (
        "roi_frame_of_reference_unresolved" if missing else "roi_frame_of_reference_mismatch"
    )


def _aggregate_row(source_label, uid, geometry, **overrides):
    return dict({"patient_id": "SYNTH_P", "course_id": "SYNTH_C", "ROI_Number": 1,
                 "ROI_Name": "Organ", "ROI_OriginalName": "Organ", "DmeanGy": 10.,
                 "Segmentation_Source": source_label, "rtstruct_sop_instance_uid": uid,
                 "roi_geometry_sha256": geometry}, **overrides)


def test_c5_equal_measurements_only_collapse_with_matching_roi_binding(tmp_path):
    rows = [
        _aggregate_row("AutoRTS", "1.2.3", "a"*64),
        _aggregate_row("Merged", "1.2.3", "a"*64),
        _aggregate_row("Custom", "1.2.4", "a"*64),
        _aggregate_row("Manual", "1.2.3", "b"*64),
    ]
    result = build_dvh_aggregate([pd.DataFrame(rows)], [("SYNTH_P", "SYNTH_C", tmp_path)])
    assert set(result.Segmentation_Source) == {"Merged", "Custom", "Manual"}


def _nifti_probe(tmp_path, monkeypatch, offsets, dose_scaling, pixel_factory, ct_origin):
    from synthetic_ct_series import write_ct_series

    course = _course(tmp_path, delivered=False)
    ct_dir = course / "DICOM/CT"
    for path in ct_dir.glob("*.dcm"):
        path.unlink()
    xs, ys, start_z = ct_origin
    ct_paths = write_ct_series(ct_dir, [start_z + n for n in range(3)], signed=False, rows=8, columns=8)
    for n, path in enumerate(ct_paths):
        image = pydicom.dcmread(path)
        image.ImagePositionPatient = [xs, ys, start_z + n]
        image.PixelSpacing = [1., 1.]
        image.SliceThickness = 1.
        image.save_as(path, enforce_file_format=True)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in ct_paths])
    reference = reader.Execute()
    mask = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(reference)
    mask[0, 0, 0] = 1
    mask_path = course / "Segmentation_TotalSegmentator/probe.nii.gz"
    mask_path.parent.mkdir()
    sitk.WriteImage(mask, str(mask_path))

    dose = pydicom.dcmread(course / RD)
    dose.NumberOfFrames = len(offsets)
    dose.GridFrameOffsetVector = offsets
    dose.FrameIncrementPointer = Tag(0x3004, 0x000C)
    dose.DoseGridScaling = dose_scaling
    shape = (len(offsets), int(dose.Rows), int(dose.Columns))
    dose.PixelData = np.broadcast_to(pixel_factory(shape), shape).astype("<u4").tobytes()
    dose.save_as(course / RD, enforce_file_format=True)
    monkeypatch.setattr(dvh, "load_course_contract", lambda _: SimpleNamespace(planning_ct_dir=ct_dir))
    rows = dvh._compute_nifti_based_dvh(course, None, dose, course / RD, None, set())
    assert len(rows) == 1 and rows[0]["dose_grid_coverage_status"] == "fully_covered"
    return rows[0]["DmeanGy"]


def test_c6_nifti_uses_declared_nonuniform_frame_offsets(tmp_path, monkeypatch):
    offsets = [*range(10), 20]
    value = _nifti_probe(tmp_path, monkeypatch, offsets, 1.,
                         lambda shape: np.array(offsets)[:, None, None], (2., 2., 10.))
    assert value == pytest.approx(10.)

    from rtpipeline.dvh_rotation import PatientDoseGrid
    grid = PatientDoseGrid(pydicom.dcmread(tmp_path / "SYNTH_P" / "SYNTH_C" / RD))
    samples = grid.sample(np.array([[-1., 2., 10.], [39., 2., 10.]]), half_voxel_edges=True)
    assert np.ma.getmaskarray(samples).tolist() == [False, True]


def test_a2_nifti_interpolates_stored_integers_in_float(tmp_path, monkeypatch):
    value = _nifti_probe(tmp_path, monkeypatch, [0, 2, 4], .5,
                         lambda shape: 100 + np.arange(shape[2]), (1., 2., 2.))
    assert value == pytest.approx(50.25)


@pytest.mark.parametrize("one_plane", [False, True])
def test_c7_a5_covered_zero_volume_and_roi_zero_get_explicit_rows(tmp_path, one_plane):
    course = _course(tmp_path, delivered=False)
    rs = pydicom.dcmread(course / RS)
    rs.StructureSetROISequence[0].ROINumber = 0
    rs.ROIContourSequence[0].ReferencedROINumber = 0
    rs.RTROIObservationsSequence[0].ReferencedROINumber = 0
    if one_plane:
        _set_box(rs, 0, (2.1, 15.9), (6.1, 17.9), (2.,))
    else:
        _set_box(rs, 0, (6.1, 6.9), (8.1, 8.9), (2., 4., 6.))
    rs.save_as(course / RS, enforce_file_format=True)
    row = _table(course).iloc[0]
    assert row.ROI_Number == 0 and row.row_status == "not_computed"
    assert row.dvh_nonmeasurement_class == (
        "insufficient_contour_planes_for_volume" if one_plane else "no_dose_grid_point_sampled"
    )
    assert pd.isna(row.DmeanGy)


def test_c8_axial_and_array_vx_use_inclusive_sample_thresholds(tmp_path):
    course = _course(tmp_path, delivered=False)
    dose = pydicom.dcmread(course / RD)
    _dose_values(dose, 9.999, scaling=.001)
    dose.save_as(course / RD, enforce_file_format=True)
    array_metrics = dvh._compute_metrics_from_arrays(np.full(100, 9.999), .04, 10., 10.)
    axial = _table(course).iloc[0]
    assert axial["V10Gy (%)"] == array_metrics["V10Gy (%)"] == 0.
    assert array_metrics["V100%Rx (%)"] == 0.
    assert dvh._compute_metrics_from_arrays(np.array([10.]), 1., 10.)["V10Gy (%)"] == 100.


@pytest.mark.parametrize("summation", ["PLAN", "PLAN_SUM", "MULTI_PLAN"])
def test_c9_tied_candidate_grids_refuse_in_either_filename_order(tmp_path, summation):
    course = _course(tmp_path, delivered=False)
    dose = pydicom.dcmread(course / RD)
    dose.DoseSummationType = summation
    first, second = tmp_path / "a.dcm", tmp_path / "z.dcm"
    dose.save_as(first, enforce_file_format=True)
    dose.SOPInstanceUID = dose.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    _dose_values(dose, 25.)
    dose.save_as(second, enforce_file_format=True)
    for paths in ([first, second], [second, first]):
        resolution = organize._classify_approved_doses([course / RP], paths)
        assert resolution.classification == "ambiguous_dose_grid"
        assert not resolution.selected_doses


def test_c10_duplicate_fraction_group_beam_exports_are_refused(tmp_path):
    course = _course(tmp_path, delivered=False)
    dose = pydicom.dcmread(course / RD)
    dose.DoseSummationType = "BEAM"
    group, beam = Dataset(), Dataset()
    group.ReferencedFractionGroupNumber = 1
    beam.ReferencedBeamNumber = 1
    group.ReferencedBeamSequence = Sequence([beam])
    dose.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence = Sequence([group])
    first, second = tmp_path / "a.dcm", tmp_path / "b.dcm"
    dose.save_as(first, enforce_file_format=True)
    dose.SOPInstanceUID = dose.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dose.save_as(second, enforce_file_format=True)
    resolution = organize._classify_approved_doses([course / RP], [first, second])
    assert resolution.classification == "duplicate_beam_dose_coverage"
    assert not resolution.selected_doses and not resolution.should_sum
    plan = pydicom.dcmread(course / RP)
    with pytest.raises(ValueError, match="duplicate_beam_dose_coverage"):
        organize._sum_doses_with_resample([first, second], plan, [plan])


def test_c11_composite_gap_is_outside_dose_support(tmp_path):
    from test_composite_course_dose import make_course
    from rtpipeline.repair_composite_course_dose import repair_course

    course = make_course(tmp_path, shift_mm=70.)
    rs = pydicom.dcmread(course / RS)
    _set_box(rs, 0, (44., 58.), (6., 20.), (2., 4., 6.))
    rs.ROIContourSequence = rs.ROIContourSequence[:1]
    rs.StructureSetROISequence = rs.StructureSetROISequence[:1]
    rs.RTROIObservationsSequence = rs.RTROIObservationsSequence[:1]
    rs.save_as(course / RS, enforce_file_format=True)
    assert repair_course(course, apply=True) == "applied"
    row = _table(course).iloc[0]
    assert row.dose_grid_coverage_status == "outside_grid"
    assert pd.isna(row.DmeanGy)
    assert row.row_status == "not_computed"


@pytest.mark.parametrize("attribute,value", [("DoseUnits", "RELATIVE"), ("DoseType", "EFFECTIVE")])
def test_c12_authoritative_root_rtdose_requires_physical_gy(tmp_path, attribute, value):
    course = _course(tmp_path)
    _root_artifacts(course)
    path = course / "RD.dcm"
    dose = pydicom.dcmread(path)
    setattr(dose, attribute, value)
    dose.save_as(path, enforce_file_format=True)
    with pytest.raises(CourseContractError, match="RTDOSE_NOT_PHYSICAL_GY"):
        load_course_contract(course)
    with pytest.raises(CourseContractError, match="RTDOSE_NOT_PHYSICAL_GY"):
        dvh.dvh_for_course(course, parallel_workers=1)


def test_c13_reexport_refuses_another_contracted_course(tmp_path, source):
    course = _course(tmp_path / "other", delivered=False)
    ct, rs, ref, *_ = source
    with pytest.raises(ValueError, match="REEXPORT_COURSE_IDENTITY_MISMATCH"):
        reexport_original(rtstruct=rs, ct_series=ct, reference_nifti=ref, output_root=course)
    assert not (course / "Segmentation_Original").exists()


def test_c13_explicit_legacy_sidecar_cannot_override_a_course_contract(tmp_path, source):
    from rtpipeline.original_segmentation import require_current_original_export, require_current_original_mask

    ct, rs, ref, *_ = source
    destination = tmp_path / "other" / "SYNTH_P" / "SYNTH_C"
    reexport_original(rtstruct=rs, ct_series=ct, reference_nifti=ref, output_root=destination)
    legacy = destination / "Segmentation_Original/ct"
    mask = next(legacy.glob("*.nii.gz"))
    require_current_original_mask(mask)
    assert _course(tmp_path / "other", delivered=False) == destination
    with pytest.raises(ValueError, match="source inputs changed"):
        require_current_original_export(legacy)
    with pytest.raises(ValueError, match="source inputs changed"):
        require_current_original_mask(mask)


def test_c14_refused_only_table_is_a_nonmeasurement_in_receipt_qc_and_aggregate(tmp_path):
    from test_dvh_rotation import analytic_dose

    course = _course(tmp_path, delivered=False)
    analytic_dose(course / RD, 5.)
    table = _table(course)
    receipt = json.loads((course / ".dvh_done").read_text())
    assert dvh_generation.validate_receipt(course, receipt)["measurement_outcome"] == "not_measured"
    qc = json.loads((course / "metadata/dvh_qc.json").read_text())
    assert qc["status"] == "degraded"
    assert table.row_status.tolist() == ["not_computed"]
    legacy = table.drop(columns="row_status").assign(patient_id="SYNTH_P", course_id="SYNTH_C")
    result = build_dvh_aggregate([legacy], [("SYNTH_P", "SYNTH_C", course)])
    assert result.row_status.tolist() == ["not_computed"]

    from rtpipeline import cli
    from rtpipeline.config import DEFAULT_MAX_TOTAL_DOSE_GY
    task = cli._DVHTask(
        course=SimpleNamespace(dirs=SimpleNamespace(root=course)),
        custom_structures=None, use_cropped=True, parallel_workers=1,
        max_total_dose_gy=DEFAULT_MAX_TOTAL_DOSE_GY,
    )
    outcome = cli._execute_dvh_task(task)
    assert outcome.status == "not_computed"
    assert outcome.output_path == course / "dvh_metrics.xlsx"


def test_c15_metrics_use_the_nominal_total_shown_in_the_row(tmp_path):
    course = _course(tmp_path, delivered=False)
    plan = pydicom.dcmread(course / RP)
    plan.DoseReferenceSequence[0].TargetPrescriptionDose = 50.
    plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose = 9.6
    plan.save_as(course / RP, enforce_file_format=True)
    dose = pydicom.dcmread(course / RD)
    _dose_values(dose, 49.)
    dose.save_as(course / RD, enforce_file_format=True)
    write_minimal_course_contract(course, selected_plans=[course / RP], selected_doses=[course / RD],
                                  authoritative_rtstruct=course / RS)
    fx.mark_fully_delivered(course)
    edit_contract(course, lambda contract: contract["delivery"].update(delivered_dose_gy=48.))
    row = _table(course).iloc[0]
    assert row.Prescribed_Dose_Gy == 50.
    assert row["V100%Rx (%)"] == 0.
    assert row.Prescription_Normalization_Gy == 50.


def test_c15_multiple_unbound_site_references_refuse_in_both_orders(tmp_path):
    from rtpipeline.prescription import resolve_plan_prescriptions

    plan = pydicom.dcmread(fx.write_plan(tmp_path / "plan.dcm", [
        fx.dose_reference(number=1, prescription_gy=20.),
        fx.dose_reference(number=2, prescription_gy=30.),
    ]))
    del plan.FractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedDoseReferenceSequence
    for refs in (list(plan.DoseReferenceSequence), list(reversed(plan.DoseReferenceSequence))):
        plan.DoseReferenceSequence = Sequence(refs)
        group, = resolve_plan_prescriptions(plan)
        assert group["resolved_prescribed_dose_total_gy"] is None


def test_a1_manual_precedence_is_local_to_patient_and_course(tmp_path):
    entries = [
        _aggregate_row("Manual", "1.2.3", "a"*64),
        _aggregate_row("Merged", "1.2.4", "b"*64),
        _aggregate_row("Merged", "1.2.5", "c"*64, course_id="SECOND"),
        _aggregate_row("Merged", "1.2.6", "d"*64, patient_id="OTHER"),
    ]
    captured = []
    namespace = _aggregate_functions()
    namespace["snakemake"] = SimpleNamespace(output=SimpleNamespace(dvh=tmp_path / "dvh.xlsx"))
    namespace["write_dvh_aggregate"] = lambda data, _: captured.append(data)
    keys = {(row["patient_id"], row["course_id"]) for row in entries}
    namespace["_write_dvh"]([pd.DataFrame(entries)], [(p, c, tmp_path) for p, c in keys])
    out, = captured
    assert set(zip(out.patient_id, out.course_id, out.Segmentation_Source)) == {
        ("SYNTH_P", "SYNTH_C", "Manual"), ("SYNTH_P", "SECOND", "Merged"), ("OTHER", "SYNTH_C", "Merged")
    }


def test_a3_axial_subcentigray_uses_unbinned_endpoints_and_hi(tmp_path):
    course = _course(tmp_path, delivered=False)
    dose = pydicom.dcmread(course / RD)
    _dose_values(dose, .004, scaling=.000001)
    rs = pydicom.dcmread(course / RS)
    _set_box(rs, 0, (1.1, 21.1), (1.1, 21.1), (0, 2, 4, 6, 8))
    histogram = dvh.dvhcalc.get_dvh(rs, dose, 1)
    values = dvh._compute_metrics(histogram, .004, lambda: dvh._rtstruct_dose_summary(rs, dose, 1))
    for key in ("DminGy", "D98Gy", "D95Gy", "D50Gy", "D2Gy", "D2ccGy", "D1ccGy", "D0.1ccGy", "D0.03ccGy"):
        assert values[key] == pytest.approx(.004)
    for key in ("Dmin%", "D98%", "D95%", "D50%", "D2%", "V100%Rx (%)"):
        assert values[key] == pytest.approx(100.)
    assert values["HI"] == 0.


@pytest.mark.parametrize("rotated", [False, True])
def test_a4_partial_grid_reports_geometry_and_covered_volume_separately(tmp_path, rotated):
    from test_dvh_rotation import analytic_dose

    course = _course(tmp_path, delivered=False)
    if rotated:
        analytic_dose(course / RD, .5)
    rs = pydicom.dcmread(course / RS)
    x = (-25., 10.) if rotated else (-15., 20.)
    _set_box(rs, 0, x, (2., 12.), (2., 4., 6.))
    rs.save_as(course / RS, enforce_file_format=True)
    row = _table(course).iloc[0]
    assert row.dose_grid_coverage_status == "partial_grid"
    assert row["Volume (cm³)"] == pytest.approx(35 * 10 * 4 / 1000)
    assert row.dose_grid_roi_volume_cm3 == row["Volume (cm³)"]
    assert row.dose_grid_covered_volume_cm3 == row["covered_Volume (cm³)"]
    assert pd.isna(row.DmeanGy) and np.isfinite(row.covered_DmeanGy)


def test_a6_name_only_readers_refuse_normalized_collisions(source):
    from test_roi_identity import builder, names
    from rtpipeline.rtstruct_geometry import rtstruct_roi_mask

    names(source[1], ["Bladder", " bladder "])
    reader = builder(source)
    with pytest.raises(ValueError, match="RTSTRUCT_AMBIGUOUS_ROI_NAME"):
        rtstruct_roi_mask(reader, "Bladder")
    assert np.array_equal(rtstruct_roi_mask(reader, roi_number=1), source[3])
    assert np.array_equal(rtstruct_roi_mask(reader, roi_number=2), source[4])


def test_a6_rs_custom_keeps_independent_dependencies_when_an_roi_name_collides(tmp_path, monkeypatch):
    from rt_utils import RTStructBuilder
    from rtpipeline import custom_structures_rtstruct as custom
    from test_custom_structures_d16_auto_harvest import _manual_only_course

    course = _manual_only_course(tmp_path, monkeypatch)
    builder = RTStructBuilder.create_from(str(course.ct), str(course.rs_manual))
    builder.add_roi(mask=course.automatic, name="PTV")
    separate = np.zeros_like(course.automatic)
    separate[7:10, 7:10, 1:3] = True
    builder.add_roi(mask=separate, name="Distinct")
    builder.save(str(course.rs_manual))
    course.config.write_text(
        "custom_structures:\n"
        "  - {name: blocked, source_structures: [PTV], operation: union}\n"
        "  - {name: independent, source_structures: [Distinct], operation: union}\n"
    )
    path = custom._create_custom_structures_rtstruct_unlocked(tmp_path, course.config, course.rs_manual, None)
    assert path and path.is_file()
    metadata = json.loads((tmp_path / "metadata/rs_custom_meta.json").read_text())
    assert metadata["custom_structure_outcomes"]["blocked"]["reason_code"] == "CUSTOM_SOURCE_AMBIGUOUS_NAME"
    assert metadata["custom_structure_outcomes"]["independent"]["status"] == "generated"
