"""Public synthetic controls for the frozen treatment-region rules."""

from __future__ import annotations

import json
import math
import stat

import numpy as np
import pydicom
import pytest

from rtpipeline import treatment_scope_geometry as geometry
from rtpipeline.treatment_scope import (
    CRITERIA,
    geometric_classes,
    load_rules,
    main,
    names_only_class,
)


def organ(lo=-100, hi=100, volume=10):
    return {
        "bbox_lps_mm": [[0, 0, lo], [1, 1, hi]],
        "centroid_lps_mm": [0, 0, 0],
        "volume_ml": volume,
    }


def target(number=1, *, z=0, volume=10, overlap=10, organ_fraction=1, typ="PTV"):
    return {
        "roi_number": number,
        "type": typ,
        "n_voxels": 1000,
        "volume_ml": volume,
        "outside_ct_contour_points": 0,
        "unsupported_contours": 0,
        "bbox_lps_mm": [[0, 0, z], [1, 1, z]],
        "z_histogram_mm_counts": [[z, 1000]],
        "centroid_lps_mm": [0, 0, z],
        "organ_overlaps": {
            "urinary_bladder": {
                "target_fraction": overlap / volume,
                "organ_fraction": organ_fraction,
                "volume_ml": overlap,
            }
        },
    }


def evidence():
    return {
        "status": "measured",
        "errors": [],
        "targets": [target()],
        "organs": {
            "hip_left": organ(-20, -10),
            "hip_right": organ(-20, -10),
            "vertebrae_L5": organ(10, 20),
            "urinary_bladder": organ(-5, 5),
            "aorta": organ(),
        },
        "target_exclusions": [],
        "target_pairwise_containment": [],
    }


def rtstruct(rows):
    ds = pydicom.Dataset()
    ds.StructureSetROISequence, ds.RTROIObservationsSequence, ds.ROIContourSequence = (
        [],
        [],
        [],
    )
    for number, name, types, count in rows:
        roi = pydicom.Dataset()
        roi.ROINumber, roi.ROIName = number, name
        ds.StructureSetROISequence.append(roi)
        for typ in types:
            observation = pydicom.Dataset()
            observation.ReferencedROINumber, observation.RTROIInterpretedType = (
                number,
                typ,
            )
            ds.RTROIObservationsSequence.append(observation)
        contour = pydicom.Dataset()
        contour.ReferencedROINumber = number
        contour.ContourSequence = []
        for _ in range(count):
            loop = pydicom.Dataset()
            loop.ContourGeometricType, loop.NumberOfContourPoints = "CLOSED_PLANAR", 4
            loop.ContourData = [20, 20, 10, 40, 20, 10, 40, 40, 10, 20, 40, 10]
            contour.ContourSequence.append(loop)
        ds.ROIContourSequence.append(contour)
    return ds


@pytest.mark.parametrize(
    "criterion,field,threshold",
    [
        ("C1", "target_fraction", 0.1),
        ("C1", "volume_ml", 5),
        ("C2", "volume_ml", 5),
        ("C3", "organ_fraction", 0.5),
        ("C4", "organ_fraction", 0.9),
    ],
)
def test_inclusive_bladder_thresholds(criterion, field, threshold):
    rule = load_rules()["geometry"]
    value = target()
    value["organ_overlaps"]["urinary_bladder"][field] = threshold
    assert geometry.bladder_hit(value, criterion, rule)
    value["organ_overlaps"]["urinary_bladder"][field] = math.nextafter(
        threshold, -math.inf
    )
    assert not geometry.bladder_hit(value, criterion, rule)


@pytest.mark.parametrize(
    "targets,expected",
    [
        ([target()], "B"),
        ([target(overlap=0, organ_fraction=0)], "P"),
        ([target(z=60)], "X"),
        ([target(), target(2, z=60)], "MX"),
    ],
)
def test_consensus_classes(targets, expected):
    value = evidence()
    value["targets"] = targets
    assert geometric_classes(value)["consensus"] == expected


def test_b_to_mp_non_nesting_counterexample():
    value = evidence()
    value["organs"]["urinary_bladder"]["volume_ml"] = 1000
    ctv = target(1, typ="CTV", volume=1000, overlap=200, organ_fraction=0.2)
    bladder_gtv = target(2, typ="GTV", volume=100, overlap=8, organ_fraction=0.008)
    bladder_gtv["organ_overlaps"]["prostate"] = {"target_fraction": 0.02}
    other_gtv = target(3, typ="GTV", volume=100, overlap=1, organ_fraction=0.001)
    other_gtv["organ_overlaps"]["prostate"] = {"target_fraction": 0.40}
    other_gtv["centroid_lps_mm"] = [30, 0, 0]
    value["targets"] = [ctv, bladder_gtv, other_gtv]
    value["target_pairwise_containment"] = [
        {"a": a, "b": b, "overlap_over_smaller": 0} for a, b in [(1, 2), (1, 3), (2, 3)]
    ]
    result = geometric_classes(value)
    assert result["classes"] == {"C1": "B", "C2": "MP", "C3": "P", "C4": "P"}
    assert result["consensus"] == "U"
    assert result["u_reason"] == "cross_criterion_disagreement"
    assert result["raw_bladder_hits"][1] == {
        "roi_number": 2,
        "C1": False,
        "C2": True,
        "C3": False,
        "C4": False,
    }
    assert result["representative_overlap"] == {
        "target_volume_cm3": 1000,
        "overlap_cm3": 200,
        "overlap_pct_target": 20.0,
        "overlap_pct_bladder": 20.0,
        "bladder_mask_volume_cm3": 1000,
    }
    assert (
        result["reasons"]["C2"]["other_primary_geometry"][0]["centroid_separation_mm"]
        == 30
    )


@pytest.mark.parametrize(
    "distance,containment,typ,expected",
    [
        (20, 0.5, "GTV", "MP"),
        (19.999, 0.5, "GTV", "B"),
        (20, 0.500001, "GTV", "B"),
        (30, 0, "CTV", "B"),
    ],
)
def test_distinct_gross_targets(distance, containment, typ, expected):
    value = evidence()
    value["targets"][0]["type"] = "GTV"
    other = target(2, overlap=0, organ_fraction=0, typ=typ)
    other["organ_overlaps"]["prostate"] = {"target_fraction": 0.3}
    other["centroid_lps_mm"] = [distance, 0, 0]
    value["targets"].append(other)
    value["target_pairwise_containment"] = [
        {"a": 1, "b": 2, "overlap_over_smaller": containment}
    ]
    assert geometric_classes(value)["consensus"] == expected


@pytest.mark.parametrize(
    "change,reason",
    [
        ("outside", "target_outside_CT_or_unsupported_contour"),
        ("unsupported", "target_outside_CT_or_unsupported_contour"),
        ("contradictory", "contradictory_target_identification"),
        ("coverage", "automatic_organs_do_not_span_target_z"),
        (
            "missing_landmarks",
            "pelvic_landmarks_absent_and_no_complete_nonpelvic_proof",
        ),
        (
            "empty_bladder",
            "empty_automatic_bladder_mask_prevents_pelvic_subclassification",
        ),
        ("error", "synthetic_failure"),
    ],
)
def test_u_reasons(change, reason):
    value = evidence()
    if change == "outside":
        value["targets"][0]["outside_ct_contour_points"] = 1
    if change == "unsupported":
        value["targets"][0]["unsupported_contours"] = 1
    if change == "contradictory":
        value["target_exclusions"] = [{"reason": "contradictory_interpreted_types"}]
    if change == "coverage":
        value["targets"] = [target(z=104)]
    if change == "missing_landmarks":
        del value["organs"]["hip_left"]
    if change == "empty_bladder":
        del value["organs"]["urinary_bladder"]
    if change == "error":
        value["errors"] = ["synthetic_failure"]
    result = geometric_classes(value)
    assert result["consensus"] == "U"
    assert result["u_reason"] == reason


@pytest.mark.parametrize(
    "fraction,expected", [(0, "X"), (0.05, "X"), (0.5, "MX"), (0.95, "B"), (1, "B")]
)
def test_pelvic_fraction_boundaries(fraction, expected):
    value = evidence()
    value["targets"][0]["z_histogram_mm_counts"] = [
        [0, int(fraction * 1000)],
        [60, 1000 - int(fraction * 1000)],
    ]
    assert geometric_classes(value)["consensus"] == expected


def test_nonpelvic_without_bladder_or_pelvic_landmarks():
    value = evidence()
    value["organs"] = {"vertebrae_T12": organ(0, 100)}
    value["targets"] = [target(z=10)]
    result = geometric_classes(value)
    assert result["consensus"] == "X"
    assert result["overlap_not_measured_reason"] == "empty_automatic_bladder_mask"
    value["targets"] = [target(z=9.999)]
    assert geometric_classes(value)["consensus"] == "U"


def test_representative_is_actual_target_with_lowest_roi_number_on_tie():
    value = evidence()
    value["targets"] = [target(9, volume=100), target(3, volume=20)]
    result = geometric_classes(value)
    assert result["representative_target_roi_number"] == 3
    assert result["representative_overlap"]["target_volume_cm3"] == 20
    assert result["representative_overlap"]["overlap_cm3"] == 10
    assert len(result["selected_target_overlaps"]) == 2
    del value["targets"][1]["organ_overlaps"]["urinary_bladder"]
    result = geometric_classes(value)
    assert result["overlap_not_measured_reason"] == "bladder_overlap_evidence_absent"
    assert set(result["representative_overlap"].values()) == {""}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("PTV pęcherz", "B"),
        ("CTV urinary_bladder", "B"),
        ("CTV prostata", "P"),
        ("CTV miednica", "P"),
        ("GTV brain", "X"),
        ("CTV pęcherzyk", "U"),
        ("GTV segm VIII", "U"),
        ("CTV cervical", "U"),
        ("PTV nodes", "U"),
        ("PTV ring bladder", "U"),
        ("PTV prechemo bladder", "U"),
    ],
)
def test_names_anatomy_and_exceptions(name, expected):
    cls, _ = names_only_class(rtstruct([(1, name, [], 1)]))
    assert cls == expected


@pytest.mark.parametrize(
    "additional,expected",
    [("CTV prostata", "MP"), ("CTV brain", "MX"), ("CTV miednica", "B"), ("PTV", "B")],
)
def test_names_mixed_fields_and_generic_targets(additional, expected):
    assert (
        names_only_class(rtstruct([(1, "CTV bladder", [], 1), (2, additional, [], 1)]))[
            0
        ]
        == expected
    )


def test_names_interpreted_types_contours_and_oars():
    ds = rtstruct(
        [
            (1, "generic bladder", ["PTV"], 1),
            (2, "PTV brain", ["ORGAN"], 1),
            (3, "CTV brain", [], 0),
        ]
    )
    cls, result = names_only_class(ds)
    assert cls == "B"
    assert [row["target"] for row in result["rois"]] == [True, False, False]
    ds = rtstruct([(1, "CTV", [], 1), (2, "bladder", ["ORGAN"], 1)])
    assert names_only_class(ds)[0] == "U"
    ds = rtstruct([(1, "CTV bladder", ["CTV", "ORGAN"], 1)])
    assert names_only_class(ds)[1]["reason"] == "contradictory_interpreted_types"
    ds = rtstruct([(1, "CTV bladder", [], 1), (1, "CTV brain", [], 1)])
    assert names_only_class(ds)[1]["reason"] == "source_resolution_error"


def test_names_saved_evidence_recomputes_selection_and_tokens():
    _, result = names_only_class(rtstruct([(1, "CTV bladder", [], 1)]))
    result["rois"][0].update(
        tokens={"nonpelvic": ["invented"]}, target=False, candidate=False
    )
    cls, actual = names_only_class(result)
    assert cls == "B"
    assert actual["rois"][0]["tokens"] == {"bladder": ["bladder"]}
    result["resolution_errors"] = ["synthetic_error"]
    assert names_only_class(result)[0] == "U"


def test_geometry_target_selection_precedence_and_helpers():
    ds = rtstruct(
        [
            (1, "generic", ["PTV"], 1),
            (2, "PTV ring", ["PTV"], 1),
            (3, "PTV named", ["ORGAN"], 1),
            (4, "CTV", [], 1),
            (5, " MARG PTV", ["PTV"], 1),
        ]
    )
    selected, excluded = geometry.select_targets(ds, load_rules()["geometry"])
    assert [row["roi_number"] for row in selected] == [1, 4]
    assert [row["roi_number"] for row in excluded] == [2, 5]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("PTV1-jelita", True),
        ("Pecherz - PTV", False),
        ("MARG_PTV", False),
        ("Z PTV", False),
        ("aPTV", False),
        ("PTV", True),
    ],
)
def test_frozen_name_candidate_dependency(name, expected):
    assert geometry._is_target_name(name) is expected


def test_custom_rules_and_fresh_loads(monkeypatch):
    first = load_rules()
    first["geometry"]["criteria"]["C2"]["intersection_ml_min"] = 11
    assert geometric_classes(evidence(), first)["classes"]["C2"] == "P"
    assert load_rules()["geometry"]["criteria"]["C2"]["intersection_ml_min"] == 5
    import rtpipeline.treatment_scope_rules as resources

    assert json.loads(resources._PACKAGED_JSON) == load_rules()

    class Missing:
        def joinpath(self, name):
            return self

        def is_file(self):
            return False

    monkeypatch.setattr(resources, "files", lambda package: Missing())
    resources._packaged_rules.cache_clear()
    assert resources.load_rules()["version"] == "1.0.0"


@pytest.mark.parametrize("sign", [1, -1])
def test_signed_grid_and_even_odd_holes(sign):
    grid = {
        "size": [10, 10, 3],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 9.0 if sign < 0 else 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0, 0.0, float(sign), 0.0, 0.0, 0.0, 1.0],
    }
    loops = []
    for loop in [
        [(2, 2, 1), (6, 2, 1), (6, 6, 1), (2, 6, 1)],
        [(3, 3, 1), (5, 3, 1), (5, 5, 1), (3, 5, 1)],
    ]:
        contour = pydicom.Dataset()
        contour.ContourGeometricType, contour.NumberOfContourPoints = "CLOSED_PLANAR", 4
        contour.ContourData = (
            geometry.physical(np.array(loop), grid).reshape(-1).tolist()
        )
        loops.append(contour)
    measured, indices = geometry.rasterize({"roi_number": 1, "contours": loops}, grid)
    expected = {
        100 + j * 10 + i
        for j in range(2, 7)
        for i in range(2, 7)
        if not (3 <= j <= 5 and 3 <= i <= 5)
    }
    assert set(indices) == expected
    assert measured["volume_ml"] == 0.016


def test_permuted_storage_uses_actual_dicom_planes():
    grid = {
        "size": [10, 3, 10],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    }
    planes = []
    for z in range(3):
        plane = pydicom.Dataset()
        plane.ImagePositionPatient, plane.ImageOrientationPatient = (
            [0, 0, z],
            [1, 0, 0, 0, 1, 0],
        )
        plane.PixelSpacing, plane.Rows, plane.Columns = [1, 1], 10, 10
        planes.append(geometry.slice_plane(plane, grid))
    ds = rtstruct([(1, "PTV", [], 1)])
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    contour.ContourData = [2, 2, 1, 4, 2, 1, 4, 4, 1, 2, 4, 1]
    result, indices = geometry.rasterize(
        {"roi_number": 1, "contours": [contour]}, grid, planes
    )
    k, j, i = np.unravel_index(indices, tuple(grid["size"][::-1]))
    points = geometry.physical(np.array([i, j, k]).T, grid)
    assert {tuple(point) for point in points} == {
        (x, y, 1) for x in range(2, 5) for y in range(2, 5)
    }
    assert result["volume_ml"] == 0.009


def test_cli_summary_excludes_source_text_and_private_output(tmp_path, capsys):
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(evidence()))
    names = tmp_path / "names.json"
    _, details = names_only_class(
        rtstruct([(1, "CTV bladder synthetic-marker", [], 1)])
    )
    names.write_text(json.dumps(details))
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--evidence",
                str(source),
                "--names-evidence",
                str(names),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "synthetic-marker" not in captured.out
    assert "roi_number" not in captured.out
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text())["geometry"]["consensus"] == "B"
    assert main(["--evidence", str(source), "--output", str(output)]) == 2


def test_existing_cli_dispatch(monkeypatch):
    import rtpipeline.cli as cli
    import rtpipeline.treatment_scope as scope

    monkeypatch.setattr(scope, "main", lambda argv: 17 if argv == ["--help"] else 99)
    assert cli.main(["scope", "--help"]) == 17


@pytest.fixture
def organized_course(tmp_path):
    """Small, synthetic DICOM/CT/mask course; no external data or downloads."""
    import SimpleITK as sitk
    from pydicom.uid import (
        generate_uid,
        ExplicitVRLittleEndian,
        CTImageStorage,
        RTStructureSetStorage,
        RTPlanStorage,
    )

    root = tmp_path / "synthetic-person" / "synthetic-course"
    for folder in [
        "metadata",
        "DICOM/CT",
        "DICOM/RTSTRUCT",
        "DICOM/RTPLAN",
        "Segmentation_TotalSegmentator/synthetic",
    ]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    frame, series, structure_uid, plan_uid = [generate_uid() for _ in range(4)]

    def save(ds, path, sop_class, uid):
        ds.file_meta = pydicom.Dataset()
        (
            ds.file_meta.TransferSyntaxUID,
            ds.file_meta.MediaStorageSOPClassUID,
            ds.file_meta.MediaStorageSOPInstanceUID,
        ) = ExplicitVRLittleEndian, sop_class, uid
        ds.SOPClassUID, ds.SOPInstanceUID = sop_class, uid
        pydicom.dcmwrite(path, ds, enforce_file_format=True)

    pixels = np.zeros((3, 10, 10), dtype=np.int16)
    for z in range(3):
        ds = pydicom.Dataset()
        ds.SeriesInstanceUID, ds.FrameOfReferenceUID = series, frame
        ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing = (
            [0, 0, z * 10],
            [1, 0, 0, 0, 1, 0],
            [10, 10],
        )
        ds.Rows, ds.Columns, ds.SamplesPerPixel = 10, 10, 1
        (
            ds.PhotometricInterpretation,
            ds.BitsAllocated,
            ds.BitsStored,
            ds.HighBit,
            ds.PixelRepresentation,
        ) = "MONOCHROME2", 16, 16, 15, 1
        ds.PixelData = pixels[z].tobytes()
        save(ds, root / f"DICOM/CT/{z}.dcm", CTImageStorage, generate_uid())

    def nifti(array, path):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing([10.0, 10.0, 10.0])
        sitk.WriteImage(image, str(path))

    nifti(pixels, root / "CT.nii.gz")
    ds = rtstruct([(1, "CTV bladder", ["CTV"], 1)])
    ds.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = frame
    save(
        ds, root / "DICOM/RTSTRUCT/structure.dcm", RTStructureSetStorage, structure_uid
    )
    plan = pydicom.Dataset()
    reference = pydicom.Dataset()
    reference.ReferencedSOPInstanceUID = structure_uid
    plan.ReferencedStructureSetSequence = [reference]
    save(plan, root / "DICOM/RTPLAN/plan.dcm", RTPlanStorage, plan_uid)
    segmentation = root / "Segmentation_TotalSegmentator/synthetic"
    masks = []
    for label in ["hip_left", "hip_right", "vertebrae_L5", "urinary_bladder"]:
        mask = np.zeros_like(pixels, dtype=np.uint8)
        if label.startswith("hip"):
            mask[0] = 1
        elif label == "vertebrae_L5":
            mask[2] = 1
        else:
            mask[1, 2:5, 2:5] = 1
        path = segmentation / (label + ".nii.gz")
        nifti(mask, path)
        masks.append(path)

    def write(path, data):
        path.write_text(json.dumps(data))

    write(
        root / "metadata/case_metadata.json",
        {
            "patient_id": root.parent.name,
            "course_id": root.name,
            "course_contract": {
                "planning_ct": {
                    "nifti_path": "CT.nii.gz",
                    "dicom_dir": "DICOM/CT",
                    "series_instance_uid": series,
                },
                "authoritative_rtstruct": {
                    "path": "DICOM/RTSTRUCT/structure.dcm",
                    "sop_instance_uid": structure_uid,
                },
                "selected_plans": [
                    {"path": "DICOM/RTPLAN/plan.dcm", "sop_instance_uid": plan_uid}
                ],
            },
        },
    )
    write(
        root / "metadata/segmentation_status.json",
        {"status": "ok", "evidence": {"segmentation_directory": str(segmentation)}},
    )
    write(
        segmentation / "manifest.json",
        {
            "models": [{"masks": [path.name for path in masks]}],
            "source_nifti_sha256": geometry.sha(root / "CT.nii.gz"),
            "source_series_instance_uid": series,
        },
    )
    write(
        root / ".segmentation_done",
        {
            "status": "ok",
            "patient_id": root.parent.name,
            "course_id": root.name,
            "outputs": [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": geometry.sha(path),
                    "size": path.stat().st_size,
                }
                for path in masks
            ],
        },
    )
    return root


def test_fresh_course_measurement_and_input_closure(organized_course):
    result = geometric_classes(organized_course)
    assert result["measurement_status"] == "measured"
    assert result["classes"] == dict.fromkeys(CRITERIA, "B")
    assert result["representative_overlap"] == {
        "target_volume_cm3": 9.0,
        "overlap_cm3": 9.0,
        "overlap_pct_target": 100.0,
        "overlap_pct_bladder": 100.0,
        "bladder_mask_volume_cm3": 9.0,
    }
    path = (
        organized_course
        / "Segmentation_TotalSegmentator/synthetic/urinary_bladder.nii.gz"
    )
    path.write_bytes(b"synthetic-mutation")
    result = geometric_classes(organized_course)
    assert result["consensus"] == "U"
    assert result["u_reason"] == "segmentation_mask_sentinel_hash_mismatch"


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_explicit_rule_file(tmp_path, extension):
    import yaml

    rule = load_rules()
    rule["geometry"]["criteria"]["C2"]["intersection_ml_min"] = 11
    path = tmp_path / ("rules." + extension)
    path.write_text(json.dumps(rule) if extension == "json" else yaml.safe_dump(rule))
    assert geometric_classes(evidence(), path)["classes"]["C2"] == "P"


def test_cli_invalid_rules_do_not_echo_source(tmp_path, capsys):
    path = tmp_path / "rules.yaml"
    path.write_text("secret-marker: [unterminated")
    assert main(["--evidence", str(tmp_path / "input.json"), "--rules", str(path)]) == 2
    captured = capsys.readouterr()
    assert "secret-marker" not in captured.err
    assert str(path) not in captured.err
