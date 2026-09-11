"""The robustness consumers must act on the outcome a course actually reached.

``robustness_for_course`` returns ``Optional[Path]``, and ``None`` covers two
opposite situations: a course that legitimately measured nothing, and a course
that found no CT or extracted nothing. These tests exercise the consumers that
now have to tell them apart - the course CLI step and its completion receipt,
and the manifest-driven cohort aggregation that accounts for every validated
course.

Everything here is synthetic and self-contained under ``tmp_path``. One real
RTSTRUCT (real DICOM bytes, so identity and content digests are real) is built
and copied per course; the CT series image, the main-radiomics identity
catalog, the mask reader and feature extraction are substituted, exactly as the
existing robustness fixtures do. What stays real is what these tests are about:
the publication path, the sidecar, the completion receipt, manifest parsing,
admission and the cohort accounting. Nothing reads clinical data, starts a
service, launches the pipeline, or writes outside the test's temporary
directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from rtpipeline import cli
from rtpipeline import custom_models
from rtpipeline import radiomics as rm
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.course_manifest import (
    CURRENT_COURSE_MANIFEST_SCHEMA,
    parse_course_manifest,
    read_course_manifest,
)
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from test_science_batch_c import _build_real_rtstruct


SHAPE = (17, 17, 17)
# One volume state and two ±4 mm translation states: a real (small) NTCV grid,
# so the consumer is exercised on the chain the study configures rather than on
# a volume-only special case.
CONDITIONS = ("ntcv_v0", "ntcv_t0_0_4_v0", "ntcv_t0_0_-4_v0")
OUTPUT_NAME = "radiomics_robustness_ct.parquet"
SENTINEL_NAME = ".radiomics_robustness_done"
ROOT = Path(__file__).resolve().parents[1]


def _roi_array() -> np.ndarray:
    array = np.zeros(SHAPE, np.uint8)
    array[6:12, 6:12, 6:12] = 1
    return array


def _nonvolumetric_row(roi_name: str, roi_number: str) -> dict:
    """A terminal, non-technical source disposition the reader would record."""
    return {
        "roi_name": roi_name,
        "roi_number": roi_number,
        "status": "nonvolumetric_nonmeasurement",
        "failure_kind": "structural",
        "structural_code": "ROI_NONVOLUMETRIC_POINT",
        "reason": f"{roi_name} is a POINT marker and encloses no volume",
    }


def _write_config(
    tmp_path: Path, *, apply_to_structures, enabled: bool = True, icc_robust=None
) -> Path:
    import yaml

    perturbation = {
        "apply_to_structures": list(apply_to_structures),
        "small_volume_changes": [0.0],
        "large_volume_changes": [-0.3, 0.0, 0.3],
        "max_translation_mm": 4.0,
        "n_random_contour_realizations": 0,
        "noise_levels": [0.0],
        "intensity": "standard",
    }
    payload = {
        "dicom_root": str(tmp_path / "Input"),
        "output_dir": str(tmp_path / "Output"),
        "logs_dir": str(tmp_path / "Logs"),
        "radiomics_robustness": {
            "enabled": enabled,
            "modes": ["segmentation_perturbation"],
            "segmentation_perturbation": perturbation,
        },
    }
    if icc_robust is not None:
        payload["radiomics_robustness"]["thresholds"] = {"icc": {"robust": icc_robust}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _rob_config(config_path: Path) -> rr.RobustnessConfig:
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return rr.RobustnessConfig.from_dict(payload["radiomics_robustness"])


class Cohort:
    """Synthetic courses plus the knobs the tests need to disturb them."""

    def __init__(self, output_root: Path, config_path: Path, courses: dict):
        self.output_root = output_root
        self.config_path = config_path
        self.courses = courses  # (patient, course) -> SimpleNamespace

    def course_dir(self, patient: str, course: str = "C1") -> Path:
        return self.courses[(patient, course)].course_dir

    def sentinel(self, patient: str, course: str = "C1") -> Path:
        return self.course_dir(patient, course) / SENTINEL_NAME

    def table(self, patient: str, course: str = "C1") -> Path:
        return self.course_dir(patient, course) / OUTPUT_NAME

    def receipt(self, patient: str, course: str = "C1") -> dict:
        return json.loads(self.sentinel(patient, course).read_text(encoding="utf-8"))

    def write_receipt(self, patient: str, payload: dict, course: str = "C1") -> None:
        self.sentinel(patient, course).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def _build_courses(tmp_path, monkeypatch, specs, *, apply_to_structures=("ROI", "Marker*")):
    """Create one course per spec and publish it through the real course CLI.

    ``specs`` maps ``(patient, course)`` to ``"measured"``, ``"source_only"`` or
    ``"unmatched"``. A measured course's mask reader yields one volumetric ROI;
    the other two yield no mask and record one non-volumetric source
    disposition, whose ROI name is either inside the requested selection
    (source-only) or outside it (unmatched).
    """
    output_root = tmp_path / "Output"
    array = _roi_array()
    mask = sitk.GetImageFromArray(array)
    image = sitk.GetImageFromArray(
        np.arange(array.size, dtype=np.float32).reshape(SHAPE)
    )

    # One real RTSTRUCT, copied per course: real DICOM bytes give every course a
    # real SOPInstanceUID and a real content digest to bind to.
    template_dir = tmp_path / "_template"
    (template_dir / "CT").mkdir(parents=True)
    rtstruct = _build_real_rtstruct(template_dir / "CT", side=8, n_slices=3)
    template_rs = template_dir / "RS.dcm"
    rtstruct.ds.save_as(template_rs)

    courses = {}
    identities = {}
    rs_paths = {}
    for (patient_id, course_id), kind in specs.items():
        course_dir = output_root / patient_id / course_id
        course_dir.mkdir(parents=True)
        rs_path = course_dir / "RS.dcm"
        shutil.copyfile(template_rs, rs_path)
        rs_paths[rs_path] = kind
        courses[(patient_id, course_id)] = SimpleNamespace(
            course_dir=course_dir, kind=kind, rs_path=rs_path
        )
        identities[course_dir] = rr.RobustnessRoiIdentity.from_mapping(
            dict(
                patient_id=patient_id,
                course_id=course_id,
                series_uid="1.2.3",
                segmentation_source="Manual",
                mask_identity=f"source-mask-{patient_id}-{course_id}",
                roi_original_name="ROI",
                stable_roi_identifier=f"roi-{patient_id}-{course_id}",
            )
        )

    monkeypatch.setattr(
        rr,
        "load_course_contract",
        lambda course: SimpleNamespace(
            planning_ct_dir=Path(course),
            planning_ct={"series_instance_uid": "1.2.3"},
        ),
    )
    monkeypatch.setattr(
        rr,
        "_load_main_ct_identity_catalog",
        lambda course_dir, **kwargs: (
            {("Manual", "ROI"): identities[Path(course_dir)]},
            {},
        ),
    )
    monkeypatch.setattr(rm, "_load_series_image", lambda _: image)
    monkeypatch.setattr(
        rm,
        "_standard_rtstruct_sources",
        lambda contract, course_dir: [
            ("Manual", Path(course_dir) / "RS.dcm", None)
        ],
    )

    def _masks(ct_dir, rs_path, *args, **kwargs):
        kind = rs_paths[Path(rs_path)]
        if kind == "measured":
            return {"ROI": array}
        sink = kwargs.get("failure_outcomes")
        roi_name = "Marker1" if kind == "source_only" else "Unrequested1"
        if sink is not None:
            sink.append(_nonvolumetric_row(roi_name, "2"))
        return {}

    monkeypatch.setattr(rm, "_rtstruct_masks", _masks)
    monkeypatch.setattr(rm, "_mask_from_array_like", lambda *a: mask)
    monkeypatch.setattr(custom_models, "list_custom_model_outputs", lambda *a: [])
    monkeypatch.setenv("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")

    # Distinct per-course value levels: identical values across subjects give
    # zero between-subject variance, which is not what this fixture is about.
    bases = {
        key: 10.0 * (index + 1) for index, key in enumerate(sorted(specs))
    }

    def _features(image_arg, masks, config, **kwargs):
        fields = kwargs["source_identity"].as_dict()
        base = bases[(fields["patient_id"], fields["course_id"])]
        rows = []
        for offset, perturbation_id in enumerate(sorted(masks)):
            identity = rr._perturbed_mask_identity(masks[perturbation_id])
            for arm in CT_EXTRACTION_ARMS:
                rows.append(
                    {
                        **fields,
                        "structure": "ROI",
                        "modality": "CT",
                        "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                        "perturbed_mask_identity": identity,
                        "perturbation_id": perturbation_id,
                        "extraction_arm": arm,
                        "run_identifier": kwargs["run_identifier"],
                        "feature_name": "original_firstorder_Mean",
                        "value": base + 0.01 * offset,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(rr, "extract_features_for_masks", _features)

    config_path = _write_config(tmp_path, apply_to_structures=apply_to_structures)
    return Cohort(output_root, config_path, courses)


def _run_course(cohort: Cohort, patient: str, course: str = "C1", *, sentinel=True):
    course_dir = cohort.course_dir(patient, course)
    argv = [
        "radiomics-robustness",
        "--course-dir",
        str(course_dir),
        "--config",
        str(cohort.config_path),
        "--output",
        str(course_dir / OUTPUT_NAME),
    ]
    if sentinel:
        argv += ["--sentinel", str(course_dir / SENTINEL_NAME)]
    return cli.main(argv)


def _write_manifest(
    cohort: Cohort,
    *,
    entries=None,
    quarantines=(),
    schema=CURRENT_COURSE_MANIFEST_SCHEMA,
    overrides=None,
) -> Path:
    if entries is None:
        entries = [
            {"patient": patient, "course": course}
            for patient, course in sorted(cohort.courses)
        ]
    payload = {
        "schema": schema,
        "intended_course_count": len(entries) + len(quarantines),
        "attempted_course_count": len(entries) + len(quarantines),
        "validated_course_count": len(entries),
        "technical_quarantine_count": len(quarantines),
        "technical_quarantines": list(quarantines),
        "courses": entries,
    }
    if overrides:
        payload.update(overrides)
    manifest = cohort.output_root / "_COURSES" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def _summary_paths(tmp_path):
    output = tmp_path / "Output" / "_RESULTS" / "radiomics_robustness_summary.xlsx"
    return output, output.parent / (output.stem + "_raw_values.parquet")


def _run_aggregate(cohort: Cohort, manifest: Path, output: Path) -> int:
    return cli.main(
        [
            "radiomics-robustness-aggregate",
            "--manifest",
            str(manifest),
            "--output-root",
            str(cohort.output_root),
            "--output",
            str(output),
            "--config",
            str(cohort.config_path),
        ]
    )


def _assert_failed_for(caplog, *expected: str) -> None:
    """A non-zero exit must be the failure the test is about, not any failure."""
    text = "\n".join(record.getMessage() for record in caplog.records)
    for fragment in expected:
        assert fragment in text, f"missing {fragment!r} in:\n{text}"


def _plant_stale_outputs(output: Path, raw: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"stale workbook")
    raw.write_bytes(b"stale raw values")


def _assert_no_nominal_outputs(output: Path, raw: Path) -> None:
    assert not output.exists(), "a rejected cohort left a nominal workbook"
    assert not raw.exists(), "a rejected cohort left nominal raw values"
    if output.parent.exists():
        assert not list(output.parent.glob("*.tmp")), "temporary outputs survived"


# ---------------------------------------------------------------------------
# The course step: an outcome, not an exit status
# ---------------------------------------------------------------------------


def test_measured_course_writes_a_receipt_bound_to_its_own_run(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})

    assert _run_course(cohort, "P1") == 0

    receipt = cohort.receipt("P1")
    assert receipt["schema"] == rc.ROBUSTNESS_COMPLETION_SCHEMA
    assert receipt["status"] == "ok"
    assert (receipt["patient_id"], receipt["course_id"]) == ("P1", "C1")
    assert receipt["measurement_outcome"] == rr.ROBUSTNESS_MEASURED_OUTCOME
    assert receipt["output_path"] == OUTPUT_NAME

    table = cohort.table("P1")
    published = pd.read_parquet(table)
    assert set(published["run_identifier"]) == {receipt["robustness_run_identifier"]}
    assert receipt["measured_output"]["sha256"] == rr._file_sha256(table)
    sidecar = rr.robustness_source_dispositions_path(cohort.course_dir("P1"))
    assert receipt["source_dispositions"]["sha256"] == rr._file_sha256(sidecar)
    assert receipt["effective_configuration_sha256"] == rr._content_sha256(
        rr.effective_robustness_configuration(
            _rob_config(cohort.config_path), output_name=OUTPUT_NAME
        )
    )

    validated = rc.read_robustness_completion_sentinel(cohort.sentinel("P1"))
    assert validated.measured and validated.output_name == OUTPUT_NAME


def test_source_only_course_completes_without_fabricating_a_table(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P2", "C1"): "source_only"})

    assert _run_course(cohort, "P2") == 0

    receipt = cohort.receipt("P2")
    assert receipt["measurement_outcome"] == rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME
    assert receipt["measured_output"] is None
    assert not cohort.table("P2").exists()
    assert receipt["source_dispositions"]["row_count"] == 1
    payload = json.loads(
        rr.robustness_source_dispositions_path(cohort.course_dir("P2")).read_text(
            encoding="utf-8"
        )
    )
    assert payload["rows"][0]["roi_name"] == "Marker1"
    assert rc.read_robustness_completion_sentinel(cohort.sentinel("P2")).measured is False


def test_unmatched_selection_blocks_the_step_and_publishes_no_receipt(
    tmp_path, monkeypatch, caplog
):
    cohort = _build_courses(tmp_path, monkeypatch, {("P3", "C1"): "unmatched"})

    assert _run_course(cohort, "P3") == 1
    _assert_failed_for(
        caplog,
        "matched no source structure",
        "not an anatomical finding",
    )

    # The step is blocked, but the run's own accounting survives untouched: an
    # unmatched selection is a configuration fact, not an anatomical exclusion.
    assert not cohort.sentinel("P3").exists()
    assert not cohort.table("P3").exists()
    payload = json.loads(
        rr.robustness_source_dispositions_path(cohort.course_dir("P3")).read_text(
            encoding="utf-8"
        )
    )
    assert payload["measurement_outcome"] == rr.ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME
    assert payload["source_only_basis"]["selection_matched_nonvolumetric_count"] == 0
    assert payload["rows"][0]["roi_name"] == "Unrequested1"


def test_absent_ct_is_a_failure_not_a_completed_course(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P4", "C1"): "measured"})
    monkeypatch.setattr(rm, "_load_series_image", lambda _: None)
    monkeypatch.setattr(
        rr,
        "load_course_contract",
        lambda course: SimpleNamespace(
            planning_ct_dir=Path(course),
            planning_ct={"series_instance_uid": "1.2.3"},
            planning_ct_nifti=None,
        ),
    )

    assert _run_course(cohort, "P4") == 1
    _assert_failed_for(caplog, "no source-disposition evidence")
    assert not cohort.sentinel("P4").exists()
    assert not rr.robustness_source_dispositions_path(
        cohort.course_dir("P4")
    ).exists()


def test_an_earlier_sidecar_cannot_certify_a_later_empty_run(tmp_path, monkeypatch):
    """A run that measures nothing must not adopt the previous run's evidence."""
    cohort = _build_courses(tmp_path, monkeypatch, {("P5", "C1"): "measured"})
    assert _run_course(cohort, "P5") == 0
    first = cohort.receipt("P5")

    # The next attempt finds no CT: it publishes nothing, so the completed
    # sidecar and receipt of the previous attempt must both be gone.
    monkeypatch.setattr(rm, "_load_series_image", lambda _: None)
    monkeypatch.setattr(
        rr,
        "load_course_contract",
        lambda course: SimpleNamespace(
            planning_ct_dir=Path(course),
            planning_ct={"series_instance_uid": "1.2.3"},
            planning_ct_nifti=None,
        ),
    )
    assert _run_course(cohort, "P5") == 1
    assert not cohort.sentinel("P5").exists()
    assert not rr.robustness_source_dispositions_path(
        cohort.course_dir("P5")
    ).exists()
    assert not cohort.table("P5").exists()
    assert first["measurement_outcome"] == rr.ROBUSTNESS_MEASURED_OUTCOME


def test_disabled_configuration_removes_a_previous_receipt(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P6", "C1"): "measured"})
    assert _run_course(cohort, "P6") == 0
    assert cohort.sentinel("P6").is_file()

    _write_config(tmp_path, apply_to_structures=("ROI",), enabled=False)

    assert _run_course(cohort, "P6") == 1
    assert not cohort.sentinel("P6").exists()


def test_mode_disabled_is_its_own_branch(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P7", "C1"): "measured"})
    rob = _rob_config(cohort.config_path)
    rob.modes = ["scan_rescan"]
    config = SimpleNamespace(radiomics_thread_limit=1)

    with pytest.raises(rr.RobustnessNotRequestedError):
        rr.run_robustness_course(
            config,
            rob,
            cohort.course_dir("P7"),
            output_path=cohort.table("P7"),
        )
    assert not rr.robustness_source_dispositions_path(
        cohort.course_dir("P7")
    ).exists()


def test_a_rerun_replaces_the_receipt_and_the_old_run_cannot_be_revived(
    tmp_path, monkeypatch
):
    cohort = _build_courses(tmp_path, monkeypatch, {("P8", "C1"): "measured"})
    assert _run_course(cohort, "P8") == 0
    first = cohort.receipt("P8")["robustness_run_identifier"]

    assert _run_course(cohort, "P8") == 0
    second = cohort.receipt("P8")["robustness_run_identifier"]

    assert first != second
    with pytest.raises(ValueError, match="stale robustness source dispositions"):
        rr.load_robustness_source_dispositions(
            cohort.course_dir("P8"),
            run_identifier=first,
            rob_config=_rob_config(cohort.config_path),
        )


def test_a_sentinel_outside_the_course_is_refused(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P9", "C1"): "measured"})
    foreign = tmp_path / SENTINEL_NAME
    foreign.write_text("stale", encoding="utf-8")

    result = cli.main(
        [
            "radiomics-robustness",
            "--course-dir",
            str(cohort.course_dir("P9")),
            "--config",
            str(cohort.config_path),
            "--output",
            str(cohort.table("P9")),
            "--sentinel",
            str(foreign),
        ]
    )

    assert result == 1
    assert foreign.read_text(encoding="utf-8") == "stale"
    assert not cohort.table("P9").exists()


def test_an_output_outside_the_course_is_refused(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("PA", "C1"): "measured"})

    assert (
        cli.main(
            [
                "radiomics-robustness",
                "--course-dir",
                str(cohort.course_dir("PA")),
                "--config",
                str(cohort.config_path),
                "--output",
                str(tmp_path / OUTPUT_NAME),
                "--sentinel",
                str(cohort.sentinel("PA")),
            ]
        )
        == 1
    )
    assert not cohort.sentinel("PA").exists()
    assert not (tmp_path / OUTPUT_NAME).exists()


# ---------------------------------------------------------------------------
# Receipt revalidation
# ---------------------------------------------------------------------------


def test_a_legacy_ok_token_is_rejected_not_migrated(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("PB", "C1"): "measured"})
    assert _run_course(cohort, "PB") == 0
    cohort.sentinel("PB").write_text("ok\n", encoding="utf-8")

    with pytest.raises(rc.LegacyRobustnessCompletionError, match="legacy"):
        rc.read_robustness_completion_sentinel(cohort.sentinel("PB"))


def test_a_disabled_marker_is_not_a_completion(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("PC", "C1"): "measured"})
    assert _run_course(cohort, "PC") == 0
    cohort.sentinel("PC").write_text("disabled\n", encoding="utf-8")

    with pytest.raises(rc.RobustnessCompletionError, match="disabled"):
        rc.read_robustness_completion_sentinel(cohort.sentinel("PC"))


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda p: p.update(patient_id="PX"), "not the course"),
        (lambda p: p.update(measurement_outcome="selection_matched_no_source_structure"), "does not complete"),
        (lambda p: p.update(status="failed"), "not a completed"),
        (lambda p: p.update(schema="rtpipeline-robustness-completion-v0"), "declares schema"),
        (lambda p: p["measured_output"].update(sha256="0" * 64), "changed after completion"),
        (lambda p: p["source_dispositions"].update(sha256="0" * 64), "changed after"),
        (lambda p: p.update(output_path="../escape.parquet"), "plain file name"),
    ],
)
def test_receipt_revalidation_rejects_a_broken_binding(
    tmp_path, monkeypatch, mutate, message
):
    cohort = _build_courses(tmp_path, monkeypatch, {("PD", "C1"): "measured"})
    assert _run_course(cohort, "PD") == 0
    payload = cohort.receipt("PD")
    mutate(payload)
    cohort.write_receipt("PD", payload)

    with pytest.raises(rc.RobustnessCompletionError, match=message):
        rc.read_robustness_completion_sentinel(cohort.sentinel("PD"))


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def _manifest_payload(**overrides):
    payload = {
        "schema": CURRENT_COURSE_MANIFEST_SCHEMA,
        "intended_course_count": 1,
        "attempted_course_count": 1,
        "validated_course_count": 1,
        "technical_quarantine_count": 0,
        "technical_quarantines": [],
        "courses": [{"patient": "P1", "course": "C1"}],
    }
    payload.update(overrides)
    return payload


def test_manifest_parsing_matches_the_general_helper_for_a_valid_manifest(tmp_path):
    payload = _manifest_payload()
    general = parse_course_manifest(
        payload, output_dir=tmp_path, manifest_path="m.json"
    )
    strict = parse_course_manifest(
        payload,
        output_dir=tmp_path,
        manifest_path="m.json",
        require_current_schema=True,
    )
    assert general[0] == strict[0]
    assert all(strict[1][key] == value for key, value in general[1].items())
    assert strict[1]["course_identities"] == frozenset({("P1", "C1")})
    assert json.loads(strict[1]["manifest_snapshot"]) == payload
    assert general[0] == [("P1", "C1", tmp_path / "P1" / "C1")]
    assert general[1]["validated_course_count"] == 1


@pytest.mark.parametrize(
    "payload,message",
    [
        (_manifest_payload(courses=[{"patient": "../etc", "course": "C1"}]), "path separator"),
        (_manifest_payload(courses=[{"patient": "P1", "course": "a/b"}]), "path separator"),
        (_manifest_payload(courses=[{"patient": "/abs", "course": "C1"}]), "path separator"),
        (_manifest_payload(courses=[{"patient": " P1", "course": "C1"}]), "whitespace"),
        (
            _manifest_payload(
                courses=[{"patient": "P1", "course": "C1"}, {"patient": "P1", "course": "C1"}],
                validated_course_count=2,
                intended_course_count=2,
                attempted_course_count=2,
            ),
            "duplicate course",
        ),
        (_manifest_payload(validated_course_count=True, intended_course_count=True, attempted_course_count=True), "must be an integer count"),
        (_manifest_payload(validated_course_count=2), "validated count does not match"),
        (_manifest_payload(schema="rtpipeline-organized-course-manifest-v1"), "requires"),
        (
            _manifest_payload(
                courses=[{"patient": "P1", "course": "C1", "path": "/somewhere/else"}]
            ),
            "declares path",
        ),
    ],
)
def test_strict_manifest_parsing_rejects_a_malformed_manifest(tmp_path, payload, message):
    with pytest.raises(RuntimeError, match=message):
        parse_course_manifest(
            payload,
            output_dir=tmp_path,
            manifest_path="m.json",
            require_current_schema=True,
        )


def test_unreadable_manifest_is_reported_as_such(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        read_course_manifest(path, output_dir=tmp_path, require_current_schema=True)


# ---------------------------------------------------------------------------
# Cohort accounting
# ---------------------------------------------------------------------------


def test_cohort_admits_measured_and_source_only_courses_together(tmp_path, monkeypatch):
    cohort = _build_courses(
        tmp_path,
        monkeypatch,
        {("P1", "C1"): "measured", ("P2", "C1"): "measured", ("P3", "C1"): "source_only"},
    )
    for patient in ("P1", "P2", "P3"):
        assert _run_course(cohort, patient) == 0
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 0

    outcomes = pd.read_excel(output, sheet_name="course_outcomes")
    # The denominator is the manifest's, not the measured subset's.
    assert len(outcomes) == 3
    assert set(zip(outcomes["patient_id"], outcomes["course_id"])) == {
        ("P1", "C1"),
        ("P2", "C1"),
        ("P3", "C1"),
    }
    measured = outcomes.set_index("patient_id")
    assert bool(measured.loc["P1", "table_present"])
    assert not bool(measured.loc["P3", "table_present"])
    assert measured.loc["P3", "measurement_outcome"] == rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME
    assert int(measured.loc["P3", "table_row_count"]) == 0
    assert int(measured.loc["P3", "measured_value_row_count"]) == 0
    assert not bool(measured.loc["P3", "contributes_measurements"])
    assert int(measured.loc["P1", "measured_value_row_count"]) == int(
        measured.loc["P1", "table_row_count"]
    )

    # A source-only course contributes accounting, never perturbation rows.
    raw_frame = pd.read_parquet(raw)
    assert set(raw_frame["patient_id"]) == {"P1", "P2"}
    assert len(raw_frame) == 2 * len(CONDITIONS) * len(CT_EXTRACTION_ARMS)

    dispositions = pd.read_excel(output, sheet_name="source_dispositions")
    assert set(dispositions["patient_id"]) == {"P3"}
    assert set(dispositions["roi_name"]) == {"Marker1"}

    summary = pd.read_excel(output, sheet_name="global_summary")
    assert set(summary["n_subjects"]) == {2}
    # Both CT extraction arms and the full requested grid survive the new path.
    assert set(summary["extraction_arm"]) == set(CT_EXTRACTION_ARMS)
    assert set(summary["n_perturbations"]) == {len(CONDITIONS)}
    assert set(raw_frame["perturbation_id"]) == set(CONDITIONS)
    assert set(raw_frame["extraction_arm"]) == set(CT_EXTRACTION_ARMS)
    per_source = pd.read_excel(output, sheet_name="per_source_summary")
    assert set(per_source["segmentation_source"]) == {"Manual"}
    # The measured statistics are exactly the ones the measured tables imply;
    # the source-only course changed no estimate.
    expected = rr.summarize_feature_stability(
        pd.concat(
            [pd.read_parquet(cohort.table(patient)) for patient in ("P1", "P2")],
            ignore_index=True,
        ),
        _rob_config(cohort.config_path),
    )
    for column in ("icc", "cov_pct", "qcd"):
        np.testing.assert_allclose(
            summary[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_a_measured_course_keeps_its_own_source_dispositions(tmp_path, monkeypatch):
    """A non-volumetric ROI in a measured course is still a published fact."""
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    real_masks = rm._rtstruct_masks

    def _mixed(ct_dir, rs_path, *args, **kwargs):
        sink = kwargs.get("failure_outcomes")
        if sink is not None:
            sink.append(_nonvolumetric_row("Marker1", "2"))
        return real_masks(ct_dir, rs_path, *args, **kwargs)

    monkeypatch.setattr(rm, "_rtstruct_masks", _mixed)
    assert _run_course(cohort, "P1") == 0
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 0

    dispositions = pd.read_excel(output, sheet_name="source_dispositions")
    assert list(dispositions["patient_id"]) == ["P1"]
    assert list(dispositions["course_measurement_outcome"]) == [
        rr.ROBUSTNESS_MEASURED_OUTCOME
    ]
    assert list(dispositions["roi_name"]) == ["Marker1"]
    outcomes = pd.read_excel(output, sheet_name="course_outcomes")
    assert int(outcomes.loc[0, "source_disposition_count"]) == 1
    assert bool(outcomes.loc[0, "contributes_measurements"])


def test_an_all_source_only_cohort_is_an_explicit_zero(tmp_path, monkeypatch):
    cohort = _build_courses(
        tmp_path,
        monkeypatch,
        {("P1", "C1"): "source_only", ("P2", "C1"): "source_only"},
    )
    for patient in ("P1", "P2"):
        assert _run_course(cohort, patient) == 0
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 0

    raw_frame = pd.read_parquet(raw)
    assert raw_frame.empty
    # Empty, but not shapeless: the raw schema is preserved.
    assert list(raw_frame.columns) == list(rr.ROBUSTNESS_RAW_VALUE_COLUMNS)

    summary = pd.read_excel(output, sheet_name="global_summary")
    assert summary.empty
    assert list(summary.columns) == list(rr.ROBUSTNESS_SUMMARY_COLUMNS)
    # No ICC estimate is invented for a cohort that measured nothing.
    assert "icc" in summary.columns and summary["icc"].empty

    outcomes = pd.read_excel(output, sheet_name="course_outcomes")
    assert len(outcomes) == 2
    assert int(outcomes["measured_value_row_count"].sum()) == 0
    assert not outcomes["table_present"].any()
    assert not outcomes["contributes_measurements"].any()
    dispositions = pd.read_excel(output, sheet_name="source_dispositions")
    assert len(dispositions) == 2


def test_a_course_directory_outside_the_manifest_is_never_admitted(tmp_path, monkeypatch):
    cohort = _build_courses(
        tmp_path, monkeypatch, {("P1", "C1"): "measured", ("P2", "C1"): "measured"}
    )
    for patient in ("P1", "P2"):
        assert _run_course(cohort, patient) == 0
    # P2 is a complete, receipted course that the manifest does not name.
    manifest = _write_manifest(cohort, entries=[{"patient": "P1", "course": "C1"}])
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 0

    outcomes = pd.read_excel(output, sheet_name="course_outcomes")
    assert list(outcomes["patient_id"]) == ["P1"]
    assert set(pd.read_parquet(raw)["patient_id"]) == {"P1"}


def test_a_manifest_course_without_a_receipt_fails_the_cohort(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(
        tmp_path, monkeypatch, {("P1", "C1"): "measured", ("P2", "C1"): "measured"}
    )
    for patient in ("P1", "P2"):
        assert _run_course(cohort, patient) == 0
    cohort.sentinel("P2").unlink()
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(caplog, "no robustness completion receipt")
    _assert_no_nominal_outputs(output, raw)


def test_a_legacy_ok_sentinel_fails_the_cohort(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    cohort.sentinel("P1").write_text("ok\n", encoding="utf-8")
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(
        caplog, "legacy", "must be re-run under the current robustness contract"
    )
    _assert_no_nominal_outputs(output, raw)


def test_an_unresolved_technical_quarantine_blocks_the_cohort(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    quarantine = {
        "patient": "P9",
        "course": "C1",
        "status": "technical_quarantine",
        "disposition_type": "technical_quarantine",
        "clinical_exclusion": False,
        "reason": "planning CT series is incomplete",
    }
    manifest = _write_manifest(cohort, quarantines=[quarantine])
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(caplog, "unresolved technical quarantine", "P9/C1")
    _assert_no_nominal_outputs(output, raw)
    # The manifest evidence is untouched: the quarantine is still recorded.
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["technical_quarantines"] == [quarantine]


def test_a_stale_run_identifier_in_the_receipt_fails_the_cohort(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    payload = cohort.receipt("P1")
    payload["robustness_run_identifier"] = "a-different-run"
    cohort.write_receipt("P1", payload)
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(caplog, "stale robustness source dispositions")
    _assert_no_nominal_outputs(output, raw)


def test_a_cohort_run_under_another_configuration_is_rejected(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)
    # Same courses, different thresholds: the same rows are not the same evidence.
    _write_config(tmp_path, apply_to_structures=("ROI", "Marker*"), icc_robust=0.5)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(caplog, "effective configuration")
    _assert_no_nominal_outputs(output, raw)


def test_a_source_that_is_gone_fails_the_cohort(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    (cohort.course_dir("P1") / "RS.dcm").unlink()
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(caplog, "no longer readable")
    _assert_no_nominal_outputs(output, raw)


def test_table_bytes_changed_after_completion_fail_the_cohort(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    frame = pd.read_parquet(cohort.table("P1"))
    frame.loc[frame.index[0], "value"] = 999.0
    frame.to_parquet(cohort.table("P1"), index=False)
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_failed_for(caplog, "changed after completion")
    _assert_no_nominal_outputs(output, raw)


def test_a_publication_failure_withdraws_the_nominal_cohort_outputs(
    tmp_path, monkeypatch
):
    cohort = _build_courses(
        tmp_path, monkeypatch, {("P1", "C1"): "measured", ("P2", "C1"): "source_only"}
    )
    for patient in ("P1", "P2"):
        assert _run_course(cohort, patient) == 0
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)

    def _failing_writer(*args, **kwargs):
        raise OSError("simulated workbook publication failure")

    monkeypatch.setattr(rr.pd, "ExcelWriter", _failing_writer)

    assert _run_aggregate(cohort, manifest, output) == 1
    _assert_no_nominal_outputs(output, raw)


def test_manifest_mode_and_explicit_inputs_are_mutually_exclusive(tmp_path, monkeypatch):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    manifest = _write_manifest(cohort)
    output, _ = _summary_paths(tmp_path)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "radiomics-robustness-aggregate",
                "--manifest",
                str(manifest),
                "--inputs",
                str(cohort.table("P1")),
                "--output",
                str(output),
                "--config",
                str(cohort.config_path),
            ]
        )


def test_manifest_mode_requires_an_output_root(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    manifest = _write_manifest(cohort)
    output, raw = _summary_paths(tmp_path)

    assert (
        cli.main(
            [
                "radiomics-robustness-aggregate",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--config",
                str(cohort.config_path),
            ]
        )
        == 1
    )
    _assert_failed_for(caplog, "--manifest requires --output-root")
    _assert_no_nominal_outputs(output, raw)


def test_explicit_inputs_do_not_take_a_scanned_root(tmp_path, monkeypatch, caplog):
    cohort = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(cohort, "P1") == 0
    output, raw = _summary_paths(tmp_path)

    assert (
        cli.main(
            [
                "radiomics-robustness-aggregate",
                "--inputs",
                str(cohort.table("P1")),
                "--output-root",
                str(cohort.output_root),
                "--output",
                str(output),
                "--config",
                str(cohort.config_path),
            ]
        )
        == 1
    )
    _assert_failed_for(caplog, "--output-root belongs to manifest mode")
    _assert_no_nominal_outputs(output, raw)


def test_explicit_inputs_keep_the_measured_only_contract(tmp_path, monkeypatch):
    cohort = _build_courses(
        tmp_path, monkeypatch, {("P1", "C1"): "measured", ("P2", "C1"): "measured"}
    )
    for patient in ("P1", "P2"):
        assert _run_course(cohort, patient) == 0
    output, raw = _summary_paths(tmp_path)

    assert (
        cli.main(
            [
                "radiomics-robustness-aggregate",
                "--inputs",
                str(cohort.table("P1")),
                str(cohort.table("P2")),
                "--output",
                str(output),
                "--config",
                str(cohort.config_path),
            ]
        )
        == 0
    )

    workbook = pd.ExcelFile(output)
    assert "course_outcomes" not in workbook.sheet_names
    assert "source_dispositions" not in workbook.sheet_names
    assert "global_summary" in workbook.sheet_names


# ---------------------------------------------------------------------------
# Workflow contracts
# ---------------------------------------------------------------------------


def _robustness_rule_bodies() -> list[str]:
    text = (ROOT / "Snakefile").read_text(encoding="utf-8")
    marker = "rule radiomics_robustness_course:"
    starts = [
        index
        for index in range(len(text))
        if text.startswith(marker, index)
    ]
    assert len(starts) == 2, "expected the container and local rule variants"
    bodies = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else text.index(
            "rule aggregate_radiomics_robustness:"
        )
        bodies.append(text[start:end])
    return bodies


def test_both_course_rule_variants_delegate_completion_to_the_cli():
    for body in _robustness_rule_bodies():
        assert '--sentinel "{output.sentinel}"' in body
        # No shell-level success token: an exit status certifies no outcome.
        assert 'echo "ok" > {output.sentinel}' not in body
        assert 'if [ ! -f "{output.sentinel}" ]; then' in body
        # The disabled branch stays explicit and separate.
        assert 'echo "disabled" > {output.sentinel}' in body


def test_the_aggregate_rule_consumes_the_manifest_without_scanning():
    text = (ROOT / "Snakefile").read_text(encoding="utf-8")
    rule = text[
        text.index("rule aggregate_radiomics_robustness:") : text.index(
            "rule aggregate_results:"
        )
    ]
    assert "--manifest {input.manifest}" in rule
    assert "--output-root {params.output_dir}" in rule
    assert "--inputs" not in rule
    assert "find " not in rule
    assert 'grep -q "^ok"' not in rule
