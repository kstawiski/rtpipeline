"""The robustness CLI must refuse a policy it was never given.

Both ``rtpipeline.cli`` robustness entry points used to replace an absent,
unreadable or non-mapping configuration with ``{}``. ``RobustnessConfig
.from_dict({})`` returns the shipped defaults with ``enabled=True``, so a run
whose configuration never arrived perturbed, measured, aggregated and published
under a policy nobody requested - and a per-course value that was present but
unusable was silently replaced by a default, quietly shrinking the grid.

Everything here is synthetic and self-contained under ``tmp_path``: YAML text,
empty course directories, and receipts written by the shipped receipt writer.
The producers (``run_robustness_course``, cohort admission, the two aggregation
functions and manifest reading) are substituted, so these tests show *where the
configuration gate sits relative to them*, not that extraction works. Nothing
reads clinical data, starts a service, launches Snakemake or a pipeline, or
writes outside the test's temporary directory. The Snakefile checks render the
rule's own shell text and run only its disabled branch, with a stub interpreter
that records any invocation instead of executing one.
"""

from __future__ import annotations

import json
import re
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rtpipeline import cli
from rtpipeline import course_manifest as cm
from rtpipeline import radiomics_conda as rc_conda
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.rt_details import DEFAULT_ROI_FAMILY_NAMES

ROOT = Path(__file__).resolve().parents[1]
SENTINEL_NAME = rc.ROBUSTNESS_COMPLETION_SENTINEL_NAME
OUTPUT_NAME = "radiomics_robustness_ct.parquet"
SHELL_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


class _ProducerRan(BaseException):
    """A producer was reached.

    Deliberately not an ``Exception``: the CLI wraps its producers in broad
    ``except Exception`` handlers, which would convert this into the very exit
    status the test is trying to attribute to the configuration gate.
    """


def _enabled_section(**overrides) -> dict:
    """The shipped standard grid, written out explicitly."""
    section = {
        "enabled": True,
        "modes": ["segmentation_perturbation"],
        "segmentation_perturbation": {
            "intensity": "standard",
            "apply_to_structures": ["GTV*", "CTV*", "PTV*"],
            "small_volume_changes": [-0.15, 0.0, 0.15],
            "max_translation_mm": 4.0,
            "n_random_contour_realizations": 2,
            "noise_levels": [0.0, 10.0, 20.0],
        },
    }
    section.update(overrides)
    return section


def _write_config(tmp_path: Path, payload, *, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _enabled_config(tmp_path: Path, **overrides) -> Path:
    return _write_config(tmp_path, {"radiomics_robustness": _enabled_section(**overrides)})


def _course_dir(tmp_path: Path, patient: str = "P1", course: str = "C1") -> Path:
    course_dir = tmp_path / "Output" / patient / course
    course_dir.mkdir(parents=True, exist_ok=True)
    return course_dir


def _plant_legitimate_receipt(course_dir: Path) -> bytes:
    """Publish a receipt a consumer accepts right now, through the real writer."""
    dispositions = rr.robustness_source_dispositions_path(course_dir)
    dispositions.parent.mkdir(parents=True, exist_ok=True)
    dispositions.write_text('{"rows": []}\n', encoding="utf-8")
    sentinel = course_dir / SENTINEL_NAME
    rc.write_robustness_completion_sentinel(
        sentinel,
        course_dir,
        patient_id=course_dir.parent.name,
        course_id=course_dir.name,
        run_identifier="earlier-legitimate-run",
        measurement_outcome="source_only_nonvolumetric",
        output_name=OUTPUT_NAME,
        dispositions_path=dispositions,
        measured_output=None,
        source_disposition_count=0,
        effective_configuration_sha256="0" * 64,
    )
    # Before the failing attempt, a consumer revalidates it without complaint.
    rc.read_robustness_completion_sentinel(sentinel)
    return sentinel.read_bytes()


@pytest.fixture
def no_producer(monkeypatch):
    """Reaching any producer is itself the failure these tests look for."""

    def _never(*_args, **_kwargs):
        raise _ProducerRan("a producer ran before the configuration was accepted")

    monkeypatch.setattr(rr, "run_robustness_course", _never)
    monkeypatch.setattr(rr, "robustness_for_course", _never)
    monkeypatch.setattr(rr, "admit_robustness_cohort_course", _never)
    monkeypatch.setattr(rr, "aggregate_robustness_cohort", _never)
    monkeypatch.setattr(rr, "aggregate_robustness_results", _never)
    monkeypatch.setattr(cm, "read_course_manifest", _never)


def _run_course(course_dir: Path, config: Path, *, sentinel: bool = True, output=None):
    argv = [
        "radiomics-robustness",
        "--course-dir",
        str(course_dir),
        "--config",
        str(config),
        "--output",
        str(output if output is not None else course_dir / OUTPUT_NAME),
    ]
    if sentinel:
        argv += ["--sentinel", str(course_dir / SENTINEL_NAME)]
    return cli.main(argv)


def _summary_pair(tmp_path: Path, *, base: str = "Output", parent: str = "_RESULTS"):
    output = tmp_path / base / parent / "radiomics_robustness_summary.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    return rr.robustness_cohort_output_paths(output)


def _plant_stale_pair(output: Path, raw: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"stale cohort workbook")
    raw.write_bytes(b"stale cohort raw values")


def _assert_pair_withdrawn(output: Path, raw: Path) -> None:
    assert not output.exists(), "a rejected cohort attempt left a nominal workbook"
    assert not raw.exists(), "a rejected cohort attempt left nominal raw values"


def _write_manifest(tmp_path: Path, output_root: Path) -> Path:
    manifest = output_root / "_COURSES" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": cm.CURRENT_COURSE_MANIFEST_SCHEMA,
                "intended_course_count": 1,
                "attempted_course_count": 1,
                "validated_course_count": 1,
                "technical_quarantine_count": 0,
                "technical_quarantines": [],
                "courses": [{"patient": "P1", "course": "C1"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def _run_aggregate_manifest(manifest: Path, output_root: Path, output: Path, config: Path):
    return cli.main(
        [
            "radiomics-robustness-aggregate",
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
            "--output",
            str(output),
            "--config",
            str(config),
        ]
    )


def _run_aggregate_inputs(inputs, output: Path, config: Path):
    return cli.main(
        [
            "radiomics-robustness-aggregate",
            "--inputs",
            *[str(item) for item in inputs],
            "--output",
            str(output),
            "--config",
            str(config),
        ]
    )


def _logged(caplog, *expected: str) -> None:
    text = "\n".join(record.getMessage() for record in caplog.records)
    for fragment in expected:
        assert fragment in text, f"missing {fragment!r} in:\n{text}"


# ---------------------------------------------------------------------------
# The course step refuses a configuration it was never given
# ---------------------------------------------------------------------------

# Every entry is (label, payload-or-None, expected log fragment). ``None``
# means the configuration file is never written at all.
UNUSABLE_CONFIGS = [
    ("absent_file", None, "does not exist"),
    ("non_mapping_root", ["radiomics_robustness"], "must be a mapping"),
    ("empty_document", {}, "declares no radiomics_robustness section"),
    ("absent_section", {"radiomics": {"enabled": True}}, "declares no radiomics_robustness section"),
    ("null_section", {"radiomics_robustness": None}, "radiomics_robustness must be a mapping"),
    (
        "non_mapping_section",
        {"radiomics_robustness": ["enabled"]},
        "radiomics_robustness must be a mapping",
    ),
    (
        "absent_enabled",
        {"radiomics_robustness": {"modes": ["segmentation_perturbation"]}},
        "sets no radiomics_robustness.enabled",
    ),
    (
        "string_enabled",
        {"radiomics_robustness": {"enabled": "true"}},
        "must be a YAML boolean",
    ),
    (
        "integer_enabled",
        {"radiomics_robustness": {"enabled": 1}},
        "must be a YAML boolean",
    ),
    ("disabled", {"radiomics_robustness": {"enabled": False}}, "is disabled in"),
]


@pytest.mark.parametrize(
    "label,payload,fragment", UNUSABLE_CONFIGS, ids=[case[0] for case in UNUSABLE_CONFIGS]
)
def test_the_course_step_refuses_an_unrequested_policy(
    tmp_path, caplog, no_producer, label, payload, fragment
):
    course_dir = _course_dir(tmp_path)
    before = _plant_legitimate_receipt(course_dir)
    assert (course_dir / SENTINEL_NAME).read_bytes() == before
    config = tmp_path / "config.yaml"
    if payload is not None:
        _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        assert _run_course(course_dir, config) == 1

    _logged(caplog, fragment)
    # The earlier legitimate receipt does not survive a failed or disabled
    # attempt: it would otherwise certify this attempt with the previous one's
    # evidence.
    assert not (course_dir / SENTINEL_NAME).exists()
    assert not (course_dir / OUTPUT_NAME).exists()


def test_the_course_step_refuses_an_unparseable_configuration(tmp_path, caplog, no_producer):
    course_dir = _course_dir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("radiomics_robustness:\n  enabled: true\n   modes: [a\n", encoding="utf-8")

    with caplog.at_level("ERROR"):
        assert _run_course(course_dir, config) == 1

    _logged(caplog, "is unreadable")


# Values that are present but unusable. ``from_dict`` reads each of these with
# ``.get(key, default)``, so before the repair each one silently substituted a
# different analysis for the requested one.
UNUSABLE_SETTINGS = [
    ("perturbation_is_a_list", {"segmentation_perturbation": ["standard"]}, "must be a mapping"),
    ("perturbation_is_null", {"segmentation_perturbation": None}, "must be a mapping"),
    (
        "empty_noise_grid",
        {"segmentation_perturbation": {"noise_levels": []}},
        "non-empty list of numbers",
    ),
    (
        "unusable_noise_level",
        {"segmentation_perturbation": {"noise_levels": [0.0, "twenty"]}},
        "noise_levels[1] must be a number",
    ),
    (
        "empty_volume_grid",
        {"segmentation_perturbation": {"small_volume_changes": []}},
        "non-empty list of numbers",
    ),
    (
        "unusable_translation",
        {"segmentation_perturbation": {"max_translation_mm": "4mm"}},
        "max_translation_mm must be a number",
    ),
    (
        "negative_translation",
        {"segmentation_perturbation": {"max_translation_mm": -4.0}},
        "max_translation_mm must be at least 0.0",
    ),
    (
        "unusable_contour_realizations",
        {"segmentation_perturbation": {"n_random_contour_realizations": 2.5}},
        "n_random_contour_realizations must be a whole number",
    ),
    (
        "empty_structure_selection",
        {"segmentation_perturbation": {"apply_to_structures": []}},
        "non-empty list of names",
    ),
    (
        "misspelt_grid_key",
        {"segmentation_perturbation": {"noise_level": [0.0, 10.0, 20.0]}},
        "does not read: noise_level",
    ),
    ("misspelt_section_key", {"metric": {}}, "does not read: metric"),
    ("empty_modes", {"modes": []}, "non-empty list of names"),
    ("metrics_is_a_list", {"metrics": ["icc"]}, "metrics must be a mapping"),
    ("icc_ci_is_a_string", {"metrics": {"icc": {"ci": "yes"}}}, "ci must be a YAML boolean"),
    (
        "unusable_icc_threshold",
        {"thresholds": {"icc": {"robust": "high"}}},
        "thresholds.icc.robust must be a number",
    ),
    (
        "misspelt_threshold_key",
        {"thresholds": {"cov": {"robust_percent": 10.0}}},
        "does not read: robust_percent",
    ),
]


@pytest.mark.parametrize(
    "label,overrides,fragment",
    UNUSABLE_SETTINGS,
    ids=[case[0] for case in UNUSABLE_SETTINGS],
)
def test_an_unusable_setting_is_refused_rather_than_defaulted(
    tmp_path, caplog, no_producer, label, overrides, fragment
):
    course_dir = _course_dir(tmp_path)
    section = _enabled_section()
    for key, value in overrides.items():
        if key == "segmentation_perturbation" and isinstance(value, dict):
            section["segmentation_perturbation"].update(value)
        else:
            section[key] = value
    config = _write_config(tmp_path, {"radiomics_robustness": section})

    with caplog.at_level("ERROR"):
        assert _run_course(course_dir, config) == 1

    _logged(caplog, fragment)


UNUSABLE_RADIOMICS = [
    ("radiomics_is_a_list", {"radiomics": ["enabled"]}, "radiomics must be a mapping"),
    (
        "unusable_env_probe_timeout",
        {"radiomics": {"env_probe_timeout": "quick"}},
        "env_probe_timeout must be a whole number",
    ),
    (
        "unusable_voxel_cap",
        {"radiomics": {"max_voxels": "lots"}},
        "max_voxels must be a whole number",
    ),
    (
        "unusable_skip_rois",
        {"radiomics": {"skip_rois": 7}},
        "skip_rois must be a string or a list of names",
    ),
    (
        "unusable_params_file",
        {"radiomics": {"params_file": ["a.yaml"]}},
        "params_file must be a non-empty string",
    ),
]


@pytest.mark.parametrize(
    "label,extra,fragment",
    UNUSABLE_RADIOMICS,
    ids=[case[0] for case in UNUSABLE_RADIOMICS],
)
def test_the_course_step_refuses_an_unusable_radiomics_section(
    tmp_path, caplog, no_producer, label, extra, fragment
):
    course_dir = _course_dir(tmp_path)
    payload = {"radiomics_robustness": _enabled_section()}
    payload.update(extra)
    config = _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        assert _run_course(course_dir, config) == 1

    _logged(caplog, fragment)


# ---------------------------------------------------------------------------
# What a valid configuration must still do
# ---------------------------------------------------------------------------


def _capture_course(monkeypatch):
    """Substitute the producer with a completing outcome and record its inputs."""
    seen = {}

    def _run(pipeline_config, rob_config, course_dir, *, output_path):
        seen["pipeline_config"] = pipeline_config
        seen["rob_config"] = rob_config
        seen["course_dir"] = course_dir
        seen["output_path"] = output_path
        return SimpleNamespace(
            completes_step=True,
            patient_id=course_dir.parent.name,
            course_id=course_dir.name,
            run_identifier="this-run",
            measurement_outcome="source_only_nonvolumetric",
            output_name=OUTPUT_NAME,
            dispositions_path=rr.robustness_source_dispositions_path(course_dir),
            measured=False,
            measured_output=None,
            source_disposition_count=0,
            effective_configuration_sha256="1" * 64,
        )

    monkeypatch.setattr(rr, "run_robustness_course", _run)
    return seen


def test_an_enabled_section_keeps_every_documented_default(tmp_path, monkeypatch):
    """Absent keys inside an explicitly enabled section are still defaults."""
    seen = _capture_course(monkeypatch)
    course_dir = _course_dir(tmp_path)
    config = _write_config(tmp_path, {"radiomics_robustness": {"enabled": True}})

    assert _run_course(course_dir, config, sentinel=False) == 0

    rob_config = seen["rob_config"]
    assert rob_config.enabled is True
    assert rob_config.modes == ["segmentation_perturbation"]
    perturbation = rob_config.perturbation
    assert perturbation.noise_levels == [0.0, 10.0, 20.0]
    assert perturbation.small_volume_changes == [-0.15, 0.0, 0.15]
    assert perturbation.large_volume_changes == [-0.30, 0.0, 0.30]
    assert perturbation.n_random_contour_realizations == 2
    assert perturbation.max_translation_mm == 4.0
    assert perturbation.intensity == "standard"
    assert perturbation.apply_to_structures == list(DEFAULT_ROI_FAMILY_NAMES)
    assert rob_config.metrics.icc.icc_type == "ICC3"
    assert rob_config.thresholds.icc_robust == 0.90


def test_the_projects_current_configuration_still_routes_the_course_step(
    tmp_path, monkeypatch
):
    """The shipped config.yaml is a positive control, not only a negative one."""
    shipped = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload = {
        "radiomics_robustness": shipped["radiomics_robustness"],
        "radiomics": shipped["radiomics"],
        "dicom_root": str(tmp_path / "Input"),
        "output_dir": str(tmp_path / "Output"),
        "logs_dir": str(tmp_path / "Logs"),
    }
    config = _write_config(tmp_path, payload)
    seen = _capture_course(monkeypatch)
    course_dir = _course_dir(tmp_path)

    assert _run_course(course_dir, config, sentinel=False) == 0

    perturbation = seen["rob_config"].perturbation
    # The complete shipped NTCV chain, not a reduced one.
    assert perturbation.noise_levels == [0.0, 10.0, 20.0]
    assert perturbation.max_translation_mm == 4.0
    assert perturbation.n_random_contour_realizations == 2
    assert perturbation.small_volume_changes == [-0.15, 0.0, 0.15]
    assert seen["pipeline_config"].radiomics_max_voxels == shipped["radiomics"]["max_voxels"]
    assert seen["pipeline_config"].radiomics_min_voxels == shipped["radiomics"]["min_voxels"]
    assert seen["course_dir"] == course_dir.resolve()


def test_a_configured_env_probe_timeout_still_reaches_the_pipeline(tmp_path, monkeypatch):
    seen = _capture_course(monkeypatch)
    course_dir = _course_dir(tmp_path)
    config = _write_config(
        tmp_path,
        {
            "radiomics_robustness": _enabled_section(),
            "radiomics": {"env_probe_timeout": 654},
        },
    )

    assert _run_course(course_dir, config, sentinel=False) == 0
    assert seen["pipeline_config"].radiomics_env_probe_timeout == 654


def test_an_env_probe_timeout_still_propagates_unchanged_as_exit_three(
    tmp_path, monkeypatch, capsys
):
    """The valid-configuration control for the typed timeout contract."""
    error = rc_conda.RadiomicsEnvironmentProbeTimeout(["conda", "run", "probe"], 321, 2)

    def _timeout(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(rr, "run_robustness_course", _timeout)
    course_dir = _course_dir(tmp_path)
    config = _enabled_config(tmp_path)

    assert (
        cli.console_main(
            [
                "radiomics-robustness",
                "--course-dir",
                str(course_dir),
                "--config",
                str(config),
                "--output",
                str(course_dir / OUTPUT_NAME),
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().err) == {
        "code": error.code,
        "message": str(error),
    }


# ---------------------------------------------------------------------------
# Symlinked study paths
# ---------------------------------------------------------------------------


def test_a_symlinked_course_ancestor_is_refused_without_following_the_link(
    tmp_path, caplog, no_producer
):
    real_output = tmp_path / "real_output"
    course_dir = real_output / "P1" / "C1"
    course_dir.mkdir(parents=True)
    before = _plant_legitimate_receipt(course_dir)
    linked_output = tmp_path / "Output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert _run_course(linked_output / "P1" / "C1", config) == 1

    _logged(caplog, "is reached through the symlink", str(linked_output))
    # Nothing was written or deleted through the link.
    assert (course_dir / SENTINEL_NAME).read_bytes() == before
    assert not (course_dir / OUTPUT_NAME).exists()


def test_a_symlinked_output_leaf_is_refused(tmp_path, caplog, no_producer):
    course_dir = _course_dir(tmp_path)
    foreign = tmp_path / "foreign.parquet"
    foreign.write_bytes(b"foreign table")
    linked_output = course_dir / OUTPUT_NAME
    linked_output.symlink_to(foreign)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert _run_course(course_dir, config, sentinel=False, output=linked_output) == 1

    _logged(caplog, "is reached through the symlink")
    assert foreign.read_bytes() == b"foreign table"


def test_a_symlinked_receipt_path_is_refused(tmp_path, caplog, no_producer):
    course_dir = _course_dir(tmp_path)
    foreign = tmp_path / "foreign_receipt.json"
    foreign.write_bytes(b"foreign receipt")
    (course_dir / SENTINEL_NAME).symlink_to(foreign)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert _run_course(course_dir, config) == 1

    _logged(caplog, "is reached through the symlink")
    assert foreign.read_bytes() == b"foreign receipt"


def test_a_foreign_receipt_path_is_refused_without_erasing_it(tmp_path, caplog, no_producer):
    """An invalid argument must not delete the artifact it names."""
    course_dir = _course_dir(tmp_path)
    foreign = tmp_path / SENTINEL_NAME
    foreign.write_text("someone else's file", encoding="utf-8")
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert (
            cli.main(
                [
                    "radiomics-robustness",
                    "--course-dir",
                    str(course_dir),
                    "--config",
                    str(config),
                    "--output",
                    str(course_dir / OUTPUT_NAME),
                    "--sentinel",
                    str(foreign),
                ]
            )
            == 1
        )

    _logged(caplog, "is not inside the course")
    assert foreign.read_text(encoding="utf-8") == "someone else's file"


def test_a_misnamed_receipt_path_is_refused_without_erasing_it(tmp_path, caplog, no_producer):
    course_dir = _course_dir(tmp_path)
    foreign = course_dir / "notes.txt"
    foreign.write_text("someone else's file", encoding="utf-8")
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert (
            cli.main(
                [
                    "radiomics-robustness",
                    "--course-dir",
                    str(course_dir),
                    "--config",
                    str(config),
                    "--output",
                    str(course_dir / OUTPUT_NAME),
                    "--sentinel",
                    str(foreign),
                ]
            )
            == 1
        )

    _logged(caplog, "the receipt is named")
    assert foreign.read_text(encoding="utf-8") == "someone else's file"


# ---------------------------------------------------------------------------
# Cohort aggregation: both modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,payload,fragment", UNUSABLE_CONFIGS, ids=[case[0] for case in UNUSABLE_CONFIGS]
)
def test_manifest_aggregation_refuses_an_unrequested_policy(
    tmp_path, caplog, no_producer, label, payload, fragment
):
    output_root = tmp_path / "Output"
    _course_dir(tmp_path)
    manifest = _write_manifest(tmp_path, output_root)
    output, raw = _summary_pair(tmp_path)
    _plant_stale_pair(output, raw)
    config = tmp_path / "config.yaml"
    if payload is not None:
        _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        assert _run_aggregate_manifest(manifest, output_root, output, config) == 1

    _logged(caplog, fragment)
    _assert_pair_withdrawn(output, raw)


@pytest.mark.parametrize(
    "label,payload,fragment", UNUSABLE_CONFIGS, ids=[case[0] for case in UNUSABLE_CONFIGS]
)
def test_explicit_input_aggregation_refuses_an_unrequested_policy(
    tmp_path, caplog, no_producer, label, payload, fragment
):
    table = tmp_path / "course_table.parquet"
    table.write_bytes(b"not read: the gate runs first")
    output, raw = _summary_pair(tmp_path)
    _plant_stale_pair(output, raw)
    config = tmp_path / "config.yaml"
    if payload is not None:
        _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        assert _run_aggregate_inputs([table], output, config) == 1

    _logged(caplog, fragment)
    _assert_pair_withdrawn(output, raw)


def test_manifest_mode_without_an_output_root_withdraws_the_stale_pair(
    tmp_path, caplog, no_producer
):
    manifest = _write_manifest(tmp_path, tmp_path / "Output")
    output, raw = _summary_pair(tmp_path)
    _plant_stale_pair(output, raw)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert (
            cli.main(
                [
                    "radiomics-robustness-aggregate",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ]
            )
            == 1
        )

    _logged(caplog, "--manifest requires --output-root")
    _assert_pair_withdrawn(output, raw)


def test_explicit_inputs_with_an_output_root_withdraw_the_stale_pair(
    tmp_path, caplog, no_producer
):
    table = tmp_path / "course_table.parquet"
    table.write_bytes(b"unused")
    output, raw = _summary_pair(tmp_path)
    _plant_stale_pair(output, raw)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert (
            cli.main(
                [
                    "radiomics-robustness-aggregate",
                    "--inputs",
                    str(table),
                    "--output-root",
                    str(tmp_path / "Output"),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ]
            )
            == 1
        )

    _logged(caplog, "--output-root belongs to manifest mode")
    _assert_pair_withdrawn(output, raw)


def test_a_symlinked_cohort_output_is_refused_and_its_target_survives(
    tmp_path, caplog, no_producer
):
    real_results = tmp_path / "real_results"
    real_results.mkdir()
    real_output, real_raw = rr.robustness_cohort_output_paths(
        real_results / "radiomics_robustness_summary.xlsx"
    )
    _plant_stale_pair(real_output, real_raw)
    linked_results = tmp_path / "_RESULTS"
    linked_results.symlink_to(real_results, target_is_directory=True)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert (
            _run_aggregate_inputs(
                [tmp_path / "table.parquet"],
                linked_results / "radiomics_robustness_summary.xlsx",
                config,
            )
            == 1
        )

    _logged(caplog, "is reached through the symlink", str(linked_results))
    # Refusing must not delete through the link.
    assert real_output.read_bytes() == b"stale cohort workbook"
    assert real_raw.read_bytes() == b"stale cohort raw values"


def test_a_symlinked_output_root_is_refused(tmp_path, caplog, no_producer):
    real_output_root = tmp_path / "real_output"
    real_output_root.mkdir()
    manifest = _write_manifest(tmp_path, real_output_root)
    linked_root = tmp_path / "Output"
    linked_root.symlink_to(real_output_root, target_is_directory=True)
    # The workbook itself sits outside the linked root, so the refusal can only
    # be about --output-root.
    output, raw = _summary_pair(tmp_path, base="results")
    _plant_stale_pair(output, raw)
    config = _enabled_config(tmp_path)

    with caplog.at_level("ERROR"):
        assert _run_aggregate_manifest(manifest, linked_root, output, config) == 1

    _logged(caplog, "cohort output root", str(linked_root))
    # The gate runs before any withdrawal, so the pair is untouched here.
    assert output.exists() and raw.exists()


def test_a_valid_configuration_still_routes_manifest_aggregation(tmp_path, monkeypatch):
    """The manifest course loop and the cohort call shape are unchanged."""
    output_root = tmp_path / "Output"
    course_dir = _course_dir(tmp_path)
    manifest = _write_manifest(tmp_path, output_root)
    output, _ = _summary_pair(tmp_path)
    config = _enabled_config(tmp_path)
    cohort_metadata = {"validated_course_count": 1}

    monkeypatch.setattr(
        cm,
        "read_course_manifest",
        lambda path, *, output_dir, require_current_schema: (
            [("P1", "C1", output_dir / "P1" / "C1")],
            cohort_metadata,
        ),
    )
    monkeypatch.setattr(
        rr,
        "admit_robustness_cohort_course",
        lambda course, *, patient_id, course_id, rob_config: (
            course,
            patient_id,
            course_id,
            rob_config,
        ),
    )
    calls = []
    monkeypatch.setattr(
        rr,
        "aggregate_robustness_cohort",
        lambda admitted, out, cfg, *, cohort: calls.append((admitted, out, cfg, cohort)),
    )

    assert _run_aggregate_manifest(manifest, output_root, output, config) == 0

    (admitted, out, cfg, cohort), = calls
    assert out == output.resolve()
    assert cohort is cohort_metadata
    assert cfg.enabled is True
    assert admitted == [(course_dir, "P1", "C1", cfg)]


def test_a_valid_configuration_still_routes_explicit_input_aggregation(tmp_path, monkeypatch):
    table = tmp_path / "course_table.parquet"
    table.write_bytes(b"unused")
    output, _ = _summary_pair(tmp_path)
    config = _enabled_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        rr,
        "aggregate_robustness_results",
        lambda inputs, out, cfg: calls.append((inputs, out, cfg)),
    )

    assert _run_aggregate_inputs([table], output, config) == 0

    (inputs, out, cfg), = calls
    assert inputs == [table]
    assert out == output.resolve()
    assert cfg.enabled is True


def test_a_failed_explicit_input_aggregation_withdraws_the_pair(tmp_path, monkeypatch):
    table = tmp_path / "course_table.parquet"
    table.write_bytes(b"unused")
    output, raw = _summary_pair(tmp_path)
    _plant_stale_pair(output, raw)
    config = _enabled_config(tmp_path)

    def _fail(*_args, **_kwargs):
        raise RuntimeError("simulated aggregation failure")

    monkeypatch.setattr(rr, "aggregate_robustness_results", _fail)

    assert _run_aggregate_inputs([table], output, config) == 1
    _assert_pair_withdrawn(output, raw)


# ---------------------------------------------------------------------------
# The workflow rules
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([^{}]+)\}")


def _snakefile() -> str:
    return (ROOT / "Snakefile").read_text(encoding="utf-8")


def _rule_texts(marker: str, end_marker: str) -> list[str]:
    text = _snakefile()
    starts = [index for index in range(len(text)) if text.startswith(marker, index)]
    assert starts, f"no rule matching {marker!r}"
    bodies = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else text.index(end_marker)
        bodies.append(text[start:end])
    return bodies


def _shell_body(rule_text: str) -> str:
    start = rule_text.index("    shell:")
    opening = rule_text.index('"""', start) + 3
    return rule_text[opening : rule_text.index('"""', opening)]


def _render(body: str, values: dict) -> str:
    def _substitute(match):
        if match.group(0) == "{{":
            return "{"
        if match.group(0) == "}}":
            return "}"
        name = match.group(1)
        assert name in values, f"unrendered placeholder {{{name}}} in rule body"
        return values[name]

    return _PLACEHOLDER.sub(_substitute, body)


def _stub_interpreter(tmp_path: Path) -> tuple[Path, Path]:
    """An executable that records any invocation instead of performing one."""
    marker = tmp_path / "interpreter-was-invoked"
    stub = tmp_path / "bin" / "python-stub"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" >> "{marker}"\n'
        "exit 97\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub, marker


def _run_shell(script: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT_SECONDS,
        env={"PATH": "/usr/bin:/bin", "HOME": str(cwd)},
    )


def _aggregate_rule() -> str:
    (rule,) = _rule_texts("rule aggregate_radiomics_robustness:", "rule aggregate_results:")
    return rule


def _course_rules() -> list[str]:
    bodies = _rule_texts(
        "rule radiomics_robustness_course:", "rule aggregate_radiomics_robustness:"
    )
    assert len(bodies) == 2, "expected the container and local rule variants"
    return bodies


def test_every_robustness_rule_body_is_valid_bash(tmp_path):
    """Rendered shell only: nothing here starts Snakemake or the pipeline."""
    values = {
        "log": str(tmp_path / "logs" / "rule.log"),
        "output.sentinel": str(tmp_path / "P1" / "C1" / SENTINEL_NAME),
        "output.summary": str(tmp_path / "_RESULTS" / "summary.xlsx"),
        "input.radiomics": str(tmp_path / "P1" / "C1" / ".radiomics_done"),
        "input.manifest": str(tmp_path / "_COURSES" / "manifest.json"),
        "threads": "1",
        "params.enabled": "True",
        "params.robustness_enabled": "True",
        "params.config": str(tmp_path / "config.yaml"),
        "params.configfile": str(tmp_path / "config.yaml"),
        "params.course_dir": str(tmp_path / "P1" / "C1"),
        "params.parquet": str(tmp_path / "P1" / "C1" / OUTPUT_NAME),
        "params.raw_values": str(tmp_path / "_RESULTS" / "summary_raw_values.parquet"),
        "params.output_dir": str(tmp_path),
        "params.root_dir": str(ROOT),
        "params.python": "/nonexistent/python",
        "params.python_bin": str(tmp_path / "bin"),
    }
    for index, rule in enumerate([*_course_rules(), _aggregate_rule()]):
        script = tmp_path / f"rule_{index}.sh"
        script.write_text(_render(_shell_body(rule), values), encoding="utf-8")
        syntax = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_the_disabled_aggregate_branch_fails_closed_and_withdraws_the_stale_pair(tmp_path):
    """A stale nominal pair plus a disabled explicit target: no blank success."""
    results = tmp_path / "_RESULTS"
    results.mkdir()
    summary, raw = rr.robustness_cohort_output_paths(
        results / "radiomics_robustness_summary.xlsx"
    )
    _plant_stale_pair(summary, raw)
    stub, marker = _stub_interpreter(tmp_path)
    log = tmp_path / "logs" / "aggregate.log"

    script = tmp_path / "aggregate_disabled.sh"
    script.write_text(
        _render(
            _shell_body(_aggregate_rule()),
            {
                "log": str(log),
                "output.summary": str(summary),
                "input.manifest": str(tmp_path / "manifest.json"),
                "params.robustness_enabled": "False",
                "params.raw_values": str(raw),
                "params.output_dir": str(tmp_path),
                "params.root_dir": str(ROOT),
                "params.configfile": str(tmp_path / "config.yaml"),
                "params.python": str(stub),
                "params.python_bin": str(stub.parent),
            },
        ),
        encoding="utf-8",
    )

    result = _run_shell(script, tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert not summary.exists(), "a disabled cohort target left a nominal workbook"
    assert not raw.exists(), "a disabled cohort target left nominal raw values"
    assert "disabled" in log.read_text(encoding="utf-8")
    assert "refusing to publish" in result.stderr
    assert not marker.exists(), "the disabled branch invoked an interpreter"


@pytest.mark.parametrize("variant", [0, 1], ids=["container", "local"])
def test_the_disabled_course_branch_keeps_no_success_receipt(tmp_path, variant):
    course_dir = tmp_path / "P1" / "C1"
    course_dir.mkdir(parents=True)
    _plant_legitimate_receipt(course_dir)
    sentinel = course_dir / SENTINEL_NAME
    stub, marker = _stub_interpreter(tmp_path)

    script = tmp_path / "course_disabled.sh"
    script.write_text(
        _render(
            _shell_body(_course_rules()[variant]),
            {
                "log": str(tmp_path / "logs" / "course.log"),
                "output.sentinel": str(sentinel),
                "input.radiomics": str(course_dir / ".radiomics_done"),
                "threads": "1",
                "params.enabled": "False",
                "params.config": str(tmp_path / "config.yaml"),
                "params.course_dir": str(course_dir),
                "params.parquet": str(course_dir / OUTPUT_NAME),
                "params.python": str(stub),
                "params.python_bin": str(stub.parent),
            },
        ),
        encoding="utf-8",
    )

    result = _run_shell(script, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text(encoding="utf-8").strip() == "disabled"
    with pytest.raises(rc.RobustnessCompletionError, match="disabled"):
        rc.read_robustness_completion_sentinel(sentinel)
    assert not marker.exists(), "the disabled branch invoked an interpreter"


def test_the_rules_raw_value_name_matches_the_published_cohort_pair(tmp_path):
    """The Snakefile spells the raw-value name; this keeps it bound to the code."""
    summary = tmp_path / "_RESULTS" / "radiomics_robustness_summary.xlsx"
    _, expected_raw = rr.robustness_cohort_output_paths(summary)
    rule = _aggregate_rule()
    declaration = rule[
        rule.index("raw_values=lambda") : rule.index("python=PYTHON_MAIN")
    ].rstrip().rstrip(",")
    assert "_raw_values.parquet" in declaration
    raw_values = eval(  # noqa: S307 - the rule's own text, evaluated verbatim
        declaration[len("raw_values=") :], {"Path": Path}
    )
    assert Path(raw_values(None, SimpleNamespace(summary=str(summary)))) == expected_raw


def test_a_disabled_full_workflow_does_not_require_the_cohort_target():
    """Failing closed on the explicit target must not break the normal run."""
    text = _snakefile()
    assert (
        'if ROBUSTNESS_ENABLED:\n'
        '    AGG_OUTPUTS["radiomics_robustness"] = RESULTS_DIR / "radiomics_robustness_summary.xlsx"'
    ) in text
    rule_all = text[text.index("rule all:") : text.index("checkpoint organize_courses:")]
    assert "*(str(path) for path in AGG_OUTPUTS.values())" in rule_all
    assert "radiomics_robustness" not in rule_all
