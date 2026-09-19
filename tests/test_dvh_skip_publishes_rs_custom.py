"""RS_custom.dcm is a bound DVH output on every DVH exit path.

A course without an authoritative dose grid emits no dose metrics, but it still
reaches radiomics. Radiomics rebuilds a missing RS_custom.dcm, and that write
lands after the DVH completion sentinel is published, leaving the sentinel
declaring an incomplete output set and blocking cohort aggregation.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rtpipeline import custom_structures_rtstruct, dvh


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_publish_uses_the_sources_radiomics_would_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course_dir = tmp_path / "patient" / "course"
    rs_auto = _touch(course_dir / "RS_auto.dcm")
    rs_manual = _touch(course_dir / "RS.dcm")
    config = _touch(tmp_path / "custom_structures.yaml", "custom_structures: []")

    calls: list[tuple] = []
    monkeypatch.setattr(
        custom_structures_rtstruct,
        "_is_rs_custom_stale",
        lambda *args: True,
    )
    monkeypatch.setattr(
        custom_structures_rtstruct,
        "_create_custom_structures_rtstruct",
        lambda *args: calls.append(args) or (course_dir / "RS_custom.dcm"),
    )

    dvh._publish_skipped_course_rs_custom(course_dir, config, rs_manual)

    assert calls == [(course_dir, config, rs_manual, rs_auto)]


def test_publish_is_a_no_op_when_rs_custom_is_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never rewrite bytes another stage already published."""

    course_dir = tmp_path / "patient" / "course"
    rs_manual = _touch(course_dir / "RS.dcm")
    _touch(course_dir / "RS_auto.dcm")
    config = _touch(tmp_path / "custom_structures.yaml", "custom_structures: []")

    monkeypatch.setattr(
        custom_structures_rtstruct,
        "_is_rs_custom_stale",
        lambda *args: False,
    )

    def _fail(*args):
        raise AssertionError("a current RS_custom.dcm must not be rebuilt")

    monkeypatch.setattr(
        custom_structures_rtstruct, "_create_custom_structures_rtstruct", _fail
    )

    dvh._publish_skipped_course_rs_custom(course_dir, config, rs_manual)


def test_publish_without_a_configuration_does_nothing(tmp_path: Path) -> None:
    course_dir = tmp_path / "patient" / "course"
    rs_manual = _touch(course_dir / "RS.dcm")

    dvh._publish_skipped_course_rs_custom(course_dir, None, rs_manual)

    assert not (course_dir / "RS_custom.dcm").exists()


def test_a_builder_failure_does_not_fail_the_skipped_course(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course_dir = tmp_path / "patient" / "course"
    rs_manual = _touch(course_dir / "RS.dcm")
    _touch(course_dir / "RS_auto.dcm")
    config = _touch(tmp_path / "custom_structures.yaml", "custom_structures: []")

    monkeypatch.setattr(
        custom_structures_rtstruct, "_is_rs_custom_stale", lambda *args: True
    )

    def _raise(*args):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(
        custom_structures_rtstruct, "_create_custom_structures_rtstruct", _raise
    )

    dvh._publish_skipped_course_rs_custom(course_dir, config, rs_manual)


@pytest.mark.parametrize(
    "dose_resolution, expected_reason",
    [
        (
            SimpleNamespace(
                ok=False,
                dose_qc_pass=True,
                dose_qc_reasons=[],
                skip_reason="no dose grid",
                reason="no dose grid",
            ),
            "no dose grid",
        ),
        (
            SimpleNamespace(
                ok=True,
                dose_qc_pass=False,
                dose_qc_reasons=["dose QC failed"],
                skip_reason=None,
                reason="qc",
            ),
            "dose QC failed",
        ),
    ],
)
def test_dose_absent_course_publishes_rs_custom_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dose_resolution: SimpleNamespace,
    expected_reason: str,
) -> None:
    """The regression: the skip path used to return before publishing."""

    course_dir = tmp_path / "patient" / "course"
    rs_manual = _touch(course_dir / "RS.dcm")
    config = _touch(tmp_path / "custom_structures.yaml", "custom_structures: []")

    contract = SimpleNamespace(
        treatment_technique={"classification": "VMAT"},
        delivery={},
        authoritative_rtstruct_path=rs_manual,
    )
    monkeypatch.setattr(dvh, "load_course_contract", lambda _dir: contract)
    monkeypatch.setattr(dvh, "_contract_dose_response_eligible", lambda _c: True)
    monkeypatch.setattr(dvh, "build_course_dirs", lambda _dir: SimpleNamespace())
    monkeypatch.setattr(dvh, "_resolve_dvh_dose", lambda *a, **k: dose_resolution)
    monkeypatch.setattr(dvh, "_invalidate_dvh_outputs", lambda _dir: None)
    monkeypatch.setattr(dvh, "_write_dvh_skip_qc", lambda *a, **k: None)

    published: list[tuple] = []
    monkeypatch.setattr(
        dvh,
        "_publish_skipped_course_rs_custom",
        lambda *args: published.append(args),
    )

    assert dvh.dvh_for_course(course_dir, custom_structures_config=config) is None
    assert published == [(course_dir, config, rs_manual)]
