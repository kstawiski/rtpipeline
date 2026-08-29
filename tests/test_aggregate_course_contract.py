from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
)
from rtpipeline.dvh import dvh_for_course


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "workflow" / "scripts" / "aggregate_results.py"


def _aggregate_functions():
    tree = ast.parse(AGGREGATE.read_text(encoding="utf-8"), filename=str(AGGREGATE))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(AGGREGATE), "exec"), namespace)
    namespace["RADIOMICS_ENABLED"] = False
    return namespace


def _validate(namespace: dict[str, object], courses: list[tuple[str, str, Path]]):
    validator = namespace["_validate_required_inputs"]
    assert callable(validator)
    return validator(courses)


def _course(tmp_path: Path) -> Path:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )
    for name in (".dvh_done", ".qc_done", ".custom_models_done"):
        (course / name).write_text("ok", encoding="utf-8")
    pd.DataFrame({"ROI_Name": ["PTV"]}).to_excel(
        course / "dvh_metrics.xlsx", index=False
    )
    return course


def test_aggregation_excludes_a_missing_contract_with_a_reason(tmp_path: Path) -> None:
    namespace = _aggregate_functions()
    course = tmp_path / "P1" / "C1"
    course.mkdir(parents=True)

    frames, errors, incomplete, noncomputed = _validate(
        namespace, [("P1", "C1", course)]
    )

    assert frames == {}
    assert ("P1", "C1") in incomplete
    assert noncomputed == {}
    assert "authoritative course contract" in errors[0]


def test_aggregation_accepts_a_valid_contract_and_required_outputs(tmp_path: Path) -> None:
    namespace = _aggregate_functions()
    course = _course(tmp_path)

    frames, errors, incomplete, noncomputed = _validate(
        namespace, [("P1", "C1", course)]
    )

    assert errors == []
    assert incomplete == {}
    assert noncomputed == {}
    assert frames[(course, "dvh")].iloc[0]["ROI_Name"] == "PTV"


def test_aggregation_excludes_a_contract_with_failed_dose_qc(tmp_path: Path) -> None:
    namespace = _aggregate_functions()
    course = _course(tmp_path)
    metadata = course / "metadata" / "case_metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["course_contract"]["delivery"].update(
        {
            "status": "no_records_at_all",
            "prescribed_dose_gy": 105.0,
            "delivered_dose_gy": None,
        }
    )
    payload["course_contract"]["dose_qc"] = {
        "status": "fail",
        "pass": False,
        "threshold_gy": 100.0,
        "reasons": ["implausible dose total"],
    }
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    frames, errors, incomplete, noncomputed = _validate(
        namespace, [("P1", "C1", course)]
    )

    assert frames == {}
    assert ("P1", "C1") in incomplete
    assert noncomputed == {}
    assert "dose QC failed" in errors[0]
    assert "implausible dose total" in errors[0]


def test_aggregation_excludes_stale_contract_identity(tmp_path: Path) -> None:
    namespace = _aggregate_functions()
    course = _course(tmp_path)
    metadata = course / "metadata" / "case_metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["course_contract"]["course_id"] = "different-course"
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    frames, errors, incomplete, noncomputed = _validate(
        namespace, [("P1", "C1", course)]
    )

    assert frames == {}
    assert ("P1", "C1") in incomplete
    assert noncomputed == {}
    assert "authoritative course contract" in errors[0]


@pytest.mark.parametrize("sentinel", [".dvh_done", ".qc_done", ".custom_models_done"])
def test_aggregation_still_requires_stage_sentinels(tmp_path: Path, sentinel: str) -> None:
    namespace = _aggregate_functions()
    course = _course(tmp_path)
    (course / sentinel).unlink()

    _frames, errors, incomplete, _noncomputed = namespace["_validate_required_inputs"](
        [("P1", "C1", course)]
    )

    assert ("P1", "C1") in incomplete
    assert any(sentinel in error for error in errors)


def test_aggregation_accepts_plan_only_not_computed_and_records_reason(
    tmp_path: Path,
) -> None:
    namespace = _aggregate_functions()
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    dose.unlink()
    write_minimal_course_contract(course, selected_plans=[plan], selected_doses=[])
    assert dvh_for_course(course) is None
    for name in (".dvh_done", ".qc_done", ".custom_models_done"):
        (course / name).write_text("ok", encoding="utf-8")

    frames, errors, incomplete, noncomputed = _validate(
        namespace, [("P1", "C1", course)]
    )

    assert errors == []
    assert incomplete == {}
    assert frames[(course, "dvh")].empty
    assert noncomputed == {
        ("P1", "C1"): "plan_only_no_authoritative_dose_grid"
    }

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    namespace["RESULTS_DIR"] = results_dir
    writer = namespace["_write_campaign_attrition"]
    assert callable(writer)
    writer([("P1", "C1", course)], incomplete, noncomputed)
    attrition = pd.read_csv(results_dir / "campaign_attrition.csv")
    assert attrition.loc[0, "status"] == "not_computed"
    assert attrition.loc[0, "reasons"] == "plan_only_no_authoritative_dose_grid"
