"""Synthetic regressions for selected membership versus artifact provenance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rtpipeline.organize import _reconcile_published_plan_dispositions


def course(selected, artifact, patient="P1", course_id="C1"):
    return SimpleNamespace(
        patient_id=patient, course_id=course_id,
        selected_plan_contract=[{"sop_instance_uid": uid} for uid in selected],
        source_plan_uids=list(artifact),
    )


def member(uid="A", patient="P1"):
    return {"patient": patient, "plan_uid": uid, "course_key": "source-key",
            "disposition_type": "course_member", "reason_code": "TARGET_BEARING_COURSE_MEMBER",
            "clinical_exclusion": False}


def reconcile(courses, rows):
    payload = {"plans": rows, "metadata": "preserved"}
    original = deepcopy(payload)
    original_courses = deepcopy([vars(c) for c in courses])
    result = _reconcile_published_plan_dispositions(courses, payload)
    assert payload == original
    assert [vars(c) for c in courses] == original_courses
    assert result["metadata"] == "preserved"
    return result["plans"]


@pytest.mark.parametrize("selected,artifact", [
    (["A", "B"], ["A"]),  # copied root, multiple selected plans
    (["A"], ["A"]),
    (["A", "B"], ["A", "B"]),  # derived sum
    (["A", "A"], ["A"]),  # repeat within one course is harmless
])
def test_selected_members_survive(selected, artifact):
    rows = reconcile([course(selected, artifact)], [member(uid) for uid in set(selected)])
    assert all(row["disposition_type"] == "course_member" for row in rows)
    assert all(row["course_id"] == "C1" for row in rows)


@pytest.mark.parametrize("courses", [
    [],
    [course(["A"], ["A"], patient="P2")],
    [course([], ["A"])],  # never fall back to artifact provenance
])
def test_unpublished_members_are_held(courses):
    row = member()
    row["course_id"] = "stale"
    result, = reconcile(courses, [row])
    assert result["disposition_type"] == "technical_hold"
    assert result["reason_code"] == "COURSE_NOT_PUBLISHED"
    assert result["clinical_exclusion"] is False
    assert "course_id" not in result


def test_cross_course_ambiguity_is_order_independent():
    courses = [course(["A"], ["A"], course_id=c) for c in ("C1", "C2")]
    row = member()
    row["course_id"] = "stale"
    first = reconcile(courses, [row])
    assert reconcile(list(reversed(courses)), [row]) == first
    result, = first
    assert result["disposition_type"] == "technical_hold"
    assert result["reason_code"] == "COURSE_PUBLICATION_AMBIGUOUS"
    assert result["clinical_exclusion"] is False
    assert "course_id" not in result


@pytest.mark.parametrize("state", ["technical_hold", "non_measurable_delivery", "excluded"])
def test_nonmembers_remain_unchanged(state):
    row = member()
    row.update(disposition_type=state, reason_code="existing-reason")
    assert reconcile([course(["A"], ["A"])], [row]) == [row]
