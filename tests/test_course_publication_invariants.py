"""Course-level denominator and publication invariants around delivered dose.

Behavioural checks call ``rtpipeline.organize`` imported from the checkout
that contains this directory. Static checks read the same checkout and compare
selected definitions and supporting modules with digests pinned from the
sources reviewed before the record-delivery adjudication change. Every fixture
value is synthetic.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rtpipeline import organize as org
from record_delivery_synthetic_rt import PER_FRACTION, PLAN_A, PLAN_B, Event, complete_session, make_plan

ROOT = Path(__file__).resolve().parents[1]

# Canonical AST digests (see _ast_digest) of definitions the adjudication change
# must leave untouched. Re-pin one only after reviewing a deliberate change to it.
BASE_FUNCTION_AST_SHA256 = {
    ("organize", "_reconcile_published_plan_dispositions"): "323073372fbaedd32b1a7ebf2b051ba134d53c33675ecb68ffe17866cba20e18",
    ("organize", "_scope_aware_course_dose_publication"): "193664705e1cb29ad12cf12b158d357d8d265e3bbd1b5003609f894c9f494fe7",
    ("organize", "_course_dose_response_eligible"): "55873e6db473d70d9a0d0241ff6a4bbe796faebf1866903768630587cfb03625",
    ("organize", "_clinical_delivery_publication"): "3f10beb7a07faa7323cf293a8191cc17fc004d412c9903b14b493a5fd27c699c",
    ("organize", "_per_plan_delivery_contract"): "a49ae005ca8cabc4e31b23fe0bcfcab91db116a2e0f9fe79e5b2b5d87647e358",
    ("course_contract", "build_dvh_decision"): "b903f7d6092a61eff3a05385e0438bd0eb6bed9379c13bf1f72cf65953a26c22",
    ("course_contract", "classify_course_dose_completeness"): "4f06cdfa1c98ecd91dd65561ea3f624510adad4b6c3a1f9854f7d9b96afd2fe6",
    ("course_contract", "_record_delivery_session_evidence"): "4e48f8be957c5090c4ae6dd8e426c7473c008f39583bef9f4c4ca6a37182fd56",
    ("course_contract", "_record_delivery_session_key"): "073f56044f63b4bb231c24018b4fe4abb945fd3843af2b360a89586eda974c08",
}

# SHA-256 of supporting modules the adjudication change must leave byte-identical.
BASE_MODULE_SHA256 = {
    "course_contract.py": "28c4b43b9b4562ae5dc7343b645ef4f0929088e4acc5635edd2559098295187b",
    "prescription.py": "723c8657c60f48c50b0d31decf0eb458c93d435f597c56724ee748c1b264ce7a",
    "plan_approval.py": "fc213a8ef85e9ec2698c6bfec7bee3877df61b6b7d835257038cd2a470c21aed",
}


def _checked_root() -> Path:
    location = Path(org.__file__).resolve()
    if location != ROOT / "rtpipeline" / "organize.py":
        pytest.fail(f"rtpipeline.organize was imported from {location}, not from the checkout at {ROOT}")
    return ROOT


@pytest.fixture(scope="module")
def organize():
    """Look up definitions on the organize module imported from this checkout."""
    _checked_root()
    return lambda name: getattr(org, name)


@pytest.fixture(scope="module")
def source_root() -> Path:
    """The checkout whose organize module the behavioural tests call."""
    return _checked_root()


def _ast_digest(node: ast.AST) -> str:
    """SHA-256 of an AST without positions, leaving out empty fields.

    Empty fields are left out so that fields added by later Python releases,
    such as ``type_params``, do not change the digest of unchanged code.
    """

    def canonical(value):
        if isinstance(value, ast.AST):
            fields = []
            for name in value._fields:
                item = getattr(value, name, None)
                if item is None or (isinstance(item, list) and not item):
                    continue
                fields.append((name, canonical(item)))
            return (type(value).__name__, tuple(fields))
        if isinstance(value, list):
            return tuple(canonical(item) for item in value)
        return (type(value).__name__, repr(value))

    return hashlib.sha256(repr(canonical(node)).encode("utf-8")).hexdigest()


def _function(root: Path, module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((root / "rtpipeline" / f"{module}.py").read_text())
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, name
    return matches[0]


def _detail(summary, plan_uid):
    return next(item for item in summary["delivery_plan_details"] if item["plan_sop_uid"] == plan_uid)


# Concern (e): replacement-chain fraction denominator.


@pytest.mark.concern_e
def test_remainder_chain_withholds_course_scalar_and_keeps_plan_doses(tmp_path: Path, organize) -> None:
    anchor = make_plan(tmp_path / "anchor.dcm", fractions=9, target_rx=19.8)
    remainder = make_plan(tmp_path / "remainder.dcm", plan_uid=PLAN_B)
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    for fraction in range(1, 8):
        records += complete_session(tmp_path, fraction, plan_uid=PLAN_B, day=10 + fraction)
    summarize = organize("_calculate_delivery_summary")

    unclassified = summarize([anchor, remainder], records)
    assert unclassified["delivered_dose_gy"] == pytest.approx(9 * PER_FRACTION)
    assert unclassified["planned_fraction_count"] == 16
    assert unclassified["delivery_status"] == "partially_delivered"

    chain = summarize([anchor, remainder], records, classification=organize("DELIVERED_REMAINDER_CHAIN"))
    assert [hold["reason_code"] for hold in chain["delivery_course_holds"]] == ["COURSE_REPLACEMENT_DENOMINATOR_UNADJUDICATED"]
    assert chain["delivered_dose_gy"] is None
    assert chain["delivery_status"] == "delivery_unresolved"
    assert chain["delivery_method"] == "course_dose_adjudication_hold"
    assert _detail(chain, PLAN_A)["delivered_dose_gy"] == pytest.approx(2 * PER_FRACTION)
    assert _detail(chain, PLAN_B)["delivered_dose_gy"] == pytest.approx(7 * PER_FRACTION)


@pytest.mark.concern_e
def test_course_fraction_totals_are_withheld_for_remainder_and_independent_memberships(organize) -> None:
    totals = organize("_course_fraction_totals")
    remainder = totals(organize("DELIVERED_REMAINDER_CHAIN"), 9, 16)
    assert remainder["delivered_fraction_count"] is None
    assert remainder["planned_fraction_count"] is None
    assert remainder["basis"] == "withheld_replacement_denominator_unadjudicated"
    independent = totals(organize("INDEPENDENT_DELIVERY_UNRECONCILED"), 9, 16)
    assert independent["basis"] == "withheld_independent_delivery_unreconciled"
    phases = totals("sequential_phases_summed", 9, 16)
    assert (phases["delivered_fraction_count"], phases["planned_fraction_count"]) == (9, 16)
    assert phases["basis"] == "selected_membership_sum"
    assert set(organize("COURSE_FRACTION_TOTALS_WITHHELD_BASES")) == {
        remainder["basis"], independent["basis"],
    }


@pytest.mark.concern_e
def test_classifier_and_summary_share_the_remainder_chain_constant(source_root: Path) -> None:
    classify = _function(source_root, "organize", "_classify_approved_doses")
    literals = {node.value for node in ast.walk(classify) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "delivered_remainder_plan_doses_accumulated" not in literals
    names = {node.id for node in ast.walk(classify) if isinstance(node, ast.Name)}
    assert "DELIVERED_REMAINDER_CHAIN" in names


# Concern (f): membership, provenance and publication invariants.


@pytest.mark.concern_f
def test_independent_delivery_classification_never_sums_gy(tmp_path: Path, organize) -> None:
    first = make_plan(tmp_path / "a.dcm")
    second = make_plan(tmp_path / "b.dcm", plan_uid=PLAN_B)
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    records += complete_session(tmp_path, 1, plan_uid=PLAN_B) + complete_session(tmp_path, 2, plan_uid=PLAN_B)
    summary = organize("_calculate_delivery_summary")(
        [first, second], records, classification=organize("INDEPENDENT_DELIVERY_UNRECONCILED")
    )
    assert [hold["reason_code"] for hold in summary["delivery_course_holds"]] == ["COURSE_INDEPENDENT_DELIVERY_UNRECONCILED"]
    assert summary["delivered_dose_gy"] is None
    assert _detail(summary, PLAN_A)["delivered_fraction_count"] == 2
    assert _detail(summary, PLAN_B)["delivered_fraction_count"] == 2


@pytest.mark.concern_f
@pytest.mark.parametrize("scope", ["UNRESOLVED_INDEPENDENT_DELIVERY", "UNRESOLVED_REPLACEMENT_CHAIN", "UNRESOLVED_COMPONENT"])
def test_unresolved_course_scope_publication_withholds_delivered_gy(organize, scope: str) -> None:
    published = organize("_scope_aware_course_dose_publication")(
        prescribed_dose_scope=scope,
        course_prescribed_dose_gy=19.8,
        course_resolved_prescribed_dose_total_gy=19.8,
        plan_prescribed_dose_gy=15.4,
        plan_resolved_prescribed_dose_total_gy=15.4,
        delivered_dose_gy=8.8,
        delivery_status="partially_delivered",
        delivery_method="record_fraction_weighted_prescription",
    )
    assert published["delivered_dose_gy"] is None
    assert published["delivery_status"] == "delivery_unresolved"


@pytest.mark.concern_f
def test_shared_selected_plan_uid_across_courses_is_publication_ambiguous(organize) -> None:
    reconcile = organize("_reconcile_published_plan_dispositions")
    shared = "2.25.418006270091000000000000000000000041"
    provenance_only = "2.25.418006270091000000000000000000000042"
    courses = [
        SimpleNamespace(patient_id="SYN", course_id="course_one",
                        selected_plan_contract=[{"sop_instance_uid": shared}], source_plan_uids=[provenance_only]),
        SimpleNamespace(patient_id="SYN", course_id="course_two",
                        selected_plan_contract=[{"sop_instance_uid": shared}], source_plan_uids=[provenance_only]),
    ]
    rows = {"plans": [
        {"patient": "SYN", "plan_uid": shared, "disposition_type": "course_member"},
        {"patient": "SYN", "plan_uid": provenance_only, "disposition_type": "course_member"},
    ]}
    result = reconcile(courses, rows)
    by_uid = {row["plan_uid"]: row for row in result["plans"]}
    assert by_uid[shared]["reason_code"] == "COURSE_PUBLICATION_AMBIGUOUS"
    assert by_uid[provenance_only]["reason_code"] == "COURSE_NOT_PUBLISHED"
    assert rows["plans"][0]["disposition_type"] == "course_member"


@pytest.mark.concern_f
@pytest.mark.parametrize(
    "module, name",
    [
        ("organize", "_reconcile_published_plan_dispositions"),
        ("organize", "_scope_aware_course_dose_publication"),
        ("organize", "_course_dose_response_eligible"),
        ("organize", "_clinical_delivery_publication"),
        ("organize", "_per_plan_delivery_contract"),
        ("course_contract", "build_dvh_decision"),
        ("course_contract", "classify_course_dose_completeness"),
        ("course_contract", "_record_delivery_session_evidence"),
        ("course_contract", "_record_delivery_session_key"),
    ],
)
def test_publication_and_membership_functions_are_unchanged(source_root: Path, module: str, name: str) -> None:
    assert _ast_digest(_function(source_root, module, name)) == BASE_FUNCTION_AST_SHA256[(module, name)]


@pytest.mark.concern_f
@pytest.mark.parametrize("module", ["course_contract.py", "prescription.py", "plan_approval.py"])
def test_supporting_modules_are_byte_identical(source_root: Path, module: str) -> None:
    digest = lambda root: hashlib.sha256((root / "rtpipeline" / module).read_bytes()).hexdigest()  # noqa: E731
    assert digest(source_root) == BASE_MODULE_SHA256[module]


@pytest.mark.concern_f
@pytest.mark.parametrize("name", ["_calculate_delivery_summary", "_adjudicate_plan_record_delivery", "_record_reference_binding"])
def test_dose_functions_use_membership_not_provenance(source_root: Path, name: str) -> None:
    node = _function(source_root, "organize", name)
    text = ast.unparse(node)
    assert "source_plan_uids" not in text
    assert "metrics_status" not in text


@pytest.mark.concern_e
@pytest.mark.concern_f
def test_course_organizer_passes_classification_to_delivery_summary(source_root: Path) -> None:
    tree = ast.parse((source_root / "rtpipeline" / "organize.py").read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_calculate_delivery_summary"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert "classification" in keywords
    assert "dose_classification_info" in ast.unparse(keywords["classification"])


# Adjudicator checks for conflicting copies, SETUP items, exact duplicates and continuations.


def _event(beam: str, *, dose: float, delivery_type: str = "TREATMENT", start: str = "T01"):
    return {
        "event_key": ("beam", "TreatmentSessionBeamSequence", beam, delivery_type, "NORMAL", start),
        "kind": "beam", "identity": beam, "delivery_type": delivery_type, "termination_status": "NORMAL",
        "event_time": start, "delivered_meterset": 10.0,
        "dose_references": [{"reference_number": "7", "dose_gy": dose, "value_present": True}],
    }


def _adjudicate(organize, rows, *, expected=(("beam", "1"),), per_fraction=None):
    return organize("_adjudicate_plan_record_delivery")(
        rows, validated_sessions={"S"}, expected_items=list(expected),
        expected_items_status="DEFINED_SINGLE_FRACTION_GROUP", reference_numbers={"7"},
        per_fraction_reference_gy=per_fraction,
    )


def _rows(*event_lists):
    return [{"session_key": "S", "delivery_events": list(events), "session_components": []} for events in event_lists]


@pytest.mark.helper_parity
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_helper_parity_conflicting_copies_are_rejected(organize, order: str) -> None:
    low, high = _event("1", dose=0.41), _event("1", dose=0.44)
    rows = _rows([low], [high]) if order == "forward" else _rows([high], [low])
    result = _adjudicate(organize, rows, per_fraction=0.42)
    assert result["record_dose_status"] == "CONTRADICTED"
    assert result["record_dose_gy"] is None


@pytest.mark.helper_parity
def test_helper_parity_setup_items_are_not_events(tmp_path: Path, organize) -> None:
    import pydicom

    from record_delivery_synthetic_rt import make_record

    record = make_record(tmp_path / "setup_mix.dcm", events=[
        Event(1, dose=0.41, reference_number="7", start="071600"),
        Event(9, delivery_type="SETUP", dose=0.06, reference_number="7", start="070500", meterset=2.0),
    ])
    events = organize("_record_session_delivery_events")(pydicom.dcmread(str(record), force=True))
    assert [(event["identity"], event["delivery_type"]) for event in events] == [("1", "TREATMENT")]
    result = _adjudicate(organize, _rows(events), per_fraction=0.41)
    assert result["record_dose_gy"] == pytest.approx(0.41)


@pytest.mark.helper_parity
def test_helper_parity_exact_duplicate_is_retained_once(organize) -> None:
    first = _event("1", dose=0.41)
    result = _adjudicate(organize, _rows([first], [dict(first)]), per_fraction=0.41)
    assert result["record_dose_status"] == "COMPLETE"
    assert result["record_dose_gy"] == pytest.approx(0.41)


@pytest.mark.helper_parity
def test_helper_parity_distinct_continuation_events_are_preserved(organize) -> None:
    stopped = _event("2", dose=0.12, start="T02")
    stopped["termination_status"] = "MACHINE"
    stopped["event_key"] = ("beam", "TreatmentSessionBeamSequence", "2", "TREATMENT", "MACHINE", "T02")
    rows = _rows([_event("1", dose=0.41)], [stopped], [_event("2", dose=0.31, delivery_type="CONTINUATION", start="T03")])
    result = _adjudicate(organize, rows, expected=(("beam", "1"), ("beam", "2")), per_fraction=0.84)
    assert result["completeness_status"] == "ALL_SESSIONS_COMPLETE"
    assert result["record_dose_gy"] == pytest.approx(0.84)
