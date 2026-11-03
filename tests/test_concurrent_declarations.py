"""Synthetic CSV and selection JSON only; no ARIA export is read."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

# Keep the input fixture schema local so base runs reach each regression even
# before the standalone generator exists.
REQUIRED_COLUMNS = {
    "case_id", "plan_uid", "course_id", "included_in_course", "plan_status",
    "fractions_planned", "fractions_treated", "refpoint_id", "refpoint_planned",
    "refpoint_delivered", "first_day_of_treatment", "last_day_of_treatment",
}


def generate_declarations(*args, **kwargs):
    from scripts.concurrent_declarations import generate_declarations as generate
    return generate(*args, **kwargs)


def row(uid, **changes):
    return dict(case_id="SYNTH_CASE", plan_uid=uid, course_id="SYNTH_COURSE",
                included_in_course="true", plan_status="Completed", fractions_planned="30",
                fractions_treated="30", refpoint_id=f"REF_{uid}", refpoint_planned="60",
                refpoint_delivered="60", first_day_of_treatment="2024-01-01",
                last_day_of_treatment="2024-02-12", **changes)


def files(tmp_path, rows, selection=None):
    history = tmp_path / "history.csv"
    with history.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection if selection is not None else {"SYNTH_CASE": ["1.2.3", "1.2.4"]}))
    return history, selection_path


def test_complete_selected_group_and_basis(tmp_path):
    history, selection = files(tmp_path, [row("1.2.3"), row("1.2.4"), row("1.2.5")])
    payload, counts = generate_declarations(history, selection, "2026-11-02")
    assert counts == dict(rows_read=3, cases_selected=1, declarations_written=1, cases_skipped=0)
    declaration, = payload["declarations"]
    assert declaration["plan_sop_instance_uids"] == ["1.2.3", "1.2.4"]
    assert declaration["declared_on"] == "2026-11-02"
    for fact in ["Completed", "fractions_treated = fractions_planned > 0", "30, 30",
                 "2024-01-01", "2024-02-12", "distinct nonempty refpoint_id", "SYNTH_COURSE"]:
        assert fact in declaration["basis"]


@pytest.mark.parametrize("field,value", [
    ("plan_status", "Approved"), ("plan_status", "completed"),
    ("fractions_treated", "29"), ("fractions_treated", "31"),
    ("fractions_planned", "0"), ("fractions_treated", "-1"),
    ("fractions_planned", "NaN"), ("fractions_treated", "Infinity"),
    ("fractions_planned", "bad"), ("fractions_planned", "30.5"),
    ("first_day_of_treatment", "2024-01-02"), ("first_day_of_treatment", ""),
    ("last_day_of_treatment", "2024-02-13"), ("last_day_of_treatment", ""),
    ("refpoint_id", "REF_1.2.3"), ("refpoint_id", ""),
    ("course_id", "OTHER"), ("course_id", ""), ("included_in_course", "false"),
])
def test_ineligible_case_is_wholly_skipped(tmp_path, field, value):
    second = row("1.2.4")
    second[field] = value
    history, selection = files(tmp_path, [row("1.2.3"), second])
    payload, counts = generate_declarations(history, selection, "2026-11-02")
    assert payload == {"declarations": []}
    assert counts["cases_skipped"] == 1


@pytest.mark.parametrize("mode", ["single", "missing", "duplicate_history", "two_cases"])
def test_missing_ambiguous_or_separate_case_membership(tmp_path, mode):
    rows = [row("1.2.3"), row("1.2.4")]
    selection = {"SYNTH_CASE": ["1.2.3", "1.2.4"]}
    if mode == "single":
        selection["SYNTH_CASE"] = ["1.2.3"]
    elif mode == "missing":
        rows.pop()
    elif mode == "duplicate_history":
        rows.append(row("1.2.4"))
    else:
        rows[1]["case_id"] = "OTHER_CASE"
        selection = {"SYNTH_CASE": ["1.2.3"], "OTHER_CASE": ["1.2.4"]}
    history, selection_path = files(tmp_path, rows, selection)
    payload, _ = generate_declarations(history, selection_path, "2026-11-02")
    assert payload == {"declarations": []}


def test_three_plan_group_different_per_plan_counts_and_decimal_csv(tmp_path):
    rows = [row(uid) for uid in ["1.2.3", "1.2.4", "1.2.5"]]
    rows[1].update(fractions_planned="20.0", fractions_treated="20.0", included_in_course="1")
    rows[2]["included_in_course"] = "yes"
    history, selection = files(tmp_path, rows, {"SYNTH_CASE": [r["plan_uid"] for r in rows]})
    payload, _ = generate_declarations(history, selection, "2026-11-02")
    assert len(payload["declarations"][0]["plan_sop_instance_uids"]) == 3
    assert "30, 20, 30" in payload["declarations"][0]["basis"]


@pytest.mark.parametrize("selection", [[], {"SYNTH_CASE": ["1.2.3", "1.2.3"]}, {"SYNTH_CASE": "1.2.3"},
                                       {" SYNTH_CASE ": ["1.2.3", "1.2.4"]}])
def test_invalid_selection_is_an_error(tmp_path, selection):
    history, selected = files(tmp_path, [row("1.2.3")], selection)
    with pytest.raises(ValueError):
        generate_declarations(history, selected, "2026-11-02")


def test_cli_output_contains_counts_only(tmp_path):
    history, selection = files(tmp_path, [row("1.2.3"), row("1.2.4")])
    output = tmp_path / "declarations.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "concurrent_declarations.py"
    command = [sys.executable, str(script), "--history", str(history), "--selection", str(selection),
               "--output", str(output), "--declared-on", "2026-11-02", "--issued-by", "Lucjan manager"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert re.fullmatch(r"(?:[a-z_]+=\d+\s*)+", result.stdout)
    assert len(json.loads(output.read_text())["declarations"]) == 1
    selection.write_text("{")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "errors=1\n"


@pytest.mark.parametrize("day", ["20261102", "2026-02-30"])
def test_invalid_declaration_date_is_an_error(tmp_path, day):
    history, selection = files(tmp_path, [row("1.2.3"), row("1.2.4")])
    with pytest.raises(ValueError):
        generate_declarations(history, selection, day)


def test_missing_csv_column_is_an_error(tmp_path):
    history, selection = files(tmp_path, [row("1.2.3"), row("1.2.4")])
    history.write_text("case_id,plan_uid\nSYNTH_CASE,1.2.3\n")
    with pytest.raises(ValueError):
        generate_declarations(history, selection, "2026-11-02")



def test_declarations_record_their_issuer(tmp_path):
    """Lucjan review 2026-11-02: every declaration names who issued it (the classifier refuses one without)."""
    history, selection = files(tmp_path, [row("1.2.3"), row("1.2.4")])
    payload, _ = generate_declarations(history, selection, "2026-11-02", "Lucjan manager")
    assert payload["declarations"] and all(d["issued_by"] == "Lucjan manager" for d in payload["declarations"])


@pytest.mark.parametrize("first,last", [("foo", "foo"), ("2024-02-12", "2024-01-01"), ("2024-1-1", "2024-02-12"),
                                        ("2024-01-01 08:00", "2024-02-12")])
def test_treatment_days_must_be_ordered_iso_dates(tmp_path, first, last):
    """Review r3 (2026-11-03): identical strings that are not ISO dates, or a last day before the first, are skipped."""
    rows = [row(uid, ) for uid in ("1.2.3", "1.2.4")]
    for r in rows:
        r.update(first_day_of_treatment=first, last_day_of_treatment=last)
    history, selection = files(tmp_path, rows)
    payload, counts = generate_declarations(history, selection, "2026-11-02")
    assert payload == {"declarations": []} and counts["cases_skipped"] == 1
