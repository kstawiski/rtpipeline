"""The public patient-key guard flags likely real course keys and accepts synthetic ones."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_patient_keys.py"
spec = importlib.util.spec_from_file_location("check_patient_keys", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "line",
    [
        'course_dir = tmp_path / "{k}" / "2021-07"',
        '[("{k}", "2024-06")]',
        "# closed live course {k}/2021-07 at publication",
        "records/{k}__2020-05__dvh.json",
        'name="test_key[{k}-2024-06]"',
        "/data/Output/{k}/2022-01/RS.dcm",
    ],
)
def test_likely_course_keys_are_flagged(tmp_path, line):
    path = tmp_path / "sample.py"
    path.write_text(line.format(k="4" + "0" * 4 + "7") + "\n")
    assert guard.scan([path], set()) == [(str(path), 1)]


@pytest.mark.parametrize(
    "line",
    [
        'course_dir = tmp_path / "SYNTH_A" / "2021-07"',
        '"sha256": "3ff8bd1540000721dba9dde4"',
        "start = 1700000.5  # 2024-06",
        'size = 400007, upload-time = "2025-08-27T15:24:02"',
        '[("x", "y")]',
    ],
)
def test_synthetic_or_unrelated_numbers_are_not_flagged(tmp_path, line):
    path = tmp_path / "sample.py"
    path.write_text(line + "\n")
    assert guard.scan([path], set()) == []


def test_allowlisted_synthetic_key_is_accepted(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text('course_dir = tmp_path / "900001" / "2021-07"\n')
    assert guard.scan([path], set()) == [(str(path), 1)]
    assert guard.scan([path], {"900001"}) == []
