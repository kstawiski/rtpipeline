"""The local restricted-ID hook helper refuses matching staged blobs and fails closed without a list."""
import subprocess
import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "check_restricted_ids.py"


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.org"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    return repo


def _run(repo, *args, stdin=""):
    return subprocess.run([sys.executable, str(HELPER), *args], cwd=repo, input=stdin, capture_output=True, text=True)


def test_staged_blob_with_listed_id_is_refused(tmp_path):
    repo = _repo(tmp_path)
    ids = tmp_path / "ids.txt"
    ids.write_text("700007\n")  # synthetic list entry
    (repo / "a.py").write_text('key = ("700007", "2021-07")\n')
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    result = _run(repo, "--staged", "--ids-file", str(ids))
    assert result.returncode == 1 and "700007" not in result.stderr


def test_staged_blob_without_listed_id_passes(tmp_path):
    repo = _repo(tmp_path)
    ids = tmp_path / "ids.txt"
    ids.write_text("700007\n")
    (repo / "a.py").write_text('key = ("SYNTH_A", "2021-07")\n')
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    assert _run(repo, "--staged", "--ids-file", str(ids)).returncode == 0


def test_missing_list_fails_closed(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert _run(repo, "--staged", "--ids-file", str(empty)).returncode == 1
