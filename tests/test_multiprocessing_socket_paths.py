"""Long scratch paths must not prevent pool, manager or listener startup."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rtpipeline import multiprocessing_utils as helpers


pytestmark = pytest.mark.skipif(os.name != "posix", reason="AF_UNIX pathname tests require POSIX")
DRIVER = Path(__file__).with_name("multiprocessing_socket_driver.py")


def run_driver(tmp_path, method, mode, workload):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} is unavailable on this platform")
    scratch = tmp_path / ("scratch-" + "x" * 120)
    scratch.mkdir()
    assert len(os.fsencode(scratch)) > 110
    env = dict(os.environ, TMPDIR=str(scratch))
    if mode == "bootstrap":
        previous = tmp_path / "previous-site"
        previous.mkdir()
        (previous / "sitecustomize.py").write_text(
            'import os\nos.environ["RTP_TEST_SITE_CUSTOMIZATION"] = "retained"\n'
        )
        env["PYTHONPATH"] = str(previous)
        original = env.copy()
        env = helpers.multiprocessing_subprocess_env(env)
        assert original["PYTHONPATH"] == str(previous)
        assert env["TMPDIR"] == str(scratch)
    completed = subprocess.run(
        [sys.executable, str(DRIVER), method, mode, workload],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["mode"] == 0o700
    assert result["scratch"] == str(scratch)
    assert len(os.fsencode(result["directory"])) < 80
    assert not Path(result["directory"]).exists(), "Socket directory must be removed at process exit"
    if result["old_directory"]:
        assert not Path(result["old_directory"]).exists()
    if mode == "bootstrap":
        assert result["previous_sitecustomize"] == "retained"
    if "skip" in result:
        pytest.skip(result["skip"])


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
@pytest.mark.parametrize("mode", ["direct", "preexisting", "bootstrap"])
@pytest.mark.parametrize("workload", ["pool", "pipe"])
def test_pools_and_sockets_with_long_tmpdir(tmp_path, method, mode, workload):
    run_driver(tmp_path, method, mode, workload)


def test_short_effective_tmpdir_is_unchanged(monkeypatch):
    config = {}
    monkeypatch.setattr(helpers.mp, "current_process", lambda: SimpleNamespace(_config=config))
    monkeypatch.setattr(helpers.tempfile, "gettempdir", lambda: "/tmp")
    assert helpers.configure_multiprocessing_tempdir() is None
    assert config == {}
    config["tempdir"] = "/tmp/pymp-existing"
    assert helpers.configure_multiprocessing_tempdir() == "/tmp/pymp-existing"


def test_socket_path_limit_counts_encoded_bytes(monkeypatch):
    config = {"tempdir": "/tmp/" + "é" * 45}
    monkeypatch.setattr(helpers.mp, "current_process", lambda: SimpleNamespace(_config=config))
    directory = helpers.configure_multiprocessing_tempdir()
    try:
        assert directory.startswith("/tmp/rtp-")
        assert config["tempdir"] == directory
    finally:
        shutil.rmtree(directory)


def test_short_directory_falls_back_when_tmp_is_unwritable(monkeypatch, tmp_path):
    config = {"tempdir": "/long/" + "x" * 110}
    monkeypatch.setattr(helpers.mp, "current_process", lambda: SimpleNamespace(_config=config))
    mkdir = helpers.tempfile.mkdtemp
    attempted = []
    def without_tmp(*, prefix, dir):
        attempted.append(dir)
        if dir == "/tmp":
            raise PermissionError("unwritable /tmp")
        return mkdir(prefix=prefix, dir=tmp_path)
    monkeypatch.setattr(helpers.tempfile, "mkdtemp", without_tmp)
    directory = None
    try:
        directory = helpers.configure_multiprocessing_tempdir()
        assert attempted == ["/tmp", "/dev/shm"]
        assert config["tempdir"] == directory
    finally:
        if directory:
            shutil.rmtree(directory)
