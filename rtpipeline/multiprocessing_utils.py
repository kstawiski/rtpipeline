"""Keep multiprocessing control sockets short without relocating scratch data."""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.util import Finalize, ForkAwareThreadLock
import os
from pathlib import Path
import shutil
import tempfile


_lock = ForkAwareThreadLock()
# Leave headroom below Linux's 107-byte and macOS's 103-byte pathname limits.
_SOCKET_PATH_BUDGET = 100
_SOCKET_NAME = "listener-xxxxxxxx"


def configure_multiprocessing_tempdir() -> str | None:
    """Prepare a private socket directory before starting processes or listeners.

    CPython's multiprocessing stores its control-directory path in the current
    process config and passes it to spawn/forkserver children. Set only that
    path: changing TMPDIR or tempfile.tempdir would also move large image files.
    The directory belongs to its creating process; multiprocessing finalizers
    remove it after workers and listeners have shut down, including in workers
    that exit through multiprocessing rather than Python's atexit handlers.
    """
    if os.name != "posix":
        return None
    with _lock:
        config = mp.current_process()._config
        existing = config.get("tempdir")
        candidate = existing or os.path.join(tempfile.gettempdir(), "pymp-xxxxxxxx")
        if len(os.fsencode(os.path.join(candidate, _SOCKET_NAME))) <= _SOCKET_PATH_BUDGET:
            return existing

        last_error = None
        for root in ("/tmp", "/dev/shm"):
            directory = None
            try:
                directory = tempfile.mkdtemp(prefix="rtp-", dir=root)
                os.chmod(directory, 0o700)
            except OSError as exc:
                if directory is not None:
                    shutil.rmtree(directory, ignore_errors=True)
                last_error = exc
                continue
            Finalize(None, shutil.rmtree, args=(directory,),
                     kwargs={"ignore_errors": True}, exitpriority=-100)
            config["tempdir"] = directory
            return directory
        raise OSError("Cannot create a short private multiprocessing socket directory") from last_error


def get_context(method: str | None = None):
    """Return a multiprocessing context with safe control-socket paths."""
    configure_multiprocessing_tempdir()
    return mp.get_context(method)


def multiprocessing_subprocess_env(env=None) -> dict[str, str]:
    """Enable the same setup in Python CLI tools launched in another environment.

    A subprocess does not inherit multiprocessing's process config. Its startup
    hook loads only this standard-library helper, so Conda tools need neither
    RTpipeline installed nor its scientific dependencies in their environment.
    """
    result = dict(os.environ if env is None else env)
    startup = str(Path(__file__).with_name("_mp_startup"))
    paths = result["PYTHONPATH"].split(os.pathsep) if result.get("PYTHONPATH") else []
    result["PYTHONPATH"] = os.pathsep.join([startup, *[p for p in paths if p != startup]])
    return result
