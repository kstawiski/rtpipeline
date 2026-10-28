"""Isolated process/socket checks, also usable as an external Python CLI."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import multiprocessing as mp
from multiprocessing.connection import Listener
from multiprocessing.util import get_temp_dir
import os
from pathlib import Path
import stat
import sys
import tempfile


def worker(value):
    with tempfile.NamedTemporaryFile() as scratch:
        return value * value, tempfile.gettempdir(), str(Path(scratch.name).parent), get_temp_dir()


def pipe_worker(connection):
    try:
        connection.send(worker(2))
    finally:
        connection.close()


def main():
    method, mode, workload = sys.argv[1:]
    scratch = tempfile.gettempdir()
    old_directory = get_temp_dir() if mode == "preexisting" else None
    if mode == "bootstrap":
        # Simulate third-party code: no RTpipeline import or explicit setup.
        context = mp.get_context(method)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rtpipeline.multiprocessing_utils import get_context

        # Course-level worker threads may all create pools at the same time.
        with ThreadPoolExecutor(max_workers=8) as threads:
            contexts = list(threads.map(get_context, [method] * 16))
        context = contexts[0]

    directory = get_temp_dir()
    with Listener(family="AF_UNIX") as listener:
        assert len(os.fsencode(listener.address)) <= 100
    with context.Manager() as manager:
        shared = manager.list([1, 2])
        assert list(shared) == [1, 2]
    result = {
        "directory": directory,
        "old_directory": old_directory,
        "mode": stat.S_IMODE(os.stat(directory).st_mode),
        "scratch": scratch,
        "previous_sitecustomize": os.environ.get("RTP_TEST_SITE_CUSTOMIZATION"),
    }
    if workload == "pool":
        try:
            # Probe only semaphore permission failures. Socket errors must fail.
            context.Lock()
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM, errno.ENOSYS):
                result["skip"] = f"Sandbox denies multiprocessing semaphores: {exc}"
                print(json.dumps(result))
                return
            raise
        with context.Pool(2) as pool:
            values = pool.map_async(worker, [1, 2, 3]).get(timeout=30)
        assert [item[0] for item in values] == [1, 4, 9]
    else:
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=pipe_worker, args=(child,))
        try:
            process.start()
            child.close()
            assert parent.poll(30), "Worker did not return its socket and scratch paths"
            values = [parent.recv()]
            process.join(30)
            assert process.exitcode == 0
            assert values[0][0] == 4
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            child.close()
            parent.close()
    assert all(item[1] == item[2] == scratch for item in values)
    assert all(item[3] == directory for item in values)
    assert tempfile.gettempdir() == os.environ["TMPDIR"] == scratch
    print(json.dumps(result))


if __name__ == "__main__":
    main()
