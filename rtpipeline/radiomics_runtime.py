"""Run the entire radiomics driver in its isolated interpreter."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile


def needs_radiomics_process() -> bool:
    import numpy as np
    return int(np.__version__.split(".")[0]) >= 2


def run_course_driver(driver: str, *args, **kwargs):
    """Transfer trusted in-process configuration, never measurement masks."""
    from .radiomics_conda import (
        _radiomics_env_prefix, _conda_subprocess_env, RadiomicsEnvironmentProbeTimeout,
    )
    from .radiomics_outcomes import RadiomicsCourseExtractionError

    prefix = _radiomics_env_prefix()
    if prefix is None:
        raise RadiomicsCourseExtractionError(
            "Radiomics cannot rasterize in a NumPy 2 process. Set "
            "RTPIPELINE_RADIOMICS_ENV_PREFIX to the isolated NumPy 1.26 "
            "radiomics environment containing bin/python."
        )
    env = _conda_subprocess_env()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH", ""))))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(Path(prefix) / "lib"), env.get("LD_LIBRARY_PATH", ""))))
    # Private IPC files contain only this caller's trusted Python objects. They
    # are removed on success and on every failure; stdout remains usable for logs.
    with tempfile.TemporaryDirectory(prefix="radiomics-driver-") as scratch:
        request = Path(scratch) / "request.pkl"
        result = Path(scratch) / "result.pkl"
        request.write_bytes(pickle.dumps((driver, args, kwargs)))
        completed = subprocess.run(
            [str(Path(prefix) / "bin" / "python"), "-m", "rtpipeline.radiomics_runtime", str(request), str(result)],
            env=env, check=False,
        )
        if not result.is_file():
            raise RadiomicsCourseExtractionError(
                f"Isolated radiomics course driver failed (exit {completed.returncode}); "
                "check the child rasterizer diagnosis and RTPIPELINE_RADIOMICS_ENV_PREFIX"
            )
        payload = pickle.loads(result.read_bytes())
        if not payload["ok"]:
            if payload["type"] == "RadiomicsEnvironmentProbeTimeout":
                details = payload["details"]
                error = RadiomicsEnvironmentProbeTimeout(
                    details["command"], details["timeout"], details["attempts"]
                )
                error.code = payload["code"]
                raise error
            if payload["type"] == "RadiomicsCourseExtractionError":
                raise RadiomicsCourseExtractionError(payload["message"])
            raise RadiomicsCourseExtractionError(
                f"{payload['type']}: {payload['message']}"
            )
        if completed.returncode:
            raise RadiomicsCourseExtractionError(
                f"Isolated radiomics course driver exited {completed.returncode} after writing a result"
            )
        return payload["result"]


def main() -> int:
    request, result = map(Path, sys.argv[1:])
    try:
        if needs_radiomics_process():
            raise RuntimeError("Configured radiomics interpreter uses NumPy 2; select a NumPy 1.26 environment")
        # Polygon readers enforce the pin before filling. Stored voxel masks
        # remain exempt, including courses with no applicable polygon source.
        driver, args, kwargs = pickle.loads(request.read_bytes())
        from . import radiomics_conda
        drivers = {"ct": radiomics_conda.radiomics_for_course, "mr": radiomics_conda.radiomics_for_course_mr}
        output = drivers[driver](*args, **kwargs)
        payload = {"ok": True, "result": output}
    except Exception as exc:
        payload = {
            "ok": False, "type": type(exc).__name__, "message": str(exc),
            "code": getattr(exc, "code", None),
        }
        from .radiomics_conda import RadiomicsEnvironmentProbeTimeout
        if isinstance(exc, RadiomicsEnvironmentProbeTimeout):
            payload["details"] = {
                "command": list(exc.command), "timeout": exc.timeout, "attempts": exc.attempts,
            }
        result.write_bytes(pickle.dumps(payload))
        return 1
    result.write_bytes(pickle.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
