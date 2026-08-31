from __future__ import annotations

"""Dependency-light bridge from Snakemake scripts to the pipeline interpreter."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


DELEGATE_SCHEMA = "rtpipeline-workflow-delegate-v1"


class DelegatedOperationError(RuntimeError):
    """A pipeline-interpreter operation failed or returned an invalid result."""


def runtime_environment(params: object) -> dict[str, str]:
    """Build the environment used by dependency-bearing pipeline subprocesses."""

    env = os.environ.copy()
    root_dir = str(getattr(params, "root_dir", "") or "").strip()
    if root_dir:
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            if root_dir not in existing_pythonpath.split(os.pathsep):
                env["PYTHONPATH"] = os.pathsep.join([root_dir, existing_pythonpath])
        else:
            env["PYTHONPATH"] = root_dir

    configfile = str(getattr(params, "configfile", "") or "").strip()
    if configfile:
        env["RTPIPELINE_CONFIGFILE"] = configfile
    radiomics_env = str(getattr(params, "radiomics_env", "") or "").strip()
    if radiomics_env:
        env["RTPIPELINE_RADIOMICS_ENV"] = radiomics_env

    python_bin = str(getattr(params, "python_bin", "") or "").strip()
    if python_bin:
        current_path = env.get("PATH", "")
        if python_bin not in current_path.split(os.pathsep):
            env["PATH"] = os.pathsep.join([python_bin, current_path])
    return env


def invoke(
    *,
    python: str,
    operation: str,
    arguments: Iterable[str],
    result_dir: Path,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Run one pipeline operation and return its structured payload."""

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    handle, result_name = tempfile.mkstemp(
        dir=str(result_dir), prefix=f".{operation}.", suffix=".json"
    )
    os.close(handle)
    result_path = Path(result_name)
    result_path.unlink(missing_ok=True)
    command = [
        str(python),
        "-m",
        "rtpipeline.workflow_delegate",
        "--result-path",
        str(result_path),
        operation,
        *[str(value) for value in arguments],
    ]
    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=dict(env) if env is not None else None,
            )
        except OSError as exc:
            raise DelegatedOperationError(
                f"could not launch pipeline interpreter for {operation}: {exc}"
            ) from exc
        output, _ = process.communicate()
        try:
            envelope = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            detail = (output or "").strip()
            suffix = f": {detail}" if detail else ""
            raise DelegatedOperationError(
                f"pipeline operation {operation} returned no readable structured result"
                f"{suffix}"
            ) from exc
        if not isinstance(envelope, dict):
            raise DelegatedOperationError(
                f"pipeline operation {operation} returned a non-object result"
            )
        if envelope.get("schema") != DELEGATE_SCHEMA:
            raise DelegatedOperationError(
                f"pipeline operation {operation} returned unsupported schema "
                f"{envelope.get('schema')!r}"
            )
        if envelope.get("operation") != operation:
            raise DelegatedOperationError(
                f"pipeline operation result identity mismatch: "
                f"expected {operation!r}, found {envelope.get('operation')!r}"
            )
        status = envelope.get("status")
        if process.returncode != 0 or status != "ok":
            error_type = str(envelope.get("error_type") or "RuntimeError")
            error = str(envelope.get("error") or "unknown delegated failure")
            raise DelegatedOperationError(f"{error_type}: {error}")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise DelegatedOperationError(
                f"pipeline operation {operation} returned no object payload"
            )
        return payload
    finally:
        result_path.unlink(missing_ok=True)
