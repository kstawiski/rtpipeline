from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Sequence


def run_totalsegmentator_prediction(
    *,
    input_path: Path,
    output_dir: Path,
    command: str = "TotalSegmentator",
    task: str,
    device: str = "gpu",
    fast: bool = False,
    force_split: bool = False,
    extra_args: Sequence[str] | None = None,
    command_prefix: str = "",
    env: Mapping[str, str] | None = None,
    runner: Callable[..., bool] | None = None,
) -> None:
    """Run a TotalSegmentator task as a custom model backend."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_arg = str(device or "cpu").lower()
    cmd_parts = [
        command,
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "-ta",
        task,
        "--device",
        device_arg,
    ]
    if fast:
        cmd_parts.append("--fast")
    if force_split:
        cmd_parts.append("--force_split")
    cmd_parts.extend(str(arg) for arg in (extra_args or []))

    cmd = f"{command_prefix}{' '.join(shlex.quote(part) for part in cmd_parts)}"
    if runner is not None:
        ok = runner(cmd, env=dict(env or {}))
    else:
        completed = subprocess.run(cmd, shell=True, env=dict(env or {}), check=False)
        ok = completed.returncode == 0
    if not ok:
        raise RuntimeError(f"TotalSegmentator task '{task}' failed")
