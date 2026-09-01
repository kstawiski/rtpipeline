"""Delegate cohort aggregation to the dependency-bearing pipeline interpreter."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from rtpipeline.snakemake_delegate import invoke, runtime_environment


def _workflow_context(workflow: Any) -> dict:
    outputs = {
        name: str(getattr(workflow.output, name))
        for name in ("dvh", "fractions", "metadata", "qc")
    }
    outputs["dvh_parquet"] = str(
        getattr(
            workflow.output,
            "dvh_parquet",
            Path(str(workflow.output.dvh)).with_suffix(".parquet"),
        )
    )
    for name in ("radiomics", "radiomics_mr"):
        value = getattr(workflow.output, name, None)
        if value is not None:
            outputs[name] = str(value)
    return {
        "input": {"manifest": str(workflow.input.manifest)},
        "output": outputs,
        "log": [str(value) for value in workflow.log],
        "params": {
            "output_dir": str(workflow.params.output_dir),
            "results_dir": str(workflow.params.results_dir),
            "radiomics_enabled": bool(workflow.params.radiomics_enabled),
            "campaign_mode": bool(
                getattr(workflow.params, "campaign_mode", False)
            ),
            "campaign_min_completion_fraction": float(
                getattr(
                    workflow.params, "campaign_min_completion_fraction", 0.5
                )
            ),
            "campaign_require_all_courses": bool(
                getattr(workflow.params, "campaign_require_all_courses", False)
            ),
            "worker_budget": int(workflow.params.worker_budget),
            "auto_worker_budget": int(workflow.params.auto_worker_budget),
            "aggregation_threads": int(workflow.params.aggregation_threads),
        },
    }


def _write_context(directory: Path, context: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    handle, context_name = tempfile.mkstemp(
        dir=str(directory), prefix=".aggregate-context.", suffix=".json"
    )
    context_path = Path(context_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(context, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        context_path.unlink(missing_ok=True)
        raise
    return context_path


def main(workflow: Any) -> None:
    log_dir = Path(workflow.log[0]).parent
    context_path = _write_context(log_dir, _workflow_context(workflow))
    try:
        payload = invoke(
            python=str(getattr(workflow.params, "python", sys.executable)),
            operation="aggregate",
            arguments=("--context-path", str(context_path)),
            result_dir=log_dir,
            env=runtime_environment(workflow.params),
        )
    finally:
        context_path.unlink(missing_ok=True)
    if payload.get("completed") is not True:
        raise RuntimeError("pipeline aggregate delegate did not report completion")


if "snakemake" in globals():
    main(snakemake)  # type: ignore[name-defined]
