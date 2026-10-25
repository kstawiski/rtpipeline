"""Skip environment-bound tests only when their external runtime is absent.

The radiomics extraction backend runs PyRadiomics in a separate NumPy 1.x
environment, and workflow tests launch the Snakemake executable. The container
CI job provides both, so these tests run there; a plain virtual environment
without them reports the skip reason instead of a spurious failure.
"""
from __future__ import annotations

import os
import shutil

import pytest


def radiomics_environment_available() -> bool:
    from rtpipeline import radiomics_conda

    return (
        radiomics_conda._radiomics_env_prefix() is not None
        or shutil.which(radiomics_conda.CONDA_EXE) is not None
    )


def snakemake_executable() -> str | None:
    return os.environ.get("RTPIPELINE_TEST_SNAKEMAKE") or shutil.which("snakemake")


def pytest_collection_modifyitems(config, items):
    radiomics = snakemake = None
    for item in items:
        if item.get_closest_marker("requires_radiomics_env"):
            if radiomics is None:
                radiomics = radiomics_environment_available()
            if not radiomics:
                item.add_marker(pytest.mark.skip(
                    reason="radiomics conda environment (rtpipeline-radiomics) is not installed"))
        if item.get_closest_marker("requires_snakemake"):
            if snakemake is None:
                snakemake = snakemake_executable() is not None
            if not snakemake:
                item.add_marker(pytest.mark.skip(reason="snakemake executable is not installed"))
