# Contributing to RTpipeline

Thank you for improving RTpipeline. Before opening a pull request:

1. Run the test suite on the minimum supported Python:
   `uv sync --locked --group test --python 3.10 && uv run --no-sync python -m pytest -q`.
   Tests that need the PyRadiomics conda environment or the Snakemake executable report
   an explicit skip reason without them; the container job in CI runs them.
2. Run `ruff check .` and `python scripts/check_public_boundary.py`.
3. Build the documentation when it changes: `pip install mkdocs-material` and
   `mkdocs build --strict`.
4. Keep `pyproject.toml`, `rtpipeline/__init__.py`, `CITATION.cff`, `codemeta.json`,
   the Dockerfile label and `rtpipeline.def` on the same version when preparing a release,
   and add a section to `docs/release_notes.md`.

Never commit patient data, DICOM files, generated cohort outputs, model weights,
credentials or manuscript workspace material; `scripts/check_public_boundary.py` refuses
the latter.

`rtpipeline/segmenters/rtsegmentator_client.py` is vendored byte for byte from
RTsegmentator. Change it upstream first, then copy it and update the pinned digest in
`tests/test_rtsegmentator_integration.py`.

Participation is governed by the [code of conduct](CODE_OF_CONDUCT.md). Report
vulnerabilities privately as described in the [security guide](docs/SECURITY.md).
