"""Archive metadata (CodeMeta, Zenodo) agrees with the package version and citation."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import rtpipeline

ROOT = Path(__file__).resolve().parents[1]


def test_codemeta_and_zenodo_agree_with_release():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    codemeta = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert codemeta["version"] == codemeta["softwareVersion"] == rtpipeline.__version__
    assert str(citation["version"]) == rtpipeline.__version__
    assert zenodo["upload_type"] == "software"
    assert zenodo["license"] == citation["license"] == "MIT"
    assert codemeta["license"].endswith("/MIT")


def test_release_notes_have_a_section_for_the_version():
    notes = (ROOT / "docs" / "release_notes.md").read_text(encoding="utf-8")
    assert f"\n## {rtpipeline.__version__} — " in notes
