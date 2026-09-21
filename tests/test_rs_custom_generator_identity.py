"""RS_custom.dcm must be rebuilt when the code that generates it changes.

_is_rs_custom_stale checked a manual integer epoch, _RS_CUSTOM_META_VERSION.
That only invalidates a publication when someone remembers to bump it. On
2026-09-21 a source-scope fix changed which ROIs resolve, nobody bumped the
epoch, and every course reused an RS_custom.dcm built under the old rule. Each
was short 28 structures -- pelvic_bones, sacrum, hip_left, hip_right, bowel_bag
among them -- and every course whose configuration required one aborted: 39 of
122 produced no radiomics at all, and the cohort's measured urinary_bladder
fell from 88 courses to 60.
"""

import pytest

from rtpipeline import custom_structures_rtstruct as module
from rtpipeline.custom_structures_rtstruct import (
    RS_CUSTOM_GENERATOR_CODE_SOURCES,
    rs_custom_generator_sha256,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    module._RS_CUSTOM_GENERATOR_RESOLVED = False
    module._RS_CUSTOM_GENERATOR_SHA256 = None
    yield
    module._RS_CUSTOM_GENERATOR_RESOLVED = False
    module._RS_CUSTOM_GENERATOR_SHA256 = None


def test_the_identity_is_a_stable_digest():
    first = rs_custom_generator_sha256()

    assert isinstance(first, str) and len(first) == 64
    module._RS_CUSTOM_GENERATOR_RESOLVED = False
    assert rs_custom_generator_sha256() == first


def test_the_geometry_validator_decides_the_identity():
    """dcbb476 changed rtstruct_geometry and silently changed RS_custom."""

    assert "rtstruct_geometry.py" in RS_CUSTOM_GENERATOR_CODE_SOURCES
    assert "custom_structures_rtstruct.py" in RS_CUSTOM_GENERATOR_CODE_SOURCES
    assert "custom_structures.py" in RS_CUSTOM_GENERATOR_CODE_SOURCES


def test_unrelated_modules_do_not_decide_the_identity():
    """A Snakefile or CLI edit must not rebuild every RS_custom in a cohort."""

    for unrelated in ("Snakefile", "cli.py", "stage_completion.py", "dvh.py"):
        assert unrelated not in RS_CUSTOM_GENERATOR_CODE_SOURCES


@pytest.mark.parametrize("edited", ["rtstruct_geometry.py", "custom_structures.py"])
def test_changing_a_generator_source_changes_the_identity(monkeypatch, tmp_path, edited):
    baseline = rs_custom_generator_sha256()

    real = module.Path(module.__file__).resolve().parent
    shadow = tmp_path / "pkg"
    shadow.mkdir()
    for relative in RS_CUSTOM_GENERATOR_CODE_SOURCES:
        (shadow / relative).write_bytes((real / relative).read_bytes())
    (shadow / edited).write_bytes(b"# edited\n" + (real / edited).read_bytes())

    monkeypatch.setattr(module, "__file__", str(shadow / "custom_structures_rtstruct.py"))
    module._RS_CUSTOM_GENERATOR_RESOLVED = False

    assert rs_custom_generator_sha256() != baseline


def test_an_unreadable_installation_yields_no_identity(monkeypatch, tmp_path):
    """Absent sources are a broken install, not a stale publication."""

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(module, "__file__", str(empty / "custom_structures_rtstruct.py"))
    module._RS_CUSTOM_GENERATOR_RESOLVED = False

    assert rs_custom_generator_sha256() is None
