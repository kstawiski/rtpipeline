"""Generator-epoch reuse checks on freshly generated, uncropped synthetic courses."""
import json
import os
from pathlib import Path

import pytest

from rtpipeline import custom_structures_rtstruct as custom
from test_custom_structures_d16_auto_harvest import _publish_clean_course


@pytest.fixture
def published(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(Path("/tmp"))
    course = _publish_clean_course(tmp_path, monkeypatch)
    assert not (tmp_path / "RS_auto_cropped.dcm").exists()
    assert not custom._is_rs_custom_stale(
        course.out, course.config, course.rs_manual, course.rs_auto
    ), "fresh producer output must be reusable before metadata mutation"
    return course


def _stale(course, *, allow_contractless=False):
    assert not (course.out.parent / "RS_auto_cropped.dcm").exists()
    return custom._is_rs_custom_stale(
        course.out, course.config, course.rs_manual, course.rs_auto,
        allow_contractless=allow_contractless,
    )


def _rewrite_metadata(course, mutation):
    path = course.out.parent / "metadata" / "rs_custom_meta.json"
    before = path.stat()
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Old manual substitutions predate this field; its absence must not admit v3.
    payload.pop("totalseg_fallback_sources", None)
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_current_generator_output_is_reusable(published):
    path = published.out.parent / "metadata" / "rs_custom_meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert type(payload["version"]) is int
    assert payload["version"] == custom._RS_CUSTOM_META_VERSION
    assert "unread_source_rois" in payload
    assert not _stale(published)


def test_v3_without_fallback_field_must_regenerate(published):
    _rewrite_metadata(published, lambda payload: payload.update(version=3))
    assert _stale(published)


def test_missing_metadata_must_regenerate(published):
    (published.out.parent / "metadata" / "rs_custom_meta.json").unlink()
    assert _stale(published)


@pytest.mark.parametrize("version", [None, True, "4", 4.0, [], {}, 0, 2, 999])
def test_malformed_or_unsupported_version_must_regenerate(published, version):
    _rewrite_metadata(published, lambda payload: payload.update(version=version))
    assert _stale(published)


def test_absent_version_must_regenerate(published):
    _rewrite_metadata(published, lambda payload: payload.pop("version"))
    assert _stale(published)


@pytest.mark.parametrize("metadata", ["missing", "v3", "absent-version", "malformed-version"])
def test_contractless_uncropped_utility_keeps_historical_reuse(published, metadata):
    if metadata == "missing":
        (published.out.parent / "metadata" / "rs_custom_meta.json").unlink()
    elif metadata == "absent-version":
        _rewrite_metadata(published, lambda payload: payload.pop("version"))
    else:
        version = 3 if metadata == "v3" else "not-an-epoch"
        _rewrite_metadata(published, lambda payload: payload.update(version=version))
    assert not _stale(published, allow_contractless=True)


@pytest.mark.parametrize("allow_contractless", [False, True])
def test_recorded_mask_substitution_still_rejects_current_epoch(published, allow_contractless):
    _rewrite_metadata(
        published, lambda payload: payload.update(totalseg_fallback_sources={"PTV": {}})
    )
    assert _stale(published, allow_contractless=allow_contractless)


@pytest.mark.parametrize("version, expected", [(2, True), (3, False)])
def test_contractless_cropped_utility_keeps_historical_epoch_check(published, version, expected):
    _rewrite_metadata(published, lambda payload: payload.update(version=version))
    # This legacy check only tests existence; no fabricated cropped DICOM is read.
    (published.out.parent / "RS_auto_cropped.dcm").touch()
    assert custom._is_rs_custom_stale(
        published.out, published.config, published.rs_manual, published.rs_auto,
        allow_contractless=True,
    ) is expected
