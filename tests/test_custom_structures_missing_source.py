"""Regression test: custom-structure set operations must fail closed when a
required source mask is missing, rather than silently producing a mathematically
incorrect structure (over-inclusive intersection, under-subtracted subtract, wrong
xor). Union of an available subset is incomplete but not spurious, so it is allowed.
"""
import numpy as np
from rtpipeline.custom_structures import CustomStructureProcessor, CustomStructureConfig


def _masks():
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    b = np.zeros((4, 4, 4), dtype=np.uint8)
    a[1:3, 1:3, 1:3] = 1
    b[2:4, 2:4, 2:4] = 1
    return {"a": a, "b": b}  # "c" intentionally absent


def test_intersection_with_missing_source_returns_none():
    proc = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))
    cfg = CustomStructureConfig(name="ix", operation="intersection",
                                source_structures=["a", "b", "c"])
    assert proc.process_custom_structure(cfg, _masks()) is None


def test_subtract_with_missing_source_returns_none():
    proc = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))
    cfg = CustomStructureConfig(name="sub", operation="subtract",
                                source_structures=["a", "b", "c"])
    assert proc.process_custom_structure(cfg, _masks()) is None


def test_xor_with_missing_source_returns_none():
    proc = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))
    cfg = CustomStructureConfig(name="x", operation="xor",
                                source_structures=["a", "b", "c"])
    assert proc.process_custom_structure(cfg, _masks()) is None


def test_union_with_missing_source_still_produces_partial():
    proc = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))
    cfg = CustomStructureConfig(name="uni", operation="union",
                                source_structures=["a", "b", "c"])
    result = proc.process_custom_structure(cfg, _masks())
    assert result is not None
    assert int(np.sum(result > 0)) > 0
    assert "uni" in proc.partial_structures  # flagged as partial


def test_intersection_all_sources_present_succeeds():
    proc = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))
    m = _masks()
    m["c"] = np.ones((4, 4, 4), dtype=np.uint8)
    cfg = CustomStructureConfig(name="ix_ok", operation="intersection",
                                source_structures=["a", "b", "c"])
    result = proc.process_custom_structure(cfg, m)
    # a∩b is the single voxel where both overlap; with c=all-ones it is a∩b
    assert result is not None
    assert int(np.sum(result > 0)) == int(np.sum((m["a"] > 0) & (m["b"] > 0)))
