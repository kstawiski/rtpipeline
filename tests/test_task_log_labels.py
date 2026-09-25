"""Task log labels never print whole objects (2026-09-25)."""

from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset

from rtpipeline.utils import _get_item_desc


def test_dataset_arguments_are_named_not_dumped():
    ds = Dataset()
    ds.PatientName = "SYNTHETIC^PATIENT"
    ds.ReviewerName = "SYNTHETIC^REVIEWER"
    label = _get_item_desc((46, "z_avoid", "AVOIDANCE", "Manual", 35.0, ds, Path("/tmp/course")))
    assert label == "46, z_avoid, AVOIDANCE, Manual, 35.0, <Dataset>, /tmp/course"
    assert "SYNTHETIC" not in label


def test_arrays_long_strings_and_objects():
    label = _get_item_desc((np.zeros((2, 3)), "x" * 500, object()))
    assert label.startswith("array(2, 3), " + "x" * 200 + "...")
    assert label.endswith("<object>")


def test_items_with_dir_attribute_and_plain_items():
    class Course:
        dir = Path("/tmp/c1")
    assert _get_item_desc(Course()) == "/tmp/c1"
    assert _get_item_desc("course-1") == "course-1"
    assert _get_item_desc(Dataset()) == "<Dataset>"
