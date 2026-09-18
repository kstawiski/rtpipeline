"""Robustness selection must not sweep in ROIs baseline never measures.

Observed in production: 104 courses voided at the completeness gate whose
missing tasks were exclusively Manual Boolean subtraction shells
(PTV2-PTV1, PTV3-PTV2, ...). Baseline radiomics retains those shells for
inventory only (no feature rows), so no baseline exists to be robust
against, and the isolated worker returns no features for them. Selecting
them converts one unmeasurable ROI into a whole-course failure.
"""
from rtpipeline.radiomics_robustness import robustness_roi_in_publication_base


def test_boolean_helper_shell_is_excluded_with_reason() -> None:
    keep, roi_class, reason = robustness_roi_in_publication_base(
        "Manual", "PTV2 - PTV1"
    )
    assert keep is False
    assert roi_class == "planning_helper"
    assert "inventory only" in reason


def test_published_targets_and_organs_are_kept() -> None:
    for source, name in [
        ("Manual", "PTV1"),
        ("Manual", "Pecherz"),
        ("Manual", "gl"),
        ("AutoRTS_total", "urinary_bladder"),
    ]:
        keep, _, _ = robustness_roi_in_publication_base(source, name)
        assert keep is True, name


def test_unclassifiable_roi_is_kept_for_measurement() -> None:
    # A classification failure must fail open toward measurement (the
    # completeness gate stays fail-closed downstream), never silently drop.
    keep, _, reason = robustness_roi_in_publication_base(
        "Manual", "definitely-not-a-real-roi-xyz"
    )
    assert keep is True
