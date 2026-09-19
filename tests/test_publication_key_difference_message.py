"""An identity-set mismatch must say which identities differ.

Two sets of equal size can still disagree. Reporting only their lengths yields
"expected 212, found 212" -- true, and impossible to act on. A course that fails
this check is dropped from the cohort, so the message is the only evidence an
analyst has for why.
"""

import pytest

from rtpipeline.radiomics_ct_contract import describe_publication_key_difference


def test_equal_sizes_that_disagree_still_name_the_difference() -> None:
    expected = {("p", "c", "bladder", "primary"), ("p", "c", "rectum", "primary")}
    found = {("p", "c", "bladder", "primary"), ("p", "c", "rectum", "sensitivity")}

    message = describe_publication_key_difference(expected, found)

    assert "expected 2, found 2" in message
    assert "p/c/rectum/primary" in message
    assert "p/c/rectum/sensitivity" in message
    assert "1 expected identity(ies) absent" in message
    assert "1 unexpected identity(ies)" in message


def test_a_missing_identity_is_reported_as_absent() -> None:
    expected = {("p", "c", "bladder", "primary"), ("p", "c", "rectum", "primary")}
    found = {("p", "c", "bladder", "primary")}

    message = describe_publication_key_difference(expected, found)

    assert "expected 2, found 1" in message
    assert "p/c/rectum/primary" in message
    assert "unexpected" not in message


def test_the_sample_is_bounded_and_the_remainder_counted() -> None:
    expected = {("p", "c", f"roi{i}", "primary") for i in range(10)}
    found: set[tuple[str, ...]] = set()

    message = describe_publication_key_difference(expected, found, sample=3)

    assert "10 expected identity(ies) absent" in message
    assert "+7 more" in message


def test_identical_sets_describe_no_difference() -> None:
    keys = {("p", "c", "bladder", "primary")}

    message = describe_publication_key_difference(keys, keys)

    assert message == "(expected 1, found 1)"


@pytest.mark.parametrize("sample", [1, 5])
def test_the_message_is_a_single_parenthesised_clause(sample: int) -> None:
    expected = {("p", "c", "a", "primary"), ("p", "c", "b", "primary")}
    found = {("p", "c", "a", "primary"), ("p", "c", "z", "primary")}

    message = describe_publication_key_difference(expected, found, sample=sample)

    assert message.startswith("(") and message.endswith(")")
