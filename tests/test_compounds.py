"""Tests for compound / derivable-word detection."""

from __future__ import annotations

from course_compiler.compounds import is_derivable_compound, split_compound

DUTCH_LINKERS = ("s", "en", "e", "n")


def test_splits_simple_compound():
    known = {"koffie", "pot", "koffiepot"}
    assert split_compound("koffiepot", known) == ["koffie", "pot"]


def test_splits_compound_with_linking_s():
    known = {"station", "plein", "stationsplein"}
    assert split_compound("stationsplein", known, linkers=DUTCH_LINKERS) == [
        "station",
        "plein",
    ]


def test_word_itself_is_never_used_as_a_part():
    # The whole word is in the known set but must not be returned as a 1-part split.
    known = {"koffiepot"}
    assert split_compound("koffiepot", known) == []


def test_non_decomposable_word_is_not_a_compound():
    known = {"tafel", "stoel", "koffie"}
    assert split_compound("tafel", known) == []
    assert is_derivable_compound("tafel", known) is False


def test_derivable_when_split_into_two_known_parts():
    known = {"koffie", "pot", "koffiepot"}
    assert is_derivable_compound("koffiepot", known) is True


def test_opaque_compound_is_not_derivable_even_if_splittable():
    """An opaque compound (hand+schoen = glove) still counts as a new word."""
    known = {"hand", "schoen", "handschoen"}
    assert split_compound("handschoen", known) == ["hand", "schoen"]
    assert is_derivable_compound("handschoen", known, opaque={"handschoen"}) is False


def test_min_part_length_rejects_tiny_fragments():
    # 'ba' + 'naan' would be a false positive; min_part_len keeps it out.
    known = {"ba", "naan", "banaan"}
    assert split_compound("banaan", known, min_part_len=3) == []


def test_splitting_is_deterministic():
    known = {"koffie", "pot", "koffiepot", "station", "plein", "stationsplein"}
    a = split_compound("stationsplein", known, linkers=DUTCH_LINKERS)
    b = split_compound("stationsplein", known, linkers=DUTCH_LINKERS)
    assert a == b == ["station", "plein"]
