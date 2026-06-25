"""Tests for VocabularyValidator."""

from __future__ import annotations

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.validator import VocabularyValidator


class _MapLemmatizer(Lemmatizer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    @property
    def language(self) -> str:
        return "test"

    def lemmatize(self, token: str) -> str | None:
        return self._mapping.get(token.lower())


# ---------------------------------------------------------------------------
# Basic allow/reject
# ---------------------------------------------------------------------------


def test_valid_lesson_is_valid():
    lem = _MapLemmatizer({"huis": "huis", "is": "zijn", "groot": "groot"})
    validator = VocabularyValidator(lem)
    result = validator.validate("huis is groot", {"huis", "zijn", "groot"})
    assert result.is_valid


def test_unknown_word_without_cefr_info_is_violation():
    lem = _MapLemmatizer({"kat": "kat", "loopt": "lopen", "snel": "snel"})
    validator = VocabularyValidator(lem)
    result = validator.validate("kat loopt snel", {"kat", "lopen"})
    assert "snel" in result.violations
    assert not result.is_valid


def test_punctuation_stripped():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    assert validator.validate("huis, huis!", {"huis"}).is_valid


def test_case_insensitive():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    assert validator.validate("HUIS Huis huis", {"huis"}).is_valid


def test_empty_text_is_valid():
    lem = _MapLemmatizer({})
    validator = VocabularyValidator(lem)
    assert validator.validate("", {"huis"}).is_valid


def test_multiple_unknown_words():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    result = validator.validate("huis abc def", {"huis"})
    assert "abc" in result.violations
    assert "def" in result.violations


def test_function_lemmas_are_exempt():
    lem = _MapLemmatizer({"de": "de", "huis": "huis", "is": "zijn"})
    validator = VocabularyValidator(lem, function_lemmas={"de", "zijn"})
    result = validator.validate("de huis is", {"huis"})
    assert result.is_valid


def test_function_lemmas_default_empty():
    lem = _MapLemmatizer({"de": "de", "huis": "huis"})
    validator = VocabularyValidator(lem)
    result = validator.validate("de huis", {"huis"})
    assert "de" in result.violations


# ---------------------------------------------------------------------------
# CEFR-aware tolerance
# ---------------------------------------------------------------------------

_CEFR = {"tafel": "A1", "stoel": "A1", "appartement": "B1", "huis": "A1"}


def test_extra_word_at_same_cefr_is_tolerated():
    """An extra A1 word in an A1 lesson is tolerated (within 50% budget)."""
    lem = _MapLemmatizer({k: k for k in _CEFR})
    validator = VocabularyValidator(lem)
    # allowed = {"huis"}, extra = "tafel" (A1) → tolerated
    result = validator.validate(
        "huis tafel",
        {"huis"},
        cefr_target="A1",
        cefr_lookup=_CEFR,
        new_word_count=10,
    )
    assert result.is_valid
    assert "tafel" in result.tolerated


def test_extra_word_above_cefr_is_violation():
    """A B1 word in an A1 lesson is always a violation."""
    lem = _MapLemmatizer({k: k for k in _CEFR})
    validator = VocabularyValidator(lem)
    result = validator.validate(
        "huis appartement",
        {"huis"},
        cefr_target="A1",
        cefr_lookup=_CEFR,
        new_word_count=10,
    )
    assert "appartement" in result.violations
    assert not result.is_valid


def test_tolerated_extras_respect_budget():
    """Extras beyond 50% of new_word_count become violations even at correct CEFR."""
    # new_word_count=2, budget=1 → second extra is a violation
    lem = _MapLemmatizer({k: k for k in _CEFR})
    validator = VocabularyValidator(lem)
    result = validator.validate(
        "huis tafel stoel",  # 2 extras: tafel, stoel (both A1)
        {"huis"},
        cefr_target="A1",
        cefr_lookup=_CEFR,
        extra_tolerance=0.5,
        new_word_count=2,  # budget = 1
    )
    # One should be tolerated, one should be a violation
    assert len(result.tolerated) == 1
    assert len(result.violations) == 1


def test_unlimited_in_level_tolerance_accepts_all_in_level_extras():
    """extra_tolerance=None tolerates every in-level extra, but still rejects above-CEFR."""
    lem = _MapLemmatizer({k: k for k in _CEFR})
    validator = VocabularyValidator(lem)
    result = validator.validate(
        "huis tafel stoel appartement",  # 3 A1 extras + 1 B1 extra
        {"huis"},
        cefr_target="A1",
        cefr_lookup=_CEFR,
        extra_tolerance=None,  # uncapped for in-level words
        new_word_count=1,
    )
    assert result.tolerated == frozenset({"tafel", "stoel"})
    assert result.violations == frozenset({"appartement"})  # above CEFR still rejected
    assert not result.is_valid


def test_no_cefr_info_all_extras_are_violations():
    """Without cefr_lookup, every extra is a violation."""
    lem = _MapLemmatizer({"huis": "huis", "tafel": "tafel"})
    validator = VocabularyValidator(lem)
    result = validator.validate("huis tafel", {"huis"})
    assert "tafel" in result.violations


def test_proper_name_not_sentence_initial_is_exempt():
    lem = _MapLemmatizer(
        {"ik": "ik", "spreek": "spreken", "met": "met", "huis": "huis"}
    )
    validator = VocabularyValidator(lem, function_lemmas={"ik", "spreken", "met"})
    result = validator.validate("ik spreek met Mark huis", {"huis"})
    assert result.is_valid


def test_sentence_initial_capitalized_word_is_not_auto_exempt():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    result = validator.validate("Mark huis", {"huis"})
    assert "mark" in result.violations
