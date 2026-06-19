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


def test_valid_lesson_returns_empty_set():
    lem = _MapLemmatizer({"huis": "huis", "is": "zijn", "groot": "groot"})
    validator = VocabularyValidator(lem)
    allowed = {"huis", "zijn", "groot"}
    unknown = validator.validate("huis is groot", allowed)
    assert unknown == set()


def test_unknown_word_detected():
    lem = _MapLemmatizer({"kat": "kat", "loopt": "lopen"})
    validator = VocabularyValidator(lem)
    unknown = validator.validate("kat loopt snel", {"kat", "lopen"})
    assert "snel" in unknown


def test_punctuation_stripped():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    unknown = validator.validate("huis, huis!", {"huis"})
    assert unknown == set()


def test_case_insensitive():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    unknown = validator.validate("HUIS Huis huis", {"huis"})
    assert unknown == set()


def test_empty_text_is_valid():
    lem = _MapLemmatizer({})
    validator = VocabularyValidator(lem)
    assert validator.validate("", {"huis"}) == set()


def test_multiple_unknown_words():
    lem = _MapLemmatizer({"huis": "huis"})
    validator = VocabularyValidator(lem)
    unknown = validator.validate("huis abc def", {"huis"})
    assert "abc" in unknown
    assert "def" in unknown


def test_function_lemmas_are_exempt():
    """Tokens in function_lemmas are skipped even if not in allowed."""
    lem = _MapLemmatizer({"de": "de", "huis": "huis", "is": "zijn"})
    validator = VocabularyValidator(lem, function_lemmas={"de", "zijn"})
    # "de" and "is" (→ "zijn") are function words — should not appear in unknown
    unknown = validator.validate("de huis is", {"huis"})
    assert unknown == set()


def test_function_lemmas_default_empty():
    """Without function_lemmas every token is validated."""
    lem = _MapLemmatizer({"de": "de", "huis": "huis"})
    validator = VocabularyValidator(lem)
    unknown = validator.validate("de huis", {"huis"})
    assert "de" in unknown
