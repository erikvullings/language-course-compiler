"""Tests for the pluggable Lemmatizer registry."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import (
    Lemmatizer,
    LemmatizerError,
    create_lemmatizer,
    register_lemmatizer,
)


class _IdentityLemmatizer(Lemmatizer):
    """Lemmatizer that returns the token unchanged."""

    def __init__(self, language: str) -> None:
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    def lemmatize(self, token: str) -> str | None:
        return token.lower() if token.strip() else None


class _DictLemmatizer(Lemmatizer):
    """Lemmatizer backed by an explicit {form: lemma} dict."""

    def __init__(self, language: str, mapping: dict[str, str]) -> None:
        self._language = language
        self._mapping = mapping

    @property
    def language(self) -> str:
        return self._language

    def lemmatize(self, token: str) -> str | None:
        return self._mapping.get(token.lower())


def test_register_and_create(monkeypatch):
    """Registering a factory makes create_lemmatizer find it by language code."""
    import course_compiler.generation.base as _mod

    # isolate registry for this test
    monkeypatch.setattr(_mod, "_REGISTRY", {})

    register_lemmatizer("xx", lambda lang: _IdentityLemmatizer(lang))
    lem = create_lemmatizer("xx")
    assert lem.language == "xx"
    assert lem.lemmatize("Huis") == "huis"


def test_unknown_language_raises(monkeypatch):
    import course_compiler.generation.base as _mod

    monkeypatch.setattr(_mod, "_REGISTRY", {})

    with pytest.raises(LemmatizerError, match="No lemmatizer registered for language"):
        create_lemmatizer("zz")


def test_lemmatize_returns_none_for_unknown(monkeypatch):
    import course_compiler.generation.base as _mod

    monkeypatch.setattr(_mod, "_REGISTRY", {})
    register_lemmatizer("nl", lambda lang: _DictLemmatizer(lang, {"huis": "huis", "huizen": "huis"}))

    lem = create_lemmatizer("nl")
    assert lem.lemmatize("huis") == "huis"
    assert lem.lemmatize("huizen") == "huis"
    assert lem.lemmatize("xyz") is None
