"""Tests for LessonGenerator."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.lesson import LessonGenerator
from course_compiler.llm.base import LLMProvider, LLMResponse, Message, PromptInput


class _MapLemmatizer(Lemmatizer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    @property
    def language(self) -> str:
        return "test"

    def lemmatize(self, token: str) -> str | None:
        return self._mapping.get(token.lower())


class _StubProvider(LLMProvider):
    """Returns a fixed sequence of responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._calls: list[list[Message]] = []

    def complete(
        self, prompt: PromptInput, *, model: str | None = None, temperature: float | None = None, **kwargs: object
    ) -> LLMResponse:
        from course_compiler.llm.base import to_messages

        self._calls.append(to_messages(prompt))
        content = self._responses.pop(0) if self._responses else ""
        return LLMResponse(content=content, model=model or "stub", raw={})

    async def acomplete(
        self, prompt: PromptInput, *, model: str | None = None, temperature: float | None = None, **kwargs: object
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature, **kwargs)


def _lemmatizer(words: list[str]) -> _MapLemmatizer:
    return _MapLemmatizer({w: w for w in words})


def test_generate_returns_content_when_valid():
    mapping = {"huis": "huis", "zijn": "zijn", "groot": "groot"}
    provider = _StubProvider(["huis zijn groot"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        lesson_id="lesson001",
        new_words=["huis", "zijn", "groot"],
        allowed_words={"huis", "zijn", "groot"},
        language="Dutch",
        model="stub",
    )
    assert result.content == "huis zijn groot"
    assert result.lesson_id == "lesson001"


def test_generate_language_appears_in_prompt():
    """The system prompt must mention the target language."""
    mapping = {"huis": "huis"}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    system_msg = provider._calls[0][0]
    assert "Dutch" in system_msg.content


def test_generate_retries_on_vocabulary_leakage():
    """First response has an unknown content word; second is clean."""
    mapping = {"huis": "huis", "zijn": "zijn", "snel": "snel"}
    provider = _StubProvider(["huis snel zijn", "huis zijn"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        lesson_id="lesson002",
        new_words=["huis", "zijn"],
        allowed_words={"huis", "zijn"},  # "snel" is NOT allowed
        language="Dutch",
        model="stub",
    )
    assert result.content == "huis zijn"
    assert len(provider._calls) == 2


def test_function_words_are_exempt():
    """Tokens whose lemma is in function_lemmas pass without appearing in allowed_words."""
    # "de" and "is" are function words; "huis" is a content word.
    mapping = {"de": "de", "huis": "huis", "is": "zijn"}
    provider = _StubProvider(["de huis is"])
    gen = LessonGenerator(
        provider,
        _MapLemmatizer(mapping),
        function_lemmas={"de", "zijn"},  # function words
    )
    result = gen.generate(
        lesson_id="lesson003",
        new_words=["huis"],
        allowed_words={"huis"},  # only content word
        language="Dutch",
        model="stub",
    )
    assert result.content == "de huis is"


def test_generate_raises_after_max_retries():
    words = ["huis"]
    provider = _StubProvider(["huis xyz"] * 5)
    gen = LessonGenerator(provider, _lemmatizer(words), max_retries=3)
    with pytest.raises(RuntimeError, match="max_retries"):
        gen.generate(
            lesson_id="lesson004",
            new_words=words,
            allowed_words=set(words),
            language="Dutch",
            model="stub",
        )


def test_generate_uses_cache(tmp_path):
    """Second call with same inputs hits the cache, not the provider."""
    from course_compiler.generation.cache import LLMCache

    words = ["huis", "zijn"]
    mapping = {"huis": "huis", "zijn": "zijn"}
    provider = _StubProvider(["huis zijn"])
    cache = LLMCache(tmp_path)
    gen = LessonGenerator(provider, _MapLemmatizer(mapping), cache=cache)
    r1 = gen.generate("lesson001", words, set(words), language="Dutch", model="stub")
    r2 = gen.generate("lesson001", words, set(words), language="Dutch", model="stub")
    assert r1.content == r2.content
    assert len(provider._calls) == 1
