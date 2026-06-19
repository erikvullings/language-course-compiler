"""Tests for LessonGenerator."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.lesson import LessonGenerator, _target_length
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


# ---------------------------------------------------------------------------
# Target length scaling
# ---------------------------------------------------------------------------

def test_target_length_scales_with_word_count():
    assert _target_length(10) == "150 words"
    assert _target_length(5) == "75 words"
    assert _target_length(1) == "30 words"   # floor at 30


def test_target_length_floor():
    assert _target_length(0) == "30 words"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

def test_generate_language_in_system_prompt():
    mapping = {"huis": "huis"}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    system_msg = provider._calls[0][0]
    assert "Dutch" in system_msg.content


def test_generate_cefr_in_user_prompt():
    mapping = {"huis": "huis"}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A1", model="stub")
    user_msg = provider._calls[0][1]
    assert "A1" in user_msg.content


def test_generate_theme_in_user_prompt():
    mapping = {"huis": "huis"}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", theme="home", model="stub")
    user_msg = provider._calls[0][1]
    assert "home" in user_msg.content


def test_allowed_words_not_in_prompt():
    """The full allowed-words list must not appear in the prompt (it's for the validator only)."""
    mapping = {w: w for w in ["huis", "kat", "boom", "water"]}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate(
        "lesson001", ["huis"], {"huis", "kat", "boom", "water"}, language="Dutch", model="stub"
    )
    full_prompt = " ".join(m.content for m in provider._calls[0])
    # "kat", "boom", "water" are allowed but should not appear in the prompt
    assert "kat" not in full_prompt
    assert "boom" not in full_prompt
    assert "water" not in full_prompt


# ---------------------------------------------------------------------------
# Validation and retry
# ---------------------------------------------------------------------------

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


def test_generate_retries_on_vocabulary_leakage():
    mapping = {"huis": "huis", "zijn": "zijn", "snel": "snel"}
    provider = _StubProvider(["huis snel zijn", "huis zijn"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        lesson_id="lesson002",
        new_words=["huis", "zijn"],
        allowed_words={"huis", "zijn"},
        language="Dutch",
        model="stub",
    )
    assert result.content == "huis zijn"
    assert len(provider._calls) == 2


def test_function_words_are_exempt():
    mapping = {"de": "de", "huis": "huis", "is": "zijn"}
    provider = _StubProvider(["de huis is"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping), function_lemmas={"de", "zijn"})
    result = gen.generate(
        lesson_id="lesson003",
        new_words=["huis"],
        allowed_words={"huis"},
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
