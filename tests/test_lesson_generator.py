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
    """Returns a fixed sequence of responses and records all call message lists."""

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
    assert _target_length(1) == "30 words"


def test_target_length_floor():
    assert _target_length(0) == "30 words"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

def test_generate_language_in_system_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    assert "Dutch" in provider._calls[0][0].content


def test_generate_cefr_in_user_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A1", model="stub")
    assert "A1" in provider._calls[0][1].content


def test_generate_theme_in_user_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", theme="home", model="stub")
    assert "home" in provider._calls[0][1].content


def test_allowed_words_not_in_prompt():
    """The full allowed-words list must not appear in the prompt."""
    mapping = {w: w for w in ["huis", "kat", "boom", "water"]}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate("lesson001", ["huis"], {"huis", "kat", "boom", "water"}, language="Dutch", model="stub")
    full_prompt = " ".join(m.content for m in provider._calls[0])
    assert "kat" not in full_prompt
    assert "boom" not in full_prompt
    assert "water" not in full_prompt


# ---------------------------------------------------------------------------
# Validation, tolerance, and retry
# ---------------------------------------------------------------------------

def test_generate_returns_valid_content():
    provider = _StubProvider(["huis zijn groot"])
    gen = LessonGenerator(provider, _lemmatizer(["huis", "zijn", "groot"]))
    result = gen.generate(
        "lesson001", ["huis", "zijn", "groot"], {"huis", "zijn", "groot"},
        language="Dutch", model="stub",
    )
    assert result.content == "huis zijn groot"
    assert result.lesson_id == "lesson001"


def test_extra_word_at_same_cefr_is_tolerated():
    """An extra word at the same CEFR level passes without retry."""
    mapping = {"huis": "huis", "tafel": "tafel"}
    provider = _StubProvider(["huis tafel"])
    cefr_lookup = {"huis": "A1", "tafel": "A1"}
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        "lesson001", ["huis"], {"huis"},
        language="Dutch", cefr="A1", model="stub",
        cefr_lookup=cefr_lookup,
    )
    assert result.content == "huis tafel"
    assert "tafel" in result.tolerated
    assert len(provider._calls) == 1  # no retry needed


def test_extra_word_above_cefr_triggers_retry_with_feedback():
    """A B1 word in an A1 lesson triggers a retry with a feedback message."""
    mapping = {"huis": "huis", "appartement": "appartement"}
    cefr_lookup = {"huis": "A1", "appartement": "B1"}
    provider = _StubProvider(["huis appartement", "huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        "lesson001", ["huis"], {"huis"},
        language="Dutch", cefr="A1", model="stub",
        cefr_lookup=cefr_lookup,
    )
    assert result.content == "huis"
    assert len(provider._calls) == 2
    # Second call must include a feedback message mentioning the violation.
    second_call_text = " ".join(m.content for m in provider._calls[1])
    assert "appartement" in second_call_text


def test_retry_messages_form_multi_turn_conversation():
    """On retry the conversation appends assistant + user feedback, not a fresh prompt."""
    mapping = {"huis": "huis", "bad": "bad"}
    provider = _StubProvider(["huis bad", "huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate(
        "lesson001", ["huis"], {"huis"},
        language="Dutch", model="stub",
    )
    # First call: 2 messages (system + user)
    assert len(provider._calls[0]) == 2
    # Second call: 4 messages (system + user + assistant_bad + user_feedback)
    assert len(provider._calls[1]) == 4
    assert provider._calls[1][2].role.value == "assistant"
    assert provider._calls[1][3].role.value == "user"


def test_function_words_are_exempt():
    mapping = {"de": "de", "huis": "huis", "is": "zijn"}
    provider = _StubProvider(["de huis is"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping), function_lemmas={"de", "zijn"})
    result = gen.generate("lesson003", ["huis"], {"huis"}, language="Dutch", model="stub")
    assert result.content == "de huis is"


def test_generate_raises_after_max_retries():
    provider = _StubProvider(["huis xyz"] * 5)
    gen = LessonGenerator(provider, _lemmatizer(["huis"]), max_retries=3)
    with pytest.raises(RuntimeError, match="max_retries"):
        gen.generate("lesson004", ["huis"], {"huis"}, language="Dutch", model="stub")


def test_generate_uses_cache(tmp_path):
    from course_compiler.generation.cache import LLMCache

    words = ["huis", "zijn"]
    provider = _StubProvider(["huis zijn"])
    cache = LLMCache(tmp_path)
    gen = LessonGenerator(provider, _lemmatizer(words), cache=cache)
    r1 = gen.generate("lesson001", words, set(words), language="Dutch", model="stub")
    r2 = gen.generate("lesson001", words, set(words), language="Dutch", model="stub")
    assert r1.content == r2.content
    assert len(provider._calls) == 1
