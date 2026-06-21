"""Tests for LessonGenerator."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.lesson import (
    LessonGenerator,
    _parse_lesson_structure,
    _target_length,
)
from course_compiler.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    PromptInput,
)


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
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        from course_compiler.llm.base import to_messages

        self._calls.append(to_messages(prompt))
        content = self._responses.pop(0) if self._responses else ""
        return LLMResponse(content=content, model=model or "stub", raw={})

    async def acomplete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature, **kwargs)


class _ErrorProvider(LLMProvider):
    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        raise LLMError("timed out")

    async def acomplete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        raise LLMError("timed out")


def _lemmatizer(words: list[str]) -> _MapLemmatizer:
    return _MapLemmatizer({w: w for w in words})


# ---------------------------------------------------------------------------
# Target length scaling
# ---------------------------------------------------------------------------


def test_parse_strips_echoed_title_placeholder():
    """Weak models echo the literal '## Lesson Title' header; strip that prefix."""
    title, _, _ = _parse_lesson_structure(
        "## Lesson Title: Begroetingen\n\n**New words:** hallo\nHallo daar."
    )
    assert title == "Begroetingen"


def test_parse_drops_bare_title_placeholder():
    """A bare echoed placeholder yields a neutral default, not 'Lesson Title'."""
    title, _, _ = _parse_lesson_structure(
        "## Lesson Title\n\n**New words:** hallo\nHallo daar."
    )
    assert title != "Lesson Title"


def test_target_length_new_word_limited_when_vocab_is_ample():
    # With a large allowed vocabulary, length is driven by the new-word budget.
    assert _target_length(10, 1000) == "150 words"
    assert _target_length(5, 1000) == "75 words"
    assert _target_length(1, 1000) == "30 words"


def test_target_length_vocab_limited_at_cold_start():
    # Early lessons (allowed ≈ new) are capped by the sustainable-vocab budget,
    # not by 15 × new_words — otherwise we demand 150 words from a 10-word vocab.
    assert _target_length(10, 10) == "40 words"


def test_target_length_floor():
    assert _target_length(0, 0) == "30 words"


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
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A1", model="stub"
    )
    assert "A1" in provider._calls[0][1].content


def test_generate_theme_in_user_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", theme="home", model="stub"
    )
    assert "home" in provider._calls[0][1].content


def test_generate_outline_in_prompt_and_forces_narrative():
    """A lesson outline is included in the prompt and switches to narrative format."""
    provider = _StubProvider(["## T\n**New words:** huis\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},  # tiny vocab that would otherwise use the example format
        language="Dutch",
        theme="home",
        outline="Two neighbours meet and talk about their house.",
        model="stub",
    )
    full_prompt = " ".join(m.content for m in provider._calls[0]).lower()
    assert "two neighbours meet" in full_prompt
    assert "narrative" in provider._calls[0][0].content.lower()


def test_early_lesson_uses_example_format_when_vocab_is_small():
    """With little vocabulary to recombine, ask for simple example sentences."""
    provider = _StubProvider(["## T\n**New words:** huis\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    system_prompt = provider._calls[0][0].content.lower()
    assert "example sentences" in system_prompt
    assert "narrative" not in system_prompt


def test_mature_lesson_uses_narrative_format_when_vocab_is_large():
    """Once a base vocabulary exists, ask for a coherent narrative."""
    allowed = {f"w{i}" for i in range(80)}
    provider = _StubProvider(["## T\n**New words:** w0\nw0"])
    gen = LessonGenerator(provider, _MapLemmatizer({w: w for w in allowed}))
    gen.generate("lesson050", ["w0"], allowed, language="Dutch", model="stub")
    system_prompt = provider._calls[0][0].content.lower()
    assert "narrative" in system_prompt


def test_narrative_threshold_is_configurable():
    """Lowering the threshold flips a small-vocab lesson into narrative format."""
    allowed = {"huis", "deur"}
    provider = _StubProvider(["## T\n**New words:** huis\nhuis"])
    gen = LessonGenerator(
        provider, _MapLemmatizer({"huis": "huis", "deur": "deur"}),
        narrative_vocab_threshold=2,
    )
    gen.generate("lesson001", ["huis"], allowed, language="Dutch", model="stub")
    assert "narrative" in provider._calls[0][0].content.lower()


def test_low_cefr_system_prompt_allows_frequent_simple_past_forms():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A1", model="stub"
    )
    system_prompt = provider._calls[0][0].content
    assert "Prefer present tense" in system_prompt
    assert "equivalents of 'was/were'" in system_prompt


def test_allowed_words_not_in_prompt():
    """The full allowed-words list must not appear in the prompt."""
    mapping = {w: w for w in ["huis", "kat", "boom", "water"]}
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate(
        "lesson001",
        ["huis"],
        {"huis", "kat", "boom", "water"},
        language="Dutch",
        model="stub",
    )
    full_prompt = " ".join(m.content for m in provider._calls[0])
    assert "kat" not in full_prompt
    assert "boom" not in full_prompt
    assert "water" not in full_prompt


# ---------------------------------------------------------------------------
# Validation, tolerance, and retry
# ---------------------------------------------------------------------------


def test_generate_returns_valid_content():
    response = "## Home Lesson\n**New words:** huis, zijn, groot\nhuis zijn groot."
    provider = _StubProvider([response])
    gen = LessonGenerator(provider, _lemmatizer(["huis", "zijn", "groot"]))
    result = gen.generate(
        "lesson001",
        ["huis", "zijn", "groot"],
        {"huis", "zijn", "groot"},
        language="Dutch",
        model="stub",
    )
    assert result.content
    assert result.lesson_id == "lesson001"
    assert result.title == "Home Lesson"
    assert set(result.new_words) == {"huis", "zijn", "groot"}


def test_extra_word_at_same_cefr_is_tolerated():
    """An extra word at the same CEFR level passes without retry."""
    mapping = {"huis": "huis", "tafel": "tafel"}
    response = "## Home\n**New words:** huis, tafel\nhuis tafel"
    provider = _StubProvider([response])
    cefr_lookup = {"huis": "A1", "tafel": "A1"}
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},
        language="Dutch",
        cefr="A1",
        model="stub",
        cefr_lookup=cefr_lookup,
    )
    assert "tafel" in result.content or "tafel" in result.new_words
    assert "tafel" in result.tolerated
    assert len(provider._calls) == 1  # no retry needed


def test_extra_word_above_cefr_triggers_retry_with_feedback():
    """A B1 word in an A1 lesson triggers a retry with a feedback message."""
    mapping = {"huis": "huis", "appartement": "appartement"}
    cefr_lookup = {"huis": "A1", "appartement": "B1"}
    bad_response = "## Home\n**New words:** huis, appartement\nappartement"
    good_response = "## Home\n**New words:** huis\nhuis"
    provider = _StubProvider([bad_response, good_response])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    result = gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},
        language="Dutch",
        cefr="A1",
        model="stub",
        cefr_lookup=cefr_lookup,
    )
    assert "huis" in result.content
    assert len(provider._calls) == 2
    # Second call must include a feedback message mentioning the violation.
    second_call_text = " ".join(m.content for m in provider._calls[1])
    assert "appartement" in second_call_text


def test_retry_feedback_asks_for_minimal_edit_not_rewrite():
    """Feedback anchors on the previous draft (revise, don't regenerate from scratch)."""
    mapping = {"huis": "huis", "appartement": "appartement"}
    cefr_lookup = {"huis": "A1", "appartement": "B1"}
    bad = "## Home\n**New words:** huis, appartement\nappartement"
    good = "## Home\n**New words:** huis\nhuis"
    provider = _StubProvider([bad, good])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},
        language="Dutch",
        cefr="A1",
        model="stub",
        cefr_lookup=cefr_lookup,
    )
    feedback = provider._calls[1][3].content.lower()
    # Instructs a minimal revision of the previous version, not a fresh rewrite.
    assert "revise" in feedback
    assert "rewrite from scratch" in feedback
    # Still names the offending word to remove/replace.
    assert "appartement" in feedback


def test_retry_messages_form_multi_turn_conversation():
    """On retry the conversation appends assistant + user feedback, not a fresh prompt."""
    mapping = {"huis": "huis", "bad": "bad"}
    bad = "## Title\n**New words:** huis, bad\nText"
    good = "## Title\n**New words:** huis\nText"
    provider = _StubProvider([bad, good])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},
        language="Dutch",
        model="stub",
    )
    # First call: 2 messages (system + user)
    assert len(provider._calls[0]) == 2
    # Second call: 4 messages (system + user + assistant_bad + user_feedback)
    assert len(provider._calls[1]) == 4
    assert provider._calls[1][2].role.value == "assistant"
    assert provider._calls[1][3].role.value == "user"


def test_function_words_are_exempt():
    mapping = {"de": "de", "huis": "huis", "is": "zijn"}
    response = "## Title\n**New words:** huis\nhuis"
    provider = _StubProvider([response])
    gen = LessonGenerator(
        provider, _MapLemmatizer(mapping), function_lemmas={"de", "zijn"}
    )
    result = gen.generate(
        "lesson003", ["huis"], {"huis"}, language="Dutch", model="stub"
    )
    assert "huis" in result.content


def test_generate_falls_back_after_validation_retries():
    provider = _StubProvider(["huis xyz"] * 5)
    gen = LessonGenerator(provider, _lemmatizer(["huis"]), max_retries=3)
    result = gen.generate(
        "lesson006", ["huis"], {"huis"}, language="Dutch", model="stub"
    )
    assert result.content == "huis."
    assert result.attempts == 3


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


def test_generate_falls_back_on_llm_error():
    gen = LessonGenerator(_ErrorProvider(), _lemmatizer(["huis"]))
    result = gen.generate(
        "lesson005", ["huis"], {"huis"}, language="Dutch", model="stub"
    )
    assert result.content == "huis."
