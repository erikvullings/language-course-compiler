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
        self._temperatures: list[float | None] = []

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
        self._temperatures.append(temperature)
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


class _DraftThenErrorProvider(LLMProvider):
    """Returns one draft on the first call, then raises ``LLMError`` thereafter."""

    def __init__(self, first: str) -> None:
        self._first = first
        self._calls = 0

    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        self._calls += 1
        if self._calls == 1:
            return LLMResponse(content=self._first, model=model or "stub", raw={})
        raise LLMError("timed out")

    async def acomplete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature, **kwargs)


def _lemmatizer(words: list[str]) -> _MapLemmatizer:
    return _MapLemmatizer({w: w for w in words})


# ---------------------------------------------------------------------------
# Target length scaling
# ---------------------------------------------------------------------------


def test_parse_strips_echoed_title_placeholder():
    """Weak models echo the literal '## Lesson Title' header; strip that prefix."""
    title, _ = _parse_lesson_structure(
        "## Lesson Title: Begroetingen\n\nHallo daar."
    )
    assert title == "Begroetingen"


def test_parse_drops_bare_title_placeholder():
    """A bare echoed placeholder yields a neutral default, not 'Lesson Title'."""
    title, _ = _parse_lesson_structure(
        "## Lesson Title\n\nHallo daar."
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


def test_generate_language_in_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    assert "Dutch" in provider._calls[0][0].content


def test_generate_cefr_in_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A1", model="stub"
    )
    assert "A1" in provider._calls[0][0].content


def test_generate_theme_in_prompt():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", theme="home", model="stub"
    )
    assert "home" in provider._calls[0][0].content


def test_generate_outline_in_prompt_and_forces_narrative():
    """A lesson outline is included in the prompt and switches to narrative format."""
    provider = _StubProvider(["## T\nhuis"])
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
    prompt = provider._calls[0][0].content.lower()
    assert "two neighbours meet" in prompt
    assert "narrative" in prompt


def test_generate_shows_english_glosses_for_new_words():
    """New words are annotated with their English meaning to fix the sense."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001",
        ["huis", "eten"],
        {"huis", "eten"},
        language="Dutch",
        model="stub",
        glosses={"huis": "house", "eten": "to eat"},
    )
    prompt = provider._calls[0][0].content
    assert "huis (house)" in prompt
    assert "eten (to eat)" in prompt


def test_generate_lists_verbs_with_soft_instruction():
    """Selected verbs are surfaced, but their use is encouraged, not mandated.

    A few may be left out if the text reads better — the validator never enforces
    their presence, so over-constraining the writer only degrades naturalness.
    """
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001",
        ["huis", "hebben"],
        {"huis", "hebben"},
        language="Dutch",
        model="stub",
        verb_lemmas=["hebben"],
    )
    prompt = provider._calls[0][0].content.lower()
    assert "hebben" in prompt
    # No longer a hard "build your sentences around" mandate.
    assert "build your sentences around these verbs" not in prompt
    # Soft phrasing: most verbs, where they fit; some may be omitted.
    assert "where they fit naturally" in prompt
    assert "left out" in prompt or "omit" in prompt


def test_verbs_not_duplicated_in_word_list():
    """Verbs appear only in their dedicated line, not also in the general word list."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001",
        ["huis", "lopen", "kijken"],
        {"huis", "lopen", "kijken"},
        language="Dutch",
        model="stub",
        verb_lemmas=["lopen", "kijken"],
    )
    prompt = provider._calls[0][0].content
    assert prompt.count("lopen") == 1
    assert prompt.count("kijken") == 1
    assert "huis" in prompt  # the non-verb word still appears


def test_narrative_prompt_states_theme_and_outline_together():
    """The narrative directive names both the theme and the scene/outline."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},
        language="Dutch",
        theme="Time",
        outline="A traveller asks about the time at a station.",
        model="stub",
    )
    prompt = provider._calls[0][0].content
    assert 'theme is "Time"' in prompt
    assert "scene is: A traveller asks about the time" in prompt


def test_a2_tense_guidance_is_lighter_than_a1():
    """A2 lessons drop the present-tense-first mandate but still steer simple tenses."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A2", model="stub"
    )
    prompt = provider._calls[0][0].content
    assert "Prefer present tense" not in prompt
    assert "simple past" in prompt.lower()


def test_b1_has_no_tense_restriction():
    """B1+ lessons carry no tense guidance at all."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", cefr="B1", model="stub"
    )
    prompt = provider._calls[0][0].content
    assert "Prefer present tense" not in prompt
    assert "simple past" not in prompt.lower()


def test_mature_lesson_relaxes_a1_tense_constraints():
    """Once the learner has a large vocabulary, A1 tense constraints relax."""
    allowed = {f"w{i}" for i in range(400)}
    provider = _StubProvider(["## T\nw0"])
    gen = LessonGenerator(
        provider,
        _MapLemmatizer({w: w for w in allowed}),
        mature_vocab_threshold=300,
    )
    gen.generate("lesson050", ["w0"], allowed, language="Dutch", cefr="A1", model="stub")
    prompt = provider._calls[0][0].content
    assert "Prefer present tense" not in prompt
    assert "Avoid complex tense combinations" not in prompt


def test_early_lesson_uses_example_format_when_vocab_is_small():
    """With little vocabulary to recombine, ask for simple example sentences."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    prompt = provider._calls[0][0].content.lower()
    assert "example sentences" in prompt
    assert "short narrative" not in prompt


def test_mature_lesson_uses_narrative_format_when_vocab_is_large():
    """Once a base vocabulary exists, ask for a coherent narrative."""
    allowed = {f"w{i}" for i in range(80)}
    provider = _StubProvider(["## T\nw0"])
    gen = LessonGenerator(provider, _MapLemmatizer({w: w for w in allowed}))
    gen.generate("lesson050", ["w0"], allowed, language="Dutch", model="stub")
    prompt = provider._calls[0][0].content.lower()
    assert "narrative" in prompt


def test_narrative_threshold_is_configurable():
    """Lowering the threshold flips a small-vocab lesson into narrative format."""
    allowed = {"huis", "deur"}
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(
        provider, _MapLemmatizer({"huis": "huis", "deur": "deur"}),
        narrative_vocab_threshold=2,
    )
    gen.generate("lesson001", ["huis"], allowed, language="Dutch", model="stub")
    assert "narrative" in provider._calls[0][0].content.lower()


def test_low_cefr_prompt_allows_frequent_simple_past_forms():
    provider = _StubProvider(["huis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", cefr="A1", model="stub"
    )
    prompt = provider._calls[0][0].content
    assert "Prefer present tense" in prompt
    assert "equivalents of 'was/were'" in prompt


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
    response = "## Home Lesson\nhuis zijn groot."
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
    response = "## Home\nhuis tafel"
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
    bad_response = "## Home\nappartement"
    good_response = "## Home\nhuis"
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
    bad = "## Home\nappartement"
    good = "## Home\nhuis"
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
    # Second call: [user_prompt, assistant_bad, user_feedback]
    feedback = provider._calls[1][2].content.lower()
    # Instructs a minimal revision of the previous version, not a fresh rewrite.
    assert "revise" in feedback
    assert "rewrite from scratch" in feedback
    # Still names the offending word to remove/replace.
    assert "appartement" in feedback


def test_retry_messages_form_multi_turn_conversation():
    """On retry the conversation appends assistant + user feedback, not a fresh prompt."""
    mapping = {"huis": "huis", "bad": "bad"}
    bad = "## Title\nbad"
    good = "## Title\nhuis"
    provider = _StubProvider([bad, good])
    gen = LessonGenerator(provider, _MapLemmatizer(mapping))
    gen.generate(
        "lesson001",
        ["huis"],
        {"huis"},
        language="Dutch",
        model="stub",
    )
    # First call: 1 message (user prompt only)
    assert len(provider._calls[0]) == 1
    # Second call: 3 messages (user_prompt + assistant_bad + user_feedback)
    assert len(provider._calls[1]) == 3
    assert provider._calls[1][1].role.value == "assistant"
    assert provider._calls[1][2].role.value == "user"


def test_many_violations_restart_fresh_without_history():
    """When a draft leaks many words, the next attempt re-issues the original prompt.

    Accumulating a badly-broken draft anchors the model to it; past the revise
    threshold we drop the history and resample from the original prompt instead.
    """
    mapping = {"huis": "huis"}
    bad = "## T\naaa bbb ccc"  # three unknown content words → 3 violations
    good = "## T\nhuis"
    provider = _StubProvider([bad, good])
    gen = LessonGenerator(
        provider, _MapLemmatizer(mapping), revise_violation_threshold=2
    )
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    # Second call is a fresh single-message prompt identical to the first — no
    # appended assistant draft or feedback turn.
    assert len(provider._calls[1]) == 1
    assert provider._calls[1][0].content == provider._calls[0][0].content


def test_few_violations_still_revise_with_history():
    """At or below the revise threshold, keep the minimal-revision-with-history path."""
    mapping = {"huis": "huis", "bad": "bad"}
    bad = "## T\nbad"  # single violation
    good = "## T\nhuis"
    provider = _StubProvider([bad, good])
    gen = LessonGenerator(
        provider, _MapLemmatizer(mapping), revise_violation_threshold=2
    )
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    assert len(provider._calls[1]) == 3  # prompt + assistant draft + feedback


def test_fresh_restart_resamples_with_higher_temperature():
    """A fresh restart bumps the sampling temperature to escape the bad draft."""
    mapping = {"huis": "huis"}
    bad = "## T\naaa bbb ccc"
    good = "## T\nhuis"
    provider = _StubProvider([bad, good])
    gen = LessonGenerator(
        provider, _MapLemmatizer(mapping), revise_violation_threshold=2
    )
    gen.generate(
        "lesson001", ["huis"], {"huis"}, language="Dutch", model="stub", temperature=0.6
    )
    assert provider._temperatures[0] == 0.6
    assert provider._temperatures[1] > 0.6


def test_final_attempt_restarts_fresh_even_with_few_violations():
    """The last retry always goes fresh rather than burning it on a doomed revision."""
    mapping = {"huis": "huis", "bad": "bad"}
    # Every attempt leaks the single word 'bad'; with max_retries=3 the third
    # (final) attempt must be a fresh prompt, not a third feedback turn.
    provider = _StubProvider(["## T\nbad"] * 3)
    gen = LessonGenerator(
        provider,
        _MapLemmatizer(mapping),
        max_retries=3,
        revise_violation_threshold=2,
    )
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    # Attempt 2 (after 1 light failure): revise with history.
    assert len(provider._calls[1]) == 3
    # Attempt 3 (final): fresh restart — single original prompt message.
    assert len(provider._calls[2]) == 1
    assert provider._calls[2][0].content == provider._calls[0][0].content


def test_example_format_word_coverage_instruction_is_soft():
    """Example-format lessons should cover most new words, not force every one."""
    provider = _StubProvider(["## T\nhuis"])
    gen = LessonGenerator(provider, _lemmatizer(["huis"]))
    gen.generate("lesson001", ["huis"], {"huis"}, language="Dutch", model="stub")
    prompt = provider._calls[0][0].content.lower()
    assert "each new word should appear in at least one sentence" not in prompt
    assert "most of the new words" in prompt


def test_function_words_are_exempt():
    mapping = {"de": "de", "huis": "huis", "is": "zijn"}
    response = "## Title\nhuis"
    provider = _StubProvider([response])
    gen = LessonGenerator(
        provider, _MapLemmatizer(mapping), function_lemmas={"de", "zijn"}
    )
    result = gen.generate(
        "lesson003", ["huis"], {"huis"}, language="Dutch", model="stub"
    )
    assert "huis" in result.content


def test_validation_fallback_uses_best_draft_not_word_soup():
    """When no draft validates, return the least-bad draft, not a word placeholder."""
    mapping = {"huis": "huis"}
    responses = [
        "## A\nhuis bad1 bad2 bad3",  # 3 violations
        "## B\nhuis bad1",  # 1 violation — the best draft
        "## C\nhuis bad1 bad2",  # 2 violations
        "## D\nhuis bad1 bad2 bad3 bad4",  # 4 violations
        "## E\nhuis bad1 bad2 bad3",  # 3 violations
    ]
    provider = _StubProvider(responses)
    gen = LessonGenerator(provider, _MapLemmatizer(mapping), max_retries=5)
    result = gen.generate(
        "lesson006", ["huis"], {"huis"}, language="Dutch", model="stub"
    )
    assert result.content == "huis bad1"  # narrative of the least-bad draft
    assert result.title == "B"
    assert result.fallback is True
    assert result.violations == frozenset({"bad1"})
    assert result.attempts == 5


def test_validation_fallback_logs_severity(caplog):
    """How badly the lesson failed (violation count + words) is logged at WARNING."""
    import logging

    provider = _StubProvider(["## A\nhuis bad1 bad2"] * 3)
    gen = LessonGenerator(provider, _MapLemmatizer({"huis": "huis"}), max_retries=3)
    with caplog.at_level(logging.WARNING):
        gen.generate("lesson006", ["huis"], {"huis"}, language="Dutch", model="stub")
    text = caplog.text.lower()
    assert "bad1" in text and "bad2" in text
    assert "lesson006" in text


def test_llm_error_falls_back_to_best_prior_draft():
    """A timeout after a usable draft returns that draft, not a placeholder."""
    provider = _DraftThenErrorProvider("## A\nhuis bad1")
    gen = LessonGenerator(provider, _MapLemmatizer({"huis": "huis"}))
    result = gen.generate(
        "lesson007", ["huis"], {"huis"}, language="Dutch", model="stub"
    )
    assert result.content == "huis bad1"
    assert result.title == "A"
    assert result.fallback is True
    assert result.violations == frozenset({"bad1"})


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
