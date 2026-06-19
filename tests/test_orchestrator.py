"""Tests for LessonOrchestrator."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.lesson import LessonGenerator
from course_compiler.generation.orchestrator import LessonOrchestrator
from course_compiler.generation.themes import ThemeAssigner
from course_compiler.llm.base import LLMProvider, LLMResponse, PromptInput
from course_compiler.models import Frequency, PartOfSpeech, Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUNCTION_POS = {PartOfSpeech.ARTICLE, PartOfSpeech.CONJUNCTION, PartOfSpeech.PREPOSITION}


def _word(lemma: str, pos: PartOfSpeech = PartOfSpeech.NOUN, cefr: str | None = "A1", rank: int = 1) -> Word:
    return Word(
        id=lemma,
        language="nl",
        lemma=lemma,
        normalized=lemma,
        part_of_speech=pos,
        cefr=cefr,
        frequency=Frequency(rank=rank),
    )


class _IdentityLemmatizer(Lemmatizer):
    @property
    def language(self) -> str:
        return "nl"

    def lemmatize(self, token: str) -> str | None:
        return token.lower()


class _StubThemeAssigner(ThemeAssigner):
    """Returns a fixed theme mapping."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping  # theme -> [lemma, ...]

    def assign(self, words: list[Word]) -> dict[str, list[Word]]:
        by_lemma = {w.lemma: w for w in words}
        result: dict[str, list[Word]] = {}
        assigned: set[str] = set()
        for theme, lemmas in self._mapping.items():
            result[theme] = [by_lemma[l] for l in lemmas if l in by_lemma]
            assigned.update(lemmas)
        leftover = [w for w in words if w.lemma not in assigned]
        if leftover:
            result.setdefault("misc", []).extend(leftover)
        return result


class _StubProvider(LLMProvider):
    """Returns the new-words list joined by spaces — always passes content-word validation."""

    def complete(self, prompt: PromptInput, *, model=None, temperature=None, **kwargs) -> LLMResponse:
        from course_compiler.llm.base import to_messages
        # Extract the new words from the user message and echo them.
        messages = to_messages(prompt)
        user_msg = next((m.content for m in messages if m.role.value == "user"), "")
        # Pull "New content words introduced in this lesson: w1, w2, ..." line.
        for line in user_msg.splitlines():
            if line.startswith("New content words"):
                words = line.split(":", 1)[1].strip()
                return LLMResponse(content=words, model=model or "stub", raw={})
        return LLMResponse(content="", model=model or "stub", raw={})

    async def acomplete(self, prompt: PromptInput, *, model=None, temperature=None, **kwargs) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature)


def _make_orchestrator(theme_mapping: dict[str, list[str]], words_per_lesson: int = 10) -> LessonOrchestrator:
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubThemeAssigner(theme_mapping)
    return LessonOrchestrator(generator, assigner, words_per_lesson=words_per_lesson)


# ---------------------------------------------------------------------------
# Tests: plan()
# ---------------------------------------------------------------------------

def test_plan_filters_by_cefr():
    words = [_word("huis", cefr="A1"), _word("appartement", cefr="B1")]
    orc = _make_orchestrator({"home": ["huis", "appartement"]})
    plans = orc.plan(words, cefr="A1")
    all_lemmas = {w.lemma for p in plans for w in p.new_words}
    assert "huis" in all_lemmas
    assert "appartement" not in all_lemmas


def test_plan_excludes_words_without_cefr():
    words = [_word("huis", cefr="A1"), _word("xyz", cefr=None)]
    orc = _make_orchestrator({"home": ["huis", "xyz"]})
    plans = orc.plan(words, cefr="A1")
    all_lemmas = {w.lemma for p in plans for w in p.new_words}
    assert "xyz" not in all_lemmas


def test_plan_function_words_not_in_new_words():
    article = _word("de", pos=PartOfSpeech.ARTICLE, cefr="A1")
    noun = _word("huis", pos=PartOfSpeech.NOUN, cefr="A1")
    orc = _make_orchestrator({"home": ["huis", "de"]})
    plans = orc.plan([article, noun], cefr="A1")
    new_lemmas = {w.lemma for p in plans for w in p.new_words}
    assert "huis" in new_lemmas
    assert "de" not in new_lemmas  # function word — never a "new" word in a lesson


def test_plan_allowed_lemmas_grows_across_lessons():
    """allowed_lemmas for lesson N must include all lemmas from lessons 1..N-1."""
    words = [_word(f"word{i}", rank=i) for i in range(1, 6)]
    theme_map = {"misc": [f"word{i}" for i in range(1, 6)]}
    orc = _make_orchestrator(theme_map, words_per_lesson=2)
    plans = orc.plan(words, cefr="A1")
    # Each plan's allowed_lemmas is a superset of all prior plans' new word lemmas.
    seen: set[str] = set()
    for plan in plans:
        assert seen.issubset(plan.allowed_lemmas)
        seen.update(w.lemma for w in plan.new_words)


def test_plan_sorted_by_frequency():
    """Within a theme, words with lower rank (more frequent) come first."""
    words = [_word("rare", rank=100), _word("common", rank=1)]
    orc = _make_orchestrator({"misc": ["rare", "common"]})
    plans = orc.plan(words, cefr="A1")
    all_new = [w.lemma for p in plans for w in p.new_words]
    assert all_new.index("common") < all_new.index("rare")


def test_plan_words_per_lesson_respected():
    words = [_word(f"w{i}", rank=i) for i in range(1, 11)]
    orc = _make_orchestrator({"misc": [f"w{i}" for i in range(1, 11)]}, words_per_lesson=3)
    plans = orc.plan(words, cefr="A1")
    for plan in plans[:-1]:  # last lesson may have fewer
        assert len(plan.new_words) <= 3


def test_plan_lesson_ids_are_unique_and_sequential():
    words = [_word(f"w{i}", rank=i) for i in range(1, 6)]
    orc = _make_orchestrator({"misc": [f"w{i}" for i in range(1, 6)]}, words_per_lesson=2)
    plans = orc.plan(words, cefr="A1")
    ids = [p.lesson_id for p in plans]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Tests: generate()
# ---------------------------------------------------------------------------

def test_generate_returns_one_lesson_per_plan():
    words = [_word("huis"), _word("deur")]
    orc = _make_orchestrator({"home": ["huis", "deur"]})
    lessons = orc.generate(words, language="Dutch", cefr="A1", model="stub")
    assert len(lessons) == 1


def test_generate_function_lemmas_passed_through():
    """function_lemmas from the plan are forwarded to the generator/validator."""
    article = _word("de", pos=PartOfSpeech.ARTICLE, cefr="A1")
    noun = _word("huis", pos=PartOfSpeech.NOUN, cefr="A1")

    # Provider returns a response that contains "de" (a function word).
    # Without function_lemmas wired through, "de" would fail validation.
    class _FixedProvider(LLMProvider):
        def complete(self, prompt: PromptInput, *, model=None, temperature=None, **kwargs) -> LLMResponse:
            return LLMResponse(content="de huis", model=model or "stub", raw={})

        async def acomplete(self, prompt: PromptInput, *, model=None, temperature=None, **kwargs) -> LLMResponse:
            return self.complete(prompt, model=model, temperature=temperature)

    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(lemmatizer=lemmatizer, provider=_FixedProvider())
    assigner = _StubThemeAssigner({"home": ["huis"]})  # "de" filtered as function word
    orc = LessonOrchestrator(generator, assigner)
    # Should not raise — "de" passes because it's in function_lemmas derived from the article.
    lessons = orc.generate([article, noun], language="Dutch", cefr="A1", model="stub")
    assert lessons[0].content == "de huis"
