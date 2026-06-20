"""Tests for LessonOrchestrator."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.lesson import LessonGenerator
from course_compiler.generation.orchestrator import LessonOrchestrator
from course_compiler.generation.themes import LessonThemePlan, ThemeAssigner
from course_compiler.llm.base import LLMProvider, LLMResponse, PromptInput
from course_compiler.models import Frequency, PartOfSpeech, Verb, Word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUNCTION_POS = {
    PartOfSpeech.ARTICLE,
    PartOfSpeech.CONJUNCTION,
    PartOfSpeech.PREPOSITION,
}


def _word(
    lemma: str,
    pos: PartOfSpeech = PartOfSpeech.NOUN,
    cefr: str | None = "A1",
    rank: int = 1,
) -> Word:
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


class _StubPlannerThemeAssigner(_StubThemeAssigner):
    def __init__(
        self, mapping: dict[str, list[str]], plans: list[LessonThemePlan]
    ) -> None:
        super().__init__(mapping)
        self._plans = plans

    def plan_lessons(
        self,
        words: list[Word],
        *,
        cefr: str,
        words_per_lesson: int,
    ) -> list[LessonThemePlan]:
        return list(self._plans)


class _StubSelectingThemeAssigner(_StubThemeAssigner):
    def __init__(
        self,
        mapping: dict[str, list[str]],
        selections_by_theme: dict[str, list[str]],
    ) -> None:
        super().__init__(mapping)
        self._selections_by_theme = selections_by_theme
        self.calls: list[tuple[str, list[str], int]] = []

    def select_seed_lemmas_for_theme(
        self,
        *,
        cefr: str,
        theme: str,
        communicative_goals: list[str],
        target_count: int,
        already_used: list[str],
        candidate_lemmas: list[str],
    ) -> list[str]:
        self.calls.append((theme, communicative_goals, target_count))
        return list(self._selections_by_theme.get(theme, []))


class _StubProposingThemeAssigner(_StubThemeAssigner):
    def __init__(
        self,
        mapping: dict[str, list[str]],
        proposals_by_theme: dict[str, list[str]],
    ) -> None:
        super().__init__(mapping)
        self._proposals_by_theme = proposals_by_theme
        self.propose_calls: list[tuple[str, int, str]] = []

    def propose_theme_vocabulary(
        self,
        *,
        cefr: str,
        theme: str,
        communicative_goals: list[str],
        target_count: int,
        already_used: list[str],
        language: str = "",
    ) -> list[str]:
        self.propose_calls.append((theme, target_count, language))
        return list(self._proposals_by_theme.get(theme, []))


class _StubProvider(LLMProvider):
    """Returns the new-words list joined by spaces — always passes content-word validation."""

    def complete(
        self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
    ) -> LLMResponse:
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

    async def acomplete(
        self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature)


def _make_orchestrator(
    theme_mapping: dict[str, list[str]], words_per_lesson: int = 10
) -> LessonOrchestrator:
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubThemeAssigner(theme_mapping)
    return LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=words_per_lesson,
        predefined_themes={},
    )


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
    orc = _make_orchestrator(
        {"misc": [f"w{i}" for i in range(1, 11)]}, words_per_lesson=3
    )
    plans = orc.plan(words, cefr="A1")
    for plan in plans[:-1]:  # last lesson may have fewer
        assert len(plan.new_words) <= 3


def test_front_loaded_budget_gives_early_lessons_more_words():
    """Opt-in front-loading: lesson 1 introduces more words, tapering to steady state."""
    words = [_word(f"w{i:03d}", rank=i) for i in range(1, 101)]  # 100 words
    provider = _StubProvider()
    generator = LessonGenerator(provider, _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": [w.lemma for w in words]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        first_lesson_words=40,
        front_load_lessons=3,
        predefined_themes={},
    )

    plans = orc.plan(words, cefr="A1")
    counts = [len(p.new_words) for p in plans]

    # Linear taper: L1=40, L2=25, L3+=10 (steady state).
    assert counts[0] == 40
    assert counts[1] == 25
    assert counts[2] == 10
    assert counts[0] > counts[-1]
    # No word is dropped or duplicated.
    all_lemmas = [w.lemma for p in plans for w in p.new_words]
    assert len(all_lemmas) == len(set(all_lemmas)) == 100


def test_uniform_budget_is_the_default():
    """Without front-load params, every lesson uses words_per_lesson (back-compat)."""
    words = [_word(f"w{i:02d}", rank=i) for i in range(1, 21)]
    provider = _StubProvider()
    generator = LessonGenerator(provider, _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": [w.lemma for w in words]})
    orc = LessonOrchestrator(
        generator, assigner, words_per_lesson=5, predefined_themes={}
    )

    plans = orc.plan(words, cefr="A1")

    assert [len(p.new_words) for p in plans] == [5, 5, 5, 5]


def test_plan_lesson_ids_are_unique_and_sequential():
    words = [_word(f"w{i}", rank=i) for i in range(1, 6)]
    orc = _make_orchestrator(
        {"misc": [f"w{i}" for i in range(1, 6)]}, words_per_lesson=2
    )
    plans = orc.plan(words, cefr="A1")
    ids = [p.lesson_id for p in plans]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_plan_uses_lesson_blueprints_when_available():
    words = [_word("huis", rank=1), _word("deur", rank=2), _word("tafel", rank=3)]
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubPlannerThemeAssigner(
        {"misc": ["huis", "deur", "tafel"]},
        plans=[LessonThemePlan(theme="home", seed_lemmas=["huis", "deur"])],
    )
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,
        predefined_themes={},
    )

    plans = orc.plan(words, cefr="A1")

    assert plans[0].theme == "home"
    assert {w.lemma for w in plans[0].new_words} == {"huis", "deur"}


def test_plan_blueprints_still_cover_all_content_words():
    words = [_word("huis", rank=1), _word("deur", rank=2), _word("tafel", rank=3)]
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubPlannerThemeAssigner(
        {"misc": ["huis", "deur", "tafel"]},
        plans=[LessonThemePlan(theme="home", seed_lemmas=["huis"])],
    )
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,
        predefined_themes={},
    )

    plans = orc.plan(words, cefr="A1")

    all_lemmas = {w.lemma for p in plans for w in p.new_words}
    assert all_lemmas == {"huis", "deur", "tafel"}
    # ceil(3/2) = 2 lessons should be created overall.
    assert len(plans) == 2


# ---------------------------------------------------------------------------
# Tests: generate()
# ---------------------------------------------------------------------------


def test_generate_returns_one_lesson_per_plan():
    words = [_word("huis"), _word("deur")]
    orc = _make_orchestrator({"home": ["huis", "deur"]})
    lessons = orc.generate(words, language="Dutch", cefr="A1", model="stub")
    assert len(lessons) == 1


def _verb(infinitive: str, cefr: str = "A1", rank: int = 1, **forms: str) -> Verb:
    present = forms or {"ik": f"{infinitive}t", "jij": f"{infinitive}t"}
    return Verb(
        id=infinitive,
        language="nl",
        lemma=infinitive,
        infinitive=infinitive,
        cefr=cefr,
        frequency=Frequency(rank=rank),
        present=present,
    )


# ---------------------------------------------------------------------------
# Tests: plan() with verbs
# ---------------------------------------------------------------------------


def test_plan_verbs_appear_as_new_word_lemmas():
    """Verb infinitives must appear in the combined new-words list for a lesson."""
    words = [_word("huis")]
    verb = _verb("lopen", cefr="A1", rank=2)
    orc = _make_orchestrator({"misc": ["huis", "lopen"]})
    plans = orc.plan(words, cefr="A1", verbs=[verb])
    all_lemmas = {w.lemma for p in plans for w in p.new_words}
    verb_lemmas = {v.infinitive for p in plans for v in p.new_verbs}
    assert "huis" in all_lemmas
    assert "lopen" in verb_lemmas


def test_plan_verb_surface_forms_in_allowed_forms():
    """All conjugated forms of an introduced verb must appear in allowed_forms."""
    verb = _verb("lopen", cefr="A1", **{"ik": "loop", "jij": "loopt"})
    orc = _make_orchestrator({"misc": ["lopen"]})
    plans = orc.plan([], cefr="A1", verbs=[verb])
    assert len(plans) == 1
    assert "loop" in plans[0].allowed_forms
    assert "loopt" in plans[0].allowed_forms


def test_plan_verb_cefr_filtering():
    """Verbs with wrong CEFR level must be excluded."""
    verb_a1 = _verb("lopen", cefr="A1")
    verb_b1 = _verb("rennen", cefr="B1")
    orc = _make_orchestrator({"misc": ["lopen", "rennen"]})
    plans = orc.plan([], cefr="A1", verbs=[verb_a1, verb_b1])
    verb_lemmas = {v.infinitive for p in plans for v in p.new_verbs}
    assert "lopen" in verb_lemmas
    assert "rennen" not in verb_lemmas


def test_plan_verb_allowed_forms_accumulate_across_lessons():
    """allowed_forms for lesson N must include forms from all prior lessons' verbs."""
    verb1 = _verb("lopen", cefr="A1", rank=1, **{"ik": "loop"})
    verb2 = _verb("rennen", cefr="A1", rank=2, **{"ik": "ren"})
    orc = _make_orchestrator({"misc": ["lopen", "rennen"]}, words_per_lesson=1)
    plans = orc.plan([], cefr="A1", verbs=[verb1, verb2])
    assert len(plans) == 2
    assert "loop" in plans[0].allowed_forms
    assert "loop" in plans[1].allowed_forms  # accumulated
    assert "ren" in plans[1].allowed_forms


def test_generate_verb_forms_exempt_from_validation():
    """Conjugated forms of introduced verbs must not trigger validation violations."""
    verb = _verb("lopen", cefr="A1", **{"ik": "loop", "jij": "loopt"})

    class _VerbFormProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return LLMResponse(
                content="loop loopt lopen", model=model or "stub", raw={}
            )

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return self.complete(prompt, model=model, temperature=temperature)

    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(lemmatizer=lemmatizer, provider=_VerbFormProvider())
    assigner = _StubThemeAssigner({"misc": ["lopen"]})
    orc = LessonOrchestrator(generator, assigner, predefined_themes={})
    # Should not raise — verb forms are exempt via allowed_forms.
    lessons = orc.generate([], language="Dutch", cefr="A1", verbs=[verb], model="stub")
    assert lessons[0].content == "loop loopt lopen"


def test_generate_function_lemmas_passed_through():
    """function_lemmas from the plan are forwarded to the generator/validator."""
    article = _word("de", pos=PartOfSpeech.ARTICLE, cefr="A1")
    noun = _word("huis", pos=PartOfSpeech.NOUN, cefr="A1")

    # Provider returns a response that contains "de" (a function word).
    # Without function_lemmas wired through, "de" would fail validation.
    class _FixedProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return LLMResponse(content="de huis", model=model or "stub", raw={})

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return self.complete(prompt, model=model, temperature=temperature)

    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(lemmatizer=lemmatizer, provider=_FixedProvider())
    assigner = _StubThemeAssigner({"home": ["huis"]})  # "de" filtered as function word
    orc = LessonOrchestrator(generator, assigner, predefined_themes={})
    # Should not raise — "de" passes because it's in function_lemmas derived from the article.
    lessons = orc.generate([article, noun], language="Dutch", cefr="A1", model="stub")
    assert lessons[0].content == "de huis"


def test_plan_uses_predefined_themes_for_cefr_when_available():
    words = [_word("huis", rank=1), _word("deur", rank=2), _word("tafel", rank=3)]
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubPlannerThemeAssigner(
        {"misc": ["huis", "deur", "tafel"]},
        plans=[LessonThemePlan(theme="llm-home", seed_lemmas=["huis", "deur"])],
    )
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,
        predefined_themes={"A1": ["Configured Theme 1", "Configured Theme 2"]},
    )

    plans = orc.plan(words, cefr="A1")

    assert [p.theme for p in plans] == ["Configured Theme 1", "Configured Theme 2"]


def test_plan_spreads_content_across_all_predefined_themes_when_more_than_implied():
    words = [
        _word("w1", rank=1),
        _word("w2", rank=2),
        _word("w3", rank=3),
        _word("w4", rank=4),
    ]
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubThemeAssigner({"misc": ["w1", "w2", "w3", "w4"]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={"A1": ["T1", "T2", "T3", "T4"]},
    )

    plans = orc.plan(words, cefr="A1")

    assert [p.theme for p in plans] == ["T1", "T2", "T3", "T4"]
    assert [len(p.new_words) for p in plans] == [1, 1, 1, 1]


def test_predefined_themes_cover_all_content_even_when_few_themes():
    """One lesson per theme distributes ALL vocabulary; nothing is dropped."""
    words = [_word(f"w{i:02d}", rank=i) for i in range(1, 11)]  # 10 words
    provider = _StubProvider()
    generator = LessonGenerator(provider, _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": [w.lemma for w in words]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,  # would, if naively chunked, cover only 2*2=4 words
        predefined_themes={"A1": ["T1", "T2"]},
    )

    plans = orc.plan(words, cefr="A1")

    assert [p.theme for p in plans] == ["T1", "T2"]
    all_lemmas = [w.lemma for p in plans for w in p.new_words]
    assert len(all_lemmas) == len(set(all_lemmas)) == 10  # every word, once


def test_plan_predefined_themes_prefer_proposed_vocabulary_filtered_to_lexicon():
    """LLM proposes theme words; orchestrator keeps only lexicon hits, freq-ranked."""
    words = [_word("koffie", rank=2), _word("thee", rank=1), _word("water", rank=3)]
    provider = _StubProvider()
    generator = LessonGenerator(provider, _IdentityLemmatizer())
    assigner = _StubProposingThemeAssigner(
        {"misc": ["koffie", "thee", "water"]},
        proposals_by_theme={
            # 'espresso' is theme-relevant but absent from our lexicon -> dropped.
            "Cafe": ["espresso", "thee", "koffie"],
        },
    )
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,
        predefined_themes={
            "A1": [
                LessonThemePlan(
                    theme="Cafe",
                    seed_lemmas=[],
                    communicative_goals=["order drinks"],
                )
            ]
        },
    )

    plans = orc.plan(words, cefr="A1", language="Dutch")

    # Out-of-lexicon proposal ('espresso') dropped; the three in-lexicon words are
    # kept, frequency-ranked (thee=1 < koffie=2 < water=3).
    assert [w.lemma for w in plans[0].new_words] == ["thee", "koffie", "water"]
    all_lemmas = {w.lemma for p in plans for w in p.new_words}
    assert "espresso" not in all_lemmas
    # The proposer was consulted with the lesson's target count and language.
    assert assigner.propose_calls and assigner.propose_calls[0][0] == "Cafe"
    assert assigner.propose_calls[0][2] == "Dutch"


def test_plan_predefined_themes_use_goal_aware_seed_selection_with_fallback():
    words = [
        _word("koffie", rank=1),
        _word("thee", rank=2),
        _word("water", rank=3),
        _word("brood", rank=4),
    ]
    provider = _StubProvider()
    lemmatizer = _IdentityLemmatizer()
    generator = LessonGenerator(provider, lemmatizer)
    assigner = _StubSelectingThemeAssigner(
        {"misc": ["koffie", "thee", "water", "brood"]},
        selections_by_theme={
            "Cafe": ["koffie", "thee"],
            # Includes an unknown lemma to assert filtering + deterministic fallback.
            "Bakery": ["onbekend"],
        },
    )

    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,
        predefined_themes={
            "A1": [
                LessonThemePlan(
                    theme="Cafe",
                    seed_lemmas=[],
                    communicative_goals=["order drinks"],
                ),
                LessonThemePlan(
                    theme="Bakery",
                    seed_lemmas=[],
                    communicative_goals=["buy bread"],
                ),
            ]
        },
    )

    plans = orc.plan(words, cefr="A1")

    assert [p.theme for p in plans] == ["Cafe", "Bakery"]
    assert {w.lemma for w in plans[0].new_words} == {"koffie", "thee"}
    assert {w.lemma for w in plans[1].new_words} == {"water", "brood"}
    assert assigner.calls == [
        ("Cafe", ["order drinks"], 2),
        ("Bakery", ["buy bread"], 2),
    ]
