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
    en: str | None = None,
) -> Word:
    return Word(
        id=lemma,
        language="nl",
        lemma=lemma,
        normalized=lemma,
        part_of_speech=pos,
        cefr=cefr,
        frequency=Frequency(rank=rank),
        translations={"en": en} if en else {},
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
        self.seed_words_calls: list[list[str]] = []

    def propose_theme_vocabulary(
        self,
        *,
        cefr: str,
        theme: str,
        communicative_goals: list[str],
        target_count: int,
        already_used: list[str],
        language: str = "",
        seed_words: list[str] | None = None,
    ) -> list[str]:
        self.propose_calls.append((theme, target_count, language))
        self.seed_words_calls.append(list(seed_words or []))
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


def test_resolve_seed_pairs_disambiguates_by_theme():
    """A seed with multiple gloss matches resolves to the theme-relevant sense.

    Two lemmas share the primary English gloss 'bank'; the financial sense is the
    rarer (higher-rank) word, so a pure frequency pick would wrongly choose the
    seat. With theme context, the lemma whose metadata overlaps the theme wins.
    """
    from course_compiler.generation.orchestrator import _resolve_seed_pairs

    seat = _word("zitbank", en="bank (a long seat)", rank=1)
    seat.synonyms = ["chair"]
    fin = _word("geldbank", en="bank (financial institution)", rank=9)
    fin.synonyms = ["account"]
    words = [seat, fin]

    # Without theme context: frequency wins → the seat.
    assert _resolve_seed_pairs(words, ["bank"]) == [("bank", "zitbank")]

    # With theme tokens that overlap the financial sense: the bank account wins.
    pairs = _resolve_seed_pairs(words, ["bank"], theme_tokens={"money", "account"})
    assert pairs == [("bank", "geldbank")]


def test_resolve_seed_pairs_frequency_tiebreak_within_theme():
    """When theme overlap ties, the more frequent gloss match is chosen."""
    from course_compiler.generation.orchestrator import _resolve_seed_pairs

    a = _word("banka", en="bank", rank=5)
    b = _word("bankb", en="bank", rank=2)
    pairs = _resolve_seed_pairs([a, b], ["bank"], theme_tokens={"unrelated"})
    assert pairs == [("bank", "bankb")]


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


def test_generate_iter_streams_plan_and_lesson_pairs():
    """generate_iter yields (plan, lesson) one at a time for incremental writing."""
    words = [_word(f"w{i}", rank=i) for i in range(1, 6)]
    orc = _make_orchestrator(
        {"misc": [f"w{i}" for i in range(1, 6)]}, words_per_lesson=2
    )
    pairs = list(orc.generate_iter(words, language="Dutch", cefr="A1", model="stub"))
    assert pairs  # at least one lesson
    for plan, lesson in pairs:
        assert plan.lesson_id == lesson.lesson_id  # paired correctly
    # Same lessons as the batch API, preserving order.
    batch = orc.generate(words, language="Dutch", cefr="A1", model="stub")
    assert [lesson.lesson_id for _p, lesson in pairs] == [
        lesson.lesson_id for lesson in batch
    ]


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


def test_generate_non_introduced_verb_forms_are_in_level_not_unresolved():
    """Conjugations of CEFR-level verbs stay valid even when the infinitive is not introduced yet."""

    class _VerbFormProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return LLMResponse(content="ben is huis", model=model or "stub", raw={})

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return self.complete(prompt, model=model, temperature=temperature)

    words = [_word("huis", cefr="A1")]
    # 'zijn' exists in the CEFR-level verb inventory, but is not introduced in this lesson.
    zijn = _verb("zijn", cefr="A1", ik="ben", hij="is")

    generator = LessonGenerator(
        lemmatizer=_IdentityLemmatizer(),
        provider=_VerbFormProvider(),
        extra_tolerance=None,
    )
    assigner = _StubThemeAssigner({"home": ["huis"]})
    orc = LessonOrchestrator(generator, assigner, predefined_themes={})

    lessons = orc.generate(
        words, language="Dutch", cefr="A1", verbs=[zijn], model="stub"
    )
    assert lessons[0].fallback is False
    assert lessons[0].violations == frozenset()


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


def test_predefined_themes_cap_lesson_size_and_teach_a_subset():
    """Each lesson introduces at most ``words_per_lesson`` new words.

    A lexicon larger than ``themes × budget`` is taught as a curated subset
    (most-frequent-first) rather than crammed into a few enormous lessons — the
    point of a graded course. (Previously this path distributed *all* vocabulary,
    which dumped hundreds of new words into early lessons.)
    """
    words = [_word(f"w{i:02d}", rank=i) for i in range(1, 11)]  # 10 words
    provider = _StubProvider()
    generator = LessonGenerator(provider, _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": [w.lemma for w in words]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=2,
        predefined_themes={"A1": ["T1", "T2"]},
    )

    plans = orc.plan(words, cefr="A1")

    assert [p.theme for p in plans] == ["T1", "T2"]
    # Capped at 2 new words per lesson: 2 lessons × 2 = 4 of the 10 words, the most
    # frequent first; the surplus is simply not taught in this run.
    assert [len(p.new_words) for p in plans] == [2, 2]
    all_lemmas = [w.lemma for p in plans for w in p.new_words]
    assert len(all_lemmas) == len(set(all_lemmas)) == 4
    assert set(all_lemmas) == {"w01", "w02", "w03", "w04"}


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

    # Out-of-lexicon proposal ('espresso') dropped; the surviving in-lexicon words
    # are kept frequency-ranked (thee=1 < koffie=2) and capped at words_per_lesson=2,
    # so 'water' is not reached in this single capped lesson.
    assert [w.lemma for w in plans[0].new_words] == ["thee", "koffie"]
    all_lemmas = {w.lemma for p in plans for w in p.new_words}
    assert "espresso" not in all_lemmas
    # The proposer was consulted with the lesson's target count and language.
    assert assigner.propose_calls and assigner.propose_calls[0][0] == "Cafe"
    assert assigner.propose_calls[0][2] == "Dutch"


# ---------------------------------------------------------------------------
# Tests: noun/verb homographs (item identity = (lemma, pos))
# ---------------------------------------------------------------------------


class _BothSensesAssigner(ThemeAssigner):
    """Theme assigner that returns every content item (no lemma collapsing)."""

    def assign(self, words: list[Word]) -> dict[str, list[Word]]:
        return {"food": list(words)}


def test_homograph_catalog_path_teaches_both_noun_and_verb():
    """A lemma that is both a noun and a verb yields two items; neither is dropped."""
    noun = _word("eten", pos=PartOfSpeech.NOUN, cefr="A1", rank=1)
    other = _word("huis", pos=PartOfSpeech.NOUN, cefr="A1", rank=3)
    verb = _verb("eten", cefr="A1", rank=2, **{"ik": "eet", "jij": "eet"})

    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": ["eten", "huis"]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={"A1": ["T1"]},
    )

    plans = orc.plan([noun, other], cefr="A1", verbs=[verb])

    noun_lemmas = {w.lemma for p in plans for w in p.new_words}
    verb_lemmas = {v.infinitive for p in plans for v in p.new_verbs}
    assert "eten" in noun_lemmas  # the noun sense is taught
    assert "eten" in verb_lemmas  # the verb sense is taught
    assert "huis" in noun_lemmas


def test_homograph_blueprint_path_teaches_both_noun_and_verb():
    noun = _word("eten", pos=PartOfSpeech.NOUN, cefr="A1", rank=1)
    verb = _verb("eten", cefr="A1", rank=2, **{"ik": "eet"})
    assigner = _StubPlannerThemeAssigner(
        {"misc": ["eten"]},
        plans=[LessonThemePlan(theme="food", seed_lemmas=["eten"])],
    )
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    orc = LessonOrchestrator(
        generator, assigner, words_per_lesson=10, predefined_themes={}
    )

    plans = orc.plan([noun], cefr="A1", verbs=[verb])

    noun_lemmas = {w.lemma for p in plans for w in p.new_words}
    verb_lemmas = {v.infinitive for p in plans for v in p.new_verbs}
    assert "eten" in noun_lemmas
    assert "eten" in verb_lemmas


def test_homograph_default_theme_path_keeps_noun_sense():
    noun = _word("eten", pos=PartOfSpeech.NOUN, cefr="A1", rank=1)
    verb = _verb("eten", cefr="A1", rank=2, **{"ik": "eet"})
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    orc = LessonOrchestrator(
        generator, _BothSensesAssigner(), words_per_lesson=10, predefined_themes={}
    )

    plans = orc.plan([noun], cefr="A1", verbs=[verb])

    noun_lemmas = {w.lemma for p in plans for w in p.new_words}
    verb_lemmas = {v.infinitive for p in plans for v in p.new_verbs}
    assert "eten" in noun_lemmas
    assert "eten" in verb_lemmas


def test_non_homograph_lemma_yields_exactly_one_item():
    """A lemma that is only a noun (or only a verb) is not duplicated."""
    noun = _word("huis", pos=PartOfSpeech.NOUN, cefr="A1", rank=1)
    verb = _verb("lopen", cefr="A1", rank=2, **{"ik": "loop"})
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": ["huis", "lopen"]})
    orc = LessonOrchestrator(
        generator, assigner, words_per_lesson=10, predefined_themes={"A1": ["T1"]}
    )

    plans = orc.plan([noun], cefr="A1", verbs=[verb])

    noun_lemmas = [w.lemma for p in plans for w in p.new_words]
    verb_lemmas = [v.infinitive for p in plans for v in p.new_verbs]
    assert noun_lemmas.count("huis") == 1
    assert verb_lemmas.count("lopen") == 1


def test_homograph_both_senses_pass_validation_once_taught():
    """Both the noun form and a verb form are accepted by the validator."""
    noun = _word("eten", pos=PartOfSpeech.NOUN, cefr="A1", rank=1)
    verb = _verb("eten", cefr="A1", rank=2, **{"ik": "eet", "jij": "eet"})

    class _NounAndFormProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return LLMResponse(content="eten eet", model=model or "stub", raw={})

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return self.complete(prompt, model=model, temperature=temperature)

    generator = LessonGenerator(
        lemmatizer=_IdentityLemmatizer(), provider=_NounAndFormProvider()
    )
    assigner = _StubThemeAssigner({"misc": ["eten"]})
    orc = LessonOrchestrator(generator, assigner, predefined_themes={"A1": ["T1"]})

    lessons = orc.generate(
        [noun], language="Dutch", cefr="A1", verbs=[verb], model="stub"
    )
    assert lessons[0].content == "eten eet"


def test_homograph_plan_is_deterministic():
    noun = _word("eten", pos=PartOfSpeech.NOUN, cefr="A1", rank=1)
    other = _word("huis", pos=PartOfSpeech.NOUN, cefr="A1", rank=3)
    verb = _verb("eten", cefr="A1", rank=2, **{"ik": "eet"})
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": ["eten", "huis"]})
    orc = LessonOrchestrator(
        generator, assigner, words_per_lesson=10, predefined_themes={"A1": ["T1"]}
    )

    first = orc.plan([noun, other], cefr="A1", verbs=[verb])
    second = orc.plan([noun, other], cefr="A1", verbs=[verb])

    def shape(plans):
        return [
            (
                p.lesson_id,
                p.theme,
                tuple(w.lemma for w in p.new_words),
                tuple(v.infinitive for v in p.new_verbs),
                tuple(sorted(p.allowed_lemmas)),
                tuple(sorted(p.allowed_forms)),
            )
            for p in plans
        ]

    assert shape(first) == shape(second)


def test_seed_words_resolve_to_lexicon_via_english_glosses():
    """English seedWords select concrete lemmas directly, beating frequency fallback."""
    # High-frequency fillers (no English gloss match) vs less-frequent concrete nouns.
    words = [
        _word("nu", rank=1, en="now"),
        _word("er", rank=2, en="there"),
        _word("ja", rank=3, en="yes"),
        _word("straat", rank=50, en="street (a paved road)"),
        _word("naam", rank=60, en="name"),
        _word("huis", rank=70, en="house"),
    ]
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    # Proposer returns nothing, so without seedWords selection would fall back to
    # the most frequent words (nu, er, ja).
    assigner = _StubProposingThemeAssigner({"misc": []}, proposals_by_theme={})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={
            "A1": [
                LessonThemePlan(
                    theme="Home",
                    seed_lemmas=[],
                    english_seed_words=["street", "name", "house"],
                ),
                LessonThemePlan(theme="Filler", seed_lemmas=[]),
            ]
        },
    )

    plans = orc.plan(words, cefr="A1", language="Dutch")

    # Lesson 1 is anchored by the resolved concrete nouns, not the frequent fillers.
    lesson1_lemmas = {w.lemma for w in plans[0].new_words}
    assert {"straat", "naam", "huis"} <= lesson1_lemmas  # seeds are present
    assert "er" not in lesson1_lemmas  # seed_owner reservation protects 'er'-less seeds


def test_verb_hints_resolve_to_verbs_via_english_glosses():
    """A catalog ``verbs`` hint selects the matching verb (most frequent) into the lesson."""
    words = [_word("huis", rank=50, en="house")]
    # Two verbs glossing 'have'; the most frequent (lower rank) must win.
    # The 'have' verb lives in a separate file as a stub without seedWords; before
    # the stub carried its gloss, an English verb hint could never reach it.
    have = Verb(
        id="hebben",
        language="nl",
        lemma="hebben",
        infinitive="hebben",
        cefr="A1",
        frequency=Frequency(rank=42),
        present={"ik": "heb"},
        translations={"en": "to have"},
    )
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubProposingThemeAssigner({"misc": []}, proposals_by_theme={})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={
            "A1": [
                LessonThemePlan(
                    theme="Greetings",
                    seed_lemmas=[],
                    english_seed_words=["house"],
                    english_verbs=["have"],
                )
            ]
        },
    )

    plans = orc.plan(words, cefr="A1", verbs=[have], language="Dutch")

    # The English verb hint resolved to the verb stub via its gloss and was taught.
    verb_infinitives = {v.infinitive for v in plans[0].new_verbs}
    assert "hebben" in verb_infinitives


def test_form_pointer_verbs_are_excluded_and_seed_glosses_recorded():
    """Inflected-form verb entries are dropped; resolved lemmas record their meaning."""
    words = [_word("winkel", rank=50, en="shop")]
    pay = Verb(
        id="betalen",
        language="nl",
        lemma="betalen",
        infinitive="betalen",
        cefr="A1",
        frequency=Frequency(rank=100),
        present={"ik": "betaal"},
        translations={"en": "pay"},
    )
    # Bogus verb entry: an inflected form of 'winkelen', not a real infinitive.
    bogus = Verb(
        id="winkel",
        language="nl",
        lemma="winkel",
        infinitive="winkel",
        cefr="A1",
        frequency=Frequency(rank=60),
        present={"ik": "winkel"},
        translations={"en": "inflection of winkelen:"},
    )
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubProposingThemeAssigner({"misc": []}, proposals_by_theme={})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={
            "A1": [
                LessonThemePlan(
                    theme="Shopping",
                    seed_lemmas=[],
                    english_seed_words=["shop"],
                    english_verbs=["pay"],
                )
            ]
        },
    )

    plans = orc.plan(words, cefr="A1", verbs=[pay, bogus], language="Dutch")

    verb_infinitives = {v.infinitive for v in plans[0].new_verbs}
    assert "betalen" in verb_infinitives
    assert "winkel" not in verb_infinitives  # form-pointer verb dropped
    # The English meaning each lemma resolved from is recorded for the prompt.
    # Note: english_seed_words are now handled by the LLM for context-aware
    # translation, so only english_verbs are recorded in seed_glosses.
    assert "winkel" not in plans[0].seed_glosses  # seed word no longer resolved here
    assert plans[0].seed_glosses.get("betalen") == "pay"


def test_catalog_verbs_field_is_parsed(tmp_path):
    """An English ``verbs`` list in the YAML catalog becomes ``english_verbs``."""
    from course_compiler.generation.orchestrator import _load_predefined_themes

    catalog = tmp_path / "themes.yaml"
    catalog.write_text(
        "A1:\n"
        "  lesson001:\n"
        "    theme: Greetings\n"
        "    seedWords: [hello, name]\n"
        "    verbs: [be, have, greet]\n",
        encoding="utf-8",
    )

    loaded = _load_predefined_themes(catalog)

    plan = loaded["A1"][0]
    assert plan.english_seed_words == ["hello", "name"]
    assert plan.english_verbs == ["be", "have", "greet"]


def test_seed_words_are_reserved_from_earlier_themes():
    """An earlier theme's frequency fallback must not steal a later theme's anchor."""
    words = [
        _word("hond", rank=1, en="dog"),  # high frequency, but anchors the Pets theme
        _word("nu", rank=2, en="now"),
        _word("hier", rank=3, en="here"),
        _word("regen", rank=4, en="rain"),
    ]
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubProposingThemeAssigner({"misc": []}, proposals_by_theme={})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={
            "A1": [
                # First theme's seed doesn't resolve, so it would otherwise pad with
                # the most frequent word (hond) — which belongs to the Pets theme.
                LessonThemePlan(
                    theme="Abstract", seed_lemmas=[], english_seed_words=["sunshine"]
                ),
                LessonThemePlan(
                    theme="Pets", seed_lemmas=[], english_seed_words=["dog"]
                ),
            ]
        },
    )

    plans = orc.plan(words, cefr="A1", language="Dutch")

    lemmas0 = {w.lemma for w in plans[0].new_words}
    lemmas1 = {w.lemma for w in plans[1].new_words}
    assert "hond" not in lemmas0  # not stolen by the earlier theme
    assert "hond" in lemmas1  # reserved for its owning (Pets) theme
    assert lemmas0 | lemmas1 == {"hond", "nu", "hier", "regen"}  # coverage preserved


def test_catalog_outline_and_seed_words_flow_through():
    """An outline reaches the LessonPlan; English seed words reach the proposer."""
    words = [_word("huis", rank=1), _word("straat", rank=2)]
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubProposingThemeAssigner(
        {"misc": ["huis", "straat"]},
        proposals_by_theme={"Home": ["huis", "straat"]},
    )
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={
            "A1": [
                LessonThemePlan(
                    theme="Home",
                    seed_lemmas=[],
                    communicative_goals=["describe your home"],
                    english_seed_words=["house", "street"],
                    outline="A person describes their house on a quiet street.",
                )
            ]
        },
    )

    plans = orc.plan(words, cefr="A1", language="Dutch")

    assert plans[0].outline == "A person describes their house on a quiet street."
    assert assigner.seed_words_calls[0] == ["house", "street"]


def test_catalog_path_front_loads_when_configured():
    """Predefined-theme runs front-load: the first lesson is largest, then tapers."""
    words = [_word(f"w{i:03d}", rank=i) for i in range(1, 101)]  # 100 words
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": [w.lemma for w in words]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        first_lesson_words=40,
        front_load_lessons=3,
        predefined_themes={"A1": [f"T{i}" for i in range(1, 11)]},  # 10 themes
    )

    plans = orc.plan(words, cefr="A1")
    counts = [len(p.new_words) for p in plans]

    assert counts[0] == max(counts)  # first lesson is the largest
    assert counts[0] > counts[1] > counts[2]  # tapers over the front-load window
    assert min(counts) >= 1  # rounding never yields a zero-size lesson
    all_lemmas = [w.lemma for p in plans for w in p.new_words]
    assert len(all_lemmas) == len(set(all_lemmas)) == 100  # nothing lost/duplicated


def test_catalog_path_is_even_split_without_front_load():
    """Regression: without first_lesson_words, the catalog path splits evenly."""
    words = [_word(f"w{i:03d}", rank=i) for i in range(1, 101)]
    generator = LessonGenerator(_StubProvider(), _IdentityLemmatizer())
    assigner = _StubThemeAssigner({"misc": [w.lemma for w in words]})
    orc = LessonOrchestrator(
        generator,
        assigner,
        words_per_lesson=10,
        predefined_themes={"A1": [f"T{i}" for i in range(1, 11)]},
    )

    plans = orc.plan(words, cefr="A1")

    assert [len(p.new_words) for p in plans] == [10] * 10


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
