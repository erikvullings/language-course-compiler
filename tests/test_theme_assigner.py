"""Tests for LLMThemeAssigner."""

from __future__ import annotations

import json

from course_compiler.generation.themes import LLMThemeAssigner
from course_compiler.llm.base import LLMError, LLMProvider, LLMResponse, PromptInput
from course_compiler.models import PartOfSpeech, Word


def _word(lemma: str, pos: PartOfSpeech = PartOfSpeech.NOUN, rank: int = 1) -> Word:
    from course_compiler.models import Frequency

    return Word(
        id=lemma,
        language="nl",
        lemma=lemma,
        normalized=lemma,
        part_of_speech=pos,
        frequency=Frequency(rank=rank),
    )


class _StubProvider(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: int = 0

    def complete(
        self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self._response, model=model or "stub", raw={})

    async def acomplete(
        self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature)


def test_assign_returns_themes_from_llm_json():
    words = [_word("huis"), _word("deur"), _word("eten"), _word("brood")]
    payload = json.dumps({"home": ["huis", "deur"], "food": ["eten", "brood"]})
    provider = _StubProvider(payload)
    assigner = LLMThemeAssigner(provider, model="stub")
    themes = assigner.assign(words)
    assert set(themes.keys()) == {"home", "food"}
    assert {w.lemma for w in themes["home"]} == {"huis", "deur"}
    assert {w.lemma for w in themes["food"]} == {"eten", "brood"}


def test_unassigned_words_go_to_misc():
    """Words not mentioned in the LLM response land in a 'misc' theme."""
    words = [_word("huis"), _word("xyz")]
    payload = json.dumps({"home": ["huis"]})
    provider = _StubProvider(payload)
    assigner = LLMThemeAssigner(provider, model="stub")
    themes = assigner.assign(words)
    assert "huis" in {w.lemma for w in themes["home"]}
    misc_lemmas = {w.lemma for w in themes.get("misc", [])}
    assert "xyz" in misc_lemmas


def test_assign_uses_cache(tmp_path):
    """Second call with the same word list hits the cache, not the provider."""
    from course_compiler.generation.cache import LLMCache

    words = [_word("huis"), _word("deur")]
    payload = json.dumps({"home": ["huis", "deur"]})
    provider = _StubProvider(payload)
    cache = LLMCache(tmp_path)
    assigner = LLMThemeAssigner(provider, model="stub", cache=cache)
    assigner.assign(words)
    assigner.assign(words)
    assert provider.calls == 1


def test_llm_json_wrapped_in_markdown_is_parsed():
    """LLMs often wrap JSON in ```json ... ``` — strip fences before parsing."""
    words = [_word("huis")]
    payload = '```json\n{"home": ["huis"]}\n```'
    provider = _StubProvider(payload)
    assigner = LLMThemeAssigner(provider, model="stub")
    themes = assigner.assign(words)
    assert "home" in themes


def test_assign_falls_back_to_misc_on_provider_error():
    class _FailingProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            raise LLMError("timeout")

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            raise LLMError("timeout")

    words = [_word("huis"), _word("deur")]
    assigner = LLMThemeAssigner(_FailingProvider(), model="stub")

    themes = assigner.assign(words)

    assert set(themes.keys()) == {"misc"}
    assert [w.lemma for w in themes["misc"]] == ["deur", "huis"]


def test_plan_lessons_returns_sanitized_lesson_blueprints():
    words = [
        _word("huis"),
        _word("deur"),
        _word("eten"),
        _word("brood"),
        _word("lopen"),
    ]
    payload = json.dumps(
        {
            "lessons": [
                {"theme": "home", "seed_lemmas": ["huis"]},
                {"theme": "food", "seed_lemmas": ["eten", "brood", "x-unknown"]},
            ]
        }
    )
    provider = _StubProvider(payload)
    assigner = LLMThemeAssigner(provider, model="stub")

    plans = assigner.plan_lessons(words, cefr="A1", words_per_lesson=3)

    # ceil(5/3) = 2 lessons
    assert len(plans) == 2
    # Sanitizer keeps plans valid without force-mixing unrelated leftovers.
    assert 1 <= len(plans[0].seed_lemmas) <= 3
    assert 1 <= len(plans[1].seed_lemmas) <= 3
    assert plans[0].theme == "home"
    assert plans[1].theme == "food"
    assert set(plans[0].seed_lemmas).issubset({"huis", "deur", "eten", "brood", "lopen"})
    assert set(plans[1].seed_lemmas).issubset({"huis", "deur", "eten", "brood", "lopen"})
    # Unknown lemma from LLM output must be removed.
    assert "x-unknown" not in plans[0].seed_lemmas
    assert "x-unknown" not in plans[1].seed_lemmas


def test_propose_theme_vocabulary_returns_parsed_lemmas():
    """The LLM generates theme-relevant words from its own knowledge (not a pool)."""
    payload = json.dumps({"vocabulary": ["brood", "appel", "melk", "koffie"]})
    provider = _StubProvider(payload)
    assigner = LLMThemeAssigner(provider, model="stub")

    result = assigner.propose_theme_vocabulary(
        cefr="A1",
        theme="food and drink",
        communicative_goals=["order food in a cafe"],
        target_count=2,
        already_used=[],
    )

    assert result == ["brood", "appel", "melk", "koffie"]


def test_propose_theme_vocabulary_includes_english_seed_words_in_prompt():
    """English seed-word anchors are passed to the proposer to bias selection."""

    class _CapturingProvider(LLMProvider):
        def __init__(self, response: str) -> None:
            self._response = response
            self.last_prompt = ""

        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            from course_compiler.llm.base import to_messages

            self.last_prompt = " ".join(m.content for m in to_messages(prompt))
            return LLMResponse(content=self._response, model=model or "stub", raw={})

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            return self.complete(prompt, model=model)

    provider = _CapturingProvider(json.dumps({"vocabulary": ["huis"]}))
    assigner = LLMThemeAssigner(provider, model="stub")

    assigner.propose_theme_vocabulary(
        cefr="A1",
        theme="home",
        communicative_goals=["describe your home"],
        target_count=3,
        already_used=[],
        seed_words=["house", "room", "street"],
    )

    assert "house" in provider.last_prompt
    assert "room" in provider.last_prompt


def test_propose_theme_vocabulary_returns_empty_on_provider_error():
    class _FailingProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            raise LLMError("timeout")

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            raise LLMError("timeout")

    assigner = LLMThemeAssigner(_FailingProvider(), model="stub")
    assert (
        assigner.propose_theme_vocabulary(
            cefr="A1",
            theme="food",
            communicative_goals=[],
            target_count=5,
            already_used=[],
        )
        == []
    )


def test_propose_theme_vocabulary_uses_cache(tmp_path):
    from course_compiler.generation.cache import LLMCache

    payload = json.dumps({"vocabulary": ["brood", "appel"]})
    provider = _StubProvider(payload)
    cache = LLMCache(tmp_path)
    assigner = LLMThemeAssigner(provider, model="stub", cache=cache)

    for _ in range(2):
        assigner.propose_theme_vocabulary(
            cefr="A1",
            theme="food",
            communicative_goals=["order food"],
            target_count=2,
            already_used=[],
        )

    assert provider.calls == 1


def test_plan_lessons_returns_empty_on_provider_error():
    class _FailingProvider(LLMProvider):
        def complete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            raise LLMError("timeout")

        async def acomplete(
            self, prompt: PromptInput, *, model=None, temperature=None, **kwargs
        ) -> LLMResponse:
            raise LLMError("timeout")

    words = [_word("huis"), _word("deur")]
    assigner = LLMThemeAssigner(_FailingProvider(), model="stub")
    assert assigner.plan_lessons(words, cefr="A1", words_per_lesson=2) == []
