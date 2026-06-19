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
