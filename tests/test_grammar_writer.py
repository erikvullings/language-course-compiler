"""Tests for GrammarWriter content generation."""

from __future__ import annotations

import json

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.grammar import GrammarTopic
from course_compiler.generation.grammar_writer import GrammarWriter
from course_compiler.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    PromptInput,
    to_messages,
)
from course_compiler.models import Grammar


class _MapLemmatizer(Lemmatizer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    @property
    def language(self) -> str:
        return "test"

    def lemmatize(self, token: str) -> str | None:
        return self._mapping.get(token.lower())


class _StubProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        self.calls.append(to_messages(prompt))
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
    def complete(self, *a: object, **k: object) -> LLMResponse:
        raise LLMError("timed out")

    async def acomplete(self, *a: object, **k: object) -> LLMResponse:
        raise LLMError("timed out")


def _payload(**kwargs: object) -> str:
    return json.dumps(kwargs)


def _topic() -> GrammarTopic:
    return GrammarTopic(
        id="present-tense",
        language="nl",
        title="Present tense",
        cefr="A1",
        depends_on=[],
    )


def _lemmatizer() -> _MapLemmatizer:
    # Known target-language lemmas (lowercase surface -> lemma).
    return _MapLemmatizer({w: w for w in ["man", "lopen", "huis", "rennen"]})


# ---------------------------------------------------------------------------


def test_generates_grammar_with_clean_examples():
    provider = _StubProvider(
        [
            _payload(
                title="Present tense",
                description="In Dutch the verb usually ends in -t or -en.",
                rules=["Add -t for hij/zij."],
                examples=["de man loopt", "het huis"],
                signalWords=["man"],
                commonMistakes=["Forgetting the -t ending."],
                exceptions=["zijn is irregular"],
            )
        ]
    )
    writer = GrammarWriter(provider, _lemmatizer())

    page = writer.generate(
        _topic(),
        allowed_words={"man", "lopen", "huis"},
        language="Dutch",
        cefr="A1",
        cefr_lookup={"man": "A1", "lopen": "A1", "huis": "A1", "rennen": "B1"},
    )

    assert isinstance(page, Grammar)
    assert page.id == "present-tense"
    assert page.language == "nl"
    assert page.title == "Present tense"
    assert "Dutch" in page.description
    assert page.examples == ["de man loopt", "het huis"]
    assert page.signal_words == ["man"]
    assert page.common_mistakes == ["Forgetting the -t ending."]
    assert page.fallback is False
    assert not page.violations


def test_signal_words_are_vocab_validated():
    # A signal word above level (rennen=B1) must fail validation like an example.
    provider = _StubProvider(
        [_payload(description="x", examples=["de man loopt"], signalWords=["rennen"])]
    )
    writer = GrammarWriter(provider, _lemmatizer(), max_retries=1)

    page = writer.generate(
        _topic(),
        allowed_words={"man", "lopen"},
        language="Dutch",
        cefr="A1",
        cefr_lookup={"man": "A1", "lopen": "A1", "rennen": "B1"},
    )

    assert page.fallback is True
    assert "rennen" in page.violations


def test_english_prose_is_not_vocab_checked():
    # Description/rules contain English words far outside the allowed set; only
    # the target-language examples are validated, so this must pass.
    provider = _StubProvider(
        [
            _payload(
                description="Conjugation depends on the grammatical subject pronoun.",
                rules=["Inflection follows person and number."],
                examples=["de man loopt"],
            )
        ]
    )
    writer = GrammarWriter(provider, _lemmatizer())

    page = writer.generate(
        _topic(),
        allowed_words={"man", "lopen"},
        language="Dutch",
        cefr="A1",
        cefr_lookup={"man": "A1", "lopen": "A1"},
    )

    assert page.fallback is False
    assert not page.violations


def test_retries_then_accepts_clean_draft():
    leaky = _payload(description="x", examples=["de man rennen"])  # rennen=B1 leak
    clean = _payload(description="x", examples=["de man loopt"])
    provider = _StubProvider([leaky, clean])
    writer = GrammarWriter(provider, _lemmatizer(), max_retries=3)

    page = writer.generate(
        _topic(),
        allowed_words={"man", "lopen"},
        language="Dutch",
        cefr="A1",
        cefr_lookup={"man": "A1", "lopen": "A1", "rennen": "B1"},
    )

    assert page.fallback is False
    assert not page.violations
    assert len(provider.calls) == 2


def test_fail_open_when_all_drafts_leak():
    leaky = _payload(description="x", examples=["de man rennen"])
    provider = _StubProvider([leaky, leaky, leaky])
    writer = GrammarWriter(provider, _lemmatizer(), max_retries=3)

    page = writer.generate(
        _topic(),
        allowed_words={"man", "lopen"},
        language="Dutch",
        cefr="A1",
        cefr_lookup={"man": "A1", "lopen": "A1", "rennen": "B1"},
    )

    assert page.fallback is True
    assert "rennen" in page.violations


def test_fail_open_on_llm_error():
    writer = GrammarWriter(_ErrorProvider(), _lemmatizer())

    page = writer.generate(
        _topic(),
        allowed_words={"man"},
        language="Dutch",
        cefr="A1",
    )

    assert page.fallback is True


def test_first_attempt_is_cached(tmp_path):
    cache = LLMCache(tmp_path)
    response = _payload(description="x", examples=["de man loopt"])
    provider = _StubProvider([response])
    writer = GrammarWriter(provider, _lemmatizer(), cache=cache)

    args = dict(
        allowed_words={"man", "lopen"},
        language="Dutch",
        cefr="A1",
        cefr_lookup={"man": "A1", "lopen": "A1"},
    )
    first = writer.generate(_topic(), **args)
    assert len(provider.calls) == 1

    # Second writer with an empty provider must hit the cache (no call needed).
    writer2 = GrammarWriter(_StubProvider([]), _lemmatizer(), cache=cache)
    second = writer2.generate(_topic(), **args)
    assert second.examples == first.examples
