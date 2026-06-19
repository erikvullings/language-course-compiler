"""Tests for example sentence generation with vocabulary constraints."""

from __future__ import annotations

import pytest

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.examples import ExampleGenerator, ExampleParseError
from course_compiler.llm.base import LLMProvider, LLMResponse, Message, PromptInput


class _MapLemmatizer(Lemmatizer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    @property
    def language(self) -> str:
        return "nl"

    def lemmatize(self, token: str) -> str | None:
        return self._mapping.get(token.lower())


class _StubProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self, prompt: PromptInput, *, model: str | None = None, temperature: float | None = None, **kwargs: object
    ) -> LLMResponse:
        from course_compiler.llm.base import to_messages

        self.calls.append(to_messages(prompt))
        content = self._responses.pop(0)
        return LLMResponse(content=content, model=model or "stub", raw={})

    async def acomplete(
        self, prompt: PromptInput, *, model: str | None = None, temperature: float | None = None, **kwargs: object
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature, **kwargs)


def test_generate_accepts_example_when_vocab_is_allowed():
    provider = _StubProvider(["nl: huis groot\nen: big house"])
    lemmatizer = _MapLemmatizer({"huis": "huis", "groot": "groot"})
    generator = ExampleGenerator(provider=provider, lemmatizer=lemmatizer)

    example = generator.generate(
        example_id="ex001",
        lesson_id="lesson001",
        language_code="nl",
        interface_languages=["en"],
        allowed_words={"huis", "groot"},
        difficulty="beginner",
        word_ids=["huis"],
    )

    assert example.sentences["nl"] == "huis groot"
    assert example.sentences["en"] == "big house"
    assert example.attempts == 1


def test_generate_retries_when_sentence_has_vocab_leakage():
    provider = _StubProvider([
        "nl: huis appartement\nen: house apartment",
        "nl: huis groot\nen: big house",
    ])
    lemmatizer = _MapLemmatizer({"huis": "huis", "appartement": "appartement", "groot": "groot"})
    generator = ExampleGenerator(provider=provider, lemmatizer=lemmatizer, max_retries=2)

    example = generator.generate(
        example_id="ex002",
        lesson_id="lesson001",
        language_code="nl",
        interface_languages=["en"],
        allowed_words={"huis", "groot"},
        difficulty="beginner",
        word_ids=["huis"],
    )

    assert example.sentences["nl"] == "huis groot"
    assert example.attempts == 2
    second_call_text = " ".join(msg.content for msg in provider.calls[1])
    assert "appartement" in second_call_text


def test_generate_raises_when_required_language_line_missing():
    provider = _StubProvider(["en: big house"])
    lemmatizer = _MapLemmatizer({"huis": "huis", "groot": "groot"})
    generator = ExampleGenerator(provider=provider, lemmatizer=lemmatizer)

    with pytest.raises(ExampleParseError, match="missing required language"):
        generator.generate(
            example_id="ex003",
            lesson_id="lesson001",
            language_code="nl",
            interface_languages=["en"],
            allowed_words={"huis", "groot"},
            difficulty="beginner",
            word_ids=["huis"],
        )
