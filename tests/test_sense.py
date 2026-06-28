"""Batched LLM sense picker: parsing, caching, and fail-open behavior."""

from __future__ import annotations

from course_compiler.generation.annotate import SenseQuery
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.sense import make_llm_sense_picker
from course_compiler.llm.base import LLMError, LLMResponse


class FakeProvider:
    def __init__(self, content="", *, error=False):
        self.content = content
        self.error = error
        self.calls = 0

    def complete(self, messages, *, model=None, temperature=None, **kwargs):
        self.calls += 1
        if self.error:
            raise LLMError("boom")
        return LLMResponse(content=self.content, model=model or "test")

    async def acomplete(self, *a, **k):  # pragma: no cover - unused
        raise NotImplementedError


def _queries():
    return [
        SenseQuery(
            token_index=3,
            lemma="bank",
            pos="noun",
            sentence="De bank is dicht.",
            candidates=["riverbank", "financial institution"],
        )
    ]


def test_picker_parses_choice_into_gloss():
    provider = FakeProvider(content='{"3": 1}')
    picker = make_llm_sense_picker(provider)
    assert picker(_queries()) == {3: "financial institution"}


def test_picker_extracts_json_from_prose():
    provider = FakeProvider(content='Sure!\n{"3": 0}\nHope that helps.')
    picker = make_llm_sense_picker(provider)
    assert picker(_queries()) == {3: "riverbank"}


def test_picker_ignores_out_of_range_and_bad_ids():
    provider = FakeProvider(content='{"3": 9, "x": 0}')
    picker = make_llm_sense_picker(provider)
    assert picker(_queries()) == {}


def test_picker_fails_open_on_llm_error():
    provider = FakeProvider(error=True)
    picker = make_llm_sense_picker(provider)
    assert picker(_queries()) == {}


def test_picker_no_queries_makes_no_call():
    provider = FakeProvider(content='{"3": 1}')
    picker = make_llm_sense_picker(provider)
    assert picker([]) == {}
    assert provider.calls == 0


def test_picker_uses_cache_on_second_call(tmp_path):
    provider = FakeProvider(content='{"3": 1}')
    cache = LLMCache(tmp_path)
    picker = make_llm_sense_picker(provider, cache=cache)
    assert picker(_queries()) == {3: "financial institution"}
    assert picker(_queries()) == {3: "financial institution"}
    assert provider.calls == 1  # second call served from cache
