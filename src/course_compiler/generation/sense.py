"""Batched, cached LLM fallback for same-POS word-sense disambiguation.

POS tagging resolves most tokens; what remains is genuine same-POS ambiguity
(e.g. ``bank`` = financial institution vs riverbank). This module builds **one**
prompt per lesson listing every still-ambiguous token in its sentence with its
candidate glosses, asks the model to pick one each, and returns a
``{token_index: gloss}`` map — the :data:`~course_compiler.generation.annotate.SensePicker`
the annotator calls. Responses go through :class:`~course_compiler.generation.cache.LLMCache`
so the (deterministic) per-lesson prompt is resolved at most once.
"""

from __future__ import annotations

import json
import re

from course_compiler.generation.annotate import SensePicker, SenseQuery
from course_compiler.generation.cache import LLMCache
from course_compiler.llm.base import LLMError, LLMProvider, Message, Role

_SYSTEM = (
    "You disambiguate word senses for a language course. For each item you are given "
    "a target word, the sentence it appears in, and a numbered list of candidate "
    "English meanings. Choose the single best meaning for that occurrence. "
    'Reply ONLY with compact JSON: {"<id>": <option_index>, ...} (zero-based index).'
)


def _build_messages(queries: list[SenseQuery]) -> list[Message]:
    items = [
        {
            "id": q.token_index,
            "word": q.lemma,
            "sentence": q.sentence,
            "options": q.candidates,
        }
        for q in queries
    ]
    user = "Disambiguate each item:\n" + json.dumps(items, ensure_ascii=False, indent=2)
    return [
        Message(role=Role.SYSTEM, content=_SYSTEM),
        Message(role=Role.USER, content=user),
    ]


def _parse(content: str, queries: list[SenseQuery]) -> dict[int, str]:
    """Map the model's ``{id: index}`` answer onto ``{token_index: gloss}``."""
    by_index = {q.token_index: q for q in queries}
    chosen: dict[int, str] = {}
    data = _extract_json_object(content)
    if not isinstance(data, dict):
        return chosen
    for raw_id, raw_choice in data.items():
        try:
            token_index = int(raw_id)
            choice = int(raw_choice)
        except (TypeError, ValueError):
            continue
        query = by_index.get(token_index)
        if query is None or not (0 <= choice < len(query.candidates)):
            continue
        chosen[token_index] = query.candidates[choice]
    return chosen


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(content: str) -> object:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_OBJECT_RE.search(content or "")
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def make_llm_sense_picker(
    provider: LLMProvider,
    *,
    model: str | None = None,
    cache: LLMCache | None = None,
    temperature: float | None = 0.0,
) -> SensePicker:
    """Return a :data:`SensePicker` backed by *provider* (cached, fail-open).

    A provider error or unparseable reply yields an empty map, so the annotator
    keeps the deterministic first-candidate gloss rather than failing the lesson.
    """

    def picker(queries: list[SenseQuery]) -> dict[int, str]:
        if not queries:
            return {}
        messages = _build_messages(queries)
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""

        if cache is not None:
            cached = cache.get(resolved_model, raw_messages)
            if cached is not None:
                return _parse(cached.content, queries)

        try:
            response = provider.complete(messages, model=model, temperature=temperature)
        except LLMError:
            return {}

        if cache is not None:
            cache.put(resolved_model, raw_messages, response)
        return _parse(response.content, queries)

    return picker
