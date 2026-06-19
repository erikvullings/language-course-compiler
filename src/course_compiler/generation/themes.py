"""Theme assignment: group Word objects into semantic clusters.

The ``ThemeAssigner`` protocol is the extension point; ``LLMThemeAssigner``
implements it using an LLM call (with optional disk caching for reproducibility).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from course_compiler.generation.cache import LLMCache
from course_compiler.llm.base import LLMError, LLMProvider, Message, Role
from course_compiler.models import Word

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a language-teaching curriculum designer. "
    "You will receive a list of vocabulary lemmas and must group them into "
    "semantic themes suitable for language lessons (e.g. 'home', 'food', 'transport'). "
    "Respond with a single JSON object mapping theme names to arrays of lemmas. "
    "Every input lemma must appear in exactly one theme. "
    "Use concise English theme names. No explanation, only JSON."
)


class ThemeAssigner(Protocol):
    """Assign a list of :class:`~course_compiler.models.Word` objects to themes.

    Returns ``{theme_name: [Word, ...]}``.  Every input word must appear in
    exactly one theme; implementors should collect unassigned words under a
    ``"misc"`` key.
    """

    def assign(self, words: list[Word]) -> dict[str, list[Word]]:
        ...


def _strip_fences(text: str) -> str:
    """Remove optional ```json ... ``` markdown fences from LLM output."""
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text.strip()


def _cache_key(model: str, lemmas: list[str]) -> str:
    payload = json.dumps({"model": model, "lemmas": sorted(lemmas)}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMThemeAssigner:
    """Cluster words into themes using an LLM, with optional disk cache.

    The cache key is derived from the model name and the sorted list of lemmas,
    so the same vocabulary always produces the same theme grouping.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str | None = None,
        *,
        cache: LLMCache | None = None,
    ) -> None:
        self._provider = provider
        self._model = model or ""
        self._cache = cache

    def assign(self, words: list[Word]) -> dict[str, list[Word]]:
        by_lemma = {w.lemma: w for w in words}
        lemmas = sorted(by_lemma)

        # Build a synthetic single-turn message list for cache keying.
        raw_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(lemmas, ensure_ascii=False)},
        ]

        if self._cache is not None:
            cached = self._cache.get(self._model, raw_messages)
            if cached is not None:
                return self._parse(cached.content, by_lemma)

        messages = [
            Message(Role.SYSTEM, _SYSTEM_PROMPT),
            Message(Role.USER, json.dumps(lemmas, ensure_ascii=False)),
        ]

        try:
            response = self._provider.complete(messages, model=self._model or None)
            parsed = self._parse(response.content, by_lemma)
        except (LLMError, json.JSONDecodeError, TypeError, ValueError):
            # Keep lesson generation moving even when theme clustering fails.
            # The fallback is deterministic and includes all input words.
            return {"misc": sorted(by_lemma.values(), key=lambda w: w.lemma)}

        if self._cache is not None:
            self._cache.put(self._model, raw_messages, response)

        return parsed

    def _parse(self, text: str, by_lemma: dict[str, Word]) -> dict[str, list[Word]]:
        raw = json.loads(_strip_fences(text))
        result: dict[str, list[Word]] = {}
        assigned: set[str] = set()
        for theme, lemma_list in raw.items():
            result[theme] = [by_lemma[l] for l in lemma_list if l in by_lemma]
            assigned.update(lemma_list)
        leftover = [w for lemma, w in by_lemma.items() if lemma not in assigned]
        if leftover:
            result.setdefault("misc", []).extend(leftover)
        return result
