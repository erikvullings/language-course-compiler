"""Theme assignment: group Word objects into semantic clusters.

The ``ThemeAssigner`` protocol is the extension point; ``LLMThemeAssigner``
implements it using an LLM call (with optional disk caching for reproducibility).
"""

from __future__ import annotations

import math
import hashlib
import json
import re
from dataclasses import dataclass, field
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

_LESSON_PLAN_SYSTEM_PROMPT = (
    "You are a language-course planner. "
    "You will receive CEFR level, vocabulary list, and words-per-lesson (n). "
    "Design themes typically used in beginner language courses, centered on practical daily-life situations. "
    "Prefer themes like: greetings and introductions, family and people, home and rooms, food and drink, shopping and money, "
    "time and dates, school/work, travel and transport, weather and seasons, health and body, hobbies and sports, "
    "city and directions, services and errands. "
    "Each lesson must be semantically coherent and usable as a real beginner lesson unit. "
    "Do not group by spelling, alphabetic order, or arbitrary word similarity. "
    "Avoid mixed buckets that combine unrelated domains. "
    "First determine lesson_count = ceil(vocabulary_size / n). "
    "Then produce exactly lesson_count lesson plans, each with a concise English theme "
    "and seed_lemmas chosen from the provided vocabulary list only. "
    "Each lesson should include between n/2 and n seed lemmas (rounded down for n/2, min 1). "
    "Some seed lemmas may already be known to the learner; that is acceptable. "
    "Use concrete theme names (no placeholders like 'theme-64'). "
    'Respond as JSON only with shape: {"lessons": [{"theme": str, "seed_lemmas": [str]}]}.'
)

_THEME_SEED_SYSTEM_PROMPT = (
    "You are selecting vocabulary for one language lesson. "
    "You will receive CEFR level, lesson theme, communicative goals and target seed count n. "
    "Propose semantically relevant lemmas for this lesson. "
    "Return at least n lemmas when possible. "
    "Respond as JSON only with shape: {\"seed_lemmas\": [str]}."
)

_PROPOSE_VOCAB_SYSTEM_PROMPT = (
    "You design vocabulary for a single beginner language lesson. "
    "You will receive the target language, CEFR level, lesson theme, communicative "
    "goals, a target word count n, and a list of already-taught lemmas to avoid. "
    "Propose the most useful, communicatively central words a learner needs to talk "
    "about this theme and reach these goals — drawn from your own knowledge of the "
    "language, NOT from any provided list. "
    "Give base dictionary forms (lemmas), lower-case, in the target language. "
    "Prefer concrete, high-frequency, everyday words at or below the CEFR level. "
    "If 'anchor_concepts_english' is provided, make sure the target-language "
    "translations of those concrete concepts are among your proposals. "
    "Do not repeat any already-taught lemma. "
    "Return about 5×n candidates so the course can select the best ones. "
    'Respond as JSON only with shape: {"vocabulary": [str]}.'
)


@dataclass(frozen=True)
class LessonThemePlan:
    """LLM-proposed lesson theme with seed lemmas for prompting the writer.

    ``english_seed_words`` and ``outline`` are optional **English** authoring hints
    from the theme catalog (language-agnostic): the seed words anchor vocabulary
    selection toward concrete nouns; the outline is a brief scenario that shapes a
    coherent narrative. Both are realized into the target language by the LLM.
    """

    theme: str
    seed_lemmas: list[str]
    communicative_goals: list[str] = field(default_factory=list)
    english_seed_words: list[str] = field(default_factory=list)
    english_verbs: list[str] = field(default_factory=list)
    outline: str = ""


class ThemeAssigner(Protocol):
    """Assign a list of :class:`~course_compiler.models.Word` objects to themes.

    Returns ``{theme_name: [Word, ...]}``.  Every input word must appear in
    exactly one theme; implementors should collect unassigned words under a
    ``"misc"`` key.

    Only :meth:`assign` is required. An assigner *may* additionally provide
    ``plan_lessons``, ``select_seed_lemmas_for_theme`` or
    ``propose_theme_vocabulary`` to influence catalog-driven planning; the
    orchestrator discovers these optionally (via ``getattr``) and falls back to
    deterministic seed-word + frequency selection when they are absent (see
    :class:`DeterministicThemeAssigner`).
    """

    def assign(self, words: list[Word]) -> dict[str, list[Word]]: ...


class DeterministicThemeAssigner:
    """Plan lessons without any LLM calls — the default assigner.

    Implements only :meth:`assign` (a single deterministic bucket, used when no
    theme catalog is configured). It deliberately omits ``propose_theme_vocabulary``
    and ``select_seed_lemmas_for_theme`` so the orchestrator's catalog path selects
    each lesson's vocabulary purely from the catalog's English seed words (resolved
    via the lexicon's glosses) and frequency ranking — fast, reproducible, offline.

    This replaced LLM-backed theme proposing, which made ~2 model calls per theme
    (minutes of latency *before any lesson was written*) without improving on
    seed-word-anchored selection.
    """

    def assign(self, words: list[Word]) -> dict[str, list[Word]]:
        return {"general": sorted(words, key=lambda w: w.lemma)}


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

    def plan_lessons(
        self,
        words: list[Word],
        *,
        cefr: str,
        words_per_lesson: int,
    ) -> list[LessonThemePlan]:
        """Ask the LLM to plan lesson themes + seed lemmas from vocabulary size.

        Returns an empty list when planning fails so callers can fall back to
        deterministic non-LLM planning.
        """
        if not words or words_per_lesson < 1:
            return []

        lemmas = sorted({w.lemma for w in words})
        lesson_count = math.ceil(len(lemmas) / words_per_lesson)
        min_seed = max(1, words_per_lesson // 2)

        user_payload = {
            "cefr": cefr,
            "vocabulary_size": len(lemmas),
            "words_per_lesson": words_per_lesson,
            "min_seed_lemmas": min_seed,
            "max_seed_lemmas": words_per_lesson,
            "lesson_count": lesson_count,
            "lemmas": lemmas,
        }
        raw_messages = [
            {"role": "system", "content": _LESSON_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        if self._cache is not None:
            cached = self._cache.get(self._model, raw_messages)
            if cached is not None:
                return self._parse_lesson_plan(
                    cached.content,
                    lemmas=lemmas,
                    words_per_lesson=words_per_lesson,
                )

        messages = [
            Message(Role.SYSTEM, _LESSON_PLAN_SYSTEM_PROMPT),
            Message(Role.USER, json.dumps(user_payload, ensure_ascii=False)),
        ]

        try:
            response = self._provider.complete(messages, model=self._model or None)
            parsed = self._parse_lesson_plan(
                response.content,
                lemmas=lemmas,
                words_per_lesson=words_per_lesson,
            )
        except (LLMError, json.JSONDecodeError, TypeError, ValueError):
            return []

        if self._cache is not None:
            self._cache.put(self._model, raw_messages, response)

        return parsed

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
        if target_count < 1:
            return []

        payload = {
            "cefr": cefr,
            "theme": theme,
            "communicative_goals": communicative_goals,
            "target_count": target_count,
            "already_used_lemmas": already_used,
            "candidate_lemmas": candidate_lemmas,
        }
        raw_messages = [
            {"role": "system", "content": _THEME_SEED_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        if self._cache is not None:
            cached = self._cache.get(self._model, raw_messages)
            if cached is not None:
                return self._parse_seed_selection(cached.content)

        messages = [
            Message(Role.SYSTEM, _THEME_SEED_SYSTEM_PROMPT),
            Message(Role.USER, json.dumps(payload, ensure_ascii=False)),
        ]

        try:
            response = self._provider.complete(messages, model=self._model or None)
            selected = self._parse_seed_selection(response.content)
        except (LLMError, json.JSONDecodeError, TypeError, ValueError):
            return []

        if self._cache is not None:
            self._cache.put(self._model, raw_messages, response)
        return selected

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
        """Ask the LLM to *generate* ~5×n theme-relevant lemmas from its own knowledge.

        Unlike :meth:`select_seed_lemmas_for_theme`, no candidate list is supplied —
        the LLM proposes communicatively central words for the theme, which the
        caller then filters against the lexicon. ``seed_words`` are optional English
        anchor concepts (from the theme catalog) that bias the proposal toward
        concrete, scene-grounding vocabulary. Cached for reproducibility; returns
        ``[]`` on provider/parse error so callers can fall back.
        """
        if target_count < 1:
            return []

        payload = {
            "language": language,
            "cefr": cefr,
            "theme": theme,
            "communicative_goals": communicative_goals,
            "target_count": target_count,
            "oversample_count": target_count * 5,
            "already_used_lemmas": already_used,
            "anchor_concepts_english": seed_words or [],
        }
        raw_messages = [
            {"role": "system", "content": _PROPOSE_VOCAB_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        if self._cache is not None:
            cached = self._cache.get(self._model, raw_messages)
            if cached is not None:
                return self._parse_proposed_vocabulary(cached.content)

        messages = [
            Message(Role.SYSTEM, _PROPOSE_VOCAB_SYSTEM_PROMPT),
            Message(Role.USER, json.dumps(payload, ensure_ascii=False)),
        ]

        try:
            response = self._provider.complete(messages, model=self._model or None)
            proposed = self._parse_proposed_vocabulary(response.content)
        except (LLMError, json.JSONDecodeError, TypeError, ValueError):
            return []

        if self._cache is not None:
            self._cache.put(self._model, raw_messages, response)
        return proposed

    def _parse_proposed_vocabulary(self, text: str) -> list[str]:
        raw = json.loads(_strip_fences(text))
        if not isinstance(raw, dict) or not isinstance(raw.get("vocabulary"), list):
            raise ValueError("Invalid proposed-vocabulary response shape.")

        out: list[str] = []
        for item in raw["vocabulary"]:
            if isinstance(item, str):
                lemma = item.strip()
                if lemma and lemma not in out:
                    out.append(lemma)
        return out

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

    def _parse_lesson_plan(
        self,
        text: str,
        *,
        lemmas: list[str],
        words_per_lesson: int,
    ) -> list[LessonThemePlan]:
        """Parse and sanitize lesson plans to deterministic valid output."""
        raw = json.loads(_strip_fences(text))
        if not isinstance(raw, dict) or not isinstance(raw.get("lessons"), list):
            raise ValueError("Invalid lesson plan response shape.")

        valid_lemmas = set(lemmas)
        lesson_count = math.ceil(len(lemmas) / words_per_lesson)
        max_seed = max(1, words_per_lesson)

        lessons_raw = raw["lessons"]
        lessons: list[LessonThemePlan] = []
        used: set[str] = set()

        for index in range(lesson_count):
            entry = lessons_raw[index] if index < len(lessons_raw) else {}
            theme = str(entry.get("theme") or f"theme-{index + 1:02d}")
            seeds_raw = entry.get("seed_lemmas")
            seeds: list[str] = []
            if isinstance(seeds_raw, list):
                for lemma in seeds_raw:
                    if not isinstance(lemma, str):
                        continue
                    if lemma in valid_lemmas and lemma not in seeds:
                        seeds.append(lemma)

            if len(seeds) > max_seed:
                seeds = seeds[:max_seed]

            # Keep themes coherent: only force a seed when the lesson is empty,
            # do not pad to n/2 with potentially unrelated leftovers.
            if not seeds:
                candidate = next(
                    (l for l in lemmas if l not in used and l not in seeds), None
                )
                if candidate is None:
                    candidate = next((l for l in lemmas if l not in seeds), None)
                if candidate is None:
                    continue
                seeds.append(candidate)

            used.update(seeds)
            lessons.append(LessonThemePlan(theme=theme, seed_lemmas=seeds))

        return lessons

    def _parse_seed_selection(self, text: str) -> list[str]:
        raw = json.loads(_strip_fences(text))
        if not isinstance(raw, dict) or not isinstance(raw.get("seed_lemmas"), list):
            raise ValueError("Invalid seed selection response shape.")

        out: list[str] = []
        for item in raw["seed_lemmas"]:
            if isinstance(item, str):
                lemma = item.strip()
                if lemma and lemma not in out:
                    out.append(lemma)
        return out
