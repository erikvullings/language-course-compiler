"""GrammarWriter: build grammar pages via LLM with example vocabulary validation.

A grammar page explains one topic in the learner's interface language (English),
but its *target-language examples* must use only vocabulary the learner has
already seen. The explanation prose is never vocabulary-checked; only the
``examples`` strings are validated against the allowed set, reusing the same
:class:`~course_compiler.generation.validator.VocabularyValidator` that guards
lesson text. Structure and retry/fail-open/caching behaviour mirror
:class:`~course_compiler.generation.lesson.LessonGenerator` for consistency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.grammar import GrammarTopic
from course_compiler.generation.validator import ValidationResult, VocabularyValidator
from course_compiler.llm.base import LLMError, LLMProvider, Message, Role
from course_compiler.models import Grammar

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

#: Cap on allowed words listed in the prompt — enough for grounded examples
#: without overwhelming the context as the vocabulary grows.
_MAX_PROMPT_WORDS = 250


def _system_prompt(language: str) -> str:
    return (
        "You are a language-teaching grammar writer. "
        "You write a concise grammar mini-lesson about a single topic. "
        "Write the explanation, rules, common mistakes, and exceptions in clear "
        "English (the learner's language). "
        f"Write every example sentence in {language}. "
        f"Crucially, examples may use ONLY the {language} words listed under "
        "'Allowed vocabulary' (plus articles, prepositions, conjunctions, and "
        "common pronouns). Do not introduce any other content words in examples. "
        "Keep examples short and natural. "
        "Return strict JSON only with keys: "
        '"title" (short, 2-6 words), '
        '"description" (a few sentences of English explanation, minimal Markdown), '
        '"rules" (array of short English strings), '
        f'"examples" (array of short {language} sentences), '
        f'"signalWords" (array of {language} cue words that typically signal this '
        "structure, e.g. time adverbs for a tense; empty array if none apply), "
        '"commonMistakes" (array of short English strings), '
        '"exceptions" (array of short English strings). '
        "No markdown fences, no extra keys, no commentary."
    )


def _user_content(
    topic: GrammarTopic, focus: str, language: str, allowed_words: set[str]
) -> str:
    words = sorted(allowed_words)[:_MAX_PROMPT_WORDS]
    return (
        f"Topic: {topic.title}\n"
        f"CEFR level: {topic.cefr}\n"
        f"Target language: {language}\n"
        f"Focus: {focus or '-'}\n"
        f"Allowed vocabulary ({len(words)} words): {', '.join(words) if words else '-'}\n\n"
        "Write the grammar mini-lesson now."
    )


def _extract_payload(raw: str) -> dict[str, object]:
    text = raw.strip()
    candidates = [text]
    fenced = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        candidates.append(fenced.group(1).strip())
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


@dataclass(frozen=True)
class _Parsed:
    title: str
    description: str
    rules: list[str]
    examples: list[str]
    signal_words: list[str]
    common_mistakes: list[str]
    exceptions: list[str]


def _parse(raw: str, topic: GrammarTopic) -> _Parsed:
    payload = _extract_payload(raw)
    title = str(payload.get("title") or "").strip() or topic.title
    return _Parsed(
        title=title,
        description=str(payload.get("description") or "").strip(),
        rules=_str_list(payload.get("rules")),
        examples=_str_list(payload.get("examples")),
        signal_words=_str_list(
            payload.get("signalWords", payload.get("signal_words"))
        ),
        common_mistakes=_str_list(
            payload.get("commonMistakes", payload.get("common_mistakes"))
        ),
        exceptions=_str_list(payload.get("exceptions")),
    )


def _feedback_message(result: ValidationResult, cefr: str) -> str:
    words = ", ".join(sorted(result.violations))
    return (
        "Your previous examples used words outside the allowed vocabulary. "
        "Rewrite the grammar mini-lesson; the examples must not contain these "
        f"words (above {cefr} level or not yet taught): {words}."
    )


class GrammarWriter:
    """Generate a :class:`~course_compiler.models.Grammar` page for one topic.

    The explanation is English prose; the target-language examples are validated
    against ``allowed_words`` and resampled on leakage. On unresolved leakage (or
    a provider error) it fails open to a best-effort page flagged ``fallback``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        lemmatizer: Lemmatizer,
        *,
        function_lemmas: set[str] | None = None,
        cache: LLMCache | None = None,
        max_retries: int = 3,
        extra_tolerance: float | None = None,
        retry_strategy: Literal["natural", "corrective"] = "natural",
        fail_open_on_llm_error: bool = True,
        fail_open_on_validation_error: bool = True,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._cache = cache
        self._max_retries = max_retries
        self._extra_tolerance = extra_tolerance
        self._retry_strategy: Literal["natural", "corrective"] = retry_strategy
        self._fail_open_on_llm_error = fail_open_on_llm_error
        self._fail_open_on_validation_error = fail_open_on_validation_error

    def generate(
        self,
        topic: GrammarTopic,
        *,
        allowed_words: set[str],
        language: str,
        cefr: str | None = None,
        focus: str = "",
        introduced_in_lesson: int | None = None,
        cefr_lookup: dict[str, str] | None = None,
        function_lemmas: set[str] | None = None,
        related_grammar: list[str] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> Grammar:
        """Generate and validate a grammar page for *topic*.

        Args:
            topic: The grammar unit to explain (id/title/cefr/language).
            allowed_words: Content-word lemmas the examples may use. Listed in the
                prompt *and* enforced by the validator.
            language: Human-readable target language name (e.g. ``"Dutch"``).
            cefr: CEFR target for example validation; defaults to ``topic.cefr``.
            focus: English hint from the catalog steering the explanation.
            introduced_in_lesson: Lesson index this topic is pegged to (stored on
                the page for downstream linking).
            cefr_lookup: ``{lemma: cefr_level}`` for CEFR-aware example tolerance.
        """
        target_cefr = cefr or topic.cefr
        messages = [
            Message(Role.SYSTEM, _system_prompt(language)),
            Message(Role.USER, _user_content(topic, focus, language, allowed_words)),
        ]
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""

        best: _Parsed | None = None
        best_result: ValidationResult | None = None

        for attempt in range(1, self._max_retries + 1):
            if attempt == 1 and self._cache is not None:
                cached = self._cache.get(resolved_model, raw_messages)
                if cached is not None:
                    parsed = _parse(cached.content, topic)
                    return self._page(
                        topic, parsed, target_cefr, introduced_in_lesson,
                        related_grammar, fallback=False, violations=frozenset(),
                    )

            try:
                response = self._provider.complete(
                    messages, model=model, temperature=temperature
                )
            except LLMError:
                if not self._fail_open_on_llm_error:
                    raise
                return self._fallback_page(
                    topic, target_cefr, introduced_in_lesson, related_grammar,
                    best, best_result,
                )

            if self._cache is not None and attempt == 1:
                self._cache.put(resolved_model, raw_messages, response)

            parsed = _parse(response.content, topic)
            # Signal words are target-language cue words, so they obey the same
            # vocabulary discipline as examples — validate them together.
            result = self._validator.validate(
                " \n".join([*parsed.examples, *parsed.signal_words]),
                allowed_words,
                extra_function_lemmas=function_lemmas,
                cefr_target=target_cefr,
                cefr_lookup=cefr_lookup,
                extra_tolerance=self._extra_tolerance,
                new_word_count=len(parsed.examples) + len(parsed.signal_words),
            )

            if best_result is None or len(result.violations) < len(best_result.violations):
                best, best_result = parsed, result

            if result.is_valid:
                return self._page(
                    topic, parsed, target_cefr, introduced_in_lesson,
                    related_grammar, fallback=False, violations=frozenset(),
                )

            if self._retry_strategy == "corrective":
                messages = [
                    *messages,
                    Message(Role.ASSISTANT, response.content),
                    Message(Role.USER, _feedback_message(result, target_cefr)),
                ]

        if not self._fail_open_on_validation_error:
            raise RuntimeError(
                f"GrammarWriter exceeded max_retries={self._max_retries} for "
                f"topic {topic.id!r}; example vocabulary could not be resolved."
            )
        return self._fallback_page(
            topic, target_cefr, introduced_in_lesson, related_grammar,
            best, best_result,
        )

    def _page(
        self,
        topic: GrammarTopic,
        parsed: _Parsed,
        cefr: str,
        introduced_in_lesson: int | None,
        related_grammar: list[str] | None,
        *,
        fallback: bool,
        violations: frozenset[str],
    ) -> Grammar:
        return Grammar(
            id=topic.id,
            language=topic.language,
            cefr=cefr,
            title=parsed.title,
            description=parsed.description,
            rules=parsed.rules,
            examples=parsed.examples,
            signal_words=parsed.signal_words,
            common_mistakes=parsed.common_mistakes,
            exceptions=parsed.exceptions,
            related_grammar=list(related_grammar or topic.depends_on),
            introduced_in_lesson=introduced_in_lesson,
            fallback=fallback,
            violations=sorted(violations),
        )

    def _fallback_page(
        self,
        topic: GrammarTopic,
        cefr: str,
        introduced_in_lesson: int | None,
        related_grammar: list[str] | None,
        best: _Parsed | None,
        best_result: ValidationResult | None,
    ) -> Grammar:
        parsed = best or _Parsed(topic.title, "", [], [], [], [], [])
        violations = best_result.violations if best_result is not None else frozenset()
        return self._page(
            topic, parsed, cefr, introduced_in_lesson, related_grammar,
            fallback=True, violations=violations,
        )
