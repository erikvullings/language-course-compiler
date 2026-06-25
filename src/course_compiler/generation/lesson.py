"""LessonGenerator: build lesson content via LLM with vocabulary validation + retry."""

from __future__ import annotations

from dataclasses import dataclass

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.validator import ValidationResult, VocabularyValidator
from course_compiler.llm.base import LLMError, LLMProvider, Message, Role

#: Words of lesson text generated per new content word introduced.
WORDS_PER_NEW_WORD = 15


def _target_length(new_word_count: int) -> str:
    return f"{max(new_word_count * WORDS_PER_NEW_WORD, 30)} words"


@dataclass(frozen=True)
class AttemptDiagnostics:
    attempt: int
    violations: frozenset[str]
    tolerated: frozenset[str]


@dataclass(frozen=True)
class GeneratedLesson:
    lesson_id: str
    content: str
    attempts: int
    tolerated: frozenset[str] = frozenset()
    fallback: bool = False
    violations: frozenset[str] = frozenset()
    best_attempt: int | None = None
    diagnostics: tuple[AttemptDiagnostics, ...] = ()


_LOW_CEFR: frozenset[str] = frozenset({"A1", "A2"})


def _tense_guidance(cefr: str) -> str:
    if cefr in _LOW_CEFR:
        return (
            "Prefer present tense. "
            "Simple past of very frequent forms "
            "(for example equivalents of 'was/were') is allowed when natural. "
            "Avoid complex tense combinations. "
        )
    return ""


def _system_prompt(language: str, target_length: str, cefr: str = "") -> str:
    return (
        f"You are a language-learning content writer. "
        f"Write a short lesson in {language}, approximately {target_length}. "
        f"Introduce each new word naturally in context. "
        f"Keep the vocabulary at the stated CEFR level — do not use advanced words. "
        f"{_tense_guidance(cefr)}"
        f"Articles, prepositions, conjunctions, and common pronouns may be used freely. "
        f"Respond with lesson text only — no headings, no word lists, no meta-commentary."
    )


def _feedback_message(result: ValidationResult, cefr_target: str) -> str:
    parts: list[str] = [
        "Your previous response contained vocabulary problems. Please rewrite the lesson."
    ]
    if result.violations:
        words = ", ".join(sorted(result.violations))
        parts.append(
            f"These words must not appear — they are above {cefr_target} level "
            f"or outside the allowed vocabulary: {words}."
        )
    return " ".join(parts)


def _fallback_content(new_words: list[str], *, lesson_id: str, reason: str) -> str:
    """Return deterministic placeholder lesson content when the provider fails."""
    words = " ".join(new_words).strip()
    if words:
        return f"{words}."
    return f"{lesson_id}: lesson unavailable ({reason})."


class LessonGenerator:
    """Generate lesson text via an :class:`~course_compiler.llm.base.LLMProvider`.

    Validates content words against the allowed vocabulary and retries on leakage,
    using multi-turn feedback so the LLM knows exactly what went wrong.

    Extra words at or below the target CEFR level are tolerated up to
    ``extra_tolerance`` × ``len(new_words)`` (default 50 %).  Words above the
    target CEFR are always rejected.

    Optionally caches LLM responses (first-attempt only) for reproducibility.
    """

    def __init__(
        self,
        provider: LLMProvider,
        lemmatizer: Lemmatizer,
        *,
        function_lemmas: set[str] | None = None,
        cache: LLMCache | None = None,
        max_retries: int = 5,
        extra_tolerance: float = 0.5,
        fail_open_on_llm_error: bool = True,
        fail_open_on_validation_error: bool = True,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._cache = cache
        self._max_retries = max_retries
        self._extra_tolerance = extra_tolerance
        self._fail_open_on_llm_error = fail_open_on_llm_error
        self._fail_open_on_validation_error = fail_open_on_validation_error

    def _build_initial_messages(
        self,
        lesson_id: str,
        language: str,
        cefr: str,
        theme: str,
        target_length: str,
        new_words: list[str],
        *,
        outline: str = "",
        communicative_goals: list[str] | None = None,
        glosses: dict[str, str] | None = None,
        verb_lemmas: list[str] | None = None,
    ) -> list[Message]:
        rendered_words = [
            (
                f"{lemma} ({glosses[lemma]})"
                if glosses is not None and lemma in glosses and glosses[lemma]
                else lemma
            )
            for lemma in new_words
        ]
        goals = communicative_goals or []
        verbs = verb_lemmas or []
        user_content = (
            f"Lesson ID: {lesson_id}\n"
            f"CEFR level: {cefr}\n"
            f"Theme: {theme}\n"
            f"Communicative goals: {', '.join(goals) if goals else '-'}\n"
            f"Focus verbs: {', '.join(verbs) if verbs else '-'}\n"
            f"Story outline: {outline or '-'}\n"
            f"New words to introduce: {', '.join(rendered_words)}\n\n"
            "Write the lesson now."
        )
        return [
            Message(Role.SYSTEM, _system_prompt(language, target_length, cefr)),
            Message(Role.USER, user_content),
        ]

    def generate(
        self,
        lesson_id: str,
        new_words: list[str],
        allowed_words: set[str],
        *,
        language: str,
        cefr: str = "A1",
        theme: str = "general",
        model: str | None = None,
        temperature: float = 0.7,
        function_lemmas: set[str] | None = None,
        cefr_lookup: dict[str, str] | None = None,
        outline: str = "",
        communicative_goals: list[str] | None = None,
        glosses: dict[str, str] | None = None,
        verb_lemmas: list[str] | None = None,
    ) -> GeneratedLesson:
        """Generate and validate lesson content, retrying with feedback on violation.

        Args:
            lesson_id: Unique identifier for this lesson (e.g. ``"lesson001"``).
            new_words: Content-word lemmas being introduced for the first time.
            allowed_words: Full set of content-word lemmas the validator accepts.
                Not sent to the LLM — enforced by the validator only.
            language: Human-readable target language name (e.g. ``"Dutch"``).
            cefr: CEFR level (e.g. ``"A1"``). Included in the prompt and used to
                classify extra words as tolerated vs. violations.
            theme: Semantic theme (e.g. ``"home"``). Included in the prompt.
            model: LLM model identifier; provider default used if omitted.
            temperature: Sampling temperature forwarded to the provider.
            function_lemmas: Extra per-call exempt lemmas (e.g. verb surface forms).
            cefr_lookup: ``{lemma: cefr_level}`` mapping from the imported lexicon.
                Required for CEFR-aware tolerance; without it all extras are violations.

        Raises:
            RuntimeError: If violations persist after ``max_retries``.
        """
        target_length = _target_length(len(new_words))
        messages = self._build_initial_messages(
            lesson_id,
            language,
            cefr,
            theme,
            target_length,
            new_words,
            outline=outline,
            communicative_goals=communicative_goals,
            glosses=glosses,
            verb_lemmas=verb_lemmas,
        )
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""
        diagnostics: list[AttemptDiagnostics] = []
        best_content: str | None = None
        best_result: ValidationResult | None = None
        best_attempt: int | None = None

        for attempt in range(1, self._max_retries + 1):
            # Cache: only the first attempt (deterministic prompt) is cached.
            if attempt == 1 and self._cache is not None:
                cached = self._cache.get(resolved_model, raw_messages)
                if cached is not None:
                    return GeneratedLesson(
                        lesson_id=lesson_id,
                        content=cached.content,
                        attempts=0,
                        best_attempt=0,
                    )

            try:
                response = self._provider.complete(
                    messages, model=model, temperature=temperature
                )
            except LLMError:
                if not self._fail_open_on_llm_error:
                    raise
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=_fallback_content(
                        new_words, lesson_id=lesson_id, reason="llm-timeout"
                    ),
                    attempts=attempt,
                    fallback=True,
                    best_attempt=attempt,
                    diagnostics=tuple(diagnostics),
                )

            if self._cache is not None and attempt == 1:
                self._cache.put(resolved_model, raw_messages, response)

            result: ValidationResult = self._validator.validate(
                response.content,
                allowed_words,
                extra_function_lemmas=function_lemmas,
                cefr_target=cefr,
                cefr_lookup=cefr_lookup,
                extra_tolerance=self._extra_tolerance,
                new_word_count=len(new_words),
            )
            diagnostics.append(
                AttemptDiagnostics(
                    attempt=attempt,
                    violations=result.violations,
                    tolerated=result.tolerated,
                )
            )

            if (
                best_result is None
                or len(result.violations) < len(best_result.violations)
                or (
                    len(result.violations) == len(best_result.violations)
                    and response.content.strip()
                    and best_content is not None
                    and len(response.content) > len(best_content)
                )
            ):
                best_content = response.content
                best_result = result
                best_attempt = attempt

            if result.is_valid:
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=response.content,
                    attempts=attempt,
                    tolerated=result.tolerated,
                    best_attempt=attempt,
                    diagnostics=tuple(diagnostics),
                )

            # Append the bad response and a correction message for the next attempt.
            messages = [
                *messages,
                Message(Role.ASSISTANT, response.content),
                Message(Role.USER, _feedback_message(result, cefr)),
            ]

        if self._fail_open_on_validation_error:
            if (
                best_content is not None
                and best_result is not None
                and best_attempt is not None
            ):
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=best_content,
                    attempts=self._max_retries,
                    tolerated=best_result.tolerated,
                    fallback=True,
                    violations=best_result.violations,
                    best_attempt=best_attempt,
                    diagnostics=tuple(diagnostics),
                )

            return GeneratedLesson(
                lesson_id=lesson_id,
                content=_fallback_content(
                    new_words, lesson_id=lesson_id, reason="validation-retries"
                ),
                attempts=self._max_retries,
                fallback=True,
                best_attempt=self._max_retries,
                diagnostics=tuple(diagnostics),
            )

        raise RuntimeError(
            f"LessonGenerator exceeded max_retries={self._max_retries} for lesson {lesson_id!r}. "
            "Vocabulary violations could not be resolved."
        )
