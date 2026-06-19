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
class GeneratedLesson:
    lesson_id: str
    content: str
    attempts: int
    tolerated: frozenset[str] = frozenset()
    title: str = ""
    theme: str = ""
    new_words: frozenset[str] = frozenset()


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
        f"Write a short lesson in {language} with this exact structure:\n\n"
        f"## Lesson Title\n\n"
        f"**New words:** comma-separated list of the new lemmas\n\n"
        f"[Coherent narrative text, approximately {target_length}. "
        f"Introduce each new word naturally in context. "
        f"Keep the vocabulary at the stated CEFR level — do not use advanced words. "
        f"{_tense_guidance(cefr)}"
        f"Articles, prepositions, conjunctions, and common pronouns may be used freely.]\n\n"
        f"Return ONLY the lesson structure above with no other commentary."
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


def _parse_lesson_structure(
    text: str,
) -> tuple[str, list[str], str]:
    """Parse structured lesson markdown into (title, new_words, narrative).

    Expected format:
    ## Lesson Title
    **New words:** word1, word2, word3
    [Narrative text]

    Returns (title, [lemmas], narrative_text). Gracefully handles malformed input.
    """
    lines = text.strip().split("\n")
    title = ""
    new_words_str = ""
    narrative_lines: list[str] = []
    in_narrative = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            title = stripped[2:].strip()
        elif stripped.startswith("**New words:**"):
            new_words_str = stripped[len("**New words:**") :].strip()
        elif stripped:
            if new_words_str or title:
                in_narrative = True
            if in_narrative:
                narrative_lines.append(line)

    new_words = [w.strip() for w in new_words_str.split(",") if w.strip()]
    narrative = "\n".join(narrative_lines).strip()

    return title or "Untitled Lesson", new_words, narrative or text


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
    ) -> list[Message]:
        user_content = (
            f"Lesson ID: {lesson_id}\n"
            f"CEFR level: {cefr}\n"
            f"Theme: {theme}\n"
            f"New words to introduce: {', '.join(new_words)}\n\n"
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
            lesson_id, language, cefr, theme, target_length, new_words
        )
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""

        for attempt in range(1, self._max_retries + 1):
            # Cache: only the first attempt (deterministic prompt) is cached.
            if attempt == 1 and self._cache is not None:
                cached = self._cache.get(resolved_model, raw_messages)
                if cached is not None:
                    title, parsed_new_words, narrative = _parse_lesson_structure(
                        cached.content
                    )
                    return GeneratedLesson(
                        lesson_id=lesson_id,
                        content=narrative,
                        attempts=0,
                        title=title,
                        theme=theme,
                        new_words=frozenset(parsed_new_words),
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
                    title="Untitled",
                    theme=theme,
                    new_words=frozenset(new_words),
                )

            if self._cache is not None and attempt == 1:
                self._cache.put(resolved_model, raw_messages, response)

            # Parse the structured lesson response first, then validate narrative
            title, parsed_new_words, narrative = _parse_lesson_structure(
                response.content
            )

            result: ValidationResult = self._validator.validate(
                narrative,
                allowed_words,
                extra_function_lemmas=function_lemmas,
                cefr_target=cefr,
                cefr_lookup=cefr_lookup,
                extra_tolerance=self._extra_tolerance,
                new_word_count=len(new_words),
            )

            if result.is_valid:
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=narrative,
                    attempts=attempt,
                    tolerated=result.tolerated,
                    title=title,
                    theme=theme,
                    new_words=frozenset(parsed_new_words),
                )

            # Append the bad response and a correction message for the next attempt.
            messages = [
                *messages,
                Message(Role.ASSISTANT, response.content),
                Message(Role.USER, _feedback_message(result, cefr)),
            ]

        if self._fail_open_on_validation_error:
            return GeneratedLesson(
                lesson_id=lesson_id,
                content=_fallback_content(
                    new_words, lesson_id=lesson_id, reason="validation-retries"
                ),
                attempts=self._max_retries,
                title="Untitled",
                theme=theme,
                new_words=frozenset(new_words),
            )

        raise RuntimeError(
            f"LessonGenerator exceeded max_retries={self._max_retries} for lesson {lesson_id!r}. "
            "Vocabulary violations could not be resolved."
        )
