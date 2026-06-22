"""LessonGenerator: build lesson content via LLM with vocabulary validation + retry."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.validator import ValidationResult, VocabularyValidator
from course_compiler.llm.base import LLMError, LLMProvider, Message, Role

log = logging.getLogger(__name__)

#: Words of lesson text generated per new content word introduced.
WORDS_PER_NEW_WORD = 15

#: Words of natural, low-repetition text each available content word can sustain.
#: Caps the requested length so we never demand long prose from a tiny vocabulary
#: (the cold-start problem: lesson 1 has only its own new words to recombine).
WORDS_PER_ALLOWED_WORD = 4

#: Smallest lesson text we will ever request.
MIN_TARGET_WORDS = 30


def _target_length(new_word_count: int, allowed_word_count: int) -> str:
    """Requested lesson length: the smaller of two budgets, floored.

    ``by_new`` gives each new word room to be introduced in context; ``by_vocab``
    caps the text at what the recombinant vocabulary can sustain naturally. Early
    lessons are vocab-limited (short, natural); mature lessons, where the allowed
    set dwarfs the new words, stay new-word-limited (the original behaviour).
    """
    by_new = new_word_count * WORDS_PER_NEW_WORD
    by_vocab = allowed_word_count * WORDS_PER_ALLOWED_WORD
    return f"{max(min(by_new, by_vocab), MIN_TARGET_WORDS)} words"


@dataclass(frozen=True)
class GeneratedLesson:
    lesson_id: str
    content: str
    attempts: int
    tolerated: frozenset[str] = frozenset()
    title: str = ""
    theme: str = ""
    new_words: frozenset[str] = frozenset()
    #: True when ``content`` is the deterministic placeholder produced after the
    #: provider failed or validation could not be satisfied — not real lesson text.
    fallback: bool = False


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


#: Lesson body formats, selected by how much vocabulary the learner can recombine.
FORMAT_EXAMPLES = "examples"
FORMAT_NARRATIVE = "narrative"


def _format_new_words(new_words: list[str], glosses: dict[str, str] | None) -> str:
    """Render the new-word list, annotating each with its English meaning.

    Showing ``lemma (meaning)`` keeps the writer on the intended sense — e.g. the
    verb ``eten (to eat)`` rather than the noun ``eten (food)`` — without leaking
    any other language into the output, which must stay in the target language.
    """
    if not glosses:
        return ", ".join(new_words)
    parts: list[str] = []
    for word in new_words:
        meaning = glosses.get(word)
        parts.append(f"{word} ({meaning})" if meaning else word)
    return ", ".join(parts)


def _user_prompt(
    language: str,
    new_words: list[str],
    target_length: str,
    cefr: str,
    theme: str,
    fmt: str,
    outline: str = "",
    glosses: dict[str, str] | None = None,
    verb_lemmas: list[str] | None = None,
) -> str:
    if fmt == FORMAT_EXAMPLES:
        write_instruction = (
            f"Write several short, simple example sentences or a brief dialogue "
            f"— not more than {target_length} in total. "
            "Keep sentences short and concrete; each new word should appear in at least one sentence."
        )
    else:
        write_instruction = (
            f"Write a short narrative — a story or dialogue — not more than {target_length}, "
            "that reads naturally from start to finish."
        )

    theme_line = f'The theme is "{theme}"'
    if outline:
        theme_line += f": {outline.strip()}"

    word_line = (
        f"Try to use these {language} words, or derivatives, naturally in context "
        f"(the English meaning is given in brackets to fix the intended sense — do "
        f"not write the English in your text): {_format_new_words(new_words, glosses)}."
    )

    parts = [
        f"You are a {language} writer. {write_instruction}",
        f"Keep the vocabulary at the CEFR {cefr} level — do not use advanced words, "
        f"but you may freely use other common words at or below this level so the text reads naturally. "
        f"Conjugate verbs correctly for their subject and tense. "
        f"{_tense_guidance(cefr)}"
        "Articles, prepositions, conjunctions, and common pronouns may be used freely.",
        theme_line + ".",
        word_line,
    ]
    if verb_lemmas:
        parts.append(
            "Build your sentences around these verbs, conjugated as the context "
            f"requires: {_format_new_words(verb_lemmas, glosses)}."
        )
    parts.append(
        f"Only output the title and text in {language}, using Markdown with this exact structure:\n\n"
        "## <SHORT_DESCRIPTIVE_TITLE>\n\n"
        "<NARRATIVE>"
    )
    return "\n\n".join(parts)


def _feedback_message(result: ValidationResult, cefr_target: str) -> str:
    parts: list[str] = [
        "Your previous lesson had vocabulary problems. Revise that version with the "
        "smallest possible change — keep the title, structure, and as much of the "
        "wording and length as you can. Do not rewrite from scratch."
    ]
    if result.violations:
        words = ", ".join(sorted(result.violations))
        parts.append(
            f"Remove or replace only these words, because they are above "
            f"{cefr_target} level or outside the allowed vocabulary: {words}. "
            f"Swap each for a simpler allowed word, or drop just that phrase."
        )
    return " ".join(parts)


def _fallback_content(new_words: list[str], *, lesson_id: str, reason: str) -> str:
    """Return deterministic placeholder lesson content when the provider fails."""
    words = " ".join(new_words).strip()
    if words:
        return f"{words}."
    return f"{lesson_id}: lesson unavailable ({reason})."


def _clean_title(raw: str) -> str:
    """Strip the echoed ``Lesson Title`` placeholder weak models reproduce verbatim.

    ``"Lesson Title: Begroetingen"`` → ``"Begroetingen"``; a bare ``"Lesson Title"``
    → ``""`` (so the caller's default kicks in instead of a literal placeholder).
    """
    cleaned = raw.strip()
    low = cleaned.lower()
    if low.startswith("lesson title"):
        cleaned = cleaned[len("lesson title") :].lstrip(" :-—").strip()
    return cleaned


def _parse_lesson_structure(text: str) -> tuple[str, str]:
    """Parse structured lesson markdown into (title, narrative).

    Expected format:
    ## Lesson Title
    [Narrative text]

    Returns (title, narrative_text). Gracefully handles malformed input.
    """
    lines = text.strip().split("\n")
    title = ""
    narrative_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            title = _clean_title(stripped[2:].strip())
        elif title:
            narrative_lines.append(line)

    narrative = "\n".join(narrative_lines).strip()
    return title or "Untitled Lesson", narrative or text


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
        extra_tolerance: float | None = 0.5,
        narrative_vocab_threshold: int = 60,
        fail_open_on_llm_error: bool = True,
        fail_open_on_validation_error: bool = True,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._cache = cache
        self._max_retries = max_retries
        self._extra_tolerance = extra_tolerance
        self._narrative_vocab_threshold = narrative_vocab_threshold
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
        fmt: str,
        outline: str = "",
        glosses: dict[str, str] | None = None,
        verb_lemmas: list[str] | None = None,
    ) -> list[Message]:
        return [
            Message(
                Role.USER,
                _user_prompt(
                    language,
                    new_words,
                    target_length,
                    cefr,
                    theme,
                    fmt,
                    outline,
                    glosses,
                    verb_lemmas,
                ),
            )
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
        outline: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        function_lemmas: set[str] | None = None,
        cefr_lookup: dict[str, str] | None = None,
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
            outline: Optional brief scenario (English) shaping a coherent narrative;
                when present the lesson is written as narrative rather than examples.
            model: LLM model identifier; provider default used if omitted.
            temperature: Sampling temperature forwarded to the provider.
            function_lemmas: Extra per-call exempt lemmas (e.g. verb surface forms).
            cefr_lookup: ``{lemma: cefr_level}`` mapping from the imported lexicon.
                Required for CEFR-aware tolerance; without it all extras are violations.

        Raises:
            RuntimeError: If violations persist after ``max_retries``.
        """
        target_length = _target_length(len(new_words), len(allowed_words))
        # An outline gives the model a concrete scenario, so even an early lesson
        # can sustain a coherent narrative rather than loose example sentences.
        fmt = (
            FORMAT_NARRATIVE
            if outline or len(allowed_words) >= self._narrative_vocab_threshold
            else FORMAT_EXAMPLES
        )
        messages = self._build_initial_messages(
            lesson_id, language, cefr, theme, target_length, new_words, fmt, outline,
            glosses, verb_lemmas,
        )
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""

        log.debug(
            "[%s] Starting generation — cefr=%s theme=%r fmt=%s target=%s "
            "new_words=%d allowed_words=%d model=%r outline=%s",
            lesson_id, cefr, theme, fmt, target_length,
            len(new_words), len(allowed_words), resolved_model,
            repr(outline[:80] + "…") if len(outline) > 80 else repr(outline),
        )
        for msg in messages:
            log.debug("[%s] %s PROMPT:\n%s", lesson_id, msg.role.value.upper(), msg.content)

        for attempt in range(1, self._max_retries + 1):
            # Cache: only the first attempt (deterministic prompt) is cached.
            if attempt == 1 and self._cache is not None:
                cached = self._cache.get(resolved_model, raw_messages)
                if cached is not None:
                    log.debug("[%s] Cache hit — skipping LLM call", lesson_id)
                    title, narrative = _parse_lesson_structure(cached.content)
                    return GeneratedLesson(
                        lesson_id=lesson_id,
                        content=narrative,
                        attempts=0,
                        title=title,
                        theme=theme,
                        new_words=frozenset(new_words),
                    )

            log.debug("[%s] Attempt %d/%d — calling LLM", lesson_id, attempt, self._max_retries)
            try:
                response = self._provider.complete(
                    messages, model=model, temperature=temperature
                )
            except LLMError:
                if not self._fail_open_on_llm_error:
                    raise
                log.debug("[%s] LLM error on attempt %d — using fallback", lesson_id, attempt)
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=_fallback_content(
                        new_words, lesson_id=lesson_id, reason="llm-timeout"
                    ),
                    attempts=attempt,
                    title="Untitled",
                    theme=theme,
                    new_words=frozenset(new_words),
                    fallback=True,
                )

            log.debug("[%s] RESPONSE (attempt %d):\n%s", lesson_id, attempt, response.content)

            if self._cache is not None and attempt == 1:
                self._cache.put(resolved_model, raw_messages, response)

            # Parse the structured lesson response first, then validate narrative
            title, narrative = _parse_lesson_structure(response.content)

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
                log.debug(
                    "[%s] Validation passed on attempt %d (tolerated=%d)",
                    lesson_id, attempt, len(result.tolerated),
                )
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=narrative,
                    attempts=attempt,
                    tolerated=result.tolerated,
                    title=title,
                    theme=theme,
                    new_words=frozenset(new_words),
                )

            log.debug(
                "[%s] Validation failed on attempt %d — violations: %s",
                lesson_id, attempt, sorted(result.violations),
            )
            feedback = _feedback_message(result, cefr)
            log.debug("[%s] FEEDBACK MESSAGE:\n%s", lesson_id, feedback)

            # Append the bad response and a correction message for the next attempt.
            messages = [
                *messages,
                Message(Role.ASSISTANT, response.content),
                Message(Role.USER, feedback),
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
                fallback=True,
            )

        raise RuntimeError(
            f"LessonGenerator exceeded max_retries={self._max_retries} for lesson {lesson_id!r}. "
            "Vocabulary violations could not be resolved."
        )
