"""LessonGenerator: build lesson content via LLM with vocabulary validation + retry."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Literal

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
    title: str | None = None
    tolerated: frozenset[str] = frozenset()
    fallback: bool = False
    violations: frozenset[str] = frozenset()
    best_attempt: int | None = None
    diagnostics: tuple[AttemptDiagnostics, ...] = ()


_LOW_CEFR: frozenset[str] = frozenset({"A1", "A2"})
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TITLE_TEXT_RE = re.compile(
    r"(?is)^\s*TITLE:\s*(.*?)\s*^\s*TEXT:\s*(.*)$", re.MULTILINE
)


def _tense_guidance(cefr: str) -> str:
    if cefr in _LOW_CEFR:
        return (
            "Prefer present tense. "
            "Simple past of very frequent forms "
            "(for example equivalents of 'was/were') is allowed when natural. "
            "Avoid complex tense combinations. "
        )
    return ""


def _user_instructions(language: str, target_length: str, cefr: str = "") -> str:
    return (
        "You are a language course lesson writer. "
        f"Write a short story in {language} with less than {target_length} at CEFR {cefr}. "
        "Introduce each new word naturally in context. "
        f"Keep the vocabulary at the stated CEFR level — do not use advanced words. "
        f"{_tense_guidance(cefr)}"
        "Before returning, verify spelling and grammar agreement "
        "(including subject-verb agreement) "
        "and correct any errors. "
        f"Articles, prepositions, conjunctions, and common pronouns may be used freely. "
        "Use plain paragraphs with minimal Markdown. "
        "Do not italicize or bold full sentences/lines. "
        "If you add emphasis, keep it sparse (at most a few words, not whole dialogue). "
        "Before returning, apply this Dutch grammar checklist: "
        "(1) Every verb agrees with its subject (ik ben, jij bent, hij is, etc.). "
        "(2) Every sentence has a complete structure (subject-verb-object where applicable). "
        "(3) No fragments or incomplete thoughts. "
        "(4) Correct prepositions and word order. "
        "Fix all errors found. "
        "Return the lesson as a JSON object with exactly these two fields: "
        '{"title": "<title>", "text": "<markdown_text>"}. '
        "The title MUST be 2-6 words exactly, noun-phrase style, "
        "not a sentence fragment, and not a question. "
        "Only output the JSON object, no markdown fences or commentary."
    )


def _extract_json_payload(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if not text:
        return None
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
    return None


def _extract_labeled_payload(raw: str) -> tuple[str | None, str] | None:
    match = _TITLE_TEXT_RE.search(raw.strip())
    if match is None:
        return None
    title = match.group(1).strip()
    text = match.group(2).strip()
    if not text:
        return None
    return (title or None), text


def _shorten_title(title: str, fallback_text: str) -> str:
    base = title.strip()
    if not base:
        for line in fallback_text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                base = stripped
                break

    # Remove trailing punctuation
    while base and base[-1] in ".!?,;:":
        base = base[:-1].strip()

    words = [w for w in base.split() if w]

    # If title is already suspiciously long (looks like a full sentence or paragraph),
    # take only the first few words as a conservative estimate.
    if len(words) > 8:
        base = " ".join(words[:4])  # Reduce to first 4 words for long inputs
    elif len(words) > 6:
        base = " ".join(words[:6])

    return base.strip()


def _extract_title_and_text(raw: str) -> tuple[str | None, str]:
    labeled = _extract_labeled_payload(raw)
    if labeled is not None:
        title, text = labeled
        return _shorten_title(title or "", text) or None, text

    payload = _extract_json_payload(raw)
    if payload is None:
        return None, raw

    text_value = payload.get("text")
    text = text_value.strip() if isinstance(text_value, str) else ""
    if not text:
        return None, raw

    title_value = payload.get("title")
    title = title_value.strip() if isinstance(title_value, str) else ""
    return _shorten_title(title, text) or None, text


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

    Validates content words against the allowed vocabulary and samples multiple
    drafts, selecting the one with the fewest violations.

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
        retry_strategy: Literal["natural", "corrective"] = "natural",
        fail_open_on_llm_error: bool = True,
        fail_open_on_validation_error: bool = True,
        verbose: bool = False,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._cache = cache
        self._max_retries = max_retries
        self._extra_tolerance = extra_tolerance
        self._retry_strategy: Literal["natural", "corrective"] = retry_strategy
        self._fail_open_on_llm_error = fail_open_on_llm_error
        self._fail_open_on_validation_error = fail_open_on_validation_error
        self._verbose = verbose

    def _log_messages(self, attempt: int, messages: list[Message]) -> None:
        if not self._verbose:
            return
        print(
            f"[llm][lesson][attempt {attempt}] prompt follows",
            file=sys.stderr,
        )
        for index, message in enumerate(messages, start=1):
            print(
                f"--- message {index} ({message.role.value}) ---",
                file=sys.stderr,
            )
            print(message.content, file=sys.stderr)

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
        english_seed_words: list[str] | None = None,
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
        seeds = english_seed_words or []
        seed_instruction = ""
        if seeds:
            seed_instruction = (
                f"\nKey scene-setting words (in English) to weave naturally into the story "
                f"(translate them to {language} in context): {', '.join(seeds)}."
            )
        user_content = (
            f"{_user_instructions(language, target_length, cefr)}\n\n"
            f"Lesson ID: {lesson_id}\n"
            f"CEFR level: {cefr}\n"
            f"Theme: {theme}\n"
            f"Communicative goals: {', '.join(goals) if goals else '-'}\n"
            f"Focus verbs: {', '.join(verbs) if verbs else '-'}\n"
            f"Story outline: {outline or '-'}\n"
            f"New words to introduce: {', '.join(rendered_words)}"
            f"{seed_instruction}\n\n"
            "Write the lesson now."
        )
        return [Message(Role.USER, user_content)]

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
        english_seed_words: list[str] | None = None,
    ) -> GeneratedLesson:
        """Generate and validate lesson content, sampling multiple drafts.

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
            english_seed_words=english_seed_words,
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
                    self._log_messages(attempt, messages)
                    return GeneratedLesson(
                        lesson_id=lesson_id,
                        content=cached.content,
                        attempts=1,
                        best_attempt=1,
                    )

            try:
                self._log_messages(attempt, messages)
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

            parsed_title, parsed_text = _extract_title_and_text(response.content)

            result: ValidationResult = self._validator.validate(
                parsed_text,
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
                    and parsed_text.strip()
                    and best_content is not None
                    and len(parsed_text) > len(best_content)
                )
            ):
                best_content = parsed_text
                best_result = result
                best_attempt = attempt

            if result.is_valid:
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=parsed_text,
                    title=parsed_title,
                    attempts=attempt,
                    tolerated=result.tolerated,
                    best_attempt=attempt,
                    diagnostics=tuple(diagnostics),
                )

            if self._retry_strategy == "corrective":
                # Multi-turn corrective refinement: ask the model to revise the
                # previous draft by explicitly removing current violations.
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
