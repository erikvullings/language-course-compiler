"""LessonGenerator: build lesson content via LLM with vocabulary validation + retry."""

from __future__ import annotations

from dataclasses import dataclass

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.validator import VocabularyValidator
from course_compiler.llm.base import LLMProvider, Message, Role

#: Words of lesson text generated per new content word introduced.
WORDS_PER_NEW_WORD = 15


def _target_length(new_word_count: int) -> str:
    return f"{max(new_word_count * WORDS_PER_NEW_WORD, 30)} words"


@dataclass(frozen=True)
class GeneratedLesson:
    lesson_id: str
    content: str
    attempts: int


def _system_prompt(language: str, target_length: str) -> str:
    return (
        f"You are a language-learning content writer. "
        f"Write a short lesson in {language}, approximately {target_length}. "
        f"Introduce each new word naturally in context. "
        f"Keep the vocabulary at the stated CEFR level — do not use advanced words. "
        f"Articles, prepositions, conjunctions, and common pronouns may be used freely. "
        f"Respond with lesson text only — no headings, no word lists, no meta-commentary."
    )


class LessonGenerator:
    """Generate lesson text via an :class:`~course_compiler.llm.base.LLMProvider`.

    Validates content words against the allowed vocabulary and retries on leakage.
    Function words (articles, prepositions, etc.) are exempt from validation;
    supply them via ``function_lemmas`` (derive from the imported lexicon by POS).

    Optionally caches LLM responses for reproducibility.
    """

    def __init__(
        self,
        provider: LLMProvider,
        lemmatizer: Lemmatizer,
        *,
        function_lemmas: set[str] | None = None,
        cache: LLMCache | None = None,
        max_retries: int = 5,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._cache = cache
        self._max_retries = max_retries

    def _build_messages(
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
            Message(Role.SYSTEM, _system_prompt(language, target_length)),
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
    ) -> GeneratedLesson:
        """Generate and validate lesson content, retrying on vocabulary leakage.

        Args:
            lesson_id: Unique identifier for this lesson (e.g. ``"lesson001"``).
            new_words: Content-word lemmas being introduced for the first time.
            allowed_words: Full set of content-word lemmas the validator accepts
                (all prior lessons + current lesson words). Not sent to the LLM.
            language: Human-readable target language name (e.g. ``"Dutch"``).
            cefr: CEFR level string (e.g. ``"A1"``). Included in the prompt so
                the LLM self-regulates vocabulary complexity.
            theme: Semantic theme for this lesson (e.g. ``"home"``). Helps the
                LLM write coherent, contextually consistent text.
            model: LLM model identifier; provider default used if omitted.
            temperature: Sampling temperature forwarded to the provider.
            function_lemmas: Extra lemmas exempt from validation for this call
                (typically verb surface forms derived by the orchestrator).

        Raises:
            RuntimeError: If vocabulary leakage persists after ``max_retries``.
        """
        target_length = _target_length(len(new_words))
        messages = self._build_messages(lesson_id, language, cefr, theme, target_length, new_words)
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""

        for attempt in range(1, self._max_retries + 1):
            if attempt == 1 and self._cache is not None:
                cached = self._cache.get(resolved_model, raw_messages)
                if cached is not None:
                    return GeneratedLesson(lesson_id=lesson_id, content=cached.content, attempts=0)

            response = self._provider.complete(messages, model=model, temperature=temperature)

            if self._cache is not None and attempt == 1:
                self._cache.put(resolved_model, raw_messages, response)

            unknown = self._validator.validate(response.content, allowed_words, extra_function_lemmas=function_lemmas)
            if not unknown:
                return GeneratedLesson(lesson_id=lesson_id, content=response.content, attempts=attempt)

        raise RuntimeError(
            f"LessonGenerator exceeded max_retries={self._max_retries} for lesson {lesson_id!r}. "
            "Vocabulary leakage could not be eliminated."
        )
