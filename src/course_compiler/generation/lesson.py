"""LessonGenerator: build lesson content via LLM with vocabulary validation + retry."""

from __future__ import annotations

from dataclasses import dataclass

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.validator import VocabularyValidator
from course_compiler.llm.base import LLMProvider, Message, Role


@dataclass(frozen=True)
class GeneratedLesson:
    lesson_id: str
    content: str
    attempts: int


def _system_prompt(language: str, target_length: str) -> str:
    return (
        f"You are a language-learning content writer. "
        f"Write a lesson in {language}, approximately {target_length}. "
        f"Use ONLY the content words (nouns, verbs, adjectives, adverbs) listed by the "
        f"user — articles, prepositions, conjunctions, and common pronouns may be used freely. "
        f"Respond with lesson text only — no explanations, no meta-commentary."
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
        target_length: str,
        new_words: list[str],
        allowed_words: set[str],
    ) -> list[Message]:
        word_list = ", ".join(sorted(allowed_words))
        user_content = (
            f"Lesson ID: {lesson_id}\n"
            f"New content words introduced in this lesson: {', '.join(new_words)}\n"
            f"Full allowed content vocabulary: {word_list}\n\n"
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
        model: str | None = None,
        temperature: float = 0.7,
        target_length: str = "180 words",
        function_lemmas: set[str] | None = None,
    ) -> GeneratedLesson:
        """Generate and validate lesson content, retrying on vocabulary leakage.

        Args:
            lesson_id: Unique identifier for this lesson (e.g. ``"lesson001"``).
            new_words: Content-word lemmas being introduced for the first time.
            allowed_words: Full set of content-word lemmas the LLM may use
                (all prior lessons + current lesson words).
            language: Human-readable target language name (e.g. ``"Dutch"``).
                Passed directly into the system prompt.
            model: LLM model identifier; provider default used if omitted.
            temperature: Sampling temperature forwarded to the provider.
            target_length: Length hint included in the system prompt
                (e.g. ``"180 words"``).  Override to produce longer or shorter lessons.

        Raises:
            RuntimeError: If vocabulary leakage persists after ``max_retries``.
        """
        messages = self._build_messages(lesson_id, language, target_length, new_words, allowed_words)
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
