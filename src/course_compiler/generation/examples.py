"""Example sentence generation with vocabulary-constrained validation."""

from __future__ import annotations

from dataclasses import dataclass

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.validator import VocabularyValidator
from course_compiler.llm.base import LLMProvider, Message, Role


class ExampleParseError(RuntimeError):
    """Raised when an example response cannot be parsed."""


@dataclass(frozen=True)
class GeneratedExample:
    id: str
    lesson_id: str
    difficulty: str
    word_ids: list[str]
    sentences: dict[str, str]
    attempts: int


class ExampleGenerator:
    """Generate multilingual example sentences and validate vocabulary use."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        lemmatizer: Lemmatizer,
        function_lemmas: set[str] | None = None,
        max_retries: int = 3,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._max_retries = max_retries

    def _initial_messages(
        self,
        *,
        language_code: str,
        interface_languages: list[str],
        difficulty: str,
        word_ids: list[str],
        lesson_id: str,
        example_id: str,
    ) -> list[Message]:
        all_langs = [language_code, *interface_languages]
        language_list = ", ".join(all_langs)
        user_content = (
            f"Example ID: {example_id}\n"
            f"Lesson ID: {lesson_id}\n"
            f"Difficulty: {difficulty}\n"
            f"Focus word IDs: {', '.join(word_ids)}\n\n"
            f"Return one line per language in this exact format: <code>: <sentence>\n"
            f"Required language codes: {language_list}"
        )
        return [
            Message(
                Role.SYSTEM,
                "You generate concise language-learning example sentences. "
                "Return plain text only with one '<code>: <sentence>' line per required language.",
            ),
            Message(Role.USER, user_content),
        ]

    def _parse(self, response_text: str, required_codes: set[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in response_text.splitlines():
            if ":" not in line:
                continue
            code, sentence = line.split(":", 1)
            code_clean = code.strip()
            sentence_clean = sentence.strip()
            if code_clean and sentence_clean:
                result[code_clean] = sentence_clean

        missing = sorted(required_codes - set(result))
        if missing:
            raise ExampleParseError(f"Example response missing required language lines: {', '.join(missing)}")
        return result

    def generate(
        self,
        *,
        example_id: str,
        lesson_id: str,
        language_code: str,
        interface_languages: list[str],
        allowed_words: set[str],
        difficulty: str,
        word_ids: list[str],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> GeneratedExample:
        required_codes = {language_code, *interface_languages}
        messages = self._initial_messages(
            language_code=language_code,
            interface_languages=interface_languages,
            difficulty=difficulty,
            word_ids=word_ids,
            lesson_id=lesson_id,
            example_id=example_id,
        )

        for attempt in range(1, self._max_retries + 1):
            response = self._provider.complete(messages, model=model, temperature=temperature)
            parsed = self._parse(response.content, required_codes)

            validation = self._validator.validate(parsed[language_code], allowed_words)
            if validation.is_valid:
                return GeneratedExample(
                    id=example_id,
                    lesson_id=lesson_id,
                    difficulty=difficulty,
                    word_ids=list(word_ids),
                    sentences=parsed,
                    attempts=attempt,
                )

            leaked = ", ".join(sorted(validation.violations))
            messages = [
                *messages,
                Message(Role.ASSISTANT, response.content),
                Message(
                    Role.USER,
                    "The target-language sentence contains vocabulary outside the allowed set: "
                    f"{leaked}. Rewrite and keep all required language lines.",
                ),
            ]

        raise RuntimeError(
            f"ExampleGenerator exceeded max_retries={self._max_retries} for example {example_id!r}."
        )
