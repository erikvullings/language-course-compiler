"""Exercise specification generation.

Exercises reference lesson/word/grammar ids only; they do not duplicate lexical
or lesson content payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ExerciseType(StrEnum):
    FILL_IN_THE_BLANK = "fill-in-the-blank"
    TYPING = "typing"
    LISTENING = "listening"
    WORD_ORDERING = "word-ordering"
    TRANSLATION = "translation"
    REVERSE_TRANSLATION = "reverse-translation"
    MULTIPLE_CHOICE = "multiple-choice"
    PRONUNCIATION = "pronunciation"
    MATCHING = "matching"
    FLASHCARDS = "flashcards"
    CONJUGATION = "conjugation"
    GRAMMAR_QUIZZES = "grammar-quizzes"
    DICTATION = "dictation"
    READING_COMPREHENSION = "reading-comprehension"


SUPPORTED_EXERCISE_TYPES: tuple[ExerciseType, ...] = (
    ExerciseType.FILL_IN_THE_BLANK,
    ExerciseType.TYPING,
    ExerciseType.LISTENING,
    ExerciseType.WORD_ORDERING,
    ExerciseType.TRANSLATION,
    ExerciseType.REVERSE_TRANSLATION,
    ExerciseType.MULTIPLE_CHOICE,
    ExerciseType.PRONUNCIATION,
    ExerciseType.MATCHING,
    ExerciseType.FLASHCARDS,
    ExerciseType.CONJUGATION,
    ExerciseType.GRAMMAR_QUIZZES,
    ExerciseType.DICTATION,
    ExerciseType.READING_COMPREHENSION,
)


@dataclass(frozen=True)
class ExerciseSpec:
    id: str
    type: ExerciseType
    lesson_id: str
    word_ids: list[str]
    grammar_ids: list[str]
    payload: dict[str, object] = field(default_factory=dict)


class ExerciseGenerator:
    """Create deterministic exercise specs from ids only."""

    def generate_for_lesson(
        self,
        *,
        lesson_id: str,
        introduced_word_ids: list[str],
        grammar_ids: list[str],
        types: list[ExerciseType],
    ) -> list[ExerciseSpec]:
        ordered_types = sorted(types, key=lambda value: value.value)
        return [
            ExerciseSpec(
                id=f"{lesson_id}-exercise{index:03d}",
                type=exercise_type,
                lesson_id=lesson_id,
                word_ids=list(introduced_word_ids),
                grammar_ids=list(grammar_ids),
                payload={},
            )
            for index, exercise_type in enumerate(ordered_types, start=1)
        ]

    def generate_all_supported_for_lesson(
        self,
        *,
        lesson_id: str,
        introduced_word_ids: list[str],
        grammar_ids: list[str],
    ) -> list[ExerciseSpec]:
        return self.generate_for_lesson(
            lesson_id=lesson_id,
            introduced_word_ids=introduced_word_ids,
            grammar_ids=grammar_ids,
            types=list(SUPPORTED_EXERCISE_TYPES),
        )
