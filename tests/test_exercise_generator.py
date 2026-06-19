"""Tests for exercise generation scaffolding."""

from __future__ import annotations

from course_compiler.generation.exercises import (
    ExerciseGenerator,
    ExerciseSpec,
    ExerciseType,
    SUPPORTED_EXERCISE_TYPES,
)


def test_supported_exercise_types_match_project_spec():
    expected = {
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
    }
    assert set(SUPPORTED_EXERCISE_TYPES) == expected


def test_generate_for_lesson_references_only_ids():
    generator = ExerciseGenerator()

    exercises = generator.generate_for_lesson(
        lesson_id="lesson001",
        introduced_word_ids=["huis", "lopen"],
        grammar_ids=["present-tense"],
        types=[ExerciseType.FILL_IN_THE_BLANK, ExerciseType.MULTIPLE_CHOICE],
    )

    assert len(exercises) == 2
    first = exercises[0]
    assert first.lesson_id == "lesson001"
    assert first.word_ids == ["huis", "lopen"]
    assert first.grammar_ids == ["present-tense"]
    assert first.payload == {}


def test_generate_for_lesson_is_deterministic():
    generator = ExerciseGenerator()

    args = dict(
        lesson_id="lesson003",
        introduced_word_ids=["huis", "lopen"],
        grammar_ids=["present-tense"],
        types=[ExerciseType.TYPING, ExerciseType.FILL_IN_THE_BLANK],
    )
    a = generator.generate_for_lesson(**args)
    b = generator.generate_for_lesson(**args)

    assert a == b


def test_generate_all_types_for_lesson():
    generator = ExerciseGenerator()

    specs = generator.generate_all_supported_for_lesson(
        lesson_id="lesson009",
        introduced_word_ids=["huis"],
        grammar_ids=["articles"],
    )

    assert len(specs) == len(SUPPORTED_EXERCISE_TYPES)
    assert all(isinstance(spec, ExerciseSpec) for spec in specs)
