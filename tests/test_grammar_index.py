"""Tests for the export-time grammar/verb review indices."""

from __future__ import annotations

from course_compiler.cli import _common_verbs_by_level, _grammar_by_lesson


def _grammar() -> dict[str, dict]:
    return {
        "present-tense": {"id": "present-tense", "cefr": "A1", "introducedInLesson": 1},
        "articles": {"id": "articles", "cefr": "A1", "introducedInLesson": 3},
        "past-tense": {"id": "past-tense", "cefr": "A2", "introducedInLesson": 2},
    }


def test_grammar_by_lesson_maps_new_and_available():
    level_lessons = {
        "A1": {"lesson001": {}, "lesson002": {}, "lesson003": {}, "lesson010": {}},
    }

    index = _grammar_by_lesson(level_lessons, _grammar())["A1"]

    # Lesson 1 introduces present-tense; it is also available.
    assert index["lesson001"] == {
        "new": ["present-tense"],
        "available": ["present-tense"],
    }
    # Lesson 2 introduces nothing new but present-tense is available (review).
    assert index["lesson002"] == {"new": [], "available": ["present-tense"]}
    # Lesson 3 introduces articles; both topics now available.
    assert index["lesson003"]["new"] == ["articles"]
    assert index["lesson003"]["available"] == ["articles", "present-tense"]
    # A later review lesson has all A1 grammar available, nothing new.
    assert index["lesson010"]["new"] == []
    assert index["lesson010"]["available"] == ["articles", "present-tense"]


def test_grammar_by_lesson_includes_lower_levels_as_available():
    level_lessons = {"A2": {"lesson001": {}, "lesson002": {}}}

    index = _grammar_by_lesson(level_lessons, _grammar())["A2"]

    # At A2, all A1 grammar is available from the start; A2 past-tense at lesson 2.
    assert index["lesson001"]["available"] == ["articles", "present-tense"]
    assert index["lesson002"]["new"] == ["past-tense"]
    assert index["lesson002"]["available"] == [
        "articles",
        "present-tense",
        "past-tense",
    ]


def test_common_verbs_by_level_ranks_by_frequency_and_references_ids():
    verbs = {
        "lopen": {"id": "lopen", "cefr": "A1", "frequency": {"rank": 30}},
        "zijn": {"id": "zijn", "cefr": "A1", "frequency": {"rank": 1}},
        "hebben": {"id": "hebben", "cefr": "A1", "frequency": {"rank": 2}},
        "verdwijnen": {"id": "verdwijnen", "cefr": "B1", "frequency": {"rank": 5}},
    }

    result = _common_verbs_by_level(verbs, ["A1", "B1"], limit=2)

    assert result["A1"] == ["zijn", "hebben"]  # most frequent A1 verbs, ids only
    assert result["B1"] == ["verdwijnen"]
