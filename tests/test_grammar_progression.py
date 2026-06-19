"""Tests for grammar dependency planning and cycle detection."""

from __future__ import annotations

import pytest

from course_compiler.generation.grammar import GrammarDependencyError, GrammarProgressionPlanner, GrammarTopic


def _topic(topic_id: str, *, depends_on: list[str] | None = None) -> GrammarTopic:
    return GrammarTopic(
        id=topic_id,
        language="nl",
        title=topic_id.replace("-", " ").title(),
        cefr="A1",
        depends_on=depends_on or [],
    )


def test_plan_orders_topics_by_dependencies():
    topics = [
        _topic("past-tense", depends_on=["word-order"]),
        _topic("present-tense"),
        _topic("word-order", depends_on=["present-tense"]),
    ]

    planner = GrammarProgressionPlanner()
    ordered = planner.plan(topics)

    assert [t.id for t in ordered] == ["present-tense", "word-order", "past-tense"]


def test_plan_is_deterministic_for_independent_topics():
    topics = [
        _topic("articles"),
        _topic("present-tense"),
        _topic("plural-nouns"),
    ]

    planner = GrammarProgressionPlanner()
    ordered = planner.plan(topics)

    assert [t.id for t in ordered] == ["articles", "plural-nouns", "present-tense"]


def test_plan_raises_cycle_error_when_graph_has_cycle():
    topics = [
        _topic("present-tense", depends_on=["word-order"]),
        _topic("word-order", depends_on=["present-tense"]),
    ]

    planner = GrammarProgressionPlanner()

    with pytest.raises(GrammarDependencyError, match="cycle"):
        planner.plan(topics)


def test_plan_raises_for_unknown_dependency():
    topics = [_topic("past-tense", depends_on=["word-order"])]

    planner = GrammarProgressionPlanner()

    with pytest.raises(GrammarDependencyError, match="Unknown dependency"):
        planner.plan(topics)
