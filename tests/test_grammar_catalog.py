"""Tests for the per-language grammar catalog loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from course_compiler.generation.grammar import (
    GrammarDependencyError,
    load_grammar_catalog,
)

_CATALOG = """
A1:
  present-tense:
    title: Present tense
    dependsOn: []
    introducedInLesson: 1
    focus: Regular present-tense conjugation.
  articles:
    title: Articles (de / het)
    dependsOn: [present-tense]
    introducedInLesson: 3
A2:
  past-tense:
    title: Simple past
    dependsOn: [present-tense]
    introducedInLesson: 12
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "nl.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_topics_for_level_in_dependency_order(tmp_path: Path):
    path = _write(tmp_path, _CATALOG)

    plans = load_grammar_catalog(path, language="nl", cefr="A1")

    assert [p.topic.id for p in plans] == ["present-tense", "articles"]
    assert plans[0].topic.language == "nl"
    assert plans[0].topic.cefr == "A1"
    assert plans[0].focus == "Regular present-tense conjugation."
    assert plans[0].introduced_in_lesson == 1
    assert plans[1].introduced_in_lesson == 3


def test_returns_only_requested_level_but_resolves_lower_level_deps(tmp_path: Path):
    path = _write(tmp_path, _CATALOG)

    plans = load_grammar_catalog(path, language="nl", cefr="A2")

    # past-tense depends on A1's present-tense (a lower level); the cross-level
    # dependency must validate, but only the A2 page is returned for generation.
    assert [p.topic.id for p in plans] == ["past-tense"]
    assert plans[0].topic.cefr == "A2"


def test_unknown_dependency_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """
A1:
  articles:
    title: Articles
    dependsOn: [present-tense]
""",
    )

    with pytest.raises(GrammarDependencyError, match="Unknown dependency"):
        load_grammar_catalog(path, language="nl", cefr="A1")


def test_cycle_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """
A1:
  a:
    title: A
    dependsOn: [b]
  b:
    title: B
    dependsOn: [a]
""",
    )

    with pytest.raises(GrammarDependencyError, match="cycle"):
        load_grammar_catalog(path, language="nl", cefr="A1")


def test_missing_level_returns_empty(tmp_path: Path):
    path = _write(tmp_path, _CATALOG)

    assert load_grammar_catalog(path, language="nl", cefr="B1") == []
