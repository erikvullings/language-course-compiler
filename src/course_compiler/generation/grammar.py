"""Grammar progression planning with dependency validation.

This module keeps grammar sequencing language-agnostic by operating on generic
grammar-topic records and dependency ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class GrammarDependencyError(RuntimeError):
    """Raised when grammar dependencies are invalid or cyclic."""


@dataclass(frozen=True)
class GrammarTopic:
    """A grammar unit that can depend on earlier grammar units."""

    id: str
    language: str
    title: str
    cefr: str
    depends_on: list[str] = field(default_factory=list)


class GrammarProgressionPlanner:
    """Produce an acyclic, deterministic grammar progression order."""

    def plan(self, topics: list[GrammarTopic]) -> list[GrammarTopic]:
        """Return topics ordered so dependencies always come first.

        Raises:
            GrammarDependencyError: For unknown dependencies or dependency cycles.
        """
        by_id = {topic.id: topic for topic in topics}
        if len(by_id) != len(topics):
            raise GrammarDependencyError("Duplicate grammar topic id detected.")

        for topic in topics:
            for dep in topic.depends_on:
                if dep not in by_id:
                    raise GrammarDependencyError(
                        f"Unknown dependency {dep!r} referenced by grammar topic {topic.id!r}."
                    )

        in_degree: dict[str, int] = {topic.id: 0 for topic in topics}
        outgoing: dict[str, set[str]] = {topic.id: set() for topic in topics}

        for topic in topics:
            for dep in topic.depends_on:
                outgoing[dep].add(topic.id)
                in_degree[topic.id] += 1

        ready = sorted([topic_id for topic_id, degree in in_degree.items() if degree == 0])
        ordered_ids: list[str] = []

        while ready:
            current = ready.pop(0)
            ordered_ids.append(current)
            for follower in sorted(outgoing[current]):
                in_degree[follower] -= 1
                if in_degree[follower] == 0:
                    ready.append(follower)
            ready.sort()

        if len(ordered_ids) != len(topics):
            blocked = sorted(topic_id for topic_id, degree in in_degree.items() if degree > 0)
            raise GrammarDependencyError(
                "Grammar dependency cycle detected; unresolved topics: " + ", ".join(blocked)
            )

        return [by_id[topic_id] for topic_id in ordered_ids]
