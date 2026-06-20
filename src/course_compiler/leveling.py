"""Assign CEFR levels by a cumulative frequency budget (language-agnostic).

Instead of bucketing words by their raw lexical-resource tag (which dumps a huge
mid-frequency tail into one level), levels are assigned so that a learner knows a
controlled number of words per level — e.g. ~2000 by the end of A2. The budgets
are *cumulative* counts per level (``{"A1": 750, "A2": 2000, ...}``) and are plain
configuration, never code, so the policy stays language-independent.

Two signals combine:

* **Frequency** is the primary signal: items are filled into levels most-frequent
  first, consuming each level's capacity (the budget increment) in turn.
* A per-item **floor** (e.g. the earliest level at which a resource attests the
  word) is a *minimum*: an item is never placed below its floor, but it may roll
  *forward* into a higher level when its floor level's budget is already spent.

Items beyond the highest budget (the rare long tail) are excluded — they appear
in no level. The unit of counting is whatever ``key`` the caller chooses; per the
project this is a ``(lemma, part-of-speech)`` item, so noun/verb homographs each
consume budget independently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Canonical CEFR ordering — lower index = lower (earlier) level.
CEFR_ORDER: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")


@dataclass(frozen=True)
class LevelItem:
    """One budget-consuming unit.

    Args:
        key: Stable, unique identifier (used both as result key and tie-break).
        rank: Frequency rank, 1 = most frequent. ``None`` sorts after all ranked
            items (least frequent).
        floor: Lowest CEFR level the item may be assigned to, or ``None`` for no
            floor. Unknown values are treated as no floor.
    """

    key: str
    rank: int | None = None
    floor: str | None = None


def assign_levels(
    items: Iterable[LevelItem],
    budgets: Mapping[str, int],
) -> dict[str, str]:
    """Map item keys to CEFR levels by cumulative frequency budget.

    Args:
        items: The units to place. Order is irrelevant — assignment is by
            ``(rank, key)`` so the result is deterministic.
        budgets: Cumulative item count per level, e.g.
            ``{"A1": 750, "A2": 2000, "B1": 3500, "B2": 5500}``. Only levels named
            here are used; they are processed in canonical CEFR order.

    Returns:
        ``{key: level}`` for every placed item. Items that find no slot at or above
        their floor (beyond the top budget) are omitted entirely.
    """
    levels = [level for level in CEFR_ORDER if level in budgets]
    if not levels:
        return {}

    # Capacity of each level = the increment of the cumulative budget. Clamp at 0
    # and track the running maximum so a non-monotonic budget can't grant negative
    # or rewound capacity.
    remaining: dict[str, int] = {}
    running = 0
    for level in levels:
        cumulative = budgets[level]
        remaining[level] = max(0, cumulative - running)
        running = max(running, cumulative)

    def floor_start(floor: str | None) -> int:
        """Index into ``levels`` of the lowest level at or above ``floor``."""
        if floor is None or floor not in CEFR_ORDER:
            return 0
        floor_rank = CEFR_ORDER.index(floor)
        for i, level in enumerate(levels):
            if CEFR_ORDER.index(level) >= floor_rank:
                return i
        return len(levels)  # floor is above every budgeted level -> excluded

    def sort_key(item: LevelItem) -> tuple[float, str]:
        rank = float(item.rank) if item.rank is not None else float("inf")
        return (rank, item.key)

    result: dict[str, str] = {}
    for item in sorted(items, key=sort_key):
        for level in levels[floor_start(item.floor) :]:
            if remaining[level] > 0:
                result[item.key] = level
                remaining[level] -= 1
                break
    return result
