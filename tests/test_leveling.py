"""Tests for cumulative-frequency-budget CEFR level assignment."""

from __future__ import annotations

from course_compiler.leveling import LevelItem, assign_levels


def _items(*keys: str) -> list[LevelItem]:
    """Items ranked by frequency in the order given (rank 1 = most frequent)."""
    return [LevelItem(key=k, rank=i) for i, k in enumerate(keys, start=1)]


def test_cumulative_budget_splits_by_frequency():
    """3 most frequent -> A1; next 2 -> A2 (cumulative budget = 5)."""
    items = _items("w1", "w2", "w3", "w4", "w5")
    result = assign_levels(items, budgets={"A1": 3, "A2": 5})
    assert result == {
        "w1": "A1",
        "w2": "A1",
        "w3": "A1",
        "w4": "A2",
        "w5": "A2",
    }


def test_items_beyond_highest_budget_are_excluded():
    items = _items("w1", "w2", "w3")
    result = assign_levels(items, budgets={"A1": 2})
    assert result == {"w1": "A1", "w2": "A1"}
    assert "w3" not in result  # beyond budget -> no level


def test_nt2lex_floor_prevents_placement_below_attested_level():
    """A very frequent word attested only at B1 is not placed in A1."""
    items = [
        LevelItem(key="rare_but_b1", rank=1, floor="B1"),
        LevelItem(key="common1", rank=2),
        LevelItem(key="common2", rank=3),
    ]
    result = assign_levels(items, budgets={"A1": 2, "A2": 3, "B1": 4})
    assert result["rare_but_b1"] == "B1"  # floor respected, not A1
    assert result["common1"] == "A1"
    assert result["common2"] == "A1"


def test_floored_word_rolls_forward_when_its_level_is_full():
    """A2-attested word rolls into B1 when the A2 budget is already spent."""
    items = [
        LevelItem(key="a1", rank=1),
        LevelItem(key="a2", rank=2),
        LevelItem(key="floor_a2_first", rank=3, floor="A2"),
        LevelItem(key="floor_a2_second", rank=4, floor="A2"),
    ]
    # caps: A1=2, A2=1, B1=2
    result = assign_levels(items, budgets={"A1": 2, "A2": 3, "B1": 5})
    assert result["a1"] == "A1"
    assert result["a2"] == "A1"
    assert result["floor_a2_first"] == "A2"
    assert result["floor_a2_second"] == "B1"  # A2 full -> rolls forward, never below floor


def test_floor_above_all_budgets_excludes_item():
    items = [LevelItem(key="c1word", rank=1, floor="C1")]
    result = assign_levels(items, budgets={"A1": 5, "A2": 10})
    assert "c1word" not in result


def test_assignment_is_deterministic_with_stable_tie_break():
    """Equal-rank items break ties by key, so output is reproducible."""
    items_a = [
        LevelItem(key="beta", rank=1),
        LevelItem(key="alpha", rank=1),
    ]
    items_b = list(reversed(items_a))
    budgets = {"A1": 1}
    result_a = assign_levels(items_a, budgets=budgets)
    result_b = assign_levels(items_b, budgets=budgets)
    assert result_a == result_b
    # "alpha" sorts before "beta" on the tie, so it wins the single A1 slot.
    assert result_a == {"alpha": "A1"}


def test_unranked_items_sort_after_ranked():
    items = [
        LevelItem(key="ranked", rank=5),
        LevelItem(key="unranked", rank=None),
    ]
    result = assign_levels(items, budgets={"A1": 1})
    assert result == {"ranked": "A1"}  # the ranked word takes the only slot


def test_budgets_are_plain_configuration():
    """Different budget dicts reshape the split without code changes."""
    items = _items("w1", "w2", "w3", "w4")
    tight = assign_levels(items, budgets={"A1": 1, "A2": 2})
    loose = assign_levels(items, budgets={"A1": 4})
    assert tight == {"w1": "A1", "w2": "A2"}
    assert loose == {"w1": "A1", "w2": "A1", "w3": "A1", "w4": "A1"}


def test_realistic_dutch_budgets():
    """The agreed A1/A2/B1/B2 cumulative budgets bucket a frequency ladder."""
    budgets = {"A1": 750, "A2": 2000, "B1": 3500, "B2": 5500}
    items = _items(*(f"w{i:05d}" for i in range(1, 6001)))
    result = assign_levels(items, budgets=budgets)
    by_level: dict[str, int] = {}
    for level in result.values():
        by_level[level] = by_level.get(level, 0) + 1
    assert by_level == {"A1": 750, "A2": 1250, "B1": 1500, "B2": 2000}
    assert result["w00001"] == "A1"
    assert result["w05500"] == "B2"
    assert "w05501" not in result  # beyond top budget -> excluded
