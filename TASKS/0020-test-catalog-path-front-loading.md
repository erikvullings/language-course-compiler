# 0020 Test/verify front-loading on the catalog path

Status: open
Priority: low
Owner: Erik Vullings
Agent: unassigned
Area: generation
Depends on: 0016

## Context

`_distribute` already honours `first_lesson_words` by weighting the per-theme
slice sizes (early themes larger) when front-loading is configured; the even
split is used otherwise. This means catalog-driven runs *should* front-load, but
that path is currently **untested** — `test_front_loaded_budget_*` only covers
the non-catalog default path. Confirm it works end-to-end and lock it with a test
(or fix if the weighted distribution misbehaves at scale / rounding).

## Test plan (TDD — one behavior per cycle)

- With predefined themes + `first_lesson_words=40, words_per_lesson=10`, the
  first lesson’s item count is largest and tapers toward the steady state, while
  the slices still sum to the full vocabulary (nothing dropped).
- Without `first_lesson_words`, the catalog path remains an even split
  (regression guard).
- Rounding never produces a zero-size lesson or loses/duplicates an item.

## Implementation Notes

- `orchestrator.py::_distribute` (weighted branch). Likely test-only.

## Agent Notes

- Carved out of 0016 follow-ups, 2026-06-20.
