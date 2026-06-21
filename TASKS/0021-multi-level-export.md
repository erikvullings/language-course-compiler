# 0021 Multi-level export (lesson IDs collide across levels)

Status: done
Priority: medium
Owner: Erik Vullings
Agent: unassigned
Area: export
Depends on: 0016

## Context

Lessons are now written per level: `courses/<lang>/<CEFR>/lessons/lessonNNN.json`.
But lesson IDs restart per level (`lesson001` exists under A1 *and* A2), and
`export` keys lessons by ID into one flat bundle — so a multi-level export would
have A2's `lesson001` overwrite A1's. The current fallback only handles the
single-level case (reads per-level dirs when the flat dir is empty).

Make export level-aware: namespace lessons by level (e.g. `A1/lesson001`) or emit
per-level lesson bundles, and reflect that in `manifest.json` so the SPA can load
a multi-level course.

## Test plan (TDD — one behavior per cycle)

- Export a course with both `A1/lessons/lesson001.json` and
  `A2/lessons/lesson001.json` → both survive in the bundle, distinguishable by
  level (no overwrite).
- Single-level course still exports as today (back-compat).
- Manifest lists the levels present.

## Implementation Notes

- `cli.py` export handler + `_load_lessons_for_export` callsite; decide bundle
  shape (namespaced ids vs per-level lesson dirs in `export/`).

## Agent Notes

- Carved out of 0016 follow-ups, 2026-06-20.
- Done 2026-06-20 (Claude). Bundle shape chosen: **per-level lesson dirs** in the
  export. `cli.py` export handler now:
  - `_discover_lesson_levels(course_dir)` finds `<course>/lessons/<LEVEL>` dirs
    named after a real CEFR level (ordered by `leveling.CEFR_ORDER`). (Layout
    updated 2026-06-20: lessons are written under `lessons/<LEVEL>/` rather than
    `<LEVEL>/lessons/`, matching the generator's output dir.)
  - When levels exist, each level's lessons are written to
    `export/lessons/<LEVEL>/<id>.json` with a `"level"` field added to the payload,
    so A1/lesson001 and A2/lesson001 no longer collide.
  - Legacy flat `<course>/lessons` still exports to `export/lessons/<id>.json`
    unchanged (back-compat).
  - `manifest.json` gained `"levels": [...]` (the ordered levels present; `[]` for
    a flat single-level course).
  - Tests: `test_export_is_level_aware_and_does_not_overwrite` (both survive,
    payloads carry level, manifest lists levels) and an assertion on the existing
    flat-export test that `levels == []`.
