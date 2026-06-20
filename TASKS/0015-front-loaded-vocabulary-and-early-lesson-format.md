# 0015 Front-loaded vocabulary curve + early-lesson format

Status: done
Priority: medium
Owner: Erik Vullings
Agent: claude
Area: generation
Depends on: 0012

## Context

Follow-up to the cold-start discussion. After 0012 (length decoupled from
new-word count) early lessons no longer demand impossible prose, but two
structural improvements remain, both inspired by the Delft Method:

1. **Front-loaded word budget.** `words_per_lesson` is a uniform 10
   (`orchestrator.py`). With no prior vocabulary to recombine you need critical
   mass to say anything, so early lessons should introduce more words (~40,
   Delft-style), tapering to a ~10 steady state. Make the per-lesson budget a
   descending curve rather than a constant.

2. **Format by lesson stage.** `_system_prompt` hardcodes "Coherent narrative
   text" (`lesson.py`). With a near-empty base, narrative is the wrong shape;
   short dialogues or labelled example sentences ("Ik ben Anna. Jij bent Tom.")
   tolerate sparse vocabulary far better. Switch format by stage and converge to
   recombinant narratives once a base (~100–150 words) exists. Optionally seed a
   "lesson 0" survival-kit word list (top ~50 lemmas) marked allowed.

## Acceptance Criteria

- Per-lesson new-word budget can follow a front-loaded curve (configurable;
  default tapers from a high first-lesson count to a steady state).
- Lesson format is a function of accumulated vocabulary size / lesson stage, not
  hardcoded to narrative.
- Reproducible and language-agnostic (curve + thresholds are config, not code
  branches per language).
- CLI surface (`--words-per-lesson`) reconciled with the curve.

## Implementation Notes

- `orchestrator.py` constructor + slicing logic (interacts with predefined-theme
  slicing in `_plan_with_theme_sequence`).
- `lesson.py` `_system_prompt` / format selection.
- Larger, more structural than 0012–0014; kept separate so it can iterate on its
  own. Deferred by Erik — 0012 already removes the hard blocker.

## Agent Notes

- Initial note: parked as the next step after the three content-generation
  changes (0012–0014) land.
- Done 2026-06-20.
  - **Front-load (A):** `orchestrator.py` constructor gains `first_lesson_words`
    and `front_load_lessons`; `_budget_for(lesson)` tapers linearly from
    `first_lesson_words` (L1) to `words_per_lesson` over `front_load_lessons`,
    then holds. `_schedule_slices` applies it. Wired into the default `plan()`
    chunk loop (global lesson counter) and the `_plan_with_theme_sequence`
    non-spread branch. Blueprint/spread paths keep uniform. Default
    (`first_lesson_words=None`) = uniform, so all prior tests pass unchanged.
  - **Format (B):** `lesson.py` `_system_prompt` now takes a `fmt`; `generate`
    picks `examples` vs `narrative` from `len(allowed_words)` vs the configurable
    `narrative_vocab_threshold` (default 60). Structure markers unchanged, so the
    parser is unaffected.
  - **CLI:** added `--first-lesson-words` and `--front-load-lessons`;
    `--words-per-lesson` documented as the steady-state count.
  - Tests: `test_orchestrator.py` (front-loaded taper + uniform default),
    `test_lesson_generator.py` (example vs narrative + configurable threshold).
    155 passed; lesson.py + orchestrator.py ruff clean.
  - Not done (future): "lesson 0" survival-kit word list; blueprint-path
    front-loading. Neither blocks the cold-start goal.
