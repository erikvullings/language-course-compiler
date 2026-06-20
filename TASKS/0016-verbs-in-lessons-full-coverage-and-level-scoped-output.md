# 0016 Verbs in lessons, full theme coverage, level-scoped output

Status: done
Priority: high
Owner: Erik Vullings
Agent: claude
Area: generation
Depends on: 0013

## Context

Three issues found while reviewing the A1 blueprint:

1. **Verbs were missing from generated lessons.** The orchestrator supports
   `plan(..., verbs=...)`/`generate(..., verbs=...)`, but the `generate-lessons`
   CLI never loaded or passed verbs. So lessons covered only the 658 A1 *content
   words*, not the 294 A1 verbs.
2. **The catalog path silently dropped vocabulary.** `_plan_with_theme_sequence`
   had two branches; the "scheduled" one capped coverage at
   `themes × words_per_lesson` and dropped the tail. Harmless for A1-without-verbs
   (spread branch), but A2 (3393 content words) was dropping ~2600 words, and
   wiring verbs would have made A1 drop ~150 too.
3. **Output path was not level-scoped.** Lessons were written to
   `courses/<lang>/lessons/`; they must be `courses/<lang>/<CEFR>/lessons/`.

## Acceptance Criteria

- `generate-lessons` loads verbs (`verbs.json` / `verbs/*.yaml`) and passes them
  to `plan` and `generate`; verbs appear in lesson seed vocabulary. ✅
- The catalog path distributes **all** content across the themes (one lesson per
  theme), dropping nothing. ✅
- Lessons are written under `<lexicon>/<CEFR>/lessons/` by default; export reads
  level-scoped dirs (falls back to the legacy flat dir). ✅
- Tests cover verb inclusion, full coverage, and the level-scoped path. ✅

## Implementation Notes

- `cli.py`: `_load_verbs_from_lexicon`; verbs threaded into plan/generate;
  `out_dir = lexicon_dir / args.cefr / "lessons"`; export merges per-level dirs.
- `orchestrator.py`: replaced the scheduled/spread branch with `_distribute`
  (always one lesson per theme, even split by default, front-loaded when
  configured); removed `_schedule_slices`; dropped the now-unused `math` import.
- Updated `test_orchestrator.py` 0013 case (1 theme/3 words now keeps all 3).

## Agent Notes

- Done (code) 2026-06-20. 158 tests pass; orchestrator ruff-clean.
- Re-running A1 preview to refresh seed lemmas (proposer cache misses because
  target_count/already_used changed once verbs are included).
- Follow-ups (separate tasks): front-loading on the catalog path (currently only
  the non-catalog default path honours it); multi-level export (lesson IDs
  collide across levels); vocabulary sizing (see 0017) and compounds (0018).
- VERIFIED 2026-06-20 against real Dutch data: A1 = 762 *unique* lemmas (the
  earlier "952" double-counted 190 noun/verb homographs like `antwoord`, `been`).
  Planner covers all 762 → 65 lessons, ~12 items each, verbs included. No
  coverage bug; the apparent shortfall was a miscount. A2 = 4226 unique.
