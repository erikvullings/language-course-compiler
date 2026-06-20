# 0013 Theme-driven word selection (generate-then-filter)

Status: done
Priority: high
Owner: Erik Vullings
Agent: claude
Area: generation
Depends on: none

## Context

Lesson seed words feel arbitrary and disconnected from the theme. The current
dataflow runs backwards: `_theme_candidate_pool` (`orchestrator.py`) scores
**our lexicon** against the theme by literal token overlap
(`theme_tokens & lexical_tokens`), then the LLM picks from that pool. Token
overlap barely fires — theme "food and drink" shares zero tokens with *brood,
appel, melk* — so the pool collapses to "most frequent untaught words", i.e.
arbitrary.

Invert it (Erik's proposal): ask the LLM to **generate** ~5n communicatively
relevant words for the theme/goals from its own knowledge, then **filter** those
against our lexicon (which holds CEFR, frequency, translations, audio,
conjugations) and rank survivors by frequency, taking n. The LLM supplies
semantic relevance; our data supplies level-appropriateness + resource
availability. Generating 5n gives margin for proposals that fall out (not in
lexicon / already taught).

Preserve:
- **Coverage** — rank survivors by frequency and backfill from the existing
  candidate pool / frequency order so the long tail still gets taught.
- **Reproducibility** — this LLM call now *determines the curriculum*, so it must
  be cached keyed on (cefr, theme, goals, target_count, already_used), like the
  other `LLMThemeAssigner` calls.

## Acceptance Criteria

- New `LLMThemeAssigner.propose_theme_vocabulary(...)` asks the LLM for ~5×
  target_count theme-relevant lemmas, parses JSON (`{"vocabulary": [str]}`),
  is disk-cached, and returns `[]` on provider/parse error.
- `LessonOrchestrator._plan_with_theme_sequence` prefers proposals: intersect
  with the lexicon (case-insensitive lemma match), drop already-used, rank by
  frequency, take target_count; fall back to the existing candidate-pool
  selector and frequency backfill when proposals yield too few.
- Proposed lemmas not in the lexicon are dropped (never appear as new words).
- Selection stays deterministic given cached LLM responses.
- Tests cover propose (parse, cache, error-fallback) and the orchestrator
  filtering/ranking behaviour. Suite green, ruff clean.

## Implementation Notes

- `src/course_compiler/generation/themes.py`: add `propose_theme_vocabulary` +
  system prompt + parser + `ThemeAssigner` protocol entry.
- `src/course_compiler/generation/orchestrator.py`: wire proposer into
  `_plan_with_theme_sequence` via `getattr` (backward-compatible with assigners
  lacking the method), before the existing `select_seed_lemmas_for_theme` path.
- Match LLM proposals to `by_lemma` exactly and lowercased.

## Agent Notes

- Initial note: second of the three agreed changes. Keep the candidate-pool path
  as fallback rather than deleting it — it guards coverage when proposals miss.
- Done 2026-06-20. `themes.py`: added `propose_theme_vocabulary` (+ system prompt,
  `_parse_proposed_vocabulary`, protocol entry), cached, oversamples 5×n, `[]` on
  error. `orchestrator.py`: `_plan_with_theme_sequence` now tries the proposer
  first — filters proposals to the lexicon (exact + lowercased), drops used,
  frequency-ranks survivors — then falls back to `select_seed_lemmas_for_theme`
  and frequency backfill. Threaded `language` through `plan`/`generate` and the
  CLI preview call. Tests: `test_theme_assigner.py` (parse/cache/error),
  `test_orchestrator.py` (lexicon filtering + frequency ranking). 149 passed.
  Introduced no new ruff errors (repo baseline already had 13 pre-existing).
