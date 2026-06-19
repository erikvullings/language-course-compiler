# 0009 Verb integration in orchestrator

Status: open
Priority: medium
Owner: unassigned
Agent: unassigned
Area: generation
Depends on: 0008

## Context
The orchestrator (task 0008) only loads `words.json`. Verbs live in `verbs.json`
as separate `Verb` model objects (infinitive, conjugation tables, CEFR, frequency).
Each verb lemma should count as exactly one new word in a lesson; its conjugated
forms (loopt, liep, gelopen, …) must be accepted by the vocabulary validator
without triggering a retry.

## Acceptance Criteria
- `LessonOrchestrator` (or its CLI entry point) loads both `words.json` and
  `verbs.json` and merges them for planning
- Each verb contributes one entry (its lemma / infinitive) to the new-words list
- All surface forms for an introduced verb (all values from `present`, `past`,
  `perfect`, `imperative`, `future`, `conditional`, `subjunctive` tables) are
  added to the function-lemmas-equivalent exempt set so the validator never
  rejects them
- CEFR and frequency filtering works identically for verbs as for words
- Existing tests remain green; new tests cover verb loading and form exemption

## Implementation Notes
- The exempt set for verb forms is not quite "function lemmas" (which are POS-based)
  — consider a separate `allowed_forms: set[str]` parameter on `VocabularyValidator`
  or extend `extra_function_lemmas` to cover this use case (simpler, same effect).
- The `Lemmatizer` should ideally map conjugated forms back to the infinitive; until
  a language-specific lemmatizer exists, adding all surface forms to the exempt set
  is the correct fallback.
- CLI: no new flags needed; the orchestrator loads verbs automatically when
  `verbs.json` is present alongside `words.json`.

## Agent Notes
- Not started.
