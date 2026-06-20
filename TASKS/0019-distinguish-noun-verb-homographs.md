# 0019 Distinguish noun/verb homographs as separate items

Status: open
Priority: high
Owner: Erik Vullings
Agent: unassigned
Area: generation
Depends on: 0016

## Context

Some Dutch lemmas are both a noun and a verb with the same form — e.g. `eten`
(food / to eat), and ~190 such forms at A1, ~650 at A2. The planner currently
collapses vocabulary by **lemma**: `by_lemma = {w.lemma: w}` keeps one entry, and
a homograph is treated only as a verb (`w.lemma in verb_lookup`). So if the noun
`eten` is "known", the learner may **never be taught the verb** `eten` (and vice
versa), even though the meaning is different.

Fix: the learnable unit is **(lemma, part-of-speech)**, not lemma. A noun and a
verb sharing a form are two distinct items, each taught (possibly in different
lessons), so neither sense is skipped.

Relationship to other tasks: this defines the *item identity* that 0017's
frequency budget counts. Decide counting policy: the second sense shares the
*form* but carries new *meaning*, so it likely still counts (unlike a derivable
compound — see 0018). Settle here.

## Test plan (TDD — one behavior per cycle)

- A lemma present as both a `Word` (noun) and a `Verb` yields **two** new-word
  items across the plan; neither is dropped.
- The two items can land in different lessons; both appear in the blueprint.
- A lemma that is only a noun (or only a verb) still yields exactly one item
  (no regression / no spurious duplication).
- The vocabulary validator accepts both the noun and the verb forms once taught
  (allowed set keyed by sense, not collapsed).
- Determinism preserved: same inputs → same plan.

## Implementation Notes

- Touch `LessonOrchestrator` de-dup (`by_lemma`, `verb_lookup`, `all_content`)
  to key by `(lemma, pos)`; keep `models.py` language-agnostic.
- Watch the allowed-vocabulary/validator path — it currently reasons in lemmas.
- Reuse the existing verb-stub mechanism; the change is identity, not new schema.

## Agent Notes

- Proposed 2026-06-20 (Erik): "if I learn the noun 'antwoord', I may never learn
  the verb — treat them distinctly."
