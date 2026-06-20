# 0019 Distinguish noun/verb homographs as separate items

Status: done
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
- Done 2026-06-20 (Claude): item identity is now `(lemma, pos)` inside the
  orchestrator. Implementation:
  - `_is_verb_item(word, verb_lookup)` decides verb-vs-noun by the item's own POS
    (`pos == VERB and lemma in verb_lookup`), replacing the old
    `lemma not in verb_lookup` test that dropped a noun whenever a same-form verb
    existed. `_split_batch` centralizes the verb/non-verb/forms split.
  - `_group_by_lemma` makes the per-path `by_lemma` a multimap
    (`dict[str, list[Word]]`), so the noun stub and verb stub for a shared form
    both survive de-dup. Selecting a lemma expands to **all** its senses.
  - All three planning paths updated: `_plan_from_blueprints`,
    `_plan_with_theme_sequence` (catalog), and the default theme path in `plan()`.
  - `models.py` untouched (stays language-agnostic); change is identity only,
    no new schema. Validator unchanged — it reasons on shared surface forms, and
    once both senses are taught the noun lemma + verb forms are already in the
    allowed/forms sets.
- **Counting policy for 0017 (settled here):** a homograph's second sense **counts**
  as a separate item against the cumulative frequency budget. It shares the form
  but carries new meaning, so it is genuine new learning (unlike a derivable
  compound, see 0018, which should *not* count). Practically: noun+verb land in
  the same lesson via the catalog/blueprint paths (lemma expansion), so each adds
  to that lesson's new-item total; in the default/leftover chunking they may split
  across lessons. Either way both appear in the plan.
- Tests added in `tests/test_orchestrator.py` (section "noun/verb homographs"):
  both senses taught across catalog/blueprint/default paths, single-sense lemmas
  not duplicated, both forms pass validation once taught, and determinism.
