# 0017 Level assignment by cumulative frequency budget

Status: open
Priority: high
Owner: Erik Vullings
Agent: unassigned
Area: converters
Depends on: 0019

## Context

Words are currently bucketed by NT2Lex's **earliest-attested** CEFR level, which
dumps a huge mid-frequency tail into A2 (4226 unique items) and gives no control
over how many words a learner knows per level.

Erik's target: a learner knows **~2000 words by the end of A2**. Achieve this by
assigning levels by a **cumulative frequency budget** instead of the raw NT2Lex
tag. Agreed starting budgets (verbs counted as words; tune later):

- A1 = 750, A2 = 2000 (cumulative), B1 = 3500, B2 = 5500

A lemma belongs to the level where the cumulative budget reaches it — so words
NT2Lex tagged A2 but beyond A2's budget **roll forward into B1/B2**, not lost.
Only words beyond the top budget (the rare long tail) are excluded. Use the
NT2Lex level as a **floor** (never introduce a word *below* its attested level
even if frequent): frequency = primary signal, NT2Lex = floor.

Counts the budget operates on are **(lemma, pos) items** per 0019, and exclude
derivable compounds per 0018 (a compound doesn't consume budget).

## Test plan (TDD — one behavior per cycle)

- Given lemmas ranked by frequency and budgets `{A1:3, A2:5}`, the 3 most
  frequent get A1; the next 2 get A2 (cumulative = 5).
- A frequent lemma with an NT2Lex floor of B1 is **not** placed below B1 even if
  its frequency would put it in A1.
- Lemmas beyond the highest budget are excluded (assigned no level).
- Assignment is deterministic for identical inputs (stable tie-break, e.g. by
  lemma) and reproducible.
- Budgets are configuration (per-level dict), not language-specific code.

## Implementation Notes

- A post-import / converter-side level-assignment pass; interacts with
  `converters/` CEFR handling and `frequency.py`. Keep `models.py` generic.
- Reference counts today (unique, verbs incl.): A1 = 762, A2 = 4226.

## Agent Notes

- Proposed 2026-06-20. Do after 0019 (item identity) and before 0018 (compounds
  need the per-level known set this defines).
