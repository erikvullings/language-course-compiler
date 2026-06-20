# 0018 Compound / derivable-word detection (introduce but don't count)

Status: open
Priority: medium
Owner: Erik Vullings
Agent: unassigned
Area: converters
Depends on: 0017

## Context

Dutch (like German) forms transparent compounds: `koffie` + `pot` → `koffiepot`.
Knowing the parts makes the compound nearly free. A measurement on the real
lexicon (greedy splitter vs the known-lemma set, length ≥5) found ~47% of A2
content words and ~34% of A1 decomposable into ≥2 known lemmas — an upper bound
(false positives like `balkon`/`banaan`, and particle-prefixed forms like
`aanval` = aan+vallen), so the truly transparent rate is plausibly ~25–35%.

Erik's decision: **compounds may be introduced in lessons, but should not count
as new words** — they don't consume the frequency budget (0017), because the
learner already knows the parts. Opaque compounds (`vliegtuig` = fly+gear =
airplane; `handschoen` = hand+shoe = glove) are NOT free and should still count.

## Test plan (TDD — one behavior per cycle)

- Splitter: `koffiepot` → `[koffie, pot]`; `stationsplein` → `[station, plein]`
  (linking `-s-`); the word itself is never used as a part.
- A non-decomposable lemma (`tafel`) is reported as not-a-compound.
- A lemma marked derivable does **not** consume the frequency budget in 0017,
  yet may still appear as allowed vocabulary / be introduced in a lesson.
- An opaque compound (configured/return-flagged as non-transparent) still counts
  as a new word.
- Splitting is deterministic and language-pluggable (linking morphemes per lang).

## Implementation Notes

- Splitter is language-dependent (linking morphemes -s-, -e(n)-) → pluggable
  analyzer / converter side, not generic `models.py`.
- Prototype drafted in conversation (DP over known lemmas + linkers, excluding
  the word itself). Add a transparency confirmation step (list/LLM) to filter the
  short-fragment false positives before excluding from the budget.
- Combined with 0017, expected to bring A2's genuine new-word count from ~4226
  toward the ~1250 that hits "2000 words by end of A2".

## Agent Notes

- Proposed 2026-06-20 (Erik): "compound words can be introduced in the lessons,
  but they should not be counted as new words."
