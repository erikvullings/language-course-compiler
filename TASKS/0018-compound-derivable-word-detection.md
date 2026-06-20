# 0018 Compound / derivable-word detection (introduce but don't count)

Status: done
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
- Done 2026-06-20 (Claude). Implementation:
  - New generic module `src/course_compiler/compounds.py`: `split_compound(word,
    known_lemmas, *, linkers, min_part_len)` — greedy longest-leading-part DP over
    the known-lemma set, with optional linking morphemes tried between parts; the
    word itself is excluded so it never "splits" into one copy of itself; returns
    `[]` unless ≥2 parts. `is_derivable_compound(...)` adds an `opaque` set so
    opaque compounds (e.g. `handschoen`) report False and keep counting.
    Language-pluggable: linkers are a caller argument, nothing in `models.py`.
  - Integrated with the 0017 budget pass: `reassign_cefr_by_budget` gained
    `linkers`/`opaque`. Transparent compounds are dropped from the counting set so
    they don't consume budget, then levelled to `max(level of parts)` so they can
    still be introduced/allowed. Dutch linkers `("en","s","e","n")` live in
    `dutch.py` as `DUTCH_LINKERS`.
  - `convert_iterables` takes `linkers`/`opaque`; `convert` takes
    `detect_compounds`/`opaque`. CLI: `course import --budgets ... --compounds`.
  - Tests: `tests/test_compounds.py` (splitter bullets + opaque + min-part-len +
    determinism) and two converter integration tests (transparent frees a slot &
    is levelled; opaque still counts).
- NOT done (deliberately deferred): the transparency *confirmation* step
  (curated list / LLM) to weed out short-fragment false positives like
  `balkon`→`bal+kon` or particle-prefixed `aanval`→`aan+vallen`. The hook exists
  (`opaque` set + `min_part_len`); populating a real opaque list against the live
  lexicon is follow-up data work, not code. Splitter currently has no verb-stem /
  particle awareness, so run with a vetted `opaque` list on real data.
