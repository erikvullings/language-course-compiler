# 0022 Coherent lesson text via English outline + seed anchors

Status: done
Priority: high
Owner: Erik Vullings
Agent: Claude
Area: generation

## Context

Generated A1 lesson text reads as loose, often nonsensical sentences
(`Ik heet ik en zij heet zij`, `Het land is een land en de taal is taal`), not as
a story. Root causes (from real `courses/nl/lessons/A1/lesson001.json`):

1. Lesson 1's "new words" are all high-frequency *light* words (greetings + modal
   verbs + bland adjectives) with **no concrete nouns** to anchor a scene.
2. Below 60 allowed words the prompt asks for "several short example sentences" —
   loose by design.
3. Strict vocabulary discipline + retry (4–5 attempts) degrades prose into
   circular filler; inflections of allowed lemmas are rejected.

Decision (Erik): pursue **Option 1 (hybrid)** — keep the cumulative lesson
structure but make the text coherent and CEFR-bounded rather than strictly
limited to the prior+current set. Because this is a **multi-language** generator,
the new authoring hints live in `themes.yaml` in **English** (language-agnostic
metadata), translated/realized by the LLM into the target language.

## Plan

- Extend `themes.yaml` per-lesson with two optional English fields:
  - `seedWords`: concrete anchor words (drive vocabulary selection).
  - `outline`: a brief scenario (drives the narrative).
- `seedWords` → `propose_theme_vocabulary` prompt so concrete nouns are selected
  into early lessons.
- `outline` → lesson-writer prompt; always write a coherent narrative/dialogue.
- Validation: reject only **above-CEFR** words; tolerate every in-level word
  (uncapped) so coherence wins over strict prior-only discipline.

## Test plan (TDD — one behavior per cycle)

- `_load_predefined_themes` parses `seedWords` + `outline` into `LessonThemePlan`
  (absent → empty, back-compat).
- The orchestrator passes a lesson's `outline` through to the generator and the
  English `seedWords` through to `propose_theme_vocabulary`.
- `VocabularyValidator` with `extra_tolerance=None` tolerates all at/below-CEFR
  extras and still rejects above-CEFR words.
- `LessonGenerator` includes the outline in the prompt and writes narrative.
- Determinism preserved; existing tests still pass.

## Implementation Notes

- `themes.py` (`LessonThemePlan`, `_PROPOSE_VOCAB_SYSTEM_PROMPT`,
  `propose_theme_vocabulary` signature), `orchestrator.py`
  (`_load_predefined_themes`, `LessonPlan`, `_plan_with_theme_sequence`,
  `generate`), `lesson.py` (prompt + tolerance), `validator.py` (None tolerance).
- Keep `models.py` language-agnostic. Hints are English in the catalog only.

## Agent Notes

- Proposed 2026-06-20 (Erik): explore Option 1; put seed words / outline in
  English in `themes.yaml` since this is a multi-language generator.
- Done 2026-06-20 (Claude). Implemented:
  - `themes.yaml` per-lesson optional `seedWords` (English list) + `outline`
    (English string); documented in a header comment; populated A1 lessons 1–10
    as working examples (rest fall back gracefully to theme + goals).
  - `LessonThemePlan` gained `english_seed_words` + `outline`;
    `_load_predefined_themes` parses them; `LessonPlan` carries `outline`;
    `_plan_with_theme_sequence` passes seed words to the proposer (try/except for
    back-compat) and sets the outline; `generate()` forwards `outline` to the
    writer.
  - `propose_theme_vocabulary` takes `seed_words` → `anchor_concepts_english` in
    the payload + a system-prompt line so concrete nouns are proposed.
  - `LessonGenerator.generate(outline=...)`: outline goes into the prompt and
    forces narrative format even for tiny vocab; narrative instruction reworded to
    demand a single coherent story/dialogue; system prompt now permits other
    in-level words.
  - Validation leniency: `VocabularyValidator.validate(extra_tolerance=None)`
    tolerates every at/below-CEFR extra (only above-CEFR are violations); the CLI
    generate-lessons path constructs `LessonGenerator(..., extra_tolerance=None)`.
  - Also relocated lesson output to `courses/<lang>/lessons/<LEVEL>/` (was
    `<LEVEL>/lessons/`); export reader + existing local A1 lessons moved to match.
- Cannot demonstrate regenerated text offline (needs a live LLM); the user
  regenerates with `course generate-lessons`. Determinism preserved (caching).
- Follow-up if still loose: fill `seedWords`/`outline` for more lessons, or add a
  one-off pass that LLM-drafts outlines from theme+goals into `themes.yaml`.
- 2026-06-21 follow-up (after a real Ollama run still produced adverb-only
  lessons): root cause was that `seedWords` only nudged the **weak local
  proposer**, whose suggestions were too sparse after CEFR filtering, so selection
  fell back to frequency (A1 top = function words). Confirmed the cache was NOT at
  fault — it stores responses keyed by the prompt hash, so changed prompts already
  regenerate. Fixes:
  - `_resolve_seed_words` + `_gloss_primary_term` in `orchestrator.py`: map English
    `seedWords` directly to lexicon lemmas via each word's `translations.en`
    headword (deterministic, no LLM), and select them as the **highest-priority**
    words in `_plan_with_theme_sequence` (before proposer/frequency). Verified on
    the real `courses/nl` lexicon: lesson001 now = hallo, naam, buur, straat,
    welkom, goedemiddag…; lesson004 = moeder, vader, broer, zus, kind; lesson005 =
    huis, kamer, keuken, tuin, deur, raam.
  - `lesson.py`: strip the echoed `## Lesson Title` placeholder (`_clean_title`),
    reword the title placeholder to angle-bracket form, and tell the writer to
    conjugate verbs (the infinitive-only complaint).
- Remaining prose quality (coherence, conjugation) depends on the LLM; the local
  `gemma` model is the weak link. Selection + anchors are now deterministic and
  correct, so a stronger model (or OpenAI) should yield good text.
