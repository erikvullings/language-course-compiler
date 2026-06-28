# 0023 POS-tagged, sense-linked lesson tokens

Status: in_progress
Priority: high
Owner: erik.vullings
Agent: claude
Area: generation / converters / cli
Depends on: 0019, 0021

## Context
The `zin` frontend mis-translates words because lessons persist only
`new_words: [lemma]` and `words.json` is keyed by lemma, discarding POS/sense. Root
cause in this repo: the Dutch importer dedupes words by lemma, dropping the second
POS of a homograph (e.g. `morgen` adverb → only "morning" survives). See the plan at
`~/.claude/plans/generating-lessons-audio-and-playful-cat.md`.

## Decisions
- spaCy as primary per-token tagger (optional `[nlp]` extra), constrained by the
  lesson's closed vocabulary; verb-form map as deterministic fallback.
- Lexicon keyed by `(lemma, POS)`; words get composite id `lemma|pos`.
- Batched, cached LLM fallback only for same-POS ambiguity.
- Embedded `tokens[]` + `vocabulary[]` in each lesson JSON.
- A separate, re-runnable `course annotate` command (works on existing lessons; can
  re-tag a single manually edited lesson). NOT welded into generation.

## Acceptance Criteria
- [x] `Word`/`Verb` carry `glosses: list[str]`; word id is `lemma|pos`; homograph POS survive import.
- [x] `separable-verbs.json` emitted by the Dutch importer.
- [x] `nlp/` package: `PosTagger` ABC + registry + spaCy nl plugin (guarded import).
- [x] Pure `TokenAnnotator` resolves per-token (ref, pos, gloss), fuses separable verbs.
- [x] Batched cached LLM sense fallback for same-POS ambiguity.
- [x] `Lesson` schema: `tokens[]` + `vocabulary[]`.
- [x] `course annotate --lang --cefr [--only]` rewrites lesson JSON in place; idempotent.
- [x] Tests pass offline (fake tagger + mocked LLM); `ruff` clean (new files).
- [x] Item 7: optional `lessonNNN.meta.yaml` override sidecar (`linkAs`/`glossOverrides`/
      `separableVerbs`) via `LessonOverrides`, applied in `annotate()` and loaded by the CLI.
- [x] Item 8: frontend handoff written to `docs/frontend-integration.md` (the `zin`
      coding agent implements the consuming side in its own repo).
- [ ] Bonus (optional): register the spaCy lemmatizer in `generation/base.py` to upgrade
      the validator's lemmatization — not done; out of scope for this task.

## Agent Notes
- Converter (`converters/dutch.py`): dedup is now by composite word id `lemma|pos`
  (`word_from_kaikki`); homograph POS survive. Added `_gloss_fragments`/`_english_glosses`
  (clean candidate list; `translations.en` display string unchanged) + `Word.glosses`/
  `Verb.glosses`. Added `detect_separable`/`annotate_separable_verbs` → writes
  `separable-verbs.json`; verbs gain `separable`/`prefix`/`reflexive`. `audio.json` stays
  lemma-keyed. Per-entry word YAML filename is now `<lemma>.<pos>.yaml`.
- `nlp/` package: `PosTagger` ABC + registry (`base.py`); `spacy_nl.py` registers an `nl`
  factory that imports spaCy lazily (raises `PosTaggerError` if extra/model missing) and
  maps UPOS→PartOfSpeech, reads `compound:prt` for particle links, provides de/het articles.
  Optional `[nlp]` extra in pyproject (`spacy>=3.7` + `python -m spacy download nl_core_news_lg`).
- `generation/annotate.py` (pure): `LessonVocab` + `build_lesson_vocab`; `annotate()` resolves
  tokens (spaCy first, verb-form map fallback, separable stem+particle override via
  `separable_by_stem`), emits `LessonToken|str` stream; `build_vocabulary()` builds
  `LessonWord[]`. `SenseQuery`/`SensePicker` hook for same-POS ambiguity.
- `generation/sense.py`: `make_llm_sense_picker` — one cached, fail-open batched call per lesson.
- `cli.py`: `course annotate` command (mirrors `generate-audio`); rebuilds cumulative allowed
  vocab from prior lessons' `new_words`, rewrites each lesson JSON via the `Lesson` model.
- Tests: `test_annotate.py`, `test_sense.py`, `test_cli_annotate.py`, plus new cases in
  `test_dutch_converter.py`/`test_cli_import.py`. 45 task tests pass.
- Pre-existing (not mine): `test_lesson_generator.py::test_user_prompt_requests_title_text_format_and_grammar_check`
  fails (prompt no longer contains "exactly 2 to 6 words"); repo also has pre-existing ruff
  errors in untouched files.
- To get full benefit on real data: re-run `course import` (regenerates `words.json` with
  composite ids + glosses + `separable-verbs.json`), then `course annotate`.
- Fix (post-review, lesson003 "zijn ... aan" → bogus "aanzijn/visit"): (1) `_verb_infinitive`
  no longer coerces a token into a verb when spaCy confidently tags it a closed-class
  non-verb (DET/PRON/ADP/CCONJ/ART/NUM/INTJ) — a possessive `zijn` (DET) stays a determiner
  rather than the verb `zijn`; (2) added `TaggedDoc.parsed` — when the backend ran a parse,
  separable `particle_links` are authoritative and the dictionary scan-ahead is skipped, so a
  stray preposition (`aan de tafel`) is never fused into a separable verb. Scan-ahead remains
  for parser-less taggers. Regression tests added in `test_annotate.py`.
