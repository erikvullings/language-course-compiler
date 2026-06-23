# Environment Setup
- Local macOS capabilities and optimized CLI tools are mapped in `~/.config/ai/tools.md`. Read this file to use optimized search/replace and parsing binaries.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A language-agnostic **course compiler**: it generates complete, reproducible
language-learning courses (vocabulary, grammar, examples, exercises, audio) from
open lexical resources, using LLMs. Output feeds a separate free/open-source
SPA/PWA. The full design lives in `INITIAL_INSTRUCTIONS.md`; the implementation
roadmap is tracked as task files in `TASKS/`.

Current state: initial scaffold — settings + the LLM module are implemented; the
pipeline stages (import → generate → validate → export) are not yet built.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"   # one-time setup
uv run pytest                            # full test suite
uv run pytest tests/test_ollama.py       # one file
uv run pytest tests/test_factory.py::test_creates_ollama_by_default  # one test
uv run ruff check .                      # lint (also `ruff format .`)
uv run course ask "..."                  # exercise the configured LLM end-to-end
```

Python ≥ 3.11 (uses `enum.StrEnum`). Dependencies are intentionally minimal:
`httpx` + `python-dotenv` at runtime.

## Non-negotiable design constraints (from INITIAL_INSTRUCTIONS.md)

These shape every change — violating them is a bug:

- **No language-specific logic in the compiler.** Dutch/German/French/etc. are
  config + datasets, never code branches. Anything language-dependent
  (lemmatizers, grammar order, gender sets) is pluggable data/plugins.
- **Reproducible.** Same inputs → byte-identical output. Avoid nondeterminism;
  cache LLM/TTS responses so generation is repeatable and tests stay offline.
- **Pluggable providers/stages.** New importers, generators, exporters, and
  LLM/TTS providers are added by registration, not by editing calling code.
- **No binary in JSON.** Audio/images are referenced by path only.
- **Vocabulary discipline.** Generated lessons may use only allowed vocabulary
  (all prior words + current lesson) and are validated + regenerated on leakage.

## Architecture

`src/course_compiler/` (src-layout; package imports as `course_compiler`):

- **`llm/`** — provider-agnostic LLM access. `base.py` defines the data models
  (`Message`, `Role`, `LLMResponse`, `LLMError`, `to_messages`) and the abstract
  `LLMProvider` with both `complete` (sync) and `acomplete` (async). `ollama.py`
  and `openai.py` implement it; `factory.py` is a registry
  (`register_provider` / `create_provider`). This registry pattern is the
  template for the other pluggable providers (TTS, importers, exporters).
- **`models.py`** — the canonical, **language-agnostic** Pydantic schema (`Word`,
  `Verb`, `Frequency`, ...). Serializes camelCase via `to_yaml`. Conjugation
  tables are `{slot: form}` mappings, not fixed pronoun fields, so the schema
  isn't tied to one language. This module must never gain language-specific logic.
- **`converters/`** — **language-dependent** importers that map source datasets
  onto `models`. `dutch.py` maps kaikki.org Wiktionary JSONL (primary: pos,
  gender, plural/diminutive, IPA, syllables, verb conjugations, English glosses →
  `translations.en`) + Open Dutch WordNet XML (synonyms via shared synsets) +
  wordfreq frequency + NT2Lex `.tsv` (CEFR level = earliest attested level per
  lemma). Per-entry mappers (`word_from_kaikki`, `verb_from_kaikki`)
  are pure; `convert` streams files, `convert_iterables` is the I/O-free variant.
- **`generation/`** — the lesson pipeline (language-agnostic). `orchestrator.py`
  plans a CEFR level into a lesson sequence (filter → theme → select seed words →
  accumulate allowed vocabulary); `themes.py` proposes themes + vocabulary via
  LLM; `lesson.py` writes and validates lesson text with feedback-driven retry;
  `validator.py` enforces vocabulary discipline. Three design choices worth
  knowing: (1) requested text length scales with the **allowed** (recombinant)
  vocabulary, not the new-word count, so early lessons stay short and natural
  (`_target_length`); (2) seed words are **generate-then-filter** — the LLM
  proposes ~5n theme-relevant words (`propose_theme_vocabulary`) and the
  orchestrator keeps only lexicon hits, frequency-ranked, falling back to a
  candidate pool for coverage; (3) on vocabulary leakage the retry strategy is
  **violation-count-aware** (`revise_violation_threshold`): a near-miss draft (few
  violations) gets a **minimal revision** of the prior draft, while a heavily
  broken one — or the final attempt — is **restarted from the original prompt** and
  resampled at a higher temperature, so the model isn't anchored to wrong text.
  All LLM calls are
  cached for reproducibility. Two cold-start aids: the per-lesson word budget can
  be **front-loaded** (`first_lesson_words` tapers to `words_per_lesson` over
  `front_load_lessons`, via `_budget_for`) so early lessons have critical mass,
  and the lesson **format adapts to stage** — below `narrative_vocab_threshold`
  allowed words the prompt asks for short example sentences/dialogue instead of a
  narrative, and the writer prompt's **tense guidance relaxes** by CEFR level and
  once the allowed vocabulary passes `mature_vocab_threshold` (cold-start A1 is
  present-tense-first; A2 and mature A1 get only a gentle steer; B1+ none).
  Both are opt-in/config so output stays reproducible and
  language-agnostic. For coherence, the theme catalog (`themes.yaml`) may carry
  two **optional English** hints per lesson — `seedWords` (concrete anchors fed to
  `propose_theme_vocabulary` so early lessons get scene-grounding nouns) and
  `outline` (a brief scenario fed to the writer prompt, forcing a narrative); the
  generation path validates leniently (`extra_tolerance=None` → only above-CEFR
  words are violations, every in-level word is allowed) so the text reads
  naturally instead of degrading into over-constrained filler.
- **`frequency.py`** — reader for wordfreq `cBpack` files (generic).
- **`leveling.py`** — generic CEFR assignment by **cumulative frequency budget**
  (`assign_levels`): items fill levels most-frequent-first up to each level's
  budget increment; a per-item floor (e.g. a resource's attested level) is a
  minimum that lets an item roll forward when its level is full; items past the
  top budget are excluded. Budgets are config, not code. The Dutch converter calls
  this (`reassign_cefr_by_budget`) with each `(lemma, pos)` as a separate item.
- **`compounds.py`** — generic, language-pluggable compound splitter
  (`split_compound` / `is_derivable_compound`): a word splitting into ≥2 known
  lemmas (linkers passed in by the caller) is a **transparent** compound and is
  introduced without consuming budget; `opaque` words still count. The converter
  supplies Dutch linkers and levels transparent compounds to `max(level of parts)`.
- **`settings.py`** — `Settings.load(env=...)` reads config via python-dotenv.
- **`cli.py`** — `course` entry point (`ask`, `import`; grows per `TASKS/`).

The split between `models.py` (generic) and `converters/<lang>.py` (specific) is
the load-bearing boundary: anything that knows Dutch (pronouns, gender mapping,
form-tag quirks) belongs in the converter, never in the models or the pipeline.

Source datasets are not in git (large): `data/nl/` holds the kaikki JSONL, ODWN
XML and wordfreq msgpack used by the Dutch importer.

### Two testability patterns to follow when extending

1. **Inject the I/O boundary.** Providers accept optional `httpx.Client` /
   `httpx.AsyncClient`; tests pass clients backed by `httpx.MockTransport`
   (see `tests/conftest.py::make_clients`) so no network is touched. HTTP calls
   use absolute URLs built from `base_url`, so injected clients need no base URL.
2. **Inject config.** `Settings.load(env={...})` reads from an explicit dict, so
   tests never mutate the process environment or require a `.env`.

When adding a new provider: implement the interface, register it in its module
at import time (as `factory.py` registers ollama/openai), and add a `.env` key +
default to `settings.py` and `.env.example`.

## Workflow expectations

- Track non-trivial work in `TASKS/` (see the task-tracking skill). Set a task
  `in_progress` before starting and `done` when finished; the file is the
  resumable source of truth.
- Build with TDD: one behavior at a time, test → minimal code → repeat. Tests
  assert behavior through public interfaces (the `*Provider` / `Settings` APIs),
  not internals, so they survive refactors.
