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
- **`settings.py`** — `Settings.load(env=...)` reads config via python-dotenv.
- **`cli.py`** — `course` entry point (skeleton; subcommands grow per `TASKS/`).

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
