# 0001 Project scaffold and LLM module

Status: done
Priority: high
Owner: erik.vullings
Agent: claude
Area: foundation
Depends on: none

## Context
Bootstrap the repository: a uv/ruff/pytest Python project with `.env`-based
settings and a provider-agnostic LLM module (sync + async) supporting Ollama and
OpenAI, designed to be easy to test.

## Acceptance Criteria
- [x] `uv venv` + `uv pip install -e ".[dev]"` works
- [x] `uv run pytest` green; `uv run ruff check .` clean
- [x] `course_compiler.llm` exposes a sync `complete` and async `acomplete`
- [x] Ollama and OpenAI providers, selectable via `.env`
- [x] Settings loaded via python-dotenv, testable with an explicit env dict
- [x] HTTP layer mockable (httpx clients injectable) — no network in tests

## Implementation Notes
- `src/course_compiler/llm/base.py` — `Message`, `Role`, `LLMResponse`,
  `LLMError`, `to_messages`, abstract `LLMProvider`.
- `ollama.py` / `openai.py` — httpx-based, absolute URLs from `base_url`, clients
  injectable for `httpx.MockTransport` testing.
- `factory.py` — `register_provider` / `create_provider` registry so new
  providers need no calling-code changes.
- `settings.py` — `Settings.load(env=...)`.

## Agent Notes
- Done. 23 tests passing. OpenAI provider speaks the OpenAI wire format over
  plain httpx, so it also works against compatible gateways via `OPENAI_BASE_URL`.
