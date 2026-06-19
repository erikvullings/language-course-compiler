# Language Course Compiler

A language-agnostic course compiler that generates complete, reproducible
language-learning courses from open lexical resources, using LLMs for lesson,
grammar, example and exercise generation. The output feeds a free, open-source
language-learning app (SPA / PWA).

The compiler contains **no language-specific logic** — Dutch, German, French,
Italian and Spanish are just different configurations and input datasets. See
[`INITIAL_INSTRUCTIONS.md`](INITIAL_INSTRUCTIONS.md) for the full design and
[`TASKS/`](TASKS/) for the tracked roadmap.

## Status

Initial scaffold. Implemented so far:

- uv / ruff / pytest project setup
- `.env`-based settings (`course_compiler.settings`)
- A provider-agnostic LLM module (`course_compiler.llm`) with synchronous and
  asynchronous calls, and built-in **Ollama** and **OpenAI** providers.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv venv
uv sync
cp .env.example .env   # then edit
```

Alternative for editable installs:

```bash
uv pip install -e ".[dev]"
```

## Develop

```bash
uv run pytest                 # run the test suite
uv run pytest tests/test_ollama.py::test_complete_returns_message_content  # single test
uv run ruff check .           # lint
uv run ruff format .          # format
```

## Using the LLM module

```python
from course_compiler.llm import create_provider
from course_compiler.settings import Settings

provider = create_provider(Settings.load())   # picks Ollama or OpenAI from .env

print(provider.complete("Translate 'huis' to English.").content)        # sync
# result = await provider.acomplete("Translate 'huis' to English.")     # async
```

Or via the CLI:

```bash
course ask "Translate 'huis' to English."
```

## License

MIT
