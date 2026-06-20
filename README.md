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

Implemented so far:

- uv / ruff / pytest project setup
- `.env`-based settings (`course_compiler.settings`)
- Provider-agnostic LLM module (`course_compiler.llm`) — Ollama and OpenAI, sync + async
- Canonical, language-agnostic lexicon schema (`course_compiler.models`)
- Dutch importer (`course_compiler.converters.dutch`) — kaikki.org, ODWN, wordfreq, NT2Lex
- Lesson generation pipeline (`course_compiler.generation`):
  - Pluggable `Lemmatizer` registry (mirrors the LLM factory pattern)
  - Disk-based LLM response cache for reproducible, offline-safe generation
  - `VocabularyValidator` — tokenize → lemmatize → reject unknown content words
  - `LessonGenerator` — LLM call + validation; text length scales with the
    accumulated (recombinant) vocabulary, lesson format adapts to stage (example
    sentences early, narrative once a base exists), and leakage triggers a
    minimal-edit retry of the prior draft rather than a fresh rewrite
  - `LLMThemeAssigner` — clusters vocabulary into themes and proposes
    theme-relevant seed words (generate-then-filter against the lexicon), cached
  - `LessonOrchestrator` — filters by CEFR, assigns themes, sequences lessons,
    accumulates allowed vocabulary, derives function-word exemptions from POS,
    and supports an optional front-loaded per-lesson word budget
- `course generate-images` — generates lesson cover images via a local Flux.1 API
- `course download-audio` — downloads MP3 pronunciation files from `audio.json`

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

## CLI reference

### Ask the LLM a question

```bash
course ask "Translate 'huis' to English."
```

### Import a lexicon (Dutch)

The canonical lexicon schema is defined as language-agnostic Pydantic models in
`course_compiler.models`. Language-specific importers in
`course_compiler.converters` map source datasets onto it.

The Dutch importer combines four open datasets (placed in `data/nl/`):

- **kaikki.org Dutch JSONL** (machine-readable Wiktionary) — part of speech,
  gender, plural/diminutive, IPA, syllables, verb conjugations, English glosses.
- **Open Dutch WordNet (LMF XML)** — synonyms (lemmas sharing a synset).
- **wordfreq cBpack** — frequency rank.
- **NT2Lex (`.tsv`)** — CEFR level (A1–C1), taken as the earliest level at which
  a lemma is attested in the CEFR-graded corpus.

```bash
course import \
  --kaikki    data/nl/kaikki.org-dictionary-Dutch.jsonl \
  --wordnet   data/nl/odwn_orbn_gwg-LMF_1.3.xml \
  --frequency data/nl/large_nl.msgpack \
  --nt2lex    data/nl/NT2Lex-extracted/NT2Lex-main/resource/NT2Lex-CGN+ODWN-v01.tsv \
  --out       courses/nl
```

This writes canonical YAML entries into `courses/nl/words/` and `courses/nl/verbs/`
(use `--limit N` for a quick smoke run). It also writes aggregate
`courses/nl/words.json` and `courses/nl/verbs.json` indexes for faster loading in
generation/export commands.

### Generate lessons

```bash
course generate-lessons --lang nl --cefr A1
```

`--lang` is the only required flag. Defaults: lexicon at `courses/<lang>`,
output at `<lexicon>/lessons`, language name derived from the lang code,
10 words per lesson. Override any of these explicitly:

```bash
course generate-lessons \
  --lang nl --cefr A1 \
  --lexicon courses/nl \
  --language-name Dutch \
  --words-per-lesson 10 \
  --out courses/nl/lessons
```

To **front-load** vocabulary (Delft-style), introduce more words in the first
lessons and taper to the steady-state count — useful because early lessons have
no prior vocabulary to recombine:

```bash
course generate-lessons --lang nl --cefr A1 \
  --first-lesson-words 40 --front-load-lessons 3 --words-per-lesson 10
```

This makes lesson 1 introduce 40 words, lesson 2 ~25, and lesson 3 onward 10.
Omit `--first-lesson-words` for a uniform budget. Independently, early lessons
(while little vocabulary has accumulated) are written as short example
sentences/dialogue rather than a narrative, which reads more naturally with a
sparse word set; the compiler switches to narrative once enough vocabulary
exists.

Preview the computed lesson blueprint first (count + theme + seed lemmas):

```bash
course generate-lessons --lang nl --cefr A1 --preview
```

To preview and then continue in one run:

```bash
course generate-lessons --lang nl --cefr A1 --preview --approve
```

To use a specific predefined theme catalog YAML:

```bash
course generate-lessons --lang nl --cefr A1 --themes-file themes.yaml
```

`--themes-file` first checks the provided path. For a bare filename (like
`themes.yaml`), it also falls back to the bundled catalog at
`src/course_compiler/generation/themes.yaml`. If no file is found, the command
fails with an explicit error.

Without `--themes-file`, `generate-lessons` auto-discovers in this order:

1. `themes.yaml` in the repository root
2. `themes.yaml` in the selected lexicon directory (for example `courses/nl/themes.yaml`)
3. bundled `src/course_compiler/generation/themes.yaml`

Note: the catalog controls lesson **theme names/order** and optional
**communicative goals**. Seed words are selected automatically: the LLM proposes
theme-relevant vocabulary, the compiler keeps only the words present in the CEFR
lexicon (frequency-ranked), and lesson text length scales with the accumulated
(recombinant) vocabulary so early lessons stay short and natural.

One lesson file per lesson is written to the output directory as
`lesson001.json`, `lesson002.json`, … Run once per CEFR level to build a
full A1 → B2 course. LLM responses (theme clustering and lesson text) are
cached in `courses/nl/.llm_cache/` so subsequent runs are fast and
byte-identical.

To regenerate lessons from scratch (ignoring cache):

```bash
course generate-lessons --lang nl --cefr A1 --no-cache
```

Or manually clear the cache:

```bash
rm -rf courses/nl/.llm_cache
```

`generate-lessons` prefers `words.json` when present (falling back to
`words/*.yaml`), so preview mode starts much faster on large lexicons.

### Generate lesson images

Generates a cover illustration for every lesson by posting to a locally running
[Flux.1](https://blackforestlabs.ai/) image API (Automatic1111-compatible,
default port 7860). Images are written to `courses/img/<LEVEL>/<LESSON>.png` and
are language-independent, so one image set covers all target languages.

#### Installing the image generation service

Open the model page on Hugging Face at [black-forest-labs](https://huggingface.co/black-forest-labs) and accept the conditions.

```bash
brew install python@3.11
export HF_TOKEN=<YOUR_TOKEN> # Your Token from Hugging Face, should have Read access to all models

huggingface-cli download black-forest-labs/FLUX.1-dev \
    --local-dir ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev

git clone https://github.com/voipnuggets/flux-generator.git
cd flux-generator
chmod +x run_flux.sh

# Run the setup script in local-only secure mode
./run_flux.sh
```

Now you can use it

```bash
course generate-images --model dev
```

Existing images are skipped. Use `--force` to regenerate them. Narrow the run
with `--level` and/or `--lesson`:

```bash
course generate-images --level A1 --lesson lesson001  --model dev --force
```

Override defaults:

```bash
course generate-images \
  --themes-file themes.yaml \
  --out courses/img \
  --api-url http://127.0.0.1:7860/sdapi/v1/txt2img \
  --width 1024 --height 576 --steps 4 --cfg-scale 4.0
```

Each image seed is derived deterministically from the level + lesson ID, so
re-running without `--force` produces the same images.

### Download audio files

Downloads MP3 pronunciation files listed in `courses/<lang>/audio.json` (a
`{ word: url }` map built by `course import`) and saves them locally as
`courses/<lang>/audio/<word>.mp3`. Spaces in word keys are replaced with
underscores; slashes with underscores.

```bash
course download-audio --lang nl
```

Existing files are skipped. Use `--force` to re-download. For a quick test:

```bash
course download-audio --lang nl --limit 100 --dry-run
```

Override defaults:

```bash
course download-audio \
  --lang nl \
  --audio-json courses/nl/audio.json \
  --out courses/nl/audio \
  --force
```

### Export split JSON bundles

```bash
course export --lang nl --course-dir courses/nl --out courses/nl/export
```

This writes:

- `manifest.json`
- `words.json`
- `verbs.json`
- `grammar.json`
- `exercises.json`
- `lessons/lesson001.json`, `lessons/lesson002.json`, ...

## Using the LLM module directly

```python
from course_compiler.llm import create_provider
from course_compiler.settings import Settings

provider = create_provider(Settings.load())   # picks Ollama or OpenAI from .env

print(provider.complete("Translate 'huis' to English.").content)        # sync
# result = await provider.acomplete("Translate 'huis' to English.")     # async
```

## License

MIT
