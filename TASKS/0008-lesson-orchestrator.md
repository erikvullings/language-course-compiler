# 0008 Lesson orchestrator

Status: done
Priority: high
Owner: erik.vullings
Agent: claude
Area: generation
Depends on: 0003

## Context
The lesson generator (task 0003) produces content given a word list, but nothing
yet decides *which* words go into which lesson, in what order, and around what
theme. The orchestrator fills that gap: it reads the imported lexicon, filters by
CEFR level, groups words into thematic clusters, sequences lessons, and feeds each
batch to `LessonGenerator`.

## Acceptance Criteria
- Given an imported lexicon and a target CEFR level (e.g. `"A1"`), produce an
  ordered list of lessons, each with `new_words` and `allowed_words` populated
- Words within a lesson share a semantic theme (e.g. "home", "food", "transport")
- Words are ordered by frequency (most frequent first) within a theme
- The accumulated `allowed_words` set grows correctly across lessons
- `function_lemmas` (POS ∈ {article, conjunction, preposition, pronoun,
  determiner}) is derived automatically from the lexicon and passed to
  `LessonGenerator`
- CEFR level is a CLI argument (`--cefr A1`); one run produces one course
- Deterministic: same lexicon + same CEFR → same lesson sequence

## Implementation Notes
- Theme assignment: prefer data-driven (NT2Lex or a domain/topic tag on the
  imported `Word`) over LLM clustering; fall back to LLM clustering only if no
  tag is available, and cache the result
- Words with no CEFR tag should be excluded (or placed at the end of the
  sequence as a configurable option)
- The orchestrator should be language-agnostic: it operates on `Word` objects
  and the pluggable `Lemmatizer` / `LessonGenerator`; language identity flows
  in as a parameter
- New words per lesson: configurable (default 10), passed as a setting
- Hook into `cli.py` under `course generate-lessons --cefr A1 --lang nl`

## Agent Notes
- Implemented in `src/course_compiler/generation/`:
  - `themes.py` — `ThemeAssigner` Protocol + `LLMThemeAssigner` (JSON clustering via LLM,
    markdown-fence stripping, disk-cached by sorted lemma list)
  - `orchestrator.py` — `LessonPlan` dataclass + `LessonOrchestrator.plan()` (filter CEFR,
    split function/content words, theme, sort by frequency, slice per lesson, accumulate
    `allowed_lemmas`) and `generate()` (calls `LessonGenerator` with `function_lemmas` wired)
  - `validator.py` extended with `extra_function_lemmas` per-call override so orchestrator
    can pass lexicon-derived function lemmas without rebuilding the generator
  - `lesson.py` extended with `function_lemmas` parameter on `generate()`
- CLI: `course generate-lessons --lexicon <dir> --cefr A1 --lang nl [--words-per-lesson N]`
  reads `words.json`, runs orchestrator, writes `lessons/lessonNNN.txt`
- 13 new tests (4 theme assigner + 9 orchestrator); 79 total passing
