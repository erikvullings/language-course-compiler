# 0004 Grammar, examples and exercise generation

Status: in_progress
Priority: medium
Owner: erik.vullings
Agent: copilot
Area: generation
Depends on: 0003

## Context
Generate grammar lessons (with a dependency graph for progression), example
sentences (beginner/intermediate/advanced, translated to interface languages),
and exercises referencing lesson/vocabulary IDs.

## Acceptance Criteria
- Grammar dependency graph with cycle detection
- Example generation constrained to introduced vocabulary
- Exercise generators for the types listed in `INITIAL_INSTRUCTIONS.md`
- Generated grammar/examples are validated before acceptance

## Implementation Notes
- Exercises reference ids, never duplicate data.

## Agent Notes
- Implemented grammar dependency planning in `src/course_compiler/generation/grammar.py`:
	- Added `GrammarTopic` + `GrammarProgressionPlanner.plan()` with deterministic topological ordering.
	- Added explicit validation for duplicate topic IDs, unknown dependencies, and cycle detection (`GrammarDependencyError`).
- Implemented exercise generation scaffolding in `src/course_compiler/generation/exercises.py`:
	- Added `ExerciseType` enum covering all exercise types listed in `INITIAL_INSTRUCTIONS.md`.
	- Added deterministic `ExerciseGenerator` producing ID-referenced `ExerciseSpec` objects only (no duplicated lesson/word payload data).
- Implemented constrained example generation in `src/course_compiler/generation/examples.py`:
	- Added multilingual `ExampleGenerator` with strict target-language vocabulary validation via `VocabularyValidator`.
	- Added retry-with-feedback behavior on vocabulary leakage and parse checks for required language lines.
- Added tests:
	- `tests/test_grammar_progression.py`
	- `tests/test_exercise_generator.py`
	- `tests/test_example_generator.py`
- Validation state:
	- Targeted tests for each new module pass.
	- Full suite passes (`106 passed`).
- Grammar content generation layer implemented:
	- Added `Grammar` model in `src/course_compiler/models.py` (language-agnostic;
	  English explanation prose + target-language `examples`, with `fallback`/`violations`).
	- Added per-language grammar catalog: `load_grammar_catalog()` + `GrammarPlan`
	  in `src/course_compiler/generation/grammar.py` (CEFR-keyed YAML like
	  `themes.yaml`; cross-level deps validated and ordered via the existing
	  `GrammarProgressionPlanner`; only the requested level's pages are returned).
	- Added `GrammarWriter` in `src/course_compiler/generation/grammar_writer.py`:
	  LLM writes explanation in English (not vocab-checked) and target-language
	  examples validated against the allowed set with the existing
	  `VocabularyValidator`; retry + fail-open + first-attempt caching mirror
	  `LessonGenerator`.
	- Wired two CLI subcommands in `src/course_compiler/cli.py`: `plan-grammar`
	  (LLM bootstraps a curated-then-committed catalog) and `generate-grammar`
	  (pegs each topic to a lesson via `introducedInLesson`, drawing on the lesson
	  plan's accumulated vocabulary plus all lower-level words).
	- Committed a curated `grammar/nl.yaml` starter (Dutch A1, ~23 topics,
	  front-loaded into lessons 1-18; later lessons review rather than introduce).
	- Signal words: `Grammar.signal_words` (target-language cue words like
	  *gisteren*/*morgen*); the writer emits and validates them with the examples.
	- Review mapping at export: `indices.json` carries `grammarByLesson`
	  (each lesson -> newly-introduced + cumulatively-available grammar, so the ~66
	  vocabulary-only lessons map to grammar review) and `commonVerbsByLevel`
	  (frequency-ranked verb ids per level; conjugation tables stay in `verbs.json`,
	  referenced by id, not duplicated).
	- Tests: `tests/test_grammar_catalog.py`, `tests/test_grammar_writer.py`,
	  `tests/test_cli_generate_grammar.py`. Full suite passes (`226 passed`).

Design notes (confirmed with the user):
- Grammar is **decoupled from lesson theme** (its own dependency order), pegged to
  a lesson index only so examples reuse vocabulary the learner has seen.
- Per-language progression lives in the **catalog data file**, never in code.
- Explanations are in the learner's **L1/English**; only examples are vocab-checked.

Remaining:
- Cross-level grammar at A2+ currently folds in all lower-level words as a baseline
  allowed set (since lesson `plan()` is within-level); revisit if lesson planning
  later accumulates across levels.
- Grammar pages already flow through `course export` → `grammar.json` (no change needed).
