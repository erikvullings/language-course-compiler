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
- Remaining for full task completion:
	- Grammar lesson/explanation generation content layer (currently progression planning only).
	- Integration into CLI/pipeline stages once task 0007 wiring is in scope.
