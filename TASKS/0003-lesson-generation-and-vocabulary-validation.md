# 0003 Lesson generation and vocabulary validation

Status: done
Priority: high
Owner: erik.vullings
Agent: claude
Area: generation
Depends on: 0002

## Context
Generate lessons with the LLM module, each introducing a configurable number of
new words and using ONLY allowed vocabulary (all previous words + current
lesson). Validate every generated lesson and regenerate on vocabulary leakage.

## Acceptance Criteria
- Lesson generator uses `course_compiler.llm`
- Validator: tokenize → lemmatize → compare lemmas against allowed vocabulary
- Lessons containing unknown vocabulary are rejected and regenerated
- Deterministic given a fixed seed / cached LLM responses

## Implementation Notes
- Lemmatization is language-specific data, not compiler logic — keep pluggable.
- Cache LLM responses for reproducibility and cheap tests.

## Agent Notes
- Implemented `src/course_compiler/generation/` with four modules:
  - `base.py` — abstract `Lemmatizer` + `_REGISTRY` / `register_lemmatizer` / `create_lemmatizer` (mirrors LLM factory)
  - `cache.py` — `LLMCache`: SHA-256-keyed JSON files for deterministic LLM responses
  - `validator.py` — `VocabularyValidator`: regex tokenizer → lemmatizer → set diff against allowed vocab
  - `lesson.py` — `LessonGenerator`: builds prompt, calls LLM through optional cache, validates, retries up to `max_retries`; raises `RuntimeError` on exhaustion
- 18 new tests in `tests/test_lemmatizer.py`, `test_llm_cache.py`, `test_vocabulary_validator.py`, `test_lesson_generator.py`; all 62 tests pass
- Design notes: validator maps `None`-lemma tokens to their lowercase form for reporting; cache is keyed on (model, messages) so same prompt across runs hits disk
