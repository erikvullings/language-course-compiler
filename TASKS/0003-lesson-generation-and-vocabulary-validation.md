# 0003 Lesson generation and vocabulary validation

Status: open
Priority: high
Owner: unassigned
Agent: unassigned
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
- Not started.
