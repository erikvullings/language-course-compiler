# 0005 Validation engine

Status: open
Priority: medium
Owner: unassigned
Agent: unassigned
Area: pipeline
Depends on: 0002

## Context
Cross-cutting validator detecting the issues listed in `INITIAL_INSTRUCTIONS.md`:
missing translations/IPA/audio/examples, duplicate lemmas/IDs, broken references,
invalid lesson order, grammar dependency cycles, unknown/unused vocabulary.

## Acceptance Criteria
- `course validate` runs all checks and reports actionable errors
- Each check is independently testable
- Non-zero exit on validation failure

## Agent Notes
- Not started.
