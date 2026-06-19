# 0004 Grammar, examples and exercise generation

Status: open
Priority: medium
Owner: unassigned
Agent: unassigned
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
- Not started.
