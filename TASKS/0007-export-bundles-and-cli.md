# 0007 JSON export bundles and CLI commands

Status: open
Priority: medium
Owner: unassigned
Agent: unassigned
Area: export
Depends on: 0002, 0005

## Context
Export an optimized, indexed output for the SPA/PWA: a manifest plus multiple
JSON bundles (manifest.json, words.json, verbs.json, grammar.json,
lessons/lessonNNN.json, exercises.json). Wire up the full `course` CLI.

## Acceptance Criteria
- Objects indexed by id; no duplication; references by id only
- Manifest + per-bundle output; reproducible byte-for-byte
- CLI commands: build, validate, generate-*, import, export, stats

## Implementation Notes
- See JSON schema + "Future Architecture" in `INITIAL_INSTRUCTIONS.md`.
- CLI skeleton exists in `src/course_compiler/cli.py`.

## Agent Notes
- Not started.
