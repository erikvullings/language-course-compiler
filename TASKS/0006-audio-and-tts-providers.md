# 0006 Audio generation and TTS providers

Status: open
Priority: low
Owner: unassigned
Agent: unassigned
Area: media
Depends on: 0002

## Context
Pluggable TTS providers (Azure, Google, OpenAI, Piper, Coqui). Generate audio
during compilation; store MP3/Opus paths only, never embed binary in JSON.

## Acceptance Criteria
- Pluggable `TTSProvider` interface mirroring the LLM provider registry
- `course generate-audio` fills missing audio
- Paths like `audio/<lang>/words/<id>.mp3`; no binary in JSON

## Implementation Notes
- Mirror the LLM `register_provider`/`create_provider` pattern.

## Agent Notes
- Not started.
