"""Audio tooling (TTS + transcript clients)."""

from course_compiler.audio.voxtral_client import (
    OpenAISpeechRequest,
    VoxtralClient,
    VoxtralError,
    VoxtralTranscriptRequest,
    VoxtralTranscriptResponse,
)

__all__ = [
    "OpenAISpeechRequest",
    "VoxtralClient",
    "VoxtralError",
    "VoxtralTranscriptRequest",
    "VoxtralTranscriptResponse",
]
