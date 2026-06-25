"""AUTO-GENERATED Voxtral OpenAPI client.

Generated from: http://localhost:8001/openapi.json
OpenAPI title: Voxtral TTS Translation Layer
OpenAPI version: 1.0.0

Do not hand-edit this file; re-generate from the running service spec.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel


class VoxtralError(RuntimeError):
    """Raised when Voxtral API calls fail."""


class ValidationError(BaseModel):
    loc: list[Any]
    msg: str
    type: str
    input: Any = None
    ctx: dict[str, Any] = None


class HTTPValidationError(BaseModel):
    detail: list[ValidationError] = None


class OpenAISpeechRequest(BaseModel):
    model: str = "voxtral"
    input: str
    voice: str = "nl_female"
    language: str = "nl"
    response_format: str = "mp3"
    speed: float = 1.0


class VoxtralExtendedRequest(BaseModel):
    text: str
    voice_reference_path: str | None = "nl_female"
    language: str = "nl"
    emotion: str = "neutral"
    nfe_steps: int = 16
    temperature: float = 0.7
    output_filename: str = "output.mp3"


class TranscriptWord(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSentence(BaseModel):
    id: str
    text: str
    start: float
    end: float
    words: list[TranscriptWord]


class TranscriptDocument(BaseModel):
    lesson_id: str
    sentences: list[TranscriptSentence]


class VoxtralTranscriptRequest(BaseModel):
    audio_path: str
    text: str
    language: str | None = "nl"
    lesson_id: str | None = None
    transcript_filename: str | None = None
    alignment_model_size: str = "small"
    beam_size: int = 5


class VoxtralTranscriptResponse(BaseModel):
    lesson_id: str
    sentences: list[TranscriptSentence]
    audio_path: str
    transcript_path: str


class VoxtralClient:
    """Client for Voxtral TTS + transcript endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_base_url = self._normalize_base_url(base_url)
        self._client = client or httpx.Client(
            base_url=normalized_base_url, timeout=timeout
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VoxtralClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def openai_compatible_speech_v1_audio_speech_post(
        self,
        request: OpenAISpeechRequest,
    ) -> bytes:
        """POST /v1/audio/speech"""
        response = self._client.post(
            "/v1/audio/speech",
            json=request.model_dump(exclude_none=True, mode="json"),
        )
        self._raise_for_status(response)
        return response.content

    def voxtral_dedicated_speech_v1_voxtral_speech_post(
        self,
        request: VoxtralExtendedRequest,
    ) -> bytes:
        """POST /v1/voxtral/speech"""
        response = self._client.post(
            "/v1/voxtral/speech",
            json=request.model_dump(exclude_none=True, mode="json"),
        )
        self._raise_for_status(response)
        return response.content

    def voxtral_generate_transcript_v1_voxtral_transcript_post(
        self,
        request: VoxtralTranscriptRequest,
    ) -> VoxtralTranscriptResponse:
        """POST /v1/voxtral/transcript"""
        response = self._client.post(
            "/v1/voxtral/transcript",
            json=request.model_dump(exclude_none=True, mode="json"),
        )
        self._raise_for_status(response)
        return VoxtralTranscriptResponse.model_validate(response.json())

    def voxtral_get_transcript_v1_voxtral_transcript__lesson_id__get(
        self,
        lesson_id: str,
    ) -> TranscriptDocument:
        """GET /v1/voxtral/transcript/{lesson_id}"""
        response = self._client.get(f"/v1/voxtral/transcript/{lesson_id}")
        self._raise_for_status(response)
        return TranscriptDocument.model_validate(response.json())

    # Convenience aliases used by the CLI.
    def synthesize_speech(self, request: OpenAISpeechRequest) -> bytes:
        return self.openai_compatible_speech_v1_audio_speech_post(request)

    def generate_transcript(
        self,
        request: VoxtralTranscriptRequest,
    ) -> VoxtralTranscriptResponse:
        return self.voxtral_generate_transcript_v1_voxtral_transcript_post(request)

    def get_transcript(self, lesson_id: str) -> TranscriptDocument:
        return self.voxtral_get_transcript_v1_voxtral_transcript__lesson_id__get(
            lesson_id
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = exc.response.text.strip()
            if len(details) > 1000:
                details = details[:1000] + "..."
            raise VoxtralError(
                f"Voxtral API request failed ({exc.response.status_code}): {details}"
            ) from exc

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        raw = base_url.strip()
        if not raw:
            raise ValueError("base_url must not be empty")
        if "://" not in raw:
            raw = "http://" + raw

        parsed = urlparse(raw)
        path = (parsed.path or "").rstrip("/")
        if path.endswith("/docs"):
            path = path[: -len("/docs")]
        elif path.endswith("/openapi.json"):
            path = path[: -len("/openapi.json")]

        clean = parsed._replace(path=path or "", params="", query="", fragment="")
        return clean.geturl().rstrip("/")
