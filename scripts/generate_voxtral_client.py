"""Generate a standalone Voxtral client from the live OpenAPI spec.

Usage:
    uv run python scripts/generate_voxtral_client.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

OPENAPI_URL = "http://localhost:8001/openapi.json"
OUT = Path("src/course_compiler/audio/voxtral_client.py")


def _py_type(schema: dict) -> str:
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]

    any_of = schema.get("anyOf")
    if any_of:
        non_null = [s for s in any_of if s.get("type") != "null"]
        if len(non_null) == 1:
            return f"{_py_type(non_null[0])} | None"
        return "Any"

    type_name = schema.get("type")
    if type_name == "string":
        return "str"
    if type_name == "integer":
        return "int"
    if type_name == "number":
        return "float"
    if type_name == "boolean":
        return "bool"
    if type_name == "array":
        return f"list[{_py_type(schema.get('items', {}))}]"
    if type_name == "object":
        return "dict[str, Any]"
    return "Any"


def _render_model(name: str, schema: dict) -> str:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [f"class {name}(BaseModel):"]

    if not properties:
        lines.append("    pass")
        return "\n".join(lines)

    for prop, prop_schema in properties.items():
        annotation = _py_type(prop_schema)
        if prop in required:
            lines.append(f"    {prop}: {annotation}")
            continue

        default = prop_schema.get("default")
        if default is None:
            lines.append(f"    {prop}: {annotation} = None")
        else:
            lines.append(f"    {prop}: {annotation} = {default!r}")

    return "\n".join(lines)


def main() -> int:
    spec = json.load(urlopen(OPENAPI_URL))
    schemas = spec["components"]["schemas"]

    model_order = [
        "ValidationError",
        "HTTPValidationError",
        "OpenAISpeechRequest",
        "VoxtralExtendedRequest",
        "TranscriptWord",
        "TranscriptSentence",
        "TranscriptDocument",
        "VoxtralTranscriptRequest",
        "VoxtralTranscriptResponse",
    ]

    rendered_models = []
    for name in model_order:
        if name in schemas:
            rendered_models.append(_render_model(name, schemas[name]))

    code = f'''"""AUTO-GENERATED Voxtral OpenAPI client.

Generated from: {OPENAPI_URL}
OpenAPI title: {spec.get("info", {}).get("title", "")}
OpenAPI version: {spec.get("info", {}).get("version", "")}

Do not hand-edit this file; re-generate from the running service spec.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel


class VoxtralError(RuntimeError):
    """Raised when Voxtral API calls fail."""


{chr(10).join(rendered_models)}


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
        self._client = client or httpx.Client(base_url=normalized_base_url, timeout=timeout)
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
        """GET /v1/voxtral/transcript/{{lesson_id}}"""
        response = self._client.get(f"/v1/voxtral/transcript/{{lesson_id}}")
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
        return self.voxtral_get_transcript_v1_voxtral_transcript__lesson_id__get(lesson_id)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = exc.response.text.strip()
            if len(details) > 1000:
                details = details[:1000] + "..."
            raise VoxtralError(
                f"Voxtral API request failed ({{exc.response.status_code}}): {{details}}"
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
'''

    OUT.write_text(code, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
