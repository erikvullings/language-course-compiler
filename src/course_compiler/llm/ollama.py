"""Ollama LLM provider (local models via the Ollama HTTP API)."""

from __future__ import annotations

import httpx

from course_compiler.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    PromptInput,
    to_messages,
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


class OllamaProvider(LLMProvider):
    """Talks to a local (or remote) Ollama server's ``/api/chat`` endpoint.

    The httpx clients can be injected to make the provider trivially testable
    with :class:`httpx.MockTransport`; otherwise they are created lazily.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._client = client
        self._async_client = async_client

    def _payload(
        self,
        prompt: PromptInput,
        model: str | None,
        temperature: float | None,
        extra: dict[str, object],
    ) -> dict[str, object]:
        messages = [m.as_dict() for m in to_messages(prompt)]
        options: dict[str, object] = {
            "temperature": self.temperature if temperature is None else temperature
        }
        return {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": options,
            **extra,
        }

    @staticmethod
    def _parse(data: dict) -> LLMResponse:
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {data!r}") from exc
        return LLMResponse(content=content, model=data.get("model", ""), raw=data)

    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            resp = client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(prompt, model, temperature, kwargs),
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        finally:
            if self._client is None:
                client.close()

    async def acomplete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(prompt, model, temperature, kwargs),
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        finally:
            if self._async_client is None:
                await client.aclose()
