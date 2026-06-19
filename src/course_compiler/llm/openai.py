"""OpenAI LLM provider (OpenAI-compatible chat completions API)."""

from __future__ import annotations

import httpx

from course_compiler.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    PromptInput,
    to_messages,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    """Calls the ``/chat/completions`` endpoint of any OpenAI-compatible API.

    Because it speaks the OpenAI wire format over plain httpx, it also works
    against compatible gateways (Azure OpenAI proxies, LiteLLM, vLLM, ...) by
    pointing ``base_url`` elsewhere. httpx clients can be injected for testing.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIProvider requires an api_key")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._client = client
        self._async_client = async_client

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _payload(
        self,
        prompt: PromptInput,
        model: str | None,
        temperature: float | None,
        extra: dict[str, object],
    ) -> dict[str, object]:
        return {
            "model": model or self.model,
            "messages": [m.as_dict() for m in to_messages(prompt)],
            "temperature": self.temperature if temperature is None else temperature,
            **extra,
        }

    @staticmethod
    def _parse(data: dict) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response: {data!r}") from exc
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
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(prompt, model, temperature, kwargs),
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
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
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(prompt, model, temperature, kwargs),
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        finally:
            if self._async_client is None:
                await client.aclose()
