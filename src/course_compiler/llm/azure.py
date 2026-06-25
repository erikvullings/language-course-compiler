"""Azure OpenAI LLM provider."""

from __future__ import annotations

import httpx

from course_compiler.llm.base import PromptInput, LLMResponse
from course_compiler.llm.openai import OpenAIProvider

DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI chat completions.

    Azure differs from OpenAI in three ways:
    - auth uses ``api-key`` header instead of ``Authorization: Bearer``
    - URL is ``{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...``
    - ``model`` is the deployment name and is *not* sent in the request body
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = DEFAULT_API_VERSION,
        temperature: float = 0.7,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("AzureOpenAIProvider requires an api_key")
        if not endpoint:
            raise ValueError("AzureOpenAIProvider requires an endpoint")
        if not deployment:
            raise ValueError("AzureOpenAIProvider requires a deployment name")
        # Pass a dummy base_url; we override the URL in _chat_url.
        super().__init__(
            api_key=api_key,
            model=deployment,
            base_url=endpoint,
            temperature=temperature,
            timeout=timeout,
            client=client,
            async_client=async_client,
        )
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version

    @property
    def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key}

    @property
    def _chat_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

    def _payload(
        self,
        prompt: PromptInput,
        model: str | None,
        temperature: float | None,
        extra: dict[str, object],
    ) -> dict[str, object]:
        payload = super()._payload(prompt, model, temperature, extra)
        # Azure does not accept a ``model`` field in the body.
        payload.pop("model", None)
        return payload

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
                self._chat_url,
                headers=self._headers,
                json=self._payload(prompt, model, temperature, kwargs),
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as exc:
            from course_compiler.llm.base import LLMError
            raise LLMError(f"Azure OpenAI request failed: {exc}") from exc
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
                self._chat_url,
                headers=self._headers,
                json=self._payload(prompt, model, temperature, kwargs),
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as exc:
            from course_compiler.llm.base import LLMError
            raise LLMError(f"Azure OpenAI request failed: {exc}") from exc
        finally:
            if self._async_client is None:
                await client.aclose()
