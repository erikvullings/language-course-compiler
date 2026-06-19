"""OpenAI provider behavior, with the HTTP layer stubbed."""

from __future__ import annotations

import json

import httpx
import pytest

from course_compiler.llm import LLMError, OpenAIProvider

from .conftest import json_response


def _chat_response(content: str, model: str = "gpt-4o-mini") -> httpx.Response:
    return json_response(
        {"model": model, "choices": [{"message": {"role": "assistant", "content": content}}]}
    )


def test_complete_returns_choice_content(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return _chat_response("house")

    provider = OpenAIProvider(api_key="sk-test", **make_clients(handler))
    result = provider.complete("translate: huis")

    assert result.content == "house"
    assert result.model == "gpt-4o-mini"


def test_authorization_header_is_sent(make_clients):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return _chat_response("ok")

    provider = OpenAIProvider(api_key="sk-secret", model="gpt-4o-mini", **make_clients(handler))
    provider.complete("hi", model="gpt-4o")

    assert captured["auth"] == "Bearer sk-secret"
    assert captured["body"]["model"] == "gpt-4o"  # per-call override wins


async def test_acomplete_returns_content(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response("async house")

    provider = OpenAIProvider(api_key="sk-test", **make_clients(handler))
    result = await provider.acomplete("hi")

    assert result.content == "async house"


def test_missing_api_key_is_rejected():
    with pytest.raises(ValueError):
        OpenAIProvider(api_key="")


def test_malformed_response_raises_llm_error(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"choices": []})

    provider = OpenAIProvider(api_key="sk-test", **make_clients(handler))
    with pytest.raises(LLMError):
        provider.complete("hi")
