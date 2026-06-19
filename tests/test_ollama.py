"""Ollama provider behavior, with the HTTP layer stubbed."""

from __future__ import annotations

import httpx
import pytest

from course_compiler.llm import LLMError, Message, OllamaProvider, Role

from .conftest import json_response


def test_complete_returns_message_content(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = httpx.Request("POST", request.url, content=request.content).read()
        assert b"loop" in body  # prompt is forwarded
        return json_response(
            {"model": "llama3", "message": {"role": "assistant", "content": "hoi"}}
        )

    clients = make_clients(handler)
    provider = OllamaProvider(model="llama3", **clients)

    result = provider.complete("translate: loop")

    assert result.content == "hoi"
    assert result.model == "llama3"


def test_complete_sends_chat_payload(make_clients):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return json_response({"model": "llama3", "message": {"content": "x"}})

    provider = OllamaProvider(model="llama3", temperature=0.1, **make_clients(handler))
    provider.complete([Message(Role.SYSTEM, "ctx"), Message(Role.USER, "hi")], temperature=0.9)

    assert captured["body"]["model"] == "llama3"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.9
    assert captured["body"]["messages"][0] == {"role": "system", "content": "ctx"}


async def test_acomplete_returns_content(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"model": "llama3", "message": {"content": "async hoi"}})

    provider = OllamaProvider(**make_clients(handler))
    result = await provider.acomplete("hi")

    assert result.content == "async hoi"


def test_http_error_is_wrapped(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "boom"}, status_code=500)

    provider = OllamaProvider(**make_clients(handler))
    with pytest.raises(LLMError):
        provider.complete("hi")


def test_malformed_response_raises_llm_error(make_clients):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"unexpected": True})

    provider = OllamaProvider(**make_clients(handler))
    with pytest.raises(LLMError):
        provider.complete("hi")
