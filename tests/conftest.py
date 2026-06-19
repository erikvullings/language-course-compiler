"""Shared test helpers for stubbing httpx-based LLM providers."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest


@pytest.fixture
def make_clients() -> Callable[[Callable[[httpx.Request], httpx.Response]], dict]:
    """Return a factory producing sync+async httpx clients backed by a handler.

    Usage::

        clients = make_clients(handler)
        OllamaProvider(client=clients["client"], async_client=clients["async_client"])

    The handler receives each request (so tests can assert on the body) and
    returns the response to feed back.
    """

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> dict:
        return {
            "client": httpx.Client(transport=httpx.MockTransport(handler)),
            "async_client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        }

    return factory


def json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=json.dumps(payload))
