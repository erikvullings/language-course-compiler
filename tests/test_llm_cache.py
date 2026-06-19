"""Tests for the disk-based LLM response cache."""

from __future__ import annotations

import json

from course_compiler.generation.cache import LLMCache
from course_compiler.llm.base import LLMResponse


def test_cache_miss_returns_none(tmp_path):
    cache = LLMCache(tmp_path)
    assert cache.get("model-x", [{"role": "user", "content": "hi"}]) is None


def test_cache_roundtrip(tmp_path):
    cache = LLMCache(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    response = LLMResponse(content="world", model="model-x", raw={})
    cache.put("model-x", messages, response)
    retrieved = cache.get("model-x", messages)
    assert retrieved is not None
    assert retrieved.content == "world"
    assert retrieved.model == "model-x"


def test_cache_key_includes_model(tmp_path):
    cache = LLMCache(tmp_path)
    messages = [{"role": "user", "content": "same prompt"}]
    r1 = LLMResponse(content="from model A", model="model-a", raw={})
    r2 = LLMResponse(content="from model B", model="model-b", raw={})
    cache.put("model-a", messages, r1)
    cache.put("model-b", messages, r2)
    assert cache.get("model-a", messages).content == "from model A"
    assert cache.get("model-b", messages).content == "from model B"


def test_cache_persists_to_disk(tmp_path):
    """A second LLMCache instance at the same path sees entries from the first."""
    messages = [{"role": "user", "content": "persist me"}]
    response = LLMResponse(content="persisted", model="m", raw={})
    LLMCache(tmp_path).put("m", messages, response)
    assert LLMCache(tmp_path).get("m", messages).content == "persisted"


def test_cache_hit_is_deterministic(tmp_path):
    cache = LLMCache(tmp_path)
    messages = [{"role": "user", "content": "deterministic"}]
    response = LLMResponse(content="same every time", model="m", raw={})
    cache.put("m", messages, response)
    r1 = cache.get("m", messages)
    r2 = cache.get("m", messages)
    assert r1.content == r2.content
