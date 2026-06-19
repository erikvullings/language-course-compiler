"""The factory builds the provider selected by settings."""

from __future__ import annotations

import pytest

from course_compiler.llm import (
    LLMError,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
)
from course_compiler.settings import Settings


def test_creates_ollama_by_default():
    provider = create_provider(Settings.load(env={}))
    assert isinstance(provider, OllamaProvider)


def test_creates_openai_when_selected():
    settings = Settings.load(env={"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-1"})
    provider = create_provider(settings)
    assert isinstance(provider, OpenAIProvider)


def test_explicit_name_overrides_settings():
    settings = Settings.load(env={"OPENAI_API_KEY": "sk-1"})  # provider defaults to ollama
    provider = create_provider(settings, name="openai")
    assert isinstance(provider, OpenAIProvider)


def test_unknown_provider_raises():
    with pytest.raises(LLMError):
        create_provider(Settings.load(env={"LLM_PROVIDER": "nope"}))


def test_settings_flow_through_to_provider():
    settings = Settings.load(
        env={"OLLAMA_MODEL": "mistral", "OLLAMA_BASE_URL": "http://host:1234", "LLM_TIMEOUT": "5"}
    )
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "mistral"
    assert provider.base_url == "http://host:1234"
    assert provider.timeout == 5.0
