"""Construct an :class:`LLMProvider` from :class:`Settings`.

Providers register themselves here, so new providers can be added without
changing calling code -- in line with the pluggable-provider goal in
``INITIAL_INSTRUCTIONS.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from course_compiler.llm.base import LLMError, LLMProvider
from course_compiler.llm.ollama import OllamaProvider
from course_compiler.llm.openai import OpenAIProvider

if TYPE_CHECKING:
    from course_compiler.settings import Settings

ProviderFactory = Callable[["Settings"], LLMProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under ``name`` (case-insensitive)."""

    _REGISTRY[name.lower()] = factory


def create_provider(settings: Settings, name: str | None = None) -> LLMProvider:
    """Create the provider named by ``settings.llm_provider`` (or ``name``)."""

    key = (name or settings.llm_provider).lower()
    try:
        factory = _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise LLMError(
            f"Unknown LLM provider {key!r}. Known providers: {known}"
        ) from None
    return factory(settings)


def _build_ollama(settings: Settings) -> OllamaProvider:
    return OllamaProvider(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        thinking=settings.llm_thinking,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _build_openai(settings: Settings) -> OpenAIProvider:
    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


register_provider("ollama", _build_ollama)
register_provider("openai", _build_openai)
