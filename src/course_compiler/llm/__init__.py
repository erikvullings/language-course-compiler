"""LLM provider abstraction.

A small, provider-agnostic interface for calling large language models both
synchronously and asynchronously. Concrete providers (Ollama, OpenAI) live in
sibling modules and are constructed via :func:`create_provider`.
"""

from course_compiler.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    to_messages,
)
from course_compiler.llm.factory import create_provider, register_provider
from course_compiler.llm.ollama import OllamaProvider
from course_compiler.llm.openai import OpenAIProvider

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "OllamaProvider",
    "OpenAIProvider",
    "Role",
    "create_provider",
    "register_provider",
    "to_messages",
]
