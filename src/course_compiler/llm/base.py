"""Provider-agnostic LLM interface and data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    """Conversation roles understood by chat-style LLM APIs."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """A single chat message."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class LLMResponse:
    """The result of a completion call.

    ``raw`` keeps the decoded provider payload so callers can access
    provider-specific fields (token usage, finish reason, ...) without the
    interface having to model every provider.
    """

    content: str
    model: str
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when a provider call fails (transport error or bad response)."""


PromptInput = str | Message | Sequence[Message]


def to_messages(prompt: PromptInput) -> list[Message]:
    """Normalize a prompt into a list of :class:`Message`.

    Accepts a bare string (treated as a single user message), a single
    ``Message``, or any sequence of messages.
    """

    if isinstance(prompt, str):
        return [Message(Role.USER, prompt)]
    if isinstance(prompt, Message):
        return [prompt]
    messages = list(prompt)
    if not messages:
        raise ValueError("prompt must contain at least one message")
    return messages


class LLMProvider(ABC):
    """Base class for synchronous + asynchronous LLM providers.

    Subclasses implement :meth:`complete` and :meth:`acomplete`. Both accept the
    same normalized inputs so calling code can switch between sync and async
    without changing how prompts are built.
    """

    @abstractmethod
    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        """Run a completion synchronously."""

    @abstractmethod
    async def acomplete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        """Run a completion asynchronously."""
