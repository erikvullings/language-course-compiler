"""Abstract Lemmatizer interface and language registry.

Language-specific lemmatizers are registered by language code at import time,
mirroring the LLM provider factory pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class LemmatizerError(RuntimeError):
    """Raised when no lemmatizer is registered for the requested language."""


class Lemmatizer(ABC):
    """Map surface forms to their dictionary lemma.

    Returns ``None`` for tokens not found in the lemmatizer's vocabulary.
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """BCP-47 language code this lemmatizer covers (e.g. ``"nl"``)."""

    @abstractmethod
    def lemmatize(self, token: str) -> str | None:
        """Return the lemma for *token*, or ``None`` if unknown."""


LemmatizerFactory = Callable[[str], Lemmatizer]

_REGISTRY: dict[str, LemmatizerFactory] = {}


class _LowercaseLemmatizer(Lemmatizer):
    """Conservative fallback lemmatizer used when no NLP lemmatizer is configured.

    It lowercases non-empty tokens and returns them as-is.
    """

    def __init__(self, language: str) -> None:
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    def lemmatize(self, token: str) -> str | None:
        normalized = token.strip().lower()
        return normalized or None


def register_lemmatizer(language: str, factory: LemmatizerFactory) -> None:
    """Register *factory* under *language*.  Overwrites any prior registration."""
    _REGISTRY[language] = factory


def create_lemmatizer(language: str) -> Lemmatizer:
    """Instantiate the lemmatizer registered for *language*.

    Raises :exc:`LemmatizerError` if no factory has been registered.
    """
    factory = _REGISTRY.get(language)
    if factory is None:
        raise LemmatizerError(f"No lemmatizer registered for language: {language!r}")
    return factory(language)


# Register conservative defaults for currently targeted languages.
for _lang in ("nl", "de", "fr", "it", "es"):
    register_lemmatizer(_lang, lambda language: _LowercaseLemmatizer(language))
