"""Part-of-speech tagger abstraction and per-language registry.

A :class:`PosTagger` turns target-language text into ordered, char-offset token
annotations (surface, lemma, POS) plus the separable-verb particle links its
parser can recover. Language-specific taggers register by language code at import
time, mirroring the LLM provider factory and the lemmatizer registry.

This module is language-agnostic: it depends only on the canonical
:class:`~course_compiler.models.PartOfSpeech` enum, never on a concrete language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from course_compiler.models import Gender, PartOfSpeech


class PosTaggerError(RuntimeError):
    """Raised when no tagger is registered (or its backend is unavailable)."""


@dataclass(frozen=True)
class TokenTag:
    """One tagged surface token with its character span in the source text."""

    surface: str
    start: int
    end: int
    lemma: str
    #: Canonical POS, or ``None`` when the backend's tag does not map (e.g. PUNCT,
    #: SYM, or PROPN which we deliberately leave unlinked as a proper name).
    pos: PartOfSpeech | None = None
    #: Raw backend tag (e.g. spaCy UPOS) kept for debugging/diagnostics.
    upos: str = ""

    @property
    def is_word(self) -> bool:
        """True for alphabetic, linkable tokens (not punctuation/whitespace)."""
        return any(ch.isalpha() for ch in self.surface)


@dataclass(frozen=True)
class TaggedDoc:
    """Result of tagging a text: ordered tokens + separable-particle links."""

    tokens: list[TokenTag]
    #: ``(verb_index, particle_index)`` pairs the parser identified as a separable
    #: verb and its detached prefix (e.g. ``stelt … voor``). Empty when the backend
    #: has no dependency parse; the annotator then falls back to a dictionary scan.
    particle_links: list[tuple[int, int]] = field(default_factory=list)
    #: ``True`` when the backend ran a dependency parse, so ``particle_links`` is
    #: authoritative — the annotator must NOT invent separable particles by scanning
    #: (which would fuse a stray preposition). ``False`` (e.g. a parser-less tagger)
    #: enables the dictionary scan-ahead fallback.
    parsed: bool = False


class PosTagger(ABC):
    """Tag target-language text with POS, lemma, and separable-verb links."""

    @property
    @abstractmethod
    def language(self) -> str:
        """BCP-47 language code this tagger covers (e.g. ``"nl"``)."""

    @abstractmethod
    def tag(self, text: str) -> TaggedDoc:
        """Tokenize and tag *text*."""

    def article_for_gender(self, gender: Gender | str | None) -> str | None:
        """Map a grammatical gender to its display article, if the language has one.

        Language plugins override this (e.g. Dutch ``de``/``het``); the default is
        ``None`` so the core stays language-agnostic.
        """
        return None


PosTaggerFactory = Callable[[str], PosTagger]

_REGISTRY: dict[str, PosTaggerFactory] = {}


def register_tagger(language: str, factory: PosTaggerFactory) -> None:
    """Register *factory* under *language* (overwrites any prior registration)."""
    _REGISTRY[language] = factory


def create_tagger(language: str) -> PosTagger:
    """Instantiate the tagger registered for *language*.

    Raises :exc:`PosTaggerError` if no factory is registered or its backend
    (e.g. the spaCy model) is not installed.
    """
    factory = _REGISTRY.get(language)
    if factory is None:
        raise PosTaggerError(
            f"No POS tagger registered for language {language!r}. "
            "Install the optional 'nlp' extra and a model, e.g. "
            "`uv pip install -e '.[nlp]'` and `python -m spacy download nl_core_news_lg`."
        )
    return factory(language)


def is_registered(language: str) -> bool:
    return language in _REGISTRY
