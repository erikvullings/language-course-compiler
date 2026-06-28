"""Pluggable POS taggers.

Importing this package registers the built-in taggers (currently the Dutch spaCy
plugin). Registration does not import the heavy backend — spaCy is loaded lazily
only when a tagger is instantiated via :func:`create_tagger`.
"""

from __future__ import annotations

# Register language plugins (factory only — no backend import at module load).
from course_compiler.nlp import spacy_nl  # noqa: E402,F401  (side-effect import)
from course_compiler.nlp.base import (
    PosTagger,
    PosTaggerError,
    TaggedDoc,
    TokenTag,
    create_tagger,
    is_registered,
    register_tagger,
)

__all__ = [
    "PosTagger",
    "PosTaggerError",
    "TaggedDoc",
    "TokenTag",
    "create_tagger",
    "is_registered",
    "register_tagger",
]
