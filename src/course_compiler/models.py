"""Canonical, language-agnostic lexicon models (Pydantic).

These models define the internal/serialized schema of a compiled course. They
contain **no language-specific logic** -- any language's importer (see
``course_compiler.converters``) maps its source data onto these shapes.

Fields serialize as ``camelCase`` (e.g. ``partOfSpeech``) to match the YAML
schema in ``INITIAL_INSTRUCTIONS.md``; use :func:`to_yaml` / ``model_dump`` with
``by_alias=True``. Conjugation tables are modelled as ``{slot: form}`` mappings
rather than fixed language-specific pronoun fields so the schema stays generic
(the Dutch importer fills ``ik``/``jij``/... keys, another language fills its own).
"""

from __future__ import annotations

from enum import StrEnum

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Gender(StrEnum):
    """Grammatical gender. ``C`` = common (de-word), ``U`` = unspecified."""

    MASCULINE = "m"
    FEMININE = "f"
    NEUTER = "n"
    COMMON = "c"
    UNSPECIFIED = "u"


class PartOfSpeech(StrEnum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    ARTICLE = "article"
    NUMERAL = "numeral"
    INTERJECTION = "interjection"
    DETERMINER = "determiner"
    OTHER = "other"


class _Model(BaseModel):
    """Base: camelCase aliases on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Frequency(_Model):
    rank: int | None = None
    occurrences: int | None = None
    zipf: float | None = None
    source: str | None = None


class Plural(_Model):
    regular: str | None = None
    alternatives: list[str] = []


class Diminutive(_Model):
    regular: str | None = None
    alternatives: list[str] = []


class Audio(_Model):
    generated: str | None = None
    recorded: str | None = None


class VerbAudio(_Model):
    word: str | None = None


class Example(_Model):
    id: str
    # language code -> sentence, e.g. {"nl": "...", "en": "..."}
    sentences: dict[str, str] = {}
    audio: str | None = None


class Word(_Model):
    id: str
    language: str
    lemma: str
    normalized: str
    part_of_speech: PartOfSpeech
    translations: dict[str, str] = {}
    gender: Gender | None = None
    plural: Plural | None = None
    diminutive: Diminutive | None = None
    ipa: str | None = None
    syllables: list[str] = []
    stress: int | None = None
    frequency: Frequency | None = None
    cefr: str | None = None
    audio: Audio | None = None
    examples: list[Example] = []
    related: list[str] = []
    synonyms: list[str] = []
    antonyms: list[str] = []
    tags: list[str] = []
    introduced_in_lesson: int | None = None
    review_weight: float | None = None


class Verb(_Model):
    id: str
    language: str
    lemma: str
    infinitive: str
    translations: dict[str, str] = {}
    auxiliary: str | None = None
    # Tense tables keyed by slot (pronoun / "singular" / "participle" / ...).
    present: dict[str, str] = {}
    past: dict[str, str] = {}
    perfect: dict[str, str] = {}
    imperative: dict[str, str] = {}
    future: dict[str, str] = {}
    conditional: dict[str, str] = {}
    subjunctive: dict[str, str] = {}
    irregular: bool = False
    frequency: Frequency | None = None
    cefr: str | None = None
    audio: VerbAudio | None = None
    tags: list[str] = []


class Lesson(_Model):
    id: str
    language: str
    cefr: str
    title: str
    theme: str = ""
    new_words: list[str] = []
    text: str
    attempts: int = 1
    tolerated: list[str] = []
    #: True when ``text`` is placeholder content written after the provider failed
    #: or validation could not be satisfied — lets ``--regenerate-fallbacks`` find
    #: these lessons without guessing from the title.
    fallback: bool = False


def to_yaml(model: BaseModel) -> str:
    """Serialize a model to deterministic YAML using camelCase keys.

    ``None`` fields are dropped; key order follows model definition order so the
    same input always yields byte-identical output.
    """

    data = model.model_dump(by_alias=True, exclude_none=True, mode="json")
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
