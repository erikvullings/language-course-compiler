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
    #: Cleaned, de-duplicated English sense fragments (e.g. ["morning", "tomorrow"]).
    #: ``translations.en`` stays the joined display default; this list is the candidate
    #: set used for per-token sense selection.
    glosses: list[str] = []
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
    #: Cleaned English sense fragments (see :attr:`Word.glosses`).
    glosses: list[str] = []
    #: ``True`` for separable verbs (e.g. ``voorstellen`` → ``stelt … voor``).
    separable: bool = False
    #: Detached prefix for a separable verb (e.g. ``voor``), else ``None``.
    prefix: str | None = None
    #: ``True`` for reflexive verbs (used with ``zich``), e.g. ``zich voelen``.
    reflexive: bool = False
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


class LessonWord(_Model):
    """One vocabulary item introduced by a lesson, with its resolved sense.

    Built from the lexicon entry the compiler selected — POS, gender, gloss and a
    stable ``ref`` are all known here, so the frontend never has to guess.
    """

    lemma: str
    pos: str | None = None
    #: Stable lexicon key the token ``ref`` points at (``lemma|pos`` for words,
    #: the infinitive for verbs).
    ref: str | None = None
    gloss: str | None = None
    gender: str | None = None
    #: Display article for the learner's target language (e.g. ``de``/``het``); only
    #: set when the language plugin provides a gender→article mapping.
    article: str | None = None


class LessonToken(_Model):
    """A single linked word in a lesson's annotated token stream.

    The stream (``Lesson.tokens``) is a list of ``LessonToken | str``: plain strings
    are the inter-word gaps (whitespace/punctuation), objects are linkable words —
    mirroring the frontend's ``Token = string | {w, ref}`` model.
    """

    #: Surface form as it appears in the text.
    w: str
    #: Lexicon key this surface form resolves to, or ``None`` when unlinked
    #: (proper names, out-of-lexicon words).
    ref: str | None = None
    pos: str | None = None
    gloss: str | None = None
    #: Surface pieces of a fused separable verb (e.g. ``["stelt", "voor"]``); ``None``
    #: for ordinary single-word tokens.
    span: list[str] | None = None


class Lesson(_Model):
    id: str
    language: str
    cefr: str
    title: str
    theme: str = ""
    new_words: list[str] = []
    #: Resolved vocabulary for the lesson (lemma + POS + sense + ref).
    vocabulary: list[LessonWord] = []
    #: Annotated token stream of ``text`` (strings for gaps, objects for words).
    tokens: list[LessonToken | str] = []
    text: str
    attempts: int = 1
    tolerated: list[str] = []
    #: True when ``text`` is placeholder content written after the provider failed
    #: or validation could not be satisfied — lets ``--regenerate-fallbacks`` find
    #: these lessons without guessing from the title.
    fallback: bool = False
    #: Unresolved content words in a fallback lesson (above level / out of vocabulary).
    #: Empty unless ``fallback`` is a best-effort draft that still leaked vocabulary.
    violations: list[str] = []


class Grammar(_Model):
    """A grammar page: explanation prose plus target-language examples.

    Language-agnostic like every other model here — the per-language topic set
    and ordering live in a grammar catalog (data), never in this schema.
    ``description``/``rules``/``commonMistakes`` are written in the learner's
    interface language (L1); ``examples`` are target-language strings that are
    vocabulary-validated against what the learner has seen by
    ``introducedInLesson``.
    """

    id: str
    language: str
    cefr: str
    title: str
    description: str = ""
    rules: list[str] = []
    examples: list[str] = []
    #: Target-language cue words that signal this structure (e.g. *gisteren* for
    #: the past, *morgen* for the future). Validated like ``examples``.
    signal_words: list[str] = []
    common_mistakes: list[str] = []
    exceptions: list[str] = []
    related_grammar: list[str] = []
    introduced_in_lesson: int | None = None
    exercises: list[str] = []
    #: True when ``description``/``examples`` are best-effort after the provider
    #: failed or example validation could not be satisfied.
    fallback: bool = False
    #: Example words above level / outside allowed vocabulary in a fallback page.
    violations: list[str] = []


def to_yaml(model: BaseModel) -> str:
    """Serialize a model to deterministic YAML using camelCase keys.

    ``None`` fields are dropped; key order follows model definition order so the
    same input always yields byte-identical output.
    """

    data = model.model_dump(by_alias=True, exclude_none=True, mode="json")
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
