"""spaCy-backed Dutch POS tagger (optional ``[nlp]`` extra).

Importing this module only *registers* the factory — spaCy itself is imported
lazily when a tagger is first instantiated, so the package import never fails when
the optional dependency is absent. The factory raises a clear
:class:`~course_compiler.nlp.base.PosTaggerError` if spaCy or its model is missing.
"""

from __future__ import annotations

import os

from course_compiler.models import Gender, PartOfSpeech
from course_compiler.nlp.base import (
    PosTagger,
    PosTaggerError,
    TaggedDoc,
    TokenTag,
    register_tagger,
)

#: Preferred Dutch model; override with ``COURSE_SPACY_MODEL``. The lg model gives
#: the best tagging/parse quality (we don't ship it — install separately).
_DEFAULT_MODEL = "nl_core_news_lg"

# spaCy UPOS -> canonical PartOfSpeech. PROPN maps to None so proper names stay
# unlinked; PUNCT/SYM/SPACE/X are absent (treated as gaps).
_UPOS_MAP: dict[str, PartOfSpeech] = {
    "NOUN": PartOfSpeech.NOUN,
    "VERB": PartOfSpeech.VERB,
    "AUX": PartOfSpeech.VERB,
    "ADJ": PartOfSpeech.ADJECTIVE,
    "ADV": PartOfSpeech.ADVERB,
    "PRON": PartOfSpeech.PRONOUN,
    "DET": PartOfSpeech.DETERMINER,
    "ADP": PartOfSpeech.PREPOSITION,
    "CCONJ": PartOfSpeech.CONJUNCTION,
    "SCONJ": PartOfSpeech.CONJUNCTION,
    "NUM": PartOfSpeech.NUMERAL,
    "INTJ": PartOfSpeech.INTERJECTION,
}

# Dutch articles by gender: de-words (m/f/c) vs het-words (n).
_ARTICLE_BY_GENDER = {
    Gender.MASCULINE: "de",
    Gender.FEMININE: "de",
    Gender.COMMON: "de",
    Gender.NEUTER: "het",
}


class SpacyPosTagger(PosTagger):
    """Dutch tagger wrapping a loaded spaCy pipeline."""

    def __init__(self, language: str = "nl", model: str | None = None) -> None:
        self._language = language
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise PosTaggerError(
                "spaCy is not installed. Install the optional extra: "
                "`uv pip install -e '.[nlp]'`."
            ) from exc

        model_name = model or os.environ.get("COURSE_SPACY_MODEL", _DEFAULT_MODEL)
        try:
            self._nlp = spacy.load(model_name)
        except OSError as exc:  # pragma: no cover - depends on installed model
            raise PosTaggerError(
                f"spaCy model {model_name!r} is not installed. Run: "
                f"`python -m spacy download {model_name}`."
            ) from exc

    @property
    def language(self) -> str:
        return self._language

    def tag(self, text: str) -> TaggedDoc:
        doc = self._nlp(text)
        tokens: list[TokenTag] = []
        index_of: dict[int, int] = {}  # spaCy token.i -> position in ``tokens``
        for token in doc:
            index_of[token.i] = len(tokens)
            tokens.append(
                TokenTag(
                    surface=token.text,
                    start=token.idx,
                    end=token.idx + len(token.text),
                    lemma=(token.lemma_ or token.text).lower(),
                    pos=_UPOS_MAP.get(token.pos_),
                    upos=token.pos_,
                )
            )

        # Separable verbs: the detached prefix attaches to its verb via compound:prt.
        particle_links: list[tuple[int, int]] = []
        for token in doc:
            if token.dep_ in ("compound:prt", "svp") and token.head is not None:
                verb_i = index_of.get(token.head.i)
                part_i = index_of.get(token.i)
                if verb_i is not None and part_i is not None:
                    particle_links.append((verb_i, part_i))

        return TaggedDoc(tokens=tokens, particle_links=particle_links, parsed=True)

    def article_for_gender(self, gender: Gender | str | None) -> str | None:
        if gender is None:
            return None
        if isinstance(gender, str):
            try:
                gender = Gender(gender)
            except ValueError:
                return None
        return _ARTICLE_BY_GENDER.get(gender)


register_tagger("nl", lambda language: SpacyPosTagger(language))
