"""Token annotator: resolve each word in a lesson to a lexicon ref + sense.

Pure and I/O-free. Given a lesson's text, a prepared closed-vocabulary context
(:class:`LessonVocab`) and a :class:`~course_compiler.nlp.base.PosTagger`, it
produces the lesson's annotated token stream (``list[LessonToken | str]``) and the
resolved :class:`~course_compiler.models.LessonWord` vocabulary list.

Resolution per token (spaCy is the primary driver):
  1. spaCy ``(lemma, pos)`` matched against the lesson's ``(lemma, pos)`` lexicon.
  2. a verb-form map (built from conjugation tables) as a deterministic fallback —
     a conjugated form beats a homograph noun.
  3. otherwise snap to any lexicon entry for the lemma; else leave unlinked.

Separable verbs are fused via the parser's particle links (or a dictionary
scan-ahead fallback); the base form and detached prefix both link to the full
infinitive and share a ``span``. Same-POS ambiguity (e.g. ``bank``) is deferred to
an optional ``sense_picker`` callback (the cached-LLM fallback lives in
``generation.sense``); without one, the first candidate gloss is used.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from course_compiler.models import (
    LessonToken,
    LessonWord,
    PartOfSpeech,
    Verb,
    Word,
)
from course_compiler.nlp.base import PosTagger, TokenTag

_SENTENCE_END = ".!?"


# --------------------------------------------------------------------------- #
# Closed-vocabulary context
# --------------------------------------------------------------------------- #
@dataclass
class LessonVocab:
    """The closed vocabulary a lesson's tokens may resolve against."""

    #: word ref ("lemma|pos") -> Word
    word_entries: dict[str, Word] = field(default_factory=dict)
    #: verb infinitive -> Verb
    verb_entries: dict[str, Verb] = field(default_factory=dict)
    #: (lemma, pos value) -> word ref
    words_by_lemma_pos: dict[tuple[str, str], str] = field(default_factory=dict)
    #: lemma -> [word ref, ...] across POS
    words_by_lemma: dict[str, list[str]] = field(default_factory=dict)
    #: surface conjugated form -> verb infinitive
    form_to_verb: dict[str, str] = field(default_factory=dict)
    #: verb infinitive -> {"prefix", "stem"}
    separable: dict[str, dict[str, str]] = field(default_factory=dict)
    #: stem infinitive -> [(prefix, full separable infinitive), ...]. Recovers a
    #: separable verb from its lemmatized stem + detached particle (spaCy lemmatizes
    #: ``stelt`` → ``stellen``, so ``("stellen", "voor")`` → ``voorstellen``).
    separable_by_stem: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    #: lemmas the lesson may freely use (soft preference when disambiguating)
    allowed_lemmas: set[str] = field(default_factory=set)


def build_lesson_vocab(
    words: list[Word],
    verbs: list[Verb],
    *,
    separable: dict[str, dict[str, str]] | None = None,
    allowed_lemmas: set[str] | None = None,
) -> LessonVocab:
    """Index loaded ``Word``/``Verb`` models into a :class:`LessonVocab`."""
    vocab = LessonVocab(
        separable=dict(separable or {}),
        allowed_lemmas=set(allowed_lemmas or set()),
    )
    for inf, info in vocab.separable.items():
        stem = (info.get("stem") or "").lower()
        prefix = (info.get("prefix") or "").lower()
        if stem and prefix:
            vocab.separable_by_stem.setdefault(stem, []).append((prefix, inf))
    for word in words:
        ref = word.id
        vocab.word_entries[ref] = word
        lemma = word.normalized or word.lemma.lower()
        pos = word.part_of_speech.value
        vocab.words_by_lemma_pos.setdefault((lemma, pos), ref)
        vocab.words_by_lemma.setdefault(lemma, []).append(ref)

    for verb in verbs:
        inf = (verb.infinitive or verb.lemma).lower()
        vocab.verb_entries[inf] = verb
        vocab.form_to_verb.setdefault(inf, inf)
        for table in (
            verb.present,
            verb.past,
            verb.perfect,
            verb.imperative,
            verb.future,
            verb.conditional,
            verb.subjunctive,
        ):
            for form in table.values():
                if form:
                    vocab.form_to_verb.setdefault(form.lower(), inf)
    return vocab


# --------------------------------------------------------------------------- #
# Sense-disambiguation hook
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SenseQuery:
    """A token left ambiguous after POS resolution (same lemma+POS, many senses)."""

    token_index: int  # index of the LessonToken in the output stream
    lemma: str
    pos: str
    sentence: str
    candidates: list[str]


#: Picks one gloss per ambiguous token: ``{token_index: chosen_gloss}``.
SensePicker = Callable[[list[SenseQuery]], dict[int, str]]


# --------------------------------------------------------------------------- #
# Authoring overrides (the ``lessonNNN.meta.yaml`` escape hatch)
# --------------------------------------------------------------------------- #
@dataclass
class LessonOverrides:
    """Per-lesson manual corrections for the rare case the pipeline gets wrong.

    All keys are surface forms / refs as they appear in the lesson, lowercased.
    """

    #: surface form -> forced ref ("lemma|pos" or a verb infinitive). Use an empty
    #: string to force-unlink (e.g. a mis-linked proper name).
    link_as: dict[str, str] = field(default_factory=dict)
    #: ref OR lemma -> forced display gloss (overrides POS/LLM sense selection).
    gloss_overrides: dict[str, str] = field(default_factory=dict)
    #: forced separable fusions: ``[{"surface": "stelt voor", "lemma": "voorstellen"}]``.
    separable_spans: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict | None) -> LessonOverrides:
        data = data or {}
        return cls(
            link_as={
                str(k).lower(): str(v) for k, v in (data.get("linkAs") or {}).items()
            },
            gloss_overrides={
                str(k).lower(): str(v)
                for k, v in (data.get("glossOverrides") or {}).items()
            },
            separable_spans=[dict(s) for s in (data.get("separableVerbs") or [])],
        )

    def is_empty(self) -> bool:
        return not (self.link_as or self.gloss_overrides or self.separable_spans)


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #
def annotate(
    text: str,
    vocab: LessonVocab,
    tagger: PosTagger,
    *,
    sense_picker: SensePicker | None = None,
    overrides: LessonOverrides | None = None,
) -> list[LessonToken | str]:
    """Return the annotated token stream for *text* (strings for gaps)."""
    overrides = overrides or LessonOverrides()
    doc = tagger.tag(text)
    tags = doc.tokens
    particle_of = dict(doc.particle_links)

    resolved: list[LessonToken | None] = [None] * len(tags)
    consumed: set[int] = set()
    # tag index -> (candidate glosses, sentence) for ambiguous tokens
    ambiguity: dict[int, tuple[list[str], str]] = {}

    for i, tag in enumerate(tags):
        if i in consumed or resolved[i] is not None:
            continue
        if not tag.is_word or tag.upos == "PROPN":
            continue
        _resolve_token(
            i, tag, tags, text, vocab, particle_of, resolved, consumed, ambiguity,
            parsed=doc.parsed,
        )

    # Authoring overrides win over automatic resolution.
    _apply_forced_separable(overrides, tags, vocab, resolved, consumed, ambiguity)
    _apply_link_as(overrides, tags, vocab, resolved, consumed, ambiguity)

    stream, queries = _build_stream(text, tags, resolved, ambiguity)

    if overrides.gloss_overrides:
        queries = _apply_gloss_overrides(stream, queries, overrides.gloss_overrides)

    if sense_picker and queries:
        chosen = sense_picker(queries)
        for query in queries:
            gloss = chosen.get(query.token_index)
            if gloss:
                token = stream[query.token_index]
                if isinstance(token, LessonToken):
                    token.gloss = gloss
    return stream


def _token_from_ref(surface: str, ref: str, vocab: LessonVocab) -> LessonToken:
    """Build a token forced to *ref*, taking pos/gloss from the lexicon entry."""
    if ref in vocab.verb_entries:
        glosses = _candidate_glosses(vocab.verb_entries[ref])
        return LessonToken(
            w=surface, ref=ref, pos="verb", gloss=glosses[0] if glosses else None
        )
    word = vocab.word_entries.get(ref)
    if word is not None:
        glosses = _candidate_glosses(word)
        return LessonToken(
            w=surface,
            ref=ref,
            pos=word.part_of_speech.value,
            gloss=glosses[0] if glosses else None,
        )
    return LessonToken(w=surface, ref=ref)  # ref unknown to lexicon: link anyway


def _apply_link_as(
    overrides: LessonOverrides,
    tags: list[TokenTag],
    vocab: LessonVocab,
    resolved: list[LessonToken | None],
    consumed: set[int],
    ambiguity: dict[int, tuple[list[str], str]],
) -> None:
    if not overrides.link_as:
        return
    for i, tag in enumerate(tags):
        if i in consumed or not tag.is_word:
            continue
        forced = overrides.link_as.get(tag.surface.lower())
        if forced is None:
            continue
        ambiguity.pop(i, None)
        resolved[i] = _token_from_ref(tag.surface, forced, vocab) if forced else None


def _apply_forced_separable(
    overrides: LessonOverrides,
    tags: list[TokenTag],
    vocab: LessonVocab,
    resolved: list[LessonToken | None],
    consumed: set[int],
    ambiguity: dict[int, tuple[list[str], str]],
) -> None:
    for span in overrides.separable_spans:
        surface = str(span.get("surface", "")).strip()
        lemma = str(span.get("lemma", "")).strip()
        parts = surface.split()
        if not lemma or len(parts) < 2:
            continue
        base, particle = parts[0].lower(), parts[-1].lower()
        base_idx = _find_word(tags, base, start=0, skip=consumed)
        if base_idx is None:
            continue
        part_idx = _find_word(tags, particle, start=base_idx + 1, skip=consumed)
        if part_idx is None:
            continue
        verb = vocab.verb_entries.get(lemma)
        glosses = _candidate_glosses(verb) if verb else []
        gloss = glosses[0] if glosses else None
        full_span = [tags[base_idx].surface, tags[part_idx].surface]
        ambiguity.pop(base_idx, None)
        resolved[base_idx] = LessonToken(
            w=tags[base_idx].surface, ref=lemma, pos="verb", gloss=gloss, span=full_span
        )
        resolved[part_idx] = LessonToken(
            w=tags[part_idx].surface, ref=lemma, pos="verb", gloss=gloss, span=full_span
        )
        consumed.add(part_idx)


def _find_word(
    tags: list[TokenTag], surface_lower: str, *, start: int, skip: set[int]
) -> int | None:
    for j in range(start, len(tags)):
        if j in skip:
            continue
        if tags[j].is_word and tags[j].surface.lower() == surface_lower:
            return j
    return None


def _apply_gloss_overrides(
    stream: list[LessonToken | str],
    queries: list[SenseQuery],
    gloss_overrides: dict[str, str],
) -> list[SenseQuery]:
    overridden: set[int] = set()
    for idx, token in enumerate(stream):
        if not isinstance(token, LessonToken) or not token.ref:
            continue
        keys = {token.ref.lower(), token.ref.split("|")[0].lower()}
        for key in keys:
            if key in gloss_overrides:
                token.gloss = gloss_overrides[key]
                overridden.add(idx)
                break
    if not overridden:
        return queries
    return [q for q in queries if q.token_index not in overridden]


def _resolve_token(
    i: int,
    tag: TokenTag,
    tags: list[TokenTag],
    text: str,
    vocab: LessonVocab,
    particle_of: dict[int, int],
    resolved: list[LessonToken | None],
    consumed: set[int],
    ambiguity: dict[int, tuple[list[str], str]],
    *,
    parsed: bool,
) -> None:
    surface = tag.surface
    sl = surface.lower()
    lemma = tag.lemma or sl

    # --- verb path (POS says verb, or the surface/lemma is a known verb form) ---
    base_inf = _verb_infinitive(tag, sl, lemma, vocab)
    # Even if the bare stem isn't itself in the lesson vocab, the spaCy lemma may be
    # the stem of a separable verb that is (``stellen`` → ``voorstellen``).
    if base_inf is None and tag.pos is PartOfSpeech.VERB and lemma in vocab.separable_by_stem:
        base_inf = lemma
    if base_inf is not None:
        fused_inf, part_idx = _separable_override(
            i, base_inf, tags, vocab, particle_of, parsed=parsed
        )
        inf = fused_inf if (fused_inf and fused_inf in vocab.verb_entries) else base_inf
        if inf in vocab.verb_entries:
            verb = vocab.verb_entries[inf]
            glosses = _candidate_glosses(verb)
            gloss = glosses[0] if glosses else None
            span = None
            if fused_inf == inf and part_idx is not None and part_idx not in consumed:
                span = [tags[i].surface, tags[part_idx].surface]
                resolved[part_idx] = LessonToken(
                    w=tags[part_idx].surface, ref=inf, pos="verb", gloss=gloss, span=span
                )
                consumed.add(part_idx)
            resolved[i] = LessonToken(
                w=surface, ref=inf, pos="verb", gloss=gloss, span=span
            )
            if len(glosses) > 1:
                ambiguity[i] = (glosses, _sentence_for(text, tag.start))
            return
        # Not a resolvable verb entry — fall through to the word path.

    # --- word path ---
    ref = _word_ref(tag, sl, lemma, vocab)
    if ref is not None:
        entry = vocab.word_entries[ref]
        glosses = _candidate_glosses(entry)
        gloss = glosses[0] if glosses else None
        resolved[i] = LessonToken(
            w=surface, ref=ref, pos=entry.part_of_speech.value, gloss=gloss
        )
        if len(glosses) > 1:
            ambiguity[i] = (glosses, _sentence_for(text, tag.start))


# POS tags we trust as confidently NON-verb, so the verb-form map never coerces
# them into a verb. These are exactly the closed-class homographs that bite us
# (``zijn`` DET = "his" vs the verb "to be"; ``aan``/``op`` ADP vs separable
# particles; ``een`` NUM vs article). spaCy is reliable on these classes.
_TRUSTED_NON_VERB_POS = frozenset(
    {
        PartOfSpeech.DETERMINER,
        PartOfSpeech.PRONOUN,
        PartOfSpeech.PREPOSITION,
        PartOfSpeech.CONJUNCTION,
        PartOfSpeech.ARTICLE,
        PartOfSpeech.NUMERAL,
        PartOfSpeech.INTERJECTION,
    }
)


def _verb_infinitive(
    tag: TokenTag, sl: str, lemma: str, vocab: LessonVocab
) -> str | None:
    """Resolve a token to a verb infinitive, or ``None`` if it is not a verb here."""
    if tag.pos is PartOfSpeech.VERB:
        if lemma in vocab.verb_entries:
            return lemma
        if sl in vocab.form_to_verb:
            return vocab.form_to_verb[sl]
        if lemma in vocab.form_to_verb:
            return vocab.form_to_verb[lemma]
        return None
    if tag.pos in _TRUSTED_NON_VERB_POS:
        # e.g. possessive "zijn" (DET) must not become the verb "zijn" (to be).
        return None
    # spaCy is unsure (None) or tagged a plausibly-mistaggable content category
    # (NOUN/ADJ/ADV): a known conjugated form beats a homograph noun ("loopt"→lopen).
    if sl in vocab.form_to_verb:
        return vocab.form_to_verb[sl]
    return None


def _word_ref(tag: TokenTag, sl: str, lemma: str, vocab: LessonVocab) -> str | None:
    pos = tag.pos.value if tag.pos else None
    if pos is not None:
        ref = vocab.words_by_lemma_pos.get((lemma, pos))
        if ref is None:
            ref = vocab.words_by_lemma_pos.get((sl, pos))
        if ref is not None:
            return ref
    # No POS match: any entry for this lemma (prefer in-lesson / matching POS).
    candidates = vocab.words_by_lemma.get(lemma) or vocab.words_by_lemma.get(sl)
    if candidates:
        return _prefer(candidates, pos, vocab)
    return None


def _prefer(refs: list[str], pos: str | None, vocab: LessonVocab) -> str:
    if pos is not None:
        for ref in refs:
            if vocab.word_entries[ref].part_of_speech.value == pos:
                return ref
    for ref in refs:
        if vocab.word_entries[ref].normalized in vocab.allowed_lemmas:
            return ref
    return refs[0]


def _separable_override(
    verb_idx: int,
    base_inf: str,
    tags: list[TokenTag],
    vocab: LessonVocab,
    particle_of: dict[int, int],
    *,
    parsed: bool,
) -> tuple[str | None, int | None]:
    """Map a base/stem verb + detached particle to its separable infinitive.

    Returns ``(separable_infinitive, particle_token_index)`` when the token is the
    finite form of a separable verb with its prefix detached later in the clause;
    otherwise ``(None, None)``. Uses the parser's particle link when present. When
    the backend *parsed* the text, that link is authoritative and we do NOT scan —
    inventing a particle would fuse a stray preposition (e.g. ``zijn`` + ``aan de
    tafel`` → bogus ``aanzijn``). The dictionary scan-ahead is only for parser-less
    taggers (``parsed=False``).
    """
    candidates = vocab.separable_by_stem.get(base_inf)
    if not candidates:
        return None, None

    linked = particle_of.get(verb_idx)
    if linked is not None and tags[linked].is_word:
        prefix = tags[linked].surface.lower()
        for cand_prefix, inf in candidates:
            if cand_prefix == prefix:
                return inf, linked

    if parsed:
        return None, None  # trust the parser: no link means no separable particle

    for cand_prefix, inf in candidates:
        idx = _scan_particle(tags, verb_idx, cand_prefix)
        if idx is not None:
            return inf, idx
    return None, None


def _scan_particle(tags: list[TokenTag], verb_idx: int, prefix: str) -> int | None:
    """Find the detached particle after the verb, before the sentence ends."""
    if not prefix:
        return None
    for j in range(verb_idx + 1, len(tags)):
        surface = tags[j].surface
        if surface in _SENTENCE_END:
            return None
        if tags[j].is_word and surface.lower() == prefix:
            return j
    return None


def _build_stream(
    text: str,
    tags: list[TokenTag],
    resolved: list[LessonToken | None],
    ambiguity: dict[int, tuple[list[str], str]],
) -> tuple[list[LessonToken | str], list[SenseQuery]]:
    """Interleave resolved tokens with the verbatim inter-token gap text."""
    stream: list[LessonToken | str] = []
    queries: list[SenseQuery] = []
    buf: list[str] = []
    cursor = 0

    def flush() -> None:
        if buf:
            stream.append("".join(buf))
            buf.clear()

    for i, tag in enumerate(tags):
        token = resolved[i]
        if token is None:
            continue  # gap text captured lazily via the cursor
        if tag.start > cursor:
            buf.append(text[cursor : tag.start])
        flush()
        if i in ambiguity:
            glosses, sentence = ambiguity[i]
            queries.append(
                SenseQuery(
                    token_index=len(stream),
                    lemma=(token.ref or "").split("|")[0],
                    pos=token.pos or "",
                    sentence=sentence,
                    candidates=glosses,
                )
            )
        stream.append(token)
        cursor = max(cursor, tag.end)

    if cursor < len(text):
        buf.append(text[cursor:])
    flush()
    return stream, queries


def _candidate_glosses(entry: Word | Verb) -> list[str]:
    if entry.glosses:
        return list(entry.glosses)
    en = entry.translations.get("en")
    if en:
        return [g.strip() for g in en.split(" ; ") if g.strip()]
    return []


_SENT_SPLIT = re.compile(r"[.!?]+")


def _sentence_for(text: str, offset: int) -> str:
    """The sentence (between .!? boundaries) containing *offset*."""
    start = 0
    for match in _SENT_SPLIT.finditer(text):
        if match.end() > offset:
            break
        start = match.end()
    end = len(text)
    tail = _SENT_SPLIT.search(text, offset)
    if tail:
        end = tail.end()
    return text[start:end].strip()


# --------------------------------------------------------------------------- #
# Vocabulary list
# --------------------------------------------------------------------------- #
def build_vocabulary(
    new_words: list[str],
    vocab: LessonVocab,
    stream: list[LessonToken | str],
    tagger: PosTagger,
) -> list[LessonWord]:
    """Resolve each new-word lemma to a :class:`LessonWord`.

    POS/sense are taken from how the word actually resolved in the text (the token
    stream); a word that never appears falls back to its best lexicon entry.
    """
    by_lemma: dict[str, LessonToken] = {}
    for token in stream:
        if isinstance(token, LessonToken) and token.ref:
            lemma = token.ref.split("|")[0]
            by_lemma.setdefault(lemma, token)

    out: list[LessonWord] = []
    for lemma in new_words:
        key = lemma.lower()
        token = by_lemma.get(key)
        if token is not None and token.ref:
            out.append(_lesson_word_from_ref(lemma, token.ref, token.gloss, vocab, tagger))
            continue
        ref = _best_ref_for_lemma(key, vocab)
        gloss = None
        if ref is not None:
            entry = _entry_for_ref(ref, vocab)
            glosses = _candidate_glosses(entry) if entry else []
            gloss = glosses[0] if glosses else None
        out.append(_lesson_word_from_ref(lemma, ref, gloss, vocab, tagger))
    return out


def _best_ref_for_lemma(lemma: str, vocab: LessonVocab) -> str | None:
    if lemma in vocab.verb_entries:
        return lemma
    refs = vocab.words_by_lemma.get(lemma)
    if refs:
        return refs[0]
    return None


def _entry_for_ref(ref: str, vocab: LessonVocab) -> Word | Verb | None:
    if ref in vocab.verb_entries:
        return vocab.verb_entries[ref]
    return vocab.word_entries.get(ref)


def _lesson_word_from_ref(
    lemma: str,
    ref: str | None,
    gloss: str | None,
    vocab: LessonVocab,
    tagger: PosTagger,
) -> LessonWord:
    if ref is None:
        return LessonWord(lemma=lemma, gloss=gloss)
    if ref in vocab.verb_entries:
        return LessonWord(lemma=lemma, pos="verb", ref=ref, gloss=gloss)
    word = vocab.word_entries.get(ref)
    if word is None:
        return LessonWord(lemma=lemma, ref=ref, gloss=gloss)
    gender = word.gender.value if word.gender else None
    return LessonWord(
        lemma=lemma,
        pos=word.part_of_speech.value,
        ref=ref,
        gloss=gloss,
        gender=gender,
        article=tagger.article_for_gender(word.gender),
    )
