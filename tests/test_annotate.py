"""Token annotator behavior, exercised with a fake (offline) POS tagger."""

from __future__ import annotations

import re

from course_compiler.generation.annotate import (
    LessonOverrides,
    LessonToken,
    SenseQuery,
    annotate,
    build_lesson_vocab,
    build_vocabulary,
)
from course_compiler.models import Gender, PartOfSpeech, Verb, Word
from course_compiler.nlp.base import PosTagger, TaggedDoc, TokenTag

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+|[.!?]")


class FakeTagger(PosTagger):
    """Deterministic tagger: ``spec`` maps surface→list of (lemma, pos, upos).

    A list lets the same surface be tagged differently per occurrence (homographs).
    """

    def __init__(self, spec, particle_links=None, parsed=False):
        self._spec = spec
        self._links = particle_links or []
        self._parsed = parsed

    @property
    def language(self) -> str:
        return "nl"

    def tag(self, text: str) -> TaggedDoc:
        counter: dict[str, int] = {}
        tokens: list[TokenTag] = []
        for m in _TOKEN_RE.finditer(text):
            surface = m.group(0)
            if surface in ".!?":
                tokens.append(
                    TokenTag(surface, m.start(), m.end(), surface, None, "PUNCT")
                )
                continue
            sl = surface.lower()
            entries = self._spec.get(sl, [(sl, None, "X")])
            idx = min(counter.get(sl, 0), len(entries) - 1)
            counter[sl] = counter.get(sl, 0) + 1
            lemma, pos, upos = entries[idx]
            tokens.append(TokenTag(surface, m.start(), m.end(), lemma, pos, upos))
        return TaggedDoc(
            tokens=tokens, particle_links=list(self._links), parsed=self._parsed
        )

    def article_for_gender(self, gender):
        if gender in (Gender.MASCULINE, Gender.FEMININE, Gender.COMMON):
            return "de"
        if gender is Gender.NEUTER:
            return "het"
        return None


def _word(lemma, pos, glosses, *, gender=None):
    return Word(
        id=f"{lemma}|{pos.value}",
        language="nl",
        lemma=lemma,
        normalized=lemma,
        part_of_speech=pos,
        glosses=glosses,
        gender=gender,
    )


def _verb(infinitive, glosses, *, present=None):
    return Verb(
        id=infinitive,
        language="nl",
        lemma=infinitive,
        infinitive=infinitive,
        glosses=glosses,
        present=present or {},
    )


def _reconstruct(stream) -> str:
    return "".join(t if isinstance(t, str) else t.w for t in stream)


def _tokens(stream):
    return [t for t in stream if isinstance(t, LessonToken)]


def test_homograph_resolves_to_different_senses_per_occurrence():
    text = "De morgen. En morgen."
    vocab = build_lesson_vocab(
        [
            _word("morgen", PartOfSpeech.NOUN, ["morning"]),
            _word("morgen", PartOfSpeech.ADVERB, ["tomorrow"]),
        ],
        [],
    )
    tagger = FakeTagger(
        {
            "de": [("de", PartOfSpeech.DETERMINER, "DET")],
            "en": [("en", PartOfSpeech.CONJUNCTION, "CCONJ")],
            "morgen": [
                ("morgen", PartOfSpeech.NOUN, "NOUN"),
                ("morgen", PartOfSpeech.ADVERB, "ADV"),
            ],
        }
    )
    stream = annotate(text, vocab, tagger)
    assert _reconstruct(stream) == text
    morgen_tokens = [t for t in _tokens(stream) if t.w.lower() == "morgen"]
    assert [(t.ref, t.gloss) for t in morgen_tokens] == [
        ("morgen|noun", "morning"),
        ("morgen|adverb", "tomorrow"),
    ]


def test_conjugated_form_resolves_to_verb_over_noun_homograph():
    text = "Hij loopt."
    vocab = build_lesson_vocab(
        [_word("lopen", PartOfSpeech.NOUN, ["a dry measure"])],
        [_verb("lopen", ["walk"], present={"hij": "loopt"})],
    )
    tagger = FakeTagger(
        {
            "hij": [("hij", PartOfSpeech.PRONOUN, "PRON")],
            "loopt": [("lopen", PartOfSpeech.VERB, "VERB")],
        }
    )
    stream = annotate(text, vocab, tagger)
    [tok] = [t for t in _tokens(stream) if t.w == "loopt"]
    assert tok.ref == "lopen" and tok.pos == "verb" and tok.gloss == "walk"


def test_verb_form_map_is_fallback_when_pos_is_wrong():
    # spaCy mis-tags "loopt" as a noun; the form map still recovers the verb.
    text = "De loopt hard."
    vocab = build_lesson_vocab(
        [_word("lopen", PartOfSpeech.NOUN, ["a dry measure"])],
        [_verb("lopen", ["walk"], present={"hij": "loopt"})],
    )
    tagger = FakeTagger({"loopt": [("loopt", PartOfSpeech.NOUN, "NOUN")]})
    stream = annotate(text, vocab, tagger)
    [tok] = [t for t in _tokens(stream) if t.w == "loopt"]
    assert tok.ref == "lopen" and tok.pos == "verb"


def test_separable_verb_fuses_stem_and_detached_particle():
    text = "De man stelt zich voor."
    vocab = build_lesson_vocab(
        [],
        [_verb("voorstellen", ["introduce"])],
        separable={"voorstellen": {"prefix": "voor", "stem": "stellen"}},
    )
    tagger = FakeTagger({"stelt": [("stellen", PartOfSpeech.VERB, "VERB")]})
    stream = annotate(text, vocab, tagger)
    assert _reconstruct(stream) == text
    fused = [t for t in _tokens(stream) if t.ref == "voorstellen"]
    assert {t.w for t in fused} == {"stelt", "voor"}
    assert all(t.span == ["stelt", "voor"] for t in fused)
    assert all(t.gloss == "introduce" for t in fused)


def test_proper_name_is_left_unlinked():
    text = "Jan loopt."
    vocab = build_lesson_vocab([], [_verb("lopen", ["walk"], present={"hij": "loopt"})])
    tagger = FakeTagger(
        {
            "jan": [("jan", None, "PROPN")],
            "loopt": [("lopen", PartOfSpeech.VERB, "VERB")],
        }
    )
    stream = annotate(text, vocab, tagger)
    assert _reconstruct(stream) == text
    assert "Jan" not in [t.w for t in _tokens(stream)]
    assert [t.w for t in _tokens(stream)] == ["loopt"]


def test_sense_picker_resolves_same_pos_ambiguity():
    text = "De bank."
    vocab = build_lesson_vocab(
        [_word("bank", PartOfSpeech.NOUN, ["riverbank", "financial institution"])], []
    )
    tagger = FakeTagger({"bank": [("bank", PartOfSpeech.NOUN, "NOUN")]})

    # Default: first candidate.
    default_stream = annotate(text, vocab, tagger)
    [default_tok] = [t for t in _tokens(default_stream) if t.w == "bank"]
    assert default_tok.gloss == "riverbank"

    seen: list[SenseQuery] = []

    def picker(queries):
        seen.extend(queries)
        return {q.token_index: q.candidates[1] for q in queries}

    stream = annotate(text, vocab, tagger, sense_picker=picker)
    [tok] = [t for t in _tokens(stream) if t.w == "bank"]
    assert tok.gloss == "financial institution"
    assert seen and seen[0].lemma == "bank" and seen[0].sentence == "De bank."


def test_build_vocabulary_resolves_article_and_ref():
    text = "De bank is dicht."
    vocab = build_lesson_vocab(
        [_word("bank", PartOfSpeech.NOUN, ["bank"], gender=Gender.COMMON)],
        [_verb("zijn", ["be"], present={"hij": "is"})],
    )
    tagger = FakeTagger(
        {
            "bank": [("bank", PartOfSpeech.NOUN, "NOUN")],
            "is": [("zijn", PartOfSpeech.VERB, "VERB")],
        }
    )
    stream = annotate(text, vocab, tagger)
    vocabulary = build_vocabulary(["bank", "zijn"], vocab, stream, tagger)
    by_lemma = {w.lemma: w for w in vocabulary}
    assert by_lemma["bank"].ref == "bank|noun"
    assert by_lemma["bank"].article == "de"
    assert by_lemma["bank"].pos == "noun"
    assert by_lemma["zijn"].ref == "zijn" and by_lemma["zijn"].pos == "verb"


# --- authoring override sidecar (task 0023, item 7) -----------------------------

def test_override_link_as_forces_ref_for_homograph_form():
    # "groet" lemmatizes to the noun by default; force it to the verb groeten.
    text = "Ik groet je."
    vocab = build_lesson_vocab(
        [_word("groet", PartOfSpeech.NOUN, ["greeting"])],
        [_verb("groeten", ["greet"])],
    )
    tagger = FakeTagger({"groet": [("groet", PartOfSpeech.NOUN, "NOUN")]})

    plain = annotate(text, vocab, tagger)
    [before] = [t for t in _tokens(plain) if t.w == "groet"]
    assert before.ref == "groet|noun"

    overrides = LessonOverrides(link_as={"groet": "groeten"})
    stream = annotate(text, vocab, tagger, overrides=overrides)
    [after] = [t for t in _tokens(stream) if t.w == "groet"]
    assert after.ref == "groeten" and after.pos == "verb" and after.gloss == "greet"


def test_override_link_as_empty_string_unlinks():
    text = "De bank."
    vocab = build_lesson_vocab([_word("bank", PartOfSpeech.NOUN, ["bank"])], [])
    tagger = FakeTagger({"bank": [("bank", PartOfSpeech.NOUN, "NOUN")]})
    stream = annotate(text, vocab, tagger, overrides=LessonOverrides(link_as={"bank": ""}))
    assert "bank" not in [t.w for t in _tokens(stream)]
    assert _reconstruct(stream) == text


def test_override_gloss_wins_over_pos_default_and_skips_sense_picker():
    text = "De bank."
    vocab = build_lesson_vocab(
        [_word("bank", PartOfSpeech.NOUN, ["riverbank", "financial institution"])], []
    )
    tagger = FakeTagger({"bank": [("bank", PartOfSpeech.NOUN, "NOUN")]})

    called = []

    def picker(queries):
        called.extend(queries)
        return {}

    overrides = LessonOverrides(gloss_overrides={"bank|noun": "the place I keep money"})
    stream = annotate(text, vocab, tagger, sense_picker=picker, overrides=overrides)
    [tok] = [t for t in _tokens(stream) if t.w == "bank"]
    assert tok.gloss == "the place I keep money"
    assert called == []  # overridden token not sent to the LLM


def test_override_forced_separable_span():
    text = "De man stelt zich voor."
    vocab = build_lesson_vocab([], [_verb("voorstellen", ["introduce"])])
    # No separable dict and no particle link: only the manual override can fuse it.
    tagger = FakeTagger({"stelt": [("stelt", PartOfSpeech.VERB, "VERB")]})
    overrides = LessonOverrides(
        separable_spans=[{"surface": "stelt voor", "lemma": "voorstellen"}]
    )
    stream = annotate(text, vocab, tagger, overrides=overrides)
    fused = [t for t in _tokens(stream) if t.ref == "voorstellen"]
    assert {t.w for t in fused} == {"stelt", "voor"}
    assert all(t.span == ["stelt", "voor"] for t in fused)


# --- regression: zijn (DET) must not become the verb / a bogus separable -------

def _aanzijn_vocab():
    # A separable verb whose stem ("zijn") is also the verb "to be" and a
    # possessive determiner — the exact homograph trap from lesson003.
    return build_lesson_vocab(
        [_word("zijn", PartOfSpeech.DETERMINER, ["his"])],
        [_verb("zijn", ["be"], present={"hij": "is"}), _verb("aanzijn", ["visit"])],
        separable={"aanzijn": {"prefix": "aan", "stem": "zijn"}},
    )


def test_possessive_zijn_not_coerced_to_verb_or_separable():
    text = "En zijn zus zit aan tafel."
    vocab = _aanzijn_vocab()
    tagger = FakeTagger(
        {
            "zijn": [("zijn", PartOfSpeech.DETERMINER, "DET")],
            "aan": [("aan", PartOfSpeech.PREPOSITION, "ADP")],
        },
        parsed=True,
    )
    stream = annotate(text, vocab, tagger)
    assert _reconstruct(stream) == text
    [zijn_tok] = [t for t in _tokens(stream) if t.w == "zijn"]
    assert zijn_tok.ref == "zijn|determiner"
    assert zijn_tok.pos == "determiner"
    assert zijn_tok.span is None
    # "aan" must not have been swallowed into a bogus separable verb.
    assert not any(t.ref == "aanzijn" for t in _tokens(stream))


def test_parsed_tagger_does_not_invent_separable_from_stray_preposition():
    text = "Ik ben aan tafel."
    vocab = _aanzijn_vocab()
    tagger = FakeTagger(
        {
            "ben": [("zijn", PartOfSpeech.VERB, "VERB")],
            "aan": [("aan", PartOfSpeech.PREPOSITION, "ADP")],
        },
        parsed=True,  # parser ran and linked no particle
    )
    stream = annotate(text, vocab, tagger)
    [ben] = [t for t in _tokens(stream) if t.w == "ben"]
    assert ben.ref == "zijn" and ben.span is None  # the verb "to be", not fused
    assert not any(t.ref == "aanzijn" for t in _tokens(stream))


def test_separable_fuses_via_parser_particle_link():
    text = "De man stelt zich voor."
    vocab = build_lesson_vocab(
        [],
        [_verb("voorstellen", ["introduce"])],
        separable={"voorstellen": {"prefix": "voor", "stem": "stellen"}},
    )
    # Parsed tagger links token 2 (stelt) -> token 4 (voor); no scan-ahead used.
    tagger = FakeTagger(
        {"stelt": [("stellen", PartOfSpeech.VERB, "VERB")]},
        particle_links=[(2, 4)],
        parsed=True,
    )
    stream = annotate(text, vocab, tagger)
    fused = [t for t in _tokens(stream) if t.ref == "voorstellen"]
    assert {t.w for t in fused} == {"stelt", "voor"}
    assert all(t.span == ["stelt", "voor"] for t in fused)


# --- regression: zon (NOUN) vs a verb form; morgen cross-POS homograph ----------

def test_noun_is_not_coerced_to_verb_when_it_is_a_verb_form():
    # "zon" (sun, NOUN) is also the 1sg of "zonnen"; spaCy's NOUN must win.
    text = "De zon schijnt."
    vocab = build_lesson_vocab(
        [_word("zon", PartOfSpeech.NOUN, ["sun"])],
        [_verb("zonnen", ["contemplate"], present={"ik": "zon"})],
    )
    tagger = FakeTagger({"zon": [("zon", PartOfSpeech.NOUN, "NOUN")]})
    stream = annotate(text, vocab, tagger)
    [tok] = [t for t in _tokens(stream) if t.w == "zon"]
    assert tok.ref == "zon|noun" and tok.pos == "noun" and tok.gloss == "sun"


def test_cross_pos_homograph_defaults_to_spacy_but_llm_can_switch_ref():
    text = "Het is morgen."
    vocab = build_lesson_vocab(
        [
            _word("morgen", PartOfSpeech.NOUN, ["morning"]),
            _word("morgen", PartOfSpeech.ADVERB, ["tomorrow"]),
        ],
        [_verb("zijn", ["be"], present={"hij": "is"})],
    )
    # spaCy biases bare "morgen" to the adverb ("tomorrow").
    tagger = FakeTagger(
        {
            "het": [("het", PartOfSpeech.PRONOUN, "PRON")],
            "is": [("zijn", PartOfSpeech.VERB, "VERB")],
            "morgen": [("morgen", PartOfSpeech.ADVERB, "ADV")],
        }
    )

    # Without an LLM, the default is spaCy's adverb sense.
    default = annotate(text, vocab, tagger)
    [d] = [t for t in _tokens(default) if t.w == "morgen"]
    assert d.ref == "morgen|adverb" and d.gloss == "tomorrow"

    # The picker is offered both cross-POS senses and can switch ref + pos + gloss.
    seen: list[SenseQuery] = []

    def picker(queries):
        seen.extend(queries)
        return {q.token_index: "morning" for q in queries}

    stream = annotate(text, vocab, tagger, sense_picker=picker)
    [m] = [t for t in _tokens(stream) if t.w == "morgen"]
    assert m.ref == "morgen|noun" and m.pos == "noun" and m.gloss == "morning"
    assert seen and set(seen[0].candidates) == {"tomorrow", "morning"}


def test_function_word_homograph_is_not_sent_to_the_picker():
    # "het" (DET "the" vs PRON "it") is a function word: trust spaCy, no LLM query.
    text = "Het is het."
    vocab = build_lesson_vocab(
        [
            _word("het", PartOfSpeech.DETERMINER, ["the"]),
            _word("het", PartOfSpeech.PRONOUN, ["it"]),
        ],
        [_verb("zijn", ["be"], present={"hij": "is"})],
    )
    tagger = FakeTagger(
        {
            "het": [("het", PartOfSpeech.DETERMINER, "DET")],
            "is": [("zijn", PartOfSpeech.VERB, "VERB")],
        }
    )
    seen: list[SenseQuery] = []

    def picker(queries):
        seen.extend(queries)
        return {}

    annotate(text, vocab, tagger, sense_picker=picker)
    assert seen == []  # function-word homographs are not disambiguated by the LLM


def test_noun_after_determiner_not_coerced_to_rare_verb_form():
    # "Ze is nieuw in de buurt." — buurt (neighbourhood, NOUN) is also "ik buurt"
    # of the rare verb buurten; spaCy's NOUN must win over the verb-form fallback.
    text = "Ze is nieuw in de buurt."
    vocab = build_lesson_vocab(
        [_word("buurt", PartOfSpeech.NOUN, ["neighbourhood"], gender=Gender.COMMON)],
        [
            _verb("zijn", ["be"], present={"hij": "is"}),
            _verb("buurten", ["visit a neighbour"], present={"ik": "buurt"}),
        ],
    )
    tagger = FakeTagger(
        {
            "is": [("zijn", PartOfSpeech.VERB, "VERB")],
            "de": [("de", PartOfSpeech.DETERMINER, "DET")],
            "buurt": [("buurt", PartOfSpeech.NOUN, "NOUN")],
        }
    )
    stream = annotate(text, vocab, tagger)
    [tok] = [t for t in _tokens(stream) if t.w == "buurt"]
    assert tok.ref == "buurt|noun"
    assert tok.pos == "noun"
    assert tok.gloss == "neighbourhood"
