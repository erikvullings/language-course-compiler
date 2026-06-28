"""Dutch importer behavior, exercised on small inline fixtures."""

from __future__ import annotations

from course_compiler.converters import dutch
from course_compiler.models import Gender, PartOfSpeech

NOUN = {
    "pos": "noun",
    "word": "huis",
    "lang_code": "nl",
    "head_templates": [{"name": "nl-noun", "args": {"1": "n", "2": "-en"}}],
    "forms": [
        {"form": "huizen", "tags": ["plural"]},
        {"form": "huisje", "tags": ["diminutive", "neuter"]},
        {"form": "huys", "tags": ["alternative", "obsolete"]},
    ],
    "sounds": [
        {"ipa": "/ɦœys/"},
        {"ipa": "[ɦœys]"},
        {
            "audio": "Nl-huis.ogg",
            "ogg_url": "https://example.test/Nl-huis.ogg",
        },
    ],
    "hyphenations": [{"parts": ["huis"]}],
    "senses": [
        {
            "glosses": ["a house, home; residence"],
            "tags": ["neuter", "historical"],
        }
    ],
}

STRONG_VERB = {
    "pos": "verb",
    "word": "lopen",
    "forms": [
        {"form": "7", "tags": ["class"]},
        {"form": "lopen", "tags": ["infinitive"]},
        {"form": "loop", "tags": ["first-person", "present", "singular"]},
        {"form": "loopt", "tags": ["present", "second-person", "singular"]},
        {"form": "loopt", "tags": ["present", "singular", "third-person"]},
        {"form": "lopen", "tags": ["plural", "present"]},
        {"form": "liep", "tags": ["first-person", "past", "singular"]},
        {"form": "liepen", "tags": ["past", "plural"]},
        {"form": "gelopen", "tags": ["participle", "past"]},
        {"form": "loop", "tags": ["imperative", "present", "singular"]},
        {"form": "lope", "tags": ["archaic", "present", "singular", "subjunctive"]},
    ],
    "senses": [{"glosses": ["to walk, to move on foot"]}],
}

WEAK_VERB = {
    "pos": "verb",
    "word": "rennen",
    "forms": [
        {"form": "weak", "tags": ["table-tags"]},
        {"form": "rennen", "tags": ["infinitive"]},
        {"form": "ren", "tags": ["first-person", "present", "singular"]},
        {"form": "rende", "tags": ["first-person", "past", "singular"]},
        {"form": "renden", "tags": ["past", "plural"]},
        {"form": "gerend", "tags": ["participle", "past"]},
    ],
    "senses": [{"glosses": ["to run"]}],
}


def test_noun_maps_core_fields():
    word = dutch.word_from_kaikki(NOUN)
    assert word is not None
    assert word.id == "huis|noun"
    assert word.part_of_speech is PartOfSpeech.NOUN
    assert word.gender is Gender.NEUTER
    assert word.ipa == "/ɦœys/"  # phonemic form preferred over [..]
    assert word.syllables == ["huis"]
    assert word.audio is not None
    assert word.audio.recorded == "https://example.test/Nl-huis.ogg"


def test_noun_plural_and_diminutive_skip_marked_variants():
    word = dutch.word_from_kaikki(NOUN)
    assert word.plural.regular == "huizen"
    assert word.diminutive.regular == "huisje"
    assert "huys" not in word.plural.alternatives  # obsolete/alternative excluded


def test_english_gloss_becomes_translation():
    word = dutch.word_from_kaikki(NOUN)
    assert word.translations == {"en": "house"}  # leading article + extra senses stripped


def test_entry_tags_are_extracted():
    word = dutch.word_from_kaikki(NOUN)
    assert word is not None
    # Grammar-only tags like "neuter" are filtered; lexical tags remain.
    assert "historical" in word.tags


def test_verbs_are_not_emitted_as_words():
    assert dutch.word_from_kaikki(STRONG_VERB) is None


def test_strong_verb_conjugation_and_irregular_flag():
    verb = dutch.verb_from_kaikki(STRONG_VERB)
    assert verb is not None
    assert verb.infinitive == "lopen"
    assert verb.present == {
        "ik": "loop",
        "jij": "loopt",
        "u": "loopt",
        "hij": "loopt",
        "wij": "lopen",
        "jullie": "lopen",
        "zij": "lopen",
    }
    assert verb.past == {"singular": "liep", "plural": "liepen"}
    assert verb.perfect == {"participle": "gelopen"}
    assert verb.imperative == {"singular": "loop"}
    assert verb.irregular is True
    assert verb.translations == {"en": "walk"}


def test_weak_verb_is_not_irregular():
    verb = dutch.verb_from_kaikki(WEAK_VERB)
    assert verb.irregular is False
    assert verb.past == {"singular": "rende", "plural": "renden"}


def test_inflected_form_verb_entry_is_dropped():
    """A form-pointer gloss ('inflection of …') marks a non-infinitive; skip it."""
    inflected = {
        "word": "winkel",
        "pos": "verb",
        "forms": [],
        "senses": [{"glosses": ["inflection of winkelen:"]}],
    }
    assert dutch.verb_from_kaikki(inflected) is None

    past_form = {
        "word": "at",
        "pos": "verb",
        "forms": [],
        "senses": [{"glosses": ["singular past indicative of eten"]}],
    }
    assert dutch.verb_from_kaikki(past_form) is None


def test_unknown_pos_is_skipped():
    assert dutch.word_from_kaikki({"pos": "name", "word": "Amsterdam", "senses": [{}]}) is None


def test_convert_iterables_routes_and_enriches():
    synonyms = {"huis": ["woning"]}
    words, verbs = dutch.convert_iterables([NOUN, STRONG_VERB], synonyms=synonyms)
    assert [w.id for w in words] == ["huis|noun"]
    assert words[0].synonyms == ["woning"]
    assert [v.id for v in verbs] == ["lopen"]


def test_convert_iterables_reassigns_cefr_by_budget():
    """With budgets, CEFR is assigned by cumulative frequency (NT2Lex = floor)."""
    from course_compiler.models import Frequency

    # huis is the more frequent noun; lopen the verb. NT2Lex tags both A1 (floor).
    frequencies = {
        "huis": Frequency(rank=1),
        "lopen": Frequency(rank=2),
    }
    cefr = {"huis": "A1", "lopen": "A1"}
    words, verbs = dutch.convert_iterables(
        [NOUN, STRONG_VERB],
        frequencies=frequencies,
        cefr=cefr,
        budgets={"A1": 1, "A2": 2},
    )
    # Only one A1 slot: the most frequent item (huis) gets A1, lopen rolls to A2.
    assert words[0].cefr == "A1"
    assert verbs[0].cefr == "A2"


def test_convert_iterables_budget_floor_is_respected():
    """An item attested at B1 is never placed below B1 even if very frequent."""
    from course_compiler.models import Frequency

    frequencies = {"huis": Frequency(rank=1), "lopen": Frequency(rank=2)}
    cefr = {"huis": "B1", "lopen": "A1"}  # huis floored at B1
    words, verbs = dutch.convert_iterables(
        [NOUN, STRONG_VERB],
        frequencies=frequencies,
        cefr=cefr,
        budgets={"A1": 1, "A2": 2, "B1": 3},
    )
    assert words[0].cefr == "B1"  # floor respected despite top frequency
    assert verbs[0].cefr == "A1"


def test_convert_iterables_without_budgets_keeps_nt2lex_levels():
    """Back-compat: no budgets -> CEFR stays the NT2Lex-attested level."""
    cefr = {"huis": "A2", "lopen": "B1"}
    words, verbs = dutch.convert_iterables(
        [NOUN, STRONG_VERB], cefr=cefr
    )
    assert words[0].cefr == "A2"
    assert verbs[0].cefr == "B1"


def _noun(word: str) -> dict:
    return {"pos": "noun", "word": word, "senses": [{"glosses": ["x"]}]}


def test_transparent_compound_does_not_consume_budget_but_is_levelled():
    """koffiepot = koffie+pot: introduced (levelled) but frees a budget slot."""
    from course_compiler.models import Frequency

    entries = [_noun("koffie"), _noun("pot"), _noun("koffiepot"), _noun("thee")]
    frequencies = {
        "koffie": Frequency(rank=1),
        "pot": Frequency(rank=2),
        "koffiepot": Frequency(rank=3),
        "thee": Frequency(rank=4),
    }
    cefr = {w: "A1" for w in ["koffie", "pot", "koffiepot", "thee"]}

    words, _ = dutch.convert_iterables(
        entries,
        frequencies=frequencies,
        cefr=cefr,
        budgets={"A1": 3},
        linkers=("s", "en", "e", "n"),
    )
    levels = {w.lemma: w.cefr for w in words}
    # Compound is levelled from its parts (max of koffie/pot = A1), not counted.
    assert levels["koffiepot"] == "A1"
    # Because the compound didn't consume a slot, 'thee' still fits in the budget.
    assert levels["thee"] == "A1"
    assert levels["koffie"] == "A1"
    assert levels["pot"] == "A1"


def test_opaque_compound_still_consumes_budget():
    """handschoen is opaque (glove): it counts as a new word despite splitting."""
    from course_compiler.models import Frequency

    entries = [_noun("hand"), _noun("schoen"), _noun("handschoen"), _noun("muts")]
    frequencies = {
        "hand": Frequency(rank=1),
        "schoen": Frequency(rank=2),
        "handschoen": Frequency(rank=3),
        "muts": Frequency(rank=4),
    }
    cefr = {w: "A1" for w in ["hand", "schoen", "handschoen", "muts"]}

    words, _ = dutch.convert_iterables(
        entries,
        frequencies=frequencies,
        cefr=cefr,
        budgets={"A1": 3},
        linkers=("s", "en", "e", "n"),
        opaque={"handschoen"},
    )
    levels = {w.lemma: w.cefr for w in words}
    # handschoen counts, taking the 3rd A1 slot; the rarer 'muts' is excluded.
    assert levels["handschoen"] == "A1"
    assert levels["muts"] is None


def test_load_wordnet_synonyms(tmp_path):
    xml = tmp_path / "wn.xml"
    xml.write_text(
        """<LexicalResource><Lexicon>
        <LexicalEntry id="a" partOfSpeech="noun">
          <Lemma writtenForm="woning"/><Sense synset="s1"/>
        </LexicalEntry>
        <LexicalEntry id="b" partOfSpeech="noun">
          <Lemma writtenForm="huis"/><Sense synset="s1"/>
        </LexicalEntry>
        <LexicalEntry id="c" partOfSpeech="noun">
          <Lemma writtenForm="auto"/><Sense synset="s2"/>
        </LexicalEntry>
        </Lexicon></LexicalResource>""",
        encoding="utf-8",
    )
    syn = dutch.load_wordnet_synonyms(xml)
    assert syn["huis"] == ["woning"]
    assert syn["woning"] == ["huis"]
    assert "auto" not in syn  # no synset peers -> no synonyms


# --- POS-keyed lexicon + glosses + separable verbs (task 0023) ------------------

MORGEN_NOUN = {
    "pos": "noun",
    "word": "morgen",
    "head_templates": [{"name": "nl-noun", "args": {"1": "m"}}],
    "senses": [{"glosses": ["morning"]}],
}
MORGEN_ADV = {
    "pos": "adv",
    "word": "morgen",
    "senses": [{"glosses": ["tomorrow"]}],
}


def test_homograph_pos_entries_both_survive_with_composite_ids():
    words, _ = dutch.convert_iterables([MORGEN_NOUN, MORGEN_ADV])
    ids = sorted(w.id for w in words)
    assert ids == ["morgen|adverb", "morgen|noun"]
    by_id = {w.id: w for w in words}
    assert by_id["morgen|noun"].translations["en"] == "morning"
    assert by_id["morgen|adverb"].translations["en"] == "tomorrow"


def test_glosses_list_drops_usage_notes_but_translation_keeps_them():
    entry = {
        "pos": "verb",
        "word": "zijn",
        "forms": [{"form": "zijn", "tags": ["infinitive"]}],
        "senses": [
            {"glosses": ["to be"]},
            {"glosses": ["Used to form the perfect tense of some verbs"]},
            {"glosses": ["go"]},
        ],
    }
    verb = dutch.verb_from_kaikki(entry)
    assert verb.glosses == ["be", "go"]  # usage note filtered from candidates
    assert "Used to form" in verb.translations["en"]  # display string unchanged


def test_detect_separable_splits_known_prefix_and_stem():
    known = {"stellen", "staan", "nemen"}
    assert dutch.detect_separable("voorstellen", known) == ("voor", "stellen")
    assert dutch.detect_separable("opstaan", known) == ("op", "staan")
    assert dutch.detect_separable("begrijpen", known) is None  # be- not separable


def test_annotate_separable_verbs_flags_and_maps():
    from course_compiler.models import Verb

    verbs = [
        Verb(id="stellen", language="nl", lemma="stellen", infinitive="stellen"),
        Verb(
            id="voorstellen",
            language="nl",
            lemma="voorstellen",
            infinitive="voorstellen",
        ),
    ]
    mapping = dutch.annotate_separable_verbs(verbs)
    assert mapping == {"voorstellen": {"prefix": "voor", "stem": "stellen"}}
    assert verbs[1].separable is True
    assert verbs[1].prefix == "voor"
    assert verbs[0].separable is False
