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
    assert word.id == "huis"
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


def test_unknown_pos_is_skipped():
    assert dutch.word_from_kaikki({"pos": "name", "word": "Amsterdam", "senses": [{}]}) is None


def test_convert_iterables_routes_and_enriches():
    synonyms = {"huis": ["woning"]}
    words, verbs = dutch.convert_iterables([NOUN, STRONG_VERB], synonyms=synonyms)
    assert [w.id for w in words] == ["huis"]
    assert words[0].synonyms == ["woning"]
    assert [v.id for v in verbs] == ["lopen"]


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
