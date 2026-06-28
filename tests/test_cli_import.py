"""The `course import` command writes canonical YAML to the output directory."""

from __future__ import annotations

import json

import yaml

from course_compiler.cli import _load_words_from_lexicon, main


def test_import_writes_word_and_verb_yaml(tmp_path):
    jsonl = tmp_path / "nl.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "pos": "noun",
                    "word": "kat",
                    "head_templates": [{"name": "nl-noun", "args": {"1": "c"}}],
                    "forms": [{"form": "katten", "tags": ["plural"]}],
                    "sounds": [
                        {
                            "audio": "Nl-kat.ogg",
                            "ogg_url": "https://example.test/Nl-kat.ogg",
                        }
                    ],
                    "senses": [{"glosses": ["a cat"]}],
                },
                {
                    "pos": "verb",
                    "word": "zijn",
                    "forms": [
                        {"form": "zijn", "tags": ["infinitive"]},
                        {"form": "geweest", "tags": ["participle", "past"]},
                    ],
                    "sounds": [
                        {
                            "audio": "Nl-zijn.ogg",
                            "mp3_url": "https://example.test/Nl-zijn.mp3",
                        }
                    ],
                    "senses": [{"glosses": ["to be"]}],
                },
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "courses" / "nl"

    rc = main(["import", "--language", "nl", "--kaikki", str(jsonl), "--out", str(out)])

    assert rc == 0
    word = yaml.safe_load((out / "words" / "kat.noun.yaml").read_text())
    assert word["translations"]["en"] == "cat"
    assert word["plural"]["regular"] == "katten"
    verb = yaml.safe_load((out / "verbs" / "zijn.yaml").read_text())
    assert verb["perfect"]["participle"] == "geweest"

    words_json = json.loads((out / "words.json").read_text(encoding="utf-8"))
    verbs_json = json.loads((out / "verbs.json").read_text(encoding="utf-8"))
    audio_json = json.loads((out / "audio.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in words_json] == ["kat|noun"]
    assert [entry["id"] for entry in verbs_json] == ["zijn"]
    # Compact aggregates omit per-entry language and empty arrays.
    assert "language" not in words_json[0]
    assert "language" not in verbs_json[0]
    assert "tags" not in words_json[0]
    assert "tags" not in verbs_json[0]
    assert audio_json == {
        "kat": "https://example.test/Nl-kat.ogg",
        "zijn": "https://example.test/Nl-zijn.mp3",
    }


def test_import_with_budgets_reassigns_cefr(tmp_path):
    """`--budgets` reassigns CEFR by cumulative frequency; NT2Lex is the floor."""
    from course_compiler.cli import _parse_budgets

    assert _parse_budgets("A1=1,A2=2") == {"A1": 1, "A2": 2}
    assert _parse_budgets(None) is None
    assert _parse_budgets("") is None

    jsonl = tmp_path / "nl.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"pos": "noun", "word": w, "senses": [{"glosses": ["x"]}]})
            for w in ["kat", "hond"]
        ),
        encoding="utf-8",
    )
    # NT2Lex: both A1. Frequency: kat more frequent than hond.
    nt2lex = tmp_path / "nt2lex.tsv"
    nt2lex.write_text(
        "word\tF@A1\tF@A2\tF@B1\tF@B2\tF@C1\n"
        "kat\t5\t-\t-\t-\t-\n"
        "hond\t5\t-\t-\t-\t-\n",
        encoding="utf-8",
    )
    out = tmp_path / "courses" / "nl"

    # Without a real cBpack we cannot rank; assert the flag is plumbed by checking
    # that a single-slot A1 budget forces one item up to A2 (deterministic by id).
    rc = main(
        [
            "import",
            "--kaikki",
            str(jsonl),
            "--nt2lex",
            str(nt2lex),
            "--out",
            str(out),
            "--budgets",
            "A1=1,A2=2",
        ]
    )

    assert rc == 0
    levels = {
        yaml.safe_load((out / "words" / f"{name}.noun.yaml").read_text())[
            "lemma"
        ]: yaml.safe_load((out / "words" / f"{name}.noun.yaml").read_text()).get("cefr")
        for name in ["kat", "hond"]
    }
    # Two A1-floored items, one A1 slot then one A2 slot: tie-break is deterministic.
    assert sorted(levels.values()) == ["A1", "A2"]


def test_lemmas_with_same_safe_stem_do_not_overwrite(tmp_path):
    # "a b" and "a.b" both sanitize to "a_b"; both must survive as files.
    jsonl = tmp_path / "nl.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"pos": "noun", "word": w, "senses": [{"glosses": ["x"]}]})
            for w in ["a b", "a.b"]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "courses" / "nl"

    rc = main(["import", "--kaikki", str(jsonl), "--out", str(out)])

    assert rc == 0
    written = sorted(p.name for p in (out / "words").glob("*.yaml"))
    assert len(written) == 2  # no collision/overwrite
    lemmas = {yaml.safe_load((out / "words" / n).read_text())["lemma"] for n in written}
    assert lemmas == {"a b", "a.b"}


def test_load_words_from_yaml_layout(tmp_path):
    out = tmp_path / "courses" / "nl"
    words_dir = out / "words"
    words_dir.mkdir(parents=True)
    (words_dir / "huis.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "huis",
                "language": "nl",
                "lemma": "huis",
                "normalized": "huis",
                "partOfSpeech": "noun",
                "translations": {"en": "house"},
                "cefr": "A1",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    words = _load_words_from_lexicon(out)

    assert len(words) == 1
    assert words[0].lemma == "huis"
    assert words[0].translations["en"] == "house"


def test_load_words_from_compact_json_layout_missing_required_fields(tmp_path):
    out = tmp_path / "courses" / "nl"
    out.mkdir(parents=True)
    (out / "words.json").write_text(
        json.dumps(
            [
                {
                    "id": "'er",
                    "lemma": "'er",
                    "translations": {"en": "abbreviation of der"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    words = _load_words_from_lexicon(out)

    assert len(words) == 1
    assert words[0].language == "nl"
    assert words[0].normalized == "'er"
    assert words[0].part_of_speech.value == "other"
