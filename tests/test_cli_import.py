"""The `course import` command writes canonical YAML to the output directory."""

from __future__ import annotations

import json

import yaml

from course_compiler.cli import main


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
                    "senses": [{"glosses": ["a cat"]}],
                },
                {
                    "pos": "verb",
                    "word": "zijn",
                    "forms": [
                        {"form": "zijn", "tags": ["infinitive"]},
                        {"form": "geweest", "tags": ["participle", "past"]},
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
    word = yaml.safe_load((out / "words" / "kat.yaml").read_text())
    assert word["translations"]["en"] == "cat"
    assert word["plural"]["regular"] == "katten"
    verb = yaml.safe_load((out / "verbs" / "zijn.yaml").read_text())
    assert verb["perfect"]["participle"] == "geweest"


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
