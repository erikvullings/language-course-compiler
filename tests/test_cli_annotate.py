"""`course annotate` adds tokens[] + vocabulary[] to existing lesson JSON."""

from __future__ import annotations

import json
import re

from course_compiler.cli import main
from course_compiler.models import Gender, PartOfSpeech
from course_compiler.nlp.base import PosTagger, TaggedDoc, TokenTag, register_tagger

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+|[.!?]")
_SPEC = {
    "de": ("de", PartOfSpeech.DETERMINER, "DET"),
    "morgen": ("morgen", PartOfSpeech.NOUN, "NOUN"),
    "is": ("zijn", PartOfSpeech.VERB, "VERB"),
    "mooi": ("mooi", PartOfSpeech.ADJECTIVE, "ADJ"),
}


class _FakeTagger(PosTagger):
    @property
    def language(self) -> str:
        return "tt"

    def tag(self, text: str) -> TaggedDoc:
        tokens = []
        for m in _TOKEN_RE.finditer(text):
            s = m.group(0)
            if s in ".!?":
                tokens.append(TokenTag(s, m.start(), m.end(), s, None, "PUNCT"))
                continue
            lemma, pos, upos = _SPEC.get(s.lower(), (s.lower(), None, "X"))
            tokens.append(TokenTag(s, m.start(), m.end(), lemma, pos, upos))
        return TaggedDoc(tokens=tokens)

    def article_for_gender(self, gender):
        if gender in (Gender.MASCULINE, Gender.FEMININE, Gender.COMMON):
            return "de"
        return "het" if gender is Gender.NEUTER else None


def _setup_course(tmp_path):
    course = tmp_path / "courses" / "tt"
    (course).mkdir(parents=True)
    (course / "words.json").write_text(
        json.dumps(
            [
                {"id": "morgen|noun", "lemma": "morgen", "normalized": "morgen",
                 "partOfSpeech": "noun", "glosses": ["morning"], "gender": "m"},
                {"id": "morgen|adverb", "lemma": "morgen", "normalized": "morgen",
                 "partOfSpeech": "adverb", "glosses": ["tomorrow"]},
                {"id": "mooi|adjective", "lemma": "mooi", "normalized": "mooi",
                 "partOfSpeech": "adjective", "glosses": ["beautiful"]},
            ]
        ),
        encoding="utf-8",
    )
    (course / "verbs.json").write_text(
        json.dumps(
            [{"id": "zijn", "lemma": "zijn", "infinitive": "zijn",
              "glosses": ["be"], "present": {"hij": "is"}}]
        ),
        encoding="utf-8",
    )
    lessons = course / "lessons" / "A1"
    lessons.mkdir(parents=True)
    (lessons / "lesson001.json").write_text(
        json.dumps(
            {
                "id": "lesson001",
                "language": "tt",
                "cefr": "A1",
                "title": "Test",
                "text": "De morgen is mooi.",
                "newWords": ["morgen", "mooi", "zijn"],
            }
        ),
        encoding="utf-8",
    )
    return course, lessons


def test_annotate_writes_tokens_and_vocabulary(tmp_path):
    register_tagger("tt", lambda language: _FakeTagger())
    course, lessons = _setup_course(tmp_path)

    rc = main(
        [
            "annotate",
            "--lang", "tt",
            "--cefr", "A1",
            "--course-dir", str(course),
            "--no-llm-senses",
        ]
    )
    assert rc == 0

    payload = json.loads((lessons / "lesson001.json").read_text(encoding="utf-8"))

    # Token stream round-trips the original text exactly.
    tokens = payload["tokens"]
    rebuilt = "".join(t if isinstance(t, str) else t["w"] for t in tokens)
    assert rebuilt == "De morgen is mooi."

    linked = {t["w"]: t for t in tokens if isinstance(t, dict)}
    assert linked["morgen"]["ref"] == "morgen|noun"
    assert linked["morgen"]["gloss"] == "morning"
    assert linked["is"]["ref"] == "zijn" and linked["is"]["pos"] == "verb"
    assert linked["mooi"]["ref"] == "mooi|adjective"

    vocab = {w["lemma"]: w for w in payload["vocabulary"]}
    assert vocab["morgen"]["ref"] == "morgen|noun"
    assert vocab["morgen"]["article"] == "de"  # from gender via tagger
    assert vocab["zijn"]["ref"] == "zijn" and vocab["zijn"]["pos"] == "verb"


def test_annotate_only_filter_targets_one_lesson(tmp_path):
    register_tagger("tt", lambda language: _FakeTagger())
    course, lessons = _setup_course(tmp_path)
    # A second lesson that must remain untouched by --only.
    (lessons / "lesson002.json").write_text(
        json.dumps(
            {
                "id": "lesson002",
                "language": "tt",
                "cefr": "A1",
                "title": "Two",
                "text": "De morgen is mooi.",
                "newWords": [],
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "annotate",
            "--lang", "tt",
            "--cefr", "A1",
            "--course-dir", str(course),
            "--no-llm-senses",
            "--only", "lesson001",
        ]
    )
    assert rc == 0

    one = json.loads((lessons / "lesson001.json").read_text(encoding="utf-8"))
    two = json.loads((lessons / "lesson002.json").read_text(encoding="utf-8"))
    assert one.get("tokens")
    assert "tokens" not in two  # untouched


def test_annotate_applies_meta_yaml_overrides(tmp_path):
    register_tagger("tt", lambda language: _FakeTagger())
    course = tmp_path / "courses" / "tt"
    course.mkdir(parents=True)
    (course / "words.json").write_text(
        json.dumps(
            [{"id": "groet|noun", "lemma": "groet", "normalized": "groet",
              "partOfSpeech": "noun", "glosses": ["greeting"]}]
        ),
        encoding="utf-8",
    )
    (course / "verbs.json").write_text(
        json.dumps([{"id": "groeten", "lemma": "groeten", "infinitive": "groeten",
                     "glosses": ["greet"]}]),
        encoding="utf-8",
    )
    lessons = course / "lessons" / "A1"
    lessons.mkdir(parents=True)
    (lessons / "lesson001.json").write_text(
        json.dumps(
            {"id": "lesson001", "language": "tt", "cefr": "A1", "title": "T",
             "text": "Ik groet je.", "newWords": ["groet"]}
        ),
        encoding="utf-8",
    )
    # meta sidecar forces the surface "groet" to link to the verb groeten.
    (lessons / "lesson001.meta.yaml").write_text(
        "linkAs:\n  groet: groeten\n", encoding="utf-8"
    )

    rc = main(
        ["annotate", "--lang", "tt", "--cefr", "A1",
         "--course-dir", str(course), "--no-llm-senses"]
    )
    assert rc == 0

    payload = json.loads((lessons / "lesson001.json").read_text(encoding="utf-8"))
    linked = {t["w"]: t for t in payload["tokens"] if isinstance(t, dict)}
    assert linked["groet"]["ref"] == "groeten"
    assert linked["groet"]["pos"] == "verb"
