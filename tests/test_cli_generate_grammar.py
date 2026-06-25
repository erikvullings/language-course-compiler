"""CLI behavior for `course plan-grammar` and `course generate-grammar`."""

from __future__ import annotations

import json

import yaml

from course_compiler.cli import main
from course_compiler.llm.base import LLMProvider, LLMResponse, PromptInput, to_messages


class _GrammarProvider(LLMProvider):
    """Routes by prompt content: plan, lesson-planner, theme, and grammar writer."""

    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        messages = to_messages(prompt)
        text = " ".join(m.content for m in messages)

        if "grammar planner" in text:
            content = json.dumps(
                {
                    "topics": [
                        {
                            "id": "present-tense",
                            "title": "Present tense",
                            "dependsOn": [],
                            "introducedInLesson": 1,
                            "focus": "Regular -t / -en endings.",
                        },
                        {
                            "id": "articles",
                            "title": "Articles",
                            "dependsOn": ["present-tense"],
                            "introducedInLesson": 2,
                            "focus": "de vs het.",
                        },
                    ]
                }
            )
            return LLMResponse(content=content, model=model or "stub", raw={})

        if "grammar writer" in text:
            content = json.dumps(
                {
                    "title": "Present tense",
                    "description": "In Dutch verbs usually end in -t or -en.",
                    "rules": ["Add -t for hij/zij."],
                    "examples": ["het huis", "de deur"],
                    "commonMistakes": ["Forgetting -t."],
                    "exceptions": [],
                }
            )
            return LLMResponse(content=content, model=model or "stub", raw={})

        if "language-course planner" in text:
            content = json.dumps(
                {"lessons": [{"theme": "home", "seed_lemmas": ["huis", "deur"]}]}
            )
            return LLMResponse(content=content, model=model or "stub", raw={})

        # Theme clustering fallback.
        return LLMResponse(
            content=json.dumps({"home": ["huis", "deur"]}),
            model=model or "stub",
            raw={},
        )

    async def acomplete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        return self.complete(prompt, model=model, temperature=temperature, **kwargs)


def _write_lexicon(course_dir):
    course_dir.mkdir(parents=True)
    payload = [
        {
            "id": "huis",
            "language": "nl",
            "lemma": "huis",
            "normalized": "huis",
            "partOfSpeech": "noun",
            "cefr": "A1",
            "frequency": {"rank": 1},
        },
        {
            "id": "deur",
            "language": "nl",
            "lemma": "deur",
            "normalized": "deur",
            "partOfSpeech": "noun",
            "cefr": "A1",
            "frequency": {"rank": 2},
        },
    ]
    (course_dir / "words.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_grammar_catalog(path):
    path.write_text(
        yaml.safe_dump(
            {
                "A1": {
                    "present-tense": {
                        "title": "Present tense",
                        "dependsOn": [],
                        "introducedInLesson": 1,
                        "focus": "Regular -t / -en endings.",
                    },
                    "articles": {
                        "title": "Articles",
                        "dependsOn": ["present-tense"],
                        "introducedInLesson": 2,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_plan_grammar_writes_catalog(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider", lambda settings: _GrammarProvider()
    )
    out = tmp_path / "grammar" / "nl.yaml"

    rc = main(["plan-grammar", "--lang", "nl", "--cefr", "A1", "--out", str(out)])

    assert rc == 0
    catalog = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert set(catalog["A1"]) == {"present-tense", "articles"}
    assert catalog["A1"]["articles"]["dependsOn"] == ["present-tense"]


def test_generate_grammar_preview_lists_topics_in_order(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider", lambda settings: _GrammarProvider()
    )
    course_dir = tmp_path / "courses" / "nl"
    _write_lexicon(course_dir)
    catalog = tmp_path / "nl.yaml"
    _write_grammar_catalog(catalog)

    rc = main(
        [
            "generate-grammar",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--grammar-file",
            str(catalog),
            "--preview",
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert [t["id"] for t in rendered["topics"]] == ["present-tense", "articles"]


def test_generate_grammar_writes_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider", lambda settings: _GrammarProvider()
    )
    course_dir = tmp_path / "courses" / "nl"
    _write_lexicon(course_dir)
    catalog = tmp_path / "nl.yaml"
    _write_grammar_catalog(catalog)
    out_dir = tmp_path / "out"

    rc = main(
        [
            "generate-grammar",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--grammar-file",
            str(catalog),
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    page = json.loads((out_dir / "present-tense.json").read_text(encoding="utf-8"))
    assert page["id"] == "present-tense"
    assert page["language"] == "nl"
    assert page["cefr"] == "A1"
    assert page["title"] == "Present tense"
    assert "Dutch" in page["description"]
    assert page["examples"] == ["het huis", "de deur"]
    assert page["introducedInLesson"] == 1
    assert page["fallback"] is False
    assert (out_dir / "articles.json").exists()


def test_generate_grammar_missing_catalog_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider", lambda settings: _GrammarProvider()
    )
    course_dir = tmp_path / "courses" / "nl"
    _write_lexicon(course_dir)

    rc = main(
        [
            "generate-grammar",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--grammar-file",
            str(tmp_path / "missing.yaml"),
        ]
    )

    assert rc == 1
    assert "grammar catalog not found" in capsys.readouterr().err
