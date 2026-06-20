"""CLI behavior for `course generate-lessons`."""

from __future__ import annotations

import json

import yaml

from course_compiler.cli import main
from course_compiler.llm.base import LLMProvider, LLMResponse, Message, PromptInput


class _GenerateLessonsProvider(LLMProvider):
    def complete(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        from course_compiler.llm.base import to_messages

        messages = to_messages(prompt)
        system = messages[0].content if messages else ""

        if "language-course planner" in system:
            content = json.dumps(
                {"lessons": [{"theme": "home", "seed_lemmas": ["huis", "deur"]}]}
            )
            return LLMResponse(content=content, model=model or "stub", raw={})

        if "language-learning content writer" in system:
            return LLMResponse(
                content="## Home Lesson\n**New words:** huis, deur\nhuis deur",
                model=model or "stub",
                raw={},
            )

        # Legacy theme clustering path fallback.
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


def _write_minimal_lexicon(course_dir):
    words_dir = course_dir / "words"
    words_dir.mkdir(parents=True)

    for lemma, rank in (("huis", 1), ("deur", 2)):
        (words_dir / f"{lemma}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": lemma,
                    "language": "nl",
                    "lemma": lemma,
                    "normalized": lemma,
                    "partOfSpeech": "noun",
                    "cefr": "A1",
                    "frequency": {"rank": rank},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )


def _write_minimal_lexicon_json(course_dir):
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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_themes_file(path, a1_theme: str):
    path.write_text(
        yaml.safe_dump(
            {
                "A1": {
                    "lesson001": {
                        "theme": a1_theme,
                        "communicativeGoals": ["goal1"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_generate_lessons_preview_prints_blueprint_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider",
        lambda settings: _GenerateLessonsProvider(),
    )

    course_dir = tmp_path / "courses" / "nl"
    _write_minimal_lexicon(course_dir)

    out_dir = tmp_path / "out"
    rc = main(
        [
            "generate-lessons",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--out",
            str(out_dir),
            "--preview",
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["lessonCount"] == 2
    assert rendered["lessons"][0]["theme"] == "Greetings"
    assert rendered["lessons"][0]["seedLemmas"] == ["huis"]
    assert rendered["lessons"][1]["theme"] == "Personal Information"
    assert rendered["lessons"][1]["seedLemmas"] == ["deur"]
    assert not out_dir.exists()


def test_generate_lessons_writes_json_files_with_lesson_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider",
        lambda settings: _GenerateLessonsProvider(),
    )

    course_dir = tmp_path / "courses" / "nl"
    _write_minimal_lexicon(course_dir)

    out_dir = tmp_path / "out"
    rc = main(
        [
            "generate-lessons",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--out",
            str(out_dir),
            "--preview",
            "--approve",
        ]
    )

    assert rc == 0

    payload = json.loads((out_dir / "lesson001.json").read_text(encoding="utf-8"))
    assert payload["id"] == "lesson001"
    assert payload["language"] == "nl"
    assert payload["cefr"] == "A1"
    assert "title" in payload
    assert "Home" in payload["title"]
    assert "huis" in payload["newWords"]
    assert payload["attempts"] >= 1
    assert isinstance(payload["tolerated"], list)
    assert not (out_dir / "lesson001.txt").exists()
    # With predefined A1 themes and 2 words, two lesson files are produced.
    assert (out_dir / "lesson002.json").exists()


def test_generate_lessons_preview_reads_words_json_layout(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider",
        lambda settings: _GenerateLessonsProvider(),
    )

    course_dir = tmp_path / "courses" / "nl"
    _write_minimal_lexicon_json(course_dir)

    rc = main(
        [
            "generate-lessons",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--preview",
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["lessonCount"] == 2
    assert rendered["lessons"][0]["seedLemmas"] == ["huis"]
    assert rendered["lessons"][1]["seedLemmas"] == ["deur"]


def test_generate_lessons_preview_uses_themes_file_override(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider",
        lambda settings: _GenerateLessonsProvider(),
    )

    course_dir = tmp_path / "courses" / "nl"
    _write_minimal_lexicon(course_dir)
    themes_file = tmp_path / "custom-themes.yaml"
    _write_themes_file(themes_file, "Custom A1 Theme")

    rc = main(
        [
            "generate-lessons",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--themes-file",
            str(themes_file),
            "--preview",
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["lessons"][0]["theme"] == "Custom A1 Theme"


def test_generate_lessons_preview_themes_file_basename_uses_bundled_catalog(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider",
        lambda settings: _GenerateLessonsProvider(),
    )

    course_dir = tmp_path / "courses" / "nl"
    _write_minimal_lexicon(course_dir)

    rc = main(
        [
            "generate-lessons",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--themes-file",
            "themes.yaml",
            "--preview",
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["lessons"][0]["theme"] == "Greetings"


def test_generate_lessons_preview_missing_themes_file_returns_error(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "course_compiler.cli.create_provider",
        lambda settings: _GenerateLessonsProvider(),
    )

    course_dir = tmp_path / "courses" / "nl"
    _write_minimal_lexicon(course_dir)

    rc = main(
        [
            "generate-lessons",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--lexicon",
            str(course_dir),
            "--themes-file",
            "missing-themes.yaml",
            "--preview",
        ]
    )

    assert rc == 1
    assert "themes file not found" in capsys.readouterr().err
