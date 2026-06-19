"""The `course export` command writes split JSON bundles."""

from __future__ import annotations

import json

import yaml

from course_compiler.cli import main


def test_export_writes_manifest_and_split_bundles(tmp_path):
    course_dir = tmp_path / "courses" / "nl"

    words_dir = course_dir / "words"
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    verbs_dir = course_dir / "verbs"
    verbs_dir.mkdir(parents=True)
    (verbs_dir / "lopen.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "lopen",
                "language": "nl",
                "lemma": "lopen",
                "infinitive": "lopen",
                "translations": {"en": "walk"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    lessons_dir = course_dir / "lessons"
    lessons_dir.mkdir(parents=True)
    (lessons_dir / "lesson001.txt").write_text("Dit is les 1.", encoding="utf-8")

    out_dir = tmp_path / "dist"
    rc = main(
        [
            "export",
            "--lang",
            "nl",
            "--course-dir",
            str(course_dir),
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["courseLanguage"] == "nl"
    assert manifest["version"] == "1.0"
    assert "compilerVersion" in manifest

    words = json.loads((out_dir / "words.json").read_text(encoding="utf-8"))
    assert words["huis"]["translations"]["en"] == "house"

    verbs = json.loads((out_dir / "verbs.json").read_text(encoding="utf-8"))
    assert verbs["lopen"]["translations"]["en"] == "walk"

    grammar = json.loads((out_dir / "grammar.json").read_text(encoding="utf-8"))
    assert grammar == {}

    exercises = json.loads((out_dir / "exercises.json").read_text(encoding="utf-8"))
    assert exercises == {}

    lesson = json.loads(
        (out_dir / "lessons" / "lesson001.json").read_text(encoding="utf-8")
    )
    assert lesson == {"id": "lesson001", "text": "Dit is les 1."}
