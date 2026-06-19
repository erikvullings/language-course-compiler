"""Canonical model serialization behavior."""

from __future__ import annotations

import yaml

from course_compiler.models import Gender, Lesson, PartOfSpeech, Word, to_yaml


def _word(**overrides) -> Word:
    base = dict(
        id="huis",
        language="nl",
        lemma="huis",
        normalized="huis",
        part_of_speech=PartOfSpeech.NOUN,
    )
    base.update(overrides)
    return Word(**base)


def test_yaml_uses_camelcase_keys():
    out = yaml.safe_load(to_yaml(_word()))
    assert out["partOfSpeech"] == "noun"
    assert "part_of_speech" not in out


def test_yaml_omits_unset_optional_fields():
    out = yaml.safe_load(to_yaml(_word()))
    # gender / ipa / frequency were never set
    assert "gender" not in out
    assert "ipa" not in out


def test_yaml_is_deterministic():
    word = _word(gender=Gender.NEUTER, translations={"en": "house"})
    assert to_yaml(word) == to_yaml(word)


def test_enum_values_serialize_as_plain_strings():
    out = yaml.safe_load(to_yaml(_word(gender=Gender.NEUTER)))
    assert out["gender"] == "n"


def test_lesson_model_serializes_to_json_shape():
    lesson = Lesson(
        id="lesson001",
        language="nl",
        cefr="A1",
        title="Home Lesson",
        text="Dit is een huis.",
        attempts=2,
        tolerated=["de"],
    )
    out = lesson.model_dump(by_alias=True, mode="json")
    assert out["id"] == "lesson001"
    assert out["language"] == "nl"
    assert out["cefr"] == "A1"
    assert out["title"] == "Home Lesson"
    assert out["text"] == "Dit is een huis."
    assert out["attempts"] == 2
    assert out["tolerated"] == ["de"]
