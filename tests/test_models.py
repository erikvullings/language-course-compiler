"""Canonical model serialization behavior."""

from __future__ import annotations

import yaml

from course_compiler.models import Gender, PartOfSpeech, Word, to_yaml


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
