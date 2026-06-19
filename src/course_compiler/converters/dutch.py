"""Dutch importer: kaikki.org Wiktionary + Open Dutch WordNet + wordfreq.

This module is intentionally Dutch-specific. It maps three open datasets onto
the canonical models in :mod:`course_compiler.models`:

* **kaikki.org Dutch JSONL** (machine-readable English Wiktionary, Dutch words) --
  the primary source: part of speech, gender, plural/diminutive, IPA, syllables,
  verb conjugation tables, and English glosses (used as ``translations.en``).
* **Open Dutch WordNet (LMF XML)** -- synonyms, derived from lemmas that share a
  WordNet synset.
* **wordfreq cBpack** -- frequency rank (see :mod:`course_compiler.frequency`).

The public entry points are the per-entry mappers (:func:`word_from_kaikki`,
:func:`verb_from_kaikki`) -- pure and unit-testable on a single decoded entry --
and :func:`convert`, which streams the files and writes YAML.
"""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from pathlib import Path

from course_compiler.frequency import load_frequencies
from course_compiler.models import (
    Diminutive,
    Frequency,
    Gender,
    PartOfSpeech,
    Plural,
    Verb,
    Word,
    to_yaml,
)

LANGUAGE = "nl"

# kaikki part-of-speech -> canonical. Unlisted values are skipped.
_POS_MAP: dict[str, PartOfSpeech] = {
    "noun": PartOfSpeech.NOUN,
    "verb": PartOfSpeech.VERB,
    "adj": PartOfSpeech.ADJECTIVE,
    "adv": PartOfSpeech.ADVERB,
    "pron": PartOfSpeech.PRONOUN,
    "prep": PartOfSpeech.PREPOSITION,
    "conj": PartOfSpeech.CONJUNCTION,
    "article": PartOfSpeech.ARTICLE,
    "num": PartOfSpeech.NUMERAL,
    "intj": PartOfSpeech.INTERJECTION,
    "det": PartOfSpeech.DETERMINER,
}

_GENDER_MAP: dict[str, Gender] = {
    "m": Gender.MASCULINE,
    "masculine": Gender.MASCULINE,
    "f": Gender.FEMININE,
    "feminine": Gender.FEMININE,
    "n": Gender.NEUTER,
    "neuter": Gender.NEUTER,
    "c": Gender.COMMON,
    "common": Gender.COMMON,
    "common-gender": Gender.COMMON,
}

# Form tags that mark a non-canonical variant we don't want as the headword form.
_MARKED = frozenset(
    {
        "archaic",
        "obsolete",
        "dated",
        "rare",
        "dialectal",
        "Flanders",
        "colloquial",
        "majestic",
        "alternative",
        "table-tags",
        "inflection-template",
        "class",
    }
)


def normalize(lemma: str) -> str:
    """Canonical key for a lemma (Dutch keeps diacritics, so just casefold)."""

    return lemma.strip().lower()


def _find_form(
    forms: list[dict], include: set[str], exclude: AbstractSet[str] = frozenset()
) -> str | None:
    """First form whose tags include all of ``include`` and none of ``exclude``."""

    blocked = set(exclude) | _MARKED
    blocked -= include  # an explicitly required tag is never blocking
    for form in forms:
        tags = set(form.get("tags", ()))
        if include <= tags and not (tags & blocked):
            value = form.get("form")
            if value:
                return value
    return None


def _all_forms(
    forms: list[dict], include: set[str], exclude: AbstractSet[str] = frozenset()
) -> list[str]:
    blocked = (set(exclude) | _MARKED) - include
    out: list[str] = []
    for form in forms:
        tags = set(form.get("tags", ()))
        if include <= tags and not (tags & blocked):
            value = form.get("form")
            if value and value not in out:
                out.append(value)
    return out


def _ipa(entry: dict) -> str | None:
    sounds = entry.get("sounds") or []
    # Prefer a phonemic transcription (/.../), else any IPA.
    for sound in sounds:
        ipa = sound.get("ipa")
        if ipa and ipa.startswith("/"):
            return ipa
    for sound in sounds:
        if sound.get("ipa"):
            return sound["ipa"]
    return None


def _syllables(entry: dict) -> list[str]:
    hyph = entry.get("hyphenations") or []
    if hyph and hyph[0].get("parts"):
        return list(hyph[0]["parts"])
    return []


def _gender(entry: dict) -> Gender | None:
    for tmpl in entry.get("head_templates") or []:
        if tmpl.get("name", "").startswith("nl-noun"):
            raw = str(tmpl.get("args", {}).get("1", "")).lower()
            if raw in _GENDER_MAP:
                return _GENDER_MAP[raw]
    for sense in entry.get("senses") or []:
        for tag in sense.get("tags") or []:
            if tag in _GENDER_MAP:
                return _GENDER_MAP[tag]
    return None


_ARTICLE_PREFIXES = ("a ", "an ", "the ", "to ")


def _english_translation(entry: dict) -> str | None:
    """Best single English equivalent, taken from the first usable gloss."""

    for sense in entry.get("senses") or []:
        for gloss in sense.get("glosses") or []:
            text = gloss.strip()
            if not text:
                continue
            lowered = text.lower()
            for prefix in _ARTICLE_PREFIXES:
                if lowered.startswith(prefix):
                    text = text[len(prefix) :]
                    break
            # Keep only the first listed sense fragment, e.g. "house, home" -> "house".
            return text.split(",")[0].split(";")[0].strip()
    return None


def word_from_kaikki(
    entry: dict, *, frequency: Frequency | None = None, cefr: str | None = None
) -> Word | None:
    """Map one decoded kaikki entry to a :class:`Word` (non-verbs).

    Returns ``None`` for entries that aren't usable as vocabulary words (unknown
    part of speech, proper nouns, or verbs -- use :func:`verb_from_kaikki`).
    """

    pos = _POS_MAP.get(entry.get("pos", ""))
    if pos is None or pos is PartOfSpeech.VERB:
        return None
    lemma = entry.get("word")
    if not lemma or not (entry.get("senses")):
        return None

    forms = entry.get("forms") or []
    plural_forms = _all_forms(forms, {"plural"})
    dim_forms = _all_forms(forms, {"diminutive"})

    translation = _english_translation(entry)

    return Word(
        id=normalize(lemma),
        language=LANGUAGE,
        lemma=lemma,
        normalized=normalize(lemma),
        part_of_speech=pos,
        translations={"en": translation} if translation else {},
        gender=_gender(entry) if pos is PartOfSpeech.NOUN else None,
        plural=Plural(regular=plural_forms[0], alternatives=plural_forms[1:])
        if plural_forms
        else None,
        diminutive=Diminutive(regular=dim_forms[0], alternatives=dim_forms[1:])
        if dim_forms
        else None,
        ipa=_ipa(entry),
        syllables=_syllables(entry),
        frequency=frequency,
        cefr=cefr,
    )


# Dutch present-tense pronoun -> form-tag selectors.
_PRESENT_SLOTS: dict[str, tuple[set[str], set[str]]] = {
    "ik": ({"first-person", "present", "singular"}, {"formal"}),
    "jij": ({"second-person", "present", "singular"}, {"formal"}),
    "u": ({"formal", "second-person", "present", "singular"}, set()),
    "hij": ({"third-person", "present", "singular"}, set()),
    "wij": ({"plural", "present"}, set()),
}


def verb_from_kaikki(
    entry: dict, *, frequency: Frequency | None = None, cefr: str | None = None
) -> Verb | None:
    """Map one decoded kaikki verb entry to a :class:`Verb`."""

    if entry.get("pos") != "verb":
        return None
    lemma = entry.get("word")
    if not lemma:
        return None
    forms = entry.get("forms") or []

    infinitive = _find_form(forms, {"infinitive"}, {"gerund"}) or lemma

    present: dict[str, str] = {}
    for pronoun, (include, exclude) in _PRESENT_SLOTS.items():
        form = _find_form(forms, include, exclude)
        if form:
            present[pronoun] = form
    # In Dutch jullie/zij share the plural present form.
    if "wij" in present:
        present.setdefault("jullie", present["wij"])
        present.setdefault("zij", present["wij"])
    if "u" not in present and "hij" in present:
        present["u"] = present["hij"]

    past: dict[str, str] = {}
    if sg := _find_form(forms, {"past", "singular"}, {"subjunctive"}):
        past["singular"] = sg
    if pl := _find_form(forms, {"past", "plural"}, {"subjunctive"}):
        past["plural"] = pl

    perfect: dict[str, str] = {}
    if participle := _find_form(forms, {"participle", "past"}):
        perfect["participle"] = participle

    imperative: dict[str, str] = {}
    if imp := _find_form(forms, {"imperative", "singular"}):
        imperative["singular"] = imp

    # Strong verbs carry a "class" form tag; weak verbs are regular.
    irregular = any("class" in (f.get("tags") or ()) for f in forms)

    translation = _english_translation(entry)

    return Verb(
        id=normalize(lemma),
        language=LANGUAGE,
        lemma=lemma,
        infinitive=infinitive,
        translations={"en": translation} if translation else {},
        present=present,
        past=past,
        perfect=perfect,
        imperative=imperative,
        future={"infinitive": infinitive},
        irregular=irregular,
        frequency=frequency,
        cefr=cefr,
    )


def iter_kaikki(path: str | Path) -> Iterator[dict]:
    """Yield decoded JSON objects from a kaikki ``.jsonl`` dump."""

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# CEFR levels in ascending order, as graded by NT2Lex (no C2 in the resource).
_CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1")


def load_cefr_levels(path: str | Path) -> dict[str, str]:
    """Derive a CEFR level per lemma from an NT2Lex ``.tsv`` resource.

    A lemma's level is the *earliest* CEFR level at which any of its senses is
    attested (a non-``-`` raw frequency ``F@<level>``). Returns
    ``{normalized_lemma: "A1" | ... | "C1"}``.
    """

    levels: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        freq_cols = {lvl: f"F@{lvl}" for lvl in _CEFR_ORDER}
        for row in reader:
            lemma = (row.get("word") or "").strip()
            if not lemma:
                continue
            for lvl in _CEFR_ORDER:
                value = (row.get(freq_cols[lvl]) or "-").strip()
                if value and value != "-":
                    key = normalize(lemma)
                    current = levels.get(key)
                    if current is None or _CEFR_ORDER.index(lvl) < _CEFR_ORDER.index(current):
                        levels[key] = lvl
                    break
    return levels


def load_wordnet_synonyms(path: str | Path) -> dict[str, list[str]]:
    """Group Open Dutch WordNet lemmas by shared synset to derive synonyms.

    Returns ``{normalized_lemma: [synonym, ...]}`` (synonyms sorted, excluding
    the lemma itself).
    """

    synset_members: dict[str, set[str]] = defaultdict(set)
    lemma_synsets: dict[str, set[str]] = defaultdict(set)

    for _event, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "LexicalEntry":
            continue
        lemma_el = elem.find("Lemma")
        written = lemma_el.get("writtenForm") if lemma_el is not None else None
        if written:
            key = normalize(written)
            for sense in elem.findall("Sense"):
                synset = sense.get("synset")
                if synset:
                    synset_members[synset].add(key)
                    lemma_synsets[key].add(synset)
        elem.clear()

    synonyms: dict[str, list[str]] = {}
    for lemma, synsets in lemma_synsets.items():
        related: set[str] = set()
        for synset in synsets:
            related |= synset_members[synset]
        related.discard(lemma)
        if related:
            synonyms[lemma] = sorted(related)
    return synonyms


def convert(
    kaikki_path: str | Path,
    output_dir: str | Path,
    *,
    wordnet_path: str | Path | None = None,
    frequency_path: str | Path | None = None,
    nt2lex_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Convert Dutch sources to YAML entries plus aggregate JSON indexes.

    Returns counts ``{"words": n, "verbs": m}``. ``limit`` caps the number of
    processed kaikki entries (useful for smoke tests).
    """

    out = Path(output_dir)
    words_dir = out / "words"
    verbs_dir = out / "verbs"
    words_dir.mkdir(parents=True, exist_ok=True)
    verbs_dir.mkdir(parents=True, exist_ok=True)

    freqs = load_frequencies(frequency_path) if frequency_path else {}
    synonyms = load_wordnet_synonyms(wordnet_path) if wordnet_path else {}
    cefr = load_cefr_levels(nt2lex_path) if nt2lex_path else {}

    counts = {"words": 0, "verbs": 0}
    seen_words: set[str] = set()
    seen_verbs: set[str] = set()
    words_json: list[dict] = []
    verbs_json: list[dict] = []

    for i, entry in enumerate(iter_kaikki(kaikki_path)):
        if limit is not None and i >= limit:
            break
        key = normalize(entry.get("word", ""))
        freq = freqs.get(key)
        level = cefr.get(key)

        if entry.get("pos") == "verb":
            verb = verb_from_kaikki(entry, frequency=freq, cefr=level)
            if verb is None or verb.id in seen_verbs:
                continue
            seen_verbs.add(verb.id)
            (verbs_dir / f"{_safe_name(verb.id)}.yaml").write_text(to_yaml(verb), encoding="utf-8")
            verbs_json.append(verb.model_dump(by_alias=True, exclude_none=True, mode="json"))
            counts["verbs"] += 1
        else:
            word = word_from_kaikki(entry, frequency=freq, cefr=level)
            if word is None or word.id in seen_words:
                continue
            if key in synonyms:
                word.synonyms = synonyms[key]
            seen_words.add(word.id)
            (words_dir / f"{_safe_name(word.id)}.yaml").write_text(to_yaml(word), encoding="utf-8")
            words_json.append(word.model_dump(by_alias=True, exclude_none=True, mode="json"))
            counts["words"] += 1

    _write_json_array(out / "words.json", words_json)
    _write_json_array(out / "verbs.json", verbs_json)

    return counts


def _write_json_array(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: str(row.get("id", "")))
    path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    """Filesystem-safe, collision-free file stem for an id.

    Already-safe ids (alphanumerics, ``-``, ``_``) keep their readable name.
    When sanitizing would change the id, a short stable hash of the original id
    is appended so two distinct ids can never collapse to the same filename
    (e.g. ``"co op"`` and ``"co.op"`` both sanitize to ``co_op``). Deterministic,
    so repeated runs produce identical filenames.
    """

    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    if stem == value and stem:
        return stem
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{stem or '_'}_{digest}"


def convert_iterables(
    entries: Iterable[dict],
    *,
    frequencies: dict[str, Frequency] | None = None,
    synonyms: dict[str, list[str]] | None = None,
    cefr: dict[str, str] | None = None,
) -> tuple[list[Word], list[Verb]]:
    """In-memory variant of :func:`convert` for tests: returns models, no I/O."""

    frequencies = frequencies or {}
    synonyms = synonyms or {}
    cefr = cefr or {}
    words: list[Word] = []
    verbs: list[Verb] = []
    for entry in entries:
        key = normalize(entry.get("word", ""))
        freq = frequencies.get(key)
        level = cefr.get(key)
        if entry.get("pos") == "verb":
            verb = verb_from_kaikki(entry, frequency=freq, cefr=level)
            if verb:
                verbs.append(verb)
        else:
            word = word_from_kaikki(entry, frequency=freq, cefr=level)
            if word:
                if key in synonyms:
                    word.synonyms = synonyms[key]
                words.append(word)
    return words, verbs
