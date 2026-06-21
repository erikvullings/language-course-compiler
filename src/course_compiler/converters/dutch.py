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
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path

from course_compiler.compounds import (
    build_known_parts,
    is_derivable_with_known,
    split_with_known,
)
from course_compiler.frequency import load_frequencies
from course_compiler.leveling import CEFR_ORDER, LevelItem, assign_levels
from course_compiler.models import (
    Audio,
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

_NOISY_CATEGORY_PREFIXES = (
    "pages with",
    "dutch entries",
)

_NON_THEMATIC_TAGS = frozenset(
    {
        "singular",
        "plural",
        "first-person",
        "second-person",
        "third-person",
        "present",
        "past",
        "future",
        "imperative",
        "indicative",
        "subjunctive",
        "participle",
        "form-of",
        "neuter",
        "masculine",
        "feminine",
        "common",
        "not-comparable",
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


def _audio_url(entry: dict) -> str | None:
    """Best audio URL from a kaikki entry's sound objects."""
    for sound in entry.get("sounds") or []:
        for key in ("mp3_url", "ogg_url", "audio_url", "wav_url", "url"):
            value = sound.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        audio = sound.get("audio")
        if isinstance(audio, str) and audio.startswith(("http://", "https://")):
            return audio
    return None


def _extract_tags(entry: dict) -> list[str]:
    """Extract lightweight lexical tags from sense tags/categories."""
    tags: set[str] = set()

    for raw in entry.get("tags") or []:
        if isinstance(raw, str):
            norm = raw.strip().lower()
            if norm and norm not in _NON_THEMATIC_TAGS:
                tags.add(norm)

    for sense in entry.get("senses") or []:
        for raw in sense.get("tags") or []:
            if isinstance(raw, str):
                norm = raw.strip().lower()
                if norm and norm not in _NON_THEMATIC_TAGS:
                    tags.add(norm)

        for category in sense.get("categories") or []:
            if not isinstance(category, dict):
                continue
            name = category.get("name")
            if not isinstance(name, str):
                continue
            norm = name.strip().lower()
            if not norm or norm.startswith(_NOISY_CATEGORY_PREFIXES):
                continue
            tags.add(norm)

    return sorted(tags)


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
    audio_url = _audio_url(entry)

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
        audio=Audio(recorded=audio_url) if audio_url else None,
        tags=_extract_tags(entry),
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
        tags=_extract_tags(entry),
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


# Dutch linking morphemes for compound splitting (longest first for determinism).
DUTCH_LINKERS: tuple[str, ...] = ("en", "s", "e", "n")


def reassign_cefr_by_budget(
    words: list[Word],
    verbs: list[Verb],
    budgets: Mapping[str, int],
    *,
    linkers: Sequence[str] = (),
    opaque: Collection[str] = (),
) -> None:
    """Reassign each item's CEFR level by cumulative frequency budget, in place.

    The current ``cefr`` (the NT2Lex earliest-attested level) becomes the *floor*:
    an item is never placed below it but may roll forward when its level's budget
    is spent. Items beyond the highest budget have their ``cefr`` cleared (excluded
    from all levels). A noun and a verb sharing a form are distinct ``(lemma, pos)``
    items and consume budget independently (cf. task 0019).

    When ``linkers`` is given, *transparent compounds* (a word that splits into ≥2
    known lemmas, excluding any in ``opaque``) do **not** consume budget: the
    learner already knows the parts. Such a compound is still levelled — it becomes
    available at the highest level among its parts — so it can be introduced in a
    lesson without inflating the new-word count (cf. task 0018).
    """
    known = {obj.lemma for obj in (*words, *verbs)}
    # Build the candidate-part set ONCE; rebuilding it per word turns the compound
    # pass into O(items × lexicon) (tens of billions of ops on a full import).
    known_parts = build_known_parts(known) if linkers else frozenset()

    items: list[LevelItem] = []
    by_key: dict[str, Word | Verb] = {}
    lemma_of: dict[str, str] = {}
    transparent: set[str] = set()
    for kind, objects in (("w", words), ("v", verbs)):
        for index, obj in enumerate(objects):
            key = f"{kind}{index}"
            by_key[key] = obj
            lemma_of[key] = obj.lemma
            rank = obj.frequency.rank if obj.frequency else None
            items.append(LevelItem(key=key, rank=rank, floor=obj.cefr))
            if linkers and is_derivable_with_known(
                obj.lemma, known_parts, linkers=linkers, opaque=opaque
            ):
                transparent.add(key)

    # Transparent compounds are excluded from the budget; everything else counts.
    counting = [item for item in items if item.key not in transparent]
    assigned = assign_levels(counting, budgets)

    lemma_level: dict[str, str] = {}
    for key, level in assigned.items():
        lemma = lemma_of[key]
        current = lemma_level.get(lemma)
        if current is None or CEFR_ORDER.index(level) > CEFR_ORDER.index(current):
            lemma_level[lemma] = level

    for key, obj in by_key.items():
        if key in transparent:
            continue
        obj.cefr = assigned.get(key)

    for key in transparent:
        obj = by_key[key]
        parts = split_with_known(obj.lemma, known_parts, linkers=linkers)
        part_levels = [lemma_level[p] for p in parts if p in lemma_level]
        if part_levels:
            obj.cefr = max(part_levels, key=CEFR_ORDER.index)
        # else: a part has no level — leave the original (floor) cefr untouched.


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
    budgets: Mapping[str, int] | None = None,
    detect_compounds: bool = False,
    opaque: Collection[str] = (),
    limit: int | None = None,
) -> dict[str, int]:
    """Convert Dutch sources to YAML entries plus aggregate JSON indexes.

    Returns counts ``{"words": n, "verbs": m}``. ``limit`` caps the number of
    processed kaikki entries (useful for smoke tests). When ``budgets`` is given,
    CEFR levels are reassigned by cumulative frequency budget (NT2Lex = floor);
    with ``detect_compounds`` transparent Dutch compounds are introduced without
    consuming budget (``opaque`` lists compounds to keep counting).
    """

    out = Path(output_dir)
    words_dir = out / "words"
    verbs_dir = out / "verbs"
    words_dir.mkdir(parents=True, exist_ok=True)
    verbs_dir.mkdir(parents=True, exist_ok=True)

    freqs = load_frequencies(frequency_path) if frequency_path else {}
    synonyms = load_wordnet_synonyms(wordnet_path) if wordnet_path else {}
    cefr = load_cefr_levels(nt2lex_path) if nt2lex_path else {}

    seen_words: set[str] = set()
    seen_verbs: set[str] = set()
    words: list[Word] = []
    verbs: list[Verb] = []
    audio_json: dict[str, str] = {}

    # Scan first so the budget pass (if any) sees the whole lexicon before levels
    # are assigned; YAML/JSON are written afterwards with the final CEFR levels.
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
            verbs.append(verb)
            if audio := _audio_url(entry):
                audio_json[verb.id] = audio
        else:
            word = word_from_kaikki(entry, frequency=freq, cefr=level)
            if word is None or word.id in seen_words:
                continue
            if key in synonyms:
                word.synonyms = synonyms[key]
            seen_words.add(word.id)
            words.append(word)
            if audio := _audio_url(entry):
                audio_json[word.id] = audio

    if budgets:
        reassign_cefr_by_budget(
            words,
            verbs,
            budgets,
            linkers=DUTCH_LINKERS if detect_compounds else (),
            opaque=opaque,
        )

    words_json: list[dict] = []
    verbs_json: list[dict] = []
    for verb in verbs:
        (verbs_dir / f"{_safe_name(verb.id)}.yaml").write_text(to_yaml(verb), encoding="utf-8")
        verbs_json.append(
            _compact_aggregate_row(
                verb.model_dump(by_alias=True, exclude_none=True, mode="json")
            )
        )
    for word in words:
        (words_dir / f"{_safe_name(word.id)}.yaml").write_text(to_yaml(word), encoding="utf-8")
        words_json.append(
            _compact_aggregate_row(
                word.model_dump(by_alias=True, exclude_none=True, mode="json")
            )
        )

    counts = {"words": len(words), "verbs": len(verbs)}
    _write_json_array(out / "words.json", words_json)
    _write_json_array(out / "verbs.json", verbs_json)
    _write_json_object(out / "audio.json", audio_json)

    return counts


def _write_json_array(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: str(row.get("id", "")))
    path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_object(path: Path, rows: dict[str, str]) -> None:
    ordered = {key: rows[key] for key in sorted(rows)}
    path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_aggregate_row(row: dict) -> dict:
    """Compact aggregate JSON payloads by dropping empty/default-style values."""

    def _compact(value):
        if isinstance(value, dict):
            out: dict = {}
            for key, child in value.items():
                if key == "language":
                    continue
                compacted = _compact(child)
                if compacted in (None, [], {}):
                    continue
                out[key] = compacted
            return out
        if isinstance(value, list):
            out_list = [_compact(item) for item in value]
            out_list = [item for item in out_list if item not in (None, [], {})]
            return out_list
        return value

    compacted_row = _compact(row)
    return compacted_row if isinstance(compacted_row, dict) else row


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
    budgets: Mapping[str, int] | None = None,
    linkers: Sequence[str] = (),
    opaque: Collection[str] = (),
) -> tuple[list[Word], list[Verb]]:
    """In-memory variant of :func:`convert` for tests: returns models, no I/O.

    When ``budgets`` is given, CEFR levels are reassigned by cumulative frequency
    budget (the NT2Lex ``cefr`` becomes the floor); otherwise the NT2Lex level is
    kept as-is. ``linkers`` enables transparent-compound detection so those words
    don't consume budget (see :func:`reassign_cefr_by_budget`).
    """

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

    if budgets:
        reassign_cefr_by_budget(
            words, verbs, budgets, linkers=linkers, opaque=opaque
        )
    return words, verbs
