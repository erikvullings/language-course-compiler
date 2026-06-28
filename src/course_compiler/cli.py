"""Command-line entry point.

This is an initial skeleton. The full command surface described in
``INITIAL_INSTRUCTIONS.md`` (build, validate, generate-lessons, ...) is tracked
in ``TASKS/`` and will be added incrementally.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from course_compiler import __version__
from course_compiler.llm import create_provider
from course_compiler.settings import Settings

# BCP-47 → human-readable name used in LLM prompts.
_LANG_NAMES: dict[str, str] = {
    "nl": "Dutch",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
}


def _default_lexicon_language(lexicon_dir: Path, default: str | None = None) -> str:
    if default:
        return default
    guess = lexicon_dir.name.strip()
    return guess or "nl"


def _hydrate_compact_word_entry(entry: dict, *, default_language: str) -> dict:
    """Fill required Word fields when loading compact aggregate JSON rows."""
    hydrated = dict(entry)
    lemma = str(hydrated.get("lemma") or hydrated.get("id") or "")
    hydrated.setdefault("language", default_language)
    hydrated.setdefault("normalized", lemma)
    hydrated.setdefault("partOfSpeech", "other")
    hydrated.setdefault("translations", {})
    return hydrated


def _load_words_from_lexicon(lexicon_dir: Path, *, default_language: str | None = None):
    from course_compiler.models import Word

    words_json = lexicon_dir / "words.json"
    words_yaml_dir = lexicon_dir / "words"

    if words_json.exists():
        raw = json.loads(words_json.read_text(encoding="utf-8"))
        language = _default_lexicon_language(lexicon_dir, default_language)
        return [
            Word.model_validate(
                _hydrate_compact_word_entry(entry, default_language=language)
            )
            for entry in raw
        ]

    if words_yaml_dir.is_dir():
        word_files = sorted(
            [*words_yaml_dir.glob("*.yaml"), *words_yaml_dir.glob("*.yml")]
        )
        if not word_files:
            raise FileNotFoundError(f"no word entries found in {words_yaml_dir}")
        return [
            Word.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in word_files
        ]

    raise FileNotFoundError(f"neither {words_json} nor {words_yaml_dir} found")


def _load_separable_verbs(course_dir: Path) -> dict[str, dict[str, str]]:
    """Load ``separable-verbs.json`` (``{lemma: {prefix, stem}}``), or empty."""
    path = course_dir / "separable-verbs.json"
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _build_sense_picker(course_dir: Path, *, no_cache: bool):
    """Construct a cached LLM sense picker, or ``None`` if no provider is available."""
    try:
        from course_compiler.generation.cache import LLMCache
        from course_compiler.generation.sense import make_llm_sense_picker

        settings = Settings.load()
        provider = create_provider(settings)
    except Exception as exc:  # noqa: BLE001 - optional feature, degrade gracefully
        print(
            f"Warning: LLM sense fallback disabled ({exc}); POS tagging only.",
            file=sys.stderr,
        )
        return None
    cache = None if no_cache else LLMCache(course_dir / ".llm_cache")
    return make_llm_sense_picker(provider, model=None, cache=cache)


def _hydrate_compact_verb_entry(entry: dict, *, default_language: str) -> dict:
    """Fill required Verb fields when loading compact aggregate JSON rows."""
    hydrated = dict(entry)
    lemma = str(
        hydrated.get("lemma") or hydrated.get("infinitive") or hydrated.get("id") or ""
    )
    hydrated.setdefault("language", default_language)
    hydrated.setdefault("lemma", lemma)
    hydrated.setdefault("infinitive", lemma)
    hydrated.setdefault("translations", {})
    return hydrated


def _load_verbs_from_lexicon(lexicon_dir: Path, *, default_language: str | None = None):
    from course_compiler.models import Verb

    verbs_json = lexicon_dir / "verbs.json"
    verbs_yaml_dir = lexicon_dir / "verbs"

    if verbs_json.exists():
        raw = json.loads(verbs_json.read_text(encoding="utf-8"))
        language = _default_lexicon_language(lexicon_dir, default_language)
        return [
            Verb.model_validate(
                _hydrate_compact_verb_entry(entry, default_language=language)
            )
            for entry in raw
        ]

    if verbs_yaml_dir.is_dir():
        verb_files = sorted(
            [*verbs_yaml_dir.glob("*.yaml"), *verbs_yaml_dir.glob("*.yml")]
        )
        if not verb_files:
            return []
        return [
            Verb.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in verb_files
        ]

    return []


def _derive_title(text: str, fallback: str) -> str:
    def shorten(candidate: str) -> str:
        cleaned = candidate.strip()
        if not cleaned:
            return ""

        sentence = cleaned.split(".", 1)[0].split("!", 1)[0].split("?", 1)[0].strip()
        words = [w for w in sentence.split() if w]
        if len(words) > 8:
            return " ".join(words[:4])
        if len(words) > 6:
            return " ".join(words[:6])
        return sentence

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("TITLE:"):
            stripped = stripped[6:].strip()
        if stripped.upper().startswith("TEXT:"):
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped:
            return shorten(stripped) or fallback
    return shorten(fallback) or fallback


def _lesson_seed(level: str, lesson_id: str) -> int:
    """Deterministic 31-bit seed from CEFR level + lesson id."""
    payload = f"{level}:{lesson_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _audio_filename(word: str) -> str:
    """Normalize a word key into a filesystem-safe mp3 stem."""
    return word.strip().replace(" ", "_").replace("/", "_")


def _compose_audio_sample_text(title: str, text: str) -> str:
    """Build the spoken sample text, with title first and a brief pause."""
    body = text.strip()
    spoken_title = title.strip()
    if not spoken_title:
        return body

    if body.lower().startswith(spoken_title.lower()):
        return body

    if spoken_title[-1] not in ".!?":
        spoken_title = f"{spoken_title}."

    # Blank line yields a small pause for most TTS engines.
    return f"{spoken_title}\n\n{body}"


def _parse_budgets(spec: str | None) -> dict[str, int] | None:
    """Parse budgets like ``A1=2000,A2=4000`` into ``{level: cumulative}``."""
    if not spec:
        return None
    parsed: dict[str, int] = {}
    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid budget item: {chunk!r}")
        level_raw, count_raw = chunk.split("=", 1)
        level = level_raw.strip().upper()
        count = int(count_raw.strip())
        parsed[level] = count
    return parsed or None


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_lesson_payload(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"lesson payload must be an object: {path}")
    return loaded


def _load_entries_from_layout(lexicon_dir: Path, stem: str) -> dict[str, dict]:
    json_file = lexicon_dir / f"{stem}.json"
    yaml_dir = lexicon_dir / stem

    if json_file.exists():
        raw = json.loads(json_file.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return {
                str(entry["id"]): entry
                for entry in raw
                if isinstance(entry, dict) and "id" in entry
            }
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        return {}

    if not yaml_dir.is_dir():
        return {}

    entries: dict[str, dict] = {}
    files = sorted([*yaml_dir.glob("*.yaml"), *yaml_dir.glob("*.yml")])
    for path in files:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        entry_id = str(loaded.get("id") or path.stem)
        entries[entry_id] = loaded
    return entries


def _load_lessons_for_export(lessons_dir: Path) -> dict[str, dict]:
    if not lessons_dir.is_dir():
        return {}

    lessons: dict[str, dict] = {}

    for path in sorted(lessons_dir.glob("*.json")):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        lesson_id = str(loaded.get("id") or path.stem)
        loaded.setdefault("id", lesson_id)
        lessons[lesson_id] = loaded

    for path in sorted([*lessons_dir.glob("*.txt"), *lessons_dir.glob("*.md")]):
        lesson_id = path.stem
        if lesson_id in lessons:
            continue
        lessons[lesson_id] = {"id": lesson_id, "text": path.read_text(encoding="utf-8")}

    return lessons


def _load_fallback_lesson_ids(lessons_dir: Path) -> set[str]:
    """Return lesson ids marked with ``fallback: true`` in ``lessons_dir`` JSON files."""
    if not lessons_dir.is_dir():
        return set()

    fallback_ids: set[str] = set()
    for path in sorted(lessons_dir.glob("*.json")):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        if loaded.get("fallback") is not True:
            continue
        lesson_id = str(loaded.get("id") or path.stem)
        fallback_ids.add(lesson_id)
    return fallback_ids


def _parse_lesson_id_filter(raw: str) -> set[str]:
    """Parse ``--only`` values into a normalized lesson-id set.

    Accepts comma-separated ids and trims whitespace.
    Example: ``lesson003, lesson004`` -> {"lesson003", "lesson004"}.
    """
    result: set[str] = set()
    for part in raw.split(","):
        lesson_id = part.strip()
        if lesson_id:
            result.add(lesson_id)
    return result


def _load_level_lessons_for_export(
    lessons_dir: Path,
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Load level-scoped lessons from ``lessons/<LEVEL>/`` directories."""
    if not lessons_dir.is_dir():
        return {}, []

    result: dict[str, dict[str, dict]] = {}
    for child in sorted(lessons_dir.iterdir()):
        if not child.is_dir():
            continue
        level = child.name
        lesson_payloads = _load_lessons_for_export(child)
        if lesson_payloads:
            result[level] = lesson_payloads

    levels = sorted(result.keys())
    return result, levels


def _load_themes_catalog(path: Path) -> dict[str, dict[str, dict]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    result: dict[str, dict[str, dict]] = {}
    for level, lessons in loaded.items():
        if not isinstance(level, str) or not isinstance(lessons, dict):
            continue
        filtered: dict[str, dict] = {}
        for lesson_id, data in lessons.items():
            if isinstance(lesson_id, str) and isinstance(data, dict):
                filtered[lesson_id] = data
        if filtered:
            result[level] = filtered
    return result


def _lesson_blueprint(plans: Sequence[object]) -> dict[str, object]:
    lessons: list[dict[str, object]] = []
    for plan in plans:
        lesson_id = getattr(plan, "lesson_id", "")
        theme = getattr(plan, "theme", "misc")
        new_words = [w.lemma for w in getattr(plan, "new_words", [])]
        new_verbs = [v.infinitive for v in getattr(plan, "new_verbs", [])]
        seed_lemmas = new_words + new_verbs
        lessons.append(
            {
                "lessonId": lesson_id,
                "theme": theme,
                "seedLemmas": seed_lemmas,
            }
        )
    return {"lessonCount": len(lessons), "lessons": lessons}


def _cefr_below(cefr: str) -> set[str]:
    """Return CEFR levels strictly below *cefr* (already known to the learner)."""
    from course_compiler.generation.validator import CEFR_ORDER

    try:
        idx = CEFR_ORDER.index(cefr)
    except ValueError:
        return set()
    return set(CEFR_ORDER[:idx])


def _grammar_blueprint(plans: Sequence[object]) -> dict[str, object]:
    topics: list[dict[str, object]] = []
    for plan in plans:
        topic = getattr(plan, "topic", None)
        topics.append(
            {
                "id": getattr(topic, "id", ""),
                "title": getattr(topic, "title", ""),
                "cefr": getattr(topic, "cefr", ""),
                "dependsOn": list(getattr(topic, "depends_on", [])),
                "introducedInLesson": getattr(plan, "introduced_in_lesson", None),
            }
        )
    return {"topicCount": len(topics), "topics": topics}


def _allowed_for_topic(
    lesson_plans: Sequence[object],
    lesson_index: int | None,
    base_lemmas: set[str],
) -> tuple[set[str], set[str]]:
    """Vocabulary a grammar topic's examples may use at its lesson peg.

    Returns ``(allowed_content_lemmas, exempt_forms)``. ``base_lemmas`` are
    lower-level words the learner already knows; the lesson plan at *lesson_index*
    (1-based; clamped, defaulting to the last lesson) contributes the within-level
    vocabulary accumulated up to that point.
    """
    allowed = set(base_lemmas)
    forms: set[str] = set()
    if lesson_plans:
        idx = (lesson_index or len(lesson_plans)) - 1
        idx = max(0, min(idx, len(lesson_plans) - 1))
        plan = lesson_plans[idx]
        allowed |= getattr(plan, "allowed_lemmas", set())
        forms |= getattr(plan, "function_lemmas", set()) | getattr(
            plan, "allowed_forms", set()
        )
    return allowed, forms


def _grammar_plan_prompt(language: str, cefr: str) -> str:
    return (
        "You are a language-curriculum grammar planner. "
        f"Propose the grammar topics a {language} learner needs to reach CEFR "
        f"level {cefr}, ordered by dependency (foundational topics first). "
        "Each topic needs: a kebab-case id, a short English title, a list of "
        "prerequisite topic ids ('dependsOn', referencing earlier ids only), an "
        "'introducedInLesson' integer (roughly one new topic per lesson, "
        "increasing), and a one-sentence English 'focus' hint. "
        'Respond with strict JSON only: {"topics": [{"id": str, "title": str, '
        '"dependsOn": [str], "introducedInLesson": int, "focus": str}]}. '
        "No markdown fences, no commentary."
    )


def _parse_grammar_plan(raw: str) -> dict[str, dict]:
    """Parse an LLM grammar plan into an ordered ``{id: catalog_entry}`` mapping."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidates = [text] + ([fenced.group(1).strip()] if fenced else [])
    payload: dict | None = None
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            payload = loaded
            break
    if payload is None:
        return {}

    topics = payload.get("topics")
    if not isinstance(topics, list):
        return {}

    catalog: dict[str, dict] = {}
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic_id = str(item.get("id") or "").strip()
        if not topic_id:
            continue
        entry: dict[str, object] = {"title": str(item.get("title") or topic_id)}
        depends_on = item.get("dependsOn") or item.get("depends_on") or []
        entry["dependsOn"] = [str(d) for d in depends_on if str(d).strip()]
        lesson = item.get("introducedInLesson", item.get("introduced_in_lesson"))
        if lesson is not None:
            with contextlib.suppress(TypeError, ValueError):
                entry["introducedInLesson"] = int(lesson)
        focus = str(item.get("focus") or "").strip()
        if focus:
            entry["focus"] = focus
        catalog[topic_id] = entry
    return catalog


def _lesson_number(lesson_id: str) -> int | None:
    """Extract the trailing integer from a lesson id (``lesson007`` -> 7)."""
    match = re.search(r"(\d+)\s*$", lesson_id)
    return int(match.group(1)) if match else None


def _grammar_by_lesson(
    level_lessons: dict[str, dict[str, dict]],
    grammar: dict[str, dict],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Map each lesson to its newly-introduced and cumulatively-available grammar.

    Grammar is decoupled from lesson themes: only a handful of lessons introduce a
    new topic (via ``introducedInLesson``). Every other lesson maps to the grammar
    already available — all lower-level topics plus same-level topics taught up to
    that lesson — so the SPA can surface review/application material rather than a
    blank grammar slot. Returns ``{level: {lessonId: {"new": [...],
    "available": [...]}}}``.
    """
    from course_compiler.generation.validator import CEFR_ORDER

    def level_rank(level: str) -> int:
        try:
            return CEFR_ORDER.index(level)
        except ValueError:
            return len(CEFR_ORDER)

    topics = [g for g in grammar.values() if isinstance(g, dict) and g.get("id")]

    index: dict[str, dict[str, dict[str, list[str]]]] = {}
    for level, lessons in level_lessons.items():
        rank = level_rank(level)
        lower = sorted(
            str(g["id"]) for g in topics if level_rank(str(g.get("cefr", ""))) < rank
        )
        this_level = [g for g in topics if str(g.get("cefr", "")) == level]

        level_index: dict[str, dict[str, list[str]]] = {}
        for lesson_id in lessons:
            number = _lesson_number(lesson_id)
            new = sorted(
                str(g["id"])
                for g in this_level
                if g.get("introducedInLesson") == number and number is not None
            )
            available = lower + sorted(
                str(g["id"])
                for g in this_level
                if isinstance(g.get("introducedInLesson"), int)
                and number is not None
                and g["introducedInLesson"] <= number
            )
            level_index[lesson_id] = {"new": new, "available": available}
        index[level] = level_index
    return index


def _common_verbs_by_level(
    verbs: dict[str, dict], levels: list[str], *, limit: int = 20
) -> dict[str, list[str]]:
    """Most frequent verb ids per CEFR level, for review/conjugation drills.

    References ids only — the conjugation tables live in ``verbs.json`` (no data
    duplication). A learner reviewing grammar at a level can drill the common
    verbs available by then; the SPA pulls each verb's tables from ``verbs.json``.
    """

    def rank(entry: dict) -> tuple[int, str]:
        freq = entry.get("frequency")
        rank_value = freq.get("rank") if isinstance(freq, dict) else None
        return (
            rank_value if isinstance(rank_value, int) else 10**9,
            str(entry.get("id", "")),
        )

    result: dict[str, list[str]] = {}
    for level in levels:
        at_level = [
            v
            for v in verbs.values()
            if isinstance(v, dict) and str(v.get("cefr", "")) == level and v.get("id")
        ]
        result[level] = [str(v["id"]) for v in sorted(at_level, key=rank)[:limit]]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="course", description="Language Course Compiler"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Send a one-off prompt to the configured LLM")
    ask.add_argument("prompt", help="The prompt text")

    gen = sub.add_parser(
        "generate-lessons", help="Generate lessons from an imported lexicon"
    )
    gen.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    gen.add_argument("--cefr", default="A1", help="Target CEFR level (A1, A2, B1, …)")
    gen.add_argument(
        "--lexicon", default=None, help="Lexicon directory (defaults to courses/<lang>)"
    )
    gen.add_argument(
        "--language-name",
        default=None,
        help="LLM prompt name (defaults to known name for --lang)",
    )
    gen.add_argument(
        "--words-per-lesson", type=int, default=10, help="New content words per lesson"
    )
    gen.add_argument(
        "--out", default=None, help="Output directory (defaults to <lexicon>/lessons)"
    )
    gen.add_argument(
        "--preview",
        action="store_true",
        help="Print the computed lesson blueprint (count/themes/seed lemmas)",
    )
    gen.add_argument(
        "--approve",
        action="store_true",
        help="When used with --preview, continue to generation after printing the blueprint",
    )
    gen.add_argument(
        "--themes-file",
        default="themes.yaml",
        help="Theme catalog YAML path (default: bundled themes.yaml)",
    )
    gen.add_argument(
        "--regenerate-fallbacks",
        action="store_true",
        help="Regenerate only lessons previously marked with fallback=true",
    )
    gen.add_argument(
        "--only",
        default=None,
        help=(
            "Generate only these lesson ids (comma-separated), "
            "e.g. lesson003 or lesson003,lesson004"
        ),
    )
    gen.add_argument(
        "--retry-strategy",
        choices=["natural", "corrective"],
        default="natural",
        help=(
            "Retry mode: natural samples fresh drafts; corrective asks the model "
            "to rewrite using violation feedback"
        ),
    )
    gen.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk LLM cache for this run",
    )
    gen.add_argument(
        "--verbose",
        action="store_true",
        help="Print LLM prompts to stderr",
    )

    plan_g = sub.add_parser(
        "plan-grammar",
        help="Draft a per-language grammar catalog with an LLM (review before use)",
    )
    plan_g.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    plan_g.add_argument(
        "--cefr", default="A1", help="Target CEFR level (A1, A2, B1, …)"
    )
    plan_g.add_argument(
        "--language-name",
        default=None,
        help="LLM prompt name (defaults to known name for --lang)",
    )
    plan_g.add_argument(
        "--out",
        default=None,
        help="Catalog YAML path (defaults to grammar/<lang>.yaml)",
    )

    gen_g = sub.add_parser(
        "generate-grammar",
        help="Generate grammar pages from a grammar catalog and imported lexicon",
    )
    gen_g.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    gen_g.add_argument("--cefr", default="A1", help="Target CEFR level (A1, A2, B1, …)")
    gen_g.add_argument(
        "--lexicon", default=None, help="Lexicon directory (defaults to courses/<lang>)"
    )
    gen_g.add_argument(
        "--language-name",
        default=None,
        help="LLM prompt name (defaults to known name for --lang)",
    )
    gen_g.add_argument(
        "--words-per-lesson",
        type=int,
        default=10,
        help="New content words per lesson (must match the lesson run for pegging)",
    )
    gen_g.add_argument(
        "--grammar-file",
        default=None,
        help="Grammar catalog YAML (defaults to grammar/<lang>.yaml)",
    )
    gen_g.add_argument(
        "--themes-file",
        default="themes.yaml",
        help="Theme catalog YAML used to plan lesson vocabulary (default: themes.yaml)",
    )
    gen_g.add_argument(
        "--out",
        default=None,
        help="Output directory (defaults to <lexicon>/grammar/<cefr>)",
    )
    gen_g.add_argument(
        "--preview",
        action="store_true",
        help="Print the ordered grammar topics (id/title/lesson) and exit",
    )
    gen_g.add_argument(
        "--approve",
        action="store_true",
        help="When used with --preview, continue to generation after printing",
    )
    gen_g.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk LLM cache for this run",
    )

    imp = sub.add_parser("import", help="Import lexical sources into canonical YAML")
    imp.add_argument("--language", default="nl", choices=["nl"], help="Source language")
    imp.add_argument("--kaikki", required=True, help="Path to kaikki.org JSONL dump")
    imp.add_argument("--wordnet", help="Path to Open WordNet LMF XML (synonyms)")
    imp.add_argument("--frequency", help="Path to wordfreq cBpack file")
    imp.add_argument("--nt2lex", help="Path to NT2Lex .tsv resource (CEFR levels)")
    imp.add_argument("--out", default="courses/nl", help="Output course directory")
    imp.add_argument("--limit", type=int, help="Only process the first N entries")
    imp.add_argument(
        "--budgets",
        default=None,
        help="CEFR cumulative budgets, e.g. A1=2000,A2=4000,B1=6000",
    )

    exp = sub.add_parser("export", help="Export a course into split JSON bundles")
    exp.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    exp.add_argument(
        "--course-dir",
        default=None,
        help="Course directory (defaults to courses/<lang>)",
    )
    exp.add_argument(
        "--out", default=None, help="Output directory (defaults to <course-dir>/export)"
    )
    exp.add_argument(
        "--version", default="1.0", help="Course version for manifest.json"
    )

    img = sub.add_parser(
        "generate-images", help="Generate lesson illustration images from theme catalog"
    )
    img.add_argument(
        "--themes-file",
        default="themes.yaml",
        help="Theme catalog YAML path",
    )
    img.add_argument("--out", required=True, help="Output image directory")
    img.add_argument("--level", default=None, help="Optional CEFR level filter")
    img.add_argument("--force", action="store_true", help="Overwrite existing images")
    img.add_argument(
        "--no-llm-prompt",
        action="store_true",
        help="Use deterministic local prompt instead of LLM prompt generation",
    )

    aud = sub.add_parser("download-audio", help="Download audio files from audio.json")
    aud.add_argument("--audio-json", required=True, help="Path to audio.json mapping")
    aud.add_argument("--out", required=True, help="Output audio directory")
    aud.add_argument("--dry-run", action="store_true", help="Show actions only")
    aud.add_argument(
        "--limit", type=int, default=None, help="Maximum files to download"
    )
    aud.add_argument("--force", action="store_true", help="Overwrite existing files")

    gen_audio = sub.add_parser(
        "generate-audio",
        help="Generate mp3 + karaoke transcript for lesson JSON files using Voxtral",
    )
    gen_audio.add_argument(
        "--lang", required=True, help="BCP-47 language code (e.g. nl)"
    )
    gen_audio.add_argument(
        "--cefr", required=True, help="Target CEFR level (A1, A2, B1, ...)"
    )
    gen_audio.add_argument(
        "--course-dir",
        default=None,
        help="Course directory (defaults to courses/<lang>)",
    )
    gen_audio.add_argument(
        "--lessons-dir",
        default=None,
        help="Lessons directory (defaults to <course-dir>/lessons/<cefr>)",
    )
    gen_audio.add_argument(
        "--only",
        default=None,
        help="Generate audio only for lesson ids (comma-separated)",
    )
    gen_audio.add_argument(
        "--lesson-id",
        dest="lesson_id",
        default=None,
        help=argparse.SUPPRESS,
    )
    gen_audio.add_argument(
        "--voice",
        default="nl_female",
        help="Voxtral voice identifier",
    )
    gen_audio.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed",
    )
    gen_audio.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing mp3/transcript files",
    )
    gen_audio.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing mp3/transcript outputs and regenerate",
    )

    ann = sub.add_parser(
        "annotate",
        help="POS-tag lessons: add tokens[] + vocabulary[] to existing lesson JSON",
    )
    ann.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    ann.add_argument("--cefr", required=True, help="Target CEFR level (A1, A2, ...)")
    ann.add_argument(
        "--course-dir",
        default=None,
        help="Course directory (defaults to courses/<lang>)",
    )
    ann.add_argument(
        "--lessons-dir",
        default=None,
        help="Lessons directory (defaults to <course-dir>/lessons/<cefr>)",
    )
    ann.add_argument(
        "--only",
        default=None,
        help="Annotate only these lesson ids (comma-separated); useful after a manual edit",
    )
    ann.add_argument(
        "--no-llm-senses",
        action="store_true",
        help="Skip the LLM same-POS sense fallback (POS tagging only)",
    )
    ann.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the LLM sense cache",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        settings = Settings.load()
        provider = create_provider(settings)
        print(provider.complete(args.prompt).content)
        return 0

    if args.command == "generate-lessons":
        from course_compiler.generation.lesson import LessonGenerator
        from course_compiler.generation.orchestrator import LessonOrchestrator
        from course_compiler.generation.themes import LLMThemeAssigner
        from course_compiler.models import Lesson

        settings = Settings.load()
        provider = create_provider(settings)

        lexicon_dir = Path(args.lexicon or f"courses/{args.lang}")
        language_name = args.language_name or _LANG_NAMES.get(args.lang) or args.lang
        try:
            words = _load_words_from_lexicon(lexicon_dir, default_language=args.lang)
        except FileNotFoundError:
            print(
                "Error: neither "
                f"{lexicon_dir / 'words.json'} nor {lexicon_dir / 'words'} found. "
                "Run 'course import' first.",
                file=sys.stderr,
            )
            return 1

        verbs = _load_verbs_from_lexicon(lexicon_dir, default_language=args.lang)

        themes_path = Path(args.themes_file)
        if not themes_path.exists():
            print(f"Error: themes file not found: {themes_path}", file=sys.stderr)
            return 1

        # Theme assigner — LLM-backed with disk cache next to the lexicon.
        from course_compiler.generation.base import create_lemmatizer
        from course_compiler.generation.cache import LLMCache

        cache_dir = lexicon_dir / ".llm_cache"
        # Targeted regeneration must be fresh: ``--only`` always bypasses cache.
        disable_cache = args.no_cache or bool(args.only)
        cache = None if disable_cache else LLMCache(cache_dir)
        assigner = LLMThemeAssigner(
            provider,
            model=None,
            cache=cache,
            verbose=args.verbose,
        )

        lemmatizer = create_lemmatizer(args.lang)
        generator = LessonGenerator(
            provider,
            lemmatizer,
            cache=cache,
            # Coherence-first lesson generation: in-level extra words are tolerated
            # (only above-CEFR words remain violations).
            extra_tolerance=None,
            retry_strategy=args.retry_strategy,
            verbose=args.verbose,
        )
        orchestrator = LessonOrchestrator(
            generator,
            assigner,
            words_per_lesson=args.words_per_lesson,
            predefined_themes_path=themes_path,
        )

        plans = orchestrator.plan(
            words, cefr=args.cefr, verbs=verbs, language=language_name
        )
        if args.preview:
            print(json.dumps(_lesson_blueprint(plans), ensure_ascii=False, indent=2))
            if not args.approve:
                return 0

        out_dir = Path(args.out) if args.out else lexicon_dir / "lessons" / args.cefr
        out_dir.mkdir(parents=True, exist_ok=True)

        only_lesson_ids: set[str] | None = None
        if args.only:
            only_lesson_ids = _parse_lesson_id_filter(args.only)
            if not only_lesson_ids:
                print(
                    "Error: --only was provided but no valid lesson ids were parsed.",
                    file=sys.stderr,
                )
                return 1

        if args.regenerate_fallbacks:
            fallback_only = _load_fallback_lesson_ids(out_dir)
            if not fallback_only:
                print(f"No fallback lessons found in {out_dir}; nothing to regenerate.")
                return 0
            if only_lesson_ids is None:
                only_lesson_ids = fallback_only
            else:
                only_lesson_ids = only_lesson_ids & fallback_only
                if not only_lesson_ids:
                    print(
                        "No lessons matched both --only and --regenerate-fallbacks; "
                        "nothing to regenerate."
                    )
                    return 0

        print("Planning (deterministic, no LLM) and generating all lessons...")

        generated_count = 0
        for index, (plan, lesson) in enumerate(
            orchestrator.generate_iter(
                words,
                language=language_name,
                cefr=args.cefr,
                verbs=verbs,
                only=only_lesson_ids,
            ),
            start=1,
        ):
            generated_count += 1
            new_count = len(plan.new_words) + len(plan.new_verbs)
            status = (
                f"  [{index}] {plan.lesson_id} {plan.theme} "
                f"({new_count} new words, {lesson.attempts} attempt(s))"
            )

            if lesson.fallback:
                unresolved = sorted(lesson.violations)
                unresolved_str = ", ".join(unresolved)
                print(
                    f"[{plan.lesson_id}] No valid draft after {lesson.attempts} attempts; "
                    f"using best-effort draft (attempt {lesson.best_attempt}, "
                    f"{len(unresolved)} unresolved word(s): {unresolved})"
                )
                status += (
                    f" -- BEST-EFFORT ({len(unresolved)} unresolved word(s): "
                    f"{unresolved_str})"
                )
            elif lesson.attempts > 1 and lesson.diagnostics:
                first = lesson.diagnostics[0]
                last = lesson.diagnostics[-1]
                status += (
                    f" -- retried (violations: {len(first.violations)} -> "
                    f"{len(last.violations)})"
                )

            print(status)

            payload = Lesson(
                id=lesson.lesson_id,
                language=args.lang,
                cefr=args.cefr,
                title=lesson.title or _derive_title(lesson.content, plan.theme),
                theme=plan.theme,
                new_words=[w.lemma for w in plan.new_words]
                + [v.infinitive for v in plan.new_verbs],
                text=lesson.content,
                attempts=lesson.attempts,
                tolerated=sorted(lesson.tolerated),
                fallback=lesson.fallback,
                violations=sorted(lesson.violations),
            )
            _write_json(
                out_dir / f"{lesson.lesson_id}.json",
                payload.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )

        print(f"Generated {generated_count} lessons into {out_dir}")
        return 0

    if args.command == "plan-grammar":
        from course_compiler.generation.grammar import (
            GrammarDependencyError,
            GrammarProgressionPlanner,
            GrammarTopic,
        )

        settings = Settings.load()
        provider = create_provider(settings)
        language_name = args.language_name or _LANG_NAMES.get(args.lang) or args.lang
        out_path = Path(args.out) if args.out else Path("grammar") / f"{args.lang}.yaml"

        response = provider.complete(_grammar_plan_prompt(language_name, args.cefr))
        topics = _parse_grammar_plan(response.content)
        if not topics:
            print(
                "Error: could not parse a grammar plan from the LLM response.",
                file=sys.stderr,
            )
            return 1

        try:
            GrammarProgressionPlanner().plan(
                [
                    GrammarTopic(
                        id=tid,
                        language=args.lang,
                        title=str(data.get("title") or tid),
                        cefr=args.cefr,
                        depends_on=list(data.get("dependsOn", [])),
                    )
                    for tid, data in topics.items()
                ]
            )
        except GrammarDependencyError as exc:
            print(f"Error: invalid grammar plan from LLM: {exc}", file=sys.stderr)
            return 1

        catalog: dict = {}
        if out_path.exists():
            existing = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
            if isinstance(existing, dict):
                catalog = existing
        catalog[args.cefr] = topics

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"Wrote {len(topics)} grammar topics for {args.cefr} to {out_path}. "
            "Review and curate before running 'course generate-grammar'."
        )
        return 0

    if args.command == "generate-grammar":
        from course_compiler.generation.base import create_lemmatizer
        from course_compiler.generation.cache import LLMCache
        from course_compiler.generation.grammar import (
            GrammarDependencyError,
            load_grammar_catalog,
        )
        from course_compiler.generation.grammar_writer import GrammarWriter
        from course_compiler.generation.lesson import LessonGenerator
        from course_compiler.generation.orchestrator import (
            LessonOrchestrator,
            _tokens,
            _verb_surface_forms,
        )
        from course_compiler.generation.themes import LLMThemeAssigner

        settings = Settings.load()
        provider = create_provider(settings)
        lexicon_dir = Path(args.lexicon or f"courses/{args.lang}")
        language_name = args.language_name or _LANG_NAMES.get(args.lang) or args.lang

        grammar_path = (
            Path(args.grammar_file)
            if args.grammar_file
            else Path("grammar") / f"{args.lang}.yaml"
        )
        if not grammar_path.exists():
            print(
                f"Error: grammar catalog not found: {grammar_path}. "
                "Run 'course plan-grammar' first.",
                file=sys.stderr,
            )
            return 1

        try:
            words = _load_words_from_lexicon(lexicon_dir, default_language=args.lang)
        except FileNotFoundError:
            print(
                f"Error: neither {lexicon_dir / 'words.json'} nor "
                f"{lexicon_dir / 'words'} found. Run 'course import' first.",
                file=sys.stderr,
            )
            return 1
        verbs = _load_verbs_from_lexicon(lexicon_dir, default_language=args.lang)

        try:
            grammar_plans = load_grammar_catalog(
                grammar_path, language=args.lang, cefr=args.cefr
            )
        except GrammarDependencyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if not grammar_plans:
            print(f"No grammar topics for {args.cefr} in {grammar_path}.")
            return 0

        if args.preview:
            print(
                json.dumps(
                    _grammar_blueprint(grammar_plans), ensure_ascii=False, indent=2
                )
            )
            if not args.approve:
                return 0

        cache = None if args.no_cache else LLMCache(lexicon_dir / ".llm_cache")
        lemmatizer = create_lemmatizer(args.lang)

        # Plan the lesson sequence (no generation) so each grammar topic can be
        # pegged to the vocabulary allowed by its introducedInLesson index.
        themes_path = Path(args.themes_file)
        orchestrator = LessonOrchestrator(
            LessonGenerator(provider, lemmatizer, cache=cache, extra_tolerance=None),
            LLMThemeAssigner(provider, model=None, cache=cache),
            words_per_lesson=args.words_per_lesson,
            predefined_themes_path=themes_path if themes_path.exists() else None,
        )
        lesson_plans = orchestrator.plan(
            words, cefr=args.cefr, verbs=verbs, language=language_name
        )

        # CEFR lookup across all levels so example extras can be classified.
        cefr_lookup: dict[str, str] = {
            w.lemma: w.cefr for w in words if w.cefr is not None
        }
        for verb in verbs:
            if verb.cefr is not None:
                cefr_lookup[verb.infinitive] = verb.cefr
                for form in _verb_surface_forms(verb):
                    for token in _tokens(form):
                        cefr_lookup[token] = verb.cefr

        below = _cefr_below(args.cefr)
        base_lemmas = {w.lemma for w in words if w.cefr in below}
        base_lemmas |= {v.infinitive for v in verbs if v.cefr in below}

        writer = GrammarWriter(provider, lemmatizer, cache=cache)
        out_dir = Path(args.out) if args.out else lexicon_dir / "grammar" / args.cefr
        out_dir.mkdir(parents=True, exist_ok=True)

        for plan in grammar_plans:
            allowed, forms = _allowed_for_topic(
                lesson_plans, plan.introduced_in_lesson, base_lemmas
            )
            page = writer.generate(
                plan.topic,
                allowed_words=allowed,
                language=language_name,
                cefr=args.cefr,
                focus=plan.focus,
                introduced_in_lesson=plan.introduced_in_lesson,
                cefr_lookup=cefr_lookup,
                function_lemmas=forms,
            )
            _write_json(
                out_dir / f"{plan.topic.id}.json",
                page.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )
            marker = (
                f" -- BEST-EFFORT ({len(page.violations)} unresolved)"
                if page.fallback
                else ""
            )
            print(f"  {plan.topic.id} ({plan.topic.title}){marker}")

        print(f"Generated {len(grammar_plans)} grammar pages into {out_dir}")
        return 0

    if args.command == "import":
        # Language-specific importers live in course_compiler.converters.
        from course_compiler.converters import dutch

        try:
            budgets = _parse_budgets(args.budgets)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        counts = dutch.convert(
            args.kaikki,
            args.out,
            wordnet_path=args.wordnet,
            frequency_path=args.frequency,
            nt2lex_path=args.nt2lex,
            budgets=budgets,
            limit=args.limit,
        )
        print(
            f"Imported {counts['words']} words and {counts['verbs']} verbs into {args.out}"
        )
        return 0

    if args.command == "export":
        course_dir = Path(args.course_dir or f"courses/{args.lang}")
        out_dir = Path(args.out) if args.out else course_dir / "export"
        lessons_out = out_dir / "lessons"

        words = _load_entries_from_layout(course_dir, "words")
        verbs = _load_entries_from_layout(course_dir, "verbs")
        grammar = _load_entries_from_layout(course_dir, "grammar")
        exercises = _load_entries_from_layout(course_dir, "exercises")
        lessons = _load_lessons_for_export(course_dir / "lessons")
        level_lessons, levels = _load_level_lessons_for_export(course_dir / "lessons")

        manifest = {
            "courseLanguage": args.lang,
            "compilerVersion": __version__,
            "version": args.version,
            "levels": levels,
        }

        indices = {
            "grammarByLesson": _grammar_by_lesson(level_lessons, grammar),
            "commonVerbsByLevel": _common_verbs_by_level(verbs, levels),
        }

        _write_json(out_dir / "manifest.json", manifest)
        _write_json(out_dir / "words.json", words)
        _write_json(out_dir / "verbs.json", verbs)
        _write_json(out_dir / "grammar.json", grammar)
        _write_json(out_dir / "exercises.json", exercises)
        _write_json(out_dir / "indices.json", indices)
        if levels:
            for level in levels:
                for lesson_id, payload in sorted(level_lessons[level].items()):
                    merged = dict(payload)
                    merged["level"] = level
                    _write_json(lessons_out / level / f"{lesson_id}.json", merged)
        else:
            for lesson_id, payload in sorted(lessons.items()):
                _write_json(lessons_out / f"{lesson_id}.json", payload)

        print(
            f"Exported course bundles into {out_dir} "
            "(words="
            f"{len(words)}, verbs={len(verbs)}, grammar={len(grammar)}, "
            f"exercises={len(exercises)}, lessons={len(lessons)})"
        )
        return 0

    if args.command == "generate-images":
        import httpx

        themes_path = Path(args.themes_file)
        if not themes_path.exists():
            print(f"Error: themes file not found: {themes_path}", file=sys.stderr)
            return 1

        catalog = _load_themes_catalog(themes_path)
        out_dir = Path(args.out)
        level_filter = args.level.upper() if isinstance(args.level, str) else None

        with httpx.Client(timeout=60.0) as client:
            for level, lessons in sorted(catalog.items()):
                if level_filter and level.upper() != level_filter:
                    continue
                for lesson_id, lesson_data in sorted(lessons.items()):
                    target = out_dir / level / f"{lesson_id}.png"
                    if target.exists() and not args.force:
                        continue

                    theme = str(lesson_data.get("theme") or lesson_id)
                    goals = lesson_data.get("communicativeGoals")
                    goals_text = ""
                    if isinstance(goals, list):
                        goals_text = ", ".join(
                            str(g) for g in goals if isinstance(g, str)
                        )

                    if args.no_llm_prompt:
                        prompt = (
                            f"Illustration for {level} lesson {lesson_id}: "
                            f"{theme}. {goals_text}"
                        ).strip()
                    else:
                        prompt = (
                            f"Language lesson illustration, {theme}, {goals_text}"
                        ).strip()

                    body = {
                        "prompt": prompt,
                        "seed": _lesson_seed(level, lesson_id),
                    }
                    response = client.post(
                        "http://localhost:7860/sdapi/v1/txt2img", json=body
                    )
                    response.raise_for_status()
                    payload = response.json()
                    images = (
                        payload.get("images") if isinstance(payload, dict) else None
                    )
                    if not isinstance(images, list) or not images:
                        continue

                    raw = base64.b64decode(images[0])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)

        return 0

    if args.command == "download-audio":
        import httpx

        audio_json = Path(args.audio_json)
        if not audio_json.exists():
            print(f"Error: audio json not found: {audio_json}", file=sys.stderr)
            return 1

        loaded = json.loads(audio_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            print(
                "Error: audio json must be an object mapping word->url", file=sys.stderr
            )
            return 1

        items = [(str(word), str(url)) for word, url in loaded.items()]
        if args.limit is not None:
            items = items[: max(args.limit, 0)]

        out_dir = Path(args.out)
        if args.dry_run:
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=60.0) as client:
            for word, url in items:
                file_name = f"{_audio_filename(word)}.mp3"
                target = out_dir / file_name
                if target.exists() and not args.force:
                    continue
                response = client.get(url)
                response.raise_for_status()
                target.write_bytes(response.content)

        return 0

    if args.command == "generate-audio":
        from course_compiler.audio import (
            OpenAISpeechRequest,
            VoxtralClient,
            VoxtralTranscriptRequest,
        )

        settings = Settings.load()

        course_dir = Path(args.course_dir or f"courses/{args.lang}")
        lessons_dir = (
            Path(args.lessons_dir)
            if args.lessons_dir
            else course_dir / "lessons" / args.cefr
        )
        if not lessons_dir.is_dir():
            print(f"Error: lessons directory not found: {lessons_dir}", file=sys.stderr)
            return 1

        if args.only and args.lesson_id:
            print(
                "Error: use only one of --only or --lesson-id.",
                file=sys.stderr,
            )
            return 1

        lesson_filter_raw = args.only or args.lesson_id
        only_lesson_ids: set[str] | None = None
        if lesson_filter_raw:
            only_lesson_ids = _parse_lesson_id_filter(lesson_filter_raw)
            if not only_lesson_ids:
                print(
                    "Error: --only was provided but no valid lesson ids were parsed.",
                    file=sys.stderr,
                )
                return 1

        lesson_files = sorted(lessons_dir.glob("*.json"))
        if only_lesson_ids is not None:
            lesson_files = [p for p in lesson_files if p.stem in only_lesson_ids]
            if not lesson_files:
                print(
                    f"Error: no lessons from --only found in {lessons_dir}",
                    file=sys.stderr,
                )
                return 1

        audio_dir = course_dir / "audio" / args.cefr
        transcript_dir = course_dir / "audio" / "transcripts" / args.cefr
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        disable_cache = args.no_cache or only_lesson_ids is not None
        with VoxtralClient(
            base_url=settings.voxtral_base_url,
            timeout=settings.voxtral_timeout,
        ) as client:
            for lesson_path in lesson_files:
                payload = _load_lesson_payload(lesson_path)
                lesson_id = str(payload.get("id") or lesson_path.stem)
                text = str(payload.get("text") or "").strip()
                if not text:
                    print(
                        f"Skipping {lesson_id}: empty lesson text in {lesson_path}",
                        file=sys.stderr,
                    )
                    continue

                mp3_path = audio_dir / f"{lesson_id}.mp3"
                transcript_path = transcript_dir / f"{lesson_id}.json"
                if (
                    mp3_path.exists()
                    and transcript_path.exists()
                    and not args.force
                    and not disable_cache
                ):
                    continue

                title = str(payload.get("title") or "").strip()
                sample_text = _compose_audio_sample_text(title, text)

                speech = OpenAISpeechRequest(
                    input=sample_text,
                    language=args.lang,
                    voice=args.voice,
                    speed=args.speed,
                    response_format="mp3",
                )
                audio_bytes = client.synthesize_speech(speech)
                mp3_path.write_bytes(audio_bytes)

                transcript = client.generate_transcript(
                    VoxtralTranscriptRequest(
                        audio_path=str(mp3_path.resolve()),
                        text=sample_text,
                        language=args.lang,
                        lesson_id=lesson_id,
                    )
                )
                _write_json(transcript_path, transcript.model_dump(mode="json"))
                generated += 1

        print(
            f"Generated audio for {generated} lesson(s) into {audio_dir} "
            f"with transcripts in {transcript_dir}"
        )
        return 0

    if args.command == "annotate":
        from course_compiler.generation.annotate import (
            LessonOverrides,
            build_lesson_vocab,
            build_vocabulary,
        )
        from course_compiler.generation.annotate import (
            annotate as annotate_text,
        )
        from course_compiler.models import Lesson
        from course_compiler.nlp import PosTaggerError, create_tagger

        course_dir = Path(args.course_dir or f"courses/{args.lang}")
        lessons_dir = (
            Path(args.lessons_dir)
            if args.lessons_dir
            else course_dir / "lessons" / args.cefr
        )
        if not lessons_dir.is_dir():
            print(f"Error: lessons directory not found: {lessons_dir}", file=sys.stderr)
            return 1

        only_lesson_ids = _parse_lesson_id_filter(args.only) if args.only else None

        try:
            tagger = create_tagger(args.lang)
        except PosTaggerError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        words = _load_words_from_lexicon(course_dir, default_language=args.lang)
        verbs = _load_verbs_from_lexicon(course_dir, default_language=args.lang)
        separable = _load_separable_verbs(course_dir)
        vocab = build_lesson_vocab(words, verbs, separable=separable)

        sense_picker = None
        if not args.no_llm_senses:
            sense_picker = _build_sense_picker(course_dir, no_cache=args.no_cache)

        annotated = 0
        cumulative: set[str] = set()
        for lesson_path in sorted(lessons_dir.glob("*.json")):
            payload = _load_lesson_payload(lesson_path)
            lesson_id = str(payload.get("id") or lesson_path.stem)
            new_words = payload.get("newWords") or payload.get("new_words") or []
            cumulative |= {w.lower() for w in new_words}

            if only_lesson_ids is not None and lesson_id not in only_lesson_ids:
                continue  # still accumulated above so allowed vocab stays correct

            text = str(payload.get("text") or "").strip()
            if not text:
                print(f"Skipping {lesson_id}: empty text", file=sys.stderr)
                continue

            meta_path = lessons_dir / f"{lesson_id}.meta.yaml"
            overrides = LessonOverrides.from_dict(
                yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists()
                else None
            )

            vocab.allowed_lemmas = cumulative
            stream = annotate_text(
                text, vocab, tagger, sense_picker=sense_picker, overrides=overrides
            )
            vocabulary = build_vocabulary(list(new_words), vocab, stream, tagger)

            lesson = Lesson.model_validate(payload)
            lesson.tokens = stream
            lesson.vocabulary = vocabulary
            _write_json(
                lesson_path,
                lesson.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )
            annotated += 1

        print(f"Annotated {annotated} lesson(s) in {lessons_dir}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
