"""Command-line entry point.

This is an initial skeleton. The full command surface described in
``INITIAL_INSTRUCTIONS.md`` (build, validate, generate-lessons, ...) is tracked
in ``TASKS/`` and will be added incrementally.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
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
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped:
            return stripped
    return fallback


def _lesson_seed(level: str, lesson_id: str) -> int:
    """Deterministic 31-bit seed from CEFR level + lesson id."""
    payload = f"{level}:{lesson_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _audio_filename(word: str) -> str:
    """Normalize a word key into a filesystem-safe mp3 stem."""
    return word.strip().replace(" ", "_").replace("/", "_")


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
        cache = None if args.no_cache else LLMCache(cache_dir)
        assigner = LLMThemeAssigner(provider, model=None, cache=cache)

        lemmatizer = create_lemmatizer(args.lang)
        generator = LessonGenerator(
            provider,
            lemmatizer,
            cache=cache,
            retry_strategy=args.retry_strategy,
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
        if args.regenerate_fallbacks:
            only_lesson_ids = _load_fallback_lesson_ids(out_dir)
            if not only_lesson_ids:
                print(f"No fallback lessons found in {out_dir}; nothing to regenerate.")
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
                title=_derive_title(lesson.content, plan.theme),
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

        _write_json(out_dir / "manifest.json", manifest)
        _write_json(out_dir / "words.json", words)
        _write_json(out_dir / "verbs.json", verbs)
        _write_json(out_dir / "grammar.json", grammar)
        _write_json(out_dir / "exercises.json", exercises)
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
