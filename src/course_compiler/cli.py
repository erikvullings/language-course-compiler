"""Command-line entry point.

This is an initial skeleton. The full command surface described in
``INITIAL_INSTRUCTIONS.md`` (build, validate, generate-lessons, ...) is tracked
in ``TASKS/`` and will be added incrementally.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _load_words_from_lexicon(lexicon_dir: Path):
    from course_compiler.models import Word

    words_json = lexicon_dir / "words.json"
    words_yaml_dir = lexicon_dir / "words"

    if words_json.exists():
        raw = json.loads(words_json.read_text(encoding="utf-8"))
        return [Word.model_validate(entry) for entry in raw]

    if words_yaml_dir.is_dir():
        word_files = sorted([*words_yaml_dir.glob("*.yaml"), *words_yaml_dir.glob("*.yml")])
        if not word_files:
            raise FileNotFoundError(f"no word entries found in {words_yaml_dir}")
        return [
            Word.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in word_files
        ]

    raise FileNotFoundError(f"neither {words_json} nor {words_yaml_dir} found")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_entries_from_layout(lexicon_dir: Path, stem: str) -> dict[str, dict]:
    json_file = lexicon_dir / f"{stem}.json"
    yaml_dir = lexicon_dir / stem

    if json_file.exists():
        raw = json.loads(json_file.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return {str(entry["id"]): entry for entry in raw if isinstance(entry, dict) and "id" in entry}
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="course", description="Language Course Compiler")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Send a one-off prompt to the configured LLM")
    ask.add_argument("prompt", help="The prompt text")

    gen = sub.add_parser("generate-lessons", help="Generate lessons from an imported lexicon")
    gen.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    gen.add_argument("--cefr", default="A1", help="Target CEFR level (A1, A2, B1, …)")
    gen.add_argument("--lexicon", default=None, help="Lexicon directory (defaults to courses/<lang>)")
    gen.add_argument("--language-name", default=None, help="LLM prompt name (defaults to known name for --lang)")
    gen.add_argument("--words-per-lesson", type=int, default=10, help="New content words per lesson")
    gen.add_argument("--out", default=None, help="Output directory (defaults to <lexicon>/lessons)")

    imp = sub.add_parser("import", help="Import lexical sources into canonical YAML")
    imp.add_argument("--language", default="nl", choices=["nl"], help="Source language")
    imp.add_argument("--kaikki", required=True, help="Path to kaikki.org JSONL dump")
    imp.add_argument("--wordnet", help="Path to Open WordNet LMF XML (synonyms)")
    imp.add_argument("--frequency", help="Path to wordfreq cBpack file")
    imp.add_argument("--nt2lex", help="Path to NT2Lex .tsv resource (CEFR levels)")
    imp.add_argument("--out", default="courses/nl", help="Output course directory")
    imp.add_argument("--limit", type=int, help="Only process the first N entries")

    exp = sub.add_parser("export", help="Export a course into split JSON bundles")
    exp.add_argument("--lang", required=True, help="BCP-47 language code (e.g. nl)")
    exp.add_argument("--course-dir", default=None, help="Course directory (defaults to courses/<lang>)")
    exp.add_argument("--out", default=None, help="Output directory (defaults to <course-dir>/export)")
    exp.add_argument("--version", default="1.0", help="Course version for manifest.json")

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

        settings = Settings.load()
        provider = create_provider(settings)

        lexicon_dir = Path(args.lexicon or f"courses/{args.lang}")
        language_name = args.language_name or _LANG_NAMES.get(args.lang) or args.lang
        try:
            words = _load_words_from_lexicon(lexicon_dir)
        except FileNotFoundError:
            print(
                f"Error: neither {lexicon_dir / 'words.json'} nor {lexicon_dir / 'words'} found. Run 'course import' first.",
                file=sys.stderr,
            )
            return 1

        # Theme assigner — LLM-backed with disk cache next to the lexicon.
        from course_compiler.generation.cache import LLMCache
        from course_compiler.generation.base import create_lemmatizer

        cache_dir = lexicon_dir / ".llm_cache"
        cache = LLMCache(cache_dir)
        assigner = LLMThemeAssigner(provider, model=None, cache=cache)

        lemmatizer = create_lemmatizer(args.lang)
        generator = LessonGenerator(provider, lemmatizer, cache=cache)
        orchestrator = LessonOrchestrator(generator, assigner, words_per_lesson=args.words_per_lesson)

        lessons = orchestrator.generate(
            words,
            language=language_name,
            cefr=args.cefr,
        )

        out_dir = Path(args.out) if args.out else lexicon_dir / "lessons"
        out_dir.mkdir(parents=True, exist_ok=True)
        for lesson in lessons:
            (out_dir / f"{lesson.lesson_id}.txt").write_text(lesson.content, encoding="utf-8")

        print(f"Generated {len(lessons)} lessons into {out_dir}")
        return 0

    if args.command == "import":
        # Language-specific importers live in course_compiler.converters.
        from course_compiler.converters import dutch

        counts = dutch.convert(
            args.kaikki,
            args.out,
            wordnet_path=args.wordnet,
            frequency_path=args.frequency,
            nt2lex_path=args.nt2lex,
            limit=args.limit,
        )
        print(f"Imported {counts['words']} words and {counts['verbs']} verbs into {args.out}")
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

        manifest = {
            "courseLanguage": args.lang,
            "compilerVersion": __version__,
            "version": args.version,
        }

        _write_json(out_dir / "manifest.json", manifest)
        _write_json(out_dir / "words.json", words)
        _write_json(out_dir / "verbs.json", verbs)
        _write_json(out_dir / "grammar.json", grammar)
        _write_json(out_dir / "exercises.json", exercises)
        for lesson_id, payload in sorted(lessons.items()):
            _write_json(lessons_out / f"{lesson_id}.json", payload)

        print(
            f"Exported course bundles into {out_dir} "
            f"(words={len(words)}, verbs={len(verbs)}, grammar={len(grammar)}, exercises={len(exercises)}, lessons={len(lessons)})"
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
