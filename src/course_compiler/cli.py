"""Command-line entry point.

This is an initial skeleton. The full command surface described in
``INITIAL_INSTRUCTIONS.md`` (build, validate, generate-lessons, ...) is tracked
in ``TASKS/`` and will be added incrementally.
"""

from __future__ import annotations

import argparse
import sys

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
        import json
        from pathlib import Path

        from course_compiler.generation.lesson import LessonGenerator
        from course_compiler.generation.orchestrator import LessonOrchestrator
        from course_compiler.generation.themes import LLMThemeAssigner
        from course_compiler.models import Word

        settings = Settings.load()
        provider = create_provider(settings)

        lexicon_dir = Path(args.lexicon or f"courses/{args.lang}")
        language_name = args.language_name or _LANG_NAMES.get(args.lang) or args.lang
        words_file = lexicon_dir / "words.json"
        if not words_file.exists():
            print(f"Error: {words_file} not found. Run 'course import' first.", file=sys.stderr)
            return 1

        raw = json.loads(words_file.read_text(encoding="utf-8"))
        words = [Word.model_validate(entry) for entry in raw]

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
