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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="course", description="Language Course Compiler")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Send a one-off prompt to the configured LLM")
    ask.add_argument("prompt", help="The prompt text")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        settings = Settings.load()
        provider = create_provider(settings)
        print(provider.complete(args.prompt).content)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
