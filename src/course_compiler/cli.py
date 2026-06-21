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
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
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


def _parse_budgets(spec: str | None) -> dict[str, int] | None:
    """Parse a ``"A1=750,A2=2000,..."`` CEFR budget spec into a dict.

    Returns ``None`` when *spec* is empty so the importer keeps NT2Lex levels.
    """
    if not spec:
        return None
    budgets: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        level, _, count = part.partition("=")
        level = level.strip().upper()
        try:
            budgets[level] = int(count.strip())
        except ValueError as exc:
            raise SystemExit(
                f"Invalid budget entry: {part!r} (expected LEVEL=COUNT)"
            ) from exc
    return budgets or None


def _load_words_from_lexicon(lexicon_dir: Path):
    from course_compiler.models import Word

    words_json = lexicon_dir / "words.json"
    words_yaml_dir = lexicon_dir / "words"

    if words_json.exists():
        raw = json.loads(words_json.read_text(encoding="utf-8"))
        language = lexicon_dir.name
        prepared: list[dict] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            if "language" not in entry:
                entry = {"language": language, **entry}
            prepared.append(entry)
        return [Word.model_validate(entry) for entry in prepared]

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


def _load_verbs_from_lexicon(lexicon_dir: Path):
    """Load verbs from ``verbs.json`` or ``verbs/*.yaml``; empty list if none."""
    from course_compiler.models import Verb

    language = lexicon_dir.name
    raw = _load_entries_from_layout(lexicon_dir, "verbs")
    verbs = []
    for entry in raw.values():
        if "language" not in entry:
            entry = {"language": language, **entry}
        verbs.append(Verb.model_validate(entry))
    return verbs


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


def _discover_lesson_levels(course_dir: Path) -> list[str]:
    """Return CEFR levels that have a ``<course>/lessons/<LEVEL>`` dir, in order."""
    from course_compiler.leveling import CEFR_ORDER

    lessons_root = course_dir / "lessons"
    found = {
        level_dir.name.upper()
        for level_dir in lessons_root.glob("*")
        if level_dir.is_dir() and level_dir.name.upper() in CEFR_ORDER
    }
    return [level for level in CEFR_ORDER if level in found]


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


_PROMPT_TEMPLATE = (
    "A clean, modern flat-design illustration without text for an adult language-learning course, "
    "minimal linework, sophisticated muted color palette, depicting a realistic everyday scene "
    "that conveys {goals}. "
)

_IMAGE_PROMPT_SYSTEM = (
    "You are an expert at writing image generation prompts for Flux/Stable Diffusion. "
    "Given a lesson theme and communicative goals for an adult language-learning course, "
    "write a single, vivid image prompt (≤120 words) that a diffusion model can render. "
    "Style: clean flat-design illustration, no text, minimal linework, sophisticated muted "
    "color palette, realistic everyday scene. Output only the prompt — no preamble, no quotes."
)

_IMAGE_PROMPT_USER = (
    "Lesson theme: {theme}\n"
    "Communicative goals: {goals}\n\n"
    "Write the image generation prompt."
)

_NEGATIVE_PROMPT = (
    "text, letters, words, numbers, signs, labels, watermark, speech bubbles, captions, "
    "writing, typography, font, logo, banner, childish, cartoon, cute, kawaii, pastel, "
    "children, kids, toys, 3d, photo, dark, scary, violent"
)


def _build_image_prompt(theme: str, goals: str, provider) -> str:
    """Ask the LLM to craft a diffusion-model image prompt for a lesson."""
    from course_compiler.llm.base import Message, Role

    messages = [
        Message(role=Role.SYSTEM, content=_IMAGE_PROMPT_SYSTEM),
        Message(role=Role.USER, content=_IMAGE_PROMPT_USER.format(theme=theme, goals=goals or theme)),
    ]
    response = provider.complete(messages)
    return response.content.strip()


def _lesson_seed(level: str, lesson_id: str) -> int:
    digest = hashlib.md5(f"{level}-{lesson_id}".encode()).hexdigest()
    return int(digest, 16) % (2**31)


def _is_valid_theme_catalog(path: Path) -> bool:
    """True when a YAML file looks like a non-empty CEFR lesson theme catalog."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict) or not loaded:
        return False

    return any(
        isinstance(lessons, dict) and lessons for _cefr, lessons in loaded.items()
    )


def _audio_filename(word: str) -> str:
    """Sanitize a word key into a safe filename stem (spaces → underscores)."""
    return word.replace(" ", "_").replace("/", "_")


def _cmd_generate_images(args) -> int:
    themes_path: Path | None = None
    if args.themes_file:
        themes_path = Path(args.themes_file)
        if not themes_path.exists():
            bundled = Path(__file__).resolve().parent / "generation" / themes_path.name
            if bundled.exists():
                themes_path = bundled
            else:
                print(
                    f"Error: themes file not found: {args.themes_file}", file=sys.stderr
                )
                return 1
    else:
        candidates = [
            Path(__file__).resolve().parents[2] / "themes.yaml",
            Path(__file__).resolve().parent / "generation" / "themes.yaml",
        ]
        themes_path = next((p for p in candidates if p.exists()), None)
        if themes_path is None:
            print(
                "Error: no themes.yaml found. Pass --themes-file explicitly.",
                file=sys.stderr,
            )
            return 1

    catalog: dict = yaml.safe_load(themes_path.read_text(encoding="utf-8"))
    out_root = Path(args.out)

    llm_provider = None
    if not args.no_llm_prompt:
        settings = Settings.load()
        llm_provider = create_provider(settings)

    steps = args.steps if args.steps is not None else (25 if args.model == "dev" else 4)
    generated = skipped = failed = 0
    timeout = httpx.Timeout(connect=10.0, read=args.timeout, write=30.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        for level, lessons in catalog.items():
            if args.level and level != args.level:
                continue
            for lesson_id, info in lessons.items():
                if args.lesson and lesson_id != args.lesson:
                    continue

                out_path = out_root / level / f"{lesson_id}.png"
                if out_path.exists() and not args.force:
                    skipped += 1
                    continue

                theme = info.get("theme", lesson_id)
                goals = ", ".join(info.get("communicativeGoals", []))
                seed = _lesson_seed(level, lesson_id)

                print(f"  {level}/{lesson_id}: {theme}")

                if llm_provider is not None:
                    try:
                        prompt = _build_image_prompt(theme, goals, llm_provider)
                        print(f"    prompt: {prompt[:80]}...")
                    except Exception as exc:
                        print(f"    LLM prompt failed ({exc}), using template fallback", file=sys.stderr)
                        prompt = _PROMPT_TEMPLATE.format(theme=theme, goals=goals or theme)
                else:
                    prompt = _PROMPT_TEMPLATE.format(theme=theme, goals=goals or theme)

                payload = {
                    "prompt": prompt,
                    "negative_prompt": _NEGATIVE_PROMPT,
                    "width": args.width,
                    "height": args.height,
                    "steps": steps,
                    "cfg_scale": args.cfg_scale,
                    "seed": seed,
                    "model": args.model,
                }
                try:
                    resp = client.post(args.api_url, json=payload)
                    resp.raise_for_status()
                    image_b64: str = resp.json()["images"][0]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(base64.b64decode(image_b64))
                    generated += 1
                except Exception as exc:
                    print(f"    FAILED: {exc}", file=sys.stderr)
                    failed += 1

    print(f"Generated {generated}, skipped {skipped}, failed {failed}.")
    return 0 if failed == 0 else 1


def _cmd_download_audio(args) -> int:
    lang: str = args.lang
    audio_json_path = (
        Path(args.audio_json) if args.audio_json else Path(f"courses/{lang}/audio.json")
    )
    out_dir = Path(args.out) if args.out else Path(f"courses/{lang}/audio")

    if not audio_json_path.exists():
        print(f"Error: {audio_json_path} not found.", file=sys.stderr)
        return 1

    catalog: dict[str, str] = json.loads(audio_json_path.read_text(encoding="utf-8"))
    items = list(catalog.items())
    if args.limit is not None:
        items = items[: args.limit]

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0
    headers = {
        "User-Agent": (
            "language-course-compiler/1.0 "
            "(https://github.com/erikvullings/language-course-compiler; educational use)"
        )
    }
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
        for word, url in items:
            suffix = Path(url).suffix or ".mp3"
            filename = _audio_filename(word) + suffix
            dest = out_dir / filename

            if dest.exists() and not args.force:
                skipped += 1
                continue

            if args.dry_run:
                print(f"  would download: {word!r} → {dest}")
                downloaded += 1
                continue

            try:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                downloaded += 1
                time.sleep(args.delay)
            except Exception as exc:
                print(f"  WARNING: failed to download {word!r}: {exc}", file=sys.stderr)
                failed += 1

    action = "Would download" if args.dry_run else "Downloaded"
    print(f"{action} {downloaded}, skipped {skipped}, failed {failed}.")
    return 0 if failed == 0 else 1


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
        "--words-per-lesson",
        type=int,
        default=10,
        help="New content words per lesson (steady-state count)",
    )
    gen.add_argument(
        "--first-lesson-words",
        type=int,
        default=None,
        help="Front-load: new words in lesson 1, tapering to --words-per-lesson "
        "(e.g. 40, Delft-style). Omit for a uniform budget.",
    )
    gen.add_argument(
        "--front-load-lessons",
        type=int,
        default=3,
        help="Number of early lessons over which --first-lesson-words tapers to "
        "the steady-state count (default: 3)",
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
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching (forces regeneration from scratch)",
    )
    gen.add_argument(
        "--themes-file",
        default=None,
        help="Path to predefined lesson themes YAML (overrides auto-discovery)",
    )

    imp = sub.add_parser("import", help="Import lexical sources into canonical YAML")
    imp.add_argument("--language", default="nl", choices=["nl"], help="Source language")
    imp.add_argument("--kaikki", required=True, help="Path to kaikki.org JSONL dump")
    imp.add_argument("--wordnet", help="Path to Open WordNet LMF XML (synonyms)")
    imp.add_argument("--frequency", help="Path to wordfreq cBpack file")
    imp.add_argument("--nt2lex", help="Path to NT2Lex .tsv resource (CEFR levels)")
    imp.add_argument("--out", default="courses/nl", help="Output course directory")
    imp.add_argument(
        "--budgets",
        help=(
            "Reassign CEFR by cumulative frequency budget, e.g. "
            "'A1=750,A2=2000,B1=3500,B2=5500'. NT2Lex level is used as a floor. "
            "Omit to keep the NT2Lex-attested level."
        ),
    )
    imp.add_argument(
        "--compounds",
        action="store_true",
        help=(
            "With --budgets, introduce transparent compounds (e.g. koffie+pot) "
            "without consuming budget; their level is derived from their parts."
        ),
    )
    imp.add_argument("--limit", type=int, help="Only process the first N entries")

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
        "generate-images", help="Generate lesson images via a local Flux.1 schnell API"
    )
    img.add_argument(
        "--themes-file",
        default=None,
        help="Path to themes YAML (defaults to bundled themes.yaml)",
    )
    img.add_argument(
        "--out",
        default="courses/img",
        help="Output directory for images (default: courses/img)",
    )
    img.add_argument(
        "--level",
        default=None,
        help="Only generate images for this CEFR level (e.g. A1)",
    )
    img.add_argument(
        "--lesson", default=None, help="Only generate this lesson (e.g. lesson001)"
    )
    img.add_argument("--force", action="store_true", help="Overwrite existing images")
    img.add_argument(
        "--api-url",
        default="http://127.0.0.1:7860/sdapi/v1/txt2img",
        help="Flux.1 API endpoint",
    )
    img.add_argument("--width", type=int, default=1024)
    img.add_argument("--height", type=int, default=576)
    img.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps (default: 4 for schnell, 25 for dev)",
    )
    img.add_argument("--cfg-scale", type=float, default=4.0)
    img.add_argument(
        "--model",
        default="schnell",
        choices=["schnell", "dev"],
        help="Flux model (default: schnell)",
    )
    img.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    img.add_argument(
        "--no-llm-prompt",
        action="store_true",
        help="Skip LLM prompt refinement and use the built-in template directly",
    )

    dl = sub.add_parser(
        "download-audio", help="Download audio files listed in audio.json"
    )
    dl.add_argument("--lang", default="nl", help="BCP-47 language code (default: nl)")
    dl.add_argument(
        "--audio-json",
        default=None,
        help="Path to audio.json (defaults to courses/<lang>/audio.json)",
    )
    dl.add_argument(
        "--out",
        default=None,
        help="Output directory (defaults to courses/<lang>/audio/)",
    )
    dl.add_argument("--force", action="store_true", help="Re-download existing files")
    dl.add_argument(
        "--dry-run", action="store_true", help="Print what would be downloaded"
    )
    dl.add_argument(
        "--limit", type=int, default=None, help="Download only first N entries"
    )
    dl.add_argument(
        "--delay",
        type=float,
        default=2,
        help="Seconds to wait between requests (default: 2)",
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
            words = _load_words_from_lexicon(lexicon_dir)
        except FileNotFoundError:
            print(
                (
                    f"Error: neither {lexicon_dir / 'words.json'} nor "
                    f"{lexicon_dir / 'words'} found. Run 'course import' first."
                ),
                file=sys.stderr,
            )
            return 1

        verbs = _load_verbs_from_lexicon(lexicon_dir)

        # Theme assigner — LLM-backed with disk cache next to the lexicon.
        from course_compiler.generation.base import create_lemmatizer
        from course_compiler.generation.cache import LLMCache

        cache_dir = lexicon_dir / ".llm_cache"
        cache = None if args.no_cache else LLMCache(cache_dir)
        assigner = LLMThemeAssigner(provider, model=None, cache=cache)

        lemmatizer = create_lemmatizer(args.lang)
        # Coherence over strict prior-only discipline: tolerate every in-level word
        # (reject only above-CEFR), so the writer can produce natural narrative text.
        generator = LessonGenerator(
            provider, lemmatizer, cache=cache, extra_tolerance=None
        )
        selected_theme_catalog: Path | None = None
        if args.themes_file is not None:
            requested = Path(args.themes_file)
            candidate_paths = [requested]
            if not requested.is_absolute() and requested.parent == Path("."):
                # Convenience: allow "--themes-file themes.yaml" to resolve to
                # the bundled catalog when no local file is present.
                candidate_paths.append(
                    Path(__file__).resolve().parent / "generation" / requested.name
                )

            predefined_themes_path = next(
                (
                    candidate
                    for candidate in candidate_paths
                    if _is_valid_theme_catalog(candidate)
                ),
                None,
            )
            if predefined_themes_path is None:
                print(
                    f"Error: themes file not found or invalid: {args.themes_file}",
                    file=sys.stderr,
                )
                return 1
            selected_theme_catalog = predefined_themes_path
        else:
            theme_candidates = [
                Path(__file__).resolve().parents[2] / "themes.yaml",
                lexicon_dir / "themes.yaml",
                Path(__file__).resolve().parent / "generation" / "themes.yaml",
            ]
            predefined_themes_path = next(
                (
                    candidate
                    for candidate in theme_candidates
                    if _is_valid_theme_catalog(candidate)
                ),
                None,
            )
            selected_theme_catalog = predefined_themes_path

        if selected_theme_catalog is not None:
            print(f"Using theme catalog: {selected_theme_catalog}", file=sys.stderr)
        else:
            print(
                "No predefined theme catalog found; using LLM theme planning.",
                file=sys.stderr,
            )
        orchestrator = LessonOrchestrator(
            generator,
            assigner,
            words_per_lesson=args.words_per_lesson,
            first_lesson_words=args.first_lesson_words,
            front_load_lessons=args.front_load_lessons,
            predefined_themes_path=predefined_themes_path,
        )

        plans = orchestrator.plan(
            words, cefr=args.cefr, verbs=verbs, language=language_name
        )
        if args.preview:
            print(json.dumps(_lesson_blueprint(plans), ensure_ascii=False, indent=2))
            if not args.approve:
                return 0

        out_dir = (
            Path(args.out)
            if args.out
            else lexicon_dir / "lessons" / args.cefr.upper()
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stream: write and report each lesson as it is generated so a long run
        # shows progress and leaves usable partial output if interrupted.
        print(
            "Planning and generating lessons (first run queries the model per "
            "theme and per lesson; results are cached for fast reruns)…",
            file=sys.stderr,
            flush=True,
        )
        count = 0
        for plan, lesson in orchestrator.generate_iter(
            words,
            language=language_name,
            cefr=args.cefr,
            verbs=verbs,
        ):
            payload = Lesson(
                id=lesson.lesson_id,
                language=args.lang,
                cefr=args.cefr,
                title=lesson.title,
                theme=lesson.theme,
                new_words=sorted(lesson.new_words),
                text=lesson.content,
                attempts=lesson.attempts,
                tolerated=sorted(lesson.tolerated),
            )
            _write_json(
                out_dir / f"{lesson.lesson_id}.json",
                payload.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )
            count += 1
            print(
                f"  [{count}] {lesson.lesson_id} {plan.theme} "
                f"({lesson.attempts} attempt(s))",
                file=sys.stderr,
                flush=True,
            )

        print(f"Generated {count} lessons into {out_dir}")
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
            budgets=_parse_budgets(args.budgets),
            detect_compounds=args.compounds,
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

        # Lessons live under <course-dir>/lessons/<LEVEL> (level-scoped). When such
        # level dirs exist, export per level so ids that repeat across levels
        # (every level restarts at lesson001) don't overwrite each other. A legacy
        # flat <course-dir>/lessons is exported as today (single, unleveled bundle).
        levels = _discover_lesson_levels(course_dir)
        lesson_count = 0
        if levels:
            for level in levels:
                level_lessons = _load_lessons_for_export(
                    course_dir / "lessons" / level
                )
                for lesson_id, payload in sorted(level_lessons.items()):
                    payload.setdefault("id", lesson_id)
                    payload["level"] = level
                    _write_json(lessons_out / level / f"{lesson_id}.json", payload)
                    lesson_count += 1
        else:
            lessons = _load_lessons_for_export(course_dir / "lessons")
            for lesson_id, payload in sorted(lessons.items()):
                _write_json(lessons_out / f"{lesson_id}.json", payload)
            lesson_count = len(lessons)

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

        print(
            f"Exported course bundles into {out_dir} "
            f"(words={len(words)}, verbs={len(verbs)}, grammar={len(grammar)}, "
            f"exercises={len(exercises)}, lessons={lesson_count}, "
            f"levels={levels or ['(flat)']})"
        )
        return 0

    if args.command == "generate-images":
        return _cmd_generate_images(args)

    if args.command == "download-audio":
        return _cmd_download_audio(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
