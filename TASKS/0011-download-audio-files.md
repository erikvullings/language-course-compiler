# 0011 Download audio files from audio.json

Status: done
Priority: medium
Owner: erik.vullings
Agent: unassigned
Area: cli
Depends on: none

## Context

`courses/nl/audio.json` maps word strings to Wikimedia MP3 URLs:

```json
{
  "huis": "https://upload.wikimedia.org/wikipedia/commons/transcoded/.../Nl-huis.ogg.mp3",
  ...
}
```

We need a `course download-audio` CLI sub-command (or standalone script) that
fetches each URL and saves it to `courses/nl/audio/<ID>.mp3`, where `<ID>` is the
JSON key (the word). This makes audio resolution trivial in the SPA/PWA: given a
word, construct the path `audio/<word>.mp3`.

## Acceptance Criteria

- `uv run course download-audio --lang nl` reads `courses/nl/audio.json` and
  downloads each URL to `courses/nl/audio/<word>.mp3`.
- Files that already exist on disk are skipped unless `--force` is passed.
- The word (key) is used verbatim as the filename stem; the extension comes from
  the URL (`.mp3`).
- Failed downloads (non-2xx, network error) are logged as warnings and skipped —
  the script does not abort on a single failure.
- A final summary is printed: `Downloaded N, skipped M, failed F`.
- `--dry-run` flag prints what would be downloaded without writing anything.
- Optional `--limit N` flag for testing (download only the first N entries).

## Implementation Notes

- Wire as `course download-audio` in `src/course_compiler/cli.py` (consistent with
  the existing `course ask` and `course import` sub-commands).
- Input path: `courses/{lang}/audio.json`; output dir: `courses/{lang}/audio/`.
- Use `httpx` (already a runtime dependency) for HTTP — optionally async with
  `httpx.AsyncClient` + `asyncio` for concurrency.
- Filenames: sanitize the word key so it is safe on all OSes. Characters like
  `'`, spaces, `/` need escaping or replacing (e.g. URL-encode or replace spaces
  with `_`). Whatever scheme is chosen, document it so the SPA can reproduce it.
- Create the output directory with `pathlib.Path.mkdir(parents=True, exist_ok=True)`.
- The `--lang` flag defaults to `nl`; the path convention is `courses/<lang>/`.

## Agent Notes

- Task created 2026-06-19. No implementation started.
- audio.json location confirmed: `courses/nl/audio.json` (note: `courses/`, not `course/`).
- Keys include special characters: `'ie`, `'s avonds`, spaces — filename sanitization is required.
- Implemented 2026-06-19 as `course download-audio` subcommand in `src/course_compiler/cli.py`.
- Sanitization scheme: `word.replace(" ", "_").replace("/", "_")` — simple and SPA-reproducible.
- Tests in `tests/test_cli_images_audio.py` covering create, skip, force, dry-run, limit, missing-json.
