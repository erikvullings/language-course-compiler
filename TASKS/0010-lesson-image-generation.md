# 0010 Lesson image generation

Status: done
Priority: medium
Owner: erik.vullings
Agent: unassigned
Area: generation
Depends on: none

## Context

Each lesson in the course needs a representative illustration for use in the SPA/PWA
front end. Images are generated locally using Flux.1 schnell (via an Automatic1111-
compatible API running at `http://127.0.0.1:7860`).

Input: `src/course_compiler/generation/themes.yaml` — CEFR level → lesson → `theme`
and `communicativeGoals` list.

Output: `course/img/<LEVEL>/<LESSON>.png` (e.g. `course/img/A1/lesson001.png`).
Output paths are language-independent so one image set serves all target languages.

The script POSTs to:

```
POST http://127.0.0.1:7860/sdapi/v1/txt2img
```

with a consistent prompt template:

```
A simple, clean 2D vector comic illustration style, flat colors, bold outlines,
educational language course graphic, minimalist, with the theme '{THEME}' and
communicative goals: {GOALS}.
```

The response `images[0]` is a base64-encoded PNG that is decoded and saved to disk.

Generation parameters (configurable via CLI flags):

| Parameter   | Default |
|-------------|---------|
| width       | 1024    |
| height      | 576     |
| steps       | 4       |
| cfg_scale   | 4.0     |
| seed        | 42      |
| model       | schnell |

The script should:
- Skip existing images unless `--force` is passed.
- Accept optional `--level` and `--lesson` filters (e.g. `--level A1 --lesson lesson001`).
- Print progress to stdout (level/lesson/theme).
- Exit non-zero on HTTP or decode errors.

## Acceptance Criteria

- `uv run course generate-images` (or `python scripts/generate_images.py`) reads
  `themes.yaml` and writes PNGs to `course/img/<LEVEL>/<LESSON>.png`.
- Existing images are skipped unless `--force` is used.
- `--level` and `--lesson` filters work correctly.
- The script is tested offline (mock HTTP) via a pytest test in `tests/`.
- No language-specific logic — prompt uses only `theme` and `communicativeGoals`
  from the YAML.

## Implementation Notes

- Add script at `scripts/generate_images.py` (standalone, no src-layout import
  required — may import `course_compiler` if already installed).
- Or wire it as a `course generate-images` CLI sub-command in `src/course_compiler/cli.py`.
- Use `httpx` (already a dependency) for the POST request rather than `subprocess`/`curl`.
- Base64 decode: `base64.b64decode(images[0])`.
- Output directory: create with `pathlib.Path.mkdir(parents=True, exist_ok=True)`.
- Seed should be deterministic per lesson (e.g. derived from level+lesson string) so
  re-runs produce the same image without `--force`.

## Agent Notes

- Task created 2026-06-19. No implementation started.
- Implemented 2026-06-19 as `course generate-images` subcommand in `src/course_compiler/cli.py`.
- Seed is derived deterministically via MD5 of `"{level}-{lesson_id}"` → int % 2^31.
- Prompt template: `_PROMPT_TEMPLATE` constant in cli.py.
- 15 tests in `tests/test_cli_images_audio.py` covering create, skip, force, level/lesson filter.
