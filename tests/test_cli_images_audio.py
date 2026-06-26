"""Tests for generate-images and download-audio CLI commands."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
import yaml

from course_compiler.cli import _audio_filename, _lesson_seed, main

# ---------------------------------------------------------------------------
# _lesson_seed
# ---------------------------------------------------------------------------


def test_lesson_seed_is_deterministic():
    assert _lesson_seed("A1", "lesson001") == _lesson_seed("A1", "lesson001")


def test_lesson_seed_differs_by_level_and_lesson():
    assert _lesson_seed("A1", "lesson001") != _lesson_seed("A1", "lesson002")
    assert _lesson_seed("A1", "lesson001") != _lesson_seed("A2", "lesson001")


def test_lesson_seed_in_valid_range():
    seed = _lesson_seed("B1", "lesson060")
    assert 0 <= seed < 2**31


# ---------------------------------------------------------------------------
# _audio_filename
# ---------------------------------------------------------------------------


def test_audio_filename_replaces_spaces():
    assert _audio_filename("'s avonds") == "'s_avonds"


def test_audio_filename_replaces_slashes():
    assert _audio_filename("a/b") == "a_b"


def test_audio_filename_plain_word():
    assert _audio_filename("huis") == "huis"


# ---------------------------------------------------------------------------
# generate-images
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_themes_yaml(tmp_path: Path) -> Path:
    data = {
        "A1": {
            "lesson001": {
                "theme": "Greetings",
                "communicativeGoals": ["greet someone", "say goodbye"],
            },
            "lesson002": {
                "theme": "Numbers",
                "communicativeGoals": ["count from 1 to 20"],
            },
        }
    }
    p = tmp_path / "themes.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def _make_flux_handler(png_bytes: bytes):
    """Return an httpx mock handler that returns a fake Flux API response."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "prompt" in body
        payload = {
            "images": [base64.b64encode(png_bytes).decode()],
            "parameters": body,
            "info": "mock",
        }
        return httpx.Response(200, content=json.dumps(payload).encode())

    return handler


def test_generate_images_creates_files(
    tmp_path: Path, minimal_themes_yaml: Path, monkeypatch
):
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # minimal fake PNG bytes
    out_dir = tmp_path / "img"

    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(
            transport=_httpx.MockTransport(_make_flux_handler(fake_png))
        ),
    )

    rc = main(
        [
            "generate-images",
            "--themes-file",
            str(minimal_themes_yaml),
            "--out",
            str(out_dir),
            "--no-llm-prompt",
        ]
    )

    assert rc == 0
    assert (out_dir / "A1" / "lesson001.png").read_bytes() == fake_png
    assert (out_dir / "A1" / "lesson002.png").read_bytes() == fake_png


def test_generate_images_skips_existing(
    tmp_path: Path, minimal_themes_yaml: Path, monkeypatch
):
    out_dir = tmp_path / "img"
    existing = out_dir / "A1" / "lesson001.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"original")

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body.get("prompt", ""))
        payload = {
            "images": [base64.b64encode(b"new").decode()],
            "parameters": {},
            "info": "",
        }
        return httpx.Response(200, content=json.dumps(payload).encode())

    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(transport=_httpx.MockTransport(handler)),
    )

    main(
        [
            "generate-images",
            "--themes-file",
            str(minimal_themes_yaml),
            "--out",
            str(out_dir),
            "--no-llm-prompt",
        ]
    )

    # lesson001 skipped, only lesson002 fetched
    assert len(calls) == 1
    assert existing.read_bytes() == b"original"


def test_generate_images_force_overwrites(
    tmp_path: Path, minimal_themes_yaml: Path, monkeypatch
):
    out_dir = tmp_path / "img"
    existing = out_dir / "A1" / "lesson001.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"original")

    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(
            transport=_httpx.MockTransport(_make_flux_handler(b"replaced"))
        ),
    )

    main(
        [
            "generate-images",
            "--themes-file",
            str(minimal_themes_yaml),
            "--out",
            str(out_dir),
            "--force",
            "--no-llm-prompt",
        ]
    )

    assert existing.read_bytes() == b"replaced"


def test_generate_images_level_filter(
    tmp_path: Path, minimal_themes_yaml: Path, monkeypatch
):
    """Add a second level to the YAML and verify --level filters it out."""
    data = yaml.safe_load(minimal_themes_yaml.read_text())
    data["A2"] = {
        "lesson001": {"theme": "Getting Reacquainted", "communicativeGoals": []}
    }
    minimal_themes_yaml.write_text(yaml.dump(data))

    out_dir = tmp_path / "img"
    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(
            transport=_httpx.MockTransport(_make_flux_handler(b"png"))
        ),
    )

    main(
        [
            "generate-images",
            "--themes-file",
            str(minimal_themes_yaml),
            "--out",
            str(out_dir),
            "--level",
            "A1",
            "--no-llm-prompt",
        ]
    )

    assert (out_dir / "A1" / "lesson001.png").exists()
    assert not (out_dir / "A2").exists()


# ---------------------------------------------------------------------------
# download-audio
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_json(tmp_path: Path) -> Path:
    data = {
        "huis": "https://example.com/audio/huis.mp3",
        "'s avonds": "https://example.com/audio/s_avonds.mp3",
        "fiets": "https://example.com/audio/fiets.mp3",
    }
    p = tmp_path / "audio.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _audio_handler(request: httpx.Request) -> httpx.Response:
    word = request.url.path.rstrip("/").split("/")[-1].replace(".mp3", "")
    return httpx.Response(200, content=f"audio:{word}".encode())


def test_download_audio_creates_files(tmp_path: Path, audio_json: Path, monkeypatch):
    out_dir = tmp_path / "audio"
    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(transport=_httpx.MockTransport(_audio_handler)),
    )

    rc = main(
        [
            "download-audio",
            "--audio-json",
            str(audio_json),
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "huis.mp3").exists()
    assert (out_dir / "'s_avonds.mp3").exists()
    assert (out_dir / "fiets.mp3").exists()


def test_download_audio_skips_existing(tmp_path: Path, audio_json: Path, monkeypatch):
    out_dir = tmp_path / "audio"
    out_dir.mkdir()
    (out_dir / "huis.mp3").write_bytes(b"original")

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"new")

    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(transport=_httpx.MockTransport(handler)),
    )

    main(["download-audio", "--audio-json", str(audio_json), "--out", str(out_dir)])

    assert (out_dir / "huis.mp3").read_bytes() == b"original"
    assert len(calls) == 2  # 's avonds and fiets only


def test_download_audio_dry_run(tmp_path: Path, audio_json: Path, monkeypatch):
    out_dir = tmp_path / "audio"
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"data")

    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(transport=_httpx.MockTransport(handler)),
    )

    main(
        [
            "download-audio",
            "--audio-json",
            str(audio_json),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )

    assert not out_dir.exists()
    assert calls == []


def test_download_audio_limit(tmp_path: Path, audio_json: Path, monkeypatch):
    out_dir = tmp_path / "audio"
    import httpx as _httpx

    _RealClient = _httpx.Client

    monkeypatch.setattr(
        _httpx,
        "Client",
        lambda **_: _RealClient(transport=_httpx.MockTransport(_audio_handler)),
    )

    main(
        [
            "download-audio",
            "--audio-json",
            str(audio_json),
            "--out",
            str(out_dir),
            "--limit",
            "1",
        ]
    )

    files = list(out_dir.iterdir())
    assert len(files) == 1


def test_download_audio_missing_json(tmp_path: Path):
    rc = main(
        [
            "download-audio",
            "--audio-json",
            str(tmp_path / "nonexistent.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# generate-audio
# ---------------------------------------------------------------------------


class _FakeTranscript:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict:
        return dict(self._payload)


class _FakeVoxtralClient:
    instances: list["_FakeVoxtralClient"] = []

    def __init__(self, *, base_url: str, timeout: float):
        self.base_url = base_url
        self.timeout = timeout
        self.calls: list[tuple[str, object]] = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def synthesize_speech(self, request) -> bytes:
        self.calls.append(("speech", request))
        return b"ID3fake"

    def generate_transcript(self, request):
        self.calls.append(("transcript", request))
        return _FakeTranscript(
            {
                "lesson_id": request.lesson_id,
                "audio_path": request.audio_path,
                "transcript_path": request.audio_path.replace(".mp3", ".json"),
                "sentences": [
                    {
                        "id": "s1",
                        "text": "Hallo wereld",
                        "start": 0.0,
                        "end": 1.0,
                        "words": [
                            {"text": "Hallo", "start": 0.0, "end": 0.4},
                            {"text": "wereld", "start": 0.4, "end": 1.0},
                        ],
                    }
                ],
            }
        )


def _write_lesson(path: Path, *, lesson_id: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": lesson_id,
                "language": "nl",
                "cefr": "A1",
                "title": "t",
                "theme": "Numbers",
                "newWords": ["een"],
                "text": text,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_generate_audio_creates_mp3_and_transcript(tmp_path: Path, monkeypatch):
    _FakeVoxtralClient.instances.clear()
    course_dir = tmp_path / "courses" / "nl"
    lessons_dir = course_dir / "lessons" / "A1"
    _write_lesson(
        lessons_dir / "lesson001.json", lesson_id="lesson001", text="Hallo daar"
    )

    monkeypatch.setattr("course_compiler.audio.VoxtralClient", _FakeVoxtralClient)

    rc = main(
        [
            "generate-audio",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--course-dir",
            str(course_dir),
        ]
    )

    assert rc == 0
    mp3 = course_dir / "audio" / "A1" / "lesson001.mp3"
    transcript = course_dir / "audio" / "transcripts" / "A1" / "lesson001.json"
    assert mp3.read_bytes() == b"ID3fake"
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    assert payload["lesson_id"] == "lesson001"
    assert payload["sentences"][0]["id"] == "s1"

    client = _FakeVoxtralClient.instances[-1]
    speech_req = next(req for kind, req in client.calls if kind == "speech")
    transcript_req = next(req for kind, req in client.calls if kind == "transcript")
    assert speech_req.input == "t.\n\nHallo daar"
    assert transcript_req.text == "t.\n\nHallo daar"


def test_generate_audio_lesson_id_filter(tmp_path: Path, monkeypatch):
    _FakeVoxtralClient.instances.clear()
    course_dir = tmp_path / "courses" / "nl"
    lessons_dir = course_dir / "lessons" / "A1"
    _write_lesson(lessons_dir / "lesson001.json", lesson_id="lesson001", text="Een")
    _write_lesson(lessons_dir / "lesson002.json", lesson_id="lesson002", text="Twee")

    monkeypatch.setattr("course_compiler.audio.VoxtralClient", _FakeVoxtralClient)

    rc = main(
        [
            "generate-audio",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--course-dir",
            str(course_dir),
            "--only",
            "lesson002",
        ]
    )

    assert rc == 0
    assert not (course_dir / "audio" / "A1" / "lesson001.mp3").exists()
    assert (course_dir / "audio" / "A1" / "lesson002.mp3").exists()


def test_generate_audio_only_bypasses_existing_outputs(tmp_path: Path, monkeypatch):
    _FakeVoxtralClient.instances.clear()
    course_dir = tmp_path / "courses" / "nl"
    lessons_dir = course_dir / "lessons" / "A1"
    _write_lesson(
        lessons_dir / "lesson001.json", lesson_id="lesson001", text="Hallo daar"
    )

    audio_dir = course_dir / "audio" / "A1"
    transcript_dir = course_dir / "audio" / "transcripts" / "A1"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "lesson001.mp3").write_bytes(b"cached")
    (transcript_dir / "lesson001.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("course_compiler.audio.VoxtralClient", _FakeVoxtralClient)

    rc = main(
        [
            "generate-audio",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--course-dir",
            str(course_dir),
            "--only",
            "lesson001",
        ]
    )

    assert rc == 0
    assert (audio_dir / "lesson001.mp3").read_bytes() == b"ID3fake"
    payload = json.loads((transcript_dir / "lesson001.json").read_text(encoding="utf-8"))
    assert payload["lesson_id"] == "lesson001"


def test_generate_audio_no_cache_bypasses_existing_outputs(tmp_path: Path, monkeypatch):
    _FakeVoxtralClient.instances.clear()
    course_dir = tmp_path / "courses" / "nl"
    lessons_dir = course_dir / "lessons" / "A1"
    _write_lesson(lessons_dir / "lesson001.json", lesson_id="lesson001", text="Een")

    audio_dir = course_dir / "audio" / "A1"
    transcript_dir = course_dir / "audio" / "transcripts" / "A1"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "lesson001.mp3").write_bytes(b"cached")
    (transcript_dir / "lesson001.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("course_compiler.audio.VoxtralClient", _FakeVoxtralClient)

    rc = main(
        [
            "generate-audio",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--course-dir",
            str(course_dir),
            "--no-cache",
        ]
    )

    assert rc == 0
    assert (audio_dir / "lesson001.mp3").read_bytes() == b"ID3fake"


def test_generate_audio_only_and_lesson_id_conflict(tmp_path: Path, monkeypatch):
    _FakeVoxtralClient.instances.clear()
    course_dir = tmp_path / "courses" / "nl"
    lessons_dir = course_dir / "lessons" / "A1"
    _write_lesson(lessons_dir / "lesson001.json", lesson_id="lesson001", text="Een")

    monkeypatch.setattr("course_compiler.audio.VoxtralClient", _FakeVoxtralClient)

    rc = main(
        [
            "generate-audio",
            "--lang",
            "nl",
            "--cefr",
            "A1",
            "--course-dir",
            str(course_dir),
            "--only",
            "lesson001",
            "--lesson-id",
            "lesson001",
        ]
    )

    assert rc == 1
