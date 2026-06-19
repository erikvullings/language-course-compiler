"""Disk-based LLM response cache for reproducible generation.

Cache entries are JSON files named by a SHA-256 digest of (model, messages).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from course_compiler.llm.base import LLMResponse


class LLMCache:
    """Persist and retrieve :class:`~course_compiler.llm.base.LLMResponse` objects.

    Keyed by model + message list so the same prompt always returns the same
    cached response across runs, making generation deterministic and tests
    network-free.
    """

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, model: str, messages: list[dict[str, str]]) -> Path:
        return self._dir / f"{self._key(model, messages)}.json"

    def get(self, model: str, messages: list[dict[str, str]]) -> LLMResponse | None:
        """Return cached response, or ``None`` on a miss."""
        path = self._path(model, messages)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(content=data["content"], model=data["model"], raw=data.get("raw", {}))

    def put(self, model: str, messages: list[dict[str, str]], response: LLMResponse) -> None:
        """Store *response* under the (model, messages) key."""
        path = self._path(model, messages)
        path.write_text(
            json.dumps({"content": response.content, "model": response.model, "raw": response.raw}, ensure_ascii=False),
            encoding="utf-8",
        )
