"""Application settings loaded from environment variables / a ``.env`` file.

Uses python-dotenv so configuration stays out of code. Call :meth:`Settings.load`
to read the current environment (optionally loading a ``.env`` first).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from course_compiler.llm.ollama import DEFAULT_BASE_URL as OLLAMA_BASE_URL
from course_compiler.llm.ollama import DEFAULT_MODEL as OLLAMA_MODEL
from course_compiler.llm.openai import DEFAULT_BASE_URL as OPENAI_BASE_URL
from course_compiler.llm.openai import DEFAULT_MODEL as OPENAI_MODEL


def _get(env: dict[str, str], key: str, default: str) -> str:
    value = env.get(key)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    llm_provider: str = "ollama"
    llm_temperature: float = 0.7
    llm_timeout: float = 60.0

    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_model: str = OLLAMA_MODEL

    openai_api_key: str = ""
    openai_base_url: str = OPENAI_BASE_URL
    openai_model: str = OPENAI_MODEL

    @classmethod
    def load(
        cls,
        env: dict[str, str] | None = None,
        *,
        dotenv_path: str | None = None,
        use_dotenv: bool = True,
    ) -> Settings:
        """Build settings from ``env`` (defaults to ``os.environ``).

        When ``env`` is omitted and ``use_dotenv`` is true, a ``.env`` file is
        loaded into the process environment first. Pass an explicit ``env`` dict
        (e.g. in tests) to read from it directly without touching the process.
        """

        if env is None:
            if use_dotenv:
                load_dotenv(dotenv_path)
            env = dict(os.environ)

        return cls(
            llm_provider=_get(env, "LLM_PROVIDER", cls.llm_provider).lower(),
            llm_temperature=float(_get(env, "LLM_TEMPERATURE", str(cls.llm_temperature))),
            llm_timeout=float(_get(env, "LLM_TIMEOUT", str(cls.llm_timeout))),
            ollama_base_url=_get(env, "OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=_get(env, "OLLAMA_MODEL", cls.ollama_model),
            openai_api_key=_get(env, "OPENAI_API_KEY", cls.openai_api_key),
            openai_base_url=_get(env, "OPENAI_BASE_URL", cls.openai_base_url),
            openai_model=_get(env, "OPENAI_MODEL", cls.openai_model),
        )
