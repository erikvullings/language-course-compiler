"""Settings load from an explicit env dict without touching the process env."""

from __future__ import annotations

from course_compiler.settings import Settings


def test_defaults_when_env_is_empty():
    settings = Settings.load(env={})

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.llm_timeout == 300.0
    assert settings.llm_max_retries == 2


def test_env_values_override_defaults():
    settings = Settings.load(
        env={
            "LLM_PROVIDER": "OpenAI",
            "OPENAI_API_KEY": "sk-123",
            "OPENAI_MODEL": "gpt-4o",
            "LLM_TEMPERATURE": "0.2",
        }
    )

    assert settings.llm_provider == "openai"  # normalized to lowercase
    assert settings.openai_api_key == "sk-123"
    assert settings.openai_model == "gpt-4o"
    assert settings.llm_temperature == 0.2


def test_blank_values_fall_back_to_defaults():
    settings = Settings.load(env={"OLLAMA_MODEL": ""})
    assert settings.ollama_model == "llama3"
