"""Settings load from an explicit env dict without touching the process env."""

from __future__ import annotations

from course_compiler.settings import Settings


def test_defaults_when_env_is_empty():
    settings = Settings.load(env={})

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.llm_thinking is False
    assert settings.llm_timeout == 300.0
    assert settings.llm_max_retries == 2
    assert settings.voxtral_base_url == "http://localhost:8001"
    assert settings.voxtral_timeout == 120.0


def test_env_values_override_defaults():
    settings = Settings.load(
        env={
            "LLM_PROVIDER": "OpenAI",
            "OPENAI_API_KEY": "sk-123",
            "OPENAI_MODEL": "gpt-4o",
            "LLM_TEMPERATURE": "0.2",
            "LLM_THINKING": "true",
        }
    )

    assert settings.llm_provider == "openai"  # normalized to lowercase
    assert settings.openai_api_key == "sk-123"
    assert settings.openai_model == "gpt-4o"
    assert settings.llm_temperature == 0.2
    assert settings.llm_thinking is True


def test_blank_values_fall_back_to_defaults():
    settings = Settings.load(env={"OLLAMA_MODEL": ""})
    assert settings.ollama_model == "llama3"


def test_voxtral_env_values_override_defaults():
    settings = Settings.load(
        env={
            "VOXTRAL_BASE_URL": "http://localhost:9000/docs",
            "VOXTRAL_TIMEOUT": "42",
        }
    )
    assert settings.voxtral_base_url == "http://localhost:9000/docs"
    assert settings.voxtral_timeout == 42.0
