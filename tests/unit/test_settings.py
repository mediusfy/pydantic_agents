from pathlib import Path

from pydantic_agents.core.settings import Settings


def test_settings_defaults(monkeypatch):
    """The .env file overrides hard-coded defaults at the module level.
    When skipping .env with _env_file=None we get the code defaults."""
    monkeypatch.delenv("BA_MODEL", raising=False)
    monkeypatch.delenv("MAX_RETRIES", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_LLM_CALLS", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("CHECKPOINT_DIR", raising=False)

    s = Settings(_env_file=None)
    assert s.ba_model == "deepseek-v4-pro"
    assert s.max_retries == 3
    assert s.max_concurrent_llm_calls == 5
    assert s.output_dir == Path("./generated-app")
    assert s.checkpoint_dir == Path("./.claw-checkpoints")


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("BA_MODEL", "gpt-4o")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/my-output")
    s = Settings(_env_file=None)
    assert s.ba_model == "gpt-4o"
    assert s.max_retries == 5
    assert s.output_dir == Path("/tmp/my-output")


def test_settings_api_keys_empty_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.openai_api_key is None
    assert s.anthropic_api_key is None
    assert s.gemini_api_key is None


def test_settings_api_keys_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "***")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
    s = Settings(_env_file=None)
    assert s.openai_api_key == "***"
    assert s.anthropic_api_key == "***"
