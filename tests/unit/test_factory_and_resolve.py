"""Model factory laziness and fail-loud behaviour, and per-agent model resolution."""

import pytest

from pydantic_agents.core.exceptions import ConfigurationError
from pydantic_agents.core.models.factory import LazyModel, create_model
from pydantic_agents.core.settings import Settings


@pytest.fixture
def litellm(monkeypatch):
    s = Settings(_env_file=None, model_provider="litellm")
    monkeypatch.setattr("pydantic_agents.core.models.factory.settings", s)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return s


def test_plain_names_are_returned_as_is():
    assert create_model("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


def test_prefixed_specs_are_lazy_and_do_not_need_a_key_to_construct(litellm, monkeypatch):
    monkeypatch.delenv("LITELLM_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    model = create_model("openai:claude-sonnet-5")  # must not raise at import time
    assert isinstance(model, LazyModel)
    assert model.model_name == "claude-sonnet-5" and model.system == "openai"


def test_missing_litellm_key_fails_loudly_at_use_instead_of_targeting_openai(litellm, monkeypatch):
    monkeypatch.delenv("LITELLM_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="LITELLM_VIRTUAL_KEY"):
        _ = create_model("openai:claude-sonnet-5").wrapped


def test_environment_changes_after_construction_are_honoured(litellm, monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "first")
    model = create_model("openai:claude-sonnet-5")  # built before the URL is known
    monkeypatch.setenv("LITELLM_BASE_URL", "http://proxy.internal:4000")
    assert str(model.wrapped.client.base_url).startswith("http://proxy.internal:4000/v1")


@pytest.mark.parametrize(
    "provider, model_name, expected",
    [
        ("litellm", "claude-sonnet-5", "openai:claude-sonnet-5"),
        ("openai", "gpt-4.1", "openai:gpt-4.1"),
        ("openai", "o3-mini", "openai:o3-mini"),
        ("openai", "claude-sonnet-5", "openai:gpt-4o"),
        ("gemini", "gemini-2.5-pro", "gemini:gemini-2.5-pro"),
        ("gemini", "claude-sonnet-5", "gemini:gemini-2.0-flash"),
        ("vertexai", "gemini-2.5-flash", "vertexai:gemini-2.5-flash"),
        ("vertexai", "claude-sonnet-5", "vertexai:gemini-2.0-flash"),
        ("openai", "openai:gpt-4.1-mini", "openai:gpt-4.1-mini"),
        ("deepseek", "deepseek-reasoner", "deepseek:deepseek-reasoner"),
    ],
)
def test_resolve_model_honours_per_agent_names(provider, model_name, expected):
    assert Settings(_env_file=None, model_provider=provider).resolve_model(model_name) == expected
