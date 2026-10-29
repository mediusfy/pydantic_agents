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
    assert model.model_name == "claude-sonnet-5"
    assert model.system == "openai"


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


def test_a_failed_first_build_is_retried_not_cached(litellm, monkeypatch):
    monkeypatch.delenv("LITELLM_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    model = create_model("openai:claude-sonnet-5")
    with pytest.raises(ConfigurationError):
        _ = model.wrapped

    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "now-set")
    assert model.wrapped.model_name == "claude-sonnet-5"


@pytest.mark.parametrize(
    "provider, model_name",
    [("openai", "claude-sonnet-5"), ("gemini", "gpt-4o"), ("vertexai", "claude-sonnet-5")],
)
def test_falling_back_to_the_provider_default_warns(caplog, provider, model_name):
    from pydantic_agents.core import settings as settings_module

    settings_module._warn_model_fallback.cache_clear()
    with caplog.at_level("WARNING", logger="pydantic_agents.core.settings"):
        Settings(_env_file=None, model_provider=provider).resolve_model(model_name)
    assert model_name in caplog.text
    assert provider in caplog.text


def test_matching_names_do_not_warn(caplog):
    from pydantic_agents.core import settings as settings_module

    settings_module._warn_model_fallback.cache_clear()
    with caplog.at_level("WARNING", logger="pydantic_agents.core.settings"):
        Settings(_env_file=None, model_provider="openai").resolve_model("gpt-4.1")
    assert caplog.text == ""


@pytest.mark.parametrize(
    "provider, model_name, expected",
    [
        (None, "deepseek-reasoner", "deepseek:deepseek-reasoner"),
        (None, "claude-sonnet-5", "claude-sonnet-5"),
        (None, "anthropic:claude-x", "anthropic:claude-x"),
        ("deepseek", "openai:gpt-4.1", "openai:gpt-4.1"),
        ("anthropic", "claude-x", "anthropic:claude-x"),
        ("litellm", "openai:gpt-4.1", "openai:openai:gpt-4.1"),
    ],
)
def test_resolve_model_other_providers(provider, model_name, expected):
    assert Settings(_env_file=None, model_provider=provider).resolve_model(model_name) == expected


@pytest.mark.parametrize(
    "env, expected_base",
    [
        ({}, "http://localhost:4000/v1"),
        ({"LITELLM_ENV": "cloud"}, "https://litellm-proxy.run.app/v1"),
        ({"LITELLM_ENV": "cloud", "LITELLM_CLOUD_URL": "https://gw.example.com/"}, "https://gw.example.com/v1"),
        ({"LITELLM_BASE_URL": "http://proxy:4000/v1"}, "http://proxy:4000/v1"),
        ({"OPENAI_BASE_URL": "http://other:9000"}, "http://other:9000/v1"),
    ],
)
def test_litellm_connection_resolves_the_base_url(monkeypatch, env, expected_base):
    from pydantic_agents.core.models.factory import _litellm_connection

    for name in ("LITELLM_BASE_URL", "OPENAI_BASE_URL", "LITELLM_ENV", "LITELLM_CLOUD_URL", "LITELLM_MASTER_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "k")
    assert _litellm_connection("openai:m") == (expected_base, "k")


def test_litellm_master_key_is_accepted(monkeypatch):
    from pydantic_agents.core.models.factory import _litellm_connection

    monkeypatch.delenv("LITELLM_VIRTUAL_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master")
    assert _litellm_connection("openai:m")[1] == "master"


def test_litellm_connection_reads_settings_loaded_from_dotenv(monkeypatch):
    from pydantic_agents.core.models import factory

    for name in ("LITELLM_BASE_URL", "OPENAI_BASE_URL", "LITELLM_VIRTUAL_KEY", "LITELLM_MASTER_KEY"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(
        _env_file=None,
        model_provider="litellm",
        litellm_base_url="http://dotenv-proxy:4000",
        litellm_virtual_key="sk-from-dotenv",
    )
    monkeypatch.setattr(factory, "settings", s)
    assert factory._litellm_connection("openai:m") == ("http://dotenv-proxy:4000/v1", "sk-from-dotenv")


@pytest.mark.parametrize(
    "provider, key, expected",
    [
        ("deepseek", "sk-d", ("https://api.deepseek.com/v1", "sk-d")),
        ("openai", "sk-o", ("https://api.openai.com/v1", "sk-o")),
        ("deepseek", None, None),
        ("mystery", "sk-x", None),
    ],
)
def test_provider_connection(monkeypatch, provider, key, expected):
    from pydantic_agents.core.models import factory

    fake = Settings(_env_file=None, deepseek_api_key=key if provider == "deepseek" else None)
    fake.openai_api_key = key if provider == "openai" else None
    monkeypatch.setattr(factory, "settings", fake)
    assert factory._provider_connection(provider, f"{provider}:m") == expected


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
def test_make_chat_model_builds_an_openai_compatible_model(provider):
    from pydantic_agents.core.models.factory import _make_chat_model

    model = _make_chat_model(provider, "some-model", "https://example.com/v1", "sk-test")
    assert model.model_name == "some-model"


def test_unknown_provider_prefix_uses_native_inference(monkeypatch):
    from pydantic_ai.models.test import TestModel

    from pydantic_agents.core.models import factory

    monkeypatch.setattr(factory, "settings", Settings(_env_file=None, model_provider="openai"))
    monkeypatch.setattr("pydantic_ai.models.infer_model", lambda _spec: TestModel())
    assert isinstance(factory._build_model("mystery:model"), TestModel)
