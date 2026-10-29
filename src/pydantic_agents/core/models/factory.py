"""Model factory for creating pydantic_ai model instances.

Supports multiple providers through a unified interface:
- Native: returns the model name string, letting pydantic_ai infer the provider
- deepseek: uses DeepSeek's OpenAI-compatible API

Provider-prefixed specs are built lazily, on the first request, so environment changes made after import (keys,
LiteLLM URL) are honoured and a missing key surfaces as a ConfigurationError at use instead of silently falling
back to a different provider.
"""

import logging
import os
from functools import cached_property

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel

from pydantic_agents.core.exceptions import ConfigurationError
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

PROVIDER_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}

PROVIDER_KEY_VARS: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _get_api_key(provider: str) -> str | None:
    """Get the API key for a provider from settings or env."""
    key_map: dict[str, str | None] = {
        "deepseek": settings.deepseek_api_key,
        "openai": settings.openai_api_key,
    }
    return key_map.get(provider)


def create_model(model_spec: str):
    """Return a pydantic_ai model for a spec.

    - Plain model name (e.g. "claude-sonnet-4-20250514"): returned as-is for pydantic_ai's own provider inference.
    - Provider-prefixed (e.g. "deepseek:deepseek-reasoner", "openai:claude-sonnet-5" via LiteLLM): a lazy model
      that is built from the current settings/environment on first use.
    """
    if ":" not in model_spec:
        return model_spec
    return LazyModel(model_spec)


class LazyModel(WrapperModel):
    """A model whose underlying client is built on first use."""

    def __init__(self, spec: str):
        Model.__init__(self)
        self.spec = spec

    @cached_property
    def wrapped(self) -> Model:  # type: ignore[override]
        return _build_model(self.spec)

    @property
    def model_name(self) -> str:
        # Avoid building the client (and demanding a key) just to name the model.
        return self.spec.partition(":")[2] or self.spec

    @property
    def system(self) -> str:
        return self.spec.partition(":")[0]


def _litellm_connection(model_spec: str) -> tuple[str, str]:
    """Base URL and key for the LiteLLM proxy, read from the environment (or ``settings`` loaded from ``.env``)."""
    base_url = os.environ.get("LITELLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or settings.litellm_base_url
    if not base_url:
        env_mode = os.environ.get("LITELLM_ENV") or settings.litellm_env or "local"
        if env_mode.lower() == "cloud":
            base_url = os.environ.get("LITELLM_CLOUD_URL") or settings.litellm_cloud_url
        else:
            base_url = "http://localhost:4000"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    api_key = (
        os.environ.get("LITELLM_VIRTUAL_KEY")
        or os.environ.get("LITELLM_MASTER_KEY")
        or settings.litellm_virtual_key
        or settings.litellm_master_key
    )
    if not api_key:
        # Falling back to "openai:<name>" here would send the request to api.openai.com.
        raise ConfigurationError(
            f"MODEL_PROVIDER=litellm but no API key is configured for '{model_spec}'. "
            "Set LITELLM_VIRTUAL_KEY or LITELLM_MASTER_KEY."
        )
    return base_url, api_key


def _provider_connection(provider: str, model_spec: str) -> tuple[str, str] | None:
    """Base URL and key for a directly-addressed provider, or None to use pydantic_ai's native inference."""
    base_url = PROVIDER_BASE_URLS.get(provider)
    if not base_url:
        logger.warning("Unknown provider prefix '%s' in model spec '%s', using native inference", provider, model_spec)
        return None
    api_key = _get_api_key(provider)
    if not api_key:
        logger.warning(
            "No API key found for provider '%s' (set %s in .env), falling back to native inference",
            provider,
            PROVIDER_KEY_VARS.get(provider, f"{provider.upper()}_API_KEY"),
        )
        return None
    return base_url, api_key


def _http_client():
    """An httpx client with the configured per-request timeout, or None to use pydantic_ai's default client."""
    if settings.agent_http_timeout_seconds <= 0:
        return None
    import httpx

    return httpx.AsyncClient(timeout=httpx.Timeout(settings.agent_http_timeout_seconds, connect=10.0))


def _make_chat_model(provider: str, model_name: str, base_url: str, api_key: str) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel

    logger.info("Creating %s model '%s' via %s", provider, model_name, base_url)
    client = _http_client()
    extra = {"http_client": client} if client is not None else {}
    if provider == "deepseek":
        from pydantic_ai.providers.deepseek import DeepSeekProvider

        return OpenAIChatModel(model_name, provider=DeepSeekProvider(api_key=api_key, **extra))

    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key, **extra))


def _build_model(model_spec: str) -> Model:
    from pydantic_ai.models import infer_model

    provider, _, model_name = model_spec.partition(":")
    provider = provider.lower()

    if provider == "openai" and settings.model_provider == "litellm":
        connection: tuple[str, str] | None = _litellm_connection(model_spec)
    else:
        connection = _provider_connection(provider, model_spec)
    if connection is None:
        return infer_model(model_spec)
    return _make_chat_model(provider, model_name, *connection)
