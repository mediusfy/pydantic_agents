"""Model factory for creating pydantic_ai model instances.

Supports multiple providers through a unified interface:
- Native: returns the model name string, letting pydantic_ai infer the provider
- deepseek: uses DeepSeek's OpenAI-compatible API
"""

import logging
import os

from orchestration.settings import settings

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
    """Create a pydantic_ai model instance.

    Supports two forms:
    - Plain model name (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514"):
      returns the string as-is for pydantic_ai's built-in provider inference.
    - Provider-prefixed (e.g. "deepseek:deepseek-reasoner"):
      creates an OpenAIModel pointed at the provider's base URL.
    """
    if ":" not in model_spec:
        return model_spec

    provider, _, model_name = model_spec.partition(":")
    provider = provider.lower()

    base_url: str | None = None
    api_key: str | None = None

    if provider == "openai" and settings.model_provider == "litellm":
        base_url = os.environ.get("LITELLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            env = os.environ.get("LITELLM_ENV", "local").lower()
            if env == "cloud":
                base_url = os.environ.get("LITELLM_CLOUD_URL", "https://litellm-proxy.run.app")
            else:
                base_url = "http://localhost:4000"
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        api_key = os.environ.get("LITELLM_VIRTUAL_KEY") or os.environ.get("LITELLM_MASTER_KEY")
        if not api_key:
            import sys

            if "pytest" in sys.modules:
                api_key = "mock-key"
            else:
                logger.warning(
                    "LiteLLM provider selected but no API key configured. "
                    "Set LITELLM_VIRTUAL_KEY or LITELLM_MASTER_KEY."
                )
                return model_spec
    else:
        base_url = PROVIDER_BASE_URLS.get(provider)
        if not base_url:
            logger.warning(
                "Unknown provider prefix '%s' in model spec '%s', using native inference",
                provider,
                model_spec,
            )
            return model_spec

        api_key = _get_api_key(provider)
        if not api_key:
            logger.warning(
                "No API key found for provider '%s' (set %s in .env), falling back to native inference",
                provider,
                PROVIDER_KEY_VARS.get(provider, f"{provider.upper()}_API_KEY"),
            )
            return model_spec

    from pydantic_ai.models.openai import OpenAIChatModel

    if provider == "deepseek":
        from pydantic_ai.providers.deepseek import DeepSeekProvider

        logger.info("Creating %s model '%s' via %s", provider, model_name, base_url)
        return OpenAIChatModel(
            model_name,
            provider=DeepSeekProvider(api_key=api_key),
        )

    from pydantic_ai.providers.openai import OpenAIProvider

    logger.info("Creating %s model '%s' via %s", provider, model_name, base_url)
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )
