"""Centralized configuration via pydantic-settings.

All environment variables are validated and typed here.
Usage:
    from pydantic_agents.core.settings import settings
    settings.ba_model       # "deepseek-reasoner" by default
    settings.max_retries    # 2 by default
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    logfire_token: str | None = None
    logfire_api_key: str | None = None

    database_url: str | None = None

    # Supported providers: deepseek, gemini, vertexai, openai, litellm
    model_provider: str | None = "litellm"

    redis_iris_url: str | None = None
    redis_iris_store_id: str | None = None
    redis_iris_token: str | None = None

    # Agent Model Overrides
    # NOTE: All agents are pinned to the same model ("claude-sonnet-5", an
    # alias the LiteLLM proxy resolves to Anthropic Claude Sonnet with its
    # own fallback chain) rather than split across fast/reasoner tiers. See
    # resolve_model() below for how this is turned into a pydantic_ai model
    # spec depending on model_provider.
    ba_model: str = "claude-sonnet-5"
    backend_model: str = "claude-sonnet-5"
    frontend_model: str = "claude-sonnet-5"
    test_model: str = "claude-sonnet-5"
    sre_model: str = "claude-sonnet-5"
    review_model: str = "claude-sonnet-5"
    database_model: str = "claude-sonnet-5"
    devops_model: str = "claude-sonnet-5"
    docs_model: str = "claude-sonnet-5"
    security_model: str = "claude-sonnet-5"
    ux_model: str = "claude-sonnet-5"
    cto_model: str = "claude-sonnet-5"

    holiday_planner_model: str = "claude-sonnet-5"
    region_analyzer_model: str = "claude-sonnet-5"
    asymmetric_reviewer_model: str = "claude-sonnet-5"
    current_events_model: str = "claude-sonnet-5"
    weather_model: str = "claude-sonnet-5"
    accountant_model: str = "claude-sonnet-5"
    price_analyst_model: str = "claude-sonnet-5"
    market_shark_model: str = "claude-sonnet-5"
    crypto_oracle_model: str = "claude-sonnet-5"

    # Pipeline Configuration
    output_dir: Path = Path("./generated-app")
    max_retries: int = 3  # Autonomous Remediation Loop: max retries before escalating to Lead Orchestrator
    max_concurrent_llm_calls: int = 5
    checkpoint_dir: Path = Path("./.claw-checkpoints")
    # Pipeline checkpoint backend: "disk" (JSON files) or "turso" (embedded Turso DB; needs the `turso` extra).
    state_backend: Literal["disk", "turso"] = "disk"
    vector_db_dir: Path = Path("./.claw-vectordb")

    go_version: str = "1.26"

    # NOTE: The _FAST_TIER/_REASONER_TIER capability-tier frozensets that used
    # to live here were removed. They were never actually read by
    # resolve_model() (or anything else in this module) — despite the old
    # docstring's claim, resolve_model() only ever maps the per-agent model
    # name through model_provider, it never special-cased agents by tier.
    # Now that every *_model field is pinned to the same "claude-sonnet-5"
    # value, keeping unused fast/reasoner groupings around would only
    # mislead readers into thinking agents still get different models.

    def resolve_model(self, model_name: str) -> str:
        """Resolve an agent-level model name to a fully-qualified pydantic_ai model spec.

        Every agent currently defaults to the same model_name
        ("claude-sonnet-5"); this method is only responsible for turning that
        name into a provider-qualified pydantic_ai model spec, based on
        model_provider. It does not apply any per-agent tiering.

        When MODEL_PROVIDER is set, each model name is prefixed with (or
        otherwise mapped through) that provider.

        Examples with MODEL_PROVIDER=litellm (the default):
        - backend_model="claude-sonnet-5"  → "openai:claude-sonnet-5"
          (LiteLLM's OpenAI-compatible proxy resolves the "claude-sonnet-5"
          alias to Anthropic Claude Sonnet, with its own fallback chain.)

        With MODEL_PROVIDER=deepseek:
        - backend_model="deepseek-reasoner"  → "deepseek:deepseek-reasoner"

        With MODEL_PROVIDER=openai:
        - backend_model="deepseek-reasoner"  → "openai:gpt-4o"
        """
        provider = self.model_provider
        if provider:
            if provider == "litellm":
                # Route everything through LiteLLM's OpenAI-compatible proxy interface
                return f"openai:{model_name}"
            if provider == "deepseek":
                return f"deepseek:{model_name}" if ":" not in model_name else model_name
            if provider == "gemini":
                return "gemini:gemini-2.0-flash"
            if provider == "vertexai":
                return "vertexai:gemini-2.0-flash"
            if provider == "openai":
                return "openai:gpt-4o"
            if ":" in model_name:
                return model_name
            return f"{provider}:{model_name}"

        # No provider override — add prefix for non-native models
        if ":" in model_name:
            return model_name
        if model_name.startswith("deepseek-"):
            return f"deepseek:{model_name}"
        return model_name


settings = Settings()
