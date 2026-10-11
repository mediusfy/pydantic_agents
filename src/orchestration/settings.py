"""Centralized configuration via pydantic-settings.

All environment variables are validated and typed here.
Usage:
    from orchestration.settings import settings
    settings.ba_model       # "deepseek-reasoner" by default
    settings.max_retries    # 2 by default
"""

from pathlib import Path

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
    # NOTE: When model_provider is set (e.g. "deepseek"), resolve_model() remaps ALL
    # model names through the provider. So these defaults serve as capability tiers
    # and are resolved at runtime — actual model used depends on model_provider and
    # the resolve_model mapping below.
    ba_model: str = "local-failover"
    backend_model: str = "deepseek-reasoner"
    frontend_model: str = "local-failover"
    test_model: str = "deepseek-reasoner"
    sre_model: str = "deepseek-reasoner"
    review_model: str = "deepseek-reasoner"
    database_model: str = "deepseek-reasoner"
    devops_model: str = "deepseek-reasoner"
    docs_model: str = "local-failover"
    security_model: str = "deepseek-reasoner"
    ux_model: str = "local-failover"
    cto_model: str = "deepseek-reasoner"

    holiday_planner_model: str = "local-failover"
    region_analyzer_model: str = "local-failover"
    asymmetric_reviewer_model: str = "deepseek-reasoner"
    current_events_model: str = "local-failover"
    weather_model: str = "local-failover"
    accountant_model: str = "deepseek-reasoner"
    price_analyst_model: str = "deepseek-reasoner"
    market_shark_model: str = "deepseek-reasoner"
    crypto_oracle_model: str = "deepseek-reasoner"

    # Pipeline Configuration
    output_dir: Path = Path("./generated-app")
    max_retries: int = 3  # Autonomous Remediation Loop: max retries before escalating to Lead Orchestrator
    max_concurrent_llm_calls: int = 5
    checkpoint_dir: Path = Path("./.claw-checkpoints")
    vector_db_dir: Path = Path("./.claw-vectordb")

    go_version: str = "1.26"

    # ── Model Capability Tiers ──────────────────────────────────────────────
    # These define which agents use fast/cheap models vs reasoning models.
    # They are consulted by resolve_model() unless MODEL_PROVIDER_ENFORCE_ALL
    # is set (which overrides all to the same provider for strict consistency).
    _FAST_TIER: frozenset[str] = frozenset(
        {
            "frontend",
            "docs",
            "ux",
            "ba",
            "holiday_planner",
            "region_analyzer",
            "current_events",
            "weather",
        }
    )
    _REASONER_TIER: frozenset[str] = frozenset(
        {
            "backend",
            "test",
            "review",
            "security",
            "quality_gate",
            "sre",
            "database",
            "devops",
            "cto",
            "asymmetric_reviewer",
            "accountant",
            "price_analyst",
            "market_shark",
            "crypto_oracle",
        }
    )

    def resolve_model(self, model_name: str) -> str:
        """Resolve an agent-level model name to a fully-qualified pydantic_ai model spec.

        Capability-tier mapping ensures fast agents (frontend, docs, ux) use
        cheaper/faster models, while reasoning agents (backend, review, security)
        use DeepSeek-reasoner. This is the default behavior.

        When MODEL_PROVIDER is set, each model name is prefixed with that
        provider. The tiered defaults per agent are preserved.

        When MODEL_PROVIDER_ENFORCE_ALL is set to true, ALL agents use the
        same model (provider's top model). This overrides tier decisions for
        strict consistency (e.g. when MCP JSON handoff requires identical
        output structure).

        Examples with MODEL_PROVIDER=deepseek:
        - backend_model="deepseek-reasoner"  → "deepseek:deepseek-reasoner"
        - frontend_model="deepseek-chat"     → "deepseek:deepseek-chat"
        - ux_model="deepseek-chat"           → "deepseek:deepseek-chat"
        - docs_model="deepseek-chat"         → "deepseek:deepseek-chat"

        With MODEL_PROVIDER=openai:
        - backend_model="deepseek-reasoner"  → "openai:gpt-4o"
        - frontend_model="deepseek-chat"     → "openai:gpt-4o"
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
