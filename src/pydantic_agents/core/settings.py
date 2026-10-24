"""Centralized configuration via pydantic-settings.

All environment variables are validated and typed here.
Usage:
    from pydantic_agents.core.settings import settings
    settings.ba_model       # "deepseek-reasoner" by default
    settings.max_retries    # 2 by default
"""

import logging
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Model-name prefixes that identify a provider's own models, and the default used when an agent's configured name
# (e.g. a LiteLLM group such as "deepseek-v4-pro") is not one of them.
logger = logging.getLogger(__name__)

_NATIVE_MODEL_PREFIXES = {
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "gemini": ("gemini-",),
    "vertexai": ("gemini-",),
}
_PROVIDER_DEFAULT_MODELS = {"openai": "gpt-4o", "gemini": "gemini-2.0-flash", "vertexai": "gemini-2.0-flash"}


@cache
def _warn_model_fallback(provider: str, model_name: str, default: str) -> None:
    """Warn once per (provider, name): a typo'd *_MODEL value must not silently become a valid-looking model."""
    logger.warning(
        "Model name %r is not a %s model; using %s. Set the agent's *_MODEL to a %s model name to choose another.",
        model_name,
        provider,
        default,
        provider,
    )


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
    # NOTE: these are real LiteLLM model groups, not Claude aliases. The proxy
    # has no Anthropic credential (Claude is reachable only through Claude Code's
    # OAuth login), so the old shared "claude-sonnet-5" alias silently meant
    # deepseek-v4-pro and then mistral-large. Reasoning roles use
    # "deepseek-v4-pro"; docs and the personal utility agents use the cheap
    # "subagent" group. See resolve_model() below for how a name becomes a
    # pydantic_ai model spec depending on model_provider.
    ba_model: str = "deepseek-v4-pro"
    backend_model: str = "deepseek-v4-pro"
    frontend_model: str = "deepseek-v4-pro"
    test_model: str = "deepseek-v4-pro"
    sre_model: str = "deepseek-v4-pro"
    review_model: str = "deepseek-v4-pro"
    database_model: str = "deepseek-v4-pro"
    devops_model: str = "deepseek-v4-pro"
    docs_model: str = "subagent"
    security_model: str = "deepseek-v4-pro"
    ux_model: str = "deepseek-v4-pro"
    cto_model: str = "deepseek-v4-pro"
    planner_model: str = "deepseek-v4-pro"
    quality_gate_model: str = "deepseek-v4-pro"
    service_engineer_model: str = "deepseek-v4-pro"

    holiday_planner_model: str = "subagent"
    region_analyzer_model: str = "subagent"
    asymmetric_reviewer_model: str = "deepseek-v4-pro"
    current_events_model: str = "subagent"
    weather_model: str = "subagent"
    accountant_model: str = "subagent"
    price_analyst_model: str = "subagent"
    market_shark_model: str = "subagent"
    crypto_oracle_model: str = "subagent"

    # Pipeline Configuration
    output_dir: Path = Path("./generated-app")
    max_retries: int = 3  # Autonomous Remediation Loop: max retries before escalating to Lead Orchestrator
    max_concurrent_llm_calls: int = 5
    # Per-agent-run limits (see core/limits.py). Timeouts are not retried; 429/5xx/connection errors are.
    agent_retries: int = 2  # pydantic_ai output/tool validation retries
    agent_request_limit: int = 40
    # Cumulative input+output tokens per run: every request resends its context, so this is much larger than any
    # single response (the database agent caps one response at 8192 tokens).
    agent_total_tokens_limit: int = 2_000_000
    # Total tries for 429/5xx/connection errors (separate from agent_retries). 1 because LiteLLM already retries
    # once and walks its fallback chain: stacking more here multiplied one call into ~12 upstream attempts.
    agent_transport_attempts: int = 1
    agent_timeout_seconds: float = 600.0
    checkpoint_dir: Path = Path("./.claw-checkpoints")
    # Pipeline checkpoint backend: "disk" (JSON files) or "turso" (embedded Turso DB; needs the `turso` extra).
    state_backend: Literal["disk", "turso"] = "disk"
    vector_db_dir: Path = Path("./.claw-vectordb")

    go_version: str = "1.26"
    # One convention for the Go service entry point, shared by the backend prompt, the implementation stage and
    # the devops templates, so generated CI and Makefiles build the generated backend.
    go_entrypoint: str = "cmd/server/main.go"
    # (directory of the entry point, derived so a bare "main.go" still yields a valid "./." build path)
    # Pinned tool versions for generated CI (the review agent rejects unpinned 'latest').
    golangci_lint_version: str = "v1.64.8"
    govulncheck_version: str = "v1.1.4"

    # NOTE: The _FAST_TIER/_REASONER_TIER capability-tier frozensets that used
    # to live here were removed. They were never actually read by
    # resolve_model() (or anything else in this module) — despite the old
    # docstring's claim, resolve_model() only ever maps the per-agent model
    # name through model_provider, it never special-cased agents by tier.
    # Unused fast/reasoner groupings would only mislead readers; the tiering
    # is the per-agent *_model defaults above.

    @property
    def go_entrypoint_dir(self) -> str:
        """Directory that holds the Go entry point, e.g. "cmd/server" ("." for a bare "main.go")."""
        return PurePosixPath(self.go_entrypoint).parent.as_posix()

    def resolve_model(self, model_name: str) -> str:
        """Resolve an agent-level model name to a fully-qualified pydantic_ai model spec.

        Each agent has its own *_model default (see above); this method is
        only responsible for turning that name into a provider-qualified
        pydantic_ai model spec, based on model_provider.

        When MODEL_PROVIDER is set, each model name is prefixed with (or
        otherwise mapped through) that provider.

        Examples with MODEL_PROVIDER=litellm (the default):
        - backend_model="deepseek-v4-pro"  → "openai:deepseek-v4-pro"
          (LiteLLM's OpenAI-compatible proxy serves the model group of that
          name, with its own fallback chain.)

        With MODEL_PROVIDER=deepseek:
        - backend_model="deepseek-reasoner"  → "deepseek:deepseek-reasoner"

        With MODEL_PROVIDER=openai (likewise gemini, vertexai):
        - backend_model="deepseek-v4-pro"  → "openai:gpt-4o" (not an OpenAI model name, so the default is used)
        - backend_model="gpt-4.1"  → "openai:gpt-4.1"
        """
        provider = self.model_provider
        if provider == "litellm":
            # Route everything through LiteLLM's OpenAI-compatible proxy interface
            return f"openai:{model_name}"
        if ":" in model_name:
            return model_name
        if not provider:
            # No provider override: only DeepSeek's native names need a prefix.
            return f"deepseek:{model_name}" if model_name.startswith("deepseek-") else model_name
        if provider in _NATIVE_MODEL_PREFIXES:
            return self._resolve_native(provider, model_name)
        return f"{provider}:{model_name}"

    @staticmethod
    def _resolve_native(provider: str, model_name: str) -> str:
        """Keep a name that belongs to the provider; otherwise fall back to its default, and say so."""
        if model_name.startswith(_NATIVE_MODEL_PREFIXES[provider]):
            return f"{provider}:{model_name}"
        default = _PROVIDER_DEFAULT_MODELS[provider]
        _warn_model_fallback(provider, model_name, default)
        return f"{provider}:{default}"


settings = Settings()
