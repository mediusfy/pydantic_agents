"""Centralized configuration via pydantic-settings.

All environment variables are validated and typed here.
Usage:
    from pydantic_agents.core.settings import settings
    settings.ba_model       # "deepseek-v4-pro" by default
    settings.max_retries    # 3 by default
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
    "deepseek": ("deepseek-chat", "deepseek-reasoner", "deepseek-coder"),
}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "vertexai": "gemini-2.0-flash",
    "deepseek": "deepseek-chat",
}


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
    litellm_base_url: str | None = None
    litellm_virtual_key: str | None = None
    litellm_master_key: str | None = None
    litellm_env: str = "local"
    litellm_cloud_url: str = "https://litellm-proxy.run.app"
    logfire_token: str | None = None
    logfire_api_key: str | None = None

    database_url: str | None = None

    # Supported providers: deepseek, gemini, vertexai, openai, litellm
    model_provider: str | None = "litellm"

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
    innovation_scout_model: str = "deepseek-v4-pro"
    qa_executor_model: str = "deepseek-v4-pro"
    observer_model: str = "deepseek-v4-pro"
    sre_autopilot_model: str = "deepseek-v4-pro"
    product_strategist_model: str = "deepseek-v4-pro"
    critic_model: str = "deepseek-v4-pro"
    api_contract_model: str = "deepseek-v4-pro"
    observability_engineer_model: str = "deepseek-v4-pro"
    finops_model: str = "subagent"
    release_manager_model: str = "subagent"

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
    # Directories the MCP tool seed_project_context may read, separated by os.pathsep (":" on Linux/macOS). Empty means
    # only output_dir. scripts/seed_projects.py is not restricted: it seeds clones it made itself.
    seed_roots: str = ""
    # Whether an approved run may merge its files into the server's own working directory (the "default" project's
    # target root is "."). Off: such a run keeps its output under output_dir and the merge stage does nothing.
    merge_into_server_tree: bool = False
    # Host and Origin header patterns accepted by the MCP server's DNS-rebinding middleware (comma-separated).
    mcp_allowed_hosts: str = "127.0.0.1:*,localhost:*,[::1]:*"
    mcp_allowed_origins: str = "http://127.0.0.1:*,http://localhost:*,http://[::1]:*"
    max_retries: int = 3  # Autonomous Remediation Loop: max retries before escalating to Lead Orchestrator
    # QaCrew's bounded fix loop (KAN-487): go_test gets at most this many rounds to make a failing TestRunReport
    # pass before the review stage runs regardless; coverage below the floor fails the gate even if tests pass.
    qa_max_fix_rounds: int = 2
    qa_min_coverage_percent: float = 80.0
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
    # Per-request HTTP timeout on the model client (read/write/pool); a hung upstream must not hold a concurrency
    # permit for the whole agent_timeout_seconds. 0 disables it.
    agent_http_timeout_seconds: float = 300.0
    # Output cap sent with every model call (max_tokens). 0 sends none.
    agent_max_output_tokens: int = 16384
    # Push activity logs into the local Claude Code / Antigravity memory directories after a pipeline run. Off by
    # default: in the pod there are no such directories, and on a workstation it writes into every agent's memory.
    memory_bridge_sync: bool = False
    checkpoint_dir: Path = Path("./.claw-checkpoints")
    # Pipeline checkpoint backend: "disk" (JSON files) or "turso" (embedded Turso DB; needs the `turso` extra).
    state_backend: Literal["disk", "turso"] = "disk"
    vector_db_dir: Path = Path("./.claw-vectordb")
    # Embeddings for vector_db: "local" runs sentence-transformers in-process, "litellm" calls the proxy's
    # OpenAI-compatible /embeddings endpoint. The pgvector column is vector(embedding_dim), so changing the provider,
    # model or dimension changes the vector space: every collection must be re-seeded afterwards.
    embedding_provider: Literal["local", "litellm"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Rerank the fused candidates through the proxy's /rerank endpoint (litellm provider only); empty disables it.
    rerank_model: str = ""
    # Fuse pgvector cosine search with Postgres full-text search (reciprocal rank fusion); False keeps vector-only.
    retrieval_hybrid: bool = True

    go_version: str = "1.26"
    # One convention for the Go service entry point, shared by the backend prompt, the implementation stage and
    # the devops templates, so generated CI and Makefiles build the generated backend.
    go_entrypoint: str = "cmd/server/main.go"
    # (directory of the entry point, derived so a bare "main.go" still yields a valid "./." build path)
    # Pinned tool versions for generated CI (the review agent rejects unpinned 'latest').
    golangci_lint_version: str = "v1.64.8"
    govulncheck_version: str = "v1.1.4"
    # Module proxy and checksum DB for go builds/tests of generated code (core/utils.py::sanitized_subprocess_env).
    # A trust boundary: no `direct`, so generated go.mod files can only pull public modules through this proxy.
    # An air-gapped deployment points it at its public-module mirror, never at a proxy that serves private modules.
    untrusted_goproxy: str = "https://proxy.golang.org"
    untrusted_gosumdb: str = "sum.golang.org"

    # Default deps for SDLC agents called without any (standalone MCP calls, delegation, evals). The CTO verifies
    # architecture against these, so they are an organisation policy: the defaults are placeholders, set them per
    # deployment. A pipeline that passes its own OrgContext / ProjectState is not affected.
    org_tech_stack: str = "Go,Vue 3,PostgreSQL,NATS,gRPC"  # comma-separated
    org_compliance_tier: str = "standard"
    org_monthly_cloud_budget: float = 5000.0  # USD
    org_environment: str = "development"
    org_team_owner: str = "unassigned"

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
        if provider == "test":
            return "test"
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
