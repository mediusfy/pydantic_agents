# pydantic-agents

Pydantic AI agents routed through a [LiteLLM](https://github.com/BerriAI/litellm) proxy,
extracted from the `mediusfy/agents` orchestration monorepo
(`src/orchestration/agents/` plus the dependency closure it needs to run standalone).

## Architecture

Each agent is a [pydantic_ai](https://ai.pydantic.dev/) `Agent` with a typed `output_type`
(a pydantic model in `orchestration/models/core.py`) and a model resolved through
`orchestration/models/factory.py::create_model`.

Model routing is provider-agnostic via `Settings.resolve_model()` in `orchestration/settings.py`:

- `MODEL_PROVIDER=litellm` (default) — every agent call goes through an OpenAI-compatible
  LiteLLM proxy (`LITELLM_BASE_URL`, auth via `LITELLM_VIRTUAL_KEY`/`LITELLM_MASTER_KEY`).
- `MODEL_PROVIDER=deepseek|openai|gemini|vertexai` — routes natively to that provider.
- Per-agent model fields (`backend_model`, `frontend_model`, ...) plus `_FAST_TIER` /
  `_REASONER_TIER` sets in `settings.py` assign cheap vs. reasoning models per agent role.

Agents are registered into a shared `AgentRegistry` (`orchestration/registry.py`) via
`orchestration/agents/registration.py`, which is how the parent project exposes them as
MCP tools (`mcp__litellm__agents-run_<name>`).

## Layout

```
src/orchestration/
  agents/            # 17 SDLC agents (cto, ba, backend, frontend, test_eng, security, ...)
    non_technical/   # 9 utility agents (weather, accountant, crypto_oracle, ...)
  models/
    core.py          # typed output_type models (Artifact, AgentAdvice, ReviewResult, ...)
    factory.py       # create_model(): routes model specs through LiteLLM/native providers
  adapters/agents.py # PydanticAIAgentProvider used by the registry
  registry.py        # AgentRegistry — agent registration/lookup
  session_recorder.py, telemetry.py, exceptions.py, planning.py, pipeline/context.py
  settings.py        # pydantic-settings config, model routing/tiering
```

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # fill in LITELLM_VIRTUAL_KEY or a direct provider key
```

```python
from orchestration.agents.cto import cto_agent

result = await cto_agent.run("Review this proposed service architecture...")
```

## Known improvement areas (carried over from extraction review)

This was extracted as-is; the following were identified during review and are **not**
yet fixed here — tracked for follow-up:

1. **Prompts promise tools that don't exist.** `crypto_oracle.py`, `database.py`,
   `devops.py` system prompts mandate tool calls (`search_news`, `get_historical_prices`,
   `update_documentation`, ...) that are never registered via `@agent.tool`. Either wire
   real tools or strip those imperative steps from the prompts.
2. **`factory.py` silently degrades on missing LiteLLM key** — falls back to returning
   the raw model spec string instead of raising, which can silently bypass the proxy
   (and its cost tracking/guardrails) using whatever ambient API key is present.
3. **`"pytest" in sys.modules` test-detection hack** in `factory.py` — replace with
   explicit dependency injection (a settings flag or injected key provider).
4. **Hardcoded fallback URL** `https://litellm-proxy.run.app` in `factory.py` — should
   fail loudly if `LITELLM_CLOUD_URL` is unset in a cloud environment instead of guessing.
5. **No HTTP timeout/retry/observability on the LiteLLM client** — `OpenAIProvider` is
   constructed with no explicit `httpx` timeout; add one plus structured per-call logging
   (settings already carries unused `logfire_token`/`logfire_api_key`).
6. **~150 lines of duplicated `Agent(...)` boilerplate** across ~20 agent files — extract
   a shared `make_agent(model_setting, prompt, output_type, deps_type=None, retries=5)`
   factory.
7. **Module-level singleton agents** (`cto_agent`, `ba_agent`, ...) instantiated at import
   time make test isolation/mocking harder — consider exposing `build_<name>_agent()`
   constructors alongside the default singleton.
8. **Inconsistent `deps_factory` usage in `registration.py`** — utility agents require a
   `deps_factory` lambda, SDLC agents never pass one even when they declare `deps_type`.
9. **Three manual touch points to add one new agent**: a `settings.py` field, a tier-set
   entry (`_FAST_TIER`/`_REASONER_TIER`), and a `registration.py` entry — no validation
   keeps these in sync.

## Relationship to pydantic_ai + LiteLLM

This repo is a live example of the pattern the parent org was evaluating: pydantic_ai's
typed `Agent`/`output_type` system gives structured, validated outputs per agent role,
while LiteLLM centralizes provider auth, routing, and cost/usage tracking behind one
OpenAI-compatible endpoint. The pattern already works end-to-end here; the improvements
above are about hardening the provider-routing edge cases and reducing per-agent
boilerplate, not about the overall approach.
