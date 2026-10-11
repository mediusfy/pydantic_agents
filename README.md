# pydantic-agents

Pydantic AI agents routed through a [LiteLLM](https://github.com/BerriAI/litellm) proxy,
extracted from the `mediusfy/agents` orchestration monorepo as a standalone, installable
package (`pydantic_agents.core` + `pydantic_agents.agents`), with the same module identity
(`registry`, `settings`, ...) usable by any consumer that installs it.

## Architecture

Each agent is a [pydantic_ai](https://ai.pydantic.dev/) `Agent` with a typed `output_type`
(a pydantic model in `pydantic_agents/core/models/core.py`) and a model resolved through
`pydantic_agents/core/models/factory.py::create_model`.

Model routing is provider-agnostic via `Settings.resolve_model()` in
`pydantic_agents/core/settings.py`:

- `MODEL_PROVIDER=litellm` (default) — every agent call goes through an OpenAI-compatible
  LiteLLM proxy (`LITELLM_BASE_URL`, auth via `LITELLM_VIRTUAL_KEY`/`LITELLM_MASTER_KEY`).
- `MODEL_PROVIDER=deepseek|openai|gemini|vertexai` — routes natively to that provider.
- Per-agent model fields (`backend_model`, `frontend_model`, ...) in `settings.py` currently
  all default to the same `"claude-sonnet-5"` alias (resolved by the LiteLLM proxy to
  Anthropic Claude Sonnet, with its own fallback chain) rather than being split by
  fast/reasoner tier. The `_FAST_TIER`/`_REASONER_TIER` frozensets that used to live here
  were removed — they were never actually read by `resolve_model()`.

Agents are registered into a shared `AgentRegistry` (`pydantic_agents/core/registry.py`) via
`pydantic_agents/agents/registration.py`. **Tools are attached lazily, not via `@agent.tool`
decorators in the agent files**: `registry.get(name)` triggers `AgentRegistry._lazy_init`,
which looks up `AGENT_TOOL_CATEGORIES[name]` in `tool_assignments.py` and calls
`core/tools/registry.py::apply_tools`, which attaches every tool in `core/tools/*.py` whose
category matches. E.g. `crypto_oracle` gets category `finance`/`news`, which is how it ends up
with `search_news`/market-data tools even though `crypto_oracle.py` itself has no `@agent.tool`
decorators. This is how the parent monorepo exposes agents as MCP tools
(`mcp__litellm__agents-run_<name>`).

## Layout

```
src/pydantic_agents/
  core/
    settings.py          # pydantic-settings config, model routing/tiering
    registry.py           # AgentRegistry — agent registration/lookup, lazy tool attachment
    tool_assignments.py    # AGENT_TOOL_CATEGORIES: agent name -> tool categories
    tools/                 # ~15 tool modules (web, search, security, finance, memory, a2a, ...)
    vector_db.py, utils.py, telemetry.py, exceptions.py, session_recorder.py, planning.py
    pipeline/context.py
    adapters/agents.py     # PydanticAIAgentProvider used by the registry
    models/
      core.py              # typed output_type models (Artifact, AgentAdvice, ReviewResult, ...)
      factory.py           # create_model(): routes model specs through LiteLLM/native providers
  agents/
    registration.py
    (17 SDLC agents: cto, ba, backend, frontend, test_eng, security, ...)
    non_technical/         # 9 utility agents (weather, accountant, crypto_oracle, ...)
```

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # fill in LITELLM_VIRTUAL_KEY or a direct provider key
```

```python
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.registry import registry

register_all_agents()
agent = registry.get("cto")  # tools are attached here, lazily
result = await agent.run("Review this proposed service architecture...")
```

## Known improvement areas (carried over from extraction review)

This was extracted as-is; the following were identified during review and are **not**
yet fixed here — tracked for follow-up:

1. ~~Prompts promise tools that don't exist~~ — **correction**: tools are attached lazily by
   category via `registry.get()` → `_lazy_init` → `apply_tools` (see Architecture above), not
   via `@agent.tool` decorators in the agent files. `crypto_oracle`/`database`/`devops` do get
   real tools through `tool_assignments.py`. The actual gap here is invisibility, not absence:
   nothing in `crypto_oracle.py` itself hints that its capabilities come from an external
   category mapping, so a reader (or an agent-generated review) can't tell what tools an agent
   has without also reading `tool_assignments.py` + `tools/registry.py::get_tools_for_agent`.
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
9. **Two manual touch points to add one new agent**: a `settings.py` field and a
   `registration.py` entry — no validation keeps these in sync.

## Relationship to pydantic_ai + LiteLLM

This repo is a live example of the pattern the parent org was evaluating: pydantic_ai's
typed `Agent`/`output_type` system gives structured, validated outputs per agent role,
while LiteLLM centralizes provider auth, routing, and cost/usage tracking behind one
OpenAI-compatible endpoint. The pattern already works end-to-end here; the improvements
above are about hardening the provider-routing edge cases and reducing per-agent
boilerplate, not about the overall approach.
