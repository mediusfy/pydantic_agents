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

Common tasks are wrapped in the `Makefile`; run `make help` for the list (`make install`,
`make test`, `make lint`, `make ci`, `make serve`, `make seed`).

```python
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.registry import registry

register_all_agents()
agent = registry.get("cto")  # tools are attached here, lazily
result = await agent.run("Review this proposed service architecture...")
```

## Running evals

`make eval` (or `promptfoo eval`) runs the promptfoo suite in `promptfooconfig.yaml`. The
provider in `evals/provider.py` runs registered agents in-process, so only the LiteLLM
gateway is needed (no Postgres), and the `llm-rubric` grader goes through the same gateway.
The key must be allowed to call `claude-sonnet-5`, which the gateway aliases to
`deepseek-v4-pro`:

```bash
export LITELLM_BASE_URL=https://litellm.jocham.io
export LITELLM_VIRTUAL_KEY=sk-...   # e.g. LITELLM_JUDGE_KEY from ~/.openclaw/.env, quotes stripped
promptfoo eval
```

Quick check that a key can call chat models (an empty `data` list means it cannot, and
evals will fail with a 403):

```bash
curl -s "$LITELLM_BASE_URL/v1/models" -H "Authorization: Bearer $LITELLM_VIRTUAL_KEY"
```

Add a case by appending to `tests:` in `promptfooconfig.yaml`, or cover another agent with
a new `providers:` entry whose `config.agent` is its registry name.

## Where the agents run

In production the agents, the orchestration pipeline and the MCP server run in-cluster
as the `pydantic-agents` Helm release (namespace `pydantic-agents`), and are exposed as
MCP tools through the LiteLLM MCP gateway as `pydantic_agents-run_<name>`
(e.g. `mcp__litellm__pydantic_agents-run_review`).

- **Entrypoint:** the image runs `uvicorn pydantic_agents.mcp_server:mcp.sse_app` on port 8080.
- **Model calls:** routed through the in-cluster LiteLLM proxy, so no upstream provider key
  reaches the agent pod.
- **Context:** each call auto-prepends vector-indexed context (ADRs, architecture rules,
  conventions) for that agent, read from the pgvector `agents` database. See
  [Seeding the knowledge base](#seeding-the-knowledge-base) for how it is refreshed.

### How it is deployed

> **Dependency lockfile.** The image installs from `requirements.lock`, a hash-pinned
> (`--require-hashes`, wheels only) resolve for linux/arm64 + Python 3.12 including the `vector` and
> `turso` extras. After changing dependencies or extras in `pyproject.toml`, run `make lock` and commit
> the result; `make lock-check` fails if it is stale. `uv.lock` (what `make install` / `uv sync` use for local
> development) is tracked too: run `uv lock` after changing dependencies and commit it; `make lock-check` checks it as well.
> `requirements-build.in` lists the build backend
> (hatchling) so the project itself can be installed with `--no-build-isolation`.

1. Merging to `main` triggers `.github/workflows/docker.yml`, which builds the `Dockerfile`
   (`linux/arm64`, since the cluster node is ARM) and pushes
   `us-east1-docker.pkg.dev/home-lab-ai-pipelines/mediusfy/pydantic-agents:<short-sha>`
   (and `:latest`) to Artifact Registry.
2. The Helm chart (`k8s/charts/agents`) and its values (`values-pydantic-agents.yaml`) live in
   the `mediusfy/infrastructure` repo, not here. Roll out a new version by updating the image
   tag there.
3. CI for this repo (`.github/workflows/ci.yml`) runs lint, tests and an MCP smoke check; run
   the same locally with `make ci`.

### Checkpoint state backend (`STATE_BACKEND`)

Pipeline checkpoints and stage events are stored on local disk by default. Set
`STATE_BACKEND=turso` to use an embedded [Turso](https://docs.turso.tech/guides/agent-databases)
database instead (design and rationale: `docs/turso-agent-databases-plan.md`).

| Setting | Values | Default | Effect |
|---|---|---|---|
| `STATE_BACKEND` | `disk`, `turso` | `disk` | `disk`: one JSON file per `{run_id}_{stage}` under `CHECKPOINT_DIR/<project>/`. `turso`: a single `CHECKPOINT_DIR/<project>/state.db` holding `pipeline_checkpoints` and `pipeline_stage_events`. |
| `CHECKPOINT_DIR` | path | `./.claw-checkpoints` | Parent directory for either backend. |

Prerequisites and caveats for enabling it in the cluster:

- **Install the extra.** `turso` needs the optional `pyturso` dependency
  (`pip install "pydantic-agents[turso]"`). The image is built from `requirements.lock`, which includes it, so
  the image is ready; a build without the extra fails at startup with a `RuntimeError` when
  `STATE_BACKEND=turso`.
- **Persistence.** `state.db` lives under `CHECKPOINT_DIR`. Unless that path is on a volume,
  it is lost on pod restart, exactly like the disk backend. Surviving restarts (a PVC or
  Turso Sync) is not set up yet and is tracked in AP-36.
- **Single writer.** One process should own a project's `state.db`; do not run several
  replicas against the same file.
- **Stage events** (retries and quality-gate verdicts per run) are only recorded with the
  `turso` backend. Query them with `TursoStageEventRecorder(project).list_events(run_id)`.
- **Secrets.** Embedded mode needs none. If Turso Sync is added later, its URL/auth token must
  come from a SOPS-managed secret in the Helm release, never plaintext in values or this repo.
- **Rollback.** Unset `STATE_BACKEND` (or set `disk`); existing runs on the other backend are
  not migrated.

### More information

- Deployment config: `mediusfy/infrastructure` (`k8s/charts/agents`, `values-pydantic-agents.yaml`); set `STATE_BACKEND` there as an env value (see above).
- Health: `kubectl -n pydantic-agents get pods`; the image needs the `vector` extra
  (`pip install pydantic-agents[vector]`) for context retrieval.
- Invocation guidance (which agents to call and when): `rules/mediusfy-agents.md` in the
  `agent-hub` tools repo.
- Migrated from the retired `mediusfy/agents` repo (archived 2026-10-14); its `agents-run_<name>`
  tools are replaced by `pydantic_agents-run_<name>`.

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


## Seeding the knowledge base

The agents prepend vector-indexed context (ADRs, architecture, conventions, READMEs) to each call.
`scripts/seed_projects.py` refreshes it from recently pushed GitHub repos (dry run by default):

```
kubectl -n pgvector port-forward pod/pgvector-0 15432:5432 &
DATABASE_URL=postgresql://<user>:<pw>@127.0.0.1:15432/agents python scripts/seed_projects.py --apply
```

Chunks are upserted, then only ones whose source docs no longer exist are deleted; any error aborts before deleting. `chatbot` is excluded by
default (`--exclude`). Use `query_context(..., project="<repo>")` to scope a search to one repo.
