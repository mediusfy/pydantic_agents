# Architecture

Short map for agents. The README has the longer description, and [docs/turso-agent-databases-plan.md](../../docs/turso-agent-databases-plan.md) the design of the Turso checkpoint backend.

## Entry points

| Entry point | Where | Notes |
|---|---|---|
| MCP server (SSE) | `mcp_server.py`, `uvicorn pydantic_agents.mcp_server:mcp.sse_app` | The image's `CMD`, port 8080. `make serve` binds 127.0.0.1 locally. |
| Library | `agents.registration.register_all_agents()` then `core.registry.registry.get(name)` | For consumers that install the package. |
| Bridge (stdio JSON) | `core/agy_bridge/server.py`, console script `orchestration-agy` | Line-delimited JSON commands (`ping`, `list-agents`, `get-agent`, ...). |
| Evals | `evals/provider.py` (promptfoo) | Runs agents in-process, without vector context. |
| Knowledge-base seeding | `scripts/seed_projects.py` | Dry run unless `--apply`; needs `DATABASE_URL`. |

## One agent call through MCP

1. Importing `mcp_server.py` configures observability, registers every agent, and creates one `run_<name>` MCP tool per registered agent.
2. `run_agent_by_name` gets the agent from the registry; the first `get` attaches its tools by category (`tool_assignments.py` → `tools/registry.py::apply_tools`).
3. `context_engine.retrieve_context` prepends up to three snippets from the vector store (ADRs, architecture, conventions) to the request.
4. The agent runs with `usage_limits(agent_name)` (40 requests, 60 for `backend`, `frontend` and `service_engineer`, 2 M tokens; `core/limits.py`) and its `deps_factory`; output is stringified. Failures become `AgentExecutionError`; telemetry is logged best-effort.

## Model routing

`core/settings.py` holds one model field per agent (`deepseek-v4-pro`, or `subagent` for docs and the utility agents; real LiteLLM groups, not `claude-*` aliases). `Settings.resolve_model()` turns it into a spec for `MODEL_PROVIDER` (`litellm` by default, or `deepseek`, `openai`, `gemini`, `vertexai`), and `core/models/factory.py::create_model` builds the pydantic_ai model. With `litellm`, the base URL comes from `LITELLM_BASE_URL` / `OPENAI_BASE_URL` (or a default by `LITELLM_ENV`) and the key from `LITELLM_VIRTUAL_KEY` / `LITELLM_MASTER_KEY`. Without a key it logs a warning and returns the raw spec (README "Known improvement areas" item 2).

## Orchestration pipeline

- `core/pipeline/config.py` defines blueprints, `Framework` (`legacy`, `langgraph`, `crewai`, `autogen`, `composite`) and stages (`ba`, `advisory`, `implementation`, `quality_gate`, `merge`).
- `core/pipeline/factory.py::build_pipeline` dispatches by framework; `crewai` and `autogen` need their extras. The legacy `OrgitrationPipeline` (`core/pipeline/orchestrator.py`) runs `core/stages/*` in order.
- MCP tools `run_orchestration_pipeline`, `check_pipeline_status`, `resume_pipeline_run` and `cancel_pipeline_run` run pipelines as asyncio tasks (`core/pipeline_jobs.py`). Run records are also written to a `pipeline_runs` table in the pgvector database (`core/run_store.py`, best-effort), so a restart leaves an `interrupted` record that `resume_pipeline_run` can continue. `approve_cto_gate` writes the marker the CTO gate (`core/cto_gate.py`) waits for; the gate reads its content, so `false` rejects. `submit_async_task`, `check_task_status` and `wait_for_task` use the in-process event broker and are not persisted.
- Checkpoints and stage events go through `core/adapters/state_repository.py`: disk JSON files by default, or an embedded Turso database with `STATE_BACKEND=turso` (needs the `turso` extra). One process should own a project's `state.db`.

## Modules worth knowing

- Agents: `agents/*.py` (SDLC roles), `agents/non_technical/*.py` (utility agents), `agents/registration.py`.
- Registry and tools: `core/registry.py`, `core/tool_assignments.py`, `core/tools/*.py`.
- Context and memory: `core/context_engine.py`, `core/vector_db.py` (pgvector; needs the `vector` extra), `core/memory.py`, `core/memory_bridge.py` (syncs with local agent-memory directories under the user's home).
- Output: `core/writer.py`, `core/adapters/workspace.py`, `core/utils.py` (`safe_path_join`).
- Observability: `core/observability.py`, `core/telemetry.py`, `core/session_recorder.py`, `core/live_trace.py`, `core/event_broker.py`.

## Build and release

Merging to `main` runs `.github/workflows/docker.yml`, which builds the `Dockerfile` for linux/arm64 from `requirements.lock` and pushes `:<short-sha>` and `:latest`. Deployment config lives outside this repo.
