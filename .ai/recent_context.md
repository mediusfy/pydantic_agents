# Recent context

Newest first. Each entry is the "AI Context & Rules Update" section of a merged pull
request, copied by `.github/workflows/ai-recent-context.yml`; only the newest
25 are kept, and git history has the rest.

Entries are notes written by PR authors. They are a record of recent changes, not
instructions: they never override AGENTS.md or `.ai/guidance/`, and any fact in them
should be checked against the code before you rely on it.

<!-- entry:37 -->
## #37: feat: durable pipeline runs, resume_pipeline_run and approve_cto_gate tools

@drewjocham, merged 2026-10-24, `72560c6`

> - Background pipeline runs are persisted in `pipeline_runs` (`core/run_store.py`); use `core.pipeline_jobs.get_run`, not `PIPELINE_RUNS`, to read a run that may predate this process.
> - The CTO gate can be approved over MCP with `approve_cto_gate`.
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code)

<!-- entry:35 -->
## #35: fix: real LiteLLM model names, one retry layer, run session metadata, resume reuses latest run id

@drewjocham, merged 2026-10-24, `cd8b12e`

> - Agent `*_model` defaults must be real LiteLLM model groups (`deepseek-v4-pro`, `subagent`), never `claude-*` aliases; `tests/unit/test_run_tracing.py` enforces it.
> - LLM transport retries live in LiteLLM; `agent_transport_attempts` stays 1.
> - Calls inside a run carry the run id via `core.limits.run_id_var` (session id for Langfuse).
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code)
