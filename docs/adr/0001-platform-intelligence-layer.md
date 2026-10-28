# ADR-0001: Platform Intelligence Layer (performance/quality observation, innovation, product ideation, self-healing SRE)

- Status: Proposed
- Date: 2026-10-28
- Owners: cto

## Context

The platform (this repo) registers each SDLC role as a `pydantic_ai.Agent` exposed via MCP (`agents/registration.py`
→ `mcp_server.py`), orchestrated through blueprint pipelines (`core/pipeline/config.py`, `core/pipeline/orchestrator.py`).
Several quality and health signals already exist but are siloed and not aggregated:

- `quality_gate_agent` and `review_agent` produce a structured verdict per run, but it is not persisted or trended —
  only the pipeline's own result for that one run sees it.
- `core/telemetry.py` appends `{agent, duration, success}` to a local JSONL file. Nothing reads it back; it has no
  database, no trends, no way to ask "which agent regressed this week".
- Logfire instrumentation (`core/observability.py`) emits per-call/tool spans when a token is configured, but the
  platform itself never queries those spans.
- `core/healthcheck.py` runs a one-shot startup check (API keys, directories, DB reachability). There is no
  continuous health loop and no automatic remediation.
- There is no mechanism that looks across pipeline runs to find recurring failure patterns, suggest adopting a new
  technique, or propose new agent/product capabilities from accumulated requirement history.

The user has asked for a layer that:
1. Observes agent performance and output quality to surface gaps/bugs proactively.
2. Scouts for new techniques/technologies and proposes platform improvements.
3. Proposes new customer-facing products/capabilities based on usage patterns.
4. Monitors overall system health and addresses issues automatically.
5. Operates autonomously (scheduled, not only on-demand).

## Decision

Add four new agent roles, following the existing registration/tool-category pattern exactly (`registry.register(...)`
+ an entry in `AGENT_TOOL_CATEGORIES`), fed by upgraded persistence of signals that already exist. No change to any
existing SDLC agent, prompt, or pipeline stage.

| Role | Responsibility | Primary inputs | Output |
|---|---|---|---|
| `observer` | Continuous quality/performance review across runs | `platform_metrics` (new table), `quality_gate`/`review` outputs, Logfire spans | Rows in a new `platform_findings` table: regressions, recurring failure patterns, flaky agents, latency/cost drift |
| `innovation_scout` | Scans for new techniques/frameworks relevant to this stack, proposes adoption | ADR store (`save_adr`/`search_adrs`), architecture docs, `web_search` | Draft ADRs routed through the existing `cto_gate` approval flow |
| `product_strategist` | Mines requirement/BA history and `platform_findings` for customer-facing product ideas | `pipeline_runs.requirement`, BA stage outputs, `platform_findings` | A written proposal report (not code) for human/CTO review |
| `sre_autopilot` | Continuous health loop (extends `healthcheck.py` from one-shot to recurring) | `healthcheck.py` checks, Logfire error rate, broker/DB health | Auto-remediates only a pre-approved, explicit allowlist of reversible actions (e.g. resume an interrupted `pipeline_runs` row, retry one stuck broker task); everything else escalates through `cto_gate`, same as other risky pipeline decisions today |

### Data model additions

- `platform_metrics` (pgvector Postgres, replaces the local JSONL as the source of truth): per-call agent name,
  duration, success, token/cost usage, and the `quality_gate`/`review` verdict when available. `core/telemetry.py`
  keeps writing JSONL for local/offline use; the MCP call path additionally upserts into this table, best-effort,
  matching the existing pattern in `run_store.py` (`enabled()` gate on `DATABASE_URL`, never raises).
- `platform_findings`: structured findings written by `observer`/`sre_autopilot` (kind, severity, agent, summary,
  evidence, status: open/acknowledged/resolved).

### Autonomy / scheduling

This repo has no internal scheduler and stays a stateless MCP server. Recurring execution is driven externally via
OpenClaw's own automation/cron mechanism calling the new MCP tools (`run_observer`, `run_sre_autopilot_check`, etc.)
on a timer — the same way `run_orchestration_pipeline` is already triggered today, just on a schedule instead of
on-demand. No scheduler is added to the Python package.

### Guardrails

- `observer`, `innovation_scout`, `product_strategist` are read-only with respect to the platform: they produce
  findings/drafts/reports, never direct code or config changes.
- `innovation_scout`'s ADRs and `product_strategist`'s proposals require explicit human/CTO approval before any
  follow-on implementation work is created.
- `sre_autopilot`'s remediation allowlist starts intentionally small and is itself tracked in an ADR amendment before
  being extended; actions outside the allowlist escalate via `cto_gate` instead of acting.

## Consequences

- Positive: quality/performance regressions become visible across runs instead of only within a single pipeline
  result; technology and product proposals get a durable, reviewable trail (ADRs, findings); health issues get
  caught and fixed between human check-ins for the narrow cases that are safe to automate.
- Negative: one more database table and two new best-effort write paths to keep consistent with `run_store.py`'s
  "never raises, degrades to no-op without `DATABASE_URL`" contract; four more agents to keep in the
  `tests/unit/test_agent_consistency.py` three-way check (registration, settings model field, tool category).
- Deferred: `sre_autopilot`'s allowlist, and whether `product_strategist` output should also open Jira tickets
  automatically, are explicitly out of scope for this ADR and tracked as follow-up tickets (AP-TBD, filled in once
  created).

## Alternatives considered

- Bolt observation directly onto the existing `quality_gate`/`sre` agents instead of new roles: rejected — conflates
  per-run gating (blocking a specific pipeline) with cross-run trend analysis (a different cadence and audience).
- Let `sre_autopilot` act without an allowlist, trusting the model's judgment: rejected — no case for unreviewed
  autonomous write access to shared infrastructure; matches the "ask before destructive/hard-to-reverse actions"
  policy already in place for `cto_gate`.
