# Observability

How to tell whether the agents succeed, how long they take, what they cost and whether their verdicts make sense.

## Where each signal goes

| Signal | Backend | Enabled by | Holds |
|---|---|---|---|
| Spans (`mcp.*`, `pipeline.*`, `agent.run`, `tool.*`, pydantic-ai GenAI spans with token counts) | Langfuse (OTLP/HTTP to `{LANGFUSE_HOST}/api/public/otel`) | `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (all three) | traces, per-model-call tokens, LiteLLM generations nested under `agent.run` |
| Same spans | Logfire cloud | `LOGFIRE_TOKEN`, or `logfire auth` credentials | traces, queryable with SQL |
| One row per agent run | Postgres table `platform_metrics` | `DATABASE_URL` | agent, duration, success, input/output tokens, verdict, `created_at` |

Both span backends can be on at once. With neither, spans are created but exported nowhere; `configure_observability`
logs a WARNING at startup and the startup health check (`core/healthcheck.py`, used by `sre_autopilot`) reports
`no telemetry backend configured`. `OTEL_SDK_DISABLED=true` turns telemetry off on purpose and is not reported.

## Verifying

1. **The tool.** Call the MCP tool `agent_usage_summary(days=7, agent="")` (read-only, `days` clamped to 1..90). Per
   agent it returns runs, success rate, p50/p95 duration, total input/output tokens, verdict passed/failed counts and
   the most recent failure time. Rows written before tokens were recorded have NULL tokens; the summary says
   "n/a" or "only N of M runs reported tokens" instead of reporting zeros.
2. **The SQL**, in psql against the agents database:

   ```sql
   SELECT agent, count(*) AS runs, count(*) FILTER (WHERE success) AS ok,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY duration) AS p95_s,
          sum(input_tokens) AS in_tok, sum(output_tokens) AS out_tok, max(created_at) AS last_run
   FROM platform_metrics WHERE created_at >= now() - interval '7 days'
   GROUP BY agent ORDER BY runs DESC;
   ```
3. **Langfuse.** Traces list: filter by name `pydantic-agents/<agent>` (set by `agent.run`); sessions group by run id
   (`langfuse.session.id`). Generations from LiteLLM appear under the matching `agent.run` span.
4. **Logfire.** Query `records` for `span_name = 'agent.run'`, or use the Logfire MCP `query_run`. An empty project
   means `LOGFIRE_TOKEN` is not set on the pod (the pod log shows `LogfireNotConfiguredWarning`).

## Limitations

- Only runs that go through `PydanticAIAgentProvider` are recorded in `platform_metrics`. Anything that calls a
  pydantic-ai agent directly (or a tool that runs its own model call) is not counted.
- `cost_usd` is not populated by the provider; cost has to be derived from tokens and the LiteLLM price list, or read
  from LiteLLM/Langfuse.
- Setting the env vars on the deployed pod is an infrastructure change (Helm values), not something this repo does.
