# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
`project.version` in `pyproject.toml`.

Every pull request that changes behaviour, the image, CI, dependencies or
docs adds a bullet under **Unreleased** (CI enforces this; see `AGENTS.md`).
When a version is cut, rename **Unreleased** to the version and date and open a
fresh **Unreleased** section above it.

## [Unreleased]

### Added
- `agent_usage_summary(days=7, agent="")` MCP tool and `platform_metrics.usage_summary`: a read-only per-agent summary
  (runs, success rate, p50/p95 duration, input/output tokens, verdict passed/failed, last failure time) so an
  operator or agent can answer "how often do agents succeed, how long, at what cost" without psql. Decisions: the
  query lives in `platform_metrics.py` and runs through `vector_db.with_connection` with bound parameters only;
  `days` is clamped to 1..90 rather than rejected, so a model's out-of-range guess still gets an answer; the tool is
  `async` (the pool work is bridged to the background loop) so it never blocks the MCP event loop; rows with NULL
  tokens (written before the provider recorded them) are reported as "n/a" / "only N of M runs", never as zeros;
  "no database" and "no rows" return distinct messages. `docs/observability.md` explains backends, env vars and
  verification, and the limit that only runs through `PydanticAIAgentProvider` are recorded.
- `agents/innovation_scout.py` (AP-81, ADR-0001 Platform Intelligence Layer): registered agent with a settings
  field (`innovation_scout_model`), tool category (`"innovation_scout": ["web", "search", "memory",
  "documentation"]`), and `run_innovation_scout` MCP tool. Its own tools search past ADRs (`search_existing_adrs`)
  and persist a draft ADR (`propose_adr`, status: proposed, id prefixed `draft-` so a draft can never overwrite
  an existing ADR) to the ADR store. Read-only with respect to the rest
  of the platform other than that one draft-write path; drafts are never auto-adopted — any follow-on
  implementation work still requires an explicit human/CTO approval through the existing CTO approval gate
  (`core/cto_gate.py`, `approve_cto_gate`), which this agent has no code path to bypass.
- `agents/sre_autopilot.py` (AP-83, ADR-0001 Platform Intelligence Layer): registered agent with a settings field
  (`sre_autopilot_model`), tool category (`"sre_autopilot": ["security", "validation", "memory"]`), and
  `run_sre_autopilot` MCP tool. Extends `core/healthcheck.py`'s checks to a repeatable health-scan path and adds
  `pipeline_jobs.list_interrupted_runs()` (stored in-flight runs last written before this process started, the ones
  `get_run` reports as interrupted, via a new `run_store.list_runs`). `core/sre_remediation.py` (pure, no I/O) classifies each detected issue
  against a literal v1 allowlist of exactly one action — auto-resuming an interrupted pipeline run via the
  existing `resume_pipeline_run` — and escalates everything else (startup-check failures) as an open
  `platform_findings` row instead of acting (once per open issue; a refused resume is reported as escalated). ADR-0001 also named "retry one stuck broker task" as an allowlisted
  action; dropped here because the event broker it targeted was removed from the codebase the same day (#60)
  before this ticket was picked up.
- `agents/qa_executor.py` (`qa_executor` agent, capability `qa_executor`/`test_execution`, `qa_executor_model` setting):
  runs test suites `test_gen` produced via `execute_test_suite` and returns a `TestRunReport` (pass/fail per
  acceptance-criterion id, coverage, unexecuted ids, `all_passed` derived in code),
  without writing or fixing tests itself. Part of the QA Engineer pipeline (KAN-482/KAN-483).
- `agents/observer.py` (AP-80, ADR-0001 Platform Intelligence Layer): registered agent with a settings field
  (`observer_model`), tool category (`"observer": ["memory", "search"]`), and `run_observer` MCP tool. Its own
  tools read `platform_metrics`/open `platform_findings` and deterministically detect regressions, flaky agents,
  and latency drift (`core/observer_analysis.py`, pure functions, no I/O), persisting each as an open
  `platform_findings` row via `save_finding`, skipping a finding whose kind and agent already have an open row so
  scheduled scans do not pile up duplicates. Read-only with respect to the rest of the platform.
- `core/models/core.py`: `Finding` (typed, versioned, requires a non-blank `file:line` or resource id `location`; severity is
  case-insensitive so a lowercase "critical" still reaches the gate),
  `TestRunReport`/`CriterionResult` (versioned test-execution report keyed by acceptance-criterion id, with
  `all_passed` derived from the results and any unexecuted ids), and `AgentAdvice.findings` — `AgentAdvice` now
  forces `approved_to_proceed=False` in code while any CRITICAL finding is open, closing the gap where
  `ArchitectureReviewCrew`'s CTO gate relied on the prompt alone (KAN-485, part of epic KAN-482).
- `AuditCrew`'s consolidation task (`core/pipeline/crewai/crews.py`) now outputs a JSON array of `Finding`
  objects instead of free-text markdown, and the bridge's new `_apply_findings` (`core/pipeline/crewai/bridge.py`)
  enforces the ADR rule in code: findings missing a `file:line`/resource id are dropped, and the whole batch is
  rejected (fail-closed, like the existing review/advice gates) if any other finding is invalid or a surviving
  finding is missing an owner or due date (KAN-491, part of epic KAN-482).
- `agents/product_strategist.py` (AP-82, ADR-0001 Platform Intelligence Layer): registered agent with a
  settings field (`product_strategist_model`), tool category (`"product_strategist": ["memory"]` —
  deliberately narrower than the ADR's literal `web, search, memory, documentation`: `documentation` grants
  `update_documentation`, a real filesystem write, and `a2a` would let it delegate to agents that write, either
  of which would contradict AP-82's "no direct writes to the repo" acceptance criterion), and `run_product_strategist` MCP tool. Fully read-only — its two tools (`recent_requirements`, backed
  by a new `run_store.recent_requirements()` read query over `pipeline_runs`, and `open_findings`, backed by the
  existing `platform_findings.list_findings()`) are its only inputs, and it has no write tool at all. Output is a
  `ProductStrategistReport` (proposals + narrative) — a written report for human/CTO review, never code or a
  ticket.
- `core/platform_metrics.py` (`platform_metrics` table) and `core/platform_findings.py` (`platform_findings` table,
  `save_finding`/`list_findings`): best-effort pgvector persistence for ADR-0001's Platform Intelligence Layer
  (AP-77), following `run_store.py`'s pattern (`enabled()` gated on `DATABASE_URL`, never raises). The MCP call path
  (`run_agent_by_name`) now also upserts a `platform_metrics` row per call, alongside the existing `telemetry.py`
  JSONL write, capturing a `passed`/`failed` verdict when the agent's output is a `QualityGateResult`/`ReviewResult`.
- `QaCrew` now runs qa_executor/go_test in a bounded fix loop (`settings.qa_max_fix_rounds`, default 2) instead
  of a single unconditional fix step: `qa_executor` reports a `TestRunReport` after every round, `go_test` is
  retried only on the failing/unexecuted criterion ids, and review's sign-off runs regardless but `state.approved`
  is forced false in code if the final report isn't `all_passed` with coverage at or above
  `settings.qa_min_coverage_percent` (default 80%), closing the gap where the retry was unbounded and ungated
  (KAN-487, part of epic KAN-482). `PipelineState.test_run_report` is the last parsed report, or None when
  `qa_executor`'s output never parsed; the bridge picks the staged runner by crew class, not by name.
- `AgentAdvice.adr_id`: approval also now forces `approved_to_proceed=False` in code while `adr_id` is empty, so
  the CTO gate can't approve a decision with no ADR to point at. `approve_cto_gate` (MCP tool) enforces the same
  rule for the human-approval path (`approved=true` requires `adr_id`), rejects `adr_content` given without an
  `adr_id` to save it under, and when `adr_content` is given, saves it under `adr_id` via `save_adr` in the same
  call instead of needing a second tool call — a `save_adr` failure is caught and returned as an error instead
  of silently leaving the approval marker unwritten with no explanation (KAN-489, epic KAN-482).
- `registry.run(name, prompt, deps=None, timeout_seconds=None, usage=None)`: the one guarded way to run an agent
  (tools attached, deps resolved, permit, timeout, transport retry, usage caps, `max_tokens`, Langfuse session tags,
  an `agent.run` span). `run_agent_by_name`, `modify_existing_code` and `plan_and_execute` all go through it; they
  used to call `Agent.run` directly and skipped every guard.
- Langfuse OTLP export (`core/observability.py`): with `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` set, the whole in-process span tree is exported to `{host}/api/public/otel`, tagged with
  `langfuse.session.id` = the run id; the httpx instrumentation's `traceparent` lets LiteLLM's generations attach
  under the agent span. Before this the agents pod (no Logfire token) dropped every in-process span.
- Every MCP agent call gets its own run id (`call_<hex>`), so its model calls are one Langfuse session; the
  `run_<agent>` tools take an optional `project` argument that scopes context retrieval to one seeded project.
- `core/verifier.py`: deterministic build, vet and test with the real Go toolchain (`verify_artifacts` for
  generated services, `verify_tree` for an existing codebase with the changes overlaid on a copy). The quality
  gate stage runs it before any model judges the code and feeds its report to the security, quality-gate and
  review agents; `modify_existing_code` does the same and retries on a failed build.
- Per-attempt checkpoints: `OrgitrationPipeline` saves `implementation.attempt<n>` after every implementation
  attempt and its gate, and a resume continues after the last saved attempt. The pipeline run id is the
  checkpoint id (`PipelineConfig.run_id`), so a resume continues its own run, not the project's newest file.
  Checkpoint files are named `{run_id}__{stage}.json`; files in the earlier `{uuid}_{stage}.json` form are
  renamed to the new form when the repository opens, so runs checkpointed by an older version still resume.
- `run_orchestration_pipeline(project_name, requirement, blueprint="new_feature", framework="")` runs any
  blueprint (LangGraph, CrewAI, AutoGen), and `project_name` is validated as a plain name at the MCP boundary
  and in `PipelineConfig` (it becomes a directory under the output, checkpoint and approval roots).
- Run status `waiting_cto_approval` while the CTO gate polls for a decision (`pipeline_jobs.set_run_status`).
- `ReviewResult` withdraws an approval whose `risk_level` is HIGH or CRITICAL or whose `security_score` is below
  70, and says so in `feedback`.
- Five agents: `critic` (checks an agent's claims against the evidence of its run; `CriticReport` with
  `unsupported_count`/`passed` derived in code), `api_contract` (OpenAPI/proto vs handlers and frontend types;
  `ContractReport`), `observability_engineer` (spans, RED metrics, slog, probes; emits alert rules and a review),
  `finops` (LiteLLM spend per model and key via the new `litellm_spend_report` tool; `CostReport` with the total
  computed in code) and `release_manager` (changelog fragment, version note, PR body with the `agent-ws` marker and
  the attribution line). Settings fields `critic_model`, `api_contract_model`, `observability_engineer_model`,
  `finops_model`, `release_manager_model`.
- Three CrewAI crews with blueprints: `crew_qa` (BA criteria → Go/Playwright tests → fixes → sign-off, keyed by
  criterion id), `crew_architecture_review` (BA, security, database and SRE findings merged by the CTO into an
  `AgentAdvice`, blocked while a CRITICAL finding is open) and `crew_audit` (secrets/OWASP, CI and container
  hardening, runtime findings consolidated into an owned, dated table).
- Configurable embedding provider for the vector store (`EMBEDDING_PROVIDER=local|litellm`, `EMBEDDING_MODEL`,
  `EMBEDDING_DIM`): `litellm` embeds through the proxy's `/embeddings` endpoint in one batched call. Switching the
  provider, model or dimension changes the vector space and requires re-seeding every collection; a dimension
  mismatch against an existing table raises `ConfigurationError` instead of inserting incompatible vectors.
- Hybrid retrieval (`RETRIEVAL_HYBRID`, default on): pgvector cosine candidates and Postgres full-text candidates
  (generated `content_tsv` column plus GIN index) fused with reciprocal rank fusion; optional `RERANK_MODEL`
  reorders them through the proxy's `/rerank` endpoint and falls back to the fused order on any error.
- Promptfoo cases for the pipeline agents `ba`, `cto`, `quality_gate`, `backend`, `service_engineer` and
  `critic` (none of the core agents had an eval); not yet run live.
- `tests/unit/test_review_fixes.py`: regression tests for every finding below (concurrent recorder spans, the
  guarded run path, shared delegation budgets, the review verdict, context scoping, the LangGraph thread id, plan
  data flow, project-name validation, Langfuse export, the verifier).
- Full-repository review remediation (2026-10-30). `MERGE_INTO_SERVER_TREE` setting (default false) and
  `core/pipeline/crewai/approval.py`, `utils.rotate_log` / `utils.kill_and_reap`, `limits.run_blocking_with_permit`,
  `context_engine.astore_artifact_contexts`, `vector_db.async_search_domains` and `platform_metrics.verdict_of` /
  `token_counts` as the shared pieces the fixes below use. `pipeline_runs` gains `blueprint`, `framework` and
  `resume_attempts` columns plus an index (added with `ALTER TABLE ... IF NOT EXISTS`, so existing tables upgrade on
  first use), and `platform_metrics` gains indexes on `(agent, created_at)` and `created_at`.

### Changed
- Missing telemetry configuration is now loud. `configure_observability` logs at WARNING (was INFO) when neither
  Langfuse keys nor a Logfire token is set, and `healthcheck.validate_telemetry` (part of `run_startup_checks`, so
  also `sre_autopilot`) reports "no telemetry backend configured". Why: the deployed pod ran with neither, the
  INFO line was never seen, and the Logfire project stayed empty. An explicit `OTEL_SDK_DISABLED=true` is treated as
  a decision and not reported, to avoid `sre_autopilot` escalating an intentional setting.
- `ba` and `review` prompts recalibrated after a live test of the deployed service (calibration PR, eval not yet run).
  Why: one request (a Go `GET /healthz`) produced six stories and six events (including `health.ReadinessCheckFailed`,
  which the agent itself called an anti-pattern), and a 9-line handler was rejected as HIGH risk with score 28.
  `ba`: asks for the fewest stories that cover the request (default one, cap two for a single capability, about five
  for several; failure paths are Gherkin scenarios, not stories), makes events optional (an empty `event_triggers`
  is valid for probes, queries and UI changes, and failures and probes are not domain events), and tells the agent
  that tools, workspace and retrieved context describe the platform rather than the target, so it must not block or
  mark a story provisional because the target service is not visible. `UserStory` gains `open_questions` and
  `assumptions` (default empty, so stored stories and every consumer keep working) so the agent no longer refers to
  flags that exist nowhere in its output. Decision: fields on the story rather than a wrapper output type, because
  `BAStage`, the crew bridge and the MCP tool all consume a plain `list[UserStory]`.
  `review`: reviews only the submitted artifact, runs only the checks for its kind and lists skipped checks in the new
  `ReviewResult.not_applicable` (default empty) instead of failing them, no longer rejects for provenance or a missing
  workspace, takes the stated purpose (readiness versus liveness) from the request, and states that net/http recovers
  handler panics. Every finding needs a concrete failure scenario, and severity, `risk_level` and `security_score`
  have defined bands so three minor findings on a short handler land in the 70s or 80s, not HIGH/28. Decision:
  `ReviewCriteria` defaults to no blueprint and non-strict, so a direct review makes no gRPC/CloudEvents/hexagonal
  demands; the pipeline's `QualityGateStage` now passes `REVIEW_BLUEPRINT_SUMMARY` and `strict=True` explicitly, so
  pipeline reviews keep their previous conformance behaviour. The approval rule (risk below HIGH, score at least 70)
  and the `approved`/`feedback` fields used by `quality_gate_stage`, the crew bridge and verdict extraction are
  unchanged. Evidence so far: unit tests only; the new promptfoo cases (`ba` and `review` health-endpoint cases)
  are written but unexecuted, so whether the prompts behave better is unverified until `make eval` runs.
- `context_engine.retrieve_context` is `async`, takes `project`, caps the query at 1,000 characters (the embedding
  model reads about 256 word pieces) and never returns the store's own unavailability notice as a snippet. The
  stages query with the requirement and story titles, not the artifact dump, and pass `config.project`.
- The quality gate stage reviews artifacts in batches that fit the 80k-character prompt budget instead of
  truncating the dump and judging what was left; the SRE agent receives the requirement, stories, services and
  CTO advice (it used to get a fixed one-line prompt); structural validation and verification run first.
- `OrgitrationPipeline.execute` returns the final `PipelineState` and stops after BA when there are no stories
  and after advisory when the CTO did not approve (`failure_report` says why). `pipeline_jobs` derives the status
  from that state: `completed` only when approved, `failed` with the reason otherwise. It used to report
  `completed` whenever `execute()` returned.
- `PipelineState.cto_approved` defaults to `False`, and the LangGraph `route_after_advisory` also stops on
  `errors`, so a crashed advisory stage no longer proceeds to implementation.
- `_GraphPipelineWrapper.execute` passes `configurable.thread_id` (the run id); a graph compiled with a
  checkpointer (`resume=True`) used to raise.
- The implementation stage runs each story's tests against every source file of the story (a suite that touches
  two files compiles now), treats a suite that could not run (toolchain, module setup, timeout) as a failure
  instead of a pass, re-runs the frontend and database agents on a retry after a gate rejection (only the backend
  after a test failure), and orders every prompt with the run-wide blocks first (repository context, local
  guidelines, CTO advice) and the story last so a provider-side prefix cache can reuse them.
- `execute_test_suite` is built on `run_test_suite(files, test_files) -> TestRunResult` (`passed`, `output`,
  `error`); the tool keeps its string contract.
- `SessionRecorder.record_span` yields its own record (`RecordingSpan`) instead of sharing one `_current` slot:
  concurrent spans (advisory, per-story implementation) used to overwrite each other and lose records.
- `delegate_to_agent` passes the caller's `usage` through `instrumented_query` and `registry.query`, so a
  delegation tree spends one run's caps instead of a fresh 40 requests and 2M tokens per node.
- `review`, `security` and `quality_gate` lose the `a2a` tool category: a verifier cannot delegate to the agent
  it judges.
- `plan_and_execute` appends the results of a task's dependencies to its prompt and skips a task whose
  dependency failed; the planner runs through `registry.run`.
- The sandbox root for model-driven file tools is a context variable (`core/utils.py::sandbox_root_var`) set by
  `run_existing_code_pipeline` for its own task; the process-wide `ORCHESTRATION_CODEBASE_ROOT` is only the
  server default. Two concurrent pipelines on different codebases no longer share or widen each other's root.
- `modify_existing_code` writes the changed files and tests under `generated-app/<codebase>/<run>/` and reports
  the path and the verification result; the files used to be discarded after the summary. An approved review's
  remarks are no longer appended to the errors the next attempt is told to fix.
- `_sync_memory_after_pipeline` only runs with `MEMORY_BRIDGE_SYNC=true` (default off): it wrote activity logs
  into every Claude Code agent's memory directory on the workstation after each run.
- `core/limits.py::run_model_settings` sends `max_tokens` (`AGENT_MAX_OUTPUT_TOKENS`, 16384) with every call and
  the Langfuse session tags inside a run; `factory._make_chat_model` gives the OpenAI client a per-request httpx
  timeout (`AGENT_HTTP_TIMEOUT_SECONDS`, 300 s, connect 10 s).
- The JSONL telemetry line is written once per call, by the provider; `instrumented_query` and
  `run_agent_by_name` no longer write a second one (`live_trace` double-counted every pipeline call).
- `PipelineConfig` adopts the blueprint's framework when none is given (a `crewai` blueprint used to build the
  legacy pipeline). Checkpoint files are named `<run_id>__<stage>.json` (run ids may contain `_`); the old
  single-underscore UUID files are still read.
- CrewAI bridge: pydantic_ai tools are actually bridged (the toolset dict was iterated by key), `RunContext` tools
  are skipped, async tools run correctly, and the CrewAI LLM is built as `openai/<model>` against the LiteLLM proxy
  instead of a bare model name the litellm SDK rejects. Every crew takes the `PipelineState` (`ResearchCrew` and
  `SecurityReviewCrew` expected strings), hierarchical crews set a `manager_llm`, and the review gate parses a
  `ReviewResult` JSON verdict instead of substring-matching "approved" (which approved "not approved").
  A crew's artifact tasks merge into one artifact per owner (a later task's files replace the same paths), and
  bridged tool calls share one bounded thread pool instead of starting a pool per call.
- Local sentence-transformers embedding runs in a worker thread in the async paths; `vector_db.search` returns
  `[]` (with a logged warning) when the vector extras are missing, instead of an explanatory string that callers
  pasted into prompts.
- `.env.example`, README and `.ai/guidance/architecture.md` describe the Langfuse export, the run guards, the
  verifier, the per-attempt checkpoints, the honest run status and the new settings.
- **Blueprints now run only their own stages (review finding 3).** `build_default_pipeline` registered all four
  stages whatever the blueprint said, so `quick_backend`, `docs_update` and `security_review` ran BA and the CTO gate
  too. Decision: honour `stages` in the legacy pipeline; a blueprint without BA implements the requirement as one
  story (`US-001`) because the implementation stage cannot run on zero stories; a gate-only blueprint now really runs
  its gate (it was skipped as "handled in the retry loop"). Not changed: the LangGraph graph is still fixed, and
  an unknown or stage-less blueprint now raises instead of registering every stage (fail closed: a typo would have
  run the full SDLC, CTO gate included). `composite` is still the same graph as `langgraph` (no CrewAI implementation step exists to wire in), which is a
  known gap and not a fix.
- **A CTO decision marker answers one gate (finding 1).** `.cto_approved` is deleted once the gate reads it. Decision:
  consume rather than scope by run id, because `approve_cto_gate` is keyed by project name and an operator who
  `touch`es the file has no run id to give; consuming keeps pre-approval working for exactly one run. A rejection
  likewise stops one run, not the project forever. `ORCHESTRATION_CTO_OVERRIDE` now needs a true value
  (`1/true/yes/on`): `false` and `0` used to bypass the gate.
- **The three quality gates run side by side (performance).** Security, quality-gate and review are independent, so
  their latency is no longer the sum of the three; the first rejection in the old order is reported. Trade-off
  accepted: a rejection no longer saves the calls of the gates after it. Retrieval is also done once per gate and
  run instead of once per batch.
- **Context retrieval embeds the request once and searches its domains concurrently, then takes the best of each
  domain round-robin.** Concatenating the domains let the first one fill `top_k` and discarded the rest after paying
  for them. The never-seeded `dependencies` domain is dropped from `AGENT_CONTEXT_DOMAINS`. Retrieval failures log
  at WARNING (they were DEBUG, so a dimension mismatch looked like "no context").
- **Lexical search ORs the query words** (`to_tsquery`) instead of `plainto_tsquery`, which ANDs every word and so
  matched nothing for request-length queries. New tables get an HNSW index instead of ivfflat built on an empty table
  (no training data, poor recall until rebuilt); the old index is dropped when HNSW is created, and pgvector without
  HNSW falls back to ivfflat. Documents are inserted in one transaction with `executemany`.
- **`platform_metrics` rows are written by the agent provider, once per agent run, with tokens and verdict.** They
  were written only for `run_<agent>` MCP calls and never carried token counts, so the `observer` agent saw a
  fraction of the traffic. Consequence: delegated and pipeline runs now add rows, so absolute counts rise.
- **Blocking work is off the event loop.** `search_adrs`, `save_adr`, `query_context`, `seed_project_context`,
  `sync_memory` and `memory_bridge_status` stay plain functions but are registered as async tools that run in a
  worker thread (FastMCP runs sync tools inline, stalling every request and background run); `approve_cto_gate`,
  the merge stage (file writes and a single batched index call) and URL/PDF ingestion do the same.
- **Merge stage indexes with stable ids** (`merge:<project>:<path>`, body capped at 4,000 characters) so a file
  merged again replaces its row instead of adding a duplicate each run, and it refuses to merge into the server's own
  working directory unless `MERGE_INTO_SERVER_TREE=true` (the `default` project's target root is `.`).
- **Crew and group-chat runs report a real status.** `ResearchCrew`, `AuditCrew`, `ArchitectureReviewCrew` and the
  AutoGen chats never set `approved`, so a finished run was reported as "failed: finished without approval". They are
  now approved when they end without errors (the architecture review also needs the CTO's approval); crews with a
  verdict or test gate still take it from that.
- **A timed-out CrewAI/AutoGen thread keeps its LLM permit until it ends** (`run_blocking_with_permit`): a thread
  cannot be stopped, so releasing the permit at the timeout let the concurrency limit be exceeded.
- **Logs are pruned.** Rotated `telemetry.jsonl` and `session_trace.jsonl` archives were never deleted; the newest 5
  are kept.
- `fetch_url` reads at most 2 MB of a response instead of loading the whole body before truncating it. Go test runs
  pass `-count=1`, and the vitest path is skipped (it has no `package.json`, so it always failed the story).
- The `pipeline.background_run` span and `delegate_to_orchestration_agent` calls now carry the run id as the Langfuse
  session; CrewAI and AutoGen runs use the run's own id instead of a fresh `run-xxxx`.

### Fixed
- `ArtifactWriter.write` (`core/writer.py`) now validates `artifact.agent_name` through `safe_path_join` and rejects
  multi-segment or empty names, preventing a model-chosen `agent_name` (`../...` or `/abs/...`) from writing outside
  `base_dir`.
- `vector_db` validates `collection_name` as a plain ASCII SQL identifier (`[A-Za-z_]\w{0,60}`) before interpolating
  it into `v_<collection_name>` DDL/DML queries, and `mcp_server.query_context` restricts `domain` to the known
  knowledge domains (`adrs`, `architecture`, `conventions`, `project`), closing a SQL injection vector.
- Build, vet, lint and test subprocesses (`core/tools/test_tools.py`, `core/verifier.py`, `core/tools/go_common.py`,
  `core/tools/gobuild.py`, `core/tools/frontend_tools.py`) now run with `sanitized_subprocess_env()`, stripping API
  keys, `DATABASE_URL`, and Go/Node/XDG redirect variables (`GOPRIVATE`, `GONOPROXY`, `GONOSUMDB`, `GOFLAGS`,
  `NODE_PATH`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`), forcing `GOENV=off`, and pinning `GOPROXY`/`GOSUMDB` from
  `UNTRUSTED_GOPROXY`/`UNTRUSTED_GOSUMDB` so generated code/tests cannot exfiltrate credentials or reach internal
  module proxies.
- `mcp_server` populates `TransportSecuritySettings.allowed_hosts` and `allowed_origins` from
  `Settings.mcp_allowed_hosts` / `Settings.mcp_allowed_origins` (defaulting to `127.0.0.1:*`, `localhost:*`,
  `[::1]:*`) instead of leaving them empty and rejecting every HTTP/SSE request with HTTP 421.
- `product_strategist` is registered under the `product_strategy` capability instead of colliding with
  `innovation_scout` on `platform_intelligence`, and `test_agent_consistency.py` asserts every capability has a
  single owner.
- AutoGen adapter (`core/adapters/autogen_bridge.py`, `core/pipeline/autogen/groups.py`,
  `core/pipeline/autogen/bridge.py`): `_build_llm_config` and `_get_manager_llm_config` now unwrap `LazyModel` and
  resolve the LiteLLM proxy `base_url`/`api_key` via `_litellm_connection`, `_register_tool_functions` iterates
  `_function_toolset.tools.values()`, skips `RunContext` tools and bridges async tools through `_call_tool_function`,
  and both `run_autogen_pipeline` and `run_crewai_pipeline` acquire `llm_semaphore()` and enforce
  `settings.agent_timeout_seconds`.
- `fetch_url` (`core/tools/web_fetch.py`), `read_and_memorize_url` (`core/tools/memory.py`) and
  `vector_db.ingest_url`/`async_ingest_url` now validate URLs with `validate_public_http_url` (blocking loopback,
  RFC1918, link-local/cloud metadata, and internal hostnames on both the initial request and any redirect hop),
  enforce timeouts, and cap response size.
- `scan_for_secrets` (`core/tools/secrets.py`) redacts matched credentials in `snippet` (`[REDACTED]`) and catches
  `sk-proj-`/`sk-ant-` keys, spaces around `=`/`:`, unquoted `.env` keys, and dotted/hyphenated database hosts.
- `mcp_server.modify_existing_code` validates `codebase_path` against `_check_codebase_root` (`SEED_ROOTS`,
  `output_dir`, or `ORCHESTRATION_CODEBASE_ROOT`), and `scaffold_and_build_service` validates `service_name` and
  constructs the pipeline via `build_pipeline(config)`.
- `Settings` declares `litellm_base_url`, `litellm_virtual_key`, `litellm_master_key`, `litellm_env` and
  `litellm_cloud_url` so `LITELLM_*` entries in `.env` are honoured by `factory._litellm_connection`.
- Tool grants and sandboxing (`core/tool_assignments.py`, `core/tools/memory.py`, `core/tools/docs.py`):
  `read_and_memorize_url` and `read_and_memorize_pdf` are no longer attached to the `memory` category, `review` and
  `cto` no longer receive `documentation` (`update_documentation`), the nine personal utility agents no longer
  receive local codebase `search`, and `update_documentation`/`read_and_memorize_pdf` resolve paths against
  `sandbox_root()`.
- `core/pydantic_ai_compat.py` centralizes access to `pydantic_ai`'s private tool storage (`_function_toolset.tools`,
  `Tool.function`) for `autogen_bridge`, `crewai_bridge`, `tools.registry.apply_tools`, and `BridgeAgent`, and runs a
  startup compatibility probe (`check_pydantic_ai_compatibility`) from `register_all_agents()` that raises
  `PydanticAICompatibilityError` instead of silently dropping tools if a future `pydantic_ai` upgrade moves those
  attributes.
- `agents/registration.py` provides default `deps_factory` callables for `cto` (`default_org_context`, backed by
  `ORG_*` settings), `backend` (`default_project_state`, rooted at `sandbox_root()`), `review`, `database`,
  `devops` and `security`, `cto_agent` has `search_existing_adrs`, and `qa_executor`'s prompt no longer attributes
  source/test fixes to `go_test`.
- `.github/workflows/changelog.yml` passes `github.base_ref` through an environment variable instead of inline shell
  interpolation, and both `ci.yml` and `changelog.yml` declare `permissions: contents: read`.
- `vector_db.ingest_url`, `ingest_pdf`, `async_ingest_url` and `async_ingest_pdf` compute deterministic SHA-256 document
  IDs instead of using `id(s)` memory addresses.
- `MergeStage._merge_file` resolves `target_root` before calling `dest.relative_to(...)`, and `sandbox_root()` returns
  `root.resolve()` when `sandbox_root_var` is set.
- `memory_bridge` honours `all_runtimes` and `max_entries` across both sync directions, skips symlinks in
  `_memory_files`, and handles vector DB errors in `get_bridge_status`.
- `healthcheck.validate_api_keys` checks `settings.litellm_virtual_key`/`litellm_master_key` and `deepseek_api_key`,
  `Settings.resolve_model` maps `deepseek` through `_NATIVE_MODEL_PREFIXES` and handles `provider="test"`, and
  `PydanticAIAgentProvider.run_agent` preserves an agent's own `_model_settings["max_tokens"]`.
- `tests/unit/test_web_fetch.py`, `tests/unit/test_secrets_scan.py`, `tests/unit/test_vector_db_collection_names.py`,
  and `tests/unit/test_review_2026_10_29.py`: dedicated unit tests for SSRF validation/redirects/size caps
  (`web_fetch.py` branch coverage 56% → 94%), secret scanning and redaction, pgvector SQL identifier validation, and
  the review's Medium/Low and CTO review fixes.
- `LocalDiskStateRepository.clear` runs `_ensure_migrated()` and matches `f"{run_id}__"` so `clear("run1")` does not
  delete `run10` checkpoints, `run_existing_code_pipeline` closes its `recorder` in `finally:`, and
  `mcp_server.cancel_pipeline_run` falls back to `get_run(run_id)`.
- `AgentRunRecord.to_dict` slices `d["prompt"]` to 100,000 characters when `prompt_truncated` is `True`, and
  `observability._instrument` calls `logfire.instrument_pydantic_ai(include_content=False)`.
- `BridgeAgent` loads hooks registered via `agy_bridge.hooks` (`get_hooks(name)`), and `agy_bridge.server._respond_to`
  rejects non-dict JSON payloads cleanly instead of raising `AttributeError`.
- `scripts/check_changelog.py` and `scripts/seed_projects.py` enforce subprocess timeouts and validate org/repo names
  with `safe_path_join`.
- `TestRunResult` sets `__test__ = False`, `test_langgraph_pipeline.py` uses `MagicMock` for `record_span.__aenter__`
  (clearing 6 pytest warnings), and `test_state_repository_contract.py` runs `LocalDiskStateRepository` contract tests
  even when the optional `turso` extra is not installed.
- `PydanticAIAgentProvider.run_agent` reads the agent's public `model_settings` (pydantic-ai has no `_model_settings`),
  so `database_agent`'s 8192 `max_tokens` cap now actually survives the per-call 16384 default; the CTO's
  `search_existing_adrs` is best-effort and reports an unreachable ADR store instead of aborting the review.
- `tests/unit/test_review_2026_10_29.py`: regression tests for the review's Medium/Low fixes plus the SSRF guard's DNS,
  redirect, size-cap and error paths (`web_fetch.py` branch coverage 56% → 94%).
- Promptfoo: `qa_executor` provider with two cases (per-criterion results and the derived verdict from supplied
  output; no test rewrite when asked for one) and two more `cto` cases (a sound design inside the approved stack is
  approved; a component outside the approved stack is flagged).
- `run_store`, `platform_metrics` and `platform_findings` now reach the database through
  `vector_db.with_connection`, which runs the query on the background loop that owns the cached asyncpg pool.
  They used to acquire that pool from the caller's loop, so once a vector search had created it there, every
  pipeline-run save and every `platform_metrics` row failed with a cross-loop error that the best-effort
  `except` swallowed.

- Checkpoint files written by a version before the `{run_id}__{stage}.json` name (`{uuid}_{stage}.json`) are
  renamed to the current name before the first read of a project's repository, so runs checkpointed by that
  version resume instead of restarting. The rename runs before the first read, is retried on a later read if it
  failed, and tolerates a second instance renaming the same file first. Files that are neither form are listed in
  one warning and ignored.
- A checkpoint file that is valid JSON but not a `PipelineState` is treated as corrupt (logged, `load` returns
  `None`) instead of raising a validation error out of the resume path.
- **A run can no longer finish approved with no code (finding 2).** The quality gate rejects an empty artifact list
  (the SRE agent's own output used to stand in for the code); the implementation stage reports a
  `runtime_error` failure when every generation agent failed and the pipeline stops instead of retrying against no
  feedback; a test agent that crashed, a story with no test files, or one with no source files now fails the story
  instead of being skipped and logged as "All tests passed". The retry loop's failure report carries the tail of
  each failed story's test output, so the next attempt sees the assertions and not just a story id.
- **Resume after a restart rebuilds the same pipeline (finding 4).** `blueprint` and `framework` are persisted (a
  resumed `crew_audit` run restarted as `new_feature`). The task handle is stored before the first `await`, so two
  concurrent resumes start one task instead of two with one uncancellable. `sre_autopilot` resumes a run at most
  `MAX_AUTO_RESUMES` (3) times, then escalates it as a finding: a run that kills the server (out of memory) was
  resumed on every restart. Finished runs beyond 200 are evicted from `PIPELINE_RUNS`. Not fixed: checkpoints are
  still on local disk by default, so resume after a pod restart needs `CHECKPOINT_DIR` on a persistent volume.
- `pipeline.factory._ensure_dependency` looked for the module `pyautogen`, which is imported as `autogen`, so every
  AutoGen blueprint failed with a missing-package error.
- **Tracing records failures as failures.** `record_span` wrote `success=True` for stages that caught their own
  exception and for cancelled runs (`CancelledError` is not an `Exception`). Stages now call `abort_run`, the span
  catches `BaseException`, and a record is written once (the first finish wins).
- Build and test subprocesses start in their own session and are killed as a group and reaped on timeout; killing
  only `go` left the test binary running and the process as a zombie.
- `modify_existing_code`: a retry whose engineer returns no files no longer counts as a pass over the failures it was
  asked to fix.
- Sonar cognitive-complexity findings on this PR: `_run_qa_staged_gate` is split into `_run_qa_tests` and
  `_qa_execute_and_fix`, `query_context` takes its per-domain work from `_query_domain`, and an exception test with
  two calls in one `pytest.raises` builds its argument outside the block; a second round of the same kind:
  `fetch_url` (now `_get_following_redirects` and `_render_body`), `ImplementationStage._run_test_phase` (now
  `_test_story`) and one more exception test. Behaviour is unchanged.
- `.claw-checkpoints/` is git-ignored: the test suite writes run state there and an earlier commit of this change
  picked it up by mistake (removed).
- Not addressed, by decision: LLM-generated Go still runs unsandboxed on the host (needs a sandbox such as gVisor, not
  a code tweak); `fetch_url` validates DNS before the client resolves it again (rebinding window); an empty `project`
  still retrieves context from every seeded project (it is the documented behaviour); the LiteLLM HTTP client is still
  created per embedding call (its tests patch the class, and a shared client needs lifecycle handling).

### Removed
- The event broker (`core/event_broker.py`) and the MCP tools `submit_async_task`, `check_task_status`,
  `wait_for_task` and `task_context`: tasks were submitted but nothing consumed them, so `wait_for_task` always
  timed out.
- `RedisIrisMemory` and the `REDIS_IRIS_*` settings: both methods were stubs.

### Changed (earlier)
- CI installs with `uv sync --frozen --no-build --no-install-project` and runs tools with `uv run --no-build --no-sync` (Sonar
  S8541): only prebuilt wheels are installed, so no dependency build script runs, and the project is used from `src`
  (`PYTHONPATH=src` for the test and smoke jobs). The image's `useradd` shares a `RUN` with the project install (S7031).
- `check_changelog.py` uses `\w` with `re.ASCII`, and the heading regex in `ai_context.py` no longer has overlapping
  quantifiers (S6353, S8786); `cto_gate.read_marker` logs the length of an unrecognised marker, not its content (S5145).
- Cognitive complexity reduced (Sonar S3776) in `live_trace.show_status`, `Itinerary._consistent_plan` and
  `mcp_server._chunk_text` by extracting helpers; no behaviour change. `show_status`'s output and the chunk edge cases are
  now pinned by tests.
- Cognitive complexity reduced (Sonar S3776) in the four largest functions, by splitting them into small helpers:
  `ImplementationStage.run` (73), `run_existing_code_pipeline` (52), `parse_code_blocks_to_files` (39) and
  `OrgitrationPipeline.execute` (37). Each now has tests that were written first and run against the old code
  (none had any for its main path). Prompts, recorded events and results are unchanged.
- `OrgitrationPipeline` runs a stage normally when a checkpoint is listed but cannot be loaded, instead of carrying `None` forward as its state.
- Cognitive complexity reduced (Sonar S3776) in `WorkspaceContext.get_repo_context` and `get_markdown_context`, the
  Go and Vue/TSX validators, the AutoGen bridge's `_apply_group_result` and `MergeStage.run`, by extracting helpers. New
  tests (written first and run against the old code) pin each one's output and messages.
- Cognitive complexity reduced (Sonar S3776) in ten functions by extracting helpers, with no behaviour change: BridgeAgent
  `get_system_prompt` and `_wrap_tools`, `run_server_loop`, `retrieve_context`, `cto_gate.apply`,
  `sync_memory_to_claude`, `sync_claude_to_pgvector`, `build_full_graph`, `format_result_summary` and `format_artifacts`.
  New tests pin the behaviour of the ones that had none (tool hooks, system-prompt lookup, the server loop, the memory
  bridge, the summary format, the graph's nodes); they also pass against the pre-refactor code.
- **CI:** every Python job installs from the committed `uv.lock` with `uv run --frozen` (via `astral-sh/setup-uv`)
  instead of an unpinned `pip install`, so CI runs the same versions as `make` and a new upstream release cannot change
  a result. The lint job now checks `scripts` as the Makefile does, and uses the locked ruff instead of the latest one.
  Third-party actions (`docker/*`, `astral-sh/setup-uv`) are pinned to a commit SHA (Sonar S7637, S8541, S8544).
- **Renamed parameters:** `AgentRegistry.query`, `instrumented_query`, `cto_gate.apply` and
  `cto_gate._poll_for_file_approval` take `timeout_seconds` instead of `timeout` (Sonar S7483); callers inside the repo
  are updated. `tools.registry.apply_tools` no longer takes the unused `agent_name`.
- Sonar code smells cleared with no behaviour change: failures in pipeline stages and bridges are logged with
  `logger.exception` (traceback included), chained `startswith`/`endswith` use a tuple, no-op `except: raise` clauses
  and a redundant `PermissionError` handler are gone, duplicated literals are constants, a heading regex that could
  backtrack in `.github/scripts/ai_context.py` is replaced by a plain match plus a strip, and `_write_files` and
  `discover_agent_capabilities` are plain functions because they never awaited. `RedisIrisMemory` and
  `sync_agent_definitions` now do their blocking work in a thread; `RedisIrisMemory.add_memory` logs that it dropped the
  memory instead of silently doing nothing.
- Two promptfoo cases were recalibrated after the first live run: the "clean code" security snippet is now a
  self-contained function (the previous payment function legitimately drew a missing-authorization finding), and the
  `market_shark` rubric allows arithmetic on the supplied figures and labelled scenario assumptions while still failing
  on invented facts or outside targets.
- Docs brought in line with the code: README and `.ai/guidance/architecture.md` describe the real model
  defaults, run limits, `AGENT_TRANSPORT_ATTEMPTS`, durable pipeline runs, `resume_pipeline_run`, the CTO gate
  and `approve_cto_gate`; the "known improvement areas" list marks the fixed items; the settings docstring and
  the promptfoo grader name `deepseek-v4-pro`; `.env.example` lists the run-limit and model overrides.
- **Behaviour change:** `crypto_oracle`, `market_shark` and `price_analyst` return typed results instead of free text:
  `Forecast`, `MarketAnalysis` and `PurchaseAdvice` (JSON through the MCP server and `registry.query`). The rules their
  prompts asked for are now enforced by validation, so a violation makes the model retry: scenario probabilities add
  up to 1 (within 0.02), price ranges are ordered, at least one source is cited, and `disclaimer` is always the
  standard not-financial-advice line, set by code. `crypto_oracle`'s accountability step is the `previous_forecast`
  field. This affects every caller, not only MCP: `registry.query`, `delegate_to_agent` (another agent receiving the
  reply) and the `run_<agent>` MCP tools now return the JSON text of the model (`model_dump_json()`), so code that
  treated the reply as prose must `json.loads` it. The exact shapes are the models in `core/models/analysis.py`.
- **Behaviour change:** `holiday_planner` and `accountant` return typed results instead of free text: `Itinerary` and
  `CostBreakdown` (`core/models/personal.py`). Same caller impact as the investing agents: `registry.query`,
  `delegate_to_agent` and the `run_<agent>` MCP tools return the JSON text of the model, so code that treated the reply
  as prose must `json.loads` it. Enforced by validation (a violation makes the model retry): an itinerary's days are
  numbered from 1 with ordered ISO dates inside the trip dates, an unverified item says where to confirm it, and it
  cites a source; a breakdown's fixed costs are exact, estimates have ordered ranges, exchange rates are dated, and it
  cites a source. Totals are computed in code (`estimated_total`, `within_budget`, `total_low`, `total_high`), the
  accountant's `disclaimer` is set by code, and a request missing key details is answered with `open_questions`
  instead of an invented plan.
- Agent model defaults name real LiteLLM groups: reasoning roles use `deepseek-v4-pro`, docs and the personal
  utility agents use `subagent`. The old shared `claude-sonnet-5` alias silently resolved to DeepSeek (the proxy
  has no Anthropic credential), so prompts and evals tuned on the belief that agents ran on Claude were wrong.
- `AGENT_TRANSPORT_ATTEMPTS` defaults to 1: LiteLLM already retries and walks its fallback chain, and stacking a
  second retry layer turned one call into up to ~12 upstream attempts.
- With `MODEL_PROVIDER=litellm`, calls made inside a pipeline run send `metadata.session_id` (the run id) and
  `metadata.trace_name` so Langfuse groups a run's generations.

### Added
- `SEED_ROOTS` setting (directories the `seed_project_context` MCP tool may read; empty means `OUTPUT_DIR`).
- Promptfoo cases for `holiday_planner` (a complete request gives an `Itinerary` inside the dates with a derived budget
  check; a vague one gets questions, not a plan) and `accountant` (a `CostBreakdown` whose totals are summed from lines
  in one currency, with the supplied rate listed; a vague request gets questions). Figures are supplied in the request
  and tools are ruled out, so they do not depend on live data.
- `tests/unit/test_limits_under_load.py` drives the real provider with many concurrent callers and a fake model: the
  concurrency cap is never exceeded, permits are released after errors, timeouts and cancellations, retries stay inside
  the permit and cost at most `attempts` runs per call, and a delegation fan-out does not deadlock. It documents that
  children run under the parent's permit, so the cap bounds root runs, not the model calls a delegating run makes.
- `tests/unit/test_agent_consistency.py` cross-checks every agent's registry entry, `*_model` settings field and
  tool categories (no unassigned or orphaned agents, no categories without a tool, no unread model fields).

### Fixed
- `MergeStage` checked that a file stays inside the target project with a string-prefix test, so a path such as `../proj-evil/x` was accepted for a project at `.../proj`. It now uses `Path.is_relative_to`.
- `sync_memory_to_claude` no longer raises `FileNotFoundError` after pushing the Claude entries when Antigravity's implicit-memory directory does not exist; that step is skipped.
- **Security (Sonar findings):**
  - The MCP tool `seed_project_context` indexed any directory a caller named. It now only reads inside `SEED_ROOTS`
    (default `OUTPUT_DIR`) and skips symlinks that leave the project. `scripts/seed_projects.py` uses the new
    unrestricted `mcp_server.seed_project_files` for its own clones.
  - `live_trace` reads only `.jsonl` files inside the current directory; `check_changelog.py` refuses a base ref that
    could be read as a git option; an unrecognised `.cto_approved` marker is logged by length only (never its content).
  - The image runs as UID 10001 instead of root, `COPY` names `README.md` instead of a glob, and CI checks the image
    is unprivileged and its working directory is writable.
- Sonar test smells cleared: composite assertions split (S9073), `pytest.raises` blocks hold a single call (S5778) and a
  lambda patch became `AsyncMock(return_value=...)` (S9081). No behaviour change.
- `make smoke-deploy` prints one line when the endpoint cannot be used (for example `the endpoint answered HTTP 503
  (Service Unavailable)`) instead of an exception-group traceback, and still exits 1.
- `make smoke-deploy` asks `crypto_oracle` for a forecast from supplied figures with tools ruled out, instead of an
  open-ended one. The open-ended request let the agent research freely and hit the pod key's tokens-per-minute limit
  (HTTP 429), so the check failed for a reason unrelated to the deploy.
- `execute_test_suite` started `go mod init` and `go mod tidy` without waiting for them, so `go test` could run
  before the module existed. It now reuses the same awaited, time-limited setup as `run_go_build` / `run_go_vet`
  (`init_and_tidy`, now in `tools/go_common.py`, which also reaps a process it kills on timeout); a slow setup or a missing `go` toolchain returns a failure message instead of hanging or raising.
- `run_go_build`, `run_go_vet`, `validate_frontend_code` and `execute_test_suite` return a failure message for any
  file name that cannot be written (NUL byte, a path whose parent is a file, a name that is too long), not only for
  traversal attempts; they used to raise out of the tool. The shared error tuple is `WRITE_NAME_ERRORS`.
- Resuming a pipeline reuses the latest checkpointed run id instead of a fresh random one that never matched an
  earlier checkpoint, and logs a warning when `resume` is requested but no checkpoint exists.

### Added
- `make smoke-deploy` (`scripts/post_deploy_smoke.py`): after a deploy, calls `run_review`, `run_security` and
  `run_crypto_oracle` through the MCP endpoint and checks each result parses as `ReviewResult`, `SecurityReport` and
  `Forecast`. Costs tokens, so it is not part of `make ci`.
- Promptfoo cases for the `security` score bands (clean code passes, critical findings score 39 or less, a high finding
  fails at 69 or less, `passed` follows the score) and for the typed investing agents (`crypto_oracle`, `market_shark`,
  `price_analyst`: parseable JSON, grounded figures, no guaranteed outcomes). The two existing cases are scoped to
  `review`/`security` with `providers:`. Not run in CI; they need the LiteLLM key.
- CI `typecheck` job runs `mypy src` and must report no errors; `make ci` includes `typecheck`. `[tool.mypy]` ignores missing
  imports for optional or untyped packages (turso, redis_agent_memory, crewai, autogen, sentence_transformers, asyncpg,
  yfinance), so the result no longer depends on which extras are installed. The two real errors (`yaml = None` in
  `tools/validate.py`, the Wikipedia wrapper call in `tools/langchain_bridge.py`) are fixed.
- Durable pipeline runs: every status change of a background run is written to a `pipeline_runs` table in the
  pgvector database (best-effort; memory only when `DATABASE_URL` is unset). `check_pipeline_status` still finds a
  run after a restart and reports one that was in flight as `interrupted`.
- MCP tools `resume_pipeline_run` (restart an interrupted, failed or cancelled run under the same id from its
  checkpoints) and `approve_cto_gate` (writes the approval marker the CTO gate polls for, so a caller outside the
  pod can approve a run; the project name is checked against path traversal).

### Fixed
- The CTO gate no longer treats any existing `.cto_approved` file as approval: the content is read, so a marker
  containing `false`/`0`/`no` stops the run (it used to pass the existence check), and the poller ends on an
  explicit rejection instead of waiting for the timeout.
- `submit_pipeline_run` rejects a caller-supplied run id that is not `pipe_` plus 12 hex characters, and
  refuses to start a run id whose previous task is still alive (its handle would become unreachable).
- Resuming a pipeline reuses the latest checkpointed run id instead of a fresh random one that never matched an
  earlier checkpoint, and logs a warning when `resume` is requested but no checkpoint exists.

### Changed
- Agent model defaults name real LiteLLM groups: reasoning roles use `deepseek-v4-pro`, docs and the personal
  utility agents use `subagent`. The old shared `claude-sonnet-5` alias silently resolved to DeepSeek (the proxy
  has no Anthropic credential), so prompts and evals tuned on the belief that agents ran on Claude were wrong.
- `AGENT_TRANSPORT_ATTEMPTS` defaults to 1: LiteLLM already retries and walks its fallback chain, and stacking a
  second retry layer turned one call into up to ~12 upstream attempts.
- With `MODEL_PROVIDER=litellm`, calls made inside a pipeline run send `metadata.session_id` (the run id) and
  `metadata.trace_name` so Langfuse groups a run's generations.

### Added
- `EventTrigger` now carries `owner`, `consumers` and `payload_schema`, `UserStory` carries `statement` and
  `bounded_context`, and `AgentAdvice` carries `architectural_constraints`: the BA and CTO prompts already
  asked for this data and it was being discarded. The advisory stage passes the constraints on to downstream agents.
- `ReviewCriteria` deps for the review agent; the pipeline passes the bounded contexts found by the BA stage.
- promptfoo agent evals: `promptfooconfig.yaml`, an in-process provider in
  `evals/provider.py`, and `make eval` (#18).
- README section on running the evals, including the gateway key requirements.
- Eval: the clean-snippet case now fails if the agent reports any hardcoded secret, instead of a
  loose rubric that a rejection for unrelated reasons could satisfy.
- SonarCloud automatic analysis skips `promptfooconfig.yaml` (eval data, not code).
- CI image smoke test: builds the linux/arm64 image on pull requests, gates its
  size and checks torch/turso/Go/Node inside it (#19).
- Changelog, `AGENTS.md`, a pull request template and a CI check that every
  pull request updates this file.
- Agent guidance in `.ai/` (coding standards, testing, architecture, security, a
  full-repository review prompt), `CLAUDE.md`, and a workflow that copies each merged
  PR's "AI Context & Rules Update" section into `.ai/recent_context.md`.

### Changed
- `SecurityReport.passed` is derived from `score >= SECURITY_PASS_SCORE` (70) in code, as `QualityGateResult.passed`
  already is from its complexity score; the value the model supplies is ignored. The security agent's prompt now
  defines the score bands (critical at most 39, high at most 69, medium/low or clean 70 and above), so "no
  critical/high findings" and a passing score mean the same thing. This fixes verdicts such as `passed: false` with
  score 55 and "no critical/high findings". Behaviour change: an audit whose score is below 70 now blocks the
  quality gate even if the model said `passed: true`.
- `log_activity` and `query_history` are best-effort: when the memory store is unreachable (for example Postgres
  rejects the connection) they log a warning and return a short message instead of raising, so the agent run that
  called them continues. Previously one failed memory write aborted the whole run (seen in `make eval`, where the
  review agent happened to call `log_activity` with no database configured). The message names only the exception
  type, not its text.
- **Behaviour change:** with `MODEL_PROVIDER=litellm` (the default) a missing `LITELLM_VIRTUAL_KEY` /
  `LITELLM_MASTER_KEY` now raises `ConfigurationError` when the model is first used, instead of silently
  sending the request to api.openai.com. Prefixed models are built lazily on first use, so environment changes
  after import are honoured; the `"pytest" in sys.modules` check is gone (no test key is needed now that models are lazy).
- `MODEL_PROVIDER=openai|gemini|vertexai` keep a per-agent model name that belongs to the provider (e.g.
  `BACKEND_MODEL=gpt-4.1`) instead of always using `gpt-4o` / `gemini-2.0-flash`; other names still fall back to
  that default; the fallback is logged once per name so a typo'd `*_MODEL` is visible.
- Agent results returned through `registry.query`, the MCP server (whose `run_<agent>` tool descriptions now say so)
  and the bridge agent are JSON for models, lists
  and dicts (they were Python reprs such as `agent_name='x' files={...}`). The legacy `result.data` branch is gone.
- `query_history` returns only the calling agent's entries for the current run (optionally one `pass_num`, or all
  agents' entries with `all_agents=true`); it used to search every agent's logs and ignored `pass_num`.
  `log_activity` now tags entries with agent, run and pass. Entries logged before this change carry no tags and
  are not returned unless `include_legacy=true` is passed (they cannot be attributed, so they come from any agent).
- Non-technical agents (accountant, weather, region analyzer, holiday planner, current events, price analyst,
  asymmetric reviewer, market shark, crypto oracle): prompts now require `get_current_datetime`, sourced numbers
  and clarifying questions; the weather agent is told about `get_weather_forecast`; the investment agents give
  scenarios with probabilities and a not-financial-advice line instead of "definitive" advice and price targets;
  the asymmetric reviewer is framed around consumer rights rather than "exploiting loopholes".
- Run limits are tighter and configurable (`AGENT_RETRIES=2`, `AGENT_REQUEST_LIMIT=40`,
  `AGENT_TOTAL_TOKENS_LIMIT=2000000`, `AGENT_TIMEOUT_SECONDS=600`, `AGENT_TRANSPORT_ATTEMPTS=3`; was 5 retries, 500 requests, 3600 s). The
  backend, service_engineer and frontend agents get a request cap of 60.
- The provider retries only 429/5xx/connection errors (up to 3 attempts); timeouts are no longer retried, so a
  stuck run can no longer burn three full timeouts.
- All stages, the registry and the LangGraph bridge share one LLM concurrency limiter sized by
  `MAX_CONCURRENT_LLM_CALLS` (the BA and advisory stages used private `Semaphore(10)`s).
- CTO and backend agents declare `deps_type` and render `OrgContext` / `ProjectState` into their instructions, so
  the CTO can actually verify against the tech stack, compliance tier and budget it is given.
- Tool `RunContext` annotations are `RunContext[Any]` instead of `RunContext[str]`, which removes the tool-signature `arg-type` errors mypy reported (the remaining mypy errors
  are missing optional packages and stubs; mypy is not run in CI).
- Review agent no longer hardcodes the `payments`/`notifications`/`users` contexts, and its prompt asks for mTLS on
  service-to-service traffic rather than "everywhere".
- Tool grants are explicit: an agent's own name no longer acts as a tool category, so a category named `docs` or
  `devops` can't silently grant tools to the agent of the same name. New `go_build`, `test_run`,
  `frontend_validation` and `config_validation` categories trim over-broad sets: backend loses
  `validate_frontend_code`/`execute_test_suite`, SRE loses the Go build tools, test agents lose the frontend validator.
- `register_tool` raises on a duplicate tool name (`search_files` was registered twice and the last import won).
- `planner`, `quality_gate` and `service_engineer` have their own `*_model` settings instead of borrowing
  `cto_model`, `review_model` and `backend_model`.
- Generated CI, Makefile, Dockerfile and implementation prompt share one entry point (`GO_ENTRYPOINT`, default
  `cmd/server/main.go`); the CI pins golangci-lint and govulncheck (`GOLANGCI_LINT_VERSION`,
  `GOVULNCHECK_VERSION`); compose uses the current file format and `docker compose`.
- `generate_test_suite`, `generate_go_table_driven_tests` and `create_component` make one structured call to a
  private tool-less agent (they were two calls through registered agents that could delegate or write files) and
  pass the parent's usage so its limits apply. The Go test generator now receives the source it is testing.
- `uv.lock` is now tracked (local `uv sync` / `make install` reproduce CI's resolution), and
  `make lock-check` fails when it is stale.
- Image dependencies install from a hash-pinned lockfile with CPU-only torch;
  the image shrank from 10.8 GB to 3.6 GB (#16).

### Removed
- `core/adapters/mcp_client.py` (`MCPClientPort`, `StandardMCPClient`): an unused stub whose methods were TODOs.
- Placeholder tools that returned empty or comment-only files: `scaffold_go_service`, `provision_k8s_cluster`, and
  the BA agent's `define_user_stories`/`map_event_triggers` (nested LLM calls that duplicated the agent's own
  structured output). `apply_helm_chart` is now `generate_helm_chart` and returns a real chart; `generate_dockerfile`
  returns a real multi-stage, non-root Dockerfile.

### Fixed
- `execute_test_suite` wrote model-chosen test and target file names into its temp directory with `os.path.join`, so
  a name like `../x` or an absolute path escaped it (the same bug fixed for `run_go_build` and
  `validate_frontend_code` in #26). It now uses `safe_write`, returns `TEST EXECUTION FAILED` for such names, and
  writes off the event loop.
- Fixes for the six bugs a local Sonar scan reported: the run-context `ContextVar` default is an immutable mapping
  instead of a shared `{}`; `BridgeAgent.run` raises `AgentExecutionError` for an unresolved agent instead of an
  `assert` inside a `try`; `update_documentation` and `execute_test_suite` no longer block the event loop with
  synchronous file I/O.
- Delegated agent calls run under their parent's concurrency permit instead of acquiring a second one, removing
  a possible deadlock when concurrent delegation chains exhaust the limiter.
- `registry.query` passes no deps (instead of `{}`) when an agent has no deps factory, and records the running
  agent's name so `log_activity` no longer logs `agent_name="system"`.
- Prompts no longer claim "no local I/O" (frontend, sre, test_eng) or demand "100% test coverage" (test_eng), and
  the reviewer is asked to verify earlier feedback is resolved rather than to "address" it.
- The generated database migration's trigger is idempotent (`DROP TRIGGER IF EXISTS` before `CREATE TRIGGER`).
- Dependencies: dropped the nonexistent `pydantic-ai-slim[deepseek]` extra and capped `pydantic-ai-slim` at `<2.0`; an unbounded spec made pip backtrack for 25+ minutes (AP-69).
- Quality gate no longer crashes on a failed security audit (it read a nonexistent `vulnerabilities` field);
  the audit's `feedback` now goes to the retry loop.
- Gates fail closed: `AgentAdvice.approved_to_proceed` defaults to false, a CTO agent error blocks the
  pipeline (an explicit human override still applies), `QualityGateResult.passed` is computed from
  `complexity_score <= 60` instead of trusted from the model, and `SecurityReport.score` is bounded 0-100.
- Sandboxing of model-chosen paths and arguments: `read_file`/`list_dir`/`write_file` are confined to
  `ORCHESTRATION_CODEBASE_ROOT` (or CWD); `search_codebase` can no longer be made to run commands via
  ripgrep flags such as `--pre`; `run_gofmt_check`, `run_go_build`, `run_go_vet` and
  `validate_frontend_code` reject paths that escape their root or temp dir; and `safe_path_join` no longer
  accepts sibling directories that share a prefix (e.g. `/repo-evil`).
- `write_file` is no longer attached to 25 of 27 agents. It now sits in an `fs_write` category that no agent
  is assigned; code-facing agents get `read_file`/`list_dir` through `fs_read`.

## [0.2.0]

### Added
- Go and Node toolchains in the image (#15).
- `turso` extra installed in the image (AP-37, #14).
- Optional embedded Turso checkpoint backend (`STATE_BACKEND=turso`) and
  pipeline stage events recorded in Turso (AP-33, AP-34, AP-35; #11, #12).
- `scripts/seed_projects.py` to refresh the vector knowledge base from active
  repos, and a project filter for vector search and `query_context` (#7, #8).
- Makefile and docs on where the agents run (#10).

### Fixed
- Seed script upserts before pruning on the persistent loop and aborts on
  errors without deleting (#9).
