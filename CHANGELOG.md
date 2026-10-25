# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
`project.version` in `pyproject.toml`.

Every pull request that changes behaviour, the image, CI, dependencies or
docs adds a bullet under **Unreleased** (CI enforces this; see `AGENTS.md`).
When a version is cut, rename **Unreleased** to the version and date and open a
fresh **Unreleased** section above it.

## [Unreleased]

### Changed
- CI installs with `uv sync --frozen --no-build --no-install-project` and runs tools with `uv run --no-sync` (Sonar
  S8541): only prebuilt wheels are installed, so no dependency build script runs, and the project is used from `src`
  (`PYTHONPATH=src` for the test and smoke jobs). The image's `useradd` shares a `RUN` with the project install (S7031).
- `check_changelog.py` uses `\w` with `re.ASCII`, and the heading regex in `ai_context.py` no longer has overlapping
  quantifiers (S6353, S8786); `cto_gate.read_marker` logs the length of an unrecognised marker, not its content (S5145).
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
