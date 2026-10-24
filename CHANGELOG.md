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
- Agent model defaults name real LiteLLM groups: reasoning roles use `deepseek-v4-pro`, docs and the personal
  utility agents use `subagent`. The old shared `claude-sonnet-5` alias silently resolved to DeepSeek (the proxy
  has no Anthropic credential), so prompts and evals tuned on the belief that agents ran on Claude were wrong.
- `AGENT_TRANSPORT_ATTEMPTS` defaults to 1: LiteLLM already retries and walks its fallback chain, and stacking a
  second retry layer turned one call into up to ~12 upstream attempts.
- With `MODEL_PROVIDER=litellm`, calls made inside a pipeline run send `metadata.session_id` (the run id) and
  `metadata.trace_name` so Langfuse groups a run's generations.

### Fixed
- `execute_test_suite` started `go mod init` and `go mod tidy` without waiting for them, so `go test` could run
  before the module existed. It now reuses the same awaited, time-limited setup as `run_go_build` / `run_go_vet`
  (`init_and_tidy`); a slow setup or a missing `go` toolchain returns a failure message instead of hanging or raising.
- `run_go_build`, `run_go_vet`, `validate_frontend_code` and `execute_test_suite` return a failure message for any
  file name that cannot be written (NUL byte, a path whose parent is a file, a name that is too long), not only for
  traversal attempts; they used to raise out of the tool. The shared error tuple is `WRITE_NAME_ERRORS`.
- Resuming a pipeline reuses the latest checkpointed run id instead of a fresh random one that never matched an
  earlier checkpoint, and logs a warning when `resume` is requested but no checkpoint exists.

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
