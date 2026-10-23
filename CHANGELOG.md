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
