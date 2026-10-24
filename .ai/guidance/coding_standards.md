# Coding standards

## Language and tooling

- Python `>=3.11` (`requires-python`); CI tests on 3.11 and 3.12, and the image runs 3.12. Use `uv run` (the Makefile does) so the locked `uv.lock` environment is used.
- ruff: line length 120, target `py311`, rule sets `E`, `F`, `I`, `N`, `W`, `UP`, `B`, `SIM`, `ARG`, `C4`, `T20`. `T20` bans `print`; the exceptions are listed in `[tool.ruff.lint.per-file-ignores]`. `make lint` checks, `make lint-fix` and `make format` change files.
- mypy runs on `src` (`make typecheck`) and CI's `typecheck` job enforces zero errors. `[tool.mypy]` in `pyproject.toml` only ignores missing imports for optional or untyped packages (turso, crewai, autogen, asyncpg, yfinance, ...); fix new errors rather than adding ignores.
- Package layout is `src/pydantic_agents/` (hatchling wheel). `core/` holds the framework, `agents/` the agent definitions, `mcp_server.py` the MCP entry point.

## Dependencies

- Runtime dependencies and extras (`vector`, `ingest`, `turso`, `crewai`, `autogen`, `dev`) are declared in `pyproject.toml`. Code that uses an optional extra imports it lazily and fails with a clear error when it is missing (see `core/pipeline/factory.py::_ensure_dependency`).
- Two lockfiles are tracked: `uv.lock` for local development, and `requirements.lock` (hash-pinned, wheels only, linux/arm64, CPython 3.12, CPU-only torch) for the Docker image. After a dependency change run `uv lock` and `make lock`, and commit both; `make lock-check` fails when either is stale.
- The image size is gated in CI (`IMAGE_SIZE_LIMIT_GB` in `ci.yml`). A dependency that pulls in CUDA torch or similar will fail it.

## Agents and tools

- Each agent is a module-level `pydantic_ai.Agent` with a typed `output_type` from `core/models/core.py`, and a model from `settings.<name>_model` through `core/models/factory.py::create_model`.
- Agents are registered in `agents/registration.py`. Tools are registered with `register_tool(name, fn, description, categories)` in `core/tools/*.py` and attached at first `registry.get(name)` according to `AGENT_TOOL_CATEGORIES` in `core/tool_assignments.py`. Agent-specific tools are defined with `@<name>_agent.tool` in the agent module (`ba`, `backend`, `database`, `devops`, `frontend`, `service_engineer`, `sre`, `test_eng` do this); anything another agent could use belongs in `core/tools/` with a category instead. The README's statement that agent files have no `@agent.tool` decorators is out of date.
- Tool names are one flat namespace: `register_tool` overwrites an existing name without warning, and modules are imported in the order listed in `core/tools/__init__.py`. `search_files` is registered by both `fs_tools.py` and `search.py`, so only the `search.py` version (and its categories) is live. Check for a clash before adding a tool.
- A tool whose first parameter is `RunContext` is attached with `agent.tool`, otherwise with `agent.tool_plain`. Every tool is wrapped with a logfire span (`core/tools/registry.py`).
- Tools return an error string rather than raising for expected failures (bad path, HTTP error), so the model can recover.

## Error handling

- Domain errors live in `core/exceptions.py` (`AgentNotFoundError`, `AgentExecutionError`, `PathTraversalError`, ...). The MCP server wraps agent failures in `AgentExecutionError`.
- Telemetry and observability calls are best-effort and must not fail an agent run (`mcp_server.run_agent_by_name` suppresses telemetry errors).
- Settings come from `core/settings.py` (pydantic-settings, `.env`, case-insensitive). Read configuration through `settings`, not `os.environ`, in new code.

## Changelog

- Every PR adds at least one bullet to `CHANGELOG.md` under `## [Unreleased]`, in the matching `### Added` / `### Changed` / `### Fixed` / `### Removed` subsection (create the subsection if it is missing).
- Write for a reader of the release notes: what changed and why it matters, not which files you touched. Reference the PR or ticket, for example `(#123)`.
- This applies to behaviour, agents, MCP tools, the image, CI, dependencies and docs.
- `scripts/check_changelog.py` (the `changelog` workflow) fails the PR if there is no new bullet under Unreleased. For a change with genuinely nothing to record (typo, formatting), a maintainer can apply the `skip-changelog` label; do not add the label yourself to get past the check.
- When a version is cut, rename Unreleased to the version and date and open a fresh Unreleased section; the version follows `project.version` in `pyproject.toml`.
