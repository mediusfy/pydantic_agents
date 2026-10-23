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

### Changed
- **Behaviour change:** with `MODEL_PROVIDER=litellm` (the default) a missing `LITELLM_VIRTUAL_KEY` /
  `LITELLM_MASTER_KEY` now raises `ConfigurationError` when the model is first used, instead of silently
  sending the request to api.openai.com. Prefixed models are built lazily on first use, so environment changes
  after import are honoured; the `"pytest" in sys.modules` check is gone (tests set a mock key in `conftest.py`).
- `MODEL_PROVIDER=openai|gemini|vertexai` keep a per-agent model name that belongs to the provider (e.g.
  `BACKEND_MODEL=gpt-4.1`) instead of always using `gpt-4o` / `gemini-2.0-flash`; other names still fall back to
  that default.
- Agent results returned through `registry.query`, the MCP server and the bridge agent are JSON for models, lists
  and dicts (they were Python reprs such as `agent_name='x' files={...}`). The legacy `result.data` branch is gone.
- `query_history` returns only the calling agent's entries for the current run (optionally one `pass_num`, or all
  agents' entries with `all_agents=true`); it used to search every agent's logs and ignored `pass_num`.
  `log_activity` now tags entries with agent, run and pass. Entries logged before this change carry no tags and
  are not returned by `query_history`.
- Non-technical agents (accountant, weather, region analyzer, holiday planner, current events, price analyst,
  asymmetric reviewer, market shark, crypto oracle): prompts now require `get_current_datetime`, sourced numbers
  and clarifying questions; the weather agent is told about `get_weather_forecast`; the investment agents give
  scenarios with probabilities and a not-financial-advice line instead of "definitive" advice and price targets;
  the asymmetric reviewer is framed around consumer rights rather than "exploiting loopholes".
- `uv.lock` is now tracked (local `uv sync` / `make install` reproduce CI's resolution), and
  `make lock-check` fails when it is stale.
- Image dependencies install from a hash-pinned lockfile with CPU-only torch;
  the image shrank from 10.8 GB to 3.6 GB (#16).

### Fixed
- Dependencies: dropped the nonexistent `pydantic-ai-slim[deepseek]` extra and capped `pydantic-ai-slim` at `<2.0`; an unbounded spec made pip backtrack for 25+ minutes (AP-69).

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
