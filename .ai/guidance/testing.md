# Testing and evals

## Unit and integration tests (free, no real LLM calls)

- `make test` runs `pytest tests -q`; `make test-unit` and `make test-integration` run one directory. CI's `test` job runs `pytest tests -q` on Python 3.11 and 3.12 after `pip install -e ".[dev,vector]"`.
- `tests/unit/` holds one file per module or feature (`test_<module>.py`, plus `agy_bridge/`). `tests/integration/` runs the pipeline end to end with mocked agent providers (`AsyncMock`), not a live model. `tests/evaluation/` holds shared fixtures and evaluators (`fixtures.py`, `decision_evaluator.py`, `hallucination_detector.py`) used by both.
- `pytest` runs with `asyncio_mode = "auto"`, so async tests need no marker. Markers `unit` and `integration` exist; only a few tests use them.
- `core/models/factory.py` returns a `"mock-key"` LiteLLM key when `pytest` is imported and no key is set. Tests must not rely on reaching a real proxy, provider, Postgres or the network; mock them.
- `tests/unit/test_check_changelog.py` covers `scripts/check_changelog.py`, and `tests/unit/test_ai_context.py` covers `.github/scripts/ai_context.py` (loaded by path with `importlib`).

## Smoke

- `make smoke` imports `pydantic_agents.mcp_server` and counts its tools. CI's `smoke` job runs the same with only the base install, so an import of an optional extra at module level fails it.
- CI's `image-smoke` job builds the linux/arm64 image without pushing, gates its size, and checks inside it that torch is the CPU build, the MCP server lists tools, and Go, Node and ripgrep are present.

## Evals (cost tokens)

- `make eval` runs promptfoo against `promptfooconfig.yaml`. `evals/provider.py` runs registered agents in-process (no vector context, no Postgres); `llm-rubric` grading goes through the same LiteLLM gateway. It needs `LITELLM_BASE_URL` and `LITELLM_VIRTUAL_KEY` in the environment.
- Current cases: `review` and `security` (a hardcoded secret must be flagged, an environment-read key must not be), the `security` score bands (clean code passes with a score of 70 or more, critical findings score 39 or less, a high finding fails at 69 or less, `passed` always follows the score), and the typed investing agents `crypto_oracle`, `market_shark` and `price_analyst` (JSON parses, figures are grounded in the request, no guarantees), and the typed personal agents `holiday_planner` (dates stay inside the trip, the budget check follows the costs, unverified items say where to confirm, a vague request gets questions instead of a plan) and `accountant` (totals are the sum of the lines in one currency, the supplied rate is listed, a vague request gets questions). A test names its agents with `providers: [label]`; without it the case runs against every provider. The investing and personal cases pass their figures in the request and tell the agent not to call tools, so they do not depend on market data.
- First full live run (2026-10-25, `deepseek-v4-pro` grader): 16 of 16 cases pass after two cases were recalibrated (see the changelog).
- Evals do not run in CI. Run them for changes to agent prompts, `output_type` models, model routing (`settings.py`, `factory.py`) or tool assignments, and report the result on the PR.

## Merge and verification criteria

A PR is ready for review or merge only when every step below passes. If one doesn't apply, the PR says so and why.

1. **Green CI.** On the PR's latest commit, `CI` (`lint`, `test` on 3.11 and 3.12, `smoke`, `lock`, `image-smoke`) and `Changelog` pass. A red job that isn't caused by this PR is reported on the PR, never ignored.
2. **Sonar addressed.** The repo uses SonarCloud automatic analysis (`.sonarcloud.properties`). Its quality gate passes and the PR adds no new blocker, critical or major issues; fix lower-severity issues too, or say on the PR why one stays.
3. **Tests pass at every level that applies:**
   - **Unit:** `make lint && make typecheck && make test-unit` locally (CI's `typecheck` job runs mypy too).
   - **Integration:** `make test-integration` for changes to `core/pipeline/`, `core/stages/`, the registry or the MCP server.
   - **Evals:** `make eval` for prompt, output model, model routing or tool assignment changes.
   - **Smoke:** `make smoke`; for Dockerfile or dependency changes, `make lock-check` and the `image-smoke` job.
4. **Coverage and gaps.**
   - Run `make cov` (`pytest --cov=pydantic_agents --cov-report=term-missing`); add `--cov-branch` to see branches. No coverage floor is enforced, so check by hand that the lines and branches the PR adds or changes are covered.
   - Every new or changed branch needs a test, including error paths, empty and boundary inputs, path traversal attempts and malformed tool arguments.
   - Write down any gap you leave and why.
