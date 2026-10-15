.DEFAULT_GOAL := help

IMAGE ?= pydantic-agents:dev
PORT  ?= 8080

.PHONY: help install lint lint-fix format typecheck test test-unit test-integration cov smoke serve seed seed-apply docker-build lock lock-check ci clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Sync the venv with dev + vector extras (uv)
	uv sync --extra dev --extra vector

lint: ## ruff check (same scope as CI)
	uv run ruff check src tests scripts

lint-fix: ## ruff check --fix
	uv run ruff check --fix src tests scripts

format: ## ruff format
	uv run ruff format src tests scripts

typecheck: ## mypy over src
	uv run mypy src

test: ## Full test suite
	uv run pytest tests -q

test-unit: ## Unit tests only
	uv run pytest tests/unit -q

test-integration: ## Integration tests only
	uv run pytest tests/integration -q

cov: ## Tests with coverage report
	uv run pytest tests -q --cov=pydantic_agents --cov-report=term-missing

smoke: ## Import the MCP server and count its tools (mirrors CI smoke job)
	uv run python -c "import asyncio; from pydantic_agents.mcp_server import mcp; print(len(asyncio.run(mcp.list_tools())), 'tools')"

serve: ## Run the MCP server locally on $(PORT)
	uv run uvicorn pydantic_agents.mcp_server:mcp.sse_app --host 127.0.0.1 --port $(PORT)

seed: ## Dry-run the vector knowledge-base refresh
	uv run python scripts/seed_projects.py

seed-apply: ## Apply the refresh (needs DATABASE_URL, e.g. via kubectl port-forward)
	uv run python scripts/seed_projects.py --apply

docker-build: ## Build the image locally ($(IMAGE))
	docker build -t $(IMAGE) .

LOCK_ARGS = pyproject.toml requirements-build.in --extra vector --extra turso --generate-hashes \
	--python-platform aarch64-unknown-linux-gnu --python-version 3.12 --only-binary :all:

lock: ## Regenerate requirements.lock (hash-pinned, linux/arm64 py3.12) used by the Dockerfile
	uv pip compile $(LOCK_ARGS) -o requirements.lock

lock-check: ## Fail if requirements.lock is stale relative to pyproject.toml
	uv pip compile $(LOCK_ARGS) -o .requirements.lock.check -q
	@grep -v '^#' requirements.lock > .requirements.lock.a; grep -v '^#' .requirements.lock.check > .requirements.lock.b; \
		diff -q .requirements.lock.a .requirements.lock.b >/dev/null; rc=$$?; \
		rm -f .requirements.lock.a .requirements.lock.b .requirements.lock.check; \
		if [ $$rc -ne 0 ]; then echo "requirements.lock is stale; run 'make lock'"; exit 1; fi

ci: lint test smoke ## Everything CI runs

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage dist build
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
