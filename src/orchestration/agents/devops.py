from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class DevopsDeps:
    agent_name: str = "devops"


devops_agent = Agent(
    create_model(settings.resolve_model(settings.devops_model)),
    deps_type=DevopsDeps,
    system_prompt="""You are a Principal DevOps Engineer.
Your objective is to design and generate complete CI/CD pipelines (GitHub Actions, GitLab CI), comprehensive Docker Compose setups for local development, and advanced Makefiles.
Prioritize secure, fast, and repeatable builds.
Return Artifact with files dict mapping relative paths to file contents.


CRITICAL WORKFLOW:
1. **History Review:** Use `query_history` to recall previous architectural and contextual decisions before generating logic.
2. **Log Progress:** Call `log_activity` after significant actions.
3. **Proactive Docs:** Keep `docs/` updated via `update_documentation` when implementing major features.
""",
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)


@devops_agent.tool
def generate_github_action(ctx: RunContext[str], service_name: str) -> str:
    """Returns an enterprise-grade GitHub Actions pipeline for a Go service using centralized settings."""
    return f"""name: {service_name} CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: "{settings.go_version}"
          cache: true # Ensures fast builds by caching modules and build outputs

      - name: Verify dependencies
        run: go mod verify

      - name: Run golangci-lint
        uses: golangci/golangci-lint-action@v4
        with:
          version: latest
          args: --timeout=5m

      - name: Run unit tests
        run: go test -v -race -coverprofile=coverage.out ./...

      - name: Run vulnerability scanner
        run: |
          go install golang.org/x/vuln/cmd/govulncheck@latest
          govulncheck ./...

      - name: Build binary
        run: go build -o bin/server ./cmd/{service_name}/main.go
"""


@devops_agent.tool
def generate_docker_compose(ctx: RunContext[str], service_name: str) -> str:
    """Returns a local Docker Compose setup including PostgreSQL and Redis dependencies."""
    return f"""version: '3.8'

services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=postgres
      - DB_PASSWORD=${{DB_PASSWORD:-changeme}}
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${{DB_PASSWORD:-changeme}}
      - POSTGRES_DB={service_name}_db
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
"""


@devops_agent.tool
def generate_makefile(ctx: RunContext[str], service_name: str) -> str:
    """Returns a standardized Makefile for local development and build automation."""
    return f"""# Makefile for {service_name}

.PHONY: all build test lint clean docker-up docker-down

all: lint test build

build:
	@echo "Building..."
	go build -o bin/server ./cmd/{service_name}/main.go

test:
	@echo "Running tests..."
	go test -v -race ./...

lint:
	@echo "Linting..."
	golangci-lint run

clean:
	@echo "Cleaning..."
	rm -rf bin/
	go clean -testcache

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down
"""
