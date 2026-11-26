# syntax=docker/dockerfile:1.7
# Toolchains the agents shell out to: `go mod init/tidy/build/vet/test` (Go 1.26, matching
# settings.go_version) and pinned offline `eslint` / `vue-tsc` / `vitest` (frontend & test runners).
FROM golang:1.26-bookworm AS gotoolchain
FROM node:22-bookworm-slim AS nodetoolchain
RUN npm install -g --ignore-scripts --no-audit --no-fund \
    eslint@9.39.5 \
    typescript@5.9.3 \
    vue-tsc@3.3.14 \
    vitest@4.1.12

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap ca-certificates curl git ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gotoolchain /usr/local/go /usr/local/go
COPY --from=nodetoolchain /usr/local/bin/ /usr/local/bin/
COPY --from=nodetoolchain /usr/local/lib/node_modules /usr/local/lib/node_modules

# Caches under /tmp so the toolchains also work when root and /opt/pydantic-agents are read-only.
# CGO off: there is no C compiler in this image.
ENV PATH=/usr/local/go/bin:$PATH \
    GOCACHE=/tmp/go-cache \
    GOPATH=/tmp/go \
    CGO_ENABLED=0 \
    NPM_CONFIG_CACHE=/tmp/npm-cache

WORKDIR /opt/pydantic-agents

# Dependencies come from the hash-pinned lockfile (requirements.lock, linux/arm64, cp312), so the
# build is reproducible and tamper-evident. --only-binary :all: refuses sdists, so no third-party
# build scripts run. Regenerate with `make lock` after changing dependencies in pyproject.toml.
# The lock is platform-specific: build this image for linux/arm64 only (an amd64 build fails on wheel
# selection, by design).
# Extras baked in: vector (ADR/architecture retrieval) and turso (STATE_BACKEND=turso, inert by default).
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes --only-binary :all: -r requirements.lock

# Install the project itself without resolving anything: deps are already locked, and hatchling
# (its build backend) is in the lock too, hence --no-build-isolation.
COPY pyproject.toml README.md ./
COPY src ./src
# Keep /opt/pydantic-agents owned by root and read-only (0555/0444) so neither the server UID nor any
# child process can modify installed code. Runtime outputs (generated-app/, checkpoints, traces) live under
# /var/lib/pydantic-agents, and default file-tool sandboxing points to an empty /var/lib/pydantic-agents/workspace.
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && chmod -R a-w /opt/pydantic-agents \
    && useradd --system --uid 10001 --user-group --create-home --home-dir /home/agent agent \
    && install -d -m 0700 -o 10001 -g 10001 /var/lib/pydantic-agents /var/lib/pydantic-agents/workspace

WORKDIR /var/lib/pydantic-agents
USER 10001:10001

EXPOSE 8080

ENV LOGFIRE_SEND_TO_LOGFIRE=false \
    ORCHESTRATION_CODEBASE_ROOT=/var/lib/pydantic-agents/workspace

CMD ["uvicorn", "pydantic_agents.mcp_server:mcp.sse_app", "--host", "0.0.0.0", "--port", "8080"]
