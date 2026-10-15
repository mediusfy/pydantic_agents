# syntax=docker/dockerfile:1.7
# Toolchains the agents shell out to: `go mod init/tidy/build/vet` (backend,
# go_test, test_gen) and `npx eslint` / `npx vue-tsc` (frontend). Copied from
# the official images so versions are current; without them those agents fail
# with "No such file or directory: 'go'" / "'npx'".
FROM golang:1.24-bookworm AS gotoolchain
FROM node:22-bookworm-slim AS nodetoolchain

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gotoolchain /usr/local/go /usr/local/go
COPY --from=nodetoolchain /usr/local/bin/node /usr/local/bin/node
COPY --from=nodetoolchain /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Caches under /tmp so the toolchains also work if the root filesystem is
# ever made read-only. CGO off: there is no C compiler in this image.
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
COPY pyproject.toml README* ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

EXPOSE 8080

ENV LOGFIRE_SEND_TO_LOGFIRE=false

CMD ["uvicorn", "pydantic_agents.mcp_server:mcp.sse_app", "--host", "0.0.0.0", "--port", "8080"]
