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
COPY pyproject.toml README* ./
COPY src ./src

# vector extra: pgvector/sentence-transformers for ADR/architecture context retrieval.
# turso extra: pyturso for the optional embedded checkpoint backend (STATE_BACKEND=turso).
# Inert unless that setting is enabled; the default backend is still disk.
# --only-binary :all: refuses sdists, so no third-party setup.py/build scripts run during the image
# build. All dependencies ship linux/arm64 wheels for Python 3.12 (verified with a wheel-only resolve).
RUN pip install --no-cache-dir --only-binary :all: ".[turso,vector]" uvicorn

EXPOSE 8080

ENV LOGFIRE_SEND_TO_LOGFIRE=false

CMD ["uvicorn", "pydantic_agents.mcp_server:mcp.sse_app", "--host", "0.0.0.0", "--port", "8080"]
