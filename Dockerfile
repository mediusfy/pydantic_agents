# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pydantic-agents
COPY pyproject.toml README* ./
COPY src ./src

# vector extra: pgvector/sentence-transformers for ADR/architecture context retrieval.
# turso extra: pyturso for the optional embedded checkpoint backend (STATE_BACKEND=turso).
# Inert unless that setting is enabled; the default backend is still disk.
# --only-binary :all: refuses sdists, so no third-party setup.py/build scripts run during the image
# build. All dependencies ship linux/arm64 wheels for Python 3.12 (verified with a wheel-only resolve).
RUN pip install --no-cache-dir --only-binary :all: ".[vector,turso]" uvicorn

EXPOSE 8080

ENV LOGFIRE_SEND_TO_LOGFIRE=false

CMD ["uvicorn", "pydantic_agents.mcp_server:mcp.sse_app", "--host", "0.0.0.0", "--port", "8080"]
