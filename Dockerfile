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
RUN pip install --no-cache-dir ".[vector]" uvicorn

EXPOSE 8080

ENV LOGFIRE_SEND_TO_LOGFIRE=false

CMD ["uvicorn", "pydantic_agents.mcp_server:mcp.sse_app", "--host", "0.0.0.0", "--port", "8080"]
